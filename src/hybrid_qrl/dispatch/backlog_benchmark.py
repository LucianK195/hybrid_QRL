"""Scale-aware and stable-backlog extension of the dispatch benchmark.

The study separates two necessary questions. First, can a fixed best-of-K
Rydberg-surrogate batch retain high reward as the binary action dimension
grows? Second, can those candidates be used after a realistic asynchronous
delay when only long-lived jobs are reserved for a future decision epoch?

The standardized detuning map and stochastic blockade process remain a
classical surrogate. Improvements reported here are algorithmic evidence, not
neutral-atom hardware evidence. Dense, QuTiP, manual-backend calibration and a
measured QPU latency trace remain independent physical-claim requirements.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from math import ceil, log, sqrt
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import numpy as np

from ..core import Action
from .baselines import (
    CandidateBatch,
    ProposalConfig,
    generate_candidates,
    proposal_weights,
    repair_action,
    solve_weighted_independent_set,
)
from .benchmark import oracle_weights, realized_step_reward
from .environment import (
    DispatchConfig,
    DispatchEnvironment,
    DispatchState,
    induced_dispatch_state,
)
from .latency_benchmark import (
    LatencyObservation,
    LatencyTrace,
    remap_candidate_job_ids,
    summarize_latency,
)
from .learning import ActorCriticModel


@dataclass(frozen=True)
class BacklogBenchmarkConfig:
    """Frozen selection, confirmation, and future-batch settings."""

    sizes: tuple[int, ...] = (20, 40, 60, 80, 100)
    training_episodes: int = 800
    training_seed: int = 8_831
    selection_sizes: tuple[int, ...] = (20, 60, 100)
    selection_seeds: int = 6
    confirmation_seeds: int = 20
    dynamic_sizes: tuple[int, ...] = (40, 100)
    dynamic_seeds: int = 12
    horizon: int = 18
    warmup_steps: int = 3
    candidate_budget: int = 16
    density: float = 0.12
    decision_step_ms: float = 1_000.0
    future_deadline_ms: float = 6_000.0
    stable_guard_steps: int = 1
    minimum_stable_jobs: int = 8
    stable_target_fraction: float = 0.25
    maximum_stable_target_jobs: int = 20
    fallback_budget_ms: float = 50.0
    oracle_time_limit_ms: float = 250.0
    epsilon: float = 0.05
    seed: int = 84_211

    def __post_init__(self) -> None:
        if not self.sizes or any(size < 8 or size > 100 for size in self.sizes):
            raise ValueError("sizes must contain values in [8, 100]")
        if self.training_episodes <= 0:
            raise ValueError("training_episodes must be positive")
        if self.selection_seeds <= 0 or self.confirmation_seeds <= 0:
            raise ValueError("selection and confirmation seeds must be positive")
        if self.dynamic_seeds <= 0 or self.horizon < 4:
            raise ValueError("dynamic settings must be positive")
        if self.candidate_budget <= 0:
            raise ValueError("candidate_budget must be positive")
        if self.decision_step_ms <= 0 or self.future_deadline_ms <= 0:
            raise ValueError("physical timing values must be positive")
        if self.stable_guard_steps < 0 or self.minimum_stable_jobs <= 0:
            raise ValueError("stable backlog settings are invalid")
        if not 0.0 < self.stable_target_fraction <= 1.0:
            raise ValueError("stable_target_fraction must lie in (0, 1]")
        if self.maximum_stable_target_jobs < self.minimum_stable_jobs:
            raise ValueError("maximum stable target is smaller than its minimum")


@dataclass(frozen=True)
class SamplerRegime:
    """Utility encoding and pulse schedule selected before confirmation."""

    name: str
    utility_encoding: str
    detuning_gain: float
    pulse_schedule: str

    def proposal(self, candidates: int) -> ProposalConfig:
        """Build the matching fixed-K proposal configuration."""

        return ProposalConfig(
            candidates=candidates,
            max_runtime_ms=2_000.0,
            utility_encoding=self.utility_encoding,
            detuning_gain=self.detuning_gain,
            pulse_schedule=self.pulse_schedule,
            geometry_error=0.0,
            cache_precision=2,
        )


@dataclass(frozen=True)
class PendingFutureBatch:
    """Candidate batch delayed until its preregistered future decision epoch."""

    issue_step: int
    arrival_step: int
    deadline_step: int
    observation: LatencyObservation
    reserved_job_ids: tuple[int, ...]
    candidate_job_ids: tuple[tuple[int, ...], ...]
    raw_generated: int
    raw_feasible: int
    projected_best_k_ratio: float
    projected_p_epsilon: float

    @property
    def deadline_met(self) -> bool:
        """Return whether retrieval precedes the future-batch deadline."""

        return self.arrival_step <= self.deadline_step


REGIME_GRID = (
    SamplerRegime("legacy-balanced", "mean", 0.5, "balanced"),
    SamplerRegime("standardized-025-balanced", "standardized", 0.25, "balanced"),
    SamplerRegime("standardized-025-adiabatic", "standardized", 0.25, "adiabatic"),
    SamplerRegime("standardized-050-balanced", "standardized", 0.50, "balanced"),
    SamplerRegime("standardized-050-adiabatic", "standardized", 0.50, "adiabatic"),
    SamplerRegime("standardized-075-adiabatic", "standardized", 0.75, "adiabatic"),
    SamplerRegime("standardized-025-extended", "standardized", 0.25, "extended"),
    SamplerRegime("standardized-050-extended", "standardized", 0.50, "extended"),
    SamplerRegime("standardized-075-extended", "standardized", 0.75, "extended"),
    SamplerRegime("standardized-100-extended", "standardized", 1.00, "extended"),
)

SCALING_METHODS = (
    "legacy_frozen",
    "multi_size_legacy",
    "scale_aware",
    "beam_search",
)

FUTURE_POLICIES = (
    "beam_immediate",
    "future_beam",
    "future_rydberg_legacy",
    "future_rydberg_scale_aware",
)


def _mean_ci(values: Iterable[float]) -> tuple[float, float]:
    items = np.asarray(list(values), dtype=float)
    average = float(np.mean(items))
    interval = (
        float(1.96 * np.std(items, ddof=1) / sqrt(len(items)))
        if len(items) > 1
        else 0.0
    )
    return average, interval


def _held_out_state(
    baseline_model: ActorCriticModel,
    size: int,
    seed: int,
    config: BacklogBenchmarkConfig,
) -> DispatchState:
    environment = DispatchEnvironment(
        DispatchConfig(
            n_jobs=size,
            density=config.density,
            horizon=max(config.warmup_steps + 2, 8),
        ),
        seed=seed,
    )
    state = environment.state()
    rng = np.random.default_rng(seed + 7_919)
    for _ in range(config.warmup_steps):
        action = baseline_model.actor.sample(state, rng)
        state, _, done, _ = environment.step(action)
        if done:
            break
    return state


def _pairwise_diversity(actions: tuple[Action, ...]) -> float:
    if len(actions) < 2:
        return 0.0
    distances = []
    for left in range(len(actions)):
        for right in range(left + 1, len(actions)):
            distances.append(
                float(
                    np.mean(
                        np.asarray(actions[left]) != np.asarray(actions[right])
                    )
                )
            )
    return float(np.mean(distances))


def _batch_record(
    *,
    batch: CandidateBatch,
    state: DispatchState,
    model: ActorCriticModel,
    reference_reward: float,
    epsilon: float,
) -> dict[str, float | None]:
    if not batch.actions:
        return {
            "best_k_ratio": 0.0,
            "critic_selected_ratio": 0.0,
            "utility_selected_ratio": 0.0,
            "epsilon_coverage": 0.0,
            "p_epsilon": 0.0,
            "k95": None,
            "raw_feasible_rate": 0.0,
            "diversity": 0.0,
        }
    ratios = [
        realized_step_reward(state, action, 1.0)
        / max(reference_reward, 1e-12)
        for action in batch.repaired_actions
    ]
    unique_ratios = [
        realized_step_reward(state, action, 1.0)
        / max(reference_reward, 1e-12)
        for action in batch.actions
    ]
    critic = model.best_action(state, list(batch.actions))
    weights = proposal_weights(model, state)
    utility = max(
        batch.actions,
        key=lambda action: float(np.asarray(action) @ weights),
    )
    p_epsilon = float(np.mean(np.asarray(ratios) >= 1.0 - epsilon))
    k95 = (
        int(ceil(log(0.05) / log(1.0 - p_epsilon)))
        if 0.0 < p_epsilon < 1.0
        else (1 if p_epsilon >= 1.0 else None)
    )
    return {
        "best_k_ratio": max(unique_ratios),
        "critic_selected_ratio": realized_step_reward(state, critic, 1.0)
        / max(reference_reward, 1e-12),
        "utility_selected_ratio": realized_step_reward(state, utility, 1.0)
        / max(reference_reward, 1e-12),
        "epsilon_coverage": float(max(unique_ratios) >= 1.0 - epsilon),
        "p_epsilon": p_epsilon,
        "k95": k95,
        "raw_feasible_rate": batch.raw_feasible / max(batch.raw_generated, 1),
        "diversity": _pairwise_diversity(batch.actions),
    }


def _selection_records(
    model: ActorCriticModel,
    baseline_model: ActorCriticModel,
    config: BacklogBenchmarkConfig,
) -> list[dict[str, Any]]:
    records = []
    for size in config.selection_sizes:
        for seed_index in range(config.selection_seeds):
            seed = config.seed + 110_003 + size * 101 + seed_index * 10_007
            state = _held_out_state(baseline_model, size, seed, config)
            oracle = solve_weighted_independent_set(
                state,
                oracle_weights(state, 1.0),
                config.oracle_time_limit_ms,
            )
            reference = realized_step_reward(state, oracle.action, 1.0)
            for regime_index, regime in enumerate(REGIME_GRID):
                batch = generate_candidates(
                    "rydberg_surrogate",
                    state,
                    model,
                    regime.proposal(config.candidate_budget),
                    np.random.default_rng(seed + 701 * regime_index),
                )
                metrics = _batch_record(
                    batch=batch,
                    state=state,
                    model=model,
                    reference_reward=reference,
                    epsilon=config.epsilon,
                )
                records.append(
                    {
                        "regime": regime.name,
                        "size": size,
                        "seed_index": seed_index,
                        "oracle_exact": oracle.success,
                        **metrics,
                    }
                )
    return records


def _select_regime(records: list[dict[str, Any]]) -> SamplerRegime:
    by_regime: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_regime[str(record["regime"])].append(record)
    ranked = []
    for name, items in by_regime.items():
        sizes = sorted({int(item["size"]) for item in items})
        size_means = [
            mean(
                float(item["best_k_ratio"])
                for item in items
                if int(item["size"]) == size
            )
            for size in sizes
        ]
        ranked.append((min(size_means), mean(size_means), name))
    selected_name = max(ranked)[2]
    return next(regime for regime in REGIME_GRID if regime.name == selected_name)


def _confirmation_records(
    *,
    model: ActorCriticModel,
    baseline_model: ActorCriticModel,
    selected_regime: SamplerRegime,
    config: BacklogBenchmarkConfig,
) -> list[dict[str, Any]]:
    records = []
    for size in config.sizes:
        for seed_index in range(config.confirmation_seeds):
            seed = config.seed + 5_000_009 + size * 103 + seed_index * 10_009
            state = _held_out_state(baseline_model, size, seed, config)
            oracle = solve_weighted_independent_set(
                state,
                oracle_weights(state, 1.0),
                config.oracle_time_limit_ms,
            )
            reference = realized_step_reward(state, oracle.action, 1.0)
            definitions = (
                (
                    "legacy_frozen",
                    "rydberg_surrogate",
                    baseline_model,
                    ProposalConfig(candidates=config.candidate_budget),
                ),
                (
                    "multi_size_legacy",
                    "rydberg_surrogate",
                    model,
                    ProposalConfig(candidates=config.candidate_budget),
                ),
                (
                    "scale_aware",
                    "rydberg_surrogate",
                    model,
                    selected_regime.proposal(config.candidate_budget),
                ),
                (
                    "beam_search",
                    "beam_search",
                    model,
                    ProposalConfig(
                        candidates=config.candidate_budget,
                        latency_ms=config.fallback_budget_ms,
                    ),
                ),
            )
            for method_index, (label, method, active_model, proposal) in enumerate(
                definitions
            ):
                batch = generate_candidates(
                    method,
                    state,
                    active_model,
                    proposal,
                    np.random.default_rng(seed + 907 * method_index),
                )
                metrics = _batch_record(
                    batch=batch,
                    state=state,
                    model=active_model,
                    reference_reward=reference,
                    epsilon=config.epsilon,
                )
                records.append(
                    {
                        "method": label,
                        "size": size,
                        "seed_index": seed_index,
                        "oracle_exact": oracle.success,
                        "proposal_latency_ms": batch.elapsed_ms,
                        **metrics,
                    }
                )
    return records


def _job_id_candidates(
    actions: Iterable[Action],
    state: DispatchState,
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(
            int(state.job_ids[node])
            for node, bit in enumerate(action)
            if bit
        )
        for action in actions
    )


def _expand_subaction(
    action: Action,
    nodes: np.ndarray,
    full_size: int,
) -> Action:
    bits = np.zeros(full_size, dtype=int)
    bits[nodes] = np.asarray(action, dtype=int)
    return tuple(int(bit) for bit in bits)


def _fallback_action(
    state: DispatchState,
    model: ActorCriticModel,
    blocked_job_ids: set[int],
    config: BacklogBenchmarkConfig,
    seed: int,
) -> tuple[Action, float]:
    nodes = np.asarray(
        [
            node
            for node, job_id in enumerate(state.job_ids)
            if int(job_id) not in blocked_job_ids
        ],
        dtype=int,
    )
    if len(nodes) == 0:
        return tuple(0 for _ in range(state.n_jobs)), 0.0
    substate = induced_dispatch_state(state, nodes)
    batch = generate_candidates(
        "beam_search",
        substate,
        model,
        ProposalConfig(
            candidates=config.candidate_budget,
            latency_ms=config.fallback_budget_ms,
        ),
        np.random.default_rng(seed),
    )
    if not batch.actions:
        return tuple(0 for _ in range(state.n_jobs)), batch.elapsed_ms
    selected = model.best_action(substate, list(batch.actions))
    return _expand_subaction(selected, nodes, state.n_jobs), batch.elapsed_ms


def _issue_future_batch(
    *,
    policy: str,
    state: DispatchState,
    model: ActorCriticModel,
    selected_regime: SamplerRegime,
    observation: LatencyObservation,
    step: int,
    config: BacklogBenchmarkConfig,
    seed: int,
) -> PendingFutureBatch | None:
    deadline_steps = int(
        ceil(config.future_deadline_ms / config.decision_step_ms)
    )
    arrival_steps = max(
        1,
        int(ceil(observation.total_ms / config.decision_step_ms)),
    )
    threshold = deadline_steps + config.stable_guard_steps + 1
    stable_nodes = np.flatnonzero(state.remaining >= threshold)
    if len(stable_nodes) < config.minimum_stable_jobs:
        return None
    target_count = min(
        config.maximum_stable_target_jobs,
        max(
            config.minimum_stable_jobs,
            int(ceil(config.stable_target_fraction * state.n_jobs)),
        ),
    )
    if len(stable_nodes) > target_count:
        priorities = proposal_weights(model, state)[stable_nodes]
        stable_nodes = stable_nodes[np.argsort(-priorities)[:target_count]]
    projection_steps = min(arrival_steps, deadline_steps)
    projected = induced_dispatch_state(
        state,
        stable_nodes,
        future_steps=projection_steps,
    )
    if policy == "future_beam":
        method = "beam_search"
        proposal = ProposalConfig(
            candidates=config.candidate_budget,
            latency_ms=config.fallback_budget_ms,
        )
    elif policy == "future_rydberg_legacy":
        method = "rydberg_surrogate"
        proposal = ProposalConfig(candidates=config.candidate_budget)
    elif policy == "future_rydberg_scale_aware":
        method = "rydberg_surrogate"
        proposal = selected_regime.proposal(config.candidate_budget)
    else:
        raise ValueError(f"unknown future policy: {policy}")
    batch = generate_candidates(
        method,
        projected,
        model,
        proposal,
        np.random.default_rng(seed),
    )
    oracle = solve_weighted_independent_set(
        projected,
        oracle_weights(projected, 1.0),
        config.oracle_time_limit_ms,
    )
    reference = realized_step_reward(projected, oracle.action, 1.0)
    metrics = _batch_record(
        batch=batch,
        state=projected,
        model=model,
        reference_reward=reference,
        epsilon=config.epsilon,
    )
    return PendingFutureBatch(
        issue_step=step,
        arrival_step=step + arrival_steps,
        deadline_step=step + deadline_steps,
        observation=observation,
        reserved_job_ids=tuple(int(item) for item in projected.job_ids),
        candidate_job_ids=_job_id_candidates(batch.repaired_actions, projected),
        raw_generated=batch.raw_generated,
        raw_feasible=batch.raw_feasible,
        projected_best_k_ratio=float(metrics["best_k_ratio"]),
        projected_p_epsilon=float(metrics["p_epsilon"]),
    )


def _reference_episode(
    environment_config: DispatchConfig,
    seed: int,
    config: BacklogBenchmarkConfig,
) -> tuple[float, float]:
    environment = DispatchEnvironment(environment_config, seed=seed)
    state = environment.state()
    total = 0.0
    exact = 0
    done = False
    while not done:
        oracle = solve_weighted_independent_set(
            state,
            oracle_weights(state, 1.0),
            config.oracle_time_limit_ms,
        )
        state, reward, done, _ = environment.step(oracle.action)
        total += reward
        exact += int(oracle.success)
    return total, exact / environment_config.horizon


def _run_future_episode(
    *,
    policy: str,
    model: ActorCriticModel,
    selected_regime: SamplerRegime,
    trace: LatencyTrace,
    environment_config: DispatchConfig,
    episode_seed: int,
    trace_offset: int,
    reference_return: float,
    config: BacklogBenchmarkConfig,
) -> dict[str, Any]:
    environment = DispatchEnvironment(environment_config, seed=episode_seed)
    state = environment.state()
    pending: PendingFutureBatch | None = None
    requests = 0
    deadline_misses = 0
    arrivals = 0
    used = 0
    total_reward = 0.0
    total_missed = 0.0
    raw_generated = 0
    raw_feasible = 0
    arrival_candidates = 0
    arrival_feasible = 0
    original_selected = 0
    surviving_selected = 0
    stable_sizes: list[int] = []
    projected_best: list[float] = []
    projected_p_epsilon: list[float] = []
    fallback_latencies: list[float] = []
    observed_latencies: list[float] = []
    shots = 0
    step = 0
    done = False

    while not done:
        arrived_ids: tuple[tuple[int, ...], ...] = ()
        arrival_reservation: set[int] = set()
        if pending is not None:
            if pending.deadline_met and pending.arrival_step <= step:
                arrivals += 1
                arrived_ids = pending.candidate_job_ids
                arrival_reservation = set(pending.reserved_job_ids)
                pending = None
            elif not pending.deadline_met and pending.deadline_step <= step:
                pending = None

        uses_future = policy != "beam_immediate"
        deadline_steps = int(
            ceil(config.future_deadline_ms / config.decision_step_ms)
        )
        can_resolve = step + deadline_steps < config.horizon
        if (
            uses_future
            and pending is None
            and not arrived_ids
            and can_resolve
        ):
            observation = trace.observation(trace_offset + requests)
            issued = _issue_future_batch(
                policy=policy,
                state=state,
                model=model,
                selected_regime=selected_regime,
                observation=observation,
                step=step,
                config=config,
                seed=episode_seed + 70_001 + requests,
            )
            if issued is not None:
                pending = issued
                requests += 1
                deadline_misses += int(not issued.deadline_met)
                raw_generated += issued.raw_generated
                raw_feasible += issued.raw_feasible
                stable_sizes.append(len(issued.reserved_job_ids))
                projected_best.append(issued.projected_best_k_ratio)
                projected_p_epsilon.append(issued.projected_p_epsilon)
                observed_latencies.append(observation.total_ms)
                shots += observation.shots

        blocked = (
            set(pending.reserved_job_ids)
            if pending is not None
            else arrival_reservation
        )
        fallback, fallback_latency = _fallback_action(
            state,
            model,
            blocked if uses_future else set(),
            config,
            episode_seed + 10_003 * step,
        )
        fallback_latencies.append(fallback_latency)
        action = fallback

        if arrived_ids:
            weights = proposal_weights(model, state)
            merged_actions: list[Action] = []
            returned_parts: list[Action] = []
            for identities in arrived_ids:
                raw = remap_candidate_job_ids(identities, state)
                original_selected += len(identities)
                surviving_selected += sum(raw)
                arrival_candidates += 1
                safe_returned = repair_action(raw, state.graph, weights)
                arrival_feasible += int(state.graph.is_feasible(safe_returned))
                merged = tuple(
                    int(left or right)
                    for left, right in zip(fallback, safe_returned)
                )
                merged_actions.append(repair_action(merged, state.graph, weights))
                returned_parts.append(safe_returned)
            unique = list(dict.fromkeys(merged_actions))
            if unique:
                action = model.best_action(state, unique)
                selected_ids = {
                    int(state.job_ids[node])
                    for node, bit in enumerate(action)
                    if bit
                }
                contributed = any(
                    selected_ids
                    & {
                        int(state.job_ids[node])
                        for node, bit in enumerate(part)
                        if bit
                    }
                    for part in returned_parts
                )
                used += int(contributed)

        if not state.graph.is_feasible(action):
            raise RuntimeError("post-repair future action is infeasible")
        state, reward, done, info = environment.step(action)
        total_reward += reward
        total_missed += info["missed_value"]
        step += 1

    request_denominator = max(requests, 1)
    candidate_denominator = max(arrival_candidates, 1)
    return {
        "policy": policy,
        "n_jobs": environment_config.n_jobs,
        "episode_seed": episode_seed,
        "episode_return": total_reward,
        "reference_return": reference_return,
        "reward_ratio": total_reward / max(reference_return, 1e-12),
        "missed_value": total_missed,
        "requests_issued": requests,
        "deadline_misses": deadline_misses,
        "deadline_compliance": 1.0 - deadline_misses / request_denominator,
        "results_arrived": arrivals,
        "results_used": used,
        "future_result_utilization": used / request_denominator,
        "eligible_result_utilization": used / max(arrivals, 1),
        "raw_feasible_rate": raw_feasible / max(raw_generated, 1),
        "post_repair_feasible_rate": arrival_feasible / candidate_denominator,
        "selected_identity_survival_rate": surviving_selected
        / max(original_selected, 1),
        "mean_stable_pool_size": mean(stable_sizes) if stable_sizes else 0.0,
        "projected_best_k_ratio": mean(projected_best)
        if projected_best
        else 0.0,
        "projected_p_epsilon": mean(projected_p_epsilon)
        if projected_p_epsilon
        else 0.0,
        "mean_observed_latency_ms": mean(observed_latencies)
        if observed_latencies
        else 0.0,
        "mean_fallback_latency_ms": mean(fallback_latencies),
        "shots_total": shots,
    }


def _future_records(
    *,
    model: ActorCriticModel,
    selected_regime: SamplerRegime,
    trace: LatencyTrace,
    config: BacklogBenchmarkConfig,
) -> list[dict[str, Any]]:
    records = []
    for size_index, size in enumerate(config.dynamic_sizes):
        environment_config = DispatchConfig(
            n_jobs=size,
            density=config.density,
            horizon=config.horizon,
        )
        for seed_index in range(config.dynamic_seeds):
            episode_seed = (
                config.seed + 9_000_011 + size * 107 + seed_index * 10_037
            )
            reference, exact_rate = _reference_episode(
                environment_config,
                episode_seed,
                config,
            )
            trace_offset = (
                size_index * config.dynamic_seeds + seed_index
            ) * config.horizon
            for policy in FUTURE_POLICIES:
                record = _run_future_episode(
                    policy=policy,
                    model=model,
                    selected_regime=selected_regime,
                    trace=trace,
                    environment_config=environment_config,
                    episode_seed=episode_seed,
                    trace_offset=trace_offset,
                    reference_return=reference,
                    config=config,
                )
                record["seed_index"] = seed_index
                record["reference_oracle_exact_rate"] = exact_rate
                records.append(record)
    return records


def _summarize(
    records: list[dict[str, Any]],
    keys: tuple[str, ...],
    metrics: tuple[str, ...],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[tuple(record[key] for key in keys)].append(record)
    output = []
    for group, items in sorted(groups.items()):
        row = {key: value for key, value in zip(keys, group)}
        row["trials"] = len(items)
        for metric in metrics:
            average, interval = _mean_ci(float(item[metric]) for item in items)
            row[f"{metric}_mean"] = average
            row[f"{metric}_ci95"] = interval
        output.append(row)
    return output


def _scaling_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _summarize(
        records,
        ("method", "size"),
        (
            "best_k_ratio",
            "critic_selected_ratio",
            "utility_selected_ratio",
            "epsilon_coverage",
            "p_epsilon",
            "raw_feasible_rate",
            "diversity",
            "proposal_latency_ms",
        ),
    )


def _future_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _summarize(
        records,
        ("policy", "n_jobs"),
        (
            "reward_ratio",
            "missed_value",
            "deadline_compliance",
            "future_result_utilization",
            "eligible_result_utilization",
            "raw_feasible_rate",
            "post_repair_feasible_rate",
            "selected_identity_survival_rate",
            "mean_stable_pool_size",
            "projected_best_k_ratio",
            "projected_p_epsilon",
            "mean_observed_latency_ms",
            "mean_fallback_latency_ms",
            "shots_total",
        ),
    )


def _find(
    summary: list[dict[str, Any]],
    **matching: Any,
) -> dict[str, Any]:
    return next(
        row
        for row in summary
        if all(row[key] == value for key, value in matching.items())
    )


def _build_gates(
    *,
    scaling: list[dict[str, Any]],
    future: list[dict[str, Any]],
    latency: dict[str, Any],
    conditional_gate: dict[str, Any],
    config: BacklogBenchmarkConfig,
) -> dict[str, Any]:
    scaled = [row for row in scaling if row["method"] == "scale_aware"]
    proposal_scaling_mean = min(
        row["best_k_ratio_mean"] for row in scaled
    ) >= 0.90
    proposal_scaling_ci = min(
        row["best_k_ratio_mean"] - row["best_k_ratio_ci95"]
        for row in scaled
    ) >= 0.90
    deployable_critic_mean = min(
        row["critic_selected_ratio_mean"] for row in scaled
    ) >= 0.90
    deployable_critic_ci = min(
        row["critic_selected_ratio_mean"]
        - row["critic_selected_ratio_ci95"]
        for row in scaled
    ) >= 0.90
    future_scaled = [
        row
        for row in future
        if row["policy"] == "future_rydberg_scale_aware"
    ]
    future_reward = min(
        row["reward_ratio_mean"] - row["reward_ratio_ci95"]
        for row in future_scaled
    ) >= 0.90
    identity = min(
        row["selected_identity_survival_rate_mean"]
        - row["selected_identity_survival_rate_ci95"]
        for row in future_scaled
    ) >= 0.80
    utilization = min(
        row["future_result_utilization_mean"]
        - row["future_result_utilization_ci95"]
        for row in future_scaled
    ) >= 0.10
    safety = min(
        row["post_repair_feasible_rate_mean"] for row in future_scaled
    ) >= 1.0 - 1e-12
    deadline = latency["deadline_compliance"] >= 0.95
    algorithmic_pass = proposal_scaling_ci and deployable_critic_ci
    asynchronous_pass = future_reward and identity and utilization and safety
    physical_pass = (
        algorithmic_pass
        and asynchronous_pass
        and deadline
        and bool(latency["measured_qpu"])
        and bool(conditional_gate["calibration_transfer_pass"])
        and bool(conditional_gate["manual_quality_pass"])
    )
    return {
        "algorithmic_pass": algorithmic_pass,
        "asynchronous_pass": asynchronous_pass,
        "physical_pass": physical_pass,
        "checks": {
            "best_k_mean_at_least_0_90": proposal_scaling_mean,
            "best_k_lower_ci_at_least_0_90": proposal_scaling_ci,
            "critic_mean_at_least_0_90": deployable_critic_mean,
            "critic_lower_ci_at_least_0_90": deployable_critic_ci,
            "future_reward_at_least_0_90": future_reward,
            "identity_survival_at_least_0_80": identity,
            "future_utilization_at_least_0_10": utilization,
            "post_repair_safety_equals_1": safety,
            "deadline_compliance_at_least_0_95": deadline,
            "measured_qpu_latency": bool(latency["measured_qpu"]),
            "calibration_transfer": bool(
                conditional_gate["calibration_transfer_pass"]
            ),
            "manual_backend_quality": bool(
                conditional_gate["manual_quality_pass"]
            ),
        },
        "thresholds": {
            "best_k_ratio_lower_ci": 0.90,
            "critic_selected_ratio_lower_ci": 0.90,
            "future_reward_ratio": 0.90,
            "identity_survival": 0.80,
            "future_result_utilization": 0.10,
            "post_repair_feasibility": 1.0,
            "deadline_compliance": 0.95,
            "future_deadline_ms": config.future_deadline_ms,
        },
    }


def _fmt(mean_value: float, interval: float) -> str:
    return f"{mean_value:.3f} +/- {interval:.3f}"


def render_backlog_report(results: dict[str, Any]) -> str:
    """Render a concise, auditable Markdown result report."""

    config = results["config"]
    selected = results["selected_regime"]
    scaling = results["scaling_summary"]
    future = results["future_summary"]
    latency = results["latency_summary"]
    gates = results["gates"]
    lines = [
        "# Scale-aware stable-backlog dispatch report",
        "",
        "## Claim boundary",
        "",
        "This experiment improves and evaluates a classical Rydberg-blockade "
        "surrogate. It does not establish neutral-atom hardware performance. "
        "The selected encoding must still transfer to dense, QuTiP, manual, "
        "and eventually measured-QPU executions.",
        "",
        f"**Algorithmic gate: {'PASS' if gates['algorithmic_pass'] else 'HOLD'}.**",
        f"**Asynchronous gate: {'PASS' if gates['asynchronous_pass'] else 'HOLD'}.**",
        f"**Physical gate: {'PASS' if gates['physical_pass'] else 'HOLD'}.**",
        "",
        "## Protocol",
        "",
        f"- K = {config['candidate_budget']} with no K increase.",
        f"- Reward-only multi-size training episodes = "
        f"{config['training_episodes']} (seed {config['training_seed']}).",
        f"- Sizes = {config['sizes']} and confirmation seeds = "
        f"{config['confirmation_seeds']} per size.",
        f"- Future deadline = {config['future_deadline_ms']:.0f} ms; "
        f"step duration = {config['decision_step_ms']:.0f} ms.",
        f"- Stable backlog guard = {config['stable_guard_steps']} step(s).",
        f"- Quantum target block = top {config['stable_target_fraction']:.0%} "
        f"of jobs, capped at {config['maximum_stable_target_jobs']} stable jobs.",
        f"- Selected on disjoint seeds: `{selected['name']}` "
        f"({selected['utility_encoding']}, gain={selected['detuning_gain']}, "
        f"schedule={selected['pulse_schedule']}).",
        "",
        "## Held-out best-of-K scaling",
        "",
        "Best-of-K is a diagnostic upper bound: it uses realized reward to "
        "identify the best sampled candidate. Critic and utility columns are "
        "deployable rerankers and do not see the oracle.",
        "",
        "| method | n | best-of-16 ratio | critic ratio | utility ratio | "
        "eps-5% coverage | candidate p(eps) | raw feasible | local ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in scaling:
        lines.append(
            "| {method} | {size} | {best} | {critic} | {utility} | "
            "{coverage} | {probability} | {feasible} | {latency} |".format(
                method=row["method"],
                size=row["size"],
                best=_fmt(
                    row["best_k_ratio_mean"],
                    row["best_k_ratio_ci95"],
                ),
                critic=_fmt(
                    row["critic_selected_ratio_mean"],
                    row["critic_selected_ratio_ci95"],
                ),
                utility=_fmt(
                    row["utility_selected_ratio_mean"],
                    row["utility_selected_ratio_ci95"],
                ),
                coverage=_fmt(
                    row["epsilon_coverage_mean"],
                    row["epsilon_coverage_ci95"],
                ),
                probability=_fmt(
                    row["p_epsilon_mean"],
                    row["p_epsilon_ci95"],
                ),
                feasible=_fmt(
                    row["raw_feasible_rate_mean"],
                    row["raw_feasible_rate_ci95"],
                ),
                latency=f"{row['proposal_latency_ms_mean']:.2f}",
            )
        )
    lines.extend(
        [
            "",
            "## Stable-backlog future-batch rollouts",
            "",
            "Beam handles unreserved jobs immediately. A future planner sees "
            "only jobs guaranteed to survive the six-step deadline plus guard. "
            "The highest-priority bounded target block is reserved until the "
            "result arrives or expires, after which persistent IDs are remapped "
            "and repaired.",
            "",
            "| policy | n | return/ref. | future best-K | identity survival | "
            "utilization | deadline | post-repair | stable jobs | shots |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in future:
        lines.append(
            "| {policy} | {size} | {reward} | {best} | {identity} | "
            "{use} | {deadline} | {safe} | {stable} | {shots:.0f} |".format(
                policy=row["policy"],
                size=row["n_jobs"],
                reward=_fmt(row["reward_ratio_mean"], row["reward_ratio_ci95"]),
                best=_fmt(
                    row["projected_best_k_ratio_mean"],
                    row["projected_best_k_ratio_ci95"],
                ),
                identity=_fmt(
                    row["selected_identity_survival_rate_mean"],
                    row["selected_identity_survival_rate_ci95"],
                ),
                use=_fmt(
                    row["future_result_utilization_mean"],
                    row["future_result_utilization_ci95"],
                ),
                deadline=_fmt(
                    row["deadline_compliance_mean"],
                    row["deadline_compliance_ci95"],
                ),
                safe=f"{row['post_repair_feasible_rate_mean']:.3f}",
                stable=f"{row['mean_stable_pool_size_mean']:.1f}",
                shots=row["shots_total_mean"],
            )
        )
    lines.extend(
        [
            "",
            "## Latency evidence",
            "",
            f"The trace source is `{latency['source_kind']}` and measured QPU "
            f"evidence is **{latency['measured_qpu']}**. Mean/p95/p99 total "
            f"latency is {latency['total_mean_ms']:.1f}/"
            f"{latency['total_p95_ms']:.1f}/{latency['total_p99_ms']:.1f} ms. "
            f"Compliance with the {config['future_deadline_ms']:.0f} ms future "
            f"deadline is {latency['deadline_compliance']:.1%}.",
            "",
            "## Gates",
            "",
            "| check | pass |",
            "|---|---:|",
        ]
    )
    for name, value in gates["checks"].items():
        lines.append(f"| {name} | {value} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The report shows both nominal means and lower-95%-confidence gates. "
            "A best-of-K mean pass is not a statistically secure scaling pass "
            "when its lower bound misses 0.90. It is also insufficient if the "
            "learned critic cannot find that candidate. Stable-backlog "
            "utilization demonstrates "
            "that delayed results can enter the pipeline, but the reservation "
            "cost must still preserve end-to-end return. Physical advancement "
            "requires the unchanged calibration and measured-latency gates.",
            "",
        ]
    )
    return "\n".join(lines)


def run_backlog_benchmark(
    *,
    model: ActorCriticModel,
    baseline_model: ActorCriticModel,
    conditional_gate: dict[str, Any],
    trace: LatencyTrace,
    config: BacklogBenchmarkConfig,
    output_json: Path | None = None,
    output_report: Path | None = None,
) -> dict[str, Any]:
    """Select on training seeds, confirm once, and run future-batch rollouts."""

    selection_records = _selection_records(
        model,
        baseline_model,
        config,
    )
    selected = _select_regime(selection_records)
    confirmation_records = _confirmation_records(
        model=model,
        baseline_model=baseline_model,
        selected_regime=selected,
        config=config,
    )
    future_records = _future_records(
        model=model,
        selected_regime=selected,
        trace=trace,
        config=config,
    )
    scaling_summary = _scaling_summary(confirmation_records)
    future_summary = _future_summary(future_records)
    latency_summary = summarize_latency(trace, config.future_deadline_ms)
    gates = _build_gates(
        scaling=scaling_summary,
        future=future_summary,
        latency=latency_summary,
        conditional_gate=conditional_gate,
        config=config,
    )
    results = {
        "schema_version": 1,
        "study": "scale_aware_stable_backlog_v1",
        "claim_boundary": (
            "Classical surrogate trend only; physical claims require measured "
            "QPU latency and unchanged calibration-transfer gates."
        ),
        "config": asdict(config),
        "selected_regime": asdict(selected),
        "trained_model": model.to_dict(),
        "selection_records": selection_records,
        "confirmation_records": confirmation_records,
        "scaling_summary": scaling_summary,
        "future_records": future_records,
        "future_summary": future_summary,
        "latency_summary": latency_summary,
        "retained_conditional_gate": conditional_gate,
        "gates": gates,
    }
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(results, indent=2) + "\n",
            encoding="utf-8",
        )
    if output_report is not None:
        output_report.parent.mkdir(parents=True, exist_ok=True)
        output_report.write_text(
            render_backlog_report(results),
            encoding="utf-8",
        )
    return results
