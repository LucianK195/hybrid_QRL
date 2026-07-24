"""Reproducible scaling and robustness study for dynamic dispatch.

The default study trains one reward-only actor-critic, freezes it, and then
uses disjoint seeds for three evaluations:

* equal-K and equal-end-to-end-latency comparisons at 20, 40, 60, and 100
  binary decisions;
* dynamic multi-step returns for all candidate generators; and
* one-factor-at-a-time robustness sweeps over density, geometry error, readout
  noise, utility distribution, pulse schedule, cache precision, and combined
  distribution shifts.

The one-step reference is a time-limited MILP that directly maximizes realized
dispatch reward.  Candidate methods instead receive learned actor priorities
and are reranked by the learned action critic.  Keeping these roles separate
prevents the policy from learning through oracle labels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import numpy as np

from ..core import Action
from ..utilities.reports.dispatch import render_dispatch_report
from ..utilities.results import ResultWriter
from .baselines import (
    METHODS,
    ProposalConfig,
    generate_candidates,
    proposal_weights,
    solve_weighted_independent_set,
)
from .environment import DispatchConfig, DispatchEnvironment, DispatchState
from .learning import ActorCriticModel, TrainingConfig, train_actor_critic


@dataclass(frozen=True)
class BenchmarkConfig:
    """Top-level experiment settings and held-out seed policy."""

    seeds: int = 20
    train_episodes: int = 320
    sizes: tuple[int, ...] = (20, 40, 60, 100)
    candidate_budget: int = 16
    latency_budget_ms: float = 20.0
    oracle_time_limit_ms: float = 500.0
    warmup_steps: int = 4
    rollout_horizon: int = 12
    seed: int = 2026

    def __post_init__(self) -> None:
        if self.seeds < 20:
            raise ValueError("the research benchmark requires at least 20 seeds")
        if self.train_episodes <= 0:
            raise ValueError("train_episodes must be positive")
        if any(size < 20 or size > 100 for size in self.sizes):
            raise ValueError("all benchmark sizes must lie in [20, 100]")


def realized_step_reward(
    state: DispatchState, action: Action, miss_penalty: float
) -> float:
    """Evaluate the exact immediate reward without mutating an environment."""

    selected = np.asarray(action, dtype=bool)
    completion = np.sum(
        state.values[selected]
        * (1.0 + state.ages[selected] / np.maximum(state.deadlines[selected], 1))
    )
    expires_now = (~selected) & (state.remaining <= 1)
    missed = np.sum(state.values[expires_now])
    return float((completion - miss_penalty * missed) / state.n_jobs)


def oracle_weights(state: DispatchState, miss_penalty: float) -> np.ndarray:
    """Linear weights whose MWIS maximizes exact immediate environment reward."""

    completion = state.values * (
        1.0 + state.ages / np.maximum(state.deadlines, 1)
    )
    avoided_miss = miss_penalty * state.values * (state.remaining <= 1)
    return completion + avoided_miss


def _held_out_state(
    model: ActorCriticModel,
    environment_config: DispatchConfig,
    seed: int,
    warmup_steps: int,
) -> DispatchState:
    environment = DispatchEnvironment(environment_config, seed=seed)
    rng = np.random.default_rng(seed + 7_919)
    state = environment.state()
    for _ in range(warmup_steps):
        action = model.actor.sample(state, rng)
        state, _, done, _ = environment.step(action)
        if done:
            break
    return state


def _mean_pairwise_hamming(actions: tuple[Action, ...]) -> float:
    if len(actions) < 2:
        return 0.0
    values: list[float] = []
    for left in range(len(actions)):
        for right in range(left + 1, len(actions)):
            values.append(
                float(
                    np.mean(
                        np.asarray(actions[left]) != np.asarray(actions[right])
                    )
                )
            )
    return float(np.mean(values))


def _evaluate_method(
    *,
    method: str,
    state: DispatchState,
    model: ActorCriticModel,
    proposal_config: ProposalConfig,
    seed: int,
    oracle_action: Action,
    miss_penalty: float,
    metadata: dict[str, Any],
    end_to_end_budget_ms: float | None = None,
) -> dict[str, Any]:
    start = perf_counter()
    batch = generate_candidates(
        method,
        state,
        model,
        proposal_config,
        np.random.default_rng(seed),
    )
    if batch.actions:
        chosen = model.best_action(state, list(batch.actions))
        fallback = False
    else:
        chosen = tuple(0 for _ in range(state.n_jobs))
        fallback = True
    total_latency = (perf_counter() - start) * 1_000.0
    reward = realized_step_reward(state, chosen, miss_penalty)
    reference = realized_step_reward(state, oracle_action, miss_penalty)
    regret = max(0.0, reference - reward)
    scale = max(abs(reference), 1e-9)
    record: dict[str, Any] = {
        **metadata,
        "method": method,
        "raw_generated": batch.raw_generated,
        "raw_feasible_rate": (
            batch.raw_feasible / batch.raw_generated if batch.raw_generated else 0.0
        ),
        "unique_feasible": batch.unique_feasible,
        "candidate_diversity": _mean_pairwise_hamming(batch.actions),
        "fallback": fallback,
        "proposal_latency_ms": batch.elapsed_ms,
        "end_to_end_latency_ms": total_latency,
        "latency_compliant": (
            True
            if end_to_end_budget_ms is None
            else total_latency <= 1.05 * end_to_end_budget_ms
        ),
        "critic_value": model.q_value(state, chosen),
        "proposal_objective": float(
            np.asarray(chosen) @ proposal_weights(model, state)
        ),
        "realized_reward": reward,
        "reference_reward": reference,
        "reward_ratio": reward / reference if reference > 1e-9 else 1.0,
        "normalized_regret": regret / scale,
        "optimum_coverage": float(regret <= 1e-8),
        "selected": int(sum(chosen)),
    }
    return record


def _paired_state_records(
    *,
    model: ActorCriticModel,
    benchmark: BenchmarkConfig,
    environment_config: DispatchConfig,
    seed_index: int,
    methods: Iterable[str],
    mode: str,
    proposal_overrides: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    seed_offset: int = 0,
) -> list[dict[str, Any]]:
    held_out_seed = (
        benchmark.seed + 1_000_003 + (seed_offset + seed_index) * 10_007
    )
    state = _held_out_state(
        model,
        environment_config,
        held_out_seed,
        benchmark.warmup_steps,
    )
    oracle = solve_weighted_independent_set(
        state,
        oracle_weights(state, environment_config.miss_penalty),
        benchmark.oracle_time_limit_ms,
    )
    overrides = proposal_overrides or {}
    if mode == "equal_k":
        proposal = ProposalConfig(
            candidates=benchmark.candidate_budget,
            max_runtime_ms=2_000.0,
            **overrides,
        )
        end_to_end_budget = None
    elif mode == "equal_latency":
        # Cap retained candidates and reserve half for repair and Q reranking.
        proposal = ProposalConfig(
            candidates=96,
            latency_ms=0.5 * benchmark.latency_budget_ms,
            **overrides,
        )
        end_to_end_budget = benchmark.latency_budget_ms
    else:
        raise ValueError(f"unknown comparison mode: {mode}")

    common = {
        "seed_index": seed_index,
        "held_out_seed": held_out_seed,
        "mode": mode,
        "n_jobs": environment_config.n_jobs,
        "density": environment_config.density,
        "graph_family": environment_config.graph_family,
        "utility_distribution": environment_config.utility_distribution,
        "candidate_budget": benchmark.candidate_budget if mode == "equal_k" else None,
        "latency_budget_ms": end_to_end_budget,
        "oracle_latency_ms": oracle.elapsed_ms,
        "oracle_exact": oracle.success,
        "oracle_status": oracle.status,
        "oracle_mip_gap": oracle.mip_gap,
        **(metadata or {}),
    }
    output = []
    for method_index, method in enumerate(methods):
        output.append(
            _evaluate_method(
                method=method,
                state=state,
                model=model,
                proposal_config=proposal,
                seed=held_out_seed + 101 * method_index,
                oracle_action=oracle.action,
                miss_penalty=environment_config.miss_penalty,
                metadata=common,
                end_to_end_budget_ms=end_to_end_budget,
            )
        )
    return output


def _run_scaling(
    model: ActorCriticModel,
    benchmark: BenchmarkConfig,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for size in benchmark.sizes:
        environment_config = DispatchConfig(
            n_jobs=size,
            density=0.12,
            horizon=max(benchmark.warmup_steps + 2, 8),
        )
        for seed_index in range(benchmark.seeds):
            for mode in ("equal_k", "equal_latency"):
                records.extend(
                    _paired_state_records(
                        model=model,
                        benchmark=benchmark,
                        environment_config=environment_config,
                        seed_index=seed_index,
                        methods=METHODS,
                        mode=mode,
                        metadata={
                            "study": "scaling",
                            "axis": "size",
                            "level": str(size),
                        },
                    )
                )
    return records


def _robustness_scenarios() -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    for density in (0.05, 0.12, 0.25):
        scenarios.append({"axis": "density", "level": str(density), "density": density})
    for value in (0.0, 0.03, 0.08):
        scenarios.append(
            {
                "axis": "geometry_error",
                "level": str(value),
                "geometry_error": value,
            }
        )
    for value in (0.0, 0.01, 0.05):
        scenarios.append(
            {
                "axis": "readout_noise",
                "level": str(value),
                "readout_noise": value,
            }
        )
    for distribution in ("uniform", "lognormal", "bimodal"):
        scenarios.append(
            {
                "axis": "utility_distribution",
                "level": distribution,
                "utility_distribution": distribution,
            }
        )
    for schedule in ("short", "balanced", "adiabatic"):
        scenarios.append(
            {
                "axis": "pulse_schedule",
                "level": schedule,
                "pulse_schedule": schedule,
            }
        )
    for precision in (None, 0, 1, 2, 4):
        scenarios.append(
            {
                "axis": "cache_precision",
                "level": str(precision),
                "cache_precision": precision,
            }
        )
    shifts = (
        ("in_distribution", "unit_disk", 0.12, "uniform"),
        ("grid", "grid", 0.12, "uniform"),
        ("dense_grid", "grid", 0.25, "uniform"),
        ("combined_shift", "grid", 0.25, "bimodal"),
    )
    for level, family, density, distribution in shifts:
        scenarios.append(
            {
                "axis": "distribution_shift",
                "level": level,
                "graph_family": family,
                "density": density,
                "utility_distribution": distribution,
            }
        )
    return scenarios


def _run_robustness(
    model: ActorCriticModel,
    benchmark: BenchmarkConfig,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    baseline_environment = DispatchConfig(n_jobs=60, density=0.12, horizon=8)
    for scenario in _robustness_scenarios():
        environment = replace(
            baseline_environment,
            density=float(scenario.get("density", baseline_environment.density)),
            graph_family=str(
                scenario.get("graph_family", baseline_environment.graph_family)
            ),
            utility_distribution=str(
                scenario.get(
                    "utility_distribution", baseline_environment.utility_distribution
                )
            ),
        )
        overrides = {
            key: scenario[key]
            for key in (
                "geometry_error",
                "readout_noise",
                "pulse_schedule",
                "cache_precision",
            )
            if key in scenario
        }
        methods: Iterable[str]
        if scenario["axis"] == "distribution_shift":
            methods = METHODS
        else:
            methods = ("rydberg_surrogate", "autoregressive", "greedy")
        for seed_index in range(benchmark.seeds):
            records.extend(
                _paired_state_records(
                    model=model,
                    benchmark=benchmark,
                    environment_config=environment,
                    seed_index=seed_index,
                    seed_offset=20_000,
                    methods=methods,
                    mode="equal_k",
                    proposal_overrides=overrides,
                    metadata={
                        "study": "robustness",
                        "axis": scenario["axis"],
                        "level": scenario["level"],
                    },
                )
            )
    return records


def _run_dynamic_rollouts(
    model: ActorCriticModel,
    benchmark: BenchmarkConfig,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    environment_config = DispatchConfig(
        n_jobs=40,
        density=0.12,
        horizon=benchmark.rollout_horizon,
    )
    for seed_index in range(benchmark.seeds):
        episode_seed = benchmark.seed + 3_000_017 + 1_009 * seed_index
        for method_index, method in enumerate(METHODS):
            environment = DispatchEnvironment(environment_config, seed=episode_seed)
            state = environment.state()
            rng_seed = episode_seed + 97 * method_index
            total_reward = 0.0
            total_missed = 0.0
            total_completed = 0.0
            latencies: list[float] = []
            fallbacks = 0
            done = False
            step = 0
            while not done:
                batch = generate_candidates(
                    method,
                    state,
                    model,
                    ProposalConfig(
                        candidates=benchmark.candidate_budget,
                        max_runtime_ms=75.0,
                    ),
                    np.random.default_rng(rng_seed + step),
                )
                if batch.actions:
                    action = model.best_action(state, list(batch.actions))
                else:
                    action = tuple(0 for _ in range(state.n_jobs))
                    fallbacks += 1
                state, reward, done, info = environment.step(action)
                total_reward += reward
                total_missed += info["missed_value"]
                total_completed += info["completion_value"]
                latencies.append(batch.elapsed_ms)
                step += 1
            output.append(
                {
                    "study": "dynamic_rollout",
                    "seed_index": seed_index,
                    "method": method,
                    "n_jobs": environment_config.n_jobs,
                    "density": environment_config.density,
                    "horizon": benchmark.rollout_horizon,
                    "candidate_budget": benchmark.candidate_budget,
                    "episode_return": total_reward,
                    "completion_value": total_completed,
                    "missed_value": total_missed,
                    "mean_decision_latency_ms": float(np.mean(latencies)),
                    "fallback_steps": fallbacks,
                }
            )
    return output


def run_benchmark(
    config: BenchmarkConfig,
    output_json: Path,
    output_report: Path,
) -> dict[str, Any]:
    """Train, evaluate, serialize raw records, and write a Markdown report."""

    training_config = TrainingConfig(
        episodes=config.train_episodes,
        horizon=24,
        seed=config.seed,
    )
    model, history = train_actor_critic(training_config)
    scaling = _run_scaling(model, config)
    rollouts = _run_dynamic_rollouts(model, config)
    robustness = _run_robustness(model, config)
    results = {
        "schema_version": 1,
        "config": asdict(config),
        "training_config": asdict(training_config),
        "model": model.to_dict(),
        "training_history": history,
        "scaling_records": scaling,
        "rollout_records": rollouts,
        "robustness_records": robustness,
    }
    ResultWriter(json_trailing_newline=False).artifacts(
        json_path=output_json,
        report_path=output_report,
        payload=results,
        render_report=render_dispatch_report,
    )
    return results
