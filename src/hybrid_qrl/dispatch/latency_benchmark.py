"""Latency-aware extension of the synthetic dynamic dispatch benchmark.

The original benchmark measures local proposal and reranking time while every
action is applied immediately. This module adds an application clock, a
timestamped QPU-latency trace, job identities, delayed execution, stale-action
repair, and asynchronous classical fallback policies.

Real hardware evidence is accepted only when latency observations are labelled
``measured_qpu`` and contain submission, start, completion, and retrieval
timestamps. The built-in stress trace is deterministic and useful for testing
the control architecture, but it deliberately fails the hardware-evidence gate.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
from math import ceil, sqrt
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import numpy as np

from ..core import Action
from ..utilities.metrics import shots_for_95_percent
from ..utilities.reports.latency import render_latency_report
from ..utilities.results import ResultWriter
from .baselines import (
    ProposalConfig,
    generate_candidates,
    proposal_weights,
    repair_action,
    solve_weighted_independent_set,
)
from .benchmark import oracle_weights, realized_step_reward
from .environment import DispatchConfig, DispatchEnvironment, DispatchState
from .learning import ActorCriticModel


POLICIES = (
    "beam_immediate",
    "greedy_immediate",
    "quantum_delayed",
    "async_beam_quantum",
    "async_greedy_quantum",
)


@dataclass(frozen=True)
class LatencyObservation:
    """One timestamped remote task observation in milliseconds."""

    request_id: str
    submitted_at_ms: float
    started_at_ms: float
    completed_at_ms: float
    retrieved_at_ms: float
    shots: int

    def __post_init__(self) -> None:
        timestamps = (
            self.submitted_at_ms,
            self.started_at_ms,
            self.completed_at_ms,
            self.retrieved_at_ms,
        )
        if not all(np.isfinite(value) for value in timestamps):
            raise ValueError("latency timestamps must be finite")
        if not (
            self.submitted_at_ms
            <= self.started_at_ms
            <= self.completed_at_ms
            <= self.retrieved_at_ms
        ):
            raise ValueError("latency timestamps must be monotonic")
        if self.shots <= 0:
            raise ValueError("shots must be positive")

    @property
    def queue_ms(self) -> float:
        """Time from task submission until device processing starts."""

        return self.started_at_ms - self.submitted_at_ms

    @property
    def execution_ms(self) -> float:
        """Provider-side preparation, shots, measurement, and finalization time."""

        return self.completed_at_ms - self.started_at_ms

    @property
    def retrieval_ms(self) -> float:
        """Time from task completion until the result becomes available."""

        return self.retrieved_at_ms - self.completed_at_ms

    @property
    def total_ms(self) -> float:
        """Submission-to-retrieval end-to-end latency."""

        return self.retrieved_at_ms - self.submitted_at_ms


@dataclass(frozen=True)
class LatencyTrace:
    """Reusable distribution of timestamped task observations."""

    source_kind: str
    source_name: str
    device: str
    observations: tuple[LatencyObservation, ...]

    def __post_init__(self) -> None:
        if self.source_kind not in {"measured_qpu", "synthetic_stress"}:
            raise ValueError("source_kind must be measured_qpu or synthetic_stress")
        if not self.observations:
            raise ValueError("latency trace must contain at least one observation")

    @property
    def is_measured_qpu(self) -> bool:
        """Whether this trace is eligible to support a physical-latency claim."""

        return self.source_kind == "measured_qpu"

    def observation(self, index: int) -> LatencyObservation:
        """Return one observation, cycling deterministically when necessary."""

        return self.observations[index % len(self.observations)]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable trace payload."""

        return {
            "schema_version": 1,
            "source_kind": self.source_kind,
            "source_name": self.source_name,
            "device": self.device,
            "observations": [asdict(item) for item in self.observations],
        }

    @classmethod
    def from_json(cls, path: Path) -> "LatencyTrace":
        """Load and validate timestamped observations from JSON."""

        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            source_kind=str(payload["source_kind"]),
            source_name=str(payload.get("source_name", path.name)),
            device=str(payload.get("device", "unspecified")),
            observations=tuple(
                LatencyObservation(**item) for item in payload["observations"]
            ),
        )


@dataclass(frozen=True)
class LatencyAwareConfig:
    """Preregistered timing and evaluation settings."""

    seeds: int = 20
    n_jobs: int = 40
    density: float = 0.12
    horizon: int = 18
    candidate_budget: int = 16
    decision_step_ms: float = 1_000.0
    quantum_deadline_ms: float = 3_000.0
    fallback_budget_ms: float = 50.0
    oracle_time_limit_ms: float = 250.0
    minimum_deadline_compliance: float = 0.95
    minimum_reward_ratio: float = 0.90
    minimum_quantum_utilization: float = 0.05
    epsilon: float = 0.05
    seed: int = 31_337

    def __post_init__(self) -> None:
        if self.seeds <= 0 or self.horizon < 4:
            raise ValueError("seeds must be positive and horizon must be at least four")
        if not 8 <= self.n_jobs <= 100:
            raise ValueError("n_jobs must lie in [8, 100]")
        if self.candidate_budget <= 0:
            raise ValueError("candidate_budget must be positive")
        if self.decision_step_ms <= 0 or self.quantum_deadline_ms <= 0:
            raise ValueError("physical timing values must be positive")
        if not 0.0 < self.minimum_deadline_compliance <= 1.0:
            raise ValueError("minimum_deadline_compliance must lie in (0, 1]")


@dataclass(frozen=True)
class PendingQuantumRequest:
    """Quantum candidate batch represented by persistent job identities."""

    request_id: str
    issue_step: int
    arrival_step: int
    deadline_step: int
    observation: LatencyObservation
    candidate_job_ids: tuple[tuple[int, ...], ...]
    raw_generated: int
    raw_feasible: int

    @property
    def deadline_met(self) -> bool:
        """Whether the recorded result was retrieved before its deadline."""

        return self.arrival_step <= self.deadline_step


def make_preregistered_stress_trace(
    *,
    count: int = 1_024,
    seed: int = 9_031,
    shots: int = 100,
) -> LatencyTrace:
    """Create the deterministic non-hardware latency stress distribution.

    Queue, provider-side execution, and retrieval components are sampled before
    held-out dispatch seeds are evaluated. The values are scenario assumptions,
    not measurements of QuEra, Amazon Braket, or another physical QPU.
    """

    if count <= 0:
        raise ValueError("count must be positive")
    rng = np.random.default_rng(seed)
    queue = rng.lognormal(mean=np.log(1_150.0), sigma=0.72, size=count)
    execution = np.maximum(80.0, rng.normal(loc=520.0, scale=110.0, size=count))
    retrieval = rng.lognormal(mean=np.log(95.0), sigma=0.45, size=count)
    observations = []
    for index in range(count):
        started = float(queue[index])
        completed = started + float(execution[index])
        retrieved = completed + float(retrieval[index])
        observations.append(
            LatencyObservation(
                request_id=f"stress-{index:05d}",
                submitted_at_ms=0.0,
                started_at_ms=started,
                completed_at_ms=completed,
                retrieved_at_ms=retrieved,
                shots=shots,
            )
        )
    return LatencyTrace(
        source_kind="synthetic_stress",
        source_name="preregistered-lognormal-queue-v1",
        device="not-a-physical-device",
        observations=tuple(observations),
    )


def write_latency_trace(trace: LatencyTrace, path: Path) -> None:
    """Persist a latency trace for audit and exact replay."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trace.to_dict(), indent=2) + "\n", encoding="utf-8")


def _percentile(values: Iterable[float], percentile: float) -> float:
    items = np.asarray(list(values), dtype=float)
    return float(np.percentile(items, percentile))


def summarize_latency(
    trace: LatencyTrace,
    deadline_ms: float,
) -> dict[str, Any]:
    """Summarize observed latency, queue, execution, retrieval, and shots."""

    observations = trace.observations
    total = [item.total_ms for item in observations]
    queue = [item.queue_ms for item in observations]
    execution = [item.execution_ms for item in observations]
    retrieval = [item.retrieval_ms for item in observations]
    canonical = json.dumps(
        trace.to_dict(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "source_kind": trace.source_kind,
        "source_name": trace.source_name,
        "device": trace.device,
        "trace_sha256": sha256(canonical).hexdigest(),
        "observations": len(observations),
        "measured_qpu": trace.is_measured_qpu,
        "deadline_ms": deadline_ms,
        "deadline_compliance": float(np.mean(np.asarray(total) <= deadline_ms)),
        "deadline_misses": int(np.count_nonzero(np.asarray(total) > deadline_ms)),
        "total_mean_ms": mean(total),
        "total_p50_ms": _percentile(total, 50),
        "total_p95_ms": _percentile(total, 95),
        "total_p99_ms": _percentile(total, 99),
        "queue_mean_ms": mean(queue),
        "queue_p95_ms": _percentile(queue, 95),
        "queue_p99_ms": _percentile(queue, 99),
        "execution_mean_ms": mean(execution),
        "execution_p95_ms": _percentile(execution, 95),
        "execution_p99_ms": _percentile(execution, 99),
        "retrieval_mean_ms": mean(retrieval),
        "retrieval_p95_ms": _percentile(retrieval, 95),
        "retrieval_p99_ms": _percentile(retrieval, 99),
        "shots_mean": mean(item.shots for item in observations),
        "shots_total": int(sum(item.shots for item in observations)),
    }


def remap_candidate_job_ids(
    candidate_job_ids: Iterable[int],
    state: DispatchState,
) -> Action:
    """Map a stale candidate to current slots, dropping replaced jobs."""

    selected = {int(job_id) for job_id in candidate_job_ids}
    return tuple(int(int(job_id) in selected) for job_id in state.job_ids)


def _job_id_candidates(
    actions: Iterable[Action],
    state: DispatchState,
) -> tuple[tuple[int, ...], ...]:
    output = []
    for action in actions:
        output.append(
            tuple(
                int(state.job_ids[node])
                for node, bit in enumerate(action)
                if bit
            )
        )
    return tuple(output)


def _issue_quantum_request(
    *,
    state: DispatchState,
    model: ActorCriticModel,
    observation: LatencyObservation,
    request_index: int,
    step: int,
    config: LatencyAwareConfig,
    seed: int,
) -> PendingQuantumRequest:
    """Generate a surrogate batch and attach recorded hardware timing."""

    batch = generate_candidates(
        "rydberg_surrogate",
        state,
        model,
        ProposalConfig(
            candidates=config.candidate_budget,
            max_runtime_ms=2_000.0,
            geometry_error=0.06,
            blockade_radius_scale=1.0,
            pulse_schedule="balanced",
            cache_precision=2,
        ),
        np.random.default_rng(seed),
    )
    arrival_steps = max(1, int(ceil(observation.total_ms / config.decision_step_ms)))
    deadline_steps = max(
        1, int(ceil(config.quantum_deadline_ms / config.decision_step_ms))
    )
    return PendingQuantumRequest(
        request_id=f"episode-{seed}-request-{request_index}",
        issue_step=step,
        arrival_step=step + arrival_steps,
        deadline_step=step + deadline_steps,
        observation=observation,
        candidate_job_ids=_job_id_candidates(batch.repaired_actions, state),
        raw_generated=batch.raw_generated,
        raw_feasible=batch.raw_feasible,
    )


def _arrival_candidates(
    request: PendingQuantumRequest,
    state: DispatchState,
    model: ActorCriticModel,
    reference_reward: float,
    epsilon: float,
) -> tuple[list[Action], dict[str, float]]:
    """Remap, repair, and evaluate every candidate at its arrival state."""

    weights = proposal_weights(model, state)
    repaired: list[Action] = []
    stale_raw_feasible = 0
    post_repair_feasible = 0
    repair_changes = 0
    epsilon_hits = 0
    original_selected = 0
    surviving_selected = 0
    for identities in request.candidate_job_ids:
        raw = remap_candidate_job_ids(identities, state)
        original_selected += len(identities)
        surviving_selected += sum(raw)
        stale_raw_feasible += int(state.graph.is_feasible(raw))
        safe = repair_action(raw, state.graph, weights)
        repair_changes += int(safe != raw)
        post_repair_feasible += int(state.graph.is_feasible(safe))
        repaired.append(safe)
        ratio = realized_step_reward(state, safe, miss_penalty=1.0) / max(
            reference_reward, 1e-12
        )
        epsilon_hits += int(ratio >= 1.0 - epsilon)
    unique = list(dict.fromkeys(repaired))
    count = max(len(repaired), 1)
    return unique, {
        "stale_raw_feasible": float(stale_raw_feasible),
        "post_repair_feasible": float(post_repair_feasible),
        "repair_changes": float(repair_changes),
        "arrival_candidates": float(len(repaired)),
        "epsilon_candidate_hits": float(epsilon_hits),
        "epsilon_batch_hit": float(epsilon_hits > 0),
        "original_selected": float(original_selected),
        "surviving_selected": float(surviving_selected),
        "stale_raw_feasible_rate": stale_raw_feasible / count,
        "post_repair_feasible_rate": post_repair_feasible / count,
    }


def _immediate_action(
    method: str,
    state: DispatchState,
    model: ActorCriticModel,
    config: LatencyAwareConfig,
    seed: int,
) -> tuple[Action, float]:
    """Generate and rerank one immediate classical fallback batch."""

    batch = generate_candidates(
        method,
        state,
        model,
        ProposalConfig(
            candidates=config.candidate_budget,
            latency_ms=config.fallback_budget_ms,
        ),
        np.random.default_rng(seed),
    )
    action = (
        model.best_action(state, list(batch.actions))
        if batch.actions
        else tuple(0 for _ in range(state.n_jobs))
    )
    return action, batch.elapsed_ms


def _oracle_action(
    state: DispatchState,
    config: LatencyAwareConfig,
) -> tuple[Action, float, bool]:
    solution = solve_weighted_independent_set(
        state,
        oracle_weights(state, miss_penalty=1.0),
        time_limit_ms=config.oracle_time_limit_ms,
    )
    reward = realized_step_reward(state, solution.action, miss_penalty=1.0)
    return solution.action, reward, solution.success


def _reference_episode(
    environment_config: DispatchConfig,
    seed: int,
    config: LatencyAwareConfig,
) -> tuple[float, float]:
    environment = DispatchEnvironment(environment_config, seed=seed)
    state = environment.state()
    total = 0.0
    exact = 0
    done = False
    while not done:
        action, _, success = _oracle_action(state, config)
        exact += int(success)
        state, reward, done, _ = environment.step(action)
        total += reward
    return total, exact / environment_config.horizon


def _run_policy_episode(
    *,
    policy: str,
    model: ActorCriticModel,
    trace: LatencyTrace,
    environment_config: DispatchConfig,
    episode_seed: int,
    seed_index: int,
    reference_return: float,
    config: LatencyAwareConfig,
) -> dict[str, Any]:
    """Execute one immediate, delayed, or asynchronous policy episode."""

    if policy not in POLICIES:
        raise ValueError(f"unknown policy: {policy}")
    environment = DispatchEnvironment(environment_config, seed=episode_seed)
    state = environment.state()
    pending: PendingQuantumRequest | None = None
    request_count = 0
    deadline_misses = 0
    requests_issued = 0
    results_arrived = 0
    results_used = 0
    results_eligible = 0
    censored_requests = 0
    raw_generated = 0
    raw_feasible = 0
    arrival_candidates = 0.0
    stale_raw_feasible = 0.0
    post_repair_feasible = 0.0
    repair_changes = 0.0
    original_selected = 0.0
    surviving_selected = 0.0
    epsilon_candidate_hits = 0.0
    epsilon_batch_hits = 0.0
    overall_epsilon_hits = 0
    total_reward = 0.0
    total_completed = 0.0
    total_missed = 0.0
    total_shots = 0
    staleness_steps: list[int] = []
    fallback_latencies: list[float] = []
    observed_total_latency: list[float] = []
    observed_queue_latency: list[float] = []
    oracle_exact_steps = 0
    step = 0
    done = False

    while not done:
        _, current_reference, exact = _oracle_action(state, config)
        oracle_exact_steps += int(exact)
        quantum_actions: list[Action] = []
        if pending is not None:
            if pending.deadline_met and pending.arrival_step <= step:
                results_arrived += 1
                results_eligible += 1
                quantum_actions, metrics = _arrival_candidates(
                    pending,
                    state,
                    model,
                    current_reference,
                    config.epsilon,
                )
                arrival_candidates += metrics["arrival_candidates"]
                stale_raw_feasible += metrics["stale_raw_feasible"]
                post_repair_feasible += metrics["post_repair_feasible"]
                repair_changes += metrics["repair_changes"]
                original_selected += metrics["original_selected"]
                surviving_selected += metrics["surviving_selected"]
                epsilon_candidate_hits += metrics["epsilon_candidate_hits"]
                epsilon_batch_hits += metrics["epsilon_batch_hit"]
                staleness_steps.append(step - pending.issue_step)
                pending = None
            elif not pending.deadline_met and pending.deadline_step <= step:
                pending = None

        immediate_method = None
        if policy in {"beam_immediate", "async_beam_quantum"}:
            immediate_method = "beam_search"
        elif policy in {"greedy_immediate", "async_greedy_quantum"}:
            immediate_method = "greedy"

        if immediate_method is not None:
            fallback, fallback_latency = _immediate_action(
                immediate_method,
                state,
                model,
                config,
                episode_seed
                + 10_003 * step
                + int(immediate_method == "greedy"),
            )
            fallback_latencies.append(fallback_latency)
        else:
            fallback = tuple(0 for _ in range(state.n_jobs))

        if policy.startswith("async_") and quantum_actions:
            quantum_best = model.best_action(state, quantum_actions)
            action = model.best_action(state, [fallback, quantum_best])
            if action == quantum_best:
                results_used += 1
        elif policy == "quantum_delayed" and quantum_actions:
            action = model.best_action(state, quantum_actions)
            results_used += 1
        else:
            action = fallback

        ratio = realized_step_reward(state, action, 1.0) / max(
            current_reference, 1e-12
        )
        overall_epsilon_hits += int(ratio >= 1.0 - config.epsilon)

        uses_quantum = policy in {
            "quantum_delayed",
            "async_beam_quantum",
            "async_greedy_quantum",
        }
        deadline_steps = int(ceil(config.quantum_deadline_ms / config.decision_step_ms))
        can_resolve = step + deadline_steps < config.horizon
        if uses_quantum and pending is None and can_resolve:
            trace_index = seed_index * config.horizon + request_count
            observation = trace.observation(trace_index)
            pending = _issue_quantum_request(
                state=state,
                model=model,
                observation=observation,
                request_index=request_count,
                step=step,
                config=config,
                seed=episode_seed + 70_001 + request_count,
            )
            request_count += 1
            requests_issued += 1
            raw_generated += pending.raw_generated
            raw_feasible += pending.raw_feasible
            total_shots += observation.shots
            observed_total_latency.append(observation.total_ms)
            observed_queue_latency.append(observation.queue_ms)
            deadline_misses += int(not pending.deadline_met)
        elif uses_quantum and pending is None and not can_resolve:
            censored_requests += 1

        state, reward, done, info = environment.step(action)
        total_reward += reward
        total_completed += info["completion_value"]
        total_missed += info["missed_value"]
        step += 1

    candidate_denominator = max(arrival_candidates, 1.0)
    request_denominator = max(requests_issued, 1)
    eligible_denominator = max(results_eligible, 1)
    p_epsilon = epsilon_candidate_hits / candidate_denominator
    return {
        "study": "latency_aware_rollout",
        "seed_index": seed_index,
        "episode_seed": episode_seed,
        "policy": policy,
        "episode_return": total_reward,
        "reference_return": reference_return,
        "reward_ratio": total_reward / max(reference_return, 1e-12),
        "completion_value": total_completed,
        "missed_value": total_missed,
        "overall_epsilon_hit_rate": overall_epsilon_hits / config.horizon,
        "quantum_candidate_p_epsilon": p_epsilon,
        "quantum_batch_epsilon_coverage": epsilon_batch_hits / eligible_denominator,
        "quantum_k95": shots_for_95_percent(p_epsilon),
        "requests_issued": requests_issued,
        "deadline_misses": deadline_misses,
        "deadline_miss_rate": deadline_misses / request_denominator,
        "results_arrived": results_arrived,
        "results_eligible": results_eligible,
        "results_used": results_used,
        "quantum_result_utilization": results_used / request_denominator,
        "eligible_result_utilization": results_used / eligible_denominator,
        "censored_request_steps": censored_requests,
        "generation_raw_feasible_rate": raw_feasible / max(raw_generated, 1),
        "stale_raw_feasible_rate": stale_raw_feasible / candidate_denominator,
        "post_repair_feasible_rate": post_repair_feasible / candidate_denominator,
        "repair_change_rate": repair_changes / candidate_denominator,
        "selected_identity_survival_rate": surviving_selected
        / max(original_selected, 1.0),
        "mean_staleness_steps": (
            float(mean(staleness_steps)) if staleness_steps else 0.0
        ),
        "fallback_latency_mean_ms": (
            float(mean(fallback_latencies)) if fallback_latencies else 0.0
        ),
        "observed_total_latency_p95_ms": (
            _percentile(observed_total_latency, 95)
            if observed_total_latency
            else 0.0
        ),
        "observed_total_latency_p99_ms": (
            _percentile(observed_total_latency, 99)
            if observed_total_latency
            else 0.0
        ),
        "observed_queue_p95_ms": (
            _percentile(observed_queue_latency, 95)
            if observed_queue_latency
            else 0.0
        ),
        "shots_total": total_shots,
        "shots_per_request": total_shots / request_denominator,
        "oracle_exact_step_rate": oracle_exact_steps / config.horizon,
    }


def _aggregate(
    records: list[dict[str, Any]],
    group_key: str,
    metrics: tuple[str, ...],
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record[group_key])].append(record)
    output = []
    for key, items in sorted(groups.items()):
        row: dict[str, Any] = {group_key: key, "trials": len(items)}
        for metric in metrics:
            values = np.asarray([float(item[metric]) for item in items])
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_ci95"] = (
                float(1.96 * np.std(values, ddof=1) / sqrt(len(values)))
                if len(values) > 1
                else 0.0
            )
            row[f"{metric}_sum"] = float(np.sum(values))
        output.append(row)
    return output


SUMMARY_METRICS = (
    "episode_return",
    "reward_ratio",
    "missed_value",
    "overall_epsilon_hit_rate",
    "quantum_candidate_p_epsilon",
    "quantum_batch_epsilon_coverage",
    "requests_issued",
    "deadline_misses",
    "deadline_miss_rate",
    "quantum_result_utilization",
    "eligible_result_utilization",
    "generation_raw_feasible_rate",
    "stale_raw_feasible_rate",
    "post_repair_feasible_rate",
    "repair_change_rate",
    "selected_identity_survival_rate",
    "mean_staleness_steps",
    "fallback_latency_mean_ms",
    "observed_total_latency_p95_ms",
    "observed_total_latency_p99_ms",
    "observed_queue_p95_ms",
    "shots_total",
    "shots_per_request",
    "oracle_exact_step_rate",
)


def _summary_by_policy(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _aggregate(records, "policy", SUMMARY_METRICS)


def _row(summary: list[dict[str, Any]], policy: str) -> dict[str, Any]:
    return next(item for item in summary if item["policy"] == policy)


def build_latency_gate(
    *,
    config: LatencyAwareConfig,
    trace_summary: dict[str, Any],
    policy_summary: list[dict[str, Any]],
    conditional_gate: dict[str, Any],
) -> dict[str, Any]:
    """Combine preregistered latency checks with retained conditional gates."""

    asynchronous = _row(policy_summary, "async_beam_quantum")
    latency_source_pass = bool(trace_summary["measured_qpu"])
    deadline_pass = (
        trace_summary["deadline_compliance"]
        >= config.minimum_deadline_compliance
    )
    reward_pass = asynchronous["reward_ratio_mean"] >= config.minimum_reward_ratio
    safety_pass = asynchronous["post_repair_feasible_rate_mean"] >= 1.0 - 1e-12
    utilization_pass = (
        asynchronous["quantum_result_utilization_mean"]
        >= config.minimum_quantum_utilization
    )
    retained = {
        "reward_ratio_pass": bool(conditional_gate["acceptable_return"]),
        "epsilon_coverage_pass": bool(conditional_gate["coverage_competitive"]),
        "calibration_transfer_pass": bool(
            conditional_gate["calibration_transfer_pass"]
        ),
        "manual_backend_pass": bool(conditional_gate["manual_quality_pass"]),
    }
    new_checks = {
        "measured_qpu_latency_pass": latency_source_pass,
        "deadline_compliance_pass": deadline_pass,
        "latency_aware_reward_pass": reward_pass,
        "post_repair_safety_pass": safety_pass,
        "quantum_utilization_pass": utilization_pass,
    }
    return {
        "pass": all(retained.values()) and all(new_checks.values()),
        "retained_conditional_gates": retained,
        "latency_aware_gates": new_checks,
        "thresholds": {
            "minimum_deadline_compliance": config.minimum_deadline_compliance,
            "minimum_reward_ratio": config.minimum_reward_ratio,
            "minimum_quantum_utilization": config.minimum_quantum_utilization,
        },
        "evidence": {
            "deadline_compliance": trace_summary["deadline_compliance"],
            "async_beam_reward_ratio": asynchronous["reward_ratio_mean"],
            "async_beam_post_repair_feasible": asynchronous[
                "post_repair_feasible_rate_mean"
            ],
            "async_beam_quantum_utilization": asynchronous[
                "quantum_result_utilization_mean"
            ],
            "surrogate_mean_tv": conditional_gate[
                "surrogate_mean_tv_to_reference"
            ],
            "manual_mean_critic_ratio": conditional_gate[
                "manual_mean_critic_ratio"
            ],
        },
    }


def run_latency_aware_benchmark(
    *,
    model: ActorCriticModel,
    conditional_gate: dict[str, Any],
    trace: LatencyTrace,
    config: LatencyAwareConfig,
    output_json: Path,
    output_report: Path,
) -> dict[str, Any]:
    """Run held-out delayed/asynchronous rollouts and write auditable outputs."""

    environment_config = DispatchConfig(
        n_jobs=config.n_jobs,
        density=config.density,
        horizon=config.horizon,
        utility_correlation="spatial",
    )
    reference: dict[int, tuple[float, float]] = {}
    records: list[dict[str, Any]] = []
    for seed_index in range(config.seeds):
        episode_seed = config.seed + 1_000_003 + 10_007 * seed_index
        reference[seed_index] = _reference_episode(
            environment_config,
            episode_seed,
            config,
        )
        for policy in POLICIES:
            record = _run_policy_episode(
                policy=policy,
                model=model,
                trace=trace,
                environment_config=environment_config,
                episode_seed=episode_seed,
                seed_index=seed_index,
                reference_return=reference[seed_index][0],
                config=config,
            )
            record["reference_oracle_exact_step_rate"] = reference[seed_index][1]
            records.append(record)
    policy_summary = _summary_by_policy(records)
    latency_summary = summarize_latency(trace, config.quantum_deadline_ms)
    gate = build_latency_gate(
        config=config,
        trace_summary=latency_summary,
        policy_summary=policy_summary,
        conditional_gate=conditional_gate,
    )
    results = {
        "schema_version": 1,
        "claim_boundary": (
            "Latency-aware synthetic dispatch extension; physical claims require "
            "a measured_qpu latency trace and every retained conditional gate."
        ),
        "config": asdict(config),
        "latency_summary": latency_summary,
        "policy_records": records,
        "policy_summary": policy_summary,
        "retained_conditional_gate": conditional_gate,
        "gate": gate,
    }
    ResultWriter().artifacts(
        json_path=output_json,
        report_path=output_report,
        payload=results,
        render_report=render_latency_report,
    )
    return results
