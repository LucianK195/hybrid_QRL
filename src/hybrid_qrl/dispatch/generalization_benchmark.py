"""Held-out stress tests for dispatch size, candidate budget, and constraints.

This module evaluates a frozen actor-critic and a previously selected Rydberg
surrogate regime. It deliberately performs no training or hyperparameter
selection. Every method sees the same held-out state and is normalized by the
same per-state MILP reward reference.

The Rydberg path remains a classical surrogate. These experiments diagnose
candidate quality and generalization; they are not evidence of quantum speedup
or hardware performance.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import json
from math import sqrt
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..core import Action
from .backlog_benchmark import SamplerRegime
from .baselines import (
    CandidateBatch,
    ProposalConfig,
    generate_candidates,
    proposal_weights,
    solve_weighted_independent_set,
)
from .benchmark import oracle_weights, realized_step_reward
from .dataset import model_from_dict
from .environment import DispatchConfig, DispatchEnvironment, DispatchState
from .learning import ActorCriticModel


@dataclass(frozen=True)
class GeneralizationBenchmarkConfig:
    """Preregistered settings for the held-out dispatch stress test."""

    seeds: int = 20
    sizes: tuple[int, ...] = (20, 40, 60, 80, 100)
    k_values: tuple[int, ...] = (1, 4, 8, 16, 32, 64)
    constraint_sizes: tuple[int, ...] = (40, 100)
    densities: tuple[float, ...] = (0.05, 0.12, 0.25, 0.40)
    dataset_size: int = 60
    fixed_k: int = 16
    warmup_steps: int = 3
    oracle_time_limit_ms: float = 1_000.0
    epsilon: float = 0.05
    seed: int = 731_021

    def __post_init__(self) -> None:
        if self.seeds <= 0:
            raise ValueError("seeds must be positive")
        all_sizes = (*self.sizes, *self.constraint_sizes, self.dataset_size)
        if any(size < 8 or size > 100 for size in all_sizes):
            raise ValueError("all job counts must lie in [8, 100]")
        if not self.k_values or any(value <= 0 for value in self.k_values):
            raise ValueError("all candidate budgets must be positive")
        if tuple(sorted(set(self.k_values))) != self.k_values:
            raise ValueError("k_values must be unique and increasing")
        if any(not 0.0 < density < 1.0 for density in self.densities):
            raise ValueError("densities must lie in (0, 1)")
        if self.fixed_k <= 0 or self.warmup_steps < 0:
            raise ValueError("fixed_k must be positive and warmup non-negative")
        if self.oracle_time_limit_ms <= 0:
            raise ValueError("oracle time limit must be positive")
        if not 0.0 < self.epsilon < 1.0:
            raise ValueError("epsilon must lie in (0, 1)")


DATASET_SETTINGS: tuple[dict[str, Any], ...] = (
    {
        "setting": "in_distribution",
        "graph_family": "unit_disk",
        "density": 0.12,
        "utility_distribution": "uniform",
        "utility_correlation": "none",
    },
    {
        "setting": "grid",
        "graph_family": "grid",
        "density": 0.12,
        "utility_distribution": "uniform",
        "utility_correlation": "none",
    },
    {
        "setting": "clustered",
        "graph_family": "clustered",
        "density": 0.12,
        "utility_distribution": "uniform",
        "utility_correlation": "none",
    },
    {
        "setting": "lognormal",
        "graph_family": "unit_disk",
        "density": 0.12,
        "utility_distribution": "lognormal",
        "utility_correlation": "none",
    },
    {
        "setting": "bimodal",
        "graph_family": "unit_disk",
        "density": 0.12,
        "utility_distribution": "bimodal",
        "utility_correlation": "none",
    },
    {
        "setting": "spatial_correlation",
        "graph_family": "unit_disk",
        "density": 0.12,
        "utility_distribution": "uniform",
        "utility_correlation": "spatial",
    },
    {
        "setting": "degree_correlation",
        "graph_family": "unit_disk",
        "density": 0.12,
        "utility_distribution": "uniform",
        "utility_correlation": "degree",
    },
    {
        "setting": "combined_shift",
        "graph_family": "clustered",
        "density": 0.25,
        "utility_distribution": "bimodal",
        "utility_correlation": "degree",
    },
)

DEADLINE_SETTINGS = (
    ("default", 3, 12),
    ("tight", 2, 5),
)


def _mean_ci(values: Iterable[float]) -> tuple[float, float]:
    items = np.asarray(list(values), dtype=float)
    if len(items) == 0:
        raise ValueError("cannot summarize an empty collection")
    average = float(np.mean(items))
    interval = (
        float(1.96 * np.std(items, ddof=1) / sqrt(len(items)))
        if len(items) > 1
        else 0.0
    )
    return average, interval


def _pairwise_diversity(actions: tuple[Action, ...]) -> float:
    if len(actions) < 2:
        return 0.0
    distances = [
        float(np.mean(np.asarray(actions[left]) != np.asarray(actions[right])))
        for left in range(len(actions))
        for right in range(left + 1, len(actions))
    ]
    return float(np.mean(distances))


def _held_out_state(
    warmup_model: ActorCriticModel,
    environment_config: DispatchConfig,
    seed: int,
    warmup_steps: int,
) -> DispatchState:
    """Create a deterministic policy-induced state on a held-out stream."""

    environment = DispatchEnvironment(environment_config, seed=seed)
    state = environment.state()
    rng = np.random.default_rng(seed + 7_919)
    for _ in range(warmup_steps):
        action = warmup_model.actor.sample(state, rng)
        state, _, done, _ = environment.step(action)
        if done:
            break
    return state


def _oracle_record(
    state: DispatchState,
    environment_config: DispatchConfig,
    time_limit_ms: float,
    metadata: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    oracle = solve_weighted_independent_set(
        state,
        oracle_weights(state, environment_config.miss_penalty),
        time_limit_ms,
    )
    reference = realized_step_reward(
        state, oracle.action, environment_config.miss_penalty
    )
    exact = bool(
        oracle.success
        and oracle.mip_gap is not None
        and oracle.mip_gap <= 1e-9
    )
    record = {
        **metadata,
        "reference_reward": reference,
        "oracle_exact": exact,
        "oracle_success": oracle.success,
        "oracle_status": oracle.status,
        "oracle_mip_gap": oracle.mip_gap,
        "oracle_latency_ms": oracle.elapsed_ms,
    }
    return reference, record


def _candidate_metrics(
    *,
    batch: CandidateBatch,
    state: DispatchState,
    model: ActorCriticModel,
    reference_reward: float,
    miss_penalty: float,
    epsilon: float,
) -> dict[str, Any]:
    """Compute proposal-upper-bound and deployed-selection diagnostics."""

    empty_reward = realized_step_reward(
        state, tuple(0 for _ in range(state.n_jobs)), miss_penalty
    )
    reference_positive = reference_reward > 1e-9
    opportunity_denominator = reference_reward - empty_reward

    def scores(action: Action) -> tuple[float | None, float]:
        reward = realized_step_reward(state, action, miss_penalty)
        ratio = reward / reference_reward if reference_positive else None
        opportunity = (
            (reward - empty_reward) / opportunity_denominator
            if opportunity_denominator > 1e-12
            else float(reward >= reference_reward - 1e-12)
        )
        return ratio, opportunity

    repaired_scores = [scores(action) for action in batch.repaired_actions]
    unique_scores = [scores(action) for action in batch.actions]
    repaired_ratios = [item[0] for item in repaired_scores]
    repaired_opportunity = [item[1] for item in repaired_scores]
    unique_ratios = [item[0] for item in unique_scores]
    unique_opportunity = [item[1] for item in unique_scores]
    if batch.actions:
        critic_action = model.best_action(state, list(batch.actions))
        weights = proposal_weights(model, state)
        utility_action = max(
            batch.actions,
            key=lambda action: float(np.asarray(action) @ weights),
        )
        critic_ratio, critic_opportunity = scores(critic_action)
        utility_ratio, utility_opportunity = scores(utility_action)
        selected_count = int(sum(critic_action))
    else:
        critic_ratio = 0.0 if reference_positive else None
        utility_ratio = 0.0 if reference_positive else None
        critic_opportunity = 0.0
        utility_opportunity = 0.0
        selected_count = 0
    threshold = 1.0 - epsilon
    valid_unique_ratios = [value for value in unique_ratios if value is not None]
    valid_repaired_ratios = [
        value for value in repaired_ratios if value is not None
    ]
    return {
        "reference_reward": reference_reward,
        "reference_positive": reference_positive,
        "empty_action_reward": empty_reward,
        "best_k_ratio": (
            max(valid_unique_ratios, default=0.0) if reference_positive else None
        ),
        "critic_selected_ratio": critic_ratio,
        "utility_selected_ratio": utility_ratio,
        "epsilon_coverage": (
            float(max(valid_unique_ratios, default=0.0) >= threshold)
            if reference_positive
            else None
        ),
        "p_epsilon": (
            float(np.mean(np.asarray(valid_repaired_ratios) >= threshold))
            if valid_repaired_ratios
            else None
        ),
        "best_k_opportunity_ratio": max(unique_opportunity, default=0.0),
        "critic_selected_opportunity_ratio": critic_opportunity,
        "utility_selected_opportunity_ratio": utility_opportunity,
        "opportunity_epsilon_coverage": float(
            max(unique_opportunity, default=0.0) >= threshold
        ),
        "p_opportunity_epsilon": float(
            np.mean(np.asarray(repaired_opportunity) >= threshold)
            if repaired_opportunity
            else 0.0
        ),
        "raw_generated": batch.raw_generated,
        "raw_feasible_rate": batch.raw_feasible / max(batch.raw_generated, 1),
        "post_repair_feasible_rate": float(
            all(state.graph.is_feasible(action) for action in batch.repaired_actions)
        ),
        "unique_feasible": batch.unique_feasible,
        "diversity": _pairwise_diversity(batch.actions),
        "proposal_latency_ms": batch.elapsed_ms,
        "critic_selected_count": selected_count,
    }


def _method_batch(
    *,
    method: str,
    state: DispatchState,
    model: ActorCriticModel,
    regime: SamplerRegime,
    candidates: int,
    seed: int,
) -> CandidateBatch:
    if method == "scale_aware":
        backend = "rydberg_surrogate"
        proposal = regime.proposal(candidates)
    elif method == "beam_search":
        backend = "beam_search"
        proposal = ProposalConfig(
            candidates=candidates,
            max_runtime_ms=2_000.0,
            beam_width=max(64, candidates),
        )
    else:
        raise ValueError(f"unknown stress-test method: {method}")
    return generate_candidates(
        backend,
        state,
        model,
        proposal,
        np.random.default_rng(seed),
    )


def _evaluate_methods(
    *,
    state: DispatchState,
    environment_config: DispatchConfig,
    model: ActorCriticModel,
    regime: SamplerRegime,
    reference_reward: float,
    candidates: int,
    seed: int,
    epsilon: float,
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    records = []
    for method_index, method in enumerate(("scale_aware", "beam_search")):
        batch = _method_batch(
            method=method,
            state=state,
            model=model,
            regime=regime,
            candidates=candidates,
            seed=seed + method_index * 1_000_003,
        )
        records.append(
            {
                **metadata,
                "method": method,
                "k": candidates,
                **_candidate_metrics(
                    batch=batch,
                    state=state,
                    model=model,
                    reference_reward=reference_reward,
                    miss_penalty=environment_config.miss_penalty,
                    epsilon=epsilon,
                ),
            }
        )
    return records


def _size_k_records(
    *,
    model: ActorCriticModel,
    warmup_model: ActorCriticModel,
    regime: SamplerRegime,
    config: GeneralizationBenchmarkConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    oracles: list[dict[str, Any]] = []
    for size in config.sizes:
        environment_config = DispatchConfig(
            n_jobs=size,
            density=0.12,
            horizon=max(config.warmup_steps + 2, 8),
        )
        for seed_index in range(config.seeds):
            seed = config.seed + 100_003 + size * 101 + seed_index * 10_007
            state = _held_out_state(
                warmup_model, environment_config, seed, config.warmup_steps
            )
            metadata = {
                "study": "size_k",
                "state_id": f"size-k-n{size}-s{seed_index}",
                "seed_index": seed_index,
                "held_out_seed": seed,
                "size": size,
                "density": environment_config.density,
                "deadline_profile": "default",
            }
            reference, oracle = _oracle_record(
                state, environment_config, config.oracle_time_limit_ms, metadata
            )
            oracles.append(oracle)
            for candidates in config.k_values:
                records.extend(
                    _evaluate_methods(
                        state=state,
                        environment_config=environment_config,
                        model=model,
                        regime=regime,
                        reference_reward=reference,
                        candidates=candidates,
                        seed=seed + 701,
                        epsilon=config.epsilon,
                        metadata=metadata,
                    )
                )
    return records, oracles


def _constraint_records(
    *,
    model: ActorCriticModel,
    warmup_model: ActorCriticModel,
    regime: SamplerRegime,
    config: GeneralizationBenchmarkConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    oracles: list[dict[str, Any]] = []
    for size in config.constraint_sizes:
        for density in config.densities:
            for deadline_profile, minimum, maximum in DEADLINE_SETTINGS:
                environment_config = DispatchConfig(
                    n_jobs=size,
                    density=density,
                    min_deadline=minimum,
                    max_deadline=maximum,
                    horizon=max(config.warmup_steps + 2, 8),
                )
                for seed_index in range(config.seeds):
                    cell = (
                        size * 10_000
                        + int(round(density * 1_000)) * 10
                        + (1 if deadline_profile == "tight" else 0)
                    )
                    seed = (
                        config.seed
                        + 2_000_003
                        + cell * 101
                        + seed_index * 10_009
                    )
                    state = _held_out_state(
                        warmup_model,
                        environment_config,
                        seed,
                        config.warmup_steps,
                    )
                    metadata = {
                        "study": "constraint",
                        "state_id": (
                            f"constraint-n{size}-d{density:.2f}-"
                            f"{deadline_profile}-s{seed_index}"
                        ),
                        "seed_index": seed_index,
                        "held_out_seed": seed,
                        "size": size,
                        "density": density,
                        "deadline_profile": deadline_profile,
                        "min_deadline": minimum,
                        "max_deadline": maximum,
                    }
                    reference, oracle = _oracle_record(
                        state,
                        environment_config,
                        config.oracle_time_limit_ms,
                        metadata,
                    )
                    oracles.append(oracle)
                    records.extend(
                        _evaluate_methods(
                            state=state,
                            environment_config=environment_config,
                            model=model,
                            regime=regime,
                            reference_reward=reference,
                            candidates=config.fixed_k,
                            seed=seed + 1_301,
                            epsilon=config.epsilon,
                            metadata=metadata,
                        )
                    )
    return records, oracles


def _dataset_records(
    *,
    model: ActorCriticModel,
    warmup_model: ActorCriticModel,
    regime: SamplerRegime,
    config: GeneralizationBenchmarkConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    oracles: list[dict[str, Any]] = []
    for setting_index, setting in enumerate(DATASET_SETTINGS):
        environment_config = DispatchConfig(
            n_jobs=config.dataset_size,
            density=float(setting["density"]),
            graph_family=str(setting["graph_family"]),
            utility_distribution=str(setting["utility_distribution"]),
            utility_correlation=str(setting["utility_correlation"]),
            horizon=max(config.warmup_steps + 2, 8),
        )
        for seed_index in range(config.seeds):
            seed = (
                config.seed
                + 4_000_037
                + setting_index * 100_003
                + seed_index * 10_019
            )
            state = _held_out_state(
                warmup_model, environment_config, seed, config.warmup_steps
            )
            metadata = {
                "study": "dataset_shift",
                "state_id": f"dataset-{setting['setting']}-s{seed_index}",
                "seed_index": seed_index,
                "held_out_seed": seed,
                "size": config.dataset_size,
                "setting": setting["setting"],
                "graph_family": setting["graph_family"],
                "density": setting["density"],
                "utility_distribution": setting["utility_distribution"],
                "utility_correlation": setting["utility_correlation"],
                "deadline_profile": "default",
            }
            reference, oracle = _oracle_record(
                state, environment_config, config.oracle_time_limit_ms, metadata
            )
            oracles.append(oracle)
            records.extend(
                _evaluate_methods(
                    state=state,
                    environment_config=environment_config,
                    model=model,
                    regime=regime,
                    reference_reward=reference,
                    candidates=config.fixed_k,
                    seed=seed + 1_909,
                    epsilon=config.epsilon,
                    metadata=metadata,
                )
            )
    return records, oracles


SUMMARY_METRICS = (
    "best_k_ratio",
    "critic_selected_ratio",
    "utility_selected_ratio",
    "epsilon_coverage",
    "p_epsilon",
    "best_k_opportunity_ratio",
    "critic_selected_opportunity_ratio",
    "utility_selected_opportunity_ratio",
    "opportunity_epsilon_coverage",
    "p_opportunity_epsilon",
    "raw_feasible_rate",
    "post_repair_feasible_rate",
    "unique_feasible",
    "diversity",
    "proposal_latency_ms",
    "critic_selected_count",
)


def _summarize(
    records: list[dict[str, Any]],
    keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[tuple(record[key] for key in keys)].append(record)
    output = []
    for group, items in sorted(grouped.items()):
        row = {key: value for key, value in zip(keys, group)}
        row["trials"] = len(items)
        for metric in SUMMARY_METRICS:
            values = [
                float(item[metric])
                for item in items
                if item[metric] is not None
            ]
            row[f"{metric}_valid_trials"] = len(values)
            if values:
                average, interval = _mean_ci(values)
                row[f"{metric}_mean"] = average
                row[f"{metric}_ci95"] = interval
            else:
                row[f"{metric}_mean"] = None
                row[f"{metric}_ci95"] = None
        output.append(row)
    return output


def _oracle_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = np.asarray([item["oracle_latency_ms"] for item in records])
    gaps = [
        float(item["oracle_mip_gap"])
        for item in records
        if item["oracle_mip_gap"] is not None
    ]
    return {
        "states": len(records),
        "exact": sum(bool(item["oracle_exact"]) for item in records),
        "exact_rate": float(np.mean([item["oracle_exact"] for item in records])),
        "positive_reference_states": sum(
            float(item["reference_reward"]) > 1e-9 for item in records
        ),
        "maximum_mip_gap": max(gaps, default=None),
        "latency_ms_mean": float(np.mean(latencies)),
        "latency_ms_p95": float(np.quantile(latencies, 0.95)),
        "latency_ms_p99": float(np.quantile(latencies, 0.99)),
    }


def _find(summary: list[dict[str, Any]], **matching: Any) -> dict[str, Any]:
    return next(
        row
        for row in summary
        if all(row.get(key) == value for key, value in matching.items())
    )


def _build_gates(results: dict[str, Any]) -> dict[str, Any]:
    config = results["config"]
    target = _find(
        results["size_k_summary"],
        method="scale_aware",
        size=max(config["sizes"]),
        k=config["fixed_k"],
    )
    constraint_rows = [
        row
        for row in results["constraint_summary"]
        if row["method"] == "scale_aware"
    ]
    dataset_rows = [
        row
        for row in results["dataset_summary"]
        if row["method"] == "scale_aware"
    ]
    all_records = (
        results["size_k_records"]
        + results["constraint_records"]
        + results["dataset_records"]
    )
    constraint_ratios_defined = all(
        row["best_k_ratio_valid_trials"] == row["trials"]
        for row in constraint_rows
    )

    def lower(row: dict[str, Any], metric: str) -> float:
        mean = row[f"{metric}_mean"]
        interval = row[f"{metric}_ci95"]
        if mean is None or interval is None:
            return float("-inf")
        return float(mean) - float(interval)

    checks = {
        "milp_exact_rate_equals_1": results["oracle_summary"]["exact_rate"] >= 1.0,
        "post_repair_feasibility_equals_1": min(
            float(row["post_repair_feasible_rate"]) for row in all_records
        )
        >= 1.0,
        "n100_k16_best_mean_at_least_0_90": target["best_k_ratio_mean"] >= 0.90,
        "n100_k16_best_lower_ci_at_least_0_90": (
            lower(target, "best_k_ratio") >= 0.90
        ),
        "n100_k16_critic_lower_ci_at_least_0_90": (
            lower(target, "critic_selected_ratio") >= 0.90
        ),
        "all_constraint_reward_ratios_defined": constraint_ratios_defined,
        "all_constraint_best_lower_ci_at_least_0_90": min(
            lower(row, "best_k_ratio")
            for row in constraint_rows
        )
        >= 0.90,
        "all_constraint_opportunity_lower_ci_at_least_0_90": min(
            lower(row, "best_k_opportunity_ratio")
            for row in constraint_rows
        )
        >= 0.90,
        "all_dataset_best_lower_ci_at_least_0_90": min(
            lower(row, "best_k_ratio")
            for row in dataset_rows
        )
        >= 0.90,
    }
    return {
        "checks": checks,
        "candidate_generation_pass": bool(
            checks["milp_exact_rate_equals_1"]
            and checks["post_repair_feasibility_equals_1"]
            and checks["n100_k16_best_lower_ci_at_least_0_90"]
            and checks["all_constraint_reward_ratios_defined"]
            and checks["all_constraint_best_lower_ci_at_least_0_90"]
            and checks["all_constraint_opportunity_lower_ci_at_least_0_90"]
            and checks["all_dataset_best_lower_ci_at_least_0_90"]
        ),
        "deployable_critic_pass": bool(
            checks["n100_k16_critic_lower_ci_at_least_0_90"]
        ),
    }


def _metric(row: dict[str, Any], name: str) -> str:
    mean = row[name + "_mean"]
    interval = row[name + "_ci95"]
    if mean is None or interval is None:
        return "n/a"
    valid = row.get(name + "_valid_trials", row["trials"])
    suffix = "" if valid == row["trials"] else f" ({valid}/{row['trials']})"
    return f"{mean:.3f} +/- {interval:.3f}{suffix}"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def build_report(results: dict[str, Any]) -> str:
    """Build the human-readable report from raw and summarized records."""

    config = results["config"]
    size_k = results["size_k_summary"]
    constraints = results["constraint_summary"]
    datasets = results["dataset_summary"]
    n100 = [
        row
        for row in size_k
        if row["method"] == "scale_aware" and row["size"] == max(config["sizes"])
    ]
    k1 = _find(n100, k=min(config["k_values"]))
    k16 = _find(n100, k=config["fixed_k"])
    k64 = _find(n100, k=max(config["k_values"]))
    beam16 = _find(
        size_k,
        method="beam_search",
        size=max(config["sizes"]),
        k=config["fixed_k"],
    )
    worst_constraint = min(
        (row for row in constraints if row["method"] == "scale_aware"),
        key=lambda row: row["best_k_opportunity_ratio_mean"],
    )
    worst_dataset = min(
        (row for row in datasets if row["method"] == "scale_aware"),
        key=lambda row: row["best_k_ratio_mean"],
    )
    status = (
        "PASS" if results["gates"]["candidate_generation_pass"] else "HOLD"
    )
    lines = [
        "# Dispatch generalization and candidate-budget stress test",
        "",
        f"## Result: {status}",
        "",
        (
            "The frozen scale-aware Rydberg surrogate was tested without "
            "retraining or retuning. At 100 jobs, best-of-K increased from "
            f"{k1['best_k_ratio_mean']:.3f} at K=1 to "
            f"{k16['best_k_ratio_mean']:.3f} at K=16 and "
            f"{k64['best_k_ratio_mean']:.3f} at K=64. The paired beam-search "
            f"best-of-16 ratio was {beam16['best_k_ratio_mean']:.3f}."
        ),
        "",
        (
            "The worst constraint cell for the surrogate was "
            f"n={worst_constraint['size']}, density={worst_constraint['density']}, "
            f"deadlines={worst_constraint['deadline_profile']}, with best-of-16 "
            "opportunity score "
            f"{_metric(worst_constraint, 'best_k_opportunity_ratio')}. The worst "
            f"dataset shift was `{worst_dataset['setting']}`, with ratio "
            f"{_metric(worst_dataset, 'best_k_ratio')}."
        ),
        "",
        "## Frozen protocol",
        "",
        (
            f"The study used {config['seeds']} paired held-out seeds per cell, "
            "the frozen multi-size model, the previously selected "
            f"`{results['selected_regime']['name']}` regime, and a paired MILP "
            "reward reference. No new setting was selected on these records."
        ),
        "",
        "## Job-count and K scaling",
        "",
        _table(
            [
                "jobs", "K", "method", "best/MILP", "critic/MILP",
                "eps-5% coverage", "opportunity", "raw feasible", "unique",
                "latency ms",
            ],
            [
                [
                    str(row["size"]), str(row["k"]), str(row["method"]),
                    _metric(row, "best_k_ratio"),
                    _metric(row, "critic_selected_ratio"),
                    _metric(row, "epsilon_coverage"),
                    _metric(row, "best_k_opportunity_ratio"),
                    _metric(row, "raw_feasible_rate"),
                    f"{row['unique_feasible_mean']:.1f}",
                    f"{row['proposal_latency_ms_mean']:.2f}",
                ]
                for row in size_k
            ],
        ),
        "",
        "## Constraint-pressure sweep at K=16",
        "",
        _table(
            [
                "jobs", "density", "deadlines", "method", "best/MILP",
                "critic/MILP", "opportunity", "opp. eps-5% coverage",
                "raw feasible",
            ],
            [
                [
                    str(row["size"]), f"{row['density']:.2f}",
                    str(row["deadline_profile"]), str(row["method"]),
                    _metric(row, "best_k_ratio"),
                    _metric(row, "critic_selected_ratio"),
                    _metric(row, "best_k_opportunity_ratio"),
                    _metric(row, "opportunity_epsilon_coverage"),
                    _metric(row, "raw_feasible_rate"),
                ]
                for row in constraints
            ],
        ),
        "",
        "## Dataset-shift sweep at n=60 and K=16",
        "",
        _table(
            [
                "setting", "method", "best/MILP", "critic/MILP",
                "eps-5% coverage", "opportunity", "diversity",
            ],
            [
                [
                    str(row["setting"]), str(row["method"]),
                    _metric(row, "best_k_ratio"),
                    _metric(row, "critic_selected_ratio"),
                    _metric(row, "epsilon_coverage"),
                    _metric(row, "best_k_opportunity_ratio"),
                    _metric(row, "diversity"),
                ]
                for row in datasets
            ],
        ),
        "",
        "## Reference and safety checks",
        "",
        _table(
            ["check", "result"],
            [
                [key.replace("_", " "), str(value)]
                for key, value in results["gates"]["checks"].items()
            ],
        ),
        "",
        (
            f"MILP completed exactly on {results['oracle_summary']['exact']} of "
            f"{results['oracle_summary']['states']} distinct held-out states. "
            f"The MILP reward was positive on "
            f"{results['oracle_summary']['positive_reference_states']} states; "
            "reward/MILP is marked `n/a` for non-positive references. "
            "Post-repair feasibility is evaluated against the authoritative "
            "application conflict graph."
        ),
        "",
        "## Interpretation boundary",
        "",
        (
            "Best-of-K measures whether a strong action was present in the batch; "
            "critic-selected ratio measures whether the current learned critic "
            "would actually deploy it. The Rydberg generator evaluated here is a "
            "classical stochastic surrogate, so its quality and local runtime do "
            "not establish neutral-atom hardware performance or quantum advantage."
        ),
        "",
        (
            "For non-positive MILP rewards, the opportunity score is "
            "(reward - empty-action reward) / (MILP reward - empty-action reward). "
            "It maps the empty action to 0 and the MILP optimum to 1 without "
            "dividing by a zero or negative optimum."
        ),
        "",
    ]
    return "\n".join(lines)


def run_generalization_benchmark(
    *,
    stable_results_path: Path,
    baseline_results_path: Path,
    output_json: Path,
    output_report: Path,
    config: GeneralizationBenchmarkConfig = GeneralizationBenchmarkConfig(),
) -> dict[str, Any]:
    """Run all preregistered cells and write raw JSON plus Markdown report."""

    stable = json.loads(stable_results_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_results_path.read_text(encoding="utf-8"))
    model = model_from_dict(stable["trained_model"])
    warmup_model = model_from_dict(baseline["model"])
    regime = SamplerRegime(**stable["selected_regime"])

    size_k_records, size_oracles = _size_k_records(
        model=model, warmup_model=warmup_model, regime=regime, config=config
    )
    constraint_records, constraint_oracles = _constraint_records(
        model=model, warmup_model=warmup_model, regime=regime, config=config
    )
    dataset_records, dataset_oracles = _dataset_records(
        model=model, warmup_model=warmup_model, regime=regime, config=config
    )
    oracle_records = size_oracles + constraint_oracles + dataset_oracles
    results: dict[str, Any] = {
        "schema_version": 1,
        "study": "dispatch_generalization_stress",
        "claim_boundary": (
            "Frozen-model classical-surrogate generalization study; not a "
            "hardware or quantum-advantage result."
        ),
        "config": asdict(config),
        "selected_regime": asdict(regime),
        "source_files": {
            "stable_results": str(stable_results_path.resolve()),
            "baseline_results": str(baseline_results_path.resolve()),
        },
        "size_k_records": size_k_records,
        "constraint_records": constraint_records,
        "dataset_records": dataset_records,
        "oracle_records": oracle_records,
        "size_k_summary": _summarize(size_k_records, ("method", "size", "k")),
        "constraint_summary": _summarize(
            constraint_records,
            ("method", "size", "density", "deadline_profile"),
        ),
        "dataset_summary": _summarize(
            dataset_records, ("method", "setting")
        ),
        "oracle_summary": _oracle_summary(oracle_records),
    }
    results["gates"] = _build_gates(results)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    output_report.write_text(build_report(results), encoding="utf-8")
    return results
