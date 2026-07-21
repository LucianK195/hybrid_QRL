"""Conditional quantum-assisted advantage study and report generation.

This study is the bridge between the scalable classical blockade surrogate and
small-system quantum evolution.  It performs four linked experiments:

1. train actor utilities through the sampled, repaired, critic-reranked return;
2. compare that learner with direct autoregressive actor-critic training;
3. calibrate surrogate distributions against dense, QuTiP, and downloaded
   neutral-atom evolutions at 8--12 binary decisions; and
4. search a 16-decision hardware-compatible phase map using 20 held-out seeds
   per regime and serious classical baselines.

The output is evidence for a conditional pipeline claim, not quantum supremacy.
Emulator wall time is reported for reproducibility and is never interpreted as
neutral-atom hardware latency.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from math import ceil, log, sqrt
from pathlib import Path
from time import perf_counter
from typing import Any
import json
import sys

import numpy as np

from ..core import Action
from ..quantum import (
    DenseRydbergStatevectorSampler,
    ManualNeutralAtomBackendSampler,
    PulseSchedule,
    QuTiPRydbergSampler,
)
from .baselines import (
    ProposalConfig,
    generate_candidates,
    proposal_weights,
    repair_action,
    solve_weighted_independent_set,
)
from .benchmark import oracle_weights, realized_step_reward
from .environment import DispatchConfig, DispatchEnvironment, DispatchState
from .learning import ActorCriticModel, TrainingConfig, train_actor_critic
from .sampler_loop import (
    SamplerLoopTrainingConfig,
    train_sampler_in_loop,
)


EPSILONS = (0.01, 0.05, 0.10)
PHASE_METHODS = (
    "beam_search",
    "local_search",
    "simulated_annealing",
    "rydberg_surrogate",
)


@dataclass(frozen=True)
class ConditionalAdvantageConfig:
    """Configuration for training, calibration, and phase-map evaluation."""

    seeds: int = 20
    sampler_training_iterations: int = 140
    candidate_budget: int = 16
    surrogate_probability_shots: int = 4_096
    calibration_sizes: tuple[int, ...] = (8, 10, 12)
    calibration_seeds: int = 20
    calibration_pulse_steps: int = 4
    pipeline_seeds: int = 10
    pipeline_horizon: int = 6
    phase_size: int = 16
    phase_seeds: int = 40
    seed: int = 8_117

    def __post_init__(self) -> None:
        if self.seeds < 20 or self.phase_seeds < 40:
            raise ValueError(
                "training requires 20 seeds and phase cells require 20+20 seeds"
            )
        if self.calibration_seeds < 10:
            raise ValueError("calibration requires at least 10 held-out seeds")
        if self.pipeline_seeds < 5:
            raise ValueError("pipeline proof requires at least five seeds")
        if self.candidate_budget <= 0:
            raise ValueError("candidate_budget must be positive")


def epsilon_field(epsilon: float) -> str:
    """Return the stable JSON suffix for an epsilon threshold."""

    return f"epsilon_{int(round(100 * epsilon)):02d}"


def shots_for_95_percent(success_probability: float) -> int | None:
    """Return iid shots required for 95% chance of at least one success."""

    if success_probability <= 0.0:
        return None
    if success_probability >= 1.0:
        return 1
    return int(ceil(log(0.05) / log(1.0 - success_probability)))


def _held_out_state(
    model: ActorCriticModel,
    config: DispatchConfig,
    seed: int,
    warmup_steps: int = 2,
) -> DispatchState:
    environment = DispatchEnvironment(config, seed=seed)
    state = environment.state()
    rng = np.random.default_rng(seed + 99_991)
    for _ in range(warmup_steps):
        action = model.actor.sample(state, rng)
        state, _, done, _ = environment.step(action)
        if done:
            break
    return state


def _action_distribution(actions: tuple[Action, ...]) -> dict[Action, float]:
    counts: dict[Action, float] = defaultdict(float)
    for action in actions:
        counts[action] += 1.0
    total = max(float(len(actions)), 1.0)
    return {action: count / total for action, count in counts.items()}


def _distribution_from_probabilities(
    probabilities: np.ndarray,
    state: DispatchState,
    weights: np.ndarray,
    bit_order: str,
) -> tuple[dict[Action, float], float]:
    repaired: dict[Action, float] = defaultdict(float)
    raw_feasible = 0.0
    for basis, probability in enumerate(probabilities):
        if probability <= 0.0:
            continue
        if bit_order == "least_significant_node_zero":
            raw = tuple((basis >> node) & 1 for node in range(state.n_jobs))
        elif bit_order == "most_significant_node_zero":
            raw = tuple(
                int(bit) for bit in format(basis, f"0{state.n_jobs}b")
            )
        else:
            raise ValueError(f"unknown bit order: {bit_order}")
        if state.graph.is_feasible(raw):
            raw_feasible += float(probability)
        safe = repair_action(raw, state.graph, weights)
        repaired[safe] += float(probability)
    return dict(repaired), raw_feasible


def _expected_hamming_diversity(distribution: dict[Action, float]) -> float:
    if not distribution:
        return 0.0
    nodes = len(next(iter(distribution)))
    one_probability = np.zeros(nodes, dtype=float)
    for action, probability in distribution.items():
        one_probability += probability * np.asarray(action, dtype=float)
    return float(np.mean(2.0 * one_probability * (1.0 - one_probability)))


def _total_variation(
    left: dict[Action, float], right: dict[Action, float]
) -> float:
    support = set(left) | set(right)
    return 0.5 * float(
        sum(abs(left.get(action, 0.0) - right.get(action, 0.0)) for action in support)
    )


def _expected_best_ratio(
    distribution: dict[Action, float],
    reward_ratio: dict[Action, float],
    candidates: int,
) -> float:
    grouped: dict[float, float] = defaultdict(float)
    for action, probability in distribution.items():
        grouped[reward_ratio[action]] += probability
    cumulative = 0.0
    previous_power = 0.0
    expected = 0.0
    for value, probability in sorted(grouped.items()):
        cumulative += probability
        current_power = cumulative**candidates
        expected += value * (current_power - previous_power)
        previous_power = current_power
    return float(expected)


def _critic_selected_ratio(
    distribution: dict[Action, float],
    reward_ratio: dict[Action, float],
    model: ActorCriticModel,
    state: DispatchState,
    candidates: int,
    rng: np.random.Generator,
    batches: int = 256,
) -> float:
    actions = list(distribution)
    probabilities = np.asarray([distribution[action] for action in actions])
    selected_ratios = []
    for _ in range(batches):
        indices = rng.choice(
            len(actions), size=candidates, replace=True, p=probabilities
        )
        batch = list(dict.fromkeys(actions[int(index)] for index in indices))
        selected = model.best_action(state, batch)
        selected_ratios.append(reward_ratio[selected])
    return float(np.mean(selected_ratios))


def _distribution_metrics(
    distribution: dict[Action, float],
    raw_feasible_probability: float,
    state: DispatchState,
    model: ActorCriticModel,
    reference_reward: float,
    candidates: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    reward_ratio = {
        action: realized_step_reward(state, action, miss_penalty=1.0)
        / max(reference_reward, 1e-12)
        for action in distribution
    }
    metrics: dict[str, Any] = {
        "raw_feasible_probability": raw_feasible_probability,
        "expected_reward_ratio": float(
            sum(distribution[action] * reward_ratio[action] for action in distribution)
        ),
        "expected_best_k_ratio": _expected_best_ratio(
            distribution, reward_ratio, candidates
        ),
        "critic_selected_ratio": _critic_selected_ratio(
            distribution,
            reward_ratio,
            model,
            state,
            candidates,
            rng,
        ),
        "expected_hamming_diversity": _expected_hamming_diversity(distribution),
        "support_size": len(distribution),
    }
    for epsilon in EPSILONS:
        probability = float(
            sum(
                distribution[action]
                for action in distribution
                if reward_ratio[action] >= 1.0 - epsilon
            )
        )
        suffix = epsilon_field(epsilon)
        metrics[f"p_{suffix}"] = probability
        metrics[f"k95_{suffix}"] = shots_for_95_percent(probability)
        metrics[f"coverage_k_{suffix}"] = 1.0 - (1.0 - probability) ** candidates
    return metrics


def _manual_protocol(steps: int) -> object:
    workspace = Path(__file__).resolve().parents[4]
    source = workspace / "QML-Platform-for-Neutral-Atom" / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from neutral_atom.simulator import AdiabaticProtocol

    return AdiabaticProtocol(
        total_time=2.0,
        n_steps=max(steps + 1, 8),
        omega_max=1.5,
        delta_g_initial=-3.0,
        delta_l_max=3.0,
    )


def _small_backend_objects(
    state: DispatchState,
    pulse_steps: int,
    include_dense: bool,
) -> list[tuple[str, object, str]]:
    schedule = PulseSchedule(
        duration=2.0,
        steps=pulse_steps,
        omega_max=1.5,
        delta_start=-3.0,
        delta_end=3.0,
        blockade=10.0,
    )
    output: list[tuple[str, object, str]] = []
    if include_dense:
        output.append(
            (
                "dense",
                DenseRydbergStatevectorSampler(schedule, cache_decimals=3),
                "least_significant_node_zero",
            )
        )
    output.append(
        (
            "qutip",
            QuTiPRydbergSampler(schedule, cache_decimals=3),
            "most_significant_node_zero",
        )
    )
    pair_distances = [
        float(np.linalg.norm(state.positions[left] - state.positions[right]))
        for left in range(state.n_jobs)
        for right in range(left + 1, state.n_jobs)
    ]
    minimum_separation = max(min(pair_distances), 1e-9)
    physical_positions = state.positions / minimum_separation
    manual = ManualNeutralAtomBackendSampler(
        positions=physical_positions,
        C6=10.0,
        protocol=_manual_protocol(pulse_steps),
        backend_source=Path(__file__).resolve().parents[4]
        / "QML-Platform-for-Neutral-Atom",
        cache_decimals=3,
    )
    output.append(("manual", manual, "most_significant_node_zero"))
    return output


def _run_training_comparison(
    config: ConditionalAdvantageConfig,
) -> tuple[
    ActorCriticModel,
    dict[str, list[float]],
    ActorCriticModel,
    dict[str, list[float]],
    list[dict[str, Any]],
]:
    direct_config = TrainingConfig(
        episodes=2 * config.sampler_training_iterations,
        horizon=12,
        train_sizes=(12, 16),
        densities=(0.08, 0.12, 0.20),
        graph_families=("unit_disk", "grid", "clustered"),
        utility_correlations=("none", "spatial", "degree"),
        seed=config.seed,
    )
    direct_model, direct_history = train_actor_critic(direct_config)
    sampler_config = SamplerLoopTrainingConfig(
        iterations=config.sampler_training_iterations,
        horizon=12,
        candidates=config.candidate_budget,
        seed=config.seed,
    )
    sampler_model, sampler_history = train_sampler_in_loop(sampler_config)

    records: list[dict[str, Any]] = []
    for seed_index in range(config.seeds):
        environment_seed = config.seed + 2_000_003 + seed_index * 10_007
        for model_name, model in (
            ("direct_actor", direct_model),
            ("sampler_in_loop", sampler_model),
        ):
            environment = DispatchEnvironment(
                DispatchConfig(
                    n_jobs=16,
                    density=0.12,
                    graph_family="clustered",
                    utility_correlation="spatial",
                    horizon=12,
                ),
                seed=environment_seed,
            )
            state = environment.state()
            total_return = 0.0
            feasible_rates = []
            done = False
            step = 0
            while not done:
                batch = generate_candidates(
                    "rydberg_surrogate",
                    state,
                    model,
                    ProposalConfig(candidates=config.candidate_budget),
                    np.random.default_rng(environment_seed + 101 * step),
                )
                action = model.best_action(state, list(batch.actions))
                state, reward, done, _ = environment.step(action)
                total_return += reward
                feasible_rates.append(
                    batch.raw_feasible / max(batch.raw_generated, 1)
                )
                step += 1
            records.append(
                {
                    "seed_index": seed_index,
                    "model": model_name,
                    "episode_return": total_return,
                    "raw_feasible_rate": float(np.mean(feasible_rates)),
                }
            )
    return (
        direct_model,
        direct_history,
        sampler_model,
        sampler_history,
        records,
    )


def _run_calibration(
    model: ActorCriticModel,
    config: ConditionalAdvantageConfig,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for size in config.calibration_sizes:
        for seed_index in range(config.calibration_seeds):
            seed = config.seed + 4_000_037 + 20_011 * seed_index + 101 * size
            state = _held_out_state(
                model,
                DispatchConfig(
                    n_jobs=size,
                    density=0.14,
                    graph_family="grid" if seed_index % 2 else "unit_disk",
                    utility_correlation="spatial",
                    horizon=6,
                ),
                seed,
            )
            weights = proposal_weights(model, state)
            oracle = solve_weighted_independent_set(
                state,
                oracle_weights(state, miss_penalty=1.0),
                time_limit_ms=1_000.0,
            )
            reference_reward = realized_step_reward(
                state, oracle.action, miss_penalty=1.0
            )
            distributions: dict[str, dict[Action, float]] = {}
            pending: list[dict[str, Any]] = []

            for backend, sampler, bit_order in _small_backend_objects(
                state,
                config.calibration_pulse_steps,
                include_dense=size <= 10,
            ):
                start = perf_counter()
                probabilities = sampler.probabilities(weights, state.graph)
                elapsed_ms = (perf_counter() - start) * 1_000.0
                distribution, raw_feasible = _distribution_from_probabilities(
                    probabilities,
                    state,
                    weights,
                    bit_order,
                )
                distributions[backend] = distribution
                metrics = _distribution_metrics(
                    distribution,
                    raw_feasible,
                    state,
                    model,
                    reference_reward,
                    config.candidate_budget,
                    np.random.default_rng(seed + 17 * len(pending)),
                )
                geometry_separation = None
                if backend == "manual":
                    report = sampler.geometry_report(state.graph)
                    geometry_separation = report["edge_to_nonedge_separation"]
                pending.append(
                    {
                        "seed_index": seed_index,
                        "n_jobs": size,
                        "backend": backend,
                        "graph_family": (
                            "grid" if seed_index % 2 else "unit_disk"
                        ),
                        "backend_evolution_ms": elapsed_ms,
                        "oracle_exact": oracle.success,
                        "oracle_mip_gap": oracle.mip_gap,
                        "geometry_separation": geometry_separation,
                        **metrics,
                    }
                )

            surrogate = generate_candidates(
                "rydberg_surrogate",
                state,
                model,
                ProposalConfig(
                    candidates=config.surrogate_probability_shots,
                    max_runtime_ms=30_000.0,
                ),
                np.random.default_rng(seed + 8_080),
            )
            surrogate_distribution = _action_distribution(
                surrogate.repaired_actions
            )
            distributions["surrogate"] = surrogate_distribution
            surrogate_metrics = _distribution_metrics(
                surrogate_distribution,
                surrogate.raw_feasible / max(surrogate.raw_generated, 1),
                state,
                model,
                reference_reward,
                config.candidate_budget,
                np.random.default_rng(seed + 9_090),
            )
            pending.append(
                {
                    "seed_index": seed_index,
                    "n_jobs": size,
                    "backend": "surrogate",
                    "graph_family": "grid" if seed_index % 2 else "unit_disk",
                    "backend_evolution_ms": surrogate.elapsed_ms,
                    "oracle_exact": oracle.success,
                    "oracle_mip_gap": oracle.mip_gap,
                    "geometry_separation": None,
                    **surrogate_metrics,
                }
            )
            reference_backend = "dense" if "dense" in distributions else "qutip"
            for record in pending:
                record["distribution_reference"] = reference_backend
                record["total_variation_to_reference"] = _total_variation(
                    distributions[record["backend"]],
                    distributions[reference_backend],
                )
                records.append(record)
    return records


def _run_pipeline_proof(
    model: ActorCriticModel,
    config: ConditionalAdvantageConfig,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for seed_index in range(config.pipeline_seeds):
        seed = config.seed + 6_000_011 + seed_index * 10_009
        environment_config = DispatchConfig(
            n_jobs=8,
            density=0.14,
            graph_family="grid",
            utility_correlation="spatial",
            horizon=config.pipeline_horizon,
        )
        initial_environment = DispatchEnvironment(environment_config, seed=seed)
        initial_state = initial_environment.state()
        backend_objects = _small_backend_objects(
            initial_state,
            config.calibration_pulse_steps,
            include_dense=True,
        )
        backend_objects.append(("surrogate", None, ""))
        for backend_index, (backend, sampler, _) in enumerate(backend_objects):
            environment = DispatchEnvironment(environment_config, seed=seed)
            state = environment.state()
            total_return = 0.0
            raw_feasibility = []
            latencies = []
            executed_safe = True
            done = False
            step = 0
            while not done:
                weights = proposal_weights(model, state)
                start = perf_counter()
                if backend == "surrogate":
                    batch = generate_candidates(
                        "rydberg_surrogate",
                        state,
                        model,
                        ProposalConfig(candidates=config.candidate_budget),
                        np.random.default_rng(seed + 101 * step),
                    )
                    candidates = list(batch.actions)
                    raw_rate = batch.raw_feasible / max(batch.raw_generated, 1)
                else:
                    raw = sampler.sample(
                        weights,
                        state.graph,
                        config.candidate_budget,
                        np.random.default_rng(
                            seed + 1_003 * backend_index + 101 * step
                        ),
                    )
                    raw_rate = float(
                        np.mean([state.graph.is_feasible(action) for action in raw])
                    )
                    repaired = [
                        repair_action(action, state.graph, weights) for action in raw
                    ]
                    candidates = list(dict.fromkeys(repaired))
                elapsed_ms = (perf_counter() - start) * 1_000.0
                action = model.best_action(state, candidates)
                executed_safe = executed_safe and state.graph.is_feasible(action)
                state, reward, done, _ = environment.step(action)
                total_return += reward
                raw_feasibility.append(raw_rate)
                latencies.append(elapsed_ms)
                step += 1
            records.append(
                {
                    "seed_index": seed_index,
                    "backend": backend,
                    "episode_return": total_return,
                    "raw_feasible_rate": float(np.mean(raw_feasibility)),
                    "mean_emulator_step_ms": float(np.mean(latencies)),
                    "executed_safe": executed_safe,
                }
            )
    return records


def _phase_regimes() -> list[dict[str, Any]]:
    regimes: list[dict[str, Any]] = []
    for density in (0.06, 0.12, 0.22):
        for radius_scale in (0.9, 1.0, 1.1):
            for pulse in ("short", "balanced", "adiabatic"):
                regimes.append(
                    {
                        "regime": (
                            f"physical:d={density}:r={radius_scale}:p={pulse}"
                        ),
                        "axis": "physical_phase",
                        "density": density,
                        "graph_family": "unit_disk",
                        "utility_distribution": "uniform",
                        "utility_correlation": "none",
                        "blockade_radius_scale": radius_scale,
                        "pulse_schedule": pulse,
                    }
                )
    for family in ("unit_disk", "grid", "clustered"):
        for correlation in ("none", "spatial", "degree"):
            regimes.append(
                {
                    "regime": f"structure:g={family}:c={correlation}",
                    "axis": "structure_phase",
                    "density": 0.12,
                    "graph_family": family,
                    "utility_distribution": "uniform",
                    "utility_correlation": correlation,
                    "blockade_radius_scale": 1.0,
                    "pulse_schedule": "balanced",
                }
            )
    for distribution in ("uniform", "lognormal", "bimodal"):
        regimes.append(
            {
                "regime": f"utility:u={distribution}",
                "axis": "utility_phase",
                "density": 0.12,
                "graph_family": "unit_disk",
                "utility_distribution": distribution,
                "utility_correlation": "none",
                "blockade_radius_scale": 1.0,
                "pulse_schedule": "balanced",
            }
        )
    return regimes


def _batch_metrics(
    state: DispatchState,
    model: ActorCriticModel,
    repaired_actions: tuple[Action, ...],
    reference_reward: float,
) -> dict[str, Any]:
    distribution = _action_distribution(repaired_actions)
    ratios = [
        realized_step_reward(state, action, miss_penalty=1.0)
        / max(reference_reward, 1e-12)
        for action in repaired_actions
    ]
    output: dict[str, Any] = {
        "best_batch_reward_ratio": max(ratios, default=0.0),
        "candidate_hamming_diversity": _expected_hamming_diversity(distribution),
    }
    for epsilon in EPSILONS:
        probability = float(np.mean(np.asarray(ratios) >= 1.0 - epsilon))
        suffix = epsilon_field(epsilon)
        output[f"p_{suffix}"] = probability
        output[f"k95_{suffix}"] = shots_for_95_percent(probability)
        output[f"hit_{suffix}"] = float(probability > 0.0)
    return output


def _run_phase_search(
    model: ActorCriticModel,
    config: ConditionalAdvantageConfig,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for regime_index, regime in enumerate(_phase_regimes()):
        for seed_index in range(config.phase_seeds):
            seed = (
                config.seed
                + 8_000_021
                + regime_index * 1_000_003
                + seed_index * 10_007
            )
            state = _held_out_state(
                model,
                DispatchConfig(
                    n_jobs=config.phase_size,
                    density=regime["density"],
                    graph_family=regime["graph_family"],
                    utility_distribution=regime["utility_distribution"],
                    utility_correlation=regime["utility_correlation"],
                    horizon=6,
                ),
                seed,
            )
            oracle = solve_weighted_independent_set(
                state,
                oracle_weights(state, miss_penalty=1.0),
                time_limit_ms=1_000.0,
            )
            reference_reward = realized_step_reward(
                state, oracle.action, miss_penalty=1.0
            )
            for method_index, method in enumerate(PHASE_METHODS):
                batch = generate_candidates(
                    method,
                    state,
                    model,
                    ProposalConfig(
                        candidates=config.candidate_budget,
                        max_runtime_ms=2_000.0,
                        blockade_radius_scale=regime[
                            "blockade_radius_scale"
                        ],
                        pulse_schedule=regime["pulse_schedule"],
                    ),
                    np.random.default_rng(seed + 101 * method_index),
                )
                selected = model.best_action(state, list(batch.actions))
                selected_ratio = realized_step_reward(
                    state, selected, miss_penalty=1.0
                ) / max(reference_reward, 1e-12)
                records.append(
                    {
                        "regime": regime["regime"],
                        "axis": regime["axis"],
                        "seed_index": seed_index,
                        "split": (
                            "selection"
                            if seed_index < config.phase_seeds // 2
                            else "confirmation"
                        ),
                        "method": method,
                        "n_jobs": config.phase_size,
                        "density": regime["density"],
                        "graph_family": regime["graph_family"],
                        "utility_distribution": regime[
                            "utility_distribution"
                        ],
                        "utility_correlation": regime[
                            "utility_correlation"
                        ],
                        "blockade_radius_scale": regime[
                            "blockade_radius_scale"
                        ],
                        "pulse_schedule": regime["pulse_schedule"],
                        "critic_selected_reward_ratio": selected_ratio,
                        "raw_feasible_rate": (
                            batch.raw_feasible / max(batch.raw_generated, 1)
                        ),
                        "unique_feasible": batch.unique_feasible,
                        "end_to_end_latency_ms": batch.elapsed_ms,
                        "oracle_exact": oracle.success,
                        "oracle_mip_gap": oracle.mip_gap,
                        **_batch_metrics(
                            state,
                            model,
                            batch.repaired_actions,
                            reference_reward,
                        ),
                    }
                )
    return records


def _aggregate(
    records: list[dict[str, Any]],
    keys: tuple[str, ...],
    metrics: tuple[str, ...],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[tuple(record[key] for key in keys)].append(record)
    output = []
    for group, items in groups.items():
        row = {key: value for key, value in zip(keys, group)}
        row["trials"] = len(items)
        for metric in metrics:
            values = np.asarray([float(item[metric]) for item in items])
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_ci95"] = (
                float(1.96 * values.std(ddof=1) / sqrt(len(values)))
                if len(values) > 1
                else 0.0
            )
        output.append(row)
    return output


def _fmt(mean: float, ci: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} ± {ci:.{digits}f}"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _gate_result(
    selection_summary: list[dict[str, Any]],
    confirmation_summary: list[dict[str, Any]],
    calibration_records: list[dict[str, Any]],
    pipeline_records: list[dict[str, Any]],
) -> dict[str, Any]:
    selection_surrogate_rows = [
        row
        for row in selection_summary
        if row["method"] == "rydberg_surrogate"
    ]
    selected = max(
        selection_surrogate_rows,
        key=lambda row: row["critic_selected_reward_ratio_mean"],
    )
    peers = {
        row["method"]: row
        for row in confirmation_summary
        if row["regime"] == selected["regime"]
    }
    best = peers["rydberg_surrogate"]
    acceptable_return = best["critic_selected_reward_ratio_mean"] >= 0.90
    beam = peers["beam_search"]
    local = peers["local_search"]
    latency_win = best["end_to_end_latency_ms_mean"] < min(
        beam["end_to_end_latency_ms_mean"],
        local["end_to_end_latency_ms_mean"],
    )
    diversity_win = best["candidate_hamming_diversity_mean"] > 1.05 * max(
        beam["candidate_hamming_diversity_mean"],
        local["candidate_hamming_diversity_mean"],
    )
    coverage_competitive = best["hit_epsilon_05_mean"] >= max(
        beam["hit_epsilon_05_mean"],
        local["hit_epsilon_05_mean"],
    )
    surrogate_opportunity_pass = (
        acceptable_return
        and coverage_competitive
        and (latency_win or diversity_win)
    )
    surrogate_tv = float(
        np.mean(
            [
                row["total_variation_to_reference"]
                for row in calibration_records
                if row["backend"] == "surrogate"
            ]
        )
    )
    manual_ratio = float(
        np.mean(
            [
                row["critic_selected_ratio"]
                for row in calibration_records
                if row["backend"] == "manual"
            ]
        )
    )
    qutip_ratio = float(
        np.mean(
            [
                row["critic_selected_ratio"]
                for row in calibration_records
                if row["backend"] == "qutip"
            ]
        )
    )
    pipeline_proof_pass = all(
        bool(row["executed_safe"]) for row in pipeline_records
    )
    calibration_transfer_pass = surrogate_tv <= 0.15
    manual_quality_pass = manual_ratio >= 0.90
    overall_pass = (
        pipeline_proof_pass
        and surrogate_opportunity_pass
        and calibration_transfer_pass
        and manual_quality_pass
    )
    return {
        "pass": overall_pass,
        "pipeline_proof_pass": pipeline_proof_pass,
        "surrogate_opportunity_pass": surrogate_opportunity_pass,
        "calibration_transfer_pass": calibration_transfer_pass,
        "manual_quality_pass": manual_quality_pass,
        "best_regime": selected["regime"],
        "acceptable_return": acceptable_return,
        "coverage_competitive": coverage_competitive,
        "latency_win": latency_win,
        "diversity_win": diversity_win,
        "surrogate_mean_tv_to_reference": surrogate_tv,
        "manual_mean_critic_ratio": manual_ratio,
        "qutip_mean_critic_ratio": qutip_ratio,
        "thresholds": {
            "acceptable_return": 0.90,
            "maximum_transfer_tv": 0.15,
            "minimum_manual_ratio": 0.90,
        },
        "selection_surrogate": selected,
        "surrogate": best,
        "peers": peers,
    }


def build_conditional_report(results: dict[str, Any]) -> str:
    """Create the Markdown result report and conditional gate decision."""

    training_summary = _aggregate(
        results["training_comparison_records"],
        ("model",),
        ("episode_return", "raw_feasible_rate"),
    )
    pipeline_summary = _aggregate(
        results["pipeline_records"],
        ("backend",),
        ("episode_return", "raw_feasible_rate", "mean_emulator_step_ms"),
    )
    calibration_summary = _aggregate(
        results["calibration_records"],
        ("n_jobs", "backend"),
        (
            "critic_selected_ratio",
            "expected_best_k_ratio",
            "p_epsilon_05",
            "coverage_k_epsilon_05",
            "raw_feasible_probability",
            "expected_hamming_diversity",
            "total_variation_to_reference",
            "backend_evolution_ms",
        ),
    )
    phase_metrics = (
        "critic_selected_reward_ratio",
        "best_batch_reward_ratio",
        "hit_epsilon_05",
        "p_epsilon_05",
        "candidate_hamming_diversity",
        "raw_feasible_rate",
        "end_to_end_latency_ms",
    )
    selection_summary = _aggregate(
        [
            row
            for row in results["phase_records"]
            if row["split"] == "selection"
        ],
        ("regime", "method"),
        phase_metrics,
    )
    confirmation_summary = _aggregate(
        [
            row
            for row in results["phase_records"]
            if row["split"] == "confirmation"
        ],
        ("regime", "method"),
        phase_metrics,
    )
    gate = _gate_result(
        selection_summary,
        confirmation_summary,
        results["calibration_records"],
        results["pipeline_records"],
    )
    results["gate"] = gate
    top_surrogate = sorted(
        (
            row
            for row in selection_summary
            if row["method"] == "rydberg_surrogate"
        ),
        key=lambda row: row["critic_selected_reward_ratio_mean"],
        reverse=True,
    )[:10]
    status = "PASS" if gate["pass"] else "HOLD — NOT YET ESTABLISHED"
    confirmed_ratio = gate["surrogate"]["critic_selected_reward_ratio_mean"]
    confirmed_hit = gate["surrogate"]["hit_epsilon_05_mean"]
    paired_differences = []
    training_by_seed: dict[int, dict[str, float]] = defaultdict(dict)
    for record in results["training_comparison_records"]:
        training_by_seed[int(record["seed_index"])][record["model"]] = float(
            record["episode_return"]
        )
    for values in training_by_seed.values():
        paired_differences.append(
            values["sampler_in_loop"] - values["direct_actor"]
        )
    paired_values = np.asarray(paired_differences)
    paired_mean = float(paired_values.mean())
    paired_ci = float(
        1.96 * paired_values.std(ddof=1) / sqrt(len(paired_values))
    )
    lines = [
        "# Conditional quantum-assisted advantage study",
        "",
        "## Claim boundary",
        "",
        "This experiment tests whether the complete hybrid proposal pipeline can "
        "work and whether a favorable conditional regime exists. Dense, QuTiP, "
        "and manual results are classical quantum-system simulations; the scalable "
        "Rydberg path is a classical blockade surrogate. No hardware quantum "
        "advantage or asymptotic supremacy is claimed.",
        "",
        "## Gate decision",
        "",
        f"**Overall conditional-advantage status: {status}.** The safe pipeline "
        f"proof passed and the best surrogate regime was `{gate['best_regime']}`, "
        "but the confirmed opportunity and calibration requirements did not all "
        "pass.",
        "",
        _table(
            ["gate", "result", "evidence"],
            [
                [
                    "safe pipeline proof",
                    str(gate["pipeline_proof_pass"]),
                    "all executed backend actions passed application safety",
                ],
                [
                    "surrogate opportunity",
                    str(gate["surrogate_opportunity_pass"]),
                    (
                        f"confirmed ratio "
                        f"{confirmed_ratio:.3f}; eps-5% hit {confirmed_hit:.2f}"
                    ),
                ],
                [
                    "distribution transfer",
                    str(gate["calibration_transfer_pass"]),
                    (
                        f"mean TV {gate['surrogate_mean_tv_to_reference']:.3f} "
                        "versus <=0.15 requirement"
                    ),
                ],
                [
                    "manual-backend quality",
                    str(gate["manual_quality_pass"]),
                    (
                        f"mean critic ratio {gate['manual_mean_critic_ratio']:.3f} "
                        "versus >=0.90 requirement"
                    ),
                ],
            ],
        ),
        "",
        "The 0.15 TV and 0.90 manual-quality checks are conservative calibration "
        "requirements, not evidence of quantum advantage by themselves. The current "
        "results support a working safe pipeline and a promising surrogate tradeoff, "
        "but not transfer of that tradeoff to the geometry-driven quantum backend.",
        "",
        "## Sampler-in-the-loop training",
        "",
        _table(
            ["training", "episode return", "raw feasible"],
            [
                [
                    row["model"],
                    _fmt(
                        row["episode_return_mean"],
                        row["episode_return_ci95"],
                    ),
                    _fmt(
                        row["raw_feasible_rate_mean"],
                        row["raw_feasible_rate_ci95"],
                    ),
                ]
                for row in sorted(training_summary, key=lambda item: item["model"])
            ],
        ),
        "",
        "Both variants use environment rewards only. The sampler-in-loop actor is "
        "updated by paired SPSA rollouts through sampling, repair, Q reranking, and "
        "dispatch execution; it receives no MILP or heuristic action labels.",
        "",
        f"The paired sampler-in-loop minus direct-actor return difference was "
        f"{paired_mean:.3f} ± {paired_ci:.3f} across held-out seeds.",
        "",
        "## Eight-qubit dynamic pipeline proof",
        "",
        _table(
            ["backend", "episode return", "raw feasible", "emulator ms/step"],
            [
                [
                    row["backend"],
                    _fmt(
                        row["episode_return_mean"],
                        row["episode_return_ci95"],
                    ),
                    _fmt(
                        row["raw_feasible_rate_mean"],
                        row["raw_feasible_rate_ci95"],
                    ),
                    _fmt(
                        row["mean_emulator_step_ms_mean"],
                        row["mean_emulator_step_ms_ci95"],
                        2,
                    ),
                ]
                for row in sorted(pipeline_summary, key=lambda item: item["backend"])
            ],
        ),
        "",
        "All executed actions passed the authoritative application-graph safety "
        "check. Emulator timings are not hardware latency estimates.",
        "",
        "## Small-backend calibration",
        "",
        _table(
            [
                "n",
                "backend",
                "critic ratio",
                "best-of-16 ratio",
                "p(eps=5%)",
                "K95",
                "K=16 coverage",
                "raw feasible",
                "TV to reference",
                "evolution ms",
            ],
            [
                [
                    str(row["n_jobs"]),
                    row["backend"],
                    _fmt(
                        row["critic_selected_ratio_mean"],
                        row["critic_selected_ratio_ci95"],
                    ),
                    _fmt(
                        row["expected_best_k_ratio_mean"],
                        row["expected_best_k_ratio_ci95"],
                    ),
                    _fmt(
                        row["p_epsilon_05_mean"],
                        row["p_epsilon_05_ci95"],
                    ),
                    str(shots_for_95_percent(row["p_epsilon_05_mean"])),
                    _fmt(
                        row["coverage_k_epsilon_05_mean"],
                        row["coverage_k_epsilon_05_ci95"],
                    ),
                    _fmt(
                        row["raw_feasible_probability_mean"],
                        row["raw_feasible_probability_ci95"],
                    ),
                    _fmt(
                        row["total_variation_to_reference_mean"],
                        row["total_variation_to_reference_ci95"],
                    ),
                    _fmt(
                        row["backend_evolution_ms_mean"],
                        row["backend_evolution_ms_ci95"],
                        1,
                    ),
                ]
                for row in sorted(
                    calibration_summary,
                    key=lambda item: (item["n_jobs"], item["backend"]),
                )
            ],
        ),
        "",
        "Dense evolution is intentionally omitted at n=12 because its dense "
        "2^n-by-2^n matrix exponentials are not a responsible repeated benchmark. "
        "At n=12, QuTiP is the distribution-distance reference.",
        "",
        "## Selection split: best surrogate phase-map regimes",
        "",
        _table(
            [
                "regime",
                "critic ratio",
                "best batch",
                "eps-5% hit",
                "candidate p(eps)",
                "K95",
                "diversity",
                "raw feasible",
                "latency ms",
            ],
            [
                [
                    row["regime"],
                    _fmt(
                        row["critic_selected_reward_ratio_mean"],
                        row["critic_selected_reward_ratio_ci95"],
                    ),
                    _fmt(
                        row["best_batch_reward_ratio_mean"],
                        row["best_batch_reward_ratio_ci95"],
                    ),
                    _fmt(
                        row["hit_epsilon_05_mean"],
                        row["hit_epsilon_05_ci95"],
                    ),
                    _fmt(
                        row["p_epsilon_05_mean"],
                        row["p_epsilon_05_ci95"],
                    ),
                    str(shots_for_95_percent(row["p_epsilon_05_mean"])),
                    _fmt(
                        row["candidate_hamming_diversity_mean"],
                        row["candidate_hamming_diversity_ci95"],
                    ),
                    _fmt(
                        row["raw_feasible_rate_mean"],
                        row["raw_feasible_rate_ci95"],
                    ),
                    _fmt(
                        row["end_to_end_latency_ms_mean"],
                        row["end_to_end_latency_ms_ci95"],
                        2,
                    ),
                ]
                for row in top_surrogate
            ],
        ),
        "",
        "## Independent confirmation: best-regime baseline comparison",
        "",
        _table(
            [
                "method",
                "critic ratio",
                "best batch",
                "eps-5% hit",
                "K95",
                "diversity",
                "latency ms",
            ],
            [
                [
                    method,
                    _fmt(
                        row["critic_selected_reward_ratio_mean"],
                        row["critic_selected_reward_ratio_ci95"],
                    ),
                    _fmt(
                        row["best_batch_reward_ratio_mean"],
                        row["best_batch_reward_ratio_ci95"],
                    ),
                    _fmt(
                        row["hit_epsilon_05_mean"],
                        row["hit_epsilon_05_ci95"],
                    ),
                    str(shots_for_95_percent(row["p_epsilon_05_mean"])),
                    _fmt(
                        row["candidate_hamming_diversity_mean"],
                        row["candidate_hamming_diversity_ci95"],
                    ),
                    _fmt(
                        row["end_to_end_latency_ms_mean"],
                        row["end_to_end_latency_ms_ci95"],
                        2,
                    ),
                ]
                for method, row in sorted(gate["peers"].items())
            ],
        ),
        "",
        "## Interpretation and next gate",
        "",
        "The pipeline proof answers whether quantum-backend candidates can flow "
        "through repair, learned reranking, and environment execution safely. The "
        "conditional gate is stricter: it asks whether the calibrated proposal "
        "path preserves acceptable return and beats tuned classical searches on a "
        "preregistered end-to-end dimension.",
        "",
        "Because the overall gate is on hold, the next experiment should optimize "
        "the physical geometry, pulse, and "
        "utility-to-detuning map against reward on training instances, then repeat "
        "the unchanged held-out gate. It would be invalid to redefine the gate "
        "after inspecting held-out results.",
        "",
        "## Limitations",
        "",
        "- All quantum-backend results are simulations, not QPU measurements.",
        "- The manual backend derives all C6/r^6 interactions from geometry, while "
        "dense and QuTiP receive exact graph edges; TV distance therefore includes "
        "a real model mismatch.",
        "- K95 assumes iid samples. Correlated hardware shots require an effective "
        "sample-size correction.",
        "- The 16-decision phase map uses a classical surrogate and supports regime "
        "selection only after small-backend calibration.",
        "- The workload remains synthetic and does not yet model Alibaba cumulative "
        "CPU and memory constraints.",
        "",
    ]
    return "\n".join(lines)


def run_conditional_study(
    config: ConditionalAdvantageConfig,
    output_json: Path,
    output_report: Path,
) -> dict[str, Any]:
    """Run the study, serialize raw records, and produce the Markdown report."""

    (
        direct_model,
        direct_history,
        sampler_model,
        sampler_history,
        training_records,
    ) = _run_training_comparison(config)
    calibration_records = _run_calibration(sampler_model, config)
    pipeline_records = _run_pipeline_proof(sampler_model, config)
    phase_records = _run_phase_search(sampler_model, config)
    results: dict[str, Any] = {
        "schema_version": 1,
        "config": asdict(config),
        "claim_boundary": (
            "Conditional end-to-end quantum-assisted pipeline study; no quantum "
            "supremacy or hardware latency claim."
        ),
        "direct_model": direct_model.to_dict(),
        "sampler_in_loop_model": sampler_model.to_dict(),
        "direct_training_history": direct_history,
        "sampler_training_history": sampler_history,
        "training_comparison_records": training_records,
        "calibration_records": calibration_records,
        "pipeline_records": pipeline_records,
        "phase_records": phase_records,
    }
    report = build_conditional_report(results)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    output_report.write_text(report, encoding="utf-8")
    return results
