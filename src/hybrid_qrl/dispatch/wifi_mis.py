"""Neutral-atom candidate sampling for public Wi-Fi MWIS scheduling.

The existing dispatch sampler already operates on a weighted unit-disk
conflict graph.  This module gives that abstraction a concrete, public-facing
interpretation: each vertex is a queued wireless transmission and each edge
means that the two transmissions should not share one airtime slot.  Selecting
the slot is therefore a maximum-weight independent-set (MWIS) problem.

The benchmark deliberately separates three claims:

* application validity: every executed candidate is an independent set;
* sampler opportunity: an ideal Rydberg distribution can beat low-compute
  randomized classical proposals at the same candidate count K; and
* physical advantage: this is *not* established without measured QPU data.

QuTiP performs the ideal quantum evolution.  Its runtime is classical emulator
cost and is never used as a neutral-atom hardware latency estimate.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from math import cos, pi, sin
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import numpy as np

from ..core import Action, ConflictGraph
from ..quantum import PulseSchedule, QuTiPRydbergSampler
from ..utilities.results import ResultWriter
from .baselines import repair_action


FAMILIES = ("bottleneck", "random", "crowded", "corridor")
METHODS = (
    "ideal_rydberg",
    "randomized_greedy",
    "one_swap_local_search",
    "simulated_annealing",
)


@dataclass(frozen=True)
class WifiMISConfig:
    """Configuration for the train/freeze/test Wi-Fi MWIS benchmark."""

    nodes: int = 12
    pulse_training_seeds: int = 12
    test_seeds: int = 40
    classical_probability_samples: int = 4_096
    candidate_budgets: tuple[int, ...] = (1, 2, 4, 8, 16)
    advantage_budget: int = 4
    near_optimal_epsilon: float = 0.01
    seed: int = 24_081

    def __post_init__(self) -> None:
        if self.nodes != 12:
            raise ValueError("the public-hotspot benchmark currently uses 12 nodes")
        if self.pulse_training_seeds < 4 or self.test_seeds < 10:
            raise ValueError("use at least 4 pulse seeds and 10 held-out seeds")
        if self.classical_probability_samples < 256:
            raise ValueError("classical_probability_samples must be at least 256")
        if not self.candidate_budgets or any(k <= 0 for k in self.candidate_budgets):
            raise ValueError("candidate budgets must be positive")
        if self.advantage_budget not in self.candidate_budgets:
            raise ValueError("advantage_budget must be one of candidate_budgets")
        if not 0.0 < self.near_optimal_epsilon < 1.0:
            raise ValueError("near_optimal_epsilon must lie in (0, 1)")


@dataclass(frozen=True)
class PulseRegime:
    """Named pulse used in the disjoint calibration split."""

    label: str
    schedule: PulseSchedule


@dataclass(frozen=True)
class WifiSlotInstance:
    """One busy-hotspot airtime frame and its MWIS representation."""

    family: str
    seed: int
    positions: np.ndarray
    interference_radius: float
    graph: ConflictGraph
    queue_packets: np.ndarray
    latency_slack_ms: np.ndarray
    service_priority: np.ndarray
    utilities: np.ndarray

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"unknown Wi-Fi instance family: {self.family}")
        n = self.graph.nodes
        for name in (
            "queue_packets",
            "latency_slack_ms",
            "service_priority",
            "utilities",
        ):
            if np.asarray(getattr(self, name)).shape != (n,):
                raise ValueError(f"{name} must contain one value per node")
        if np.asarray(self.positions).shape != (n, 2):
            raise ValueError("positions must have shape (nodes, 2)")


def _graph_from_positions(
    positions: np.ndarray, interference_radius: float
) -> ConflictGraph:
    edges = tuple(
        (left, right)
        for left in range(len(positions))
        for right in range(left + 1, len(positions))
        if float(np.linalg.norm(positions[left] - positions[right]))
        <= interference_radius
    )
    return ConflictGraph(nodes=len(positions), edges=edges)


def _radius_for_density(positions: np.ndarray, density: float) -> float:
    distances = np.asarray(
        [
            float(np.linalg.norm(positions[left] - positions[right]))
            for left in range(len(positions))
            for right in range(left + 1, len(positions))
        ]
    )
    edge_count = max(1, int(round(density * len(distances))))
    return float(np.partition(distances, edge_count - 1)[edge_count - 1])


def _wifi_metadata(
    rng: np.random.Generator, nodes: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    queue = rng.integers(4, 65, size=nodes).astype(float)
    slack = rng.uniform(8.0, 120.0, size=nodes)
    service = rng.choice((0.35, 0.65, 1.0), size=nodes, p=(0.45, 0.35, 0.20))
    queue_signal = np.log1p(queue) / np.log(65.0)
    urgency = 1.0 - np.clip(slack / 120.0, 0.0, 1.0)
    raw = 0.45 * queue_signal + 0.35 * urgency + 0.20 * service
    utilities = 0.45 + raw
    return queue, slack, service, utilities


def make_wifi_instance(
    family: str, seed: int, nodes: int = 12
) -> WifiSlotInstance:
    """Create a reproducible unit-disk wireless-interference instance.

    ``bottleneck`` models two contention domains in which one urgent central
    transmitter blocks five mutually compatible edge transmitters.  It is a
    realistic failure mode for priority-first local scheduling: a locally
    attractive transmission can sacrifice substantial spatial reuse.
    ``random``, ``crowded``, and ``corridor`` are non-targeted controls.  The
    corridor family places devices along two sides of a hallway, creating a
    ladder-like local-interference graph common in apartments, hotels, and
    campus buildings.
    """

    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}")
    if nodes != 12:
        raise ValueError("the current geometry is defined for 12 nodes")
    rng = np.random.default_rng(seed)
    queue, slack, service, utilities = _wifi_metadata(rng, nodes)

    if family == "bottleneck":
        centers = np.asarray(((0.28, 0.50), (0.72, 0.50)), dtype=float)
        cluster_radius = 0.104
        points: list[tuple[float, float]] = []
        for center in centers:
            points.append((float(center[0]), float(center[1])))
            rotation = rng.uniform(-0.025, 0.025)
            for leaf in range(5):
                angle = 2.0 * pi * leaf / 5.0 + rotation
                points.append(
                    (
                        float(center[0] + cluster_radius * cos(angle)),
                        float(center[1] + cluster_radius * sin(angle)),
                    )
                )
        positions = np.asarray(points)
        interference_radius = 0.119
        # Central voice/video uploads look best one at a time; collectively,
        # the edge devices carry more useful airtime.  Jitter prevents ties.
        for hub in (0, 6):
            queue[hub] = rng.integers(52, 65)
            slack[hub] = rng.uniform(8.0, 16.0)
            service[hub] = 1.0
            utilities[hub] = rng.uniform(1.28, 1.38)
            leaves = slice(hub + 1, hub + 6)
            queue[leaves] = rng.integers(14, 39, size=5)
            slack[leaves] = rng.uniform(24.0, 75.0, size=5)
            service[leaves] = rng.choice((0.35, 0.65), size=5)
            utilities[leaves] = rng.uniform(0.58, 0.78, size=5)
    elif family == "random":
        positions = rng.uniform(0.05, 0.95, size=(nodes, 2))
        interference_radius = _radius_for_density(positions, 0.18)
    elif family == "crowded":
        centers = np.asarray(((0.30, 0.35), (0.70, 0.38), (0.50, 0.72)))
        assignment = rng.integers(0, len(centers), size=nodes)
        positions = centers[assignment] + rng.normal(scale=0.12, size=(nodes, 2))
        positions = np.clip(positions, 0.03, 0.97)
        interference_radius = _radius_for_density(positions, 0.32)
    else:
        hallway_x = np.linspace(0.08, 0.92, 6)
        upper = np.column_stack((hallway_x, np.full(6, 0.45)))
        lower = np.column_stack((hallway_x, np.full(6, 0.55)))
        positions = np.vstack((upper, lower))
        positions += rng.normal(scale=0.004, size=positions.shape)
        positions = np.clip(positions, 0.03, 0.97)
        interference_radius = 0.18

    graph = _graph_from_positions(positions, interference_radius)
    return WifiSlotInstance(
        family=family,
        seed=seed,
        positions=positions,
        interference_radius=interference_radius,
        graph=graph,
        queue_packets=queue,
        latency_slack_ms=slack,
        service_priority=service,
        utilities=utilities,
    )


def action_value(action: Action, utilities: np.ndarray) -> float:
    """Return the total scheduled QoS utility."""

    return float(np.asarray(action, dtype=float) @ utilities)


def exact_mwis(instance: WifiSlotInstance) -> tuple[float, Action, int]:
    """Enumerate the 12-node test oracle and count optimal actions."""

    best_value = -np.inf
    best_action: Action = tuple(0 for _ in range(instance.graph.nodes))
    optimum_count = 0
    for basis in range(1 << instance.graph.nodes):
        action = tuple((basis >> node) & 1 for node in range(instance.graph.nodes))
        if not instance.graph.is_feasible(action):
            continue
        value = action_value(action, instance.utilities)
        if value > best_value + 1e-12:
            best_value = value
            best_action = action
            optimum_count = 1
        elif abs(value - best_value) <= 1e-12:
            optimum_count += 1
    return float(best_value), best_action, optimum_count


def _greedy_action(
    graph: ConflictGraph, utilities: np.ndarray, priorities: np.ndarray
) -> Action:
    adjacency = graph.adjacency()
    selected: set[int] = set()
    for raw_node in np.argsort(-priorities):
        node = int(raw_node)
        if not (selected & adjacency[node]):
            selected.add(node)
    return tuple(int(node in selected) for node in range(graph.nodes))


def _one_swap_action(
    graph: ConflictGraph,
    utilities: np.ndarray,
    rng: np.random.Generator,
) -> Action:
    priorities = np.log(np.maximum(utilities, 1e-12)) + rng.gumbel(size=graph.nodes)
    selected = set(np.flatnonzero(_greedy_action(graph, utilities, priorities)))
    adjacency = graph.adjacency()
    improved = True
    while improved:
        improved = False
        best_delta = 1e-12
        best_move: tuple[int | None, int] | None = None
        for raw_node in rng.permutation(graph.nodes):
            node = int(raw_node)
            if node in selected:
                continue
            conflicts = selected & adjacency[node]
            if len(conflicts) <= 1:
                removed = next(iter(conflicts)) if conflicts else None
                removed_value = 0.0 if removed is None else utilities[removed]
                delta = float(utilities[node] - removed_value)
                if delta > best_delta:
                    best_delta = delta
                    best_move = (removed, node)
        if best_move is not None:
            removed, added = best_move
            if removed is not None:
                selected.remove(removed)
            selected.add(added)
            improved = True
    return tuple(int(node in selected) for node in range(graph.nodes))


def _annealed_action(
    graph: ConflictGraph,
    utilities: np.ndarray,
    rng: np.random.Generator,
) -> Action:
    adjacency = graph.adjacency()
    selected: set[int] = set()
    moves = 12 * graph.nodes
    temperature_scale = float(np.mean(utilities))
    for move in range(moves):
        node = int(rng.integers(graph.nodes))
        temperature = max(0.015, 1.3 * (1.0 - move / moves)) * temperature_scale
        if node in selected:
            delta = -float(utilities[node])
            if rng.random() < np.exp(min(0.0, delta / temperature)):
                selected.remove(node)
            continue
        conflicts = selected & adjacency[node]
        delta = float(utilities[node] - sum(utilities[item] for item in conflicts))
        if delta >= 0.0 or rng.random() < np.exp(delta / temperature):
            selected.difference_update(conflicts)
            selected.add(node)
    return tuple(int(node in selected) for node in range(graph.nodes))


def _empirical_distribution(
    instance: WifiSlotInstance,
    method: str,
    samples: int,
    seed: int,
) -> tuple[dict[Action, float], float]:
    rng = np.random.default_rng(seed)
    counts: dict[Action, float] = defaultdict(float)
    start = perf_counter()
    for _ in range(samples):
        if method == "randomized_greedy":
            priorities = np.log(np.maximum(instance.utilities, 1e-12)) + rng.gumbel(
                size=instance.graph.nodes
            )
            action = _greedy_action(instance.graph, instance.utilities, priorities)
        elif method == "one_swap_local_search":
            action = _one_swap_action(instance.graph, instance.utilities, rng)
        elif method == "simulated_annealing":
            action = _annealed_action(instance.graph, instance.utilities, rng)
        else:
            raise ValueError(f"unsupported empirical method: {method}")
        counts[action] += 1.0
    elapsed_ms = (perf_counter() - start) * 1_000.0
    return ({action: count / samples for action, count in counts.items()}, elapsed_ms)


def _rydberg_distribution(
    instance: WifiSlotInstance,
    sampler: QuTiPRydbergSampler,
) -> tuple[dict[Action, float], float, float]:
    start = perf_counter()
    probabilities = sampler.probabilities(instance.utilities, instance.graph)
    elapsed_ms = (perf_counter() - start) * 1_000.0
    repaired: dict[Action, float] = defaultdict(float)
    raw_feasible = 0.0
    for basis, probability in enumerate(probabilities):
        if probability <= 0.0:
            continue
        raw = tuple(
            int(bit) for bit in format(basis, f"0{instance.graph.nodes}b")
        )
        if instance.graph.is_feasible(raw):
            raw_feasible += float(probability)
        safe = repair_action(raw, instance.graph, instance.utilities)
        repaired[safe] += float(probability)
    total = sum(repaired.values())
    normalized = {action: probability / total for action, probability in repaired.items()}
    return normalized, float(raw_feasible), elapsed_ms


def expected_hamming_diversity(distribution: dict[Action, float]) -> float:
    """Expected normalized Hamming distance between two independent shots."""

    if not distribution:
        return 0.0
    nodes = len(next(iter(distribution)))
    marginals = np.zeros(nodes, dtype=float)
    for action, probability in distribution.items():
        marginals += probability * np.asarray(action, dtype=float)
    return float(np.mean(2.0 * marginals * (1.0 - marginals)))


def distribution_metrics(
    distribution: dict[Action, float],
    instance: WifiSlotInstance,
    optimum: float,
    budgets: Iterable[int],
    epsilon: float,
) -> dict[str, Any]:
    """Calculate exact best-of-K metrics from a discrete distribution."""

    grouped: dict[float, float] = defaultdict(float)
    near_probability = 0.0
    for action, probability in distribution.items():
        ratio = action_value(action, instance.utilities) / max(optimum, 1e-12)
        grouped[float(ratio)] += probability
        if ratio >= 1.0 - epsilon:
            near_probability += probability

    budget_rows = []
    for budget in budgets:
        cumulative = 0.0
        previous_power = 0.0
        expected_best_ratio = 0.0
        for ratio, probability in sorted(grouped.items()):
            cumulative += probability
            current_power = cumulative**budget
            expected_best_ratio += ratio * (current_power - previous_power)
            previous_power = current_power
        budget_rows.append(
            {
                "candidates_k": int(budget),
                "expected_best_ratio": float(expected_best_ratio),
                "near_optimal_hit_probability": float(
                    1.0 - (1.0 - near_probability) ** budget
                ),
            }
        )

    expected_one_shot = float(
        sum(ratio * probability for ratio, probability in grouped.items())
    )
    return {
        "expected_one_shot_ratio": expected_one_shot,
        "single_shot_near_optimal_probability": float(near_probability),
        "expected_hamming_diversity": expected_hamming_diversity(distribution),
        "support_size": len(distribution),
        "budgets": budget_rows,
    }


def _beam_action(
    graph: ConflictGraph, utilities: np.ndarray, beam_width: int = 16
) -> Action:
    order = [int(node) for node in np.argsort(-utilities)]
    adjacency = graph.adjacency()
    beam: list[tuple[float, frozenset[int]]] = [(0.0, frozenset())]
    for node in order:
        expanded = list(beam)
        for value, selected in beam:
            if not (adjacency[node] & selected):
                expanded.append((value + float(utilities[node]), selected | {node}))
        expanded.sort(key=lambda item: item[0], reverse=True)
        unique: dict[frozenset[int], float] = {}
        for value, selected in expanded:
            unique.setdefault(selected, value)
            if len(unique) >= beam_width:
                break
        beam = [(value, selected) for selected, value in unique.items()]
    _, selected = max(beam, key=lambda item: item[0])
    return tuple(int(node in selected) for node in range(graph.nodes))


def _pulse_regimes() -> tuple[PulseRegime, ...]:
    return (
        PulseRegime(
            "short",
            PulseSchedule(
                duration=4.0,
                steps=16,
                omega_max=1.7,
                delta_start=-3.5,
                delta_end=4.0,
                blockade=16.0,
            ),
        ),
        PulseRegime(
            "balanced",
            PulseSchedule(
                duration=8.0,
                steps=24,
                omega_max=1.5,
                delta_start=-4.0,
                delta_end=4.5,
                blockade=18.0,
            ),
        ),
        PulseRegime(
            "adiabatic",
            PulseSchedule(
                duration=14.0,
                steps=32,
                omega_max=1.35,
                delta_start=-4.5,
                delta_end=5.0,
                blockade=20.0,
            ),
        ),
    )


def _mean_ci(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    mean = float(np.mean(array)) if len(array) else 0.0
    if len(array) < 2:
        return mean, 0.0
    return mean, float(1.96 * np.std(array, ddof=1) / np.sqrt(len(array)))


def _select_pulse(config: WifiMISConfig) -> tuple[PulseRegime, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for regime in _pulse_regimes():
        sampler = QuTiPRydbergSampler(schedule=regime.schedule, cache_decimals=None)
        ratios = []
        near_hits = []
        raw_feasible = []
        elapsed = []
        for index in range(config.pulse_training_seeds):
            seed = config.seed + index
            instance = make_wifi_instance("bottleneck", seed, config.nodes)
            optimum, _, _ = exact_mwis(instance)
            distribution, feasible, elapsed_ms = _rydberg_distribution(instance, sampler)
            metrics = distribution_metrics(
                distribution,
                instance,
                optimum,
                (config.advantage_budget,),
                config.near_optimal_epsilon,
            )
            ratios.append(metrics["budgets"][0]["expected_best_ratio"])
            near_hits.append(
                metrics["budgets"][0]["near_optimal_hit_probability"]
            )
            raw_feasible.append(feasible)
            elapsed.append(elapsed_ms)
        ratio_mean, ratio_ci = _mean_ci(ratios)
        hit_mean, hit_ci = _mean_ci(near_hits)
        feasible_mean, feasible_ci = _mean_ci(raw_feasible)
        time_mean, time_ci = _mean_ci(elapsed)
        rows.append(
            {
                "label": regime.label,
                "schedule": asdict(regime.schedule),
                "selection_objective_mean": ratio_mean,
                "selection_objective_ci95": ratio_ci,
                "near_optimal_hit_mean": hit_mean,
                "near_optimal_hit_ci95": hit_ci,
                "raw_feasible_mean": feasible_mean,
                "raw_feasible_ci95": feasible_ci,
                "classical_emulator_ms_mean": time_mean,
                "classical_emulator_ms_ci95": time_ci,
            }
        )
    best_objective = max(row["selection_objective_mean"] for row in rows)
    # A quality-only argmax tends to select a many-times slower pulse for
    # sub-per-mille gains that are immaterial to the declared 1% metric.
    # Freeze the fastest regime within 0.002 of the best training objective.
    eligible = [
        row
        for row in rows
        if row["selection_objective_mean"] >= best_objective - 0.002
    ]
    chosen_row = min(eligible, key=lambda row: row["classical_emulator_ms_mean"])
    chosen = next(regime for regime in _pulse_regimes() if regime.label == chosen_row["label"])
    return chosen, rows


def _instance_record(
    instance: WifiSlotInstance,
    sampler: QuTiPRydbergSampler,
    config: WifiMISConfig,
    record_index: int,
) -> dict[str, Any]:
    optimum, optimum_action, optimum_count = exact_mwis(instance)
    quantum_distribution, raw_feasible, quantum_ms = _rydberg_distribution(
        instance, sampler
    )
    method_records: dict[str, Any] = {
        "ideal_rydberg": distribution_metrics(
            quantum_distribution,
            instance,
            optimum,
            config.candidate_budgets,
            config.near_optimal_epsilon,
        )
        | {
            "raw_feasible_probability": raw_feasible,
            "classical_emulator_ms": quantum_ms,
            "evidence_type": "ideal_qutip_statevector",
        }
    }

    for method_index, method in enumerate(METHODS[1:], start=1):
        distribution, elapsed_ms = _empirical_distribution(
            instance,
            method,
            config.classical_probability_samples,
            config.seed + 1_000_003 * record_index + 10_007 * method_index,
        )
        method_records[method] = distribution_metrics(
            distribution,
            instance,
            optimum,
            config.candidate_budgets,
            config.near_optimal_epsilon,
        ) | {
            "raw_feasible_probability": 1.0,
            "empirical_pool_ms": elapsed_ms,
            "empirical_probability_samples": config.classical_probability_samples,
            "evidence_type": "classical_algorithm",
        }

    priority_action = _greedy_action(
        instance.graph, instance.utilities, instance.utilities
    )
    beam_action = _beam_action(instance.graph, instance.utilities, beam_width=16)
    density = 2.0 * len(instance.graph.edges) / (
        instance.graph.nodes * (instance.graph.nodes - 1)
    )
    return {
        "split": "held_out_test",
        "family": instance.family,
        "seed": instance.seed,
        "nodes": instance.graph.nodes,
        "edges": len(instance.graph.edges),
        "edge_density": density,
        "interference_radius": instance.interference_radius,
        "positions": instance.positions.tolist(),
        "queue_packets": instance.queue_packets.tolist(),
        "latency_slack_ms": instance.latency_slack_ms.tolist(),
        "service_priority": instance.service_priority.tolist(),
        "utilities": instance.utilities.tolist(),
        "graph_edges": [list(edge) for edge in instance.graph.edges],
        "exact_reference": {
            "optimal_value": optimum,
            "optimal_action": list(optimum_action),
            "degeneracy": optimum_count,
        },
        "deterministic_controls": {
            "priority_greedy": {
                "ratio": action_value(priority_action, instance.utilities) / optimum,
                "action": list(priority_action),
            },
            "beam_width_16": {
                "ratio": action_value(beam_action, instance.utilities) / optimum,
                "action": list(beam_action),
            },
            "exact_enumeration": {
                "ratio": 1.0,
                "action": list(optimum_action),
            },
        },
        "methods": method_records,
    }


def _summary(records: list[dict[str, Any]], config: WifiMISConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        family_records = [record for record in records if record["family"] == family]
        for method in METHODS:
            for budget in config.candidate_budgets:
                budget_rows = [
                    next(
                        item
                        for item in record["methods"][method]["budgets"]
                        if item["candidates_k"] == budget
                    )
                    for record in family_records
                ]
                ratios = [item["expected_best_ratio"] for item in budget_rows]
                hits = [item["near_optimal_hit_probability"] for item in budget_rows]
                ratio_mean, ratio_ci = _mean_ci(ratios)
                hit_mean, hit_ci = _mean_ci(hits)
                rows.append(
                    {
                        "family": family,
                        "method": method,
                        "candidates_k": budget,
                        "expected_best_ratio_mean": ratio_mean,
                        "expected_best_ratio_ci95": ratio_ci,
                        "near_optimal_hit_mean": hit_mean,
                        "near_optimal_hit_ci95": hit_ci,
                    }
                )
    return rows


def _paired_evidence(
    records: list[dict[str, Any]], config: WifiMISConfig
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        family_records = [record for record in records if record["family"] == family]
        for comparator in METHODS[1:]:
            for budget in config.candidate_budgets:
                deltas = []
                wins = []
                for record in family_records:
                    quantum = next(
                        row["expected_best_ratio"]
                        for row in record["methods"]["ideal_rydberg"]["budgets"]
                        if row["candidates_k"] == budget
                    )
                    classical = next(
                        row["expected_best_ratio"]
                        for row in record["methods"][comparator]["budgets"]
                        if row["candidates_k"] == budget
                    )
                    deltas.append(quantum - classical)
                    wins.append(float(quantum > classical + 1e-12))
                delta_mean, delta_ci = _mean_ci(deltas)
                win_mean, win_ci = _mean_ci(wins)
                rows.append(
                    {
                        "family": family,
                        "comparator": comparator,
                        "candidates_k": budget,
                        "quantum_minus_classical_mean": delta_mean,
                        "quantum_minus_classical_ci95": delta_ci,
                        "paired_win_rate": win_mean,
                        "paired_win_rate_ci95": win_ci,
                    }
                )
    return rows


def _deterministic_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for family in FAMILIES:
        family_records = [record for record in records if record["family"] == family]
        for method in ("priority_greedy", "beam_width_16", "exact_enumeration"):
            values = [record["deterministic_controls"][method]["ratio"] for record in family_records]
            mean, ci = _mean_ci(values)
            rows.append(
                {
                    "family": family,
                    "method": method,
                    "ratio_mean": mean,
                    "ratio_ci95": ci,
                }
            )
    return rows


def _find(
    rows: list[dict[str, Any]], **matching: Any
) -> dict[str, Any]:
    return next(
        row
        for row in rows
        if all(row.get(key) == value for key, value in matching.items())
    )


def _gates(results: dict[str, Any]) -> dict[str, Any]:
    config = results["config"]
    budget = config["advantage_budget"]
    paired = results["paired_evidence"]
    summary = results["summary"]
    vs_greedy = _find(
        paired,
        family="bottleneck",
        comparator="randomized_greedy",
        candidates_k=budget,
    )
    vs_local = _find(
        paired,
        family="bottleneck",
        comparator="one_swap_local_search",
        candidates_k=budget,
    )
    vs_annealing = _find(
        paired,
        family="bottleneck",
        comparator="simulated_annealing",
        candidates_k=budget,
    )
    quantum = _find(
        summary,
        family="bottleneck",
        method="ideal_rydberg",
        candidates_k=budget,
    )
    test_records = [
        record for record in results["records"] if record["family"] == "bottleneck"
    ]
    raw_values = [
        record["methods"]["ideal_rydberg"]["raw_feasible_probability"]
        for record in test_records
    ]
    raw_mean, raw_ci = _mean_ci(raw_values)
    checks = {
        "beats_randomized_greedy_paired_lower_ci": (
            vs_greedy["quantum_minus_classical_mean"]
            - vs_greedy["quantum_minus_classical_ci95"]
            > 0.0
        ),
        "beats_one_swap_local_paired_lower_ci": (
            vs_local["quantum_minus_classical_mean"]
            - vs_local["quantum_minus_classical_ci95"]
            > 0.0
        ),
        "ideal_ratio_at_least_0_95": quantum["expected_best_ratio_mean"] >= 0.95,
        "raw_feasible_mean_at_least_0_90": raw_mean >= 0.90,
    }
    return {
        "advantage_family": "bottleneck",
        "candidate_budget": budget,
        "limited_ideal_sampler_advantage_pass": all(checks.values()),
        "checks": checks,
        "quantum_minus_randomized_greedy": vs_greedy,
        "quantum_minus_one_swap_local": vs_local,
        "quantum_minus_simulated_annealing": vs_annealing,
        "raw_feasible_mean": raw_mean,
        "raw_feasible_ci95": raw_ci,
        "beats_simulated_annealing_paired_lower_ci": (
            vs_annealing["quantum_minus_classical_mean"]
            - vs_annealing["quantum_minus_classical_ci95"]
            > 0.0
        ),
        "physical_qpu_evidence": False,
        "end_to_end_latency_evidence": False,
    }


def run_wifi_mis_benchmark(
    config: WifiMISConfig,
    output_json: Path,
    output_report: Path,
    figure_dir: Path | None = None,
) -> dict[str, Any]:
    """Select a pulse on training seeds and evaluate frozen held-out frames."""

    chosen, pulse_rows = _select_pulse(config)
    sampler = QuTiPRydbergSampler(schedule=chosen.schedule, cache_decimals=None)
    records: list[dict[str, Any]] = []
    for family_index, family in enumerate(FAMILIES):
        for index in range(config.test_seeds):
            seed = config.seed + 100_000 + 10_000 * family_index + index
            instance = make_wifi_instance(family, seed, config.nodes)
            records.append(
                _instance_record(instance, sampler, config, len(records))
            )

    results: dict[str, Any] = {
        "study": "public_wifi_neutral_atom_mwis",
        "claim_boundary": (
            "Conditional ideal-sampler advantage on held-out Wi-Fi MWIS frames. "
            "QuTiP is a classical ideal quantum-system simulation; no hardware, "
            "wall-clock, energy, or asymptotic quantum advantage is claimed."
        ),
        "application": {
            "public_scenario": "busy public Wi-Fi airtime scheduling",
            "vertex": "one queued device transmission",
            "edge": "two transmissions that interfere in the same airtime slot",
            "weight": "queue, latency urgency, and service priority utility",
            "action": "a safe simultaneous-transmission independent set",
        },
        "config": asdict(config),
        "pulse_selection": {
            "split": "bottleneck_training_only",
            "quality_tolerance_from_best": 0.002,
            "tie_break": "fastest classical emulator within quality tolerance",
            "chosen_label": chosen.label,
            "chosen_schedule": asdict(chosen.schedule),
            "regimes": pulse_rows,
        },
        "records": records,
    }
    results["summary"] = _summary(records, config)
    results["paired_evidence"] = _paired_evidence(records, config)
    results["deterministic_summary"] = _deterministic_summary(records)
    results["gates"] = _gates(results)

    from ..utilities.reports.wifi_mis import render_wifi_mis_report

    ResultWriter().artifacts(
        json_path=output_json,
        report_path=output_report,
        payload=results,
        render_report=render_wifi_mis_report,
    )
    if figure_dir is not None:
        from ..utilities.wifi_mis_plotting import plot_wifi_mis_figures

        plot_wifi_mis_figures(results, figure_dir)
    return results


__all__ = [
    "FAMILIES",
    "METHODS",
    "PulseRegime",
    "WifiMISConfig",
    "WifiSlotInstance",
    "action_value",
    "distribution_metrics",
    "exact_mwis",
    "make_wifi_instance",
    "run_wifi_mis_benchmark",
]
