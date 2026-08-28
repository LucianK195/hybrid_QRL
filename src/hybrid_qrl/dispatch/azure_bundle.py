"""Configuration-based Azure packing for pairwise neutral-atom proposals.

The raw Azure benchmark represents every VM request by one binary variable.
Its four cumulative capacity constraints are not pairwise and therefore do
not match a Rydberg-blockade conflict graph.  This module implements the
configuration, or bundle, reformulation discussed in the project report:

1. a classical preprocessing stage generates complete capacity-feasible VM
   bundles for each machine slot;
2. one graph node represents ``(machine slot, bundle)``;
3. two nodes share an edge when they target the same machine or contain the
   same VM request; and
4. a sampler selects a weighted independent set of compatible bundles.

Every independent set is a safe allocation because capacity is verified
inside each bundle and the graph prevents both duplicate VM placement and
multiple configurations on one machine.  The benchmark separately reports
coverage of the generated bundle library against a direct job-by-machine
MILP.  This separation is essential: proposal quality inside a weak library
must not be mistaken for end-to-end packing quality.

The ``rydberg_geometry`` method uses a two-dimensional distance graph fitted
to the exact bundle conflict graph.  ``blockade_exact_graph`` is an
algorithmic upper-bound that runs the same stochastic blockade schedule on
the authoritative graph directly; it is not a hardware-realizable result.
Both remain classical surrogates, and neither measures QPU latency.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
from math import sqrt
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix

from ..core import Action, ConflictGraph
from ..utilities.reports.azure_bundle import (
    render_azure_bundle_report,
    render_external_portfolio_report,
)
from ..utilities.results import ResultWriter
from .azure_packing import (
    AzureTraceWindow,
    RESOURCE_NAMES,
    load_trace_windows,
    trace_profile,
)
from .backlog_benchmark import REGIME_GRID, SamplerRegime
from .baselines import (
    ProposalConfig,
    generate_candidates,
    proposal_weights,
    repair_action,
    solve_weighted_independent_set,
)
from .environment import DispatchState, graph_from_positions


BUNDLE_METHODS = (
    "layout_grover_qaoa",
    "randomized_layout",
    "deterministic_layout",
    "quantum_portfolio",
    "paired_grover_qaoa",
    "modular_xy_qaoa",
    "modular_rydberg",
    "rydberg_geometry",
    "blockade_exact_graph",
    "repair_only",
    "beam_search",
    "randomized_greedy",
)


@dataclass(frozen=True)
class QuantumWalkRegime:
    """Angles for a one-hot, constraint-preserving quantum-walk module."""

    gamma: float = 1.0
    beta: float = 1.0
    depth: int = 2

    def __post_init__(self) -> None:
        if not np.isfinite(self.gamma) or self.gamma <= 0:
            raise ValueError("quantum-walk gamma must be positive and finite")
        if not np.isfinite(self.beta) or self.beta <= 0:
            raise ValueError("quantum-walk beta must be positive and finite")
        if self.depth <= 0:
            raise ValueError("quantum-walk depth must be positive")


@dataclass(frozen=True)
class ExternalPortfolioConfig:
    """Frozen cross-generation validation for the quantum-module portfolio."""

    machine_ids: tuple[int, ...] = (11, 23)
    machine_slots: int = 8
    raw_jobs: int = 200
    bundle_nodes: int = 96
    capacity: float = 0.25
    candidates: int = 8
    train_windows: int = 30
    test_windows: int = 20
    train_day_start: float = 0.25
    train_day_end: float = 5.5
    test_day_start: float = 10.0
    test_day_end: float = 13.75
    direct_milp_time_limit_ms: float = 10_000.0
    direct_milp_retry_limit_ms: float = 30_000.0
    bundle_milp_time_limit_ms: float = 5_000.0
    quantum_walk_gamma: float = 0.8
    quantum_walk_beta: float = 1.2
    quantum_walk_depth: int = 3
    seed: int = 1_700_000

    def __post_init__(self) -> None:
        if not self.machine_ids or len(set(self.machine_ids)) != len(
            self.machine_ids
        ):
            raise ValueError("machine_ids must be non-empty and unique")
        if self.machine_slots < 2 or self.raw_jobs <= 0:
            raise ValueError("external portfolio dimensions are invalid")
        if self.bundle_nodes <= 0 or self.bundle_nodes % self.machine_slots:
            raise ValueError("bundle_nodes must divide across machine slots")
        if not 0.0 < self.capacity <= 1.0 or self.candidates <= 0:
            raise ValueError("external capacity or candidate budget is invalid")
        if self.train_windows <= 0 or self.test_windows <= 0:
            raise ValueError("external window counts must be positive")
        if self.train_day_end > self.test_day_start:
            raise ValueError("external train and test splits overlap")
        if (
            self.direct_milp_time_limit_ms <= 0
            or self.direct_milp_retry_limit_ms
            < self.direct_milp_time_limit_ms
            or self.bundle_milp_time_limit_ms <= 0
        ):
            raise ValueError("external MILP limits must be positive")
        QuantumWalkRegime(
            self.quantum_walk_gamma,
            self.quantum_walk_beta,
            self.quantum_walk_depth,
        )


@dataclass(frozen=True)
class AzureBundleConfig:
    """Frozen split, bundle-library, candidate, and reference settings.

    ``raw_jobs`` is deliberately distinct from ``bundle_nodes``.  The former
    is the number of Azure VM requests presented to the classical bundle
    generator, whereas the latter is the number of binary decisions exposed
    to the sampler.
    """

    machine_id: int = 16
    machine_slots: int = 2
    raw_jobs: int = 200
    bundle_nodes: tuple[int, ...] = (20, 40, 60, 80, 100)
    capacities: tuple[float, ...] = (0.75, 1.00)
    k_values: tuple[int, ...] = (4, 16, 64)
    train_windows: int = 30
    test_windows: int = 20
    train_day_start: float = 0.25
    train_day_end: float = 9.75
    test_day_start: float = 10.0
    test_day_end: float = 13.75
    epsilon: float = 0.05
    direct_milp_time_limit_ms: float = 5_000.0
    bundle_milp_time_limit_ms: float = 2_000.0
    sampler_regime: str = "stable"
    primary_method: str = "rydberg_geometry"
    comparison_method: str = "repair_only"
    primary_k: int = 16
    quantum_walk_gamma: float = 1.0
    quantum_walk_beta: float = 1.0
    quantum_walk_depth: int = 2
    seed: int = 710_923

    def __post_init__(self) -> None:
        if self.machine_id < 0:
            raise ValueError("machine_id must be non-negative")
        if self.machine_slots < 2:
            raise ValueError("machine_slots must be at least two")
        if self.raw_jobs < 20:
            raise ValueError("raw_jobs must be at least 20")
        if not self.bundle_nodes:
            raise ValueError("bundle_nodes must not be empty")
        if tuple(sorted(set(self.bundle_nodes))) != self.bundle_nodes:
            raise ValueError("bundle_nodes must be unique and increasing")
        if any(value <= 0 for value in self.bundle_nodes):
            raise ValueError("bundle node counts must be positive")
        if any(value % self.machine_slots for value in self.bundle_nodes):
            raise ValueError(
                "every bundle node count must divide evenly over machine slots"
            )
        if any(not 0.0 < value <= 1.0 for value in self.capacities):
            raise ValueError("capacity scales must lie in (0, 1]")
        if tuple(sorted(set(self.capacities))) != self.capacities:
            raise ValueError("capacities must be unique and increasing")
        if any(value <= 0 for value in self.k_values):
            raise ValueError("candidate budgets must be positive")
        if tuple(sorted(set(self.k_values))) != self.k_values:
            raise ValueError("k_values must be unique and increasing")
        if self.train_windows <= 0 or self.test_windows <= 0:
            raise ValueError("window counts must be positive")
        if not self.train_day_start < self.train_day_end:
            raise ValueError("invalid training interval")
        if not self.test_day_start < self.test_day_end:
            raise ValueError("invalid test interval")
        if self.train_day_end > self.test_day_start:
            raise ValueError("training and test intervals must not overlap")
        if not 0.0 < self.epsilon < 1.0:
            raise ValueError("epsilon must lie in (0, 1)")
        if (
            self.direct_milp_time_limit_ms <= 0
            or self.bundle_milp_time_limit_ms <= 0
        ):
            raise ValueError("MILP time limits must be positive")
        regime_names = {regime.name for regime in REGIME_GRID}
        if (
            self.sampler_regime != "stable"
            and self.sampler_regime not in regime_names
        ):
            raise ValueError("sampler_regime must be 'stable' or a known regime")
        if self.primary_method not in BUNDLE_METHODS:
            raise ValueError("primary_method must be a known bundle method")
        if self.comparison_method not in BUNDLE_METHODS:
            raise ValueError("comparison_method must be a known bundle method")
        if self.primary_method == self.comparison_method:
            raise ValueError("primary and comparison methods must differ")
        if self.primary_k not in self.k_values:
            raise ValueError("primary_k must be included in k_values")
        QuantumWalkRegime(
            gamma=self.quantum_walk_gamma,
            beta=self.quantum_walk_beta,
            depth=self.quantum_walk_depth,
        )


@dataclass(frozen=True)
class AzureBundleNode:
    """One complete feasible VM configuration assigned to one machine slot."""

    machine: int
    layout: int
    members: tuple[int, ...]
    usage: tuple[float, ...]
    utility: float
    predicted_utility: float


@dataclass(frozen=True)
class AzureBundleInstance:
    """Bundle graph plus the raw trace data required for authoritative checks."""

    window: AzureTraceWindow
    capacity: float
    nodes: tuple[AzureBundleNode, ...]
    state: DispatchState
    physical_graph: ConflictGraph
    geometry_jaccard: float
    compatibility_recall: float


@dataclass(frozen=True)
class AzureBundleCandidateBatch:
    """Repaired bundle candidates and diagnostics retained before reranking."""

    actions: tuple[Action, ...]
    raw_generated: int
    raw_feasible: int
    mean_removed_fraction: float
    elapsed_ms: float


@dataclass(frozen=True)
class AzureAssignmentReference:
    """Direct job-by-machine or restricted bundle MILP result."""

    objective: float
    incumbent_objective: float
    success: bool
    status: int
    mip_gap: float | None
    elapsed_ms: float

    @property
    def exact(self) -> bool:
        """Return whether HiGHS completed with zero reported relative gap."""

        return bool(
            self.success
            and self.mip_gap is not None
            and self.mip_gap <= 1e-9
        )

    @property
    def bounded(self) -> bool:
        """Return whether finite incumbent and conservative bound are present."""

        return bool(
            np.isfinite(self.objective)
            and np.isfinite(self.incumbent_objective)
            and self.objective + 1e-9 >= self.incumbent_objective
        )


@dataclass
class AzureBundleValueModel:
    """Reward-trained linear VM value head and additive bundle reranker.

    Training targets are the trace-derived offline VM utilities, never MILP
    actions or objectives.  At evaluation time, bundle proposal and reranking
    scores are sums of the predicted member values.
    """

    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray

    @staticmethod
    def job_features(window: AzureTraceWindow) -> np.ndarray:
        """Return priority, normalized lifetime, and four resource features."""

        lifetime = np.log1p(window.lifetime_days * 24.0) / np.log1p(
            90.0 * 24.0
        )
        return np.column_stack(
            (
                (window.priorities == 0).astype(float),
                lifetime,
                window.resources,
            )
        )

    def predict_jobs(self, window: AzureTraceWindow) -> np.ndarray:
        """Predict positive trace utility for every request in a window."""

        standardized = (self.job_features(window) - self.mean) / self.scale
        design = np.column_stack(
            (standardized, np.ones(window.jobs, dtype=float))
        )
        return np.maximum(design @ self.weights, 1e-6)

    def utility_logits(self, state: DispatchState) -> np.ndarray:
        """Return standardized learned bundle values for proposal encoding."""

        values = state.node_features[:, 0]
        deviation = float(np.std(values))
        if deviation <= 1e-12:
            return np.zeros(state.n_jobs, dtype=float)
        return (values - float(np.mean(values))) / deviation

    def q_value(self, state: DispatchState, action: Action) -> float:
        """Return the additive reward-trained value of a bundle allocation."""

        return float(
            np.asarray(action, dtype=float) @ state.node_features[:, 1]
        )

    def best_action(
        self,
        state: DispatchState,
        actions: list[Action],
    ) -> Action:
        """Select the candidate with the largest learned additive value."""

        if not actions:
            raise ValueError("cannot rerank an empty action collection")
        return max(actions, key=lambda action: self.q_value(state, action))

    def to_dict(self) -> dict[str, list[float]]:
        """Serialize the fitted reward-head parameters."""

        return {
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "weights": self.weights.tolist(),
        }


def _anchors(start: float, end: float, count: int) -> np.ndarray:
    if count == 1:
        return np.asarray(((start + end) / 2.0,), dtype=float)
    return np.linspace(start, end, count, dtype=float)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _method_seed_offset(method: str) -> int:
    """Return an order-independent deterministic RNG stream for one method."""

    digest = sha256(method.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def _ridge(
    features: np.ndarray,
    targets: np.ndarray,
    regularization: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale = np.where(scale > 1e-9, scale, 1.0)
    standardized = (features - mean) / scale
    design = np.column_stack(
        (standardized, np.ones(len(standardized), dtype=float))
    )
    penalty = np.eye(design.shape[1], dtype=float) * regularization
    penalty[-1, -1] = 0.0
    weights = np.linalg.solve(
        design.T @ design + penalty,
        design.T @ targets,
    )
    return mean, scale, weights


def fit_bundle_value_model(
    train_windows: list[AzureTraceWindow],
) -> tuple[AzureBundleValueModel, dict[str, Any]]:
    """Fit the reward-only linear value head on chronological training data."""

    features = np.vstack(
        [AzureBundleValueModel.job_features(window) for window in train_windows]
    )
    targets = np.concatenate([window.utility for window in train_windows])
    mean, scale, weights = _ridge(features, targets, regularization=1e-4)
    model = AzureBundleValueModel(mean=mean, scale=scale, weights=weights)
    fitted = np.concatenate(
        [model.predict_jobs(window) for window in train_windows]
    )
    residual = targets - fitted
    return model, {
        "training_requests": int(len(targets)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r2": float(
            1.0
            - np.sum(residual**2)
            / max(np.sum((targets - np.mean(targets)) ** 2), 1e-12)
        ),
        "uses_milp_labels": False,
    }


def _pack_partition(
    members: np.ndarray,
    resources: np.ndarray,
    capacity: float,
) -> tuple[int, ...]:
    usage = np.zeros(len(RESOURCE_NAMES), dtype=float)
    selected: list[int] = []
    for raw_index in members:
        index = int(raw_index)
        proposed = usage + resources[index]
        if np.all(proposed <= capacity + 1e-10):
            selected.append(index)
            usage = proposed
    return tuple(sorted(selected))


def generate_bundle_library(
    window: AzureTraceWindow,
    *,
    model: AzureBundleValueModel,
    machine_slots: int,
    capacity: float,
    target_nodes: int,
    seed: int,
) -> tuple[AzureBundleNode, ...]:
    """Generate nested, complete, capacity-feasible machine configurations.

    Each stochastic layout partitions requests between machine slots before
    packing.  Consequently, the bundles produced in the same layout are
    mutually compatible, ensuring that the restricted master problem always
    contains at least one complete multi-machine allocation.
    """

    if target_nodes % machine_slots:
        raise ValueError("target_nodes must divide evenly over machine slots")
    layouts = target_nodes // machine_slots
    rng = np.random.default_rng(seed)
    predicted = model.predict_jobs(window)
    normalized_resources = window.resources / max(capacity, 1e-12)
    burden = np.mean(normalized_resources, axis=1)
    density_score = predicted / (0.02 + burden)
    log_score = np.log(np.maximum(density_score, 1e-12))
    accepted: list[AzureBundleNode] = []
    seen: set[tuple[int, tuple[int, ...]]] = set()
    attempts = 0
    while len(accepted) < target_nodes:
        attempts += 1
        if attempts > max(2_000, layouts * 200):
            raise RuntimeError("could not generate enough unique bundle layouts")
        assignment_order = np.argsort(
            -(log_score + rng.gumbel(size=window.jobs))
        )
        offset = int(rng.integers(machine_slots))
        partitions = [
            assignment_order[
                (np.arange(window.jobs) + offset) % machine_slots == machine
            ]
            for machine in range(machine_slots)
        ]
        proposed: list[AzureBundleNode] = []
        valid_layout = True
        for machine, partition in enumerate(partitions):
            local_priority = (
                log_score[partition] + 0.75 * rng.gumbel(size=len(partition))
            )
            local_order = partition[np.argsort(-local_priority)]
            members = _pack_partition(
                local_order,
                window.resources,
                capacity,
            )
            key = (machine, members)
            if not members or key in seen:
                valid_layout = False
                break
            usage = tuple(
                float(value)
                for value in np.sum(window.resources[list(members)], axis=0)
            )
            proposed.append(
                AzureBundleNode(
                    machine=machine,
                    layout=len(accepted) // machine_slots,
                    members=members,
                    usage=usage,
                    utility=float(np.sum(window.utility[list(members)])),
                    predicted_utility=float(np.sum(predicted[list(members)])),
                )
            )
        if not valid_layout:
            continue
        accepted.extend(proposed)
        seen.update((node.machine, node.members) for node in proposed)
    return tuple(accepted)


def _bundle_graph(nodes: tuple[AzureBundleNode, ...]) -> ConflictGraph:
    member_sets = [set(node.members) for node in nodes]
    edges = tuple(
        (left, right)
        for left in range(len(nodes))
        for right in range(left + 1, len(nodes))
        if (
            nodes[left].machine == nodes[right].machine
            or not member_sets[left].isdisjoint(member_sets[right])
        )
    )
    return ConflictGraph(nodes=len(nodes), edges=edges)


def _fit_unit_disk_embedding(
    graph: ConflictGraph,
    seed: int,
) -> tuple[np.ndarray, float, ConflictGraph, float]:
    """Fit a two-dimensional distance graph and return its edge Jaccard.

    Classical multidimensional scaling embeds a target distance of one for
    edges and two for non-edges.  The blockade radius is selected on this
    instance to maximize edge-set Jaccard, making this an optimistic geometry
    calibration rather than an arbitrary layout.
    """

    count = graph.nodes
    adjacency = np.zeros((count, count), dtype=float)
    for left, right in graph.edges:
        adjacency[left, right] = 1.0
        adjacency[right, left] = 1.0
    target_distance = np.where(adjacency > 0.5, 1.0, 2.0)
    np.fill_diagonal(target_distance, 0.0)
    centering = np.eye(count) - np.ones((count, count)) / count
    gram = -0.5 * centering @ (target_distance**2) @ centering
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1][:2]
    positive = np.maximum(eigenvalues[order], 0.0)
    positions = eigenvectors[:, order] * np.sqrt(positive)
    if np.ptp(positions) <= 1e-12:
        positions = np.random.default_rng(seed).normal(
            scale=1e-4, size=(count, 2)
        )
    pair_distances = np.asarray(
        [
            np.linalg.norm(positions[left] - positions[right])
            for left in range(count)
            for right in range(left + 1, count)
        ],
        dtype=float,
    )
    exact_edges = set(graph.edges)
    labels = np.asarray(
        [
            int((left, right) in exact_edges)
            for left in range(count)
            for right in range(left + 1, count)
        ],
        dtype=int,
    )
    target_edges = int(np.sum(labels))
    thresholds = np.unique(pair_distances)
    best_jaccard = -1.0
    radius = 1e-12
    for threshold in thresholds:
        predicted = pair_distances <= threshold + 1e-12
        intersection = int(np.sum(labels[predicted]))
        union = target_edges + int(np.sum(predicted)) - intersection
        jaccard = intersection / max(union, 1)
        if jaccard > best_jaccard:
            best_jaccard = jaccard
            radius = float(threshold + 1e-12)
    physical = graph_from_positions(positions, radius)
    physical_edges = set(physical.edges)
    union = exact_edges | physical_edges
    measured = (
        len(exact_edges & physical_edges) / len(union) if union else 1.0
    )
    return positions, radius, physical, measured


def make_bundle_instance(
    window: AzureTraceWindow,
    nodes: tuple[AzureBundleNode, ...],
    *,
    capacity: float,
    seed: int,
) -> AzureBundleInstance:
    """Build the exact set-packing state and calibrated physical geometry."""

    graph = _bundle_graph(nodes)
    positions, radius, physical, jaccard = _fit_unit_disk_embedding(graph, seed)
    possible_pairs = len(nodes) * (len(nodes) - 1) // 2
    exact_edges = set(graph.edges)
    physical_edges = set(physical.edges)
    exact_compatible = possible_pairs - len(exact_edges)
    preserved_compatible = possible_pairs - len(exact_edges | physical_edges)
    compatibility_recall = (
        preserved_compatible / exact_compatible
        if exact_compatible
        else 1.0
    )
    features = np.asarray(
        [
            (
                node.predicted_utility,
                node.predicted_utility,
                len(node.members) / window.jobs,
                *node.usage,
                node.machine,
            )
            for node in nodes
        ],
        dtype=float,
    )
    values = np.asarray([node.utility for node in nodes], dtype=float)
    state = DispatchState(
        graph=graph,
        positions=positions,
        blockade_radius=radius,
        values=values,
        ages=np.zeros(len(nodes), dtype=int),
        deadlines=np.ones(len(nodes), dtype=int),
        remaining=np.ones(len(nodes), dtype=int),
        node_features=features,
        job_ids=np.arange(len(nodes), dtype=np.int64),
        step_index=0,
    )
    return AzureBundleInstance(
        window=window,
        capacity=capacity,
        nodes=nodes,
        state=state,
        physical_graph=physical,
        geometry_jaccard=jaccard,
        compatibility_recall=compatibility_recall,
    )


def bundle_allocation_feasible(
    instance: AzureBundleInstance,
    action: Action,
) -> bool:
    """Check capacity, unique-machine, and unique-request constraints."""

    if not instance.state.graph.is_valid_shape(action):
        return False
    machines: set[int] = set()
    requests: set[int] = set()
    for index, bit in enumerate(action):
        if not bit:
            continue
        node = instance.nodes[index]
        if node.machine in machines:
            return False
        if any(value > instance.capacity + 1e-10 for value in node.usage):
            return False
        if requests.intersection(node.members):
            return False
        machines.add(node.machine)
        requests.update(node.members)
    return True


def bundle_reward(instance: AzureBundleInstance, action: Action) -> float:
    """Return the trace utility of the distinct VMs in selected bundles."""

    return float(np.asarray(action, dtype=float) @ instance.state.values)


def allocated_requests(
    instance: AzureBundleInstance,
    action: Action,
) -> set[int]:
    """Return raw request indices covered by the selected configurations."""

    output: set[int] = set()
    for index, bit in enumerate(action):
        if bit:
            output.update(instance.nodes[index].members)
    return output


def solve_direct_assignment_milp(
    window: AzureTraceWindow,
    *,
    machine_slots: int,
    capacity: float,
    time_limit_ms: float,
) -> AzureAssignmentReference:
    """Solve the original job-by-machine vector-packing formulation."""

    start = perf_counter()
    jobs = window.jobs
    variables = machine_slots * jobs
    rows = jobs + machine_slots * len(RESOURCE_NAMES)
    matrix = np.zeros((rows, variables), dtype=float)
    upper = np.ones(rows, dtype=float)
    for job in range(jobs):
        for machine in range(machine_slots):
            matrix[job, machine * jobs + job] = 1.0
    row = jobs
    for machine in range(machine_slots):
        columns = slice(machine * jobs, (machine + 1) * jobs)
        for resource in range(len(RESOURCE_NAMES)):
            matrix[row, columns] = window.resources[:, resource]
            upper[row] = capacity
            row += 1
    result = milp(
        c=-np.tile(window.utility, machine_slots),
        integrality=np.ones(variables),
        bounds=Bounds(np.zeros(variables), np.ones(variables)),
        constraints=LinearConstraint(
            csr_matrix(matrix),
            np.full(rows, -np.inf),
            upper,
        ),
        options={
            "time_limit": time_limit_ms / 1_000.0,
            "mip_rel_gap": 0.0,
        },
    )
    incumbent = (
        0.0
        if result.x is None
        else float(
            np.dot(
                np.tile(window.utility, machine_slots),
                np.asarray(result.x) >= 0.5,
            )
        )
    )
    dual_value = getattr(result, "mip_dual_bound", None)
    conservative_upper = (
        incumbent
        if dual_value is None or not np.isfinite(float(dual_value))
        else max(incumbent, -float(dual_value))
    )
    gap_value = getattr(result, "mip_gap", None)
    return AzureAssignmentReference(
        objective=conservative_upper,
        incumbent_objective=incumbent,
        success=bool(result.success),
        status=int(result.status),
        mip_gap=None if gap_value is None else float(gap_value),
        elapsed_ms=(perf_counter() - start) * 1_000.0,
    )


def solve_bundle_milp(
    instance: AzureBundleInstance,
    time_limit_ms: float,
) -> AzureAssignmentReference:
    """Solve the restricted bundle-library weighted independent set."""

    solution = solve_weighted_independent_set(
        instance.state,
        instance.state.values,
        time_limit_ms=time_limit_ms,
    )
    return AzureAssignmentReference(
        objective=bundle_reward(instance, solution.action),
        incumbent_objective=bundle_reward(instance, solution.action),
        success=solution.success,
        status=solution.status,
        mip_gap=solution.mip_gap,
        elapsed_ms=solution.elapsed_ms,
    )


def _blockade_candidates(
    graph: ConflictGraph,
    weights: np.ndarray,
    candidates: int,
    regime: SamplerRegime,
    rng: np.random.Generator,
) -> list[Action]:
    """Run the project's stochastic blockade schedule on a supplied graph."""

    adjacency = graph.adjacency()
    cached = np.round(weights, decimals=2)
    if regime.utility_encoding == "mean":
        normalized = cached / (float(np.mean(cached)) + 1e-9)
    else:
        deviation = float(np.std(cached))
        if deviation <= 1e-9:
            normalized = np.ones_like(cached)
        else:
            standardized = (cached - float(np.mean(cached))) / deviation
            encoded = np.exp(
                np.clip(regime.detuning_gain * standardized, -6.0, 6.0)
            )
            normalized = encoded / (float(np.mean(encoded)) + 1e-9)
    sweep_count, beta_max, detuning_start, detuning_end = {
        "short": (2, 2.0, 1.15, 0.50),
        "balanced": (6, 5.0, 1.15, 0.50),
        "adiabatic": (12, 8.0, 1.15, 0.50),
        "extended": (16, 10.0, 1.30, 0.60),
    }[regime.pulse_schedule]
    output: list[Action] = []
    for _ in range(candidates):
        selected: set[int] = set()
        for sweep in range(sweep_count):
            beta = beta_max * (sweep + 1) / sweep_count
            progress = (sweep + 1) / sweep_count
            detuning = detuning_start + (
                detuning_end - detuning_start
            ) * progress
            for raw_node in rng.permutation(graph.nodes):
                node = int(raw_node)
                if selected & adjacency[node]:
                    selected.discard(node)
                    continue
                exponent = np.clip(
                    beta * (normalized[node] - detuning), -40.0, 40.0
                )
                probability = 1.0 / (1.0 + np.exp(-exponent))
                if rng.random() < probability:
                    selected.add(node)
                else:
                    selected.discard(node)
        output.append(
            tuple(int(node in selected) for node in range(graph.nodes))
        )
    return output


def _modular_blockade_candidates(
    instance: AzureBundleInstance,
    weights: np.ndarray,
    candidates: int,
    regime: SamplerRegime,
    rng: np.random.Generator,
) -> list[Action]:
    """Sample one capacity-feasible bundle per machine with feed-forward masks.

    The full bundle conflict graph is generally not a two-dimensional unit-disk
    graph: each machine contributes a clique, while shared VM requests create
    sparse cross-machine exclusions.  Embedding all of those edges in one
    register creates false blockade edges.  This model instead assigns one
    blockade clique to each machine slot.  After a clique is measured, bundles
    containing an already allocated request are masked from subsequent
    registers.  Alternating the machine order avoids privileging one slot.

    Every returned action is feasible before the authoritative repair layer.
    The routine remains a classical surrogate for sequential neutral-atom
    modules, not a hardware execution.
    """

    machine_nodes: dict[int, list[int]] = defaultdict(list)
    for index, node in enumerate(instance.nodes):
        machine_nodes[node.machine].append(index)
    machines = tuple(sorted(machine_nodes))
    output: list[Action] = []
    paired_order: list[int] | None = None
    for candidate_index in range(candidates):
        selected: list[int] = []
        allocated: set[int] = set()
        if candidate_index % 2 == 0:
            order = [int(value) for value in rng.permutation(machines)]
            paired_order = order
        else:
            # Antithetic ordering gives every early/late machine decision a
            # matched counterpart while exploring more than the two fixed
            # orders used by the initial prototype.
            order = list(reversed(paired_order or list(machines)))
        for machine in order:
            eligible = [
                index
                for index in machine_nodes[machine]
                if allocated.isdisjoint(instance.nodes[index].members)
            ]
            if not eligible:
                continue
            clique = ConflictGraph(
                nodes=len(eligible),
                edges=tuple(
                    (left, right)
                    for left in range(len(eligible))
                    for right in range(left + 1, len(eligible))
                ),
                max_selected=1,
            )
            local = _blockade_candidates(
                clique,
                weights[np.asarray(eligible, dtype=int)],
                1,
                regime,
                rng,
            )[0]
            chosen = [
                eligible[index] for index, bit in enumerate(local) if bit
            ]
            if not chosen:
                continue
            node_index = chosen[0]
            selected.append(node_index)
            allocated.update(instance.nodes[node_index].members)
        output.append(
            tuple(int(index in selected) for index in range(len(instance.nodes)))
        )
    return output


def _one_hot_xy_probabilities(
    weights: np.ndarray,
    regime: QuantumWalkRegime,
) -> np.ndarray:
    """Evolve an ideal XY/QAOA walk inside the one-excitation subspace.

    A machine with ``m`` eligible bundles is represented by ``m`` qubits, but
    the state remains in the one-hot subspace spanned by ``|100...0>`` through
    ``|000...1>``.  The cost phase encodes learned bundle utility and the XY
    complete-graph mixer preserves excitation number exactly.  Working in the
    reduced subspace is an exact ideal-state simulation of that module.
    """

    values = np.asarray(weights, dtype=float)
    count = len(values)
    if count == 0:
        return np.empty(0, dtype=float)
    if count == 1:
        return np.ones(1, dtype=float)
    deviation = float(np.std(values))
    encoded = (
        np.zeros(count, dtype=float)
        if deviation <= 1e-12
        else (values - float(np.mean(values))) / deviation
    )
    state = np.full(count, 1.0 / np.sqrt(count), dtype=np.complex128)
    orthogonal_phase = np.exp(1j * regime.beta / (count - 1))
    uniform_phase = np.exp(-1j * regime.beta)
    for layer in range(regime.depth):
        progress = (layer + 1) / regime.depth
        state *= np.exp(-1j * regime.gamma * progress * encoded)
        mean_amplitude = np.mean(state)
        state = (
            orthogonal_phase * state
            + (uniform_phase - orthogonal_phase) * mean_amplitude
        )
    probabilities = np.abs(state) ** 2
    return probabilities / float(np.sum(probabilities))


def _modular_xy_qaoa_candidates(
    instance: AzureBundleInstance,
    weights: np.ndarray,
    candidates: int,
    regime: QuantumWalkRegime,
    rng: np.random.Generator,
) -> list[Action]:
    """Compose one-hot XY/QAOA modules with classical feed-forward masks."""

    machine_nodes: dict[int, list[int]] = defaultdict(list)
    for index, node in enumerate(instance.nodes):
        machine_nodes[node.machine].append(index)
    machines = tuple(sorted(machine_nodes))
    output: list[Action] = []
    paired_order: list[int] | None = None
    for candidate_index in range(candidates):
        allocated: set[int] = set()
        selected: list[int] = []
        if candidate_index % 2 == 0:
            order = [int(value) for value in rng.permutation(machines)]
            paired_order = order
        else:
            order = list(reversed(paired_order or list(machines)))
        for machine in order:
            eligible = [
                index
                for index in machine_nodes[machine]
                if allocated.isdisjoint(instance.nodes[index].members)
            ]
            if not eligible:
                continue
            probabilities = _one_hot_xy_probabilities(
                weights[np.asarray(eligible, dtype=int)], regime
            )
            local = int(rng.choice(len(eligible), p=probabilities))
            node_index = eligible[local]
            selected.append(node_index)
            allocated.update(instance.nodes[node_index].members)
        output.append(
            tuple(int(index in selected) for index in range(len(instance.nodes)))
        )
    return output


def _grover_qaoa_probabilities(
    values: np.ndarray,
    regime: QuantumWalkRegime,
) -> np.ndarray:
    """Evolve uniform feasible states with cost and Grover-mixer phases."""

    costs = np.asarray(values, dtype=float)
    count = len(costs)
    if count == 0:
        return np.empty(0, dtype=float)
    if count == 1:
        return np.ones(1, dtype=float)
    deviation = float(np.std(costs))
    encoded = (
        np.zeros(count, dtype=float)
        if deviation <= 1e-12
        else (costs - float(np.mean(costs))) / deviation
    )
    state = np.full(count, 1.0 / np.sqrt(count), dtype=np.complex128)
    mixer_phase = np.exp(-1j * regime.beta)
    for layer in range(regime.depth):
        progress = (layer + 1) / regime.depth
        state *= np.exp(-1j * regime.gamma * progress * encoded)
        projection = np.mean(state)
        state += (mixer_phase - 1.0) * projection
    probabilities = np.abs(state) ** 2
    return probabilities / float(np.sum(probabilities))


def _paired_grover_qaoa_candidates(
    instance: AzureBundleInstance,
    weights: np.ndarray,
    candidates: int,
    regime: QuantumWalkRegime,
    rng: np.random.Generator,
) -> list[Action]:
    """Jointly sample compatible bundles for pairs of machine slots.

    Each module's computational basis is restricted to compatible pairs of
    complete machine bundles.  Cost phases encode their summed learned value;
    a Grover mixer creates interference across all feasible pair assignments.
    Thus cross-machine duplicate-request constraints enter the quantum state
    preparation rather than being left to repair after independent samples.
    """

    machine_nodes: dict[int, list[int]] = defaultdict(list)
    for index, node in enumerate(instance.nodes):
        machine_nodes[node.machine].append(index)
    machines = tuple(sorted(machine_nodes))
    output: list[Action] = []
    paired_order: list[int] | None = None
    member_sets = [set(node.members) for node in instance.nodes]
    for candidate_index in range(candidates):
        allocated: set[int] = set()
        selected: list[int] = []
        if candidate_index % 2 == 0:
            order = [int(value) for value in rng.permutation(machines)]
            paired_order = order
        else:
            order = list(reversed(paired_order or list(machines)))
        for offset in range(0, len(order), 2):
            first = order[offset]
            first_nodes = [
                index
                for index in machine_nodes[first]
                if allocated.isdisjoint(member_sets[index])
            ]
            if not first_nodes:
                continue
            if offset + 1 == len(order):
                probabilities = _one_hot_xy_probabilities(
                    weights[np.asarray(first_nodes, dtype=int)], regime
                )
                choice = first_nodes[
                    int(rng.choice(len(first_nodes), p=probabilities))
                ]
                selected.append(choice)
                allocated.update(member_sets[choice])
                continue
            second = order[offset + 1]
            second_nodes = [
                index
                for index in machine_nodes[second]
                if allocated.isdisjoint(member_sets[index])
            ]
            pairs = [
                (left, right)
                for left in first_nodes
                for right in second_nodes
                if member_sets[left].isdisjoint(member_sets[right])
            ]
            if not pairs:
                continue
            pair_values = np.asarray(
                [weights[left] + weights[right] for left, right in pairs]
            )
            probabilities = _grover_qaoa_probabilities(pair_values, regime)
            left, right = pairs[
                int(rng.choice(len(pairs), p=probabilities))
            ]
            selected.extend((left, right))
            allocated.update(member_sets[left])
            allocated.update(member_sets[right])
        output.append(
            tuple(int(index in selected) for index in range(len(instance.nodes)))
        )
    return output


def _layout_actions(instance: AzureBundleInstance) -> list[Action]:
    """Return complete feasible allocations retained from library layouts."""

    grouped: dict[int, list[int]] = defaultdict(list)
    for index, node in enumerate(instance.nodes):
        grouped[node.layout].append(index)
    actions: list[Action] = []
    for indices in grouped.values():
        action = tuple(
            int(index in indices) for index in range(len(instance.nodes))
        )
        if bundle_allocation_feasible(instance, action):
            actions.append(action)
    return actions


def _layout_grover_qaoa_candidates(
    instance: AzureBundleInstance,
    weights: np.ndarray,
    candidates: int,
    regime: QuantumWalkRegime,
    rng: np.random.Generator,
) -> list[Action]:
    """Sample complete feasible allocation layouts with a Grover mixer."""

    layouts = _layout_actions(instance)
    if not layouts:
        return [tuple(0 for _ in instance.nodes) for _ in range(candidates)]
    values = np.asarray(
        [np.dot(np.asarray(action, dtype=float), weights) for action in layouts]
    )
    probabilities = _grover_qaoa_probabilities(values, regime)
    choices = rng.choice(len(layouts), size=candidates, p=probabilities)
    return [layouts[int(index)] for index in choices]


def _randomized_layout_candidates(
    instance: AzureBundleInstance,
    candidates: int,
    rng: np.random.Generator,
) -> list[Action]:
    """Uniformly sample the same complete layouts as the quantum module."""

    layouts = _layout_actions(instance)
    if not layouts:
        return [tuple(0 for _ in instance.nodes) for _ in range(candidates)]
    choices = rng.integers(0, len(layouts), size=candidates)
    return [layouts[int(index)] for index in choices]


def _deterministic_layout_candidates(
    instance: AzureBundleInstance,
    weights: np.ndarray,
    candidates: int,
) -> list[Action]:
    """Return the highest learned-score complete layout as a hard control."""

    layouts = _layout_actions(instance)
    if not layouts:
        empty = tuple(0 for _ in instance.nodes)
        return [empty for _ in range(candidates)]
    best = max(
        layouts,
        key=lambda action: float(
            np.dot(np.asarray(action, dtype=float), weights)
        ),
    )
    return [best for _ in range(candidates)]


def _quantum_portfolio_candidates(
    instance: AzureBundleInstance,
    weights: np.ndarray,
    candidates: int,
    rydberg: SamplerRegime,
    quantum_walk: QuantumWalkRegime,
    rng: np.random.Generator,
) -> list[Action]:
    """Allocate a fixed shot budget across complementary quantum modules."""

    modules = (
        _layout_grover_qaoa_candidates,
        _paired_grover_qaoa_candidates,
        _modular_xy_qaoa_candidates,
        _modular_blockade_candidates,
    )
    counts = [candidates // len(modules) for _ in modules]
    for index in range(candidates % len(modules)):
        counts[index] += 1
    output: list[Action] = []
    for index, (module, count) in enumerate(zip(modules, counts, strict=True)):
        if count == 0:
            continue
        child = np.random.default_rng(int(rng.integers(0, 2**32 - 1)))
        if module is _modular_blockade_candidates:
            output.extend(module(instance, weights, count, rydberg, child))
        else:
            output.extend(
                module(instance, weights, count, quantum_walk, child)
            )
    return output


def generate_bundle_candidates(
    method: str,
    *,
    instance: AzureBundleInstance,
    model: AzureBundleValueModel,
    regime: SamplerRegime,
    candidates: int,
    seed: int,
    quantum_walk: QuantumWalkRegime = QuantumWalkRegime(),
) -> AzureBundleCandidateBatch:
    """Generate equal-K bundle proposals and repair exact set conflicts."""

    if method not in BUNDLE_METHODS:
        raise ValueError(f"unknown bundle method: {method}")
    start = perf_counter()
    rng = np.random.default_rng(seed)
    state = instance.state
    weights = proposal_weights(model, state)
    if method == "layout_grover_qaoa":
        raw = _layout_grover_qaoa_candidates(
            instance, weights, candidates, quantum_walk, rng
        )
    elif method == "randomized_layout":
        raw = _randomized_layout_candidates(instance, candidates, rng)
    elif method == "deterministic_layout":
        raw = _deterministic_layout_candidates(instance, weights, candidates)
    elif method == "quantum_portfolio":
        raw = _quantum_portfolio_candidates(
            instance,
            weights,
            candidates,
            regime,
            quantum_walk,
            rng,
        )
    elif method == "paired_grover_qaoa":
        raw = _paired_grover_qaoa_candidates(
            instance, weights, candidates, quantum_walk, rng
        )
    elif method == "modular_xy_qaoa":
        raw = _modular_xy_qaoa_candidates(
            instance, weights, candidates, quantum_walk, rng
        )
    elif method == "modular_rydberg":
        raw = _modular_blockade_candidates(
            instance, weights, candidates, regime, rng
        )
    elif method == "rydberg_geometry":
        raw = _blockade_candidates(
            instance.physical_graph, weights, candidates, regime, rng
        )
    elif method == "blockade_exact_graph":
        raw = _blockade_candidates(
            state.graph, weights, candidates, regime, rng
        )
    elif method == "repair_only":
        raw = [tuple(1 for _ in range(state.n_jobs)) for _ in range(candidates)]
    else:
        baseline = "beam_search" if method == "beam_search" else "greedy"
        batch = generate_candidates(
            baseline,
            state,
            model,
            ProposalConfig(
                candidates=candidates,
                max_runtime_ms=2_000.0,
                beam_width=max(128, 4 * candidates),
            ),
            rng,
        )
        raw = list(batch.repaired_actions)
    raw_feasible = sum(
        bundle_allocation_feasible(instance, action) for action in raw
    )
    repaired: list[Action] = []
    removal: list[float] = []
    for action in raw:
        safe = repair_action(action, state.graph, weights)
        selected = max(sum(action), 1)
        removal.append(max(sum(action) - sum(safe), 0) / selected)
        repaired.append(safe)
    unique = tuple(
        dict.fromkeys(
            action
            for action in repaired
            if bundle_allocation_feasible(instance, action)
        )
    )
    return AzureBundleCandidateBatch(
        actions=unique,
        raw_generated=len(raw),
        raw_feasible=raw_feasible,
        mean_removed_fraction=float(np.mean(removal) if removal else 0.0),
        elapsed_ms=(perf_counter() - start) * 1_000.0,
    )


def _diversity(actions: tuple[Action, ...]) -> float:
    if len(actions) < 2:
        return 0.0
    distances = [
        float(np.mean(np.asarray(actions[left]) != np.asarray(actions[right])))
        for left in range(len(actions))
        for right in range(left + 1, len(actions))
    ]
    return float(np.mean(distances))


def evaluate_bundle_candidates(
    *,
    batch: AzureBundleCandidateBatch,
    instance: AzureBundleInstance,
    model: AzureBundleValueModel,
    bundle_reference: float,
    direct_reference: float,
    epsilon: float,
) -> dict[str, Any]:
    """Measure restricted-master quality, end-to-end quality, and safety."""

    rewards = [
        bundle_reward(instance, action) for action in batch.actions
    ]
    if batch.actions:
        reranked = model.best_action(instance.state, list(batch.actions))
    else:
        reranked = tuple(0 for _ in range(instance.state.n_jobs))
    reranked_reward = bundle_reward(instance, reranked)
    ratios = [
        reward / max(bundle_reference, 1e-12) for reward in rewards
    ]
    return {
        "best_bundle_ratio": max(ratios, default=0.0),
        "reranked_bundle_ratio": (
            reranked_reward / max(bundle_reference, 1e-12)
        ),
        "best_end_to_end_ratio": (
            max(rewards, default=0.0) / max(direct_reference, 1e-12)
        ),
        "reranked_end_to_end_ratio": (
            reranked_reward / max(direct_reference, 1e-12)
        ),
        "epsilon_coverage": float(
            max(ratios, default=0.0) >= 1.0 - epsilon
        ),
        "p_epsilon": float(
            np.mean(np.asarray(ratios) >= 1.0 - epsilon)
            if ratios
            else 0.0
        ),
        "raw_feasible_rate": (
            batch.raw_feasible / max(batch.raw_generated, 1)
        ),
        "post_repair_feasible_rate": float(
            bool(batch.actions)
            and all(
                bundle_allocation_feasible(instance, action)
                for action in batch.actions
            )
        ),
        "repair_removed_fraction": batch.mean_removed_fraction,
        "unique_feasible": len(batch.actions),
        "diversity": _diversity(batch.actions),
        "selected_bundles": int(sum(reranked)),
        "allocated_vms": len(allocated_requests(instance, reranked)),
        "proposal_latency_ms": batch.elapsed_ms,
    }


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


SUMMARY_METRICS = (
    "best_bundle_ratio",
    "reranked_bundle_ratio",
    "best_end_to_end_ratio",
    "reranked_end_to_end_ratio",
    "epsilon_coverage",
    "p_epsilon",
    "raw_feasible_rate",
    "post_repair_feasible_rate",
    "repair_removed_fraction",
    "unique_feasible",
    "diversity",
    "selected_bundles",
    "allocated_vms",
    "proposal_latency_ms",
    "library_coverage",
    "geometry_jaccard",
    "compatibility_recall",
    "false_blockade_rate",
    "exact_graph_density",
    "physical_graph_density",
)


def _summarize(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            record["method"],
            record["bundle_nodes"],
            record["capacity"],
            record["k"],
        )
        grouped[key].append(record)
    summary: list[dict[str, Any]] = []
    for key, items in sorted(grouped.items(), key=lambda item: item[0]):
        row: dict[str, Any] = {
            "method": key[0],
            "bundle_nodes": key[1],
            "capacity": key[2],
            "k": key[3],
            "trials": len(items),
        }
        for metric in SUMMARY_METRICS:
            mean, ci95 = _mean_ci(float(item[metric]) for item in items)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci95"] = ci95
        summary.append(row)
    return summary


def _find(summary: list[dict[str, Any]], **matching: Any) -> dict[str, Any]:
    return next(
        row
        for row in summary
        if all(row[key] == value for key, value in matching.items())
    )


def _paired(
    records: list[dict[str, Any]],
    *,
    left: str,
    right: str,
    bundle_nodes: int,
    capacity: float,
    k: int,
    metric: str,
) -> dict[str, Any]:
    indexed = {
        (
            record["window_index"],
            record["method"],
            record["bundle_nodes"],
            record["capacity"],
            record["k"],
        ): record
        for record in records
    }
    windows = sorted({int(record["window_index"]) for record in records})
    differences = [
        float(
            indexed[(window, left, bundle_nodes, capacity, k)][metric]
            - indexed[(window, right, bundle_nodes, capacity, k)][metric]
        )
        for window in windows
    ]
    mean, ci95 = _mean_ci(differences)
    return {"mean": mean, "ci95": ci95, "trials": len(differences)}


def _paired_evidence(
    records: list[dict[str, Any]],
    *,
    left: str,
    right: str,
    bundle_nodes: int,
    capacity: float,
    k: int,
    metric: str,
    seed: int,
) -> dict[str, Any]:
    """Return paired effect, bootstrap interval, and exact sign-flip p-value."""

    indexed = {
        (
            record["window_index"],
            record["method"],
            record["bundle_nodes"],
            record["capacity"],
            record["k"],
        ): record
        for record in records
    }
    windows = sorted({int(record["window_index"]) for record in records})
    differences = np.asarray(
        [
            indexed[(window, left, bundle_nodes, capacity, k)][metric]
            - indexed[(window, right, bundle_nodes, capacity, k)][metric]
            for window in windows
        ],
        dtype=float,
    )
    rng = np.random.default_rng(seed)
    resampled = differences[
        rng.integers(0, len(differences), size=(100_000, len(differences)))
    ]
    bootstrap = np.quantile(np.mean(resampled, axis=1), (0.025, 0.975))
    observed = float(np.mean(differences))
    permutations = 1 << len(differences)
    exceedances = 0
    bit_positions = np.arange(len(differences), dtype=np.uint64)
    for start in range(0, permutations, 65_536):
        stop = min(start + 65_536, permutations)
        masks = np.arange(start, stop, dtype=np.uint64)[:, None]
        signs = 1.0 - 2.0 * ((masks >> bit_positions) & 1)
        permuted = signs @ differences / len(differences)
        exceedances += int(np.sum(permuted >= observed - 1e-15))
    return {
        "left": left,
        "right": right,
        "metric": metric,
        "k": k,
        "trials": len(differences),
        "mean": observed,
        "bootstrap_ci95_low": float(bootstrap[0]),
        "bootstrap_ci95_high": float(bootstrap[1]),
        "exact_one_sided_sign_flip_p": exceedances / permutations,
        "wins": int(np.sum(differences > 1e-12)),
        "ties": int(np.sum(np.abs(differences) <= 1e-12)),
        "losses": int(np.sum(differences < -1e-12)),
    }


def _candidate_efficiency(
    records: list[dict[str, Any]],
    *,
    method: str,
    bundle_nodes: int,
    capacity: float,
    k_values: Iterable[int],
    threshold: float = 0.95,
) -> dict[str, Any]:
    """Summarize the smallest recorded K reaching a quality threshold."""

    ordered_k = tuple(sorted(int(value) for value in k_values))
    relevant = [
        record
        for record in records
        if record["method"] == method
        and record["bundle_nodes"] == bundle_nodes
        and record["capacity"] == capacity
    ]
    windows = sorted({int(record["window_index"]) for record in relevant})
    indexed = {
        (int(record["window_index"]), int(record["k"])): record
        for record in relevant
    }
    first_hits: list[int | None] = []
    for window in windows:
        first_hits.append(
            next(
                (
                    k
                    for k in ordered_k
                    if indexed[(window, k)]["best_bundle_ratio"] >= threshold
                ),
                None,
            )
        )
    observed_hits = [value for value in first_hits if value is not None]
    return {
        "method": method,
        "quality_threshold": threshold,
        "trials": len(first_hits),
        "median_k": (
            float(np.median(observed_hits)) if observed_hits else None
        ),
        "hit_by_k": {
            str(k): float(
                np.mean(
                    [value is not None and value <= k for value in first_hits]
                )
            )
            for k in ordered_k
        },
        "not_reached_by_max_k": int(
            sum(value is None for value in first_hits)
        ),
    }


def _build_gates(results: dict[str, Any]) -> dict[str, Any]:
    config = results["config"]
    target_nodes = max(config["bundle_nodes"])
    target_capacity = max(config["capacities"])
    primary_method = config.get("primary_method", "rydberg_geometry")
    comparison_method = config.get("comparison_method", "repair_only")
    primary_k = int(config.get("primary_k", 16))
    target = _find(
        results["summary"],
        method=primary_method,
        bundle_nodes=target_nodes,
        capacity=target_capacity,
        k=primary_k,
    )
    if primary_method == "modular_rydberg":
        evidence = results["primary_evidence"]
        paired = evidence["paired_best_bundle"]
        primary_efficiency = evidence["candidate_efficiency"][primary_method]
        comparison_efficiency = evidence["candidate_efficiency"][
            comparison_method
        ]
        checks = {
            "all_direct_assignment_bounds_within_1pct": (
                results["oracle_summary"]["direct_gap_within_1pct_rate"] == 1.0
            ),
            "all_bundle_milp_exact": (
                results["oracle_summary"]["bundle_exact_rate"] == 1.0
            ),
            "all_executed_actions_safe": (
                min(
                    row["post_repair_feasible_rate_mean"]
                    for row in results["summary"]
                )
                == 1.0
            ),
        }
        proposal_checks = {
            "raw_feasible_rate_equals_1": (
                target["raw_feasible_rate_mean"] == 1.0
            ),
            "repair_removed_fraction_equals_0": (
                target["repair_removed_fraction_mean"] == 0.0
            ),
            "paired_bootstrap_lower_ci_above_comparison": (
                paired["bootstrap_ci95_low"] > 0.0
            ),
            "median_k95_below_comparison": (
                primary_efficiency["median_k"] is not None
                and comparison_efficiency["median_k"] is not None
                and primary_efficiency["median_k"]
                < comparison_efficiency["median_k"]
            ),
        }
        return {
            "pipeline_pass": all(checks.values()),
            "sampler_contribution_pass": all(proposal_checks.values()),
            "hardware_claim_pass": False,
            "checks": checks,
            "sampler_contribution_checks": proposal_checks,
            "hardware_note": (
                "The modular Rydberg path is a sequential classical surrogate; "
                "no physical QPU distribution or latency was measured."
            ),
        }
    paired = results["paired_comparisons"][
        "geometry_minus_repair_end_to_end"
    ]
    checks = {
        "all_direct_assignment_bounds_within_1pct": (
            results["oracle_summary"]["direct_gap_within_1pct_rate"] == 1.0
        ),
        "all_bundle_milp_exact": (
            results["oracle_summary"]["bundle_exact_rate"] == 1.0
        ),
        "all_executed_actions_safe": (
            min(
                row["post_repair_feasible_rate_mean"]
                for row in results["summary"]
            )
            == 1.0
        ),
        "library_coverage_lower_ci_at_least_090": (
            target["library_coverage_mean"]
            - target["library_coverage_ci95"]
            >= 0.90
        ),
        "geometry_end_to_end_lower_ci_at_least_090": (
            target["best_end_to_end_ratio_mean"]
            - target["best_end_to_end_ratio_ci95"]
            >= 0.90
        ),
    }
    proposal_checks = {
        "raw_feasible_rate_at_least_020": (
            target["raw_feasible_rate_mean"] >= 0.20
        ),
        "repair_removed_fraction_at_most_010": (
            target["repair_removed_fraction_mean"] <= 0.10
        ),
        "positive_paired_lower_ci_over_repair_only": (
            paired["mean"] - paired["ci95"] > 0.0
        ),
        "compatible_pair_recall_at_least_080": (
            target["compatibility_recall_mean"] >= 0.80
        ),
    }
    return {
        "pipeline_pass": all(checks.values()),
        "sampler_contribution_pass": all(proposal_checks.values()),
        "hardware_claim_pass": False,
        "checks": checks,
        "sampler_contribution_checks": proposal_checks,
        "hardware_note": (
            "The Rydberg paths are classical surrogates; no physical "
            "distribution-transfer calibration or QPU latency was measured."
        ),
    }


def run_azure_bundle_benchmark(
    *,
    sqlite_path: Path,
    stable_results_path: Path,
    output_json: Path,
    output_report: Path,
    config: AzureBundleConfig = AzureBundleConfig(),
) -> dict[str, Any]:
    """Train on early trace windows and evaluate held-out bundle graphs."""

    stable = json.loads(stable_results_path.read_text(encoding="utf-8"))
    if config.sampler_regime == "stable":
        regime = SamplerRegime(**stable["selected_regime"])
    else:
        regime = next(
            item for item in REGIME_GRID if item.name == config.sampler_regime
        )
    train = load_trace_windows(
        sqlite_path,
        machine_id=config.machine_id,
        anchors=_anchors(
            config.train_day_start,
            config.train_day_end,
            config.train_windows,
        ),
        end_day=config.test_day_start,
        jobs=config.raw_jobs,
    )
    test = load_trace_windows(
        sqlite_path,
        machine_id=config.machine_id,
        anchors=_anchors(
            config.test_day_start,
            config.test_day_end,
            config.test_windows,
        ),
        end_day=14.0,
        jobs=config.raw_jobs,
    )
    model, training = fit_bundle_value_model(train)
    records: list[dict[str, Any]] = []
    direct_oracles: list[dict[str, Any]] = []
    bundle_oracles: list[dict[str, Any]] = []
    window_summaries: list[dict[str, Any]] = []
    max_nodes = max(config.bundle_nodes)
    for window_index, window in enumerate(test):
        window_summaries.append(
            {
                "window_index": window_index,
                "anchor_day": window.anchor_day,
                "high_priority_fraction": float(
                    np.mean(window.priorities == 0)
                ),
                "unique_tenants": int(len(np.unique(window.tenant_ids))),
                "vm_ids_sha256": sha256(window.vm_ids.tobytes()).hexdigest(),
            }
        )
        for capacity in config.capacities:
            seed = (
                config.seed
                + window_index * 1_000_003
                + int(capacity * 10_000)
            )
            library = generate_bundle_library(
                window,
                model=model,
                machine_slots=config.machine_slots,
                capacity=capacity,
                target_nodes=max_nodes,
                seed=seed,
            )
            direct = solve_direct_assignment_milp(
                window,
                machine_slots=config.machine_slots,
                capacity=capacity,
                time_limit_ms=config.direct_milp_time_limit_ms,
            )
            direct_oracles.append(
                {
                    "window_index": window_index,
                    "capacity": capacity,
                    "objective": direct.objective,
                    "incumbent_objective": direct.incumbent_objective,
                    "exact": direct.exact,
                    "bounded": direct.bounded,
                    "status": direct.status,
                    "mip_gap": direct.mip_gap,
                    "latency_ms": direct.elapsed_ms,
                }
            )
            for node_count in config.bundle_nodes:
                nodes = library[:node_count]
                instance = make_bundle_instance(
                    window,
                    nodes,
                    capacity=capacity,
                    seed=seed + node_count * 101,
                )
                bundle = solve_bundle_milp(
                    instance, config.bundle_milp_time_limit_ms
                )
                possible_edges = node_count * (node_count - 1) / 2
                graph_density = (
                    len(instance.state.graph.edges) / max(possible_edges, 1)
                )
                physical_density = (
                    len(instance.physical_graph.edges) / max(possible_edges, 1)
                )
                library_coverage = (
                    bundle.objective / max(direct.objective, 1e-12)
                )
                bundle_oracles.append(
                    {
                        "window_index": window_index,
                        "capacity": capacity,
                        "bundle_nodes": node_count,
                        "objective": bundle.objective,
                        "incumbent_objective": bundle.incumbent_objective,
                        "exact": bundle.exact,
                        "status": bundle.status,
                        "mip_gap": bundle.mip_gap,
                        "latency_ms": bundle.elapsed_ms,
                        "library_coverage": library_coverage,
                    }
                )
                common = {
                    "window_index": window_index,
                    "anchor_day": window.anchor_day,
                    "raw_jobs": config.raw_jobs,
                    "machine_slots": config.machine_slots,
                    "bundle_nodes": node_count,
                    "capacity": capacity,
                    "direct_reference": direct.objective,
                    "direct_incumbent": direct.incumbent_objective,
                    "bundle_reference": bundle.objective,
                    "library_coverage": library_coverage,
                    "direct_oracle_exact": direct.exact,
                    "bundle_oracle_exact": bundle.exact,
                    "geometry_jaccard": instance.geometry_jaccard,
                    "compatibility_recall": instance.compatibility_recall,
                    "false_blockade_rate": (
                        1.0 - instance.compatibility_recall
                    ),
                    "exact_graph_density": graph_density,
                    "physical_graph_density": physical_density,
                    "mean_bundle_size": float(
                        np.mean([len(node.members) for node in nodes])
                    ),
                    "all_bundles_capacity_feasible": float(
                        all(
                            all(
                                value <= capacity + 1e-10
                                for value in node.usage
                            )
                            for node in nodes
                        )
                    ),
                }
                for k in config.k_values:
                    for method in BUNDLE_METHODS:
                        batch = generate_bundle_candidates(
                            method,
                            instance=instance,
                            model=model,
                            regime=regime,
                            candidates=k,
                            seed=seed
                            + node_count * 10_007
                            + k * 503
                            + _method_seed_offset(method),
                            quantum_walk=QuantumWalkRegime(
                                gamma=config.quantum_walk_gamma,
                                beta=config.quantum_walk_beta,
                                depth=config.quantum_walk_depth,
                            ),
                        )
                        records.append(
                            {
                                **common,
                                "method": method,
                                "k": k,
                                **evaluate_bundle_candidates(
                                    batch=batch,
                                    instance=instance,
                                    model=model,
                                    bundle_reference=bundle.objective,
                                    direct_reference=direct.objective,
                                    epsilon=config.epsilon,
                                ),
                            }
                        )
    summary = _summarize(records)
    target_nodes = max(config.bundle_nodes)
    target_capacity = max(config.capacities)
    paired = {
        "geometry_minus_repair_end_to_end": _paired(
            records,
            left="rydberg_geometry",
            right="repair_only",
            bundle_nodes=target_nodes,
            capacity=target_capacity,
            k=16,
            metric="best_end_to_end_ratio",
        ),
        "exact_graph_minus_geometry_end_to_end": _paired(
            records,
            left="blockade_exact_graph",
            right="rydberg_geometry",
            bundle_nodes=target_nodes,
            capacity=target_capacity,
            k=16,
            metric="best_end_to_end_ratio",
        ),
    }
    primary_evidence = {
        "paired_best_bundle": _paired_evidence(
            records,
            left=config.primary_method,
            right=config.comparison_method,
            bundle_nodes=target_nodes,
            capacity=target_capacity,
            k=config.primary_k,
            metric="best_bundle_ratio",
            seed=config.seed + 8_081,
        ),
        "paired_end_to_end": _paired_evidence(
            records,
            left=config.primary_method,
            right=config.comparison_method,
            bundle_nodes=target_nodes,
            capacity=target_capacity,
            k=config.primary_k,
            metric="best_end_to_end_ratio",
            seed=config.seed + 8_083,
        ),
        "candidate_efficiency": {
            method: _candidate_efficiency(
                records,
                method=method,
                bundle_nodes=target_nodes,
                capacity=target_capacity,
                k_values=config.k_values,
            )
            for method in (config.primary_method, config.comparison_method)
        },
    }
    direct_latencies = np.asarray(
        [row["latency_ms"] for row in direct_oracles], dtype=float
    )
    bundle_latencies = np.asarray(
        [row["latency_ms"] for row in bundle_oracles], dtype=float
    )
    results: dict[str, Any] = {
        "schema_version": 1,
        "study": "azure_packing_bundle_conflict",
        "claim_boundary": (
            "Official trace, configuration-based offline benchmark, classical "
            "Rydberg surrogates, no measured QPU or latency claim."
        ),
        "config": asdict(config),
        "source": {
            "official_documentation": (
                "https://github.com/Azure/AzurePublicDataset/"
                "blob/master/AzureTracesForPacking2020.md"
            ),
            "sqlite_path": str(sqlite_path.resolve()),
            "sqlite_sha256": _file_sha256(sqlite_path),
            "stable_results_path": str(stable_results_path.resolve()),
        },
        "trace_profile": trace_profile(sqlite_path, config.machine_id),
        "selected_regime": asdict(regime),
        "training": training,
        "model": model.to_dict(),
        "test_windows": window_summaries,
        "records": records,
        "direct_oracles": direct_oracles,
        "bundle_oracles": bundle_oracles,
        "summary": summary,
        "paired_comparisons": paired,
        "primary_evidence": primary_evidence,
        "oracle_summary": {
            "direct_states": len(direct_oracles),
            "direct_exact": int(sum(row["exact"] for row in direct_oracles)),
            "direct_exact_rate": float(
                np.mean([row["exact"] for row in direct_oracles])
            ),
            "direct_gap_within_1pct": int(
                sum(
                    row["bounded"]
                    and row["mip_gap"] is not None
                    and row["mip_gap"] <= 0.01
                    for row in direct_oracles
                )
            ),
            "direct_gap_within_1pct_rate": float(
                np.mean(
                    [
                        row["bounded"]
                        and row["mip_gap"] is not None
                        and row["mip_gap"] <= 0.01
                        for row in direct_oracles
                    ]
                )
            ),
            "direct_maximum_mip_gap": float(
                max(
                    row["mip_gap"]
                    for row in direct_oracles
                    if row["mip_gap"] is not None
                )
            ),
            "direct_latency_ms_mean": float(np.mean(direct_latencies)),
            "direct_latency_ms_p95": float(
                np.quantile(direct_latencies, 0.95)
            ),
            "bundle_states": len(bundle_oracles),
            "bundle_exact": int(sum(row["exact"] for row in bundle_oracles)),
            "bundle_exact_rate": float(
                np.mean([row["exact"] for row in bundle_oracles])
            ),
            "bundle_latency_ms_mean": float(np.mean(bundle_latencies)),
            "bundle_latency_ms_p95": float(
                np.quantile(bundle_latencies, 0.95)
            ),
        },
    }
    results["gates"] = _build_gates(results)
    ResultWriter().artifacts(
        json_path=output_json,
        report_path=output_report,
        payload=results,
        render_report=render_azure_bundle_report,
    )
    return results


EXTERNAL_PORTFOLIO_METHODS = (
    "quantum_portfolio",
    "randomized_greedy",
    "randomized_layout",
    "deterministic_layout",
    "beam_search",
)


def run_external_portfolio_benchmark(
    *,
    sqlite_path: Path,
    stable_results_path: Path,
    output_json: Path,
    output_report: Path,
    config: ExternalPortfolioConfig = ExternalPortfolioConfig(),
) -> dict[str, Any]:
    """Validate a frozen quantum-module portfolio on unseen generations."""

    stable = json.loads(stable_results_path.read_text(encoding="utf-8"))
    selected = SamplerRegime(**stable["selected_regime"])
    # The adiabatic pulse was selected on generation 16 validation days.  The
    # external generations remain untouched by all model and pulse choices.
    rydberg = SamplerRegime(
        "standardized-050-adiabatic",
        "standardized",
        0.5,
        "adiabatic",
    )
    quantum_walk = QuantumWalkRegime(
        config.quantum_walk_gamma,
        config.quantum_walk_beta,
        config.quantum_walk_depth,
    )
    records: list[dict[str, Any]] = []
    source_profiles: dict[str, Any] = {}
    training_summaries: dict[str, Any] = {}
    oracle_records: list[dict[str, Any]] = []
    for machine_id in config.machine_ids:
        train = load_trace_windows(
            sqlite_path,
            machine_id=machine_id,
            anchors=_anchors(
                config.train_day_start,
                config.train_day_end,
                config.train_windows,
            ),
            end_day=6.0,
            jobs=config.raw_jobs,
        )
        test = load_trace_windows(
            sqlite_path,
            machine_id=machine_id,
            anchors=_anchors(
                config.test_day_start,
                config.test_day_end,
                config.test_windows,
            ),
            end_day=14.0,
            jobs=config.raw_jobs,
        )
        model, training = fit_bundle_value_model(train)
        training_summaries[str(machine_id)] = training
        source_profiles[str(machine_id)] = trace_profile(sqlite_path, machine_id)
        for window_index, window in enumerate(test):
            seed = (
                config.seed
                + machine_id * 10_000_019
                + window_index * 1_000_003
            )
            nodes = generate_bundle_library(
                window,
                model=model,
                machine_slots=config.machine_slots,
                capacity=config.capacity,
                target_nodes=config.bundle_nodes,
                seed=seed,
            )
            instance = make_bundle_instance(
                window,
                nodes,
                capacity=config.capacity,
                seed=seed + 97,
            )
            bundle = solve_bundle_milp(
                instance, config.bundle_milp_time_limit_ms
            )
            direct = solve_direct_assignment_milp(
                window,
                machine_slots=config.machine_slots,
                capacity=config.capacity,
                time_limit_ms=config.direct_milp_time_limit_ms,
            )
            if (
                not direct.bounded
                or direct.mip_gap is None
                or direct.mip_gap > 0.01
            ):
                direct = solve_direct_assignment_milp(
                    window,
                    machine_slots=config.machine_slots,
                    capacity=config.capacity,
                    time_limit_ms=config.direct_milp_retry_limit_ms,
                )
            oracle_records.append(
                {
                    "machine_id": machine_id,
                    "window_index": window_index,
                    "bundle_exact": bundle.exact,
                    "bundle_reference": bundle.objective,
                    "direct_reference": direct.objective,
                    "direct_incumbent": direct.incumbent_objective,
                    "direct_gap": direct.mip_gap,
                    "direct_bounded": direct.bounded,
                }
            )
            common = {
                "machine_id": machine_id,
                "window_index": window_index,
                "anchor_day": window.anchor_day,
                "k": config.candidates,
                "bundle_nodes": config.bundle_nodes,
                "capacity": config.capacity,
                "bundle_reference": bundle.objective,
                "direct_reference": direct.objective,
                "library_coverage": (
                    bundle.objective / max(direct.objective, 1e-12)
                ),
            }
            for method in EXTERNAL_PORTFOLIO_METHODS:
                batch = generate_bundle_candidates(
                    method,
                    instance=instance,
                    model=model,
                    regime=rydberg,
                    candidates=config.candidates,
                    seed=(
                        seed
                        + config.candidates * 503
                        + _method_seed_offset(method)
                    ),
                    quantum_walk=quantum_walk,
                )
                records.append(
                    {
                        **common,
                        "method": method,
                        **evaluate_bundle_candidates(
                            batch=batch,
                            instance=instance,
                            model=model,
                            bundle_reference=bundle.objective,
                            direct_reference=direct.objective,
                            epsilon=0.05,
                        ),
                    }
                )
    summary: list[dict[str, Any]] = []
    comparisons: dict[str, Any] = {}
    for machine_id in config.machine_ids:
        generation_records = [
            row for row in records if row["machine_id"] == machine_id
        ]
        for method in EXTERNAL_PORTFOLIO_METHODS:
            method_records = [
                row for row in generation_records if row["method"] == method
            ]
            row: dict[str, Any] = {
                "machine_id": machine_id,
                "method": method,
                "trials": len(method_records),
            }
            for metric in (
                "best_bundle_ratio",
                "best_end_to_end_ratio",
                "epsilon_coverage",
                "raw_feasible_rate",
                "repair_removed_fraction",
                "proposal_latency_ms",
            ):
                mean, ci95 = _mean_ci(item[metric] for item in method_records)
                row[f"{metric}_mean"] = mean
                row[f"{metric}_ci95"] = ci95
            summary.append(row)
        comparisons[str(machine_id)] = {
            method: {
                "bundle": _paired_evidence(
                    generation_records,
                    left="quantum_portfolio",
                    right=method,
                    bundle_nodes=config.bundle_nodes,
                    capacity=config.capacity,
                    k=config.candidates,
                    metric="best_bundle_ratio",
                    seed=config.seed + machine_id + _method_seed_offset(method),
                ),
                "end_to_end": _paired_evidence(
                    generation_records,
                    left="quantum_portfolio",
                    right=method,
                    bundle_nodes=config.bundle_nodes,
                    capacity=config.capacity,
                    k=config.candidates,
                    metric="best_end_to_end_ratio",
                    seed=(
                        config.seed
                        + machine_id
                        + _method_seed_offset(method)
                        + 17
                    ),
                ),
            }
            for method in EXTERNAL_PORTFOLIO_METHODS
            if method != "quantum_portfolio"
        }
    oracle_checks = {
        "all_bundle_milp_exact": all(
            row["bundle_exact"] for row in oracle_records
        ),
        "all_direct_bounds_within_1pct": all(
            row["direct_bounded"]
            and row["direct_gap"] is not None
            and row["direct_gap"] <= 0.01
            for row in oracle_records
        ),
    }
    sampler_checks = {
        "portfolio_raw_feasible_equals_1": all(
            row["raw_feasible_rate"] == 1.0
            for row in records
            if row["method"] == "quantum_portfolio"
        ),
        "portfolio_repair_fraction_equals_0": all(
            row["repair_removed_fraction"] == 0.0
            for row in records
            if row["method"] == "quantum_portfolio"
        ),
        "beats_randomized_greedy_on_every_dataset": all(
            comparisons[str(machine_id)]["randomized_greedy"]["bundle"][
                "bootstrap_ci95_low"
            ]
            > 0.0
            for machine_id in config.machine_ids
        ),
        "beats_randomized_layout_on_every_dataset": all(
            comparisons[str(machine_id)]["randomized_layout"]["bundle"][
                "bootstrap_ci95_low"
            ]
            > 0.0
            for machine_id in config.machine_ids
        ),
    }
    strong_checks = {
        "beats_deterministic_layout_on_every_dataset": all(
            comparisons[str(machine_id)]["deterministic_layout"]["bundle"][
                "bootstrap_ci95_low"
            ]
            > 0.0
            for machine_id in config.machine_ids
        ),
        "beats_beam_on_every_dataset": all(
            comparisons[str(machine_id)]["beam_search"]["bundle"][
                "bootstrap_ci95_low"
            ]
            > 0.0
            for machine_id in config.machine_ids
        ),
    }
    results: dict[str, Any] = {
        "schema_version": 1,
        "study": "azure_external_quantum_portfolio",
        "claim_boundary": (
            "External Azure hardware-generation validation of ideal quantum-"
            "module surrogates; no physical QPU or hardware timing claim."
        ),
        "config": asdict(config),
        "source": {
            "sqlite_path": str(sqlite_path.resolve()),
            "sqlite_sha256": _file_sha256(sqlite_path),
            "stable_results_path": str(stable_results_path.resolve()),
        },
        "source_profiles": source_profiles,
        "training": training_summaries,
        "generation16_selected_regime": asdict(selected),
        "portfolio_rydberg_regime": asdict(rydberg),
        "quantum_walk_regime": asdict(quantum_walk),
        "records": records,
        "oracles": oracle_records,
        "summary": summary,
        "comparisons": comparisons,
        "gates": {
            "pipeline_pass": all(oracle_checks.values()),
            "potential_advantage_pass": (
                all(sampler_checks.values())
            ),
            "strong_classical_advantage_pass": all(strong_checks.values()),
            "hardware_claim_pass": False,
            "oracle_checks": oracle_checks,
            "sampler_checks": sampler_checks,
            "strong_baseline_checks": strong_checks,
        },
    }
    ResultWriter().artifacts(
        json_path=output_json,
        report_path=output_report,
        payload=results,
        render_report=render_external_portfolio_report,
    )
    return results
