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

from collections import defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import sqrt
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix

from ..core import Action, ConflictGraph
from .azure_packing import (
    AzureTraceWindow,
    RESOURCE_NAMES,
    load_trace_windows,
    trace_profile,
)
from .backlog_benchmark import SamplerRegime
from .baselines import (
    ProposalConfig,
    generate_candidates,
    proposal_weights,
    repair_action,
    solve_weighted_independent_set,
)
from .environment import DispatchState, graph_from_positions


BUNDLE_METHODS = (
    "rydberg_geometry",
    "blockade_exact_graph",
    "repair_only",
    "beam_search",
    "randomized_greedy",
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


def generate_bundle_candidates(
    method: str,
    *,
    instance: AzureBundleInstance,
    model: AzureBundleValueModel,
    regime: SamplerRegime,
    candidates: int,
    seed: int,
) -> AzureBundleCandidateBatch:
    """Generate equal-K bundle proposals and repair exact set conflicts."""

    if method not in BUNDLE_METHODS:
        raise ValueError(f"unknown bundle method: {method}")
    start = perf_counter()
    rng = np.random.default_rng(seed)
    state = instance.state
    weights = proposal_weights(model, state)
    if method == "rydberg_geometry":
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


def _metric(row: dict[str, Any], metric: str) -> str:
    return (
        f"{row[f'{metric}_mean']:.3f} "
        f"+/- {row[f'{metric}_ci95']:.3f}"
    )


def _table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    header = "| " + " | ".join(
        value.ljust(widths[index]) for index, value in enumerate(headers)
    ) + " |"
    divider = "|" + "|".join("-" * (width + 2) for width in widths) + "|"
    body = [
        "| " + " | ".join(
            value.ljust(widths[index]) for index, value in enumerate(row)
        ) + " |"
        for row in rows
    ]
    return "\n".join((header, divider, *body))


def _build_gates(results: dict[str, Any]) -> dict[str, Any]:
    config = results["config"]
    target_nodes = max(config["bundle_nodes"])
    target_capacity = max(config["capacities"])
    target = _find(
        results["summary"],
        method="rydberg_geometry",
        bundle_nodes=target_nodes,
        capacity=target_capacity,
        k=16,
    )
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


def build_bundle_report(results: dict[str, Any]) -> str:
    """Render the concise benchmark report from the recorded results."""

    config = results["config"]
    summary = results["summary"]
    target_nodes = max(config["bundle_nodes"])
    capacity = max(config["capacities"])
    target = _find(
        summary,
        method="rydberg_geometry",
        bundle_nodes=target_nodes,
        capacity=capacity,
        k=16,
    )
    ideal = _find(
        summary,
        method="blockade_exact_graph",
        bundle_nodes=target_nodes,
        capacity=capacity,
        k=16,
    )
    beam = _find(
        summary,
        method="beam_search",
        bundle_nodes=target_nodes,
        capacity=capacity,
        k=16,
    )
    repair = _find(
        summary,
        method="repair_only",
        bundle_nodes=target_nodes,
        capacity=capacity,
        k=16,
    )
    paired = results["paired_comparisons"]
    full_capacity_k16 = [
        row
        for row in summary
        if row["capacity"] == capacity and row["k"] == 16
    ]
    k_sweep = [
        row
        for row in summary
        if row["bundle_nodes"] == target_nodes
        and row["capacity"] == capacity
    ]
    pipeline = "PASS" if results["gates"]["pipeline_pass"] else "HOLD"
    contribution = (
        "PASS" if results["gates"]["sampler_contribution_pass"] else "HOLD"
    )
    lines = [
        "# Azure bundle-conflict benchmark",
        "",
        f"## Trace-to-bundle pipeline: {pipeline}",
        "",
        f"## Geometry Rydberg proposal contribution: {contribution}",
        "",
        (
            f"Two hundred held-out Azure requests were converted into up to "
            f"{target_nodes} capacity-feasible `(machine, bundle)` nodes for "
            f"{config['machine_slots']} machine slots. At full per-machine "
            f"capacity and K=16, the geometry Rydberg surrogate achieved "
            f"{_metric(target, 'best_bundle_ratio')} of the bundle-library "
            f"MILP and {_metric(target, 'best_end_to_end_ratio')} of the "
            "direct job-by-machine MILP."
        ),
        "",
        (
            "The direct assignment reference used a five-second time limit: "
            f"{results['oracle_summary']['direct_exact']} of "
            f"{results['oracle_summary']['direct_states']} solves were exact, "
            "and every remaining conservative upper bound was within 1% of "
            "its incumbent (maximum gap "
            f"{results['oracle_summary']['direct_maximum_mip_gap']:.3%}). "
            "All end-to-end ratios use the upper bound, not the incumbent."
        ),
        "",
        (
            f"The bundle library itself covered "
            f"{_metric(target, 'library_coverage')} of the direct MILP. "
            f"Raw geometry-sampler feasibility was "
            f"{_metric(target, 'raw_feasible_rate')}, and authoritative "
            f"repair removed {_metric(target, 'repair_removed_fraction')} "
            "of selected bundle nodes."
        ),
        "",
        (
            "The same blockade schedule on the exact conflict graph reached "
            f"{_metric(ideal, 'best_end_to_end_ratio')}; beam reached "
            f"{_metric(beam, 'best_end_to_end_ratio')}; and the repair-only "
            f"control reached {_metric(repair, 'best_end_to_end_ratio')}. "
            "The ideal-graph path is an algorithmic diagnostic, not a "
            "hardware-compatible result."
        ),
        "",
        (
            "The paired geometry-minus-repair-only end-to-end difference was "
            f"{paired['geometry_minus_repair_end_to_end']['mean']:.4f} +/- "
            f"{paired['geometry_minus_repair_end_to_end']['ci95']:.4f}. "
            "The paired exact-graph-minus-geometry difference was "
            f"{paired['exact_graph_minus_geometry_end_to_end']['mean']:.4f} "
            f"+/- "
            f"{paired['exact_graph_minus_geometry_end_to_end']['ci95']:.4f}."
        ),
        "",
        "## What changed from the raw-job benchmark",
        "",
        (
            "Capacity is now enforced before sampling: every node is a complete "
            "four-resource-feasible machine configuration. Same-machine and "
            "shared-request conflicts are exactly pairwise, so an independent "
            "set is an authoritative safe allocation without cumulative "
            "capacity repair. A separate direct assignment MILP measures the "
            "solution-space loss introduced by the finite bundle library."
        ),
        "",
        "## Full-capacity results at K=16",
        "",
        _table(
            [
                "nodes",
                "method",
                "best/bundle",
                "best/direct",
                "rerank/direct",
                "raw feasible",
                "repair removed",
                "library coverage",
                "latency ms",
            ],
            [
                [
                    str(row["bundle_nodes"]),
                    str(row["method"]),
                    _metric(row, "best_bundle_ratio"),
                    _metric(row, "best_end_to_end_ratio"),
                    _metric(row, "reranked_end_to_end_ratio"),
                    _metric(row, "raw_feasible_rate"),
                    _metric(row, "repair_removed_fraction"),
                    _metric(row, "library_coverage"),
                    f"{row['proposal_latency_ms_mean']:.2f}",
                ]
                for row in full_capacity_k16
            ],
        ),
        "",
        "## K sweep at 100 bundle nodes and full capacity",
        "",
        _table(
            [
                "K",
                "method",
                "best/direct",
                "eps-5% coverage",
                "unique",
                "diversity",
                "latency ms",
            ],
            [
                [
                    str(row["k"]),
                    str(row["method"]),
                    _metric(row, "best_end_to_end_ratio"),
                    _metric(row, "epsilon_coverage"),
                    _metric(row, "unique_feasible"),
                    _metric(row, "diversity"),
                    f"{row['proposal_latency_ms_mean']:.2f}",
                ]
                for row in k_sweep
            ],
        ),
        "",
        "## Geometry transfer",
        "",
        (
            f"At the target setting, exact-to-unit-disk edge Jaccard was "
            f"{_metric(target, 'geometry_jaccard')}. The exact bundle graph "
            f"density was {_metric(target, 'exact_graph_density')}, versus "
            f"{_metric(target, 'physical_graph_density')} for the fitted "
            "two-dimensional blockade graph."
        ),
        "",
        (
            "Because this graph is dense, edge Jaccard alone is misleading. "
            "Only "
            f"{_metric(target, 'compatibility_recall')} of exact compatible "
            "bundle pairs remained non-edges in the physical graph; the "
            f"false-blockade rate was {_metric(target, 'false_blockade_rate')}."
        ),
        "",
        "## Gates",
        "",
        _table(
            ["pipeline check", "result"],
            [
                [key.replace("_", " "), str(value)]
                for key, value in results["gates"]["checks"].items()
            ],
        ),
        "",
        _table(
            ["sampler check", "result"],
            [
                [key.replace("_", " "), str(value)]
                for key, value in results["gates"][
                    "sampler_contribution_checks"
                ].items()
            ],
        ),
        "",
        "## Claim boundary",
        "",
        (
            "This experiment validates the bundle reformulation on the official "
            "trace. It does not reproduce Azure production scheduling, use a "
            "physical neutral-atom backend, measure hardware latency, or "
            "establish quantum advantage."
        ),
        "",
    ]
    return "\n".join(lines)


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
    regime = SamplerRegime(**stable["selected_regime"])
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
                    for method_index, method in enumerate(BUNDLE_METHODS):
                        batch = generate_bundle_candidates(
                            method,
                            instance=instance,
                            model=model,
                            regime=regime,
                            candidates=k,
                            seed=seed
                            + node_count * 10_007
                            + k * 503
                            + method_index * 100_003,
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
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    output_report.write_text(
        build_bundle_report(results), encoding="utf-8"
    )
    return results
