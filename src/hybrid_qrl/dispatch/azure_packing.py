"""Trace-driven Azure VM packing benchmark with cumulative safety constraints.

The official Azure Packing 2020 trace supplies VM arrivals, priorities,
lifetimes, and normalized resource requirements for compatible hardware
generations. This module turns chronological request windows into binary
admission problems with CPU, memory, SSD, and NIC capacity constraints.

The neutral-atom path remains the project's scalable classical Rydberg
surrogate. Its geometric conflict graph is only a proposal approximation.
Every candidate is repaired and verified against the authoritative cumulative
resource constraints before critic reranking.
"""

from __future__ import annotations

import json
import sqlite3
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
from ..utilities.reports.azure_packing import render_azure_packing_report
from ..utilities.results import ResultWriter
from .backlog_benchmark import SamplerRegime
from .baselines import generate_candidates, proposal_weights
from .environment import DispatchState, graph_from_positions


RESOURCE_NAMES = ("cpu", "memory", "ssd", "nic")
METHODS = (
    "rydberg_surrogate",
    "deterministic_repair",
    "beam_search",
    "randomized_greedy",
    "random_shooting",
)


@dataclass(frozen=True)
class AzurePackingConfig:
    """Frozen data split, decision sizes, budgets, and reference settings."""

    machine_id: int = 16
    train_windows: int = 30
    test_windows: int = 20
    train_day_start: float = 0.25
    train_day_end: float = 9.75
    test_day_start: float = 10.0
    test_day_end: float = 13.75
    sizes: tuple[int, ...] = (20, 40, 60, 80, 100)
    capacities: tuple[float, ...] = (0.50, 0.75, 1.00)
    k_values: tuple[int, ...] = (4, 16, 64)
    epsilon: float = 0.05
    oracle_time_limit_ms: float = 1_000.0
    critic_actions_per_state: int = 24
    seed: int = 620_417

    def __post_init__(self) -> None:
        if self.machine_id < 0:
            raise ValueError("machine_id must be non-negative")
        if self.train_windows <= 0 or self.test_windows <= 0:
            raise ValueError("window counts must be positive")
        if not self.train_day_start < self.train_day_end:
            raise ValueError("invalid training interval")
        if not self.test_day_start < self.test_day_end:
            raise ValueError("invalid test interval")
        if self.train_day_end > self.test_day_start:
            raise ValueError("training and test intervals must not overlap")
        if any(size < 8 or size > 100 for size in self.sizes):
            raise ValueError("sizes must lie in [8, 100]")
        if tuple(sorted(set(self.sizes))) != self.sizes:
            raise ValueError("sizes must be unique and increasing")
        if any(not 0.0 < value <= 1.0 for value in self.capacities):
            raise ValueError("capacity scales must lie in (0, 1]")
        if any(value <= 0 for value in self.k_values):
            raise ValueError("candidate budgets must be positive")
        if tuple(sorted(set(self.k_values))) != self.k_values:
            raise ValueError("k_values must be unique and increasing")
        if not 0.0 < self.epsilon < 1.0:
            raise ValueError("epsilon must lie in (0, 1)")
        if self.oracle_time_limit_ms <= 0:
            raise ValueError("oracle_time_limit_ms must be positive")
        if self.critic_actions_per_state < 4:
            raise ValueError("critic_actions_per_state must be at least four")


@dataclass(frozen=True)
class AzureTraceWindow:
    """One chronological request window extracted from the SQLite trace."""

    anchor_day: float
    vm_ids: np.ndarray
    tenant_ids: np.ndarray
    vm_type_ids: np.ndarray
    priorities: np.ndarray
    start_days: np.ndarray
    end_days: np.ndarray
    lifetime_days: np.ndarray
    resources: np.ndarray
    utility: np.ndarray

    @property
    def jobs(self) -> int:
        """Return the number of VM requests in the window."""

        return int(len(self.vm_ids))

    def prefix(self, jobs: int) -> "AzureTraceWindow":
        """Return the first ``jobs`` requests while preserving chronology."""

        if not 1 <= jobs <= self.jobs:
            raise ValueError("prefix size is outside the window")
        return AzureTraceWindow(
            anchor_day=self.anchor_day,
            vm_ids=self.vm_ids[:jobs].copy(),
            tenant_ids=self.tenant_ids[:jobs].copy(),
            vm_type_ids=self.vm_type_ids[:jobs].copy(),
            priorities=self.priorities[:jobs].copy(),
            start_days=self.start_days[:jobs].copy(),
            end_days=self.end_days[:jobs].copy(),
            lifetime_days=self.lifetime_days[:jobs].copy(),
            resources=self.resources[:jobs].copy(),
            utility=self.utility[:jobs].copy(),
        )


@dataclass(frozen=True)
class AzureCandidateBatch:
    """Capacity-repaired candidate batch and proposal diagnostics."""

    actions: tuple[Action, ...]
    raw_generated: int
    raw_capacity_feasible: int
    mean_removed_fraction: float
    elapsed_ms: float


@dataclass(frozen=True)
class AzureMilpReference:
    """Exact or incumbent cumulative-capacity MILP result."""

    action: Action
    objective: float
    success: bool
    status: int
    mip_gap: float | None
    elapsed_ms: float

    @property
    def exact(self) -> bool:
        """Return whether the solver completed with zero reported MIP gap."""

        return bool(
            self.success
            and self.mip_gap is not None
            and self.mip_gap <= 1e-9
        )


@dataclass
class AzurePackingModel:
    """Reward-trained node utility head and action-value critic."""

    actor_mean: np.ndarray
    actor_scale: np.ndarray
    actor_weights: np.ndarray
    critic_mean: np.ndarray
    critic_scale: np.ndarray
    critic_weights: np.ndarray

    def utility_logits(self, state: DispatchState) -> np.ndarray:
        """Predict per-request proposal logits from trace node features."""

        standardized = (
            state.node_features - self.actor_mean
        ) / self.actor_scale
        design = np.column_stack(
            (standardized, np.ones(state.n_jobs, dtype=float))
        )
        return np.clip(design @ self.actor_weights, -20.0, 20.0)

    @staticmethod
    def raw_action_features(
        state: DispatchState,
        action: Action,
    ) -> np.ndarray:
        """Aggregate selected priority, lifetime, and resource features."""

        bits = np.asarray(action, dtype=float)
        count = float(bits.sum())
        selected = bits > 0.5
        sums = bits @ state.node_features
        means = (
            np.mean(state.node_features[selected], axis=0)
            if np.any(selected)
            else np.zeros(state.node_features.shape[1], dtype=float)
        )
        utilization = sums[2:6]
        return np.concatenate(
            (
                np.asarray((count / state.n_jobs,), dtype=float),
                sums / state.n_jobs,
                means,
                np.asarray(
                    (
                        float(np.mean(utilization)),
                        float(np.max(utilization, initial=0.0)),
                    )
                ),
            )
        )

    def q_value(self, state: DispatchState, action: Action) -> float:
        """Predict trace-derived packing reward for one feasible action."""

        raw = self.raw_action_features(state, action)
        standardized = (raw - self.critic_mean) / self.critic_scale
        design = np.concatenate((standardized, np.asarray((1.0,))))
        return float(design @ self.critic_weights)

    def best_action(
        self,
        state: DispatchState,
        actions: list[Action],
    ) -> Action:
        """Return the candidate with the largest learned critic value."""

        if not actions:
            raise ValueError("cannot rerank an empty candidate list")
        return max(actions, key=lambda action: self.q_value(state, action))

    def to_dict(self) -> dict[str, list[float]]:
        """Serialize fitted parameters for reproducibility."""

        return {
            "actor_mean": self.actor_mean.tolist(),
            "actor_scale": self.actor_scale.tolist(),
            "actor_weights": self.actor_weights.tolist(),
            "critic_mean": self.critic_mean.tolist(),
            "critic_scale": self.critic_scale.tolist(),
            "critic_weights": self.critic_weights.tolist(),
        }


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _anchors(start: float, end: float, count: int) -> np.ndarray:
    if count == 1:
        return np.asarray(((start + end) / 2.0,))
    return np.linspace(start, end, count, dtype=float)


def _offline_utility(
    priorities: np.ndarray,
    lifetime_days: np.ndarray,
) -> np.ndarray:
    """Return the preregistered priority-weighted offline value proxy."""

    priority_weight = np.where(priorities == 0, 4.0, 1.0)
    lifetime_hours = np.clip(lifetime_days * 24.0, 1.0 / 60.0, 90.0 * 24.0)
    lifetime_signal = np.log1p(lifetime_hours) / np.log1p(90.0 * 24.0)
    return priority_weight * (0.75 + 0.25 * lifetime_signal)


def load_trace_windows(
    sqlite_path: Path,
    *,
    machine_id: int,
    anchors: Iterable[float],
    end_day: float,
    jobs: int,
) -> list[AzureTraceWindow]:
    """Load deterministic chronological windows from the official trace.

    The SQLite database is opened read-only. Only VM types supported by the
    selected hardware generation are returned. Right-censored lifetimes use
    the trace documentation's 90-day anonymization cap.
    """

    if jobs <= 0:
        raise ValueError("jobs must be positive")
    uri = f"file:{sqlite_path.resolve().as_posix()}?mode=ro"
    query = """
        SELECT
            v.vmId,
            v.tenantId,
            v.vmTypeId,
            v.priority,
            v.starttime,
            v.endtime,
            t.core,
            t.memory,
            t.ssd,
            t.nic
        FROM vm AS v
        JOIN vmType AS t
          ON t.vmTypeId = v.vmTypeId
         AND t.machineId = ?
        WHERE v.starttime >= ?
          AND v.starttime < ?
        ORDER BY v.starttime, v.vmId
        LIMIT ?
    """
    output: list[AzureTraceWindow] = []
    with sqlite3.connect(uri, uri=True) as connection:
        for raw_anchor in anchors:
            anchor = float(raw_anchor)
            rows = connection.execute(
                query, (machine_id, anchor, end_day, jobs)
            ).fetchall()
            if len(rows) != jobs:
                raise ValueError(
                    f"anchor {anchor} returned {len(rows)} rows, expected {jobs}"
                )
            vm_ids = np.asarray([row[0] for row in rows], dtype=np.int64)
            tenant_ids = np.asarray([row[1] for row in rows], dtype=np.int64)
            vm_type_ids = np.asarray([row[2] for row in rows], dtype=np.int64)
            priorities = np.asarray([row[3] for row in rows], dtype=int)
            start_days = np.asarray([row[4] for row in rows], dtype=float)
            end_days = np.asarray(
                [np.nan if row[5] is None else row[5] for row in rows],
                dtype=float,
            )
            lifetime_days = np.where(
                np.isnan(end_days),
                90.0,
                np.clip(end_days - start_days, 1.0 / 1_440.0, 90.0),
            )
            resources = np.asarray(
                [
                    [
                        0.0 if value is None else float(value)
                        for value in row[6:10]
                    ]
                    for row in rows
                ],
                dtype=float,
            )
            resources = np.clip(resources, 0.0, 1.0)
            output.append(
                AzureTraceWindow(
                    anchor_day=anchor,
                    vm_ids=vm_ids,
                    tenant_ids=tenant_ids,
                    vm_type_ids=vm_type_ids,
                    priorities=priorities,
                    start_days=start_days,
                    end_days=end_days,
                    lifetime_days=lifetime_days,
                    resources=resources,
                    utility=_offline_utility(priorities, lifetime_days),
                )
            )
    return output


def trace_profile(sqlite_path: Path, machine_id: int) -> dict[str, Any]:
    """Return official-table counts and selected-hardware coverage."""

    uri = f"file:{sqlite_path.resolve().as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        vm_count = int(connection.execute("SELECT COUNT(*) FROM vm").fetchone()[0])
        type_rows = int(
            connection.execute("SELECT COUNT(*) FROM vmType").fetchone()[0]
        )
        machine_types = int(
            connection.execute(
                """
                SELECT COUNT(DISTINCT vmTypeId)
                FROM vmType
                WHERE machineId = ?
                """,
                (machine_id,),
            ).fetchone()[0]
        )
        machine_count = int(
            connection.execute(
                "SELECT COUNT(DISTINCT machineId) FROM vmType"
            ).fetchone()[0]
        )
        high, low = connection.execute(
            """
            SELECT
                SUM(CASE WHEN priority = 0 THEN 1 ELSE 0 END),
                SUM(CASE WHEN priority = 1 THEN 1 ELSE 0 END)
            FROM vm
            """
        ).fetchone()
        start_min, start_max = connection.execute(
            "SELECT MIN(starttime), MAX(starttime) FROM vm"
        ).fetchone()
    return {
        "vm_requests": vm_count,
        "vm_type_rows": type_rows,
        "machine_generations": machine_count,
        "selected_machine_id": machine_id,
        "selected_machine_vm_types": machine_types,
        "high_priority_requests": int(high),
        "low_priority_requests": int(low),
        "minimum_start_day": float(start_min),
        "maximum_start_day": float(start_max),
    }


def _pairwise_capacity_graph(
    resources: np.ndarray,
    capacities: np.ndarray,
) -> ConflictGraph:
    jobs = int(len(resources))
    edges = tuple(
        (left, right)
        for left in range(jobs)
        for right in range(left + 1, jobs)
        if np.any(resources[left] + resources[right] > capacities + 1e-12)
    )
    return ConflictGraph(nodes=jobs, edges=edges)


def _resource_embedding(
    resources: np.ndarray,
    target_edges: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    """Embed resource vectors in two dimensions with matched edge density."""

    rng = np.random.default_rng(seed)
    centered = resources - np.mean(resources, axis=0, keepdims=True)
    if np.linalg.norm(centered) <= 1e-12:
        positions = rng.uniform(0.0, 1.0, size=(len(resources), 2))
    else:
        _, _, right = np.linalg.svd(centered, full_matrices=False)
        components = min(2, right.shape[0])
        positions = centered @ right[:components].T
        if components == 1:
            positions = np.column_stack((positions[:, 0], np.zeros(len(resources))))
        minimum = np.min(positions, axis=0)
        span = np.ptp(positions, axis=0)
        positions = (positions - minimum) / np.where(span > 1e-12, span, 1.0)
        positions += rng.normal(scale=1e-4, size=positions.shape)
    distances = np.asarray(
        [
            np.linalg.norm(positions[left] - positions[right])
            for left in range(len(positions))
            for right in range(left + 1, len(positions))
        ],
        dtype=float,
    )
    if len(distances) == 0:
        radius = 1e-9
    elif target_edges <= 0:
        positive = distances[distances > 0]
        radius = (
            float(np.min(positive) * 0.5) if len(positive) else 1e-9
        )
    else:
        index = min(target_edges - 1, len(distances) - 1)
        radius = float(np.partition(distances, index)[index])
    return positions, radius


def make_packing_state(
    window: AzureTraceWindow,
    capacity_scale: float,
    seed: int,
) -> tuple[DispatchState, np.ndarray, dict[str, float]]:
    """Build the sampler state and authoritative cumulative capacities."""

    capacities = np.full(len(RESOURCE_NAMES), capacity_scale, dtype=float)
    graph = _pairwise_capacity_graph(window.resources, capacities)
    positions, radius = _resource_embedding(
        window.resources, len(graph.edges), seed
    )
    physical_graph = graph_from_positions(positions, radius)
    authoritative = set(graph.edges)
    physical = set(physical_graph.edges)
    union = authoritative | physical
    geometry_jaccard = (
        len(authoritative & physical) / len(union) if union else 1.0
    )
    lifetime_signal = np.log1p(window.lifetime_days * 24.0) / np.log1p(
        90.0 * 24.0
    )
    high_priority = (window.priorities == 0).astype(float)
    node_features = np.column_stack(
        (
            high_priority,
            lifetime_signal,
            window.resources,
        )
    )
    lifetime_bucket = np.clip(
        np.ceil(1.0 + 11.0 * lifetime_signal), 1, 12
    ).astype(int)
    state = DispatchState(
        graph=graph,
        positions=positions,
        blockade_radius=radius,
        values=window.utility.copy(),
        ages=np.zeros(window.jobs, dtype=int),
        deadlines=lifetime_bucket.copy(),
        remaining=lifetime_bucket.copy(),
        node_features=node_features,
        job_ids=window.vm_ids.copy(),
        step_index=0,
    )
    possible = window.jobs * (window.jobs - 1) / 2
    diagnostics = {
        "pairwise_density": len(graph.edges) / max(possible, 1),
        "physical_density": len(physical_graph.edges) / max(possible, 1),
        "geometry_jaccard": geometry_jaccard,
    }
    return state, capacities, diagnostics


def capacity_usage(state: DispatchState, action: Action) -> np.ndarray:
    """Return CPU, memory, SSD, and NIC usage for an action."""

    bits = np.asarray(action, dtype=float)
    return bits @ state.node_features[:, 2:6]


def capacity_feasible(
    state: DispatchState,
    action: Action,
    capacities: np.ndarray,
) -> bool:
    """Check graph shape and all authoritative cumulative constraints."""

    return bool(
        state.graph.is_valid_shape(action)
        and np.all(capacity_usage(state, action) <= capacities + 1e-10)
    )


def repair_capacity(
    state: DispatchState,
    action: Action,
    capacities: np.ndarray,
    scores: np.ndarray,
) -> tuple[Action, int]:
    """Remove low-value contributors until every capacity is satisfied."""

    bits = np.asarray(action, dtype=int).copy()
    if bits.shape != (state.n_jobs,):
        bits = np.zeros(state.n_jobs, dtype=int)
    bits = np.where(bits > 0, 1, 0)
    resources = state.node_features[:, 2:6]
    removed = 0
    while np.any(bits @ resources > capacities + 1e-10):
        usage = bits @ resources
        violated = usage > capacities + 1e-10
        selected = np.flatnonzero(bits)
        burden = np.sum(
            resources[selected][:, violated]
            / np.maximum(capacities[violated], 1e-12),
            axis=1,
        )
        efficiency = scores[selected] / np.maximum(burden, 1e-12)
        loser = int(selected[int(np.argmin(efficiency))])
        bits[loser] = 0
        removed += 1
    return tuple(int(bit) for bit in bits), removed


def packing_reward(state: DispatchState, action: Action) -> float:
    """Return the additive offline trace utility of an admitted set."""

    return float(np.asarray(action, dtype=float) @ state.values)


def solve_packing_milp(
    state: DispatchState,
    capacities: np.ndarray,
    time_limit_ms: float,
) -> AzureMilpReference:
    """Solve the priority-weighted four-resource admission problem."""

    start = perf_counter()
    constraint = LinearConstraint(
        csr_matrix(state.node_features[:, 2:6].T),
        np.full(len(RESOURCE_NAMES), -np.inf),
        capacities,
    )
    result = milp(
        c=-state.values,
        integrality=np.ones(state.n_jobs),
        bounds=Bounds(np.zeros(state.n_jobs), np.ones(state.n_jobs)),
        constraints=constraint,
        options={
            "time_limit": time_limit_ms / 1_000.0,
            "mip_rel_gap": 0.0,
        },
    )
    if result.x is None:
        scores = state.values / (
            1e-6 + np.mean(state.node_features[:, 2:6], axis=1)
        )
        action = _pack_by_order(
            state,
            capacities,
            np.argsort(-scores),
            include_probability=1.0,
            rng=np.random.default_rng(0),
        )
    else:
        action = tuple(int(value >= 0.5) for value in result.x)
    gap_value = getattr(result, "mip_gap", None)
    gap = None if gap_value is None else float(gap_value)
    return AzureMilpReference(
        action=action,
        objective=packing_reward(state, action),
        success=bool(result.success),
        status=int(result.status),
        mip_gap=gap,
        elapsed_ms=(perf_counter() - start) * 1_000.0,
    )


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
    penalty = np.eye(design.shape[1]) * regularization
    penalty[-1, -1] = 0.0
    weights = np.linalg.solve(
        design.T @ design + penalty,
        design.T @ targets,
    )
    return mean, scale, weights


def _pack_by_order(
    state: DispatchState,
    capacities: np.ndarray,
    order: Iterable[int],
    *,
    include_probability: float,
    rng: np.random.Generator,
) -> Action:
    selected = np.zeros(state.n_jobs, dtype=int)
    usage = np.zeros(len(RESOURCE_NAMES), dtype=float)
    resources = state.node_features[:, 2:6]
    for raw_node in order:
        node = int(raw_node)
        if include_probability < 1.0 and rng.random() > include_probability:
            continue
        proposed = usage + resources[node]
        if np.all(proposed <= capacities + 1e-10):
            selected[node] = 1
            usage = proposed
    return tuple(int(bit) for bit in selected)


def fit_reward_model(
    train_windows: list[AzureTraceWindow],
    config: AzurePackingConfig,
) -> tuple[AzurePackingModel, dict[str, Any]]:
    """Fit the utility head and critic without MILP labels."""

    node_features = []
    targets = []
    for window in train_windows:
        lifetime_signal = np.log1p(window.lifetime_days * 24.0) / np.log1p(
            90.0 * 24.0
        )
        features = np.column_stack(
            (
                (window.priorities == 0).astype(float),
                lifetime_signal,
                window.resources,
            )
        )
        burden = np.mean(window.resources, axis=1)
        target = np.log1p(window.utility / (0.02 + burden))
        node_features.append(features)
        targets.append(target)
    actor_mean, actor_scale, actor_weights = _ridge(
        np.vstack(node_features),
        np.concatenate(targets),
        regularization=1e-3,
    )
    critic_feature_count = 15
    model = AzurePackingModel(
        actor_mean=actor_mean,
        actor_scale=actor_scale,
        actor_weights=actor_weights,
        critic_mean=np.zeros(critic_feature_count),
        critic_scale=np.ones(critic_feature_count),
        critic_weights=np.zeros(critic_feature_count + 1),
    )

    critic_features: list[np.ndarray] = []
    critic_targets: list[float] = []
    training_sizes = tuple(
        size for size in (20, 60, 100) if size <= max(config.sizes)
    )
    for window_index, base_window in enumerate(train_windows):
        for size in training_sizes:
            state, capacities_one, _ = make_packing_state(
                base_window.prefix(size),
                1.0,
                seed=config.seed + window_index * 101 + size,
            )
            logits = model.utility_logits(state)
            for capacity_scale in config.capacities:
                capacities = capacities_one * capacity_scale
                rng = np.random.default_rng(
                    config.seed
                    + 100_003 * window_index
                    + 1_009 * size
                    + int(capacity_scale * 1_000)
                )
                for action_index in range(config.critic_actions_per_state):
                    if action_index == 0:
                        order = np.argsort(-logits)
                        include_probability = 1.0
                    elif action_index % 2:
                        order = np.argsort(
                            -(logits + rng.gumbel(size=state.n_jobs))
                        )
                        include_probability = 1.0
                    else:
                        order = rng.permutation(state.n_jobs)
                        include_probability = 0.70
                    action = _pack_by_order(
                        state,
                        capacities,
                        order,
                        include_probability=include_probability,
                        rng=rng,
                    )
                    critic_features.append(
                        model.raw_action_features(state, action)
                    )
                    critic_targets.append(
                        packing_reward(state, action) / state.n_jobs
                    )
    critic_mean, critic_scale, critic_weights = _ridge(
        np.vstack(critic_features),
        np.asarray(critic_targets),
        regularization=5e-3,
    )
    model.critic_mean = critic_mean
    model.critic_scale = critic_scale
    model.critic_weights = critic_weights
    fitted = np.asarray(
        [
            np.concatenate(
                (
                    (feature - critic_mean) / critic_scale,
                    np.asarray((1.0,)),
                )
            )
            @ critic_weights
            for feature in critic_features
        ]
    )
    actual = np.asarray(critic_targets)
    residual = actual - fitted
    diagnostics = {
        "actor_training_nodes": int(sum(window.jobs for window in train_windows)),
        "critic_training_actions": len(critic_targets),
        "critic_rmse": float(np.sqrt(np.mean(residual**2))),
        "critic_r2": float(
            1.0
            - np.sum(residual**2)
            / max(np.sum((actual - np.mean(actual)) ** 2), 1e-12)
        ),
        "uses_milp_labels": False,
    }
    return model, diagnostics


def _beam_actions(
    state: DispatchState,
    capacities: np.ndarray,
    scores: np.ndarray,
    candidates: int,
) -> list[Action]:
    order = [int(node) for node in np.argsort(-scores)]
    resources = state.node_features[:, 2:6]
    width = max(128, 4 * candidates)
    beam: list[tuple[float, frozenset[int], np.ndarray]] = [
        (0.0, frozenset(), np.zeros(len(RESOURCE_NAMES), dtype=float))
    ]
    for node in order:
        expanded = list(beam)
        for score, selected, usage in beam:
            proposed = usage + resources[node]
            if np.all(proposed <= capacities + 1e-10):
                expanded.append(
                    (
                        score + float(scores[node]),
                        selected | {node},
                        proposed,
                    )
                )
        expanded.sort(key=lambda item: item[0], reverse=True)
        unique: dict[frozenset[int], tuple[float, np.ndarray]] = {}
        for score, selected, usage in expanded:
            unique.setdefault(selected, (score, usage))
            if len(unique) >= width:
                break
        beam = [
            (score, selected, usage)
            for selected, (score, usage) in unique.items()
        ]
    beam.sort(key=lambda item: item[0], reverse=True)
    return [
        tuple(int(node in selected) for node in range(state.n_jobs))
        for _, selected, _ in beam[:candidates]
    ]


def generate_packing_candidates(
    method: str,
    *,
    state: DispatchState,
    capacities: np.ndarray,
    model: AzurePackingModel,
    regime: SamplerRegime,
    candidates: int,
    seed: int,
) -> AzureCandidateBatch:
    """Generate equal-K proposals and enforce cumulative safety."""

    if method not in METHODS:
        raise ValueError(f"unknown Azure packing method: {method}")
    start = perf_counter()
    rng = np.random.default_rng(seed)
    weights = proposal_weights(model, state)
    if method == "rydberg_surrogate":
        batch = generate_candidates(
            "rydberg_surrogate",
            state,
            model,
            regime.proposal(candidates),
            rng,
        )
        raw = list(batch.repaired_actions)
    elif method == "deterministic_repair":
        raw = [tuple(1 for _ in range(state.n_jobs))]
    elif method == "beam_search":
        raw = _beam_actions(
            state, capacities, weights, candidates
        )
    elif method == "randomized_greedy":
        raw = [
            _pack_by_order(
                state,
                capacities,
                np.argsort(
                    -(np.log(np.maximum(weights, 1e-12)) + rng.gumbel(size=state.n_jobs))
                ),
                include_probability=1.0,
                rng=rng,
            )
            for _ in range(candidates)
        ]
    else:
        raw = [
            _pack_by_order(
                state,
                capacities,
                rng.permutation(state.n_jobs),
                include_probability=0.70,
                rng=rng,
            )
            for _ in range(candidates)
        ]
    raw_feasible = sum(
        capacity_feasible(state, action, capacities) for action in raw
    )
    repaired: list[Action] = []
    removed_fractions: list[float] = []
    for action in raw:
        safe, removed = repair_capacity(
            state, action, capacities, weights
        )
        selected = max(sum(action), 1)
        removed_fractions.append(removed / selected)
        repaired.append(safe)
    unique = tuple(
        dict.fromkeys(
            action
            for action in repaired
            if capacity_feasible(state, action, capacities)
        )
    )
    return AzureCandidateBatch(
        actions=unique,
        raw_generated=len(raw),
        raw_capacity_feasible=raw_feasible,
        mean_removed_fraction=float(
            np.mean(removed_fractions) if removed_fractions else 0.0
        ),
        elapsed_ms=(perf_counter() - start) * 1_000.0,
    )


def _diversity(actions: tuple[Action, ...]) -> float:
    if len(actions) < 2:
        return 0.0
    values = [
        float(np.mean(np.asarray(actions[left]) != np.asarray(actions[right])))
        for left in range(len(actions))
        for right in range(left + 1, len(actions))
    ]
    return float(np.mean(values))


def evaluate_candidate_batch(
    *,
    batch: AzureCandidateBatch,
    state: DispatchState,
    capacities: np.ndarray,
    model: AzurePackingModel,
    reference_reward: float,
    epsilon: float,
) -> dict[str, Any]:
    """Compute proposal, critic, safety, utilization, and diversity metrics."""

    ratios = [
        packing_reward(state, action) / max(reference_reward, 1e-12)
        for action in batch.actions
    ]
    if batch.actions:
        critic_action = model.best_action(state, list(batch.actions))
        weights = proposal_weights(model, state)
        utility_action = max(
            batch.actions,
            key=lambda action: float(np.asarray(action) @ weights),
        )
    else:
        critic_action = tuple(0 for _ in range(state.n_jobs))
        utility_action = critic_action
    critic_ratio = (
        packing_reward(state, critic_action) / max(reference_reward, 1e-12)
    )
    utility_ratio = (
        packing_reward(state, utility_action) / max(reference_reward, 1e-12)
    )
    usage = capacity_usage(state, critic_action)
    normalized_usage = usage / np.maximum(capacities, 1e-12)
    return {
        "best_k_ratio": max(ratios, default=0.0),
        "critic_selected_ratio": critic_ratio,
        "utility_selected_ratio": utility_ratio,
        "epsilon_coverage": float(max(ratios, default=0.0) >= 1.0 - epsilon),
        "p_epsilon": float(
            np.mean(np.asarray(ratios) >= 1.0 - epsilon)
            if ratios
            else 0.0
        ),
        "accepted_vms": int(sum(critic_action)),
        "mean_resource_utilization": float(np.mean(normalized_usage)),
        "peak_resource_utilization": float(
            np.max(normalized_usage, initial=0.0)
        ),
        "raw_capacity_feasible_rate": (
            batch.raw_capacity_feasible / max(batch.raw_generated, 1)
        ),
        "post_repair_feasible_rate": float(
            all(
                capacity_feasible(state, action, capacities)
                for action in batch.actions
            )
        ),
        "repair_removed_fraction": batch.mean_removed_fraction,
        "unique_feasible": len(batch.actions),
        "diversity": _diversity(batch.actions),
        "proposal_latency_ms": batch.elapsed_ms,
    }


def _mean_ci(values: Iterable[float]) -> tuple[float, float]:
    items = np.asarray(list(values), dtype=float)
    if len(items) == 0:
        raise ValueError("cannot summarize an empty collection")
    mean = float(np.mean(items))
    interval = (
        float(1.96 * np.std(items, ddof=1) / sqrt(len(items)))
        if len(items) > 1
        else 0.0
    )
    return mean, interval


SUMMARY_METRICS = (
    "best_k_ratio",
    "critic_selected_ratio",
    "utility_selected_ratio",
    "epsilon_coverage",
    "p_epsilon",
    "accepted_vms",
    "mean_resource_utilization",
    "peak_resource_utilization",
    "raw_capacity_feasible_rate",
    "post_repair_feasible_rate",
    "repair_removed_fraction",
    "unique_feasible",
    "diversity",
    "proposal_latency_ms",
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
            mean, interval = _mean_ci(float(item[metric]) for item in items)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci95"] = interval
        output.append(row)
    return output


def _find(summary: list[dict[str, Any]], **matching: Any) -> dict[str, Any]:
    return next(
        row
        for row in summary
        if all(row[key] == value for key, value in matching.items())
    )




def _build_gates(results: dict[str, Any]) -> dict[str, Any]:
    config = results["config"]
    target = _find(
        results["summary"],
        method="rydberg_surrogate",
        size=max(config["sizes"]),
        capacity=max(config["capacities"]),
        k=16,
    )
    checks = {
        "milp_exact_rate_equals_1": results["oracle_summary"]["exact_rate"] >= 1.0,
        "post_repair_feasibility_equals_1": min(
            record["post_repair_feasible_rate"]
            for record in results["records"]
        )
        >= 1.0,
        "n100_k16_best_mean_at_least_0_90": (
            target["best_k_ratio_mean"] >= 0.90
        ),
        "n100_k16_best_lower_ci_at_least_0_90": (
            target["best_k_ratio_mean"] - target["best_k_ratio_ci95"] >= 0.90
        ),
        "n100_k16_critic_lower_ci_at_least_0_90": (
            target["critic_selected_ratio_mean"]
            - target["critic_selected_ratio_ci95"]
            >= 0.90
        ),
    }
    contribution = results["paired_comparisons"][
        "n100_full_capacity_k16_rydberg_minus_repair"
    ]
    contribution_checks = {
        "rydberg_best_paired_lower_ci_above_repair": (
            contribution["mean"] - contribution["ci95"] > 0.0
        ),
        "n100_k16_raw_capacity_feasible_at_least_0_10": (
            target["raw_capacity_feasible_rate_mean"] >= 0.10
        ),
    }
    pipeline_pass = bool(all(checks.values()))
    return {
        "checks": checks,
        "sampler_contribution_checks": contribution_checks,
        "pipeline_pass": pipeline_pass,
        "sampler_contribution_pass": bool(
            pipeline_pass and all(contribution_checks.values())
        ),
        "hardware_claim_pass": False,
        "hardware_claim_reason": (
            "The neutral-atom path is a classical surrogate and no measured "
            "QPU latency or distribution-transfer calibration was run."
        ),
    }


def run_azure_packing_benchmark(
    *,
    sqlite_path: Path,
    stable_results_path: Path,
    output_json: Path,
    output_report: Path,
    config: AzurePackingConfig = AzurePackingConfig(),
) -> dict[str, Any]:
    """Fit on the training interval, evaluate held-out windows, and report."""

    stable = json.loads(stable_results_path.read_text(encoding="utf-8"))
    regime = SamplerRegime(**stable["selected_regime"])
    maximum_jobs = max(config.sizes)
    train = load_trace_windows(
        sqlite_path,
        machine_id=config.machine_id,
        anchors=_anchors(
            config.train_day_start,
            config.train_day_end,
            config.train_windows,
        ),
        end_day=config.test_day_start,
        jobs=maximum_jobs,
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
        jobs=maximum_jobs,
    )
    model, training = fit_reward_model(train, config)

    records: list[dict[str, Any]] = []
    oracle_records: list[dict[str, Any]] = []
    test_windows = []
    for window_index, base_window in enumerate(test):
        test_windows.append(
            {
                "window_index": window_index,
                "anchor_day": base_window.anchor_day,
                "first_start_day": float(base_window.start_days[0]),
                "last_start_day": float(base_window.start_days[-1]),
                "high_priority_fraction": float(
                    np.mean(base_window.priorities == 0)
                ),
                "unique_tenants": int(len(np.unique(base_window.tenant_ids))),
                "right_censored_fraction": float(
                    np.mean(np.isnan(base_window.end_days))
                ),
                "vm_ids_sha256": sha256(
                    base_window.vm_ids.tobytes()
                ).hexdigest(),
            }
        )
        for size in config.sizes:
            window = base_window.prefix(size)
            for capacity in config.capacities:
                state_seed = (
                    config.seed
                    + 100_003 * window_index
                    + 1_009 * size
                    + int(capacity * 10_000)
                )
                state, capacities, graph_metrics = make_packing_state(
                    window, capacity, state_seed
                )
                oracle = solve_packing_milp(
                    state, capacities, config.oracle_time_limit_ms
                )
                common = {
                    "window_index": window_index,
                    "anchor_day": base_window.anchor_day,
                    "size": size,
                    "capacity": capacity,
                    "reference_reward": oracle.objective,
                    "oracle_exact": oracle.exact,
                    "oracle_status": oracle.status,
                    "oracle_mip_gap": oracle.mip_gap,
                    "oracle_latency_ms": oracle.elapsed_ms,
                    **graph_metrics,
                }
                oracle_records.append(common)
                for candidates in config.k_values:
                    for method_index, method in enumerate(METHODS):
                        batch = generate_packing_candidates(
                            method,
                            state=state,
                            capacities=capacities,
                            model=model,
                            regime=regime,
                            candidates=candidates,
                            seed=state_seed + 701 + method_index * 1_000_003,
                        )
                        records.append(
                            {
                                **common,
                                "method": method,
                                "k": candidates,
                                **evaluate_candidate_batch(
                                    batch=batch,
                                    state=state,
                                    capacities=capacities,
                                    model=model,
                                    reference_reward=oracle.objective,
                                    epsilon=config.epsilon,
                                ),
                            }
                        )

    exact = sum(record["oracle_exact"] for record in oracle_records)
    oracle_latencies = np.asarray(
        [record["oracle_latency_ms"] for record in oracle_records]
    )
    paired_values = []
    indexed = {
        (
            record["window_index"],
            record["method"],
            record["size"],
            record["capacity"],
            record["k"],
        ): record
        for record in records
    }
    for window_index in range(config.test_windows):
        rydberg = indexed[
            (
                window_index,
                "rydberg_surrogate",
                maximum_jobs,
                max(config.capacities),
                16,
            )
        ]
        repair = indexed[
            (
                window_index,
                "deterministic_repair",
                maximum_jobs,
                max(config.capacities),
                16,
            )
        ]
        paired_values.append(
            rydberg["best_k_ratio"] - repair["best_k_ratio"]
        )
    paired_mean, paired_ci = _mean_ci(paired_values)
    results: dict[str, Any] = {
        "schema_version": 1,
        "study": "azure_packing_2020",
        "claim_boundary": (
            "Official trace, derived offline admission benchmark, classical "
            "Rydberg surrogate, no measured QPU claim."
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
        "test_windows": test_windows,
        "records": records,
        "oracle_records": oracle_records,
        "summary": _summarize(
            records, ("method", "size", "capacity", "k")
        ),
        "paired_comparisons": {
            "n100_full_capacity_k16_rydberg_minus_repair": {
                "mean": paired_mean,
                "ci95": paired_ci,
                "trials": len(paired_values),
            }
        },
        "oracle_summary": {
            "states": len(oracle_records),
            "exact": int(exact),
            "exact_rate": exact / max(len(oracle_records), 1),
            "latency_ms_mean": float(np.mean(oracle_latencies)),
            "latency_ms_p95": float(np.quantile(oracle_latencies, 0.95)),
            "latency_ms_p99": float(np.quantile(oracle_latencies, 0.99)),
            "maximum_mip_gap": max(
                (
                    record["oracle_mip_gap"]
                    for record in oracle_records
                    if record["oracle_mip_gap"] is not None
                ),
                default=None,
            ),
        },
    }
    results["gates"] = _build_gates(results)
    ResultWriter().artifacts(
        json_path=output_json,
        report_path=output_report,
        payload=results,
        render_report=render_azure_packing_report,
    )
    return results
