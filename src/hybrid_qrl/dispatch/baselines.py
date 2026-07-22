"""Candidate generators for equal-K and equal-latency dispatch comparisons.

The module provides seven classical baselines plus a learned autoregressive
proposal path and a Rydberg-blockade surrogate.  Every method shares the same
``generate_candidates`` interface and returns auditable counts and timing.

``milp`` uses SciPy's HiGHS mixed-integer solver with an explicit wall-clock
limit.  Repeated no-good cuts request distinct top solutions.  The Rydberg
surrogate is a classical stochastic blockade process: it is useful for 20--100
bit scaling studies, but it is not evidence of quantum speedup and is never
reported as a hardware or statevector result.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from time import perf_counter

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, vstack

from ..core import Action, ConflictGraph
from .environment import DispatchState, perturbed_physical_graph
from .learning import ActorCriticModel


METHODS = (
    "milp",
    "simulated_annealing",
    "mcmc",
    "local_search",
    "beam_search",
    "greedy",
    "autoregressive",
    "rydberg_surrogate",
)


@dataclass(frozen=True)
class ProposalConfig:
    """Shared proposal budget and physical-surrogate controls.

    ``utility_encoding`` controls only the scalable classical Rydberg
    surrogate. ``mean`` preserves the original mean-normalized detuning map.
    ``standardized`` exponentiates centered utilities before restoring unit
    mean. The latter keeps local-detuning contrast stable when graph size and
    the utility distribution change. It is a surrogate hypothesis that must
    be recalibrated against dense, QuTiP, or hardware distributions before it
    supports a physical claim.
    """

    candidates: int = 16
    latency_ms: float | None = None
    max_runtime_ms: float = 2_000.0
    geometry_error: float = 0.0
    blockade_radius_scale: float = 1.0
    readout_noise: float = 0.0
    pulse_schedule: str = "balanced"
    cache_precision: int | None = 2
    beam_width: int = 32
    utility_encoding: str = "mean"
    detuning_gain: float = 0.5

    def __post_init__(self) -> None:
        if self.candidates <= 0:
            raise ValueError("candidates must be positive")
        if self.latency_ms is not None and self.latency_ms <= 0:
            raise ValueError("latency_ms must be positive")
        if self.max_runtime_ms <= 0:
            raise ValueError("max_runtime_ms must be positive")
        if self.geometry_error < 0:
            raise ValueError("geometry_error must be non-negative")
        if self.blockade_radius_scale <= 0:
            raise ValueError("blockade_radius_scale must be positive")
        if not 0 <= self.readout_noise <= 1:
            raise ValueError("readout_noise must be in [0, 1]")
        if self.pulse_schedule not in {
            "short",
            "balanced",
            "adiabatic",
            "extended",
        }:
            raise ValueError("unknown pulse schedule")
        if self.cache_precision is not None and self.cache_precision < 0:
            raise ValueError("cache_precision must be non-negative or None")
        if self.utility_encoding not in {"mean", "standardized"}:
            raise ValueError("utility_encoding must be mean or standardized")
        if self.detuning_gain <= 0:
            raise ValueError("detuning_gain must be positive")


@dataclass(frozen=True)
class CandidateBatch:
    """Generated actions and diagnostics before critic reranking."""

    method: str
    actions: tuple[Action, ...]
    repaired_actions: tuple[Action, ...]
    raw_generated: int
    raw_feasible: int
    unique_feasible: int
    elapsed_ms: float


@dataclass(frozen=True)
class MilpSolution:
    """One HiGHS incumbent together with exactness diagnostics."""

    action: Action
    elapsed_ms: float
    status: int
    success: bool
    mip_gap: float | None
    objective: float


def proposal_weights(model: ActorCriticModel, state: DispatchState) -> np.ndarray:
    """Convert learned actor logits to positive MWIS proposal priorities.

    Softplus preserves logit order while allowing graph solvers to select a
    useful non-empty independent set even when all Bernoulli logits are below
    zero early in training.
    """

    logits = model.utility_logits(state)
    return np.logaddexp(0.0, np.clip(logits, -30.0, 30.0))


def action_score(action: Action, weights: np.ndarray) -> float:
    """Return the linear proposal objective for an action."""

    return float(np.asarray(action, dtype=float) @ weights)


def repair_action(
    action: Action,
    graph: ConflictGraph,
    weights: np.ndarray,
) -> Action:
    """Deterministically remove lower-priority endpoints of true conflicts."""

    bits = np.asarray(action, dtype=int).copy()
    if bits.shape != (graph.nodes,):
        bits = np.zeros(graph.nodes, dtype=int)
    bits = np.where(bits > 0, 1, 0)
    for left, right in graph.edges:
        if bits[left] and bits[right]:
            loser = left if weights[left] < weights[right] else right
            bits[loser] = 0
    while int(bits.sum()) > int(graph.max_selected):
        chosen = np.flatnonzero(bits)
        bits[int(chosen[np.argmin(weights[chosen])])] = 0
    return tuple(int(bit) for bit in bits)


def _deadline(config: ProposalConfig) -> float:
    duration = config.max_runtime_ms if config.latency_ms is None else config.latency_ms
    return perf_counter() + duration / 1_000.0


def _greedy_action(
    graph: ConflictGraph,
    weights: np.ndarray,
    priorities: np.ndarray,
) -> Action:
    adjacency = graph.adjacency()
    selected: set[int] = set()
    for raw_node in np.argsort(-priorities):
        node = int(raw_node)
        if len(selected) >= int(graph.max_selected):
            break
        if not (adjacency[node] & selected):
            selected.add(node)
    return tuple(int(node in selected) for node in range(graph.nodes))


def _greedy_candidates(
    state: DispatchState,
    weights: np.ndarray,
    target: int,
    rng: np.random.Generator,
    deadline: float,
) -> list[Action]:
    output: list[Action] = []
    positive = np.maximum(weights, 1e-9)
    while len(output) < target and perf_counter() < deadline:
        if not output:
            priority = positive
        else:
            priority = np.log(positive) + rng.gumbel(size=state.n_jobs)
        output.append(_greedy_action(state.graph, weights, priority))
    return output


def _local_search_candidates(
    state: DispatchState,
    weights: np.ndarray,
    target: int,
    rng: np.random.Generator,
    deadline: float,
) -> list[Action]:
    adjacency = state.graph.adjacency()
    output: list[Action] = []
    while len(output) < target and perf_counter() < deadline:
        seed_priority = np.log(np.maximum(weights, 1e-9)) + rng.gumbel(
            size=state.n_jobs
        )
        current = set(
            np.flatnonzero(_greedy_action(state.graph, weights, seed_priority))
        )
        improved = True
        while improved and perf_counter() < deadline:
            improved = False
            best_delta = 1e-12
            best_move: tuple[int | None, int] | None = None
            for add in rng.permutation(state.n_jobs):
                node = int(add)
                if node in current:
                    continue
                conflicts = current & adjacency[node]
                if len(conflicts) <= 1:
                    removed = next(iter(conflicts)) if conflicts else None
                    removed_value = 0.0 if removed is None else weights[removed]
                    delta = weights[node] - removed_value
                    if delta > best_delta:
                        best_delta = float(delta)
                        best_move = (removed, node)
            if best_move is not None:
                removed, added = best_move
                if removed is not None:
                    current.remove(removed)
                current.add(added)
                improved = True
        output.append(tuple(int(node in current) for node in range(state.n_jobs)))
    return output


def _annealing_candidates(
    state: DispatchState,
    weights: np.ndarray,
    target: int,
    rng: np.random.Generator,
    deadline: float,
) -> list[Action]:
    adjacency = state.graph.adjacency()
    output: list[Action] = []
    moves = max(80, 8 * state.n_jobs)
    scale = float(np.mean(weights) + 1e-9)
    while len(output) < target and perf_counter() < deadline:
        selected: set[int] = set()
        for move in range(moves):
            if perf_counter() >= deadline:
                break
            node = int(rng.integers(state.n_jobs))
            temperature = max(0.02, 1.2 * (1.0 - move / moves)) * scale
            if node in selected:
                delta = -weights[node]
                if delta >= 0 or rng.random() < exp(float(delta / temperature)):
                    selected.remove(node)
                continue
            conflicts = selected & adjacency[node]
            removed_weight = float(sum(weights[item] for item in conflicts))
            delta = float(weights[node] - removed_weight)
            if delta >= 0 or rng.random() < exp(delta / temperature):
                selected.difference_update(conflicts)
                selected.add(node)
        output.append(tuple(int(node in selected) for node in range(state.n_jobs)))
    return output


def _mcmc_candidates(
    state: DispatchState,
    weights: np.ndarray,
    target: int,
    rng: np.random.Generator,
    deadline: float,
) -> list[Action]:
    adjacency = state.graph.adjacency()
    selected: set[int] = set()
    output: list[Action] = []
    total_sweeps = 5 + 2 * target
    centered = weights / (float(np.mean(weights)) + 1e-9)
    for sweep in range(total_sweeps):
        if perf_counter() >= deadline or len(output) >= target:
            break
        beta = min(4.0, 0.6 + 0.25 * sweep)
        for raw_node in rng.permutation(state.n_jobs):
            node = int(raw_node)
            if selected & adjacency[node]:
                selected.discard(node)
                continue
            probability = 1.0 / (1.0 + exp(-float(beta * (centered[node] - 0.7))))
            if rng.random() < probability:
                selected.add(node)
            else:
                selected.discard(node)
        if sweep >= 5:
            output.append(tuple(int(node in selected) for node in range(state.n_jobs)))
    return output


def _beam_candidates(
    state: DispatchState,
    weights: np.ndarray,
    target: int,
    beam_width: int,
    deadline: float,
) -> list[Action]:
    order = [int(node) for node in np.argsort(-weights)]
    adjacency = state.graph.adjacency()
    beam: list[tuple[float, frozenset[int]]] = [(0.0, frozenset())]
    for node in order:
        if perf_counter() >= deadline:
            break
        expanded = list(beam)
        for score, selected in beam:
            if not (adjacency[node] & selected):
                expanded.append((score + float(weights[node]), selected | {node}))
        expanded.sort(key=lambda item: item[0], reverse=True)
        unique: dict[frozenset[int], float] = {}
        for score, selected in expanded:
            unique.setdefault(selected, score)
            if len(unique) >= max(beam_width, target):
                break
        beam = [(score, selected) for selected, score in unique.items()]
    beam.sort(key=lambda item: item[0], reverse=True)
    return [
        tuple(int(node in selected) for node in range(state.n_jobs))
        for _, selected in beam[:target]
    ]


def _milp_candidates(
    state: DispatchState,
    weights: np.ndarray,
    target: int,
    deadline: float,
) -> list[Action]:
    n = state.n_jobs
    edge_rows = np.zeros((len(state.graph.edges), n), dtype=float)
    for row, (left, right) in enumerate(state.graph.edges):
        edge_rows[row, left] = 1.0
        edge_rows[row, right] = 1.0
    base_matrix = csr_matrix(edge_rows)
    no_good: list[np.ndarray] = []
    output: list[Action] = []
    while len(output) < target:
        remaining = deadline - perf_counter()
        if remaining <= 0:
            break
        matrices = [base_matrix]
        lower = [np.full(len(state.graph.edges), -np.inf)]
        upper = [np.ones(len(state.graph.edges))]
        if no_good:
            matrices.append(csr_matrix(np.asarray(no_good)))
            lower.append(np.full(len(no_good), -np.inf))
            upper.append(
                np.asarray(
                    [int(np.count_nonzero(row > 0)) - 1 for row in no_good]
                )
            )
        matrix = vstack(matrices, format="csr")
        constraint = LinearConstraint(
            matrix, np.concatenate(lower), np.concatenate(upper)
        )
        result = milp(
            c=-weights,
            integrality=np.ones(n),
            bounds=Bounds(np.zeros(n), np.ones(n)),
            constraints=constraint,
            options={"time_limit": max(0.001, remaining), "mip_rel_gap": 0.0},
        )
        if result.x is None:
            break
        action = tuple(int(value >= 0.5) for value in result.x)
        if action in output:
            break
        output.append(action)
        bits = np.asarray(action, dtype=int)
        no_good.append(np.where(bits == 1, 1.0, -1.0))
    return output


def _autoregressive_candidates(
    state: DispatchState,
    model: ActorCriticModel,
    target: int,
    rng: np.random.Generator,
    deadline: float,
) -> list[Action]:
    output: list[Action] = []
    while len(output) < target and perf_counter() < deadline:
        output.append(model.actor.sample(state, rng))
    return output


def _rydberg_surrogate_candidates(
    state: DispatchState,
    weights: np.ndarray,
    target: int,
    rng: np.random.Generator,
    deadline: float,
    config: ProposalConfig,
) -> list[Action]:
    physical_graph = perturbed_physical_graph(
        state,
        config.geometry_error,
        rng,
        radius_scale=config.blockade_radius_scale,
    )
    adjacency = physical_graph.adjacency()
    cached = (
        weights.copy()
        if config.cache_precision is None
        else np.round(weights, decimals=config.cache_precision)
    )
    if config.utility_encoding == "mean":
        normalized = cached / (float(np.mean(cached)) + 1e-9)
    else:
        deviation = float(np.std(cached))
        if deviation <= 1e-9:
            normalized = np.ones_like(cached)
        else:
            standardized = (cached - float(np.mean(cached))) / deviation
            encoded = np.exp(
                np.clip(config.detuning_gain * standardized, -6.0, 6.0)
            )
            normalized = encoded / (float(np.mean(encoded)) + 1e-9)
    sweep_count, beta_max, detuning_start, detuning_end = {
        "short": (2, 2.0, 1.15, 0.50),
        "balanced": (6, 5.0, 1.15, 0.50),
        "adiabatic": (12, 8.0, 1.15, 0.50),
        "extended": (16, 10.0, 1.30, 0.60),
    }[config.pulse_schedule]
    output: list[Action] = []
    while len(output) < target and perf_counter() < deadline:
        selected: set[int] = set()
        for sweep in range(sweep_count):
            beta = beta_max * (sweep + 1) / sweep_count
            progress = (sweep + 1) / sweep_count
            detuning = detuning_start + (
                detuning_end - detuning_start
            ) * progress
            for raw_node in rng.permutation(state.n_jobs):
                node = int(raw_node)
                if selected & adjacency[node]:
                    selected.discard(node)
                    continue
                probability = 1.0 / (
                    1.0 + exp(-float(beta * (normalized[node] - detuning)))
                )
                if rng.random() < probability:
                    selected.add(node)
                else:
                    selected.discard(node)
        bits = np.asarray(
            [int(node in selected) for node in range(state.n_jobs)],
            dtype=int,
        )
        if config.readout_noise:
            flips = rng.random(state.n_jobs) < config.readout_noise
            bits[flips] = 1 - bits[flips]
        output.append(tuple(int(bit) for bit in bits))
    return output


def generate_candidates(
    method: str,
    state: DispatchState,
    model: ActorCriticModel,
    config: ProposalConfig,
    rng: np.random.Generator,
) -> CandidateBatch:
    """Generate, validate, repair, and deduplicate one candidate batch."""

    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    start = perf_counter()
    deadline = _deadline(config)
    weights = proposal_weights(model, state)
    target = config.candidates
    if method == "milp":
        raw = _milp_candidates(state, weights, target, deadline)
    elif method == "simulated_annealing":
        raw = _annealing_candidates(state, weights, target, rng, deadline)
    elif method == "mcmc":
        raw = _mcmc_candidates(state, weights, target, rng, deadline)
    elif method == "local_search":
        raw = _local_search_candidates(state, weights, target, rng, deadline)
    elif method == "beam_search":
        raw = _beam_candidates(
            state, weights, target, config.beam_width, deadline
        )
    elif method == "greedy":
        raw = _greedy_candidates(state, weights, target, rng, deadline)
    elif method == "autoregressive":
        raw = _autoregressive_candidates(state, model, target, rng, deadline)
    else:
        raw = _rydberg_surrogate_candidates(
            state, weights, target, rng, deadline, config
        )

    raw_feasible = sum(state.graph.is_feasible(action) for action in raw)
    repaired = [repair_action(action, state.graph, weights) for action in raw]
    unique = tuple(
        dict.fromkeys(
            action for action in repaired if state.graph.is_feasible(action)
        )
    )
    return CandidateBatch(
        method=method,
        actions=unique,
        repaired_actions=tuple(repaired),
        raw_generated=len(raw),
        raw_feasible=raw_feasible,
        unique_feasible=len(unique),
        elapsed_ms=(perf_counter() - start) * 1_000.0,
    )


def solve_weighted_independent_set(
    state: DispatchState,
    weights: np.ndarray,
    time_limit_ms: float = 1_000.0,
) -> MilpSolution:
    """Solve one weighted independent set for evaluation-oracle use.

    The returned action is the best incumbent exposed by HiGHS before the time
    limit.  Benchmark reports call it an exact oracle only when HiGHS reports a
    completed solve; the 20--100 node geometric instances used by the default
    study normally solve well within this conservative limit.
    """

    if weights.shape != (state.n_jobs,):
        raise ValueError("weights must contain one value per job")
    start = perf_counter()
    edge_rows = np.zeros((len(state.graph.edges), state.n_jobs), dtype=float)
    for row, (left, right) in enumerate(state.graph.edges):
        edge_rows[row, left] = 1.0
        edge_rows[row, right] = 1.0
    constraint = LinearConstraint(
        csr_matrix(edge_rows),
        np.full(len(state.graph.edges), -np.inf),
        np.ones(len(state.graph.edges)),
    )
    result = milp(
        c=-weights,
        integrality=np.ones(state.n_jobs),
        bounds=Bounds(np.zeros(state.n_jobs), np.ones(state.n_jobs)),
        constraints=constraint,
        options={"time_limit": time_limit_ms / 1_000.0, "mip_rel_gap": 0.0},
    )
    if result.x is None:
        action = _greedy_action(state.graph, weights, weights)
    else:
        action = tuple(int(value >= 0.5) for value in result.x)
    gap_value = getattr(result, "mip_gap", None)
    mip_gap = None if gap_value is None else float(gap_value)
    return MilpSolution(
        action=action,
        elapsed_ms=(perf_counter() - start) * 1_000.0,
        status=int(result.status),
        success=bool(result.success),
        mip_gap=mip_gap,
        objective=action_score(action, weights),
    )
