"""Dynamic scheduling environment with neutral-atom-compatible conflicts.

Each state contains ``n`` pending jobs and therefore ``n`` binary decisions.
Selecting job ``i`` completes it, but jobs connected in a unit-disk conflict
graph cannot be selected together.  Unselected jobs age and may miss their
deadline; completed and expired jobs are immediately replaced.  This creates
a continuing dispatch problem in which waiting can change future return.

The conflict graph is generated from two-dimensional positions and a blockade
radius.  ``unit_disk`` uses random positions, while ``grid`` uses a jittered
lattice.  Both are geometric graph families that can be embedded in a neutral
atom register.  The graph is the authoritative application constraint; a
perturbed physical graph can separately be constructed to study geometry
calibration error.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil, sqrt

import numpy as np

from ..core import Action, ConflictGraph


@dataclass(frozen=True)
class DispatchConfig:
    """Configuration for one dynamic dispatch environment.

    Parameters are deliberately dimensionless.  ``density`` is the target
    fraction of connected node pairs, not spatial atom density.  A quantile of
    pairwise distances chooses the blockade radius, so finite-size boundary
    effects do not silently change the requested graph density.
    """

    n_jobs: int = 40
    graph_family: str = "unit_disk"
    density: float = 0.12
    utility_distribution: str = "uniform"
    horizon: int = 24
    min_deadline: int = 3
    max_deadline: int = 12
    miss_penalty: float = 1.0
    grid_jitter: float = 0.08

    def __post_init__(self) -> None:
        if not 20 <= self.n_jobs <= 100:
            raise ValueError("n_jobs must be between 20 and 100")
        if self.graph_family not in {"unit_disk", "grid"}:
            raise ValueError("graph_family must be 'unit_disk' or 'grid'")
        if not 0.0 < self.density < 1.0:
            raise ValueError("density must lie strictly between zero and one")
        if self.utility_distribution not in {"uniform", "lognormal", "bimodal"}:
            raise ValueError("unsupported utility distribution")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if not 1 <= self.min_deadline <= self.max_deadline:
            raise ValueError("invalid deadline range")
        if self.miss_penalty < 0:
            raise ValueError("miss_penalty must be non-negative")


@dataclass(frozen=True)
class DispatchState:
    """Immutable snapshot passed to policies and solvers.

    Arrays are defensive copies made by :class:`DispatchEnvironment`.  The six
    node features are value, normalized age, urgency, value-times-urgency,
    normalized degree, and a constant bias feature.
    """

    graph: ConflictGraph
    positions: np.ndarray
    blockade_radius: float
    values: np.ndarray
    ages: np.ndarray
    deadlines: np.ndarray
    remaining: np.ndarray
    node_features: np.ndarray
    step_index: int

    @property
    def n_jobs(self) -> int:
        """Return the number of binary decisions in this state."""

        return self.graph.nodes


def graph_from_positions(
    positions: np.ndarray,
    blockade_radius: float,
) -> ConflictGraph:
    """Construct a unit-disk conflict graph from physical positions."""

    points = np.asarray(positions, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("positions must have shape (n, 2)")
    delta = points[:, None, :] - points[None, :, :]
    distance = np.sqrt(np.sum(delta * delta, axis=-1))
    edges = tuple(
        (left, right)
        for left in range(len(points))
        for right in range(left + 1, len(points))
        if distance[left, right] <= blockade_radius
    )
    return ConflictGraph(nodes=len(points), edges=edges)


def perturbed_physical_graph(
    state: DispatchState,
    relative_error: float,
    rng: np.random.Generator,
) -> ConflictGraph:
    """Build the graph seen by hardware after position-calibration error.

    Gaussian position noise is measured relative to the blockade radius.  The
    returned graph may omit true conflicts or introduce spurious ones; callers
    must still repair proposals against ``state.graph`` before execution.
    """

    if relative_error < 0:
        raise ValueError("relative_error must be non-negative")
    noisy = state.positions + rng.normal(
        scale=relative_error * state.blockade_radius,
        size=state.positions.shape,
    )
    return graph_from_positions(noisy, state.blockade_radius)


class DispatchEnvironment:
    """Continuing weighted-job dispatch process with deadline pressure."""

    def __init__(self, config: DispatchConfig, seed: int = 0) -> None:
        self.config = config
        self.rng = np.random.default_rng(seed)
        self._positions, self._radius, self._graph = self._make_graph()
        self._values = np.empty(config.n_jobs, dtype=float)
        self._ages = np.zeros(config.n_jobs, dtype=int)
        self._deadlines = np.empty(config.n_jobs, dtype=int)
        self._remaining = np.empty(config.n_jobs, dtype=int)
        self._step = 0
        self.reset()

    def _make_graph(self) -> tuple[np.ndarray, float, ConflictGraph]:
        n = self.config.n_jobs
        if self.config.graph_family == "unit_disk":
            positions = self.rng.uniform(0.0, 1.0, size=(n, 2))
        else:
            side = ceil(sqrt(n))
            lattice = np.asarray(
                [(column, row) for row in range(side) for column in range(side)],
                dtype=float,
            )[:n]
            lattice /= max(side - 1, 1)
            spacing = 1.0 / max(side - 1, 1)
            positions = lattice + self.rng.normal(
                scale=self.config.grid_jitter * spacing,
                size=lattice.shape,
            )

        distances = []
        for left in range(n):
            for right in range(left + 1, n):
                distances.append(
                    float(np.linalg.norm(positions[left] - positions[right]))
                )
        edge_count = max(1, int(round(self.config.density * len(distances))))
        radius = float(
            np.partition(np.asarray(distances), edge_count - 1)[edge_count - 1]
        )
        graph = graph_from_positions(positions, radius)
        return positions, radius, graph

    def _sample_values(self, count: int) -> np.ndarray:
        distribution = self.config.utility_distribution
        if distribution == "uniform":
            return self.rng.uniform(0.1, 1.0, size=count)
        if distribution == "lognormal":
            values = self.rng.lognormal(mean=-0.35, sigma=0.75, size=count)
            return np.clip(values / 3.0, 0.05, 1.5)
        high = self.rng.random(count) < 0.22
        return np.where(
            high,
            self.rng.uniform(0.8, 1.25, size=count),
            self.rng.uniform(0.05, 0.35, size=count),
        )

    def _replace_jobs(self, mask: np.ndarray) -> None:
        count = int(np.count_nonzero(mask))
        if count == 0:
            return
        self._values[mask] = self._sample_values(count)
        self._ages[mask] = 0
        self._deadlines[mask] = self.rng.integers(
            self.config.min_deadline,
            self.config.max_deadline + 1,
            size=count,
        )
        self._remaining[mask] = self._deadlines[mask]

    def reset(self) -> DispatchState:
        """Replace every job and return the initial state."""

        self._step = 0
        self._replace_jobs(np.ones(self.config.n_jobs, dtype=bool))
        return self.state()

    def state(self) -> DispatchState:
        """Return a defensive state snapshot for policy evaluation."""

        degree = np.asarray(
            [len(items) for items in self._graph.adjacency()], dtype=float
        )
        urgency = 1.0 / np.maximum(self._remaining, 1)
        features = np.column_stack(
            (
                self._values,
                self._ages / max(self.config.max_deadline, 1),
                urgency,
                self._values * urgency,
                degree / max(self.config.n_jobs - 1, 1),
                np.ones(self.config.n_jobs),
            )
        )
        return DispatchState(
            graph=self._graph,
            positions=self._positions.copy(),
            blockade_radius=self._radius,
            values=self._values.copy(),
            ages=self._ages.copy(),
            deadlines=self._deadlines.copy(),
            remaining=self._remaining.copy(),
            node_features=features,
            step_index=self._step,
        )

    def step(
        self, action: Action
    ) -> tuple[DispatchState, float, bool, dict[str, float]]:
        """Execute one feasible batch and advance all pending deadlines."""

        if not self._graph.is_feasible(action):
            raise ValueError("environment received an infeasible action")
        selected = np.asarray(action, dtype=bool)
        completion_value = float(
            np.sum(
                self._values[selected]
                * (1.0 + self._ages[selected] / self._deadlines[selected])
            )
        )

        waiting = ~selected
        self._ages[waiting] += 1
        self._remaining[waiting] -= 1
        expired = waiting & (self._remaining <= 0)
        missed_value = float(np.sum(self._values[expired]))
        reward = (
            completion_value - self.config.miss_penalty * missed_value
        ) / self.config.n_jobs
        self._replace_jobs(selected | expired)
        self._step += 1
        done = self._step >= self.config.horizon
        info = {
            "completion_value": completion_value,
            "missed_value": missed_value,
            "selected": float(np.count_nonzero(selected)),
            "expired": float(np.count_nonzero(expired)),
        }
        return self.state(), float(reward), done, info

    def with_distribution(self, distribution: str, seed: int) -> "DispatchEnvironment":
        """Create a matched environment with a different job-value law."""

        return DispatchEnvironment(
            replace(self.config, utility_distribution=distribution),
            seed=seed,
        )
