"""Core data types and extension interfaces for the hybrid action head."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np


Action = tuple[int, ...]


@dataclass(frozen=True)
class ConflictGraph:
    """Pairwise hard constraints for an n-bit action."""

    nodes: int
    edges: tuple[tuple[int, int], ...]
    min_selected: int = 0
    max_selected: int | None = None

    def __post_init__(self) -> None:
        if self.nodes <= 0:
            raise ValueError("nodes must be positive")
        maximum = self.nodes if self.max_selected is None else self.max_selected
        if not (0 <= self.min_selected <= maximum <= self.nodes):
            raise ValueError(
                "selection bounds must satisfy 0 <= min_selected <= "
                "max_selected <= nodes"
            )
        object.__setattr__(self, "max_selected", maximum)
        normalized: set[tuple[int, int]] = set()
        for left, right in self.edges:
            if left == right:
                raise ValueError("self-conflicts are not supported")
            if not (0 <= left < self.nodes and 0 <= right < self.nodes):
                raise ValueError(f"edge {(left, right)} is outside the graph")
            normalized.add(tuple(sorted((left, right))))
        object.__setattr__(self, "edges", tuple(sorted(normalized)))

    def is_valid_shape(self, action: Sequence[int]) -> bool:
        return len(action) == self.nodes and all(bit in (0, 1) for bit in action)

    def is_feasible(self, action: Sequence[int]) -> bool:
        selected = sum(action) if self.is_valid_shape(action) else -1
        return (
            self.is_valid_shape(action)
            and self.min_selected <= selected <= self.max_selected
            and all(
                not (action[left] and action[right]) for left, right in self.edges
            )
        )

    def adjacency(self) -> tuple[frozenset[int], ...]:
        neighbors = [set() for _ in range(self.nodes)]
        for left, right in self.edges:
            neighbors[left].add(right)
            neighbors[right].add(left)
        return tuple(frozenset(items) for items in neighbors)


@dataclass(frozen=True)
class Decision:
    """Output selected by the safety layer and classical critic."""

    action: Action
    critic_value: float
    sampler: str
    requested_candidates: int
    raw_candidates: int
    feasible_candidates: int
    unique_feasible_candidates: int
    used_fallback: bool = False


class StateEncoder(Protocol):
    def encode(self, observation: Any) -> np.ndarray:
        """Convert the environment observation to a compact numeric state."""


class UtilityHead(Protocol):
    def utilities(self, encoded_state: np.ndarray, graph: ConflictGraph) -> np.ndarray:
        """Return one state-dependent utility per binary decision."""


class CandidateSampler(Protocol):
    name: str

    def sample(
        self,
        utilities: np.ndarray,
        graph: ConflictGraph,
        candidates: int,
        rng: np.random.Generator,
    ) -> list[Action]:
        """Generate candidate actions; the safety filter remains authoritative."""


class Critic(Protocol):
    def value(
        self,
        encoded_state: np.ndarray,
        action: Action,
        utilities: np.ndarray,
    ) -> float:
        """Estimate long-horizon value for a feasible candidate."""
