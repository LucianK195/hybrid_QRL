"""Core contracts and immutable records for hybrid action selection.

This module defines the backend-independent language used by the package.  An
``Action`` is a binary tuple in graph-node order: bit ``i`` indicates whether
decision node ``i`` is selected.  :class:`ConflictGraph` describes pairwise
exclusions and optional cardinality bounds, while the protocol classes define
the replaceable encoder, utility, sampler, and critic stages.

The protocols use structural typing.  A user component does not need to
inherit from them; implementing the documented attributes and methods is
sufficient.  This keeps learned PyTorch/JAX models and external sampler
adapters independent of the small reference implementation.

Notes
-----
``ConflictGraph`` only models pairwise conflicts plus global selection-count
bounds.  Cumulative capacity, routing, temporal, or domain-specific constraints
must be checked by an extended safety filter before an action is executed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np


Action = tuple[int, ...]


@dataclass(frozen=True)
class ConflictGraph:
    """Normalized pairwise constraints for a fixed-length binary action.

    Parameters
    ----------
    nodes:
        Number of binary decision variables.  Valid action indices are
        ``0`` through ``nodes - 1``.
    edges:
        Pairs of mutually exclusive node indices.  Edge direction is ignored;
        duplicates are removed and stored in sorted canonical order.
    min_selected:
        Minimum permitted Hamming weight of an action.  Defaults to zero.
    max_selected:
        Maximum permitted Hamming weight.  ``None`` means ``nodes``.

    Raises
    ------
    ValueError
        If the node count or cardinality bounds are invalid, an edge references
        an unknown node, or an edge is a self-conflict.

    Notes
    -----
    Instances are frozen so the same graph can safely be used in sampler cache
    keys.  Construction normalizes ``edges`` and resolves ``max_selected`` to
    an integer.
    """

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
        """Check whether ``action`` is a binary vector of the expected length.

        This method validates representation only; it does not evaluate edges
        or cardinality bounds.

        Parameters
        ----------
        action:
            Candidate sequence to validate.

        Returns
        -------
        bool
            ``True`` when the sequence has ``nodes`` entries and every entry is
            equal to integer-like ``0`` or ``1``.
        """
        return len(action) == self.nodes and all(bit in (0, 1) for bit in action)

    def is_feasible(self, action: Sequence[int]) -> bool:
        """Evaluate all constraints represented by this graph.

        Parameters
        ----------
        action:
            Candidate bit sequence in graph-node order.

        Returns
        -------
        bool
            ``True`` only when the shape is valid, the number of selected bits
            lies within the configured bounds, and no edge has both endpoints
            selected.

        Notes
        -----
        A ``False`` result is returned for malformed actions rather than an
        exception, which lets safety filtering reject untrusted backend output.
        """
        selected = sum(action) if self.is_valid_shape(action) else -1
        return (
            self.is_valid_shape(action)
            and self.min_selected <= selected <= self.max_selected
            and all(
                not (action[left] and action[right]) for left, right in self.edges
            )
        )

    def adjacency(self) -> tuple[frozenset[int], ...]:
        """Build an immutable neighbor set for every node.

        Returns
        -------
        tuple of frozenset of int
            Entry ``i`` contains every node that conflicts with node ``i``.
            Isolated nodes have an empty set.

        Notes
        -----
        The representation is created on demand.  Samplers that repeatedly use
        one large graph may cache the returned tuple themselves.
        """
        neighbors = [set() for _ in range(self.nodes)]
        for left, right in self.edges:
            neighbors[left].add(right)
            neighbors[right].add(left)
        return tuple(frozenset(items) for items in neighbors)


@dataclass(frozen=True)
class Decision:
    """Auditable result returned by :meth:`HybridActionHead.select`.

    Attributes
    ----------
    action:
        Feasible binary action selected for execution.
    critic_value:
        Critic score assigned to ``action``.  Its scale and interpretation are
        defined by the supplied critic.
    sampler:
        Primary sampler name, or a primary-to-fallback path when fallback was
        required.
    requested_candidates:
        Configured primary candidate budget ``K``.
    raw_candidates:
        Number of candidates actually returned by the primary sampler.
    feasible_candidates:
        Number of primary candidates accepted before duplicate removal.
    unique_feasible_candidates:
        Number of candidates passed to critic reranking after filtering and
        deduplication.  If fallback is used, this describes the fallback result.
    used_fallback:
        Whether the primary batch contained no usable action and the classical
        fallback path was invoked.
    """

    action: Action
    critic_value: float
    sampler: str
    requested_candidates: int
    raw_candidates: int
    feasible_candidates: int
    unique_feasible_candidates: int
    used_fallback: bool = False


class StateEncoder(Protocol):
    """Structural interface for mapping observations to numeric state vectors."""

    def encode(self, observation: Any) -> np.ndarray:
        """Convert an environment observation to a one-dimensional state.

        Parameters
        ----------
        observation:
            Environment-specific observation object.

        Returns
        -------
        numpy.ndarray
            Numeric representation consumed by the utility head and critic.
        """


class UtilityHead(Protocol):
    """Structural interface for state-dependent binary-decision utilities."""

    def utilities(self, encoded_state: np.ndarray, graph: ConflictGraph) -> np.ndarray:
        """Score every graph node for the current encoded state.

        Parameters
        ----------
        encoded_state:
            Output of the configured state encoder.
        graph:
            Constraint graph defining the expected number and order of nodes.

        Returns
        -------
        numpy.ndarray
            One-dimensional floating-point array with shape ``(graph.nodes,)``.
            Larger values should indicate more desirable selections.
        """


class CandidateSampler(Protocol):
    """Structural interface shared by classical and quantum samplers.

    Implementations must expose a human-readable ``name`` for diagnostics.
    They may return infeasible or duplicate actions: the downstream safety
    filter remains authoritative.
    """

    name: str

    def sample(
        self,
        utilities: np.ndarray,
        graph: ConflictGraph,
        candidates: int,
        rng: np.random.Generator,
    ) -> list[Action]:
        """Generate up to ``candidates`` binary actions.

        Parameters
        ----------
        utilities:
            Node utilities with shape ``(graph.nodes,)``.
        graph:
            Pairwise constraints and cardinality bounds for the decision.
        candidates:
            Requested candidate budget ``K``; implementations should reject
            non-positive values.
        rng:
            Caller-owned NumPy generator used for reproducible randomness.

        Returns
        -------
        list of Action
            Raw measured or generated actions in graph-node order.
        """


class Critic(Protocol):
    """Structural interface for reranking feasible candidate actions."""

    def value(
        self,
        encoded_state: np.ndarray,
        action: Action,
        utilities: np.ndarray,
    ) -> float:
        """Estimate the value of one feasible candidate.

        Parameters
        ----------
        encoded_state:
            State representation created once for the current decision.
        action:
            Feasible candidate that has passed the safety filter.
        utilities:
            Per-node utilities used by the sampler.

        Returns
        -------
        float
            Comparable scalar score.  :class:`HybridActionHead` maximizes it.
        """
