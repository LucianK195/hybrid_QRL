"""Reference classical components for examples, tests, and fair baselines.

These implementations intentionally contain no trainable framework dependency.
They satisfy the protocols in :mod:`hybrid_qrl.core` and can be replaced by
learned encoders, utility networks, or stronger combinatorial solvers without
changing the hybrid pipeline.

The randomized greedy sampler is an important experimental baseline: it uses
the same utility vector, candidate budget, and random-number source as a
quantum sampler while always respecting pairwise graph conflicts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .core import Action, ConflictGraph


class IdentityEncoder:
    """Convert an already-vectorized observation to floating-point NumPy form.

    This encoder is suitable for toy environments and integration tests.  Real
    applications should replace it with normalization, feature extraction, or a
    learned representation.
    """

    def encode(self, observation: Any) -> np.ndarray:
        """Return ``observation`` as a one-dimensional float array.

        Parameters
        ----------
        observation:
            Array-like environment observation.

        Returns
        -------
        numpy.ndarray
            One-dimensional floating-point view or copy as determined by
            :func:`numpy.asarray`.

        Raises
        ------
        ValueError
            If the converted observation is not one-dimensional.
        """
        encoded = np.asarray(observation, dtype=float)
        if encoded.ndim != 1:
            raise ValueError("IdentityEncoder expects a one-dimensional observation")
        return encoded


@dataclass(frozen=True)
class StaticUtilityHead:
    """Return a fixed node-utility vector independently of the state.

    Parameters
    ----------
    weights:
        Utility assigned to each graph node.  The number of weights must equal
        ``graph.nodes`` at evaluation time.

    Notes
    -----
    This component is useful for deterministic examples and small weighted
    independent-set studies.  It is not a learned reinforcement-learning head.
    """

    weights: tuple[float, ...]

    def utilities(self, encoded_state: np.ndarray, graph: ConflictGraph) -> np.ndarray:
        """Return a defensive copy of the configured utilities.

        Parameters
        ----------
        encoded_state:
            Ignored; accepted to satisfy the utility-head protocol.
        graph:
            Graph used to validate the expected utility count.

        Returns
        -------
        numpy.ndarray
            Floating-point vector with shape ``(graph.nodes,)``.

        Raises
        ------
        ValueError
            If ``weights`` contains a different number of entries than the
            graph has nodes.
        """
        del encoded_state
        values = np.asarray(self.weights, dtype=float)
        if values.shape != (graph.nodes,):
            raise ValueError("utility count must match graph.nodes")
        return values.copy()


class RandomizedWeightedGreedy:
    """Sample feasible maximal independent sets with randomized priorities.

    For every requested candidate, the sampler adds independent Gumbel noise to
    a positive transform of the node utilities, visits nodes from highest to
    lowest noisy priority, and selects a node when none of its selected
    neighbors conflicts.  Each result is maximal with respect to pairwise graph
    edges, although it is not guaranteed to maximize total utility.

    Cardinality minima are not constructed explicitly.  Consequently a graph
    with a restrictive ``min_selected`` may still require downstream filtering.
    """

    name = "randomized_weighted_greedy"

    def sample(
        self,
        utilities: np.ndarray,
        graph: ConflictGraph,
        candidates: int,
        rng: np.random.Generator,
    ) -> list[Action]:
        """Generate randomized greedy candidates.

        Parameters
        ----------
        utilities:
            One utility per graph node.  Values may be negative; an affine
            shift produces positive sampling weights without changing order.
        graph:
            Pairwise conflict graph used during greedy construction.
        candidates:
            Number of independent randomized restarts.
        rng:
            NumPy generator controlling Gumbel noise and reproducibility.

        Returns
        -------
        list of Action
            Exactly ``candidates`` binary tuples in graph-node order.

        Raises
        ------
        ValueError
            If ``candidates`` is not positive or the utility shape is invalid.
        """
        if candidates <= 0:
            raise ValueError("candidates must be positive")
        if utilities.shape != (graph.nodes,):
            raise ValueError("utilities must contain one value per graph node")

        adjacency = graph.adjacency()
        positive_scale = np.maximum(utilities - utilities.min() + 1e-6, 1e-6)
        output: list[Action] = []
        for _ in range(candidates):
            priorities = rng.gumbel(size=graph.nodes) + np.log(positive_scale)
            selected: set[int] = set()
            for node in np.argsort(-priorities):
                index = int(node)
                if not (adjacency[index] & selected):
                    selected.add(index)
            output.append(tuple(int(node in selected) for node in range(graph.nodes)))
        return output
