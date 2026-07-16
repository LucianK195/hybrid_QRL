"""Small classical components and a constraint-aware sampler baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .core import Action, ConflictGraph


class IdentityEncoder:
    """Template encoder for examples where the observation is already numeric."""

    def encode(self, observation: Any) -> np.ndarray:
        encoded = np.asarray(observation, dtype=float)
        if encoded.ndim != 1:
            raise ValueError("IdentityEncoder expects a one-dimensional observation")
        return encoded


@dataclass(frozen=True)
class StaticUtilityHead:
    """Fixed utilities; replace this with a learned state-dependent model."""

    weights: tuple[float, ...]

    def utilities(self, encoded_state: np.ndarray, graph: ConflictGraph) -> np.ndarray:
        del encoded_state
        values = np.asarray(self.weights, dtype=float)
        if values.shape != (graph.nodes,):
            raise ValueError("utility count must match graph.nodes")
        return values.copy()


class RandomizedWeightedGreedy:
    """Fast classical baseline producing feasible maximal independent sets."""

    name = "randomized_weighted_greedy"

    def sample(
        self,
        utilities: np.ndarray,
        graph: ConflictGraph,
        candidates: int,
        rng: np.random.Generator,
    ) -> list[Action]:
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
