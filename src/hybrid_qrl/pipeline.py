"""Safety filtering and critic reranking for the hybrid action pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .classical import RandomizedWeightedGreedy
from .core import (
    Action,
    CandidateSampler,
    ConflictGraph,
    Critic,
    Decision,
    StateEncoder,
    UtilityHead,
)


class SafetyFilter:
    """Reject malformed and constraint-violating candidates before execution."""

    def apply(self, actions: list[Action], graph: ConflictGraph) -> list[Action]:
        feasible = [tuple(action) for action in actions if graph.is_feasible(action)]
        # Duplicates do not help deterministic critic reranking.
        return list(dict.fromkeys(feasible))


class UtilityCritic:
    """One-step reference critic; replace with the trained RL critic."""

    def value(
        self,
        encoded_state: np.ndarray,
        action: Action,
        utilities: np.ndarray,
    ) -> float:
        del encoded_state
        return float(np.dot(utilities, np.asarray(action, dtype=float)))


@dataclass
class HybridActionHead:
    """Compose encoder, utilities, sampler, safety layer, and critic."""

    encoder: StateEncoder
    utility_head: UtilityHead
    sampler: CandidateSampler
    critic: Critic
    candidates: int = 8
    safety_filter: SafetyFilter = field(default_factory=SafetyFilter)
    fallback_sampler: CandidateSampler = field(default_factory=RandomizedWeightedGreedy)

    def select(
        self,
        observation: Any,
        graph: ConflictGraph,
        seed: int | None = None,
    ) -> Decision:
        if self.candidates <= 0:
            raise ValueError("candidates must be positive")

        encoded = self.encoder.encode(observation)
        utilities = np.asarray(
            self.utility_head.utilities(encoded, graph), dtype=float
        )
        if utilities.shape != (graph.nodes,):
            raise ValueError("utility head returned the wrong number of values")

        rng = np.random.default_rng(seed)
        raw = self.sampler.sample(utilities, graph, self.candidates, rng)
        primary_raw = list(raw)
        primary_feasible = sum(graph.is_feasible(candidate) for candidate in raw)
        safe = self.safety_filter.apply(raw, graph)
        sampler_name = self.sampler.name
        used_fallback = False

        if not safe:
            raw = self.fallback_sampler.sample(utilities, graph, 1, rng)
            safe = self.safety_filter.apply(raw, graph)
            sampler_name = f"{self.sampler.name}->fallback:{self.fallback_sampler.name}"
            used_fallback = True
        if not safe:
            raise RuntimeError("no feasible action was produced, including by fallback")

        scored = [
            (self.critic.value(encoded, action, utilities), action) for action in safe
        ]
        critic_value, action = max(scored, key=lambda item: (item[0], item[1]))
        return Decision(
            action=action,
            critic_value=float(critic_value),
            sampler=sampler_name,
            requested_candidates=self.candidates,
            raw_candidates=len(primary_raw),
            feasible_candidates=primary_feasible,
            unique_feasible_candidates=len(safe),
            used_fallback=used_fallback,
        )
