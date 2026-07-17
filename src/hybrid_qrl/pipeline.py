"""Orchestration, safety enforcement, fallback, and best-of-K reranking.

The :class:`HybridActionHead` is the package's main runtime boundary.  It runs
the following stages for one environment decision:

1. encode the observation once;
2. compute one utility per binary decision;
3. request ``K`` raw actions from the primary sampler;
4. reject malformed, infeasible, and duplicate actions;
5. invoke a classical fallback only when the primary batch is empty; and
6. select the highest-valued feasible action with the critic.

Quantum backends are proposal mechanisms, not trusted constraint validators.
The safety filter is deliberately downstream of every sampler.
"""

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
    """Validate candidates and remove duplicates while preserving order.

    Subclass or replace this component to enforce constraints that are not
    expressible by :class:`~hybrid_qrl.core.ConflictGraph`, such as cumulative
    CPU/memory limits, DAG readiness, routing rules, or hardware safety limits.
    """

    def apply(self, actions: list[Action], graph: ConflictGraph) -> list[Action]:
        """Return unique graph-feasible candidates in first-seen order.

        Parameters
        ----------
        actions:
            Untrusted actions produced by a classical or quantum sampler.
        graph:
            Authoritative pairwise and cardinality constraints.

        Returns
        -------
        list of Action
            Feasible tuples with duplicates removed.  The result may be empty.
        """
        feasible = [tuple(action) for action in actions if graph.is_feasible(action)]
        # Duplicates do not help deterministic critic reranking.
        return list(dict.fromkeys(feasible))


class UtilityCritic:
    """Score an action by its immediate sum of selected node utilities.

    This deterministic reference critic is appropriate for weighted
    independent-set examples.  It does not estimate delayed reward or a value
    function and should be replaced for a trained RL policy.
    """

    def value(
        self,
        encoded_state: np.ndarray,
        action: Action,
        utilities: np.ndarray,
    ) -> float:
        """Compute the utility dot product for one feasible action.

        Parameters
        ----------
        encoded_state:
            Ignored by this one-step critic.
        action:
            Binary candidate in graph-node order.
        utilities:
            Utility vector aligned with ``action``.

        Returns
        -------
        float
            ``dot(utilities, action)``.
        """
        del encoded_state
        return float(np.dot(utilities, np.asarray(action, dtype=float)))


@dataclass
class HybridActionHead:
    """Select one safe action from a sampled candidate batch.

    Parameters
    ----------
    encoder:
        Converts the environment observation to a numeric state vector.
    utility_head:
        Produces one state-dependent utility per graph node.
    sampler:
        Primary classical, emulated, or hardware candidate generator.
    critic:
        Scores feasible candidates; the highest score is selected.
    candidates:
        Primary candidate budget ``K``.  Must be positive when selecting.
    safety_filter:
        Authoritative validator applied before critic evaluation.
    fallback_sampler:
        Safe recovery sampler called for one candidate when the primary batch
        has no unique feasible action.

    Notes
    -----
    The same seeded :class:`numpy.random.Generator` is passed first to the
    primary sampler and then, if needed, to the fallback.  With deterministic
    components, repeating a call with the same observation, graph, and seed is
    reproducible.
    """

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
        """Run the complete candidate-selection pipeline.

        Parameters
        ----------
        observation:
            Environment-specific observation accepted by ``encoder``.
        graph:
            Constraint graph defining action length and feasibility.
        seed:
            Optional seed for the per-decision NumPy generator.  ``None`` uses
            NumPy's nondeterministic default seeding.

        Returns
        -------
        Decision
            Selected action, critic value, sampler path, and candidate-quality
            diagnostics.

        Raises
        ------
        ValueError
            If the candidate budget is not positive or the utility head returns
            a vector whose shape is not ``(graph.nodes,)``.
        RuntimeError
            If neither the primary sampler nor the fallback yields a feasible
            action.

        Notes
        -----
        Ties in critic value are resolved lexicographically by the action tuple,
        which makes deterministic critics reproducible.  ``raw_candidates`` and
        ``feasible_candidates`` in the returned record always describe the
        primary batch; ``unique_feasible_candidates`` describes the batch that
        was actually reranked.
        """
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
