from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from hybrid_qrl.classical import IdentityEncoder, StaticUtilityHead
from hybrid_qrl.core import Action, ConflictGraph
from hybrid_qrl.pipeline import HybridActionHead, SafetyFilter, UtilityCritic
from hybrid_qrl.quantum import (
    DenseRydbergStatevectorSampler,
    ManualNeutralAtomBackendSampler,
    PulseSchedule,
    QuTiPRydbergSampler,
)

QUTIP_AVAILABLE = importlib.util.find_spec("qutip") is not None


GRAPH = ConflictGraph(nodes=3, edges=((0, 1), (1, 2)))


class FixedSampler:
    name = "fixed"

    def __init__(self, actions: list[Action]):
        self.actions = actions

    def sample(self, utilities, graph, candidates, rng):
        del utilities, graph, candidates, rng
        return self.actions


class PipelineTests(unittest.TestCase):
    def test_conflict_graph_normalizes_edges(self) -> None:
        graph = ConflictGraph(nodes=3, edges=((1, 0), (0, 1), (2, 1)))
        self.assertEqual(graph.edges, ((0, 1), (1, 2)))

    def test_safety_filter_rejects_invalid_and_deduplicates(self) -> None:
        actions = [(1, 1, 0), (1, 0, 1), (1, 0, 1), (1, 0)]
        self.assertEqual(SafetyFilter().apply(actions, GRAPH), [(1, 0, 1)])

    def test_exactly_one_cardinality_constraint(self) -> None:
        graph = ConflictGraph(
            nodes=2, edges=((0, 1),), min_selected=1, max_selected=1
        )
        self.assertFalse(graph.is_feasible((0, 0)))
        self.assertTrue(graph.is_feasible((1, 0)))
        self.assertTrue(graph.is_feasible((0, 1)))
        self.assertFalse(graph.is_feasible((1, 1)))

    def test_critic_reranks_feasible_candidates(self) -> None:
        head = HybridActionHead(
            encoder=IdentityEncoder(),
            utility_head=StaticUtilityHead((1.0, 2.0, 3.0)),
            sampler=FixedSampler([(1, 0, 0), (0, 1, 0), (1, 0, 1)]),
            critic=UtilityCritic(),
            candidates=3,
        )
        decision = head.select(np.zeros(3), GRAPH, seed=1)
        self.assertEqual(decision.action, (1, 0, 1))
        self.assertEqual(decision.critic_value, 4.0)

    def test_fallback_runs_when_quantum_candidates_are_invalid(self) -> None:
        head = HybridActionHead(
            encoder=IdentityEncoder(),
            utility_head=StaticUtilityHead((1.0, 2.0, 3.0)),
            sampler=FixedSampler([(1, 1, 1)]),
            critic=UtilityCritic(),
            candidates=1,
        )
        decision = head.select(np.zeros(3), GRAPH, seed=1)
        self.assertIn("fallback", decision.sampler)
        self.assertTrue(GRAPH.is_feasible(decision.action))

    def test_dense_rydberg_probabilities_are_normalized(self) -> None:
        graph = ConflictGraph(
            nodes=2, edges=((0, 1),), min_selected=1, max_selected=1
        )
        sampler = DenseRydbergStatevectorSampler(
            schedule=PulseSchedule(duration=1.0, steps=2)
        )
        probabilities = sampler.probabilities(np.array([1.2, 0.8]), graph)
        self.assertEqual(probabilities.shape, (4,))
        self.assertTrue(np.all(probabilities >= 0.0))
        self.assertAlmostEqual(float(probabilities.sum()), 1.0)

    @unittest.skipUnless(QUTIP_AVAILABLE, "QuTiP is not installed")
    def test_qutip_sampler_normalization_order_and_cache(self) -> None:
        graph = ConflictGraph(
            nodes=2, edges=((0, 1),), min_selected=1, max_selected=1
        )
        sampler = QuTiPRydbergSampler(cache_decimals=3)
        utilities = np.array([1.2, 0.8])
        probabilities = sampler.probabilities(utilities, graph)
        self.assertEqual(probabilities.shape, (4,))
        self.assertAlmostEqual(float(probabilities.sum()), 1.0)
        # Basis order is |00>, |01>, |10>, |11>; node 0 has higher utility.
        self.assertGreater(probabilities[2], probabilities[1])
        cached = sampler.probabilities(utilities, graph)
        self.assertIs(probabilities, cached)
        self.assertEqual(sampler.cache_hits, 1)

    @unittest.skipUnless(QUTIP_AVAILABLE, "QuTiP is not installed")
    def test_downloaded_neutral_atom_backend_adapter(self) -> None:
        graph = ConflictGraph(
            nodes=2, edges=((0, 1),), min_selected=1, max_selected=1
        )
        sampler = ManualNeutralAtomBackendSampler(
            positions=np.array([[0.0, 0.0], [1.0, 0.0]]),
            C6=10.0,
            cache_decimals=3,
        )
        utilities = np.array([1.2, 0.8])
        probabilities = sampler.probabilities(utilities, graph)
        self.assertEqual(probabilities.shape, (4,))
        self.assertAlmostEqual(float(probabilities.sum()), 1.0)
        self.assertLess(probabilities[3], 1e-3)
        cached = sampler.probabilities(utilities, graph)
        self.assertIs(probabilities, cached)
        self.assertEqual(sampler.cache_hits, 1)


if __name__ == "__main__":
    unittest.main()
