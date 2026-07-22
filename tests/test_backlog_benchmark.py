"""Regression tests for scale-aware and stable-backlog dispatch helpers."""

from __future__ import annotations

import unittest

import numpy as np

from hybrid_qrl.dispatch.backlog_benchmark import (
    BacklogBenchmarkConfig,
    SamplerRegime,
    _issue_future_batch,
)
from hybrid_qrl.dispatch.baselines import ProposalConfig, generate_candidates
from hybrid_qrl.dispatch.environment import (
    DispatchConfig,
    DispatchEnvironment,
    induced_dispatch_state,
)
from hybrid_qrl.dispatch.latency_benchmark import LatencyObservation
from hybrid_qrl.dispatch.learning import TrainingConfig, train_actor_critic


class BacklogBenchmarkTests(unittest.TestCase):
    """Check detuning encoding, state projection, and stable reservation."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.model, _ = train_actor_critic(
            TrainingConfig(
                episodes=3,
                horizon=3,
                train_sizes=(20,),
                densities=(0.12,),
            )
        )

    def test_standardized_detuning_returns_safe_fixed_k_batch(self) -> None:
        state = DispatchEnvironment(DispatchConfig(n_jobs=20), seed=41).state()
        batch = generate_candidates(
            "rydberg_surrogate",
            state,
            self.model,
            ProposalConfig(
                candidates=5,
                utility_encoding="standardized",
                detuning_gain=0.5,
                pulse_schedule="adiabatic",
            ),
            np.random.default_rng(8),
        )

        self.assertEqual(batch.raw_generated, 5)
        self.assertEqual(len(batch.repaired_actions), 5)
        self.assertTrue(
            all(state.graph.is_feasible(action) for action in batch.actions)
        )

    def test_induced_future_state_preserves_identity_and_induces_edges(self) -> None:
        state = DispatchEnvironment(DispatchConfig(n_jobs=20), seed=51).state()
        nodes = np.flatnonzero(state.remaining > 2)[:10]
        projected = induced_dispatch_state(state, nodes, future_steps=2)

        np.testing.assert_array_equal(projected.job_ids, state.job_ids[nodes])
        np.testing.assert_array_equal(
            projected.remaining,
            state.remaining[nodes] - 2,
        )
        self.assertEqual(projected.n_jobs, len(nodes))
        expected_edges = {
            (left, right)
            for left in range(len(nodes))
            for right in range(left + 1, len(nodes))
            if tuple(sorted((int(nodes[left]), int(nodes[right]))))
            in state.graph.edges
        }
        self.assertEqual(
            set(projected.graph.edges),
            expected_edges,
        )

    def test_future_request_reserves_only_jobs_that_survive_deadline(self) -> None:
        state = DispatchEnvironment(DispatchConfig(n_jobs=20), seed=61).state()
        observation = LatencyObservation(
            request_id="test",
            submitted_at_ms=0.0,
            started_at_ms=500.0,
            completed_at_ms=1_200.0,
            retrieved_at_ms=1_500.0,
            shots=100,
        )
        config = BacklogBenchmarkConfig(
            sizes=(20,),
            selection_sizes=(20,),
            selection_seeds=1,
            confirmation_seeds=1,
            dynamic_sizes=(20,),
            dynamic_seeds=1,
            horizon=8,
            future_deadline_ms=2_000.0,
            minimum_stable_jobs=2,
            stable_target_fraction=1.0,
            maximum_stable_target_jobs=20,
        )
        request = _issue_future_batch(
            policy="future_rydberg_scale_aware",
            state=state,
            model=self.model,
            selected_regime=SamplerRegime(
                "test",
                "standardized",
                0.5,
                "adiabatic",
            ),
            observation=observation,
            step=0,
            config=config,
            seed=3,
        )

        self.assertIsNotNone(request)
        assert request is not None
        reserved = set(request.reserved_job_ids)
        threshold = 2 + config.stable_guard_steps + 1
        expected = {
            int(state.job_ids[node])
            for node in np.flatnonzero(state.remaining >= threshold)
        }
        self.assertEqual(reserved, expected)
        self.assertTrue(
            all(set(candidate) <= reserved for candidate in request.candidate_job_ids)
        )


if __name__ == "__main__":
    unittest.main()
