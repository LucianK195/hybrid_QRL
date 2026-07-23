"""Regression tests for the dispatch generalization stress-test helpers."""

from __future__ import annotations

import unittest

import numpy as np

from hybrid_qrl.dispatch.baselines import ProposalConfig, generate_candidates
from hybrid_qrl.dispatch.benchmark import realized_step_reward
from hybrid_qrl.dispatch.environment import DispatchConfig, DispatchEnvironment
from hybrid_qrl.dispatch.generalization_benchmark import (
    GeneralizationBenchmarkConfig,
    _candidate_metrics,
    _summarize,
)
from hybrid_qrl.dispatch.learning import TrainingConfig, train_actor_critic


class GeneralizationBenchmarkTests(unittest.TestCase):
    """Check validation, safety metrics, and paired summary statistics."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.model, _ = train_actor_critic(
            TrainingConfig(episodes=2, horizon=2, train_sizes=(20,))
        )

    def test_candidate_metrics_preserve_post_repair_safety(self) -> None:
        state = DispatchEnvironment(DispatchConfig(n_jobs=20), seed=13).state()
        batch = generate_candidates(
            "rydberg_surrogate",
            state,
            self.model,
            ProposalConfig(candidates=8, readout_noise=0.15),
            np.random.default_rng(91),
        )
        reference = max(
            realized_step_reward(state, action, 1.0)
            for action in batch.actions
        )
        metrics = _candidate_metrics(
            batch=batch,
            state=state,
            model=self.model,
            reference_reward=reference,
            miss_penalty=1.0,
            epsilon=0.05,
        )
        self.assertEqual(metrics["post_repair_feasible_rate"], 1.0)
        self.assertEqual(metrics["raw_generated"], 8)
        self.assertLessEqual(metrics["best_k_ratio"], 1.0 + 1e-12)

    def test_summary_returns_mean_and_confidence_interval(self) -> None:
        records = []
        for value in (0.8, 1.0):
            row = {"method": "scale_aware", "size": 20, "k": 4}
            for metric in (
                "best_k_ratio",
                "critic_selected_ratio",
                "utility_selected_ratio",
                "epsilon_coverage",
                "p_epsilon",
                "best_k_opportunity_ratio",
                "critic_selected_opportunity_ratio",
                "utility_selected_opportunity_ratio",
                "opportunity_epsilon_coverage",
                "p_opportunity_epsilon",
                "raw_feasible_rate",
                "post_repair_feasible_rate",
                "unique_feasible",
                "diversity",
                "proposal_latency_ms",
                "critic_selected_count",
            ):
                row[metric] = value
            records.append(row)
        summary = _summarize(records, ("method", "size", "k"))
        self.assertEqual(len(summary), 1)
        self.assertAlmostEqual(summary[0]["best_k_ratio_mean"], 0.9)
        self.assertGreater(summary[0]["best_k_ratio_ci95"], 0.0)

    def test_non_positive_reference_uses_opportunity_score(self) -> None:
        state = DispatchEnvironment(
            DispatchConfig(n_jobs=20, min_deadline=1, max_deadline=1), seed=19
        ).state()
        batch = generate_candidates(
            "rydberg_surrogate",
            state,
            self.model,
            ProposalConfig(candidates=4),
            np.random.default_rng(7),
        )
        metrics = _candidate_metrics(
            batch=batch,
            state=state,
            model=self.model,
            reference_reward=-0.1,
            miss_penalty=1.0,
            epsilon=0.05,
        )
        self.assertIsNone(metrics["best_k_ratio"])
        self.assertIsNotNone(metrics["best_k_opportunity_ratio"])

    def test_config_rejects_unsorted_k_values(self) -> None:
        with self.assertRaises(ValueError):
            GeneralizationBenchmarkConfig(k_values=(4, 1))


if __name__ == "__main__":
    unittest.main()
