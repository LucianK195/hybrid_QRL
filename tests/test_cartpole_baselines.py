from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from cartpole_benchmark import (  # noqa: E402
    EpsilonGreedyPolicy,
    LinearPolicy,
    SampledUtilityPolicy,
)


class CartPoleClassicalBaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = LinearPolicy(
            mean=np.zeros(4),
            scale=np.ones(4),
            weights=np.array([0.0, 0.0, 1.0, 0.0]),
            bias=0.0,
        )
        self.prefer_right = np.array([0.0, 0.0, 1.0, 0.0])

    def test_zero_epsilon_matches_argmax(self) -> None:
        policy = EpsilonGreedyPolicy(self.model, epsilon=0.0, seed=7)
        expected = self.model.action(self.prefer_right)
        self.assertEqual(policy(self.prefer_right), expected)

    def test_best_of_k_uniform_reranks_sampled_actions(self) -> None:
        policy = SampledUtilityPolicy(
            self.model,
            candidates=32,
            seed=7,
            proposal="uniform",
        )
        self.assertEqual(policy(self.prefer_right), 1)

    def test_stochastic_policy_is_reproducible_from_seed(self) -> None:
        left = SampledUtilityPolicy(
            self.model,
            candidates=1,
            seed=11,
            proposal="softmax",
            temperature=0.25,
        )
        right = SampledUtilityPolicy(
            self.model,
            candidates=1,
            seed=11,
            proposal="softmax",
            temperature=0.25,
        )
        left_actions = [left(np.zeros(4)) for _ in range(20)]
        right_actions = [right(np.zeros(4)) for _ in range(20)]
        self.assertEqual(left_actions, right_actions)

    def test_invalid_baseline_parameters_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            EpsilonGreedyPolicy(self.model, epsilon=1.1, seed=7)
        with self.assertRaises(ValueError):
            SampledUtilityPolicy(
                self.model,
                candidates=0,
                seed=7,
                proposal="uniform",
            )
        with self.assertRaises(ValueError):
            SampledUtilityPolicy(
                self.model,
                candidates=1,
                seed=7,
                proposal="softmax",
                temperature=0.0,
            )


if __name__ == "__main__":
    unittest.main()
