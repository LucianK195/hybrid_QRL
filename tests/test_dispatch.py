"""Tests for the dynamic dispatch benchmark and its candidate baselines."""

from __future__ import annotations

import unittest

import numpy as np

from hybrid_qrl.dispatch import (
    DispatchConfig,
    DispatchEnvironment,
    TrainingConfig,
    train_actor_critic,
)
from hybrid_qrl.dispatch.baselines import (
    METHODS,
    ProposalConfig,
    generate_candidates,
    proposal_weights,
    solve_weighted_independent_set,
)
from hybrid_qrl.dispatch.conditional_benchmark import shots_for_95_percent
from hybrid_qrl.dispatch.sampler_loop import (
    SamplerLoopTrainingConfig,
    train_sampler_in_loop,
)


class DispatchBenchmarkTests(unittest.TestCase):
    """Regression tests for environment, learning, and candidate safety."""

    def test_environment_has_requested_scale_and_safe_transition(self) -> None:
        environment = DispatchEnvironment(
            DispatchConfig(n_jobs=20, density=0.15, horizon=2),
            seed=4,
        )
        state = environment.state()
        self.assertEqual(state.node_features.shape, (20, 6))
        self.assertEqual(len(state.graph.edges), round(0.15 * 20 * 19 / 2))
        empty = tuple(0 for _ in range(20))
        next_state, reward, done, info = environment.step(empty)
        self.assertEqual(next_state.step_index, 1)
        self.assertFalse(done)
        self.assertTrue(np.isfinite(reward))
        self.assertEqual(info["selected"], 0)

    def test_small_calibration_scale_and_clustered_graph(self) -> None:
        state = DispatchEnvironment(
            DispatchConfig(
                n_jobs=8,
                graph_family="clustered",
                utility_correlation="spatial",
            ),
            seed=9,
        ).state()
        self.assertEqual(state.n_jobs, 8)
        self.assertEqual(state.node_features.shape, (8, 6))

    def test_actor_critic_updates_without_teacher_actions(self) -> None:
        model, history = train_actor_critic(
            TrainingConfig(
                episodes=4,
                horizon=4,
                train_sizes=(20,),
                densities=(0.12,),
            )
        )
        self.assertEqual(len(history["episode_return"]), 4)
        self.assertTrue(np.any(model.actor.weights[:-1] != 0.0))
        self.assertTrue(np.any(model.action_critic.weights != 0.0))

    def test_every_candidate_method_returns_safe_actions(self) -> None:
        model, _ = train_actor_critic(
            TrainingConfig(
                episodes=4,
                horizon=4,
                train_sizes=(20,),
                densities=(0.12,),
            )
        )
        state = DispatchEnvironment(DispatchConfig(n_jobs=20), seed=11).state()
        for index, method in enumerate(METHODS):
            batch = generate_candidates(
                method,
                state,
                model,
                ProposalConfig(candidates=3, max_runtime_ms=250),
                np.random.default_rng(100 + index),
            )
            self.assertTrue(batch.actions)
            self.assertEqual(len(batch.repaired_actions), 3)
            self.assertTrue(
                all(state.graph.is_feasible(action) for action in batch.actions)
            )

    def test_epsilon_shot_requirement_handles_boundary_cases(self) -> None:
        self.assertIsNone(shots_for_95_percent(0.0))
        self.assertEqual(shots_for_95_percent(1.0), 1)
        self.assertEqual(shots_for_95_percent(0.1), 29)

    def test_sampler_in_loop_training_updates_actor_and_critics(self) -> None:
        model, history = train_sampler_in_loop(
            SamplerLoopTrainingConfig(
                iterations=4,
                horizon=3,
                train_sizes=(8,),
                densities=(0.12,),
                graph_families=("grid",),
                utility_correlations=("none",),
                candidates=4,
            )
        )
        self.assertEqual(len(history["paired_mean_return"]), 4)
        self.assertTrue(np.any(model.actor.weights[:-1] != 0.0))
        self.assertTrue(np.any(model.action_critic.weights != 0.0))

    def test_milp_oracle_dominates_one_greedy_restart(self) -> None:
        model, _ = train_actor_critic(
            TrainingConfig(
                episodes=2,
                horizon=3,
                train_sizes=(20,),
                densities=(0.18,),
            )
        )
        state = DispatchEnvironment(
            DispatchConfig(n_jobs=20, density=0.18), seed=23
        ).state()
        weights = proposal_weights(model, state)
        oracle = solve_weighted_independent_set(state, weights)
        greedy = generate_candidates(
            "greedy",
            state,
            model,
            ProposalConfig(candidates=1),
            np.random.default_rng(2),
        ).actions[0]
        self.assertGreaterEqual(
            np.asarray(oracle.action) @ weights,
            np.asarray(greedy) @ weights - 1e-8,
        )
