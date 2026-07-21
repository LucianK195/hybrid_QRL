"""Tests for deterministic dispatch graph-dataset export helpers."""

from __future__ import annotations

import numpy as np

from hybrid_qrl.dispatch.dataset import replay_held_out_state
from hybrid_qrl.dispatch.environment import DispatchConfig
from hybrid_qrl.dispatch.learning import (
    ActorCriticModel,
    LinearAutoregressiveActor,
    LinearCritic,
)


def _model() -> ActorCriticModel:
    return ActorCriticModel(
        actor=LinearAutoregressiveActor.initialize(),
        value_critic=LinearCritic(np.zeros(11)),
        action_critic=LinearCritic(np.zeros(23)),
    )


def test_replay_held_out_state_is_deterministic() -> None:
    config = DispatchConfig(n_jobs=20, density=0.12, horizon=8)
    first = replay_held_out_state(_model(), config, 1234, 4)
    second = replay_held_out_state(_model(), config, 1234, 4)

    assert first.graph == second.graph
    assert first.step_index == second.step_index == 4
    np.testing.assert_allclose(first.positions, second.positions)
    np.testing.assert_allclose(first.node_features, second.node_features)
