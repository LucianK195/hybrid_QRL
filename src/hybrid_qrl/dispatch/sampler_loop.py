"""Reward-only actor-critic training with a candidate sampler in the loop.

The standard reference learner samples actions directly from its masked
autoregressive actor.  This module instead executes the complete proposal path
at every environment step: actor utilities, Rydberg-blockade surrogate,
constraint repair, learned-Q reranking, and dispatch reward.

The sampler is discrete and non-differentiable, so actor parameters are updated
with simultaneous perturbation stochastic approximation (SPSA).  Paired
positive and negative perturbations use common environment and sampler seeds.
No MILP solution, greedy action, or teacher label enters training.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .baselines import ProposalConfig, generate_candidates
from .environment import DispatchConfig, DispatchEnvironment
from .learning import (
    ActorCriticModel,
    LinearAutoregressiveActor,
    LinearCritic,
    action_features,
    state_features,
)


@dataclass(frozen=True)
class SamplerLoopTrainingConfig:
    """Hyperparameters for sampler-in-the-loop SPSA actor-critic training."""

    iterations: int = 140
    horizon: int = 12
    train_sizes: tuple[int, ...] = (12, 16)
    densities: tuple[float, ...] = (0.08, 0.12, 0.20)
    graph_families: tuple[str, ...] = ("unit_disk", "grid", "clustered")
    utility_correlations: tuple[str, ...] = ("none", "spatial", "degree")
    candidates: int = 16
    gamma: float = 0.97
    actor_learning_rate: float = 0.10
    perturbation: float = 0.12
    stability: float = 8.0
    critic_learning_rate: float = 0.03
    seed: int = 7_301

    def __post_init__(self) -> None:
        if self.iterations <= 0 or self.horizon <= 0:
            raise ValueError("iterations and horizon must be positive")
        if any(size < 8 or size > 100 for size in self.train_sizes):
            raise ValueError("train sizes must lie in [8, 100]")
        if self.candidates <= 0:
            raise ValueError("candidates must be positive")
        if self.actor_learning_rate <= 0 or self.perturbation <= 0:
            raise ValueError("SPSA scales must be positive")


@dataclass
class _SamplerTrajectory:
    """Internal trajectory record used for SPSA scores and critic updates."""

    transitions: list[tuple[np.ndarray, np.ndarray, float]]
    episode_return: float
    mean_raw_feasible: float
    mean_unique: float


def _new_model() -> ActorCriticModel:
    actor = LinearAutoregressiveActor.initialize()
    value_critic = LinearCritic(np.zeros(11, dtype=float))
    action_critic = LinearCritic(np.zeros(23, dtype=float))
    return ActorCriticModel(actor, value_critic, action_critic)


def _sampler_trajectory(
    model: ActorCriticModel,
    environment_config: DispatchConfig,
    environment_seed: int,
    sampler_seed: int,
    candidates: int,
) -> _SamplerTrajectory:
    environment = DispatchEnvironment(environment_config, seed=environment_seed)
    state = environment.state()
    transitions: list[tuple[np.ndarray, np.ndarray, float]] = []
    raw_feasible: list[float] = []
    unique_counts: list[float] = []
    done = False
    step = 0
    while not done:
        batch = generate_candidates(
            "rydberg_surrogate",
            state,
            model,
            ProposalConfig(candidates=candidates, max_runtime_ms=1_000.0),
            np.random.default_rng(sampler_seed + step),
        )
        if batch.actions:
            action = model.best_action(state, list(batch.actions))
        else:
            action = tuple(0 for _ in range(state.n_jobs))
        state_vector = state_features(state)
        action_vector = action_features(state, action)
        state, reward, done, _ = environment.step(action)
        transitions.append((state_vector, action_vector, reward))
        raw_feasible.append(
            batch.raw_feasible / max(batch.raw_generated, 1)
        )
        unique_counts.append(float(batch.unique_feasible))
        step += 1
    return _SamplerTrajectory(
        transitions=transitions,
        episode_return=float(sum(item[2] for item in transitions)),
        mean_raw_feasible=float(np.mean(raw_feasible)),
        mean_unique=float(np.mean(unique_counts)),
    )


def _update_critics(
    model: ActorCriticModel,
    trajectory: _SamplerTrajectory,
    gamma: float,
    learning_rate: float,
) -> None:
    returns = np.zeros(len(trajectory.transitions), dtype=float)
    running = 0.0
    for index in range(len(trajectory.transitions) - 1, -1, -1):
        running = trajectory.transitions[index][2] + gamma * running
        returns[index] = running
    for (state_vector, action_vector, _), target in zip(
        trajectory.transitions, returns
    ):
        model.value_critic.update(state_vector, float(target), learning_rate)
        model.action_critic.update(action_vector, float(target), learning_rate)


def train_sampler_in_loop(
    config: SamplerLoopTrainingConfig = SamplerLoopTrainingConfig(),
) -> tuple[ActorCriticModel, dict[str, list[float]]]:
    """Train utilities and critics through the complete surrogate pipeline.

    Each SPSA iteration runs matched positive and negative actor perturbations.
    The difference in actual episode return estimates the gradient of the
    sampler-mediated objective.  Critics are updated from both trajectories'
    discounted Monte Carlo returns after the actor comparison is complete.
    """

    rng = np.random.default_rng(config.seed)
    model = _new_model()
    history: dict[str, list[float]] = {
        "paired_mean_return": [],
        "return_difference": [],
        "mean_raw_feasible": [],
        "mean_unique": [],
        "actor_step_norm": [],
    }

    for iteration in range(config.iterations):
        environment_config = DispatchConfig(
            n_jobs=int(rng.choice(config.train_sizes)),
            density=float(rng.choice(config.densities)),
            graph_family=str(rng.choice(config.graph_families)),
            utility_correlation=str(rng.choice(config.utility_correlations)),
            horizon=config.horizon,
        )
        environment_seed = config.seed + 20_011 * iteration
        sampler_seed = config.seed + 900_001 + 1_009 * iteration
        base_weights = model.actor.weights.copy()
        direction = rng.choice((-1.0, 1.0), size=len(base_weights))
        perturbation = config.perturbation / (iteration + 1.0) ** 0.101

        model.actor.weights[:] = base_weights + perturbation * direction
        positive = _sampler_trajectory(
            model,
            environment_config,
            environment_seed,
            sampler_seed,
            config.candidates,
        )
        model.actor.weights[:] = base_weights - perturbation * direction
        negative = _sampler_trajectory(
            model,
            environment_config,
            environment_seed,
            sampler_seed,
            config.candidates,
        )

        return_difference = positive.episode_return - negative.episode_return
        gradient_scale = return_difference / (2.0 * perturbation)
        learning_rate = config.actor_learning_rate / (
            iteration + 1.0 + config.stability
        ) ** 0.602
        actor_step = learning_rate * np.clip(gradient_scale, -4.0, 4.0)
        model.actor.weights[:] = np.clip(
            base_weights + actor_step * direction,
            -8.0,
            8.0,
        )
        _update_critics(
            model,
            positive,
            config.gamma,
            config.critic_learning_rate,
        )
        _update_critics(
            model,
            negative,
            config.gamma,
            config.critic_learning_rate,
        )

        history["paired_mean_return"].append(
            0.5 * (positive.episode_return + negative.episode_return)
        )
        history["return_difference"].append(return_difference)
        history["mean_raw_feasible"].append(
            0.5 * (positive.mean_raw_feasible + negative.mean_raw_feasible)
        )
        history["mean_unique"].append(
            0.5 * (positive.mean_unique + negative.mean_unique)
        )
        history["actor_step_norm"].append(
            float(np.linalg.norm(actor_step * direction))
        )

    return model, history
