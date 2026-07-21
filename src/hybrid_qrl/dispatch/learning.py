"""Framework-free Monte Carlo actor-critic for the dispatch benchmark.

This module intentionally learns only from environment trajectories.  The
actor never receives greedy, MILP, or teacher actions.  A linear node policy is
used so the same parameters apply to 20 or 100 decisions.  The policy visits
nodes in random order, samples a Bernoulli decision for each currently
unblocked node, and masks its graph neighbors after selection.

Two critics are trained from discounted returns: a state-value baseline lowers
policy-gradient variance, and an action-value critic reranks best-of-K
candidate batches at evaluation time.  They are deliberately small enough to
make the benchmark easy to inspect and reproduce without PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core import Action
from .environment import DispatchConfig, DispatchEnvironment, DispatchState


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0))))


def state_features(state: DispatchState) -> np.ndarray:
    """Aggregate a variable-size dispatch state into fixed-size features."""

    useful = state.node_features[:, :-1]
    return np.concatenate((useful.mean(axis=0), useful.max(axis=0), [1.0]))


def action_features(state: DispatchState, action: Action) -> np.ndarray:
    """Build fixed-size state-action features for the learned Q critic."""

    selected = np.asarray(action, dtype=bool)
    if np.any(selected):
        chosen = state.node_features[selected, :-1]
        chosen_sum = chosen.sum(axis=0) / state.n_jobs
        chosen_max = chosen.max(axis=0)
    else:
        chosen_sum = np.zeros(state.node_features.shape[1] - 1)
        chosen_max = np.zeros_like(chosen_sum)
    fraction = float(np.mean(selected))
    return np.concatenate(
        (state_features(state), chosen_sum, chosen_max, [fraction, fraction * fraction])
    )


@dataclass
class LinearAutoregressiveActor:
    """Nodewise Bernoulli policy with graph masking and shared parameters."""

    weights: np.ndarray

    @classmethod
    def initialize(cls, feature_count: int = 6) -> "LinearAutoregressiveActor":
        weights = np.zeros(feature_count, dtype=float)
        weights[-1] = -0.35
        return cls(weights)

    def logits(self, state: DispatchState) -> np.ndarray:
        """Return the learned node logits used by every proposal method."""

        return state.node_features @ self.weights

    def sample(
        self,
        state: DispatchState,
        rng: np.random.Generator,
        *,
        return_gradient: bool = False,
    ) -> Action | tuple[Action, np.ndarray]:
        """Draw one feasible action and optionally its score-function gradient."""

        selected: set[int] = set()
        blocked: set[int] = set()
        gradient = np.zeros_like(self.weights)
        logits = self.logits(state)
        adjacency = state.graph.adjacency()
        for raw_node in rng.permutation(state.n_jobs):
            node = int(raw_node)
            if node in blocked:
                continue
            probability = _sigmoid(float(logits[node]))
            decision = int(rng.random() < probability)
            gradient += (decision - probability) * state.node_features[node]
            if decision:
                selected.add(node)
                blocked.update(adjacency[node])
        action = tuple(int(node in selected) for node in range(state.n_jobs))
        if return_gradient:
            return action, gradient
        return action


@dataclass
class LinearCritic:
    """Online linear regressor used for V(s) or Q(s, a)."""

    weights: np.ndarray

    def predict(self, features: np.ndarray) -> float:
        return float(features @ self.weights)

    def update(
        self, features: np.ndarray, target: float, learning_rate: float
    ) -> float:
        error = float(target - self.predict(features))
        self.weights += learning_rate * np.clip(error, -3.0, 3.0) * features
        return error


@dataclass
class ActorCriticModel:
    """Learned policy, state baseline, and action-reranking critic."""

    actor: LinearAutoregressiveActor
    value_critic: LinearCritic
    action_critic: LinearCritic

    def utility_logits(self, state: DispatchState) -> np.ndarray:
        """Expose actor logits as shared proposal utilities."""

        return self.actor.logits(state)

    def q_value(self, state: DispatchState, action: Action) -> float:
        """Estimate discounted return for a feasible state-action pair."""

        return self.action_critic.predict(action_features(state, action))

    def best_action(self, state: DispatchState, actions: list[Action]) -> Action:
        """Rerank a non-empty candidate batch using the learned Q critic."""

        if not actions:
            raise ValueError("cannot rerank an empty candidate batch")
        return max(actions, key=lambda action: self.q_value(state, action))

    def to_dict(self) -> dict[str, list[float]]:
        """Return JSON-serializable learned parameters."""

        return {
            "actor": self.actor.weights.tolist(),
            "value_critic": self.value_critic.weights.tolist(),
            "action_critic": self.action_critic.weights.tolist(),
        }


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters for reward-only Monte Carlo actor-critic training."""

    episodes: int = 320
    horizon: int = 24
    train_sizes: tuple[int, ...] = (20, 40, 60)
    densities: tuple[float, ...] = (0.08, 0.12, 0.18)
    graph_families: tuple[str, ...] = ("unit_disk",)
    utility_correlations: tuple[str, ...] = ("none",)
    gamma: float = 0.97
    actor_learning_rate: float = 0.012
    critic_learning_rate: float = 0.035
    seed: int = 2026


def train_actor_critic(
    config: TrainingConfig = TrainingConfig(),
) -> tuple[ActorCriticModel, dict[str, list[float]]]:
    """Train an actor-critic exclusively from sampled dispatch trajectories.

    Returns the learned model and episode-level diagnostics.  Each episode uses
    a new graph and job stream; sizes and densities are sampled from the
    configured training support.  Discounted Monte Carlo returns are targets
    for both critics, while the actor uses the state-value advantage.
    """

    rng = np.random.default_rng(config.seed)
    actor = LinearAutoregressiveActor.initialize()
    value_critic = LinearCritic(np.zeros(11, dtype=float))
    q_feature_count = 11 + 5 + 5 + 2
    action_critic = LinearCritic(np.zeros(q_feature_count, dtype=float))
    model = ActorCriticModel(actor, value_critic, action_critic)
    history: dict[str, list[float]] = {
        "episode_return": [],
        "mean_selected": [],
        "mean_expired": [],
    }

    for episode in range(config.episodes):
        n_jobs = int(rng.choice(config.train_sizes))
        density = float(rng.choice(config.densities))
        environment = DispatchEnvironment(
            DispatchConfig(
                n_jobs=n_jobs,
                density=density,
                graph_family=str(rng.choice(config.graph_families)),
                utility_correlation=str(rng.choice(config.utility_correlations)),
                horizon=config.horizon,
            ),
            seed=config.seed + 10_003 * episode,
        )
        state = environment.state()
        transitions: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
        selected_counts: list[float] = []
        expired_counts: list[float] = []
        done = False
        while not done:
            action, policy_gradient = actor.sample(state, rng, return_gradient=True)
            state_vector = state_features(state)
            q_vector = action_features(state, action)
            next_state, reward, done, info = environment.step(action)
            transitions.append((state_vector, q_vector, policy_gradient, reward))
            selected_counts.append(info["selected"])
            expired_counts.append(info["expired"])
            state = next_state

        returns = np.zeros(len(transitions), dtype=float)
        running = 0.0
        for index in range(len(transitions) - 1, -1, -1):
            running = transitions[index][3] + config.gamma * running
            returns[index] = running

        advantages = np.asarray(
            [
                target - value_critic.predict(state_vector)
                for (state_vector, _, _, _), target in zip(transitions, returns)
            ]
        )
        advantage_scale = float(np.std(advantages) + 1e-6)
        for (state_vector, q_vector, gradient, _), target, advantage in zip(
            transitions, returns, advantages
        ):
            actor.weights += (
                config.actor_learning_rate
                * np.clip(advantage / advantage_scale, -4.0, 4.0)
                * gradient
                / n_jobs
            )
            value_critic.update(
                state_vector, float(target), config.critic_learning_rate
            )
            action_critic.update(q_vector, float(target), config.critic_learning_rate)
        actor.weights[:] = np.clip(actor.weights, -8.0, 8.0)

        history["episode_return"].append(float(sum(item[3] for item in transitions)))
        history["mean_selected"].append(float(np.mean(selected_counts)))
        history["mean_expired"].append(float(np.mean(expired_counts)))

    return model, history
