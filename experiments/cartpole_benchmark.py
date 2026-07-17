"""Offline CartPole sanity test for the hybrid constrained-action model.

The standard CartPole-v1 equations are implemented locally to avoid adding a
Gym dependency. A linear policy is fitted to a reproducible trajectory dataset
labelled by a stabilizing controller. At evaluation time the same learned
utilities drive direct classical selection, randomized-greedy candidate
selection, and a two-qubit idealized Rydberg candidate sampler.

CartPole has only two actions, so this validates integration and safety rather
than quantum scaling or advantage.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from hybrid_qrl.classical import IdentityEncoder, RandomizedWeightedGreedy
from hybrid_qrl.core import ConflictGraph
from hybrid_qrl.pipeline import HybridActionHead, UtilityCritic
from hybrid_qrl.quantum import (
    DenseRydbergStatevectorSampler,
    ManualNeutralAtomBackendSampler,
    QuTiPRydbergSampler,
)


@dataclass(frozen=True)
class CartPoleConfig:
    gravity: float = 9.8
    mass_cart: float = 1.0
    mass_pole: float = 0.1
    half_pole_length: float = 0.5
    force_magnitude: float = 10.0
    time_step: float = 0.02
    angle_threshold_radians: float = 12.0 * 2.0 * math.pi / 360.0
    position_threshold: float = 2.4
    max_steps: int = 500


class CartPoleEnv:
    """Minimal deterministic-seed implementation of the CartPole-v1 dynamics."""

    def __init__(self, config: CartPoleConfig | None = None):
        self.config = config or CartPoleConfig()
        self.state = np.zeros(4, dtype=float)
        self.steps = 0

    def reset(self, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        self.state = rng.uniform(-0.05, 0.05, size=4)
        self.steps = 0
        return self.state.copy()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool]:
        if action not in (0, 1):
            raise ValueError("CartPole action must be 0 (left) or 1 (right)")

        cfg = self.config
        x, x_velocity, angle, angle_velocity = self.state
        force = cfg.force_magnitude if action == 1 else -cfg.force_magnitude
        total_mass = cfg.mass_cart + cfg.mass_pole
        pole_mass_length = cfg.mass_pole * cfg.half_pole_length
        cosine = math.cos(angle)
        sine = math.sin(angle)
        temporary = (
            force + pole_mass_length * angle_velocity**2 * sine
        ) / total_mass
        angle_acceleration = (
            cfg.gravity * sine - cosine * temporary
        ) / (
            cfg.half_pole_length
            * (4.0 / 3.0 - cfg.mass_pole * cosine**2 / total_mass)
        )
        x_acceleration = (
            temporary
            - pole_mass_length * angle_acceleration * cosine / total_mass
        )

        x += cfg.time_step * x_velocity
        x_velocity += cfg.time_step * x_acceleration
        angle += cfg.time_step * angle_velocity
        angle_velocity += cfg.time_step * angle_acceleration
        self.state = np.array([x, x_velocity, angle, angle_velocity], dtype=float)
        self.steps += 1

        terminated = bool(
            abs(x) > cfg.position_threshold
            or abs(angle) > cfg.angle_threshold_radians
        )
        truncated = self.steps >= cfg.max_steps
        return self.state.copy(), 1.0, terminated, truncated


# A simple stabilizing teacher, used only to label the offline dataset.
EXPERT_GAINS = np.array((-0.5, -1.0, 8.0, 2.0), dtype=float)


def expert_action(state: np.ndarray) -> int:
    return int(float(np.dot(EXPERT_GAINS, state)) >= 0.0)


def collect_dataset(
    episodes: int, seed_start: int
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    env = CartPoleEnv()
    observations: list[np.ndarray] = []
    actions: list[int] = []
    returns: list[int] = []
    for episode in range(episodes):
        state = env.reset(seed_start + episode)
        total = 0
        while True:
            action = expert_action(state)
            observations.append(state)
            actions.append(action)
            state, _, terminated, truncated = env.step(action)
            total += 1
            if terminated or truncated:
                break
        returns.append(total)
    return np.asarray(observations), np.asarray(actions, dtype=int), returns


@dataclass
class LinearPolicy:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: float
    confidence: float = 4.0

    @classmethod
    def fit(
        cls, observations: np.ndarray, actions: np.ndarray, ridge: float = 1e-4
    ) -> "LinearPolicy":
        mean = observations.mean(axis=0)
        scale = observations.std(axis=0)
        scale[scale < 1e-8] = 1.0
        standardized = (observations - mean) / scale
        design = np.column_stack((standardized, np.ones(len(standardized))))
        targets = 2.0 * actions.astype(float) - 1.0
        regularizer = ridge * np.eye(design.shape[1])
        regularizer[-1, -1] = 0.0
        parameters = np.linalg.solve(
            design.T @ design + regularizer, design.T @ targets
        )
        return cls(mean, scale, parameters[:-1], float(parameters[-1]))

    def margin(self, state: np.ndarray) -> float:
        standardized = (np.asarray(state, dtype=float) - self.mean) / self.scale
        return float(np.dot(self.weights, standardized) + self.bias)

    def action(self, state: np.ndarray) -> int:
        return int(self.margin(state) >= 0.0)

    def utilities(self, state: np.ndarray) -> np.ndarray:
        logit = float(np.clip(self.confidence * self.margin(state), -20.0, 20.0))
        right_probability = 1.0 / (1.0 + math.exp(-logit))
        # A common positive offset encourages exactly one Rydberg excitation;
        # the relative ordering still comes from the learned policy.
        return np.array(
            [1.5 - right_probability, 0.5 + right_probability], dtype=float
        )


@dataclass(frozen=True)
class LinearUtilityHead:
    policy: LinearPolicy

    def utilities(self, encoded_state: np.ndarray, graph: ConflictGraph) -> np.ndarray:
        if graph.nodes != 2:
            raise ValueError("CartPole utility head expects two action nodes")
        return self.policy.utilities(encoded_state)


CARTPOLE_ACTION_GRAPH = ConflictGraph(
    nodes=2,
    edges=((0, 1),),
    min_selected=1,
    max_selected=1,
)


def one_hot_to_action(action: tuple[int, ...]) -> int:
    if action == (1, 0):
        return 0
    if action == (0, 1):
        return 1
    raise ValueError(f"expected an exactly-one action, received {action}")


class HybridPolicy:
    def __init__(
        self,
        model: LinearPolicy,
        sampler,
        candidates: int,
        seed: int,
    ):
        self.head = HybridActionHead(
            encoder=IdentityEncoder(),
            utility_head=LinearUtilityHead(model),
            sampler=sampler,
            critic=UtilityCritic(),
            candidates=candidates,
        )
        self.rng = np.random.default_rng(seed)
        self.steps = 0
        self.raw_feasible = 0
        self.raw_candidates = 0
        self.fallbacks = 0
        self.unique_feasible = 0

    def __call__(self, state: np.ndarray) -> int:
        decision = self.head.select(
            state,
            CARTPOLE_ACTION_GRAPH,
            seed=int(self.rng.integers(0, np.iinfo(np.int32).max)),
        )
        self.steps += 1
        self.raw_feasible += decision.feasible_candidates
        self.raw_candidates += decision.raw_candidates
        self.fallbacks += int(decision.used_fallback)
        self.unique_feasible += decision.unique_feasible_candidates
        return one_hot_to_action(decision.action)

    def diagnostics(self) -> dict[str, float | int]:
        return {
            "action_steps": self.steps,
            "raw_feasible_rate": self.raw_feasible / max(1, self.raw_candidates),
            "fallback_rate": self.fallbacks / max(1, self.steps),
            "mean_unique_feasible_candidates": (
                self.unique_feasible / max(1, self.steps)
            ),
        }


def evaluate_policy(
    policy: Callable[[np.ndarray], int], seeds: list[int]
) -> dict[str, object]:
    returns = []
    env = CartPoleEnv()
    for seed in seeds:
        state = env.reset(seed)
        total = 0
        while True:
            state, _, terminated, truncated = env.step(policy(state))
            total += 1
            if terminated or truncated:
                break
        returns.append(total)
    values = np.asarray(returns, dtype=float)
    return {
        "episodes": len(returns),
        "mean_return": float(values.mean()),
        "std_return": float(values.std()),
        "median_return": float(np.median(values)),
        "min_return": int(values.min()),
        "max_return": int(values.max()),
        "solved_rate_return_at_least_475": float(np.mean(values >= 475.0)),
        "episode_returns": returns,
    }


def heldout_agreement(
    policy: Callable[[np.ndarray], int], observations: np.ndarray, labels: np.ndarray
) -> float:
    predictions = np.array([policy(state) for state in observations], dtype=int)
    return float(np.mean(predictions == labels))


def heldout_agreements(
    policy: Callable[[np.ndarray], int],
    observations: np.ndarray,
    labels: np.ndarray,
    reference_policy: Callable[[np.ndarray], int],
) -> tuple[float, float]:
    """Compare one stochastic policy with teacher labels and classical argmax."""
    predictions = np.array([policy(state) for state in observations], dtype=int)
    references = np.array(
        [reference_policy(state) for state in observations], dtype=int
    )
    return (
        float(np.mean(predictions == labels)),
        float(np.mean(predictions == references)),
    )


def run_benchmark(
    training_episodes: int,
    evaluation_episodes: int,
    candidates: int,
    seed: int,
    dataset_output: Path | None,
    quantum_backend: str = "dense",
) -> dict[str, object]:
    observations, labels, teacher_returns = collect_dataset(
        training_episodes, seed_start=seed + 1_000
    )
    split_rng = np.random.default_rng(seed + 2_000)
    indices = split_rng.permutation(len(observations))
    split = int(0.8 * len(indices))
    train_indices, test_indices = indices[:split], indices[split:]
    model = LinearPolicy.fit(observations[train_indices], labels[train_indices])

    if dataset_output is not None:
        dataset_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            dataset_output,
            observations=observations,
            actions=labels,
            train_indices=train_indices,
            test_indices=test_indices,
        )

    evaluation_seeds = list(range(seed + 10_000, seed + 10_000 + evaluation_episodes))
    direct = lambda state: model.action(state)
    random_rng = np.random.default_rng(seed + 3_000)
    random_policy = lambda state: int(random_rng.integers(0, 2))

    classical_candidate = HybridPolicy(
        model,
        RandomizedWeightedGreedy(),
        candidates=candidates,
        seed=seed + 4_000,
    )
    if quantum_backend == "qutip":
        quantum_sampler = QuTiPRydbergSampler(cache_decimals=2)
    elif quantum_backend == "manual":
        quantum_sampler = ManualNeutralAtomBackendSampler(
            positions=np.array([[0.0, 0.0], [1.0, 0.0]]),
            C6=10.0,
            cache_decimals=2,
        )
    elif quantum_backend == "dense":
        quantum_sampler = DenseRydbergStatevectorSampler(cache_decimals=2)
    else:
        raise ValueError("quantum_backend must be 'dense', 'qutip', or 'manual'")
    quantum_candidate = HybridPolicy(
        model,
        quantum_sampler,
        candidates=candidates,
        seed=seed + 5_000,
    )

    test_observations = observations[test_indices]
    test_labels = labels[test_indices]
    # Cap stochastic agreement evaluation so it does not dominate the rollout.
    agreement_count = min(2_000, len(test_observations))
    agreement_slice = slice(0, agreement_count)
    classical_teacher_agreement, classical_linear_agreement = heldout_agreements(
        classical_candidate,
        test_observations[agreement_slice],
        test_labels[agreement_slice],
        direct,
    )
    quantum_teacher_agreement, quantum_linear_agreement = heldout_agreements(
        quantum_candidate,
        test_observations[agreement_slice],
        test_labels[agreement_slice],
        direct,
    )
    dataset_metrics = {
        "samples": len(observations),
        "training_samples": len(train_indices),
        "test_samples": len(test_indices),
        "teacher_mean_return": float(np.mean(teacher_returns)),
        "teacher_solved_rate": float(np.mean(np.asarray(teacher_returns) >= 475)),
        "linear_test_accuracy": heldout_agreement(
            direct, test_observations, test_labels
        ),
        "classical_candidate_test_agreement": classical_teacher_agreement,
        "classical_candidate_to_linear_test_agreement": (
            classical_linear_agreement
        ),
        "quantum_candidate_test_agreement": quantum_teacher_agreement,
        "quantum_candidate_to_linear_test_agreement": quantum_linear_agreement,
        "stochastic_agreement_samples": agreement_count,
    }

    # Fresh wrappers keep rollout diagnostics separate from dataset agreement.
    classical_candidate = HybridPolicy(
        model,
        RandomizedWeightedGreedy(),
        candidates=candidates,
        seed=seed + 6_000,
    )
    quantum_candidate = HybridPolicy(
        model,
        quantum_sampler,
        candidates=candidates,
        seed=seed + 7_000,
    )
    classical_result = evaluate_policy(classical_candidate, evaluation_seeds)
    classical_result["sampler_diagnostics"] = classical_candidate.diagnostics()
    quantum_result = evaluate_policy(quantum_candidate, evaluation_seeds)
    quantum_result["sampler_diagnostics"] = quantum_candidate.diagnostics()
    quantum_result["emulator_cache"] = {
        "hits": quantum_sampler.cache_hits,
        "misses": quantum_sampler.cache_misses,
        "rounded_utility_vectors": len(quantum_sampler._probability_cache),
    }

    return {
        "experiment": "CartPole offline linear policy with hybrid action heads",
        "interpretation": (
            "Integration sanity test only: CartPole has two actions and no "
            "large combinatorial action space."
        ),
        "environment": asdict(CartPoleConfig()),
        "config": {
            "seed": seed,
            "training_episodes": training_episodes,
            "evaluation_episodes": evaluation_episodes,
            "candidate_budget_K": candidates,
            "quantum_emulator_cache_decimals": quantum_sampler.cache_decimals,
            "quantum_backend": quantum_backend,
        },
        "dataset": dataset_metrics,
        "learned_linear_policy": {
            "normalized_weights": model.weights.tolist(),
            "bias": model.bias,
            "mean": model.mean.tolist(),
            "scale": model.scale.tolist(),
        },
        "evaluation": {
            "random": evaluate_policy(random_policy, evaluation_seeds),
            "classical_linear_argmax": evaluate_policy(direct, evaluation_seeds),
            "classical_greedy_candidates": classical_result,
            "hybrid_rydberg_candidates": quantum_result,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-episodes", type=int, default=64)
    parser.add_argument("--evaluation-episodes", type=int, default=30)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--quantum-backend",
        choices=("dense", "qutip", "manual"),
        default="dense",
    )
    parser.add_argument(
        "--dataset-output",
        type=Path,
        default=Path("results/cartpole_offline_dataset.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/cartpole_hybrid_comparison.json"),
    )
    args = parser.parse_args()
    if args.training_episodes <= 0 or args.evaluation_episodes <= 0:
        parser.error("episode counts must be positive")
    if args.candidates <= 0:
        parser.error("--candidates must be positive")

    report = run_benchmark(
        args.training_episodes,
        args.evaluation_episodes,
        args.candidates,
        args.seed,
        args.dataset_output,
        args.quantum_backend,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    compact = {
        "dataset": report["dataset"],
        "evaluation": {
            name: {
                "mean_return": metrics["mean_return"],
                "std_return": metrics["std_return"],
                "solved_rate": metrics["solved_rate_return_at_least_475"],
            }
            for name, metrics in report["evaluation"].items()
        },
        "output": str(args.output),
        "dataset_output": str(args.dataset_output),
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
