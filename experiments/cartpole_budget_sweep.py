"""Run the CartPole hybrid benchmark for several candidate budgets K."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cartpole_benchmark import run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--training-episodes", type=int, default=64)
    parser.add_argument("--evaluation-episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--softmax-temperature", type=float, default=0.25)
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
        default=Path("results/cartpole_candidate_budget_sweep.json"),
    )
    args = parser.parse_args()
    if not args.budgets or any(budget <= 0 for budget in args.budgets):
        parser.error("all candidate budgets must be positive")
    if not 0.0 <= args.epsilon <= 1.0:
        parser.error("--epsilon must be between zero and one")
    if args.softmax_temperature <= 0.0:
        parser.error("--softmax-temperature must be positive")

    reports = {}
    for index, budget in enumerate(args.budgets):
        report = run_benchmark(
            training_episodes=args.training_episodes,
            evaluation_episodes=args.evaluation_episodes,
            candidates=budget,
            seed=args.seed,
            dataset_output=args.dataset_output if index == 0 else None,
            quantum_backend=args.quantum_backend,
            epsilon=args.epsilon,
            softmax_temperature=args.softmax_temperature,
        )
        reports[str(budget)] = report

    first = reports[str(args.budgets[0])]
    compact_rows = []
    for budget in args.budgets:
        evaluation = reports[str(budget)]["evaluation"]
        quantum = evaluation["hybrid_rydberg_candidates"]
        compact_rows.append(
            {
                "candidate_budget_K": budget,
                "classical_linear_mean_return": evaluation[
                    "classical_linear_argmax"
                ]["mean_return"],
                "classical_epsilon_greedy_mean_return": evaluation[
                    "classical_epsilon_greedy"
                ]["mean_return"],
                "classical_softmax_mean_return": evaluation[
                    "classical_softmax"
                ]["mean_return"],
                "classical_uniform_best_of_k_mean_return": evaluation[
                    "classical_uniform_best_of_k"
                ]["mean_return"],
                "classical_softmax_best_of_k_mean_return": evaluation[
                    "classical_softmax_best_of_k"
                ]["mean_return"],
                "classical_greedy_mean_return": evaluation[
                    "classical_greedy_candidates"
                ]["mean_return"],
                "hybrid_rydberg_mean_return": quantum["mean_return"],
                "hybrid_rydberg_solved_rate": quantum[
                    "solved_rate_return_at_least_475"
                ],
                "hybrid_raw_feasible_rate": quantum["sampler_diagnostics"][
                    "raw_feasible_rate"
                ],
                "hybrid_fallback_rate": quantum["sampler_diagnostics"][
                    "fallback_rate"
                ],
            }
        )

    output = {
        "experiment": "CartPole candidate-budget sweep",
        "interpretation": first["interpretation"],
        "shared_config": {
            "seed": args.seed,
            "training_episodes": args.training_episodes,
            "evaluation_episodes": args.evaluation_episodes,
            "budgets": args.budgets,
            "quantum_backend": args.quantum_backend,
            "epsilon": args.epsilon,
            "softmax_temperature": args.softmax_temperature,
        },
        "dataset": first["dataset"],
        "summary": compact_rows,
        "reports_by_candidate_budget": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(compact_rows, indent=2))


if __name__ == "__main__":
    main()
