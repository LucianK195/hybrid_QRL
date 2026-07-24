"""CartPole dataset experiment commands."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ..cartpole.benchmark import run_benchmark
from ..utilities.cartpole_reporting import (
    _compact_trial,
    aggregate_trials,
    render_markdown,
)
from ..utilities import ExperimentApplication, ExperimentCommand, ResultWriter


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--training-episodes", type=int, default=64)
    parser.add_argument("--evaluation-episodes", type=int, default=30)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--softmax-temperature", type=float, default=0.25)


def _validate_common(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    if args.training_episodes <= 0 or args.evaluation_episodes <= 0:
        parser.error("episode counts must be positive")
    if not 0.0 <= args.epsilon <= 1.0:
        parser.error("--epsilon must be between zero and one")
    if args.softmax_temperature <= 0.0:
        parser.error("--softmax-temperature must be positive")


class BenchmarkCommand(ExperimentCommand):
    name = "benchmark"
    help = "Run one offline CartPole integration benchmark."

    def configure(self, parser: argparse.ArgumentParser) -> None:
        _add_common_arguments(parser)
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

    def validate(
        self,
        parser: argparse.ArgumentParser,
        args: argparse.Namespace,
    ) -> None:
        _validate_common(parser, args)
        if args.candidates <= 0:
            parser.error("--candidates must be positive")

    def execute(self, args: argparse.Namespace) -> None:
        report = run_benchmark(
            args.training_episodes,
            args.evaluation_episodes,
            args.candidates,
            args.seed,
            args.dataset_output,
            args.quantum_backend,
            args.epsilon,
            args.softmax_temperature,
        )
        ResultWriter().json(args.output, report)
        compact = {
            "dataset": report["dataset"],
            "evaluation": {
                name: {
                    "mean_return": metrics["mean_return"],
                    "std_return": metrics["std_return"],
                    "solved_rate": metrics[
                        "solved_rate_return_at_least_475"
                    ],
                }
                for name, metrics in report["evaluation"].items()
            },
            "output": str(args.output),
            "dataset_output": str(args.dataset_output),
        }
        print(json.dumps(compact, indent=2))


class BudgetSweepCommand(ExperimentCommand):
    name = "budget-sweep"
    help = "Compare the CartPole sampler across candidate budgets K."

    def configure(self, parser: argparse.ArgumentParser) -> None:
        _add_common_arguments(parser)
        parser.add_argument(
            "--budgets",
            type=int,
            nargs="+",
            default=[1, 2, 4, 8],
        )
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
            default=Path("results/cartpole_candidate_budget_sweep.json"),
        )

    def validate(
        self,
        parser: argparse.ArgumentParser,
        args: argparse.Namespace,
    ) -> None:
        _validate_common(parser, args)
        if not args.budgets or any(budget <= 0 for budget in args.budgets):
            parser.error("all candidate budgets must be positive")

    def execute(self, args: argparse.Namespace) -> None:
        reports = {
            str(budget): run_benchmark(
                training_episodes=args.training_episodes,
                evaluation_episodes=args.evaluation_episodes,
                candidates=budget,
                seed=args.seed,
                dataset_output=args.dataset_output if index == 0 else None,
                quantum_backend=args.quantum_backend,
                epsilon=args.epsilon,
                softmax_temperature=args.softmax_temperature,
            )
            for index, budget in enumerate(args.budgets)
        }
        first = reports[str(args.budgets[0])]
        compact_rows = [
            self._compact_budget(budget, reports[str(budget)])
            for budget in args.budgets
        ]
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
        ResultWriter().json(args.output, output)
        print(json.dumps(compact_rows, indent=2))

    @staticmethod
    def _compact_budget(
        budget: int,
        report: dict[str, object],
    ) -> dict[str, object]:
        evaluation = report["evaluation"]
        quantum = evaluation["hybrid_rydberg_candidates"]
        return {
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


class MultiSeedCommand(ExperimentCommand):
    name = "multiseed"
    help = "Run the backend-by-budget paired multi-seed study."

    def configure(self, parser: argparse.ArgumentParser) -> None:
        _add_common_arguments(parser)
        parser.add_argument(
            "--backends",
            nargs="+",
            choices=("dense", "qutip", "manual"),
            default=["dense", "qutip", "manual"],
        )
        parser.add_argument(
            "--budgets",
            type=int,
            nargs="+",
            default=[1, 2, 4, 8, 16],
        )
        parser.add_argument(
            "--seeds",
            type=int,
            nargs="+",
            default=[17, 29, 43, 71],
        )
        parser.add_argument(
            "--output",
            type=Path,
            default=Path("results/cartpole_multiseed_results.json"),
        )
        parser.add_argument(
            "--report",
            type=Path,
            default=Path("results/cartpole_multiseed_report.md"),
        )

    def validate(
        self,
        parser: argparse.ArgumentParser,
        args: argparse.Namespace,
    ) -> None:
        _validate_common(parser, args)
        if any(value <= 0 for value in args.budgets):
            parser.error("candidate budgets must be positive")

    def execute(self, args: argparse.Namespace) -> None:
        config = {
            "backends": args.backends,
            "budgets": args.budgets,
            "seeds": args.seeds,
            "training_episodes": args.training_episodes,
            "evaluation_episodes": args.evaluation_episodes,
            "epsilon": args.epsilon,
            "softmax_temperature": args.softmax_temperature,
            "cache_decimals": 2,
        }
        combinations = [
            (backend, budget, seed)
            for backend in args.backends
            for budget in args.budgets
            for seed in args.seeds
        ]
        trials = []
        study_start = time.perf_counter()
        for index, (backend, budget, seed) in enumerate(combinations, start=1):
            print(
                f"[{index:02d}/{len(combinations)}] "
                f"backend={backend} K={budget} seed={seed}",
                flush=True,
            )
            trial_start = time.perf_counter()
            report = run_benchmark(
                training_episodes=args.training_episodes,
                evaluation_episodes=args.evaluation_episodes,
                candidates=budget,
                seed=seed,
                dataset_output=None,
                quantum_backend=backend,
                epsilon=args.epsilon,
                softmax_temperature=args.softmax_temperature,
            )
            trials.append(
                _compact_trial(report, time.perf_counter() - trial_start)
            )

        total_seconds = time.perf_counter() - study_start
        aggregate = aggregate_trials(trials)
        output = {
            "experiment": "Multi-seed CartPole backend-by-budget study",
            "interpretation": (
                "Two-action integration and best-of-K reliability study; not a "
                "quantum-scaling or advantage experiment."
            ),
            "config": config,
            "total_trials": len(trials),
            "total_elapsed_seconds": total_seconds,
            "aggregate": aggregate,
            "trials": trials,
        }
        writer = ResultWriter()
        writer.json(args.output, output)
        writer.text(
            args.report,
            render_markdown(config, aggregate, total_seconds),
        )
        print(
            json.dumps(
                {
                    "total_trials": len(trials),
                    "total_elapsed_seconds": total_seconds,
                    "output": str(args.output),
                    "report": str(args.report),
                },
                indent=2,
            )
        )


CartPoleApplication = ExperimentApplication(
    description="Run experiments derived from the CartPole trajectory dataset.",
    commands=(BenchmarkCommand(), BudgetSweepCommand(), MultiSeedCommand()),
    default_command="benchmark",
)
