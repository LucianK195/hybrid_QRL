"""Run the frozen-model dispatch generalization and candidate-budget sweep."""

from __future__ import annotations

import argparse
from pathlib import Path

from hybrid_qrl.dispatch.generalization_benchmark import (
    GeneralizationBenchmarkConfig,
    run_generalization_benchmark,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    """Parse source-result, test-size, and output arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stable-results",
        type=Path,
        default=PROJECT_ROOT / "results" / "stable_backlog_scaling_results.json",
    )
    parser.add_argument(
        "--baseline-results",
        type=Path,
        default=PROJECT_ROOT / "results" / "dispatch_benchmark_results.json",
    )
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument(
        "--json",
        type=Path,
        default=PROJECT_ROOT / "results" / "dispatch_generalization_stress.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "results" / "dispatch_generalization_stress.md",
    )
    return parser.parse_args()


def main() -> None:
    """Execute the preregistered matrix using frozen model parameters."""

    args = parse_args()
    results = run_generalization_benchmark(
        stable_results_path=args.stable_results,
        baseline_results_path=args.baseline_results,
        output_json=args.json,
        output_report=args.report,
        config=GeneralizationBenchmarkConfig(seeds=args.seeds),
    )
    print(f"candidate generation pass: {results['gates']['candidate_generation_pass']}")
    print(f"deployable critic pass: {results['gates']['deployable_critic_pass']}")
    print(
        f"MILP exact: {results['oracle_summary']['exact']}/"
        f"{results['oracle_summary']['states']}"
    )
    print(args.json)
    print(args.report)


if __name__ == "__main__":
    main()

