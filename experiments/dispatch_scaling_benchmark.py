"""Command-line entry point for the real dynamic dispatch benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from hybrid_qrl.dispatch.benchmark import BenchmarkConfig, run_benchmark


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--train-episodes", type=int, default=320)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--latency-ms", type=float, default=20.0)
    parser.add_argument(
        "--json",
        type=Path,
        default=PROJECT_ROOT / "results" / "dispatch_benchmark_results.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "results" / "dispatch_benchmark_report.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BenchmarkConfig(
        seeds=args.seeds,
        train_episodes=args.train_episodes,
        candidate_budget=args.k,
        latency_budget_ms=args.latency_ms,
    )
    results = run_benchmark(config, args.json, args.report)
    print(
        f"wrote {len(results['scaling_records'])} scaling, "
        f"{len(results['rollout_records'])} rollout, and "
        f"{len(results['robustness_records'])} robustness records"
    )
    print(args.json)
    print(args.report)


if __name__ == "__main__":
    main()
