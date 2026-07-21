"""Run sampler-in-loop training, backend calibration, and phase-map search."""

from __future__ import annotations

import argparse
from pathlib import Path

from hybrid_qrl.dispatch.conditional_benchmark import (
    ConditionalAdvantageConfig,
    run_conditional_study,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--training-iterations", type=int, default=140)
    parser.add_argument("--calibration-seeds", type=int, default=20)
    parser.add_argument("--phase-seeds", type=int, default=40)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument(
        "--json",
        type=Path,
        default=PROJECT_ROOT / "results" / "conditional_advantage_results.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "results" / "conditional_advantage_report.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ConditionalAdvantageConfig(
        seeds=args.seeds,
        sampler_training_iterations=args.training_iterations,
        candidate_budget=args.k,
        calibration_seeds=args.calibration_seeds,
        phase_seeds=args.phase_seeds,
    )
    results = run_conditional_study(config, args.json, args.report)
    print(
        f"wrote {len(results['training_comparison_records'])} training, "
        f"{len(results['calibration_records'])} calibration, "
        f"{len(results['pipeline_records'])} pipeline, and "
        f"{len(results['phase_records'])} phase records"
    )
    print(args.json)
    print(args.report)


if __name__ == "__main__":
    main()
