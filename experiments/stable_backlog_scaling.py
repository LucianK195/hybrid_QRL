"""Run scale-aware fixed-K confirmation and stable-backlog rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hybrid_qrl.dispatch.backlog_benchmark import (
    BacklogBenchmarkConfig,
    run_backlog_benchmark,
)
from hybrid_qrl.dispatch.dataset import model_from_dict
from hybrid_qrl.dispatch.latency_benchmark import (
    LatencyTrace,
    make_preregistered_stress_trace,
)
from hybrid_qrl.dispatch.learning import TrainingConfig, train_actor_critic


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    """Parse frozen-model, latency, training, and output arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-results",
        type=Path,
        default=PROJECT_ROOT / "results" / "dispatch_benchmark_results.json",
    )
    parser.add_argument(
        "--conditional-results",
        type=Path,
        default=PROJECT_ROOT / "results" / "conditional_advantage_results.json",
    )
    parser.add_argument(
        "--latency-trace",
        type=Path,
        help="Measured timestamp JSON; omit to replay the synthetic stress trace.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=PROJECT_ROOT / "results" / "stable_backlog_scaling_results.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "results" / "stable_backlog_scaling_report.md",
    )
    parser.add_argument("--training-episodes", type=int, default=800)
    parser.add_argument("--confirmation-seeds", type=int, default=20)
    parser.add_argument("--dynamic-seeds", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=18)
    parser.add_argument("--k", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    """Train on environment reward, freeze the model, and run held-out tests."""

    args = parse_args()
    baseline_payload = json.loads(
        args.baseline_results.read_text(encoding="utf-8")
    )
    conditional_payload = json.loads(
        args.conditional_results.read_text(encoding="utf-8")
    )
    baseline_model = model_from_dict(baseline_payload["model"])
    config = BacklogBenchmarkConfig(
        training_episodes=args.training_episodes,
        confirmation_seeds=args.confirmation_seeds,
        dynamic_seeds=args.dynamic_seeds,
        horizon=args.horizon,
        candidate_budget=args.k,
    )
    model, history = train_actor_critic(
        TrainingConfig(
            episodes=config.training_episodes,
            horizon=18,
            train_sizes=(20, 40, 60, 80, 100),
            densities=(0.08, 0.12, 0.18),
            graph_families=("unit_disk", "grid", "clustered"),
            utility_correlations=("none", "spatial", "degree"),
            seed=config.training_seed,
        )
    )
    trace = (
        LatencyTrace.from_json(args.latency_trace)
        if args.latency_trace is not None
        else make_preregistered_stress_trace()
    )
    results = run_backlog_benchmark(
        model=model,
        baseline_model=baseline_model,
        conditional_gate=conditional_payload["gate"],
        trace=trace,
        config=config,
        output_json=args.json,
        output_report=args.report,
    )
    results["training_history_tail"] = {
        key: values[-20:] for key, values in history.items()
    }
    args.json.write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"selected regime: {results['selected_regime']['name']}")
    print(f"algorithmic pass: {results['gates']['algorithmic_pass']}")
    print(f"asynchronous pass: {results['gates']['asynchronous_pass']}")
    print(f"physical pass: {results['gates']['physical_pass']}")
    print(args.json)
    print(args.report)


if __name__ == "__main__":
    main()
