"""Run delayed and asynchronous dispatch with timestamped QPU latency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hybrid_qrl.dispatch.dataset import model_from_dict
from hybrid_qrl.dispatch.latency_benchmark import (
    LatencyAwareConfig,
    LatencyTrace,
    make_preregistered_stress_trace,
    run_latency_aware_benchmark,
    write_latency_trace,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    """Parse benchmark, latency-trace, and output settings."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conditional-results",
        type=Path,
        default=PROJECT_ROOT / "results" / "conditional_advantage_results.json",
    )
    parser.add_argument(
        "--latency-trace",
        type=Path,
        help=(
            "Timestamped measured-QPU JSON. If omitted, use the preregistered "
            "synthetic stress trace, which cannot pass the physical evidence gate."
        ),
    )
    parser.add_argument(
        "--stress-trace-output",
        type=Path,
        default=PROJECT_ROOT / "results" / "dispatch_latency_stress_trace.json",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=PROJECT_ROOT / "results" / "latency_aware_dispatch_results.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "results" / "latency_aware_dispatch_report.md",
    )
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--jobs", type=int, default=40)
    parser.add_argument("--horizon", type=int, default=18)
    parser.add_argument("--step-ms", type=float, default=1_000.0)
    parser.add_argument("--deadline-ms", type=float, default=3_000.0)
    parser.add_argument("--k", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    """Load the frozen model/gates, run the extension, and print outputs."""

    args = parse_args()
    conditional = json.loads(args.conditional_results.read_text(encoding="utf-8"))
    model = model_from_dict(conditional["sampler_in_loop_model"])
    if args.latency_trace is None:
        trace = make_preregistered_stress_trace()
        write_latency_trace(trace, args.stress_trace_output)
        trace_path = args.stress_trace_output
    else:
        trace = LatencyTrace.from_json(args.latency_trace)
        trace_path = args.latency_trace
    config = LatencyAwareConfig(
        seeds=args.seeds,
        n_jobs=args.jobs,
        horizon=args.horizon,
        candidate_budget=args.k,
        decision_step_ms=args.step_ms,
        quantum_deadline_ms=args.deadline_ms,
    )
    results = run_latency_aware_benchmark(
        model=model,
        conditional_gate=conditional["gate"],
        trace=trace,
        config=config,
        output_json=args.json,
        output_report=args.report,
    )
    print(f"latency trace: {trace_path}")
    print(f"wrote {len(results['policy_records'])} policy episodes")
    print(args.json)
    print(args.report)
    print(f"combined gate pass: {results['gate']['pass']}")


if __name__ == "__main__":
    main()
