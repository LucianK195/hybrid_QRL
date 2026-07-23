"""Run the official Azure Packing 2020 trace benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from hybrid_qrl.dispatch.azure_packing import (
    AzurePackingConfig,
    run_azure_packing_benchmark,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    """Parse trace, source-model, and output paths."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sqlite",
        type=Path,
        default=(
            PROJECT_ROOT
            / "datasets"
            / "azure_packing"
            / "raw"
            / "packing_trace_zone_a_v1.sqlite"
        ),
    )
    parser.add_argument(
        "--stable-results",
        type=Path,
        default=PROJECT_ROOT / "results" / "stable_backlog_scaling_results.json",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=PROJECT_ROOT / "results" / "azure_packing_benchmark.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "results" / "azure_packing_benchmark.md",
    )
    return parser.parse_args()


def main() -> None:
    """Execute the frozen trace split and print gate outcomes."""

    args = parse_args()
    results = run_azure_packing_benchmark(
        sqlite_path=args.sqlite,
        stable_results_path=args.stable_results,
        output_json=args.json,
        output_report=args.report,
        config=AzurePackingConfig(),
    )
    print(f"pipeline pass: {results['gates']['pipeline_pass']}")
    print(
        "sampler contribution pass: "
        f"{results['gates']['sampler_contribution_pass']}"
    )
    print(
        f"MILP exact: {results['oracle_summary']['exact']}/"
        f"{results['oracle_summary']['states']}"
    )
    print(args.json)
    print(args.report)


if __name__ == "__main__":
    main()
