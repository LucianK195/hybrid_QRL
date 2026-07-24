"""Azure Packing trace dataset application."""

from __future__ import annotations

import argparse
from typing import Any

from ..dispatch.azure_packing import (
    AzurePackingConfig,
    run_azure_packing_benchmark,
)
from ..utilities import ExperimentApplication
from ._azure import AzureTraceCommand


class PackingBenchmarkCommand(AzureTraceCommand):
    """Run the official Azure Packing 2020 trace benchmark."""

    result_filename = "azure_packing_benchmark.json"
    report_filename = "azure_packing_benchmark.md"

    def run_benchmark(self, args: argparse.Namespace) -> dict[str, Any]:
        return run_azure_packing_benchmark(
            sqlite_path=args.sqlite,
            stable_results_path=args.stable_results,
            output_json=args.json,
            output_report=args.report,
            config=AzurePackingConfig(),
        )

    def print_summary(self, results: dict[str, Any]) -> None:
        print(f"pipeline pass: {results['gates']['pipeline_pass']}")
        print(
            "sampler contribution pass: "
            f"{results['gates']['sampler_contribution_pass']}"
        )
        print(
            f"MILP exact: {results['oracle_summary']['exact']}/"
            f"{results['oracle_summary']['states']}"
        )


AzurePackingApplication = ExperimentApplication(
    description="Run experiments on the Azure Packing trace dataset.",
    commands=(PackingBenchmarkCommand(),),
    default_command="benchmark",
)
