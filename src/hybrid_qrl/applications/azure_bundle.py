"""Azure bundle-conflict dataset application."""

from __future__ import annotations

import argparse
from typing import Any

from ..dispatch.azure_bundle import (
    AzureBundleConfig,
    run_azure_bundle_benchmark,
)
from ..utilities import ExperimentApplication
from ._azure import AzureTraceCommand


class BundleBenchmarkCommand(AzureTraceCommand):
    """Run the configuration-based Azure bundle-conflict benchmark."""

    result_filename = "azure_bundle_benchmark.json"
    report_filename = "azure_bundle_benchmark.md"

    def run_benchmark(self, args: argparse.Namespace) -> dict[str, Any]:
        return run_azure_bundle_benchmark(
            sqlite_path=args.sqlite,
            stable_results_path=args.stable_results,
            output_json=args.json,
            output_report=args.report,
            config=AzureBundleConfig(),
        )

    def print_summary(self, results: dict[str, Any]) -> None:
        print(f"pipeline pass: {results['gates']['pipeline_pass']}")
        print(
            "sampler contribution pass: "
            f"{results['gates']['sampler_contribution_pass']}"
        )
        print(
            f"direct MILP exact: {results['oracle_summary']['direct_exact']}/"
            f"{results['oracle_summary']['direct_states']}"
        )
        print(
            f"bundle MILP exact: {results['oracle_summary']['bundle_exact']}/"
            f"{results['oracle_summary']['bundle_states']}"
        )


AzureBundleApplication = ExperimentApplication(
    description="Run experiments on the Azure bundle-conflict graph dataset.",
    commands=(BundleBenchmarkCommand(),),
    default_command="benchmark",
)
