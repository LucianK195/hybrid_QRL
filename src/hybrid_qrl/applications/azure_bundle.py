"""Azure bundle-conflict dataset application."""

from __future__ import annotations

import argparse
from typing import Any

from ..dispatch.azure_bundle import (
    AzureBundleConfig,
    ExternalPortfolioConfig,
    run_azure_bundle_benchmark,
    run_external_portfolio_benchmark,
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


class ModularBundleBenchmarkCommand(BundleBenchmarkCommand):
    """Run the frozen eight-machine modular Rydberg comparison."""

    name = "modular"
    result_filename = "azure_modular_rydberg_results.json"
    report_filename = "azure_modular_rydberg_report.md"

    def run_benchmark(self, args: argparse.Namespace) -> dict[str, Any]:
        return run_azure_bundle_benchmark(
            sqlite_path=args.sqlite,
            stable_results_path=args.stable_results,
            output_json=args.json,
            output_report=args.report,
            config=AzureBundleConfig(
                machine_slots=8,
                bundle_nodes=(96,),
                capacities=(0.25,),
                k_values=(1, 4, 8, 16, 32, 64),
                train_day_start=0.25,
                train_day_end=5.5,
                test_day_start=10.0,
                test_day_end=13.75,
                direct_milp_time_limit_ms=10_000.0,
                bundle_milp_time_limit_ms=5_000.0,
                sampler_regime="standardized-050-adiabatic",
                primary_method="modular_rydberg",
                comparison_method="randomized_greedy",
                primary_k=8,
            ),
        )


class XYQAOABundleBenchmarkCommand(BundleBenchmarkCommand):
    """Run the frozen one-hot XY/QAOA Azure comparison."""

    name = "xy-qaoa"
    result_filename = "azure_xy_qaoa_results.json"
    report_filename = "azure_xy_qaoa_report.md"

    def run_benchmark(self, args: argparse.Namespace) -> dict[str, Any]:
        return run_azure_bundle_benchmark(
            sqlite_path=args.sqlite,
            stable_results_path=args.stable_results,
            output_json=args.json,
            output_report=args.report,
            config=AzureBundleConfig(
                machine_slots=8,
                bundle_nodes=(96,),
                capacities=(0.25,),
                k_values=(1, 4, 8, 16, 32, 64),
                train_day_start=0.25,
                train_day_end=5.5,
                test_day_start=10.0,
                test_day_end=13.75,
                direct_milp_time_limit_ms=10_000.0,
                bundle_milp_time_limit_ms=5_000.0,
                sampler_regime="standardized-050-adiabatic",
                primary_method="modular_xy_qaoa",
                comparison_method="randomized_greedy",
                primary_k=8,
                quantum_walk_gamma=0.8,
                quantum_walk_beta=1.0,
                quantum_walk_depth=2,
            ),
        )


class PairedGroverBundleBenchmarkCommand(BundleBenchmarkCommand):
    """Run the frozen paired-machine Grover/QAOA comparison."""

    name = "paired-grover"
    result_filename = "azure_paired_grover_results.json"
    report_filename = "azure_paired_grover_report.md"

    def run_benchmark(self, args: argparse.Namespace) -> dict[str, Any]:
        return run_azure_bundle_benchmark(
            sqlite_path=args.sqlite,
            stable_results_path=args.stable_results,
            output_json=args.json,
            output_report=args.report,
            config=AzureBundleConfig(
                machine_slots=8,
                bundle_nodes=(96,),
                capacities=(0.25,),
                k_values=(1, 4, 8, 16, 32, 64),
                train_day_start=0.25,
                train_day_end=5.5,
                test_day_start=10.0,
                test_day_end=13.75,
                direct_milp_time_limit_ms=10_000.0,
                bundle_milp_time_limit_ms=5_000.0,
                sampler_regime="standardized-050-adiabatic",
                primary_method="paired_grover_qaoa",
                comparison_method="randomized_greedy",
                primary_k=8,
                quantum_walk_gamma=0.8,
                quantum_walk_beta=1.2,
                quantum_walk_depth=3,
            ),
        )


class ExternalPortfolioBenchmarkCommand(BundleBenchmarkCommand):
    """Validate the frozen quantum portfolio on unseen Azure generations."""

    name = "external-portfolio"
    result_filename = "azure_external_portfolio_results.json"
    report_filename = "azure_external_portfolio_report.md"

    def run_benchmark(self, args: argparse.Namespace) -> dict[str, Any]:
        return run_external_portfolio_benchmark(
            sqlite_path=args.sqlite,
            stable_results_path=args.stable_results,
            output_json=args.json,
            output_report=args.report,
            config=ExternalPortfolioConfig(),
        )

    def print_summary(self, results: dict[str, Any]) -> None:
        gates = results["gates"]
        print(f"pipeline pass: {gates['pipeline_pass']}")
        print(
            "potential advantage pass: "
            f"{gates['potential_advantage_pass']}"
        )
        print(
            "strong classical advantage pass: "
            f"{gates['strong_classical_advantage_pass']}"
        )


AzureBundleApplication = ExperimentApplication(
    description="Run experiments on the Azure bundle-conflict graph dataset.",
    commands=(
        BundleBenchmarkCommand(),
        ModularBundleBenchmarkCommand(),
        XYQAOABundleBenchmarkCommand(),
        PairedGroverBundleBenchmarkCommand(),
        ExternalPortfolioBenchmarkCommand(),
    ),
    default_command="benchmark",
)
