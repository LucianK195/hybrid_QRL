"""Command-line application for the public Wi-Fi MWIS study."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..dispatch.wifi_mis import WifiMISConfig, run_wifi_mis_benchmark
from ..utilities import ExperimentApplication, ExperimentCommand, workspace_root


class WifiMISBenchmarkCommand(ExperimentCommand):
    """Run pulse selection, frozen evaluation, report, and figure generation."""

    name = "benchmark"
    help = "benchmark ideal Rydberg and classical samplers on Wi-Fi MWIS frames"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        root = workspace_root()
        parser.add_argument(
            "--json-path",
            type=str,
            default=str(root / "results" / "wifi_mis_results.json"),
        )
        parser.add_argument(
            "--report",
            type=str,
            default=str(root / "results" / "wifi_mis_academic_report.md"),
        )
        parser.add_argument(
            "--figures",
            type=str,
            default=str(root / "figures" / "wifi_mis"),
        )
        parser.add_argument("--pulse-seeds", type=int, default=12)
        parser.add_argument("--test-seeds", type=int, default=40)
        parser.add_argument("--classical-samples", type=int, default=4_096)
        parser.add_argument("--seed", type=int, default=24_081)
        parser.add_argument("--skip-figures", action="store_true")

    def validate(
        self, parser: argparse.ArgumentParser, args: argparse.Namespace
    ) -> None:
        if args.pulse_seeds < 4:
            parser.error("--pulse-seeds must be at least 4")
        if args.test_seeds < 10:
            parser.error("--test-seeds must be at least 10")
        if args.classical_samples < 256:
            parser.error("--classical-samples must be at least 256")

    def execute(self, args: argparse.Namespace) -> None:
        results = run_wifi_mis_benchmark(
            config=WifiMISConfig(
                pulse_training_seeds=args.pulse_seeds,
                test_seeds=args.test_seeds,
                classical_probability_samples=args.classical_samples,
                seed=args.seed,
            ),
            output_json=Path(args.json_path),
            output_report=Path(args.report),
            figure_dir=None if args.skip_figures else Path(args.figures),
        )
        gates = results["gates"]
        print(
            "limited ideal-sampler advantage pass: "
            f"{gates['limited_ideal_sampler_advantage_pass']}"
        )
        print(
            "beats simulated annealing paired lower CI: "
            f"{gates['beats_simulated_annealing_paired_lower_ci']}"
        )
        print(f"physical QPU evidence: {gates['physical_qpu_evidence']}")


WifiMISApplication = ExperimentApplication(
    description="Public Wi-Fi neutral-atom MWIS benchmark",
    commands=(WifiMISBenchmarkCommand(),),
    default_command="benchmark",
)


__all__ = ["WifiMISApplication", "WifiMISBenchmarkCommand"]
