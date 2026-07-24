"""Shared Azure trace command scaffolding."""

from __future__ import annotations

import argparse
from abc import abstractmethod
from pathlib import Path
from typing import Any

from ..utilities import ExperimentCommand, workspace_root


class AzureTraceCommand(ExperimentCommand):
    """Common paths and lifecycle for Azure-derived benchmark datasets."""

    name = "benchmark"
    result_filename: str
    report_filename: str

    @property
    def help(self) -> str:
        return self.__doc__ or "Run the Azure trace benchmark."

    def configure(self, parser: argparse.ArgumentParser) -> None:
        root = workspace_root()
        parser.add_argument(
            "--sqlite",
            type=Path,
            default=(
                root
                / "datasets"
                / "azure_packing"
                / "raw"
                / "packing_trace_zone_a_v1.sqlite"
            ),
        )
        parser.add_argument(
            "--stable-results",
            type=Path,
            default=root / "results" / "stable_backlog_scaling_results.json",
        )
        parser.add_argument(
            "--json",
            type=Path,
            default=root / "results" / self.result_filename,
        )
        parser.add_argument(
            "--report",
            type=Path,
            default=root / "results" / self.report_filename,
        )

    def execute(self, args: argparse.Namespace) -> None:
        results = self.run_benchmark(args)
        self.print_summary(results)
        print(args.json)
        print(args.report)

    @abstractmethod
    def run_benchmark(self, args: argparse.Namespace) -> dict[str, Any]:
        """Execute the concrete Azure formulation."""

    @abstractmethod
    def print_summary(self, results: dict[str, Any]) -> None:
        """Print formulation-specific gate and oracle counts."""
