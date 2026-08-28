from __future__ import annotations

import ast
import unittest
from pathlib import Path

from hybrid_qrl.applications.cartpole import CartPoleApplication
from hybrid_qrl.applications.dispatch import DispatchApplication
from hybrid_qrl.applications.azure_bundle import AzureBundleApplication
from hybrid_qrl.applications.wifi_mis import WifiMISApplication


class ExperimentLayoutTests(unittest.TestCase):
    def test_experiments_has_one_python_entry_per_dataset(self) -> None:
        experiments = Path(__file__).resolve().parents[1] / "experiments"
        entries = {path.name for path in experiments.glob("*.py")}
        self.assertEqual(
            entries,
            {
                "azure_bundle.py",
                "azure_packing.py",
                "cartpole.py",
                "dispatch.py",
                "wifi_mis.py",
            },
        )

    def test_cartpole_stages_remain_available(self) -> None:
        self.assertEqual(
            {command.name for command in CartPoleApplication.commands},
            {"benchmark", "budget-sweep", "multiseed"},
        )

    def test_dispatch_stages_remain_available(self) -> None:
        self.assertEqual(
            {command.name for command in DispatchApplication.commands},
            {
                "scaling",
                "conditional",
                "latency",
                "backlog",
                "generalization",
                "export",
                "plot",
            },
        )

    def test_azure_bundle_stages_remain_available(self) -> None:
        self.assertEqual(
            {command.name for command in AzureBundleApplication.commands},
            {
                "benchmark",
                "modular",
                "xy-qaoa",
                "paired-grover",
                "external-portfolio",
            },
        )

    def test_wifi_mis_stage_remains_available(self) -> None:
        self.assertEqual(
            {command.name for command in WifiMISApplication.commands},
            {"benchmark"},
        )

    def test_dispatch_modules_do_not_define_report_renderers(self) -> None:
        dispatch = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "hybrid_qrl"
            / "dispatch"
        )
        forbidden = []
        for path in dispatch.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in tree.body:
                if not isinstance(node, ast.FunctionDef):
                    continue
                if "report" in node.name:
                    forbidden.append(f"{path.name}:{node.name}")
        self.assertEqual(forbidden, [])

    def test_all_dispatch_reporters_live_in_utilities(self) -> None:
        reports = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "hybrid_qrl"
            / "utilities"
            / "reports"
        )
        self.assertEqual(
            {path.name for path in reports.glob("*.py")},
            {
                "__init__.py",
                "azure_bundle.py",
                "azure_packing.py",
                "backlog.py",
                "conditional.py",
                "dispatch.py",
                "generalization.py",
                "latency.py",
                "wifi_mis.py",
            },
        )


if __name__ == "__main__":
    unittest.main()
