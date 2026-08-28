"""Run shallow-QAOA versus direct-Rydberg Wi-Fi/MWIS comparison."""

from pathlib import Path

from hybrid_qrl.dispatch.wifi_mis_qaoa import run_qaoa_vs_rydberg


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    results = run_qaoa_vs_rydberg(
        baseline_json=root / "results" / "wifi_mis_results.json",
        output_json=root / "results" / "qaoa_vs_rydberg_results.json",
        output_report=root / "results" / "qaoa_vs_rydberg_report.md",
        figure_dir=root / "figures" / "wifi_mis",
    )
    print("comparison complete")
    print("p=1:", results["training"]["qaoa_p1"]["chosen"])
    print("p=2:", results["training"]["qaoa_p2"]["chosen"])
