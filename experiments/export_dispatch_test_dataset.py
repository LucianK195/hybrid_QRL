"""Export the benchmark's held-out graph states as a reusable test dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from hybrid_qrl.dispatch.dataset import export_test_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    """Parse source-result and dataset destination paths."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=PROJECT_ROOT / "results" / "dispatch_benchmark_results.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "dispatch_test_v1.jsonl",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "dispatch_test_v1_manifest.json",
    )
    return parser.parse_args()


def main() -> None:
    """Create the frozen test set and print its reproducibility digest."""

    args = parse_args()
    summary = export_test_dataset(args.results, args.output, args.manifest)
    print(f"wrote {summary.records} held-out graph instances")
    print(summary.output_path)
    print(summary.manifest_path)
    print(f"sha256 {summary.sha256}")


if __name__ == "__main__":
    main()
