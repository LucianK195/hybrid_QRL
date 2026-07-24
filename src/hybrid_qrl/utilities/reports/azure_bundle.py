"""Azure bundle-conflict report renderer."""

from __future__ import annotations

from typing import Any

from ..reporting import aligned_markdown_table as _table
from ..reporting import find_summary_row as _find

def _metric(row: dict[str, Any], metric: str) -> str:
    return (
        f"{row[f'{metric}_mean']:.3f} "
        f"+/- {row[f'{metric}_ci95']:.3f}"
    )


def render_azure_bundle_report(results: dict[str, Any]) -> str:
    """Render the concise benchmark report from the recorded results."""

    config = results["config"]
    summary = results["summary"]
    target_nodes = max(config["bundle_nodes"])
    capacity = max(config["capacities"])
    target = _find(
        summary,
        method="rydberg_geometry",
        bundle_nodes=target_nodes,
        capacity=capacity,
        k=16,
    )
    ideal = _find(
        summary,
        method="blockade_exact_graph",
        bundle_nodes=target_nodes,
        capacity=capacity,
        k=16,
    )
    beam = _find(
        summary,
        method="beam_search",
        bundle_nodes=target_nodes,
        capacity=capacity,
        k=16,
    )
    repair = _find(
        summary,
        method="repair_only",
        bundle_nodes=target_nodes,
        capacity=capacity,
        k=16,
    )
    paired = results["paired_comparisons"]
    full_capacity_k16 = [
        row
        for row in summary
        if row["capacity"] == capacity and row["k"] == 16
    ]
    k_sweep = [
        row
        for row in summary
        if row["bundle_nodes"] == target_nodes
        and row["capacity"] == capacity
    ]
    pipeline = "PASS" if results["gates"]["pipeline_pass"] else "HOLD"
    contribution = (
        "PASS" if results["gates"]["sampler_contribution_pass"] else "HOLD"
    )
    lines = [
        "# Azure bundle-conflict benchmark",
        "",
        f"## Trace-to-bundle pipeline: {pipeline}",
        "",
        f"## Geometry Rydberg proposal contribution: {contribution}",
        "",
        (
            f"Two hundred held-out Azure requests were converted into up to "
            f"{target_nodes} capacity-feasible `(machine, bundle)` nodes for "
            f"{config['machine_slots']} machine slots. At full per-machine "
            f"capacity and K=16, the geometry Rydberg surrogate achieved "
            f"{_metric(target, 'best_bundle_ratio')} of the bundle-library "
            f"MILP and {_metric(target, 'best_end_to_end_ratio')} of the "
            "direct job-by-machine MILP."
        ),
        "",
        (
            "The direct assignment reference used a five-second time limit: "
            f"{results['oracle_summary']['direct_exact']} of "
            f"{results['oracle_summary']['direct_states']} solves were exact, "
            "and every remaining conservative upper bound was within 1% of "
            "its incumbent (maximum gap "
            f"{results['oracle_summary']['direct_maximum_mip_gap']:.3%}). "
            "All end-to-end ratios use the upper bound, not the incumbent."
        ),
        "",
        (
            f"The bundle library itself covered "
            f"{_metric(target, 'library_coverage')} of the direct MILP. "
            f"Raw geometry-sampler feasibility was "
            f"{_metric(target, 'raw_feasible_rate')}, and authoritative "
            f"repair removed {_metric(target, 'repair_removed_fraction')} "
            "of selected bundle nodes."
        ),
        "",
        (
            "The same blockade schedule on the exact conflict graph reached "
            f"{_metric(ideal, 'best_end_to_end_ratio')}; beam reached "
            f"{_metric(beam, 'best_end_to_end_ratio')}; and the repair-only "
            f"control reached {_metric(repair, 'best_end_to_end_ratio')}. "
            "The ideal-graph path is an algorithmic diagnostic, not a "
            "hardware-compatible result."
        ),
        "",
        (
            "The paired geometry-minus-repair-only end-to-end difference was "
            f"{paired['geometry_minus_repair_end_to_end']['mean']:.4f} +/- "
            f"{paired['geometry_minus_repair_end_to_end']['ci95']:.4f}. "
            "The paired exact-graph-minus-geometry difference was "
            f"{paired['exact_graph_minus_geometry_end_to_end']['mean']:.4f} "
            f"+/- "
            f"{paired['exact_graph_minus_geometry_end_to_end']['ci95']:.4f}."
        ),
        "",
        "## What changed from the raw-job benchmark",
        "",
        (
            "Capacity is now enforced before sampling: every node is a complete "
            "four-resource-feasible machine configuration. Same-machine and "
            "shared-request conflicts are exactly pairwise, so an independent "
            "set is an authoritative safe allocation without cumulative "
            "capacity repair. A separate direct assignment MILP measures the "
            "solution-space loss introduced by the finite bundle library."
        ),
        "",
        "## Full-capacity results at K=16",
        "",
        _table(
            [
                "nodes",
                "method",
                "best/bundle",
                "best/direct",
                "rerank/direct",
                "raw feasible",
                "repair removed",
                "library coverage",
                "latency ms",
            ],
            [
                [
                    str(row["bundle_nodes"]),
                    str(row["method"]),
                    _metric(row, "best_bundle_ratio"),
                    _metric(row, "best_end_to_end_ratio"),
                    _metric(row, "reranked_end_to_end_ratio"),
                    _metric(row, "raw_feasible_rate"),
                    _metric(row, "repair_removed_fraction"),
                    _metric(row, "library_coverage"),
                    f"{row['proposal_latency_ms_mean']:.2f}",
                ]
                for row in full_capacity_k16
            ],
        ),
        "",
        "## K sweep at 100 bundle nodes and full capacity",
        "",
        _table(
            [
                "K",
                "method",
                "best/direct",
                "eps-5% coverage",
                "unique",
                "diversity",
                "latency ms",
            ],
            [
                [
                    str(row["k"]),
                    str(row["method"]),
                    _metric(row, "best_end_to_end_ratio"),
                    _metric(row, "epsilon_coverage"),
                    _metric(row, "unique_feasible"),
                    _metric(row, "diversity"),
                    f"{row['proposal_latency_ms_mean']:.2f}",
                ]
                for row in k_sweep
            ],
        ),
        "",
        "## Geometry transfer",
        "",
        (
            f"At the target setting, exact-to-unit-disk edge Jaccard was "
            f"{_metric(target, 'geometry_jaccard')}. The exact bundle graph "
            f"density was {_metric(target, 'exact_graph_density')}, versus "
            f"{_metric(target, 'physical_graph_density')} for the fitted "
            "two-dimensional blockade graph."
        ),
        "",
        (
            "Because this graph is dense, edge Jaccard alone is misleading. "
            "Only "
            f"{_metric(target, 'compatibility_recall')} of exact compatible "
            "bundle pairs remained non-edges in the physical graph; the "
            f"false-blockade rate was {_metric(target, 'false_blockade_rate')}."
        ),
        "",
        "## Gates",
        "",
        _table(
            ["pipeline check", "result"],
            [
                [key.replace("_", " "), str(value)]
                for key, value in results["gates"]["checks"].items()
            ],
        ),
        "",
        _table(
            ["sampler check", "result"],
            [
                [key.replace("_", " "), str(value)]
                for key, value in results["gates"][
                    "sampler_contribution_checks"
                ].items()
            ],
        ),
        "",
        "## Claim boundary",
        "",
        (
            "This experiment validates the bundle reformulation on the official "
            "trace. It does not reproduce Azure production scheduling, use a "
            "physical neutral-atom backend, measure hardware latency, or "
            "establish quantum advantage."
        ),
        "",
    ]
    return "\n".join(lines)
