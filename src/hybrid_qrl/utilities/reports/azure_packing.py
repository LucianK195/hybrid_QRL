"""Azure Packing report renderer."""

from __future__ import annotations

from functools import partial
from typing import Any

from ..reporting import find_summary_row as _find
from ..reporting import markdown_table

_table = partial(markdown_table, padded_divider=True)

def _metric(row: dict[str, Any], name: str) -> str:
    return f"{row[name + '_mean']:.3f} +/- {row[name + '_ci95']:.3f}"


def render_azure_packing_report(results: dict[str, Any]) -> str:
    """Build the concise Markdown report from completed raw records."""

    config = results["config"]
    summary = results["summary"]
    full_capacity = max(config["capacities"])
    max_size = max(config["sizes"])
    primary = [
        row
        for row in summary
        if row["capacity"] == full_capacity and row["k"] == 16
    ]
    k_sweep = [
        row
        for row in summary
        if row["size"] == max_size and row["capacity"] == full_capacity
    ]
    capacity_sweep = [
        row
        for row in summary
        if row["size"] == max_size and row["k"] == 16
    ]
    target = _find(
        summary,
        method="rydberg_surrogate",
        size=max_size,
        capacity=full_capacity,
        k=16,
    )
    beam = _find(
        summary,
        method="beam_search",
        size=max_size,
        capacity=full_capacity,
        k=16,
    )
    repair = _find(
        summary,
        method="deterministic_repair",
        size=max_size,
        capacity=full_capacity,
        k=16,
    )
    pipeline_status = (
        "PASS" if results["gates"]["pipeline_pass"] else "HOLD"
    )
    contribution_status = (
        "PASS"
        if results["gates"]["sampler_contribution_pass"]
        else "HOLD"
    )
    paired = results["paired_comparisons"][
        "n100_full_capacity_k16_rydberg_minus_repair"
    ]
    lines = [
        "# Azure Packing 2020 hybrid benchmark",
        "",
        f"## Trace pipeline: {pipeline_status}",
        "",
        f"## Rydberg sampler contribution: {contribution_status}",
        "",
        (
            "The complete trace-to-action pipeline executed on official Azure "
            "VM requests with four cumulative resource constraints. At 100 "
            f"requests, full capacity, and K=16, the Rydberg surrogate reached "
            f"best-of-16 ratio {_metric(target, 'best_k_ratio')} and "
            f"critic-selected ratio {_metric(target, 'critic_selected_ratio')}. "
            f"Capacity-aware beam reached {_metric(beam, 'best_k_ratio')}, "
            "while applying the same repair to one deterministic all-selected "
            f"proposal reached {_metric(repair, 'best_k_ratio')}."
        ),
        "",
        (
            "The paired Rydberg-minus-deterministic-repair best-action difference "
            f"was {paired['mean']:.4f} +/- {paired['ci95']:.4f}. The Rydberg "
            f"raw capacity-feasibility rate was "
            f"{target['raw_capacity_feasible_rate_mean']:.1%}; therefore the "
            "high post-repair score is not evidence that the sampler itself "
            "improved this packing problem."
        ),
        "",
        (
            f"All {results['oracle_summary']['states']} MILP references completed "
            "with zero gap and every executed candidate was feasible after "
            "capacity repair."
        ),
        "",
        "## Data and model",
        "",
        (
            f"The SQLite trace contains {results['trace_profile']['vm_requests']:,} "
            "VM requests. The benchmark uses hardware generation "
            f"{config['machine_id']} with "
            f"{results['trace_profile']['selected_machine_vm_types']} compatible "
            "VM types. Thirty chronological windows from days 0.25--9.75 train "
            "the reward-only utility head and critic; 20 windows from days "
            "10.00--13.75 are held out."
        ),
        "",
        (
            f"The critic used {results['training']['critic_training_actions']:,} "
            "sampled feasible actions, achieved training "
            f"R2={results['training']['critic_r2']:.3f}, and received no MILP labels."
        ),
        "",
        "## Full-capacity comparison at K=16",
        "",
        _table(
            [
                "jobs",
                "method",
                "best/MILP",
                "critic/MILP",
                "eps-5% coverage",
                "raw capacity feasible",
                "repair removed",
                "utilization",
                "latency ms",
            ],
            [
                [
                    str(row["size"]),
                    str(row["method"]),
                    _metric(row, "best_k_ratio"),
                    _metric(row, "critic_selected_ratio"),
                    _metric(row, "epsilon_coverage"),
                    _metric(row, "raw_capacity_feasible_rate"),
                    _metric(row, "repair_removed_fraction"),
                    _metric(row, "mean_resource_utilization"),
                    f"{row['proposal_latency_ms_mean']:.2f}",
                ]
                for row in primary
            ],
        ),
        "",
        "## Candidate-budget sweep at 100 requests and full capacity",
        "",
        _table(
            [
                "K",
                "method",
                "best/MILP",
                "critic/MILP",
                "eps-5% coverage",
                "unique",
                "diversity",
                "latency ms",
            ],
            [
                [
                    str(row["k"]),
                    str(row["method"]),
                    _metric(row, "best_k_ratio"),
                    _metric(row, "critic_selected_ratio"),
                    _metric(row, "epsilon_coverage"),
                    f"{row['unique_feasible_mean']:.1f}",
                    _metric(row, "diversity"),
                    f"{row['proposal_latency_ms_mean']:.2f}",
                ]
                for row in k_sweep
            ],
        ),
        "",
        "## Capacity-pressure sweep at 100 requests and K=16",
        "",
        _table(
            [
                "capacity",
                "method",
                "best/MILP",
                "critic/MILP",
                "raw capacity feasible",
                "repair removed",
                "accepted VMs",
            ],
            [
                [
                    f"{row['capacity']:.2f}",
                    str(row["method"]),
                    _metric(row, "best_k_ratio"),
                    _metric(row, "critic_selected_ratio"),
                    _metric(row, "raw_capacity_feasible_rate"),
                    _metric(row, "repair_removed_fraction"),
                    _metric(row, "accepted_vms"),
                ]
                for row in capacity_sweep
            ],
        ),
        "",
        "## Gates",
        "",
        _table(
            ["check", "result"],
            [
                [key.replace("_", " "), str(value)]
                for key, value in results["gates"]["checks"].items()
            ],
        ),
        "",
        _table(
            ["sampler contribution check", "result"],
            [
                [key.replace("_", " "), str(value)]
                for key, value in results["gates"][
                    "sampler_contribution_checks"
                ].items()
            ],
        ),
        "",
        "## Interpretation",
        "",
        (
            "This is a real-trace algorithmic validation: observation window -> "
            "utility encoding -> candidate generator -> cumulative capacity "
            "repair -> learned critic -> safe admission action. It is not a "
            "reconstruction of Azure's production allocator. Lifetimes are used "
            "offline, low-priority lifetime is allocation-dependent in the source "
            "trace, and the neutral-atom result is still generated by the "
            "classical surrogate."
        ),
        "",
    ]
    return "\n".join(lines)
