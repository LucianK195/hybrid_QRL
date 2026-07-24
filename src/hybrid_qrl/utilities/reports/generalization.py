"""Dispatch generalization report renderer."""

from __future__ import annotations

from functools import partial
from typing import Any

from ..reporting import find_summary_row as _find
from ..reporting import markdown_table

_table = partial(markdown_table, padded_divider=True)

def _metric(row: dict[str, Any], name: str) -> str:
    mean = row[name + "_mean"]
    interval = row[name + "_ci95"]
    if mean is None or interval is None:
        return "n/a"
    valid = row.get(name + "_valid_trials", row["trials"])
    suffix = "" if valid == row["trials"] else f" ({valid}/{row['trials']})"
    return f"{mean:.3f} +/- {interval:.3f}{suffix}"


def render_generalization_report(results: dict[str, Any]) -> str:
    """Build the human-readable report from raw and summarized records."""

    config = results["config"]
    size_k = results["size_k_summary"]
    constraints = results["constraint_summary"]
    datasets = results["dataset_summary"]
    n100 = [
        row
        for row in size_k
        if row["method"] == "scale_aware" and row["size"] == max(config["sizes"])
    ]
    k1 = _find(n100, k=min(config["k_values"]))
    k16 = _find(n100, k=config["fixed_k"])
    k64 = _find(n100, k=max(config["k_values"]))
    beam16 = _find(
        size_k,
        method="beam_search",
        size=max(config["sizes"]),
        k=config["fixed_k"],
    )
    worst_constraint = min(
        (row for row in constraints if row["method"] == "scale_aware"),
        key=lambda row: row["best_k_opportunity_ratio_mean"],
    )
    worst_dataset = min(
        (row for row in datasets if row["method"] == "scale_aware"),
        key=lambda row: row["best_k_ratio_mean"],
    )
    status = (
        "PASS" if results["gates"]["candidate_generation_pass"] else "HOLD"
    )
    lines = [
        "# Dispatch generalization and candidate-budget stress test",
        "",
        f"## Result: {status}",
        "",
        (
            "The frozen scale-aware Rydberg surrogate was tested without "
            "retraining or retuning. At 100 jobs, best-of-K increased from "
            f"{k1['best_k_ratio_mean']:.3f} at K=1 to "
            f"{k16['best_k_ratio_mean']:.3f} at K=16 and "
            f"{k64['best_k_ratio_mean']:.3f} at K=64. The paired beam-search "
            f"best-of-16 ratio was {beam16['best_k_ratio_mean']:.3f}."
        ),
        "",
        (
            "The worst constraint cell for the surrogate was "
            f"n={worst_constraint['size']}, density={worst_constraint['density']}, "
            f"deadlines={worst_constraint['deadline_profile']}, with best-of-16 "
            "opportunity score "
            f"{_metric(worst_constraint, 'best_k_opportunity_ratio')}. The worst "
            f"dataset shift was `{worst_dataset['setting']}`, with ratio "
            f"{_metric(worst_dataset, 'best_k_ratio')}."
        ),
        "",
        "## Frozen protocol",
        "",
        (
            f"The study used {config['seeds']} paired held-out seeds per cell, "
            "the frozen multi-size model, the previously selected "
            f"`{results['selected_regime']['name']}` regime, and a paired MILP "
            "reward reference. No new setting was selected on these records."
        ),
        "",
        "## Job-count and K scaling",
        "",
        _table(
            [
                "jobs", "K", "method", "best/MILP", "critic/MILP",
                "eps-5% coverage", "opportunity", "raw feasible", "unique",
                "latency ms",
            ],
            [
                [
                    str(row["size"]), str(row["k"]), str(row["method"]),
                    _metric(row, "best_k_ratio"),
                    _metric(row, "critic_selected_ratio"),
                    _metric(row, "epsilon_coverage"),
                    _metric(row, "best_k_opportunity_ratio"),
                    _metric(row, "raw_feasible_rate"),
                    f"{row['unique_feasible_mean']:.1f}",
                    f"{row['proposal_latency_ms_mean']:.2f}",
                ]
                for row in size_k
            ],
        ),
        "",
        "## Constraint-pressure sweep at K=16",
        "",
        _table(
            [
                "jobs", "density", "deadlines", "method", "best/MILP",
                "critic/MILP", "opportunity", "opp. eps-5% coverage",
                "raw feasible",
            ],
            [
                [
                    str(row["size"]), f"{row['density']:.2f}",
                    str(row["deadline_profile"]), str(row["method"]),
                    _metric(row, "best_k_ratio"),
                    _metric(row, "critic_selected_ratio"),
                    _metric(row, "best_k_opportunity_ratio"),
                    _metric(row, "opportunity_epsilon_coverage"),
                    _metric(row, "raw_feasible_rate"),
                ]
                for row in constraints
            ],
        ),
        "",
        "## Dataset-shift sweep at n=60 and K=16",
        "",
        _table(
            [
                "setting", "method", "best/MILP", "critic/MILP",
                "eps-5% coverage", "opportunity", "diversity",
            ],
            [
                [
                    str(row["setting"]), str(row["method"]),
                    _metric(row, "best_k_ratio"),
                    _metric(row, "critic_selected_ratio"),
                    _metric(row, "epsilon_coverage"),
                    _metric(row, "best_k_opportunity_ratio"),
                    _metric(row, "diversity"),
                ]
                for row in datasets
            ],
        ),
        "",
        "## Reference and safety checks",
        "",
        _table(
            ["check", "result"],
            [
                [key.replace("_", " "), str(value)]
                for key, value in results["gates"]["checks"].items()
            ],
        ),
        "",
        (
            f"MILP completed exactly on {results['oracle_summary']['exact']} of "
            f"{results['oracle_summary']['states']} distinct held-out states. "
            f"The MILP reward was positive on "
            f"{results['oracle_summary']['positive_reference_states']} states; "
            "reward/MILP is marked `n/a` for non-positive references. "
            "Post-repair feasibility is evaluated against the authoritative "
            "application conflict graph."
        ),
        "",
        "## Interpretation boundary",
        "",
        (
            "Best-of-K measures whether a strong action was present in the batch; "
            "critic-selected ratio measures whether the current learned critic "
            "would actually deploy it. The Rydberg generator evaluated here is a "
            "classical stochastic surrogate, so its quality and local runtime do "
            "not establish neutral-atom hardware performance or quantum advantage."
        ),
        "",
        (
            "For non-positive MILP rewards, the opportunity score is "
            "(reward - empty-action reward) / (MILP reward - empty-action reward). "
            "It maps the empty action to 0 and the MILP optimum to 1 without "
            "dividing by a zero or negative optimum."
        ),
        "",
    ]
    return "\n".join(lines)
