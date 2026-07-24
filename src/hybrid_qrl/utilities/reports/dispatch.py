"""Dispatch scaling benchmark report renderer."""

from __future__ import annotations

from math import sqrt
from typing import Any

import numpy as np

from ..reporting import (
    format_mean_ci as _metric,
    markdown_table as _markdown_table,
)

def _aggregate(
    records: list[dict[str, Any]],
    group_keys: tuple[str, ...],
    metrics: tuple[str, ...],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in records:
        key = tuple(record[item] for item in group_keys)
        groups.setdefault(key, []).append(record)
    output = []
    for key, items in sorted(groups.items(), key=lambda pair: tuple(map(str, pair[0]))):
        row = {name: value for name, value in zip(group_keys, key)}
        row["trials"] = len(items)
        for metric in metrics:
            values = np.asarray([float(item[metric]) for item in items], dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_ci95"] = float(
                1.96 * np.std(values, ddof=1) / sqrt(len(values))
            ) if len(values) > 1 else 0.0
        output.append(row)
    return output


def render_dispatch_report(results: dict[str, Any]) -> str:
    """Render an auditable Markdown report from raw benchmark records."""

    scaling = results["scaling_records"]
    rollouts = results["rollout_records"]
    robustness = results["robustness_records"]
    training = results["training_history"]["episode_return"]
    metrics = (
        "reward_ratio",
        "normalized_regret",
        "optimum_coverage",
        "raw_feasible_rate",
        "unique_feasible",
        "end_to_end_latency_ms",
        "latency_compliant",
    )
    scale_summary = _aggregate(
        scaling,
        ("mode", "n_jobs", "method"),
        metrics,
    )
    equal_latency = [item for item in scaling if item["mode"] == "equal_latency"]
    compliance = float(np.mean([item["latency_compliant"] for item in equal_latency]))
    exact_rate = float(np.mean([item["oracle_exact"] for item in scaling + robustness]))

    def scale_mean(mode: str, size: int, method: str, metric: str) -> float:
        values = [
            float(item[metric])
            for item in scaling
            if item["mode"] == mode
            and item["n_jobs"] == size
            and item["method"] == method
        ]
        return float(np.mean(values))

    rollout_means = {
        method: float(
            np.mean(
                [
                    item["episode_return"]
                    for item in rollouts
                    if item["method"] == method
                ]
            )
        )
        for method in {item["method"] for item in rollouts}
    }
    best_rollout = max(rollout_means, key=rollout_means.get)
    lines = [
        "# Dynamic dispatch benchmark",
        "",
        "## Scope and claim boundary",
        "",
        f"The study uses {results['config']['seeds']} held-out seeds per cell and "
        "20--100 binary decisions on unit-disk conflict graphs. The policy and both "
        "critics are trained only from environment rewards. The Rydberg result is a "
        "classical blockade-dynamics surrogate, not quantum hardware evidence.",
        "",
        "The MILP reference directly maximizes realized one-step dispatch reward. "
        "The time-limited MILP baseline, like the other candidate generators, uses "
        "the frozen learned actor priorities and the learned Q critic for reranking.",
        "",
        "This report is a latency-aware reinterpretation of the existing records; "
        "it does not add hardware measurements or change any numerical result. The "
        "environment advances in abstract steps and does not define how much "
        "physical time one step represents.",
        "",
        "## Validity after physical-latency review",
        "",
        _markdown_table(
            ["claim", "status", "interpretation"],
            [
                [
                    "synthetic dispatch comparison",
                    "valid",
                    "paired algorithmic comparison on the defined unit-disk task",
                ],
                [
                    "reward-trained actor and critics",
                    "valid",
                    "training uses environment reward without oracle labels",
                ],
                [
                    "safe execution after repair",
                    "valid",
                    "executed actions satisfy the authoritative application graph",
                ],
                [
                    "Rydberg-surrogate scaling",
                    "valid in model",
                    "describes the implemented classical stochastic surrogate",
                ],
                [
                    "20 ms equal-latency comparison",
                    "local only",
                    "measures Python proposal, repair, and critic time on one host",
                ],
                [
                    "neutral-atom QPU latency or return",
                    "not tested",
                    "queue, preparation, shots, readout, and retrieval are absent",
                ],
                [
                    "real-time quantum advantage",
                    "not established",
                    "requires a physical deadline and hardware-in-loop evaluation",
                ],
            ],
        ),
        "",
        "The overall status therefore remains **HOLD**. Adding hardware latency is "
        "an additional gate; it cannot convert the present negative scaling and "
        "calibration evidence into a quantum-assisted advantage.",
        "",
        "## Latency accounting",
        "",
        "Recorded `end_to_end_latency_ms` values include local candidate generation, "
        "repair, deduplication, learned-Q reranking, and Python overhead. For the "
        "Rydberg surrogate they do **not** include cloud submission, queue waiting, "
        "atom preparation, physical shots, measurement, or result retrieval.",
        "",
        "A future hardware result must report both:",
        "",
        "- `T_local = T_encode + T_propose + T_repair + T_critic`; and",
        "- `T_hardware = T_submit + T_queue + T_prepare + T_shots + "
        "T_readout + T_retrieve + T_repair + T_critic`.",
        "",
        "The applicable feasibility condition is `T_hardware <= T_decision`, where "
        "`T_decision` is the scheduling deadline defined by the application.",
        "",
        "## RL training",
        "",
        f"Mean undiscounted return changed from {np.mean(training[:40]):.3f} over the "
        f"first 40 episodes to {np.mean(training[-40:]):.3f} over the final 40. "
        "No MILP, greedy, or teacher actions enter the updates.",
        "",
        "## Main findings",
        "",
        f"At equal K, the Rydberg surrogate's reward/reference ratio fell from "
        f"{scale_mean('equal_k', 20, 'rydberg_surrogate', 'reward_ratio'):.3f} at "
        f"n=20 to "
        f"{scale_mean('equal_k', 100, 'rydberg_surrogate', 'reward_ratio'):.3f} "
        f"at n=100. Beam search retained "
        f"{scale_mean('equal_k', 100, 'beam_search', 'reward_ratio'):.3f} at n=100. "
        "The observed scaling therefore does not support a quantum-advantage claim.",
        "",
        f"The best mean dynamic return was {best_rollout} at "
        f"{rollout_means[best_rollout]:.3f}. The Rydberg surrogate achieved "
        f"{rollout_means['rydberg_surrogate']:.3f}. Local equal-latency compliance "
        f"was {compliance:.1%}; this is not QPU deadline compliance. Non-compliant "
        "local trials remain identifiable in the JSON. "
        f"HiGHS completed {exact_rate:.1%} of recorded reference solves with zero "
        "reported MIP gap.",
        "",
    ]
    for mode, title in (
        ("equal_k", f"Equal K = {results['config']['candidate_budget']}"),
        (
            "equal_latency",
            f"Equal local software target = "
            f"{results['config']['latency_budget_ms']} ms",
        ),
    ):
        rows = []
        for item in scale_summary:
            if item["mode"] != mode:
                continue
            rows.append(
                [
                    str(item["n_jobs"]),
                    str(item["method"]),
                    _metric(item["reward_ratio_mean"], item["reward_ratio_ci95"]),
                    _metric(
                        item["normalized_regret_mean"],
                        item["normalized_regret_ci95"],
                    ),
                    _metric(
                        item["optimum_coverage_mean"],
                        item["optimum_coverage_ci95"],
                    ),
                    _metric(
                        item["unique_feasible_mean"],
                        item["unique_feasible_ci95"],
                        1,
                    ),
                    _metric(
                        item["raw_feasible_rate_mean"],
                        item["raw_feasible_rate_ci95"],
                    ),
                    _metric(
                        item["end_to_end_latency_ms_mean"],
                        item["end_to_end_latency_ms_ci95"],
                        2,
                    ),
                    (
                        "—"
                        if mode == "equal_k"
                        else f"{item['latency_compliant_mean']:.1%}"
                    ),
                ]
            )
        lines.extend(
            [
                f"## {title}",
                "",
                _markdown_table(
                    [
                        "n",
                        "method",
                        "reward / MILP ref.",
                        "norm. regret",
                        "opt. coverage",
                        "unique",
                        "raw feasible",
                        "local latency ms",
                        "within budget",
                    ],
                    rows,
                ),
                "",
            ]
        )

    rollout_summary = _aggregate(
        rollouts,
        ("method",),
        (
            "episode_return",
            "completion_value",
            "missed_value",
            "mean_decision_latency_ms",
        ),
    )
    lines.extend(["## Dynamic 12-step rollout", ""])
    lines.append(
        _markdown_table(
            [
                "method",
                "episode return",
                "completed value",
                "missed value",
                "local latency / step ms",
            ],
            [
                [
                    str(item["method"]),
                    _metric(item["episode_return_mean"], item["episode_return_ci95"]),
                    _metric(
                        item["completion_value_mean"], item["completion_value_ci95"]
                    ),
                    _metric(item["missed_value_mean"], item["missed_value_ci95"]),
                    _metric(
                        item["mean_decision_latency_ms_mean"],
                        item["mean_decision_latency_ms_ci95"],
                        2,
                    ),
                ]
                for item in rollout_summary
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Dynamic-latency limitation",
            "",
            "Every recorded rollout applies an action immediately to the same state "
            "that produced it. Consequently, the return values do not include stale "
            "observations or changes that occur while a remote QPU task is pending.",
            "",
            "If one environment step represents `Delta_t` milliseconds and a "
            "hardware request takes `T_hardware`, a latency-aware rollout should use "
            "`d = ceil(T_hardware / Delta_t)` delayed steps and execute an action "
            "computed from `s[t-d]` against the current state `s[t]`. The action must "
            "then be repaired and revalidated before execution.",
            "",
            "Until that experiment is run, the dynamic returns establish software "
            "pipeline quality only, not real-time hardware-in-loop performance.",
        ]
    )
    shift_records = [
        item for item in robustness if item["axis"] == "distribution_shift"
    ]
    shift_summary = _aggregate(
        shift_records,
        ("level", "method"),
        ("reward_ratio", "normalized_regret"),
    )
    lines.extend(["", "## Distribution-shift comparison", ""])
    lines.append(
        _markdown_table(
            ["shift", "method", "reward / MILP ref.", "norm. regret"],
            [
                [
                    str(item["level"]),
                    str(item["method"]),
                    _metric(item["reward_ratio_mean"], item["reward_ratio_ci95"]),
                    _metric(
                        item["normalized_regret_mean"],
                        item["normalized_regret_ci95"],
                    ),
                ]
                for item in shift_summary
            ],
        )
    )
    robust_surrogate = [
        item for item in robustness if item["method"] == "rydberg_surrogate"
    ]
    robust_summary = _aggregate(
        robust_surrogate,
        ("axis", "level"),
        ("reward_ratio", "normalized_regret", "raw_feasible_rate", "unique_feasible"),
    )
    lines.extend(["", "## Rydberg-surrogate sensitivity", ""])
    lines.append(
        _markdown_table(
            [
                "axis",
                "level",
                "reward / MILP ref.",
                "norm. regret",
                "raw feasible",
                "unique",
            ],
            [
                [
                    str(item["axis"]),
                    str(item["level"]),
                    _metric(item["reward_ratio_mean"], item["reward_ratio_ci95"]),
                    _metric(
                        item["normalized_regret_mean"],
                        item["normalized_regret_ci95"],
                    ),
                    _metric(
                        item["raw_feasible_rate_mean"],
                        item["raw_feasible_rate_ci95"],
                    ),
                    _metric(
                        item["unique_feasible_mean"],
                        item["unique_feasible_ci95"],
                        1,
                    ),
                ]
                for item in robust_summary
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Interpretation rules",
            "",
            "A growing action space (2^n) is not a quantum advantage. A promising "
            "trend requires the Rydberg path to retain reward ratio, candidate "
            "diversity, and feasibility as n and noise grow. The existing latency "
            "columns can support claims about local implementations only. They must "
            "not support a neutral-atom hardware-latency claim.",
            "",
            "Any local latency row exceeding 105% of the configured target is marked "
            "non-compliant in the raw JSON. A future physical claim additionally "
            "requires at least 95% of measured hardware requests to finish before the "
            "application deadline, together with safe post-arrival execution.",
            "",
            "MILP is exact for the default instances when HiGHS completes within the "
            "reported reference limit; otherwise its incumbent is only a lower-bound "
            "reference. The JSON retains per-instance oracle latency so such cases can "
            "be audited.",
            "",
            "## Required extension before a physical dispatch claim",
            "",
            "1. Assign a physical duration to one environment step and preregister "
            "decision deadlines.",
            "2. Measure a latency distribution rather than substituting emulator "
            "runtime for hardware time.",
            "3. Add delayed-action and stale-state rollouts with mandatory repair.",
            "4. Evaluate an asynchronous design in which beam or greedy search is the "
            "immediate fallback and quantum candidates target a future batch.",
            "5. Report reward, deadline misses, raw/post-repair feasibility, p95/p99 "
            "latency, shots, queue time, and quantum-result utilization.",
            "6. Retain the reward-ratio, epsilon-coverage, calibration-transfer, and "
            "manual-backend gates from the conditional-advantage study.",
            "",
            "## Revised conclusion",
            "",
            "The experiment remains a valid algorithmic benchmark and a safe "
            "simulated-pipeline demonstration. It shows that the present Rydberg "
            "surrogate loses relative reward as the action dimension grows and is "
            "outperformed by beam search. It does not show that a neutral-atom QPU "
            "can meet a real-time dispatch deadline. Real hardware latency and state "
            "staleness remain unmeasured, so the defensible status is **HOLD: no "
            "conditional quantum-assisted dispatch advantage is established**.",
            "",
        ]
    )
    return "\n".join(lines)
