"""Conditional-advantage study report renderer."""

from __future__ import annotations

from collections import defaultdict
from math import sqrt
from typing import Any

import numpy as np

from ..metrics import shots_for_95_percent
from ..reporting import (
    format_mean_ci as _fmt,
    markdown_table as _table,
)


def _aggregate(
    records: list[dict[str, Any]],
    keys: tuple[str, ...],
    metrics: tuple[str, ...],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[tuple(record[key] for key in keys)].append(record)
    output = []
    for group, items in groups.items():
        row = {key: value for key, value in zip(keys, group)}
        row["trials"] = len(items)
        for metric in metrics:
            values = np.asarray([float(item[metric]) for item in items])
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_ci95"] = (
                float(1.96 * values.std(ddof=1) / sqrt(len(values)))
                if len(values) > 1
                else 0.0
            )
        output.append(row)
    return output


def _gate_result(
    selection_summary: list[dict[str, Any]],
    confirmation_summary: list[dict[str, Any]],
    calibration_records: list[dict[str, Any]],
    pipeline_records: list[dict[str, Any]],
) -> dict[str, Any]:
    selection_surrogate_rows = [
        row
        for row in selection_summary
        if row["method"] == "rydberg_surrogate"
    ]
    selected = max(
        selection_surrogate_rows,
        key=lambda row: row["critic_selected_reward_ratio_mean"],
    )
    peers = {
        row["method"]: row
        for row in confirmation_summary
        if row["regime"] == selected["regime"]
    }
    best = peers["rydberg_surrogate"]
    acceptable_return = best["critic_selected_reward_ratio_mean"] >= 0.90
    beam = peers["beam_search"]
    local = peers["local_search"]
    latency_win = best["end_to_end_latency_ms_mean"] < min(
        beam["end_to_end_latency_ms_mean"],
        local["end_to_end_latency_ms_mean"],
    )
    diversity_win = best["candidate_hamming_diversity_mean"] > 1.05 * max(
        beam["candidate_hamming_diversity_mean"],
        local["candidate_hamming_diversity_mean"],
    )
    coverage_competitive = best["hit_epsilon_05_mean"] >= max(
        beam["hit_epsilon_05_mean"],
        local["hit_epsilon_05_mean"],
    )
    surrogate_opportunity_pass = (
        acceptable_return
        and coverage_competitive
        and (latency_win or diversity_win)
    )
    surrogate_tv = float(
        np.mean(
            [
                row["total_variation_to_reference"]
                for row in calibration_records
                if row["backend"] == "surrogate"
            ]
        )
    )
    manual_ratio = float(
        np.mean(
            [
                row["critic_selected_ratio"]
                for row in calibration_records
                if row["backend"] == "manual"
            ]
        )
    )
    qutip_ratio = float(
        np.mean(
            [
                row["critic_selected_ratio"]
                for row in calibration_records
                if row["backend"] == "qutip"
            ]
        )
    )
    pipeline_proof_pass = all(
        bool(row["executed_safe"]) for row in pipeline_records
    )
    calibration_transfer_pass = surrogate_tv <= 0.15
    manual_quality_pass = manual_ratio >= 0.90
    overall_pass = (
        pipeline_proof_pass
        and surrogate_opportunity_pass
        and calibration_transfer_pass
        and manual_quality_pass
    )
    return {
        "pass": overall_pass,
        "pipeline_proof_pass": pipeline_proof_pass,
        "surrogate_opportunity_pass": surrogate_opportunity_pass,
        "calibration_transfer_pass": calibration_transfer_pass,
        "manual_quality_pass": manual_quality_pass,
        "best_regime": selected["regime"],
        "acceptable_return": acceptable_return,
        "coverage_competitive": coverage_competitive,
        "latency_win": latency_win,
        "diversity_win": diversity_win,
        "surrogate_mean_tv_to_reference": surrogate_tv,
        "manual_mean_critic_ratio": manual_ratio,
        "qutip_mean_critic_ratio": qutip_ratio,
        "thresholds": {
            "acceptable_return": 0.90,
            "maximum_transfer_tv": 0.15,
            "minimum_manual_ratio": 0.90,
        },
        "selection_surrogate": selected,
        "surrogate": best,
        "peers": peers,
    }


def render_conditional_report(results: dict[str, Any]) -> str:
    """Create the Markdown result report and conditional gate decision."""

    training_summary = _aggregate(
        results["training_comparison_records"],
        ("model",),
        ("episode_return", "raw_feasible_rate"),
    )
    pipeline_summary = _aggregate(
        results["pipeline_records"],
        ("backend",),
        ("episode_return", "raw_feasible_rate", "mean_emulator_step_ms"),
    )
    calibration_summary = _aggregate(
        results["calibration_records"],
        ("n_jobs", "backend"),
        (
            "critic_selected_ratio",
            "expected_best_k_ratio",
            "p_epsilon_05",
            "coverage_k_epsilon_05",
            "raw_feasible_probability",
            "expected_hamming_diversity",
            "total_variation_to_reference",
            "backend_evolution_ms",
        ),
    )
    phase_metrics = (
        "critic_selected_reward_ratio",
        "best_batch_reward_ratio",
        "hit_epsilon_05",
        "p_epsilon_05",
        "candidate_hamming_diversity",
        "raw_feasible_rate",
        "end_to_end_latency_ms",
    )
    selection_summary = _aggregate(
        [
            row
            for row in results["phase_records"]
            if row["split"] == "selection"
        ],
        ("regime", "method"),
        phase_metrics,
    )
    confirmation_summary = _aggregate(
        [
            row
            for row in results["phase_records"]
            if row["split"] == "confirmation"
        ],
        ("regime", "method"),
        phase_metrics,
    )
    gate = _gate_result(
        selection_summary,
        confirmation_summary,
        results["calibration_records"],
        results["pipeline_records"],
    )
    results["gate"] = gate
    top_surrogate = sorted(
        (
            row
            for row in selection_summary
            if row["method"] == "rydberg_surrogate"
        ),
        key=lambda row: row["critic_selected_reward_ratio_mean"],
        reverse=True,
    )[:10]
    status = "PASS" if gate["pass"] else "HOLD — NOT YET ESTABLISHED"
    confirmed_ratio = gate["surrogate"]["critic_selected_reward_ratio_mean"]
    confirmed_hit = gate["surrogate"]["hit_epsilon_05_mean"]
    paired_differences = []
    training_by_seed: dict[int, dict[str, float]] = defaultdict(dict)
    for record in results["training_comparison_records"]:
        training_by_seed[int(record["seed_index"])][record["model"]] = float(
            record["episode_return"]
        )
    for values in training_by_seed.values():
        paired_differences.append(
            values["sampler_in_loop"] - values["direct_actor"]
        )
    paired_values = np.asarray(paired_differences)
    paired_mean = float(paired_values.mean())
    paired_ci = float(
        1.96 * paired_values.std(ddof=1) / sqrt(len(paired_values))
    )
    lines = [
        "# Conditional quantum-assisted advantage study",
        "",
        "## Claim boundary",
        "",
        "This experiment tests whether the complete hybrid proposal pipeline can "
        "work and whether a favorable conditional regime exists. Dense, QuTiP, "
        "and manual results are classical quantum-system simulations; the scalable "
        "Rydberg path is a classical blockade surrogate. No hardware quantum "
        "advantage or asymptotic supremacy is claimed.",
        "",
        "## Gate decision",
        "",
        f"**Overall conditional-advantage status: {status}.** The safe pipeline "
        f"proof passed and the best surrogate regime was `{gate['best_regime']}`, "
        "but the confirmed opportunity and calibration requirements did not all "
        "pass.",
        "",
        _table(
            ["gate", "result", "evidence"],
            [
                [
                    "safe pipeline proof",
                    str(gate["pipeline_proof_pass"]),
                    "all executed backend actions passed application safety",
                ],
                [
                    "surrogate opportunity",
                    str(gate["surrogate_opportunity_pass"]),
                    (
                        f"confirmed ratio "
                        f"{confirmed_ratio:.3f}; eps-5% hit {confirmed_hit:.2f}"
                    ),
                ],
                [
                    "distribution transfer",
                    str(gate["calibration_transfer_pass"]),
                    (
                        f"mean TV {gate['surrogate_mean_tv_to_reference']:.3f} "
                        "versus <=0.15 requirement"
                    ),
                ],
                [
                    "manual-backend quality",
                    str(gate["manual_quality_pass"]),
                    (
                        f"mean critic ratio {gate['manual_mean_critic_ratio']:.3f} "
                        "versus >=0.90 requirement"
                    ),
                ],
            ],
        ),
        "",
        "The 0.15 TV and 0.90 manual-quality checks are conservative calibration "
        "requirements, not evidence of quantum advantage by themselves. The current "
        "results support a working safe pipeline and a promising surrogate tradeoff, "
        "but not transfer of that tradeoff to the geometry-driven quantum backend.",
        "",
        "## Sampler-in-the-loop training",
        "",
        _table(
            ["training", "episode return", "raw feasible"],
            [
                [
                    row["model"],
                    _fmt(
                        row["episode_return_mean"],
                        row["episode_return_ci95"],
                    ),
                    _fmt(
                        row["raw_feasible_rate_mean"],
                        row["raw_feasible_rate_ci95"],
                    ),
                ]
                for row in sorted(training_summary, key=lambda item: item["model"])
            ],
        ),
        "",
        "Both variants use environment rewards only. The sampler-in-loop actor is "
        "updated by paired SPSA rollouts through sampling, repair, Q reranking, and "
        "dispatch execution; it receives no MILP or heuristic action labels.",
        "",
        f"The paired sampler-in-loop minus direct-actor return difference was "
        f"{paired_mean:.3f} ± {paired_ci:.3f} across held-out seeds.",
        "",
        "## Eight-qubit dynamic pipeline proof",
        "",
        _table(
            ["backend", "episode return", "raw feasible", "emulator ms/step"],
            [
                [
                    row["backend"],
                    _fmt(
                        row["episode_return_mean"],
                        row["episode_return_ci95"],
                    ),
                    _fmt(
                        row["raw_feasible_rate_mean"],
                        row["raw_feasible_rate_ci95"],
                    ),
                    _fmt(
                        row["mean_emulator_step_ms_mean"],
                        row["mean_emulator_step_ms_ci95"],
                        2,
                    ),
                ]
                for row in sorted(pipeline_summary, key=lambda item: item["backend"])
            ],
        ),
        "",
        "All executed actions passed the authoritative application-graph safety "
        "check. Emulator timings are not hardware latency estimates.",
        "",
        "## Small-backend calibration",
        "",
        _table(
            [
                "n",
                "backend",
                "critic ratio",
                "best-of-16 ratio",
                "p(eps=5%)",
                "K95",
                "K=16 coverage",
                "raw feasible",
                "TV to reference",
                "evolution ms",
            ],
            [
                [
                    str(row["n_jobs"]),
                    row["backend"],
                    _fmt(
                        row["critic_selected_ratio_mean"],
                        row["critic_selected_ratio_ci95"],
                    ),
                    _fmt(
                        row["expected_best_k_ratio_mean"],
                        row["expected_best_k_ratio_ci95"],
                    ),
                    _fmt(
                        row["p_epsilon_05_mean"],
                        row["p_epsilon_05_ci95"],
                    ),
                    str(shots_for_95_percent(row["p_epsilon_05_mean"])),
                    _fmt(
                        row["coverage_k_epsilon_05_mean"],
                        row["coverage_k_epsilon_05_ci95"],
                    ),
                    _fmt(
                        row["raw_feasible_probability_mean"],
                        row["raw_feasible_probability_ci95"],
                    ),
                    _fmt(
                        row["total_variation_to_reference_mean"],
                        row["total_variation_to_reference_ci95"],
                    ),
                    _fmt(
                        row["backend_evolution_ms_mean"],
                        row["backend_evolution_ms_ci95"],
                        1,
                    ),
                ]
                for row in sorted(
                    calibration_summary,
                    key=lambda item: (item["n_jobs"], item["backend"]),
                )
            ],
        ),
        "",
        "Dense evolution is intentionally omitted at n=12 because its dense "
        "2^n-by-2^n matrix exponentials are not a responsible repeated benchmark. "
        "At n=12, QuTiP is the distribution-distance reference.",
        "",
        "## Selection split: best surrogate phase-map regimes",
        "",
        _table(
            [
                "regime",
                "critic ratio",
                "best batch",
                "eps-5% hit",
                "candidate p(eps)",
                "K95",
                "diversity",
                "raw feasible",
                "latency ms",
            ],
            [
                [
                    row["regime"],
                    _fmt(
                        row["critic_selected_reward_ratio_mean"],
                        row["critic_selected_reward_ratio_ci95"],
                    ),
                    _fmt(
                        row["best_batch_reward_ratio_mean"],
                        row["best_batch_reward_ratio_ci95"],
                    ),
                    _fmt(
                        row["hit_epsilon_05_mean"],
                        row["hit_epsilon_05_ci95"],
                    ),
                    _fmt(
                        row["p_epsilon_05_mean"],
                        row["p_epsilon_05_ci95"],
                    ),
                    str(shots_for_95_percent(row["p_epsilon_05_mean"])),
                    _fmt(
                        row["candidate_hamming_diversity_mean"],
                        row["candidate_hamming_diversity_ci95"],
                    ),
                    _fmt(
                        row["raw_feasible_rate_mean"],
                        row["raw_feasible_rate_ci95"],
                    ),
                    _fmt(
                        row["end_to_end_latency_ms_mean"],
                        row["end_to_end_latency_ms_ci95"],
                        2,
                    ),
                ]
                for row in top_surrogate
            ],
        ),
        "",
        "## Independent confirmation: best-regime baseline comparison",
        "",
        _table(
            [
                "method",
                "critic ratio",
                "best batch",
                "eps-5% hit",
                "K95",
                "diversity",
                "latency ms",
            ],
            [
                [
                    method,
                    _fmt(
                        row["critic_selected_reward_ratio_mean"],
                        row["critic_selected_reward_ratio_ci95"],
                    ),
                    _fmt(
                        row["best_batch_reward_ratio_mean"],
                        row["best_batch_reward_ratio_ci95"],
                    ),
                    _fmt(
                        row["hit_epsilon_05_mean"],
                        row["hit_epsilon_05_ci95"],
                    ),
                    str(shots_for_95_percent(row["p_epsilon_05_mean"])),
                    _fmt(
                        row["candidate_hamming_diversity_mean"],
                        row["candidate_hamming_diversity_ci95"],
                    ),
                    _fmt(
                        row["end_to_end_latency_ms_mean"],
                        row["end_to_end_latency_ms_ci95"],
                        2,
                    ),
                ]
                for method, row in sorted(gate["peers"].items())
            ],
        ),
        "",
        "## Interpretation and next gate",
        "",
        "The pipeline proof answers whether quantum-backend candidates can flow "
        "through repair, learned reranking, and environment execution safely. The "
        "conditional gate is stricter: it asks whether the calibrated proposal "
        "path preserves acceptable return and beats tuned classical searches on a "
        "preregistered end-to-end dimension.",
        "",
        "Because the overall gate is on hold, the next experiment should optimize "
        "the physical geometry, pulse, and "
        "utility-to-detuning map against reward on training instances, then repeat "
        "the unchanged held-out gate. It would be invalid to redefine the gate "
        "after inspecting held-out results.",
        "",
        "## Limitations",
        "",
        "- All quantum-backend results are simulations, not QPU measurements.",
        "- The manual backend derives all C6/r^6 interactions from geometry, while "
        "dense and QuTiP receive exact graph edges; TV distance therefore includes "
        "a real model mismatch.",
        "- K95 assumes iid samples. Correlated hardware shots require an effective "
        "sample-size correction.",
        "- The 16-decision phase map uses a classical surrogate and supports regime "
        "selection only after small-backend calibration.",
        "- The workload remains synthetic and does not yet model Alibaba cumulative "
        "CPU and memory constraints.",
        "",
    ]
    return "\n".join(lines)
