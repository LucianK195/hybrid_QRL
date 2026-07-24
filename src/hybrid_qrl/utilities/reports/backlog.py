"""Stable-backlog study report renderer."""

from __future__ import annotations

from typing import Any

from ..reporting import format_mean_ci

def _fmt(mean_value: float, interval: float) -> str:
    return format_mean_ci(mean_value, interval, separator=" +/- ")


def render_backlog_report(results: dict[str, Any]) -> str:
    """Render a concise, auditable Markdown result report."""

    config = results["config"]
    selected = results["selected_regime"]
    scaling = results["scaling_summary"]
    future = results["future_summary"]
    latency = results["latency_summary"]
    gates = results["gates"]
    lines = [
        "# Scale-aware stable-backlog dispatch report",
        "",
        "## Claim boundary",
        "",
        "This experiment improves and evaluates a classical Rydberg-blockade "
        "surrogate. It does not establish neutral-atom hardware performance. "
        "The selected encoding must still transfer to dense, QuTiP, manual, "
        "and eventually measured-QPU executions.",
        "",
        f"**Algorithmic gate: {'PASS' if gates['algorithmic_pass'] else 'HOLD'}.**",
        f"**Asynchronous gate: {'PASS' if gates['asynchronous_pass'] else 'HOLD'}.**",
        f"**Physical gate: {'PASS' if gates['physical_pass'] else 'HOLD'}.**",
        "",
        "## Protocol",
        "",
        f"- K = {config['candidate_budget']} with no K increase.",
        f"- Reward-only multi-size training episodes = "
        f"{config['training_episodes']} (seed {config['training_seed']}).",
        f"- Sizes = {config['sizes']} and confirmation seeds = "
        f"{config['confirmation_seeds']} per size.",
        f"- Future deadline = {config['future_deadline_ms']:.0f} ms; "
        f"step duration = {config['decision_step_ms']:.0f} ms.",
        f"- Stable backlog guard = {config['stable_guard_steps']} step(s).",
        f"- Quantum target block = top {config['stable_target_fraction']:.0%} "
        f"of jobs, capped at {config['maximum_stable_target_jobs']} stable jobs.",
        f"- Selected on disjoint seeds: `{selected['name']}` "
        f"({selected['utility_encoding']}, gain={selected['detuning_gain']}, "
        f"schedule={selected['pulse_schedule']}).",
        "",
        "## Held-out best-of-K scaling",
        "",
        "Best-of-K is a diagnostic upper bound: it uses realized reward to "
        "identify the best sampled candidate. Critic and utility columns are "
        "deployable rerankers and do not see the oracle.",
        "",
        "| method | n | best-of-16 ratio | critic ratio | utility ratio | "
        "eps-5% coverage | candidate p(eps) | raw feasible | local ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in scaling:
        lines.append(
            "| {method} | {size} | {best} | {critic} | {utility} | "
            "{coverage} | {probability} | {feasible} | {latency} |".format(
                method=row["method"],
                size=row["size"],
                best=_fmt(
                    row["best_k_ratio_mean"],
                    row["best_k_ratio_ci95"],
                ),
                critic=_fmt(
                    row["critic_selected_ratio_mean"],
                    row["critic_selected_ratio_ci95"],
                ),
                utility=_fmt(
                    row["utility_selected_ratio_mean"],
                    row["utility_selected_ratio_ci95"],
                ),
                coverage=_fmt(
                    row["epsilon_coverage_mean"],
                    row["epsilon_coverage_ci95"],
                ),
                probability=_fmt(
                    row["p_epsilon_mean"],
                    row["p_epsilon_ci95"],
                ),
                feasible=_fmt(
                    row["raw_feasible_rate_mean"],
                    row["raw_feasible_rate_ci95"],
                ),
                latency=f"{row['proposal_latency_ms_mean']:.2f}",
            )
        )
    lines.extend(
        [
            "",
            "## Stable-backlog future-batch rollouts",
            "",
            "Beam handles unreserved jobs immediately. A future planner sees "
            "only jobs guaranteed to survive the six-step deadline plus guard. "
            "The highest-priority bounded target block is reserved until the "
            "result arrives or expires, after which persistent IDs are remapped "
            "and repaired.",
            "",
            "| policy | n | return/ref. | future best-K | identity survival | "
            "utilization | deadline | post-repair | stable jobs | shots |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in future:
        lines.append(
            "| {policy} | {size} | {reward} | {best} | {identity} | "
            "{use} | {deadline} | {safe} | {stable} | {shots:.0f} |".format(
                policy=row["policy"],
                size=row["n_jobs"],
                reward=_fmt(row["reward_ratio_mean"], row["reward_ratio_ci95"]),
                best=_fmt(
                    row["projected_best_k_ratio_mean"],
                    row["projected_best_k_ratio_ci95"],
                ),
                identity=_fmt(
                    row["selected_identity_survival_rate_mean"],
                    row["selected_identity_survival_rate_ci95"],
                ),
                use=_fmt(
                    row["future_result_utilization_mean"],
                    row["future_result_utilization_ci95"],
                ),
                deadline=_fmt(
                    row["deadline_compliance_mean"],
                    row["deadline_compliance_ci95"],
                ),
                safe=f"{row['post_repair_feasible_rate_mean']:.3f}",
                stable=f"{row['mean_stable_pool_size_mean']:.1f}",
                shots=row["shots_total_mean"],
            )
        )
    lines.extend(
        [
            "",
            "## Latency evidence",
            "",
            f"The trace source is `{latency['source_kind']}` and measured QPU "
            f"evidence is **{latency['measured_qpu']}**. Mean/p95/p99 total "
            f"latency is {latency['total_mean_ms']:.1f}/"
            f"{latency['total_p95_ms']:.1f}/{latency['total_p99_ms']:.1f} ms. "
            f"Compliance with the {config['future_deadline_ms']:.0f} ms future "
            f"deadline is {latency['deadline_compliance']:.1%}.",
            "",
            "## Gates",
            "",
            "| check | pass |",
            "|---|---:|",
        ]
    )
    for name, value in gates["checks"].items():
        lines.append(f"| {name} | {value} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The report shows both nominal means and lower-95%-confidence gates. "
            "A best-of-K mean pass is not a statistically secure scaling pass "
            "when its lower bound misses 0.90. It is also insufficient if the "
            "learned critic cannot find that candidate. Stable-backlog "
            "utilization demonstrates "
            "that delayed results can enter the pipeline, but the reservation "
            "cost must still preserve end-to-end return. Physical advancement "
            "requires the unchanged calibration and measured-latency gates.",
            "",
        ]
    )
    return "\n".join(lines)
