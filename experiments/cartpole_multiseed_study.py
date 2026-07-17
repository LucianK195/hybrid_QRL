"""Run and aggregate a multi-seed CartPole backend-by-budget study.

This experiment repeats the existing CartPole integration benchmark across
multiple candidate budgets, random seeds, and quantum simulator backends.  It
stores compact per-trial records, aggregates paired comparisons against the
direct classical policy, and writes both machine-readable JSON and a Markdown
report.

CartPole contains only two environment actions.  The study measures integration
reliability and best-of-K recovery, not combinatorial scaling or quantum
advantage.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.stats import t as student_t

from cartpole_benchmark import run_benchmark


POLICY_KEYS = (
    "random",
    "classical_linear_argmax",
    "classical_greedy_candidates",
    "hybrid_rydberg_candidates",
)


def _compact_trial(
    report: dict[str, object],
    elapsed_seconds: float,
) -> dict[str, object]:
    """Keep the auditable metrics needed for aggregation and paired analysis."""
    config = report["config"]
    dataset = report["dataset"]
    evaluation = report["evaluation"]
    policies = {
        name: {
            "mean_return": evaluation[name]["mean_return"],
            "std_return": evaluation[name]["std_return"],
            "solved_rate": evaluation[name][
                "solved_rate_return_at_least_475"
            ],
            "episode_returns": evaluation[name]["episode_returns"],
        }
        for name in POLICY_KEYS
    }
    hybrid = evaluation["hybrid_rydberg_candidates"]
    greedy = evaluation["classical_greedy_candidates"]
    policies["hybrid_rydberg_candidates"]["sampler_diagnostics"] = hybrid[
        "sampler_diagnostics"
    ]
    policies["hybrid_rydberg_candidates"]["emulator_cache"] = hybrid[
        "emulator_cache"
    ]
    policies["classical_greedy_candidates"]["sampler_diagnostics"] = greedy[
        "sampler_diagnostics"
    ]
    return {
        "backend": config["quantum_backend"],
        "candidate_budget_K": config["candidate_budget_K"],
        "seed": config["seed"],
        "elapsed_seconds": elapsed_seconds,
        "dataset": {
            "samples": dataset["samples"],
            "training_samples": dataset["training_samples"],
            "test_samples": dataset["test_samples"],
            "teacher_mean_return": dataset["teacher_mean_return"],
            "teacher_solved_rate": dataset["teacher_solved_rate"],
            "linear_test_accuracy": dataset["linear_test_accuracy"],
            "quantum_candidate_test_agreement": dataset[
                "quantum_candidate_test_agreement"
            ],
            "quantum_candidate_to_linear_test_agreement": dataset[
                "quantum_candidate_to_linear_test_agreement"
            ],
            "stochastic_agreement_samples": dataset[
                "stochastic_agreement_samples"
            ],
        },
        "evaluation": policies,
    }


def _mean(values: Iterable[float]) -> float:
    """Return a JSON-friendly arithmetic mean."""
    return float(np.mean(np.asarray(list(values), dtype=float)))


def _sample_std(values: Iterable[float]) -> float:
    """Return sample standard deviation, or zero for a single observation."""
    array = np.asarray(list(values), dtype=float)
    return float(array.std(ddof=1)) if len(array) > 1 else 0.0


def _paired_mean_ci95(differences: list[float]) -> tuple[float, float, float]:
    """Return paired mean and two-sided 95% Student-t confidence interval."""
    mean = _mean(differences)
    if len(differences) < 2:
        return mean, mean, mean
    standard_error = _sample_std(differences) / math.sqrt(len(differences))
    margin = float(student_t.ppf(0.975, len(differences) - 1)) * standard_error
    return mean, mean - margin, mean + margin


def aggregate_trials(trials: list[dict[str, object]]) -> list[dict[str, object]]:
    """Aggregate trials by backend and candidate budget."""
    groups: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for trial in trials:
        key = (str(trial["backend"]), int(trial["candidate_budget_K"]))
        groups[key].append(trial)

    rows = []
    backend_order = {"dense": 0, "qutip": 1, "manual": 2}
    ordered_groups = sorted(
        groups.items(),
        key=lambda item: (
            backend_order.get(item[0][0], len(backend_order)),
            item[0][1],
        ),
    )
    for (backend, budget), group in ordered_groups:
        direct_means = [
            item["evaluation"]["classical_linear_argmax"]["mean_return"]
            for item in group
        ]
        random_episodes = [
            value
            for item in group
            for value in item["evaluation"]["random"]["episode_returns"]
        ]
        greedy_episodes = [
            value
            for item in group
            for value in item["evaluation"]["classical_greedy_candidates"][
                "episode_returns"
            ]
        ]
        hybrid_means = [
            item["evaluation"]["hybrid_rydberg_candidates"]["mean_return"]
            for item in group
        ]
        differences = [
            hybrid - direct
            for hybrid, direct in zip(hybrid_means, direct_means)
        ]
        difference, ci_low, ci_high = _paired_mean_ci95(differences)

        direct_episodes = [
            value
            for item in group
            for value in item["evaluation"]["classical_linear_argmax"][
                "episode_returns"
            ]
        ]
        hybrid_episodes = [
            value
            for item in group
            for value in item["evaluation"]["hybrid_rydberg_candidates"][
                "episode_returns"
            ]
        ]
        episode_return_matches = sum(
            hybrid == direct
            for hybrid, direct in zip(hybrid_episodes, direct_episodes)
        )
        diagnostics = [
            item["evaluation"]["hybrid_rydberg_candidates"][
                "sampler_diagnostics"
            ]
            for item in group
        ]
        action_steps = [float(item["action_steps"]) for item in diagnostics]
        raw_candidates = [steps * budget for steps in action_steps]
        total_raw = sum(raw_candidates)
        total_steps = sum(action_steps)
        raw_feasible_rate = sum(
            float(item["raw_feasible_rate"]) * raw
            for item, raw in zip(diagnostics, raw_candidates)
        ) / max(1.0, total_raw)
        fallback_rate = sum(
            float(item["fallback_rate"]) * steps
            for item, steps in zip(diagnostics, action_steps)
        ) / max(1.0, total_steps)
        mean_unique = sum(
            float(item["mean_unique_feasible_candidates"]) * steps
            for item, steps in zip(diagnostics, action_steps)
        ) / max(1.0, total_steps)

        agreement_weights = [
            float(item["dataset"]["stochastic_agreement_samples"])
            for item in group
        ]
        agreement = sum(
            float(item["dataset"]["quantum_candidate_test_agreement"])
            * weight
            for item, weight in zip(group, agreement_weights)
        ) / max(1.0, sum(agreement_weights))
        linear_agreement = sum(
            float(item["dataset"][
                "quantum_candidate_to_linear_test_agreement"
            ])
            * weight
            for item, weight in zip(group, agreement_weights)
        ) / max(1.0, sum(agreement_weights))

        rows.append(
            {
                "backend": backend,
                "candidate_budget_K": budget,
                "seed_count": len(group),
                "evaluation_episodes": len(hybrid_episodes),
                "classical_argmax_mean_return": _mean(direct_episodes),
                "classical_argmax_solved_rate": _mean(
                    value >= 475 for value in direct_episodes
                ),
                "random_mean_return": _mean(random_episodes),
                "random_solved_rate": _mean(
                    value >= 475 for value in random_episodes
                ),
                "classical_greedy_mean_return": _mean(greedy_episodes),
                "classical_greedy_solved_rate": _mean(
                    value >= 475 for value in greedy_episodes
                ),
                "hybrid_mean_return": _mean(hybrid_episodes),
                "hybrid_seed_mean_std": _sample_std(hybrid_means),
                "hybrid_pooled_episode_std": _sample_std(hybrid_episodes),
                "hybrid_solved_rate": _mean(
                    value >= 475 for value in hybrid_episodes
                ),
                "paired_delta_to_argmax": difference,
                "paired_delta_ci95_low": ci_low,
                "paired_delta_ci95_high": ci_high,
                "quantum_candidate_test_agreement": agreement,
                "quantum_candidate_to_linear_test_agreement": linear_agreement,
                "episode_return_match_rate": (
                    episode_return_matches / len(hybrid_episodes)
                ),
                "raw_feasible_rate": raw_feasible_rate,
                "fallback_rate": fallback_rate,
                "mean_unique_feasible_candidates": mean_unique,
                "mean_trial_seconds": _mean(
                    item["elapsed_seconds"] for item in group
                ),
            }
        )
    return rows


def _percent(value: float, decimals: int = 1) -> str:
    """Format a ratio as a percentage with configurable decimal precision."""
    return f"{100.0 * value:.{decimals}f}%"


def render_markdown(
    config: dict[str, object],
    aggregate: list[dict[str, object]],
    total_seconds: float,
) -> str:
    """Render the experiment design, aggregate table, and cautious findings."""
    lines = [
        "# Multi-seed CartPole hybrid sampler study",
        "",
        f"Run date: {date.today().isoformat()}",
        "",
        "## Scope",
        "",
        "This is a two-action integration and reliability study, not evidence of",
        "quantum advantage. It tests whether critic-reranked best-of-K sampling",
        "recovers a learned classical controller across seeds and simulator",
        "implementations.",
        "",
        "## Experimental design",
        "",
        f"- Backends: {', '.join(config['backends'])}.",
        f"- Candidate budgets K: {config['budgets']}.",
        f"- Independent benchmark seeds: {config['seeds']}.",
        f"- Training trajectories per trial: {config['training_episodes']}.",
        f"- Evaluation episodes per seed: {config['evaluation_episodes']}.",
        "- Every backend/K condition uses the same seed-specific datasets and",
        "  evaluation environment seeds as its classical controls.",
        "- A return of at least 475 out of 500 is counted as solved.",
        "- The paired delta is hybrid mean return minus direct classical argmax",
        "  mean return. Its interval is a two-sided 95% Student-t interval over",
        "  seed-level paired differences.",
        "",
        "## Aggregate results",
        "",
        "| Backend | K | Episodes | Classical | Hybrid | Seed SD | Delta [95% CI] "
        "| Solved | Linear match | Return match | Feasible | Fallback |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        delta = (
            f"{row['paired_delta_to_argmax']:+.1f} "
            f"[{row['paired_delta_ci95_low']:+.1f}, "
            f"{row['paired_delta_ci95_high']:+.1f}]"
        )
        lines.append(
            f"| {row['backend']} | {row['candidate_budget_K']} | "
            f"{row['evaluation_episodes']} | "
            f"{row['classical_argmax_mean_return']:.1f} | "
            f"{row['hybrid_mean_return']:.1f} | "
            f"{row['hybrid_seed_mean_std']:.1f} | {delta} | "
            f"{_percent(row['hybrid_solved_rate'])} | "
            f"{_percent(row['quantum_candidate_to_linear_test_agreement'], 2)} | "
            f"{_percent(row['episode_return_match_rate'])} | "
            f"{_percent(row['raw_feasible_rate'])} | "
            f"{_percent(row['fallback_rate'])} |"
        )

    lookup = {
        (row["backend"], row["candidate_budget_K"]): row
        for row in aggregate
    }
    control_backend = config["backends"][0]
    control_rows = [
        lookup[(control_backend, budget)]
        for budget in config["budgets"]
        if (control_backend, budget) in lookup
    ]
    lines.extend(
        [
            "",
            "## Classical controls",
            "",
            "| K | Random mean | Direct argmax mean | Greedy best-of-K mean |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in control_rows:
        lines.append(
            f"| {row['candidate_budget_K']} | {row['random_mean_return']:.1f} | "
            f"{row['classical_argmax_mean_return']:.1f} | "
            f"{row['classical_greedy_mean_return']:.1f} |"
        )

    maximum_budget = max(config["budgets"])
    maximum_rows = [
        lookup[(backend, maximum_budget)]
        for backend in config["backends"]
        if (backend, maximum_budget) in lookup
    ]
    lines.extend(["", "## Main findings", ""])
    if maximum_rows and all(
        row["quantum_candidate_to_linear_test_agreement"] == 1.0
        and row["episode_return_match_rate"] == 1.0
        for row in maximum_rows
    ):
        reference = maximum_rows[0]
        lines.extend(
            [
                f"At K={maximum_budget}, every tested backend reached 100% action",
                "agreement with linear argmax and 100% paired episode-return",
                f"agreement. The hybrid mean ({reference['hybrid_mean_return']:.1f})",
                "and solved rate "
                f"({_percent(reference['hybrid_solved_rate'])}) exactly matched the",
                "direct classical controller across all evaluation episodes.",
                "",
            ]
        )

    if all((backend, 8) in lookup for backend in config["backends"]):
        lines.append("At K=8, recovery was close but backend-dependent:")
        lines.append("")
        for backend in config["backends"]:
            row = lookup[(backend, 8)]
            matched_episodes = round(
                row["episode_return_match_rate"] * row["evaluation_episodes"]
            )
            lines.append(
                f"- {backend}: mean {row['hybrid_mean_return']:.1f}, "
                f"linear-action match "
                f"{_percent(row['quantum_candidate_to_linear_test_agreement'], 2)}, "
                f"and return match {_percent(row['episode_return_match_rate'])} "
                f"({matched_episodes}/{row['evaluation_episodes']} episodes)."
            )
        lines.append("")
        lines.extend(
            [
                "The K=8 action-match rates are near 100%, but rare action errors",
                "can still alter a long sequential-control rollout. This explains",
                "why action agreement can look saturated before episode-return",
                "agreement is exact.",
                "",
            ]
        )

    below_classical = [
        row
        for row in aggregate
        if row["paired_delta_ci95_high"] < 0.0
    ]
    if below_classical:
        cells = ", ".join(
            f"{row['backend']} K={row['candidate_budget_K']}"
            for row in below_classical
        )
        lines.extend(
            [
                "The paired 95% interval remained entirely below classical for",
                f"{cells}. These conditions are resolved as worse within this",
                "four-seed study; conditions whose interval crosses zero remain",
                "statistically unresolved.",
                "",
            ]
        )

    minimum_rows = [
        lookup[(backend, min(config["budgets"]))]
        for backend in config["backends"]
        if (backend, min(config["budgets"])) in lookup
    ]
    if minimum_rows:
        feasibility = ", ".join(
            f"{row['backend']} {_percent(row['raw_feasible_rate'])}"
            for row in minimum_rows
        )
        lines.extend(
            [
                f"At K={min(config['budgets'])}, raw feasibility was still high "
                f"({feasibility}),",
                "yet return was poor. Feasibility therefore does not measure",
                "preferred-action or optimum coverage. From K=4 onward fallback",
                "was zero or negligible, so remaining errors mainly reflect omission",
                "of the critic-preferred action from the sampled batch.",
            ]
        )

    lines.extend(
        [
            "",
            "## Reading the result",
            "",
            "Increasing K gives the critic more chances to observe the preferred",
            "action. Mean return should therefore approach the direct classical",
            "policy when the sampler assigns non-negligible mass to that action.",
            "A high raw feasible rate alone is insufficient: a sampler can produce",
            "valid actions while repeatedly omitting the higher-utility action.",
            "Linear match measures action agreement with direct classical argmax on",
            "held-out states. Return match is stricter at the rollout level: it is",
            "the fraction of paired evaluation episodes with identical total return.",
            "",
            "Differences whose paired confidence interval includes zero are not",
            "resolved by this number of seeds. Positive deviations are also not",
            "evidence that sampling improves the policy; stochastic action errors",
            "can occasionally rescue a trajectory that the deterministic linear",
            "controller would lose.",
            "",
            "## Runtime and limitations",
            "",
            f"The sequential study took {total_seconds:.1f} seconds on this local",
            "machine. Per-trial timing includes dataset construction, fitting,",
            "agreement evaluation, classical controls, and hybrid rollouts. It is",
            "not isolated sampler latency and must not be interpreted as QPU runtime.",
            "",
            "CartPole has only two actions and one conflict edge. It validates the",
            "software path, safety behavior, and best-of-K mechanism, but says",
            "nothing about performance as the number of qubits or combinatorial",
            "decisions grows.",
            "",
            "## Reproduction",
            "",
            "```powershell",
            ".\\.venv\\Scripts\\python.exe "
            ".\\hybrid_qrl\\experiments\\cartpole_multiseed_study.py `",
            "  --backends " + " ".join(config["backends"]) + " `",
            "  --budgets " + " ".join(map(str, config["budgets"])) + " `",
            "  --seeds " + " ".join(map(str, config["seeds"])),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    """Parse the study matrix, execute all trials, and write both reports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=("dense", "qutip", "manual"),
        default=["dense", "qutip", "manual"],
    )
    parser.add_argument(
        "--budgets", type=int, nargs="+", default=[1, 2, 4, 8, 16]
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[17, 29, 43, 71]
    )
    parser.add_argument("--training-episodes", type=int, default=64)
    parser.add_argument("--evaluation-episodes", type=int, default=30)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/cartpole_multiseed_results.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("results/cartpole_multiseed_report.md"),
    )
    args = parser.parse_args()
    if any(value <= 0 for value in args.budgets):
        parser.error("candidate budgets must be positive")
    if args.training_episodes <= 0 or args.evaluation_episodes <= 0:
        parser.error("episode counts must be positive")

    config = {
        "backends": args.backends,
        "budgets": args.budgets,
        "seeds": args.seeds,
        "training_episodes": args.training_episodes,
        "evaluation_episodes": args.evaluation_episodes,
        "cache_decimals": 2,
    }
    combinations = [
        (backend, budget, seed)
        for backend in args.backends
        for budget in args.budgets
        for seed in args.seeds
    ]
    trials = []
    study_start = time.perf_counter()
    for index, (backend, budget, seed) in enumerate(combinations, start=1):
        print(
            f"[{index:02d}/{len(combinations)}] "
            f"backend={backend} K={budget} seed={seed}",
            flush=True,
        )
        trial_start = time.perf_counter()
        report = run_benchmark(
            training_episodes=args.training_episodes,
            evaluation_episodes=args.evaluation_episodes,
            candidates=budget,
            seed=seed,
            dataset_output=None,
            quantum_backend=backend,
        )
        trials.append(_compact_trial(report, time.perf_counter() - trial_start))

    total_seconds = time.perf_counter() - study_start
    aggregate = aggregate_trials(trials)
    output = {
        "experiment": "Multi-seed CartPole backend-by-budget study",
        "interpretation": (
            "Two-action integration and best-of-K reliability study; not a "
            "quantum-scaling or advantage experiment."
        ),
        "config": config,
        "total_trials": len(trials),
        "total_elapsed_seconds": total_seconds,
        "aggregate": aggregate,
        "trials": trials,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    args.report.write_text(
        render_markdown(config, aggregate, total_seconds),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "total_trials": len(trials),
                "total_elapsed_seconds": total_seconds,
                "output": str(args.output),
                "report": str(args.report),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
