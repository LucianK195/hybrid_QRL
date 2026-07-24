"""Synthetic dispatch graph dataset experiment commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..dispatch.backlog_benchmark import (
    BacklogBenchmarkConfig,
    run_backlog_benchmark,
)
from ..dispatch.benchmark import BenchmarkConfig, run_benchmark
from ..dispatch.conditional_benchmark import (
    ConditionalAdvantageConfig,
    run_conditional_study,
)
from ..dispatch.dataset import export_test_dataset, model_from_dict
from ..dispatch.generalization_benchmark import (
    GeneralizationBenchmarkConfig,
    run_generalization_benchmark,
)
from ..dispatch.latency_benchmark import (
    LatencyAwareConfig,
    LatencyTrace,
    make_preregistered_stress_trace,
    run_latency_aware_benchmark,
    write_latency_trace,
)
from ..dispatch.learning import TrainingConfig, train_actor_critic
from ..utilities import ExperimentApplication, ExperimentCommand, ResultWriter
from ..utilities.dispatch_plotting import write_dispatch_figures
from ..utilities.paths import workspace_root


ROOT = workspace_root()


class ScalingCommand(ExperimentCommand):
    name = "scaling"
    help = "Run scaling, rollout, and robustness evaluation."

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--seeds", type=int, default=20)
        parser.add_argument("--train-episodes", type=int, default=320)
        parser.add_argument("--k", type=int, default=16)
        parser.add_argument("--latency-ms", type=float, default=20.0)
        parser.add_argument(
            "--json",
            type=Path,
            default=ROOT / "results" / "dispatch_benchmark_results.json",
        )
        parser.add_argument(
            "--report",
            type=Path,
            default=ROOT / "results" / "dispatch_benchmark_report.md",
        )

    def execute(self, args: argparse.Namespace) -> None:
        config = BenchmarkConfig(
            seeds=args.seeds,
            train_episodes=args.train_episodes,
            candidate_budget=args.k,
            latency_budget_ms=args.latency_ms,
        )
        results = run_benchmark(config, args.json, args.report)
        print(
            f"wrote {len(results['scaling_records'])} scaling, "
            f"{len(results['rollout_records'])} rollout, and "
            f"{len(results['robustness_records'])} robustness records"
        )
        print(args.json)
        print(args.report)


class ConditionalCommand(ExperimentCommand):
    name = "conditional"
    help = "Run sampler training, calibration, and phase-map search."

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--seeds", type=int, default=20)
        parser.add_argument("--training-iterations", type=int, default=140)
        parser.add_argument("--calibration-seeds", type=int, default=20)
        parser.add_argument("--phase-seeds", type=int, default=40)
        parser.add_argument("--k", type=int, default=16)
        parser.add_argument(
            "--json",
            type=Path,
            default=ROOT / "results" / "conditional_advantage_results.json",
        )
        parser.add_argument(
            "--report",
            type=Path,
            default=ROOT / "results" / "conditional_advantage_report.md",
        )

    def execute(self, args: argparse.Namespace) -> None:
        config = ConditionalAdvantageConfig(
            seeds=args.seeds,
            sampler_training_iterations=args.training_iterations,
            candidate_budget=args.k,
            calibration_seeds=args.calibration_seeds,
            phase_seeds=args.phase_seeds,
        )
        results = run_conditional_study(config, args.json, args.report)
        print(
            f"wrote {len(results['training_comparison_records'])} training, "
            f"{len(results['calibration_records'])} calibration, "
            f"{len(results['pipeline_records'])} pipeline, and "
            f"{len(results['phase_records'])} phase records"
        )
        print(args.json)
        print(args.report)


class LatencyCommand(ExperimentCommand):
    name = "latency"
    help = "Run delayed and asynchronous dispatch using a latency trace."

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--conditional-results",
            type=Path,
            default=ROOT / "results" / "conditional_advantage_results.json",
        )
        parser.add_argument(
            "--latency-trace",
            type=Path,
            help=(
                "Timestamped measured-QPU JSON. If omitted, use the "
                "preregistered synthetic stress trace."
            ),
        )
        parser.add_argument(
            "--stress-trace-output",
            type=Path,
            default=ROOT / "results" / "dispatch_latency_stress_trace.json",
        )
        parser.add_argument(
            "--json",
            type=Path,
            default=ROOT / "results" / "latency_aware_dispatch_results.json",
        )
        parser.add_argument(
            "--report",
            type=Path,
            default=ROOT / "results" / "latency_aware_dispatch_report.md",
        )
        parser.add_argument("--seeds", type=int, default=20)
        parser.add_argument("--jobs", type=int, default=40)
        parser.add_argument("--horizon", type=int, default=18)
        parser.add_argument("--step-ms", type=float, default=1_000.0)
        parser.add_argument("--deadline-ms", type=float, default=3_000.0)
        parser.add_argument("--k", type=int, default=16)

    def execute(self, args: argparse.Namespace) -> None:
        conditional = json.loads(
            args.conditional_results.read_text(encoding="utf-8")
        )
        model = model_from_dict(conditional["sampler_in_loop_model"])
        if args.latency_trace is None:
            trace = make_preregistered_stress_trace()
            write_latency_trace(trace, args.stress_trace_output)
            trace_path = args.stress_trace_output
        else:
            trace = LatencyTrace.from_json(args.latency_trace)
            trace_path = args.latency_trace
        config = LatencyAwareConfig(
            seeds=args.seeds,
            n_jobs=args.jobs,
            horizon=args.horizon,
            candidate_budget=args.k,
            decision_step_ms=args.step_ms,
            quantum_deadline_ms=args.deadline_ms,
        )
        results = run_latency_aware_benchmark(
            model=model,
            conditional_gate=conditional["gate"],
            trace=trace,
            config=config,
            output_json=args.json,
            output_report=args.report,
        )
        print(f"latency trace: {trace_path}")
        print(f"wrote {len(results['policy_records'])} policy episodes")
        print(args.json)
        print(args.report)
        print(f"combined gate pass: {results['gate']['pass']}")


class BacklogCommand(ExperimentCommand):
    name = "backlog"
    help = "Run fixed-K scaling confirmation and stable-backlog rollouts."

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--baseline-results",
            type=Path,
            default=ROOT / "results" / "dispatch_benchmark_results.json",
        )
        parser.add_argument(
            "--conditional-results",
            type=Path,
            default=ROOT / "results" / "conditional_advantage_results.json",
        )
        parser.add_argument(
            "--latency-trace",
            type=Path,
            help="Measured timestamp JSON; omit for the synthetic stress trace.",
        )
        parser.add_argument(
            "--json",
            type=Path,
            default=ROOT / "results" / "stable_backlog_scaling_results.json",
        )
        parser.add_argument(
            "--report",
            type=Path,
            default=ROOT / "results" / "stable_backlog_scaling_report.md",
        )
        parser.add_argument("--training-episodes", type=int, default=800)
        parser.add_argument("--confirmation-seeds", type=int, default=20)
        parser.add_argument("--dynamic-seeds", type=int, default=12)
        parser.add_argument("--horizon", type=int, default=18)
        parser.add_argument("--k", type=int, default=16)

    def execute(self, args: argparse.Namespace) -> None:
        baseline_payload = json.loads(
            args.baseline_results.read_text(encoding="utf-8")
        )
        conditional_payload = json.loads(
            args.conditional_results.read_text(encoding="utf-8")
        )
        baseline_model = model_from_dict(baseline_payload["model"])
        config = BacklogBenchmarkConfig(
            training_episodes=args.training_episodes,
            confirmation_seeds=args.confirmation_seeds,
            dynamic_seeds=args.dynamic_seeds,
            horizon=args.horizon,
            candidate_budget=args.k,
        )
        model, history = train_actor_critic(
            TrainingConfig(
                episodes=config.training_episodes,
                horizon=18,
                train_sizes=(20, 40, 60, 80, 100),
                densities=(0.08, 0.12, 0.18),
                graph_families=("unit_disk", "grid", "clustered"),
                utility_correlations=("none", "spatial", "degree"),
                seed=config.training_seed,
            )
        )
        trace = (
            LatencyTrace.from_json(args.latency_trace)
            if args.latency_trace is not None
            else make_preregistered_stress_trace()
        )
        results = run_backlog_benchmark(
            model=model,
            baseline_model=baseline_model,
            conditional_gate=conditional_payload["gate"],
            trace=trace,
            config=config,
            output_json=args.json,
            output_report=args.report,
        )
        results["training_history_tail"] = {
            key: values[-20:] for key, values in history.items()
        }
        ResultWriter().json(args.json, results)
        print(f"selected regime: {results['selected_regime']['name']}")
        print(f"algorithmic pass: {results['gates']['algorithmic_pass']}")
        print(f"asynchronous pass: {results['gates']['asynchronous_pass']}")
        print(f"physical pass: {results['gates']['physical_pass']}")
        print(args.json)
        print(args.report)


class GeneralizationCommand(ExperimentCommand):
    name = "generalization"
    help = "Run the frozen-model shift and candidate-budget stress test."

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--stable-results",
            type=Path,
            default=ROOT / "results" / "stable_backlog_scaling_results.json",
        )
        parser.add_argument(
            "--baseline-results",
            type=Path,
            default=ROOT / "results" / "dispatch_benchmark_results.json",
        )
        parser.add_argument("--seeds", type=int, default=20)
        parser.add_argument(
            "--json",
            type=Path,
            default=ROOT / "results" / "dispatch_generalization_stress.json",
        )
        parser.add_argument(
            "--report",
            type=Path,
            default=ROOT / "results" / "dispatch_generalization_stress.md",
        )

    def execute(self, args: argparse.Namespace) -> None:
        results = run_generalization_benchmark(
            stable_results_path=args.stable_results,
            baseline_results_path=args.baseline_results,
            output_json=args.json,
            output_report=args.report,
            config=GeneralizationBenchmarkConfig(seeds=args.seeds),
        )
        print(
            "candidate generation pass: "
            f"{results['gates']['candidate_generation_pass']}"
        )
        print(
            "deployable critic pass: "
            f"{results['gates']['deployable_critic_pass']}"
        )
        print(
            f"MILP exact: {results['oracle_summary']['exact']}/"
            f"{results['oracle_summary']['states']}"
        )
        print(args.json)
        print(args.report)


class ExportCommand(ExperimentCommand):
    name = "export"
    help = "Freeze held-out graph states as a reusable JSON Lines dataset."

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--results",
            type=Path,
            default=ROOT / "results" / "dispatch_benchmark_results.json",
        )
        parser.add_argument(
            "--output",
            type=Path,
            default=ROOT / "datasets" / "dispatch_test_v1.jsonl",
        )
        parser.add_argument(
            "--manifest",
            type=Path,
            default=ROOT / "datasets" / "dispatch_test_v1_manifest.json",
        )

    def execute(self, args: argparse.Namespace) -> None:
        summary = export_test_dataset(args.results, args.output, args.manifest)
        print(f"wrote {summary.records} held-out graph instances")
        print(summary.output_path)
        print(summary.manifest_path)
        print(f"sha256 {summary.sha256}")


class PlotCommand(ExperimentCommand):
    name = "plot"
    help = "Generate the five editable dispatch benchmark SVG figures."

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--results",
            type=Path,
            default=ROOT / "results" / "dispatch_benchmark_results.json",
        )
        parser.add_argument(
            "--dataset",
            type=Path,
            default=ROOT / "datasets" / "dispatch_test_v1.jsonl",
        )
        parser.add_argument(
            "--output-dir",
            type=Path,
            default=ROOT / "figures" / "dispatch_benchmark",
        )

    def execute(self, args: argparse.Namespace) -> None:
        for path in write_dispatch_figures(
            args.results,
            args.dataset,
            args.output_dir,
        ):
            print(path)


DispatchApplication = ExperimentApplication(
    description="Run experiments on the synthetic dispatch graph dataset.",
    commands=(
        ScalingCommand(),
        ConditionalCommand(),
        LatencyCommand(),
        BacklogCommand(),
        GeneralizationCommand(),
        ExportCommand(),
        PlotCommand(),
    ),
    default_command="scaling",
)
