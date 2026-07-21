"""Export frozen held-out graph states for dispatch-method evaluation.

The benchmark itself generates states on demand. This module turns those
states into a portable JSON Lines dataset so candidate generators can be
tested on exactly the same graph, node features, and immediate-reward target.
It exports test instances only: actor and critic parameters are not stored in
the dataset and no record should be used for model fitting.

Every record contains the authoritative application conflict graph and its
neutral-atom geometry. ``positions`` plus ``blockade_radius`` reproduce the
unit-disk edges, while ``edges`` remain the constraints that an executed action
must satisfy. The linear objective and reward constant permit independent
verification of an action without recreating the environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .environment import DispatchConfig, DispatchEnvironment, DispatchState
from .learning import (
    ActorCriticModel,
    LinearAutoregressiveActor,
    LinearCritic,
)


DATASET_SCHEMA_VERSION = 1


def _objective_weights(state: DispatchState, miss_penalty: float) -> np.ndarray:
    """Return the linear weights for exact immediate dispatch reward."""

    completion = state.values * (
        1.0 + state.ages / np.maximum(state.deadlines, 1)
    )
    avoided_miss = miss_penalty * state.values * (state.remaining <= 1)
    return completion + avoided_miss


@dataclass(frozen=True)
class DatasetExportSummary:
    """Metadata returned after writing a held-out graph dataset."""

    records: int
    output_path: Path
    manifest_path: Path
    sha256: str


def model_from_dict(payload: dict[str, list[float]]) -> ActorCriticModel:
    """Restore the frozen linear actor-critic stored in benchmark results."""

    required = {"actor", "value_critic", "action_critic"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"model payload is missing {sorted(missing)}")
    return ActorCriticModel(
        actor=LinearAutoregressiveActor(np.asarray(payload["actor"], dtype=float)),
        value_critic=LinearCritic(np.asarray(payload["value_critic"], dtype=float)),
        action_critic=LinearCritic(
            np.asarray(payload["action_critic"], dtype=float)
        ),
    )


def replay_held_out_state(
    model: ActorCriticModel,
    config: DispatchConfig,
    held_out_seed: int,
    warmup_steps: int,
) -> DispatchState:
    """Recreate the policy-induced test state used by the benchmark.

    A dedicated random stream reproduces the four default warm-up actions.
    This isolates evaluation seeds from training and proposal randomness.
    """

    environment = DispatchEnvironment(config, seed=held_out_seed)
    rng = np.random.default_rng(held_out_seed + 7_919)
    state = environment.state()
    for _ in range(warmup_steps):
        action = model.actor.sample(state, rng)
        state, _, done, _ = environment.step(action)
        if done:
            break
    return state


def _unique_scaling_instances(
    records: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Select one metadata row per paired equal-K scaling state."""

    instances: dict[tuple[int, int], dict[str, Any]] = {}
    for row in records:
        if row.get("study") != "scaling" or row.get("mode") != "equal_k":
            continue
        key = (int(row["n_jobs"]), int(row["seed_index"]))
        instances.setdefault(key, row)
    return [instances[key] for key in sorted(instances)]


def build_test_records(results: dict[str, Any]) -> list[dict[str, Any]]:
    """Build deterministic JSON-ready records from benchmark results.

    The returned set has one state for every size/seed pair in the equal-K
    scaling study. With defaults this is 4 sizes x 20 seeds = 80 held-out
    instances. The reference reward comes from the paired time-limited MILP.
    """

    model = model_from_dict(results["model"])
    benchmark_config = results["config"]
    warmup_steps = int(benchmark_config["warmup_steps"])
    output: list[dict[str, Any]] = []

    for metadata in _unique_scaling_instances(results["scaling_records"]):
        n_jobs = int(metadata["n_jobs"])
        environment_config = DispatchConfig(
            n_jobs=n_jobs,
            density=float(metadata["density"]),
            graph_family=str(metadata["graph_family"]),
            utility_distribution=str(metadata["utility_distribution"]),
            horizon=max(warmup_steps + 2, 8),
        )
        held_out_seed = int(metadata["held_out_seed"])
        state = replay_held_out_state(
            model,
            environment_config,
            held_out_seed,
            warmup_steps,
        )
        weights = _objective_weights(state, environment_config.miss_penalty)
        expires_now = state.remaining <= 1
        reward_constant = -float(
            environment_config.miss_penalty * np.sum(state.values[expires_now])
        ) / n_jobs
        possible_edges = n_jobs * (n_jobs - 1) / 2
        reference_reward = float(metadata["reference_reward"])
        output.append(
            {
                "schema_version": DATASET_SCHEMA_VERSION,
                "instance_id": (
                    f"dispatch-{state.graph.nodes:03d}-"
                    f"seed-{int(metadata['seed_index']):02d}"
                ),
                "split": "test",
                "seed_index": int(metadata["seed_index"]),
                "held_out_seed": held_out_seed,
                "n_jobs": n_jobs,
                "graph_family": environment_config.graph_family,
                "target_density": environment_config.density,
                "realized_density": len(state.graph.edges) / possible_edges,
                "blockade_radius": state.blockade_radius,
                "positions": state.positions.tolist(),
                "edges": [list(edge) for edge in state.graph.edges],
                "step_index": state.step_index,
                "values": state.values.tolist(),
                "ages": state.ages.tolist(),
                "deadlines": state.deadlines.tolist(),
                "remaining": state.remaining.tolist(),
                "node_features": state.node_features.tolist(),
                "objective_weights": weights.tolist(),
                "reward_constant": reward_constant,
                "reference": {
                    "reward": reference_reward,
                    "linear_objective": (
                        reference_reward - reward_constant
                    )
                    * n_jobs,
                    "oracle_exact": bool(metadata["oracle_exact"]),
                    "oracle_status": str(metadata["oracle_status"]),
                    "oracle_mip_gap": metadata["oracle_mip_gap"],
                    "time_limit_ms": float(
                        benchmark_config["oracle_time_limit_ms"]
                    ),
                },
            }
        )
    return output


def export_test_dataset(
    results_path: Path,
    output_path: Path,
    manifest_path: Path,
) -> DatasetExportSummary:
    """Write JSONL instances and a compact provenance/field manifest."""

    results_bytes = results_path.read_bytes()
    results = json.loads(results_bytes)
    records = build_test_records(results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record, separators=(",", ":")) for record in records]
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    output_path.write_bytes(payload)
    dataset_digest = sha256(payload).hexdigest()

    counts_by_size: dict[str, int] = {}
    for record in records:
        key = str(record["n_jobs"])
        counts_by_size[key] = counts_by_size.get(key, 0) + 1
    manifest = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "name": "hybrid-qrl-dispatch-heldout-test-v1",
        "description": (
            "Frozen policy-induced graph states from the equal-K scaling "
            "benchmark; reserved for evaluation."
        ),
        "split": "test",
        "records": len(records),
        "counts_by_size": counts_by_size,
        "candidate_budget": results["config"]["candidate_budget"],
        "warmup_steps": results["config"]["warmup_steps"],
        "benchmark_seed": results["config"]["seed"],
        "source_results": str(results_path.resolve()),
        "source_results_sha256": sha256(results_bytes).hexdigest(),
        "dataset_file": output_path.name,
        "dataset_sha256": dataset_digest,
        "format": "JSON Lines; one graph state per line",
        "constraint_semantics": (
            "edges are authoritative; positions and blockade_radius are the "
            "physical unit-disk embedding"
        ),
        "reward_formula": (
            "reward = reward_constant + dot(objective_weights, action) / n_jobs"
        ),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return DatasetExportSummary(
        records=len(records),
        output_path=output_path,
        manifest_path=manifest_path,
        sha256=dataset_digest,
    )
