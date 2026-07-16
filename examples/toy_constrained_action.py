"""Run the reusable hybrid action head on a six-node toy graph."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import numpy as np

from hybrid_qrl.classical import (
    IdentityEncoder,
    RandomizedWeightedGreedy,
    StaticUtilityHead,
)
from hybrid_qrl.core import ConflictGraph
from hybrid_qrl.pipeline import HybridActionHead, UtilityCritic
from hybrid_qrl.quantum import (
    ManualNeutralAtomBackendSampler,
    QuTiPRydbergSampler,
    RydbergEmulatorSampler,
)


WEIGHTS = (1.00, 1.25, 0.90, 1.15, 1.05, 1.30)
GRAPH = ConflictGraph(
    nodes=6,
    edges=((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3)),
)
MANUAL_BACKEND_POSITIONS = (
    (0.201, 0.607),
    (-0.845, 0.868),
    (-1.196, -0.191),
    (-0.201, -0.607),
    (0.845, -0.868),
    (1.196, 0.191),
)


def exact_reference() -> tuple[float, str]:
    best_score = float("-inf")
    best_action = ""
    for state in range(1 << GRAPH.nodes):
        action = tuple((state >> node) & 1 for node in range(GRAPH.nodes))
        if GRAPH.is_feasible(action):
            value = sum(weight * bit for weight, bit in zip(WEIGHTS, action))
            if value > best_score:
                best_score = value
                best_action = "".join(map(str, action))
    return best_score, best_action


def run(sampler_name: str, candidates: int, seed: int) -> dict[str, object]:
    if sampler_name == "quantum":
        sampler = RydbergEmulatorSampler()
    elif sampler_name == "qutip":
        sampler = QuTiPRydbergSampler()
    elif sampler_name == "manual":
        sampler = ManualNeutralAtomBackendSampler(
            positions=np.asarray(MANUAL_BACKEND_POSITIONS, dtype=float),
            C6=10.0,
        )
    else:
        sampler = RandomizedWeightedGreedy()
    action_head = HybridActionHead(
        encoder=IdentityEncoder(),
        utility_head=StaticUtilityHead(WEIGHTS),
        sampler=sampler,
        critic=UtilityCritic(),
        candidates=candidates,
    )
    decision = action_head.select(WEIGHTS, GRAPH, seed=seed)
    output = asdict(decision)
    output["action"] = "".join(map(str, decision.action))
    if isinstance(sampler, RydbergEmulatorSampler):
        output["emulator_backend"] = sampler.last_backend
        output["emulator_seconds"] = sampler.last_elapsed_seconds
    if isinstance(sampler, QuTiPRydbergSampler):
        output["emulator_backend"] = f"QuTiP {sampler.qutip_version} sesolve"
        output["solver_stats"] = sampler.last_solver_stats
    if isinstance(sampler, ManualNeutralAtomBackendSampler):
        output["emulator_backend"] = (
            "downloaded neutral_atom.simulator factory -> QuTiPSimulator"
        )
        output["positions"] = sampler.positions.tolist()
        output["C6"] = sampler.C6
        output["geometry_report"] = sampler.geometry_report(GRAPH)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sampler",
        choices=("quantum", "qutip", "manual", "classical", "both"),
        default="both",
    )
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.candidates <= 0:
        parser.error("--candidates must be positive")

    names = ("classical", "quantum") if args.sampler == "both" else (args.sampler,)
    optimum, action = exact_reference()
    report = {
        "candidate_budget": args.candidates,
        "exact_small_system_reference": {
            "score": optimum,
            "action": action,
            "evaluation_only": True,
        },
        "decisions": {name: run(name, args.candidates, args.seed) for name in names},
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
