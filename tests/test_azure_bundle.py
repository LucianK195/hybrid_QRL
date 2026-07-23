"""Regression tests for the Azure bundle-conflict reformulation."""

from __future__ import annotations

from itertools import product
import unittest

import numpy as np

from hybrid_qrl.dispatch.azure_bundle import (
    AzureBundleConfig,
    AzureBundleValueModel,
    bundle_allocation_feasible,
    generate_bundle_library,
    make_bundle_instance,
    solve_bundle_milp,
    solve_direct_assignment_milp,
)
from hybrid_qrl.dispatch.azure_packing import AzureTraceWindow


def _window(jobs: int = 40) -> AzureTraceWindow:
    rng = np.random.default_rng(17)
    resources = rng.uniform(0.025, 0.12, size=(jobs, 4))
    priorities = np.asarray([0 if index % 3 else 1 for index in range(jobs)])
    lifetime = np.linspace(0.25, 12.0, jobs)
    utility = np.where(priorities == 0, 4.0, 1.0) * (
        0.8 + 0.01 * lifetime
    )
    return AzureTraceWindow(
        anchor_day=10.0,
        vm_ids=np.arange(jobs, dtype=np.int64),
        tenant_ids=np.arange(jobs, dtype=np.int64) // 2,
        vm_type_ids=np.arange(jobs, dtype=np.int64) % 5,
        priorities=priorities,
        start_days=np.full(jobs, 10.0),
        end_days=10.0 + lifetime,
        lifetime_days=lifetime,
        resources=resources,
        utility=utility,
    )


def _model(window: AzureTraceWindow) -> AzureBundleValueModel:
    features = AzureBundleValueModel.job_features(window)
    return AzureBundleValueModel(
        mean=np.zeros(features.shape[1]),
        scale=np.ones(features.shape[1]),
        weights=np.asarray((2.5, 0.2, 0.0, 0.0, 0.0, 0.0, 1.0)),
    )


class AzureBundleTests(unittest.TestCase):
    """Check bundle construction, exact pairwise safety, and references."""

    def test_generated_bundles_are_capacity_feasible_and_balanced(self) -> None:
        window = _window()
        nodes = generate_bundle_library(
            window,
            model=_model(window),
            machine_slots=2,
            capacity=0.5,
            target_nodes=20,
            seed=13,
        )

        self.assertEqual(len(nodes), 20)
        self.assertEqual(sum(node.machine == 0 for node in nodes), 10)
        self.assertEqual(sum(node.machine == 1 for node in nodes), 10)
        self.assertTrue(
            all(max(node.usage) <= 0.5 + 1e-10 for node in nodes)
        )

    def test_conflict_graph_is_equivalent_to_authoritative_constraints(self) -> None:
        window = _window()
        nodes = generate_bundle_library(
            window,
            model=_model(window),
            machine_slots=2,
            capacity=0.5,
            target_nodes=6,
            seed=21,
        )
        instance = make_bundle_instance(
            window, nodes, capacity=0.5, seed=22
        )

        for action in product((0, 1), repeat=len(nodes)):
            self.assertEqual(
                instance.state.graph.is_feasible(action),
                bundle_allocation_feasible(instance, action),
            )

    def test_direct_assignment_reference_dominates_bundle_library(self) -> None:
        window = _window(20)
        nodes = generate_bundle_library(
            window,
            model=_model(window),
            machine_slots=2,
            capacity=0.5,
            target_nodes=20,
            seed=31,
        )
        instance = make_bundle_instance(
            window, nodes, capacity=0.5, seed=32
        )
        direct = solve_direct_assignment_milp(
            window,
            machine_slots=2,
            capacity=0.5,
            time_limit_ms=2_000.0,
        )
        bundle = solve_bundle_milp(instance, time_limit_ms=2_000.0)

        self.assertTrue(direct.exact)
        self.assertTrue(bundle.exact)
        self.assertGreaterEqual(
            direct.objective + 1e-8,
            bundle.objective,
        )

    def test_config_requires_balanced_machine_node_counts(self) -> None:
        with self.assertRaises(ValueError):
            AzureBundleConfig(machine_slots=3, bundle_nodes=(20,))


if __name__ == "__main__":
    unittest.main()
