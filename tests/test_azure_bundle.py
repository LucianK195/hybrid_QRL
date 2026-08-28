"""Regression tests for the Azure bundle-conflict reformulation."""

from __future__ import annotations

from itertools import product
import unittest

import numpy as np

from hybrid_qrl.dispatch.azure_bundle import (
    AzureBundleConfig,
    AzureBundleValueModel,
    ExternalPortfolioConfig,
    QuantumWalkRegime,
    bundle_allocation_feasible,
    generate_bundle_library,
    generate_bundle_candidates,
    make_bundle_instance,
    solve_bundle_milp,
    solve_direct_assignment_milp,
    _one_hot_xy_probabilities,
    _grover_qaoa_probabilities,
    _method_seed_offset,
    _layout_actions,
)
from hybrid_qrl.dispatch.backlog_benchmark import SamplerRegime
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
        with self.assertRaises(ValueError):
            AzureBundleConfig(sampler_regime="unknown")
        with self.assertRaises(ValueError):
            AzureBundleConfig(primary_k=3)
        with self.assertRaises(ValueError):
            AzureBundleConfig(
                primary_method="beam_search",
                comparison_method="beam_search",
            )
        with self.assertRaises(ValueError):
            ExternalPortfolioConfig(
                direct_milp_time_limit_ms=2_000.0,
                direct_milp_retry_limit_ms=1_000.0,
            )

    def test_method_seed_stream_is_stable_and_distinct(self) -> None:
        methods = (
            "paired_grover_qaoa",
            "modular_xy_qaoa",
            "modular_rydberg",
            "randomized_greedy",
            "beam_search",
        )
        first = [_method_seed_offset(method) for method in methods]
        second = [_method_seed_offset(method) for method in reversed(methods)]

        self.assertEqual(first, list(reversed(second)))
        self.assertEqual(len(first), len(set(first)))

    def test_modular_rydberg_is_feasible_before_repair(self) -> None:
        window = _window()
        model = _model(window)
        nodes = generate_bundle_library(
            window,
            model=model,
            machine_slots=2,
            capacity=0.5,
            target_nodes=20,
            seed=41,
        )
        instance = make_bundle_instance(
            window, nodes, capacity=0.5, seed=42
        )
        batch = generate_bundle_candidates(
            "modular_rydberg",
            instance=instance,
            model=model,
            regime=SamplerRegime(
                "test", "standardized", 0.5, "extended"
            ),
            candidates=16,
            seed=43,
        )

        self.assertEqual(batch.raw_generated, 16)
        self.assertEqual(batch.raw_feasible, 16)
        self.assertEqual(batch.mean_removed_fraction, 0.0)
        self.assertTrue(batch.actions)
        self.assertTrue(
            all(
                bundle_allocation_feasible(instance, action)
                for action in batch.actions
            )
        )

    def test_xy_qaoa_distribution_is_normalized_and_nonuniform(self) -> None:
        probabilities = _one_hot_xy_probabilities(
            np.asarray((0.2, 0.5, 1.7, 2.2)),
            QuantumWalkRegime(gamma=0.8, beta=1.1, depth=2),
        )

        self.assertAlmostEqual(float(np.sum(probabilities)), 1.0)
        self.assertTrue(np.all(probabilities >= 0.0))
        self.assertGreater(float(np.ptp(probabilities)), 1e-3)

    def test_modular_xy_qaoa_is_feasible_before_repair(self) -> None:
        window = _window()
        model = _model(window)
        nodes = generate_bundle_library(
            window,
            model=model,
            machine_slots=2,
            capacity=0.5,
            target_nodes=20,
            seed=51,
        )
        instance = make_bundle_instance(
            window, nodes, capacity=0.5, seed=52
        )
        batch = generate_bundle_candidates(
            "modular_xy_qaoa",
            instance=instance,
            model=model,
            regime=SamplerRegime(
                "test", "standardized", 0.5, "extended"
            ),
            candidates=16,
            seed=53,
            quantum_walk=QuantumWalkRegime(
                gamma=0.8, beta=1.1, depth=2
            ),
        )

        self.assertEqual(batch.raw_generated, 16)
        self.assertEqual(batch.raw_feasible, 16)
        self.assertEqual(batch.mean_removed_fraction, 0.0)
        self.assertTrue(batch.actions)

    def test_grover_qaoa_distribution_is_normalized(self) -> None:
        probabilities = _grover_qaoa_probabilities(
            np.asarray((0.3, 0.7, 1.1, 2.4, 2.9)),
            QuantumWalkRegime(gamma=0.7, beta=1.2, depth=2),
        )

        self.assertAlmostEqual(float(np.sum(probabilities)), 1.0)
        self.assertTrue(np.all(probabilities >= 0.0))
        self.assertGreater(float(np.ptp(probabilities)), 1e-3)

    def test_paired_grover_qaoa_is_feasible_before_repair(self) -> None:
        window = _window()
        model = _model(window)
        nodes = generate_bundle_library(
            window,
            model=model,
            machine_slots=2,
            capacity=0.5,
            target_nodes=20,
            seed=61,
        )
        instance = make_bundle_instance(
            window, nodes, capacity=0.5, seed=62
        )
        batch = generate_bundle_candidates(
            "paired_grover_qaoa",
            instance=instance,
            model=model,
            regime=SamplerRegime(
                "test", "standardized", 0.5, "extended"
            ),
            candidates=16,
            seed=63,
            quantum_walk=QuantumWalkRegime(
                gamma=0.7, beta=1.2, depth=2
            ),
        )

        self.assertEqual(batch.raw_generated, 16)
        self.assertEqual(batch.raw_feasible, 16)
        self.assertEqual(batch.mean_removed_fraction, 0.0)
        self.assertTrue(batch.actions)

    def test_layout_quantum_and_classical_candidates_share_safe_space(self) -> None:
        window = _window()
        model = _model(window)
        nodes = generate_bundle_library(
            window,
            model=model,
            machine_slots=2,
            capacity=0.5,
            target_nodes=20,
            seed=71,
        )
        instance = make_bundle_instance(
            window, nodes, capacity=0.5, seed=72
        )
        layouts = _layout_actions(instance)

        self.assertTrue(layouts)
        self.assertTrue(
            all(bundle_allocation_feasible(instance, action) for action in layouts)
        )
        for method in (
            "layout_grover_qaoa",
            "randomized_layout",
            "deterministic_layout",
            "quantum_portfolio",
        ):
            batch = generate_bundle_candidates(
                method,
                instance=instance,
                model=model,
                regime=SamplerRegime(
                    "test", "standardized", 0.5, "adiabatic"
                ),
                candidates=16,
                seed=73,
                quantum_walk=QuantumWalkRegime(
                    gamma=0.8, beta=1.2, depth=3
                ),
            )
            self.assertEqual(batch.raw_generated, 16)
            self.assertEqual(batch.raw_feasible, 16)
            self.assertEqual(batch.mean_removed_fraction, 0.0)


if __name__ == "__main__":
    unittest.main()
