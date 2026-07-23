"""Regression tests for Azure trace packing and cumulative repair."""

from __future__ import annotations

import unittest

import numpy as np

from hybrid_qrl.dispatch.azure_packing import (
    AzurePackingConfig,
    AzureTraceWindow,
    capacity_feasible,
    make_packing_state,
    repair_capacity,
    solve_packing_milp,
)


def _window(jobs: int = 20) -> AzureTraceWindow:
    resources = np.tile(
        np.asarray((0.18, 0.16, 0.08, 0.05), dtype=float),
        (jobs, 1),
    )
    priorities = np.asarray([0 if index % 3 else 1 for index in range(jobs)])
    lifetime = np.linspace(0.25, 12.0, jobs)
    utility = np.where(priorities == 0, 4.0, 1.0)
    return AzureTraceWindow(
        anchor_day=10.0,
        vm_ids=np.arange(jobs, dtype=np.int64),
        tenant_ids=np.arange(jobs, dtype=np.int64) // 2,
        vm_type_ids=np.arange(jobs, dtype=np.int64) % 4,
        priorities=priorities,
        start_days=np.full(jobs, 10.0),
        end_days=10.0 + lifetime,
        lifetime_days=lifetime,
        resources=resources,
        utility=utility,
    )


class AzurePackingTests(unittest.TestCase):
    """Check cumulative safety, exact references, and config validation."""

    def test_capacity_repair_removes_until_all_resources_fit(self) -> None:
        state, capacities, _ = make_packing_state(_window(), 0.5, seed=7)
        unsafe = tuple(1 for _ in range(state.n_jobs))
        repaired, removed = repair_capacity(
            state,
            unsafe,
            capacities,
            scores=state.values,
        )

        self.assertGreater(removed, 0)
        self.assertTrue(capacity_feasible(state, repaired, capacities))

    def test_milp_dominates_simple_feasible_action(self) -> None:
        state, capacities, _ = make_packing_state(_window(), 1.0, seed=11)
        simple, _ = repair_capacity(
            state,
            tuple(1 for _ in range(state.n_jobs)),
            capacities,
            scores=np.ones(state.n_jobs),
        )
        oracle = solve_packing_milp(state, capacities, time_limit_ms=2_000.0)

        self.assertTrue(oracle.exact)
        self.assertGreaterEqual(
            oracle.objective + 1e-10,
            float(np.asarray(simple) @ state.values),
        )

    def test_config_rejects_overlapping_time_splits(self) -> None:
        with self.assertRaises(ValueError):
            AzurePackingConfig(train_day_end=10.5, test_day_start=10.0)


if __name__ == "__main__":
    unittest.main()
