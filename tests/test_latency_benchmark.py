"""Regression tests for latency traces and stale dispatch actions."""

from __future__ import annotations

import unittest

import numpy as np

from hybrid_qrl.dispatch import DispatchConfig, DispatchEnvironment
from hybrid_qrl.dispatch.latency_benchmark import (
    LatencyObservation,
    make_preregistered_stress_trace,
    remap_candidate_job_ids,
    summarize_latency,
)


class LatencyBenchmarkTests(unittest.TestCase):
    """Check timestamp validation, quantiles, and persistent job identity."""

    def test_latency_observation_requires_monotonic_timestamps(self) -> None:
        with self.assertRaises(ValueError):
            LatencyObservation(
                request_id="bad",
                submitted_at_ms=0.0,
                started_at_ms=5.0,
                completed_at_ms=4.0,
                retrieved_at_ms=6.0,
                shots=100,
            )

    def test_stress_trace_is_deterministic_and_not_hardware_evidence(self) -> None:
        first = make_preregistered_stress_trace(count=32, seed=5)
        second = make_preregistered_stress_trace(count=32, seed=5)
        summary = summarize_latency(first, deadline_ms=3_000.0)

        self.assertFalse(first.is_measured_qpu)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertGreater(summary["total_p99_ms"], summary["total_p50_ms"])

    def test_stale_action_drops_replaced_job_identity(self) -> None:
        environment = DispatchEnvironment(
            DispatchConfig(n_jobs=8, density=0.12, horizon=3),
            seed=91,
        )
        original = environment.state()
        selected_job = int(original.job_ids[0])
        action = (1,) + (0,) * 7
        current, _, _, _ = environment.step(action)

        remapped = remap_candidate_job_ids((selected_job,), current)
        self.assertEqual(sum(remapped), 0)
        self.assertFalse(np.any(current.job_ids == selected_job))


if __name__ == "__main__":
    unittest.main()
