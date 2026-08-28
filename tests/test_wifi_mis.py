from __future__ import annotations

import unittest

from hybrid_qrl.dispatch.wifi_mis import (
    distribution_metrics,
    exact_mwis,
    make_wifi_instance,
)


class WifiMISTests(unittest.TestCase):
    def test_bottleneck_is_two_native_unit_disk_stars(self) -> None:
        instance = make_wifi_instance("bottleneck", seed=17)
        self.assertEqual(instance.graph.nodes, 12)
        self.assertEqual(len(instance.graph.edges), 10)
        self.assertEqual(
            set(instance.graph.edges),
            {
                (0, 1),
                (0, 2),
                (0, 3),
                (0, 4),
                (0, 5),
                (6, 7),
                (6, 8),
                (6, 9),
                (6, 10),
                (6, 11),
            },
        )

    def test_exact_bottleneck_prefers_spatial_reuse(self) -> None:
        instance = make_wifi_instance("bottleneck", seed=19)
        optimum, action, degeneracy = exact_mwis(instance)
        self.assertGreater(optimum, 0.0)
        self.assertEqual(degeneracy, 1)
        self.assertEqual(action[0], 0)
        self.assertEqual(action[6], 0)
        self.assertEqual(sum(action), 10)
        self.assertTrue(instance.graph.is_feasible(action))

    def test_best_of_k_metrics_are_monotone_and_safe(self) -> None:
        instance = make_wifi_instance("bottleneck", seed=23)
        optimum, action, _ = exact_mwis(instance)
        empty = tuple(0 for _ in range(instance.graph.nodes))
        distribution = {empty: 0.75, action: 0.25}
        metrics = distribution_metrics(
            distribution,
            instance,
            optimum,
            budgets=(1, 2, 4, 8),
            epsilon=0.01,
        )
        ratios = [row["expected_best_ratio"] for row in metrics["budgets"]]
        hits = [row["near_optimal_hit_probability"] for row in metrics["budgets"]]
        self.assertEqual(ratios, sorted(ratios))
        self.assertEqual(hits, sorted(hits))
        self.assertAlmostEqual(hits[0], 0.25)
        self.assertAlmostEqual(hits[-1], 1.0 - 0.75**8)

    def test_control_families_remain_unit_disk_graphs(self) -> None:
        for family in ("random", "crowded", "corridor"):
            instance = make_wifi_instance(family, seed=31)
            for left, right in instance.graph.edges:
                distance = float(
                    ((instance.positions[left] - instance.positions[right]) ** 2).sum()
                    ** 0.5
                )
                self.assertLessEqual(distance, instance.interference_radius + 1e-12)

    def test_corridor_has_local_ladder_interference(self) -> None:
        instance = make_wifi_instance("corridor", seed=31)
        self.assertEqual(instance.graph.nodes, 12)
        self.assertGreaterEqual(len(instance.graph.edges), 14)
        self.assertLessEqual(len(instance.graph.edges), 18)
        self.assertTrue(all(abs(left - right) <= 6 for left, right in instance.graph.edges))


if __name__ == "__main__":
    unittest.main()
