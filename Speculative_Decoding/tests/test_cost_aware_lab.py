import unittest
import numpy as np
from examples.cost_aware_lab import (
    progress_curve, compare_budgets, paired_speed_interval, break_even_tokens,
)


class CostAwareTests(unittest.TestCase):
    def test_survival_tail_sum_against_explicit_round_law(self):
        alpha = np.array([.2, .8, .5])
        survival, expected = progress_curve(alpha)
        for gamma in range(4):
            probabilities = np.r_[survival[:gamma]*(1-alpha[:gamma]), survival[gamma]]
            self.assertAlmostEqual(float(probabilities.sum()), 1.)
            direct = np.arange(1, gamma+2) @ probabilities
            self.assertAlmostEqual(float(direct), expected[gamma])

    def test_shape_cliff_defeats_stopping_at_first_unprofitable_step(self):
        result = compare_budgets([.9, .9], [1., 2., 2.])
        self.assertLess(result['speedup'][1], 1.)
        self.assertEqual(result['best_gamma'], 2)
        self.assertAlmostEqual(result['speedup'][2], 1.355)
        self.assertEqual(compare_budgets([0., 0.], [1., 2., 3.])['best_gamma'], 0)

    def test_ratio_bootstrap_preserves_pairing_and_scaling(self):
        result = paired_speed_interval([1., 10., 100.], [2., 20., 200.], draws=300)
        self.assertEqual(result['aggregate_speedup'], .5)
        np.testing.assert_allclose(result['percentile_95_interval'], [.5, .5])
        scaled = paired_speed_interval([1000., 10000., 100000.], [2000., 20000., 200000.], draws=300)
        self.assertEqual(result, scaled)
        heterogeneous = paired_speed_interval([1., 100.], [1., 50.], draws=300)
        self.assertNotEqual(heterogeneous['mean_request_speedup'], heterogeneous['aggregate_speedup'])

    def test_amortization_uses_additive_cost_and_handles_no_payback(self):
        self.assertAlmostEqual(break_even_tokens(360000., .01, .007), 120000000.)
        self.assertIsNone(break_even_tokens(360000., .01, .012))
        with self.assertRaises(ValueError):
            compare_budgets([.5], [1., 0.])
        with self.assertRaises(ValueError):
            progress_curve([float('nan')])
        with self.assertRaises(ValueError):
            paired_speed_interval([1.], [1.])


if __name__ == '__main__':
    unittest.main()
