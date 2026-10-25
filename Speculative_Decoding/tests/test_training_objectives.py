"""Independent invariants for the training-objective teaching constructions."""
import unittest
import numpy as np
from examples.training_objectives_lab import (
    finite_horizon_rounds, forward_kl, overlap, two_token_survival,
    weighted_ce_gradients,
)


class TrainingObjectivesTests(unittest.TestCase):
    def test_kl_ranking_can_reverse_acceptance_ranking(self):
        p = np.array([.6, .25, .15])
        a, b = np.array([.33, .36, .31]), np.array([.52, .46, .02])
        self.assertLess(forward_kl(p, a), forward_kl(p, b))
        self.assertLess(overlap(p, a), overlap(p, b))

    def test_joint_dependence_changes_prefix_survival(self):
        first = np.array([.5, .5])
        target = np.array([[.9, .1], [.1, .9]])
        independent = np.full((2, 2), .5)
        np.testing.assert_allclose(two_token_survival(first, target, first, independent), [1., .6])
        np.testing.assert_allclose(two_token_survival(first, target, first, target), [1., 1.])
        # Both second-token marginal distributions equal the target marginal.
        np.testing.assert_allclose(first @ independent, first @ target)

    def test_detached_weights_match_unsmoothed_proxy_gradient(self):
        self.assertLess(weighted_ce_gradients(0.)['unsmoothed_gradient_max_error'], 1e-14)

    def test_smoothing_changes_the_optimization_vector_field(self):
        results = weighted_ce_gradients(.1)
        self.assertGreater(results['unsmoothed_gradient_max_error'], 1e-3)
        self.assertGreater(results['smoothed_gradient_max_error'], 1e-3)

    def test_terminal_and_bonus_boundaries(self):
        self.assertEqual(finite_horizon_rounds(np.zeros((0, 2)), 0), 0)
        for n in range(1, 12):
            for gamma in range(4):
                self.assertEqual(finite_horizon_rounds(np.zeros((n, gamma)), n), n)
                self.assertEqual(finite_horizon_rounds(np.ones((n, gamma)), n),
                                 (n + gamma) // (gamma + 1))

    def test_better_first_round_can_require_more_total_rounds(self):
        a = np.zeros((4, 2)); a[0] = [1., 0.]; a[1] = [1., 1.]
        b = a.copy(); b[0] = [0., 1.]
        self.assertEqual(finite_horizon_rounds(a, 4), 3)
        self.assertEqual(finite_horizon_rounds(b, 4), 2)


if __name__ == '__main__':
    unittest.main()
