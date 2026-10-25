"""Independent oracles for cache correctness, numeric stability and metric units."""
import unittest

import numpy as np
import torch

from examples.systems_lab import cached_causal_mask, cache_transaction_demo, fcfs_queue, streaming_metrics
from examples.tiny_transformer import probabilities


class SystemsTests(unittest.TestCase):
    def test_cached_mask_has_history_offset(self):
        np.testing.assert_array_equal(cached_causal_mask(2, 3), [
            [1, 1, 1, 0, 0], [1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])
        np.testing.assert_array_equal(cached_causal_mask(0, 3), np.tril(np.ones((3, 3))))

    def test_full_prefix_oracle_detects_stale_cache(self):
        torch.set_num_threads(1)
        result = cache_transaction_demo()
        self.assertLess(max(result['parallel_logits_errors']), 1e-12)
        self.assertLess(result['next_logits_errors']['correct_crop'], 1e-12)
        self.assertGreater(result['next_logits_errors']['no_crop'], 1e-3)
        self.assertGreater(result['next_logits_errors']['retain_one_rejected'], 1e-3)
        self.assertGreater(result['retained_storage_bytes'], result['logical_cache_bytes'])

    def test_tiny_positive_temperature_remains_a_probability_law(self):
        for logits, expected in [([2., 1., -1.], [1., 0., 0.]),
                                  ([2., 2., -1.], [.5, .5, 0.]),
                                  ([-1., -2., -3.], [1., 0., 0.])]:
            result = probabilities(torch.tensor(logits), 1e-320)
            np.testing.assert_allclose(result, expected)
        batched = probabilities(torch.tensor([[2., 1.], [-2., -1.]]), 1e-320)
        np.testing.assert_array_equal(batched, [[1., 0.], [0., 1.]])
        np.testing.assert_array_equal(probabilities(torch.tensor([0., -torch.inf]), 1.), [1., 0.])
        np.testing.assert_allclose(
            probabilities(torch.tensor([1e308, -1e308], dtype=torch.float64), 1e308),
            [1/(1+np.exp(-2)), 1/(1+np.exp(2))],
        )

    def test_nonfinite_logits_fail_before_sampling(self):
        for logits in [[float('nan'), 1.], [float('inf'), 1.], [-float('inf'), -float('inf')]]:
            with self.subTest(logits=logits), self.assertRaises(ValueError):
                probabilities(torch.tensor(logits), 1.)

    def test_same_tpot_can_hide_very_different_streaming_gaps(self):
        regular = streaming_metrics([(t, 1) for t in np.linspace(.1, .38, 8)])
        chunks = streaming_metrics([(.1, 4), (.38, 4)])
        self.assertAlmostEqual(regular['tpot_seconds'], chunks['tpot_seconds'])
        self.assertAlmostEqual(regular['max_token_gap_seconds'], .04)
        self.assertAlmostEqual(chunks['max_token_gap_seconds'], .28)
        self.assertIsNone(streaming_metrics([(.1, 1)])['tpot_seconds'])

    def test_queue_recursion_matches_hand_computation(self):
        result = fcfs_queue([0., 1., 2.], [2., .5, 1.])
        self.assertEqual(result['starts'], [0., 2., 2.5])
        self.assertEqual(result['ends'], [2., 2.5, 3.5])
        self.assertEqual(result['waits'], [0., 1., .5])
        self.assertEqual(result['response_seconds'], [2., 1.5, 1.5])

    def test_equal_mean_service_does_not_determine_wait(self):
        arrivals = np.arange(40)*.1
        constant = fcfs_queue(arrivals, [.08]*40)
        bursty = fcfs_queue(arrivals, [.02, .02, .02, .26]*10)
        self.assertAlmostEqual(constant['mean_service'], bursty['mean_service'])
        self.assertAlmostEqual(constant['mean_wait'], 0.)
        self.assertGreater(bursty['mean_wait'], .04)
        self.assertGreater(bursty['response_p95'], constant['response_p95'])


if __name__ == '__main__':
    unittest.main()
