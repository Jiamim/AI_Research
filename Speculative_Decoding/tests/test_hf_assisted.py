"""Benchmark comparisons retain every output and timing sample."""
import unittest

import torch
from examples.hf_assisted import benchmark, timing_summary


class BenchmarkTests(unittest.TestCase):
    def test_intermediate_mismatch_is_not_hidden_by_last_good_output(self):
        good, bad = torch.tensor([[1, 2]]), torch.tensor([[1, 3]])
        calls = iter([good, good, good, bad, good, good])
        with self.assertRaisesRegex(AssertionError, 'repeat 0'):
            benchmark(lambda assisted: (next(calls), .1), 2)

    def test_all_timings_and_alternating_order_are_kept(self):
        order = []
        def run(assisted):
            order.append(assisted)
            return torch.tensor([[1, 2]]), .1 if assisted else .2
        _, times = benchmark(run, 3)
        self.assertEqual(order, [False, True, False, True, True, False, False, True])
        self.assertEqual(times, {False: [.2]*3, True: [.1]*3})

    def test_invalid_measurement_cannot_be_reported_as_speedup(self):
        for duration in [0., -1., float('nan'), float('inf')]:
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                benchmark(lambda assisted: (torch.tensor([[1, 2]]), duration), 1)
        for repeats in [0, -1, True, 1.5]:
            with self.subTest(repeats=repeats), self.assertRaises(ValueError):
                benchmark(lambda assisted: (torch.tensor([[1, 2]]), .1), repeats)

    def test_pooled_rates_use_total_output_over_total_time(self):
        summary = timing_summary({False: [1., 1., 10.], True: [1., 2., 3.]}, 12)
        self.assertEqual(summary['e2e_speedup'], .5)  # median 1 / median 2
        self.assertEqual(summary['aggregate_speedup'], 2.)  # total 12 / total 6
        self.assertEqual(summary['pooled_baseline_tokens_per_second'], 3.)
        self.assertEqual(summary['pooled_assisted_tokens_per_second'], 6.)
        self.assertEqual(summary['paired_speedups'], [1., .5, 10/3])
