"""Independent counterexamples for token boundaries, precision and evaluation units."""
import itertools
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from examples.evaluation import pass_all_k, pass_at_k
from examples.mechanisms import stochastic_round_interval
from examples.retrieval import ndcg_at_k
from examples.scaling import experiment, run_design, seed_study
from examples.tokenizer import chunk_boundaries, count_pretokens, count_pretokens_file


class TokenBoundaryTests(unittest.TestCase):
    def test_parallel_counts_match_whole_text_when_specials_overlap(self):
        for text, specials in [('zz axb zz', ['x', 'axb']),
                               ('zz abcx zz', ['bc', 'ab']),
                               ('zz aaaa zz', ['aa'])]:
            with self.subTest(specials=specials), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / 'text.txt'
                path.write_text(text, encoding='utf-8')
                self.assertEqual(count_pretokens_file(path, specials, workers=2),
                                 count_pretokens(text, specials))

    def test_separator_crossing_read_block_is_not_missed(self):
        separator = b'<doc>'
        data = b'a' * 9094 + separator + b'b' * (10000 - 9094 - len(separator))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'text.txt'
            path.write_bytes(data)
            # The first read after the 5000-byte guess ends inside <doc>.
            self.assertEqual(chunk_boundaries(path, 2, separator), [0, 9094, 10000])


class RoundingContractTests(unittest.TestCase):
    def test_fp64_endpoints_do_not_round_through_default_float(self):
        upper = 1.0 + 2.0 ** -40
        x = torch.tensor([1.0, upper], dtype=torch.float64)
        self.assertTrue(torch.equal(stochastic_round_interval(x, 1.0, upper), x))

    def test_full_fp64_interval_handles_overflowing_span(self):
        maximum = torch.finfo(torch.float64).max
        x = torch.tensor([-maximum, 0.0, maximum], dtype=torch.float64)
        draws = torch.tensor([0.75, 0.25, 0.75], dtype=torch.float64)
        with mock.patch('torch.rand', return_value=draws):
            rounded = stochastic_round_interval(x, -maximum, maximum)
        self.assertTrue(torch.equal(rounded, torch.tensor([-maximum, maximum, maximum], dtype=torch.float64)))

    def test_endpoint_representability_is_required_for_unbiased_output(self):
        with self.assertRaisesRegex(ValueError, 'representable'):
            stochastic_round_interval(torch.tensor([0.15], dtype=torch.float16), 0.1, 0.2)
        with self.assertRaisesRegex(ValueError, 'floating-point'):
            stochastic_round_interval(torch.tensor([0]), 0.0, 0.5)


class ScalingAccountingTests(unittest.TestCase):
    def test_adaptive_cost_summary_includes_all_seeds(self):
        seeds = range(8)
        designs = [experiment(seed=seed, noise=0.04)['designs']['budgeted_extended_parabola'] for seed in seeds]
        summary = seed_study(seeds=seeds, noise=0.04)['summary']['budgeted_extended_parabola']
        counts = [design['queries'] for design in designs]
        self.assertGreater(max(counts), min(counts))
        self.assertIsNone(summary['queries'])
        self.assertEqual((summary['queries_min'], summary['queries_max']), (min(counts), max(counts)))
        self.assertAlmostEqual(summary['queries_mean'], sum(counts) / len(counts))
        self.assertEqual(summary['compute_spent_max'], max(design['compute_spent'] for design in designs))

    def test_observed_edge_and_extrapolated_fit_are_separate(self):
        # y=(log(N)+1)^2 has an extrapolated vertex at N=e^-1 below the grid.
        def surface(sizes, *args):
            return (np.log(sizes) + 1) ** 2
        with mock.patch('examples.scaling.loss', side_effect=surface):
            result = run_design([1., 2.], lambda c, _: [1., 2., 4.],
                                np.random.default_rng(0), 0., 'parabola')
        self.assertEqual(result['edge_minima'], [True, True])
        self.assertEqual(result['final_edge_minima'], [True, True])
        self.assertEqual(result['fit_outside_grid'], [True, True])


class RankingAndReliabilityTests(unittest.TestCase):
    def test_ndcg_rejects_nonfinite_grades_and_fractional_cutoff(self):
        for grade in (math.nan, math.inf, -1.0):
            with self.subTest(grade=grade), self.assertRaises(ValueError):
                ndcg_at_k(['a'], {'a': grade, 'b': 1}, 1)
        with self.assertRaises(ValueError):
            ndcg_at_k(['a'], {'a': 1}, 1.5)

    def test_ndcg_handles_large_and_tiny_finite_grades(self):
        self.assertEqual(ndcg_at_k(['a'], {'a': 1024.0}, 1), 1.0)
        self.assertEqual(ndcg_at_k(['a'], {'a': 1e-30}, 1), 1.0)
        swapped = ndcg_at_k(['b', 'a'], {'a': 1024.0, 'b': 1023.0}, 2)
        expected = (0.5 + 1 / math.log2(3)) / (1 + 0.5 / math.log2(3))
        self.assertAlmostEqual(swapped, expected)

    def test_pass_all_k_matches_subset_enumeration(self):
        for outcomes in itertools.product((0, 1), repeat=5):
            for k in (1, 2, 4, 5):
                subsets = list(itertools.combinations(outcomes, k))
                expected = sum(all(subset) for subset in subsets) / len(subsets)
                self.assertAlmostEqual(pass_all_k(5, sum(outcomes), k), expected)

    def test_pass_all_k_is_unbiased_for_iid_trials(self):
        n, p, k = 7, 0.35, 3
        mean = sum(math.comb(n, c) * p ** c * (1 - p) ** (n - c) * pass_all_k(n, c, k)
                   for c in range(n + 1))
        self.assertAlmostEqual(mean, p ** k)
        self.assertEqual(pass_all_k(np.int64(4), np.int64(4), 4), 1.0)
        self.assertEqual(pass_all_k(4, 0, 2), 0.0)
        with self.assertRaises(ValueError):
            pass_all_k(4, 4, 5)

    def test_extreme_counts_do_not_round_a_log_argument_to_zero(self):
        total = 10 ** 20
        self.assertTrue(math.isclose(pass_all_k(total, 1, 1), 1 / total, rel_tol=1e-13))
        self.assertEqual(pass_at_k(total, total - 1, 1), 1.0)
        self.assertTrue(math.isclose(pass_at_k(total, 1, 1), 1 / total, rel_tol=1e-13))
