"""Distribution, boundary, causal alignment, and KV rollback checks."""
import copy
import itertools
import unittest
from unittest.mock import patch
import numpy as np
from examples.speculative_sampling import (
    MarkovLM, generate, one_step_experiment, probability_vector,
    residual_distribution, speculative_step, verify_candidates, expected_tokens,
)


class SamplingTests(unittest.TestCase):
    def test_probability_mass_identity(self):
        rng = np.random.default_rng(99)
        for _ in range(200):
            p, q = rng.dirichlet(np.ones(9), size=2)
            alpha = np.minimum(p, q).sum()
            output_law = np.minimum(p, q) + (1 - alpha) * residual_distribution(p, q)
            np.testing.assert_allclose(output_law, p, atol=1e-14)

    def test_monte_carlo_and_wrong_resampling_control(self):
        result = one_step_experiment()
        np.testing.assert_allclose(result["exact_empirical"], result["target"], atol=.004)
        np.testing.assert_allclose(result["wrong_empirical"], result["wrong_theory"], atol=.004)
        self.assertGreater(abs(result["wrong_empirical"][0] - result["target"][0]), .1)

    def test_identical_and_disjoint_distributions(self):
        rng = np.random.default_rng(3)
        p = np.array([1., 0., 0.])
        result = verify_candidates([0, 0], [p, p], [p, p, p], rng)
        self.assertEqual(result.tokens, [0, 0, 0])
        self.assertFalse(result.rejected)
        q = np.array([0., 1., 0.])
        result = verify_candidates([1, 1], [q, q], [p, p, p], rng)
        self.assertEqual(result.tokens, [0])
        self.assertTrue(result.rejected)

    def test_context_dependent_joint_distribution(self):
        p = np.array([[.6, .3, .1], [.2, .5, .3], [.3, .1, .6]])
        q = np.array([[.1, .6, .3], [.7, .1, .2], [.2, .7, .1]])
        target, draft = MarkovLM(p), MarkovLM(q)
        rng = np.random.default_rng(2026)
        n = 25_000
        counts = np.zeros((3, 3, 3))
        for _ in range(n):
            prefix = [0]
            while len(prefix) < 4:
                step = speculative_step(target, draft, prefix, min(2, 3 - len(prefix)), rng)
                prefix.extend(step.tokens)
            counts[tuple(prefix[1:])] += 1
        law = np.empty_like(counts)
        for a, b, c in itertools.product(range(3), repeat=3):
            law[a, b, c] = p[0, a] * p[a, b] * p[b, c]
        tv = float(np.abs(counts / n - law).sum() / 2)
        self.assertLess(tv, .025, f"joint-sequence total variation = {tv}")

    def test_eos_length_and_gamma_zero(self):
        model = MarkovLM(np.array([[0., 1.], [0., 1.]]))
        self.assertEqual(generate(model, model, [0], 0)[0], [0])
        self.assertEqual(generate(model, model, [0], 1, gamma=9)[0], [0, 1])
        output, stats = generate(model, model, [0], 10, eos_id=1)
        self.assertEqual(output, [0, 1])
        self.assertEqual(stats["accepted"], 1)
        self.assertEqual(len(generate(model, model, [0], 7, gamma=0)[0]), 8)

    def test_invalid_q_and_zero_residual(self):
        p = np.array([1., 0.])
        with self.assertRaises(ValueError):
            verify_candidates([1], [p], [p, p], np.random.default_rng())
        with self.assertRaises(ValueError):
            residual_distribution(p, p)
        with self.assertRaises(ValueError):
            probability_vector([.1, .1])

    def test_malformed_tokens_and_all_probability_rows_rejected(self):
        p = np.array([1., 0.])
        rng = np.random.default_rng(1)
        for token in [-1, 2, 0.5, True]:
            with self.subTest(token=token), self.assertRaises(ValueError):
                verify_candidates([token], [p], [p, p], rng)
        for rows in [[p, np.array([1., 0., 0.])], [p, np.array([np.nan, 0.])]]:
            with self.assertRaises(ValueError):
                verify_candidates([0], [p], rows, rng)
        # Invalid suffixes must not succeed merely because the first token rejects.
        with self.assertRaises(ValueError):
            verify_candidates([1, -1], [1-p, p], [p, p, p], rng)
        for alpha in [float('nan'), float('inf'), -0.1, 1.1]:
            with self.assertRaises(ValueError):
                expected_tokens(alpha, 4)
        for gamma in [-1, 1.5, True]:
            with self.assertRaises(ValueError):
                expected_tokens(.5, gamma)

    def test_zero_target_probability_never_accepted_at_zero_uniform(self):
        class ZeroRNG:
            def random(self): return 0.0
            def choice(self, size, p): return int(np.argmax(p))
        p, q = np.array([1., 0.]), np.array([0., 1.])
        step = verify_candidates([1], [q], [p, p], ZeroRNG())
        self.assertEqual(step.tokens, [0])
        self.assertTrue(step.rejected)

    def test_subnormal_probability_does_not_round_acceptance_into_rejection(self):
        class FixedRNG:
            def random(self): return .4
            def choice(self, size, p): return int(np.argmax(p))
        tiny = np.nextafter(0., 1.)
        p, q = np.array([tiny, 1.]), np.array([2*tiny, 1.])
        result = verify_candidates([0], [q], [p, p], FixedRNG())
        self.assertEqual(result.accepted, 1)  # u=.4 < p/q=.5
        self.assertFalse(result.rejected)
        class HighUniform(FixedRNG):
            def random(self): return .75
        self.assertEqual(verify_candidates([0], [p], [p, p], HighUniform()).accepted, 1)

    def test_eos_stops_before_later_verification_or_bonus(self):
        class NoMoreRNG:
            def random(self): return 0.1
            def choice(self, *args, **kwargs):
                raise AssertionError('No sampling after accepted EOS')
        p = np.array([0., 1.])
        step = verify_candidates([1, 1], [p, p], [p, p, p], NoMoreRNG(), eos_id=1)
        self.assertEqual((step.tokens, step.accepted, step.rejected), ([1], 1, False))

    def test_rejected_draft_eos_and_target_bonus_eos(self):
        rng = np.random.default_rng(4)
        zero, one = np.array([1., 0.]), np.array([0., 1.])
        rejected = verify_candidates([1], [one], [zero, one], rng, eos_id=1)
        self.assertEqual((rejected.tokens, rejected.accepted, rejected.rejected), ([0], 0, True))
        bonus = verify_candidates([0], [zero], [zero, one], rng, eos_id=1)
        self.assertEqual((bonus.tokens, bonus.accepted, bonus.rejected), ([0, 1], 1, False))
        repair = verify_candidates([0], [zero], [one, zero], rng, eos_id=1)
        self.assertEqual((repair.tokens, repair.accepted, repair.rejected), ([1], 0, True))


class CacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from examples.tiny_transformer import TinyTransformer
        torch.set_num_threads(1)
        torch.manual_seed(10)
        cls.target = TinyTransformer().double().eval()
        torch.manual_seed(20)
        cls.draft = TinyTransformer(layers=1).double().eval()

    def test_parallel_logits_equal_serial_conditionals(self):
        import torch
        from examples.tiny_transformer import token_tensor
        model, prefix, candidates = self.target, [1, 2, 3], [4, 5, 6]
        with torch.inference_mode():
            _, cache = model(token_tensor(prefix[:-1], model))
            batched, _ = model(token_tensor([prefix[-1], *candidates], model), cache)
            for i in range(len(candidates) + 1):
                serial, _ = model(token_tensor(prefix + candidates[:i], model))
                torch.testing.assert_close(batched[0, i], serial[0, -1], rtol=1e-10, atol=1e-10)

    def test_rejected_suffix_removed(self):
        import torch
        from examples.tiny_transformer import token_tensor, crop_cache
        model = self.target
        with torch.inference_mode():
            _, cache = model(token_tensor([1, 2, 3, 4, 5], model))
            cache = crop_cache(cache, 3)  # reject 4, 5; replace with 7
            repaired, _ = model(token_tensor([7], model), cache)
            expected, _ = model(token_tensor([1, 2, 3, 7], model))
            torch.testing.assert_close(repaired[0, -1], expected[0, -1], rtol=1e-10, atol=1e-10)

    def test_greedy_matches_full_target(self):
        from examples.tiny_transformer import autoregressive_generate, speculative_generate
        for prompt in [[1], [1, 2, 3]]:
            expected = autoregressive_generate(self.target, prompt, 19)
            for gamma in [0, 1, 4, 8]:
                actual, trace = speculative_generate(self.target, self.draft, prompt, 19, gamma=gamma)
                self.assertEqual(actual, expected)
                self.assertTrue(all(row["cache_length"] == row["prefix_length"] + row["accepted"] for row in trace))

    def test_all_accept_bonus_and_terminal_eos(self):
        from examples.tiny_transformer import autoregressive_generate, speculative_generate
        draft = copy.deepcopy(self.target)
        actual, trace = speculative_generate(self.target, draft, [1, 2], 12, gamma=4)
        self.assertTrue(all(not row["rejected"] for row in trace))
        self.assertEqual(actual, autoregressive_generate(self.target, [1, 2], 12))
        eos = actual[2]
        stopped, _ = speculative_generate(self.target, draft, [1, 2], 12, eos_id=eos)
        self.assertEqual(stopped, [1, 2, eos])

    def test_sampling_and_zero_length(self):
        from examples.tiny_transformer import speculative_generate
        for count in [0, 1, 7]:
            output, _ = speculative_generate(self.target, self.draft, [1], count, greedy=False)
            self.assertEqual(len(output), count + 1)

    def test_zero_budget_and_disabled_draft_do_no_extra_work(self):
        from examples.tiny_transformer import autoregressive_generate, speculative_generate
        for count in [0, 1, 8]:
            with patch.object(self.draft, 'forward', side_effect=AssertionError('draft called')):
                with patch.object(self.target, 'forward', wraps=self.target.forward) as forward:
                    actual, trace = speculative_generate(self.target, self.draft, [1, 2], count, gamma=0)
                    self.assertEqual(forward.call_count, count)
                expected = autoregressive_generate(self.target, [1, 2], count)
                self.assertEqual(actual, expected)
                self.assertEqual(len(trace), count)
        with patch.object(self.target, 'forward', side_effect=AssertionError('prefill called')):
            self.assertEqual(speculative_generate(self.target, self.draft, [1, 2], 0)[0], [1, 2])
        # One requested token needs no draft even with gamma>0.
        with patch.object(self.draft, 'forward', side_effect=AssertionError('draft called')):
            speculative_generate(self.target, self.draft, [1], 1, gamma=8)

    def test_eos_trace_matches_committed_output(self):
        from examples.tiny_transformer import autoregressive_generate, speculative_generate
        identical = copy.deepcopy(self.target)
        eos = autoregressive_generate(self.target, [1, 2], 1)[-1]
        output, trace = speculative_generate(self.target, identical, [1, 2], 8, eos_id=eos)
        self.assertEqual(output, [1, 2, eos])
        row = trace[-1]
        self.assertEqual(row['accepted'], 1)
        self.assertEqual(row['emitted'], [eos])
        self.assertEqual(row['cache_length'], len(output) - 1)
        self.assertEqual(row['draft_cache_length'], len(output) - 1)
        self.assertEqual(row['stop_reason'], 'eos')

    def test_sampling_cached_probabilities_match_full_prefix_after_each_repair(self):
        import torch
        from examples import tiny_transformer as tiny
        original_verify = tiny.verify_candidates
        for temperature in [.3, 1.7]:
            prefix = [1, 2]
            seen = {'rejections': 0, 'rounds': 0}
            def checked(candidates, qs, ps, rng, **kwargs):
                with torch.inference_mode():
                    for i in range(len(candidates) + 1):
                        ids = prefix + list(candidates[:i])
                        logits, _ = self.target(tiny.token_tensor(ids, self.target))
                        np.testing.assert_allclose(ps[i], tiny.probabilities(logits[0, -1], temperature), atol=1e-12)
                        if i < len(candidates):
                            logits, _ = self.draft(tiny.token_tensor(ids, self.draft))
                            np.testing.assert_allclose(qs[i], tiny.probabilities(logits[0, -1], temperature), atol=1e-12)
                step = original_verify(candidates, qs, ps, rng, **kwargs)
                prefix.extend(step.tokens)
                seen['rounds'] += 1
                seen['rejections'] += int(step.rejected)
                return step
            with patch.object(tiny, 'verify_candidates', side_effect=checked):
                output, _ = tiny.speculative_generate(self.target, self.draft, [1, 2], 35,
                                                      gamma=4, greedy=False, temperature=temperature)
            self.assertEqual(output, prefix)
            self.assertGreater(seen['rounds'], 2)
            self.assertGreater(seen['rejections'], 0)

    def test_capacity_and_sampling_parameters(self):
        from examples.tiny_transformer import TinyTransformer, autoregressive_generate, speculative_generate
        model = TinyTransformer(vocab=3, width=8, heads=2, layers=1, max_positions=4).double()
        self.assertEqual(len(autoregressive_generate(model, [0, 1, 2, 0], 1)), 5)
        for count in [-1, 1.5, True]:
            with self.assertRaises(ValueError): autoregressive_generate(model, [0], count)
        with self.assertRaises(ValueError): autoregressive_generate(model, [0, 1, 2, 0], 2)
        for prompt in [[], [-1], [3], [1.5]]:
            with self.assertRaises(ValueError): autoregressive_generate(model, prompt, 1)
        for temperature in [0., -1., float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                autoregressive_generate(model, [0], 1, greedy=False, temperature=temperature)
        expected = autoregressive_generate(model, [0], 4)
        self.assertEqual(speculative_generate(model, model, [0], 4, temperature=0)[0], expected)

    def test_target_only_sampling_and_eos_use_same_path(self):
        from examples.tiny_transformer import autoregressive_generate, speculative_generate
        for seed in [1, 5, 42]:
            expected = autoregressive_generate(self.target, [1], 15, greedy=False, seed=seed, eos_id=3)
            output, _ = speculative_generate(self.target, None, [1], 15, gamma=0,
                                              greedy=False, seed=seed, eos_id=3)
            self.assertEqual(output, expected)


if __name__ == "__main__":
    unittest.main()
