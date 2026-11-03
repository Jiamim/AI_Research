"""Regression checks for substantive mathematical and numerical review findings."""
import math
import unittest
import torch
from examples.optim import clip_grad_norm
from examples.tokenizer import Tokenizer
from examples.model import sample_next
from examples.numerical_tools import corrected_kl, entropy_logit_differential, chapter_review_demo


class ChapterReviewTests(unittest.TestCase):
    def test_nope_depth_changes_permutation_symmetry(self):
        result = chapter_review_demo()
        self.assertLess(result['nope_last_logit_difference']['1'], 1e-12)
        self.assertGreater(result['nope_last_logit_difference']['2'], 1e-3)
        self.assertGreater(result['ppo']['ratio_after'], result['ppo']['clip_boundary'])
        self.assertEqual(result['hybrid_state_ratios']['8192'], .5)
        self.assertEqual(result['hybrid_state_ratios']['131072'], .265625)

    def test_entropy_differential_matches_autograd_and_finite_change(self):
        z = torch.tensor([.3, -.5, .8], dtype=torch.float64, requires_grad=True)
        delta = torch.tensor([.1, -.4, .2], dtype=torch.float64)
        def entropy(x):
            return -(x.softmax(-1) * x.log_softmax(-1)).sum()
        derivative, = torch.autograd.grad(entropy(z), z)
        analytic = entropy_logit_differential(z, delta)
        torch.testing.assert_close(analytic, (derivative * delta).sum(), rtol=1e-12, atol=1e-12)
        eps = 1e-5
        finite = (entropy(z + eps * delta) - entropy(z - eps * delta)) / (2 * eps)
        torch.testing.assert_close(analytic, finite, rtol=1e-8, atol=1e-10)

    def test_annotation_cost_units(self):
        cost = chapter_review_demo()['annotation_cost']
        self.assertAlmostEqual(cost['flops'] / 1e24, 2.1)
        self.assertGreater(cost['gpu_hours'], 1.4e6)
        self.assertLess(cost['gpu_hours'], 1.5e6)
        self.assertLess(cost['ratio_to_6000_hours'], 250)

    def test_control_marker_can_be_encoded_as_untrusted_content(self):
        tokenizer = Tokenizer({i: bytes([i]) for i in range(256)}, [], ['<|end|>'])
        text = 'x<|end|>y'
        self.assertIn(256, tokenizer.encode(text))
        ordinary = tokenizer.encode(text, allow_special=False)
        self.assertNotIn(256, ordinary)
        self.assertEqual(tokenizer.decode(ordinary), text)
        self.assertEqual(list(tokenizer.encode_iterable(list(text), allow_special=False)), ordinary)

    def test_custom_regex_lookahead_does_not_prematurely_settle(self):
        vocab = {i: bytes([i]) for i in range(256)} | {256: b'ab'}
        tokenizer = Tokenizer(vocab, [(b'a', b'b')], pattern=r'a(?=.{5}z)|a.|.')
        # The first 'a' changes its match extent after a distant 'z' arrives.
        text = 'abcdefz'
        for i in range(len(text)):
            self.assertEqual(list(tokenizer.encode_iterable([text[:i], text[i:]])), tokenizer.encode(text))

    def test_clipping_preserves_fp64_and_handles_large_finite_values(self):
        for dtype, magnitude in ((torch.float64, 1e200), (torch.float32, 1e20)):
            p = torch.nn.Parameter(torch.zeros(2, dtype=dtype))
            p.grad = torch.tensor([3., 4.], dtype=dtype) * magnitude
            norm = clip_grad_norm([p], 1.)
            self.assertEqual(norm.dtype, dtype)
            self.assertTrue(torch.isfinite(norm))
            torch.testing.assert_close(norm / magnitude, norm.new_tensor(5.))
            torch.testing.assert_close(p.grad, p.new_tensor([.6, .8]))

    def test_clipping_nonfinite_rejects_before_mutation(self):
        good, bad = [torch.nn.Parameter(torch.zeros(2)) for _ in range(2)]
        good.grad, bad.grad = torch.tensor([3., 4.]), torch.tensor([math.inf, 1.])
        with self.assertRaises(ValueError):
            clip_grad_norm([good, bad], 1.)
        torch.testing.assert_close(good.grad, torch.tensor([3., 4.]))
        with self.assertRaises(ValueError):
            clip_grad_norm([], -1.)

    def test_exact_kl_is_finite_when_softmax_underflows(self):
        z = torch.tensor([0., -1000., 1000.], dtype=torch.float64, requires_grad=True)
        q = torch.tensor([.2, .3, .5], dtype=torch.float64)
        actual = corrected_kl(z, q, torch.full_like(q, 1/3))
        self.assertTrue(torch.isfinite(actual))
        self.assertAlmostEqual(actual.item(), math.log(2), places=12)
        self.assertTrue(torch.isfinite(torch.autograd.grad(actual, z)[0]).all())

    def test_sampling_mask_and_tiny_temperature(self):
        for dtype in (torch.float32, torch.float64):
            logits = torch.tensor([[1000., 999., -torch.inf]], dtype=dtype)
            self.assertEqual(sample_next(logits, temperature=1e-300).item(), 0)
        for invalid in (torch.tensor([[torch.nan, 0.]]), torch.tensor([[torch.inf, 0.]]), torch.full((1, 2), -torch.inf)):
            with self.assertRaises(ValueError):
                sample_next(invalid, temperature=0)


if __name__ == '__main__':
    unittest.main()
