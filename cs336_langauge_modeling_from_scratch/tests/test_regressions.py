"""Regression tests for defects found in the 2026-11 project review.

Each test fails on the code before its fix, or on a plausible wrong implementation the
earlier tests let through (wrong RoPE frequencies, a missing beta, a missing stop-gradient).
"""
import contextlib
import importlib.util
import io
import math
import os
from pathlib import Path
import random
import tempfile
import unittest
import numpy as np
import torch
from examples.accounting import ridge_point, swiglu_width
from examples.agents import calculator
from examples.architectures import MODELS, count, kv_bytes_per_token
from examples.data import drop_repeated_lines
from examples.evaluation import pass_at_k
from examples.inference import speculative_step
from examples.long_context import yarn_inverse_frequencies
from examples.mechanisms import stochastic_round_interval
from examples.model import Config, TinyLM, generate, rope
from examples.optim import clip_grad_norm, cosine_lr, wsd_lr
from examples.posttrain import advantages, arithmetic_reward, dpo_loss, simpo_loss
from examples.rl_objectives import cispo_loss, clipped_token_fraction, dapo_loss, overlong_penalty
from examples.scaling import experiment
from examples.synthetic_data import is_correct
from examples.systems import tensor_parallel_demo
from examples.tokenizer import Tokenizer, count_pretokens, count_pretokens_file, train_bpe
from examples.train import run

torch.set_num_threads(2)


class TokenizerRegressionTests(unittest.TestCase):
    def test_special_token_with_ordinary_bytes_never_leaks_into_plain_text(self):
        vocab, merges = train_bpe("a<b a<b x", 260, ["<"])
        t = Tokenizer(vocab, merges, ["<"])
        special = t.special_ids["<"]
        self.assertEqual(t.encode("a<b", allow_special=False), [97, 60, 98])
        self.assertEqual(t.encode("a<b"), [97, special, 98])
        self.assertNotIn(special, list(t.encode_iterable(["a<", "b"], allow_special=False)))

    def test_streaming_matches_whole_text_with_overlapping_specials(self):
        cases = [(["<|eot|>", "|>x"], ["a <|eot|>", "y"]),
                 (["<|a|>", "<|a|>b<|a|>"], ["x<|a|>b<|a|>", "y"])]
        for specials, chunks in cases:
            vocab, merges = train_bpe("hello world " + " ".join(specials), 300, specials)
            t = Tokenizer(vocab, merges, specials)
            self.assertEqual(list(t.encode_iterable(chunks)), t.encode("".join(chunks)), specials)

    def test_random_overlapping_specials_stream_and_roundtrip(self):
        rng = random.Random(0)
        for _ in range(300):
            specials = list({"".join(rng.choice("ab<|>x") for _ in range(rng.randint(1, 4)))
                             for _ in range(rng.randint(1, 3))})
            vocab, merges = train_bpe("ab ab<|> x xa " + " ".join(specials), 300, specials)
            t = Tokenizer(vocab, merges, specials)
            text = "".join(rng.choice("ab<|>x e\n'" + "".join(specials)) for _ in range(rng.randint(0, 20)))
            cuts = sorted(rng.sample(range(len(text) + 1), min(3, len(text) + 1)))
            chunks = [text[a:b] for a, b in zip([0] + cuts, cuts + [len(text)])]
            for allow in (True, False):
                ids = t.encode(text, allow_special=allow)
                self.assertEqual(list(t.encode_iterable(chunks, allow_special=allow)), ids, (specials, chunks))
                self.assertEqual(t.decode(ids), text)
                if not allow:
                    self.assertFalse(set(ids) & set(t.special_ids.values()))

    def test_file_counting_without_special_tokens(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "plain.txt"
            path.write_text("one two two\nthree", encoding="utf-8")
            self.assertEqual(count_pretokens_file(path, special_tokens=(), workers=2), count_pretokens("one two two\nthree"))


class TrainingRegressionTests(unittest.TestCase):
    def test_nonfinite_gradients_end_as_divergence_with_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            result = run(steps=30, out=folder, lr=500, max_grad_norm=1e9, divergence_loss=6.0, verbose=False)
            self.assertTrue(result["diverged"])
            self.assertTrue((Path(folder) / "checkpoint.pt").exists())

    def test_token_ids_outside_vocabulary_fail_clearly(self):
        with tempfile.TemporaryDirectory() as folder:
            data = np.arange(400, dtype=np.uint16) % 300
            np.save(Path(folder) / "t.npy", data)
            with self.assertRaisesRegex(ValueError, "vocabulary"):
                run(steps=2, out=folder, train=str(Path(folder) / "t.npy"), valid=str(Path(folder) / "t.npy"), verbose=False)

    def test_clip_grad_norm_accepts_a_single_tensor(self):
        p = torch.nn.Parameter(torch.ones(3, 4))
        p.grad = torch.full((3, 4), 10.0)
        norm = clip_grad_norm(p, 1.0)
        self.assertAlmostEqual(norm.item(), math.sqrt(12) * 10, places=4)
        self.assertAlmostEqual(p.grad.norm().item(), 1.0, places=4)

    def test_degenerate_schedules_end_at_min_lr(self):
        self.assertEqual(cosine_lr(10, 1.0, 0.1, warmup=10, cosine_end=10), 0.1)
        self.assertEqual(wsd_lr(100, 1.0, 0.1, 10, decay_start=100, end=100), 0.1)
        self.assertAlmostEqual(cosine_lr(5, 1.0, 0.1, 0, 10), 0.55)
        self.assertAlmostEqual(wsd_lr(95, 1.0, 0.1, 10, 90, 100), 0.55)

    def test_generate_restores_training_mode(self):
        model = TinyLM(Config()).train()
        generate(model, torch.tensor([[1, 2]]), 2, temperature=0)
        self.assertTrue(model.training)
        model.eval()
        generate(model, torch.tensor([[1, 2]]), 2, temperature=0)
        self.assertFalse(model.training)

    def test_rope_matches_complex_rotation_of_adjacent_pairs(self):
        # Pins the base, the frequency schedule and the adjacent-pair layout, not only norm/relativity.
        torch.manual_seed(0)
        x = torch.randn(1, 1, 3, 8, dtype=torch.float64)
        positions = torch.tensor([0, 5, 17])
        freqs = 10000.0 ** (-torch.arange(0, 8, 2, dtype=torch.float64) / 8)
        z = torch.view_as_complex(x.reshape(1, 1, 3, 4, 2).contiguous())
        expected = torch.view_as_real(z * torch.polar(torch.ones(3, 4, dtype=torch.float64),
                                                      positions[:, None] * freqs)).reshape(1, 1, 3, 8)
        torch.testing.assert_close(rope(x, positions), expected)


class PosttrainRegressionTests(unittest.TestCase):
    def test_dpo_value_and_gradient_scale_with_beta(self):
        chosen = torch.tensor([-1.0], requires_grad=True)
        rejected = torch.tensor([-3.0], requires_grad=True)
        beta = 0.5
        loss = dpo_loss(chosen, rejected, torch.tensor([-2.0]), torch.tensor([-2.0]), beta=beta)
        margin = beta * ((-1 + 2) - (-3 + 2))
        self.assertAlmostEqual(loss.item(), math.log1p(math.exp(-margin)), places=6)
        loss.backward()
        self.assertAlmostEqual(chosen.grad.item(), -beta * torch.sigmoid(torch.tensor(-margin)).item(), places=6)

    def test_simpo_beta_scales_the_normalized_gap(self):
        chosen, rejected = torch.tensor([-2.0]), torch.tensor([-6.0])
        lengths = torch.tensor([2.0])
        for beta in (0.5, 2.0):
            expected = math.log1p(math.exp(-(beta * (-1 + 3) - 1.0)))
            self.assertAlmostEqual(simpo_loss(chosen, rejected, lengths, lengths, beta=beta, margin=1.0).item(),
                                   expected, places=6)

    def test_grpo_uses_population_std_on_mixed_groups_and_accepts_integer_rewards(self):
        rewards = torch.tensor([[0, 1, 1, 1]])
        a = advantages(rewards)
        std = math.sqrt(0.75 * 0.25)
        torch.testing.assert_close(a, torch.tensor([[-0.75, 0.25, 0.25, 0.25]]) / (std + 1e-6))

    def test_rewards_accept_ascii_digits_only(self):
        self.assertEqual(arithmetic_reward(1, 2, [0xC2, 0xB2, 10]), (0.0, 0.0))  # "²\n"
        self.assertEqual(arithmetic_reward(1, 2, list("٣\n".encode())), (0.0, 0.0))
        self.assertEqual(arithmetic_reward(1, 2, list(b"3\n")), (1.0, 1.0))
        self.assertFalse(is_correct("8+4=²\n"))
        self.assertTrue(is_correct("8+4=2\n"))


class RLObjectiveRegressionTests(unittest.TestCase):
    def test_cispo_stop_gradient_matters_away_from_logp_zero(self):
        logp = torch.tensor([[-0.5]], requires_grad=True)
        cispo_loss(logp, torch.tensor([[-0.6]]), torch.tensor([1.0]), torch.ones(1, 1)).backward()
        # With sg(r): d/dlogp of -(r * A * logp) is -r; without it, -(r + r * logp).
        self.assertAlmostEqual(logp.grad.item(), -math.exp(0.1), places=5)

    def test_dapo_clip_branch_is_exercised(self):
        old = torch.zeros(2, 1)
        logp = torch.log(torch.tensor([[1.5], [0.5]])).requires_grad_()
        dapo_loss(logp, old, torch.tensor([1.0, -1.0]), torch.ones(2, 1), eps_low=0.2, eps_high=0.28).backward()
        # Ratio 1.5 > 1.28 with A>0 and ratio 0.5 < 0.8 with A<0 are both clipped: no gradient.
        torch.testing.assert_close(logp.grad, torch.zeros(2, 1))
        self.assertEqual(clipped_token_fraction(logp.detach(), old, torch.tensor([1.0, -1.0]), torch.ones(2, 1),
                                                0.2, 0.28), 1.0)

    def test_rl_helpers_validate_degenerate_arguments(self):
        with self.assertRaises(ValueError):
            overlong_penalty([10], 20, 0)
        with self.assertRaises(ValueError):
            clipped_token_fraction(torch.zeros(2, 3), torch.zeros(2, 3), torch.ones(2, 1), torch.ones(2, 3))


class NumericsRegressionTests(unittest.TestCase):
    def test_stochastic_rounding_is_unbiased_in_low_precision(self):
        generator = torch.Generator().manual_seed(0)
        for dtype in (torch.bfloat16, torch.float16):
            x = torch.full((400_000,), 0.001, dtype=dtype)
            y = stochastic_round_interval(x, 0.0, 1.0, generator)
            self.assertEqual(y.dtype, dtype)
            self.assertAlmostEqual(y.double().mean().item(), x[0].double().item(), delta=3e-4)
        self.assertEqual(stochastic_round_interval(torch.tensor([0.25], dtype=torch.float64), 0., 1.).dtype,
                         torch.float64)

    def test_speculative_step_handles_identical_distributions(self):
        p = torch.tensor([[0.3, 0.7], [0.5, 0.5]])
        q = p[:1] * (1 + 1e-7)
        for seed in range(200):
            tokens = speculative_step(p, q, torch.tensor([0]), torch.Generator().manual_seed(seed))
            self.assertTrue(1 <= len(tokens) <= 2)

    def test_pass_at_k_accepts_numpy_integers(self):
        self.assertAlmostEqual(pass_at_k(np.int64(10), np.int64(3), 2), pass_at_k(10, 3, 2))
        self.assertEqual(pass_at_k(10, 0, 3), 0.0)
        self.assertFalse(math.copysign(1, pass_at_k(10, 0, 3)) < 0)
        with self.assertRaises(ValueError):
            pass_at_k(10.0, 3, 2)

    def test_calculator_is_exact_and_trims_whitespace(self):
        self.assertEqual(calculator("123456789012345678/1"), "123456789012345678")
        self.assertEqual(calculator(" 17 * 23 "), "391")
        self.assertEqual(calculator("7/2"), "3.5")

    def test_accounting_edges(self):
        self.assertEqual(swiglu_width(12), 64)
        with self.assertRaisesRegex(ValueError, "HBM"):
            ridge_point("A100")

    def test_line_dedup_keeps_paragraph_breaks(self):
        self.assertEqual(drop_repeated_lines(["a\n\nb\nnav", "c\n\nd\nnav"]), ["a\n\nb", "c\n\nd"])

    def test_tensor_parallel_demo_leaves_global_rng_alone(self):
        torch.manual_seed(5)
        expected = torch.rand(3)
        torch.manual_seed(5)
        tensor_parallel_demo(2)
        torch.testing.assert_close(torch.rand(3), expected)


class ArchitectureRegressionTests(unittest.TestCase):
    def test_minimax_m1_formula_matches_safetensors(self):
        self.assertIn("MiniMaxAI/MiniMax-M1-80k", MODELS)
        result = count("MiniMaxAI/MiniMax-M1-80k")
        self.assertEqual(result["total"], result["reported"])
        self.assertAlmostEqual(result["active"] / 1e9, 45.9, delta=0.06 * 45.9)

    def test_gpt_oss_without_context_uses_long_context_limit(self):
        self.assertLess(kv_bytes_per_token("openai/gpt-oss-120b")["bytes"],
                        kv_bytes_per_token("openai/gpt-oss-120b", context=128)["bytes"])

    def test_yarn_attention_factor_follows_deepseek_mscale_ratio(self):
        _, factor = yarn_inverse_frequencies(64, 1e4, 40, 4096, mscale=1.0, mscale_all_dim=1.0)
        self.assertEqual(factor, 1.0)
        _, plain = yarn_inverse_frequencies(64, 1e4, 40, 4096)
        self.assertAlmostEqual(plain, 0.1 * math.log(40) + 1)


class ScalingRegressionTests(unittest.TestCase):
    def test_extending_edge_minima_repairs_the_worst_case(self):
        result = experiment(seed=0)["designs"]
        # The first budget's grid is centred at sqrt(C/6) = 12.9, but the true optimum is about 39.
        self.assertTrue(result["budgeted_lowest"]["edge_minima"][0])
        self.assertTrue(result["budgeted_extended_parabola"]["edge_minima"][0])
        extended = result["budgeted_extended_parabola"]
        self.assertLessEqual(extended["queries"], result["budgeted_parabola"]["queries"] + 7)
        self.assertLess(abs(extended["exponent"] - 0.4615), 0.02)


@unittest.skipUnless(importlib.util.find_spec("triton") and os.environ.get("TRITON_INTERPRET") == "1",
                     "set TRITON_INTERPRET=1 to run the Triton kernel on CPU")
class TritonRegressionTests(unittest.TestCase):
    def test_masked_leading_block_is_not_nan(self):
        from examples.attention import triton_softmax
        x = torch.randn(3, 37) * 30
        x[0, :16] = float("-inf")
        x[1, :33] = float("-inf")
        torch.testing.assert_close(triton_softmax(x), torch.softmax(x, -1), rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
