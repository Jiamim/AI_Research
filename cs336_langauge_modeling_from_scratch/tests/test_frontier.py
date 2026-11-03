"""Checks for the frontier chapters (29-34): architectures, long context, RL objectives,
agents, synthetic data and block-scaled low precision."""
import importlib.util
import math
from pathlib import Path
import re
import unittest
import torch
from examples.architectures import MODELS, count, kv_bytes_per_token, load
from examples.long_context import (rope_inverse_frequencies, position_interpolation, yarn_inverse_frequencies,
                                   yarn_correction_range, yarn_attention_factor, rotation_wavelengths,
                                   sink_attention, streaming_kv_positions)
from examples.rl_objectives import dapo_loss, cispo_loss, dynamic_sampling_mask, overlong_penalty, clipped_token_fraction
from examples.posttrain import clipped_policy_loss
from examples.agents import calculator, run_episode, byte_tokens_and_mask, reward
from examples.synthetic_data import is_correct
from examples.low_precision import round_to_e2m1, mxfp4_quantize, fp8_tile_quantize, relative_error, int4_absmax

torch.set_num_threads(2)


class ArchitectureTests(unittest.TestCase):
    def test_every_formula_matches_published_safetensors_total(self):
        for repo in MODELS:
            result = count(repo)
            self.assertEqual(result["total"], result["reported"], repo)

    def test_active_parameters_match_model_cards(self):
        # gpt-oss card: 5.13B / 3.61B active; Qwen3 names: A22B / A3B; DeepSeek-V3 37B; Kimi K2 32B.
        expected = {"openai/gpt-oss-120b": 5.13, "openai/gpt-oss-20b": 3.61, "Qwen/Qwen3-235B-A22B": 22,
                    "Qwen/Qwen3-30B-A3B": 3, "deepseek-ai/DeepSeek-V3": 37, "moonshotai/Kimi-K2-Instruct": 32}
        for repo, billions in expected.items():
            self.assertAlmostEqual(count(repo)["active"] / 1e9, billions, delta=0.06 * billions, msg=repo)

    def test_mla_cache_is_576_values_per_layer(self):
        c, _ = load("deepseek-ai/DeepSeek-V3")
        self.assertEqual(kv_bytes_per_token("deepseek-ai/DeepSeek-V3")["bytes"], 61 * 576 * 2)
        # Without compression the same heads would cache K and V for 128 heads of width 192/128.
        mha = 61 * 128 * (192 + 128) * 2
        self.assertGreater(mha / (61 * 576 * 2), 70)

    def test_banded_layers_cap_cache_growth(self):
        short = kv_bytes_per_token("openai/gpt-oss-120b", context=128)["bytes"]
        long = kv_bytes_per_token("openai/gpt-oss-120b", context=131072)["bytes"]
        self.assertAlmostEqual(long / short, 0.5, delta=0.01)


class LongContextTests(unittest.TestCase):
    def test_pi_divides_all_frequencies(self):
        torch.testing.assert_close(position_interpolation(64, 1e4, 8), rope_inverse_frequencies(64, 1e4) / 8)

    def test_yarn_keeps_fast_dims_and_interpolates_slow_dims(self):
        inv, _ = yarn_inverse_frequencies(128, 1e4, 32, 4096)
        ref = rope_inverse_frequencies(128, 1e4)
        low, high = yarn_correction_range(128, 1e4, 4096)
        torch.testing.assert_close(inv[:low], ref[:low])
        torch.testing.assert_close(inv[high:], ref[high:] / 32)
        self.assertTrue(torch.all(inv <= ref) and torch.all(inv >= ref / 32 - 1e-15))

    @unittest.skipUnless(importlib.util.find_spec("transformers"), "transformers not installed")
    def test_yarn_matches_transformers_on_real_configs(self):
        from transformers import DeepseekV3Config, GptOssConfig
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
        for repo, config_class in [("deepseek-ai/DeepSeek-V3", DeepseekV3Config),
                                   ("moonshotai/Kimi-K2-Instruct", DeepseekV3Config), ("openai/gpt-oss-120b", GptOssConfig)]:
            c, _ = load(repo)
            skip = {"auto_map", "architectures", "quantization_config", "model_type"}
            reference, scale = ROPE_INIT_FUNCTIONS["yarn"](config_class(**{k: v for k, v in c.items() if k not in skip}), "cpu")
            rope = c["rope_scaling"]
            # gpt-oss sets "truncate": false, which keeps the ramp boundaries fractional.
            ours, _ = yarn_inverse_frequencies(c.get("qk_rope_head_dim") or c["head_dim"], c["rope_theta"], rope["factor"],
                                               rope["original_max_position_embeddings"], rope.get("beta_fast", 32),
                                               rope.get("beta_slow", 1), truncate=rope.get("truncate", True))
            self.assertLess(((ours - reference).abs() / reference).max().item(), 1e-6, repo)
            self.assertAlmostEqual(yarn_attention_factor(rope["factor"], rope.get("mscale"), rope.get("mscale_all_dim")), scale,
                                   places=6, msg=repo)

    def test_yarn_attention_factor(self):
        self.assertAlmostEqual(yarn_attention_factor(32), 0.1 * math.log(32) + 1)
        self.assertEqual(yarn_attention_factor(40, 1.0, 1.0), 1.0)
        self.assertEqual(yarn_attention_factor(1), 1.0)

    def test_wavelength_of_first_pair_is_two_pi(self):
        self.assertAlmostEqual(rotation_wavelengths(128, 1e4)[0].item(), 2 * math.pi)

    def test_sink_absorbs_attention_mass(self):
        q, k, v = (torch.randn(1, 5, 4) for _ in range(3))
        _, open_mass = sink_attention(q, k, v, torch.tensor([-50.0]))
        out, sink_mass = sink_attention(q, k, v, torch.tensor([50.0]))
        torch.testing.assert_close(open_mass, torch.ones_like(open_mass))
        self.assertLess(sink_mass.max().item(), 1e-15)
        self.assertLess(out.abs().max().item(), 1e-15)

    def test_streaming_cache_keeps_sinks_and_window(self):
        kept = streaming_kv_positions(5000, sinks=4, window=16)
        self.assertEqual(kept[:4], [0, 1, 2, 3])
        self.assertEqual(kept[4:], list(range(4984, 5000)))
        self.assertEqual(streaming_kv_positions(10, 4, 16), list(range(10)))


class RLObjectiveTests(unittest.TestCase):
    def setUp(self):
        self.mask = torch.ones(2, 3)
        self.adv = torch.tensor([1.0, -1.0])

    def test_dapo_matches_ppo_when_clip_is_symmetric_and_lengths_equal(self):
        z = torch.randn(2, 3, requires_grad=True)
        old = z.detach() - 0.1
        ppo = clipped_policy_loss(z, old, self.adv, self.mask, eps=0.2)
        dapo = dapo_loss(z, old, self.adv, self.mask, eps_low=0.2, eps_high=0.2)
        torch.testing.assert_close(ppo, dapo)

    def test_token_level_aggregation_weights_long_sequences_more(self):
        z = torch.zeros(2, 4, requires_grad=True)
        mask = torch.tensor([[1., 0, 0, 0], [1, 1, 1, 1]])
        dapo_loss(z, z.detach(), torch.tensor([1.0, 1.0]), mask).backward()
        # Every valid token gets the same weight 1/5, regardless of its sequence length.
        torch.testing.assert_close(z.grad[mask.bool()], torch.full((5,), -0.2))

    def test_clip_higher_lets_more_positive_tokens_learn(self):
        old = torch.zeros(1, 1)
        z = torch.log(torch.tensor([[1.25]])).requires_grad_()
        dapo_loss(z, old, torch.tensor([1.0]), torch.ones(1, 1), 0.2, 0.28).backward()
        self.assertLess(z.grad.item(), 0)
        z.grad = None
        dapo_loss(z, old, torch.tensor([1.0]), torch.ones(1, 1), 0.2, 0.2).backward()
        self.assertEqual(z.grad.item(), 0)

    def test_cispo_keeps_gradient_on_clipped_tokens(self):
        z = torch.zeros(1, 1, requires_grad=True)
        cispo_loss(z, torch.tensor([[-1.0]]), torch.tensor([1.0]), torch.ones(1, 1), eps_high=0.5).backward()
        self.assertAlmostEqual(z.grad.item(), -1.5, places=5)
        # The same token (ratio e, positive advantage) is clipped by PPO and DAPO: zero gradient.
        for loss in (lambda x: clipped_policy_loss(x, torch.tensor([[-1.0]]), torch.tensor([1.0]), torch.ones(1, 1)),
                     lambda x: dapo_loss(x, torch.tensor([[-1.0]]), torch.tensor([1.0]), torch.ones(1, 1))):
            x = torch.zeros(1, 1, requires_grad=True)
            loss(x).backward()
            self.assertEqual(x.grad.item(), 0.0)

    def test_dynamic_sampling_and_overlong_penalty(self):
        rewards = torch.tensor([[1., 1], [0, 0], [0, 1]])
        self.assertEqual(dynamic_sampling_mask(rewards).tolist(), [False, False, True])
        torch.testing.assert_close(overlong_penalty([10, 15, 20, 25], 20, 10), torch.tensor([0.0, -0.5, -1.0, -1.0]))

    def test_clipped_fraction_counts_only_saturated_side(self):
        old = torch.zeros(1, 2)
        z = torch.log(torch.tensor([[1.5, 0.5]]))
        self.assertEqual(clipped_token_fraction(z, old, torch.tensor([1.0]), torch.ones(1, 2)), 0.5)


class AgentTests(unittest.TestCase):
    def test_calculator_is_restricted(self):
        self.assertEqual(calculator("17 * 23 + 1"), "392")
        for bad in ["__import__('os')", "2**10", "abs(-1)", "1/0"]:
            self.assertTrue(calculator(bad).startswith("error"), bad)

    def test_loss_mask_covers_only_model_text(self):
        segments = run_episode(["<tool>2+3</tool>", "<answer>5</answer>"])
        self.assertEqual([r for r, _ in segments], ["model", "tool", "model"])
        ids, mask = byte_tokens_and_mask("Q\n", segments)
        model_bytes = sum(len(t.encode()) for r, t in segments if r == "model")
        self.assertEqual(sum(mask), model_bytes)
        self.assertEqual(len(ids), len(mask))
        self.assertEqual(reward(segments, 5)["reward"], 1.0)

    def test_episode_stops_at_call_budget(self):
        segments = run_episode(["<tool>1+1</tool>"] * 10, max_calls=3)
        self.assertEqual(sum(r == "tool" for r, _ in segments), 3)


class SyntheticDataTests(unittest.TestCase):
    def test_verifier(self):
        self.assertTrue(is_correct("7+5=2\n"))
        self.assertFalse(is_correct("7+5=3\n"))
        self.assertFalse(is_correct("7+5=2x"))


class LowPrecisionTests(unittest.TestCase):
    def test_e2m1_grid_and_ties(self):
        torch.testing.assert_close(round_to_e2m1(torch.tensor([0.25, 0.75, 2.5, 5.0, 9.0, -1.3])),
                                   torch.tensor([0.0, 1.0, 2.0, 4.0, 6.0, -1.5]))

    def test_e2m1_exact_midpoints_are_not_neighborhoods(self):
        middle = torch.tensor([0.75], dtype=torch.float64)
        below = torch.nextafter(middle, torch.zeros_like(middle))
        above = torch.nextafter(middle, torch.ones_like(middle))
        values = torch.cat([below, middle, above, -below, -middle, -above])
        expected = torch.tensor([0.5, 1., 1., -0.5, -1., -1.], dtype=torch.float64)
        self.assertTrue(torch.equal(round_to_e2m1(values), expected))
        self.assertEqual(round_to_e2m1(values).dtype, torch.float64)

    def test_quantizers_reject_nonfinite_input_instead_of_hiding_it(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            x = torch.full((2, 32), value)
            for quantize in (round_to_e2m1, mxfp4_quantize, int4_absmax,
                             lambda values: fp8_tile_quantize(values, (1, 32))):
                with self.subTest(value=value, function=quantize), self.assertRaises(ValueError):
                    quantize(x)

    def test_mxfp4_scale_stays_in_e8m0_range_at_tiny_magnitudes(self):
        x = torch.tensor([[2.**-126, 2.**-128] + [0.] * 30], dtype=torch.float32)
        quantized, scales = mxfp4_quantize(x)
        self.assertEqual(scales.item(), 2.**-127)
        self.assertTrue(torch.equal(quantized, x))
        # A rounded log2 near 8 can choose the wrong shared exponent.
        edge = torch.nextafter(torch.tensor(8.), torch.tensor(0.)).item()
        _, scales = mxfp4_quantize(torch.full((1, 32), edge))
        self.assertEqual(scales.item(), 1.0)

    def test_absmax_quantizers_preserve_small_nonzero_blocks(self):
        x = torch.full((2, 32), 1e-20)
        fp8, scales = fp8_tile_quantize(x, (1, 32))
        self.assertEqual(scales.shape, (2, 1))
        self.assertLess(relative_error(fp8, x), 1e-6)
        self.assertLess(relative_error(int4_absmax(x), x), 1e-6)
        zero = torch.zeros_like(x)
        out, scales = fp8_tile_quantize(zero, (2, 32))
        self.assertEqual(scales.shape, (1, 1))
        self.assertTrue(torch.equal(scales, torch.ones_like(scales)))
        self.assertTrue(torch.equal(out, zero))

    def test_zero_reference_has_explicit_relative_error_semantics(self):
        self.assertEqual(relative_error(torch.zeros(3), torch.zeros(3)), 0.0)
        self.assertTrue(math.isinf(relative_error(torch.ones(3), torch.zeros(3))))
        x = torch.full((3,), 3e38)
        self.assertEqual(relative_error(x, x), 0.0)

    def test_mxfp4_power_of_two_scales_and_exact_grid_values(self):
        v = torch.tensor([[6.0, 4.0, 3.0, 1.5] + [0.0] * 28])
        q, scale = mxfp4_quantize(v)
        torch.testing.assert_close(q, v)
        self.assertEqual(scale.item(), 1.0)
        q, scale = mxfp4_quantize(v * 2 ** -5)
        torch.testing.assert_close(q, v * 2 ** -5)
        self.assertEqual(scale.item(), 2 ** -5)

    def test_tiles_protect_ordinary_channels_from_an_outlier(self):
        torch.manual_seed(0)
        a = torch.randn(128, 256)
        a[:, 3] *= 1e5
        normal = torch.ones(256, dtype=torch.bool)
        normal[3] = False
        one, _ = fp8_tile_quantize(a, (128, 256))
        tiles, _ = fp8_tile_quantize(a, (1, 128))
        self.assertLess(relative_error(tiles[:, normal], a[:, normal]), relative_error(one[:, normal], a[:, normal]) / 3)


class CrossReferenceTests(unittest.TestCase):
    def test_section_and_chapter_references_resolve(self):
        """Every "第 N.M 节" and "第 N 章" in the prose must point to a heading that exists."""
        chapters = sorted((Path(__file__).resolve().parents[1] / "book/chapters").glob("*.md"))
        numbers = {int(p.name[:2]) for p in chapters}
        headings = set()
        for p in chapters:
            headings |= set(re.findall(r"^##+ (\d+\.\d+)\b", p.read_text(encoding="utf-8"), re.M))
        broken = []
        for p in chapters:
            for line_number, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
                broken += [(p.name, line_number, m) for m in re.findall(r"(\d{1,2}\.\d{1,2})\s*节", line) if m not in headings]
                broken += [(p.name, line_number, m) for m in re.findall(r"第\s*(\d{1,2})\s*章", line) if int(m) not in numbers]
        self.assertEqual(broken, [])


if __name__ == "__main__":
    unittest.main()
