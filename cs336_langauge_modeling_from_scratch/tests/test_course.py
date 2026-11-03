"""Checks tied to the official assignment conventions and the new course demos."""
import contextlib
import io
import math
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from torch.nn import functional as F
from examples.tokenizer import Tokenizer, train_bpe, count_pretokens, count_pretokens_file, train_bpe_from_counts
from examples.model import Config, TinyLM, sample_next, generate
from examples.optim import cosine_lr, wsd_lr, clip_grad_norm
from examples.attention import softmax_backward
from examples.accounting import llama2, parameters, forward_flops, activation_bytes, ridge_point, swiglu_width, Shape
from examples.data import rule_quality_check, redact_contacts, drop_repeated_lines, near_duplicate_clusters
from examples.scaling import isoflop_minimum, experiment
from examples.inference import speculative_step, PagedKVCache, paged_cache_demo
from examples.train import run

torch.set_num_threads(2)
STYLIZED = "low low low low low\nlower lower widest widest widest\nnewest newest newest newest newest newest"


class AssignmentTokenizerTests(unittest.TestCase):
    def test_stylized_bpe_example_and_tie_break(self):
        vocab, merges = train_bpe(STYLIZED, 256 + 1 + 12, ["<|endoftext|>"], pattern=r"\S+")
        self.assertEqual([f"{a.decode()} {b.decode()}" for a, b in merges],
                         ['s t', 'e st', 'o w', 'l ow', 'w est', 'n e', 'ne west', 'w i', 'wi d', 'wid est', 'low e', 'lowe r'])
        self.assertEqual(vocab[256], b"<|endoftext|>")
        vocab6, merges6 = train_bpe(STYLIZED, 256 + 1 + 6, ["<|endoftext|>"], pattern=r"\S+")
        t = Tokenizer(vocab6, merges6, ["<|endoftext|>"], pattern=r"\S+")
        self.assertEqual([t.vocab[i] for i in t.encode("newest")], [b"ne", b"west"])

    def test_special_tokens_are_boundaries_not_counts(self):
        counts = count_pretokens("ab<|endoftext|>ab", ["<|endoftext|>"])
        self.assertEqual(counts, {tuple(b"ab"): 2})
        vocab, merges = train_bpe("a<|endoftext|>b" * 50, 300, ["<|endoftext|>"])
        self.assertNotIn((b">", b"b"), merges)
        self.assertFalse(any(b"<|" in a + b for a, b in merges))

    def test_overlapping_special_tokens_prefer_longest(self):
        t = Tokenizer({i: bytes([i]) for i in range(256)}, [], ["<|a|>", "<|a|><|a|>"])
        ids = t.encode("x<|a|><|a|>y")
        self.assertEqual(len(ids), 3)
        self.assertEqual(t.decode(ids), "x<|a|><|a|>y")

    def test_invalid_utf8_decodes_with_replacement(self):
        t = Tokenizer({i: bytes([i]) for i in range(256)}, [])
        self.assertEqual(t.decode([0xE4, 0xBD]), "�")

    def test_streaming_matches_whole_text(self):
        text = (STYLIZED + "<|endoftext|>  spaced   words\n\n") * 5
        vocab, merges = train_bpe(text, 290, ["<|endoftext|>"])
        t = Tokenizer(vocab, merges, ["<|endoftext|>"])
        for width in (1, 3, 7, 13):
            chunks = [text[i:i + width] for i in range(0, len(text), width)]
            self.assertEqual(list(t.encode_iterable(chunks)), t.encode(text))

    def test_streaming_contractions_and_every_two_way_split(self):
        vocab, merges = train_bpe("we're you'll they've it's " * 30, 300)
        t = Tokenizer(vocab, merges)
        for text in ["we're", "you'll go", "they've  x", "it's\n\n"]:
            full = t.encode(text)
            for i in range(len(text) + 1):
                for j in range(i, len(text) + 1):
                    self.assertEqual(list(t.encode_iterable([text[:i], text[i:j], text[j:]])), full, (text, i, j))

    def test_parallel_file_counts_match_serial(self):
        text = (STYLIZED + "<|endoftext|>") * 40
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "corpus.txt"
            path.write_text(text, encoding="utf-8")
            self.assertEqual(count_pretokens_file(path, workers=3), count_pretokens(text, ["<|endoftext|>"]))
            self.assertEqual(train_bpe_from_counts(count_pretokens_file(path, workers=1), 280)[1],
                             train_bpe(text, 280, ["<|endoftext|>"])[1])


class ModelAblationTests(unittest.TestCase):
    def test_every_ablation_is_causal_and_cache_consistent(self):
        torch.manual_seed(0)
        ids = torch.randint(0, 64, (2, 9))
        for overrides in [dict(norm="post"), dict(norm="none"), dict(position="none"),
                          dict(ffn="silu", hidden=64), dict(tie_embeddings=True)]:
            settings = dict(vocab=64, width=16, heads=4, kv_heads=2, hidden=48) | overrides
            model = TinyLM(Config(**settings)).eval()
            with torch.no_grad():
                full = model(ids)
                changed = ids.clone()
                changed[:, 5:] = (changed[:, 5:] + 1) % 64
                torch.testing.assert_close(model(changed)[:, :5], full[:, :5], rtol=0, atol=0)
                first, cache = model(ids[:, :4], None, True)
                rest, _ = model(ids[:, 4:], cache, True)
                torch.testing.assert_close(torch.cat((first, rest), 1), full, rtol=1e-5, atol=1e-6)

    def test_nope_last_position_ignores_order_of_earlier_tokens(self):
        # One layer: the last query attends over the same set of keys/values whatever their order,
        # so without position encoding swapping two earlier tokens cannot change its output; RoPE can.
        torch.manual_seed(0)
        a, b = torch.tensor([[3, 5, 9]]), torch.tensor([[5, 3, 9]])
        for position, changes in (("none", False), ("rope", True)):
            model = TinyLM(Config(vocab=32, width=16, layers=1, heads=2, kv_heads=2, hidden=32, position=position)).eval()
            with torch.no_grad():
                difference = (model(a)[0, -1] - model(b)[0, -1]).abs().max().item()
            self.assertEqual(difference > 1e-4, changes, position)

    def test_tied_embeddings_share_storage_and_count_once(self):
        model = TinyLM(Config(vocab=50, width=16, heads=2, kv_heads=2, hidden=32, tie_embeddings=True))
        self.assertIs(model.embedding.weight, model.head.weight)
        untied = TinyLM(Config(vocab=50, width=16, heads=2, kv_heads=2, hidden=32))
        self.assertEqual(sum(p.numel() for p in untied.parameters()) - sum(p.numel() for p in model.parameters()), 50 * 16)

    def test_initialization_follows_assignment_scales(self):
        torch.manual_seed(0)
        model = TinyLM(Config(vocab=1000, width=128, heads=4, kv_heads=4, hidden=344))
        q = model.blocks[0].attn.q.weight
        sigma = math.sqrt(2 / 256)
        # A normal truncated at +-3 sigma has standard deviation 0.9866 sigma.
        self.assertAlmostEqual(q.std().item(), sigma * 0.9866, delta=0.003)
        self.assertLessEqual(q.abs().max().item(), 3 * sigma + 1e-6)
        self.assertLessEqual(model.embedding.weight.abs().max().item(), 3.0)

    def test_top_p_keeps_smallest_prefix_and_greedy(self):
        logits = torch.log(torch.tensor([[0.5, 0.3, 0.15, 0.05]]))
        g = torch.Generator().manual_seed(0)
        draws = {int(sample_next(logits, 1.0, 0.75, g)) for _ in range(300)}
        self.assertEqual(draws, {0, 1})
        self.assertEqual(int(sample_next(logits, 0.0)), 0)
        with self.assertRaises(ValueError):
            sample_next(logits, 1.0, 0.0)

    def test_generation_stops_at_eos(self):
        model = TinyLM(Config(vocab=8, width=8, heads=2, kv_heads=2, hidden=16)).eval()
        with torch.no_grad():
            model.head.weight.zero_()
            model.head.weight[3] = 0  # all logits equal; greedy picks token 0
        out = generate(model, torch.tensor([[1]]), 5, temperature=0, eos_id=0)
        self.assertEqual(out.tolist(), [[1, 0]])


class ScheduleTests(unittest.TestCase):
    def test_cosine_schedule_endpoints(self):
        self.assertEqual(cosine_lr(0, 1.0, 0.1, 10, 100), 0.0)
        self.assertEqual(cosine_lr(10, 1.0, 0.1, 10, 100), 1.0)
        self.assertAlmostEqual(cosine_lr(55, 1.0, 0.1, 10, 100), 0.55)
        self.assertAlmostEqual(cosine_lr(100, 1.0, 0.1, 10, 100), 0.1)
        self.assertEqual(cosine_lr(500, 1.0, 0.1, 10, 100), 0.1)

    def test_wsd_holds_then_decays(self):
        self.assertEqual(wsd_lr(50, 1.0, 0.0, 10, 80, 100), 1.0)
        self.assertAlmostEqual(wsd_lr(90, 1.0, 0.0, 10, 80, 100), 0.5)

    def test_global_clip_matches_torch(self):
        a, b = torch.nn.Linear(4, 3), torch.nn.Linear(4, 3)
        b.load_state_dict(a.state_dict())
        x = torch.randn(5, 4) * 100
        a(x).square().sum().backward()
        b(x).square().sum().backward()
        ours = clip_grad_norm(a.parameters(), 1.0)
        theirs = torch.nn.utils.clip_grad_norm_(b.parameters(), 1.0)
        torch.testing.assert_close(ours, theirs)
        for p, q in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-4, atol=1e-6)


class AttentionTests(unittest.TestCase):
    def test_softmax_backward_matches_autograd(self):
        torch.manual_seed(3)
        s = torch.randn(4, 37, dtype=torch.float64, requires_grad=True)
        upstream = torch.randn(4, 37, dtype=torch.float64)
        p = s.softmax(-1)
        (reference,) = torch.autograd.grad((p * upstream).sum(), s)
        torch.testing.assert_close(softmax_backward(p.detach(), upstream), reference)

    def test_triton_softmax_in_interpreter(self):
        if os.environ.get("TRITON_INTERPRET") != "1":
            self.skipTest("set TRITON_INTERPRET=1 to run the Triton kernel on CPU")
        from examples.attention import triton_softmax
        torch.manual_seed(0)
        # Widths that are and are not multiples of the block, and logits large enough to need the max shift.
        for width in (1, 16, 37):
            x = torch.randn(5, width) * 50
            torch.testing.assert_close(triton_softmax(x), x.softmax(-1), rtol=1e-5, atol=1e-6)


class AccountingTests(unittest.TestCase):
    def test_llama2_7b_matches_published_parameter_count(self):
        self.assertEqual(parameters(llama2("7b"))["total"], 6_738_415_616)
        self.assertEqual(swiglu_width(512), 1344)
        self.assertEqual(round(ridge_point()), 295)
        # Korthikanti et al. eq. 1: selective recomputation keeps 34sbh per layer, full keeps 2sbh.
        seven = llama2("7b")
        self.assertEqual(activation_bytes(seven, 1, "selective"), 32 * 34 * 4096 * 4096)
        self.assertEqual(activation_bytes(seven, 1, "full"), 32 * 2 * 4096 * 4096)

    def test_parameter_formula_matches_model(self):
        model = TinyLM(Config(vocab=300, width=32, layers=3, heads=4, kv_heads=2, hidden=88))
        shape = Shape(vocab=300, context=16, layers=3, d_model=32, heads=4, d_ff=88, kv_heads=2)
        self.assertEqual(parameters(shape)["total"], sum(p.numel() for p in model.parameters()))

    def test_attention_share_grows_with_context(self):
        short = forward_flops(llama2("7b"))
        long = forward_flops(llama2("7b", context=32768))
        share = lambda f: (f["attention_scores"] + f["attention_values"]) / f["total"]
        self.assertLess(share(short), 0.15)
        self.assertGreater(share(long), 0.55)


class DataFilterTests(unittest.TestCase):
    def test_quality_rules(self):
        good = "The children walked to the market and bought fresh bread for their grandmother. " * 5
        self.assertEqual(rule_quality_check(good), [])
        self.assertIn("word_count", rule_quality_check("too short"))
        self.assertIn("alphabetic_words", rule_quality_check("12 34 56 78 ## " * 20))
        self.assertIn("ellipsis_lines", rule_quality_check("\n".join(["more words follow here ..."] * 20)))
        self.assertIn("mean_word_length", rule_quality_check("a b c d e " * 20))

    def test_contact_redaction_validates_ip_boundaries(self):
        text, counts = redact_contacts("mail x@y.org ip 10.0.0.1 not 999.1.1.1 phone 650-555-0100")
        self.assertEqual(counts, dict(email=1, phone=1, ip=1))
        self.assertIn("999.1.1.1", text)
        # Part of a longer dotted number: not an address. Followed by a sentence-final period: an address.
        self.assertEqual(redact_contacts("version 1.2.3.4.5")[1]["ip"], 0)
        self.assertEqual(redact_contacts("My IP is 10.0.0.1.")[0], "My IP is <IP>.")

    def test_repeated_lines_are_dropped_from_every_document(self):
        documents = ["menu\nunique one\nfooter", "menu\nunique two\nfooter"]
        self.assertEqual(drop_repeated_lines(documents), ["unique one", "unique two"])

    def test_minhash_clusters_near_duplicates_only(self):
        base = " ".join(f"word{i}" for i in range(60))
        docs = [base, base + " extra tail", " ".join(f"other{i}" for i in range(60))]
        self.assertEqual(near_duplicate_clusters(docs, threshold=0.8)["keep"], [0, 2])


class ScalingDesignTests(unittest.TestCase):
    def test_parabola_vertex_recovers_minimum(self):
        sizes = np.geomspace(10, 1000, 5)
        losses = (np.log(sizes) - np.log(100)) ** 2 + 2
        n, edge = isoflop_minimum(sizes, losses, "parabola")
        self.assertAlmostEqual(n, 100, places=6)
        self.assertFalse(edge)
        self.assertTrue(isoflop_minimum(sizes, -np.log(sizes), "lowest")[1])

    def test_budgeted_design_uses_far_fewer_queries(self):
        result = experiment(seed=0)
        self.assertLess(result["designs"]["budgeted_parabola"]["queries"],
                        result["designs"]["dense_parabola"]["queries"] / 5)


class InferenceTests(unittest.TestCase):
    def test_speculative_rejection_resamples_from_residual(self):
        target = torch.tensor([[0.0, 1.0], [0.5, 0.5]])
        draft = torch.tensor([[1.0, 0.0]])
        # Draft proposes token 0, which the target never produces: always rejected, residual is token 1.
        self.assertEqual(speculative_step(target, draft, torch.tensor([0])), [1])
        same = torch.tensor([[0.3, 0.7], [0.5, 0.5]])
        self.assertEqual(len(speculative_step(same, same[:1], torch.tensor([1]))), 2)
        # Target (0.5, 0.5), draft always proposes token 0: accepted with probability 0.5, otherwise the
        # residual max(p - q, 0) puts all mass on token 1. Resampling from the target instead gives 0.75.
        g = torch.Generator().manual_seed(0)
        half = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        first = [speculative_step(half, draft, torch.tensor([0]), g)[0] for _ in range(4000)]
        self.assertAlmostEqual(sum(t == 0 for t in first) / len(first), 0.5, delta=0.03)

    def test_paged_cache_prefix_sharing_and_release(self):
        result = paged_cache_demo()
        self.assertLess(result["blocks_with_prefix_sharing"], result["blocks_without_sharing"])
        cache = PagedKVCache(num_blocks=2, block_size=4)
        cache.add_request("a", list(range(8)))
        with self.assertRaises(MemoryError):
            cache.append("a")
        cache.release("a")
        self.assertEqual(cache.used_blocks(), 0)


class PolicyGradientTests(unittest.TestCase):
    def test_baseline_variance_by_enumeration(self):
        from examples.posttrain import pg_estimator_variance, optimal_baseline
        # Three-action softmax, score along the third logit: d log pi(a) / d theta_3 = 1[a=3] - pi_3.
        pi, r = [0.05, 0.05, 0.9], [0.0, 0.5, 1.0]
        score = [float(a == 2) - pi[2] for a in range(3)]
        mean_reward = sum(p * x for p, x in zip(pi, r))
        b_star = optimal_baseline(pi, r, score)
        # A Monte Carlo estimate agrees with the enumerated variance.
        torch.manual_seed(0)
        actions = torch.multinomial(torch.tensor(pi), 200_000, replacement=True)
        z = (torch.tensor(r)[actions] - mean_reward) * torch.tensor(score)[actions]
        self.assertAlmostEqual(z.var().item(), pg_estimator_variance(pi, r, score, mean_reward), places=3)
        # The mean-reward baseline increases variance here; b* minimizes it.
        self.assertGreater(pg_estimator_variance(pi, r, score, mean_reward), pg_estimator_variance(pi, r, score, 0.0))
        for b in (0.0, mean_reward, b_star - 0.05, b_star + 0.05):
            self.assertLessEqual(pg_estimator_variance(pi, r, score, b_star), pg_estimator_variance(pi, r, score, b))

    def test_grpo_loop_runs_and_logs_finite_metrics(self):
        from examples.posttrain import grpo_arithmetic_demo
        result = grpo_arithmetic_demo(steps=20, prompts_per_step=8, pretrain_steps=60, verbose=False)
        log = result["log"]
        self.assertEqual(len(log), 3)
        for record in log[1:]:
            self.assertTrue(math.isfinite(record["entropy"]) and math.isfinite(record["grad_norm"]))
            self.assertTrue(0 <= record["mixed_groups"] <= 1)

    def test_elo_win_probability(self):
        from examples.evaluation import elo_win_probability
        self.assertAlmostEqual(elo_win_probability(1000, 1000), 0.5)
        self.assertAlmostEqual(elo_win_probability(1400, 1000), 10 / 11)


class SystemsDemoTests(unittest.TestCase):
    def test_low_precision_accumulation(self):
        from examples.systems import accumulation_demo
        result = accumulation_demo()["results"]
        self.assertAlmostEqual(result["fp32 += fp32"], 10, places=2)
        # BF16 stalls once 1e-3 is below half the spacing (2^-8 at 0.5); Kahan recovers to within one BF16 ulp.
        self.assertEqual(result["bf16 += bf16"], 0.5)
        self.assertLess(abs(result["bf16 += bf16, Kahan"] - 10), 0.0625 + 1e-6)


class TrainingLoopTests(unittest.TestCase):
    def test_token_file_training_and_divergence_detection(self):
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            pattern = np.tile(np.arange(20, dtype=np.uint16), 400)
            np.save(root / "train.npy", pattern)
            np.save(root / "valid.npy", np.tile(np.arange(20, dtype=np.uint16), 50))
            config = dict(vocab=32, width=32, heads=4, kv_heads=2, hidden=64)
            result = run(steps=60, out=root / "run", train=root / "train.npy", valid=root / "valid.npy",
                         config=config, batch_size=8, context=16, lr=3e-3, eval_every=30, verbose=False)
            self.assertLess(result["history"][-1]["val_loss"], 0.5)
            blown = run(steps=30, out=root / "blown", train=root / "train.npy", valid=root / "valid.npy",
                        config=config, batch_size=8, context=16, lr=5.0, max_grad_norm=1e9,
                        divergence_loss=6.0, verbose=False)
            self.assertTrue(blown["diverged"])


if __name__ == "__main__":
    unittest.main()
