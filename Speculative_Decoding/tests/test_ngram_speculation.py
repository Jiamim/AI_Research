import unittest
import numpy as np
from examples.ngram_speculation import CountNGramLM, prompt_lookup, verify_copied_tokens


class NGramTests(unittest.TestCase):
    def test_longest_match_and_copy(self):
        result = prompt_lookup([1, 2, 3, 4, 1, 2], max_context=3, max_draft=2)
        self.assertEqual(result.matched_length, 2)
        self.assertEqual(result.source_start, 0)
        self.assertEqual(result.candidates, [3, 4])

    def test_no_match_self_match_and_zero_budget(self):
        for tokens in [[], [1], [1,2,3]]:
            self.assertEqual(prompt_lookup(tokens).candidates, [])
        self.assertEqual(prompt_lookup([1,1], max_draft=0).candidates, [])

    def test_multiple_occurrences_and_overlap(self):
        tokens = [1,2,3,1,2,4,1,2]
        self.assertEqual(prompt_lookup(tokens,2,1,True).candidates,[4])
        self.assertEqual(prompt_lookup(tokens,2,1,False).candidates,[3])
        result=prompt_lookup([1,1,1,1], max_context=3, max_draft=5)
        self.assertEqual(result.candidates,[1])  # never read past committed history

    def test_count_distribution_and_no_document_crossing(self):
        lm=CountNGramLM(4,2).fit([[0,1,2],[0,1,3]])
        q,context=lm.distribution_with_context([0,1])
        self.assertEqual(context,(0,1))
        np.testing.assert_allclose(q,[0,0,.5,.5])
        self.assertNotIn((2,),lm.counts)  # no synthetic 2 -> next document's 0

    def test_backoff_smoothing_and_empty_corpus(self):
        lm=CountNGramLM(4,2,add_k=1).fit([[0,1,2],[3,1,2]])
        q,context=lm.distribution_with_context([2,1])
        self.assertEqual(context,(1,))
        np.testing.assert_allclose(q,np.array([1,1,3,1])/6)
        np.testing.assert_allclose(CountNGramLM(4).next_probs([1]),[.25]*4)

    def test_deterministic_copy_repair(self):
        p=np.array([.5,.3,.2]);rng=np.random.default_rng(2026)
        n=15_000;counts=np.zeros(3);accepted=0
        for _ in range(n):
            step=verify_copied_tokens([1],[p,p],3,rng)
            counts[step.tokens[0]]+=1;accepted+=step.accepted
        np.testing.assert_allclose(counts/n,p,atol=.012)
        self.assertAlmostEqual(accepted/n,.3,delta=.012)

    def test_invalid_corpus_does_not_destroy_fitted_model(self):
        model = CountNGramLM(3, 2).fit([[0, 1, 2]])
        before = model.next_probs([0, 1]).copy()
        for bad in [[-1], [3], [1.5], [True]]:
            with self.assertRaises(ValueError): model.fit([[0, 2], bad])
            np.testing.assert_array_equal(model.next_probs([0, 1]), before)
        for value in [float('nan'), float('inf')]:
            with self.assertRaises(ValueError): CountNGramLM(3, add_k=value)
        with self.assertRaises(ValueError): prompt_lookup([-1, -1])
        with self.assertRaises(ValueError):
            verify_copied_tokens([-1], [np.array([.5,.5])]*2, 2, np.random.default_rng())


class LookupGenerationTests(unittest.TestCase):
    def test_transformer_lookup_greedy_matches_target(self):
        import torch
        from examples.tiny_transformer import TinyTransformer, autoregressive_generate, lookup_generate
        torch.set_num_threads(1)
        torch.manual_seed(10)
        model = TinyTransformer(vocab=3, width=8, heads=2, layers=1).double().eval()
        for prompt in [[0], [0, 1, 0, 1], [2, 2, 2]]:
            baseline = autoregressive_generate(model, prompt, 18)
            for gamma in [0, 1, 4]:
                output, trace = lookup_generate(model, prompt, 18, gamma=gamma)
                self.assertEqual(output, baseline)
                position = len(prompt)
                for row in trace:
                    position += len(row['emitted'])
                    self.assertEqual(row['cache_length'], position - 1)

    def test_lookup_fallback_copy_and_terminal_eos(self):
        import torch
        from examples.tiny_transformer import TinyTransformer, lookup_generate
        torch.manual_seed(3)
        model = TinyTransformer(vocab=2, width=8, heads=2, layers=1).double().eval()
        with torch.no_grad(): model.head.weight.zero_()  # greedy always outputs 0
        output, trace = lookup_generate(model, [1], 8, gamma=4)
        self.assertEqual(output, [1] + [0]*8)
        self.assertEqual(trace[0]['proposed'], [])
        self.assertTrue(any(row['accepted'] > 0 for row in trace))
        output, trace = lookup_generate(model, [0, 0, 0], 8, gamma=4, eos_id=0)
        self.assertEqual(output, [0, 0, 0, 0])
        self.assertEqual(trace[0]['accepted'], 1)
        self.assertEqual(trace[0]['cache_length'], 3)

    def test_lookup_sampling_uses_point_mass_proposals_and_full_prefix_target(self):
        import torch
        from unittest.mock import patch
        from examples import tiny_transformer as tiny
        torch.manual_seed(3)
        model = tiny.TinyTransformer(vocab=3, width=8, heads=2, layers=1).double().eval()
        prefix = [0, 1, 0, 1]
        verify = tiny.verify_candidates
        def checked(candidates, qs, ps, rng, **kwargs):
            with torch.inference_mode():
                for i, p in enumerate(ps):
                    logits, _ = model(tiny.token_tensor(prefix + list(candidates[:i]), model))
                    np.testing.assert_allclose(p, tiny.probabilities(logits[0,-1], 1.0), atol=1e-12)
                for token, q in zip(candidates, qs):
                    self.assertEqual(q[token], 1.0)
                    self.assertEqual(np.count_nonzero(q), 1)
            step = verify(candidates, qs, ps, rng, **kwargs)
            prefix.extend(step.tokens)
            return step
        with patch.object(tiny, 'verify_candidates', side_effect=checked):
            output, trace = tiny.lookup_generate(model, prefix.copy(), 20, greedy=False)
        self.assertEqual(output, prefix)
        self.assertTrue(any(row['proposed'] for row in trace))


if __name__ == "__main__":
    unittest.main()
