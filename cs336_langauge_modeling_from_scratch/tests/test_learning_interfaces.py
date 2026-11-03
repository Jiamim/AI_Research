"""Independent references for the book's text, probability and trajectory interfaces."""
import itertools
import math
import unittest
from unittest import mock

import numpy as np
import torch

from examples.evaluation import paired_bootstrap, paired_cluster_bootstrap
from examples.extensions import linear_attention_explicit, linear_attention_recurrent
from examples.inference import speculative_step
from examples.mechanisms import tiled_linear_cross_entropy
from examples.numerical_tools import packing_equivalence_demo
from examples.objectives import packed_causal_mask, packed_next_token_mask
from examples.posttrain import generalized_advantages
from examples.tokenizer import Tokenizer


class StreamingTextTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Tokenizer({i:bytes([i]) for i in range(256)}, [])

    def test_codepoints_can_span_tokens_and_graphemes_can_span_outputs(self):
        text = '中🙂 e\u0301 👩\u200d💻'
        ids = list(text.encode('utf-8'))
        self.assertEqual(''.join(self.tokenizer.decode_iterable(iter(ids))), text)
        self.assertNotEqual(''.join(self.tokenizer.decode([i]) for i in ids), text)

    def test_stream_flush_matches_whole_decode_for_malformed_and_partial_bytes(self):
        self.tokenizer.vocab.update({256:b'\xe4\xb8',257:b'\xadA',258:b'\xf0\x9f'})
        for ids in ([], [256,257], [255,256,257], [256], [258], [0x80,0xc0,0xaf,65]):
            with self.subTest(ids=ids):
                self.assertEqual(''.join(self.tokenizer.decode_iterable(ids)), self.tokenizer.decode(ids))

    def test_first_completed_character_does_not_consume_the_rest_of_the_stream(self):
        ids = iter('中A'.encode('utf-8'))
        decoded = self.tokenizer.decode_iterable(ids)
        self.assertEqual(next(decoded), '中')
        self.assertEqual(next(ids), ord('A'))


class PackedDocumentTests(unittest.TestCase):
    def test_attention_and_target_masks_preserve_eos_but_remove_cross_document_targets(self):
        docs = torch.tensor([[0,0,0,1,1,1,-1,-1]])
        expected = torch.tensor([[True,True,False,True,True,False,False]])
        torch.testing.assert_close(packed_next_token_mask(docs), expected)
        roles = torch.tensor([[0,1,1,0,0,1,0,0]])
        torch.testing.assert_close(packed_next_token_mask(docs,roles),
                                   torch.tensor([[True,True,False,False,True,False,False]]))
        attention = packed_causal_mask(docs)
        self.assertFalse(bool(attention[0,3:6,:3].any()))
        self.assertFalse(bool(attention[0,6:].any()))
        self.assertFalse(bool(packed_next_token_mask(torch.full((1,3),-1)).any()))

    def test_separate_documents_and_packing_have_identical_losses_and_parameter_gradients(self):
        result = packing_equivalence_demo()
        self.assertEqual(result['targets'], 5)
        self.assertEqual(result['cross_document_targets_suppressed'], 1)
        for key in ('loss_error','gradient_error','other_document_logit_error'):
            self.assertLess(result[key], 1e-11, key)

    def test_document_ids_and_role_masks_reject_silent_coercion(self):
        for docs in (torch.tensor([[0.,0.5]]), torch.tensor([[0,-2]]), torch.tensor([0,0])):
            for function in (packed_causal_mask,packed_next_token_mask):
                with self.assertRaises(ValueError):function(docs)
        docs = torch.tensor([[0,0,1,1]])
        with self.assertRaises(ValueError):packed_next_token_mask(docs,torch.ones(1,3))
        with self.assertRaises(ValueError):packed_next_token_mask(docs,torch.full((1,4),.5))


class KernelFeatureTests(unittest.TestCase):
    def test_negative_features_are_preserved_before_normalization(self):
        for dtype in (torch.float16,torch.bfloat16,torch.float32,torch.float64):
            q = torch.ones(3,16,dtype=dtype)
            k = torch.full_like(q,-20.)
            v = torch.tensor([[2.,-3.]]*3,dtype=dtype)
            for function in (linear_attention_recurrent,linear_attention_explicit):
                with self.subTest(dtype=dtype,function=function.__name__):
                    result = function(q,k,v)
                    self.assertTrue(bool(result.isfinite().all()))
                    self.assertEqual(result.dtype,dtype)
                    torch.testing.assert_close(result,v)

    def test_underflowing_kernel_mass_stays_finite_with_fp16_inputs(self):
        for function in (linear_attention_recurrent,linear_attention_explicit):
            q = torch.full((3,4),-100.,dtype=torch.float16,requires_grad=True)
            k = q.detach().clone().requires_grad_()
            v = torch.ones(3,2,dtype=torch.float16,requires_grad=True)
            result = function(q,k,v)
            torch.testing.assert_close(result,torch.zeros_like(v))
            result.sum().backward()
            for x in (q,k,v):self.assertTrue(bool(x.grad.isfinite().all()))

    def test_enclosing_autocast_does_not_lower_the_accumulator_again(self):
        q = torch.ones(3,16)
        k = torch.full_like(q,-20.)
        v = torch.tensor([[2.,-3.]]*3)
        for function in (linear_attention_recurrent,linear_attention_explicit):
            with torch.autocast('cpu',dtype=torch.float16):
                actual = function(q,k,v)
            self.assertEqual(actual.dtype,torch.float32)
            torch.testing.assert_close(actual,v)

    def test_tiled_loss_preserves_declared_precision_during_forward_and_recomputation(self):
        h = torch.tensor([[10.1,3.7],[1.2,5.9]],requires_grad=True)
        w = torch.tensor([[6.2,2.4],[2.7,1.6],[1.5,4.3]],requires_grad=True)
        labels = torch.tensor([0,2])
        expected = torch.nn.functional.cross_entropy(h@w.T,labels)
        reference_gradients = torch.autograd.grad(expected,(h,w))
        with torch.autocast('cpu',dtype=torch.float16):
            actual = tiled_linear_cross_entropy(h,w,labels,block=2)
            gradients = torch.autograd.grad(actual,(h,w))
        torch.testing.assert_close(actual,expected,rtol=1e-4,atol=3e-6)
        for a,b in zip(gradients,reference_gradients):
            torch.testing.assert_close(a,b,rtol=1e-4,atol=2e-5)

    def test_recurrence_values_and_gradients_match_independent_dense_kernel(self):
        q = torch.tensor([[.3,-.5],[1.,.7],[-.4,1.2]],dtype=torch.float64,requires_grad=True)
        k = torch.tensor([[-20.,.2],[.4,-.9],[.7,.5]],dtype=torch.float64,requires_grad=True)
        v = torch.tensor([[2.,-3.,1.],[.1,.4,.8],[-.7,.2,1.]],dtype=torch.float64,requires_grad=True)
        phi = lambda x: torch.where(x<0,x.exp(),x+1)
        kernel = (phi(q) @ phi(k).T).tril()
        expected = (kernel @ v)/kernel.sum(-1,keepdim=True).clamp_min(1e-8)
        actual = linear_attention_recurrent(q,k,v)
        torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
        reference = torch.autograd.grad(expected.square().sum(),(q,k,v))
        gradient = torch.autograd.grad(actual.square().sum(),(q,k,v))
        for a,b in zip(gradient,reference):torch.testing.assert_close(a,b,rtol=1e-11,atol=1e-11)
        self.assertTrue(torch.autograd.gradcheck(linear_attention_recurrent,(q,k,v)))

    def test_time_or_dtype_mismatch_cannot_be_hidden_by_zip(self):
        q,k,v = torch.ones(3,2),torch.ones(3,2),torch.ones(3,4)
        for function in (linear_attention_recurrent,linear_attention_explicit):
            for args in ((q,k[:2],v),(q,k,v[:2]),(q,k.double(),v),(q,k,v[:,:0])):
                with self.assertRaises(ValueError):function(*args)


class ClusterBootstrapTests(unittest.TestCase):
    def test_matches_explicit_cluster_multiset_enumeration_for_both_estimands(self):
        a,b = np.zeros(6),np.array([1.,0.,0.,.5,.5,.5])
        groups = [[0],[1,2],[3,4,5]]
        labels = ['a','b','b','c','c','c']
        draws = list(itertools.product(range(3),repeat=3))
        for weighting in ('item','cluster'):
            expected=[]
            for draw in draws:
                rows = [index for group in draw for index in groups[group]]
                expected.append((b-a)[rows].mean() if weighting=='item'
                                else np.mean([(b-a)[groups[group]].mean() for group in draw]))
            rng = mock.Mock()
            rng.integers.side_effect=[np.array(draw) for draw in draws]
            with mock.patch('examples.evaluation.np.random.default_rng',return_value=rng):
                actual = paired_cluster_bootstrap(a,b,labels,repetitions=len(draws),weighting=weighting)
            np.testing.assert_allclose(actual['interval_95'],np.quantile(expected,[.025,.975]))
            point = b.mean() if weighting=='item' else np.mean([b[group].mean() for group in groups])
            self.assertAlmostEqual(actual['difference'],point)

    def test_duplicate_items_do_not_create_independent_clusters(self):
        a,b = np.zeros(8),np.repeat([0.,0.,1.,1.],2)
        labels = np.repeat(np.arange(4),2)
        original = paired_cluster_bootstrap(a,b,labels)
        repeated = paired_cluster_bootstrap(np.repeat(a,10),np.repeat(b,10),np.repeat(labels,10))
        self.assertEqual(original,repeated)
        independent = paired_bootstrap(np.repeat(a,10),np.repeat(b,10))
        width = lambda result:result['interval_95'][1]-result['interval_95'][0]
        self.assertGreater(width(original),width(independent))

    def test_singleton_clusters_match_item_bootstrap_and_invalid_groups_fail(self):
        a,b = [0.,1.,0.,1.],[1.,1.,0.,0.]
        clustered = paired_cluster_bootstrap(a,b,range(4))
        regular = paired_bootstrap(a,b)
        self.assertEqual(clustered['difference'],regular['difference'])
        self.assertEqual(clustered['interval_95'],regular['interval_95'])
        for labels in ([0]*4,[0,1],[0,1,None,2],[0,1,float('nan'),2]):
            with self.assertRaises(ValueError):paired_cluster_bootstrap(a,b,labels)


class TrajectoryAdvantageTests(unittest.TestCase):
    def test_lambda_one_matches_terminal_and_truncated_returns(self):
        rewards = torch.tensor([0.,0.,1.],dtype=torch.float64)
        values = torch.tensor([.2,.4,.7],dtype=torch.float64)
        following = torch.tensor([.4,.7,.6],dtype=torch.float64)
        last = torch.tensor([False,False,True])
        no = torch.zeros_like(last)
        terminal,targets = generalized_advantages(rewards,values,following,last,no,lam=1.)
        torch.testing.assert_close(terminal,torch.tensor([.8,.6,.3],dtype=torch.float64))
        torch.testing.assert_close(targets,torch.ones_like(values))
        timeout,targets = generalized_advantages(rewards,values,following,no,last,lam=1.)
        torch.testing.assert_close(timeout,torch.tensor([1.4,1.2,.9],dtype=torch.float64))
        torch.testing.assert_close(targets,torch.full_like(values,1.6))

    def test_discounted_trace_matches_a_finite_sum_stopped_at_each_boundary(self):
        r = torch.tensor([.2,-.1,1.,.3,.4,2.],dtype=torch.float64)
        v = torch.tensor([.1,.4,.5,.2,.3,.7],dtype=torch.float64)
        nxt = torch.tensor([.4,.5,999.,.3,.8,0.],dtype=torch.float64)
        terminal = torch.tensor([False,False,True,False,False,True])
        truncated = torch.tensor([False,False,False,False,True,False])
        for lam in (0.,.7,1.):
            actual,_ = generalized_advantages(r,v,nxt,terminal,truncated,gamma=.9,lam=lam)
            expected=[]
            for start in range(len(r)):
                total=0.
                for end in range(start,len(r)):
                    delta=r[end].item()+(.9*nxt[end].item() if not terminal[end] else 0)-v[end].item()
                    total+=(.9*lam)**(end-start)*delta
                    if terminal[end] or truncated[end]:break
                expected.append(total)
            torch.testing.assert_close(actual,torch.tensor(expected,dtype=torch.float64))

    def test_timeout_does_not_leak_reset_episode_reward_and_targets_are_detached(self):
        r = torch.tensor([1.,2.],requires_grad=True)
        v = torch.tensor([.2,.3],requires_grad=True)
        nxt = torch.tensor([.7,float('nan')],requires_grad=True)
        terminal = torch.tensor([False,True])
        truncated = torch.tensor([True,False])
        advantage,targets = generalized_advantages(r,v,nxt,terminal,truncated)
        changed = r.detach().clone();changed[1]=2000.
        other,_ = generalized_advantages(changed,v,nxt,terminal,truncated)
        self.assertAlmostEqual(advantage[0].item(),1.5)
        self.assertEqual(advantage[0].item(),other[0].item())
        self.assertFalse(advantage.requires_grad)
        self.assertFalse(targets.requires_grad)
        with self.assertRaises(ValueError):generalized_advantages(r,v,nxt,terminal.float(),truncated)
        with self.assertRaises(ValueError):generalized_advantages(r,v,nxt,terminal,truncated,lam=float('nan'))


class ProbabilityAssumptionTests(unittest.TestCase):
    def test_symmetric_initialization_bound_needs_more_than_label_independence(self):
        logits = torch.tensor([[2.,-1.],[-1.,2.]],dtype=torch.float64)
        mean_nll = -logits.log_softmax(-1)[:,0].mean().item()
        self.assertGreaterEqual(mean_nll,math.log(2))
        # Deterministic biased predictor and target are independent random variables.
        self.assertLess(-math.log(.9),math.log(2))

    def test_perfect_constant_calibration_does_not_identify_density_ratios(self):
        # Rows are two equally common inputs; columns are the two classes.
        joint = np.array([[.45,.05],[.05,.45]])
        prior = joint.sum(0)
        posterior = joint[:,0]/joint.sum(1)
        calibrated_constant = np.full(2,prior[0])
        self.assertEqual(calibrated_constant[0],joint[:,0].sum())
        density_ratio = (joint[:,0]/prior[0])/(joint[:,1]/prior[1])
        np.testing.assert_allclose(density_ratio,[9.,1/9])
        np.testing.assert_allclose(posterior/(1-posterior)*prior[1]/prior[0],density_ratio)
        self.assertFalse(np.allclose(calibrated_constant/(1-calibrated_constant),density_ratio))


class ConditionalSpeculationTests(unittest.TestCase):
    def test_two_token_joint_distribution_with_prefix_dependent_target(self):
        p = torch.tensor([.7,.3],dtype=torch.float64)
        q = torch.tensor([.4,.6],dtype=torch.float64)
        after = torch.tensor([[.2,.8],[.9,.1]],dtype=torch.float64)
        joint = torch.zeros(2,2,dtype=torch.float64)
        residual = (p-q).clamp_min(0);residual/=residual.sum()
        for draft in range(2):
            alpha = min(1.,(p[draft]/q[draft]).item())
            target = torch.stack((p,after[draft]))
            for second in range(2):
                with mock.patch('torch.rand',return_value=torch.tensor(0.)), \
                     mock.patch('torch.multinomial',return_value=torch.tensor([second])):
                    out = speculative_step(target,q[None],torch.tensor([draft]))
                joint[out[0],out[1]]+=q[draft]*alpha*after[draft,second]
            if alpha==1:continue
            for corrected in range(2):
                if residual[corrected]==0:continue
                with mock.patch('torch.rand',return_value=torch.tensor((1+alpha)/2)), \
                     mock.patch('torch.multinomial',return_value=torch.tensor([corrected])):
                    out = speculative_step(target,q[None],torch.tensor([draft]))
                self.assertEqual(out,[corrected])
                # Continue from the corrected prefix, never from the rejected draft.
                joint[out[0]]+=q[draft]*(1-alpha)*residual[corrected]*after[out[0]]
        torch.testing.assert_close(joint,p[:,None]*after,rtol=1e-12,atol=1e-12)

    def test_mid_round_rejection_does_not_sample_a_bonus_from_stale_rows(self):
        target = torch.tensor([[1.,0.],[0.,1.],[1.,0.]])
        draft = torch.tensor([[1.,0.],[1.,0.]])
        with mock.patch('torch.rand',return_value=torch.tensor(.5)), \
             mock.patch('torch.multinomial',return_value=torch.tensor([1])) as sample:
            out = speculative_step(target,draft,torch.tensor([0,0]))
        self.assertEqual(out,[0,1])
        self.assertEqual(sample.call_count,1)


if __name__ == '__main__':unittest.main()
