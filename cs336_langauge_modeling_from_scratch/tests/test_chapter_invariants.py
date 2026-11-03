"""Independent calculations for the assumptions added by the full chapter review."""
import itertools
import math
import unittest

import torch
from torch.nn import functional as F

from examples.extensions import clip_loss
from examples.model import Config, TinyLM, sample_next
from examples.numerical_tools import chapter_invariants_demo
from examples.objectives import sft_loss
from examples.optim import clip_grad_norm
from examples.posttrain import on_policy_loss


class ChapterInvariantTests(unittest.TestCase):
    def test_merged_softmax_has_dense_values_and_gradients(self):
        result = chapter_invariants_demo()
        self.assertLess(result['softmax_merge']['output_error'], 1e-12)
        self.assertLess(result['softmax_merge']['gradient_error'], 1e-11)
        self.assertEqual(result['kimi_yarn'], dict(unscaled_pairs=20, interpolated_pairs=12))

    def test_delta_rule_residual_stability_matches_a_gradient_step(self):
        state = torch.tensor([[.2, .3], [.4, -.1]], dtype=torch.float64, requires_grad=True)
        key = torch.tensor([2., 0.], dtype=torch.float64)
        value = torch.tensor([1., -1.], dtype=torch.float64)
        loss = .5 * (state.T @ key - value).square().sum()
        gradient, = torch.autograd.grad(loss, state)
        for rate in (.125, .25, .75):
            updated = state.detach() - rate * gradient
            actual = (updated.T @ key - value).square().sum() / (2*loss.detach())
            self.assertAlmostEqual(actual.item(), (1-rate*key.square().sum().item())**2)

    def test_nonuniform_expert_union_matches_enumeration(self):
        probabilities = [.7, .3, 0., 0.]
        exact = sum(math.prod(probabilities[i] for i in assignments)*len(set(assignments))
                    for assignments in itertools.product(range(4), repeat=3))
        result = chapter_invariants_demo()['expert_union']
        self.assertAlmostEqual(result['concentrated'], exact)
        self.assertLess(exact, result['uniform'])

    def test_siglip_common_divisor_changes_scale_not_stationary_bias(self):
        batch = 4
        labels = 2*torch.eye(batch,dtype=torch.float64)-1
        for common_score in (0., .7):
            for divisor in (batch, batch*batch):
                bias = torch.tensor(-math.log(batch-1)-common_score,dtype=torch.float64,requires_grad=True)
                loss = F.softplus(-labels*(common_score+bias)).sum()/divisor
                gradient, = torch.autograd.grad(loss,bias,create_graph=True)
                self.assertLess(abs(gradient.item()),1e-12)
                self.assertGreater(torch.autograd.grad(gradient,bias)[0].item(),0.)

    def test_replicated_ddp_gradients_are_one_logical_vector_for_clipping(self):
        gradient = torch.tensor([3.,4.],dtype=torch.float64)
        parameter = torch.nn.Parameter(torch.zeros_like(gradient))
        parameter.grad = gradient.clone()
        actual_norm = clip_grad_norm([parameter],max_norm=5.,eps=0.)
        self.assertEqual(actual_norm.item(),5.)
        torch.testing.assert_close(parameter.grad,gradient)
        # Independent shards reconstruct the logical norm; four replicas overcount it.
        sharded_norm = sum(part.square().sum() for part in gradient.split(1)).sqrt()
        duplicated_norm = gradient.repeat(4).norm()
        self.assertEqual(sharded_norm.item(),actual_norm.item())
        self.assertEqual(duplicated_norm.item(),10.)
        torch.testing.assert_close(gradient*(5/duplicated_norm),gradient/2)

    def test_separate_clip_microbatches_remove_the_cross_batch_negatives(self):
        image = torch.eye(2,dtype=torch.float64,requires_grad=True)
        text = torch.tensor([[1.,.3],[.4,1.]],dtype=torch.float64)
        full = clip_loss(image,text,temperature=1.)
        separate = sum(clip_loss(image[i:i+1],text[i:i+1],temperature=1.) for i in range(2))/2
        self.assertEqual(separate.item(),0.)
        self.assertGreater(full.item(),0.)
        self.assertGreater(torch.autograd.grad(full,image)[0].norm().item(),.1)
        self.assertEqual(torch.autograd.grad(separate,image)[0].norm().item(),0.)

    def test_fixed_length_normalization_keeps_sequence_microbatch_weights(self):
        logps = torch.tensor([[-.2,-.4,-.5],[-.1,-.3,-.8],[-.5,-.7,-.9]],dtype=torch.float64,requires_grad=True)
        mask = torch.tensor([[1,0,0],[1,1,0],[1,1,1]],dtype=torch.bool)
        advantage = torch.tensor([1.,-1.,.5],dtype=torch.float64)
        full = on_policy_loss(logps,advantage,mask,normalization='constant',max_length=3)
        split = (on_policy_loss(logps[:1],advantage[:1],mask[:1],normalization='constant',max_length=3)/3
                 +on_policy_loss(logps[1:],advantage[1:],mask[1:],normalization='constant',max_length=3)*2/3)
        torch.testing.assert_close(full,split)
        torch.testing.assert_close(torch.autograd.grad(full,logps)[0],torch.autograd.grad(split,logps)[0])

    def test_response_only_loss_still_reaches_prompt_representations(self):
        with torch.random.fork_rng():
            torch.manual_seed(4)
            model = TinyLM(Config(vocab=16,width=8,layers=1,heads=2,kv_heads=2,hidden=16)).double()
        ids = torch.tensor([[1,2,3]])
        labels = torch.tensor([[2,3,4]])
        mask = torch.tensor([[0,0,1]],dtype=torch.bool)
        sft_loss(model(ids),labels,mask).backward()
        self.assertGreater(model.embedding.weight.grad[1].norm().item(),0.)

    def test_k3_support_missing_mass_changes_the_expectation(self):
        p = torch.tensor([1.,0.],dtype=torch.float64)
        q = torch.tensor([.5,.5],dtype=torch.float64)
        valid = p>0
        ratio = q[valid]/p[valid]
        k3 = (p[valid]*(ratio-1-ratio.log())).sum()
        kl = (p[valid]*(p[valid].log()-q[valid].log())).sum()
        self.assertAlmostEqual(k3.item(),kl.item()+q[valid].sum().item()-1)
        self.assertNotAlmostEqual(k3.item(),kl.item())

    def test_temperature_limit_with_ties_and_greedy_have_different_support(self):
        logits = torch.tensor([[2.,2.,0.]],dtype=torch.float64)
        probabilities = (logits/1e-6).softmax(-1)
        torch.testing.assert_close(probabilities,torch.tensor([[.5,.5,0.]],dtype=torch.float64))
        self.assertEqual(sample_next(logits,temperature=0).item(),0)

    def test_layer_error_contains_the_cross_term(self):
        w1 = torch.tensor([[1.,2.],[0.,1.]],dtype=torch.float64)
        w2 = torch.tensor([[1.,0.],[3.,1.]],dtype=torch.float64)
        e1 = torch.tensor([[.1,0.],[0.,-.2]],dtype=torch.float64)
        e2 = torch.tensor([[0.,.3],[.2,0.]],dtype=torch.float64)
        x = torch.tensor([2.,-1.],dtype=torch.float64)
        exact = (w2+e2)@(w1+e1)@x-w2@w1@x
        decomposition = w2@e1@x+e2@w1@x+e2@e1@x
        torch.testing.assert_close(exact,decomposition)
        self.assertGreater((exact-(w2@e1@x+e2@w1@x)).norm().item(),.01)

    def test_quantized_lora_merge_can_erase_a_nonzero_update(self):
        original, update, step = 1., .1, .5
        merged = step*round((original+update)/step)
        self.assertEqual(merged,original)
        self.assertNotEqual(merged,original+update)

    def test_bounded_logit_error_bounds_probability_ratio_but_not_argmax(self):
        logits = torch.tensor([0.,.01,-2.],dtype=torch.float64)
        epsilon = .01
        perturbed = logits+torch.tensor([epsilon,-epsilon,0.],dtype=torch.float64)
        ratio = perturbed.softmax(-1)/logits.softmax(-1)
        self.assertTrue(bool(torch.all(ratio>=math.exp(-2*epsilon))))
        self.assertTrue(bool(torch.all(ratio<=math.exp(2*epsilon))))
        self.assertNotEqual(logits.argmax().item(),perturbed.argmax().item())


if __name__=='__main__':unittest.main()
