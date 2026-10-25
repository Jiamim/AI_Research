"""Independent invariants for executable explanations, not model benchmarks."""
import unittest
import numpy as np
import torch

from examples.principle_lab import (
    allocate_budget, block_context_mask, coupling_table, dpace_loss,
    expected_progress, optimal_tree_score, processed_distribution, run_all,
    tree_forward, tree_layout,
)
from examples.tiny_transformer import TinyTransformer


class PrincipleTests(unittest.TestCase):
    def test_coupling_has_both_marginals_and_maximum_diagonal(self):
        rng=np.random.default_rng(7)
        for _ in range(30):
            p,q=rng.dirichlet(np.ones(4),size=2)
            joint=coupling_table(p,q)
            np.testing.assert_allclose(joint.sum(0),p,atol=1e-14)
            np.testing.assert_allclose(joint.sum(1),q,atol=1e-14)
            self.assertAlmostEqual(joint.trace(),np.minimum(p,q).sum())
        np.testing.assert_array_equal(coupling_table([0,1],[1,0]),[[0,1],[0,0]])

    def test_top_p_keeps_crossing_token_and_mask_is_applied_first(self):
        p=processed_distribution(np.log([.5,.3,.2]),top_p=.6)
        np.testing.assert_allclose(p,[.625,.375,0])
        q=processed_distribution(np.log([.5,.3,.2]),top_k=1,allowed=[False,True,True])
        np.testing.assert_array_equal(q,[0,1,0])
        np.testing.assert_array_equal(processed_distribution([2,2,1],top_k=1),[1,0,0])
        with self.assertRaises(ValueError):processed_distribution([1,2],allowed=[False,False])

    def test_tree_mask_depth_and_invalid_parent(self):
        mask,depth=tree_layout([-1,0,0,1])
        np.testing.assert_array_equal(depth,[0,1,1,2])
        np.testing.assert_array_equal(mask[3],[True,True,False,True])
        with self.assertRaises(ValueError):tree_layout([-1,2,0])

    def test_tree_branch_mutation_cannot_change_sibling_branch_logits(self):
        torch.set_num_threads(1)
        torch.manual_seed(2)
        model=TinyTransformer(width=8,heads=2,layers=2).double().eval()
        parents=[-1,0,0,1]
        a,_=tree_forward(model,[1,2],parents,[2,3,4,5])
        b,_=tree_forward(model,[1,2],parents,[2,3,9,5])
        torch.testing.assert_close(a[:,[0,1,3]],b[:,[0,1,3]],atol=1e-12,rtol=1e-12)

    def test_tree_path_gather_and_serial_lower_layer_reuse(self):
        results=run_all()
        self.assertLess(results['tree']['max_logits_error'],1e-12)
        self.assertLess(results['tree']['gather_error'],1e-12)
        self.assertLess(results['layer_reuse']['max_logits_error'],1e-12)
        self.assertEqual(results['dependence']['independent_overlap'],.5)
        self.assertAlmostEqual(results['dependence']['conditional_overlap'],.95)

    def test_block_mask_hides_future_target_features_and_other_blocks(self):
        mask=block_context_mask(5,[2,4],3)
        self.assertTrue(mask[:3,:2].all())
        self.assertFalse(mask[:3,2:5].any())
        self.assertTrue(mask[:3,5:8].all())
        self.assertFalse(mask[:3,8:].any())
        self.assertFalse(mask[3:,5:8].any())

    def test_dpace_gradient_is_weighted_ce_with_detached_weights(self):
        z=torch.tensor([[.2,.4],[.8,-.1]],dtype=torch.float64,requires_grad=True)
        labels=torch.tensor([0,1])
        loss,weights=dpace_loss(z,labels,.1)
        loss.backward()
        expected=weights[:,None]*(z.detach().softmax(-1)-torch.nn.functional.one_hot(labels,2))
        torch.testing.assert_close(z.grad,expected)
        self.assertFalse(weights.requires_grad)
        self.assertGreaterEqual(float(weights[0]),float(weights[1]))

    def test_budget_optimization_and_progress_units(self):
        _,length=expected_progress([.8]*4)
        self.assertAlmostEqual(length,3.3616)
        result=allocate_budget([[1,1],[0,0]],[1]*7)
        self.assertEqual(result['lengths'],[2,0])
        self.assertEqual(result['expected_emitted'],4)
        self.assertEqual(result['total_verify_tokens'],4)
        result=allocate_budget([[.9]], [1,1,100])
        self.assertEqual(result['lengths'],[0])

    def test_tree_dp_reduces_to_chain_and_branches_when_depth_limited(self):
        self.assertAlmostEqual(optimal_tree_score([.6],4,4),1+.6+.6**2+.6**3)
        self.assertAlmostEqual(optimal_tree_score([.6,.3],3,1),1.9)
        self.assertEqual(optimal_tree_score([.6,.3],1,4),1)
