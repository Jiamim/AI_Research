"""Semantic and counterexample tests for the advanced chapters."""
import math
import unittest
import numpy as np
import torch
from examples.research import reweight_domains,pareto_front,factorial_interaction,gradient_transfer_demo
from examples.retrieval import BM25,ranking_metrics,ndcg_at_k,demo as retrieval_demo
from examples.reasoning import iid_pass_probability,select_candidate,verifier_experiment
from examples.compression import symmetric_quantize,low_rank_approximation,optimal_compensation,demo as compression_demo
from examples.interpretability import ToyCircuit,patch_output,recovery,demo as patching_demo


class ResearchTests(unittest.TestCase):
    def test_domain_reweighting_normalized_and_floored(self):
        result=reweight_domains([.5,.3,.2],[0.,10000.,-10000.],floor_mass=.06)
        self.assertAlmostEqual(result.sum(),1.)
        self.assertTrue(np.all(result>=.02))
        self.assertEqual(int(result.argmax()),1)

    def test_zero_update_preserves_normalized_weights_without_floor(self):
        np.testing.assert_allclose(reweight_domains([5,3,2],[1,2,3],eta=0,floor_mass=0),[.5,.3,.2])

    def test_pareto_direction_and_duplicate_points(self):
        self.assertEqual(pareto_front([[80,200],[81,300],[79,250],[80,200]],[True,False]),[0,1,3])

    def test_gradient_conflict_and_factorial_interaction(self):
        result=gradient_transfer_demo()
        self.assertLess(result['gradient_inner_product'],0)
        self.assertGreater(result['actual_old_loss_change'],0)
        self.assertLess(result['actual_new_loss_change'],0)
        self.assertEqual(factorial_interaction(70,72,73,74),-1)


class RetrievalTests(unittest.TestCase):
    def test_known_documents_retrieved_on_fixed_corpus(self):
        result=retrieval_demo()
        self.assertTrue(all(r['metrics']['mrr']==1 for r in result['queries'][:3]))
        # The paraphrase shares no characters with the relevant document: lexical recall is zero.
        self.assertEqual(result['queries'][3]['ranked'],[])
        self.assertEqual(result['queries'][3]['metrics']['recall'],0)

    def test_reciprocal_rank_fusion_uses_ranks_only(self):
        from examples.retrieval import reciprocal_rank_fusion
        self.assertEqual(reciprocal_rank_fusion([['a','b'],['b','c']]),['b','a','c'])
        self.assertEqual(reciprocal_rank_fusion([['a','a','b']]),['a','b'])

    def test_unknown_query_does_not_return_zero_score_documents(self):
        self.assertEqual(BM25([('a','alpha beta'),('b','gamma')]).search('unseen'),[])

    def test_retrieval_metrics_deduplicate_and_respect_k(self):
        result=ranking_metrics(['wrong','a','a','b'],{'a','b'},2)
        self.assertEqual(result,{'recall':.5,'mrr':.5})

    def test_ndcg_rewards_graded_order_not_just_presence(self):
        gains={'a':3,'b':1,'c':0}
        self.assertAlmostEqual(ndcg_at_k(['a','b','c'],gains,3),1.0)
        # Same two documents retrieved, so recall@2 is identical, but the marginal one ranks first.
        swapped=(1+7/math.log2(3))/(7+1/math.log2(3))
        self.assertAlmostEqual(ndcg_at_k(['b','a'],gains,2),swapped)
        self.assertEqual(ranking_metrics(['b','a'],{'a','b'},2),ranking_metrics(['a','b'],{'a','b'},2))
        self.assertEqual(ndcg_at_k(['x','a','a'],gains,1),0.0)
        with self.assertRaises(ValueError):ndcg_at_k(['a'],{'a':0},1)

    def test_document_length_normalization(self):
        index=BM25([('short','rare'),('long','rare '+'common '*100)])
        self.assertEqual(index.search('rare')[0][0],'short')


class ReasoningTests(unittest.TestCase):
    def test_iid_coverage_formula_and_boundary(self):
        self.assertEqual(iid_pass_probability(0,5),0)
        self.assertEqual(iid_pass_probability(1,5),1)
        self.assertAlmostEqual(iid_pass_probability(.25,4),1-.75**4)

    def test_verifier_can_choose_wrong_despite_correct_candidate(self):
        correctness=np.array([True,False]);scores=[1.,2.]
        self.assertTrue(correctness.any())
        self.assertFalse(correctness[select_candidate(scores)])

    def test_synthetic_selection_never_exceeds_oracle_coverage(self):
        result=verifier_experiment(trials=1000,ks=(1,4,16))
        coverage=[r['oracle_coverage'] for r in result['records']]
        self.assertEqual(coverage,sorted(coverage))
        self.assertTrue(all(r['selected_accuracy']<=r['oracle_coverage'] for r in result['records']))


class CompressionTests(unittest.TestCase):
    def test_quadratic_compensation_satisfies_constraint_and_beats_no_compensation(self):
        h=torch.tensor([[2.,1.],[1.,2.]],dtype=torch.float64)
        update=optimal_compensation(h,0,.2)
        torch.testing.assert_close(update,torch.tensor([.2,-.1],dtype=torch.float64))
        naive=torch.tensor([.2,0.],dtype=torch.float64)
        self.assertLess(update@h@update,naive@h@naive)
        self.assertAlmostEqual((h@update)[1].item(),0.)
    def test_zero_quantization_is_finite_and_exact(self):
        for channel in [True,False]:
            quantized,scale=symmetric_quantize(torch.zeros(3,4),per_output_channel=channel)
            self.assertTrue(torch.isfinite(scale).all())
            torch.testing.assert_close(quantized,torch.zeros_like(quantized))

    def test_channel_quantization_preserves_small_channel_better(self):
        w=torch.tensor([[1.,1.],[.01,.01]],dtype=torch.float64)
        tensor,_=symmetric_quantize(w,bits=2)
        channel,_=symmetric_quantize(w,bits=2,per_output_channel=True)
        self.assertLess((channel-w).square().sum(),(tensor-w).square().sum())

    def test_diagonal_scaling_preserves_unquantized_function(self):
        torch.manual_seed(12)
        w=torch.randn(3,4,dtype=torch.float64);x=torch.randn(4,5,dtype=torch.float64)
        scale=torch.tensor([.1,1.,3.,8.],dtype=torch.float64)
        torch.testing.assert_close((w*scale)@(x/scale[:,None]),w@x)

    def test_weight_error_and_output_error_can_disagree(self):
        result=compression_demo()
        self.assertLess(result['smaller_weight_mse'],result['larger_weight_mse'])
        self.assertGreater(result['smaller_weight_error_output_mse'],result['larger_weight_error_output_mse'])

    def test_low_rank_residual_matches_discarded_singular_value(self):
        w=torch.diag(torch.tensor([3.,2.,1.],dtype=torch.float64))
        approximation=low_rank_approximation(w,2)
        self.assertAlmostEqual((w-approximation).square().sum().item(),1.)


class InterpretabilityTests(unittest.TestCase):
    def test_relevant_and_irrelevant_patching_controls(self):
        result=patching_demo()
        self.assertEqual(result['recovery'],1)
        self.assertEqual(result['irrelevant_patch'],result['corrupt'])
        self.assertTrue(result['hook_cleanup_verified'])

    def test_hook_removed_on_exception(self):
        model=ToyCircuit();x=torch.tensor([[1.,2.,3.]])
        original=model(x).item()
        with self.assertRaises(RuntimeError):
            with patch_output(model.hidden,torch.zeros_like(x),torch.ones_like(x,dtype=torch.bool)):
                self.assertEqual(model(x).item(),0)
                raise RuntimeError('deliberate interruption')
        self.assertEqual(model(x).item(),original)
        self.assertEqual(len(model.hidden._forward_hooks),0)

    def test_recovery_is_not_a_probability(self):
        self.assertEqual(recovery(2,1,3),2)
        self.assertEqual(recovery(2,1,0),-1)
        with self.assertRaises(ValueError):recovery(1,1,2)


if __name__=='__main__':unittest.main()
