"""Regression tests derived from the technical review's counterexamples."""
import contextlib
import io
import itertools
import math
from pathlib import Path
import tempfile
import unittest
import torch
from examples.model import Attention, Config, RMSNorm
from examples.objectives import packed_causal_mask, sft_loss, distillation_loss
from examples.posttrain import clipped_policy_loss, gspo_loss, masked_sequence_logps, on_policy_loss, sampled_kl_metric
from examples.evaluation import pass_at_k, wilson_interval, paired_bootstrap
from examples.train import run
from examples.scaling import optimal_allocation

torch.set_num_threads(2)


class NumericalReviewTests(unittest.TestCase):
    def test_attention_preserves_double_precision_for_gradcheck(self):
        torch.manual_seed(3)
        attention=Attention(Config(width=8,heads=2,kv_heads=1,hidden=16)).double()
        x=torch.randn(1,3,8,dtype=torch.float64,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda z:attention(z)[0],(x,)))

    def test_rmsnorm_low_precision_output_contract(self):
        norm=RMSNorm(4)
        x=torch.randn(2,4,dtype=torch.bfloat16,requires_grad=True)
        result=norm(x)
        self.assertEqual(result.dtype,x.dtype)
        result.float().sum().backward()
        self.assertTrue(torch.isfinite(norm.weight.grad).all())

    def test_invalid_model_dimensions_fail_before_forward(self):
        for kwargs in [dict(heads=0),dict(width=12,heads=4),dict(heads=4,kv_heads=3),dict(layers=0)]:
            with self.assertRaises(ValueError):Config(**kwargs)

    def test_padding_nan_and_infinite_values_do_not_enter_policy_math(self):
        for objective in [clipped_policy_loss,gspo_loss]:
            z=torch.tensor([[-.3,float('nan'),1000.]],requires_grad=True)
            old=torch.tensor([[-.4,float('-inf'),-1000.]])
            mask=torch.tensor([[1.,0.,0.]])
            loss=objective(z,old,torch.tensor([1.]),mask)
            reference=objective(z[:,:1],old[:,:1],torch.tensor([1.]),mask[:,:1])
            torch.testing.assert_close(loss,reference)
            loss.backward()
            torch.testing.assert_close(z.grad[:,1:],torch.zeros(1,2))

    def test_all_padding_and_active_overflow_are_reported(self):
        for objective in [clipped_policy_loss,gspo_loss]:
            with self.assertRaises(ValueError):
                objective(torch.zeros(1,2),torch.zeros(1,2),torch.ones(1),torch.zeros(1,2))
            with self.assertRaises(ValueError):
                objective(torch.zeros(1,2),torch.full((1,2),-1000.),torch.ones(1),torch.ones(1,2))

    def test_invalid_padding_logits_are_removed_before_softmax(self):
        logits=torch.tensor([[[0.,1.],[float('nan'),float('nan')]]],dtype=torch.float64,requires_grad=True)
        result=masked_sequence_logps(logits,torch.tensor([[1,-100]]),torch.tensor([[1,0]]))
        self.assertEqual(result.dtype,torch.float64)
        self.assertAlmostEqual(result.item(),-math.log1p(math.exp(-1)),places=12)
        result.sum().backward()
        torch.testing.assert_close(logits.grad[:,1],torch.zeros(1,2,dtype=torch.float64))

    def test_on_policy_surrogate_gradient_matches_detached_ratio_at_old_policy(self):
        z=torch.tensor([[-.3,-.4],[-.7,-.2]],requires_grad=True)
        a=torch.tensor([1.,-2.]);mask=torch.ones_like(z)
        direct=on_policy_loss(z,a,mask)
        surrogate=clipped_policy_loss(z,z.detach().clone(),a,mask)
        gd=torch.autograd.grad(direct,z)[0];gs=torch.autograd.grad(surrogate,z)[0]
        torch.testing.assert_close(gd,gs)

    def test_sft_normalizations_have_different_expected_weights(self):
        logits=torch.zeros(2,3,2,requires_grad=True)
        labels=torch.zeros(2,3,dtype=torch.long);mask=torch.tensor([[1,0,0],[1,1,1]])
        token_grad=torch.autograd.grad(sft_loss(logits,labels,mask,"token"),logits)[0]
        sequence_grad=torch.autograd.grad(sft_loss(logits,labels,mask,"sequence"),logits)[0]
        self.assertAlmostEqual(token_grad[0,0,0].item(),-.125)
        self.assertAlmostEqual(sequence_grad[0,0,0].item(),-.25)
        self.assertAlmostEqual(sequence_grad[1,0,0].item(),-1/12)

    def test_packed_mask_isolates_documents_and_padding(self):
        mask=packed_causal_mask(torch.tensor([[0,0,1,1,-1]]))
        expected=torch.tensor([[[1,0,0,0,0],[1,1,0,0,0],[0,0,1,0,0],[0,0,1,1,0],[0,0,0,0,0]]],dtype=torch.bool)
        self.assertTrue(torch.equal(mask,expected))

    def test_k3_has_correct_value_but_fixed_sample_gradient_can_be_wrong(self):
        z=torch.tensor([.4,-.2],dtype=torch.float64,requires_grad=True)
        logp=z.log_softmax(-1);p=logp.exp();ref=torch.tensor([.1,.9],dtype=torch.float64)
        k3=(ref.log()-logp).exp()-1-(ref.log()-logp)
        kl=(p*(logp-ref.log())).sum()
        torch.testing.assert_close((p*k3).sum(),kl)
        true_grad=torch.autograd.grad(kl,z,retain_graph=True)[0]
        wrong_grad=torch.autograd.grad((p.detach()*k3).sum(),z)[0]
        expected=p.detach()[0]*p.detach()[1]*(.6-math.log(.1/.9))
        torch.testing.assert_close(true_grad,torch.stack((expected,-expected)))
        torch.testing.assert_close(wrong_grad,p.detach()-ref)
        self.assertGreater((true_grad-wrong_grad).abs().max().item(),.01)
        metric=sampled_kl_metric(logp[None],ref.log()[None],torch.ones(1,2))
        self.assertFalse(metric.requires_grad)


class StatisticalReviewTests(unittest.TestCase):
    def test_scaling_optimum_satisfies_stationarity_and_budget(self):
        n,d=optimal_allocation(1e6)
        self.assertAlmostEqual(6*n*d,1e6,places=6)
        self.assertAlmostEqual(.35*2*n**(-.35),.3*d**(-.3),places=12)
        loss=lambda x:1+2*x**(-.35)+(1e6/(6*x))**(-.3)
        self.assertLess(loss(n),loss(n*.8))
        self.assertLess(loss(n),loss(n*1.2))

    def test_distillation_teacher_frozen_and_student_moves_toward_teacher(self):
        student=torch.zeros(1,2,2,dtype=torch.float64,requires_grad=True)
        teacher=torch.tensor([[[2.,-2.],[float('nan'),float('nan')]]],dtype=torch.float64,requires_grad=True)
        loss=distillation_loss(student,teacher,torch.tensor([[1,0]]))
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertLess(student.grad[0,0,0].item(),0)
        torch.testing.assert_close(student.grad[:,1],torch.zeros(1,2,dtype=torch.float64))

    def test_pass_at_k_matches_enumerated_subsets(self):
        n,c,k=7,3,3
        subsets=list(itertools.combinations(range(n),k))
        exact=sum(any(i<c for i in subset) for subset in subsets)/len(subsets)
        self.assertAlmostEqual(pass_at_k(n,c,k),exact,places=12)
        self.assertEqual(pass_at_k(10,0,3),0)
        self.assertEqual(pass_at_k(10,10,3),1)

    def test_wilson_nonzero_uncertainty_at_boundaries(self):
        low,high=wilson_interval(0,10)
        self.assertAlmostEqual(low,0)
        self.assertGreater(high,.2)
        low,high=wilson_interval(10,10)
        self.assertLess(low,.8)
        self.assertAlmostEqual(high,1)

    def test_paired_bootstrap_identical_systems_has_zero_difference(self):
        result=paired_bootstrap([0,1,1,0],[0,1,1,0],repetitions=100)
        self.assertEqual(result['difference'],0)
        self.assertEqual(result['interval_95'],[0.,0.])

    def test_finite_group_baseline_bias_by_exact_enumeration(self):
        p=.3;g=3
        centered=rloo=0.
        for actions in itertools.product([0,1],repeat=g):
            probability=math.prod(p if a else 1-p for a in actions)
            mean=sum(actions)/g
            estimate=sum((a-mean)*(a-p) for a in actions)/g
            centered+=probability*estimate;rloo+=probability*estimate*g/(g-1)
        self.assertAlmostEqual(centered,(1-1/g)*p*(1-p))
        self.assertAlmostEqual(rloo,p*(1-p))


class ResumeReviewTests(unittest.TestCase):
    def test_interrupted_training_matches_continuous_training(self):
        with tempfile.TemporaryDirectory() as folder,contextlib.redirect_stdout(io.StringIO()):
            root=Path(folder)
            run(steps=8,out=root/'full')
            run(steps=8,out=root/'split',stop_after=3)
            run(steps=8,out=root/'split',resume=root/'split/checkpoint.pt')
            full=torch.load(root/'full/checkpoint.pt',weights_only=True)
            split=torch.load(root/'split/checkpoint.pt',weights_only=True)
            def equal(a,b):
                if isinstance(a,torch.Tensor):self.assertTrue(torch.equal(a,b))
                elif isinstance(a,dict):
                    self.assertEqual(a.keys(),b.keys())
                    for key in a:equal(a[key],b[key])
                elif isinstance(a,(tuple,list)):
                    self.assertEqual(len(a),len(b))
                    for x,y in zip(a,b):equal(x,y)
                else:self.assertEqual(a,b)
            equal(full,split)
            with self.assertRaises(ValueError):run(steps=9,resume=root/'split/checkpoint.pt')


if __name__=='__main__':unittest.main()
