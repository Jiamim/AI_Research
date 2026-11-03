"""Counterexample and gradient checks for the accepted external review suggestions."""
import unittest
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from examples.numerical_tools import (
    dpo_shared_parameter_demo, corrected_kl, mismatch_stats,
    bootstrap_power_prediction, fake_quantize_ste, two_of_four_mask,
    sae_encode_decode, steer_feature_with_residual, muon_shape_rms,
)


class ReviewUpdateTests(unittest.TestCase):
    def test_dpo_chosen_can_fall_while_ratio_improves(self):
        r=dpo_shared_parameter_demo()
        self.assertLess(r['chosen_after'],r['chosen_before'])
        self.assertLess(r['rejected_after'],r['rejected_before'])
        self.assertGreater(r['ratio_after'],1.)
        self.assertAlmostEqual(r['chosen_after']+r['rejected_after']+r['remaining_after'],1.)

    def test_corrected_k3_value_and_full_product_gradient_match_exact_kl(self):
        z=torch.tensor([.3,-.5,.8],dtype=torch.float64,requires_grad=True)
        q=torch.tensor([.2,.5,.3],dtype=torch.float64)
        old=torch.tensor([.6,.1,.3],dtype=torch.float64)
        actual=corrected_kl(z,q,old)
        p=z.softmax(-1);expected=(p*(p.log()-q.log())).sum()
        torch.testing.assert_close(actual,expected)
        ga=torch.autograd.grad(actual,z,retain_graph=True)[0]
        ge=torch.autograd.grad(expected,z)[0]
        torch.testing.assert_close(ga,ge)

    def test_detaching_outer_ratio_changes_kl_gradient(self):
        z=torch.tensor([.3,-.5,.8],dtype=torch.float64,requires_grad=True)
        q=torch.tensor([.2,.5,.3],dtype=torch.float64);old=torch.tensor([.6,.1,.3],dtype=torch.float64)
        p=z.softmax(-1);k3=q/p-(q/p).log()-1
        wrong=(old*(p/old).detach()*k3).sum()
        right=corrected_kl(z,q,old)
        gw=torch.autograd.grad(wrong,z,retain_graph=True)[0]
        gr=torch.autograd.grad(right,z)[0]
        self.assertGreater((gw-gr).abs().max().item(),.01)

    def test_mismatch_tail_and_invalid_padding(self):
        a=torch.tensor([0.,0.,2.,float('nan')]);b=torch.zeros(4)
        r=mismatch_stats(a,b,torch.tensor([1,1,1,0]))
        self.assertEqual(r['absolute_p50'],0.)
        self.assertEqual(r['absolute_max'],2.)
        self.assertGreater(r['absolute_p99'],r['absolute_p50'])

    def test_bootstrap_new_run_includes_residual_variation(self):
        rng=np.random.default_rng(11);c=np.geomspace(1,100,50)
        y=3*c**(-.2)*np.exp(rng.normal(0,.08,len(c)))
        r=bootstrap_power_prediction(c,y,150,repeats=500)
        mean,future=r['fitted_mean_interval'],r['new_run_prediction_interval']
        self.assertGreater(future[1]-future[0],mean[1]-mean[0])
        self.assertTrue(mean[0]<r['prediction']<mean[1])

    def test_fake_quant_forward_and_proxy_gradient_are_distinct(self):
        x=torch.tensor([.1,.9,5.],requires_grad=True)
        y=fake_quantize_ste(x,scale=.5,qmax=3)
        torch.testing.assert_close(y,torch.tensor([0.,1.,1.5]))
        y.sum().backward();torch.testing.assert_close(x.grad,torch.ones_like(x))

    def test_two_of_four_pattern_not_merely_global_half_sparsity(self):
        w=torch.tensor([[1.,2.,3.,4.,40.,30.,20.,10.]])
        mask=two_of_four_mask(w)
        self.assertTrue(torch.equal(mask.reshape(-1,4).sum(-1),torch.tensor([2,2])))
        torch.testing.assert_close(w*mask,torch.tensor([[0.,0.,3.,4.,40.,30.,0.,0.]]))

    def test_sae_steering_preserves_reconstruction_residual(self):
        h=torch.tensor([[1.,-2.,3.]])
        encoder=torch.tensor([[1.,0.],[0.,1.],[0.,0.]])
        decoder=torch.tensor([[1.,0.,0.],[0.,1.,0.]])
        z,recon=sae_encode_decode(h,encoder,decoder)
        modified=steer_feature_with_residual(h,z,decoder,0,2.)
        znew=z.clone();znew[:,0]+=2
        torch.testing.assert_close(modified-znew@decoder,h-recon)
        self.assertEqual(int((z!=0).sum()),1)

    def test_muon_shape_scaling_keeps_ideal_per_parameter_rms(self):
        torch.manual_seed(31)
        for shape in [(2,5),(5,2),(3,3)]:
            r=muon_shape_rms(torch.randn(*shape,dtype=torch.float64))
            self.assertAlmostEqual(r['unscaled_rms'],1/np.sqrt(max(shape)),places=12)
            self.assertAlmostEqual(r['scaled_rms'],.2,places=12)

    def test_activation_checkpoint_preserves_dropout_gradient(self):
        x=torch.linspace(.1,1.,32,dtype=torch.float64,requires_grad=True)
        w=torch.linspace(1.,2.,32,dtype=torch.float64,requires_grad=True)
        def f(a,b):return torch.nn.functional.dropout(a*b,p=.5,training=True).square().sum()
        torch.manual_seed(12);reference=f(x,w)
        expected=torch.autograd.grad(reference,(x,w))
        torch.manual_seed(12);actual=checkpoint(f,x,w,use_reentrant=False,preserve_rng_state=True)
        gradients=torch.autograd.grad(actual,(x,w))
        torch.testing.assert_close(actual,reference)
        for a,b in zip(gradients,expected):torch.testing.assert_close(a,b)


if __name__=='__main__':unittest.main()
