"""Independent reference checks for the guide's deeper derivations and invariants."""
import copy
import math
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from torch.nn import functional as F
from examples.mechanisms import (
    tiled_linear_cross_entropy, sharded_cross_entropy, affine_scan, mla_content_output,
    importance_diagnostics, verifier_precision, potential_shaping, stochastic_round_interval,
)
from examples.inference import PagedKVCache
from examples.train import run
from examples.objectives import sft_loss, distillation_loss
from examples.compression import input_weighted_low_rank, low_rank_approximation
from examples.numerical_tools import muon_shape_rms


class MechanismTests(unittest.TestCase):
    def test_muon_nullspace_does_not_get_an_arbitrary_update(self):
        self.assertEqual(muon_shape_rms(torch.zeros(2,3))['scaled_rms'],0.)
        rank_one=torch.tensor([[1.,0.,0.],[0.,0.,0.]],dtype=torch.float64)
        result=muon_shape_rms(rank_one)
        self.assertAlmostEqual(result['unscaled_rms'],1/math.sqrt(6),places=12)
        self.assertAlmostEqual(result['scaled_rms'],.2/math.sqrt(2),places=12)
    def test_weighted_low_rank_prefers_active_input_directions(self):
        w=torch.diag(torch.tensor([2.,1.],dtype=torch.float64))
        h=torch.diag(torch.tensor([.01,100.],dtype=torch.float64))
        weighted=input_weighted_low_rank(w,h,1)
        ordinary=low_rank_approximation(w,1)
        def error(a):
            e=w-a
            return torch.trace(e@h@e.T)
        self.assertAlmostEqual(error(weighted).item(),.04)
        self.assertAlmostEqual(error(ordinary).item(),100.)
        self.assertEqual(torch.linalg.matrix_rank(weighted).item(),1)
    def test_tiled_loss_values_gradients_and_numeric_gradcheck(self):
        torch.manual_seed(21)
        h = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
        w = torch.randn(7, 2, dtype=torch.float64, requires_grad=True)
        y = torch.tensor([0, 4, 6])
        for block in (1, 3, 20):
            ref = F.cross_entropy(h @ w.T, y)
            result = tiled_linear_cross_entropy(h, w, y, block)
            torch.testing.assert_close(result, ref)
            actual = torch.autograd.grad(result, (h, w))
            expected = torch.autograd.grad(ref, (h, w))
            for x, z in zip(actual, expected):
                torch.testing.assert_close(x, z, rtol=1e-11, atol=1e-11)
        self.assertTrue(torch.autograd.gradcheck(lambda a,b:tiled_linear_cross_entropy(a,b,y,3),(h,w)))

    def test_tiled_loss_saves_no_full_logits_tensor(self):
        h=torch.randn(5,3,requires_grad=True);w=torch.randn(17,3,requires_grad=True);y=torch.arange(5)
        saved=[]
        def pack(x):
            saved.append(tuple(x.shape))
            return x
        with torch.autograd.graph.saved_tensors_hooks(pack,lambda x:x):
            tiled_linear_cross_entropy(h,w,y,4).backward()
        self.assertNotIn((5,17),saved)
        self.assertIn((5,),saved)

    def test_virtual_vocab_shards_match_global_softmax(self):
        z=torch.tensor([[1000.,999.,998.,997.,1001.],[-2.,1.,4.,0.,5.]],dtype=torch.float64,requires_grad=True)
        y=torch.tensor([4,0])
        result=sharded_cross_entropy(list(z.split([2,1,2],dim=1)),y)
        expected=F.cross_entropy(z,y)
        torch.testing.assert_close(result,expected)
        torch.testing.assert_close(torch.autograd.grad(result,z)[0],torch.autograd.grad(expected,z)[0])

    def test_affine_scan_matches_serial_values_and_gradients(self):
        torch.manual_seed(42)
        for length in (1,7,16):
            a=torch.rand(length,3,dtype=torch.float64,requires_grad=True)
            b=torch.randn(length,3,dtype=torch.float64,requires_grad=True)
            initial=torch.randn(3,dtype=torch.float64,requires_grad=True)
            state=initial;states=[]
            for at,bt in zip(a,b):
                state=at*state+bt;states.append(state)
            expected=torch.stack(states);actual=affine_scan(a,b,initial)
            torch.testing.assert_close(actual,expected)
            for x,y in zip(torch.autograd.grad(actual.square().sum(),(a,b,initial)),
                           torch.autograd.grad(expected.square().sum(),(a,b,initial))):
                torch.testing.assert_close(x,y)

    def test_mla_absorption_preserves_output_and_all_gradients(self):
        torch.manual_seed(43)
        shapes=((2,3),(7,4),(2,3,4),(2,2,4),(2,5,2))
        args=[torch.randn(*s,dtype=torch.float64,requires_grad=True) for s in shapes]
        actual=mla_content_output(*args);reference=mla_content_output(*args,absorbed=False)
        torch.testing.assert_close(actual,reference)
        for a,b in zip(torch.autograd.grad(actual.square().sum(),args),torch.autograd.grad(reference.square().sum(),args)):
            torch.testing.assert_close(a,b)

    def test_ess_detects_concentration_but_not_common_scale(self):
        uniform=importance_diagnostics(torch.full((20,),1000.))
        self.assertAlmostEqual(uniform['ess'],20.)
        concentrated=importance_diagnostics(torch.tensor([30.]+[0.]*19))
        self.assertLess(concentrated['ess'],1.001)
        self.assertGreater(uniform['log_mean_weight'],999.)

    def test_verifier_false_positives_dominate_rare_successes(self):
        self.assertAlmostEqual(verifier_precision(.05,.9,.05),.045/.0925)
        self.assertLess(verifier_precision(.01,.9,.05),.16)
        self.assertAlmostEqual(verifier_precision(.05,.9,0.),1.)

    def test_potential_shaping_telescopes_with_discount(self):
        reward=torch.tensor([0.,0.,1.],dtype=torch.float64)
        phi=torch.tensor([.7,.1,.9,0.],dtype=torch.float64)
        gamma=.9;discount=gamma**torch.arange(3,dtype=torch.float64)
        shaped=potential_shaping(reward,phi,gamma)
        torch.testing.assert_close((shaped*discount).sum(),(reward*discount).sum()-phi[0])
        phi[-1]=.4
        torch.testing.assert_close((potential_shaping(reward,phi,gamma)*discount).sum(),
                                   (reward*discount).sum()-phi[0]+gamma**3*phi[-1])

    def test_stochastic_rounding_unbiased_in_an_interval(self):
        rng=torch.Generator().manual_seed(9)
        y=stochastic_round_interval(torch.full((50000,),.3),0.,.5,rng)
        self.assertEqual(set(y.tolist()),{0.,.5})
        self.assertLess(abs(y.mean().item()-.3),.003)
        self.assertLess(abs(y.var(correction=0).item()-.06),.001)
        with self.assertRaises(ValueError):stochastic_round_interval(torch.tensor([.6]),0.,.5)

    def test_paged_cache_failed_admission_is_atomic(self):
        cache=PagedKVCache(3,2);cache.add_request('live',[1,2])
        snapshot=copy.deepcopy(vars(cache))
        # One shared block plus three new blocks cannot fit; refcounts must not leak.
        with self.assertRaises(MemoryError):cache.add_request('failed',[1,2,3,4,5,6,7,8])
        self.assertEqual(vars(cache),snapshot)
        with self.assertRaises(ValueError):cache.add_request('live',[3,4])
        self.assertEqual(vars(cache),snapshot)
        cache.add_request('ok',[1,2,9]);cache.release('ok');cache.release('live')
        self.assertEqual(cache.used_blocks(),0)

    def test_exact_resume_checks_settings_and_dataset_content(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);train=root/'train.npy';valid=root/'valid.npy'
            np.save(train,np.arange(40)%8);np.save(valid,np.arange(12)%8)
            settings=dict(steps=3,out=root/'run',train=train,valid=valid,context=4,batch_size=2,
                          eval_windows=32,verbose=False,config=dict(vocab=8,width=8,heads=2,kv_heads=2,hidden=16,layers=1))
            run(**settings,stop_after=1)
            checkpoint=root/'run/checkpoint.pt'
            for change in (dict(lr=.01),dict(warmup=0),dict(context=3),dict(batch_size=3)):
                with self.assertRaisesRegex(ValueError,'exact resume'):run(**(settings|change),resume=checkpoint)
            data=np.load(train);data[0]=7;np.save(train,data)
            with self.assertRaisesRegex(ValueError,'train_sha256'):run(**settings,resume=checkpoint)

    def test_sft_binary_mask_dtype_does_not_change_large_denominator(self):
        logits=torch.zeros(1,70000,2)
        labels=torch.zeros(1,70000,dtype=torch.long)
        mask=torch.ones(1,70000,dtype=torch.float16)
        self.assertAlmostEqual(sft_loss(logits,labels,mask).item(),math.log(2),places=5)
        self.assertAlmostEqual(sft_loss(logits,labels,mask,'sequence').item(),math.log(2),places=5)
        with self.assertRaises(ValueError):distillation_loss(logits,logits,mask,temperature=math.nan)


if __name__=='__main__':unittest.main()
