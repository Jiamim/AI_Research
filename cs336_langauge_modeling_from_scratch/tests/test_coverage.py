"""Semantic checks for the missing-topic supplements."""
import math
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from examples.tokenizer import unigram_segment
from examples.data import mmap_next_token_batch
from examples.posttrain import simpo_loss,token_entropy
from examples.systems import rmsnorm_custom,tensor_parallel_demo


class CoverageTests(unittest.TestCase):
    def test_entropy_full_distribution_and_padding(self):
        logits=torch.zeros(1,3,4,dtype=torch.float64)
        logits[0,1]=torch.tensor([40.,-40.,-40.,-40.]);logits[0,2]=float('nan')
        entropy=token_entropy(logits,torch.tensor([[1,1,0]]))
        self.assertAlmostEqual(entropy[0,0].item(),math.log(4),places=12)
        self.assertLess(entropy[0,1].item(),1e-25)
        self.assertEqual(entropy[0,2].item(),0)
    def test_unigram_best_path_and_marginal_are_distinct(self):
        result=unigram_segment('ab',{'a':.4,'b':.4,'ab':.2})
        self.assertEqual(result['pieces'],['ab'])
        self.assertAlmostEqual(math.exp(result['best_log_probability']),.2)
        self.assertAlmostEqual(math.exp(result['log_probability']),.36)

    def test_unigram_overlapping_paths_and_unencodable_text(self):
        result=unigram_segment('aaa',{'a':.6,'aa':.4})
        self.assertAlmostEqual(math.exp(result['log_probability']),.6**3+2*.6*.4)
        self.assertEqual(unigram_segment('',{'a':1.})['log_probability'],0)
        with self.assertRaises(ValueError):unigram_segment('b',{'a':1.})

    def test_mmap_shift_large_token_ids_and_boundary(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'tokens.npy'
            data=np.array([70000,1,2,3,4,5],dtype=np.uint32);np.save(path,data)
            x,y=mmap_next_token_batch(path,[0,2],3)
            torch.testing.assert_close(x,torch.tensor([[70000,1,2],[2,3,4]]))
            torch.testing.assert_close(y,torch.tensor([[1,2,3],[3,4,5]]))
            x[0,0]=0
            self.assertEqual(np.load(path)[0],70000)
            with self.assertRaises(ValueError):mmap_next_token_batch(path,[3],3)

    def test_custom_rmsnorm_forward_and_gradient(self):
        torch.manual_seed(37)
        x=torch.randn(2,3,5,dtype=torch.float64,requires_grad=True)
        g=torch.randn(5,dtype=torch.float64,requires_grad=True)
        expected=x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-6)*g
        actual=rmsnorm_custom(x,g)
        torch.testing.assert_close(actual,expected)
        self.assertTrue(torch.autograd.gradcheck(rmsnorm_custom,(x,g)))

    def test_tensor_parallel_backward_requires_sum_of_contributions(self):
        for parts in [1,2,3]:
            result=tensor_parallel_demo(parts)
            self.assertLess(result['output_max_error'],1e-12)
            self.assertLess(result['input_gradient_max_error'],1e-12)

    def test_simpo_length_normalization_and_margin(self):
        chosen=torch.tensor([-4.],requires_grad=True)
        rejected=torch.tensor([-2.],requires_grad=True)
        loss=simpo_loss(chosen,rejected,torch.tensor([4.]),torch.tensor([2.]),beta=2,margin=1)
        self.assertAlmostEqual(loss.item(),math.log1p(math.exp(1)),places=6)
        loss.backward()
        self.assertLess(chosen.grad.item(),0)
        self.assertGreater(rejected.grad.item(),0)
        with self.assertRaises(ValueError):simpo_loss(chosen,rejected,torch.zeros(1),torch.ones(1))


if __name__=='__main__':unittest.main()
