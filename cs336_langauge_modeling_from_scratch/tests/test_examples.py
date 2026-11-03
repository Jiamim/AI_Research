"""Semantic and numerical checks for the teaching implementations."""
import math
import unittest
import torch
from torch.nn import functional as F
from examples.tokenizer import Tokenizer, merge_pair, train_bpe
from examples.model import Config, TinyLM, RMSNorm, rope
from examples.optim import AdamW
from examples.attention import online_attention, dense_attention
from examples.posttrain import advantages, clipped_policy_loss, dpo_loss, gspo_loss, masked_sequence_logps
from examples.data import exact_dedup, shingles, minhash, jaccard
from examples.extensions import TopKMoE, linear_attention_explicit, linear_attention_recurrent, clip_loss

torch.set_num_threads(2)


class TokenizerTests(unittest.TestCase):
    def test_roundtrip_unseen_unicode_and_whitespace(self):
        vocab, merges = train_bpe("banana banana<|endoftext|>你好，语言模型<|endoftext|> a\n a", 300, ["<|endoftext|>"])
        t = Tokenizer(vocab, merges, ["<|endoftext|>"])
        for s in ["", "🙂𠮷中", "\n\t ", "aaaaa", "unseen code: x += 1", "a<|endoftext|>b"]:
            self.assertEqual(t.decode(t.encode(s)), s)

    def test_nonoverlapping_merge_and_determinism(self):
        self.assertEqual(merge_pair([1]*5, (1,1), 2), [2,2,1])
        self.assertEqual(train_bpe("abac abac", 270)[1], train_bpe("abac abac", 270)[1])


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.model = TinyLM(Config(width=32, hidden=88, heads=4, kv_heads=2)).eval()
        self.ids = torch.randint(0, 256, (2, 11))

    def test_causality(self):
        changed = self.ids.clone()
        changed[:, 6:] = (changed[:, 6:] + 37) % 256
        with torch.no_grad():
            torch.testing.assert_close(self.model(self.ids)[:, :6], self.model(changed)[:, :6], rtol=0, atol=0)

    def test_cache_single_and_chunked(self):
        with torch.no_grad():
            full = self.model(self.ids)
            for chunks in [[1]*11, [4,3,4]]:
                cache, outputs, start = None, [], 0
                for size in chunks:
                    part, cache = self.model(self.ids[:, start:start+size], cache, True)
                    outputs.append(part)
                    start += size
                torch.testing.assert_close(full, torch.cat(outputs, 1), rtol=1e-5, atol=1e-6)
                self.assertEqual(cache[0][0].shape, (2,2,11,8))

    def test_parameter_accounting(self):
        c = self.model.cfg
        dkv = c.kv_heads * c.width // c.heads
        expected = 2*c.vocab*c.width + c.layers*(2*c.width**2 + 2*c.width*dkv + 3*c.width*c.hidden + 2*c.width) + c.width
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), expected)

    def test_rope_relative_position_and_norm(self):
        q, k = torch.randn(1,1,1,8, dtype=torch.float64), torch.randn(1,1,1,8,dtype=torch.float64)
        def rot(x,p): return rope(x,torch.tensor([p]))
        torch.testing.assert_close(rot(q,7).norm(), q.norm())
        torch.testing.assert_close((rot(q,7)*rot(k,11)).sum(), (q*rot(k,4)).sum())

    def test_rmsnorm_gradient(self):
        norm = RMSNorm(5).double()
        x = torch.randn(2,5,dtype=torch.float64,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(norm,(x,)))


class AttentionTests(unittest.TestCase):
    def test_values_gradients_and_irregular_tiles(self):
        torch.manual_seed(5)
        for causal in [True, False]:
            for block in [1,4,20]:
                q,k,v = [torch.randn(9,6,dtype=torch.float64,requires_grad=True) for _ in range(3)]
                a, b = online_attention(q,k,v,block,causal), dense_attention(q,k,v,causal)
                torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-10)
                upstream = torch.randn_like(a)
                ga = torch.autograd.grad((a*upstream).sum(),(q,k,v))
                gb = torch.autograd.grad((b*upstream).sum(),(q,k,v))
                for x,y in zip(ga,gb): torch.testing.assert_close(x,y,rtol=1e-9,atol=1e-9)


class OptimizationTests(unittest.TestCase):
    def test_adamw_matches_torch_multiple_steps(self):
        torch.manual_seed(7)
        a = torch.nn.Parameter(torch.randn(3,4,dtype=torch.float64))
        b = torch.nn.Parameter(a.detach().clone())
        ours = AdamW([a],lr=0.01,betas=(0.8,0.95),eps=1e-6,weight_decay=0.2)
        reference = torch.optim.AdamW([b],lr=0.01,betas=(0.8,0.95),eps=1e-6,weight_decay=0.2,foreach=False)
        for _ in range(7):
            g = torch.randn_like(a)
            a.grad, b.grad = g.clone(), g.clone()
            ours.step(); reference.step()
            torch.testing.assert_close(a,b,rtol=1e-11,atol=1e-11)

    def test_unequal_microbatch_weighting(self):
        torch.manual_seed(2)
        a = torch.nn.Linear(4,3)
        b = torch.nn.Linear(4,3); b.load_state_dict(a.state_dict())
        x,y = torch.randn(7,4),torch.randint(0,3,(7,))
        F.cross_entropy(a(x),y).backward()
        for sl in [slice(0,2),slice(2,7)]:
            (F.cross_entropy(b(x[sl]),y[sl],reduction="sum")/7).backward()
        torch.testing.assert_close(a.weight.grad,b.weight.grad)


class PosttrainTests(unittest.TestCase):
    def test_dpo_initial_loss_and_direction(self):
        chosen = torch.tensor([-4.],requires_grad=True)
        rejected = torch.tensor([-3.],requires_grad=True)
        loss = dpo_loss(chosen,rejected,chosen.detach(),rejected.detach())
        self.assertAlmostEqual(loss.item(),math.log(2),places=6)
        loss.backward()
        self.assertLess(chosen.grad.item(),0)
        self.assertGreater(rejected.grad.item(),0)

    def test_constant_groups_zero_and_leave_one_out(self):
        for method in ["grpo","dr_grpo","maxrl","rloo"]:
            a=advantages(torch.tensor([[0.,0.,0.],[1.,1.,1.]]),method)
            self.assertTrue(torch.isfinite(a).all())
            torch.testing.assert_close(a,torch.zeros_like(a))
        torch.testing.assert_close(advantages(torch.tensor([[0.,1.]]),"rloo"),torch.tensor([[-1.,1.]]))

    def test_response_mask_padding_and_gradient(self):
        z = torch.zeros(1,3,4,requires_grad=True)
        mask = torch.tensor([[1.,1.,0.]])
        val=masked_sequence_logps(z,torch.tensor([[1,2,-100]]),mask)
        self.assertAlmostEqual(val.item(),-2*math.log(4),places=6)
        val.sum().backward()
        torch.testing.assert_close(z.grad[:,2],torch.zeros_like(z.grad[:,2]))

    def test_clipping_sign_and_mask(self):
        logp = torch.log(torch.tensor([[1.5],[0.5]],requires_grad=False)).requires_grad_()
        loss=clipped_policy_loss(logp,torch.zeros_like(logp),torch.tensor([1.,-1.]),torch.ones_like(logp))
        loss.backward()
        torch.testing.assert_close(logp.grad,torch.zeros_like(logp))
        self.assertAlmostEqual(loss.item(),-0.2,places=6)

    def test_gspo_identical_policy_gradient_is_length_normalized(self):
        logp=torch.zeros(1,3,requires_grad=True)
        mask=torch.tensor([[1.,1.,0.]])
        loss=gspo_loss(logp,logp.detach(),torch.tensor([2.]),mask)
        loss.backward()
        torch.testing.assert_close(logp.grad,torch.tensor([[-1.,-1.,0.]]))


class DataTests(unittest.TestCase):
    def test_dedup_and_minhash_identity(self):
        self.assertEqual(len(exact_dedup(["a  b","a b","c"])),2)
        a=shingles("the same document")
        self.assertEqual(minhash(a),minhash(a))
        self.assertEqual(jaccard(a,a),1)


class ExtensionTests(unittest.TestCase):
    def test_linear_attention_recurrence_and_gradients(self):
        q,k,v=[torch.randn(7,4,dtype=torch.float64,requires_grad=True) for _ in range(3)]
        a,b=linear_attention_recurrent(q,k,v),linear_attention_explicit(q,k,v)
        torch.testing.assert_close(a,b)
        for ga,gb in zip(torch.autograd.grad(a.square().sum(),(q,k,v)),torch.autograd.grad(b.square().sum(),(q,k,v))):
            torch.testing.assert_close(ga,gb)

    def test_top1_router_has_task_gradient(self):
        torch.manual_seed(13)
        model=TopKMoE(topk=1)
        y,balance,load=model(torch.randn(10,16))
        y.square().sum().backward()
        self.assertGreater(model.router.weight.grad.norm().item(),0)
        self.assertAlmostEqual(load.sum().item(),1,places=6)

    def test_clip_alignment_and_pair_permutation_invariance(self):
        x=torch.eye(5)
        order=torch.tensor([2,4,0,1,3])
        self.assertLess(clip_loss(x,x),clip_loss(x,x.roll(1,0)))
        torch.testing.assert_close(clip_loss(x,x),clip_loss(x[order],x[order]))


if __name__ == "__main__":
    unittest.main()
