"""Independent boundary checks for probability mass and experiment presentation."""
import math
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from examples.inference import speculative_step
from examples.train import save_checkpoint
from scripts import run_tinystories


class ProbabilityMassTests(unittest.TestCase):
    def test_wide_vocabulary_does_not_make_probability_checks_meaningless(self):
        for dtype, vocab, mass in ((torch.bfloat16,512,.5),(torch.float16,4096,2.),
                                    (torch.float32,100000,1.01)):
            valid = torch.full((2,vocab),1/vocab,dtype=dtype)
            invalid = torch.full((1,vocab),mass/vocab,dtype=dtype)
            with self.subTest(dtype=dtype), self.assertRaisesRegex(ValueError,'sum to one'):
                speculative_step(valid,invalid,torch.tensor([0]))
            with self.subTest(dtype=dtype,role='target'), self.assertRaisesRegex(ValueError,'sum to one'):
                speculative_step(invalid.expand(2,-1),valid[:1],torch.tensor([0]))

    def test_quantized_normalized_distributions_are_still_accepted(self):
        generator=torch.Generator().manual_seed(19)
        probabilities=torch.randn(2,4096,dtype=torch.float64,generator=generator).softmax(-1)
        for dtype in (torch.bfloat16,torch.float16,torch.float32,torch.float64):
            with self.subTest(dtype=dtype):
                target=probabilities.to(dtype)
                tokens=speculative_step(target,target[:1],target[:1].argmax(-1),generator)
                self.assertEqual(len(tokens),2)

    def test_fp16_subnormal_probabilities_keep_absolute_rounding_allowance(self):
        probabilities=torch.full((2,100000),1/100000,dtype=torch.float16)
        self.assertGreater(abs(probabilities.double().sum(-1)[0].item()-1),torch.finfo(torch.float16).eps)
        result=speculative_step(probabilities,probabilities[:1],torch.tensor([0]))
        self.assertEqual(len(result),2)


class CheckpointPermissionTests(unittest.TestCase):
    @unittest.skipIf(os.name=='nt','POSIX file permissions')
    def test_save_never_changes_process_umask_and_preserves_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            ordinary=root/'ordinary'
            ordinary.write_bytes(b'')
            target=root/'checkpoint.pt'
            with mock.patch('examples.train.os.umask',side_effect=AssertionError('must not change process umask')):
                save_checkpoint({'step':1},target)
                self.assertEqual(target.stat().st_mode & 0o777,ordinary.stat().st_mode & 0o777)
                target.chmod(0o640)
                save_checkpoint({'step':2},target)
                self.assertEqual(target.stat().st_mode & 0o777,0o640)
            self.assertEqual(torch.load(target,weights_only=True)['step'],2)
            self.assertFalse(list(root.glob('.checkpoint-*')))


class TinyStoriesPresentationTests(unittest.TestCase):
    def test_failed_baseline_cannot_be_used_for_generation_or_seed_summary(self):
        for loss in (None,math.nan,math.inf):
            with self.subTest(loss=loss),self.assertRaisesRegex(RuntimeError,'baseline'):
                run_tinystories.require_baseline([dict(name='base',diverged=False,history=[dict(val_loss=loss)])],'base')
        with self.assertRaises(RuntimeError):
            run_tinystories.require_baseline([dict(name='base',diverged=True,history=[dict(val_loss=1.)])],'base')
        with self.assertRaises(RuntimeError):
            run_tinystories.require_baseline([dict(name='base',diverged=False,steps=1,total_steps=2,history=[dict(val_loss=1.)])],'base')

    def test_explicit_failed_baseline_stops_before_loading_weights_or_replacing_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            data=root/'outputs/tinystories'
            data.mkdir(parents=True)
            (root/'book').mkdir()
            summary=root/'book/tinystories-results.json'
            summary.write_text('{"previous": true}',encoding='utf-8')
            for name in ('train.npy','valid.npy','tokenizer.json'):
                (data/name).write_bytes(b'fixture')
            pool=mock.MagicMock()
            pool.__enter__.return_value=pool
            pool.map.side_effect=lambda function,jobs:[dict(name=name,diverged=name=='base',
                history=[dict(val_loss=None if name=='base' else 3.)]) for name,settings in jobs]
            with mock.patch.object(run_tinystories,'ROOT',root), \
                 mock.patch.object(run_tinystories.sys,'argv',['tinystories','--best-lr','0.01']), \
                 mock.patch.object(run_tinystories,'prepare_data',return_value={}), \
                 mock.patch.object(run_tinystories,'ProcessPoolExecutor',return_value=pool), \
                 mock.patch.object(run_tinystories.torch,'load') as load,self.assertRaisesRegex(RuntimeError,'baseline'):
                run_tinystories.main()
            self.assertFalse(load.called,'failed baseline must stop before reading weights')
            self.assertEqual(json.loads(summary.read_text(encoding='utf-8')),{'previous':True})

    def test_negative_log_probabilities_can_overflow_margin_without_positive_inputs(self):
        from examples.posttrain import dpo_loss
        chosen=torch.tensor([-40000.],dtype=torch.float16,requires_grad=True)
        rejected=torch.tensor([-1.],dtype=torch.float16,requires_grad=True)
        ref_chosen=torch.tensor([-1.],dtype=torch.float16)
        ref_rejected=torch.tensor([-40000.],dtype=torch.float16)
        naive=(chosen.detach()-ref_chosen)-(rejected.detach()-ref_rejected)
        self.assertTrue(torch.isneginf(naive).all())
        loss=dpo_loss(chosen,rejected,ref_chosen,ref_rejected,beta=.1)
        expected=torch.nn.functional.softplus(torch.tensor(7999.8))
        torch.testing.assert_close(loss,expected)

    def test_graphs_use_recorded_baseline_and_real_failure_status(self):
        from scripts import make_figures
        data=dict(base_run='base',device='cpu',train_settings={'steps':12},runs=[
            dict(name='base',lr=.01,parameters={'total':1000000},diverged=False,
                 history=[dict(step=0,val_loss=9.),dict(step=6,val_loss=6.),dict(step=12,val_loss=5.)]),
            dict(name='no_norm',lr=.01,diverged=False,history=[dict(step=12,val_loss=8.)])])
        observed={}
        def capture(figure,name):
            axis=figure.axes[0]
            observed[name]=dict(title=axis.get_title(loc='left'),labels=axis.get_legend_handles_labels()[1],
                                ys=[line.get_ydata().tolist() for line in axis.lines],ylim=axis.get_ylim())
            make_figures.plt.close(figure)
        with mock.patch.object(make_figures,'save',side_effect=capture):
            make_figures.plot_tinystories(data)
        sweep=observed['tinystories-lr.svg']
        self.assertIn('scan skipped',sweep['title'])
        self.assertIn('12 steps',sweep['title'])
        self.assertEqual(sweep['ys'],[[6.,5.]])
        ablations=observed['tinystories-ablations.svg']
        self.assertEqual(ablations['ys'],[[6.,5.],[8.]])
        self.assertTrue(any('lr 0.01' in label for label in ablations['labels']))
        self.assertFalse(any('failed' in label or 'diverged' in label for label in ablations['labels']))
        self.assertGreater(ablations['ylim'][1],8.)


if __name__=='__main__':unittest.main()
