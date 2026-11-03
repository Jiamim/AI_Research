"""Cross-module invariants: reject changed objectives, preserve state and validate reuse."""
import copy
import io
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from examples.accounting import Shape, forward_flops
from examples.architectures import kv_bytes_per_token
from examples.evaluation import elo_win_probability, wilson_interval
from examples.inference import speculative_step
from examples.long_context import yarn_attention_factor
from examples.model import Config, TinyLM
from examples.numerical_tools import bootstrap_power_prediction
from examples.optim import AdamW
from examples.posttrain import dpo_loss, simpo_loss
from examples.reasoning import iid_pass_probability
from examples.research import reweight_domains
from examples.rl_objectives import cispo_loss, dapo_loss
from examples.train import save_checkpoint, windows
from examples.train import run as train_run
from scripts import artifact_io, rebuild, run_tinystories, verify_sources


class ObjectiveContracts(unittest.TestCase):
    def test_preference_pairs_cannot_broadcast_or_be_empty(self):
        for shape in ((2, 1), (1,), (0,)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                dpo_loss(torch.zeros(2), torch.zeros(shape), torch.zeros(2), torch.zeros(2))
        with self.assertRaises(ValueError):
            dpo_loss(torch.zeros(2), torch.zeros(2), torch.zeros(2), torch.zeros(2), beta=math.nan)
        with self.assertRaises(ValueError):
            simpo_loss(torch.zeros(2), torch.zeros(2), torch.ones(2), torch.full((2,), math.inf))

    def test_low_precision_preferences_subtract_in_fp32(self):
        chosen = torch.tensor([-60000.], dtype=torch.float16, requires_grad=True)
        rejected = torch.tensor([60000.], dtype=torch.float16, requires_grad=True)
        result = dpo_loss(chosen, rejected, torch.zeros(1), torch.zeros(1))
        expected = torch.nn.functional.softplus(torch.tensor(12000.))
        torch.testing.assert_close(result, expected)
        result.backward()
        self.assertTrue(torch.isfinite(chosen.grad).all())

    def test_rl_clipping_parameters_cannot_reverse_gradient(self):
        args = (torch.tensor([[-.2]]), torch.tensor([[-.2]]), torch.ones(1), torch.ones(1, 1))
        for function in (cispo_loss, dapo_loss):
            for high in (-2., math.nan, math.inf):
                with self.subTest(function=function, high=high), self.assertRaises(ValueError):
                    function(*args, eps_high=high)


class StateContracts(unittest.TestCase):
    @unittest.skipIf(os.name == 'nt', 'POSIX permission preservation')
    def test_atomic_write_preserves_existing_access_permissions(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'report.json'
            path.write_text('{}',encoding='utf-8')
            path.chmod(0o640)
            artifact_io.write_json(path,{'ok':True})
            self.assertEqual(path.stat().st_mode & 0o777,0o640)
            new=Path(directory)/'new.json'
            artifact_io.write_json(new,{'ok':True})
            self.assertEqual(new.stat().st_mode & 0o777,0o644)

    def test_invalid_optimizer_group_fails_before_any_update(self):
        first = torch.nn.Parameter(torch.ones(2))
        second = torch.nn.Parameter(torch.ones(2))
        first.grad = torch.ones_like(first)
        second.grad = torch.ones_like(second)
        optimizer = AdamW([{'params': [first]}, {'params': [second]}])
        optimizer.param_groups[1]['lr'] = math.nan
        with self.assertRaises(ValueError):
            optimizer.step()
        self.assertTrue(torch.equal(first, torch.ones_like(first)))
        self.assertEqual(len(optimizer.state), 0)

    def test_sparse_gradient_rejection_does_not_partially_step(self):
        first=torch.nn.Parameter(torch.ones(2))
        second=torch.nn.Parameter(torch.ones(2))
        first.grad=torch.ones_like(first)
        second.grad=torch.sparse_coo_tensor(torch.tensor([[0]]),torch.tensor([1.]),(2,))
        optimizer=AdamW([first,second])
        with self.assertRaisesRegex(ValueError,'dense'):
            optimizer.step()
        self.assertTrue(torch.equal(first,torch.ones_like(first)))
        self.assertEqual(len(optimizer.state),0)

    def test_model_rejects_inconsistent_layer_caches(self):
        with self.assertRaises(ValueError):
            Config(rope_base=math.nan)
        with self.assertRaises(ValueError):
            Config(width=64.0)
        model = TinyLM(Config())
        ids = torch.ones(1, 2, dtype=torch.long)
        _, cache = model(ids, use_cache=True)
        cache[1] = tuple(value[:, :, :1] for value in cache[1])
        with self.assertRaisesRegex(ValueError, 'prefix length'):
            model(ids[:, :1], caches=cache)

    def test_windows_reject_short_or_negative_ranges(self):
        for data in (torch.arange(6), np.arange(6)):
            for starts in ([-1], [4], []):
                with self.subTest(starts=starts), self.assertRaises(ValueError):
                    windows(data, starts, 3)

    def test_checkpoint_write_failure_preserves_previous_resume_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pt'
            path.write_bytes(b'previous checkpoint')
            def fail(value, stream):
                stream.write(b'partial checkpoint')
                raise OSError('disk full')
            with mock.patch('examples.train.torch.save', side_effect=fail), self.assertRaises(OSError):
                save_checkpoint({'model': 1}, path)
            self.assertEqual(path.read_bytes(), b'previous checkpoint')
            self.assertEqual(list(path.parent.glob('.checkpoint-*')), [])

    def test_divergence_metrics_are_valid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch('examples.train.F.cross_entropy', return_value=torch.tensor(float('nan'))):
                result=train_run(steps=2,out=directory,verbose=False)
            self.assertTrue(result['diverged'])
            text=(Path(directory)/'metrics.json').read_text(encoding='utf-8')
            json.loads(text,parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
            self.assertIsNone(result['history'][-1]['val_loss'])
            self.assertTrue(result['history'][-1]['nonfinite_validation_loss'])
            self.assertTrue((Path(directory)/'checkpoint.pt').is_file())

    def test_publication_failure_rolls_back_earlier_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('one', 'two'):
                (root / name).write_bytes(('old ' + name).encode())
                (root / (name + '.new')).write_bytes(('new ' + name).encode())
            original = Path.replace
            def fail_second(path, target):
                if Path(target) == root / 'two' and path.name.startswith('.write-'):
                    raise OSError('simulated replacement failure')
                return original(path, target)
            with mock.patch.object(Path, 'replace', fail_second), self.assertRaises(OSError):
                artifact_io.publish_files([(root / (name + '.new'), root / name) for name in ('one', 'two')])
            for name in ('one', 'two'):
                self.assertEqual((root / name).read_bytes(), ('old ' + name).encode())
            self.assertEqual(list(root.glob('.previous-*')), [])

    def test_zero_new_outputs_are_left_after_failed_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'one.new').write_bytes(b'one')
            (root/'two.new').write_bytes(b'two')
            original=Path.replace
            def fail_second(path,target):
                if Path(target)==root/'two' and path.name.startswith('.write-'):
                    raise OSError('simulated failure')
                return original(path,target)
            with mock.patch.object(Path,'replace',fail_second),self.assertRaises(OSError):
                artifact_io.publish_files([(root/'one.new',root/'one'),(root/'two.new',root/'two')])
            self.assertFalse((root/'one').exists())
            self.assertFalse((root/'two').exists())


class StatisticalContracts(unittest.TestCase):
    def test_extreme_domain_weights_preserve_probabilities(self):
        np.testing.assert_allclose(reweight_domains([1e308, 1e308], [0, 0], eta=0), [.5, .5])
        np.testing.assert_allclose(reweight_domains([1e-300, 2e-300], [0, 0], floor_mass=0), [1/3, 2/3])
        self.assertGreater(iid_pass_probability(1e-20, 10), 0)
        self.assertAlmostEqual(iid_pass_probability(1e-20, 10) / 1e-19, 1)

    def test_statistical_inputs_define_an_identifiable_problem(self):
        with self.assertRaisesRegex(ValueError, 'distinct'):
            bootstrap_power_prediction([1, 1, 1, 1], [1, 2, 3, 4], 2, repeats=10)
        with self.assertRaises(ValueError):
            wilson_interval(2.5, 5)
        self.assertEqual(elo_win_probability(-1e6, 1e6), 0)
        self.assertEqual(elo_win_probability(1e6, -1e6), 1)

    def test_resource_defaults_do_not_consume_explicit_zero(self):
        shape = Shape(10, 8, 1, 4, 2, 8)
        self.assertEqual(forward_flops(shape, tokens=0)['total'], 0)
        with self.assertRaises(ValueError):
            Shape(10, 8, 1, 4, 2, 8, kv_heads=0)
        with self.assertRaises(ValueError):
            kv_bytes_per_token('openai/gpt-oss-20b', context=0)
        self.assertEqual(yarn_attention_factor(32, 0., 0.), 1.)

    def test_draft_tokens_must_be_possible_under_draft_sampler(self):
        p = torch.tensor([[.5, .5], [.5, .5]])
        with self.assertRaisesRegex(ValueError, 'positive probability'):
            speculative_step(p, torch.tensor([[1., 0.]]), torch.tensor([1]))
        with self.assertRaisesRegex(ValueError, 'sum to one'):
            speculative_step(p, torch.tensor([[.5, .7]]), torch.tensor([0]))


class ExperimentReuseContracts(unittest.TestCase):
    def test_cached_run_requires_matching_config_data_and_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'train.npy').write_bytes(b'version-a')
            run_tinystories.init_runs(root, 1, {'train.npy': artifact_io.file_sha256(root/'train.npy')})
            calls = []
            def train(**settings):
                calls.append(settings)
                out = Path(settings['out'])
                out.mkdir(parents=True, exist_ok=True)
                result = {'history': [{'val_loss': 1.0}], 'diverged': False}
                (out / 'metrics.json').write_text(json.dumps(result), encoding='utf-8')
                (out / 'checkpoint.pt').write_bytes(b'complete checkpoint')
                return result
            overrides = {'config': {'layers': 2}, 'lr': .001}
            with mock.patch.object(run_tinystories, 'run', side_effect=train):
                run_tinystories._run(('demo', overrides))
                self.assertEqual(overrides, {'config': {'layers': 2}, 'lr': .001})
                run_tinystories._run(('demo', overrides))
                self.assertEqual(len(calls), 1)
                run_tinystories._run(('demo', overrides | {'lr': .002}))
                self.assertEqual(len(calls), 2)
                (root / 'runs/demo/checkpoint.pt').write_bytes(b'corrupt')
                run_tinystories._run(('demo', overrides | {'lr': .002}))
                self.assertEqual(len(calls), 3)

    def test_download_honors_limit_when_server_ignores_range(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'train.txt'
            with mock.patch.object(run_tinystories, 'urlopen', return_value=io.BytesIO(b'0123456789')):
                run_tinystories.download('train', path, max_bytes=4)
            self.assertEqual(path.read_bytes(), b'0123')

    def test_source_hash_failure_leaves_no_download_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            document={'id':'fixture','download_url':'https://example.test/source.py','sha256':'0'*64}
            with mock.patch.object(verify_sources,'ROOT',root), \
                 mock.patch.object(verify_sources,'urlopen',return_value=io.BytesIO(b'wrong contents')):
                error=verify_sources.check_source(document,download=True)
            self.assertIn('hash mismatch',error)
            self.assertFalse((root/'sources/cache/fixture.py').exists())

    def test_spawn_worker_receives_runtime_settings(self):
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            np.save(root/'train.npy',np.arange(40,dtype=np.uint16)%8)
            np.save(root/'valid.npy',np.arange(20,dtype=np.uint16)%8)
            hashes = {name:artifact_io.file_sha256(root/name) for name in ('train.npy', 'valid.npy')}
            overrides=dict(steps=3,context=4,batch_size=2,eval_every=1,eval_windows=2,warmup=1,lr=.001,
                           config=dict(vocab=8,width=8,layers=1,heads=2,kv_heads=2,hidden=16))
            with ProcessPoolExecutor(1,mp_context=mp.get_context('spawn'),initializer=run_tinystories.init_runs,
                                     initargs=(directory,3,hashes)) as pool:
                result=pool.submit(run_tinystories.run_contract,{'vocab':10},{'steps':2}).result(timeout=60)
                trained=pool.submit(run_tinystories._run,('small',overrides)).result(timeout=60)
                cached=pool.submit(run_tinystories._run,('small',overrides)).result(timeout=60)
            self.assertEqual(trained,cached)
            self.assertEqual(trained['steps'],3)
            self.assertFalse(trained['diverged'])
            self.assertTrue((root/'runs/small/cache-contract.json').is_file())
        self.assertEqual(result['threads'],3)
        self.assertEqual(result['data'],hashes)


class RebuildAcceptanceContracts(unittest.TestCase):
    def test_no_pdf_run_remains_explicitly_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'book').mkdir()
            artifact_io.write_json(root/'book/validation.json',dict(tests_run=1,failures=0,errors=0,skipped=0,success=True,source_sha256={},runtime=artifact_io.runtime_snapshot()))
            completed=mock.Mock(returncode=0)
            with mock.patch.object(rebuild,'ROOT',root), \
                 mock.patch.object(rebuild.sys,'argv',['rebuild','--skip-checks','--without-pdf']), \
                 mock.patch.object(rebuild.subprocess,'run',return_value=completed),self.assertRaises(SystemExit) as error:
                rebuild.main()
            self.assertEqual(error.exception.code,1)
            report=json.loads((root/'book/rebuild-check.json').read_text(encoding='utf-8'))
            self.assertFalse(report['complete'])
            self.assertTrue(report['text_ready'])
            self.assertEqual(report['stages'][-1]['status'],'not_run')

    def test_failed_environment_check_does_not_regenerate_the_book(self):
        # Unit tests passed but the two-process Gloo check failed (a socket-less sandbox): chapter 35 would
        # embed "check failed", so nothing may be regenerated unless the user explicitly allows it.
        record=dict(tests_run=5,failures=0,errors=0,skipped=0,success=False,source_sha256={},
                    runtime=artifact_io.runtime_snapshot(),distributed=dict(success=False,stderr='Operation not permitted'))
        for allow,expect_build in ((False,False),(True,True)):
            with self.subTest(allow=allow), tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                (root/'book').mkdir()
                artifact_io.write_json(root/'book/validation.json',record)
                argv=['rebuild','--skip-checks','--without-pdf']+(['--allow-failed-environment-checks'] if allow else [])
                with mock.patch.object(rebuild,'ROOT',root), mock.patch.object(rebuild,'validation_snapshot',return_value={}), \
                     mock.patch.object(rebuild.sys,'argv',argv), \
                     mock.patch.object(rebuild.subprocess,'run',return_value=mock.Mock(returncode=0)) as launched, \
                     self.assertRaises(SystemExit):
                    rebuild.main()
                self.assertEqual(launched.called,expect_build)
                report=json.loads((root/'book/rebuild-check.json').read_text(encoding='utf-8'))
                self.assertFalse(report['complete'])
                if not allow:
                    self.assertEqual(report['stages'][-1]['failed'],['distributed'])

    def test_stale_validation_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'book').mkdir()
            # Current runtime, stale sources: only the source-hash check can reject this record.
            artifact_io.write_json(root/'book/validation.json',dict(tests_run=1,failures=0,errors=0,skipped=0,success=True,
                                                                     source_sha256={'old.py':'old'},runtime=artifact_io.runtime_snapshot()))
            with mock.patch.object(rebuild,'ROOT',root), \
                 mock.patch.object(rebuild.sys,'argv',['rebuild','--skip-checks']), \
                 mock.patch.object(rebuild.subprocess,'run') as run,self.assertRaises(SystemExit):
                rebuild.main()
            run.assert_not_called()
