"""Regression tests for cache identity, failed evaluations and reused validation."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from examples.numerical_tools import mismatch_stats
from examples.train import run
from scripts import artifact_io, rebuild, run_checks, run_tinystories


class DownloadIdentityTests(unittest.TestCase):
    def test_changed_limit_or_split_fetches_again_but_same_request_reuses(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'text.txt'
            with mock.patch.object(run_tinystories, 'urlopen',
                                   side_effect=lambda *a, **k: io.BytesIO(b'0123456789')) as fetch:
                run_tinystories.download('train', path, max_bytes=4)
                run_tinystories.download('train', path, max_bytes=8)
                self.assertEqual(path.read_bytes(), b'01234567')
                self.assertEqual(fetch.call_count, 2)
                run_tinystories.download('train', path, max_bytes=8)
                self.assertEqual(fetch.call_count, 2)
                run_tinystories.download('valid', path, max_bytes=8)
                self.assertEqual(fetch.call_count, 3)

    def test_corrupt_download_is_not_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'text.txt'
            with mock.patch.object(run_tinystories, 'urlopen',
                                   side_effect=lambda *a, **k: io.BytesIO(b'0123456789')) as fetch:
                run_tinystories.download('train', path, max_bytes=4)
                path.write_bytes(b'xxxx')
                run_tinystories.download('train', path, max_bytes=4)
                self.assertEqual(path.read_bytes(), b'0123')
                self.assertEqual(fetch.call_count, 2)

    def test_truncated_replacement_preserves_download_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'text.txt'
            with mock.patch.object(run_tinystories, 'urlopen', return_value=io.BytesIO(b'0123456789')):
                run_tinystories.download('train', path, max_bytes=4)
            before = {p.name:p.read_bytes() for p in root.iterdir()}
            truncated = io.BytesIO(b'123')
            truncated.headers = {'Content-Length':'10'}
            with mock.patch.object(run_tinystories, 'urlopen', return_value=truncated), self.assertRaises(OSError):
                run_tinystories.download('train', path, max_bytes=8)
            self.assertEqual({p.name:p.read_bytes() for p in root.iterdir()}, before)

    def test_untracked_existing_text_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'text.txt'
            path.write_bytes(b'user-supplied corpus')
            with mock.patch.object(run_tinystories, 'urlopen') as fetch, self.assertRaises(ValueError):
                run_tinystories.download('train', path)
            fetch.assert_not_called()
            self.assertEqual(path.read_bytes(), b'user-supplied corpus')

    def test_incomplete_partial_range_is_not_a_complete_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'text.txt'
            response = io.BytesIO(b'1234')
            response.status = 206
            response.headers = {'Content-Length':'4', 'Content-Range':'bytes 0-3/100'}
            with mock.patch.object(run_tinystories, 'urlopen', return_value=response), self.assertRaises(OSError):
                run_tinystories.download('train', path, max_bytes=8)
            self.assertFalse(path.exists())


class DataPublicationTests(unittest.TestCase):
    def test_real_preprocessing_reuses_only_unchanged_source_text(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            train, valid = root/'train.txt', root/'valid.txt'
            train.write_text('alpha beta<|endoftext|>beta alpha', encoding='utf-8')
            valid.write_text('beta alpha', encoding='utf-8')
            out=root/'tokens'
            with mock.patch.object(run_tinystories, 'BASE', {'vocab':270}):
                first=run_tinystories.prepare_data(train, valid, out, workers=1)
                with mock.patch.object(run_tinystories, 'encode_file') as encode:
                    second=run_tinystories.prepare_data(train, valid, out, workers=1)
                self.assertFalse(encode.called, 'unchanged inputs should reuse prepared token arrays')
                self.assertEqual(first, second)
                train.write_text('gamma beta<|endoftext|>beta gamma', encoding='utf-8')
                third=run_tinystories.prepare_data(train, valid, out, workers=1)
            self.assertNotEqual(third['contract']['train_sha256'], first['contract']['train_sha256'])
            self.assertEqual(third['train_tokens'], np.load(out/'train.npy').size)
            self.assertTrue(all(artifact_io.file_sha256(out/name)==digest for name,digest in third['artifact_sha256'].items()))

    def test_metadata_replacement_failure_rolls_back_tokenizer_and_token_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            train, valid = root/'train.txt', root/'valid.txt'
            train.write_text('a b a', encoding='utf-8')
            valid.write_text('b a', encoding='utf-8')
            out=root/'tokens'
            out.mkdir()
            originals = {'train.npy':b'old train', 'valid.npy':b'old valid',
                         'tokenizer.json':b'old tokenizer', 'data.json':b'{}'}
            for name, contents in originals.items():
                (out/name).write_bytes(contents)
            def encode(path, vocab, merges, target, workers):
                np.save(target, np.arange(4, dtype=np.uint16))
                return 4
            original_replace = Path.replace
            def fail_metadata(source, destination):
                if Path(destination)==out/'data.json' and source.name.startswith('.write-'):
                    raise OSError('simulated metadata write failure')
                return original_replace(source, destination)
            with mock.patch.object(run_tinystories, 'encode_file', side_effect=encode), \
                 mock.patch.object(Path, 'replace', fail_metadata), self.assertRaises(OSError):
                run_tinystories.prepare_data(train, valid, out, workers=1)
            self.assertTrue(all((out/name).read_bytes()==contents for name, contents in originals.items()))
            self.assertEqual({p.name for p in out.iterdir()}, set(originals))

    def test_input_change_during_encoding_cannot_publish_a_false_data_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            train, valid = root/'train.txt', root/'valid.txt'
            train.write_text('a b a', encoding='utf-8')
            valid.write_text('b a', encoding='utf-8')
            out=root/'tokens'
            def encode(path, vocab, merges, target, workers):
                np.save(target, np.arange(4, dtype=np.uint16))
                if path==valid:
                    train.write_text('changed while encoding', encoding='utf-8')
                return 4
            with mock.patch.object(run_tinystories, 'encode_file', side_effect=encode), \
                 self.assertRaisesRegex(RuntimeError, 'changed during preprocessing'):
                run_tinystories.prepare_data(train, valid, out, workers=1)
            self.assertEqual(list(out.iterdir()), [])

    def test_failed_training_retains_the_previous_cached_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'train.npy').write_bytes(b'train')
            run_tinystories.init_runs(root,1,{'train.npy':artifact_io.file_sha256(root/'train.npy')})
            cached=root/'runs/trial'
            cached.mkdir(parents=True)
            originals = {'checkpoint.pt':b'old checkpoint','metrics.json':b'old metrics','cache-contract.json':b'{}'}
            for name, contents in originals.items():
                (cached/name).write_bytes(contents)
            def fail(**settings):
                staging=Path(settings['out'])
                (staging/'checkpoint.pt').write_bytes(b'new checkpoint')
                raise OSError('metrics generation failed')
            with mock.patch.object(run_tinystories, 'run', side_effect=fail), self.assertRaises(OSError):
                run_tinystories._run(('trial', {'lr':.001}))
            self.assertTrue(all((cached/name).read_bytes()==contents for name, contents in originals.items()))
            self.assertEqual({p.name for p in cached.iterdir()}, set(originals))

    def test_changed_data_is_rejected_before_using_a_run_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            path=root/'train.npy'
            path.write_bytes(b'old data')
            run_tinystories.init_runs(root,1,{'train.npy':artifact_io.file_sha256(path)})
            path.write_bytes(b'changed data')
            with mock.patch.object(run_tinystories,'run') as train,self.assertRaisesRegex(ValueError,'changed after worker'):
                run_tinystories._run(('trial',{'lr':.001}))
            self.assertFalse(train.called,'stale token identity must be rejected before training')

    def test_publishing_overlapping_source_and_destination_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory)/'a', Path(directory)/'b'
            first.write_bytes(b'a')
            second.write_bytes(b'b')
            with self.assertRaisesRegex(ValueError, 'staged separately'):
                artifact_io.publish_files([(first,second),(second,first)])
            self.assertEqual(first.read_bytes(),b'a')
            self.assertEqual(second.read_bytes(),b'b')


class ValidationStateTests(unittest.TestCase):
    def test_nonfinite_initial_validation_stops_before_optimizer_update(self):
        original = torch.nn.functional.cross_entropy
        def losses(*args, **kwargs):
            return torch.tensor(float('nan')) if kwargs.get('reduction') == 'sum' else original(*args, **kwargs)
        with tempfile.TemporaryDirectory() as directory, mock.patch('examples.train.F.cross_entropy', side_effect=losses):
            result = run(steps=2, out=directory, verbose=False)
            self.assertTrue(result['diverged'])
            self.assertEqual(result['steps'], 0)
            checkpoint = torch.load(Path(directory)/'checkpoint.pt', weights_only=True)
            self.assertTrue(checkpoint['diverged'])
            self.assertEqual(len(checkpoint['optimizer']['state']), 0)

    def test_final_nonfinite_validation_marks_completed_updates_as_failed(self):
        original = torch.nn.functional.cross_entropy
        evaluations = []
        def losses(*args, **kwargs):
            if kwargs.get('reduction') == 'sum':
                evaluations.append(None)
                if len(evaluations) > 1:
                    return torch.tensor(float('inf'))
            return original(*args, **kwargs)
        with tempfile.TemporaryDirectory() as directory, mock.patch('examples.train.F.cross_entropy', side_effect=losses):
            result = run(steps=2, out=directory, verbose=False)
            self.assertEqual(result['steps'], 2)
            self.assertTrue(result['diverged'])
            self.assertIsNone(result['history'][-1]['val_loss'])
            self.assertTrue(result['history'][-1]['nonfinite_validation_loss'])

    def test_old_test_result_cannot_cross_runtime_versions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'book').mkdir()
            old_runtime = copy.deepcopy(artifact_io.runtime_snapshot())
            old_runtime['packages']['torch'] = 'different-version'
            artifact_io.write_json(root/'book/validation.json', dict(
                tests_run=1, failures=0, errors=0, skipped=0, success=True, source_sha256={}, runtime=old_runtime))
            with mock.patch.object(rebuild, 'ROOT', root), \
                 mock.patch.object(rebuild.sys, 'argv', ['rebuild', '--skip-checks', '--without-pdf']), \
                 mock.patch.object(rebuild.subprocess, 'run', return_value=mock.Mock(returncode=0)) as launched, \
                 contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                rebuild.main()
            self.assertFalse(launched.called, 'stale runtime validation must stop before launching a build')
            report = json.loads((root/'book/rebuild-check.json').read_text(encoding='utf-8'))
            self.assertFalse(report['complete'])
            self.assertIn('runtime', report['stages'][-1]['error'])

    def test_empty_test_discovery_is_a_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'book').mkdir()
            with mock.patch.object(run_checks,'ROOT',root), \
                 mock.patch.object(run_checks.unittest.defaultTestLoader,'discover',return_value=unittest.TestSuite()), \
                 mock.patch.object(run_checks.subprocess,'run') as launched, self.assertRaises(SystemExit):
                run_checks.main()
            self.assertFalse(launched.called,'no tests must not trigger external checks')
            report=json.loads((root/'book/validation.json').read_text(encoding='utf-8'))
            self.assertFalse(report['success'])
            self.assertEqual(report['tests_run'],0)

    def test_corrupt_validation_clears_previous_rebuild_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'book').mkdir()
            (root/'book/validation.json').write_text('{broken',encoding='utf-8')
            artifact_io.write_json(root/'book/rebuild-check.json',{'success':True,'complete':True})
            with mock.patch.object(rebuild,'ROOT',root), \
                 mock.patch.object(rebuild.sys,'argv',['rebuild','--skip-checks']), \
                 mock.patch.object(rebuild.subprocess,'run') as launched,self.assertRaises(SystemExit):
                rebuild.main()
            self.assertFalse(launched.called,'corrupt validation must stop before build')
            report=json.loads((root/'book/rebuild-check.json').read_text(encoding='utf-8'))
            self.assertFalse(report['success'])
            self.assertFalse(report['complete'])
            self.assertIn('unreadable',report['stages'][-1]['error'])

    def test_mismatch_promotes_before_subtraction(self):
        result = mismatch_stats(torch.tensor([60000., float('nan')], dtype=torch.float16),
                                torch.tensor([-60000., float('nan')], dtype=torch.float16), torch.tensor([1, 0]))
        self.assertEqual(result['absolute_max'], 120000.)
        with self.assertRaises(ValueError):
            mismatch_stats(torch.zeros(2), torch.ones(2), torch.tensor([1., .5]))
