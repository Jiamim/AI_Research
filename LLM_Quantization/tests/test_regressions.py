"""Regression cases for known failures; all fixtures are local and deterministic."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments import build_technique_docs as builder
from experiments import quant_lab, result_utils, technique_lab
from experiments.technique_catalog import CARDS, CHAPTERS
from experiments.technique_kernels import TinyKIVI

ROOT = Path(__file__).resolve().parents[1]


class InputContracts(unittest.TestCase):
    def test_invalid_bits_rejected(self):
        for fn, minimum in ((quant_lab.symmetric_qdq, 2), (quant_lab.affine_qdq, 1)):
            for bits in (True, np.bool_(False), 2.5, "4", minimum - 1, -1, 17):
                with self.subTest(function=fn.__name__, bits=bits), self.assertRaises(ValueError):
                    fn([0., 1.], bits=bits)

    def test_nonfinite_or_empty_arrays_rejected(self):
        for values in ([], [np.nan], [np.inf], [1., -np.inf], [1+2j]):
            for fn in (quant_lab.symmetric_qdq, quant_lab.affine_qdq):
                with self.subTest(function=fn.__name__, values=values), self.assertRaises(ValueError):
                    fn(values)

    def test_groups_and_codebooks_rejected(self):
        for group in (0, -2, 1.5, True, 3):
            with self.subTest(group=group), self.assertRaises(ValueError):
                quant_lab.group_qdq(np.ones((2, 4)), group_size=group)
        for codebook in ([], [np.nan], [[0., 1.]]):
            with self.subTest(codebook=codebook), self.assertRaises(ValueError):
                quant_lab.codebook_qdq(np.ones(4), codebook, group_size=4)

    def test_bound_cannot_expand_shape(self):
        with self.assertRaises(ValueError):
            quant_lab.symmetric_qdq([1., 2.], bound=np.ones((2, 1)))

    def test_float_range_failures_are_not_silent_invalid_codes(self):
        tiny, huge = np.nextafter(0., 1.), np.finfo(float).max
        for fn in (quant_lab.symmetric_qdq, quant_lab.affine_qdq):
            for values in ([0., tiny], [-huge, huge]):
                with self.subTest(function=fn.__name__, values=values), self.assertRaises(ValueError):
                    fn(values)
        # Clipping a huge finite input to an ordinary finite range is supported.
        decoded, codes, _ = quant_lab.symmetric_qdq([-huge, 0., huge], bound=1.)
        np.testing.assert_array_equal(codes, [-7, 0, 7])
        np.testing.assert_array_equal(decoded, [-1., 0., 1.])

    def test_valid_encodings_and_zero_groups(self):
        _, codes, scale = quant_lab.symmetric_qdq([-1., -.3, .2, .9], bits=np.int64(4))
        np.testing.assert_array_equal(codes, [-7, -2, 1, 6])
        self.assertAlmostEqual(scale.item(), 1/7)
        for fn in (quant_lab.symmetric_qdq, quant_lab.affine_qdq):
            np.testing.assert_array_equal(fn(np.zeros((2, 4)), axis=-1)[0], np.zeros((2, 4)))
        np.testing.assert_array_equal(quant_lab.affine_qdq([0., 1.], bits=1)[0], [0., 1.])

    def test_kivi_rejects_invalid_configuration(self):
        for kwargs in ({'group': 0}, {'residual': -2}, {'residual': 0}, {'dim': 0},
                       {'bits': 2.5}, {'group': True}, {'residual': 3}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TinyKIVI(**kwargs)

    def test_kivi_failed_append_does_not_change_state(self):
        cache = TinyKIVI()
        with self.assertRaises(ValueError):
            cache.append(np.ones(4), [1., 2., 3., np.nan])
        self.assertEqual(len(cache.k_recent), 0)
        for index in range(13):
            row = np.arange(4.) + index
            cache.append(row, row)
            keys, values = cache.reconstruct()
            self.assertEqual(keys.shape, (index + 1, 4))
            np.testing.assert_array_equal(values[-1], row)
            self.assertLess(len(cache.k_recent), cache.residual)
            self.assertLessEqual(len(cache.v_recent), cache.residual)

    def test_kivi_conversion_failure_rolls_back_both_streams(self):
        from copy import deepcopy
        # Fail after one key group has already been encoded in a multi-group flush.
        cache = TinyKIVI(dim=2, group=2, residual=4)
        for value in (1., 2., 3.):
            cache.append([value, -value], [value, 0.])
        before = deepcopy(cache.__dict__)
        real_encode = cache.encode
        calls = []
        def fail_second(x, axis):
            calls.append(axis)
            if len(calls) == 2:
                raise ValueError('simulated conversion failure')
            return real_encode(x, axis)
        with patch.object(cache, 'encode', side_effect=fail_second), self.assertRaises(ValueError):
            cache.append([4., -4.], [4., 0.])
        self.assertEqual(calls, [0, 0])
        for name in ('k_recent', 'v_recent', 'k_chunks', 'v_chunks'):
            self.assertEqual(len(cache.__dict__[name]), len(before[name]))
            for actual, expected in zip(cache.__dict__[name], before[name]):
                np.testing.assert_array_equal(actual, expected)
        cache.append([4., -4.], [4., 0.])
        self.assertEqual(cache.reconstruct()[0].shape, (4, 2))
        # A real float64 range failure must likewise leave the pending token absent.
        small = TinyKIVI(dim=2, group=2, residual=2)
        small.append([1., 1.], [0., 1.])
        snapshot = small.reconstruct()
        with self.assertRaises(ValueError):
            small.append([-np.finfo(float).max, np.finfo(float).max], [1., 2.])
        for actual, expected in zip(small.reconstruct(), snapshot):
            np.testing.assert_array_equal(actual, expected)

    def test_floating_quantizers_reject_invalid_input_instead_of_returning_codes(self):
        from experiments.technique_kernels import float_qdq, micro_fp4, nearest
        for value in (np.nan, np.inf, -np.inf, 1j):
            for fn, x in ((nearest, [value]), (float_qdq, [value]), (micro_fp4, [value]*16)):
                with self.subTest(function=fn.__name__, value=value), self.assertRaises(ValueError):
                    fn(x, [-1., 0., 1.]) if fn is nearest else fn(x)
        for scale in (0., -1., np.nan, np.inf, [1., 2.]):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                micro_fp4(np.ones(16), tensor_scale=scale)
        for fn, x in ((float_qdq, [np.nextafter(0., 1.)]),
                      (float_qdq, [np.finfo(float).max]),
                      (micro_fp4, [np.nextafter(0., 1.)]*16)):
            with self.subTest(function=fn.__name__), self.assertRaises(ValueError):
                fn(x)
        np.testing.assert_array_equal(float_qdq(np.zeros(16)), np.zeros(16))
        np.testing.assert_array_equal(micro_fp4(np.zeros(16))[0], np.zeros(16))

    def test_histogram_calibration_requires_probability_mass(self):
        from experiments.technique_kernels import histogram_kl
        for counts, threshold, bins in (([0, 0], 2, 2), ([2, -1], 2, 2),
                                       ([1, np.nan], 2, 2), ([1, 2], 0, 1),
                                       ([1, 2], 3, 2), ([1, 2], 2, 3), ([1, 2], 2.5, 1)):
            with self.subTest(counts=counts, threshold=threshold, bins=bins), self.assertRaises(ValueError):
                histogram_kl(counts, threshold, bins)
        normal = histogram_kl([4., 2., 1., .5], 3, 2)
        enormous = histogram_kl(np.array([4., 2., 1., .5])*1e307, 3, 2)
        self.assertAlmostEqual(normal, enormous)


class AlgorithmRegressions(unittest.TestCase):
    def test_slim_inverse_matrix_cannot_replace_cholesky_factor(self):
        card = next(c for c in CARDS if c['id'] == 'slim_llm')
        wrong = card['code'].replace('np.diag(U)[None, :]**2', 'np.diag(M)[None, :]**2')
        self.assertNotEqual(wrong, card['code'])
        with self.assertRaises(AssertionError):
            technique_lab.execute({**card, 'code': wrong})

    def test_vptq_does_not_use_input_hessian_as_output_vector_metric(self):
        card = next(c for c in CARDS if c['id'] == 'vptq')
        wrong = card['code'].replace(
            'np.sum((candidates-vector)**2, axis=-1)',
            'np.sum((candidates-vector)**2*np.array([9., 1.]), axis=-1)')
        self.assertNotEqual(wrong, card['code'])
        with self.assertRaises(AssertionError):
            technique_lab.execute({**card, 'code': wrong})

    def test_wrong_updates_and_wrong_selection_are_caught(self):
        mutations = (
            ('zeroquant', 'master -= .03*gradient', 'master += .03*gradient'),
            ('llm_qat', 'master -= .1*gradient', 'master += .1*gradient'),
            ('llm_qat', 'master -= .1*gradient', 'pass'),
            ('omni_lwc', 'best = min(candidates)', 'best = max(candidates)'),
        )
        for key, old, new in mutations:
            with self.subTest(technique=key, mutation=new):
                card = next(c for c in CARDS if c['id'] == key)
                self.assertIn(old, card['code'])
                with self.assertRaises(AssertionError):
                    technique_lab.execute({**card, 'code': card['code'].replace(old, new)})

    def test_ignoring_bit_width_is_caught(self):
        real = quant_lab.group_qdq
        def wrong(w, bits=4, group_size=4):
            return real(w, bits=8, group_size=group_size)
        card = next(c for c in CARDS if c['id'] == 'grouping')
        with patch.object(quant_lab, 'group_qdq', wrong), self.assertRaises(AssertionError):
            technique_lab.execute(card)


class DocumentRegressions(unittest.TestCase):
    def test_assessment_rejects_missing_paths_difficulty_and_wrong_scores(self):
        source = (ROOT/'docs'/'01_均匀量化与误差.md').read_text(encoding='utf-8')
        self.assertEqual(technique_lab.validate_assessment(source, '01'), 11)
        corruptions = [
            source.replace('01-01｜对称 INT4 的编码（简单｜', '01-01｜对称 INT4 的编码（', 1),
            source.replace('计算，10 分）', '计算，11 分）', 1),
            source.replace('**参考路径一：**', '参考路径一：', 1),
            source.replace('id="exercise-01-01"', 'id="exercise-01-missing"', 1),
            source.replace('<details><summary>参考答案与评分要点</summary>', '<details open><summary>参考答案与评分要点</summary>', 1),
        ]
        for candidate in corruptions:
            self.assertNotEqual(candidate, source)
            with self.assertRaises(ValueError):
                technique_lab.validate_assessment(candidate, '01')

    def test_review_exercises_remain_part_of_course_validation(self):
        source = (ROOT/'docs'/'11_练习与答案.md').read_text(encoding='utf-8')
        self.assertEqual(technique_lab.validate_assessment(source, '11'), 37)
        with self.assertRaises(ValueError):
            technique_lab.validate_assessment(source.replace('## 题 26：', '## 题 27：', 1), '11')

    def test_current_notebook_rejects_stale_outputs_and_protocol(self):
        import nbformat
        from experiments.check_project import check_notebook
        original = nbformat.read(ROOT/'experiments'/'notebooks'/'colab_qwen3_quantization.ipynb', as_version=4)
        import copy
        for mutation in ('outputs', 'protocol'):
            notebook = copy.deepcopy(original)
            if mutation == 'outputs':
                code = next(c for c in notebook.cells if c.cell_type == 'code')
                code.execution_count = 1
            else:
                notebook.metadata.quantization_validation.protocol_version = 999
            with self.subTest(mutation=mutation), patch.object(nbformat, 'read', return_value=notebook), \
                    self.assertRaises(AssertionError):
                check_notebook()

    def test_each_missing_closing_marker_is_rejected(self):
        for chapter, title in CHAPTERS.items():
            source = (ROOT/'docs'/f'{chapter}_{title}.md').read_text(encoding='utf-8')
            closing = '<!-- /inline-technique -->'
            start = 0
            while (position := source.find(closing, start)) >= 0:
                start = position + len(closing)
                with self.subTest(chapter=chapter, position=position), self.assertRaises(ValueError):
                    builder.strip_generated(source[:position] + source[start:])

    def test_stray_closing_marker_is_rejected(self):
        with self.assertRaises(ValueError):
            builder.strip_generated('Authored prose\n<!-- /inline-technique -->')

    def test_duplicate_ids_rejected_before_build(self):
        with patch.object(builder, 'CARDS', CARDS + [CARDS[0]]), self.assertRaises(ValueError):
            builder.outputs()

    def test_rebuild_preserves_prose_and_is_idempotent(self):
        for chapter, title in CHAPTERS.items():
            with self.subTest(chapter=chapter):
                source = (ROOT/'docs'/f'{chapter}_{title}.md').read_text(encoding='utf-8')
                cards = [c for c in CARDS if c['chapter'] == chapter]
                merged = builder.merged_chapter(chapter, cards, source)
                self.assertEqual(builder.strip_generated(source), builder.strip_generated(merged))
                self.assertEqual(merged, builder.merged_chapter(chapter, cards, merged))


class ResultsRegressions(unittest.TestCase):
    def test_default_commands_do_not_change_published_results(self):
        paths = [p for directory in ('experiments', 'figures') for p in (ROOT/directory).rglob('*')
                 if p.suffix in ('.json', '.md', '.png', '.svg')]
        def hashes():
            return {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        before = hashes()
        for script in ('quant_lab.py', 'technique_lab.py', 'outlook_scenarios.py', 'llm_int8_lab.py',
                       'deep_dive_lab.py'):
            result = subprocess.run([sys.executable, '-B', str(ROOT/'experiments'/script)],
                                    cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(before, hashes())

    def test_publishing_requires_document_validation(self):
        with patch('sys.argv', ['technique_lab.py', '--write-results']), \
             patch.object(technique_lab, 'validate_docs', side_effect=ValueError('bad docs')), \
             patch.object(technique_lab, 'write_json') as writer:
            with self.assertRaisesRegex(ValueError, 'bad docs'):
                technique_lab.main()
            writer.assert_not_called()

    def test_no_implicit_writes_from_output_options(self):
        for script, option in (('quant_lab.py', '--figures'), ('quant_lab.py', '--output-dir'),
                               ('technique_lab.py', '--output-dir'), ('outlook_scenarios.py', '--output-dir'),
                               ('llm_int8_lab.py', '--output-dir'), ('deep_dive_lab.py', '--output-dir')):
            args = [sys.executable, '-B', str(ROOT/'experiments'/script), option]
            if option == '--output-dir':
                args.append('runs/should-not-exist')
            completed = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 2)

    def test_custom_output_directory_for_all_runners(self):
        import re
        from urllib.parse import unquote
        with tempfile.TemporaryDirectory(prefix='quant results ') as temp:
            out = Path(temp)
            for script, filename in (('quant_lab.py', 'results.json'),
                                     ('technique_lab.py', 'technique_results.json'),
                                     ('outlook_scenarios.py', 'outlook_results.json'),
                                     ('deep_dive_lab.py', 'deep_dive_results.json')):
                result = subprocess.run([sys.executable, '-B', str(ROOT/'experiments'/script),
                                         '--write-results', '--output-dir', temp],
                                        cwd=ROOT, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                payload = json.loads((out/filename).read_text())
                self.assertEqual(payload['schema_version'], 2)
                self.assertIn('experiments/technique_kernels.py', payload['source_files_sha256'])
            self.assertEqual(json.loads((out/'technique_results.json').read_text())['checked_docs'], 93)
            for report in out.glob('*.md'):
                for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', report.read_text()):
                    self.assertTrue((report.parent/unquote(target)).exists(), target)

    def test_existing_figure_links_survive_report_refresh(self):
        metrics, checks, plot = quant_lab.run()
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            (out/'figures').mkdir()
            for name in ('uniform_and_clipping', 'scaling_and_rotation'):
                (out/'figures'/f'{name}.png').write_bytes(b'fixture')
            quant_lab.write_outputs(metrics, checks, plot, output_dir=out)
            report = (out/'实验结果.md').read_text(encoding='utf-8')
            self.assertIn('figures/uniform_and_clipping.png', report)
            payload = json.loads((out/'results.json').read_text())
            self.assertIn('created_utc', payload)
            self.assertIn('experiments/quant_lab.py', payload['source_files_sha256'])

    def test_helper_change_changes_source_fingerprint(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'experiments').mkdir()
            helper = root/'experiments'/'helper.py'
            helper.write_text('value = 4\n')
            with patch.object(result_utils, 'ROOT', root):
                before = result_utils.run_metadata({})
                helper.write_text('value = 8\n')
                after = result_utils.run_metadata({})
            self.assertNotEqual(before['source_sha256'], after['source_sha256'])

    def test_failed_json_encoding_preserves_previous_result(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'result.json'
            path.write_text('previous')
            with self.assertRaises(ValueError):
                result_utils.write_json(path, {'loss': float('nan')})
            self.assertEqual(path.read_text(), 'previous')


if __name__ == '__main__':
    unittest.main()
