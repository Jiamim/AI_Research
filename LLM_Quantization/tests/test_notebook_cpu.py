"""Execute selected notebook functions on CPU without downloading any model."""

import ast
import json
import math
import hashlib
import io
import re
from contextlib import redirect_stdout
from datetime import datetime, timezone
from numbers import Integral
from pathlib import Path
import statistics
import tempfile
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from experiments.quant_lab import group_qdq

ROOT = Path(__file__).resolve().parents[1]


def notebook_functions():
    notebook = json.loads((ROOT/'experiments'/'notebooks'/'colab_qwen3_quantization.ipynb').read_text())
    names = {
        'ppl_windows', 'perplexity', 'rtn_dequantize', 'apply_rtn', 'phase_benchmark',
        'experiment_history', 'experiment_plan', 'calibration_blocks', 'aggregate_weight_nmse',
        'logits_kl', 'check_answer', 'comparison_rows', 'ptq_config', 'prepare_ptq', 'save_json',
        'prepare_model', 'model_runtime_info', 'configure_ptq_workers',
        'generation_completion', 'generation_benchmark', 'validate_ptq_backends',
    }
    nodes = []
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        source = '\n'.join(line for line in ''.join(cell['source']).splitlines()
                           if not line.lstrip().startswith(('%', '!')))
        tree = ast.parse(source)
        nodes.extend(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names)
    if {node.name for node in nodes} != names:
        raise AssertionError('Missing tested notebook functions')
    namespace = dict(torch=torch, F=F, math=math, json=json, re=re, Integral=Integral, GPU=torch.device('cpu'),
                     statistics=statistics, time=time, NEW_TOKENS=4, TIMING_REPEATS=2)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<notebook-cpu>', 'exec'), namespace)
    return namespace


class NotebookNumerics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = notebook_functions()

    def test_window_coverage_no_duplicates(self):
        windows = self.env['ppl_windows']
        for tokens, context, stride in ((2048, 256, 256), (32768, 1024, 512), (17, 8, 3), (17, 8, 8)):
            with self.subTest(tokens=tokens, context=context, stride=stride):
                scored = []
                for begin, end, start in windows(tokens, context, stride):
                    self.assertLessEqual(end-begin, context)
                    self.assertGreater(start, begin)
                    scored.extend(range(start, end))
                self.assertEqual(len(scored), len(set(scored)))
                expected = [i for i in range(1, tokens) if stride < context or i % context]
                self.assertEqual(scored, expected)

    def test_invalid_window_configuration(self):
        for args in ((1, 4, 2), (8, 1, 1), (8, 4, 5), (8, 4, 0), (8, 4, 1.5)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                list(self.env['ppl_windows'](*args))

    def test_ppl_matches_independent_token_loss(self):
        # Logits depend on the preceding token, exposing off-by-one alignment.
        vocabulary = 5
        ids = torch.tensor([[0, 1, 4, 2, 3, 0, 2, 2, 1, 4, 0, 3, 1]])
        class Model:
            def __call__(self, input_ids, use_cache):
                return SimpleNamespace(logits=F.one_hot(input_ids, vocabulary).float()*2)
        for context, stride in ((6, 3), (6, 6)):
            result = self.env['perplexity'](Model(), ids, context, stride)
            indices = [i for i in range(1, ids.shape[1]) if stride < context or i % context]
            all_logits = F.one_hot(ids[:, :-1], vocabulary).float()*2
            losses = F.cross_entropy(all_logits.reshape(-1, vocabulary), ids[:, 1:].reshape(-1), reduction='none')
            expected = sum(losses[i-1].item() for i in indices)
            self.assertEqual(result['scored_tokens'], len(indices))
            self.assertAlmostEqual(result['nll_sum'], expected, places=5)
            self.assertAlmostEqual(result['ppl'], math.exp(expected/len(indices)), places=5)

    def test_rtn_padding_zero_groups_and_reference(self):
        quantize = self.env['rtn_dequantize']
        weight = torch.tensor([[0., 0., 0., 0., .3], [-1., -.3, .2, .9, -.5]], dtype=torch.float16)
        result = quantize(weight, group_size=4)
        padded = F.pad(weight.float(), (0, 3)).numpy()
        expected = group_qdq(padded, group_size=4)[:, :5].astype(np.float16)
        np.testing.assert_array_equal(result.numpy(), expected)
        self.assertEqual(result.shape, weight.shape)
        self.assertEqual(result.dtype, weight.dtype)

    def test_rtn_invalid_inputs(self):
        quantize = self.env['rtn_dequantize']
        for kwargs in ({'bits': 2.5}, {'bits': True}, {'bits': 1}, {'group_size': 0}, {'group_size': 2.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                quantize(torch.ones(2, 4), **kwargs)
        for weight in (torch.ones(4), torch.empty(0, 4), torch.tensor([[float('nan')]]), torch.ones(2, 4, dtype=torch.int64)):
            with self.subTest(shape=weight.shape), self.assertRaises(ValueError):
                quantize(weight)

    def test_apply_rtn_preserves_excluded_layers(self):
        torch.manual_seed(42)
        model = torch.nn.Module()
        model.proj = torch.nn.Linear(5, 3)
        model.lm_head = torch.nn.Linear(3, 5)
        model.embedding = torch.nn.Embedding(5, 3)
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        records = self.env['apply_rtn'](model, group_size=4)
        self.assertEqual([record['layer'] for record in records], ['proj'])
        self.assertFalse(torch.equal(before['proj.weight'], model.proj.weight))
        for name in ('proj.bias', 'lm_head.weight', 'lm_head.bias', 'embedding.weight'):
            torch.testing.assert_close(dict(model.named_parameters())[name], before[name], rtol=0, atol=0)

    def test_phase_timing_updates_cache_and_mask_lengths(self):
        calls = []
        class Model:
            def __call__(self, input_ids, attention_mask, use_cache, past_key_values=None):
                before = past_key_values or 0
                after = before + input_ids.shape[1]
                if attention_mask.shape[1] != after:
                    raise AssertionError('Attention mask and cache lengths differ')
                calls.append((before, after))
                return SimpleNamespace(logits=F.one_hot(input_ids % 5, 5).float(), past_key_values=after)
        self.env['prompt_inputs'] = {'input_ids': torch.tensor([[1, 2, 3]]), 'attention_mask': torch.ones(1, 3, dtype=torch.long)}
        with patch.object(torch.cuda, 'synchronize'):
            result = self.env['phase_benchmark'](Model())
        self.assertEqual(calls, [(0, 3), (3, 4), (4, 5), (5, 6)] * 3)
        self.assertEqual(result['decode_tokens'], 3)
        self.assertEqual(len(result['phase_seconds_all']), 2)

    def test_execution_archive_selects_only_latest_tutorial_cells(self):
        cells = ['unrelated user commands', '# quantization-cell: setup\nold settings']
        names = ('setup', 'data', 'evaluation', 'rtn', 'ptq', 'benchmark')
        cells.extend(f'# quantization-cell: {name}\ncurrent {name}' for name in names)
        history = self.env['experiment_history'](cells)
        self.assertNotIn('unrelated', history)
        self.assertNotIn('old settings', history)
        for name in names:
            self.assertIn(f'current {name}', history)
        with self.assertRaises(RuntimeError):
            self.env['experiment_history'](cells[:-1])

    def test_chat_template_text_is_not_given_duplicate_special_tokens(self):
        notebook = json.loads((ROOT/'experiments'/'notebooks'/'colab_qwen3_quantization.ipynb').read_text())
        checked = 0
        for cell in notebook['cells']:
            if cell['cell_type'] != 'code':
                continue
            source = '\n'.join(line for line in ''.join(cell['source']).splitlines()
                               if not line.lstrip().startswith(('%', '!')))
            for node in ast.walk(ast.parse(source)):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                        and node.func.id == 'tokenizer' and node.args
                        and isinstance(node.args[0], ast.Name) and node.args[0].id in ('text', 'prompt_text')):
                    kwargs = {kw.arg: kw.value for kw in node.keywords}
                    self.assertIn('add_special_tokens', kwargs)
                    self.assertFalse(ast.literal_eval(kwargs['add_special_tokens']))
                    checked += 1
        self.assertEqual(checked, 2)

    def test_plan_has_three_independent_rtn_groups_and_two_ptq_methods(self):
        plan = self.env['experiment_plan'](True, [32, 64, 128], True, True, 128)
        self.assertEqual([p['mode'] for p in plan], [
            'fp16', 'int8', 'nf4', 'rtn4_g32', 'rtn4_g64', 'rtn4_g128', 'gptq4_g128', 'awq4_g128',
        ])
        self.assertEqual(len({id(p) for p in plan}), len(plan))
        self.assertEqual(len(self.env['experiment_plan'](False, [128], False, False, 128)), 3)
        for groups in ([], [32, 32], [0], [True], [17], [32.0], [[32]]):
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                self.env['experiment_plan'](True, groups, True, True, 128)

    def test_calibration_blocks_are_repeatable_nonoverlapping_and_exact_length(self):
        ids = torch.arange(800).reshape(1, -1)
        prepare = self.env['calibration_blocks']
        blocks, starts = prepare(ids, count=6, length=64, seed=42)
        again, same_starts = prepare(ids, count=6, length=64, seed=42)
        self.assertEqual(tuple(blocks.shape), (6, 64))
        self.assertEqual(len(starts), len(set(starts)))
        self.assertEqual(starts, same_starts)
        torch.testing.assert_close(blocks, again)
        for block, start in zip(blocks, starts):
            self.assertEqual(start % 64, 0)
            torch.testing.assert_close(block, ids[0, start:start+64])
        with self.assertRaises(ValueError):
            prepare(ids, count=30, length=64, seed=42)

    def test_rtn_group_change_starts_from_original_weights(self):
        original = torch.nn.Linear(128, 4, bias=False)
        torch.manual_seed(45)
        original.weight.data.uniform_(-1, 1)
        original.weight.data[:, 0] = 20
        restored = []
        for group in (32, 64, 128):
            model = torch.nn.Sequential(torch.nn.Linear(128, 4, bias=False))
            model[0].load_state_dict(original.state_dict())
            records = self.env['apply_rtn'](model, group)
            restored.append(model[0].weight.detach().clone())
            expected_error = (original.weight - model[0].weight).square().sum().item()
            expected_signal = original.weight.square().sum().item()
            self.assertAlmostEqual(self.env['aggregate_weight_nmse'](records),
                                   expected_error / expected_signal, places=6)
        self.assertFalse(torch.equal(restored[0], restored[-1]))

    def test_nmse_uses_energy_weighting_instead_of_layer_mean(self):
        records = [dict(squared_error_sum=1., signal_squared_sum=1.),
                   dict(squared_error_sum=9., signal_squared_sum=999.)]
        self.assertAlmostEqual(self.env['aggregate_weight_nmse'](records), 0.01)

    def test_logits_kl_matches_reference_and_exposes_distribution_changes(self):
        p = torch.tensor([[[.8, .2]]], dtype=torch.float64)
        q = torch.tensor([[[.4, .6]]], dtype=torch.float64)
        kl = self.env['logits_kl']
        self.assertEqual(kl(p.log(), p.log()), 0.)
        expected = .8 * math.log(.8/.4) + .2 * math.log(.2/.6)
        self.assertAlmostEqual(kl(p.log(), q.log()), expected)
        with self.assertRaises(ValueError):
            kl(p.log(), q.log()[0])

    def test_answer_checks_are_strict_and_do_not_claim_general_accuracy(self):
        check = self.env['check_answer']
        self.assertTrue(check('arithmetic', '391\n')['passed'])
        self.assertFalse(check('arithmetic', '答案是391')['passed'])
        self.assertTrue(check('json_format', '{"name":"苹果", "count":3}')['passed'])
        for answer in ('```json\n{"name":"苹果","count":3}\n```',
                       '{"name":"苹果","count":3.0}', '{"name":"苹果","count":3,"extra":0}'):
            self.assertFalse(check('json_format', answer)['passed'])
        self.assertIsNone(check('instruction', 'arbitrary text')['passed'])

    def test_content_format_and_ambiguous_answers_are_separate(self):
        check = self.env['check_answer']
        for answer in ('答案是391', '17 × 23 = 391', '**391**', r'\boxed{391}'):
            result = check('arithmetic', answer)
            self.assertTrue(result['content_passed'], answer)
            self.assertFalse(result['format_passed'], answer)
        self.assertFalse(check('arithmetic', '421')['content_passed'])
        self.assertIsNone(check('arithmetic', '可能是391，但还没确定')['content_passed'])
        fenced = check('json_format', '```json\n{"name":"苹果","count":3}\n```')
        self.assertTrue(fenced['content_passed'])
        self.assertFalse(fenced['format_passed'])
        duplicate = check('json_format', '{"name":"梨","name":"苹果","count":3}')
        self.assertFalse(duplicate['passed'])

    def test_eos_at_exact_limit_is_not_reported_as_truncation(self):
        finish = self.env['generation_completion']
        self.assertFalse(finish(torch.tensor([1, 9]), [8, 9], 2)['length_limit_reached'])
        self.assertTrue(finish(torch.tensor([1, 2]), 9, 2)['length_limit_reached'])
        self.assertFalse(finish(torch.tensor([1]), 9, 2)['length_limit_reached'])

    def test_cpu_parent_pool_meets_loader_minimum(self):
        configure = self.env['configure_ptq_workers']
        for raw, expected in ((None, 2), ('1', 2), ('0', 2), ('6', 6)):
            environment = {} if raw is None else {'GPTQMODEL_CPU_WORKERS': raw}
            self.assertEqual(configure(environment), expected)
            self.assertEqual(environment['GPTQMODEL_CPU_WORKERS'], str(expected))
        with self.assertRaises(ValueError):
            configure({'GPTQMODEL_CPU_WORKERS': 'invalid'})

    def test_backend_preflight_checks_pack_and_inference_separately(self):
        calls = []
        package = ModuleType('gptqmodel')
        package.BACKEND = lambda value: value
        constants = ModuleType('gptqmodel.models._const')
        constants.DEVICE = SimpleNamespace(CUDA='cuda')
        importer = ModuleType('gptqmodel.utils.importer')
        def select(**kwargs):
            calls.append(kwargs)
            if kwargs['backend'] == 'awq_gemm_triton':
                raise ValueError('inference unavailable')
            return torch.nn.Linear
        importer.select_quant_linear = select
        modules = {'gptqmodel': package, 'gptqmodel.models._const': constants,
                   'gptqmodel.utils.importer': importer}
        settings = dict(PLAN=[dict(mode='awq4_g128', method='awq', group_size=128)],
                        AWQ_BACKEND='awq_gemm_triton', AWQ_PACK_BACKEND='awq_torch', DTYPE=torch.float16,
                        ptq_config=lambda *args: SimpleNamespace(format='gemm', method='awq'))
        with patch.dict(self.env, settings), patch.dict('sys.modules', modules):
            records = self.env['validate_ptq_backends']()
        self.assertEqual([(c['backend'], c['pack']) for c in calls],
                         [('awq_torch', True), ('awq_gemm_triton', False)])
        self.assertIn('inference unavailable', records[-1]['error'])

    def test_generation_timing_rejects_early_stop(self):
        class Model:
            def generate(self, **kwargs):
                return torch.tensor([[1, 2, 3, 4]])  # two input + two output, expected four.
        settings = dict(prompt_inputs={'input_ids': torch.tensor([[1, 2]])},
                        tokenizer=SimpleNamespace(eos_token_id=9, decode=lambda *a, **kw: 'text'))
        with patch.dict(self.env, settings), patch.object(torch.cuda, 'synchronize'), \
                patch.object(torch.cuda, 'reset_peak_memory_stats'):
            with self.assertRaisesRegex(ValueError, '生成长度'):
                self.env['generation_benchmark'](Model())

    def test_summary_excludes_fake_quant_speedup_and_checks_protocol(self):
        baseline = dict(mode='fp16', representation='fp16', group_size=None, ppl=10.,
                        scored_tokens=100, logits_kl_vs_fp16=0., model_footprint_mib=1200.,
                        peak_generation_allocated_mib=1600., generation_tokens_per_s=20.,
                        preparation_seconds=1., rule_checks_passed=1, rule_checks_total=2)
        fake = dict(baseline, mode='rtn4_g128', representation='fake_quant_fp16', ppl=11.,
                    group_size=128, generation_tokens_per_s=22.)
        packed = dict(baseline, mode='gptq4_g128', representation='packed_int4', group_size=128,
                      generation_tokens_per_s=30.)
        summary = self.env['comparison_rows']([baseline, fake, packed])
        self.assertAlmostEqual(summary[1]['ppl_delta_pct'], 10.)
        self.assertIsNone(summary[1]['speedup_vs_fp16'])
        self.assertEqual(summary[2]['speedup_vs_fp16'], 1.5)
        with self.assertRaises(ValueError):
            self.env['comparison_rows']([baseline, dict(fake, scored_tokens=99)])
        with self.assertRaises(ValueError):
            self.env['comparison_rows']([fake])

    def test_ptq_calibrates_saves_then_reloads_for_evaluation(self):
        calls = []
        tiny = torch.nn.Linear(2, 2)
        class CalibrationModel:
            def quantize(self, **kwargs):
                calls.append(('quantize', kwargs))
            def save(self, path):
                calls.append(('save', path))
                Path(path).mkdir()
                (Path(path)/'model.safetensors').write_bytes(b'fake checkpoint')
                (Path(path)/'config.json').write_text('{"quantization_config":{"bits":4}}')
        class Loader:
            @staticmethod
            def load(path, **kwargs):
                calls.append(('load', path, kwargs))
                return CalibrationModel() if path == 'original-snapshot' else SimpleNamespace(model=tiny)
        fake_package = ModuleType('gptqmodel')
        fake_package.GPTQModel = Loader
        fake_package.GPTQConfig = lambda **kwargs: kwargs
        fake_package.AWQConfig = lambda **kwargs: kwargs
        fake_package.BACKEND = lambda name: name
        calibration = [{'input_ids': [1, 2], 'attention_mask': [1, 1]}]
        with tempfile.TemporaryDirectory() as directory:
            settings = dict(ARTIFACT_DIR=Path(directory), SOURCE_DIR='original-snapshot', DTYPE=torch.float16,
                            GPTQ_BACKEND='gptq_triton', AWQ_BACKEND='awq_gemm_triton',
                            AWQ_PACK_BACKEND='awq_torch',
                            calibration_inputs=calibration, clear_gpu=lambda: None,
                            save_json=lambda name, value: None,
                            tokenizer=SimpleNamespace(save_pretrained=lambda path: None))
            with patch.dict(self.env, settings), patch.dict('sys.modules', {'gptqmodel': fake_package}), \
                    patch.object(torch.cuda, 'synchronize'):
                for method in ('gptq', 'awq'):
                    spec = dict(mode=method+'4_g128', method=method, group_size=128)
                    model, details = self.env['prepare_ptq'](spec)
                    self.assertIs(model, tiny)
                    self.assertGreater(details['checkpoint_weights_mib'], 0.)
                    with self.assertRaises(RuntimeError):
                        self.env['prepare_ptq'](spec)
        self.assertEqual([call[0] for call in calls], ['load', 'quantize', 'save', 'load'] * 2)
        for offset in (0, 4):
            source_load, quantize, _, reload = calls[offset:offset+4]
            self.assertIs(quantize[1]['calibration'], calibration)
            self.assertIsNone(quantize[1]['calibration_sort'])
            self.assertFalse(source_load[2]['quantize_config']['lm_head'])
            self.assertIsNone(source_load[2]['quantize_config']['fallback'])
            self.assertNotIn('device_map', source_load[2])
            self.assertEqual(reload[2]['device_map'], {'': 'cuda:0'})
        self.assertEqual(calls[1][1]['backend'], 'gptq_triton')
        self.assertEqual(calls[5][1]['backend'], 'awq_torch')
        self.assertEqual(calls[7][2]['backend'], 'awq_gemm_triton')

    def test_benchmark_continues_after_ptq_failure_and_exports_partial_report(self):
        """Run actual loop/report code with tiny stand-ins, without GPU or downloads."""
        notebook = json.loads((ROOT/'experiments'/'notebooks'/'colab_qwen3_quantization.ipynb').read_text())
        sources = {''.join(c['source']).splitlines()[0]: ''.join(c['source'])
                   for c in notebook['cells'] if c['cell_type'] == 'code'}
        benchmark = sources['# quantization-cell: benchmark']
        report = sources['# quantization-cell: summary']
        download = sources['# quantization-cell: download']
        plan = self.env['experiment_plan'](True, [32, 64, 128], True, True, 128)
        calls = []
        def prepare(spec):
            calls.append(spec['mode'])
            if spec['method'] == 'gptq':
                raise RuntimeError('simulated backend compilation error')
            return SimpleNamespace(get_memory_footprint=lambda: 1024), {'weight_nmse': 0.01 if spec['method'] == 'rtn' else None}
        task = {'id': 'instruction', 'prompt': 'a prompt', 'rubric': 'manual'}
        samples = [{**task, 'answer': '<script>alert(1)</script>', 'passed': None,
                    'evaluation': 'pending_manual_review'}]
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            env = dict(self.env,
                       hashlib=hashlib, datetime=datetime, timezone=timezone,
                       OUTPUT_DIR=output, ARTIFACT_DIR=output/'checkpoints',
                       PLAN=plan, metadata=dict(protocol_version=4, profile='synthetic-test', plan=plan,
                                               new_tokens=4, timing_repeats=2),
                       PROFILE='synthetic-test', PROTOCOL_VERSION=4,
                       CONFIG={'context': 4, 'stride': 2}, QUALITY_TASKS=[task],
                       eval_ids=torch.tensor([[1, 2, 3, 4]]), MEASURE_PHASES=False,
                       RUN_GPTQ=True, RUN_AWQ=True, validate_ptq_backends=lambda: [],
                       get_ipython=lambda: SimpleNamespace(history_manager=SimpleNamespace(
                           input_hist_raw=['', *sources.values()])),
                       clear_gpu=lambda: None, prepare_model=prepare,
                       ensure_gpu_model=lambda model: None,
                       model_runtime_info=lambda model, method: {},
                       probe_log_probs=lambda model: torch.tensor([[[.5, .5]]]).log(),
                       perplexity=lambda *args: dict(ppl=2., nll_sum=2.07944, scored_tokens=3),
                       generation_benchmark=lambda model: dict(generation_tokens_per_s=10.,
                                                               peak_generation_allocated_mib=1.,
                                                               generated_tokens=4, generation_seconds_all=[.4, .4]),
                       quality_samples=lambda model: samples)
            with patch.object(torch.cuda, 'synchronize'), patch.object(torch.cuda, 'reset_peak_memory_stats'), \
                    patch.object(torch.cuda, 'max_memory_allocated', return_value=1024), \
                    patch('IPython.display.display'), patch.object(plt, 'show'), redirect_stdout(io.StringIO()):
                exec(compile(benchmark, '<notebook-benchmark>', 'exec'), env)
                exec(compile(report, '<notebook-report>', 'exec'), env)
                colab = ModuleType('google.colab')
                colab.files = SimpleNamespace(download=lambda path: None)
                with patch.dict('sys.modules', {'google.colab': colab}):
                    exec(compile(download, '<notebook-download>', 'exec'), env)
            from experiments.inspect_run import inspect
            try:
                self.assertEqual(inspect(env['archive'])['status'], 'partial')
            finally:
                Path(env['archive']).unlink(missing_ok=True)
            self.assertEqual(calls, [spec['mode'] for spec in plan])
            self.assertEqual(env['metadata']['status'], 'partial')
            self.assertEqual(len(env['results']), 7)
            self.assertEqual(env['failures'][0]['mode'], 'gptq4_g128')
            self.assertEqual(env['results'][-1]['mode'], 'awq4_g128')
            for name in ('summary.csv', 'summary.json', 'comparison.png', 'rtn_groups.png', 'quality_comparison.html',
                         'failures.json', 'results.json', 'metadata.json', 'executed_cells.txt'):
                self.assertTrue((output/name).is_file(), name)
            report_text = (output/'quality_comparison.html').read_text()
            self.assertNotIn('<script>', report_text)
            self.assertIn('&lt;script&gt;', report_text)
            self.assertNotIn('gptq4_g128', (output/'summary.csv').read_text())
            self.assertNotIn('synthetic-test', ''.join(str(c.get('outputs', [])) for c in notebook['cells']))
            # A write failure is not a failed quantization mode. Stop immediately
            # instead of placing the already successful baseline in both lists.
            env['metadata']['status'] = 'prepared'
            real_replace = Path.replace
            def fail_results_write(path, target):
                if Path(target).name == 'results.json':
                    raise OSError('simulated report disk failure')
                return real_replace(path, target)
            with patch.object(torch.cuda, 'synchronize'), patch.object(torch.cuda, 'reset_peak_memory_stats'), \
                    patch.object(torch.cuda, 'max_memory_allocated', return_value=1024), \
                    patch.object(Path, 'replace', fail_results_write), redirect_stdout(io.StringIO()), \
                    self.assertRaisesRegex(OSError, 'report disk failure'):
                exec(compile(benchmark, '<notebook-write-failure>', 'exec'), env)
            self.assertEqual(env['failures'], [])
            self.assertEqual([row['mode'] for row in env['results']], ['fp16'])


if __name__ == '__main__':
    unittest.main()
