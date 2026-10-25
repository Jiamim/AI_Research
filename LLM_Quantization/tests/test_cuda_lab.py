"""CUDA lab contracts on CPU, plus explicitly skipped-unless-CUDA integration.

Triton kernels and new cases run on CPU through TRITON_INTERPRET=1 in a subprocess
(the interpreter flag must be set before triton is imported); that checks numerics
and case wiring only, never performance.
"""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest
import warnings
from unittest.mock import patch

import numpy as np
import torch

from experiments.cuda_kernels import pack_int4, unpack_int4, pack_kv, unpack_kv, error_metrics
from experiments.cuda_runtime import RunReport, sample_summary, require_cuda, Unsupported
from experiments import cuda_lab
from experiments import cuda_cases, cuda_runtime

ROOT = Path(__file__).resolve().parents[1]
HAS_TRITON = importlib.util.find_spec('triton') is not None


def run_interpreted(code):
    """在新进程中以 Triton 解释器执行代码片段；返回 stdout，失败时把 stderr 带进断言信息。"""
    env = dict(os.environ, TRITON_INTERPRET='1', PYTHONDONTWRITEBYTECODE='1')
    result = subprocess.run([sys.executable, '-B', '-c', textwrap.dedent(code)], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=900)
    if result.returncode:
        raise AssertionError(result.stdout[-2000:] + result.stderr[-4000:])
    return result.stdout


def independent_qdq(matrix, group):
    rows = np.asarray(matrix,dtype=np.float32).reshape(matrix.shape[0],-1,group)
    maximum = np.max(abs(rows),axis=-1,keepdims=True)
    scale = np.where(maximum > 0,np.maximum(maximum/7,2.**-24),1).astype(np.float16)
    codes = np.clip(np.rint(rows/scale.astype(np.float32)),-7,7)
    return (codes*scale.astype(np.float32)).reshape(matrix.shape).astype(np.float16)


class CudaContracts(unittest.TestCase):
    def test_int4_physical_layout_and_signed_decode(self):
        weight=torch.tensor([[-7.,-1.,0.,7.],[7.,0.,-1.,-7.]])
        packed=pack_int4(weight,4)
        self.assertEqual(packed.payload.tolist(),[[249,112],[7,159]])
        self.assertEqual(packed.scales.dtype,torch.float16)
        self.assertEqual(packed.storage_bytes,8)  # four bytes payload + four bytes scale
        torch.testing.assert_close(unpack_int4(packed),weight.half(),rtol=0,atol=0)

    def test_codec_matches_independent_numpy_with_saved_scales(self):
        array=np.random.default_rng(7).normal(size=(7,32)).astype(np.float32)
        array[0]=0;array[1]*=1e-8
        for group in (2,8,32):
            expected=independent_qdq(array,group)
            actual=unpack_int4(pack_int4(torch.from_numpy(array),group))
            np.testing.assert_array_equal(actual.numpy(),expected)

    def test_reject_invalid_input_and_scale_overflow(self):
        for value,group in [(torch.ones(2,7),4),(torch.ones(2,8),3),(torch.ones(2,8),True),
                            (torch.full((2,8),float('nan')),4),(torch.ones(2,8,dtype=torch.int64),4),
                            (torch.full((2,8),1e9),4),(torch.empty(0,8),4)]:
            with self.subTest(shape=value.shape,group=group),self.assertRaises(ValueError):
                pack_int4(value,group)

    def test_kv_time_feature_axes_and_recent_tail_storage(self):
        rng=np.random.default_rng(3)
        k=torch.tensor(rng.normal(size=(1,2,12,4)),dtype=torch.float16)
        v=torch.tensor(rng.normal(size=(1,2,12,4)),dtype=torch.float16)
        k[:,:,:,0]*=8
        packed=pack_kv(k,v,4,4)
        rk,rv=unpack_kv(packed)
        expected_k=k.clone();expected_v=v.clone()
        for h in range(2):
            # K每通道沿time分组；V每token沿features分组。
            expected_k[0,h,:8]=torch.from_numpy(independent_qdq(k[0,h,:8].T.numpy(),4)).T
            expected_v[0,h,:8]=torch.from_numpy(independent_qdq(v[0,h,:8].numpy(),4))
        torch.testing.assert_close(rk,expected_k,rtol=0,atol=0)
        torch.testing.assert_close(rv,expected_v,rtol=0,atol=0)
        self.assertNotEqual(packed.recent_key.untyped_storage().data_ptr(),k.untyped_storage().data_ptr())
        self.assertEqual(packed.recent_key.untyped_storage().nbytes(),packed.recent_key.nbytes)
        self.assertEqual(packed.storage_bytes,64+64+128) # two 64-value packed prefixes and two recent tails
        untouched=pack_kv(k,v,4,12)
        self.assertIsNone(untouched.key)
        torch.testing.assert_close(unpack_kv(untouched)[0],k)
        for group,residual in [(3,0),(4,1),(4,16),(4,-1)]:
            with self.assertRaises(ValueError):pack_kv(k,v,group,residual)

    def test_metrics_and_timing_reject_invalid_records(self):
        self.assertEqual(sample_summary([3.,1.,2.])['median_ms'],2.)
        for samples in ([],[0.],[-1.],[True],[float('nan')],[float('inf')]):
            with self.assertRaises(ValueError):sample_summary(samples)
        result=error_metrics(torch.tensor([1.,3.]),torch.tensor([2.,3.]))
        self.assertAlmostEqual(result['nmse'],.1)
        with self.assertRaises(ValueError):error_metrics(torch.zeros(2),torch.zeros(1))
        with self.assertRaises(ValueError):error_metrics(torch.zeros(2),torch.tensor([0.,float('inf')]))

    def test_cuda_request_never_silently_uses_cpu(self):
        with patch.object(torch.cuda,'is_available',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'不会回退'):require_cuda('cuda:0')
            with self.assertRaises(RuntimeError):require_cuda('cpu')

    def test_timer_normalizes_batches_and_preserves_raw_samples(self):
        # CUDA event test double checks accounting only, never published as a measurement.
        class Event:
            def __init__(self,**kwargs):pass
            def record(self):pass
            def synchronize(self):pass
            def elapsed_time(self,end):return 10.
        calls=[]
        bench=cuda_runtime.CudaBench(torch.device('cpu'),'smoke')
        with patch.object(torch.cuda,'Event',Event),patch.object(torch.cuda,'synchronize'), \
             patch.object(torch.cuda,'reset_peak_memory_stats'), \
             patch.object(torch.cuda,'memory_allocated',return_value=100), \
             patch.object(torch.cuda,'max_memory_allocated',return_value=164):
            rows=bench.compare('timing',[cuda_cases.variant('x',lambda:calls.append(1),flops=2_000_000)])
        self.assertEqual(len(calls),3+5*5+1)  # warmup + measured calls + peak call
        self.assertEqual(rows[0]['cuda']['samples_ms'],[2.]*5)
        self.assertEqual(rows[0]['incremental_peak_bytes'],64)
        self.assertAlmostEqual(rows[0]['effective_tflops'],.001)

    def test_flush_mode_times_each_call_after_untimed_l2_flush(self):
        class Event:
            def __init__(self,**kwargs):pass
            def record(self):log.append('event')
            def synchronize(self):pass
            def elapsed_time(self,end):return 4.
        log=[]
        bench=cuda_runtime.CudaBench(torch.device('cpu'),'smoke')
        bench.flush_l2=lambda:log.append('flush')
        with patch.object(torch.cuda,'Event',Event),patch.object(torch.cuda,'synchronize'), \
             patch.object(torch.cuda,'reset_peak_memory_stats'), \
             patch.object(torch.cuda,'memory_allocated',return_value=0), \
             patch.object(torch.cuda,'max_memory_allocated',return_value=0):
            rows=bench.compare('timing',[cuda_cases.variant('slow',lambda:log.append('call'),bytes_moved=8_000_000),
                                         cuda_cases.variant('fast',lambda:log.append('call'))],
                               l2='flush',baseline='slow')
        # 预热之后：一次初始化清理，然后每次计时调用前恰有一次未计时的清理：flush, start, call, end。
        # 末尾两次调用是计时外的显存峰值测量，不清理也不计时。
        measured=log[log.index('flush'):]
        self.assertEqual(measured[-2:],['call','call'])
        timed=measured[:-2]
        self.assertEqual(timed.count('flush'),1+2*5*5)
        self.assertEqual(timed.count('call'),2*5*5)
        for i,item in enumerate(timed):
            if item=='call':
                self.assertEqual(timed[i-2:i+2],['flush','event','call','event'])
        self.assertEqual(rows[0]['l2_mode'],'flush')
        self.assertIsNone(rows[0]['wall'])
        self.assertEqual(rows[0]['cuda']['median_ms'],4.)
        self.assertAlmostEqual(rows[0]['effective_gbps'],2.)  # 8e6 bytes / 4 ms
        self.assertEqual([r['speedup_vs_baseline'] for r in rows],[1.,1.])
        with self.assertRaises(ValueError):
            bench.compare('timing',[cuda_cases.variant('x',lambda:None)],baseline='missing')
        with self.assertRaises(ValueError):
            bench.compare('timing',[cuda_cases.variant('x',lambda:None)],l2='cold')

    def test_status_counts_variant_level_failures(self):
        passed,skipped,failed={'status':'passed'},{'status':'skipped'},{'status':'failed'}
        self.assertEqual(cuda_lab.run_status([passed],[]),'completed')
        self.assertEqual(cuda_lab.run_status([passed,skipped],[]),'completed_with_skips')
        self.assertEqual(cuda_lab.run_status([skipped],[]),'skipped')
        self.assertEqual(cuda_lab.run_status([passed,failed],[]),'partial')
        self.assertEqual(cuda_lab.run_status([failed],[]),'failed')
        self.assertEqual(cuda_lab.run_status([passed],[{'case':'x'}]),'partial')

    def test_optional_variant_isolates_failures(self):
        rows=[]
        bench=type('Bench',(),{'checkpoint':staticmethod(rows.append)})()
        self.assertEqual(cuda_cases.optional_variant(bench,'c','ok',lambda:'v'),'v')
        def unsupported():raise Unsupported('missing library')
        def broken():raise AssertionError('wrong numbers')
        self.assertIsNone(cuda_cases.optional_variant(bench,'c','opt',unsupported))
        self.assertIsNone(cuda_cases.optional_variant(bench,'c','bad',broken))
        self.assertEqual([(r['variant'],r['status']) for r in rows],[('opt','skipped'),('bad','failed')])
        self.assertIn('AssertionError',rows[1]['reason'])
        self.assertIn('wrong numbers',rows[1]['traceback'])

    def test_bandwidth_reference_annotates_other_rows(self):
        report=RunReport({})
        report.data['results']=[dict(case='bandwidth',variant='read_sum',status='passed',effective_gbps=200.),
                                dict(case='w4a16_x',variant='y',status='passed',effective_gbps=150.),
                                dict(case='w4a16_x',variant='z',status='skipped')]
        report.annotate_bandwidth()
        self.assertEqual(report.data['measured_read_gbps'],200.)
        self.assertAlmostEqual(report.data['results'][1]['bandwidth_fraction'],.75)
        self.assertNotIn('bandwidth_fraction',report.data['results'][0])
        self.assertNotIn('bandwidth_fraction',report.data['results'][2])

    def test_key_comparisons_pair_rows_within_one_case(self):
        timed=lambda case,variant,ms:dict(case=case,variant=variant,status='passed',cuda=dict(median_ms=ms))
        rows=[timed('w4a16_a','triton_w4a16_fused',1.),timed('w4a16_a','triton_fp16_same_template',3.),
              timed('w4a16_a','torch_fp16',2.),timed('w4a16_b','triton_w4a16_fused',1.),
              dict(case='w4a16_b',variant='torch_fp16',status='failed')]
        pairs=cuda_runtime.key_comparisons(rows)
        self.assertEqual([(p['case'],p['reference'],p['speedup']) for p in pairs],
                         [('w4a16_a','triton_fp16_same_template',3.),('w4a16_a','torch_fp16',2.)])

    def test_rowwise_int8_matches_int64_reference(self):
        x=torch.tensor([[1.,-2.,4.,0.],[0.,0.,0.,0.]])
        codes,scale=cuda_cases._rowwise_int8(x)
        self.assertEqual(codes.dtype,torch.int8)
        self.assertEqual(codes[0].tolist(),[32,-64,127,0])
        self.assertEqual(scale[1].item(),1.)
        self.assertEqual(codes[1].abs().sum().item(),0)
        w=torch.randn(5,4)
        qw,_=cuda_cases._rowwise_int8(w)
        # torch._int_mm 在 CPU 上同样实现：INT32 结果必须与 int64 精确一致。
        self.assertTrue(torch.all(qw.abs()<=127))
        self.assertTrue(torch.all(qw.abs().amax(dim=1)==127))

    def test_decode_stack_skeleton_and_graph_helper_shapes(self):
        shapes=cuda_cases.stack_shapes(cuda_cases.QWEN3_06B)
        self.assertEqual(shapes['q'],(2048,1024))
        self.assertEqual(shapes['down'],(1024,3072))
        # 每层 7 个线性；FP16 权重约 0.88 GB 的 28 层骨架，不含 embedding/lm_head。
        params=sum(n*k for n,k in shapes.values())*cuda_cases.QWEN3_LAYERS
        self.assertEqual(params,28*(2048*1024+2*1024*1024+1024*2048+3*3072*1024))
        layers=[{name:torch.randn(shape,dtype=torch.float64)*.01 for name,shape in shapes.items()}]
        calls=[]
        def linear(layer,name,h):
            calls.append(name)
            return h@layer[name].T
        out=cuda_cases.stack_forward(torch.randn(2,1024,dtype=torch.float64),layers,linear)
        self.assertEqual(out.shape,(2,1024))
        self.assertEqual(calls,['q','k','v','o','gate','up','down'])

    def test_decode_attention_reads_all_past_keys(self):
        q=torch.ones(1,1,1,2)
        k=torch.zeros(1,1,4,2)
        v=torch.tensor([[[[1.,1.],[2.,2.],[3.,3.],[6.,6.]]]])
        expected=torch.full_like(q,3.)
        actual=cuda_cases.eager_attention(q,k,v,causal=False)
        torch.testing.assert_close(actual,expected)
        with self.assertRaises(ValueError):cuda_cases.eager_attention(q,k,v,causal=True)
        # An unsupported backend must be skipped without invoking a fallback.
        with patch.object(torch.backends.cuda,'can_use_flash_attention',return_value=False):
            available,reason=cuda_cases.sdpa_available(q,k,v,cuda_cases.SDPBackend.FLASH_ATTENTION,False)
            self.assertFalse(available)
            self.assertTrue(reason)

    def test_reported_fp32_fallback_cannot_pass_as_int8(self):
        with self.assertRaises(RuntimeWarning),cuda_cases.no_int8_fallback():
            warnings.warn('int8_linear_matmul has no usable algo; falling back to fp32 matmul.',RuntimeWarning)

    def test_failed_preflight_preserves_report_and_manifest(self):
        with tempfile.TemporaryDirectory() as temp,contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            args=['cuda_lab.py','--write-results','--output-dir',temp]
            with patch('sys.argv',args),patch.object(cuda_lab,'require_cuda',side_effect=RuntimeError('no GPU')):
                self.assertEqual(cuda_lab.main(),1)
            folder=next(Path(temp).iterdir())
            report=json.loads((folder/'cuda_results.json').read_text())
            self.assertEqual(report['status'],'failed')
            self.assertEqual(report['errors'][0]['message'],'no GPU')
            manifest=json.loads((folder/'manifest.json').read_text())
            for name,digest in manifest['files_sha256'].items():
                self.assertEqual(hashlib.sha256((folder/name).read_bytes()).hexdigest(),digest)
            self.assertIn('sources/experiments/cuda_cases.py',manifest['files_sha256'])
            second=RunReport({},temp)
            self.assertNotEqual(second.directory,folder)
            self.assertEqual(json.loads((folder/'cuda_results.json').read_text())['status'],'failed')

    def test_numerical_failure_is_not_labeled_unsupported(self):
        def passed(bench):bench.checkpoint(dict(case='first',variant='x',status='passed'))
        def unavailable(bench):raise Unsupported('optional library missing')
        def broken(bench):raise AssertionError('wrong answer')
        class FakeBench:
            def __init__(self,*args,checkpoint,**kwargs):self.checkpoint=checkpoint
        with tempfile.TemporaryDirectory() as temp,contextlib.redirect_stdout(io.StringIO()):
            args=['cuda_lab.py','--write-results','--output-dir',temp]
            with patch('sys.argv',args),patch.object(cuda_lab,'CASES',{'first':passed,'optional':unavailable,'bad':broken}), \
                 patch.object(cuda_lab,'require_cuda',return_value=torch.device('cuda:0')), \
                 patch.object(cuda_lab,'environment',return_value={'gpu':'test double'}),patch.object(cuda_lab,'CudaBench',FakeBench), \
                 patch.object(torch.cuda,'empty_cache'):
                self.assertEqual(cuda_lab.main(),1)
            report=json.loads(next(Path(temp).glob('*/cuda_results.json')).read_text())
            self.assertEqual(report['status'],'partial')
            self.assertEqual([r['status'] for r in report['results']],['passed','skipped','failed'])
            self.assertEqual(report['errors'][0]['type'],'AssertionError')


@unittest.skipUnless(HAS_TRITON,'需要 Triton；CPU 版 PyTorch 可单独 pip install triton 以运行解释器测试')
class TritonInterpreter(unittest.TestCase):
    def test_fused_kernels_match_explicit_decode(self):
        out=run_interpreted('''
            import math, torch
            from experiments.cuda_kernels import pack_int4, unpack_int4, pack_kv, unpack_kv
            from experiments import cuda_triton as ct
            g = torch.Generator().manual_seed(5)
            for m in (1, 3, 20):
                w = (torch.randn(48, 256, generator=g)*.1).half(); x = (torch.randn(m, 256, generator=g)/16).half()
                for group in (32, 128):
                    packed = pack_int4(w, group)
                    reference = x.float()@unpack_int4(packed).float().T
                    configs = ct.candidate_configs(m, 256, group)
                    assert configs
                    for config in configs:
                        assert (ct.w4a16_matmul(x, packed, config).float()-reference).abs().max() < 2e-3, (m, group, config)
                config = ct.candidate_configs(m, 256)[0]
                assert (ct.fp16_matmul(x, w, config).float()-x.float()@w.float().T).abs().max() < 2e-3
            for length, residual, group in ((96, 0, 32), (96, 32, 32), (200, 8, 16), (64, 64, 32)):
                k = torch.randn(1, 2, length, 32, generator=g).half(); v = torch.randn(1, 2, length, 32, generator=g).half()
                q = torch.randn(1, 2, 1, 32, generator=g).half()
                cache = pack_kv(k, v, group_size=group, residual=residual)
                kq, vq = unpack_kv(cache)
                attend = lambda a, b: torch.softmax(q.float()@a.float().transpose(-1, -2)/math.sqrt(32), -1)@b.float()
                for block in (32, 64):
                    for splits in (1, 3):
                        out = ct.kv_decode_int4(q, cache, block_t=block, splits=splits)
                        assert (out.float()-attend(kq, vq)).abs().max() < 2e-3, (length, residual, block, splits)
                        out = ct.kv_decode_fp16(q, k, v, block_t=block, splits=splits)
                        assert (out.float()-attend(k, v)).abs().max() < 2e-3
            print('KERNELS OK')
        ''')
        self.assertIn('KERNELS OK',out)

    def test_new_cases_run_end_to_end_with_tiny_shapes(self):
        # 模拟 CUDA 计时/设备 API，只验证每个新实验的构建、校验与记录链路；数字不是测量。
        out=run_interpreted('''
            import contextlib, json, torch
            from unittest.mock import patch
            from experiments import cuda_cases, cuda_runtime
            from experiments.cuda_kernels import error_metrics
            class Event:
                def __init__(self, **kwargs): pass
                def record(self, *args): pass
                def synchronize(self): pass
                def elapsed_time(self, end): return 1.
            class Props:
                name = 'interpreter'; major, minor = 8, 9; multi_processor_count = 4; L2_cache_size = 1 << 20
            class Graph:
                def replay(self): pass
            class Stream:
                def __init__(self, *args, **kwargs): pass
                def wait_stream(self, other): pass
            def check_nmse(actual, expected, limit, label):
                error = error_metrics(expected.float(), actual.float())
                assert error['nmse'] <= limit, (label, error)
                return error
            tiny = dict(bandwidth_mib=1, w4a16_shapes=[(64, 256)], w4a16_batches=(1, 20), w4a16_groups=(32,),
                        lowp_shapes=[(8, 64, 128)], kv_lengths=(160,), kv_heads=2, kv_dim=64,
                        stack_model=dict(hidden=128, intermediate=256, q_out=128, kv_out=64),
                        stack_layers=2, stack_batches=(1,))
            null = lambda *args, **kwargs: None
            rows = []
            with contextlib.ExitStack() as stack:
                for target, name, value in [
                        (torch.cuda, 'Event', Event), (torch.cuda, 'synchronize', null),
                        (torch.cuda, 'reset_peak_memory_stats', null), (torch.cuda, 'memory_allocated', lambda *a: 0),
                        (torch.cuda, 'max_memory_allocated', lambda *a: 0), (torch.cuda, 'get_device_properties', lambda *a: Props()),
                        (torch.cuda, 'get_device_capability', lambda *a: (8, 9)), (torch.cuda, 'CUDAGraph', Graph),
                        (torch.cuda, 'graph', lambda graph: contextlib.nullcontext()), (torch.cuda, 'Stream', Stream),
                        (torch.cuda, 'stream', lambda s: contextlib.nullcontext()), (torch.cuda, 'current_stream', lambda *a: Stream()),
                        (cuda_cases, 'check_nmse', check_nmse)]:
                    stack.enter_context(patch.object(target, name, value))
                stack.enter_context(patch.dict(cuda_cases.SIZES, smoke=tiny))
                for case in ('bandwidth', 'w4a16', 'lowp_gemm', 'kv_fused', 'decode_stack'):
                    bench = cuda_runtime.CudaBench(torch.device('cpu'), 'smoke', checkpoint=rows.append)
                    bench.warmup = bench.repeats = bench.iterations = 1
                    bench.flush_bytes = 1 << 16
                    cuda_cases.CASES[case](bench)
            json.dumps(rows, allow_nan=False, default=str)
            failed = [(r['case'], r['variant'], r.get('reason')) for r in rows if r['status'] == 'failed']
            assert not failed, failed
            names = {(r['case'].split('_')[0], r['variant']) for r in rows if r['status'] == 'passed'}
            for expected in [('bandwidth', 'read_sum'), ('w4a16', 'triton_w4a16_fused'), ('w4a16', 'triton_fp16_same_template'),
                             ('lowp', 'int8_gemm_dequant'), ('lowp', 'fp8_gemm'), ('kv', 'triton_int4_fused'),
                             ('decode', 'triton_w4a16_cuda_graph')]:
                assert expected in names, (expected, sorted(names))
            assert all(r.get('speedup_vs_baseline') for r in rows if r['status'] == 'passed' and r['case'].startswith('w4a16'))
            print('CASES OK', len(rows))
        ''')
        self.assertIn('CASES OK',out)


@unittest.skipUnless(torch.cuda.is_available() and not torch.version.hip,'需要 NVIDIA CUDA 实测；CPU 不替代')
class CudaIntegration(unittest.TestCase):
    def test_smoke_cases_execute_on_cuda(self):
        from experiments.cuda_runtime import CudaBench
        from experiments.cuda_cases import CASES
        device=require_cuda('cuda:0')
        for name in ('bandwidth','gemm','int4','w4a16','lowp_gemm','attention','kv','kv_fused','decode_stack'):
            with self.subTest(case=name):
                rows=[]
                bench=CudaBench(device,'smoke',checkpoint=rows.append)
                bench.warmup,bench.repeats,bench.iterations=1,2,2
                CASES[name](bench)
                measured=[r for r in rows if r['status']=='passed']
                self.assertTrue(measured)
                self.assertFalse([r for r in rows if r['status']=='failed'],
                                 [(r['variant'],r.get('reason')) for r in rows if r['status']=='failed'])
                for row in measured:
                    self.assertGreater(row['cuda']['median_ms'],0)
                    self.assertGreaterEqual(row['peak_allocated_bytes'],row['resident_allocated_bytes'])


if __name__=='__main__':
    unittest.main()
