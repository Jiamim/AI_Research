"""PyTorch数值路径与原NumPy实现独立比较；有CUDA时执行相同输入的GPU测试。"""
import unittest
import ast
import os
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch

from experiments import demo_torch_kernels as gpu
from experiments import quant_lab as reference
from experiments.technique_kernels import micro_fp4, TinyKIVI
from experiments.llm_int8_lab import int8_matmul, split_matmul
from experiments.demo_runtime import DemoRuntime, same, exact


class TorchKernels(unittest.TestCase):
    def verify(self, device):
        rng = np.random.default_rng(55)
        x, w = rng.normal(size=(5, 32)), rng.normal(size=(3, 32))
        x[0] = 0
        tx, tw = [torch.tensor(v, dtype=torch.float64, device=device) for v in (x, w)]

        def matches(actual, expected):
            self.assertEqual(actual.device, torch.device(device))
            np.testing.assert_allclose(actual.detach().cpu().numpy(), expected, rtol=1e-10, atol=1e-10)

        for bits in (2, 4, 8):
            for axis in (None, 1):
                for actual, expected in zip(gpu.symmetric_qdq(tx, bits, axis), reference.symmetric_qdq(x, bits, axis)):
                    matches(actual, expected)
            matches(gpu.group_qdq(tw, bits, 8), reference.group_qdq(w, bits, 8))
        for kind in ('nv', 'mx'):
            actual, state = gpu.micro_fp4(tx, kind)
            expected, expected_state = micro_fp4(x, kind)
            matches(actual, expected)
            for name in state:
                matches(state[name], expected_state[name])
            # 非空全零输入与很小但可表示的数，比较重建和全部尺度/码字。
            for magnitude in (0., 1e-100):
                values = np.full((2, 32), magnitude)
                decoded, state = gpu.micro_fp4(tx.new_tensor(values), kind)
                expected, expected_state = micro_fp4(values, kind)
                for actual, wanted in [(decoded, expected), *[(state[k], expected_state[k]) for k in state]]:
                    matches(actual, wanted)
                self.assertTrue(bool((state['block_scale'] > 0).all()))
        # 过去会得到有限的负数，或把Inf交给argmin；CPU/CUDA必须与NumPy一同拒绝。
        for value, kind, scale in [(np.finfo(float).max, 'nv', .1), (1., 'nv', 1e-310),
                                    (np.finfo(float).max, 'nv', None),
                                    (np.nextafter(0., 1.), 'nv', None),
                                    (np.nextafter(0., 1.), 'mx', None), (1., 'nv', 1e308)]:
            with self.subTest(value=value, kind=kind, scale=scale):
                values = np.full(32, value)
                with self.assertRaises(ValueError):
                    micro_fp4(values, kind, scale)
                with self.assertRaises(ValueError):
                    gpu.micro_fp4(tx.new_tensor(values), kind, scale)
        for values, levels in [([float('nan')], [0., 1.]), ([float('inf')], [0., 1.]),
                                ([np.finfo(float).max], [-np.finfo(float).max])]:
            with self.assertRaises(ValueError):
                gpu.nearest(tx.new_tensor(values), tx.new_tensor(levels))
        matches(gpu.hadamard(8, device=device), reference.hadamard(8))
        matches(gpu.int8_matmul(tx, tw)[0], int8_matmul(x, w)[0])
        for threshold in (.01, 1., 100.):
            matches(gpu.split_matmul(tx, tw, threshold)[0], split_matmul(x, w, threshold)[0])
        cache = gpu.TinyKIVI(dim=4, group=2, residual=4, device=device)
        baseline = TinyKIVI(dim=4, group=2, residual=4)
        for k, v in zip(rng.normal(size=(9, 4)), rng.normal(size=(9, 4))):
            cache.append(torch.tensor(k, device=device), torch.tensor(v, device=device))
            baseline.append(k, v)
            for actual, expected in zip(cache.reconstruct(), baseline.reconstruct()):
                matches(actual, expected)
        before = tuple(t.clone() for t in cache.reconstruct())
        with self.assertRaises(ValueError):
            cache.append(tx.new_full((4,), float('nan')), tx.new_ones(4))
        for actual, expected in zip(cache.reconstruct(), before):
            self.assertTrue(torch.equal(actual, expected))

    def test_cpu_matches_numpy(self):
        self.verify('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), '需要CUDA设备，CPU通过不能冒充GPU验证')
    def test_cuda_matches_numpy(self):
        self.verify('cuda:0')

    def test_explicit_cuda_never_falls_back(self):
        with patch.object(torch.cuda, 'is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, '不会自动降级'):
                DemoRuntime('cuda')
            self.assertEqual(DemoRuntime('auto').device.type, 'cpu')

    def test_invalid_device_and_quantizer_inputs(self):
        for device in ('mps', 'cpu:1', 'invalid'):
            with self.assertRaises(ValueError):
                DemoRuntime(device)
        for values in ([], [float('nan')], [float('inf')]):
            with self.assertRaises(ValueError):
                gpu.symmetric_qdq(torch.tensor(values))

    def test_comparisons_preserve_types_and_shapes(self):
        for device in ['cpu'] + (['cuda:0'] if torch.cuda.is_available() else []):
            self.assertFalse(exact(torch.tensor([0], device=device), [.9]))
            self.assertFalse(exact(torch.tensor([True], device=device), [1]))
            self.assertFalse(same(torch.tensor([1., 1.], device=device), [1.]))
            self.assertFalse(same(torch.tensor([1., 1.], device=device), 1.))
            self.assertTrue(same(torch.tensor([1., 1.], device=device), 1., allow_scalar_broadcast=True))
            self.assertTrue(same(torch.tensor([1], dtype=torch.uint8, device=device), [1.]))
            self.assertTrue(exact(torch.tensor([127], dtype=torch.int8, device=device), [127]))
            self.assertFalse(exact(torch.tensor([-1], dtype=torch.int8, device=device), [255]))
            self.assertFalse(same(torch.tensor([2**53+1], device=device), [float(2**53)]))
            self.assertFalse(exact(torch.tensor([2**53+1], device=device), [2**53]))
            self.assertFalse(same(torch.tensor([float('nan')], device=device), [float('nan')]))

    def test_square_scales_zero_blocks_and_transpose(self):
        path = Path(__file__).resolve().parents[1]/'experiments/demos/training_transpose.py'
        function = next(n for n in ast.parse(path.read_text()).body
                        if isinstance(n, ast.FunctionDef) and n.name == 'square_qdq')
        env = dict(torch=torch, finite=gpu.finite, fp4_block_qdq=gpu.fp4_block_qdq)
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), env)
        for device in ['cpu'] + (['cuda:0'] if torch.cuda.is_available() else []):
            values = torch.arange(16*48, device=device, dtype=torch.float64).reshape(16, 48)/100
            values[:, :16] = 0
            decoded = env['square_qdq'](values)
            self.assertTrue(torch.equal(decoded[:, :16], torch.zeros_like(decoded[:, :16])))
            torch.testing.assert_close(decoded.T, env['square_qdq'](values.T.contiguous()), rtol=0, atol=0)
            torch.testing.assert_close(env['square_qdq'](torch.zeros_like(values)), torch.zeros_like(values), rtol=0, atol=0)
            for invalid in (values[:, :17], values.new_full(values.shape, float('nan')),
                            values.new_full(values.shape, torch.finfo(torch.float64).max)):
                with self.assertRaises(ValueError):
                    env['square_qdq'](invalid)


class CudaChapters(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), '需要在Colab GPU执行全部章节')
    def test_all_chapters_use_cuda(self):
        from experiments.chapter_demo_lab import run_all
        from experiments.chapter_demo_catalog import DEMOS
        records = run_all(device='cuda', check_docs=os.environ.get('QUANT_DEMO_PRACTICE') != '1')
        self.assertEqual(len(records), len(DEMOS))
        for record in records:
            self.assertEqual(record['runtime']['device'], 'cuda:0')
            self.assertGreater(record['checked_tensor_count'], 0)


if __name__ == '__main__':
    unittest.main()
