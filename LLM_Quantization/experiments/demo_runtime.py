"""逐章 demo 的显式设备与数据准备；数值计算使用 PyTorch CPU/CUDA。"""
from contextlib import contextmanager
from contextvars import ContextVar
import os

import numpy as np
import torch

_ACTIVE = ContextVar('chapter_demo_runtime', default=None)


class DemoRuntime:
    """只支持 CPU/CUDA；明确请求 CUDA 时不允许悄悄退回 CPU。"""

    def __init__(self, requested='auto'):
        if requested == 'auto':
            requested = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        try:
            device = torch.device(requested)
        except (RuntimeError, ValueError) as exc:
            raise ValueError('设备必须是 auto、cpu、cuda 或 cuda:N') from exc
        if device.type not in ('cpu', 'cuda') or (device.type == 'cpu' and device.index is not None):
            raise ValueError('本实操只支持 cpu 或 cuda:N')
        if device.type == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError('已请求 CUDA，但当前 PyTorch 无可用 GPU。'
                                   '请在 Colab 选择 GPU 运行时并检查 torch.cuda.is_available()；'
                                   '不会自动降级为 CPU。')
            index = 0 if device.index is None else device.index
            if index >= torch.cuda.device_count():
                raise ValueError(f'CUDA 设备 {index} 不存在')
            device = torch.device('cuda', index)
        self.device = device
        self.dtype = torch.float64  # 小规模数学核验优先数值稳定；不是性能benchmark。
        self.created_tensors = 0

    def tensor(self, values, dtype=None):
        """数据准备阶段明确搬到目标设备；整数索引须显式给dtype。"""
        result = torch.as_tensor(values, dtype=dtype or self.dtype, device=self.device)
        self.created_tensors += 1
        return result

    def rng(self, seed):
        # 在CPU用相同NumPy随机流生成fixture，然后一次搬到目标设备。
        # 这样CPU/CUDA比较的是同一份输入，而非两套不同的随机数。
        runtime = self
        generator = np.random.default_rng(seed)

        class InputGenerator:
            def normal(self, size):
                return runtime.tensor(generator.normal(size=size))

            def choice(self, values, size):
                return runtime.tensor(generator.choice(values, size=size))

            def integers(self, low, high, size):
                return runtime.tensor(generator.integers(low, high, size=size), dtype=torch.int64)

        return InputGenerator()

    def synchronize(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def metadata(self):
        return dict(device=str(self.device), dtype=str(self.dtype), torch_version=torch.__version__,
                    cuda_build=torch.version.cuda,
                    gpu_name=torch.cuda.get_device_name(self.device) if self.device.type == 'cuda' else None,
                    input_tensors_created=self.created_tensors,
                    input_preparation='Seeded NumPy fixtures on CPU, copied once to the selected device',
                    compute='PyTorch tensors on the selected device; file I/O and printing on CPU')


@contextmanager
def use_runtime(runtime):
    token = _ACTIVE.set(runtime)
    try:
        yield runtime
    finally:
        _ACTIVE.reset(token)


def get_runtime():
    """runner传入的设备优先；直接运行模块/复制代码时读取环境变量。"""
    active = _ACTIVE.get()
    if active is not None:
        return active
    runtime = DemoRuntime(os.environ.get('QUANT_DEMO_DEVICE', 'auto'))
    print('实操计算设备:', runtime.device, '；核验dtype:', runtime.dtype)
    return runtime


def _comparison_tensors(actual, expected):
    device = next((v.device for v in (actual, expected) if isinstance(v, torch.Tensor)), torch.device('cpu'))
    # Python字面量通过NumPy推断类型，保留Python float的64位精度；不转成actual的类型。
    return tuple(v.to(device=device) if isinstance(v, torch.Tensor)
                 else torch.as_tensor(np.asarray(v), device=device) for v in (actual, expected))


def same(actual, expected, atol=1e-8, rtol=1e-5, *, allow_scalar_broadcast=False):
    """浮点近似比较默认要求同形状；需要标量广播时必须显式声明。"""
    actual, expected = _comparison_tensors(actual, expected)
    if actual.shape != expected.shape:
        if not allow_scalar_broadcast or (actual.ndim != 0 and expected.ndim != 0):
            return False
        actual, expected = torch.broadcast_tensors(actual, expected)
    values = (actual, expected)
    if not any(v.is_floating_point() or v.is_complex() for v in values):
        return exact(actual, expected)
    # 混合整数/浮点时，拒绝可能在float64中丢位的大整数，避免假相等。
    for value in values:
        if not value.is_floating_point() and not value.is_complex() and value.dtype != torch.bool:
            if (value.to(torch.float64).abs() >= 2**53).any():
                return False
    dtype = torch.complex128 if any(v.is_complex() for v in values) else torch.float64
    return torch.allclose(actual.to(dtype), expected.to(dtype), atol=atol, rtol=rtol)


def exact(actual, expected):
    """码字或同类数值精确比较：检查形状，禁止整数/浮点/布尔之间隐式转换。"""
    actual, expected = _comparison_tensors(actual, expected)
    category = lambda v: (v.dtype == torch.bool, v.is_floating_point(), v.is_complex())
    if actual.shape != expected.shape or category(actual) != category(expected):
        return False
    try:
        dtype = torch.promote_types(actual.dtype, expected.dtype)
    except RuntimeError:
        return False  # 如uint64与int64，没有无损的公共整数类型。
    return torch.equal(actual.to(dtype), expected.to(dtype))
