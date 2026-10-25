"""带注释的逐章 PyTorch 数值辅助函数，所有中间张量保留输入设备。

这些是QDQ/格式模拟，不调用bitsandbytes或原生INT4/FP4 Tensor Core算子。
原有93个机制示例继续使用NumPy，作为独立参照。
"""
import math
import torch


def finite(x):
    if not isinstance(x, torch.Tensor) or not x.numel() or x.is_complex() or not torch.isfinite(x).all():
        raise ValueError('输入必须是非空、有限实数 PyTorch 张量')
    return x.to(torch.float64)


def width(bits, minimum=2, maximum=16):
    if isinstance(bits, bool) or not isinstance(bits, int) or not minimum <= bits <= maximum:
        raise ValueError(f'位宽必须是 {minimum}–{maximum} 的整数')


def symmetric_qdq(x, bits=4, axis=None):
    """axis为共享尺度的归约轴；round使用ties-to-even，全零组尺度为1。"""
    width(bits)
    x = finite(x)
    maximum = x.abs().amax(dim=axis, keepdim=True)
    scale = torch.where(maximum > 0, maximum/(2**(bits-1)-1), torch.ones_like(maximum))
    if not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('尺度下溢或溢出；请缩放输入')
    codes = torch.round(x/scale).clamp(-(2**(bits-1)-1), 2**(bits-1)-1).to(torch.int64)
    return codes*scale, codes, scale


def group_qdq(x, bits=4, group_size=4):
    """按权重行切完整组；形状变化不复制到CPU。"""
    x = finite(x)
    if (isinstance(group_size, bool) or not isinstance(group_size, int) or group_size <= 0
            or x.ndim != 2 or x.shape[1] % group_size):
        raise ValueError('输入必须是二维权重且每行可分为完整正整数组')
    blocks = x.reshape(x.shape[0], -1, group_size)
    return symmetric_qdq(blocks, bits, axis=-1)[0].reshape(x.shape)


def affine_qdq(x, bits=2, axis=None):
    width(bits, 1, 8)
    x = finite(x)
    low = x.amin(dim=axis, keepdim=True).clamp(max=0)
    high = x.amax(dim=axis, keepdim=True).clamp(min=0)
    scale = torch.where(high > low, (high-low)/(2**bits-1), torch.ones_like(high))
    if not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('仿射尺度无效；请缩放输入')
    zero = torch.round(-low/scale).clamp(0, 2**bits-1).to(torch.int64)
    codes = (torch.round(x/scale)+zero).clamp(0, 2**bits-1).to(torch.uint8)
    return codes, scale, zero


def softmax(x):
    return torch.softmax(x, dim=-1)


def hadamard(n, *, device, dtype=torch.float64):
    if not isinstance(n, int) or n <= 0 or n & (n-1):
        raise ValueError('Hadamard阶数须为2的幂')
    matrix = torch.ones((1, 1), device=device, dtype=dtype)
    while matrix.shape[0] < n:
        matrix = torch.cat([torch.cat([matrix, matrix], dim=1),
                            torch.cat([matrix, -matrix], dim=1)], dim=0)
    return matrix/math.sqrt(n)


def fp_levels(kind, x):
    # 码本是静态格式常数，Python列举后一次放到输入设备；查表/距离计算在设备上。
    if kind == 'e2m1':
        positives = [0., .5, 1., 1.5, 2., 3., 4., 6.]
    elif kind == 'e4m3':
        positives = [2.**-6*m/8 if e == 0 else 2.**(e-7)*(1+m/8)
                     for e in range(16) for m in range(8) if not (e == 15 and m == 7)]
    else:
        raise ValueError('本示例只支持 E2M1 和 E4M3')
    return x.new_tensor(sorted(set(positives+[-v for v in positives])))


def nearest(x, levels):
    # 输入有限仍可能在减法中溢出；先检查距离，不能让argmin掩盖Inf/NaN。
    x, levels = finite(x), finite(levels)
    if levels.ndim != 1 or levels.device != x.device:
        raise ValueError('码本必须是一维且与输入位于同一设备')
    distances = (x[..., None]-levels).abs()
    if not torch.isfinite(distances).all():
        raise ValueError('码字距离溢出；请缩放输入')
    # CUDA上的argmin和gather完成码字选择；平局取表中的第一个值。
    return levels[distances.argmin(dim=-1)]


def positive_scale(scale):
    """在除法之前检查尺度；where的未选分支也会执行，不能靠它遮住0/0。"""
    if not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('尺度下溢或溢出；请缩放输入')
    return scale


def fp4_block_qdq(blocks, kind='nv', tensor_scale=None):
    """末轴共享尺度的数学子步骤；物理块长由micro_fp4/二维示例分别约束。"""
    blocks = finite(blocks)
    if blocks.ndim == 0 or kind not in ('nv', 'mx'):
        raise ValueError('选择nv/mx且块输入至少一维')
    if kind == 'mx' and tensor_scale is not None:
        raise ValueError('MX没有额外张量尺度')
    maximum = blocks.abs().amax(dim=-1, keepdim=True)
    if kind == 'nv':
        if tensor_scale is None:
            global_scale = (maximum.max()/(448*6) if maximum.max() > 0
                            else blocks.new_tensor(1.))
        else:
            try:
                global_scale = (tensor_scale.to(device=blocks.device) if isinstance(tensor_scale, torch.Tensor)
                                else torch.as_tensor(tensor_scale, device=blocks.device, dtype=torch.float64))
                global_scale = finite(global_scale)
            except (TypeError, RuntimeError) as exc:
                raise ValueError('张量尺度必须是正的有限实数标量') from exc
            if global_scale.ndim != 0:
                raise ValueError('张量尺度必须是标量')
        positive_scale(global_scale)
        divisor = positive_scale(6*global_scale)
        scale = nearest(maximum/divisor, fp_levels('e4m3', blocks))
        # 零块显式使用1；非零块尺度舍入到0时采用最小E4M3次正规数。
        scale = torch.where(maximum == 0, torch.ones_like(scale), scale.clamp(min=2.**-9))
    else:
        safe = positive_scale(torch.where(maximum > 0, maximum/6, torch.ones_like(maximum)))
        scale = 2.**torch.ceil(torch.log2(safe)).clamp(-127, 127)
        global_scale = blocks.new_tensor(1.)
    decoded_scale = positive_scale(global_scale*scale)
    payload = nearest(blocks/decoded_scale, fp_levels('e2m1', blocks))
    restored = decoded_scale*payload
    if not torch.isfinite(restored).all():
        raise ValueError('FP4重建溢出；请缩放输入')
    return restored, dict(tensor_scale=global_scale, block_scale=scale, payload=payload)


def micro_fp4(x, kind='nv', tensor_scale=None):
    """与CPU参照一致的FP4数值模拟；T4也能执行，不要求原生FP4硬件。"""
    x = finite(x)
    if kind not in ('nv', 'mx') or x.ndim == 0:
        raise ValueError('选择nv/mx且输入至少一维')
    group = 16 if kind == 'nv' else 32
    if x.shape[-1] % group:
        raise ValueError('最后一维必须包含完整微块')
    blocks = x.reshape(*x.shape[:-1], -1, group)
    restored, state = fp4_block_qdq(blocks, kind, tensor_scale)
    return restored.reshape(x.shape), state


class TinyKIVI:
    """K沿时间分组、V沿特征分组；转换全部成功后才提交新状态。"""

    def __init__(self, dim=4, group=2, residual=4, bits=2, *, device):
        width(bits, 1, 8)
        if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in (dim, group, residual)):
            raise ValueError('维度、组长和尾部长必须为正整数')
        if dim % group or residual % group:
            raise ValueError('使用完整组')
        self.dim, self.group, self.residual, self.bits = dim, group, residual, bits
        self.device = torch.device(device)
        self.k_recent, self.v_recent, self.k_chunks, self.v_chunks = [], [], [], []

    @staticmethod
    def decode(chunk):
        codes, scale, zero = chunk
        return scale*(codes.to(torch.float64)-zero)

    def append(self, k, v):
        k, v = finite(k), finite(v)
        if k.shape != (self.dim,) or v.shape != (self.dim,) or k.device != self.device or v.device != self.device:
            raise ValueError('每次K/V须与缓存维度、设备一致')
        recent_k, recent_v = [*self.k_recent, k.clone()], [*self.v_recent, v.clone()]
        chunks_k, chunks_v = [], []
        if len(recent_k) == self.residual:
            rows = torch.stack(recent_k)
            chunks_k = [affine_qdq(rows[i:i+self.group], self.bits, axis=0)
                        for i in range(0, self.residual, self.group)]
            recent_k = []
        if len(recent_v) > self.residual:
            chunks_v = [affine_qdq(recent_v[0].reshape(-1, self.group), self.bits, axis=1)]
            recent_v = recent_v[1:]
        self.k_chunks.extend(chunks_k)
        self.v_chunks.extend(chunks_v)
        self.k_recent, self.v_recent = recent_k, recent_v

    def reconstruct(self):
        keys = [self.decode(c) for c in self.k_chunks]
        values = [self.decode(c).reshape(1, self.dim) for c in self.v_chunks]
        if self.k_recent:
            keys.append(torch.stack(self.k_recent))
        if self.v_recent:
            values.append(torch.stack(self.v_recent))
        empty = torch.empty((0, self.dim), dtype=torch.float64, device=self.device)
        return torch.cat(keys) if keys else empty, torch.cat(values) if values else empty


def int8_matmul(x, w):
    """INT8码，FP64精确整数点积参照；兼容CUDA不支持通用INT32 matmul的设备。

    在本例小维度中，整数和低于2^53，FP64精确承载它；验证后转为INT32。
    这不是INT8 Tensor Core性能实现，不用CPU整数GEMM冒充CUDA。
    """
    x, w = finite(x), finite(w)
    if x.ndim != 2 or w.ndim != 2 or x.shape[1] != w.shape[1] or x.device != w.device:
        raise ValueError('输入和权重必须是同设备、归约维匹配的矩阵')
    if x.shape[1]*127**2 > 2**31-1:
        raise ValueError('示例归约长度超出INT32保守界')
    _, qx, sx = symmetric_qdq(x, 8, axis=1)
    _, qw, sw = symmetric_qdq(w, 8, axis=1)
    accumulator = (qx.to(torch.float64)@qw.to(torch.float64).T).to(torch.int32)
    result = accumulator.to(torch.float64)*(sx*sw.T)
    return result, dict(accumulator=accumulator, sx=sx, sw=sw, qx=qx.to(torch.int8), qw=qw.to(torch.int8))


def split_matmul(x, w, threshold=6.):
    x, w = finite(x), finite(w)
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError('阈值必须正且有限')
    outliers = (x.abs() > threshold).any(dim=0)
    ordinary = x.new_zeros((x.shape[0], w.shape[0]))
    high = torch.zeros_like(ordinary)
    if (~outliers).any():
        ordinary, _ = int8_matmul(x[:, ~outliers], w[:, ~outliers])
    if outliers.any():
        high = x[:, outliers]@w[:, outliers].T
    return ordinary+high, dict(outliers=outliers, y_regular=ordinary, y_outlier=high)


def error_metrics(reference, candidate):
    error = (reference-candidate).square()
    return dict(mse=error.mean(), nmse=error.sum()/reference.square().sum().clamp(min=1e-30))
