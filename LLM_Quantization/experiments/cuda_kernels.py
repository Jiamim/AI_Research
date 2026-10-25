"""CUDA 实验的可移植参照：真实 INT4 字节存储，PyTorch 显式解码。

这里没有融合的 INT4 GEMM。解码结果是 FP16，矩阵乘使用浮点后端。
函数同样支持 CPU，便于用独立手算检查位布局和缓存分组轴。
"""
from dataclasses import dataclass
import torch


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return value


def finite_matrix(weight):
    if (not isinstance(weight, torch.Tensor) or weight.ndim != 2 or not weight.numel()
            or not weight.is_floating_point() or not torch.isfinite(weight).all()):
        raise ValueError('Expected a nonempty finite floating-point matrix')


@dataclass
class PackedInt4:
    payload: torch.Tensor
    scales: torch.Tensor
    rows: int
    columns: int
    group_size: int

    @property
    def storage_bytes(self):
        # payload 与实际 FP16 scales；不含 Python 对象、文件头与 allocator 对齐。
        return self.payload.nbytes + self.scales.nbytes


def pack_int4(weight, group_size=128):
    """每行沿输入轴分组；[-7,7] 对称码，补码 nibble，偶数列存低4位。"""
    finite_matrix(weight)
    positive_integer(group_size, 'group_size')
    rows, columns = weight.shape
    if group_size % 2 or columns % group_size:
        raise ValueError('Use even group_size and complete groups along the last axis')
    blocks = weight.float().reshape(rows, -1, group_size)
    maximum = blocks.abs().amax(dim=-1, keepdim=True)
    # 编码使用真正保存的 FP16 scale；极小非零尺度钳到 FP16 最小次正规数。
    scales = torch.where(maximum > 0, (maximum/7).clamp_min(2.**-24), 1.).half()
    if not torch.isfinite(scales).all():
        raise ValueError('Scale cannot be represented as finite FP16')
    codes = torch.round(blocks/scales.float()).clamp(-7, 7).to(torch.int16).reshape(rows, columns)
    unsigned = (codes & 15).to(torch.uint8)
    payload = (unsigned[:, 0::2] | (unsigned[:, 1::2] << 4)).contiguous()
    return PackedInt4(payload, scales.squeeze(-1).contiguous(), rows, columns, group_size)


def unpack_int4(packed):
    """全部解码在 payload 所在设备执行；计时范围包含临时张量分配。"""
    unsigned = torch.stack((packed.payload & 15, packed.payload >> 4), dim=-1)
    unsigned = unsigned.reshape(packed.rows, -1).to(torch.int16)
    signed = torch.where(unsigned >= 8, unsigned-16, unsigned)
    blocks = signed.reshape(packed.rows, -1, packed.group_size).float()
    return (blocks*packed.scales.float().unsqueeze(-1)).reshape(packed.rows, packed.columns).half()


@dataclass
class PackedKV:
    key: PackedInt4 | None
    value: PackedInt4 | None
    recent_key: torch.Tensor
    recent_value: torch.Tensor
    shape: tuple
    prefix: int

    @property
    def storage_bytes(self):
        return (sum(p.storage_bytes for p in (self.key, self.value) if p is not None)
                + self.recent_key.nbytes + self.recent_value.nbytes)


def pack_kv(key, value, group_size=32, residual=128):
    """静态缓存快照：K 沿时间轴、V 沿特征轴量化，最近 residual 个 token 保留 FP16。

    本函数不实现流式追加/分页；对应动态生命周期见原有 TinyKIVI 实操。
    """
    positive_integer(group_size, 'group_size')
    if isinstance(residual, bool) or not isinstance(residual, int) or residual < 0:
        raise ValueError('residual must be a nonnegative integer')
    if (key.ndim != 4 or key.shape != value.shape or key.device != value.device
            or key.dtype != torch.float16 or value.dtype != torch.float16):
        raise ValueError('K/V must be FP16 [batch, heads, time, dim] on the same device')
    if not key.numel() or not torch.isfinite(key).all() or not torch.isfinite(value).all():
        raise ValueError('K/V must be nonempty and finite')
    b, h, time, dim = key.shape
    prefix = time-residual
    if residual > time or group_size % 2 or dim % group_size or prefix % group_size:
        raise ValueError('Require residual <= time and complete K-time / V-feature groups')
    # clone 避免切片继续持有整个原始缓存的 storage。
    recent_k = key[:, :, prefix:, :].clone().contiguous()
    recent_v = value[:, :, prefix:, :].clone().contiguous()
    pk = pv = None
    if prefix:
        pk = pack_int4(key[:, :, :prefix, :].permute(0, 1, 3, 2).reshape(b*h*dim, prefix), group_size)
        pv = pack_int4(value[:, :, :prefix, :].reshape(b*h*prefix, dim), group_size)
    return PackedKV(pk, pv, recent_k, recent_v, tuple(key.shape), prefix)


def unpack_kv(cache):
    if not cache.prefix:
        return cache.recent_key, cache.recent_value
    b, h, _, dim = cache.shape
    key = unpack_int4(cache.key).reshape(b, h, dim, cache.prefix).permute(0, 1, 3, 2)
    value = unpack_int4(cache.value).reshape(b, h, cache.prefix, dim)
    return (torch.cat((key, cache.recent_key), dim=2),
            torch.cat((value, cache.recent_value), dim=2))


def error_metrics(reference, candidate):
    """只在计时外调用；NMSE 是总平方误差/参考能量，不是逐元素相对误差。"""
    if reference.shape != candidate.shape or reference.device != candidate.device:
        raise ValueError('Reference and candidate must have matching shapes and devices')
    r, c = reference.float(), candidate.float()
    if not torch.isfinite(r).all() or not torch.isfinite(c).all():
        raise ValueError('Non-finite numerical result')
    difference = c-r
    energy = r.double().square().sum().item()
    squared = difference.double().square().sum().item()
    return dict(mse=squared/r.numel(), nmse=squared/max(energy, 1e-30),
                max_abs=difference.abs().max().item())
