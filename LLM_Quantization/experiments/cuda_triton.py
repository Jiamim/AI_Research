"""Triton 融合内核：INT4 码在寄存器内解码，不先写回完整 FP16 矩阵或缓存。

每个内核都有同模板 FP16 分支，只改变数据读取与解码，便于把差异归因于字节数和解码开销。
数据布局与 cuda_kernels.pack_int4 / pack_kv 完全相同。GEMV 和组内对齐的 GEMM 先对整数码求和，
再乘一次组尺度（常见 W4A16 内核做法）；KV 与其余 GEMM 分块按 unpack_int4 先乘尺度并舍入到 FP16。
两种写法与解码后的 FP16 权重只差 FP16 舍入量级，正确性对照用容差而非逐位相等。
CPU 上可用 TRITON_INTERPRET=1 解释执行核验数值；解释器结果不代表 GPU 性能。
"""
import re

import torch
import triton
import triton.language as tl

from experiments.cuda_kernels import PackedInt4, PackedKV

# 候选配置在正式计时外由 CudaBench.tune 选择，并写入报告。
# 小 M 时输出块太少，SPLIT_K>1 沿 K 切分以占满 SM，各段 FP32 原子累加后再转 FP16。
GEMV_CONFIGS = (dict(BLOCK_N=16, BLOCK_K=256, num_warps=4, SPLIT_K=1),
                dict(BLOCK_N=32, BLOCK_K=128, num_warps=4, SPLIT_K=1),
                dict(BLOCK_N=64, BLOCK_K=128, num_warps=8, SPLIT_K=1),
                dict(BLOCK_N=8, BLOCK_K=256, num_warps=2, SPLIT_K=4),
                dict(BLOCK_N=16, BLOCK_K=256, num_warps=4, SPLIT_K=2),
                dict(BLOCK_N=16, BLOCK_K=256, num_warps=4, SPLIT_K=4),
                dict(BLOCK_N=32, BLOCK_K=256, num_warps=4, SPLIT_K=4))
GEMM_CONFIGS = (dict(BLOCK_M=16, BLOCK_N=64, BLOCK_K=64, num_warps=4, num_stages=3, SPLIT_K=1),
                dict(BLOCK_M=16, BLOCK_N=32, BLOCK_K=128, num_warps=4, num_stages=3, SPLIT_K=2),
                dict(BLOCK_M=16, BLOCK_N=64, BLOCK_K=64, num_warps=4, num_stages=3, SPLIT_K=4),
                dict(BLOCK_M=32, BLOCK_N=64, BLOCK_K=64, num_warps=4, num_stages=3, SPLIT_K=1),
                dict(BLOCK_M=32, BLOCK_N=64, BLOCK_K=64, num_warps=4, num_stages=3, SPLIT_K=4),
                dict(BLOCK_M=64, BLOCK_N=64, BLOCK_K=64, num_warps=4, num_stages=3, SPLIT_K=1),
                dict(BLOCK_M=64, BLOCK_N=128, BLOCK_K=32, num_warps=4, num_stages=3, SPLIT_K=1),
                dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, num_warps=8, num_stages=3, SPLIT_K=1))
KV_CONFIGS = (dict(BLOCK_T=32, num_warps=4), dict(BLOCK_T=64, num_warps=4), dict(BLOCK_T=128, num_warps=8))


@triton.jit
def _nibbles(byte):
    # 低4位是偶数列/偶数token，高4位是奇数列/奇数token；>=8 的补码表示负数。
    low = byte & 15
    high = (byte >> 4) & 15
    return tl.where(low >= 8, low - 16, low), tl.where(high >= 8, high - 16, high)


@triton.jit
def _store(y_ptr, offsets, acc, mask, SPLIT_K: tl.constexpr):
    # 不切分时直接写 FP16；切分时 y_ptr 指向预清零的 FP32 缓冲区，各段原子累加。
    if SPLIT_K == 1:
        tl.store(y_ptr + offsets, acc.to(tl.float16), mask=mask)
    else:
        tl.atomic_add(y_ptr + offsets, acc, mask=mask)


@triton.jit
def _gemv_kernel(x_ptr, w_ptr, s_ptr, y_ptr, N, K, stride_w, stride_s,
                 GROUP: tl.constexpr, PACKED: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                 SPLIT_K: tl.constexpr):
    """M=1：每个 program 计算 BLOCK_N 个输出在一段 K 上的部分和；不使用 tl.dot。"""
    rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    valid = rows < N
    span = K // SPLIT_K
    k_start = tl.program_id(1) * span
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(k_start, k_start + span, BLOCK_K):
        if PACKED:
            pair = tl.arange(0, BLOCK_K // 2)
            byte = tl.load(w_ptr + rows[:, None] * stride_w + k0 // 2 + pair[None, :],
                           mask=valid[:, None], other=0).to(tl.int32)
            low, high = _nibbles(byte)
            even = tl.load(x_ptr + k0 + 2 * pair).to(tl.float32)
            odd = tl.load(x_ptr + k0 + 2 * pair + 1).to(tl.float32)
            partial = low.to(tl.float32) * even[None, :] + high.to(tl.float32) * odd[None, :]
            # BLOCK_K 是 GROUP 的整数倍：组内先求和，每组只乘一次尺度。
            grouped = tl.sum(tl.reshape(partial, (BLOCK_N, BLOCK_K // GROUP, GROUP // 2)), axis=2)
            scale = tl.load(s_ptr + rows[:, None] * stride_s + k0 // GROUP + tl.arange(0, BLOCK_K // GROUP)[None, :],
                            mask=valid[:, None], other=0.).to(tl.float32)
            acc += tl.sum(grouped * scale, axis=1)
        else:
            cols = k0 + tl.arange(0, BLOCK_K)
            weight = tl.load(w_ptr + rows[:, None] * stride_w + cols[None, :],
                             mask=valid[:, None], other=0.).to(tl.float32)
            acc += tl.sum(weight * tl.load(x_ptr + cols).to(tl.float32)[None, :], axis=1)
    _store(y_ptr, rows, acc, valid, SPLIT_K)


@triton.jit
def _gemm_kernel(x_ptr, w_ptr, s_ptr, y_ptr, M, N, K, stride_x, stride_w, stride_s, stride_y,
                 GROUP: tl.constexpr, PACKED: tl.constexpr,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr):
    """M>1：分块 tl.dot。INT4 分支把偶数/奇数列拆成两个 K/2 的乘法，避免寄存器内重排。"""
    rm = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    span = K // SPLIT_K
    k_start = tl.program_id(2) * span
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(k_start, k_start + span, BLOCK_K):
        if PACKED:
            pair = tl.arange(0, BLOCK_K // 2)
            x_even = tl.load(x_ptr + rm[:, None] * stride_x + k0 + 2 * pair[None, :],
                             mask=rm[:, None] < M, other=0.)
            x_odd = tl.load(x_ptr + rm[:, None] * stride_x + k0 + 2 * pair[None, :] + 1,
                            mask=rm[:, None] < M, other=0.)
            byte = tl.load(w_ptr + rn[None, :] * stride_w + k0 // 2 + pair[:, None],
                           mask=rn[None, :] < N, other=0).to(tl.int32)
            low, high = _nibbles(byte)
            if GROUP % BLOCK_K == 0:
                # 整个 K 分块落在同一组：先用整数码做乘法（-7..7 在 FP16 中精确），再乘每列尺度。
                scale = tl.load(s_ptr + rn * stride_s + k0 // GROUP, mask=rn < N, other=0.).to(tl.float32)
                part = tl.dot(x_even, low.to(tl.float16)) + tl.dot(x_odd, high.to(tl.float16))
                acc += part * scale[None, :]
            else:
                scale = tl.load(s_ptr + rn[None, :] * stride_s + (k0 + 2 * pair[:, None]) // GROUP,
                                mask=rn[None, :] < N, other=0.).to(tl.float32)
                acc += tl.dot(x_even, (low.to(tl.float32) * scale).to(tl.float16))
                acc += tl.dot(x_odd, (high.to(tl.float32) * scale).to(tl.float16))
        else:
            cols = k0 + tl.arange(0, BLOCK_K)
            x_tile = tl.load(x_ptr + rm[:, None] * stride_x + cols[None, :], mask=rm[:, None] < M, other=0.)
            w_tile = tl.load(w_ptr + rn[None, :] * stride_w + cols[:, None], mask=rn[None, :] < N, other=0.)
            acc += tl.dot(x_tile, w_tile)
    _store(y_ptr, rm[:, None] * stride_y + rn[None, :], acc, (rm[:, None] < M) & (rn[None, :] < N), SPLIT_K)


def candidate_configs(m, k, group=None):
    """M=1 用 GEMV 模板；其余用 tl.dot，BLOCK_M 不超过 M 的二次幂上界（至少16）。

    INT4 的 GEMV 要求 BLOCK_K 是组大小的整数倍；FP16 分支传 group=None。
    """
    if m == 1:
        pool = [c for c in GEMV_CONFIGS if group is None or c['BLOCK_K'] % group == 0]
    else:
        limit = max(16, 1 << (m - 1).bit_length())
        pool = [c for c in GEMM_CONFIGS if c['BLOCK_M'] <= limit]
    return [dict(c) for c in pool if k % (c['BLOCK_K'] * c['SPLIT_K']) == 0]


def _launch(x, weight, scales, n, k, group, packed, config, return_kernel):
    if x.dtype != torch.float16 or x.ndim != 2 or x.shape[1] != k or not x.is_contiguous():
        raise ValueError('x must be contiguous FP16 [M, K] matching the weight')
    if weight.device != x.device or scales.device != x.device:
        raise ValueError('Inputs and weights must be on the same device')
    block_k, split = config['BLOCK_K'], config.get('SPLIT_K', 1)
    if block_k % 2 or k % (block_k * split):
        raise ValueError('K must be divisible by an even BLOCK_K times SPLIT_K')
    m = x.shape[0]
    # 切分时先在 FP32 缓冲区累加；清零与最后的类型转换都计入调用时间。
    out = torch.empty((m, n), dtype=torch.float16, device=x.device) if split == 1 else \
        torch.zeros((m, n), dtype=torch.float32, device=x.device)
    if m == 1:
        if packed and block_k % group:
            raise ValueError('INT4 GEMV needs BLOCK_K to be a multiple of group_size')
        grid = (triton.cdiv(n, config['BLOCK_N']), split)
        kernel = _gemv_kernel[grid](x, weight, scales, out, n, k, weight.stride(0), scales.stride(0),
                                    GROUP=group, PACKED=packed, BLOCK_N=config['BLOCK_N'], BLOCK_K=block_k,
                                    SPLIT_K=split, num_warps=config['num_warps'])
    else:
        if block_k < 32:
            raise ValueError('tl.dot needs BLOCK_K/2 >= 16')
        grid = (triton.cdiv(m, config['BLOCK_M']), triton.cdiv(n, config['BLOCK_N']), split)
        kernel = _gemm_kernel[grid](x, weight, scales, out, m, n, k, x.stride(0), weight.stride(0),
                                    scales.stride(0), out.stride(0), GROUP=group, PACKED=packed,
                                    BLOCK_M=config['BLOCK_M'], BLOCK_N=config['BLOCK_N'], BLOCK_K=block_k,
                                    SPLIT_K=split, num_warps=config['num_warps'], num_stages=config['num_stages'])
    y = out if split == 1 else out.to(torch.float16)
    return (y, kernel) if return_kernel else y


def w4a16_matmul(x, packed: PackedInt4, config, return_kernel=False):
    """y = x @ decode(packed)^T；只读取 uint8 码和 FP16 组尺度。"""
    return _launch(x, packed.payload, packed.scales, packed.rows, packed.columns,
                   packed.group_size, True, config, return_kernel)


def fp16_matmul(x, weight, config, return_kernel=False):
    """同模板 FP16 对照：分块、网格与累加相同，只把权重读取换成 FP16。"""
    if weight.dtype != torch.float16 or weight.ndim != 2 or not weight.is_contiguous():
        raise ValueError('weight must be contiguous FP16 [N, K]')
    return _launch(x, weight, weight, weight.shape[0], weight.shape[1], config['BLOCK_K'], False,
                   config, return_kernel)


def tensor_core_mma(kernel):
    """读取编译后的 PTX：True 表示含 mma/wgmma，False 表示没有，None 表示无法读取（如解释器）。"""
    asm = getattr(kernel, 'asm', None)
    ptx = asm.get('ptx') if hasattr(asm, 'get') else None
    if not isinstance(ptx, str):
        return None
    return bool(re.search(r'\bmma\.sync|\bwgmma\.mma_async', ptx))


@triton.jit
def _kv_int4_partial(q_ptr, kp_ptr, ks_ptr, vp_ptr, vs_ptr, m_ptr, l_ptr, o_ptr,
                     prefix, chunk, split_base, splits_total, sm_scale,
                     stride_kp, stride_ks, stride_vp, stride_vs,
                     D: tl.constexpr, GROUP_K: tl.constexpr, GROUP_V: tl.constexpr, BLOCK_T: tl.constexpr):
    """一个 (batch*head, split) 处理一段 INT4 前缀：K 沿时间分组，V 沿特征分组，在线 softmax。"""
    bh = tl.program_id(0)
    split = tl.program_id(1)
    start = split * chunk
    end = tl.minimum(start + chunk, prefix)
    dim = tl.arange(0, D)
    half_d = tl.arange(0, D // 2)
    pair = tl.arange(0, BLOCK_T // 2)
    k_groups = tl.arange(0, BLOCK_T // GROUP_K)
    v_groups = tl.arange(0, D // GROUP_V)
    q = tl.load(q_ptr + bh * D + dim).to(tl.float32) * sm_scale
    m_i = tl.max(tl.full((BLOCK_T // 2,), float('-inf'), tl.float32), axis=0)
    l_i = tl.sum(tl.zeros((BLOCK_T // 2,), tl.float32), axis=0)
    acc_even = tl.zeros((D // 2,), tl.float32)
    acc_odd = tl.zeros((D // 2,), tl.float32)
    k_rows = bh * D + dim
    for t0 in range(start, end, BLOCK_T):
        t_even = t0 + 2 * pair
        live = t_even < end  # prefix 与 chunk 都是偶数，一对 token 不会跨越边界
        # K：每个通道一行，沿时间每 GROUP_K 个 token 共享尺度；分块与组对齐。
        kb = tl.load(kp_ptr + k_rows[:, None] * stride_kp + t_even[None, :] // 2,
                     mask=live[None, :], other=0).to(tl.int32)
        ks = tl.load(ks_ptr + k_rows[:, None] * stride_ks + t0 // GROUP_K + k_groups[None, :],
                     mask=(t0 + k_groups * GROUP_K < end)[None, :], other=0.).to(tl.float32)
        k_even, k_odd = _nibbles(kb)
        ke = tl.reshape(tl.reshape(k_even.to(tl.float32), (D, BLOCK_T // GROUP_K, GROUP_K // 2))
                        * ks[:, :, None], (D, BLOCK_T // 2))
        ko = tl.reshape(tl.reshape(k_odd.to(tl.float32), (D, BLOCK_T // GROUP_K, GROUP_K // 2))
                        * ks[:, :, None], (D, BLOCK_T // 2))
        s_even = tl.sum(q[:, None] * ke.to(tl.float16).to(tl.float32), axis=0)
        s_odd = tl.sum(q[:, None] * ko.to(tl.float16).to(tl.float32), axis=0)
        s_even = tl.where(live, s_even, float('-inf'))
        s_odd = tl.where(live, s_odd, float('-inf'))
        m_new = tl.maximum(m_i, tl.maximum(tl.max(s_even, axis=0), tl.max(s_odd, axis=0)))
        alpha = tl.exp(m_i - m_new)
        p_even = tl.exp(s_even - m_new)
        p_odd = tl.exp(s_odd - m_new)
        l_i = l_i * alpha + tl.sum(p_even, axis=0) + tl.sum(p_odd, axis=0)
        # V：每个 token 一行，沿特征每 GROUP_V 个值共享尺度。
        v_even = bh * prefix + t_even
        vb_e = tl.load(vp_ptr + v_even[:, None] * stride_vp + half_d[None, :], mask=live[:, None], other=0).to(tl.int32)
        vb_o = tl.load(vp_ptr + (v_even + 1)[:, None] * stride_vp + half_d[None, :], mask=live[:, None], other=0).to(tl.int32)
        vs_e = tl.load(vs_ptr + v_even[:, None] * stride_vs + v_groups[None, :], mask=live[:, None], other=0.).to(tl.float32)
        vs_o = tl.load(vs_ptr + (v_even + 1)[:, None] * stride_vs + v_groups[None, :], mask=live[:, None], other=0.).to(tl.float32)
        ve_lo, ve_hi = _nibbles(vb_e)
        vo_lo, vo_hi = _nibbles(vb_o)
        ve_lo = tl.reshape(tl.reshape(ve_lo.to(tl.float32), (BLOCK_T // 2, D // GROUP_V, GROUP_V // 2))
                           * vs_e[:, :, None], (BLOCK_T // 2, D // 2))
        ve_hi = tl.reshape(tl.reshape(ve_hi.to(tl.float32), (BLOCK_T // 2, D // GROUP_V, GROUP_V // 2))
                           * vs_e[:, :, None], (BLOCK_T // 2, D // 2))
        vo_lo = tl.reshape(tl.reshape(vo_lo.to(tl.float32), (BLOCK_T // 2, D // GROUP_V, GROUP_V // 2))
                           * vs_o[:, :, None], (BLOCK_T // 2, D // 2))
        vo_hi = tl.reshape(tl.reshape(vo_hi.to(tl.float32), (BLOCK_T // 2, D // GROUP_V, GROUP_V // 2))
                           * vs_o[:, :, None], (BLOCK_T // 2, D // 2))
        acc_even = acc_even * alpha + tl.sum(p_even[:, None] * ve_lo.to(tl.float16).to(tl.float32)
                                             + p_odd[:, None] * vo_lo.to(tl.float16).to(tl.float32), axis=0)
        acc_odd = acc_odd * alpha + tl.sum(p_even[:, None] * ve_hi.to(tl.float16).to(tl.float32)
                                           + p_odd[:, None] * vo_hi.to(tl.float16).to(tl.float32), axis=0)
        m_i = m_new
    slot = bh * splits_total + split_base + split
    tl.store(m_ptr + slot, m_i)
    tl.store(l_ptr + slot, l_i)
    tl.store(o_ptr + slot * D + 2 * half_d, acc_even)
    tl.store(o_ptr + slot * D + 2 * half_d + 1, acc_odd)


@triton.jit
def _kv_fp16_partial(q_ptr, k_ptr, v_ptr, m_ptr, l_ptr, o_ptr, length, chunk, split_base, splits_total, sm_scale,
                     D: tl.constexpr, BLOCK_T: tl.constexpr):
    """同模板 FP16 分段：读取连续 [B,H,T,D] 缓存；也用于 INT4 路径的 FP16 最近窗口。"""
    bh = tl.program_id(0)
    split = tl.program_id(1)
    start = split * chunk
    end = tl.minimum(start + chunk, length)
    dim = tl.arange(0, D)
    token = tl.arange(0, BLOCK_T)
    q = tl.load(q_ptr + bh * D + dim).to(tl.float32) * sm_scale
    m_i = tl.max(tl.full((BLOCK_T,), float('-inf'), tl.float32), axis=0)
    l_i = tl.sum(tl.zeros((BLOCK_T,), tl.float32), axis=0)
    acc = tl.zeros((D,), tl.float32)
    for t0 in range(start, end, BLOCK_T):
        t = t0 + token
        live = t < end
        rows = bh * length + t
        k = tl.load(k_ptr + rows[:, None] * D + dim[None, :], mask=live[:, None], other=0.).to(tl.float32)
        s = tl.where(live, tl.sum(k * q[None, :], axis=1), float('-inf'))
        m_new = tl.maximum(m_i, tl.max(s, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        v = tl.load(v_ptr + rows[:, None] * D + dim[None, :], mask=live[:, None], other=0.).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new
    slot = bh * splits_total + split_base + split
    tl.store(m_ptr + slot, m_i)
    tl.store(l_ptr + slot, l_i)
    tl.store(o_ptr + slot * D + dim, acc)


@triton.jit
def _kv_merge(m_ptr, l_ptr, o_ptr, out_ptr, splits, D: tl.constexpr, BLOCK_S: tl.constexpr):
    """分段 softmax 合并：各段按 exp(m_s - max m) 重新加权，不能直接平均各段输出。"""
    bh = tl.program_id(0)
    s = tl.arange(0, BLOCK_S)
    live = s < splits
    m = tl.load(m_ptr + bh * splits + s, mask=live, other=float('-inf'))
    l = tl.load(l_ptr + bh * splits + s, mask=live, other=0.)
    weight = tl.exp(m - tl.max(m, axis=0))
    dim = tl.arange(0, D)
    o = tl.load(o_ptr + (bh * splits + s)[:, None] * D + dim[None, :], mask=live[:, None], other=0.)
    out = tl.sum(o * weight[:, None], axis=0) / tl.sum(l * weight, axis=0)
    tl.store(out_ptr + bh * D + dim, out.to(tl.float16))


def split_count(length, heads_total, sm_count, block_t):
    """让 B*H*splits 约为两倍 SM 数，同时每段至少一个 tile。"""
    target = max(1, -(-2 * sm_count // heads_total))
    return max(1, min(target, -(-length // block_t)))


def _chunk(length, splits, block_t):
    # 每段长度取 BLOCK_T 的倍数，再按实际长度去掉空段。
    chunk = -(-(-(-length // splits)) // block_t) * block_t
    return chunk, -(-length // chunk)


def _check_query(query):
    if query.ndim != 4 or query.shape[2] != 1 or query.dtype != torch.float16 or not query.is_contiguous():
        raise ValueError('query must be contiguous FP16 [B, H, 1, D]')
    d = query.shape[-1]
    if d < 16 or d & (d - 1):
        raise ValueError('head dim must be a power of two >= 16')
    return query.shape[0] * query.shape[1], d


def _merge(m, l, o, query):
    out = torch.empty_like(query)
    splits = m.shape[1]
    _kv_merge[(m.shape[0],)](m, l, o, out, splits, D=query.shape[-1],
                             BLOCK_S=max(2, triton.next_power_of_2(splits)), num_warps=4)
    return out


def kv_decode_fp16(query, key, value, *, block_t, splits, num_warps=4):
    """FP16 缓存 decode attention：split-K 分段 + 合并；与 INT4 路径同模板。"""
    bh, d = _check_query(query)
    if key.shape != value.shape or key.shape[:2] != query.shape[:2] or key.shape[-1] != d \
            or key.dtype != torch.float16 or not key.is_contiguous() or not value.is_contiguous():
        raise ValueError('K/V must be contiguous FP16 [B, H, T, D] matching query')
    length = key.shape[2]
    chunk, splits = _chunk(length, splits, block_t)
    m = torch.empty((bh, splits), dtype=torch.float32, device=query.device)
    l = torch.empty_like(m)
    o = torch.empty((bh, splits, d), dtype=torch.float32, device=query.device)
    _kv_fp16_partial[(bh, splits)](query, key, value, m, l, o, length, chunk, 0, splits, d ** -.5,
                                   D=d, BLOCK_T=block_t, num_warps=num_warps)
    return _merge(m, l, o, query)


def kv_decode_int4(query, cache: PackedKV, *, block_t, splits, num_warps=4):
    """INT4 前缀直接在内核内解码；FP16 最近窗口作为额外一段，再统一合并。"""
    bh, d = _check_query(query)
    if tuple(query.shape[:2]) != tuple(cache.shape[:2]) or cache.shape[-1] != d:
        raise ValueError('Packed cache does not match query batch/heads/dim')
    tail = cache.recent_key.shape[2]
    prefix_splits, chunk = 0, 0
    if cache.prefix:
        k, v = cache.key, cache.value
        if block_t % k.group_size or d % v.group_size:
            raise ValueError('BLOCK_T must be a multiple of the K group; head dim a multiple of the V group')
        chunk, prefix_splits = _chunk(cache.prefix, splits, block_t)
    total = prefix_splits + (1 if tail else 0)
    m = torch.empty((bh, total), dtype=torch.float32, device=query.device)
    l = torch.empty_like(m)
    o = torch.empty((bh, total, d), dtype=torch.float32, device=query.device)
    if cache.prefix:
        _kv_int4_partial[(bh, prefix_splits)](
            query, k.payload, k.scales, v.payload, v.scales, m, l, o,
            cache.prefix, chunk, 0, total, d ** -.5,
            k.payload.stride(0), k.scales.stride(0), v.payload.stride(0), v.scales.stride(0),
            D=d, GROUP_K=k.group_size, GROUP_V=v.group_size, BLOCK_T=block_t, num_warps=num_warps)
    if tail:
        _kv_fp16_partial[(bh, 1)](query, cache.recent_key, cache.recent_value, m, l, o, tail, tail,
                                  prefix_splits, total, d ** -.5, D=d, BLOCK_T=block_t, num_warps=num_warps)
    return _merge(m, l, o, query)
