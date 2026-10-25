"""T4/A100 CUDA 算子实验。所有随机数据为合成数据，不下载模型。"""
from contextlib import contextmanager, nullcontext
from functools import partial
import importlib.util
import math
import warnings

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from experiments.cuda_kernels import pack_int4, unpack_int4, pack_kv, unpack_kv, error_metrics
from experiments.cuda_runtime import failed, matmul_precision, skipped, Unsupported


def variant(name, fn, *, context=nullcontext, **details):
    return dict(name=name, fn=fn, context=context, details=details)


def optional_variant(bench, case, name, build):
    """构建并校验可选路径；不支持记为跳过，构建/校验出错只让该路径失败，其他路径继续测量。"""
    try:
        return build()
    except Unsupported as exc:
        bench.checkpoint(skipped(case, name, str(exc)))
    except Exception as exc:
        bench.checkpoint(failed(case, name, exc))
    return None


def check_nmse(actual, expected, limit, label):
    """尺度无关的正确性检查：总平方误差/参考能量；同时要求输出在 CUDA 上且有限。"""
    if actual.device.type != 'cuda' or not torch.isfinite(actual).all():
        raise AssertionError(f'{label}: output must be finite and on CUDA')
    error = error_metrics(expected.float(), actual.float())
    if error['nmse'] > limit:
        raise AssertionError(f'{label}: NMSE {error["nmse"]:.3g} exceeds {limit:g}')
    return error


def triton_module():
    if importlib.util.find_spec('triton') is None:
        raise Unsupported('Triton 未安装；Linux CUDA 版 PyTorch 通常自带，CPU 版没有。')
    from experiments import cuda_triton
    return cuda_triton


@contextmanager
def no_int8_fallback():
    # Some bitsandbytes releases can return INT32 after a slower FP32 fallback.
    # Reject that reported path instead of calling its timing an INT8 result.
    with warnings.catch_warnings():
        warnings.filterwarnings('error', message=r'.*fall.*back.*fp32.*', category=RuntimeWarning)
        yield


def validate_output(value, expected, *, rtol=3e-3, atol=3e-3):
    if value.device.type != 'cuda' or not torch.isfinite(value).all():
        raise AssertionError('CUDA output must be finite and on the requested device')
    torch.testing.assert_close(value.float(), expected.float(), rtol=rtol, atol=atol)


@torch.inference_mode()
def gemm(bench):
    """改变 M 区分小批 decode 风格与较大 prefill 风格的线性层形状。"""
    shapes = [(1, 1024, 1024), (128, 1024, 1024)] if bench.profile == 'smoke' else [
        (1, 4096, 4096), (32, 4096, 4096), (512, 4096, 4096)]
    capability = torch.cuda.get_device_capability(bench.device)
    for m, n, k in shapes:
        case = f'gemm_m{m}_n{n}_k{k}'
        a = bench.randn((m, k), torch.float32)/math.sqrt(k)
        w = bench.randn((n, k), torch.float32)
        with matmul_precision(False):
            reference = a@w.T
        configs = [('fp32_ieee', torch.float32, False), ('fp16', torch.float16, False)]
        if capability[0] >= 8:
            configs += [('bf16', torch.bfloat16, False), ('fp32_tf32_requested', torch.float32, True)]
        else:
            for name in ('bf16', 'fp32_tf32_requested'):
                bench.checkpoint(skipped(case, name, '该实验仅在 SM80+ 请求 BF16/TF32；T4 执行 FP32/FP16。'))
        variants = []
        for name, dtype, tf32 in configs:
            x, weight = a.to(dtype), w.to(dtype)
            fn = lambda x=x, weight=weight: x@weight.T
            with matmul_precision(tf32):
                actual = fn()
            error = error_metrics(reference, actual)
            if error['nmse'] > .01:
                raise AssertionError(f'Unexpected GEMM numerical error: {name}')
            variants.append(variant(name, fn, context=partial(matmul_precision, tf32),
                                    error=error, shape_mnk=[m,n,k], flops=2*m*n*k,
                                    input_dtype=str(dtype), output_dtype=str(actual.dtype),
                                    tf32_requested=tf32, storage_bytes=x.nbytes+weight.nbytes,
                                    bytes_moved=x.nbytes+weight.nbytes+m*n*actual.element_size(),
                                    implementation='torch.matmul / CUDA BLAS; instruction choice is backend-dependent',
                                    scope='GEMM only; fixture creation and dtype casts excluded'))
        bench.compare(case, variants, l2='flush', baseline='fp16')


@torch.inference_mode()
def int4(bench):
    n, k = (512, 1024) if bench.profile == 'smoke' else (2048, 4096)
    batch_sizes = (1, 128) if bench.profile == 'smoke' else (1, 32, 512)
    group_sizes = (32, 128)
    weight = bench.randn((n,k))*.1
    inputs = {m:bench.randn((m,k))/math.sqrt(k) for m in batch_sizes}
    for group in group_sizes:
        packed = pack_int4(weight, group)
        restored = unpack_int4(packed)
        storage = dict(storage_bytes=packed.storage_bytes, fp16_storage_bytes=weight.nbytes,
                       compression_ratio=weight.nbytes/packed.storage_bytes,
                       group_size=group, scale_dtype='torch.float16', payload_dtype='torch.uint8',
                       logical_bits_per_weight=packed.storage_bytes*8/weight.numel())
        bench.compare(f'int4_codec_g{group}', [
            variant('pack', lambda: pack_int4(weight,group), **storage,
                    implementation='PyTorch CUDA reductions, rounding and bit packing; includes validation',
                    scope='Offline encoding; not included in forward comparisons'),
            variant('unpack', lambda: unpack_int4(packed), **storage,
                    error=error_metrics(weight,restored), implementation='PyTorch CUDA eager bit operations',
                    scope='Decode the complete matrix into a newly allocated FP16 tensor'),
        ], l2='flush')
        for m in batch_sizes:
            x = inputs[m]  # Same input for g32 and g128; only representation changes.
            with matmul_precision(False):
                original = x.float()@weight.float().T
            quantized = x@restored.T
            fn = lambda: x@unpack_int4(packed).T
            validate_output(fn(), quantized)
            io = x.nbytes+m*n*2
            common = dict(shape_mnk=[m,n,k], group_size=group, flops=2*m*n*k,
                          scope='One forward; input creation/offline packing excluded',
                          implementation='FP16 GEMM; explicit unpack path is not a fused INT4 kernel')
            bench.compare(f'int4_m{m}_g{group}', [
                variant('fp16_original', lambda: x@weight.T, bytes_moved=weight.nbytes+io,
                        storage_bytes=weight.nbytes, error=error_metrics(original,x@weight.T), **common),
                variant('fp16_predecoded', lambda: x@restored.T, storage_bytes=restored.nbytes,
                        bytes_moved=restored.nbytes+io, error=error_metrics(original,quantized), **common),
                # 读 INT4、写完整 FP16、再读回 FP16；这正是融合内核要省掉的流量。
                variant('int4_unpack_then_fp16', fn, storage_bytes=packed.storage_bytes,
                        bytes_moved=packed.storage_bytes+2*weight.nbytes+io,
                        fp16_storage_bytes=weight.nbytes, compression_ratio=weight.nbytes/packed.storage_bytes,
                        error=error_metrics(original,quantized), **common),
            ], l2='flush')


def eager_attention(q,k,v,causal=False):
    scores = q.float()@k.float().transpose(-2,-1)/math.sqrt(q.shape[-1])
    if causal:
        if q.shape[-2] != k.shape[-2]:
            raise ValueError('Explicit causal reference requires square prefill attention')
        mask = torch.ones(scores.shape[-2:],dtype=torch.bool,device=q.device).triu(1)
        scores = scores.masked_fill(mask, -torch.inf)
    return (torch.softmax(scores,dim=-1)@v.float()).to(q.dtype)


def sdpa_available(q,k,v,backend,causal):
    if backend == SDPBackend.MATH:
        return True, None
    # PyTorch 2.3 predates the enable_gqa argument added to SDPAParams.
    try:
        params = torch.backends.cuda.SDPAParams(q,k,v,None,0.,causal,False)
    except TypeError:
        params = torch.backends.cuda.SDPAParams(q,k,v,None,0.,causal)
    checker = {SDPBackend.FLASH_ATTENTION:torch.backends.cuda.can_use_flash_attention,
               SDPBackend.EFFICIENT_ATTENTION:torch.backends.cuda.can_use_efficient_attention}[backend]
    if not checker(params):
        return False, 'PyTorch capability check rejected this backend for the current build/device/shape.'
    # Once capability checks accept a path, execution errors are real failures.
    return True, None


@torch.inference_mode()
def attention(bench):
    length = 256 if bench.profile == 'smoke' else 1024
    for phase, tokens, causal in [('prefill',length,True), ('decode',1,False)]:
        case = 'attention_'+phase
        q = bench.randn((1,8,tokens,64))
        k,v = bench.randn((1,8,length,64)), bench.randn((1,8,length,64))
        with matmul_precision(False):
            reference = eager_attention(q,k,v,causal)
        variants = [variant('explicit_fp32_scores', lambda: eager_attention(q,k,v,causal),
                            context=partial(matmul_precision,False), error=error_metrics(reference,reference),
                            implementation='Materialized FP32 logits and probabilities, FP16 output')]
        for name,backend in [('math',SDPBackend.MATH),('efficient',SDPBackend.EFFICIENT_ATTENTION),
                             ('flash',SDPBackend.FLASH_ATTENTION)]:
            available,reason = sdpa_available(q,k,v,backend,causal)
            if not available:
                bench.checkpoint(skipped(case,name,reason))
                continue
            fn = lambda: F.scaled_dot_product_attention(q,k,v,dropout_p=0.,is_causal=causal)
            with sdpa_kernel(backend):
                output = fn()
            validate_output(output,reference,rtol=5e-3,atol=5e-3)
            variants.append(variant(name,fn,context=partial(sdpa_kernel,backend),
                                    error=error_metrics(reference,output),
                                    implementation='PyTorch SDPA with only '+name+' enabled; no other backend fallback'))
        for item in variants:
            item['details'].update(query_shape=list(q.shape), kv_shape=list(k.shape),causal=causal,
                                   scope='Attention forward only; fixed cache, no projection/RoPE/append',
                                   decode_mask='All supplied keys are past/current tokens; is_causal=False for Q length 1')
        bench.compare(case,variants,l2='flush')


@torch.inference_mode()
def kv(bench):
    length = 512 if bench.profile == 'smoke' else 4096
    key,value = bench.randn((1,8,length,64)),bench.randn((1,8,length,64))
    query = bench.randn((1,8,1,64))
    # 固定 MATH 后端隔离缓存的表示误差，其他后端在 attention 实验单独对照。
    def attend(k,v):
        return F.scaled_dot_product_attention(query,k,v,dropout_p=0.,is_causal=False)
    context = partial(sdpa_kernel,SDPBackend.MATH)
    with context():
        reference = attend(key,value)
    original_bytes = key.nbytes+value.nbytes
    for residual in (0,128):
        packed = pack_kv(key,value,group_size=32,residual=residual)
        kq,vq = unpack_kv(packed)
        with context():
            quantized = attend(kq,vq)
        errors = error_metrics(reference,quantized)
        common = dict(kv_shape=list(key.shape),residual=residual,group_size=32,
                      key_axis='time',value_axis='features',error=errors,
                      key_error=error_metrics(key,kq),value_error=error_metrics(value,vq))
        bench.compare(f'kv_codec_r{residual}',[
            variant('pack_snapshot',lambda:pack_kv(key,value,32,residual),
                    storage_bytes=packed.storage_bytes, **common,
                    scope='Offline/static snapshot encoding, includes validation; no streaming append'),
            variant('unpack_snapshot',lambda:unpack_kv(packed),storage_bytes=packed.storage_bytes,**common,
                    scope='Decode both complete K/V tensors and concatenate FP16 recent tail'),
        ], l2='flush')
        def decoded_attention():
            keys,values=unpack_kv(packed)
            return attend(keys,values)
        with context():
            validate_output(decoded_attention(),quantized)
        bench.compare(f'kv_decode_r{residual}',[
            variant('fp16_cache',lambda:attend(key,value),context=context,storage_bytes=original_bytes,
                    bytes_moved=original_bytes,
                    error=error_metrics(reference,reference),scope='Decode attention with original FP16 cache'),
            variant('fp16_predecoded_cache',lambda:attend(kq,vq),context=context,storage_bytes=original_bytes,
                    bytes_moved=original_bytes,
                    **common,scope='Decode attention after offline dequantization; no runtime compression'),
            variant('int4_unpack_then_attention',decoded_attention,context=context,**common,
                    storage_bytes=packed.storage_bytes,fp16_storage_bytes=original_bytes,
                    bytes_moved=packed.storage_bytes+2*original_bytes,
                    compression_ratio=original_bytes/packed.storage_bytes,
                    scope='Full K/V unpack + FP16 SDPA MATH; static cache, not a fused quantized attention kernel'),
        ], l2='flush')


@torch.inference_mode()
def int8(bench):
    if importlib.util.find_spec('bitsandbytes') is None:
        raise Unsupported('Optional bitsandbytes is not installed; install it to run the real INT8 backend.')
    import bitsandbytes.functional as bnb
    required=('int8_vectorwise_quant','int8_linear_matmul','int8_mm_dequant')
    if any(not hasattr(bnb,name) for name in required):
        raise Unsupported('Installed bitsandbytes lacks required functional INT8 APIs.')
    m,n,k = (32,512,1024) if bench.profile=='smoke' else (256,2048,4096)
    x,w = bench.randn((m,k)),bench.randn((n,k))*.1
    qx,sx,_ = bnb.int8_vectorwise_quant(x,threshold=0.)
    qw,sw,_ = bnb.int8_vectorwise_quant(w,threshold=0.)
    if any(t.dtype != torch.int8 or t.device != bench.device for t in (qx,qw)):
        raise AssertionError('Expected actual CUDA INT8 operand tensors')
    if k % 4:
        raise AssertionError('INT8 shape must avoid the documented nonaligned FP32 fallback')
    try:
        with no_int8_fallback():
            accum = bnb.int8_linear_matmul(qx,qw)
    except RuntimeWarning as exc:
        raise Unsupported('bitsandbytes reported an FP32 fallback: '+str(exc)) from exc
    if accum.device != bench.device or accum.dtype != torch.int32:
        raise AssertionError('Expected actual CUDA INT32 output from INT8 operands')
    # 在计时外用 CPU int64 对完整 K 归约的一小块做精确独立核对。
    exact = qx[:8].cpu().long()@qw[:8].cpu().long().T
    torch.testing.assert_close(accum[:8,:8].cpu().long(),exact,rtol=0,atol=0)
    decoded = bnb.int8_mm_dequant(accum,sx,sw)
    oracle = (accum.float()*(sx[:,None]*sw[None,:])/(127.**2)).half()
    validate_output(decoded,oracle,rtol=2e-3,atol=2e-3)
    with matmul_precision(False):
        reference=x.float()@w.float().T
    def prequantized():
        return bnb.int8_mm_dequant(bnb.int8_linear_matmul(qx,qw),sx,sw)
    def online_activation():
        codes,stats,_=bnb.int8_vectorwise_quant(x,threshold=0.)
        return bnb.int8_mm_dequant(bnb.int8_linear_matmul(codes,qw),stats,sw)
    error = error_metrics(reference,decoded)
    common = dict(shape_mnk=[m,n,k],flops=2*m*n*k,threshold=0.,outlier_split=False,
                  weight_storage_bytes=qw.nbytes+sw.nbytes,
                  implementation='bitsandbytes functional INT8 CUDA backend; profiler needed for instruction attribution')
    bench.compare('int8',[
        variant('fp16',lambda:x@w.T,error=error_metrics(reference,x@w.T),
                storage_bytes=w.nbytes,scope='Original FP16 GEMM',shape_mnk=[m,n,k],flops=2*m*n*k),
        variant('int8_gemm_only',lambda:bnb.int8_linear_matmul(qx,qw),context=no_int8_fallback,**common,
                scope='Prequantized operands -> INT32 only; excludes quantization and dequantization'),
        variant('int8_gemm_dequant',prequantized,context=no_int8_fallback,**common,error=error,
                scope='Prequantized operands -> FP16 output; excludes input/weight quantization'),
        variant('int8_activation_quant_gemm_dequant',online_activation,context=no_int8_fallback,**common,error=error,
                scope='Online activation quantization + INT8 GEMM + output dequantization; weights prequantized'),
    ], l2='flush', baseline='fp16')


# Qwen3-0.6B 的线性层形状（hidden 1024、MLP 3072、16 个 Q 头与 8 个 KV 头、head_dim 128、28 层）。
QWEN3_06B = dict(hidden=1024, intermediate=3072, q_out=2048, kv_out=1024)
QWEN3_LAYERS = 28

# 每个新增实验的形状集中在此，便于阅读与修改；测试用缩小的表在 CPU 上演练逻辑。
SIZES = {
    'smoke': dict(bandwidth_mib=256, w4a16_shapes=[(2048, 2048)], w4a16_batches=(1, 16, 128), w4a16_groups=(128,),
                  lowp_shapes=[(32, 2048, 2048), (512, 2048, 2048)], kv_lengths=(1024, 4096),
                  kv_heads=8, kv_dim=128, stack_model=QWEN3_06B, stack_layers=QWEN3_LAYERS, stack_batches=(1,)),
    'full': dict(bandwidth_mib=1024, w4a16_shapes=[(4096, 4096), (11008, 4096)], w4a16_batches=(1, 4, 16, 64, 256),
                 w4a16_groups=(32, 128), lowp_shapes=[(32, 4096, 4096), (512, 4096, 4096), (2048, 4096, 4096)],
                 kv_lengths=(2048, 8192, 32768), kv_heads=8, kv_dim=128, stack_model=QWEN3_06B, stack_layers=QWEN3_LAYERS,
                 stack_batches=(1, 16)),
}


def attention_fp32(q, k, v):
    """不融合、FP32 计算的 decode 参照；输出保持 FP32 供误差统计。"""
    scores = q.float()@k.float().transpose(-2, -1)/math.sqrt(q.shape[-1])
    return torch.softmax(scores, dim=-1)@v.float()


@torch.inference_mode()
def bandwidth(bench):
    """实测显存读/写/复制带宽，作为本轮 roofline 与带宽比例参照；缓冲区远大于 L2。"""
    l2 = getattr(torch.cuda.get_device_properties(bench.device), 'L2_cache_size', 0) or 0
    size = max(SIZES[bench.profile]['bandwidth_mib']*2**20, 16*l2)
    source = torch.empty(size//2, dtype=torch.float16, device=bench.device).normal_(generator=bench.generator)
    target = torch.empty_like(source)
    # 计时外核对：复制逐位相等；求和与分块 FP64 求和一致。
    target.copy_(source)
    if not torch.equal(target, source):
        raise AssertionError('Device copy changed data')
    chunks = source.chunk(64)
    exact = sum(c.double().sum().item() for c in chunks)
    magnitude = sum(c.double().abs().sum().item() for c in chunks)
    if abs(torch.sum(source, dtype=torch.float32).item()-exact) > 1e-5*magnitude:
        raise AssertionError('Device reduction disagrees with chunked FP64 sum')
    common = dict(buffer_bytes=size, l2_cache_bytes=l2,
                  scope='Streaming kernels over a buffer >= 16x L2; achievable bandwidth, not vendor peak')
    bench.compare('bandwidth', [
        variant('read_sum', lambda: torch.sum(source, dtype=torch.float32), bytes_moved=size,
                implementation='torch.sum reduction: reads every byte once', **common),
        variant('write_zero', lambda: target.zero_(), bytes_moved=size,
                implementation='Tensor.zero_: writes every byte once', **common),
        variant('copy', lambda: target.copy_(source), bytes_moved=2*size,
                implementation='Tensor.copy_: one read and one write per byte', **common),
    ])


def _w4a16_variants(bench, case, ct, x, packed, weight, references, common, io):
    """同一输入下的 Triton FP16 同模板对照与 INT4 融合路径；配置在计时外选出。"""
    m, k = x.shape
    reference_fp, reference_q = references

    def build(kind):
        if kind == 'fp16':
            candidates = [(c, partial(ct.fp16_matmul, x, weight, c)) for c in ct.candidate_configs(m, k)]
        else:
            candidates = [(c, partial(ct.w4a16_matmul, x, packed, c))
                          for c in ct.candidate_configs(m, k, packed.group_size)]
        config, table = bench.tune(candidates, l2='flush')
        run = ct.fp16_matmul if kind == 'fp16' else ct.w4a16_matmul
        operand = weight if kind == 'fp16' else packed
        output, kernel = run(x, operand, config, return_kernel=True)
        expected = reference_fp if kind == 'fp16' else reference_q
        check_nmse(output, expected, 1e-4, f'{case}/{kind}')
        details = dict(config=config, tuning=table, tensor_core_mma=ct.tensor_core_mma(kernel),
                       error=error_metrics(reference_fp, output), **common)
        if kind == 'fp16':
            return variant('triton_fp16_same_template', partial(run, x, operand, config),
                           bytes_moved=weight.nbytes+io, storage_bytes=weight.nbytes,
                           implementation='Triton, same tiles as the INT4 kernel; FP16 weight loads', **details)
        return variant('triton_w4a16_fused', partial(run, x, operand, config),
                       bytes_moved=packed.storage_bytes+io, storage_bytes=packed.storage_bytes,
                       fp16_storage_bytes=weight.nbytes, compression_ratio=weight.nbytes/packed.storage_bytes,
                       error_vs_decoded=error_metrics(reference_q, output),
                       implementation='Triton: uint8 codes + FP16 group scales decoded in registers; '
                                      'no full FP16 weight is written', **details)

    return [item for item in (optional_variant(bench, case, 'triton_fp16_same_template', lambda: build('fp16')),
                              optional_variant(bench, case, 'triton_w4a16_fused', lambda: build('int4')))
            if item is not None]


@torch.inference_mode()
def w4a16(bench):
    """扫描 M：cuBLAS FP16、同模板 Triton FP16、融合 W4A16 与先解码路径；观察 decode→prefill 拐点。"""
    ct = triton_module()
    size = SIZES[bench.profile]
    shapes, batches, groups = size['w4a16_shapes'], size['w4a16_batches'], size['w4a16_groups']
    for n, k in shapes:
        weight = bench.randn((n, k))*.02
        for group in groups:
            packed = pack_int4(weight, group)
            restored = unpack_int4(packed)
            for m in batches:
                case = f'w4a16_n{n}_k{k}_m{m}_g{group}'
                x = bench.randn((m, k))
                with matmul_precision(False):
                    reference_fp = x.float()@weight.float().T
                    reference_q = x.float()@restored.float().T
                io = x.nbytes+m*n*2
                common = dict(shape_mnk=[m, n, k], group_size=group, flops=2*m*n*k,
                              scope='One linear forward with L2 flushed before each call; offline packing excluded')
                variants = [variant('torch_fp16', lambda x=x: x@weight.T, bytes_moved=weight.nbytes+io,
                                    storage_bytes=weight.nbytes, error=error_metrics(reference_fp, x@weight.T),
                                    implementation='torch.matmul / cuBLAS', **common)]
                variants += _w4a16_variants(bench, case, ct, x, packed, weight, (reference_fp, reference_q), common, io)
                variants.append(variant('int4_unpack_then_fp16', lambda x=x, packed=packed: x@unpack_int4(packed).T,
                                        bytes_moved=packed.storage_bytes+2*weight.nbytes+io,
                                        storage_bytes=packed.storage_bytes, error=error_metrics(reference_fp, x@restored.T),
                                        implementation='PyTorch unpack to a full FP16 matrix, then cuBLAS', **common))
                bench.compare(case, variants, l2='flush', baseline='torch_fp16')


def _rowwise_int8(t):
    # 每行对称 INT8，FP32 尺度；全零行用尺度 1 避免除零。
    maximum = t.float().abs().amax(dim=1, keepdim=True)
    scale = torch.where(maximum > 0, maximum/127, torch.ones_like(maximum))
    return torch.round(t.float()/scale).clamp(-127, 127).to(torch.int8), scale


def _int8_variants(bench, case, x, w, reference, common):
    if not hasattr(torch, '_int_mm'):
        raise Unsupported('当前 PyTorch 没有 torch._int_mm。')
    m, n = x.shape[0], w.shape[0]
    qx, sx = _rowwise_int8(x)
    qw, sw = _rowwise_int8(w)
    weight_t = qw.t()  # [K,N] 列主序视图，cuBLASLt INT8 路径要求的第二操作数布局
    accum = torch._int_mm(qx, weight_t)
    if accum.dtype != torch.int32 or accum.device != bench.device:
        raise AssertionError('torch._int_mm must return CUDA INT32')
    # 计时外：一小块完整 K 归约与 CPU int64 精确一致。
    exact = qx[:8].cpu().long()@qw[:8].cpu().long().T
    torch.testing.assert_close(accum[:8, :8].cpu().long(), exact, rtol=0, atol=0)

    def dequant(values, scale):
        return (values.float()*scale*sw.T).half()

    error = check_nmse(dequant(accum, sx), reference, 1e-2, f'{case}/int8')

    def dynamic():
        codes, scale = _rowwise_int8(x)
        return dequant(torch._int_mm(codes, weight_t), scale)

    int8 = dict(storage_bytes=qw.nbytes+sw.nbytes, weight_scale='per-row FP32',
                activation_scale='per-row FP32', implementation='torch._int_mm (cuBLASLt INT8, INT32 accumulate)', **common)
    return [variant('int8_gemm_only', lambda: torch._int_mm(qx, weight_t),
                    bytes_moved=qx.nbytes+qw.nbytes+m*n*4, scope='Prequantized operands -> INT32 only', **int8),
            variant('int8_gemm_dequant', lambda: dequant(torch._int_mm(qx, weight_t), sx), error=error,
                    bytes_moved=qx.nbytes+qw.nbytes+m*n*2, scope='Prequantized operands -> FP16 output', **int8),
            variant('int8_dynamic_activation', dynamic, error=error, bytes_moved=x.nbytes+qw.nbytes+m*n*2,
                    scope='Eager per-row activation quantization + INT8 GEMM + dequantization', **int8)]


def _fp8_variants(bench, case, x, w, reference, common):
    if torch.cuda.get_device_capability(bench.device) < (8, 9):
        raise Unsupported('FP8 矩阵乘需要 SM89+（如 L4、H100）；T4/A100 没有 FP8 Tensor Core。')
    if not hasattr(torch, '_scaled_mm') or not hasattr(torch, 'float8_e4m3fn'):
        raise Unsupported('当前 PyTorch 没有 torch._scaled_mm / float8_e4m3fn。')
    limit = torch.finfo(torch.float8_e4m3fn).max
    m, n = x.shape[0], w.shape[0]

    def tensorwise(t):
        # 每张量一个 FP32 尺度：absmax 映射到 E4M3 最大有限值 448。
        scale = (t.float().abs().amax()/limit).clamp_min(1e-12)
        return (t.float()/scale).to(torch.float8_e4m3fn), scale

    qx, sx = tensorwise(x)
    qw, sw = tensorwise(w)

    def matmul(codes, scale):
        result = torch._scaled_mm(codes, qw.t(), scale_a=scale, scale_b=sw, out_dtype=torch.bfloat16)
        return result[0] if isinstance(result, tuple) else result  # PyTorch < 2.4 also returned amax

    error = check_nmse(matmul(qx, sx), reference, 1e-2, f'{case}/fp8')

    def dynamic():
        codes, scale = tensorwise(x)
        return matmul(codes, scale)

    fp8 = dict(storage_bytes=qw.nbytes+4, format='E4M3 per-tensor scales, BF16 output',
               implementation='torch._scaled_mm (cuBLASLt FP8)', error=error, **common)
    return [variant('fp8_gemm', lambda: matmul(qx, sx), bytes_moved=qx.nbytes+qw.nbytes+m*n*2,
                    scope='Prequantized FP8 operands -> BF16 output', **fp8),
            variant('fp8_dynamic_activation', dynamic, bytes_moved=x.nbytes+qw.nbytes+m*n*2,
                    scope='Eager per-tensor activation quantization + FP8 GEMM', **fp8)]


@torch.inference_mode()
def lowp_gemm(bench):
    """PyTorch 原生低精度计算：INT8（torch._int_mm）与 SM89+ 的 FP8（torch._scaled_mm），对照 FP16。"""
    for m, n, k in SIZES[bench.profile]['lowp_shapes']:
        case = f'lowp_m{m}_n{n}_k{k}'
        x, w = bench.randn((m, k)), bench.randn((n, k))*.02
        with matmul_precision(False):
            reference = x.float()@w.float().T
        common = dict(shape_mnk=[m, n, k], flops=2*m*n*k)
        variants = [variant('fp16', lambda x=x, w=w: x@w.T, error=error_metrics(reference, x@w.T),
                            storage_bytes=w.nbytes, bytes_moved=x.nbytes+w.nbytes+m*n*2,
                            implementation='torch.matmul / cuBLAS FP16', **common)]
        variants += optional_variant(bench, case, 'int8',
                                     lambda: _int8_variants(bench, case, x, w, reference, common)) or []
        variants += optional_variant(bench, case, 'fp8',
                                     lambda: _fp8_variants(bench, case, x, w, reference, common)) or []
        bench.compare(case, variants, l2='flush', baseline='fp16')


@torch.inference_mode()
def kv_fused(bench):
    """INT4 KV 在 split-K decode attention 内核中直接解码，对照 SDPA、同模板 FP16 与先解码路径。"""
    ct = triton_module()
    size = SIZES[bench.profile]
    lengths, b, h, d, group, residual = size['kv_lengths'], 1, size['kv_heads'], size['kv_dim'], 32, 128
    sm_count = torch.cuda.get_device_properties(bench.device).multi_processor_count
    for length in lengths:
        case = f'kv_fused_t{length}'
        key, value = bench.randn((b, h, length, d)), bench.randn((b, h, length, d))
        query = bench.randn((b, h, 1, d))
        packed = pack_kv(key, value, group_size=group, residual=residual)
        kq, vq = unpack_kv(packed)
        reference = attention_fp32(query, key, value)
        reference_q = attention_fp32(query, kq, vq)
        original = key.nbytes+value.nbytes
        common = dict(kv_shape=[b, h, length, d], group_size=group, residual=residual,
                      key_axis='time', value_axis='features',
                      scope='Single decode query over a static cache; L2 flushed before each call; no RoPE/append/paging')

        def build(kind):
            prefix = packed.prefix if kind == 'int4' else length
            run = ct.kv_decode_int4 if kind == 'int4' else ct.kv_decode_fp16
            operands = (packed,) if kind == 'int4' else (key, value)
            candidates = []
            for c in ct.KV_CONFIGS:
                config = dict(c, splits=ct.split_count(prefix, b*h, sm_count, c['BLOCK_T']))
                candidates.append((config, partial(run, query, *operands, block_t=c['BLOCK_T'],
                                                   splits=config['splits'], num_warps=c['num_warps'])))
            config, table = bench.tune(candidates, l2='flush')
            fn = partial(run, query, *operands, block_t=config['BLOCK_T'], splits=config['splits'],
                         num_warps=config['num_warps'])
            output = fn()
            check_nmse(output, reference_q if kind == 'int4' else reference, 1e-4, f'{case}/{kind}')
            details = dict(config=config, tuning=table, error=error_metrics(reference, output), **common)
            if kind == 'int4':
                return variant('triton_int4_fused', fn, bytes_moved=packed.storage_bytes,
                               storage_bytes=packed.storage_bytes, fp16_storage_bytes=original,
                               compression_ratio=original/packed.storage_bytes,
                               error_vs_decoded=error_metrics(reference_q, output),
                               implementation='Triton split-K: INT4 prefix decoded in registers, FP16 recent tail '
                                              'as one more split, segment-softmax merge', **details)
            return variant('triton_fp16_split', fn, bytes_moved=original, storage_bytes=original,
                           implementation='Triton split-K with the same template over the FP16 cache', **details)

        sdpa = lambda: F.scaled_dot_product_attention(query, key, value, dropout_p=0., is_causal=False)
        variants = [variant('sdpa_fp16', sdpa, bytes_moved=original, storage_bytes=original,
                            error=error_metrics(reference, sdpa()),
                            implementation='PyTorch SDPA with its default backend selection', **common)]
        variants += [item for item in (optional_variant(bench, case, 'triton_fp16_split', lambda: build('fp16')),
                                       optional_variant(bench, case, 'triton_int4_fused', lambda: build('int4')))
                     if item is not None]

        def unpack_then_sdpa():
            keys, values = unpack_kv(packed)
            return F.scaled_dot_product_attention(query, keys, values, dropout_p=0., is_causal=False)

        variants.append(variant('int4_unpack_then_sdpa', unpack_then_sdpa, storage_bytes=packed.storage_bytes,
                                bytes_moved=packed.storage_bytes+2*original,
                                error=error_metrics(reference, unpack_then_sdpa()),
                                implementation='PyTorch unpack + concatenate full FP16 K/V, then SDPA; '
                                               'bytes_moved is a lower bound', **common))
        bench.compare(case, variants, l2='flush', baseline='sdpa_fp16')


def stack_shapes(config):
    h, i = config['hidden'], config['intermediate']
    return dict(q=(config['q_out'], h), k=(config['kv_out'], h), v=(config['kv_out'], h),
                o=(h, config['q_out']), gate=(i, h), up=(i, h), down=(h, i))


def _rms(x):
    return x*torch.rsqrt(x.float().square().mean(-1, keepdim=True)+1e-6).to(x.dtype)


def stack_forward(x, layers, linear):
    """只保留每层线性与逐元素运算：Attention、RoPE、KV 写入与 lm_head 省略。q 代替 Attention 输出送入 o。"""
    for layer in layers:
        hidden = _rms(x)
        q = linear(layer, 'q', hidden)
        linear(layer, 'k', hidden)
        linear(layer, 'v', hidden)
        x = x+linear(layer, 'o', q)
        hidden = _rms(x)
        x = x+linear(layer, 'down', F.silu(linear(layer, 'gate', hidden))*linear(layer, 'up', hidden))
    return x


def cuda_graph(fn, example):
    """在旁路 stream 预热后捕获；返回的 replay 复用静态输入/输出，只能在同一形状下重放。"""
    static = example.clone()
    stream = torch.cuda.Stream(device=example.device)
    stream.wait_stream(torch.cuda.current_stream(example.device))
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn(static)
    torch.cuda.current_stream(example.device).wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn(static)

    def replay():
        graph.replay()
        return output
    return replay


def _resident(build, device):
    """以 PyTorch allocator 增量计量常驻权重与量化状态，库内部的额外缓存另计。"""
    torch.cuda.synchronize(device)
    before = torch.cuda.memory_allocated(device)
    result = build()
    torch.cuda.synchronize(device)
    return result, torch.cuda.memory_allocated(device)-before


def _bnb_stack(layers, kind, device):
    if importlib.util.find_spec('bitsandbytes') is None:
        raise Unsupported('Optional bitsandbytes is not installed.')
    import bitsandbytes as bnb
    modules = []
    for layer in layers:
        built = {}
        for name, weight in layer.items():
            n, k = weight.shape
            if kind == 'int8':
                # 与 Qwen3 Notebook 相同的 LLM.int8() 阈值 6.0。
                module = bnb.nn.Linear8bitLt(k, n, bias=False, has_fp16_weights=False, threshold=6.0)
                module.weight = bnb.nn.Int8Params(weight.cpu(), requires_grad=False, has_fp16_weights=False)
            else:
                # 与 Notebook 第一轮相同：NF4、关闭 double quant、FP16 计算。
                module = bnb.nn.Linear4bit(k, n, bias=False, compute_dtype=torch.float16,
                                           compress_statistics=False, quant_type='nf4')
                module.weight = bnb.nn.Params4bit(weight.cpu(), requires_grad=False, blocksize=64,
                                                  compress_statistics=False, quant_type='nf4', module=module)
            built[name] = module.to(device).eval()  # bitsandbytes 在搬到 CUDA 时量化
        modules.append(built)
    return modules


@torch.inference_mode()
def decode_stack(bench):
    """Qwen3-0.6B 形状的 28 层线性骨架：FP16/W4A16/bnb INT8/NF4，eager 与 CUDA Graph 的 decode 步耗时。"""
    size = SIZES[bench.profile]
    config = size['stack_model']
    shapes = stack_shapes(config)
    layers = [{name: bench.randn(shape)/math.sqrt(shape[1]) for name, shape in shapes.items()}
              for _ in range(size['stack_layers'])]
    fp16_bytes = sum(w.nbytes for layer in layers for w in layer.values())
    batches = size['stack_batches']
    ct = None
    packed_layers = packed_bytes = None
    try:
        ct = triton_module()
        packed_layers = [{name: pack_int4(w, 128) for name, w in layer.items()} for layer in layers]
        packed_bytes = sum(p.storage_bytes for layer in packed_layers for p in layer.values())  # 精确 payload+scale
    except Unsupported as exc:
        bench.checkpoint(skipped('decode_stack', 'triton_w4a16', str(exc)))
    bnb_stacks = {}
    for kind in ('int8', 'nf4'):
        try:
            bnb_stacks[kind] = _resident(lambda kind=kind: _bnb_stack(layers, kind, bench.device), bench.device)
        except Unsupported as exc:
            bench.checkpoint(skipped('decode_stack', f'bnb_{kind}_eager', str(exc)))
        except Exception as exc:
            bench.checkpoint(failed('decode_stack', f'bnb_{kind}_eager', exc))
    fp16_linear = lambda layer, name, h: F.linear(h, layer[name])
    for m in batches:
        case = f'decode_stack_m{m}'
        x = bench.randn((m, config['hidden']))
        common = dict(model_shapes='Qwen3-0.6B linear layers', layers=len(layers), linear_shapes=shapes,
                      tokens=m, flops=2*m*(fp16_bytes//2),
                      scope='Linear-only decode skeleton: RMS-style normalization, SiLU-gated MLP and residuals; '
                            'attention, RoPE, KV writes, embedding and lm_head omitted')
        reference = stack_forward(x, layers, fp16_linear)
        eager = partial(stack_forward, layers=layers, linear=fp16_linear)
        variants = [variant('fp16_eager', lambda: eager(x), bytes_moved=fp16_bytes, weight_bytes=fp16_bytes,
                            implementation='F.linear (cuBLAS), one launch per op', **common)]

        def graphed(name, fn, weight_bytes, implementation):
            replay = cuda_graph(fn, x)
            # split-K 原子累加的顺序不固定，图重放与 eager 只要求在 FP16 舍入量级内一致。
            check_nmse(replay(), fn(x), 1e-5, f'{case}/{name}')
            return variant(name, replay, bytes_moved=weight_bytes, weight_bytes=weight_bytes,
                           error=error_metrics(reference.float(), replay().float()),
                           implementation=implementation, **common)

        item = optional_variant(bench, case, 'fp16_cuda_graph',
                                lambda: graphed('fp16_cuda_graph', eager, fp16_bytes, 'Same kernels replayed from one CUDA Graph'))
        variants += [item] if item else []
        if packed_layers is not None:
            def build_triton():
                chosen = {}
                for name, (n, k) in shapes.items():
                    if (n, k) not in chosen:
                        packed, sample = packed_layers[0][name], bench.randn((m, k))
                        candidates = [(c, partial(ct.w4a16_matmul, sample, packed, c))
                                      for c in ct.candidate_configs(m, k, 128)]
                        chosen[(n, k)] = bench.tune(candidates, l2='flush')
                        check_nmse(ct.w4a16_matmul(sample, packed, chosen[(n, k)][0]),
                                   sample.float()@unpack_int4(packed).float().T, 1e-4, f'{case}/triton_{name}')
                configs = {name: chosen[shape][0] for name, shape in shapes.items()}
                linear = lambda layer, name, h: ct.w4a16_matmul(h, layer[name], configs[name])
                return partial(stack_forward, layers=packed_layers, linear=linear), \
                    {f'{n}x{k}': dict(config=c, tuning=t) for (n, k), (c, t) in chosen.items()}

            built = optional_variant(bench, case, 'triton_w4a16_eager', build_triton)
            if built is not None:
                fused, tuning = built
                variants.append(variant('triton_w4a16_eager', lambda: fused(x), bytes_moved=packed_bytes,
                                        weight_bytes=packed_bytes, tuning=tuning,
                                        error=error_metrics(reference.float(), fused(x).float()),
                                        implementation='Triton fused W4A16 for every linear, group 128', **common))
                item = optional_variant(bench, case, 'triton_w4a16_cuda_graph',
                                        lambda: graphed('triton_w4a16_cuda_graph', fused, packed_bytes,
                                                        'Triton W4A16 kernels replayed from one CUDA Graph'))
                variants += [item] if item else []
        for kind, (modules, resident) in bnb_stacks.items():
            def build_bnb(kind=kind, modules=modules, resident=resident):
                linear = lambda layer, name, h: layer[name](h)
                sample = bench.randn((m, config['hidden']))
                with matmul_precision(False):
                    expected = sample.float()@layers[0]['q'].float().T
                check_nmse(modules[0]['q'](sample), expected, 1e-2 if kind == 'int8' else 5e-2, f'{case}/bnb_{kind}')
                fn = partial(stack_forward, layers=modules, linear=linear)
                label = 'LLM.int8() threshold 6.0' if kind == 'int8' else 'NF4 blocksize 64, no double quant'
                # bitsandbytes 不公开统一的字节账本：用 allocator 增量（含量化状态），不可用时不报 GB/s。
                return variant(f'bnb_{kind}_eager', lambda: fn(x), bytes_moved=resident or None,
                               weight_bytes=resident or None, weight_bytes_source='allocator delta after quantization',
                               error=error_metrics(reference.float(), fn(x).float()),
                               implementation=f'bitsandbytes nn modules, {label}', **common)
            item = optional_variant(bench, case, f'bnb_{kind}_eager', build_bnb)
            variants += [item] if item else []
        bench.compare(case, variants, baseline='fp16_eager')


CASES = {'bandwidth': bandwidth, 'gemm': gemm, 'int4': int4, 'w4a16': w4a16, 'lowp_gemm': lowp_gemm,
         'attention': attention, 'kv': kv, 'kv_fused': kv_fused, 'int8': int8, 'decode_stack': decode_stack}
DESCRIPTIONS = {
    'bandwidth': '显存读/写/复制带宽，作为本轮 roofline 参照',
    'gemm': 'FP32/FP16/BF16/TF32 GEMM 的吞吐与误差',
    'int4': 'INT4 打包/解码开销，以及先解码再 FP16 GEMM 的路径',
    'w4a16': 'Triton 融合 W4A16 与 cuBLAS FP16 随 M 的对照（需 Triton）',
    'lowp_gemm': 'torch._int_mm INT8 与 SM89+ FP8 GEMM（无需额外库）',
    'attention': 'SDPA MATH/EFFICIENT/FLASH 后端的 prefill/decode',
    'kv': 'INT4 KV 打包/解码与先解码再 Attention',
    'kv_fused': 'Triton split-K decode attention 直接读取 INT4 KV（需 Triton）',
    'int8': 'bitsandbytes INT8 GEMM 的逐级成本（需 bitsandbytes）',
    'decode_stack': 'Qwen3-0.6B 形状线性骨架：FP16/W4A16/INT8/NF4 与 CUDA Graph',
}
