"""CUDA 计时和运行记录。CPU 只用于测试统计逻辑，不可冒充 GPU 测量。"""
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import csv
import gc
import hashlib
import importlib.metadata
import io
import math
from pathlib import Path
import random
import statistics
import subprocess
import time
import traceback
import uuid

import torch
from experiments.cuda_kernels import positive_integer
from experiments.result_utils import ROOT, run_metadata, write_json, write_text


def require_cuda(requested):
    device = torch.device(requested)
    if device.type != 'cuda' or torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError('需要 NVIDIA CUDA 版 PyTorch 和可用 GPU；不会回退到 CPU。')
    index = torch.cuda.current_device() if device.index is None else device.index
    if not 0 <= index < torch.cuda.device_count():
        raise ValueError('CUDA device index is out of range')
    torch.cuda.set_device(index)
    return torch.device('cuda', index)


def nvidia_smi_inventory():
    # 时钟与功耗上限会影响 Colab 实测；旧驱动不支持扩展字段时退回基本字段。
    for fields in ('name,driver_version,memory.total,clocks.max.sm,clocks.max.memory,power.limit',
                   'name,driver_version,memory.total'):
        try:
            return subprocess.run(['nvidia-smi', f'--query-gpu={fields}', '--format=csv,noheader'],
                                  capture_output=True, text=True, timeout=5, check=True).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            continue
    return None


def environment(device):
    props = torch.cuda.get_device_properties(device)
    versions = {}
    for name in ('torch', 'numpy', 'bitsandbytes', 'triton'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True,
                                text=True, timeout=5, check=True).stdout.strip()
        dirty = bool(subprocess.run(['git', 'status', '--porcelain'], cwd=ROOT, capture_output=True,
                                   text=True, timeout=5, check=True).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        commit, dirty = None, None
    return dict(device=str(device), gpu=props.name, compute_capability=[props.major, props.minor],
                total_memory_bytes=props.total_memory, multiprocessors=props.multi_processor_count,
                l2_cache_bytes=getattr(props, 'L2_cache_size', None),
                cuda_build=torch.version.cuda, versions=versions, git_commit=commit, git_dirty=dirty,
                nvidia_smi_inventory=nvidia_smi_inventory(),
                fp16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                initial_free_bytes=torch.cuda.mem_get_info(device)[0])


def sample_summary(values):
    if not values or any(isinstance(v, bool) or not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError('Timing samples must be finite positive milliseconds')
    return dict(samples_ms=values, median_ms=statistics.median(values),
                min_ms=min(values), max_ms=max(values))


@contextmanager
def matmul_precision(tf32=False):
    # Use one precision API family at a time and restore the caller's state.
    backend = torch.backends.cuda.matmul
    modern = hasattr(backend, 'fp32_precision')
    old = backend.fp32_precision if modern else backend.allow_tf32
    if modern:
        backend.fp32_precision = 'tf32' if tf32 else 'ieee'
    else:
        backend.allow_tf32 = tf32
    try:
        yield
    finally:
        if modern:
            backend.fp32_precision = old
        else:
            backend.allow_tf32 = old


class Unsupported(RuntimeError):
    """An explicitly detected missing optional capability, not a numerical failure."""


class PersistenceError(RuntimeError):
    """Stop on a result write failure instead of reporting a backend failure."""


def skipped(case, variant, reason, **details):
    return dict(case=case, variant=variant, status='skipped', reason=reason, **details)


def failed(case, variant, exc, **details):
    """单个路径的真实执行错误；同一实验的其他路径继续测量，运行状态记为 partial。"""
    return dict(case=case, variant=variant, status='failed', reason=f'{type(exc).__name__}: {exc}',
                error_type=type(exc).__name__,
                traceback=''.join(traceback.format_exception(type(exc), exc, exc.__traceback__)), **details)


L2_MODES = ('flush', 'warm')

# 摘要中的成对对照：(候选, 参照, 含义)。比值 = 参照中位数 / 候选中位数，>1 表示候选更快。
KEY_PAIRS = (
    ('triton_w4a16_fused', 'triton_fp16_same_template', '同模板下 INT4 读取 vs FP16 读取'),
    ('triton_w4a16_fused', 'torch_fp16', '融合 W4A16 vs cuBLAS FP16'),
    ('int4_unpack_then_fp16', 'torch_fp16', '先解码再 GEMM vs cuBLAS FP16'),
    ('triton_int4_fused', 'triton_fp16_split', '同模板下 INT4 KV vs FP16 KV'),
    ('triton_int4_fused', 'sdpa_fp16', '融合 INT4 KV vs SDPA FP16'),
    ('int8_gemm_dequant', 'fp16', 'INT8 GEMM+反量化 vs FP16'),
    ('fp8_gemm', 'fp16', 'FP8 GEMM vs FP16'),
    ('fp16_cuda_graph', 'fp16_eager', 'CUDA Graph vs eager（FP16）'),
    ('triton_w4a16_cuda_graph', 'fp16_cuda_graph', 'W4A16 vs FP16（均用 CUDA Graph）'),
    ('bnb_int8_eager', 'fp16_eager', 'bitsandbytes LLM.int8() vs FP16'),
    ('bnb_nf4_eager', 'fp16_eager', 'bitsandbytes NF4 vs FP16'),
)


def key_comparisons(results):
    """同一 case 内成对路径的中位数比值；任何一侧缺失或未通过时不生成该行。"""
    timed = {}
    for row in results:
        if row.get('status') == 'passed' and row.get('cuda'):
            timed[(row['case'], row['variant'])] = row['cuda']['median_ms']
    rows = []
    for case in dict.fromkeys(row.get('case') for row in results):
        for candidate, reference, meaning in KEY_PAIRS:
            if (case, candidate) in timed and (case, reference) in timed:
                rows.append(dict(case=case, candidate=candidate, reference=reference, meaning=meaning,
                                 speedup=timed[(case, reference)]/timed[(case, candidate)]))
    return rows


class CudaBench:
    def __init__(self, device, profile='smoke', seed=42, trace_dir=None, checkpoint=None):
        if profile not in ('smoke', 'full'):
            raise ValueError('Unknown CUDA benchmark profile')
        self.device, self.profile, self.seed = device, profile, seed
        self.warmup, self.repeats, self.iterations = (3, 5, 5) if profile == 'smoke' else (5, 7, 20)
        self.trace_dir = trace_dir
        self.checkpoint = checkpoint or (lambda row: None)
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.flush_bytes = None
        self._flush_buffer = None

    def randn(self, shape, dtype=torch.float16):
        # A single FP32 fixture is rounded for dtype comparisons within a run.
        return torch.randn(shape, generator=self.generator, device=self.device, dtype=torch.float32).to(dtype)

    def flush_l2(self):
        """写满一块远大于 L2 的缓冲区，使下一次调用从显存读取输入。"""
        if self._flush_buffer is None:
            if self.flush_bytes is None:
                l2 = getattr(torch.cuda.get_device_properties(self.device), 'L2_cache_size', 0) or 0
                self.flush_bytes = max(4*l2, 64*2**20) if l2 else 256*2**20
            self._flush_buffer = torch.empty(self.flush_bytes, dtype=torch.int8, device=self.device)
        self._flush_buffer.zero_()

    def tune(self, candidates, l2='warm'):
        """在正式计时外短测每个 (config, fn)，返回最快配置与全部记录；编译/资源错误只淘汰该配置。

        选中的配置随后仍要通过正确性检查；淘汰原因写入报告，不静默丢弃。
        """
        table = []
        for config, fn in candidates:
            try:
                for _ in range(2):
                    fn()
                torch.cuda.synchronize(self.device)
                samples = []
                for _ in range(5):
                    if l2 == 'flush':
                        self.flush_l2()
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    fn()
                    end.record()
                    end.synchronize()
                    samples.append(start.elapsed_time(end))
                table.append(dict(config=config, median_ms=statistics.median(samples)))
            except Exception as exc:  # e.g. Triton OutOfResources for one tile shape
                torch.cuda.synchronize(self.device)
                table.append(dict(config=config, error=f'{type(exc).__name__}: {exc}'[:300]))
        timed = [row for row in table if 'median_ms' in row]
        if not timed:
            raise RuntimeError('No candidate configuration ran: ' + '; '.join(row['error'] for row in table))
        return min(timed, key=lambda row: row['median_ms'])['config'], table

    @torch.inference_mode()
    def compare(self, case, variants, l2='warm', baseline=None):
        """Variants share resident fixtures; randomized per-round order reduces ordering bias.

        Each sample is the mean of iterations calls. warm: CUDA events bracket
        back-to-back calls, and synchronized wall time also includes host
        submission. flush: every call follows an L2 flush that is enqueued but
        not timed; each call has its own event pair, wall time is not recorded.
        Neither is a production serving latency. baseline names the variant
        used for speedup_vs_baseline (baseline median / variant median).
        """
        mode = l2
        if baseline is not None and baseline not in {v['name'] for v in variants}:
            raise ValueError('baseline must name one of the compared variants')
        if mode not in L2_MODES:
            raise ValueError('Unknown L2 timing mode')
        for value in (self.warmup, self.repeats, self.iterations):
            positive_integer(value, 'benchmark count')
        clocks = {v['name']: dict(cuda=[], wall=[]) for v in variants}
        for variant in variants:
            with variant.get('context', nullcontext)():
                for _ in range(self.warmup):
                    variant['fn']()
        torch.cuda.synchronize(self.device)
        # Initialize event objects before measured samples to avoid lazy setup cost.
        pairs = 1 if mode == 'warm' else self.iterations
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(pairs)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(pairs)]
        for start, end in zip(starts, ends):
            start.record(); end.record()
        ends[-1].synchronize()
        if mode == 'flush':
            self.flush_l2()
        order_rng = random.Random(self.seed)
        for _ in range(self.repeats):
            order = list(variants)
            order_rng.shuffle(order)
            for variant in order:
                with variant.get('context', nullcontext)():
                    torch.cuda.synchronize(self.device)
                    if mode == 'warm':
                        wall_start = time.perf_counter()
                        starts[0].record()
                        for _ in range(self.iterations):
                            variant['fn']()
                        ends[0].record()
                        ends[0].synchronize()
                        clocks[variant['name']]['wall'].append(
                            (time.perf_counter()-wall_start)*1000/self.iterations)
                        cuda_ms = starts[0].elapsed_time(ends[0])/self.iterations
                    else:
                        # 清理在 start 之前排队，GPU 忙于清理时主机继续提交，避免把启动空隙计入。
                        for start, end in zip(starts, ends):
                            self.flush_l2()
                            start.record()
                            variant['fn']()
                            end.record()
                        ends[-1].synchronize()
                        cuda_ms = sum(s.elapsed_time(e) for s, e in zip(starts, ends))/self.iterations
                clocks[variant['name']]['cuda'].append(cuda_ms)
        rows = []
        for variant in variants:
            with variant.get('context', nullcontext)():
                # Peak is measured in a separate call after synchronization.
                gc.collect()
                torch.cuda.synchronize(self.device)
                resident = torch.cuda.memory_allocated(self.device)
                torch.cuda.reset_peak_memory_stats(self.device)
                output = variant['fn']()
                torch.cuda.synchronize(self.device)
                peak = torch.cuda.max_memory_allocated(self.device)
                del output
                trace_file = None
                if self.trace_dir is not None:
                    from torch.profiler import profile, ProfilerActivity
                    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                                 record_shapes=True, profile_memory=True) as profiler:
                        variant['fn']()
                        torch.cuda.synchronize(self.device)
                    trace_file = f'{case}__{variant["name"]}.json'
                    profiler.export_chrome_trace(str(self.trace_dir/trace_file))
            wall = clocks[variant['name']]['wall']
            row = dict(case=case, variant=variant['name'], status='passed',
                       cuda=sample_summary(clocks[variant['name']]['cuda']),
                       wall=sample_summary(wall) if wall else None, l2_mode=mode,
                       flush_bytes=self.flush_bytes if mode == 'flush' else None,
                       resident_allocated_bytes=resident, peak_allocated_bytes=peak,
                       incremental_peak_bytes=max(0, peak-resident),
                       warmup=self.warmup, repeats=self.repeats, iterations=self.iterations,
                       trace=f'traces/{trace_file}' if trace_file else None, **variant.get('details', {}))
            if row.get('flops'):
                row['effective_tflops'] = row['flops']/row['cuda']['median_ms']/1e9
            if row.get('bytes_moved'):
                # bytes / ms / 1e6 = GB/s；按声明的最少读写字节计算，不是 profiler 计数。
                row['effective_gbps'] = row['bytes_moved']/row['cuda']['median_ms']/1e6
            rows.append(row)
        if baseline is not None:
            reference = next(r for r in rows if r['variant'] == baseline)['cuda']['median_ms']
            for row in rows:
                row.update(baseline_variant=baseline, speedup_vs_baseline=reference/row['cuda']['median_ms'])
        for row in rows:
            self.checkpoint(row)
        return rows


class RunReport:
    def __init__(self, parameters, output_dir=None):
        self.directory = None
        self.data = {**run_metadata(parameters, ['requirements-cuda.txt', 'experiments/CUDA.md',
                                                'tests/test_cuda_lab.py']),
                     'schema_version': 2, 'kind': 'cuda_lab', 'run_id': uuid.uuid4().hex,
                     'status': 'running', 'active_case': None, 'results': [], 'errors': [],
                     'scope': 'Synthetic CUDA operator measurements; not model quality or serving benchmarks',
                     'timing_scope': 'Repeated calls with shared resident fixtures after warmup; per-row l2_mode states '
                                     'whether L2 was flushed before each call. Includes allocations and codec input checks; '
                                     'excludes fixture creation, config tuning, correctness comparison and profiler'}
        if output_dir is not None:
            # One new folder per invocation; a failed run never reuses a previous success.
            base = Path(output_dir).resolve()
            self.directory = base/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+self.data['run_id'][:8])
            self.directory.mkdir(parents=True, exist_ok=False)
            for name in self.data['source_files_sha256']:
                target = self.directory/'sources'/name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((ROOT/name).read_bytes())
        self.save()

    def save(self):
        self.data['updated_utc'] = datetime.now(timezone.utc).isoformat()
        if self.directory:
            try:
                write_json(self.directory/'cuda_results.json', self.data)
            except (OSError, ValueError, TypeError) as exc:
                raise PersistenceError('Could not persist CUDA run status') from exc

    def add(self, row):
        self.data['results'].append(row)
        self.save()
        median = row.get('cuda', {}).get('median_ms')
        detail = f'{median:.4f} ms' if median is not None else row.get('reason', '')
        print(f'{row["status"].upper()} {row["case"]}/{row["variant"]} {detail}', flush=True)

    def annotate_bandwidth(self):
        """以本轮 bandwidth/read_sum 的实测读带宽为参照，标出各路径达到的比例。"""
        reference = next((r.get('effective_gbps') for r in self.data['results']
                          if r.get('case') == 'bandwidth' and r.get('variant') == 'read_sum'
                          and r.get('status') == 'passed'), None)
        self.data['measured_read_gbps'] = reference
        if reference:
            for row in self.data['results']:
                if row.get('effective_gbps') and row.get('case') != 'bandwidth':
                    row['bandwidth_fraction'] = row['effective_gbps']/reference

    def finish(self, status):
        self.data.update(status=status, active_case=None)
        self.annotate_bandwidth()
        self.save()
        if not self.directory:
            return
        output = io.StringIO()
        columns = ['case','variant','status','cuda_median_ms','wall_median_ms','nmse',
                   'storage_bytes','incremental_peak_bytes','effective_tflops','effective_gbps',
                   'bandwidth_fraction','speedup_vs_baseline','baseline_variant','l2_mode','reason']
        writer = csv.DictWriter(output, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        notes = []
        lines = ['# CUDA 实验结果', '', f'状态：**{status}**；run_id：`{self.data["run_id"]}`。', '']
        if self.data.get('measured_read_gbps'):
            lines.extend([f'本轮实测读带宽参照：{self.data["measured_read_gbps"]:.1f} GB/s（bandwidth/read_sum）。', ''])
        lines.extend(['| 实验 | 路径 | 状态 | CUDA ms | Wall ms | GB/s | 带宽比例 | 相对基线 | NMSE |',
                      '|---|---|---|---:|---:|---:|---:|---:|---:|'])
        show = lambda x: '—' if x is None else f'{x:.6g}'
        for row in self.data['results']:
            record = {k:row.get(k) for k in columns}
            record.update(cuda_median_ms=(row.get('cuda') or {}).get('median_ms'),
                          wall_median_ms=(row.get('wall') or {}).get('median_ms'),
                          nmse=(row.get('error') or {}).get('nmse'))
            writer.writerow(record)
            lines.append(f'| {row["case"]} | {row["variant"]} | {row["status"]} | '
                         f'{show(record["cuda_median_ms"])} | {show(record["wall_median_ms"])} | '
                         f'{show(record["effective_gbps"])} | {show(record["bandwidth_fraction"])} | '
                         f'{show(record["speedup_vs_baseline"])} | {show(record["nmse"])} |')
            if row.get('reason'):
                notes.append(f'- {row["case"]}/{row["variant"]}：{row["reason"]}')
        pairs = key_comparisons(self.data['results'])
        self.data['key_comparisons'] = pairs
        if pairs:
            lines.extend(['', '## 关键对照', '', '比值 = 参照中位数 / 候选中位数；>1 表示候选更快。', '',
                          '| 实验 | 对照 | 比值 |', '|---|---|---:|'])
            lines.extend(f'| {r["case"]} | {r["meaning"]} | {r["speedup"]:.3g} |' for r in pairs)
        if notes:
            lines.extend(['', '## 跳过与失败', '', *notes])
        lines.extend(['', '逐次时间、格式、计时范围、跳过原因与错误见 [JSON](cuda_results.json)。',
                      '每个样本是多次调用的均值，min/max 不是服务延迟分位数。',
                      'flush 模式每次调用前清 L2 且不记 Wall；warm 模式连续调用，小输入可能留在 L2。',
                      'GB/s 按声明的最少读写字节计算；带宽比例相对本轮 read_sum，不是厂商峰值。',
                      '相对基线 = 基线中位数 / 本路径中位数，>1 表示更快；基线名见 JSON 的 baseline_variant。',
                      '显存增量来自单次调用；共享的参照张量也驻留在进程，不能将峰值当作独立部署占用。',
                      'int4/kv 是先解码再浮点计算；w4a16/kv_fused 是 Triton 寄存器内解码，仍不是原生 INT4 Tensor Core 指令。'])
        if self.data['errors']:
            lines.extend(['', '运行存在失败；详细异常见 JSON，不能将部分结果当作完整通过。'])
        self.save()  # key_comparisons 也写入 JSON
        write_text(self.directory/'summary.csv', output.getvalue())
        write_text(self.directory/'summary.md', '\n'.join(lines)+'\n')
        # Includes actual executed sources. Hashes establish consistency, not authenticity.
        files = {p.relative_to(self.directory).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(self.directory.rglob('*')) if p.is_file() and p.name != 'manifest.json'}
        write_json(self.directory/'manifest.json', dict(kind='cuda_lab', files_sha256=files))
