"""Offline systems mechanisms: cache transactions, roofline, streaming and queues.

The cache experiment runs an actual random-weight CPU Transformer. Every timing
curve and request trace below is synthetic, not a GPU or serving benchmark.
Run: python -m examples.systems_lab
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from .speculative_sampling import nonnegative_int
from .tiny_transformer import TinyTransformer, crop_cache, probabilities, token_tensor


def cached_causal_mask(history_length: int, query_length: int) -> np.ndarray:
    """Mask [N, S+N]: query i is at logical position S+i."""
    history_length = nonnegative_int(history_length, 'history_length')
    query_length = nonnegative_int(query_length, 'query_length')
    return np.arange(history_length+query_length)[None, :] <= (
        history_length+np.arange(query_length)[:, None]
    )


@torch.inference_mode()
def cache_transaction_demo() -> dict:
    """Independent full-prefix oracle distinguishes two common cache mistakes."""
    with torch.random.fork_rng():
        torch.manual_seed(314)
        model = TinyTransformer(width=16, heads=2, layers=2).double().eval()
    prompt, proposed = [1, 2, 3], [4, 5, 6]
    accepted, correction = [4], 7
    _, initial = model(token_tensor(prompt[:-1], model))
    block_logits, tentative = model(token_tensor([prompt[-1], *proposed], model), initial)
    row_errors = []
    for i in range(len(proposed)+1):
        serial, _ = model(token_tensor(prompt+proposed[:i], model))
        row_errors.append(float((block_logits[0, i]-serial[0, -1]).abs().max()))
    committed = prompt+accepted+[correction]
    cropped = crop_cache(tentative, len(committed)-1)
    actual, _ = model(token_tensor([correction], model), cropped)
    oracle, _ = model(token_tensor(committed, model))
    # Wrong 1: correct token appended after the whole tentative suffix.
    stale, _ = model(token_tensor([correction], model), tentative)
    # Wrong 2: crop by one too few: retains the rejected token 5 before 7.
    extra, _ = model(token_tensor([correction], model), crop_cache(tentative, len(committed)))
    logical_bytes = sum(t.numel()*t.element_size() for pair in cropped for t in pair)
    # Views keep their parent allocations alive. Count unique underlying storages.
    storages = {(t.untyped_storage().data_ptr(), t.untyped_storage().nbytes())
                for pair in cropped for t in pair}
    return dict(
        prompt=prompt, proposed=proposed, accepted=accepted, correction=correction,
        committed=committed, input_rows=[prompt[-1], *proposed],
        initial_cache_tokens=len(prompt)-1,
        tentative_cache_tokens=len(prompt)+len(proposed),
        final_cache_tokens=len(committed)-1,
        mask=cached_causal_mask(len(prompt)-1, len(proposed)+1).astype(int).tolist(),
        parallel_logits_errors=row_errors,
        next_logits_errors=dict(correct_crop=float((actual-oracle[:, -1:]).abs().max()),
                                no_crop=float((stale-oracle[:, -1:]).abs().max()),
                                retain_one_rejected=float((extra-oracle[:, -1:]).abs().max())),
        logical_cache_bytes=logical_bytes,
        retained_storage_bytes=sum(size for _, size in storages),
    )


def roofline_bound(*, parameters: int, layers: int, query_heads: int,
                   kv_heads: int, head_dim: int, batch: int, context: int,
                   queries: int, weight_bytes: int, kv_bytes: int,
                   bandwidth: float, flops_per_second: float,
                   extra_kv_read_fraction: float = 0.0) -> dict:
    """Optimistic bulk-cost bound, with SI bytes/s and FLOP/s.

    Read weights once. Read historical KV [1 + rho*(N-1)] times. rho=0 is
    ideal sharing among verification queries; rho=1 models no such sharing.
    Includes dense linear FLOPs and the leading historical attention FLOPs;
    excludes new-token attention, activation traffic, launch and communication.
    These omissions make this a lower-bound model, not a latency predictor.
    """
    positive = dict(parameters=parameters, layers=layers, query_heads=query_heads,
                    kv_heads=kv_heads, head_dim=head_dim, batch=batch, queries=queries,
                    weight_bytes=weight_bytes, kv_bytes=kv_bytes)
    for name, value in positive.items():
        if nonnegative_int(value, name) == 0:
            raise ValueError(f'{name} must be positive')
    context = nonnegative_int(context, 'context')
    if (not all(math.isfinite(x) and x > 0 for x in [bandwidth, flops_per_second])
            or not 0 <= extra_kv_read_fraction <= 1):
        raise ValueError('Need positive hardware rates and 0 <= KV read fraction <= 1')
    kv_read_multiplier = 1+extra_kv_read_fraction*(queries-1)
    history_kv_bytes = 2*layers*batch*context*kv_heads*head_dim*kv_bytes
    writes = 2*layers*batch*queries*kv_heads*head_dim*kv_bytes
    memory_bytes = parameters*weight_bytes+history_kv_bytes*kv_read_multiplier+writes
    dense_flops = 2*parameters*batch*queries
    attention_flops = 4*layers*batch*queries*context*query_heads*head_dim
    compute_seconds = (dense_flops+attention_flops)/flops_per_second
    memory_seconds = memory_bytes/bandwidth
    return dict(queries=queries, compute_seconds=compute_seconds,
                memory_seconds=memory_seconds,
                bound_seconds=max(compute_seconds, memory_seconds),
                arithmetic_intensity=(dense_flops+attention_flops)/memory_bytes,
                history_kv_bytes=history_kv_bytes, transferred_bytes=memory_bytes,
                kv_read_multiplier=kv_read_multiplier)


def streaming_metrics(chunks, arrival=0.0) -> dict:
    """chunks = [(receive_time_seconds, number_of_new_tokens), ...].

    Every token in one network chunk shares a timestamp: these are client-visible
    intervals, not the unknown internal per-token emission timestamps.
    """
    if not chunks or not math.isfinite(arrival):
        raise ValueError('Need nonempty chunks and a finite arrival')
    previous = arrival
    token_times, chunk_times = [], []
    for timestamp, count in chunks:
        count = nonnegative_int(count, 'chunk token count')
        if count == 0 or not math.isfinite(timestamp) or timestamp < previous:
            raise ValueError('Chunks need positive counts and nondecreasing finite times')
        token_times.extend([timestamp]*count)
        chunk_times.append(timestamp)
        previous = timestamp
    n = len(token_times)
    ttft, e2e = token_times[0]-arrival, token_times[-1]-arrival
    gaps = np.diff(token_times)
    chunk_gaps = np.diff(chunk_times)
    return dict(tokens=n, ttft_seconds=ttft, e2e_seconds=e2e,
                tpot_seconds=(e2e-ttft)/(n-1) if n > 1 else None,
                max_token_gap_seconds=float(gaps.max()) if len(gaps) else None,
                max_chunk_gap_seconds=float(chunk_gaps.max()) if len(chunk_gaps) else None,
                token_timestamps=token_times, chunk_timestamps=chunk_times)


def fcfs_queue(arrivals, service_times) -> dict:
    """Finite, single-server first-come-first-served simulation, no batching.

    Exogenous arrivals are identical across methods. Service durations are fixed
    synthetic inputs. This is a counterexample lab, not an LLM scheduler model.
    """
    arrivals, services = np.asarray(arrivals, float), np.asarray(service_times, float)
    if (arrivals.ndim != 1 or services.shape != arrivals.shape or not len(arrivals)
            or not np.isfinite(arrivals).all() or not np.isfinite(services).all()
            or np.any(np.diff(arrivals) < 0) or np.any(services <= 0)):
        raise ValueError('Need sorted finite arrivals and paired positive service times')
    finished = float(arrivals[0])
    starts, ends = [], []
    for arrival, service in zip(arrivals, services):
        start = max(float(arrival), finished)
        finished = start+float(service)
        starts.append(start)
        ends.append(finished)
    waits = np.asarray(starts)-arrivals
    responses = np.asarray(ends)-arrivals
    return dict(arrivals=arrivals.tolist(), services=services.tolist(), starts=starts, ends=ends,
                waits=waits.tolist(), response_seconds=responses.tolist(),
                mean_service=float(services.mean()), mean_wait=float(waits.mean()),
                response_p95=float(np.quantile(responses, .95)))


def run_all() -> dict:
    torch.set_num_threads(1)
    common = dict(parameters=7_000_000_000, layers=32, query_heads=32, kv_heads=8,
                  head_dim=128, batch=8, context=32768, weight_bytes=2, kv_bytes=2,
                  bandwidth=3e12, flops_per_second=300e12)
    roofline = {name: [roofline_bound(**common, queries=n, extra_kv_read_fraction=rho)
                       for n in range(1, 33)]
                for name, rho in [('ideal_reuse', 0.), ('no_reuse', 1.)]}
    arrivals = np.arange(40)*.1
    constant = fcfs_queue(arrivals, [.08]*40)
    bursty = fcfs_queue(arrivals, [.02, .02, .02, .26]*10)
    smooth = streaming_metrics([(float(t), 1) for t in np.linspace(.1, .38, 8)])
    bursts = streaming_metrics([(.1, 4), (.38, 4)])
    return dict(
        note='CPU mechanism checks; all timings and service traces are synthetic, not GPU measurements.',
        environment=dict(torch=torch.__version__, numpy=np.__version__, device='cpu'),
        cache=cache_transaction_demo(), roofline=dict(assumptions=common, curves=roofline),
        streaming=dict(smooth=smooth, bursts=bursts),
        queue=dict(constant=constant, bursty=bursty),
        temperature=dict(value=1e-320, probabilities=probabilities(torch.tensor([2., 1., -1.]), 1e-320).tolist()),
    )


if __name__ == '__main__':
    result = run_all()
    path = Path(__file__).resolve().parents[1]/'data/systems_results.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(cache=result['cache'],
                         streaming={k: {field: v[field] for field in ['tpot_seconds', 'max_token_gap_seconds']}
                                    for k, v in result['streaming'].items()},
                         queue={k: {field: v[field] for field in ['mean_service', 'mean_wait', 'response_p95']}
                                for k, v in result['queue'].items()}), indent=2))
    print(f'Saved {path}')
