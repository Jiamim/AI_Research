"""A CPU-sized version of the A1 TinyStories experiments, end to end.

Steps: train a 10K byte-level BPE on TinyStories text, encode train/valid to
uint16 .npy, then run a learning-rate sweep and the A1 architecture ablations
(post-norm, no norm at three learning rates, NoPE, ungated SiLU, tied
embeddings) in parallel CPU processes. Finished runs are cached per name. Results go to outputs/tinystories/ and a compact summary to
book/tinystories-results.json.

    python scripts/run_tinystories.py --download          # fetch data (~220 MB)
    python scripts/run_tinystories.py --train-text PATH --valid-text PATH
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import math
import multiprocessing as mp
import re
import statistics
from pathlib import Path
import sys
import tempfile
import time
from urllib.request import Request, urlopen
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.tokenizer import Tokenizer, chunk_boundaries, count_pretokens_file, train_bpe_from_counts
from examples.model import Config, TinyLM, generate
from examples.train import run
from scripts.artifact_io import atomic_output, file_sha256, publish_files, write_json

URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-{}.txt"
EOT = "<|endoftext|>"
BASE = dict(vocab=10000, width=256, layers=4, heads=8, kv_heads=8, hidden=704)
TRAIN = dict(steps=1500, batch_size=32, context=128, eval_every=100, eval_windows=512, warmup=75)


def pool_context():
    # PyTorch may already have active OpenMP threads; fork can inherit locked thread-pool state.
    return mp.get_context('spawn')


def init_runs(out, threads, data_hashes):
    global OUT, THREADS, DATA_HASHES
    OUT, THREADS, DATA_HASHES = Path(out), threads, data_hashes


def data_contract(train_text, valid_text):
    return dict(train_sha256=file_sha256(train_text), valid_sha256=file_sha256(valid_text),
                vocab_size=BASE['vocab'], special_token=EOT,
                tokenizer_sha256=file_sha256(ROOT/'examples/tokenizer.py'),
                encoding_script_sha256=file_sha256(Path(__file__)))


def run_contract(config, settings):
    return dict(config=config, settings=settings, threads=THREADS, divergence_loss=10.0,
                data=DATA_HASHES, torch=str(torch.__version__),
                runner_sha256=file_sha256(Path(__file__)),
                implementation={name:file_sha256(ROOT/'examples'/name)
                                for name in ('train.py','model.py','optim.py')})


def cache_record(path):
    try:
        value = json.loads(Path(path).read_text(encoding='utf-8'))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def final_validation_loss(result):
    """Only a completed, non-diverged run with a finite validation score is eligible."""
    if result.get('diverged') or not result.get('history'):
        return None
    loss = result['history'][-1].get('val_loss')
    if type(loss) not in (int, float) or not math.isfinite(loss):
        return None
    if 'total_steps' in result and result.get('steps') != result['total_steps']:
        return None
    return loss


def require_baseline(runs, name):
    matches = [result for result in runs if result['name'] == name]
    if len(matches) != 1 or final_validation_loss(matches[0]) is None:
        raise RuntimeError(f'baseline {name!r} did not finish with a finite validation loss; inspect its metrics before generating a summary')
    return matches[0]


def download(split, path, max_bytes=None):
    path = Path(path)
    if split not in ('train', 'valid'):
        raise ValueError('download split must be train or valid')
    if max_bytes is not None and (type(max_bytes) is not int or max_bytes < 1):
        raise ValueError('positive download byte limit required')
    request = dict(url=URL.format(split), max_bytes=max_bytes)
    metadata = path.with_name(path.name+'.download.json')
    if path.exists():
        if not metadata.is_file():
            raise ValueError('existing text has no download record; use it without --download or choose a new download path')
        try:
            previous = json.loads(metadata.read_text(encoding='utf-8'))
        except (OSError, ValueError) as error:
            raise ValueError('download record is unreadable; use the existing text without --download or choose a new path') from error
        if not isinstance(previous, dict):
            raise ValueError('download record must be a JSON object')
        if previous.get('request') == request and previous.get('sha256') == file_sha256(path) \
                and previous.get('bytes') == path.stat().st_size:
            return
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = {"Range": f"bytes=0-{max_bytes - 1}"} if max_bytes else {}
    with tempfile.TemporaryDirectory(prefix='.download-', dir=path.parent) as directory:
        staging = Path(directory)
        payload, record = staging/'payload', staging/'record.json'
        with urlopen(Request(request['url'], headers=headers), timeout=600) as response, payload.open('wb') as output:
            response_headers = getattr(response, 'headers', {})
            length = response_headers.get('Content-Length')
            declared = None if length is None else int(length)
            if declared is not None and declared < 0:
                raise OSError('negative Content-Length in download response')
            status = getattr(response, 'status', None)
            range_bytes = None
            if status == 206:
                content_range = response_headers.get('Content-Range', '')
                match = re.fullmatch(r'bytes 0-(\d+)/(\d+|\*)', content_range)
                if max_bytes is None or match is None:
                    raise OSError('partial download response does not start at the requested corpus prefix')
                range_bytes = int(match[1]) + 1
                total = None if match[2] == '*' else int(match[2])
                wanted = max_bytes if total is None else min(max_bytes, total)
                if range_bytes != wanted or (declared is not None and declared != range_bytes):
                    raise OSError('partial response does not cover the complete requested corpus prefix')
            remaining = max_bytes
            received = 0
            while remaining is None or remaining > 0:
                chunk = response.read(1 << 20 if remaining is None else min(1 << 20, remaining))
                if not chunk:
                    break
                output.write(chunk)
                received += len(chunk)
                if remaining is not None:
                    remaining -= len(chunk)
            expected = range_bytes if range_bytes is not None else (
                declared if max_bytes is None else (min(declared, max_bytes) if declared is not None else None))
            if received == 0 or (expected is not None and received != expected):
                raise OSError(f'incomplete download: received {received} bytes, expected {expected}')
        write_json(record, dict(request=request, bytes=received, sha256=file_sha256(payload)))
        publish_files([(payload, path), (record, metadata)])


_TOKENIZER = None


def _init_worker(vocab, merges):
    global _TOKENIZER
    _TOKENIZER = Tokenizer(vocab, merges, [EOT])


def _encode_span(args):
    path, start, end = args
    with open(path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")
    return np.asarray(_TOKENIZER.encode(text), dtype=np.uint16)


def encode_file(path, vocab, merges, out, workers=16):
    if not vocab or max(vocab) > np.iinfo(np.uint16).max:
        raise ValueError('token IDs must fit the uint16 dataset format')
    bounds = chunk_boundaries(path, workers * 4, EOT.encode())
    with ProcessPoolExecutor(workers, mp_context=pool_context(), initializer=_init_worker, initargs=(vocab, merges)) as pool:
        parts = list(pool.map(_encode_span, [(path, a, b) for a, b in zip(bounds, bounds[1:])]))
    ids = np.concatenate(parts)
    with atomic_output(out) as stream:
        np.save(stream, ids)
    return int(ids.size)


def _run(job):
    name, overrides = job
    overrides = dict(overrides)
    config = {**BASE, **overrides.pop("config", {})}
    settings = {**TRAIN, **overrides}
    if DATA_HASHES != {name:file_sha256(OUT/name) for name in DATA_HASHES}:
        raise ValueError('token data changed after worker initialization; restart the experiment')
    expected = run_contract(config, settings)
    cached = OUT / "runs" / name / "metrics.json"
    metadata = cached.with_name('cache-contract.json')
    checkpoint = cached.with_name('checkpoint.pt')
    if cached.exists() and metadata.exists() and checkpoint.exists():
        previous = cache_record(metadata)
        if previous.get('contract') == expected and previous.get('metrics_sha256') == file_sha256(cached) \
                and previous.get('checkpoint_sha256') == file_sha256(checkpoint):
            return json.loads(cached.read_text(encoding="utf-8")) | {"name": name}
    cached.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.run-', dir=cached.parent) as directory:
        staging = Path(directory)
        result = run(out=str(staging), train=str(OUT / "train.npy"), valid=str(OUT / "valid.npy"),
                     config=config, threads=THREADS, divergence_loss=10.0, verbose=False, **settings)
        if expected != run_contract(config, settings) or DATA_HASHES != {name:file_sha256(OUT/name) for name in DATA_HASHES}:
            raise RuntimeError('source or token data changed during training; cached run was retained')
        write_json(staging/metadata.name, dict(contract=expected,
                   metrics_sha256=file_sha256(staging/cached.name), checkpoint_sha256=file_sha256(staging/checkpoint.name)))
        publish_files([(staging/path.name, path) for path in (checkpoint, cached, metadata)])
    result["name"] = name
    print(name, "final val", result["history"][-1].get("val_loss"), "diverged", result["diverged"], flush=True)
    return result


def prepare_data(train_text, valid_text, out, workers=16):
    """Generate tokenizer, token arrays and their identity record as a single publication."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    contract = data_contract(train_text, valid_text)
    previous = cache_record(out/'data.json')
    artifacts = ('train.npy', 'valid.npy', 'tokenizer.json')
    if previous.get('contract') == contract and all((out/name).is_file() for name in artifacts) \
            and previous.get('artifact_sha256') == {name:file_sha256(out/name) for name in artifacts}:
        return previous
    started = time.perf_counter()
    counts = count_pretokens_file(train_text, [EOT], workers=workers)
    vocab, merges = train_bpe_from_counts(counts, BASE['vocab'], [EOT])
    tokenizer_seconds = time.perf_counter() - started
    with tempfile.TemporaryDirectory(prefix='.tokens-', dir=out) as directory:
        staging = Path(directory)
        write_json(staging/'tokenizer.json', dict(vocab={i:b.hex() for i,b in vocab.items()},
                   merges=[[a.hex(), b.hex()] for a,b in merges]))
        train_tokens = encode_file(train_text, vocab, merges, staging/'train.npy', workers=workers)
        valid_tokens = encode_file(valid_text, vocab, merges, staging/'valid.npy', workers=workers)
        if not train_tokens or not valid_tokens:
            raise ValueError('training and validation text must produce nonempty token streams')
        if data_contract(train_text, valid_text) != contract:
            raise RuntimeError('source text or tokenizer changed during preprocessing; prior data was retained')
        longest = max(vocab.values(), key=len)
        record = dict(contract=contract, artifact_sha256={name:file_sha256(staging/name) for name in artifacts},
                      tokenizer_seconds=round(tokenizer_seconds, 1), train_bytes=Path(train_text).stat().st_size,
                      valid_bytes=Path(valid_text).stat().st_size, train_tokens=train_tokens, valid_tokens=valid_tokens,
                      bytes_per_token=round(Path(valid_text).stat().st_size/valid_tokens, 3),
                      longest_token=longest.decode('utf-8', errors='replace'), last_merges=[
                          (a+b).decode('utf-8', errors='replace') for a,b in merges[-8:]])
        write_json(staging/'data.json', record)
        publish_files([(staging/name, out/name) for name in (*artifacts, 'data.json')])
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--train-text")
    parser.add_argument("--valid-text")
    parser.add_argument("--train-bytes", type=int, default=200 * 2**20)
    parser.add_argument("--parallel", type=int, default=6)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--best-lr", type=float, help="skip the sweep and use this learning rate")
    parser.add_argument("--seed-repeats", type=int, nargs="+", metavar="SEED",
                        help="only rerun the base configuration with these extra seeds and record the spread")
    args = parser.parse_args()
    if min(args.parallel,args.threads,args.train_bytes) < 1 or (args.best_lr is not None and (not math.isfinite(args.best_lr) or args.best_lr <= 0)):
        parser.error('positive worker/thread/byte counts and finite positive learning rate required')
    global OUT, THREADS, DATA_HASHES
    OUT, THREADS = ROOT / "outputs/tinystories", args.threads
    OUT.mkdir(parents=True, exist_ok=True)
    train_text = Path(args.train_text) if args.train_text else OUT / "train.txt"
    valid_text = Path(args.valid_text) if args.valid_text else OUT / "valid.txt"
    if args.download:
        download("train", train_text, args.train_bytes)
        download("valid", valid_text)
    results_path = ROOT / "book/tinystories-results.json"
    if args.seed_repeats:
        DATA_HASHES={name:file_sha256(OUT/name) for name in ('train.npy','valid.npy','tokenizer.json')}
        previous=json.loads(results_path.read_text(encoding='utf-8'))
        if previous.get('data',{}).get('artifact_sha256') != DATA_HASHES:
            # Records written before hashes were stored cannot prove which token data they used.
            hint = (' (the stored results predate data hashing: rerun the base experiment once, then repeat seeds)'
                    if 'artifact_sha256' not in previous.get('data',{}) else '')
            raise ValueError('seed-repeat baseline does not match current token data; rerun the base experiment'+hint)
        expected=run_contract(BASE,{**TRAIN,'lr':previous['best_lr']})
        if previous.get('baseline_contract')!=expected:
            raise ValueError('baseline settings or source changed; rerun the base experiment before seed repeats')
        record_seed_repeats(results_path, args.seed_repeats, args.parallel)
        return
    summary = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), torch=str(torch.__version__),
                   device="cpu", base_config=BASE, train_settings=TRAIN)
    artifacts=('train.npy','valid.npy','tokenizer.json')
    summary['data'] = prepare_data(train_text, valid_text, OUT)
    DATA_HASHES={name:file_sha256(OUT/name) for name in artifacts}
    with ProcessPoolExecutor(args.parallel,mp_context=pool_context(),initializer=init_runs,initargs=(str(OUT),THREADS,DATA_HASHES)) as pool:
        if args.best_lr is None:
            sweep = list(pool.map(_run, [(f"lr_{lr:g}", dict(lr=lr)) for lr in [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]]))
            finished = [r for r in sweep if final_validation_loss(r) is not None]
            if not finished:
                raise RuntimeError('all learning-rate trials diverged; no baseline can be selected')
            best_lr = min(finished, key=final_validation_loss)["hyperparameters"]["lr"]
        else:
            sweep, best_lr = [], args.best_lr
        ablations = {
            "post_norm": dict(config=dict(norm="post")),
            "no_norm": dict(config=dict(norm="none")),
            "no_norm_lr_div3": dict(config=dict(norm="none"), lr=best_lr / 3),
            "no_norm_lr_div10": dict(config=dict(norm="none"), lr=best_lr / 10),
            "nope": dict(config=dict(position="none")),
            "silu_ffn": dict(config=dict(ffn="silu", hidden=1024)),
            "tied_embeddings": dict(config=dict(tie_embeddings=True)),
        }
        jobs = [(name, {"lr": best_lr, **spec}) for name, spec in ablations.items()]
        if args.best_lr is not None:
            jobs.append(("base", dict(lr=best_lr)))
        ablation = list(pool.map(_run, jobs))
    runs = sweep + ablation
    base_name = f"lr_{best_lr:g}" if args.best_lr is None else "base"
    require_baseline(runs, base_name)
    checkpoint = torch.load(OUT / "runs" / base_name / "checkpoint.pt", weights_only=True)
    model = TinyLM(Config(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    vocab_hex = json.loads((OUT / "tokenizer.json").read_text(encoding="utf-8"))
    tokenizer = Tokenizer({int(i): bytes.fromhex(b) for i, b in vocab_hex["vocab"].items()},
                          [(bytes.fromhex(a), bytes.fromhex(b)) for a, b in vocab_hex["merges"]], [EOT])
    samples = {}
    for temperature, top_p in [(0.0, 1.0), (0.8, 0.95)]:
        prompt = torch.tensor([tokenizer.encode("Once upon a time")])
        ids = generate(model, prompt, 160, temperature, top_p, eos_id=tokenizer.special_ids[EOT],
                       generator=torch.Generator().manual_seed(0))
        samples[f"T={temperature},top_p={top_p}"] = tokenizer.decode(ids[0].tolist())
    baseline_contract=json.loads((OUT/'runs'/base_name/'cache-contract.json').read_text(encoding='utf-8'))['contract']
    summary.update(best_lr=best_lr, base_run=base_name, baseline_contract=baseline_contract, samples=samples, runs=[dict(
        name=r["name"], lr=r["hyperparameters"]["lr"], config=r["config"], parameters=r["parameters"],
        diverged=r["diverged"], wall_seconds=r["wall_seconds"], tokens_per_second=r["tokens_per_second"],
        history=[{k: (round(v, 4) if isinstance(v, float) else v) for k, v in h.items()} for h in r["history"]])
        for r in runs])
    write_json(results_path, summary)
    print(json.dumps(dict(best_lr=best_lr, samples=samples), ensure_ascii=False, indent=1))


def record_seed_repeats(results_path, seeds, parallel):
    """Rerun the base configuration (best learning rate, default seed 7) with extra seeds."""
    summary = json.loads(results_path.read_text(encoding="utf-8"))
    best_lr = summary["best_lr"]
    base = require_baseline(summary['runs'], summary['base_run'])
    seeds=list(dict.fromkeys(seeds))
    if len(set([7,*seeds])) < 2:
        raise ValueError('seed repeats require at least one seed different from baseline seed 7')
    with ProcessPoolExecutor(parallel,mp_context=pool_context(),initializer=init_runs,initargs=(str(OUT),THREADS,DATA_HASHES)) as pool:
        repeats = list(pool.map(_run, [(f"seed_{seed}", dict(lr=best_lr, seed=seed)) for seed in seeds]))
    if any(final_validation_loss(r) is None for r in repeats):
        raise RuntimeError('some seed repeats diverged; inspect their metrics before aggregating')
    losses = {"7": base["history"][-1]["val_loss"]}
    losses |= {str(seed): round(r["history"][-1]["val_loss"], 4) for seed, r in zip(seeds, repeats)}
    values = list(losses.values())
    summary["base_seed_repeats"] = dict(seeds=losses, mean=round(statistics.mean(values), 4),
                                        stdev=round(statistics.stdev(values), 4),
                                        range=round(max(values) - min(values), 4))
    write_json(results_path, summary)
    print(json.dumps(summary["base_seed_repeats"], indent=1))


if __name__ == "__main__":
    main()
