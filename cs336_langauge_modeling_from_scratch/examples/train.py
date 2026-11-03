"""Train the guide's decoder on a token stream, with exact interruption and resume.

Two data sources share one loop:
* default: an original synthetic modular-addition byte corpus (seconds on CPU);
* --train/--valid: 1-D integer token arrays saved as .npy (e.g. TinyStories IDs).

Teaching reference; coursework restrictions and this demo's scope are explained in chapter 28.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import stat
import time
import tempfile
import numpy as np
import torch
from torch.nn import functional as F
from .model import Config, TinyLM, generate
from .optim import AdamW, clip_grad_norm, cosine_lr


def corpus(seed, count):
    rng = random.Random(seed)
    rows = [f"{a}+{b}={(a+b)%10}\n" for a, b in ((rng.randrange(10), rng.randrange(10)) for _ in range(count))]
    return torch.tensor(list("".join(rows).encode()), dtype=torch.long)


def load_tokens(path):
    """Memory-map a 1-D integer .npy file; only sampled windows are read."""
    data = np.load(path, mmap_mode="r", allow_pickle=False)
    if data.ndim != 1 or not np.issubdtype(data.dtype, np.integer):
        raise ValueError("expected a one-dimensional integer token array")
    return data


def windows(data, starts, context):
    """Inputs and next-token targets for windows of context+1 tokens."""
    starts = list(starts)
    if type(context) is not int or context < 1 or not starts or any(
        not isinstance(s, (int, np.integer)) or s < 0 or s + context >= len(data) for s in starts
    ):
        raise ValueError("positive context and starts whose windows contain context+1 tokens required")
    if isinstance(data, torch.Tensor):
        full = torch.stack([data[s:s + context + 1] for s in starts])
    else:
        full = torch.from_numpy(np.stack([data[s:s + context + 1] for s in starts]).astype(np.int64))
    return full[:, :-1], full[:, 1:]


def batch(data, batch_size, context, generator):
    starts = torch.randint(len(data) - context, (batch_size,), generator=generator).tolist()
    return windows(data, starts, context)


def parameter_counts(model):
    total = sum(p.numel() for p in model.parameters())
    embedding = model.embedding.weight.numel()
    head = 0 if model.cfg.tie_embeddings else model.head.weight.numel()
    return dict(total=total, non_embedding=total - embedding - head)


def data_fingerprint(data):
    """Hash logical token content in bounded chunks, including dtype and length."""
    array = data.numpy() if isinstance(data, torch.Tensor) else data
    digest = hashlib.sha256(f"{array.dtype}:{array.shape}".encode())
    for start in range(0, len(array), 1 << 20):
        digest.update(np.asarray(array[start:start + (1 << 20)]).tobytes())
    return digest.hexdigest()


def save_checkpoint(checkpoint, path):
    """A failed serialization must leave the previous resumable checkpoint intact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else None
    # A regular file inside a private staging directory gets the process's umask
    # naturally; reading it via os.umask(0) would alter concurrent file creation.
    with tempfile.TemporaryDirectory(dir=path.parent, prefix='.checkpoint-') as directory:
        temporary = Path(directory)/'checkpoint.pt'
        with temporary.open('wb') as file:
            torch.save(checkpoint, file)
            file.flush()
            os.fsync(file.fileno())
        if mode is not None:
            temporary.chmod(mode)
        temporary.replace(path)


def run(steps=120, out="outputs/tiny", seed=7, stop_after=None, resume=None, train=None, valid=None,
        config=None, lr=2e-3, min_lr_ratio=0.1, warmup=None, batch_size=16, context=32,
        weight_decay=0.1, betas=(0.9, 0.95), max_grad_norm=1.0, eval_every=20, eval_windows=384,
        threads=2, divergence_loss=None, verbose=True):
    """steps is the planned schedule horizon, even when stopping early.

    Resume is exact on CPU: same data, configuration, optimizer state, global
    and batch RNG, and the same planned schedule horizon.
    """
    if type(steps) is not int or steps < 1 or (stop_after is not None and (type(stop_after) is not int or not 0 < stop_after <= steps)):
        raise ValueError("positive steps required; stop_after must be within the planned run")
    if any(type(x) is not int or x < 1 for x in (batch_size, context, eval_every, eval_windows, threads)) or (train is None) != (valid is None):
        raise ValueError("positive batch/context/evaluation/threads and paired train/valid files required")
    if not math.isfinite(min_lr_ratio) or not 0 <= min_lr_ratio <= 1 or not math.isfinite(max_grad_norm) or max_grad_norm < 0:
        raise ValueError("finite min_lr_ratio in [0,1] and nonnegative max_grad_norm required")
    if divergence_loss is not None and (not math.isfinite(divergence_loss) or divergence_loss <= 0):
        raise ValueError("divergence_loss must be None or finite and positive")
    warmup = min(10, steps - 1) if warmup is None else warmup
    cosine_lr(0, lr, lr * min_lr_ratio, warmup, steps)
    AdamW._validate(lr, betas, 1e-8, weight_decay)
    saved = torch.load(resume, map_location="cpu", weights_only=True) if resume else None
    if saved:
        if saved.get("total_steps") != steps:
            raise ValueError("resuming must keep the original learning-rate schedule horizon")
        seed = saved["seed"]
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed + 10)
    if train is None:
        # IID draws with overlapping arithmetic facts: a distribution-learning smoke test.
        train_data, valid_data = corpus(1, 1500), corpus(2, 200)
    else:
        train_data, valid_data = load_tokens(train), load_tokens(valid)
    if min(len(train_data), len(valid_data)) <= context:
        raise ValueError("both datasets need at least context+1 tokens")
    cfg = Config(**(config or {}))
    for name, data in (("train", train_data), ("valid", valid_data)):
        low, high = int(data.min()), int(data.max())
        if low < 0 or high >= cfg.vocab:
            raise ValueError(f"{name} token IDs span [{low}, {high}] but the model vocabulary is {cfg.vocab}")
    if train is None:
        xv, yv = batch(valid_data, 24, context, torch.Generator().manual_seed(19))
    else:
        # Fixed, evenly spaced validation windows: identical for every run and checkpoint.
        count = min(eval_windows, len(valid_data) - context)
        stride = max(1, (len(valid_data) - context - 1) // count)
        xv, yv = windows(valid_data, range(0, stride * count, stride), context)
    model = TinyLM(cfg)
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    opt = AdamW([{"params": decay}, {"params": no_decay, "weight_decay": 0.0}],
                lr=lr, betas=betas, weight_decay=weight_decay)
    contract = dict(
        lr=lr, min_lr_ratio=min_lr_ratio, warmup=warmup, batch_size=batch_size,
        context=context, weight_decay=weight_decay, betas=list(betas), max_grad_norm=max_grad_norm,
        eval_every=eval_every, eval_windows=eval_windows, threads=threads, divergence_loss=divergence_loss,
        train_sha256=data_fingerprint(train_data), valid_sha256=data_fingerprint(valid_data),
        torch=str(torch.__version__),
        implementation={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                        for name in ('train.py', 'model.py', 'optim.py')},
    )
    history, start_step, diverged, running = [], 0, False, []
    if saved:
        if saved.get('resume_contract') != contract:
            previous = saved.get('resume_contract', {})
            changed = [key for key in contract if previous.get(key) != contract[key]]
            raise ValueError("exact resume requires identical settings/data/implementation: " + ', '.join(changed))
        if saved.get('diverged', False):
            raise ValueError("cannot exactly resume a diverged checkpoint")
        if saved["config"] != asdict(cfg):
            raise ValueError("checkpoint configuration differs from this experiment")
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["torch_rng"])
        g.set_state(saved["batch_rng"])
        history, start_step = saved["history"], saved["step"]
        running.extend(saved["running_losses"])
    end_step = steps if stop_after is None else stop_after
    if end_step <= start_step:
        raise ValueError("resume endpoint must be after the saved step")
    tokens_per_step = batch_size * context
    started = time.perf_counter()

    @torch.no_grad()
    def evaluate(step, grad_norm=None, current_lr=None):
        nonlocal diverged
        model.eval()
        total = 0.0
        for i in range(0, len(xv), 64):
            logits = model(xv[i:i + 64])
            total += F.cross_entropy(logits.reshape(-1, cfg.vocab), yv[i:i + 64].reshape(-1), reduction="sum").item()
        validation_loss = total / yv.numel()
        record = dict(step=step, tokens=step * tokens_per_step,
                      val_loss=validation_loss if math.isfinite(validation_loss) else None)
        if not math.isfinite(validation_loss):
            record['nonfinite_validation_loss'] = True
            record['diverged'] = True
            diverged = True
        if running:
            # Wall time stays out of the history so resumed and continuous runs compare exactly.
            record.update(train_loss=sum(running) / len(running), lr=current_lr, grad_norm=grad_norm)
            running.clear()
        history.append(record)
        if verbose:
            print(json.dumps(record), flush=True)
        model.train()

    if not saved:
        evaluate(0)
    grad_norm = None
    completed = start_step  # Count only optimizer updates that were actually applied.
    for step in range(start_step, end_step):
        if diverged:
            break
        x, y = batch(train_data, batch_size, context, g)
        current_lr = cosine_lr(step, lr, lr * min_lr_ratio, warmup, steps)
        for group in opt.param_groups:
            group["lr"] = current_lr
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x).reshape(-1, cfg.vocab), y.reshape(-1))
        # Non-finite loss is divergence at any step; a loss ceiling applies once warmup is over.
        if not math.isfinite(loss.item()) or (divergence_loss and step >= warmup and loss.item() > divergence_loss):
            diverged, completed = True, step
            history.append(dict(step=step, tokens=step * tokens_per_step, diverged=True,
                                train_loss=loss.item() if math.isfinite(loss.item()) else None,
                                nonfinite_loss=not math.isfinite(loss.item())))
            break
        loss.backward()
        # Finite loss with overflowing gradients is divergence too: stop and checkpoint, don't crash.
        if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            diverged, completed = True, step
            history.append(dict(step=step, tokens=step * tokens_per_step, diverged=True,
                                train_loss=loss.item(), nonfinite_gradients=True))
            break
        try:
            grad_norm = clip_grad_norm(model.parameters(), max_grad_norm).item()
        except ValueError as error:
            if 'norm exceeds' not in str(error):
                raise
            diverged, completed = True, step
            history.append(dict(step=step, tokens=step * tokens_per_step, diverged=True,
                                train_loss=loss.item(), nonfinite_grad_norm=True))
            break
        opt.step()
        completed = step + 1
        running.append(loss.item())
        if (step + 1) % eval_every == 0 or step + 1 == steps:
            evaluate(step + 1, grad_norm, current_lr)
    elapsed = time.perf_counter() - started
    path = Path(out)
    path.mkdir(parents=True, exist_ok=True)
    save_checkpoint(dict(model=model.state_dict(), optimizer=opt.state_dict(), config=asdict(cfg),
                    step=completed, total_steps=steps, seed=seed, history=history, running_losses=running,
                    torch_rng=torch.get_rng_state(), batch_rng=g.get_state(),
                    resume_contract=contract, diverged=diverged), path / "checkpoint.pt")
    result = dict(seed=seed, steps=completed, total_steps=steps, torch=str(torch.__version__), device="cpu",
                  config=asdict(cfg), parameters=parameter_counts(model), history=history,
                  hyperparameters=dict(lr=lr, min_lr_ratio=min_lr_ratio, warmup=warmup, batch_size=batch_size,
                                       context=context, weight_decay=weight_decay, betas=list(betas),
                                       max_grad_norm=max_grad_norm),
                  diverged=diverged, wall_seconds=round(elapsed, 1),
                  tokens_per_second=round((completed - start_step) * tokens_per_step / elapsed, 1),
                  data="synthetic arithmetic bytes" if train is None else str(train))
    (path / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    if train is None and verbose and not diverged:
        sample = generate(model, torch.tensor([list(b"2+3=")]), new_tokens=30, temperature=0)
        print(bytes(sample[0].tolist()).decode("utf-8", errors="replace"))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--out", default="outputs/tiny")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--train", help="1-D integer .npy token file")
    parser.add_argument("--valid", help="1-D integer .npy token file")
    parser.add_argument("--config", default="{}", help='JSON Config overrides, e.g. {"norm":"post"}')
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--context", type=int, default=32)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    if (args.train is None) != (args.valid is None):
        parser.error("--train and --valid must be given together")
    run(args.steps, args.out, stop_after=args.stop_after, resume=args.resume, train=args.train,
        valid=args.valid, config=json.loads(args.config), lr=args.lr, batch_size=args.batch_size,
        context=args.context, threads=args.threads)
