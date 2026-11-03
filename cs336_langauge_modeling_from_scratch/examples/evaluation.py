"""Stable pass@k/pass^k, Wilson intervals, paired bootstrap and toy-LM diagnostics."""
import argparse
import json
import math
import operator
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .model import Config, TinyLM
from .train import corpus


def _log_fraction(numerator, denominator):
    # Near 1 use log1p; a tiny fraction can instead round its complement to exactly 1.
    if 2 * numerator > denominator:
        return math.log1p(-(denominator - numerator) / denominator)
    return math.log(numerator) - math.log(denominator)


def pass_at_k(n, c, k):
    """Unbiased pass@k from n samples with c correct: 1 - C(n-c, k) / C(n, k), in a stable product form."""
    try:
        n, c, k = (operator.index(x) for x in (n, c, k))  # accepts numpy integers, rejects floats
    except TypeError as error:
        raise ValueError("n, c and k must be integers") from error
    if not (0 <= c <= n and 1 <= k <= n):
        raise ValueError("integer counts must satisfy 0 <= c <= n and 1 <= k <= n")
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return -math.expm1(math.fsum(_log_fraction(n - c - i, n - i) for i in range(k)))


def pass_all_k(n, c, k):
    """Unbiased pass^k: all k independent attempts on the SAME task must succeed.

    Average C(c,k)/C(n,k) over tasks; raising the pooled accuracy to k loses task heterogeneity.
    """
    try:
        n, c, k = (operator.index(x) for x in (n, c, k))
    except TypeError as error:
        raise ValueError("n, c and k must be integers") from error
    if not (0 <= c <= n and 1 <= k <= n):
        raise ValueError("integer counts must satisfy 0 <= c <= n and 1 <= k <= n")
    if c < k:
        return 0.0
    if c == n:
        return 1.0
    return math.exp(math.fsum(_log_fraction(c - i, n - i) for i in range(k)))


def wilson_interval(successes, total, z=1.959963984540054):
    try:
        successes, total = operator.index(successes), operator.index(total)
    except TypeError as error:
        raise ValueError("integer trial and success counts required") from error
    if total < 1 or not 0 <= successes <= total or not math.isfinite(z) or z <= 0:
        raise ValueError("valid counts and positive z required")
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def paired_bootstrap(a, b, repetitions=2000, seed=17):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or a.size == 0 or repetitions < 1:
        raise ValueError("matching nonempty per-item score vectors and positive repetitions required")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("scores must be finite")
    rng = np.random.default_rng(seed)
    delta = b - a
    values = [delta[rng.integers(len(delta), size=len(delta))].mean() for _ in range(repetitions)]
    return dict(difference=float(delta.mean()), interval_95=np.quantile(values, [.025, .975]).tolist())


def paired_cluster_bootstrap(a, b, clusters, repetitions=2000, seed=17, weighting="item"):
    """Resample independent clusters, keeping all paired items within each draw.

    `item` estimates total paired difference / total items; `cluster` gives
    each cluster mean equal weight. Every draw samples G clusters with replacement,
    including multiplicity. This covers cluster sampling, not training-seed noise
    or model selection, and needs at least two independent clusters.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    labels = list(clusters)
    if a.ndim != 1 or a.shape != b.shape or a.size != len(labels) or not a.size \
            or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("finite matching per-item scores and one cluster ID per pair required")
    if type(repetitions) is not int or repetitions < 2 or weighting not in ("item", "cluster"):
        raise ValueError("at least two repetitions and item/cluster weighting required")
    groups = {}
    for index, label in enumerate(labels):
        if label is None or (isinstance(label, (float, np.floating)) and not math.isfinite(label)):
            raise ValueError("cluster IDs cannot be missing or nonfinite")
        try:
            groups.setdefault(label, []).append(index)
        except TypeError as error:
            raise ValueError("hashable cluster IDs required") from error
    if len(groups) < 2:
        raise ValueError("at least two independent clusters required")
    delta = b - a
    totals = np.array([delta[indices].sum() for indices in groups.values()])
    sizes = np.array([len(indices) for indices in groups.values()])
    means = totals / sizes
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(repetitions):
        draw = rng.integers(len(groups), size=len(groups))
        samples.append(totals[draw].sum() / sizes[draw].sum() if weighting == "item" else means[draw].mean())
    return dict(difference=float(delta.mean() if weighting == "item" else means.mean()),
                interval_95=np.quantile(samples, [.025, .975]).tolist(),
                clusters=len(groups), weighting=weighting)


def elo_win_probability(rating_a, rating_b):
    """Arena-style Elo: Bradley-Terry with base-10 logits scaled by 400."""
    if not math.isfinite(rating_a) or not math.isfinite(rating_b):
        raise ValueError("finite ratings required")
    delta = (rating_a - rating_b) * (math.log(10) / 400)
    small = math.exp(-abs(delta))
    return 1 / (1 + small) if delta >= 0 else small / (1 + small)


@torch.no_grad()
def arithmetic_diagnostics(checkpoint, records_per_window=5):
    """Per-role NLL and exact-answer accuracy for the synthetic arithmetic model.

    Windows hold whole 6-byte records and stay inside the training context, so
    every position the model is scored on was seen during training.
    """
    torch.set_num_threads(2)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = TinyLM(Config(**saved["config"]))
    model.load_state_dict(saved["model"])
    model.eval()
    window = 6 * records_per_window
    tokens = corpus(2, 600)[: (3600 // window) * window].reshape(-1, window)
    logits = model(tokens[:, :-1])
    nll = F.cross_entropy(logits.reshape(-1, 256), tokens[:, 1:].reshape(-1), reduction="none").reshape(len(tokens), -1)
    # Target index i predicts byte i+1 of the window; byte offset within a record decides its role.
    offsets = torch.arange(1, window) % 6
    roles = {"operand": (offsets == 0) | (offsets == 2), "format": (offsets == 1) | (offsets == 3) | (offsets == 5),
             "answer": offsets == 4}
    result = {name: dict(tokens=int(mask.sum()) * len(tokens), nll=nll[:, mask].mean().item())
              for name, mask in roles.items()}
    prompts = torch.tensor([list(f"{a}+{b}=".encode()) for a in range(10) for b in range(10)])
    targets = torch.tensor([ord(str((a + b) % 10)) for a in range(10) for b in range(10)])
    last = model(prompts)[:, -1]
    successes = int((last.argmax(-1) == targets).sum())
    return dict(checkpoint_step=saved["step"], position_metrics=result,
                all_100_prompt_accuracy=successes / 100, correct_prompts=successes,
                answer_nll_from_fresh_prompt=F.cross_entropy(last, targets).item(),
                exact_arithmetic_reference_nats_per_byte=2 * math.log(10) / 6,
                scope="all 100 facts also occur in training text; this measures fitting seen facts, not OOD arithmetic")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument("--out")
    args = parser.parse_args()
    result = arithmetic_diagnostics(args.checkpoint) if args.checkpoint else {
        "pass_at_3": pass_at_k(10, 2, 3), "wilson_50_of_100": wilson_interval(50, 100),
        "paired_example": paired_bootstrap([0, 1, 1, 0], [1, 1, 1, 0]),
        "elo_100_point_gap_win_probability": elo_win_probability(1100, 1000)}
    print(json.dumps(result, indent=2))
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
