"""Actual causal Transformer and KV rollback, runnable offline on CPU.

This pedagogical implementation uses NumPy sampling and Python loops. It does
not claim a latency speedup. Run: python -m examples.tiny_transformer
"""
from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn
from .speculative_sampling import nonnegative_int, token_ids, verify_candidates

Cache = list[tuple[torch.Tensor, torch.Tensor]] | None


class AttentionBlock(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width))

    def forward(self, x: torch.Tensor, past: tuple | None):
        b, n, width = x.shape
        qkv = self.qkv(self.norm1(x)).view(b, n, 3, self.heads, width // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        old = 0 if past is None else past[0].shape[-2]
        if past is not None:
            k = torch.cat([past[0], k], dim=-2)
            v = torch.cat([past[1], v], dim=-2)
        # Query row i is at absolute position old+i. Forbid looking into its future.
        mask = torch.arange(old + n, device=x.device)[None, :] <= (
            old + torch.arange(n, device=x.device)[:, None]
        )
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(width // self.heads)
        attention = scores.masked_fill(~mask, float("-inf")).softmax(dim=-1)
        values = (attention @ v).transpose(1, 2).reshape(b, n, width)
        x = x + self.proj(values)
        return x + self.mlp(self.norm2(x)), (k, v)


class TinyTransformer(nn.Module):
    def __init__(self, vocab: int = 11, width: int = 32, heads: int = 4, layers: int = 2,
                 max_positions: int = 512):
        super().__init__()
        for name, value in dict(vocab=vocab, width=width, heads=heads, layers=layers,
                                max_positions=max_positions).items():
            if nonnegative_int(value, name) == 0:
                raise ValueError(f"{name} must be positive")
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.vocab = vocab
        self.embedding = nn.Embedding(vocab, width)
        self.position = nn.Embedding(max_positions, width)
        self.blocks = nn.ModuleList([AttentionBlock(width, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab, bias=False)

    def forward(self, ids: torch.Tensor, cache: Cache = None):
        if ids.ndim != 2 or ids.shape[1] == 0 or ids.dtype != torch.long:
            raise ValueError("Expected a nonempty [batch, tokens] int64 tensor")
        old = cache_length(cache)
        if old + ids.shape[1] > self.position.num_embeddings:
            raise ValueError("Input exceeds the model position capacity")
        positions = torch.arange(old, old + ids.shape[1], device=ids.device)
        x = self.embedding(ids) + self.position(positions)
        new_cache = []
        for i, layer in enumerate(self.blocks):
            x, kv = layer(x, None if cache is None else cache[i])
            new_cache.append(kv)
        return self.head(self.norm(x)), new_cache


def cache_length(cache: Cache) -> int:
    return 0 if cache is None else cache[0][0].shape[-2]


def crop_cache(cache: Cache, length: int) -> Cache:
    length = nonnegative_int(length, "cache length")
    if length < 0 or length > cache_length(cache):
        raise ValueError("Invalid KV crop length")
    return None if length == 0 else [(k[..., :length, :], v[..., :length, :]) for k, v in cache]


def token_tensor(ids: list[int], model: TinyTransformer) -> torch.Tensor:
    return torch.tensor([ids], dtype=torch.long, device=next(model.parameters()).device)


def probabilities(logits: torch.Tensor, temperature: float) -> np.ndarray:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Use greedy=True, not temperature=0, for greedy decoding")
    # Softmax normally subtracts its maximum, but that happens too late if the
    # temperature division has already overflowed to +inf. Shift first: at least
    # one logit remains zero, while large negative gaps may safely become -inf.
    scores = logits.double()
    if (scores.ndim == 0 or scores.shape[-1] == 0
            or torch.isnan(scores).any() or torch.isposinf(scores).any()
            or not torch.isfinite(scores).any(dim=-1).all()):
        raise ValueError("Each logits row needs a finite value and no NaN or +inf")
    if temperature >= 1:
        # Division cannot overflow here; do it first so opposite finite values
        # near float64's limit do not overflow during the maximum subtraction.
        scores = scores / temperature
        scores = scores - scores.amax(dim=-1, keepdim=True)
    else:
        scores = (scores - scores.amax(dim=-1, keepdim=True)) / temperature
    return scores.softmax(-1).cpu().numpy()


def _validate_generation(model, prompt, count, gamma, greedy, temperature, eos_id):
    count = nonnegative_int(count, "max_new_tokens")
    gamma = nonnegative_int(gamma, "gamma")
    prompt = token_ids(prompt, model.vocab)
    if not prompt:
        raise ValueError("Need a nonempty prompt")
    if eos_id is not None:
        token_ids([eos_id], model.vocab)
    if not greedy and (not math.isfinite(temperature) or temperature <= 0):
        raise ValueError("Sampling temperature must be finite and positive")
    # The final emitted token is never fed back, so only count-1 new positions
    # are evaluated. Do not fail halfway through generation at the embedding.
    if len(prompt) + max(0, count - 1) > model.position.num_embeddings:
        raise ValueError("Requested generation exceeds the model position capacity")
    return prompt, count, gamma


def _select(logits, greedy, temperature, rng):
    if greedy:
        return int(logits.argmax())
    p = probabilities(logits, temperature)
    return int(rng.choice(len(p), p=p))


def _target_generate(model, prompt, count, greedy, temperature, rng, eos_id):
    """The same target-only path for AR and gamma=0; no draft or empty prefill."""
    output, trace = prompt.copy(), []
    cache = None
    for i in range(count):
        base = len(output)
        ids = prompt if i == 0 else [output[-1]]
        logits, cache = model(token_tensor(ids, model), cache)
        token = _select(logits[0, -1], greedy, temperature, rng)
        output.append(token)
        trace.append(dict(prefix_length=base, proposed=[], accepted=0,
                          emitted=[token], rejected=False,
                          cache_length=cache_length(cache), draft_cache_length=None,
                          stop_reason="eos" if token == eos_id else
                          "length" if i + 1 == count else None))
        if token == eos_id:
            break
    return output, trace


@torch.inference_mode()
def autoregressive_generate(
    model: TinyTransformer, prompt: list[int], count: int, *, greedy: bool = True,
    temperature: float = 1.0, seed: int = 42, eos_id: int | None = None,
) -> list[int]:
    prompt, count, _ = _validate_generation(model, prompt, count, 0, greedy, temperature, eos_id)
    model.eval()
    return _target_generate(model, prompt, count, greedy, temperature,
                            np.random.default_rng(seed), eos_id)[0]


class _NeuralProposal:
    def __init__(self, model):
        self.model, self.cache = model, None
        model.eval()

    def propose(self, prefix, depth, greedy, temperature, rng, eos_id):
        if not depth:
            return [], []
        if self.cache is None and len(prefix) > 1:
            _, self.cache = self.model(token_tensor(prefix[:-1], self.model))
        assert cache_length(self.cache) == len(prefix) - 1
        logits, self.cache = self.model(token_tensor([prefix[-1]], self.model), self.cache)
        candidates, qs = [], []
        for _ in range(depth):
            if greedy:
                token = int(logits[0, -1].argmax())
                q = np.eye(1, self.model.vocab, token, dtype=np.float64)[0]
            else:
                q = probabilities(logits[0, -1], temperature)
                token = int(rng.choice(self.model.vocab, p=q))
            candidates.append(token)
            qs.append(q)
            # Keep the reference's explicit final-token cache catch-up. This
            # makes both caches follow the same round-boundary invariant.
            logits, self.cache = self.model(token_tensor([token], self.model), self.cache)
            if token == eos_id:
                break
        return candidates, qs

    def commit(self, length):
        if self.cache is not None:
            self.cache = crop_cache(self.cache, length)
        return cache_length(self.cache) if self.cache is not None else None


class _LookupProposal:
    def __init__(self, vocab, max_context, prefer_recent):
        self.vocab, self.max_context, self.prefer_recent = vocab, max_context, prefer_recent

    def propose(self, prefix, depth, greedy, temperature, rng, eos_id):
        from .ngram_speculation import prompt_lookup
        result = prompt_lookup(prefix, self.max_context, depth, self.prefer_recent)
        candidates = result.candidates
        if eos_id in candidates:
            candidates = candidates[:candidates.index(eos_id) + 1]
        qs = [np.eye(1, self.vocab, token, dtype=np.float64)[0] for token in candidates]
        return candidates, qs

    def commit(self, length):
        return None  # lookup uses only the committed token snapshot


@torch.inference_mode()
def _generate(target, proposer, prompt, count, gamma, greedy, temperature, seed, eos_id):
    target.eval()
    rng = np.random.default_rng(seed)
    if count == 0:
        return prompt.copy(), []
    if gamma == 0 or count == 1:
        return _target_generate(target, prompt, count, greedy, temperature, rng, eos_id)
    output, trace, cache_t = prompt.copy(), [], None
    if len(prompt) > 1:
        _, cache_t = target(token_tensor(prompt[:-1], target))
    while len(output) - len(prompt) < count:
        base = len(output)
        assert cache_length(cache_t) == base - 1
        depth = min(gamma, count - (base - len(prompt)) - 1)
        candidates, qs = proposer.propose(output, depth, greedy, temperature, rng, eos_id)
        logits, cache_t = target(token_tensor([output[-1], *candidates], target), cache_t)
        if greedy:
            # Preserve argmax even at very small temperatures; no CPU softmax
            # is needed to define the deterministic target policy.
            winners = logits[0].argmax(-1).cpu().numpy()
            ps = np.zeros((len(winners), target.vocab), dtype=np.float64)
            ps[np.arange(len(winners)), winners] = 1.0
        else:
            ps = probabilities(logits[0], temperature)
        step = verify_candidates(candidates, qs, ps, rng, greedy=greedy, eos_id=eos_id)
        output.extend(step.tokens)
        # Accepted EOS has no correction/bonus: crop from actual output length,
        # not from the number accepted before a separate terminal truncation.
        crop_to = len(output) - 1
        cache_t = crop_cache(cache_t, crop_to)
        # A final zero-depth round does not touch the unused draft cache.
        draft_length = proposer.commit(crop_to) if depth else None
        stopped = step.tokens[-1] == eos_id
        trace.append(dict(prefix_length=base, proposed=candidates, accepted=step.accepted,
                          emitted=step.tokens, rejected=step.rejected,
                          cache_length=cache_length(cache_t), draft_cache_length=draft_length,
                          stop_reason="eos" if stopped else
                          "length" if len(output) - len(prompt) == count else None))
        if stopped:
            break
        assert cache_length(cache_t) == len(output) - 1
    return output, trace


def speculative_generate(
    target: TinyTransformer, draft: TinyTransformer, prompt: list[int],
    max_new_tokens: int = 20, gamma: int = 4, *, greedy: bool = True,
    temperature: float = 1.0, seed: int = 42, eos_id: int | None = None,
) -> tuple[list[int], list[dict]]:
    prompt, count, gamma = _validate_generation(
        target, prompt, max_new_tokens, gamma, greedy, temperature, eos_id)
    proposer = None
    if count > 1 and gamma:
        if target.vocab != draft.vocab:
            raise ValueError("Target and draft must share a vocabulary")
        _validate_generation(draft, prompt, count, gamma, greedy, temperature, eos_id)
        proposer = _NeuralProposal(draft)
    return _generate(target, proposer, prompt, count, gamma, greedy, temperature, seed, eos_id)


def lookup_generate(
    target: TinyTransformer, prompt: list[int], max_new_tokens: int = 20,
    gamma: int = 4, *, max_context: int = 4, prefer_recent: bool = True,
    greedy: bool = True, temperature: float = 1.0, seed: int = 42,
    eos_id: int | None = None,
) -> tuple[list[int], list[dict]]:
    """Deterministic history copying with exact target verification and KV reuse."""
    prompt, count, gamma = _validate_generation(
        target, prompt, max_new_tokens, gamma, greedy, temperature, eos_id)
    if nonnegative_int(max_context, "max_context") == 0:
        raise ValueError("max_context must be positive")
    return _generate(target, _LookupProposal(target.vocab, max_context, prefer_recent),
                     prompt, count, gamma, greedy, temperature, seed, eos_id)


def summarize_trace(trace: list[dict]) -> dict:
    """Committed counts, including EOS; utilization is not conditional acceptance."""
    rounds = len(trace)
    proposed = sum(len(row["proposed"]) for row in trace)
    accepted = sum(row["accepted"] for row in trace)
    committed = sum(len(row["emitted"]) for row in trace)
    return dict(rounds=rounds, proposed=proposed, accepted=accepted, committed=committed,
                mean_committed_per_round=committed / rounds if rounds else 0.0,
                proposal_utilization=accepted / proposed if proposed else None,
                stop_reason=trace[-1]["stop_reason"] if trace else "length")


def demo() -> dict:
    torch.set_num_threads(1)
    torch.manual_seed(10)
    target = TinyTransformer().double().eval()
    torch.manual_seed(20)
    draft = TinyTransformer(layers=1).double().eval()
    prompt = [1, 2, 3]
    baseline = autoregressive_generate(target, prompt, 24)
    result, trace = speculative_generate(target, draft, prompt, 24, gamma=4)
    assert result == baseline
    return {
        "device": "cpu", "weights": "random; no downloaded checkpoint",
        "greedy_equal": result == baseline, "new_tokens": 24,
        "verification_rounds": len(trace), "trace": trace,
        "stats": summarize_trace(trace),
        "note": "Correctness demo only; round reduction does not imply wall-clock speedup.",
    }


if __name__ == "__main__":
    import json
    print(json.dumps(demo(), indent=2))
