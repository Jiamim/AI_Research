"""Inference-serving references: speculative sampling, a paged KV allocator, prefix sharing.

* speculative_step: draft k tokens from q, verify with the target p, accept with
  min(1, p/q), resample a rejection from (p - q)+, and add a bonus token when all
  drafts pass. The output distribution equals sampling from p directly.
* PagedKVCache: fixed-size physical blocks + per-request block tables; requests
  that share a prompt prefix share blocks by reference count (copy-on-write
  is unnecessary here because cached blocks are never modified after filling).
"""
import json
import torch


def speculative_step(target_probs, draft_probs, draft_tokens, generator=None):
    """One verify round.

    target_probs: [k+1, V], target distributions along the SAME proposed prefix.
    draft_probs:  [k, V],   actual draft sampling distributions along that prefix.
    Row i is conditioned on the input prefix and draft_tokens[:i]. After the
    first rejection, later target rows are stale and must not be reused. Returns
    accepted drafts plus a correction/bonus token (1..k+1 tokens in total).
    The caller handles EOS, output budgets and rollback of rejected KV entries.
    """
    if target_probs.ndim != 2 or draft_probs.ndim != 2 or draft_tokens.ndim != 1 \
            or target_probs.shape != (len(draft_tokens) + 1, draft_probs.shape[-1]) \
            or len(draft_probs) != len(draft_tokens) or not target_probs.shape[1]:
        raise ValueError("target [k+1,V], draft [k,V] and draft token IDs [k] required")
    if draft_tokens.dtype != torch.long or draft_tokens.device != target_probs.device or draft_probs.device != target_probs.device:
        raise ValueError("int64 draft tokens and all inputs on one device required")
    dtype = torch.float64 if target_probs.dtype == torch.float64 or draft_probs.dtype == torch.float64 else torch.float32
    distributions = []
    for probabilities in (target_probs, draft_probs):
        if not probabilities.is_floating_point() or not torch.isfinite(probabilities).all() or torch.any(probabilities < 0):
            raise ValueError("finite nonnegative probability matrices required")
        work = probabilities.to(dtype)
        # For nonnegative normalized probabilities, normal-value rounding errors scale
        # with their total mass (about 1), not vocabulary size. Only subnormal absolute
        # errors add a V-dependent term. Sum in FP64 to keep reduction error separate.
        sums = probabilities.sum(-1, keepdim=True, dtype=torch.float64)
        precision = torch.finfo(probabilities.dtype)
        tolerance = max(1e-6, 0.5 * precision.eps + 0.5 * probabilities.shape[-1] * precision.tiny * precision.eps)
        if not torch.isfinite(sums).all() or torch.any(sums <= 0) or torch.any((sums - 1).abs() > tolerance):
            raise ValueError("each probability row must sum to one")
        distributions.append(work / sums.to(dtype))
    target_probs, draft_probs = distributions
    if torch.any((draft_tokens < 0) | (draft_tokens >= target_probs.shape[1])):
        raise ValueError("draft token outside vocabulary")
    if len(draft_tokens) and torch.any(draft_probs.gather(1, draft_tokens[:, None]) <= 0):
        raise ValueError("draft tokens must have positive probability under the actual draft sampler")
    out = []
    for i, token in enumerate(draft_tokens.tolist()):
        p, q = target_probs[i, token], draft_probs[i, token]
        if torch.rand((), device=target_probs.device, generator=generator) < torch.clamp(p / q, max=1.0):
            out.append(token)
            continue
        residual = (target_probs[i] - draft_probs[i]).clamp_min(0)
        # If p == q up to rounding, a rejection can leave an all-zero residual; sampling the
        # target itself is then the correct (and only well-defined) choice.
        residual = residual if residual.sum() > 0 else target_probs[i]
        out.append(int(torch.multinomial(residual / residual.sum(), 1, generator=generator)))
        return out
    # Every draft accepted: the target's next distribution is already computed.
    out.append(int(torch.multinomial(target_probs[len(draft_tokens)], 1, generator=generator)))
    return out


def speculative_distribution_check(vocab=5, k=3, trials=20_000, seed=0):
    """Empirical first-token distribution of speculative sampling vs the target."""
    g = torch.Generator().manual_seed(seed)
    target = torch.softmax(torch.randn(k + 1, vocab, generator=g) * 1.5, -1)
    draft = torch.softmax(torch.randn(k, vocab, generator=g) * 1.5, -1)
    counts = torch.zeros(vocab)
    accepted = 0
    for _ in range(trials):
        tokens = torch.stack([torch.multinomial(draft[i], 1, generator=g)[0] for i in range(k)])
        result = speculative_step(target, draft, tokens, g)
        counts[result[0]] += 1
        accepted += len(result)
    # Acceptance rate of the first position; with a shared rate alpha over k drafts the expected
    # number of tokens per round is (1 - alpha^(k+1)) / (1 - alpha) (chapter 12.6).
    alpha = torch.minimum(target[0], draft[0]).sum().item()
    return dict(max_abs_error_first_token=float((counts / trials - target[0]).abs().max()),
                mean_tokens_per_round=accepted / trials, first_position_acceptance=alpha)


class PagedKVCache:
    """Logical token positions -> (physical block, offset) via per-request block tables."""

    def __init__(self, num_blocks, block_size):
        if not isinstance(num_blocks, int) or not isinstance(block_size, int) or min(num_blocks, block_size) < 1:
            raise ValueError("positive integer block count and block size required")
        self.block_size = block_size
        self.free = list(range(num_blocks - 1, -1, -1))
        self.refcount = [0] * num_blocks
        self.tables, self.lengths = {}, {}
        self.prefix_index = {}

    def _allocate(self):
        if not self.free:
            raise MemoryError("out of KV blocks: preempt or reject a request")
        block = self.free.pop()
        self.refcount[block] = 1
        return block

    def add_request(self, request, prompt_tokens):
        """Reuse cached full blocks for an identical prompt prefix; allocate the rest."""
        if request in self.tables:
            raise ValueError("request already exists")
        # Plan allocations before changing refcounts: OOM must leave live requests intact.
        full = len(prompt_tokens) // self.block_size
        keys = [tuple(prompt_tokens[:(i + 1) * self.block_size]) for i in range(full)]
        needed = sum(key not in self.prefix_index for key in keys) + bool(len(prompt_tokens) % self.block_size)
        if needed > len(self.free):
            raise MemoryError("out of KV blocks: request admission did not modify the cache")
        table, length = [], 0
        full_blocks = len(prompt_tokens) // self.block_size
        for i in range(full_blocks):
            key = tuple(prompt_tokens[:(i + 1) * self.block_size])
            if key in self.prefix_index:
                block = self.prefix_index[key]
                self.refcount[block] += 1
            else:
                block = self._allocate()
                self.prefix_index[key] = block
            table.append(block)
            length += self.block_size
        self.tables[request], self.lengths[request] = table, length
        for _ in range(len(prompt_tokens) - length):
            self.append(request)

    def append(self, request):
        """Reserve the slot for one new token; returns (block, offset)."""
        table, length = self.tables[request], self.lengths[request]
        if length % self.block_size == 0:
            table.append(self._allocate())
        self.lengths[request] = length + 1
        return table[-1], length % self.block_size

    def release(self, request):
        for block in self.tables.pop(request):
            self.refcount[block] -= 1
            if self.refcount[block] == 0:
                self.free.append(block)
                self.prefix_index = {k: b for k, b in self.prefix_index.items() if b != block}
        del self.lengths[request]

    def used_blocks(self):
        return sum(count > 0 for count in self.refcount)


def paged_cache_demo():
    cache = PagedKVCache(num_blocks=64, block_size=16)
    system_prompt = list(range(48))  # three full blocks shared by every request
    for r in range(4):
        cache.add_request(r, system_prompt + [1000 + r] * 5)
        for _ in range(20):
            cache.append(r)
    shared = cache.used_blocks()
    unshared = sum(-(-cache.lengths[r] // 16) for r in range(4))
    cache.release(0)
    return dict(blocks_with_prefix_sharing=shared, blocks_without_sharing=unshared,
                blocks_after_releasing_one=cache.used_blocks(),
                contiguous_max_length_reservation=4 * (-(-2048 // 16)))


if __name__ == "__main__":
    print(json.dumps(dict(speculative=speculative_distribution_check(), paged_cache=paged_cache_demo()), indent=2))
