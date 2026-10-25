"""Two different n-gram drafters: deterministic copying and a probability model.

Run offline: python -m examples.ngram_speculation
Tokens are integers; the demo displays whitespace-separated words for clarity.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from typing import Sequence
import numpy as np
from examples.speculative_sampling import (
    MarkovLM, generate, nonnegative_int, token_ids, verify_candidates,
)


@dataclass
class LookupResult:
    candidates: list[int]
    matched_length: int
    source_start: int | None
    continuation_start: int | None


def prompt_lookup(history: Sequence[int], max_context: int = 4,
                  max_draft: int = 5, prefer_recent: bool = True) -> LookupResult:
    """Find the longest repeated suffix with an available continuation.

    Searches a fixed snapshot of committed history. We exclude the suffix's
    own position. Overlapping occurrences are allowed, but never copy beyond
    the snapshot. This is a transparent reference, not an efficient index.
    """
    max_context = nonnegative_int(max_context, "max_context")
    max_draft = nonnegative_int(max_draft, "max_draft")
    if max_context < 1:
        raise ValueError("max_context >= 1 and max_draft >= 0 required")
    tokens = token_ids(history)
    if max_draft == 0:
        return LookupResult([], 0, None, None)
    for n in range(min(max_context, len(tokens) - 1), 0, -1):
        suffix = tokens[-n:]
        positions = range(len(tokens) - n)
        if prefer_recent:
            positions = reversed(positions)
        for start in positions:
            if tokens[start:start + n] == suffix:
                continuation = start + n
                draft = tokens[continuation:continuation + max_draft]
                if draft:
                    return LookupResult(draft, n, start, continuation)
    return LookupResult([], 0, None, None)


def verify_copied_tokens(candidates, target_probs, vocab_size, rng, *, greedy=False, eos_id=None):
    """A fixed copied sequence has conditional proposal q_i = delta(candidate_i)."""
    if nonnegative_int(vocab_size, "vocab_size") == 0:
        raise ValueError("vocab_size must be positive")
    candidates = token_ids(candidates, vocab_size)
    qs = []
    for candidate in candidates:
        q = np.zeros(vocab_size, dtype=np.float64)
        q[candidate] = 1.0
        qs.append(q)
    return verify_candidates(candidates, qs, target_probs, rng, greedy=greedy, eos_id=eos_id)


class CountNGramLM:
    """A true stochastic n-gram drafter with longest-observed-context backoff.

    max_context=2 means up to two preceding tokens, i.e. a trigram LM.
    Sequences are indexed separately so no event crosses document boundaries.
    add_k applies explicit smoothing after context selection. It is part of q.
    """
    def __init__(self, vocab_size: int, max_context: int = 2, add_k: float = 0.0):
        vocab_size = nonnegative_int(vocab_size, "vocab_size")
        max_context = nonnegative_int(max_context, "max_context")
        if vocab_size < 1 or not np.isfinite(add_k) or add_k < 0:
            raise ValueError("Invalid vocabulary size, context length or smoothing")
        self.vocab_size = vocab_size
        self.max_context = max_context
        self.add_k = add_k
        self.counts = defaultdict(Counter)

    def fit(self, sequences: Sequence[Sequence[int]]):
        new_counts = defaultdict(Counter)
        for sequence in sequences:
            ids = token_ids(sequence, self.vocab_size)
            for i, token in enumerate(ids):
                for n in range(min(self.max_context, i) + 1):
                    context = tuple(ids[i-n:i]) if n else ()
                    new_counts[context][token] += 1
        self.counts = new_counts
        return self

    def distribution_with_context(self, prefix):
        ids = token_ids(prefix, self.vocab_size)
        for n in range(min(len(ids), self.max_context), -1, -1):
            context = tuple(ids[-n:]) if n else ()
            counts = self.counts.get(context)
            if counts:
                q = np.full(self.vocab_size, self.add_k, dtype=np.float64)
                for token, count in counts.items():
                    q[token] += count
                return q / q.sum(), context
        return np.full(self.vocab_size, 1.0 / self.vocab_size), ()

    def next_probs(self, prefix):
        return self.distribution_with_context(prefix)[0]

    def verify(self, prefix, candidates):
        return [self.next_probs([*prefix, *candidates[:i]]) for i in range(len(candidates) + 1)]


def demo():
    words = "the cat sat on the mat and the cat".split()
    vocabulary = list(dict.fromkeys(words))
    mapping = {word: i for i, word in enumerate(vocabulary)}
    history = [mapping[word] for word in words]
    lookup = prompt_lookup(history, max_context=3, max_draft=4)
    p = np.array([.5, .3, .2])
    q_count = np.array([.2, .5, .3])
    rng = np.random.default_rng(2026)
    n = 100_000
    counts = np.zeros(3, dtype=int)
    accepted = 0
    # Always copy B; its true proposal distribution is [0,1,0], not q_count.
    for _ in range(n):
        step = verify_copied_tokens([1], [p, p], 3, rng)
        counts[step.tokens[0]] += 1
        accepted += step.accepted
    probabilistic = CountNGramLM(3, max_context=1, add_k=.1).fit([
        [0, 0, 1, 0, 2, 0, 1], [1, 1, 0, 2, 2, 1], [2, 2, 0, 0, 1]
    ])
    target = MarkovLM(np.array([[.5,.3,.2],[.2,.6,.2],[.3,.2,.5]]))
    sequence, stats = generate(target, probabilistic, [0], 24, gamma=4, seed=2026)
    return {
        "lookup": {"history": words, "matched_length": lookup.matched_length,
                   "source_start_zero_based": lookup.source_start,
                   "copied_tokens": [vocabulary[token] for token in lookup.candidates]},
        "deterministic_copy_experiment": {
            "seed": 2026, "samples": n, "copied_token": "B", "target": p.tolist(),
            "true_q": [0,1,0], "empirical_output": (counts/n).tolist(),
            "empirical_acceptance": accepted/n, "theoretical_acceptance": float(p[1]),
            "count_model_overlap_if_sampled_from_it": float(np.minimum(p,q_count).sum()),
        },
        "stochastic_ngram": {"sequence": sequence, "stats": stats,
                             "actual_q_after_0": probabilistic.next_probs([0]).tolist()},
        "note": "Correctness and mechanism demo; no LLM/GPU latency measurement.",
    }


if __name__ == "__main__":
    from pathlib import Path
    result = demo()
    print(json.dumps(result, indent=2))
    (Path(__file__).resolve().parents[1]/"data"/"ngram_results.json").write_text(json.dumps(result, indent=2))
