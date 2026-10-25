"""Exact single-chain speculative sampling, with a tiny Markov language model.

Notation: p = target, q = actual proposal distribution. No GPU or downloads.
Run: python -m examples.speculative_sampling
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence
import json
from numbers import Integral
import numpy as np


def nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def token_ids(values: Sequence[int], vocab_size: int | None = None) -> list[int]:
    ids = [nonnegative_int(token, "token ID") for token in values]
    if vocab_size is not None and any(token >= vocab_size for token in ids):
        raise ValueError("Token outside vocabulary")
    return ids


def probability_vector(values: Sequence[float]) -> np.ndarray:
    p = np.asarray(values, dtype=np.float64)
    if p.ndim != 1 or p.size == 0 or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("Expected a finite, nonnegative probability vector")
    if not np.isclose(p.sum(), 1.0, rtol=1e-7, atol=1e-12):
        raise ValueError("Probabilities must sum to one")
    return p / p.sum()


def residual_distribution(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Only call after rejection; if p == q, rejection has probability zero."""
    p, q = probability_vector(p), probability_vector(q)
    if p.shape != q.shape:
        raise ValueError("Target and proposal must share a vocabulary")
    residual = np.maximum(p - q, 0.0)
    mass = residual.sum()
    if mass <= 0:
        raise ValueError("Zero residual mass: inspect the rejection logic/numerics")
    return residual / mass


@dataclass
class Step:
    tokens: list[int]       # accepted prefix + repair/bonus, unless accepted EOS ends it
    accepted: int          # excludes the repair/bonus token
    proposed: int
    rejected: bool


def verify_candidates(
    candidates: Sequence[int],
    proposal_probs: Sequence[np.ndarray],
    target_probs: Sequence[np.ndarray],
    rng: np.random.Generator,
    *,
    greedy: bool = False,
    eos_id: int | None = None,
) -> Step:
    """p[i] is conditioned on prefix + candidates[:i], never candidates[:i+1]."""
    if len(proposal_probs) != len(candidates) or len(target_probs) != len(candidates) + 1:
        raise ValueError("Need gamma proposal distributions and gamma+1 target distributions")
    # Validate every row before sampling, including discarded suffixes and bonus.
    # This prevents malformed model output from succeeding by chance.
    ps = [probability_vector(row) for row in target_probs]
    qs = [probability_vector(row) for row in proposal_probs]
    if any(row.shape != ps[0].shape for row in [*ps, *qs]):
        raise ValueError("This reference implementation requires a shared vocabulary")
    candidates = token_ids(candidates, len(ps[0]))
    if eos_id is not None:
        token_ids([eos_id], len(ps[0]))
    if not greedy and any(q[token] <= 0 for q, token in zip(qs, candidates)):
        raise ValueError("A sampled proposal cannot have zero proposal probability")
    accepted: list[int] = []
    for i, token in enumerate(candidates):
        p, q = ps[i], qs[i]
        if greedy:
            if token != int(p.argmax()):
                return Step(accepted + [int(p.argmax())], i, len(candidates), True)
        else:
            # Only divide when p < q, so the ratio is bounded by one even for
            # tiny q. Multiplying u*q can round a subnormal up and reject a
            # valid proposal (even p=q); the bounded ratio avoids that failure.
            u = rng.random()
            if p[token] < q[token] and u >= p[token] / q[token]:
                repair = int(rng.choice(len(p), p=residual_distribution(p, q)))
                return Step(accepted + [repair], i, len(candidates), True)
        accepted.append(int(token))
        if token == eos_id:
            return Step(accepted, len(accepted), len(candidates), False)
    p_bonus = ps[-1]
    bonus = int(p_bonus.argmax()) if greedy else int(rng.choice(len(p_bonus), p=p_bonus))
    return Step(accepted + [bonus], len(candidates), len(candidates), False)


class LanguageModel(Protocol):
    def next_probs(self, prefix: Sequence[int]) -> np.ndarray: ...
    def verify(self, prefix: Sequence[int], draft: Sequence[int]) -> list[np.ndarray]: ...


@dataclass
class MarkovLM:
    """Each row gives P(next_token | last_token); BOS is supplied in the prompt."""
    transitions: np.ndarray

    def __post_init__(self) -> None:
        transitions = np.asarray(self.transitions)
        if transitions.ndim != 2 or transitions.shape[0] == 0 or transitions.shape[0] != transitions.shape[1]:
            raise ValueError("Transition matrix must be nonempty and square")
        self.transitions = np.stack([probability_vector(row) for row in self.transitions])
        if self.transitions.shape[0] != self.transitions.shape[1]:
            raise ValueError("Transition matrix must be square")

    def next_probs(self, prefix: Sequence[int]) -> np.ndarray:
        if len(prefix) == 0:
            raise ValueError("Provide at least one prompt/BOS token")
        return self.transitions[token_ids(prefix, len(self.transitions))[-1]]

    def verify(self, prefix: Sequence[int], draft: Sequence[int]) -> list[np.ndarray]:
        # All conditioning tokens are known, so this is one vectorized lookup.
        # A Transformer implements this interface with one causal forward pass.
        if len(prefix) == 0:
            raise ValueError("Provide at least one prompt/BOS token")
        ids = token_ids([*prefix, *draft], len(self.transitions))
        return list(self.transitions[ids[len(prefix) - 1:]])


def speculative_step(
    target: LanguageModel, draft: LanguageModel, prefix: Sequence[int],
    gamma: int, rng: np.random.Generator, *, greedy: bool = False,
    eos_id: int | None = None,
) -> Step:
    gamma = nonnegative_int(gamma, "gamma")
    prefix = token_ids(prefix)
    if not prefix:
        raise ValueError("Need a nonempty prefix")
    candidates, qs = [], []
    for _ in range(gamma):
        q = probability_vector(draft.next_probs([*prefix, *candidates]))
        token = int(q.argmax()) if greedy else int(rng.choice(len(q), p=q))
        candidates.append(token)
        qs.append(q)
        if token == eos_id:
            break
    ps = target.verify(prefix, candidates)
    return verify_candidates(candidates, qs, ps, rng, greedy=greedy, eos_id=eos_id)


def generate(
    target: LanguageModel, draft: LanguageModel, prompt: Sequence[int],
    max_new_tokens: int, gamma: int = 4, seed: int = 0,
    *, greedy: bool = False, eos_id: int | None = None,
) -> tuple[list[int], dict]:
    max_new_tokens = nonnegative_int(max_new_tokens, "max_new_tokens")
    gamma = nonnegative_int(gamma, "gamma")
    prompt = token_ids(prompt)
    if not prompt:
        raise ValueError("Need a nonempty prompt")
    if eos_id is not None:
        token_ids([eos_id])
    rng = np.random.default_rng(seed)
    prefix = list(prompt)
    rounds = proposed = accepted = 0
    while len(prefix) - len(prompt) < max_new_tokens:
        remaining = max_new_tokens - (len(prefix) - len(prompt))
        step = speculative_step(target, draft, prefix, min(gamma, remaining - 1), rng,
                                greedy=greedy, eos_id=eos_id)
        rounds += 1
        proposed += step.proposed
        for i, token in enumerate(step.tokens):
            prefix.append(token)
            accepted += int(i < step.accepted)
            if token == eos_id:
                return prefix, dict(rounds=rounds, proposed=proposed, accepted=accepted)
    return prefix, dict(rounds=rounds, proposed=proposed, accepted=accepted)


def expected_tokens(alpha: float | np.ndarray, gamma: int) -> float | np.ndarray:
    """Assumes a constant conditional acceptance probability, ignoring EOS."""
    gamma = nonnegative_int(gamma, "gamma")
    alpha = np.asarray(alpha, dtype=np.float64)
    if not np.isfinite(alpha).all() or np.any(alpha < 0) or np.any(alpha > 1):
        raise ValueError("alpha in [0,1], gamma >= 0 required")
    return sum(np.asarray(alpha) ** i for i in range(gamma + 1))


def one_step_experiment(n: int = 200_000, seed: int = 2026) -> dict:
    """Vectorized Monte Carlo of the one-token acceptance/repair experiment."""
    if nonnegative_int(n, "samples") == 0:
        raise ValueError("samples must be positive")
    p = np.array([0.5, 0.3, 0.2])
    q = np.array([0.2, 0.5, 0.3])
    rng = np.random.default_rng(seed)
    proposed = rng.choice(3, size=n, p=q)
    keep = rng.random(n) * q[proposed] < np.minimum(p[proposed], q[proposed])
    repaired = proposed.copy()
    repaired[~keep] = rng.choice(3, size=(~keep).sum(), p=residual_distribution(p, q))
    # Deliberately wrong control: resampling from p after rejecting introduces bias.
    wrong = proposed.copy()
    wrong[~keep] = rng.choice(3, size=(~keep).sum(), p=p)
    return {
        "seed": seed, "samples": n, "target": p.tolist(), "draft": q.tolist(),
        "exact_empirical": (np.bincount(repaired, minlength=3) / n).tolist(),
        "wrong_empirical": (np.bincount(wrong, minlength=3) / n).tolist(),
        "acceptance_empirical": float(keep.mean()), "acceptance_theory": float(np.minimum(p, q).sum()),
        "wrong_theory": (np.minimum(p, q) + (1 - np.minimum(p, q).sum()) * p).tolist(),
    }


if __name__ == "__main__":
    print(json.dumps(one_step_experiment(), indent=2))
    target = MarkovLM(np.array([[.6, .3, .1], [.2, .5, .3], [.3, .1, .6]]))
    draft = MarkovLM(np.array([[.4, .4, .2], [.3, .4, .3], [.2, .2, .6]]))
    sequence, stats = generate(target, draft, [0], max_new_tokens=24, seed=7)
    print("Generated token IDs:", sequence)
    print("Algorithm counters, NOT a wall-clock benchmark:", stats)
