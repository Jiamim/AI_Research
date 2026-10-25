"""Exact small-vocabulary checks for the model/training deep-dive.

Run ``python -m examples.training_objectives_lab``. No checkpoint or GPU is used.
The finite-horizon example is an original deterministic-target construction,
not an implementation of the full EDR paper/evaluator.
"""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def overlap(p, q):
    return np.minimum(p, q).sum(axis=-1)


def forward_kl(p, q):
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    positive = p > 0
    return float(np.sum(p[positive] * np.log(p[positive] / q[positive])))


def two_token_survival(p_first, p_next, q_first, q_next):
    """Exact accepted-prefix masses, without averaging conditional rates first.

    p_next[u, v] and q_next[u, v] condition on the *same actual* first token u.
    Both samplers use classic maximal-coupling verification.
    """
    first_mass = np.minimum(p_first, q_first)
    return np.array([first_mass.sum(), np.sum(first_mass * overlap(p_next, q_next))])


def weighted_ce_gradients(smoothing=0.0):
    """Compare detached weighted CE with the unsmoothed/smoothed proxy gradients."""
    if not 0 <= smoothing <= 1:
        raise ValueError('smoothing must lie in [0, 1]')
    z = torch.tensor([[1., -.3], [-.2, .4], [.1, .2]],
                     dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([0, 1, 0])
    log_q_star = z.log_softmax(-1).gather(-1, labels[:, None]).squeeze(-1)
    q_star = log_q_star.exp()
    proxy = q_star.cumprod(0).sum()
    smoothed_proxy = ((1 - smoothing) * q_star + smoothing).cumprod(0).sum()
    weights = ((1 - smoothing) * q_star + smoothing).cumprod(0)
    weights = weights.flip(0).cumsum(0).flip(0).detach()
    loss = -(weights * log_q_star).sum()
    ce_grad = torch.autograd.grad(loss, z, retain_graph=True)[0]
    proxy_grad = torch.autograd.grad(-proxy, z, retain_graph=True)[0]
    smoothed_grad = torch.autograd.grad(-smoothed_proxy, z)[0]
    return dict(weights=weights.tolist(),
                unsmoothed_gradient_max_error=float((ce_grad - proxy_grad).abs().max()),
                smoothed_gradient_max_error=float((ce_grad - smoothed_grad).abs().max()))


def finite_horizon_rounds(acceptance, length):
    """Expected rounds for a deterministic target, with final position terminal.

    acceptance[n, d] = proposal probability of the deterministic target token
    at offset d in the round starting after n committed tokens. At first
    rejection one correction is emitted; all accepted gives one bonus unless
    the terminal token/length bound was reached. Rows n can differ: a
    parallel draft's input depends on its round start.
    """
    a = np.asarray(acceptance, dtype=float)
    if not isinstance(length, int) or length < 0:
        raise ValueError('length must be a nonnegative integer')
    if a.ndim != 2 or len(a) < length or not np.isfinite(a).all():
        raise ValueError('acceptance must be a finite [length, gamma] matrix')
    if np.any((a < 0) | (a > 1)):
        raise ValueError('acceptance must lie in [0, 1]')
    gamma = a.shape[1]

    @lru_cache(None)
    def value(n):
        if n >= length:
            return 0.
        continuation = 0.
        survival = 1.
        width = min(gamma, length - n)
        for d in range(width):
            continuation += survival * (1 - a[n, d]) * value(n + d + 1)
            survival *= a[n, d]
        continuation += survival * value(min(n + width + 1, length))
        return 1 + continuation

    return value(0)


def run_checks():
    p = np.array([.6, .25, .15])
    q_a, q_b = np.array([.33, .36, .31]), np.array([.52, .46, .02])
    first = np.array([.5, .5])
    target_next = np.array([[.9, .1], [.1, .9]])
    independent_next = np.full((2, 2), .5)
    ind = two_token_survival(first, target_next, first, independent_next)
    markov = two_token_survival(first, target_next, first, target_next)
    # First-round A accepts one token; B immediately rejects. B reaches a
    # better future round start. The final (fourth) target token is terminal.
    a = np.zeros((4, 2)); a[0] = [1., 0.]; a[1] = [1., 1.]
    b = a.copy(); b[0] = [0., 1.]
    rows = []
    for vocab in [10, 100, 1000, 10000]:
        z = torch.zeros(vocab, dtype=torch.float64, requires_grad=True)
        p_onehot = torch.zeros(vocab, dtype=torch.float64); p_onehot[0] = 1
        q = z.softmax(-1)
        ce = -z.log_softmax(-1)[0]
        tv = .5 * (p_onehot - q).abs().sum()
        ce_norm = torch.autograd.grad(ce, z, retain_graph=True)[0].norm()
        tv_norm = torch.autograd.grad(tv, z)[0].norm()
        rows.append(dict(vocab=vocab, ce_norm=float(ce_norm), tv_norm=float(tv_norm)))
    return dict(
        scope='Exact toy probability/gradient calculations; no GPU benchmark or trained drafter.',
        kl_ranking=dict(target=p.tolist(), q_a=q_a.tolist(), q_b=q_b.tolist(),
                        kl_a=forward_kl(p, q_a), kl_b=forward_kl(p, q_b),
                        acceptance_a=float(overlap(p, q_a)), acceptance_b=float(overlap(p, q_b))),
        dependence=dict(independent_survival=ind.tolist(), markov_survival=markov.tolist(),
                        independent_emitted=float(1 + ind.sum()), markov_emitted=float(1 + markov.sum()),
                        extra_round_cost_break_even=float((1 + markov.sum()) / (1 + ind.sum()) - 1)),
        gradients=rows,
        weight_identity=weighted_ce_gradients(0.),
        smoothed_weights=weighted_ce_gradients(.1),
        local_vs_global=dict(first_round_emitted_a=2, first_round_emitted_b=1,
                             total_rounds_a=finite_horizon_rounds(a, 4),
                             total_rounds_b=finite_horizon_rounds(b, 4)))


if __name__ == '__main__':
    result = run_checks()
    path = ROOT / 'data/training_objectives_results.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
