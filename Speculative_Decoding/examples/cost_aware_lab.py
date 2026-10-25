"""Offline cost/uncertainty lab. All input timings are synthetic, never GPU data.

Run: python -m examples.cost_aware_lab
"""
from pathlib import Path
import json
import numpy as np


def progress_curve(conditional_acceptance):
    """E[L_gamma] for fixed budgets, no EOS, and alpha_i=P(pass i | pass <i)."""
    alpha = np.asarray(conditional_acceptance, dtype=float)
    if alpha.ndim != 1 or not np.all(np.isfinite(alpha)) or np.any((alpha < 0) | (alpha > 1)):
        raise ValueError('Expected a finite one-dimensional probability vector')
    survival = np.r_[1., np.cumprod(alpha)]
    return survival, np.cumsum(survival)


def compare_budgets(conditional_acceptance, round_cost_ms):
    """Compare full configurations, including disabling speculation at gamma=0.

    The caller supplies measured or modeled critical-path costs. Comparing the
    same survival curve across budgets assumes budget changes do not change q.
    """
    survival, progress = progress_curve(conditional_acceptance)
    cost = np.asarray(round_cost_ms, dtype=float)
    if cost.shape != progress.shape or not np.all(np.isfinite(cost)) or np.any(cost <= 0):
        raise ValueError('Need one positive finite cost per budget, including gamma=0')
    rates = progress / cost
    best = int(np.argmax(rates))  # Tie -> smaller speculative budget.
    return {'survival': survival.tolist(), 'expected_tokens': progress.tolist(),
            'round_cost_ms': cost.tolist(), 'tokens_per_ms': rates.tolist(),
            'speedup': (rates / rates[0]).tolist(), 'best_gamma': best}


def paired_speed_interval(baseline_ms, speculative_ms, *, seed=20261025, draws=6000):
    """Percentile paired bootstrap of SUM(baseline time)/SUM(speculative time).

    Each pair must represent equal work with the same timing boundary. This is
    an interval for the request-population ratio estimated by corpus sums,
    not an average per-request speedup or uncertainty in already known sums.
    Pairs must be independent sampling units; cluster correlated prompts first.
    """
    base, spec = np.asarray(baseline_ms, dtype=float), np.asarray(speculative_ms, dtype=float)
    if base.ndim != 1 or len(base) < 2 or spec.shape != base.shape:
        raise ValueError('Need at least two paired observations')
    if not np.all(np.isfinite(base)) or not np.all(np.isfinite(spec)) or np.any(base <= 0) or np.any(spec <= 0):
        raise ValueError('Times must be positive and finite')
    if type(draws) is not int or draws < 100:
        raise ValueError('Use at least 100 bootstrap draws')
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(base), size=(draws, len(base)))
    ratios = base[indices].sum(axis=1) / spec[indices].sum(axis=1)
    low, high = np.quantile(ratios, [.025, .975])
    return {'aggregate_speedup': float(base.sum()/spec.sum()),
            'percentile_95_interval': [float(low), float(high)],
            'mean_request_speedup': float(np.mean(base/spec)),
            'pairs': len(base), 'bootstrap_draws': draws, 'seed': seed}


def break_even_tokens(training_cost, baseline_cost_per_token, speculative_cost_per_token):
    """Costs use one additive unit (e.g. GPU-seconds or money), not mixed units."""
    values = [training_cost, baseline_cost_per_token, speculative_cost_per_token]
    if not all(np.isfinite(x) for x in values) or training_cost < 0 or min(values[1:]) <= 0:
        raise ValueError('Need finite nonnegative training and positive per-token costs')
    savings = baseline_cost_per_token - speculative_cost_per_token
    return None if savings <= 0 else float(training_cost/savings)


def experiment():
    alpha = [.93, .90, .86, .83, .76, .65, .55, .40]
    # A new execution shape costs more even when it contains just one candidate.
    shape_costs = [1., 2., 2., 2.05, 3.4, 3.45, 3.5, 3.55, 5.]
    shape = compare_budgets(alpha, shape_costs)
    rng = np.random.default_rng(20261025)
    baseline = rng.lognormal(mean=4., sigma=.8, size=80)
    # Explicitly synthetic paired task heterogeneity.
    speculative = baseline*(.66 + .22*np.clip(baseline/500., 0., 1.)) + 5.
    interval = paired_speed_interval(baseline, speculative)
    return {'scope': 'Analytical/synthetic data only; no model or GPU benchmark.',
            'assumptions': ['fixed proposal conditionals across gamma', 'no EOS or length cap',
                            'full round critical-path costs', 'independent synthetic request pairs'],
            'shape_cliffs': {'conditional_acceptance': alpha, **shape},
            'paired_timing': {'baseline_ms': baseline.tolist(),
                             'speculative_ms': speculative.tolist(), **interval},
            'amortization': {'training_gpu_seconds': 360000,
                             'baseline_gpu_seconds_per_token': .01,
                             'speculative_gpu_seconds_per_token': .007,
                             'break_even_tokens': break_even_tokens(360000, .01, .007)}}


if __name__ == '__main__':
    result = experiment()
    destination = Path(__file__).resolve().parents[1]/'data/cost_aware_results.json'
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'best_gamma': result['shape_cliffs']['best_gamma'],
                      'paired_interval': result['paired_timing']['percentile_95_interval'],
                      'break_even_tokens': result['amortization']['break_even_tokens'],
                      'scope': result['scope']}, ensure_ascii=False, indent=2))
