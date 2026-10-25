"""Plot saved synthetic cost-lab results; never rerun experiments."""
import json
from pathlib import Path
import numpy as np
from .plotting import plt, save, COLORS


def make():
    data = json.loads((Path(__file__).resolve().parents[1]/'data/cost_aware_results.json').read_text())
    shape = data['shape_cliffs']
    gamma = np.arange(len(shape['speedup']))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), layout='constrained')
    axes[0].plot(gamma, shape['expected_tokens'], 'o-', label='Expected committed tokens', color=COLORS['accept'])
    axes[0].plot(gamma, shape['round_cost_ms'], 's-', label='Round cost (ms)', color=COLORS['reject'])
    axes[0].set(xlabel='Draft budget', title='Compare whole execution shapes')
    axes[0].legend(fontsize=9)
    axes[1].plot(gamma, shape['speedup'], 'o-', color=COLORS['target'])
    axes[1].axhline(1, color='gray', linestyle='--')
    best = shape['best_gamma']
    axes[1].scatter([best], [shape['speedup'][best]], color=COLORS['accept'], s=100, zorder=3)
    axes[1].set(xlabel='Draft budget', ylabel='Modeled speedup', title=f'Best budget = {best}; budget 1 loses')
    timings = data['paired_timing']
    base, spec = np.array(timings['baseline_ms']), np.array(timings['speculative_ms'])
    axes[2].scatter(base, spec, s=18, alpha=.7, color=COLORS['draft'])
    extent = [0, max(base.max(), spec.max())*1.05]
    axes[2].plot(extent, extent, '--', color='gray', label='Equal time')
    axes[2].set(xlabel='Baseline request time (ms)', ylabel='Speculative request time (ms)',
                title='Paired synthetic request times')
    axes[2].legend(fontsize=9)
    fig.supxlabel('Analytical and synthetic examples only. No GPU timings or performance claims.', fontsize=10, color=COLORS['muted'])
    save(fig, '27_cost_budget_uncertainty')


if __name__ == '__main__':
    make()
