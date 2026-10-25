"""Build figures for the exactness tutorial from saved finite-experiment data.

Run python -m examples.exactness_lab first, then this module. No experiments are
run by the figure builder, and no experimental JSON is modified here.
"""
from fractions import Fraction as F
import json
from pathlib import Path

import numpy as np
from .plotting import COLORS, plt, save


def make():
    root = Path(__file__).resolve().parents[1]
    data = json.loads((root/'data/exactness_results.json').read_text(encoding='utf-8'))
    modes = ('component', 'posterior', 'wrong_marginal')
    labels = ('Condition on Z', 'Bayes conditional', 'Wrong slot marginal')
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.2), sharey=True)
    for ax, mode, label in zip(axes, modes, labels):
        result = data['mixtures'][mode]
        matrix = np.zeros((2, 2))
        for sequence, probability in result['law'].items():
            x, y, _ = map(int, sequence.split(','))
            matrix[x, y] += float(F(probability))
        ax.imshow(matrix, cmap='Blues', vmin=0, vmax=.5)
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                    fontsize=16, color='white' if value>.33 else COLORS['target'])
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['A', 'B'],
               yticklabels=['A', 'B'], xlabel='Second output token')
        ax.set_title(f'{label}\nTV = {float(F(result["tv"])):.2f}; '
                     f'E[A] = {float(F(result["expected_accepted"])):.2f}', fontsize=12)
    axes[0].set_ylabel('First output token')
    fig.suptitle('Equal single-token marginals can hide a wrong joint law', fontsize=14)
    fig.subplots_adjust(bottom=.16, top=.76, wspace=.24)
    save(fig, '24_latent_mixture_joint')

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5))
    keys = ('1', '0,1', '0,0,1', '0,0,0')
    names = ('EOS', 'A EOS', 'A A EOS', 'A A A')
    probabilities = [float(F(data['eos_law'][key])) for key in keys]
    axes[0].bar(names, probabilities, color=[COLORS['accept']]*3+[COLORS['target']])
    for i, probability in enumerate(probabilities):
        axes[0].text(i, probability+.015, f'{probability:g}', ha='center')
    axes[0].set(ylabel='Probability', ylim=(0, .62),
                title='Stopped target law (maximum 3 tokens)')
    horizons = np.arange(1, 513)
    for epsilon in (.0001, .001, .01):
        bound = -np.expm1(horizons*np.log1p(-epsilon))
        axes[1].plot(horizons, bound, label=f'per-step TV ≤ {epsilon:g}')
    axes[1].set(xlabel='Maximum generated tokens', ylabel='Sequence TV upper bound',
                ylim=(0, 1.03), title='Uniform error bound: 1 − (1 − ε)ⁿ')
    axes[1].legend(frameon=False, fontsize=10)
    fig.text(.5, -.015, 'Exact finite law (left); worst-case theoretical bounds, not measured drift (right)',
             ha='center', color=COLORS['muted'])
    fig.tight_layout()
    save(fig, '25_stopping_and_error_bounds')


if __name__ == '__main__':
    make()
