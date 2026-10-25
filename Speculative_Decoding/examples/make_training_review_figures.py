"""Regenerate three original figures from saved training-theory checks."""
import json
from pathlib import Path
import numpy as np
from .plotting import plt, save, box, COLORS

ROOT = Path(__file__).resolve().parents[1]


def dependency():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), layout='constrained')
    matrices = [np.array([[.45, .05], [.05, .45]]), np.full((2, 2), .25),
                np.array([[.45, .05], [.05, .45]])]
    titles = ['Target: correlated tokens', 'Independent slots: correct marginals',
              'Conditional head: matched joint']
    for ax, matrix, title in zip(axes, matrices, titles):
        ax.imshow(matrix, vmin=0, vmax=.5, cmap='Blues')
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, f'{value:.2f}', ha='center', va='center', fontsize=15,
                    color='white' if value > .3 else '#172b43')
        ax.set(xticks=[0, 1], xticklabels=['A', 'B'], yticks=[0, 1], yticklabels=['A', 'B'],
               xlabel='Second token', ylabel='First token', title=title)
    fig.supxlabel('All one-token marginals are [0.5, 0.5]. Second-prefix survival: 1.0 vs 0.6 vs 1.0.',
                  fontsize=11)
    save(fig, 'review_training_joint_dependence')


def objectives(result):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout='constrained')
    r = result['kl_ranking']
    x = np.arange(2)
    axes[0].bar(x - .18, [r['kl_a'], r['kl_b']], width=.36, label='Forward KL (nats)', color=COLORS['draft'])
    axes[0].bar(x + .18, [r['acceptance_a'], r['acceptance_b']], width=.36,
                label='Single-step acceptance', color=COLORS['accept'])
    axes[0].set(xticks=x, xticklabels=['Draft A', 'Draft B'], ylim=(0, 1),
                title='Lower KL need not mean higher acceptance')
    axes[0].legend(fontsize=9)
    rows = result['gradients']
    for key, label, color in [('ce_norm', 'CE / negative log-overlap', COLORS['target']),
                               ('tv_norm', 'TV', COLORS['reject'])]:
        axes[1].loglog([r['vocab'] for r in rows], [r[key] for r in rows],
                       'o-', label=label, color=color)
    axes[1].set(xlabel='Vocabulary size V', ylabel='Logit-gradient L2 norm',
                title='Uniform draft, point-mass target')
    axes[1].legend(fontsize=9)
    fig.supxlabel('Exact toy calculations; unrestricted global optima still coincide at q = p.', fontsize=11)
    save(fig, 'review_training_objectives')


def round_starts():
    fig, axes = plt.subplots(2, 1, figsize=(12, 5.5), layout='constrained')
    for ax in axes:
        ax.set(xlim=(0, 12), ylim=(0, 2)); ax.axis('off')
    labels = [
        [('Start n=0', 0.2, 2.0, 'muted'), ('Accept x1; repair x2', 2.65, 3.0, 'draft'),
         ('Repair x3', 6.1, 2.25, 'reject'), ('Emit terminal x4', 8.8, 2.8, 'target')],
        [('Start n=0', 0.2, 2.0, 'muted'), ('Repair x1', 2.65, 3.0, 'reject'),
         ('Accept x2, x3; bonus terminal x4', 6.1, 5.5, 'accept')]]
    for ax, row, title in zip(axes, labels,
                              ['Policy A: first-round output 2; total 3 rounds',
                               'Policy B: first-round output 1; total 2 rounds']):
        ax.text(.2, 1.7, title, weight='bold', fontsize=14)
        for label, x, width, color in row:
            box(ax, x, .65, width, .6, label, COLORS[color])
        for left, right in zip(row, row[1:]):
            ax.annotate('', (right[1] - .06, .95), (left[1] + left[2] + .03, .95),
                        arrowprops={'arrowstyle': '->', 'color': COLORS['muted']})
    fig.supxlabel('Same deterministic four-token target; gamma = 2. The drafter depends on the round start.',
                  fontsize=11)
    save(fig, 'review_training_round_starts')


def main():
    result = json.loads((ROOT / 'data/training_objectives_results.json').read_text())
    dependency(); objectives(result); round_starts()


if __name__ == '__main__':
    main()
