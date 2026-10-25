"""Systems-review SVGs from saved CPU results and explicit synthetic assumptions."""
import json
from pathlib import Path

import numpy as np

from .plotting import COLORS, plt, save

ROOT = Path(__file__).resolve().parents[1]


def cache_transaction(result):
    data = result['cache']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    mask = np.asarray(data['mask'])
    ax = axes[0]
    ax.imshow(mask, cmap='Blues', vmin=0, vmax=1)
    for (i, j), value in np.ndenumerate(mask):
        ax.text(j, i, str(value), ha='center', va='center',
                color='white' if value else '#172b43')
    ax.set(xticks=range(6), xticklabels=['1', '2', '3', '4', '5', '6'],
           yticks=range(4), yticklabels=['3 -> p(4)', '4 -> p(5)', '5 -> p(6)', '6 -> bonus'],
           xlabel='Key token ID: cached history | verification input',
           title='History S=2, new queries N=4')
    ax.axvline(1.5, color=COLORS['reject'], linewidth=2)
    names = ['Correct crop', 'No crop', 'Keep rejected 5']
    values = [data['next_logits_errors'][k] for k in ['correct_crop', 'no_crop', 'retain_one_rejected']]
    axes[1].bar(names, np.maximum(values, 1e-16), color=[COLORS['accept'], COLORS['reject'], COLORS['draft']])
    axes[1].set(yscale='log', ylim=(1e-16, 10), ylabel='Max absolute logit error',
                title='Next token 7: compare with full [1,2,3,4,7]')
    for i, value in enumerate(values):
        axes[1].text(i, max(value, 1e-16)*2, f'{value:.2e}', ha='center', fontsize=9)
    fig.supxlabel('Random-weight CPU Transformer; accepted 4, rejected 5/6, correction 7 pending.', fontsize=10)
    save(fig, 'review_systems_cache')


def roofline(result):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    for name, color, label in [('ideal_reuse', 'target', 'Ideal historical KV reuse'),
                               ('no_reuse', 'reject', 'Read KV per query')]:
        curve = result['roofline']['curves'][name]
        x = [r['queries'] for r in curve]
        axes[0].plot(x, [1000*r['bound_seconds'] for r in curve], 'o-', color=COLORS[color], label=label)
    ideal = result['roofline']['curves']['ideal_reuse']
    x = [r['queries'] for r in ideal]
    axes[1].plot(x, [1000*r['compute_seconds'] for r in ideal], label='Compute bound', color=COLORS['draft'])
    axes[1].plot(x, [1000*r['memory_seconds'] for r in ideal], label='Memory bound (ideal reuse)', color=COLORS['target'])
    for ax in axes:
        ax.set(xlabel='Verification input tokens per request (includes pending token)',
               ylabel='Analytical lower bound (ms)', xticks=[1, 8, 16, 24, 32])
        ax.legend(fontsize=9)
    axes[0].set_title('Kernel reuse determines the useful region')
    axes[1].set_title('The maximum, not the sum, is the roofline bound')
    fig.supxlabel('Synthetic: 7B dense-equivalent params, B=8, S=32768, GQA=8/32, 3 TB/s, 300 TFLOP/s. No GPU measurement.', fontsize=9)
    save(fig, 'review_systems_roofline')


def serving(result):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    for name, color, label in [('smooth', 'target', '8 chunks, one token each'),
                               ('bursts', 'reject', '2 chunks, four tokens each')]:
        row = result['streaming'][name]
        axes[0].scatter(row['token_timestamps'], np.arange(1, row['tokens']+1),
                        color=COLORS[color], label=label, marker='o' if name == 'smooth' else 'x')
    axes[0].set(xlabel='Client receive time (s)', ylabel='Received token index',
                title='Same TTFT, E2E and TPOT; different silence gaps')
    axes[0].legend(fontsize=9)
    for name, color, label in [('constant', 'target', 'Constant 80 ms service'),
                               ('bursty', 'reject', '20, 20, 20, 260 ms service')]:
        row = result['queue'][name]
        axes[1].plot(row['arrivals'], np.asarray(row['response_seconds'])*1000,
                     label=label, color=COLORS[color])
    axes[1].set(xlabel='Arrival time (s)', ylabel='Response time including queue (ms)',
                title='Same mean service 80 ms; different queueing')
    axes[1].legend(fontsize=9)
    fig.supxlabel('Synthetic traces only. Right: arrivals every 100 ms, one FCFS server, no continuous batching.', fontsize=10)
    save(fig, 'review_systems_serving')


def main():
    result = json.loads((ROOT/'data/systems_results.json').read_text())
    cache_transaction(result)
    roofline(result)
    serving(result)
    print('Rebuilt 3 systems figures from data/systems_results.json.')


if __name__ == '__main__':
    main()
