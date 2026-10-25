#!/usr/bin/env python3
"""LLM.int8() teaching reference: NumPy arithmetic, not a bitsandbytes kernel.

Run without flags for numeric experiments; --write-results also saves JSON/figures.
The default outlier branch uses float64 to isolate quantization error. A separate
float16 branch experiment exposes rounding from the high-precision path.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.quant_lab import symmetric_qdq
from experiments.result_utils import run_metadata, write_json


def validate_inputs(x, w):
    x, w = np.asarray(x), np.asarray(w)
    for value in (x, w):
        if (value.ndim != 2 or 0 in value.shape or not np.isrealobj(value)
                or not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all()):
            raise ValueError("Expected nonempty, finite, real matrices")
    if x.shape[1] != w.shape[1]:
        raise ValueError("Use X=(tokens, features), W=(outputs, features)")
    # Conservative INT32 accumulator bound; typical LLM dimensions fit easily.
    if x.shape[1] * 127**2 > np.iinfo(np.int32).max:
        raise ValueError("Reference INT32 accumulator might overflow")
    return x.astype(np.float64), w.astype(np.float64)


def quantize_absmax(value, axis=None):
    """Return INT8 codes and dequantization steps (scale = absmax / 127)."""
    _, codes, scale = symmetric_qdq(value, bits=8, axis=axis)
    return codes.astype(np.int8), scale


def int8_matmul(x, w, vectorwise=True):
    """INT8 operands, INT32 sums, floating-point rescaling; no CUDA speed claim."""
    x, w = validate_inputs(x, w)
    axis = 1 if vectorwise else None
    qx, sx = quantize_absmax(x, axis=axis)
    qw, sw = quantize_absmax(w, axis=axis)
    # Casting is essential: NumPy INT8 @ INT8 would overflow in its INT8 output.
    accumulator = qx.astype(np.int32) @ qw.astype(np.int32).T
    output = accumulator.astype(np.float64) * (sx * sw.T)
    return output, {"qx": qx, "qw": qw, "sx": sx, "sw": sw, "accumulator": accumulator}


def split_matmul(x, w, threshold=6.0, high_dtype=np.float64):
    """Paper-level decomposition: preserve original outlier weights in high precision.

    Not bit-exact to bitsandbytes: no packed layouts, original outlier weights are
    available, and ordinary weight scales are recomputed after slicing W.
    A positive threshold uses strict abs(x) > threshold. To disable splitting,
    call int8_matmul instead; zero is deliberately not a special flag here.
    """
    x, w = validate_inputs(x, w)
    if (not np.isscalar(threshold) or isinstance(threshold, (bool, np.bool_))
            or not np.isfinite(threshold) or threshold <= 0):
        raise ValueError("threshold must be positive and finite")
    if np.dtype(high_dtype) not in (np.dtype(np.float16), np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("Use a floating-point dtype for the outlier branch")
    # Union over all tokens: split feature columns, not merely individual elements.
    outliers = np.any(np.abs(x) > threshold, axis=0)
    regular = ~outliers
    y_regular = np.zeros((x.shape[0], w.shape[0]), dtype=np.float64)
    qstate = None
    if regular.any():
        y_regular, qstate = int8_matmul(x[:, regular], w[:, regular])
    y_outlier = np.zeros_like(y_regular)
    if outliers.any():
        y_outlier = (x[:, outliers].astype(high_dtype)
                     @ w[:, outliers].astype(high_dtype).T).astype(np.float64)
    return y_regular + y_outlier, {
        "outliers": outliers, "regular": regular, "quantization": qstate,
        "y_regular": y_regular, "y_outlier": y_outlier,
    }


def error_metrics(reference, candidate):
    squared_error = np.sum((reference - candidate)**2)
    energy = np.sum(reference**2)
    return {"mse": float(np.mean((reference - candidate)**2)),
            "nmse": float(squared_error / max(energy, 1e-30)),
            "max_abs_error": float(np.max(np.abs(reference - candidate)))}


def hand_example():
    x = np.array([[.02, -.04, .10, 30.], [.08, -.06, .02, -25.]])
    w = np.array([[.4, -.7, .2, .05], [-.3, .1, .8, -.04]])
    reference = x @ w.T
    vector, state = int8_matmul(x, w)
    mixed, split = split_matmul(x, w)
    mixed_fp16, _ = split_matmul(x, w, high_dtype=np.float16)
    removed = x[:, split['regular']] @ w[:, split['regular']].T
    # A storage caveat, NOT a reproduction of any particular library release:
    # high-precision multiplication cannot undo previously quantized W_O.
    qw, sw = quantize_absmax(w, axis=1)
    reconstructed = qw.astype(np.float64) * sw
    high_with_quantized_weights = x[:, split['outliers']] @ reconstructed[:, split['outliers']].T
    mixed_stored_w = split['y_regular'] + high_with_quantized_weights
    return x, w, {
        "x": x.tolist(), "w": w.tolist(), "reference": reference.tolist(),
        "vector_qx": state['qx'].tolist(), "vector_sx": state['sx'].ravel().tolist(),
        "outlier_columns": np.flatnonzero(split['outliers']).tolist(),
        "regular_qx": split['quantization']['qx'].tolist(),
        "regular_sx": split['quantization']['sx'].ravel().tolist(),
        "y_regular": split['y_regular'].tolist(), "y_outlier": split['y_outlier'].tolist(),
        "outputs": {"vector_only": vector.tolist(), "mixed_exact_branch": mixed.tolist(),
                    "mixed_fp16_branch": mixed_fp16.tolist(), "drop_outliers": removed.tolist(),
                    "mixed_reconstructed_weight_example": mixed_stored_w.tolist()},
        "errors": {name: error_metrics(reference, candidate) for name, candidate in (
            ("vector_only", vector), ("mixed_exact_branch", mixed), ("mixed_fp16_branch", mixed_fp16),
            ("drop_outliers", removed), ("mixed_reconstructed_weight_example", mixed_stored_w))},
    }


def synthetic_case(seed=42):
    rng = np.random.default_rng(seed)
    x = np.clip(rng.normal(0, .5, size=(64, 256)), -1.5, 1.5)
    x *= np.geomspace(.25, 2.0, 64)[:, None]
    for index, lower, upper in ((7, 8, 12), (89, 14, 22), (201, 28, 35)):
        x[:, index] = rng.choice([-1., 1.], 64) * rng.uniform(lower, upper, 64)
    w = rng.normal(0, .18, size=(48, 256)) * np.geomspace(.3, 2., 48)[:, None]
    return x, w


def run_experiments():
    _, _, hand = hand_example()
    x, w = synthetic_case()
    reference = x @ w.T
    results = []
    for name, candidate in (
        ("tensor_INT8", int8_matmul(x, w, vectorwise=False)[0]),
        ("vector_INT8", int8_matmul(x, w)[0]),
        ("split_FP64_reference", split_matmul(x, w)[0]),
        ("split_FP16_branch", split_matmul(x, w, high_dtype=np.float16)[0]),
    ):
        results.append({"method": name, **error_metrics(reference, candidate)})
    sweep = []
    for threshold in (.5, 1., 2., 4., 6., 8., 12., 20., 30., 35., 40.):
        y, state = split_matmul(x, w, threshold=threshold)
        count = int(state['outliers'].sum())
        sweep.append({"threshold": threshold, "outlier_columns": count,
                      "outlier_column_fraction": count / x.shape[1], **error_metrics(reference, y)})
    return {"scope": "Synthetic NumPy arithmetic, not a GPU benchmark or LLM quality evaluation",
            "seed": 42, "shape_x": list(x.shape), "shape_w": list(w.shape),
            "high_branch_default": "float64 idealized reference; original outlier weights available",
            "hand_example": hand, "ablation": results, "threshold_sweep": sweep}


def figures(result, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})

    def save(fig, stem):
        for suffix in ('png', 'svg'):
            fig.savefig(directory / f'{stem}.{suffix}', dpi=170, bbox_inches='tight')
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.set(xlim=(0, 13), ylim=(0, 6.5))
    ax.axis('off')
    def box(x, y, width, height, label, color):
        ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle='round,pad=.12',
                                   facecolor=color, edgecolor='#334155'))
        ax.text(x + width/2, y + height/2, label, ha='center', va='center', fontsize=10)
    def arrow(a, b, label=None):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle='->', mutation_scale=15, color='#334155'))
        if label:
            ax.text((a[0]+b[0])/2, (a[1]+b[1])/2+.1, label, ha='center', fontsize=9)
    box(.2, 2.6, 2.0, 1.1, 'X: [tokens, features]\nW: [outputs, features]\nFloating-point inputs', '#e2e8f0')
    box(2.8, 2.6, 2.3, 1.1, 'Detect feature columns\nany token: |X[:, k]| > tau\nSplit SAME k in X and W', '#fef3c7')
    box(5.8, 4.25, 3.0, 1.3, 'Ordinary features R\nINT8 X_R and W_R\nINT32 accumulation\nScale by outer product', '#dbeafe')
    box(5.8, .8, 3.0, 1.3, 'Outlier features O\nHigher-precision matmul\nX_O @ W_O.T', '#ffedd5')
    box(9.65, 2.6, 2.8, 1.1, 'ADD two floating outputs\n+ bias (once)\nY: [tokens, outputs]', '#dcfce7')
    arrow((2.35, 3.15), (2.65, 3.15))
    arrow((5.2, 3.5), (5.65, 4.7), 'R')
    arrow((5.2, 2.8), (5.65, 1.5), 'O')
    arrow((8.95, 4.7), (9.5, 3.5))
    arrow((8.95, 1.5), (9.5, 2.8))
    ax.text(6.5, 6.1, 'LLM.int8(): vector-wise quantization + mixed-precision decomposition', ha='center', fontsize=14)
    ax.text(6.5, .12, 'Paper-level data flow. Physical layouts, weight storage and fused kernels vary by implementation.', ha='center', fontsize=10)
    save(fig, '01_dataflow')

    x, _, hand = hand_example()
    small = x[0, :3]
    step_all, step_regular = 30/127, .1/127
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), constrained_layout=True)
    positions = np.arange(3)
    for ax, step, title in zip(axes, (step_all, step_regular),
                               ('Outlier shares the scale: step = 30/127', 'Outlier removed from scale: step = 0.1/127')):
        restored = np.rint(small / step) * step
        ax.bar(positions-.17, small, width=.34, label='Original', color='#64748b')
        ax.bar(positions+.17, restored, width=.34, label='Quantize + dequantize', color='#2563eb')
        ax.axhline(0, color='black', lw=.6)
        ax.set(xticks=positions, xticklabels=['feature 0', 'feature 1', 'feature 2'], ylim=(-.065, .13), title=title)
        ax.legend(fontsize=9)
        for pos, val in zip(positions+.17, restored):
            ax.annotate(f'{val:.5f}', (pos, val), xytext=(0, 7 if val >= 0 else -13), textcoords='offset points', ha='center', fontsize=9)
    fig.suptitle('Same INT8 bit width: different scale resolution (large feature 30 omitted from bars)', fontsize=12)
    save(fig, '02_quantization_grid')

    x, w = synthetic_case()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    im = axes[0].imshow(x, aspect='auto', cmap='coolwarm', norm=SymLogNorm(linthresh=.5, vmin=-35, vmax=35))
    axes[0].set(title='Synthetic X: 3 persistent outlier columns', xlabel='Feature dimension k', ylabel='Token row t')
    fig.colorbar(im, ax=axes[0], label='Activation (symmetric-log colors)')
    values = [row['nmse'] for row in result['ablation']]
    axes[1].bar(['Tensor INT8', 'Vector INT8', 'Split / FP64', 'Split / FP16'], values,
                color=['#64748b', '#2563eb', '#16a34a', '#86efac'])
    axes[1].set(yscale='log', ylabel='Output NMSE (log scale)', title='Synthetic ablation: not model PPL')
    axes[1].tick_params(axis='x', rotation=20)
    for i, v in enumerate(values):
        axes[1].text(i, v*1.15, f'{v:.2e}', ha='center', fontsize=9)
    axes[1].set_ylim(min(values)*.5, max(values)*3)
    save(fig, '03_ablation')

    sweep = result['threshold_sweep']
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    xs = [row['threshold'] for row in sweep]
    positive = [row['nmse'] for row in sweep if row['nmse'] > 0]
    plot_floor = min(positive) / 10
    ax.plot(xs, [max(row['nmse'], plot_floor) for row in sweep], 'o-', color='#2563eb', label='Output NMSE')
    for row in sweep:
        if row['nmse'] == 0:
            ax.annotate('0 (all high precision)', (row['threshold'], plot_floor),
                        xytext=(10, 10 if row['threshold'] == .5 else 27), textcoords='offset points', fontsize=8)
    ax.set(yscale='log', xlabel='Positive threshold tau', ylabel='Output NMSE (log scale)',
           title='Synthetic threshold sweep: error vs outlier-column fraction')
    right = ax.twinx()
    right.plot(xs, [100*row['outlier_column_fraction'] for row in sweep], 's--', color='#ea580c')
    right.set_ylabel('Features assigned to high precision / %', color='#ea580c')
    right.set_ylim(-3, 105)
    ax.axvline(6, color='#64748b', linestyle=':', label='tau = 6')
    ax.legend(loc='center right')
    fig.text(.5, -.02, 'Column fraction is a work proxy, NOT measured latency. Error is not guaranteed monotonic.', ha='center', fontsize=10)
    save(fig, '04_threshold')

    measured = ROOT / 'experiments/archive/colab_t4_2026-10-15/metrics_from_notebook.csv'
    if measured.exists():
        with measured.open() as handle:
            measured_rows = {row['mode']: row for row in csv.DictReader(handle)}
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        for ax, key, title in zip(axes,
            ['ppl', 'peak_generation_allocated_mib', 'generation_tokens_per_s'],
            ['PPL (lower is better)', 'Peak allocated memory / MiB', 'Generation tokens/s']):
            vals = [float(measured_rows[name][key]) for name in ('fp16', 'int8')]
            bars = ax.bar(['FP16', 'LLM.int8()'], vals, color=['#64748b', '#2563eb'])
            ax.bar_label(bars, labels=[f'{v:.2f}' for v in vals], padding=3)
            ax.set_ylim(0, max(vals)*1.2)
            ax.set_title(title)
        fig.suptitle('Historical T4 run: Qwen3-0.6B / smoke / batch=1 / 64 generated tokens', fontsize=12)
        save(fig, '05_t4_measured')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-results', action='store_true')
    parser.add_argument('--output-dir', type=Path, help='Result directory (requires --write-results)')
    args = parser.parse_args()
    if args.output_dir is not None and not args.write_results:
        parser.error('--output-dir requires --write-results')
    result = run_experiments()
    print('Synthetic output errors (not GPU latency or LLM PPL):')
    for row in result['ablation']:
        print(f"{row['method']:24s} NMSE={row['nmse']:.8g} max_error={row['max_abs_error']:.8g}")
    print('Hand example outlier columns:', result['hand_example']['outlier_columns'])
    print('Hand example reference:', result['hand_example']['reference'])
    if args.write_results:
        output_dir = args.output_dir or ROOT/'figures/llm_int8_deep_dive'
        figures(result, output_dir)
        payload = {**run_metadata({'seed': 42, 'figures': True}, [
            'docs/专题_LLM_int8_原理与实测.md',
            'experiments/archive/colab_t4_2026-10-15/metrics_from_notebook.csv',
        ]), **result}
        write_json(output_dir/'results.json', payload)
        print('Wrote results and figures to', output_dir)


if __name__ == '__main__':
    main()
