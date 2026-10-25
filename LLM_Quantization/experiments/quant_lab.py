#!/usr/bin/env python3
"""Small CPU demonstrations of quantization mathematics, not production kernels.

Run: python experiments/quant_lab.py
Write results: python experiments/quant_lab.py --write-results
Optional figures: python experiments/quant_lab.py --write-results --figures
Only NumPy is needed unless --figures is passed. All experiments are deterministic.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.validation import finite_array, integer
from experiments.result_utils import PUBLISHED_RESULTS, file_link, run_metadata, write_json, write_text


NF4 = np.array([
    -1., -.6961928009986877, -.5250730514526367, -.39491748809814453,
    -.28444138169288635, -.18477343022823334, -.09105003625154495, 0.,
    .07958029955625534, .16093020141124725, .24611230194568634,
    .33791524171829224, .44070982933044434, .5626170039176941,
    .7229568362236023, 1.,
])


def symmetric_qdq(x, bits=4, axis=None, bound=None):
    """Symmetric integer grid [-qmax,qmax], float64 fake quantization.

    axis is the REDUCTION axis that shares a scale, not the preserved channel.
    NumPy rint uses ties-to-even. Zero groups return exact zero with scale 1.
    An explicit bound must be finite and strictly positive.
    """
    bits = integer(bits, "bits", 2, 16)
    x = finite_array(x, "x")
    qmax = 2 ** (bits - 1) - 1
    if bound is None:
        amax = np.max(np.abs(x), axis=axis, keepdims=True)
    else:
        amax = finite_array(bound, "bound")
        if not np.all(amax > 0):
            raise ValueError("An explicit clipping bound must be finite and positive.")
        if np.broadcast_shapes(x.shape, amax.shape) != x.shape:
            raise ValueError("bound must broadcast without changing the input shape.")
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
            scale = np.where(amax > 0, amax / qmax, 1.)
            if np.any(scale <= 0):
                raise ValueError('Quantization scale underflows float64; rescale the input.')
            # Clip before division: an explicit tiny bound need not overflow x/s.
            q = np.rint(np.clip(x, -amax, amax) / scale)
            q = np.clip(q, -qmax, qmax).astype(np.int64)
            restored = q * scale
    except FloatingPointError as exc:
        raise ValueError('Quantization exceeds float64 range; rescale the input.') from exc
    return restored, q, scale


def affine_qdq(x, bits=2, axis=None):
    """Integer zero-point affine quantizer, range expanded to include zero."""
    bits = integer(bits, "bits", 1, 16)
    x = finite_array(x, "x")
    lo = np.minimum(np.min(x, axis=axis, keepdims=True), 0.)
    hi = np.maximum(np.max(x, axis=axis, keepdims=True), 0.)
    qmax = 2**bits - 1
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
            scale = np.where(hi > lo, (hi - lo) / qmax, 1.)
            if np.any(scale <= 0):
                raise ValueError('Quantization scale underflows float64; rescale the input.')
            zero = np.clip(np.rint(-lo / scale), 0, qmax).astype(np.int64)
            q = np.clip(np.rint(x / scale) + zero, 0, qmax).astype(np.int64)
            restored = scale * (q - zero)
    except FloatingPointError as exc:
        raise ValueError('Quantization exceeds float64 range; rescale the input.') from exc
    return restored, q, scale, zero


def group_qdq(w, bits=4, group_size=4):
    group_size = integer(group_size, "group_size")
    w = finite_array(w, "w", ndim=2)
    if w.shape[1] % group_size:
        raise ValueError("Demo requires a 2D matrix with full groups; no padding.")
    groups = w.reshape(w.shape[0], -1, group_size)
    return symmetric_qdq(groups, bits, axis=-1)[0].reshape(w.shape)


def codebook_qdq(w, codebook, group_size=64):
    group_size = integer(group_size, "group_size")
    w = finite_array(w, "w")
    codebook = finite_array(codebook, "codebook", ndim=1)
    if w.size % group_size:
        raise ValueError("The number of weights must be divisible by group_size.")
    blocks = w.reshape(-1, group_size)
    scales = np.max(np.abs(blocks), axis=1, keepdims=True)
    scales = np.where(scales > 0, scales, 1.)
    indices = np.argmin(abs(blocks[..., None] / scales[..., None] - codebook), axis=-1)
    return (scales * codebook[indices]).reshape(np.shape(w))


def hadamard(n):
    n = integer(n, "n")
    if n < 1 or n & (n - 1):
        raise ValueError("Demo Hadamard supports powers of two only.")
    h = np.ones((1, 1))
    while len(h) < n:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(n)


def softmax(x):
    ex = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return ex / ex.sum(axis=-1, keepdims=True)


def sequential_compensation(w, h, grid):
    """Pedagogical sequential OBQ/GPTQ principle, NOT efficient GPTQ.

    Fixed grid/order; one output row; inverse downdate each iteration.
    No grouping, packing, damping, activation ordering or block updates.
    """
    current = np.array(w, dtype=float, copy=True)
    inv = np.linalg.inv(h)
    result = []
    for i in range(len(current)):
        qi = grid[np.argmin(abs(grid - current[i]))]
        delta = (qi - current[i]) * inv[:, 0] / inv[0, 0]
        current[i:] += delta
        result.append(float(qi))
        if len(inv) > 1:
            inv = inv[1:, 1:] - np.outer(inv[1:, 0], inv[0, 1:]) / inv[0, 0]
    return np.array(result)


def mse(a, b):
    return float(np.mean((np.asarray(a) - b) ** 2))


def run():
    rng = np.random.default_rng(20261010)
    metrics = {}
    checks = []

    def check(name, condition):
        if not bool(condition):
            raise AssertionError(name)
        checks.append(name)

    # 1. Hand calculation, zero groups and actual 4-bit packing.
    x = np.array([-1., -.3, .2, .9])
    xhat, q, scale = symmetric_qdq(x)
    check("INT4 手算编码", np.array_equal(q, [-7, -2, 1, 6]))
    check("全零组", np.array_equal(symmetric_qdq(np.zeros(4))[0], np.zeros(4)))
    for invalid_bound in (0., -1., np.nan, np.inf):
        try:
            symmetric_qdq([1., 2.], bound=invalid_bound)
        except ValueError:
            checks.append(f"拒绝非法裁剪范围 {invalid_bound}")
        else:
            raise AssertionError(f"Accepted invalid bound {invalid_bound}")
    check("显式正裁剪范围", np.allclose(symmetric_qdq([-2., 2.], bound=.5)[0], [-.5, .5]))
    af, aq, asc, az = affine_qdq(np.array([-1., 0., .7, 2.]))
    check("仿射手算与整数零点", np.array_equal(aq, [0, 1, 2, 3]) and int(az.item()) == 1)
    check("仿射全零组", np.array_equal(affine_qdq(np.zeros(4))[0], np.zeros(4)))
    unsigned = (q + 8).astype(np.uint8)
    packed = unsigned[::2] | (unsigned[1::2] << 4)
    unpacked = np.empty(4, dtype=np.int64)
    unpacked[::2], unpacked[1::2] = packed & 15, packed >> 4
    check("INT4 打包往返", np.array_equal(unpacked - 8, q) and packed.nbytes == 2)
    metrics["01_uniform"] = {"q": q.tolist(), "scale": scale.item(), "x_hat": xhat.tolist(),
                              "payload_bytes": packed.nbytes, "affine_x_hat": af.tolist()}

    # 2. Rounding error law is a statistical approximation.
    samples = rng.uniform(-1, 1, 200_000)
    step = .1
    noise = step * np.rint(samples / step) - samples
    observed = np.mean(noise**2)
    expected = step**2 / 12
    check("无裁剪半步长界", np.max(abs(noise)) <= step / 2 + 1e-12)
    check("均匀噪声的 s²/12 近似", abs(observed / expected - 1) < .02)
    metrics["02_noise"] = {"observed_mse": float(observed), "s2_over_12": expected}

    # 3. Clipping and group size, constructed data rather than LLM weights.
    weights = rng.normal(0, .7, 20_000)
    weights[0] = 8.
    bounds = np.linspace(.5, 8., 100)
    errors = np.array([mse(weights, symmetric_qdq(weights, bound=c)[0]) for c in bounds])
    best = int(np.argmin(errors))
    tiny = np.array([[.1, .2, .3, 10.]])
    metrics["03_clipping_groups"] = {
        "best_bound": float(bounds[best]), "best_mse": float(errors[best]),
        "minmax_mse": float(errors[-1]), "one_group_mse": mse(tiny, group_qdq(tiny, group_size=4)),
        "two_groups_mse": mse(tiny, group_qdq(tiny, group_size=2)),
    }

    # 4. Exact quadratic form and constrained compensation.
    X = np.array([[1., .5], [0., np.sqrt(3) / 2]])
    H = 2 * X.T @ X
    w = np.array([.6, .6])
    grid = np.array([0., 1.])
    rtn = np.rint(w)
    gpt = sequential_compensation(w, H, grid)
    loss = lambda wh: float(np.sum((X @ (wh - w)) ** 2))
    check("输出误差等于二次型", np.isclose(loss(rtn), .5 * (rtn-w) @ H @ (rtn-w)))
    check("GPTQ 手算例子", np.allclose(gpt, [1, 0]) and np.isclose(loss(rtn), .48) and np.isclose(loss(gpt), .28))
    M = np.linalg.inv(H)
    delta = .4 * M[:, 0] / M[0, 0]
    check("受约束最小损失", np.isclose(.5 * delta @ H @ delta, .4**2 / (2*M[0, 0])))
    schur = M[1:, 1:] - np.outer(M[1:, 0], M[0, 1:]) / M[0, 0]
    check("逆 Hessian 消元", np.allclose(schur, np.linalg.inv(H[1:, 1:])))
    metrics["04_gptq"] = {"rtn": rtn.tolist(), "compensated": gpt.tolist(),
                            "rtn_loss": loss(rtn), "compensated_loss": loss(gpt)}
    # Independent control: change X and derive H, keeping the original example intact.
    X_identity = np.eye(2) / np.sqrt(2)
    H_identity = 2 * X_identity.T @ X_identity
    gpt_identity = sequential_compensation(w, H_identity, grid)
    identity_loss = float(np.sum((X_identity @ (gpt_identity - w)) ** 2))
    check("单位 Hessian 与输入一致", np.allclose(H_identity, np.eye(2)))
    check("无耦合时补偿退化为 RTN", np.array_equal(gpt_identity, rtn) and np.isclose(identity_loss, .16))
    metrics["04_gptq"].update(identity_compensated=gpt_identity.tolist(), identity_loss=identity_loss)

    # 5. GPTAQ-style nonzero linear term, checked against independent KKT solve.
    gradient = np.array([.3, -.2])
    unit = np.array([1., 0.])
    epsilon = .4
    formula = -M @ gradient + (epsilon + unit @ M @ gradient) / M[0, 0] * (M @ unit)
    kkt = np.block([[H, unit[:, None]], [unit[None, :], np.zeros((1, 1))]])
    solution = np.linalg.solve(kkt, np.r_[-gradient, epsilon])[:2]
    check("非对称重建约束解", np.allclose(formula, solution))
    metrics["05_asymmetric_calibration"] = {"delta": formula.tolist()}

    # 6. Equivalence of scaling, translation and rotation; nonlinear counterexample.
    X = rng.normal(size=(128, 8)); X[:, 0] *= 20
    W = rng.normal(size=(6, 8)) * .2
    D = np.exp(rng.normal(size=8))
    R = hadamard(8)
    base = X @ W.T
    check("缩放等价", np.allclose((X / D) @ (W * D).T, base))
    check("旋转等价", np.allclose((X @ R) @ (W @ R).T, base))
    center = rng.normal(size=8); bias = rng.normal(size=6)
    check("平移缩放与偏置补偿", np.allclose(((X-center)/D) @ (W*D).T + bias + W @ center, base+bias))
    rms = lambda a: a / np.sqrt(np.mean(a*a, axis=-1, keepdims=True) + 1e-6)
    check("无逐通道增益的 RMSNorm 等价", np.allclose(rms(X @ R), rms(X) @ R))
    nonlinear_gap = float(np.max(abs(np.maximum(X @ R, 0) - np.maximum(X, 0) @ R)))
    check("非线性不能随意穿过旋转", nonlinear_gap > .1)
    metrics["06_transforms"] = {"rotation_identity_max_error": float(np.max(abs((X@R)@(W@R).T-base))),
                                "relu_commutation_gap": nonlinear_gap}

    # 7. AWQ principle: shared group scale matters. Not complete AWQ.
    alphas = np.linspace(0, 1, 21)
    activation_mean = np.mean(abs(X), axis=0)
    awq_errors = []
    for alpha in alphas:
        scales = activation_mean ** alpha
        scales /= np.sqrt(scales.max() * scales.min())
        approx = (X / scales) @ group_qdq(W * scales, group_size=4).T
        awq_errors.append(mse(base, approx))
    ai = int(np.argmin(awq_errors))
    col_before = symmetric_qdq(W, axis=0)[0]
    col_after = symmetric_qdq(W * D, axis=0)[0] / D
    check("独立输入列尺度可抵消 AWQ 缩放", np.allclose(col_before, col_after))
    amax = np.max(abs(X), axis=0); wmax = np.max(abs(W), axis=0)
    smooth_s = np.sqrt(amax / wmax)
    check("SmoothQuant 平衡范围", np.allclose(amax / smooth_s, wmax * smooth_s))
    unsmoothed = symmetric_qdq(X, bits=8, axis=1)[0] @ symmetric_qdq(W, bits=8, axis=1)[0].T
    smoothed = symmetric_qdq(X/smooth_s, bits=8, axis=1)[0] @ symmetric_qdq(W*smooth_s, bits=8, axis=1)[0].T
    metrics["07_scaling"] = {"awq_best_alpha": float(alphas[ai]), "awq_rtn_mse": awq_errors[0],
                              "awq_best_mse": awq_errors[ai], "w8a8_before_mse": mse(base, unsmoothed),
                              "w8a8_after_smoothing_mse": mse(base, smoothed)}
    small_a, small_b = .1, .01
    scale_at_zero = small_a**0 / small_b**1
    scale_at_one = small_a**1 / small_b**0
    activation_at_zero = small_a / scale_at_zero
    activation_at_one = small_a / scale_at_one
    check("SmoothQuant 绝对幅度非单调减小反例", activation_at_one > activation_at_zero)
    channel_a, channel_b = np.array([.1, .2, 10.]), np.array([.01, .5, 1.])
    channel_scales = [channel_a**alpha / channel_b**(1-alpha) for alpha in (0., .5, 1.)]
    channel_spreads = [np.ptp(np.log(channel_a / s)) for s in channel_scales]
    check("SmoothQuant 激活跨通道范围趋于均衡", np.all(np.diff(channel_spreads) < 0))
    metrics["07_scaling"].update(small_activation_alpha0=activation_at_zero,
                                 small_activation_alpha1=activation_at_one)

    # 8. Weighted centroid, NF4 vs uniform: distribution-dependent observation.
    values = np.array([0., 1.]); importance = np.array([9., 1.])
    centroid = float(np.sum(values * importance) / importance.sum())
    check("加权码本中心", np.isclose(centroid, .1))
    gaussian_w = rng.normal(size=(128, 64))
    uniform16 = np.linspace(-1., 1., 16)
    check("码本代表值数量及零值约定", len(NF4) == 16 and 0. in NF4 and 0. not in uniform16)
    metrics["08_codebooks"] = {"weighted_center": centroid,
        "gaussian_nf4_mse": mse(gaussian_w, codebook_qdq(gaussian_w, NF4)),
        "gaussian_symmetric_int4_15level_mse": mse(gaussian_w, symmetric_qdq(gaussian_w, axis=1)[0]),
        "gaussian_uniform16_absmax_mse": mse(gaussian_w, codebook_qdq(gaussian_w, uniform16)),
        "gaussian_affine_int4_16level_mse": mse(gaussian_w, affine_qdq(gaussian_w, bits=4, axis=1)[0])}

    # 9. Truncated SVD compensation and its exact discarded-energy identity.
    matrix = rng.normal(size=(12, 8))
    matrix_hat = group_qdq(matrix, group_size=4)
    residual = matrix - matrix_hat
    U, S, Vt = np.linalg.svd(residual, full_matrices=False)
    rank = 2
    correction = (U[:, :rank] * S[:rank]) @ Vt[:rank]
    err_after = float(np.sum((residual-correction)**2))
    check("低秩补偿的剩余奇异值能量", np.isclose(err_after, np.sum(S[rank:]**2)))
    metrics["09_loftq_principle"] = {"error_before": float(np.sum(residual**2)), "error_after_rank2": err_after}

    # 10. Softmax translation, key centering and key/value first-order error.
    Q = rng.normal(size=(3, 8)); K = rng.normal(size=(17, 8)); V = rng.normal(size=(17, 5))
    logits = Q @ K.T / np.sqrt(8)
    p = softmax(logits)
    centered_p = softmax(Q @ (K-K.mean(axis=0)).T / np.sqrt(8))
    check("Key 去中心不改变精确注意力", np.allclose(p, centered_p))
    ek = rng.normal(size=K.shape) * 1e-5; ev = rng.normal(size=V.shape) * 1e-5
    dz = Q @ ek.T / np.sqrt(8)
    dp = np.stack([(np.diag(row)-np.outer(row, row)) @ change for row, change in zip(p, dz)])
    predicted = dp @ V + p @ ev
    actual = softmax(Q@(K+ek).T/np.sqrt(8))@(V+ev) - p@V
    rel = float(np.linalg.norm(predicted-actual) / np.linalg.norm(actual))
    check("KV 一阶误差传播", rel < 1e-3)
    metrics["10_attention"] = {"first_order_relative_error": rel,
                                "key_centering_error": float(np.max(abs(p-centered_p)))}

    # 11. One-bit residual inner-product estimator. G independent of q and r.
    residual = rng.normal(size=8); query = rng.normal(size=8)
    G = rng.normal(size=(120_000, 8))
    estimates = np.linalg.norm(residual) * np.sqrt(np.pi/2) * np.sign(G@residual) * (G@query)
    target = float(query @ residual)
    mean = float(estimates.mean())
    standard_error = float(estimates.std(ddof=1) / np.sqrt(len(estimates)))
    check("QJL 内积无偏性的蒙特卡洛核对", abs(mean-target) < 6*standard_error)
    metrics["11_qjl"] = {"target_inner_product": target, "sample_mean": mean,
                          "standard_error": standard_error, "samples": len(estimates)}

    # 12. HQQ p=1 subproblem: exact soft threshold and zero update stationarity.
    residual = np.array([-.3, -.02, 0., .06, .5]); beta = 10.
    shrunk = np.sign(residual) * np.maximum(abs(residual)-1/beta, 0.)
    grid_e = np.linspace(-1, 1, 20001)
    brute_e = np.array([grid_e[np.argmin(abs(grid_e)+beta/2*(ri-grid_e)**2)] for ri in residual])
    check("p=1 收缩子问题", np.allclose(shrunk, brute_e, atol=1e-4))
    ww = np.array([-.2, .1, .45]); ss = .2; qq = np.array([0, 2, 3]); ee = np.array([-.02, 0, .01])
    zz = float(np.mean(qq-(ww-ee)/ss))
    check("HQQ offset 二次子问题驻点", abs(np.sum(ww-ss*(qq-zz)-ee)) < 1e-12)
    metrics["12_hqq_subproblem"] = {"shrunk_residual": shrunk.tolist(), "optimal_offset": zz}

    # 13. Storage arithmetic (payload and metadata are distinct).
    kv_bytes = 2 * 1 * 32 * 8192 * 8 * 128 * 16 // 8
    check("GQA KV 缓存例子", kv_bytes == 2**30)
    metrics["13_storage"] = {"kv_fp16_gib": kv_bytes/2**30, "kv_int4_payload_gib": kv_bytes/4/2**30,
        "int4_group128_fp16_scale_bpw": 4+16/128, "nf4_double_scale_overhead_bpw": 8/64+32/(64*256),
        "mxfp4_bpv": 4+8/32, "nvfp4_block_bpv": 4+8/16, "ternary_ideal_bits": float(np.log2(3)),
        "amdahl_60pct_4x": 1/(.4+.6/4)}

    # 14. LSQ STE derivative: check against surrogate, not actual round derivative.
    ww = .26; ss = .1; qs = np.rint(ww/ss); v0 = ww/ss
    surrogate = lambda st: st * (qs + ww/st-v0)
    eps = 1e-6
    numerical = (surrogate(ss+eps)-surrogate(ss-eps))/(2*eps)
    check("LSQ 的 STE 替代梯度", np.isclose(numerical, qs-v0))
    metrics["14_lsq"] = {"ste_scale_gradient": float(qs-v0), "surrogate_finite_difference": float(numerical)}

    return metrics, checks, (weights, bounds, errors, alphas, np.asarray(awq_errors))


def write_outputs(metrics, checks, plot_data, figures=False, output_dir=None):
    root = Path(output_dir).resolve() if output_dir is not None else PUBLISHED_RESULTS
    figure_dir = ROOT / "figures" if root == PUBLISHED_RESULTS else root / "figures"
    payload = {**run_metadata({"seed": 20261010, "figures": figures}),
               "seed": 20261010, "numpy": np.__version__,
               "scope": "synthetic CPU numerical demonstrations, not LLM benchmarks",
               "passed_checks": checks, "metrics": metrics}
    command = "python experiments/quant_lab.py --write-results" + (" --figures" if figures else "")
    lines = ["# 数值实验记录", "", f"运行命令：`{command}`。", "",
             f"固定随机种子：20261010；Python {platform.python_version()}；NumPy {np.__version__}。",
             f"共运行 {len(metrics)} 组数值演示，{len(checks)} 项数学与边界检查全部通过。", "",
             "这是 CPU 小矩阵实验，没有加载真实 LLM，也没有测量 GPU 加速或任务准确率。",
             "AWQ、SmoothQuant、NF4 的误差比较只对本次构造数据成立；并非完整论文算法复现。", "",
             "## 关键结果", "", "| 项目 | 数值 |", "|---|---:|"]
    for section, values in metrics.items():
        for key, value in values.items():
            if isinstance(value, (float, int)):
                lines.append(f"| {section} / {key} | {value:.9g} |")
    lines.extend(["", "完整数组和检查名称见 [results.json](results.json)。", "",
                  "## NF4 对照条件", "",
                  "四种配置使用同一份高斯数组、每组 64 个元素。NF4 有 16 个代表值且包含零；",
                  "对称 INT4 使用 [-7,7]，有 15 个代表值；uniform16 在相同 absmax 范围内均匀放置 16 个点，但不包含零；",
                  "仿射 INT4 使用 16 个编码、包含零的 min/max 范围，并需要额外 zero point 元数据。",
                  "这里比较的是重建误差，不是相同总存储预算或真实模型质量。", "", "## 已通过的检查", ""])
    lines.extend(f"- {name}" for name in checks)
    if figures:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figure_dir.mkdir(parents=True, exist_ok=True)
        weights, bounds, errors, alphas, awq_errors = plot_data
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
        xx = np.linspace(-1.1, 1.1, 1000)
        axes[0].plot(xx, xx, color="0.65", linestyle="--", label="Identity")
        axes[0].step(xx, symmetric_qdq(xx, bits=4, bound=1.)[0], where="mid", label="INT4 QDQ")
        axes[0].set(xlabel="Original value", ylabel="Reconstructed value", title="Uniform quantization staircase")
        axes[0].legend()
        axes[1].plot(bounds, errors)
        i = int(np.argmin(errors)); axes[1].scatter(bounds[i], errors[i], color="tab:red")
        axes[1].set(xlabel="Clipping bound", ylabel="Weight MSE", title="Clipping vs rounding error")
        for ax in axes: ax.grid(alpha=.2)
        for ext in ("svg", "png"): fig.savefig(figure_dir/f"uniform_and_clipping.{ext}", dpi=160)
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
        axes[0].plot(alphas, awq_errors, marker="o")
        axes[0].set(xlabel="Scaling exponent alpha", ylabel="Layer-output MSE", title="AWQ principle on synthetic inputs")
        x = np.array([12., 0, 0, 0]); xr = x @ hadamard(4)
        axes[1].bar(np.arange(4)-.18, x, width=.36, label="Before")
        axes[1].bar(np.arange(4)+.18, xr, width=.36, label="After Hadamard")
        axes[1].set(xlabel="Coordinate", ylabel="Value", title="Same norm, different coordinate peaks")
        axes[1].legend()
        for ax in axes: ax.grid(alpha=.2)
        for ext in ("svg", "png"): fig.savefig(figure_dir/f"scaling_and_rotation.{ext}", dpi=160)
        plt.close(fig)
    if all((figure_dir / f"{name}.png").exists() for name in ("uniform_and_clipping", "scaling_and_rotation")):
        lines.extend(["", "## 图表", "", "!" + file_link("量化阶梯与裁剪", figure_dir / "uniform_and_clipping.png", root), "",
                      "!" + file_link("通道缩放与旋转", figure_dir / "scaling_and_rotation.png", root), "",
                      "同目录提供 SVG 矢量版本，可以缩放或导出。坐标使用英文以避免中文字体依赖。"])
    lines.extend(["", "运行时间、参数与完整源码指纹见 results.json。",
                  "图表可能来自上次导出；修改实验后请同时使用 --figures 刷新。" if not figures else "图表已随本次结果重新导出。", ""])
    write_json(root / "results.json", payload)
    write_text(root / "实验结果.md", "\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", action="store_true", help="Also export PNG/SVG figures using Matplotlib")
    parser.add_argument("--write-results", action="store_true", help="Explicitly publish results; default is read-only")
    parser.add_argument("--output-dir", type=Path, help="Result directory (requires --write-results)")
    args = parser.parse_args()
    if (args.figures or args.output_dir is not None) and not args.write_results:
        parser.error("--figures and --output-dir require --write-results")
    metrics, checks, plot_data = run()
    if args.write_results:
        write_outputs(metrics, checks, plot_data, args.figures, args.output_dir)
    print(f"PASS: {len(metrics)} experiments, {len(checks)} numerical checks.")
    for section in ("04_gptq", "07_scaling", "10_attention", "11_qjl"):
        print(section, json.dumps(metrics[section], ensure_ascii=False))
    if args.write_results:
        print(f"Written results to: {args.output_dir or PUBLISHED_RESULTS}")


if __name__ == "__main__":
    main()
