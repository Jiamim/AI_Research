"""Small reusable numerical kernels for the executable tutorial cards.

All arrays are CPU NumPy arrays. QDQ is simulation, not a hardware kernel.
The finite floating-point LUT rounds ties toward the first sorted value;
it is not a bit-exact implementation of hardware round-to-nearest-even.
"""

from __future__ import annotations

import numpy as np
from experiments.validation import finite_array, integer

from experiments.quant_lab import (
    NF4, affine_qdq, codebook_qdq, group_qdq, hadamard, mse,
    sequential_compensation, softmax, symmetric_qdq,
)


def nearest(x, levels):
    x = finite_array(x, "x")
    levels = finite_array(levels, "levels", ndim=1)
    try:
        with np.errstate(over='raise', invalid='raise'):
            indices = np.argmin(abs(x[..., None] - levels), axis=-1)
    except FloatingPointError as exc:
        raise ValueError("Nearest-code distance exceeds float64 range; rescale the input.") from exc
    return levels[indices]


def lloyd(values, count, importance=None, iterations=30):
    """Weighted scalar Lloyd iterations; empty clusters retain their center."""
    values = finite_array(values, "values").ravel()
    count = integer(count, "count")
    iterations = integer(iterations, "iterations")
    h = np.ones_like(values) if importance is None else finite_array(importance, "importance").ravel()
    if h.shape != values.shape or np.any(h <= 0):
        raise ValueError("Use positive weights and a positive number of centers.")
    centers = np.linspace(values.min(), values.max(), count)
    losses = []
    for _ in range(iterations):
        assignment = np.argmin(abs(values[:, None] - centers), axis=1)
        for j in range(count):
            mask = assignment == j
            if mask.any():
                centers[j] = np.average(values[mask], weights=h[mask])
        reconstructed = nearest(values, centers)
        losses.append(float(np.sum(h * (values - reconstructed)**2)))
    return np.sort(centers), losses


def ldl_feedback(w, h, levels):
    """Scalar LDLQ principle, reverse order for H = L D L.T."""
    c = np.linalg.cholesky(h)
    l = c / np.diag(c)[None, :]
    result = np.array(w, dtype=float, copy=True)
    for i in range(len(w) - 1, -1, -1):
        feedback = l[i+1:, i] @ (result[i+1:] - w[i+1:])
        result[i] = nearest(w[i] - feedback, levels)
    return result, l, np.diag(c)**2


def nearest_even_sum(x):
    """Nearest point in D_n = {integer coordinates with even sum}."""
    x = np.asarray(x, dtype=float)
    z = np.rint(x)
    if int(z.sum()) % 2:
        direction = np.where(x >= z, 1., -1.)
        extra_cost = (z + direction - x)**2 - (z - x)**2
        j = int(np.argmin(extra_cost))
        z[j] += direction[j]
    return z


def nearest_e8(x):
    """Infinite E8 lattice geometry only; not the finite E8P codec of QuIP#."""
    x = np.asarray(x, dtype=float)
    if x.shape != (8,):
        raise ValueError("E8 requires one eight-dimensional vector.")
    integer = nearest_even_sum(x)
    half = nearest_even_sum(x - .5) + .5
    return min((integer, half), key=lambda candidate: np.sum((candidate-x)**2))


def fp_levels(kind):
    """Finite values of E4M3FN, E5M2, or E2M1, excluding NaN and infinity."""
    ebits, mbits, bias = {"e4m3": (4, 3, 7), "e5m2": (5, 2, 15), "e2m1": (2, 1, 1)}[kind]
    values = []
    for exponent in range(2**ebits):
        for mantissa in range(2**mbits):
            if kind == "e5m2" and exponent == 31:
                continue
            if kind == "e4m3" and exponent == 15 and mantissa == 7:
                continue
            if exponent == 0:
                value = 2.**(1-bias) * mantissa / 2**mbits
            else:
                value = 2.**(exponent-bias) * (1 + mantissa / 2**mbits)
            values.extend([value, -value])
    return np.unique(values)


def float_qdq(x, kind="e4m3", axis=None):
    x = finite_array(x, "x")
    levels = fp_levels(kind)
    maximum = np.max(abs(x), axis=axis, keepdims=True)
    try:
        with np.errstate(over='raise', divide='raise', invalid='raise', under='ignore'):
            scale = np.where(maximum > 0, maximum / levels.max(), 1.)
            if np.any(scale <= 0):
                raise ValueError("Floating-point scale underflows float64; rescale the input.")
            return scale * nearest(x/scale, levels)
    except FloatingPointError as exc:
        raise ValueError("Floating-point simulation exceeds float64 range; rescale the input.") from exc


def micro_fp4(x, kind="nv", tensor_scale=None):
    """E2M1 with group-16 E4M3 scales or group-32 power-of-two scales.

    Uses conservative ceil(log2) MX scales, not a vendor conversion recipe.
    Returns float64 reconstructions and explicit numerical scale metadata.
    """
    x = finite_array(x, "x")
    if x.ndim == 0:
        raise ValueError("Microblocks require at least one input dimension.")
    if kind not in ("nv", "mx"):
        raise ValueError("Use kind='nv' or kind='mx'.")
    if tensor_scale is not None:
        scale_value = finite_array(tensor_scale, "tensor_scale")
        if scale_value.ndim != 0 or scale_value <= 0:
            raise ValueError("The tensor scale must be a positive finite scalar.")
        if kind == 'mx':
            raise ValueError("MX blocks do not use an additional tensor scale in this demo.")
        tensor_scale = scale_value.item()
    try:
        with np.errstate(over='raise', divide='raise', invalid='raise', under='ignore'):
            return _micro_fp4(x, kind, tensor_scale)
    except FloatingPointError as exc:
        raise ValueError("FP4 simulation exceeds float64 range; rescale the input.") from exc


def _micro_fp4(x, kind, tensor_scale):
    """Numerical body after validating inputs; all scale arithmetic is guarded."""
    group = {"nv": 16, "mx": 32}[kind]
    if x.shape[-1] % group:
        raise ValueError("The final dimension must contain full microblocks.")
    blocks = x.reshape(*x.shape[:-1], -1, group)
    amax = np.max(abs(blocks), axis=-1, keepdims=True)
    if kind == "mx":
        safe = np.where(amax > 0, amax/6, 1.)
        scale = 2.**np.clip(np.ceil(np.log2(safe)), -127, 127)
        global_scale = 1.
    else:
        maximum = np.max(amax)
        global_scale = (maximum / (448*6) if maximum > 0 else 1.) if tensor_scale is None else tensor_scale
        if not np.isfinite(global_scale) or global_scale <= 0:
            raise ValueError("The tensor scale must be positive and finite.")
        divisor = 6*global_scale
        # Python float乘法不会受np.errstate保护，必须单独检查。
        if not np.isfinite(divisor) or divisor <= 0:
            raise ValueError('FP4 scale arithmetic exceeds float64 range; rescale the input.')
        ideal = amax / divisor
        scale = nearest(ideal, fp_levels("e4m3"))
        # Nonzero blocks whose scale underflows use the smallest subnormal.
        scale = np.where(amax == 0, 1., np.maximum(scale, 2.**-9))
    payload = nearest(blocks/(global_scale*scale), fp_levels("e2m1"))
    result = (global_scale*scale*payload).reshape(x.shape)
    return result, {"tensor_scale": global_scale, "block_scale": scale, "payload": payload}


def rope_rows(x, positions):
    """Toy RoPE with adjacent coordinate pairs and the standard base 10000."""
    x = np.asarray(x, dtype=float)
    pairs = x.reshape(len(x), -1, 2)
    theta = np.asarray(positions)[:, None] / 10000.**(np.arange(pairs.shape[1])*2/x.shape[1])
    c, s = np.cos(theta), np.sin(theta)
    a, b = pairs[..., 0], pairs[..., 1]
    return np.stack([a*c-b*s, a*s+b*c], axis=-1).reshape(x.shape)


class TinyKIVI:
    """One-head streaming K/V layout demonstration, without bit packing.

    Keys flush their whole residual buffer at R; values evict one old row
    when the residual exceeds R. Quantized codes use uint8 containers.
    """

    def __init__(self, dim=4, group=2, residual=4, bits=2):
        dim = integer(dim, "dim")
        group = integer(group, "group")
        residual = integer(residual, "residual")
        bits = integer(bits, "bits", 1, 8)
        if residual % group or dim % group:
            raise ValueError("Use full groups and uint8-compatible bit widths.")
        self.dim, self.group, self.residual, self.bits = dim, group, residual, bits
        self.k_recent, self.v_recent = [], []
        self.k_chunks, self.v_chunks = [], []

    def encode(self, x, axis):
        _, q, scale, zero = affine_qdq(x, bits=self.bits, axis=axis)
        return q.astype(np.uint8), scale, zero

    @staticmethod
    def decode(chunk):
        q, scale, zero = chunk
        return scale * (q.astype(float) - zero)

    def append(self, k, v):
        k, v = finite_array(k, "k", ndim=1), finite_array(v, "v", ndim=1)
        if k.shape != (self.dim,) or v.shape != (self.dim,):
            raise ValueError("One key and one value of shape (dim,) are required.")
        # Stage every encoding before committing either stream. A failed key
        # or value conversion must not leave different K/V sequence lengths.
        k_recent = [*self.k_recent, k.copy()]
        v_recent = [*self.v_recent, v.copy()]
        k_chunks, v_chunks = [], []
        if len(k_recent) == self.residual:
            rows = np.array(k_recent)
            for start in range(0, len(rows), self.group):
                k_chunks.append(self.encode(rows[start:start+self.group], axis=0))
            k_recent = []
        if len(v_recent) > self.residual:
            oldest = v_recent[0].reshape(-1, self.group)
            v_chunks.append(self.encode(oldest, axis=1))
            v_recent = v_recent[1:]
        self.k_chunks.extend(k_chunks)
        self.v_chunks.extend(v_chunks)
        self.k_recent, self.v_recent = k_recent, v_recent

    def reconstruct(self):
        keys = [self.decode(c) for c in self.k_chunks]
        values = [self.decode(c).reshape(1, self.dim) for c in self.v_chunks]
        if self.k_recent:
            keys.append(np.array(self.k_recent))
        if self.v_recent:
            values.append(np.array(self.v_recent))
        empty = np.empty((0, self.dim))
        return (np.concatenate(keys) if keys else empty,
                np.concatenate(values) if values else empty)


def histogram_kl(counts, threshold, bins=4):
    """Teaching KL calibration on a nonnegative histogram.

    Fold the tail into the last retained bin, merge contiguous bins into
    quantized buckets, then redistribute mass over nonempty source bins.
    This explicit convention is not a universal TensorRT implementation.
    """
    counts = finite_array(counts, "counts", ndim=1)
    threshold = integer(threshold, "threshold", 1, len(counts))
    bins = integer(bins, "bins", 1, threshold)
    if np.any(counts < 0) or not np.any(counts > 0):
        raise ValueError("Histogram counts must be nonnegative with positive total mass.")
    # KL is invariant to a common positive scale; avoid overflowing count sums.
    counts = counts / counts.max()
    p = counts[:threshold].copy()
    p[-1] += np.sum(counts[threshold:])
    q = np.zeros_like(p)
    for ids in np.array_split(np.arange(threshold), bins):
        active = ids[p[ids] > 0]
        if len(active):
            q[active] = p[ids].sum()/len(active)
    p /= p.sum()
    q /= q.sum()
    mask = p > 0
    return float(np.sum(p[mask]*np.log(p[mask]/q[mask])))
