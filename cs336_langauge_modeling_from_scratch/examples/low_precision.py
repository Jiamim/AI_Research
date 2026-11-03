"""Block-scaled low precision: MXFP4 (OCP microscaling) and DeepSeek-V3-style FP8 tiles.

* mxfp4_quantize follows Algorithm 1 of the MX paper (arXiv 2310.10537): blocks of 32
  values share a power-of-two scale X = 2^(floor(log2 max|v|) - emax_elem); elements
  are rounded to the nearest FP4 E2M1 value {0, 0.5, 1, 1.5, 2, 3, 4, 6} with clamping.
  Storage: 4 bits per element + 8 bits per 32 elements = 4.25 bits, the figure the
  gpt-oss card quotes for its MoE weights.
* fp8_tile_quantize uses torch.float8_e4m3fn with one FP32 scale per 1x128 activation
  tile or per 128x128 weight block (the DeepSeek-V3 recipe).

Everything is dequantized back to float to measure error; no packed kernels.
"""
import json
import operator
import torch

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
E2M1_EMAX = 2          # largest normal 6 = 1.5 * 2^2
E4M3_MAX = 448.0


def _finite_float(values):
    if not values.is_floating_point() or not values.numel() or not torch.isfinite(values).all():
        raise ValueError("nonempty finite floating-point values required")


def _fp32_blocks(values, block):
    _finite_float(values)
    try:
        block = operator.index(block)
    except TypeError as error:
        raise ValueError("positive integer block size required") from error
    if values.ndim < 1 or block < 1 or values.shape[-1] % block:
        raise ValueError("last dimension must be a multiple of a positive block size")
    result = values.reshape(*values.shape[:-1], -1, block).float()
    if not torch.isfinite(result).all():
        raise ValueError("values overflow the FP32 working dtype")
    return result


def _absmax_scale(amax, limit):
    # Handle true zero blocks separately. An arbitrary floor would erase small nonzero blocks.
    smallest = torch.nextafter(amax.new_tensor(0.), amax.new_tensor(1.))
    return torch.where(amax > 0, (amax / limit).clamp_min(smallest), torch.ones_like(amax))


def round_to_e2m1(x):
    """Nearest E2M1 value for finite inputs; exact ties choose the even code, sign kept.

    Accumulates and returns FP64 for FP64 input, otherwise FP32, on the input device.
    """
    _finite_float(x)
    work = x.to(torch.float64 if x.dtype == torch.float64 else torch.float32)
    grid = E2M1.to(device=x.device, dtype=work.dtype)
    magnitude = work.abs().clamp(max=6.0)
    distance = (magnitude[..., None] - grid).abs()
    best = distance.argmin(-1)
    # Prefer the even code index on exact ties, as round-half-to-even would.
    tie = distance == distance.gather(-1, best[..., None])
    codes = torch.arange(len(grid), device=x.device)
    even_choice = torch.where(tie & (codes % 2 == 0), codes, len(grid)).min(-1).values
    best = torch.where(tie.sum(-1) > 1, even_choice, best)
    return torch.copysign(grid[best], work)


def mxfp4_quantize(values, block=32):
    """Finite-input reference: FP32 output with E8M0 scales in [2^-127, 2^127].

    Zero blocks use scale 1; reject NaN/Inf instead of modeling their packed encodings.
    """
    v = _fp32_blocks(values, block)
    amax = v.abs().amax(-1, keepdim=True)
    # frexp gives exact floor(log2(amax)); log2 can round a value just below a power of two up.
    _, exponent = torch.frexp(amax)
    shared = (exponent - 1 - E2M1_EMAX).clamp(-127, 127)
    scale = torch.where(amax > 0, torch.exp2(shared.to(v.dtype)), torch.ones_like(amax))
    dequantized = round_to_e2m1(v / scale) * scale
    return dequantized.reshape(values.shape), scale.squeeze(-1)


def fp8_tile_quantize(values, tile):
    """Per-tile absmax scaling into float8_e4m3fn; tile is (rows, cols).

    Returns FP32 reconstruction and a [number_of_row_tiles, number_of_column_tiles] scale grid.
    Zero blocks use scale 1; a nonzero scale is bounded only by FP32 representability.
    """
    _finite_float(values)
    if values.ndim != 2 or len(tile) != 2:
        raise ValueError("a matrix and a two-dimensional tile required")
    try:
        r, c = (operator.index(size) for size in tile)
    except TypeError as error:
        raise ValueError("positive integer tile dimensions required") from error
    rows, cols = values.shape
    if min(r, c) < 1 or rows % r or cols % c:
        raise ValueError("matrix shape must be a multiple of the tile shape")
    blocks = values.float().reshape(rows // r, r, cols // c, c)
    if not torch.isfinite(blocks).all():
        raise ValueError("values overflow the FP32 working dtype")
    amax = blocks.abs().amax(dim=(1, 3), keepdim=True)
    scale = _absmax_scale(amax, E4M3_MAX)
    q = (blocks / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    limit = torch.finfo(torch.float32).max
    output = (q.float() * scale).clamp(-limit, limit)
    return output.reshape(rows, cols), scale.squeeze(3).squeeze(1)


def relative_error(approx, exact):
    """Relative L2 error; exact zero reconstructed as zero scores 0, otherwise infinity."""
    _finite_float(approx)
    _finite_float(exact)
    if approx.shape != exact.shape or approx.device != exact.device:
        raise ValueError("matching shapes and devices required")
    approx, exact = approx.double(), exact.double()
    scale = torch.maximum(approx.abs().amax(), exact.abs().amax())
    if scale == 0:
        return 0.0
    denominator = (exact / scale).norm()
    return float('inf') if denominator == 0 else (((approx / scale) - (exact / scale)).norm() / denominator).item()


def int4_absmax(values, block=32):
    """Symmetric INT4 with a free FP scale per block, for comparison with MXFP4's power-of-two scale."""
    v = _fp32_blocks(values, block)
    scale = _absmax_scale(v.abs().amax(-1, keepdim=True), 7)
    limit = torch.finfo(torch.float32).max
    return ((v / scale).round().clamp(-7, 7) * scale).clamp(-limit, limit).reshape(values.shape)


def demo():
    torch.manual_seed(0)
    weights = torch.randn(256, 512) * 0.02
    mx, scales = mxfp4_quantize(weights)
    tile_w, _ = fp8_tile_quantize(weights, (128, 128))
    bits = 4 + 8 / 32
    # Algorithm 1 floors the shared exponent, so the block maximum can exceed 6 and clamp.
    # Rounding the exponent up instead never clamps but wastes resolution on every block.
    blocks = weights.reshape(256, -1, 32)
    clamped = blocks.abs() / scales[..., None] > 6.0
    up = torch.exp2(torch.ceil(torch.log2(blocks.abs().amax(-1, keepdim=True) / 6.0)))
    no_clamp = (round_to_e2m1(blocks / up) * up).reshape(weights.shape)
    result = dict(
        mxfp4_bits_per_weight=bits,
        mxfp4_weight_relative_error=round(relative_error(mx, weights), 4),
        mxfp4_clamped_element_fraction=round(clamped.float().mean().item(), 4),
        mxfp4_blocks_with_clamping=round(clamped.any(-1).float().mean().item(), 4),
        mxfp4_ceil_exponent_relative_error=round(relative_error(no_clamp, weights), 4),
        int4_absmax_per_32_relative_error=round(relative_error(int4_absmax(weights), weights), 4),
        mxfp4_scales_are_powers_of_two=bool(torch.all(torch.log2(scales) == torch.round(torch.log2(scales)))),
        fp8_128x128_weight_relative_error=round(relative_error(tile_w, weights), 4),
        # Element counts from the gpt-oss-120b safetensors metadata (sources/model-configs/index.json):
        # the U8 entry is the 114.66B MXFP4 expert weights, the rest is BF16. The card reports 60.8 GiB.
        gpt_oss_120b_checkpoint_gib=round((114_661_785_600 * bits / 8 + 2_167_371_072 * 2) / 2**30, 1))
    # One activation channel far larger than the rest: compare a single scale with 1x128 tiles,
    # measuring error on the ordinary channels, which is where a shared scale does damage.
    normal = torch.ones(512, dtype=torch.bool)
    normal[7] = False
    for magnitude in (200.0, 1e5):
        activations = torch.randn(256, 512)
        activations[:, 7] *= magnitude
        one, _ = fp8_tile_quantize(activations, (256, 512))
        tiles, _ = fp8_tile_quantize(activations, (1, 128))
        result[f"outlier_x{magnitude:g}"] = dict(
            normal_channel_error_one_scale=round(relative_error(one[:, normal], activations[:, normal]), 4),
            normal_channel_error_1x128=round(relative_error(tiles[:, normal], activations[:, normal]), 4),
            flushed_to_zero_one_scale=round(((one[:, normal] == 0) & (activations[:, normal] != 0)).float().mean().item(), 4))
    return result


if __name__ == "__main__":
    print(json.dumps(demo(), indent=2))
