"""Long-context mechanics: RoPE scaling (PI, NTK-aware, YaRN), attention sinks, sliding windows.

* yarn_inverse_frequencies reproduces the frequency blend used by DeepSeek-V3,
  gpt-oss, Kimi K2 and Qwen long-context variants; tests/test_frontier.py compares it
  with transformers on the pinned configs (gpt-oss needs truncate=False, as its config says).
* yarn_attention_factor is the factor applied to both q and k (cos and sin are scaled),
  so attention logits are multiplied by its square.
* rotation_wavelengths shows which RoPE dimensions complete a full turn inside
  the training context: those are left alone, the slow ones are interpolated.
* sink_attention adds one learned logit per head to the softmax denominator, as in
  gpt-oss, so a head can put (almost) no weight on any real token.
* streaming_kv_positions is the StreamingLLM cache policy: keep the first few
  "sink" tokens plus a recent window.
"""
import json
import math
import torch


def rope_inverse_frequencies(dim, base):
    if type(dim) is not int or dim < 2 or dim % 2 or not math.isfinite(base) or base <= 0:
        raise ValueError("positive even rotary dimension and finite positive base required")
    return 1.0 / base ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim)


def position_interpolation(dim, base, factor):
    """PI: divide every frequency by the extension factor (equivalently, positions by factor)."""
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("positive finite interpolation factor required")
    return rope_inverse_frequencies(dim, base) / factor


def ntk_aware(dim, base, factor):
    """NTK-aware scaling: enlarge the base so the slowest frequency stretches by `factor`."""
    if dim <= 2 or not math.isfinite(factor) or factor <= 0:
        raise ValueError("NTK-aware scaling requires dim > 2 and positive finite factor")
    return rope_inverse_frequencies(dim, base * factor ** (dim / (dim - 2)))


def yarn_correction_range(dim, base, original_context, beta_fast=32, beta_slow=1, truncate=True):
    """Dimension indices where a frequency completes beta_fast and beta_slow turns in the original context."""
    rope_inverse_frequencies(dim, base)
    if base <= 1 or not math.isfinite(original_context) or original_context <= 0 \
            or not all(math.isfinite(x) and x > 0 for x in (beta_fast, beta_slow)) or beta_fast < beta_slow:
        raise ValueError("YaRN needs base > 1, positive context and beta_fast >= beta_slow > 0")
    def dim_for(turns):
        return dim * math.log(original_context / (turns * 2 * math.pi)) / (2 * math.log(base))
    low, high = dim_for(beta_fast), dim_for(beta_slow)
    if truncate:
        low, high = math.floor(low), math.ceil(high)
    return max(low, 0), min(high, dim - 1)


def yarn_mscale(factor, mscale=1.0):
    if not math.isfinite(factor) or factor <= 0 or not math.isfinite(mscale) or mscale < 0:
        raise ValueError("finite positive factor and nonnegative mscale required")
    return 1.0 if factor <= 1 else 0.1 * mscale * math.log(factor) + 1.0


def yarn_attention_factor(factor, mscale=None, mscale_all_dim=None):
    """Temperature correction applied to q and k (logits get its square). DeepSeek-style configs give mscale and mscale_all_dim, and the
    factor becomes their ratio (1.0 when both are 1, as in DeepSeek-V3 and Kimi K2, which instead
    fold the correction into the softmax scale of MLA)."""
    if mscale is not None and mscale_all_dim is not None:
        return yarn_mscale(factor, mscale) / yarn_mscale(factor, mscale_all_dim)
    return yarn_mscale(factor)


def yarn_inverse_frequencies(dim, base, factor, original_context, beta_fast=32, beta_slow=1, truncate=True,
                             mscale=None, mscale_all_dim=None):
    """Fast-rotating dims keep their frequency, slow dims are interpolated by `factor`, a linear ramp between.

    Returns (inverse frequencies, q/k temperature factor). Pass the config's mscale/mscale_all_dim
    for DeepSeek-style models: their factor is a ratio (1.0 for DeepSeek-V3 and Kimi K2).
    """
    extrapolation = rope_inverse_frequencies(dim, base)
    interpolation = extrapolation / factor
    low, high = yarn_correction_range(dim, base, original_context, beta_fast, beta_slow, truncate)
    if low == high:
        high += 0.001
    ramp = ((torch.arange(dim // 2, dtype=torch.float64) - low) / (high - low)).clamp(0, 1)
    keep = 1 - ramp
    return interpolation * (1 - keep) + extrapolation * keep, yarn_attention_factor(factor, mscale, mscale_all_dim)


def rotation_wavelengths(dim, base):
    """Tokens needed for each RoPE pair to complete one full rotation."""
    return 2 * math.pi / rope_inverse_frequencies(dim, base)


def sink_attention(q, k, v, sink_logits, causal=True):
    """Softmax over [scores, sink]; the sink column is dropped after normalization.

    q, k, v: [heads, T, d]; sink_logits: [heads]. Returns output and the attention mass on real tokens.
    """
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    if causal:
        t = q.shape[-2]
        if k.shape[-2] != t:
            raise ValueError("causal sink reference requires equal query/key lengths")
        scores = scores.masked_fill(torch.ones(t, t, dtype=torch.bool, device=q.device).triu(1), -torch.inf)
    sink = sink_logits[:, None, None].expand(-1, scores.shape[1], 1)
    probs = torch.cat([scores, sink], dim=-1).softmax(-1)[..., :-1]
    return probs @ v, probs.sum(-1)


def streaming_kv_positions(length, sinks=4, window=1024):
    """Positions kept by a StreamingLLM-style cache after `length` tokens."""
    if any(type(value) is not int or value < 0 for value in (length, sinks, window)):
        raise ValueError("nonnegative integer length, sink count and window required")
    if length <= sinks + window:
        return list(range(length))
    return list(range(sinks)) + list(range(length - window, length))


def demo():
    dim, base, original, factor = 128, 10000.0, 4096, 32  # e.g. a 4K-pretrained model extended to 128K
    wavelengths = rotation_wavelengths(dim, base)
    low, high = yarn_correction_range(dim, base, original)
    inv_yarn, mscale = yarn_inverse_frequencies(dim, base, factor, original)
    inv_rope = rope_inverse_frequencies(dim, base)
    ratio = (inv_rope / inv_yarn).tolist()
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 6, 8) for _ in range(3))
    _, mass_open = sink_attention(q, k, v, torch.tensor([-30.0, -30.0]))
    _, mass_sink = sink_attention(q, k, v, torch.tensor([8.0, 8.0]))
    return dict(
        setting=dict(head_dim=dim, base=base, original_context=original, factor=factor),
        shortest_and_longest_wavelength=[round(wavelengths[0].item(), 2), round(wavelengths[-1].item(), 1)],
        pairs_completing_a_turn_in_original_context=int((wavelengths < original).sum()),
        yarn_ramp_dims=[low, high], yarn_attention_scale=round(mscale, 4),
        frequency_divisor_first_mid_last=[round(ratio[0], 3), round(ratio[len(ratio) // 2], 3), round(ratio[-1], 3)],
        sink_mass_on_tokens=dict(sink_logit_minus30=mass_open.mean().item(), sink_logit_8=round(mass_sink.mean().item(), 4)),
        streaming_cache_size_at_100k=len(streaming_kv_positions(100_000)))


if __name__ == "__main__":
    print(json.dumps(demo(), indent=2))
