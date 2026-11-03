"""Closed-form resource accounting for decoder-only Transformers.

Worked on Llama 2 shapes (the 2026 A1 accounting problems use GPT-2 shapes and
are left to the student): parameter and FLOP breakdowns, mixed-precision model
state, activation memory after Korthikanti et al. (2022), training time against
the GPU-hours Llama 2 reports, the course's roofline numbers, and a
bandwidth-bound decode model in the style of the 2026 inference lecture.
Every function returns plain numbers so chapter tables can be regenerated.
"""
from dataclasses import dataclass
import json
import math

# Dense peak throughput (FLOP/s) and HBM bandwidth (bytes/s) quoted in the course materials.
HARDWARE = {
    "A100": dict(bf16=312e12, memory=80e9),
    "H100": dict(bf16=989e12, tf32=495e12, hbm=3.35e12, memory=80e9),
    "B200": dict(bf16=2.5e15, fp32=80e12, memory=192e9),
}


@dataclass(frozen=True)
class Shape:
    vocab: int
    context: int
    layers: int
    d_model: int
    heads: int
    d_ff: int
    kv_heads: int = None
    tied: bool = False

    def __post_init__(self):
        for value in (self.vocab, self.context, self.layers, self.d_model, self.heads, self.d_ff):
            if type(value) is not int or value < 1:
                raise ValueError("positive integer model dimensions required")
        kv = self.heads if self.kv_heads is None else self.kv_heads
        if type(kv) is not int or kv < 1 or self.d_model % self.heads or self.heads % kv:
            raise ValueError("width must divide into heads and query heads into positive KV groups")

    @property
    def d_kv(self):
        return (self.heads if self.kv_heads is None else self.kv_heads) * (self.d_model // self.heads)


def swiglu_width(d_model, multiple=64):
    """8/3 * d_model rounded to the nearest multiple (A1 convention), never below one multiple."""
    if type(d_model) is not int or type(multiple) is not int or min(d_model, multiple) < 1:
        raise ValueError("positive integer width and alignment required")
    return multiple * max(1, round(8 * d_model / 3 / multiple))


def llama2(size, context=4096):
    """Llama 2 shapes (Touvron et al., 2023): SwiGLU, RMSNorm, untied embeddings; 70B uses 8 KV heads."""
    layers, d, heads, d_ff, kv = {"7b": (32, 4096, 32, 11008, None), "13b": (40, 5120, 40, 13824, None),
                                  "70b": (80, 8192, 64, 28672, 8)}[size]
    return Shape(vocab=32000, context=context, layers=layers, d_model=d, heads=heads, d_ff=d_ff, kv_heads=kv)


def parameters(s):
    attention = 2 * s.d_model * s.d_model + 2 * s.d_model * s.d_kv
    per_layer = attention + 3 * s.d_model * s.d_ff + 2 * s.d_model
    embeddings = s.vocab * s.d_model * (1 if s.tied else 2)
    return dict(embeddings=embeddings, per_layer=per_layer, total=embeddings + s.layers * per_layer + s.d_model)


def forward_flops(s, tokens=None):
    """Matmul FLOPs for one forward pass over `tokens` positions (default: full context)."""
    t = s.context if tokens is None else tokens
    if type(t) is not int or t < 0:
        raise ValueError("nonnegative integer forward token count required")
    parts = dict(
        qkvo_projections=s.layers * 2 * t * (2 * s.d_model * s.d_model + 2 * s.d_model * s.d_kv),
        attention_scores=s.layers * 2 * t * t * s.d_model,
        attention_values=s.layers * 2 * t * t * s.d_model,
        feed_forward=s.layers * 2 * t * 3 * s.d_model * s.d_ff,
        lm_head=2 * t * s.d_model * s.vocab,
    )
    parts["total"] = sum(parts.values())
    return parts


def model_state_bytes(s, bytes_per_param=16):
    """AdamW in mixed precision: BF16 weights and gradients, FP32 master weights and two moments."""
    if not math.isfinite(bytes_per_param) or bytes_per_param <= 0:
        raise ValueError("positive finite bytes per parameter required")
    return bytes_per_param * parameters(s)["total"]


def activation_bytes(s, batch, recompute="none"):
    """Stored activations for one step in 16-bit, from Korthikanti et al. (2022), eq. 1:
    s*b*h*(34 + 5*a*s/h) bytes per layer without model parallelism.

    "selective" drops the 5as/h attention term (recomputed, or never stored with FlashAttention);
    "full" counts only saved layer boundaries (2sbh), not backward's live layer or peak memory.
    "selective" is a historical estimate, not an exact FlashAttention allocation model.
    The constants were derived for a GPT block with a
    4h GeLU MLP and dropout, so for SwiGLU blocks treat the result as an order-of-magnitude estimate.
    """
    t, d, a = s.context, s.d_model, s.heads
    if type(batch) is not int or batch < 1 or recompute not in ('none', 'selective', 'full'):
        raise ValueError("positive integer batch and a supported recomputation mode required")
    per_layer = {"none": t * batch * d * (34 + 5 * a * t / d), "selective": 34 * t * batch * d,
                 "full": 2 * t * batch * d}[recompute]
    return s.layers * per_layer


def training_flops(s, tokens):
    """Forward matmul FLOPs x 3 (backward = 2 x forward) over `tokens`, in full-context sequences."""
    if not math.isfinite(tokens) or tokens < 0:
        raise ValueError("finite nonnegative training token count required")
    return 3 * forward_flops(s)["total"] * tokens / s.context


def training_days(s, tokens, peak, mfu):
    if not math.isfinite(peak) or peak <= 0 or not math.isfinite(mfu) or not 0 < mfu <= 1:
        raise ValueError("positive finite peak and MFU in (0,1] required")
    return training_flops(s, tokens) / (peak * mfu) / 86400


def mfu(model_flops_per_step, step_seconds, devices, peak):
    if any(not math.isfinite(x) or x <= 0 for x in (step_seconds, devices, peak)) \
            or not math.isfinite(model_flops_per_step) or model_flops_per_step < 0:
        raise ValueError("finite nonnegative FLOPs and positive time/device/peak values required")
    return model_flops_per_step / (step_seconds * devices * peak)


def ridge_point(device="H100", dtype="bf16"):
    hw = HARDWARE[device]
    if dtype not in hw or "hbm" not in hw:
        raise ValueError(f"no {dtype} peak or HBM bandwidth recorded for {device}; only course-quoted numbers are kept")
    return hw[dtype] / hw["hbm"]


def decode_step(params, kv_bytes_per_token, context, batch, bandwidth, bytes_per_param=2):
    """Lower-bound decode latency: each step reads all weights plus every request's KV cache."""
    if any(not math.isfinite(x) or x < 0 for x in (params, kv_bytes_per_token, context)) \
            or any(not math.isfinite(x) or x <= 0 for x in (batch, bandwidth, bytes_per_param)):
        raise ValueError("nonnegative finite sizes and positive batch/bandwidth/parameter precision required")
    memory = bytes_per_param * params + batch * context * kv_bytes_per_token
    if memory == 0:
        raise ValueError("decode model requires some weight or KV traffic")
    latency = memory / bandwidth
    return dict(memory_gb=memory / 1e9, latency_ms=1e3 * latency, tokens_per_second=batch / latency)


def llama2_13b(kv_heads=40):
    s = Shape(vocab=32000, context=1024, layers=40, d_model=5120, heads=40, d_ff=13824, kv_heads=kv_heads)
    return s, parameters(s)["total"], 2 * 2 * s.layers * s.d_kv  # K and V, BF16


def report():
    seven = llama2("7b")
    out = dict(llama2_7b_parameters=parameters(seven), llama2_7b_bf16_load_gb=2 * parameters(seven)["total"] / 1e9)
    shares = {}
    for name, shape in [("7b", seven), ("13b", llama2("13b")), ("70b", llama2("70b")), ("7b_32k", llama2("7b", 32768))]:
        f = forward_flops(shape)
        shares[name] = {k: round(v / f["total"], 4) for k, v in f.items() if k != "total"} | {"tflops": f["total"] / 1e12}
    out["forward_flop_shares"] = shares
    out["llama2_7b_model_state_gb"] = model_state_bytes(seven) / 1e9
    out["llama2_7b_activations_gb_per_sequence"] = {mode: activation_bytes(seven, 1, mode) / 1e9
                                                   for mode in ("none", "selective", "full")}
    # Llama 2 7B: 2.0T tokens, 184,320 A100-80GB GPU-hours (Touvron et al., 2023, Tables 1-2).
    flops = training_flops(seven, 2.0e12)
    out["llama2_7b_training"] = dict(flops=flops, six_n_d=6 * parameters(seven)["total"] * 2.0e12,
                                     implied_a100_mfu=flops / (184_320 * 3600 * HARDWARE["A100"]["bf16"]))
    out["h100_bf16_ridge_flops_per_byte"] = ridge_point()
    decode = {}
    for kv_heads, batch in [(40, 1), (40, 64), (8, 64), (8, 256), (40, 256)]:
        s, n, kv = llama2_13b(kv_heads)
        decode[f"kv{kv_heads}_b{batch}"] = decode_step(n, kv, s.context, batch, HARDWARE["H100"]["hbm"])
    out["llama2_13b_h100_decode"] = decode
    out["llama2_13b_parameters"] = llama2_13b()[1]
    return out


if __name__ == "__main__":
    print(json.dumps(report(), indent=2))
