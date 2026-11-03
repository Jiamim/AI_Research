"""A deliberately small decoder: RMSNorm, RoPE, SwiGLU, GQA and KV cache.

Config switches cover the course ablations: pre/post/no normalization, RoPE or
no position information (NoPE), SwiGLU or ungated SiLU feed-forward, and tied
input/output embeddings.

Teaching reference: it uses nn.Linear, nn.Embedding and F.silu for brevity, which A1
forbids, and coursework restrictions and this demo's scope are explained in chapter 28.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    vocab: int = 256
    width: int = 64
    layers: int = 2
    heads: int = 4
    kv_heads: int = 2
    hidden: int = 176
    rope_base: float = 10000.0
    norm: str = "pre"          # "pre", "post" or "none"
    position: str = "rope"     # "rope" or "none" (NoPE)
    ffn: str = "swiglu"        # "swiglu" or "silu"
    tie_embeddings: bool = False

    def __post_init__(self):
        if any(type(value) is not int or value < 1 for value in
               (self.vocab, self.width, self.layers, self.heads, self.kv_heads, self.hidden)):
            raise ValueError("all model dimensions must be positive integers")
        if self.width % self.heads or self.heads % self.kv_heads:
            raise ValueError("width must divide into heads; query heads must divide into KV groups")
        if (self.width // self.heads) % 2 or not math.isfinite(self.rope_base) or self.rope_base <= 0:
            raise ValueError("RoPE needs an even head dimension and a positive base")
        if self.norm not in ("pre", "post", "none") or self.position not in ("rope", "none") \
                or self.ffn not in ("swiglu", "silu"):
            raise ValueError("unknown norm, position or ffn option")


def init_linear(weight):
    """N(0, 2/(d_in+d_out)) truncated at three standard deviations."""
    std = math.sqrt(2 / sum(weight.shape))
    nn.init.trunc_normal_(weight, std=std, a=-3 * std, b=3 * std)


class RMSNorm(nn.Module):
    def __init__(self, width, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        # Preserve float64 for numerical gradient tests; accumulate low precision in float32.
        work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        scaled = work * torch.rsqrt(work.square().mean(-1, keepdim=True) + self.eps)
        return (scaled * self.weight.to(work.dtype)).to(x.dtype)


def rope(x, positions, base=10000.0):
    # x: [batch, heads, time, head_dim]; adjacent coordinate pairs.
    d = x.shape[-1]
    if d % 2 or d < 2 or not x.is_floating_point() or positions.ndim != 1 or len(positions) != x.shape[-2] \
            or not math.isfinite(base) or base <= 0:
        raise ValueError("RoPE needs an even head dimension")
    work_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    freq = base ** (-torch.arange(0, d, 2, device=x.device, dtype=work_dtype) / d)
    angles = positions.to(device=x.device, dtype=work_dtype)[:, None] * freq[None, :]
    c, s = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    a, b = x[..., 0::2], x[..., 1::2]
    return torch.stack((a * c - b * s, a * s + b * c), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg.width % cfg.heads or cfg.heads % cfg.kv_heads:
            raise ValueError("width must be divisible by heads; heads must be divisible by kv_heads")
        self.h, self.hk, self.d = cfg.heads, cfg.kv_heads, cfg.width // cfg.heads
        self.base = cfg.rope_base
        self.use_rope = cfg.position == "rope"
        self.q = nn.Linear(cfg.width, self.h * self.d, bias=False)
        self.k = nn.Linear(cfg.width, self.hk * self.d, bias=False)
        self.v = nn.Linear(cfg.width, self.hk * self.d, bias=False)
        self.out = nn.Linear(self.h * self.d, cfg.width, bias=False)

    def forward(self, x, cache=None, use_cache=False):
        b, t, _ = x.shape
        offset = 0 if cache is None else cache[0].shape[-2]
        pos = torch.arange(offset, offset + t, device=x.device)
        q = self.q(x).view(b, t, self.h, self.d).transpose(1, 2)
        k = self.k(x).view(b, t, self.hk, self.d).transpose(1, 2)
        if self.use_rope:
            q, k = rope(q, pos, self.base), rope(k, pos, self.base)
        v = self.v(x).view(b, t, self.hk, self.d).transpose(1, 2)
        if cache is not None:
            k = torch.cat((cache[0], k), dim=-2)
            v = torch.cat((cache[1], v), dim=-2)
        new_cache = (k, v) if use_cache else None
        # Materialized repeats simplify the reference; production kernels share KV directly.
        kq = k.repeat_interleave(self.h // self.hk, dim=1)
        vq = v.repeat_interleave(self.h // self.hk, dim=1)
        scores = q @ kq.transpose(-1, -2) / math.sqrt(self.d)
        allowed = torch.arange(k.shape[-2], device=x.device)[None, :] <= pos[:, None]
        scores = scores.masked_fill(~allowed, -torch.inf)
        softmax_dtype = torch.float64 if scores.dtype == torch.float64 else torch.float32
        p = scores.softmax(dim=-1, dtype=softmax_dtype).to(q.dtype)
        y = (p @ vq).transpose(1, 2).contiguous().view(b, t, self.h * self.d)
        return self.out(y), new_cache


class FeedForward(nn.Module):
    """SwiGLU: down(SiLU(gate x) * up x). Ungated SiLU: down(SiLU(up x))."""
    def __init__(self, cfg):
        super().__init__()
        self.gated = cfg.ffn == "swiglu"
        self.gate = nn.Linear(cfg.width, cfg.hidden, bias=False) if self.gated else None
        self.up = nn.Linear(cfg.width, cfg.hidden, bias=False)
        self.down = nn.Linear(cfg.hidden, cfg.width, bias=False)

    def forward(self, x):
        hidden = F.silu(self.gate(x)) * self.up(x) if self.gated else F.silu(self.up(x))
        return self.down(hidden)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mode = cfg.norm
        make_norm = (lambda: RMSNorm(cfg.width)) if cfg.norm != "none" else nn.Identity
        self.norm1, self.norm2 = make_norm(), make_norm()
        self.attn = Attention(cfg)
        self.ffn = FeedForward(cfg)

    def forward(self, x, cache=None, use_cache=False):
        if self.mode == "post":
            # Original Transformer: normalize after each residual addition.
            a, cache = self.attn(x, cache, use_cache)
            x = self.norm1(x + a)
            return self.norm2(x + self.ffn(x)), cache
        a, cache = self.attn(self.norm1(x), cache, use_cache)
        x = x + a
        return x + self.ffn(self.norm2(x)), cache


class TinyLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab, cfg.width)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.layers))
        # Post-norm blocks already end in a norm; "none" removes every RMSNorm.
        self.norm = RMSNorm(cfg.width) if cfg.norm == "pre" else nn.Identity()
        self.head = nn.Linear(cfg.width, cfg.vocab, bias=False)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init_linear(module.weight)
        if cfg.tie_embeddings:
            # A shared matrix serves as both lookup and readout, so use the readout scale.
            self.embedding.weight = self.head.weight
        else:
            nn.init.trunc_normal_(self.embedding.weight, std=1.0, a=-3.0, b=3.0)

    def forward(self, ids, caches=None, use_cache=False):
        if ids.ndim != 2 or not ids.shape[0] or not ids.shape[1] or ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("nonempty integer token IDs with shape [batch, time] required")
        if caches is not None:
            if len(caches) != len(self.blocks):
                raise ValueError("one KV cache pair per model layer required")
            lengths = []
            for pair in caches:
                if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                    raise ValueError("each KV cache must contain a key and value tensor")
                key, value = pair
                expected = (len(ids), self.cfg.kv_heads)
                if key.ndim != 4 or value.shape != key.shape or key.shape[:2] != expected \
                        or key.shape[-1] != self.cfg.width // self.cfg.heads or key.device != ids.device \
                        or value.device != ids.device or key.dtype != value.dtype:
                    raise ValueError("KV cache shape, dtype or device does not match the model input")
                lengths.append(key.shape[-2])
            if len(set(lengths)) != 1:
                raise ValueError("all layers must cache the same prefix length")
        x = self.embedding(ids)
        new_caches = []
        for i, block in enumerate(self.blocks):
            x, cache = block(x, None if caches is None else caches[i], use_cache)
            new_caches.append(cache)
        logits = self.head(self.norm(x))
        return (logits, new_caches) if use_cache else logits


def sample_next(logits, temperature=1.0, top_p=1.0, generator=None):
    """Temperature then nucleus filtering; temperature 0 means greedy argmax."""
    if not math.isfinite(temperature) or temperature < 0 or not 0 < top_p <= 1:
        raise ValueError("nonnegative finite temperature and 0 < top_p <= 1 required")
    if logits.ndim != 2 or min(logits.shape) < 1 or not logits.is_floating_point() or torch.isnan(logits).any() or torch.isposinf(logits).any() or not torch.isfinite(logits).any(-1).all():
        raise ValueError("each logit row needs a finite candidate, without NaN or positive infinity")
    if temperature == 0:
        return logits.argmax(-1, keepdim=True)
    use_double = logits.dtype == torch.float64 or temperature < torch.finfo(torch.float32).tiny
    work = logits.double() if use_double else logits.float()
    # Center before dividing by a tiny temperature; negative infinity masks are valid.
    probs = ((work - work.max(-1, keepdim=True).values) / temperature).softmax(-1)
    if top_p < 1:
        sorted_probs, order = probs.sort(-1, descending=True)
        # Keep the smallest prefix whose mass reaches top_p (always keep the top token).
        keep = sorted_probs.cumsum(-1) - sorted_probs < top_p
        sorted_probs = sorted_probs * keep
        probs = torch.zeros_like(probs).scatter(-1, order, sorted_probs)
        probs = probs / probs.sum(-1, keepdim=True)
    return torch.multinomial(probs, 1, generator=generator)


@torch.no_grad()
def generate(model, prompt, new_tokens=40, temperature=1.0, top_p=1.0, eos_id=None, generator=None):
    """Sample with a KV cache in eval mode; the model's previous train/eval mode is restored."""
    if prompt.ndim != 2 or not prompt.shape[0] or not prompt.shape[1] or type(new_tokens) is not int or new_tokens < 0:
        raise ValueError("nonempty prompt and nonnegative count required")
    if eos_id is not None and (type(eos_id) is not int or not 0 <= eos_id < model.cfg.vocab):
        raise ValueError("EOS ID must be in the model vocabulary")
    was_training = model.training
    model.eval()
    try:
        return _generate(model, prompt, new_tokens, temperature, top_p, eos_id, generator)
    finally:
        model.train(was_training)


def _generate(model, prompt, new_tokens, temperature, top_p, eos_id, generator):
    ids, caches = prompt.clone(), None
    chunk = ids
    # Rows that already emitted EOS keep receiving EOS, so each row can be cut at its first EOS.
    done = torch.zeros(len(ids), dtype=torch.bool, device=ids.device)
    for _ in range(new_tokens):
        logits, caches = model(chunk, caches=caches, use_cache=True)
        token = sample_next(logits[:, -1], temperature, top_p, generator)
        if eos_id is not None:
            token = torch.where(done[:, None], torch.full_like(token, eos_id), token)
            done |= token[:, 0] == eos_id
        ids = torch.cat((ids, token), dim=1)
        if eos_id is not None and bool(done.all()):
            break
        chunk = token
    return ids
