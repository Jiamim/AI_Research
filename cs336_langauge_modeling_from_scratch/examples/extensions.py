"""Small MoE, linear-attention and image/text-alignment references.

These implementations expose mathematical structure, not production efficiency.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class TopKMoE(nn.Module):
    """All experts are evaluated for clarity; sparse *outputs*, not sparse compute."""
    def __init__(self, width=16, hidden=32, experts=4, topk=2):
        super().__init__()
        if not 1 <= topk <= experts:
            raise ValueError("1 <= topk <= experts required")
        self.topk = topk
        self.router = nn.Linear(width, experts, bias=False)
        self.experts = nn.ModuleList(nn.Sequential(nn.Linear(width, hidden), nn.SiLU(),
                                                   nn.Linear(hidden, width)) for _ in range(experts))

    def forward(self, x):
        logits = self.router(x)
        probs = logits.to(torch.float64 if logits.dtype == torch.float64 else torch.float32).softmax(-1)
        weights, indices = probs.topk(self.topk, dim=-1)
        # Retain probabilities from the full softmax, including for top-1.
        gates = torch.zeros_like(probs).scatter(-1, indices, weights).to(x.dtype)
        outputs = torch.stack([expert(x) for expert in self.experts], dim=-2)
        y = (outputs * gates[..., None]).sum(-2)
        assignments = F.one_hot(indices, num_classes=len(self.experts)).float().sum(-2)
        load = assignments.reshape(-1, len(self.experts)).mean(0) / self.topk
        mean_probability = probs.reshape(-1,len(self.experts)).mean(0)
        balance = len(self.experts) * (load.detach() * mean_probability).sum()
        return y, balance, load


def _linear_attention_inputs(q, k, v):
    if q.ndim != 2 or k.shape != q.shape or v.ndim != 2 or len(v) != len(q) \
            or min(*q.shape, v.shape[-1]) < 1:
        raise ValueError("nonempty q,k [T,Dk] and v [T,Dv] with matching time axes required")
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64) or any(
        x.dtype != q.dtype or x.device != q.device or not torch.isfinite(x).all() for x in (q, k, v)
    ):
        raise ValueError("finite matching floating dtypes and devices required")
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    def feature(x):
        x = x.to(dtype)
        # ELU(x)+1 equals exp(x) on x<0. Computing (exp(x)-1)+1
        # loses small positive values to cancellation, even in FP32.
        return torch.where(x >= 0, x + 1, x.clamp(max=0).exp())
    return feature(q), feature(k), v.to(dtype)


def linear_attention_recurrent(q, k, v):
    """Causal ELU+1 kernel, denominator floor 1e-8; FP32/64 accumulation.

    Returns the input dtype. This is not softmax attention; the denominator
    floor changes the normalized kernel when its total mass is very small.
    """
    output_dtype = q.dtype
    q, k, v = _linear_attention_inputs(q, k, v)
    # Explicit casts alone do not stop an enclosing autocast region from
    # lowering these matrix products again and erasing small feature values.
    with torch.autocast(device_type=q.device.type, enabled=False):
        state = q.new_zeros(q.shape[-1], v.shape[-1])
        normalizer = q.new_zeros(q.shape[-1])
        outputs = []
        for qt, kt, vt in zip(q,k,v):
            state = state + kt[:, None] * vt[None, :]
            normalizer = normalizer + kt
            outputs.append((qt @ state) / (qt @ normalizer).clamp_min(1e-8))
        return torch.stack(outputs).to(output_dtype)


def linear_attention_explicit(q, k, v):
    """Dense causal reference for the same kernel and denominator convention."""
    output_dtype = q.dtype
    q, k, v = _linear_attention_inputs(q, k, v)
    with torch.autocast(device_type=q.device.type, enabled=False):
        weights = (q @ k.T).tril()
        return ((weights @ v) / weights.sum(-1,keepdim=True).clamp_min(1e-8)).to(output_dtype)


def clip_loss(image_features, text_features, temperature=0.1):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive")
    if image_features.ndim != 2 or image_features.shape != text_features.shape or min(image_features.shape) < 1 \
            or image_features.device != text_features.device or not torch.isfinite(image_features).all() \
            or not torch.isfinite(text_features).all():
        raise ValueError('finite aligned [batch, features] image/text pairs on one device required')
    images, texts = F.normalize(image_features,dim=-1), F.normalize(text_features,dim=-1)
    scores = images @ texts.T / temperature
    targets = torch.arange(len(scores),device=scores.device)
    return (F.cross_entropy(scores,targets)+F.cross_entropy(scores.T,targets))/2


if __name__ == '__main__':
    torch.manual_seed(2)
    q, k, v = [torch.randn(13, 8) for _ in range(3)]
    error = (linear_attention_recurrent(q, k, v) - linear_attention_explicit(q, k, v)).abs().max().item()
    print('linear attention recurrent/explicit maximum error:', error)
    model = TopKMoE()
    y, balance, load = model(torch.randn(2, 5, 16))
    print('MoE shape:', list(y.shape), 'expert load:', load.detach().tolist(), 'balance loss:', balance.item())
    features = torch.eye(8)
    print('CLIP aligned loss:', clip_loss(features, features).item(),
          'permuted loss:', clip_loss(features, features.roll(1, 0)).item())
