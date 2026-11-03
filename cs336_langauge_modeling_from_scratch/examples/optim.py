"""Dense-gradient AdamW, cosine schedule and global-norm clipping, written out step by step.

Teaching reference: A1 asks you to implement these yourself (coursework restrictions are explained in chapter 28).
"""
import math
import torch


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1):
        self._validate(lr, betas, eps, weight_decay)
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
        for group in self.param_groups:
            self._validate(group['lr'], group['betas'], group['eps'], group['weight_decay'])

    @staticmethod
    def _validate(lr, betas, eps, weight_decay):
        if not all(math.isfinite(x) for x in (lr, eps, weight_decay)) or lr < 0 or eps <= 0 or weight_decay < 0 \
                or len(betas) != 2 or not all(math.isfinite(b) and 0 <= b < 1 for b in betas):
            raise ValueError("finite learning rate/weight decay >= 0, epsilon > 0 and two betas in [0,1) required")

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise NotImplementedError("closures are outside this teaching optimizer")
        # Check every group before mutating any weights or optimizer state.
        for group in self.param_groups:
            self._validate(group['lr'], group['betas'], group['eps'], group['weight_decay'])
            if any(p.grad is not None and p.grad.is_sparse for p in group['params']):
                raise ValueError('dense gradients only')
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if not state:
                    state.update(step=0, m=torch.zeros_like(p), v=torch.zeros_like(p))
                state["step"] += 1
                t, m, v = state["step"], state["m"], state["v"]
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                p.mul_(1 - group["lr"] * group["weight_decay"])
                mhat = m / (1 - b1 ** t)
                vhat = v / (1 - b2 ** t)
                p.addcdiv_(mhat, vhat.sqrt().add_(group["eps"]), value=-group["lr"])


def cosine_lr(t, max_lr, min_lr, warmup, cosine_end):
    """Linear warmup over [0, warmup), cosine decay to min_lr at cosine_end, then constant."""
    if not all(math.isfinite(x) for x in (t, max_lr, min_lr, warmup, cosine_end)) or t < 0 \
            or not 0 <= warmup <= cosine_end or not 0 <= min_lr <= max_lr:
        raise ValueError("require 0 <= warmup <= cosine_end and min_lr <= max_lr")
    if t < warmup:
        return max_lr * t / warmup
    if t >= cosine_end:
        return min_lr
    progress = (t - warmup) / (cosine_end - warmup)
    return min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (max_lr - min_lr)


def wsd_lr(t, max_lr, min_lr, warmup, decay_start, end):
    """Warmup-stable-decay: hold the peak, then decay linearly to min_lr at `end`."""
    if not all(math.isfinite(x) for x in (t, max_lr, min_lr, warmup, decay_start, end)) or t < 0 \
            or not 0 <= min_lr <= max_lr or not 0 <= warmup <= decay_start <= end:
        raise ValueError("require 0 <= warmup <= decay_start <= end")
    if t < warmup:
        return max_lr * t / warmup
    if t >= end:
        return min_lr
    if t < decay_start:
        return max_lr
    progress = (t - decay_start) / (end - decay_start)
    return max_lr + progress * (min_lr - max_lr)


@torch.no_grad()
def clip_grad_norm(parameters, max_norm, eps=1e-6):
    """Scale all gradients together when their joint L2 norm exceeds max_norm.

    Accepts an iterable of parameters or a single tensor, like torch.nn.utils.clip_grad_norm_.
    Returns the norm before clipping, which is the number worth logging.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    if not math.isfinite(max_norm) or max_norm < 0 or not math.isfinite(eps) or eps < 0:
        raise ValueError("finite nonnegative max_norm and eps required")
    grads = [g for g in grads if g.numel()]
    if not grads:
        return torch.tensor(0.0)
    if any(g.is_sparse for g in grads) or len({g.device for g in grads}) > 1:
        raise ValueError("dense gradients on one device required; reduce sharded norms separately")
    if any(not torch.isfinite(g).all() for g in grads):
        raise ValueError("nonfinite gradients: inspect the failed step before clipping")
    dtype = torch.float64 if any(g.dtype == torch.float64 for g in grads) else torch.float32
    # Scale before squaring: a finite gradient may overflow a naive sum of squares.
    scale = torch.stack([g.abs().max().to(dtype) for g in grads]).max()
    total = scale if scale == 0 else scale * torch.sqrt(sum((g.to(dtype) / scale).square().sum() for g in grads))
    if not torch.isfinite(total):
        raise ValueError("gradient norm exceeds the accumulation dtype range")
    coefficient = (max_norm / (total + eps)).clamp(max=1) if total > 0 else total.new_tensor(1.)
    for g in grads:
        g.mul_(coefficient)
    return total
