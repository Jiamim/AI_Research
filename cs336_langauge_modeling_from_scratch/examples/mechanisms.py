"""Executable derivations: exact loss tiling, affine scan, MLA and statistical diagnostics.

Small CPU references. They expose invariants, not production GPU performance.
"""
import math
import torch
from torch.autograd.function import once_differentiable


class _LinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, labels, block):
        # hidden [M,D], weight [V,D]; one tile [M,block] at a time.
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            lse = hidden.new_full((len(hidden),), -torch.inf)
            for start in range(0, len(weight), block):
                logits = hidden @ weight[start:start + block].T
                lse = torch.logaddexp(lse, logits.logsumexp(-1))
            target = (hidden * weight[labels]).sum(-1)
            ctx.save_for_backward(hidden, weight, labels, lse)
            ctx.block = block
            return (lse - target).mean()

    @staticmethod
    @once_differentiable
    def backward(ctx, upstream):
        hidden, weight, labels, lse = ctx.saved_tensors
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            dx, dw = torch.zeros_like(hidden), torch.empty_like(weight)
            rows = torch.arange(len(hidden), device=hidden.device)
            for start in range(0, len(weight), ctx.block):
                end = min(start + ctx.block, len(weight))
                gradient = (hidden @ weight[start:end].T - lse[:, None]).exp()
                owned = (labels >= start) & (labels < end)
                gradient[rows[owned], labels[owned] - start] -= 1
                gradient *= upstream / len(hidden)
                dx.add_(gradient @ weight[start:end])
                dw[start:end] = gradient.T @ hidden
            return dx, dw, None, None


def tiled_linear_cross_entropy(hidden, weight, labels, block=16):
    """Exact mean CE, without storing all logits. FP32/64, no bias, first-order only.

    Includes full hidden/weight gradients; no gradient filtering, label smoothing,
    ignore_index or GPU fusion. Filter padded targets before calling. Forward and
    recomputation disable autocast so the stated FP32/64 arithmetic stays consistent.
    """
    if hidden.ndim != 2 or weight.ndim != 2 or hidden.shape[1] != weight.shape[1]:
        raise ValueError("hidden [M,D] and weight [V,D] must have matching D")
    if min(*hidden.shape, len(weight)) < 1 or labels.shape != hidden.shape[:1] or labels.dtype != torch.long:
        raise ValueError("nonempty inputs and one int64 label per row required")
    if hidden.dtype not in (torch.float32, torch.float64) or weight.dtype != hidden.dtype:
        raise ValueError("matching FP32 or FP64 inputs required")
    if hidden.device != weight.device or labels.device != hidden.device:
        raise ValueError("inputs must share a device")
    if not isinstance(block, int) or block < 1 or torch.any((labels < 0) | (labels >= len(weight))):
        raise ValueError("positive tile width and in-vocabulary labels required")
    if not torch.isfinite(hidden).all() or not torch.isfinite(weight).all():
        raise ValueError("finite inputs required")
    return _LinearCrossEntropy.apply(hidden, weight, labels, block)


def sharded_cross_entropy(shards, labels):
    """Virtual vocabulary ranks: global maximum/sum, then owner-only target logit.

    All shards live in one process; autograd here is NOT a distributed collective.
    """
    if not shards or any(s.ndim != 2 or min(s.shape) < 1 for s in shards):
        raise ValueError("nonempty [M,V_rank] shards required")
    if any(s.shape[0] != shards[0].shape[0] for s in shards) or labels.shape != shards[0].shape[:1]:
        raise ValueError("matching row counts and labels required")
    if labels.dtype != torch.long or any(s.device != labels.device or not s.is_floating_point()
                                         or not torch.isfinite(s).all() for s in shards):
        raise ValueError("finite floating shards and int64 labels on one device required")
    dtype = torch.float64 if any(s.dtype == torch.float64 for s in shards) else torch.float32
    shards = [s.to(dtype) for s in shards]
    if torch.any((labels < 0) | (labels >= sum(s.shape[1] for s in shards))):
        raise ValueError("labels outside vocabulary")
    maximum = torch.stack([s.max(-1).values for s in shards]).max(0).values.detach()
    denominator = sum((s - maximum[:, None]).exp().sum(-1) for s in shards)
    target, offset = torch.zeros_like(maximum), 0
    for shard in shards:
        local = labels - offset
        owned = (local >= 0) & (local < shard.shape[1])
        selected = shard.gather(1, local.clamp(0, shard.shape[1] - 1)[:, None]).squeeze(1)
        target = target + selected.masked_fill(~owned, 0)
        offset += shard.shape[1]
    return (maximum + denominator.log() - target).mean()


def affine_scan(a, b, initial):
    """Inclusive parallel scan of h[t]=a[t]*h[t-1]+b[t], diagonal a.

    Hillis-Steele doubling: O(log T) dependency depth, O(T log T) work/storage
    with autograd. Not Mamba's work-efficient GPU kernel.
    """
    if a.ndim != 2 or a.shape != b.shape or initial.shape != a.shape[1:] or len(a) == 0:
        raise ValueError("a,b [T,D] and initial [D] required")
    aa, bb, step = a, b, 1
    while step < len(a):
        new_a = aa[step:] * aa[:-step]
        new_b = aa[step:] * bb[:-step] + bb[step:]
        aa = torch.cat((aa[:step], new_a))
        bb = torch.cat((bb[:step], new_b))
        step *= 2
    return aa * initial + bb


def mla_content_output(query, latent, up_key, up_value, output, absorbed=True):
    """Content-only MLA at one query: head-wise attention, then output projection.

    q [H,Dk], c [T,R], Wk [H,Dk,R], Wv [H,Dv,R], Wo [H,Dout,Dv].
    Positional scores and masks are intentionally omitted to isolate the identity.
    """
    if absorbed:
        q_latent = torch.einsum('hd,hdr->hr', query, up_key)
        scores = q_latent @ latent.T / math.sqrt(query.shape[-1])
        context = scores.softmax(-1) @ latent
        combined = torch.einsum('hov,hvr->hor', output, up_value)
        return torch.einsum('hor,hr->o', combined, context)
    keys = torch.einsum('tr,hdr->htd', latent, up_key)
    values = torch.einsum('tr,hvr->htv', latent, up_value)
    scores = torch.einsum('hd,htd->ht', query, keys) / math.sqrt(query.shape[-1])
    context = torch.einsum('ht,htv->hv', scores.softmax(-1), values)
    return torch.einsum('hov,hv->o', output, context)


def importance_diagnostics(log_weights):
    """Stable weight concentration diagnostics, not an unbiasedness guarantee."""
    if log_weights.ndim != 1 or not len(log_weights) or not torch.isfinite(log_weights).all():
        raise ValueError("a finite nonempty vector of log weights required")
    values = log_weights.detach().double()
    normalized = values.softmax(0)
    return dict(ess=(1 / normalized.square().sum()).item(),
                max_mass=normalized.max().item(),
                log_mean_weight=(values.logsumexp(0) - math.log(len(values))).item())


def verifier_precision(base_success, true_accept, false_accept):
    """P(correct | accepted), given generator success and verifier TPR/FPR."""
    if not all(0 <= x <= 1 for x in (base_success, true_accept, false_accept)):
        raise ValueError("probabilities must be in [0,1]")
    mass = base_success * true_accept + (1 - base_success) * false_accept
    if mass == 0:
        raise ValueError("conditioning event has zero probability")
    return base_success * true_accept / mass


def potential_shaping(rewards, potentials, gamma=1.):
    """Add gamma*Phi(s[t+1])-Phi(s[t]); terminal potential must be handled by caller."""
    if rewards.ndim != 1 or potentials.shape != (len(rewards) + 1,) or not 0 <= gamma <= 1:
        raise ValueError("T rewards, T+1 potentials and gamma in [0,1] required")
    return rewards + gamma * potentials[1:] - potentials[:-1]


def stochastic_round_interval(x, lower, upper, generator=None):
    """In-range stochastic rounding to TWO adjacent values; no clipping or STE.

    The probability and the uniform draw are computed in float64: a bf16/fp16 draw is itself
    quantized (bf16 steps of 2^-8 near 1), which would bias the rounding it is meant to unbias.
    The result keeps x's dtype; both endpoints must be exactly representable in that dtype.
    """
    if not math.isfinite(lower) or not math.isfinite(upper) or not lower < upper:
        raise ValueError("finite ordered endpoints required")
    if not x.is_floating_point():
        raise ValueError("floating-point input required")
    low, high = x.new_tensor(lower), x.new_tensor(upper)
    if low.item() != lower or high.item() != upper:
        raise ValueError("endpoints must be exactly representable in the input dtype")
    values = x.double()
    if not torch.isfinite(values).all() or torch.any((values < lower) | (values > upper)):
        raise ValueError("x must lie in the specified interval")
    span = upper - lower
    # Opposite finite FP64 endpoints can have an overflowing difference.
    probability = ((values - lower) / span if math.isfinite(span)
                   else (values / 2 - lower / 2) / (upper / 2 - lower / 2))
    draws = torch.rand(x.shape, dtype=torch.float64, device=x.device, generator=generator)
    # Scalar-only torch.where branches would first round both endpoints to the default dtype.
    return torch.where(draws < probability, high, low)


def demo():
    from torch.nn import functional as F
    rng = torch.Generator().manual_seed(336)
    h = torch.randn(7, 5, generator=rng, dtype=torch.float64, requires_grad=True)
    w = torch.randn(19, 5, generator=rng, dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([0, 3, 6, 9, 12, 15, 18])
    reference = F.cross_entropy(h @ w.T, labels)
    tiled = tiled_linear_cross_entropy(h, w, labels, 6)
    actual = torch.autograd.grad(tiled, (h, w))
    expected = torch.autograd.grad(reference, (h, w))
    logits = h.detach() @ w.detach().T
    split = sharded_cross_entropy(list(logits.split([4, 8, 7], dim=1)), labels)
    a, b = torch.rand(9, 4, generator=rng, dtype=torch.float64), torch.randn(9, 4, generator=rng, dtype=torch.float64)
    initial = torch.randn(4, generator=rng, dtype=torch.float64)
    state, sequential = initial, []
    for at, bt in zip(a, b):
        state = at * state + bt
        sequential.append(state)
    tensors = [torch.randn(*shape, generator=rng, dtype=torch.float64)
               for shape in ((2, 3), (7, 4), (2, 3, 4), (2, 2, 4), (2, 5, 2))]
    mla_error = (mla_content_output(*tensors) - mla_content_output(*tensors, absorbed=False)).abs().max()
    samples = stochastic_round_interval(torch.full((50000,), .3), 0., .5, rng)
    p = [.01, .05, .25, .5, .9]
    return dict(
        scope='Small CPU derivations and controlled sampling; no industrial training or GPU benchmark',
        linear_ce=dict(loss=tiled.item(), loss_error=abs(tiled.item() - reference.item()),
                       gradient_max_error=max((x-y).abs().max().item() for x,y in zip(actual, expected)),
                       virtual_vocab_parallel_error=abs(split.item() - reference.item())),
        affine_scan_max_error=(affine_scan(a,b,initial)-torch.stack(sequential)).abs().max().item(),
        mla_content_max_error=mla_error.item(),
        importance=dict(uniform=importance_diagnostics(torch.zeros(16)),
                        concentrated=importance_diagnostics(torch.tensor([20.]+[0.]*15))),
        verifier=dict(base_success=p, accepted_precision=[verifier_precision(x,.9,.05) for x in p]),
        rounding=dict(expected=.3, empirical_mean=samples.mean().item(), expected_variance=.3*(.5-.3),
                      empirical_variance=samples.var(correction=0).item()),
        moe_expected_experts={str(batch):128*(1-(1-4/128)**batch) for batch in (1,8,32,128)},
    )


if __name__ == '__main__':
    import json
    print(json.dumps(demo(), indent=2))
