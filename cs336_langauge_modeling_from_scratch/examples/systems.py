"""CPU references for a first-order custom backward, virtual tensor parallelism and low-precision accumulation."""
import json
import math
import torch
from torch.autograd.function import once_differentiable


class RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gain, eps):
        if x.shape[-1] != gain.numel() or gain.ndim != 1 or not math.isfinite(eps) or eps <= 0:
            raise ValueError("matching one-dimensional gain and positive epsilon required")
        dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
        work, weight = x.to(dtype), gain.to(dtype)
        inv = torch.rsqrt(work.square().mean(-1, keepdim=True) + eps)
        ctx.save_for_backward(work, weight, inv)
        ctx.x_dtype, ctx.gain_dtype = x.dtype, gain.dtype
        return (work * inv * weight).to(x.dtype)

    @staticmethod
    @once_differentiable
    def backward(ctx, upstream):
        x, gain, inv = ctx.saved_tensors
        dy = upstream.to(x.dtype)
        weighted = dy * gain
        dx = weighted * inv - x * inv ** 3 * (weighted * x).mean(-1, keepdim=True)
        # gain is shared by every row, so its gradient sums over all leading dimensions.
        dg = (dy * x * inv).reshape(-1, gain.numel()).sum(0)
        return dx.to(ctx.x_dtype), dg.to(ctx.gain_dtype), None


def rmsnorm_custom(x, gain, eps=1e-6):
    return RMSNormFunction.apply(x, gain, eps)


def tensor_parallel_demo(parts=2):
    """Virtual shards: sum output contributions and input-gradient contributions."""
    if parts < 1 or 12 % parts:
        raise ValueError("parts must divide hidden width 12")
    g = torch.Generator().manual_seed(31)  # local generator: the caller's global RNG is untouched
    x = torch.randn(5, 8, dtype=torch.float64, generator=g).requires_grad_()
    w1 = torch.randn(8, 12, dtype=torch.float64, generator=g).requires_grad_()
    w2 = torch.randn(12, 6, dtype=torch.float64, generator=g).requires_grad_()
    reference = (x @ w1).tanh() @ w2
    contributions = [(x @ a).tanh() @ b for a, b in zip(w1.chunk(parts, dim=1), w2.chunk(parts, dim=0))]
    output = sum(contributions)
    upstream = torch.randn(reference.shape, dtype=reference.dtype, generator=g)
    dx_reference = torch.autograd.grad(reference, x, upstream, retain_graph=True)[0]
    dx_shards = [torch.autograd.grad(y, x, upstream, retain_graph=True)[0] for y in contributions]
    torch.testing.assert_close(output, reference)
    torch.testing.assert_close(sum(dx_shards), dx_reference)
    return dict(parts=parts, output_max_error=(output - reference).abs().max().item(),
                input_gradient_max_error=(sum(dx_shards) - dx_reference).abs().max().item(),
                scope="virtual CPU shards; no communication-performance claim")


def accumulation_demo(steps=10000, value=1e-3):
    """Summing many small numbers: where the running sum lives decides the error.

    A different setting from the 2026 A2 mixed_precision_accumulation problem, which is
    left to the student. Kahan summation keeps the running sum in BF16 but carries the
    rounding error of every addition in a second BF16 variable.
    """
    results = {}
    for name, acc_dtype, item_dtype in [("fp32 += fp32", torch.float32, torch.float32),
                                        ("bf16 += bf16", torch.bfloat16, torch.bfloat16),
                                        ("fp32 += bf16", torch.float32, torch.bfloat16)]:
        total = torch.tensor(0, dtype=acc_dtype)
        for _ in range(steps):
            total += torch.tensor(value, dtype=item_dtype)
        results[name] = total.item()
    total = torch.tensor(0, dtype=torch.bfloat16)
    compensation = torch.tensor(0, dtype=torch.bfloat16)
    item = torch.tensor(value, dtype=torch.bfloat16)
    for _ in range(steps):
        y = item - compensation
        t = total + y
        compensation = (t - total) - y
        total = t
    results["bf16 += bf16, Kahan"] = total.item()
    return dict(exact=steps * value, bf16_item=item.item(), results=results)


if __name__ == "__main__":
    print(json.dumps(dict(tensor_parallel=tensor_parallel_demo(), accumulation=accumulation_demo()), indent=2))
