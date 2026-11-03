"""Attention references for chapter 10: the online-softmax invariant and a first Triton kernel.

* dense_attention: the naive O(T^2)-memory reference.
* online_attention: keys streamed in blocks with the running (m, l, u) state, the
  invariant FlashAttention is built on.
* softmax_backward: the softmax vector-Jacobian product dS = P * (dP - rowsum(P * dP)),
  the identity behind FlashAttention's backward.
* triton_softmax: a row softmax written as one Triton kernel that streams each row in
  column blocks with the same (m, l) recurrence. It teaches the Triton programming
  model and runs on CPU when TRITON_INTERPRET=1 is set before triton is imported.

These are teaching references, not the 2026 A2 FlashAttention-2 deliverables: the tiled
forward, the logsumexp-based backward and the Triton attention kernel are left to the
assignment (coursework restrictions and this demo's scope are explained in chapter 28).
"""
import math
import os
import torch


def dense_attention(q, k, v, causal=True):
    s = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    if causal:
        t_q, t_k = q.shape[-2], k.shape[-2]
        if t_q != t_k:
            raise ValueError("this reference uses square causal attention")
        mask = torch.arange(t_k, device=q.device)[None, :] <= torch.arange(t_q, device=q.device)[:, None]
        s = s.masked_fill(~mask, -torch.inf)
    return s.softmax(-1) @ v


def online_attention(q, k, v, block_size=16, causal=True):
    """All queries at once, keys streamed in blocks: shows the running (m, l, u) state."""
    if type(block_size) is not int or block_size < 1 or q.ndim != 2 or k.ndim != 2 or v.ndim != 2 \
            or len(k) == 0 or len(q) == 0 or len(k) != len(v) or q.shape[1] != k.shape[1]:
        raise ValueError("positive block size and nonempty input required")
    if causal and len(q) != len(k):
        raise ValueError("this reference uses square causal attention")
    m = q.new_full((len(q), 1), -torch.inf)
    denom = q.new_zeros((len(q), 1))
    numer = q.new_zeros((len(q), v.shape[1]))
    for start in range(0, len(k), block_size):
        kb, vb = k[start:start + block_size], v[start:start + block_size]
        scores = q @ kb.T / math.sqrt(q.shape[1])
        if causal:
            mask = torch.arange(start, start + len(kb), device=q.device)[None, :] <= torch.arange(len(q), device=q.device)[:, None]
            scores = scores.masked_fill(~mask, -torch.inf)
        new_m = torch.maximum(m, scores.max(-1, keepdim=True).values)
        alpha = (m - new_m).exp()
        p = (scores - new_m).exp()
        numer = alpha * numer + p @ vb
        denom = alpha * denom + p.sum(-1, keepdim=True)
        m = new_m
    return numer / denom


def softmax_backward(p, grad_p):
    """Gradient w.r.t. the logits of a row softmax, given P and dL/dP (last dimension)."""
    return p * (grad_p - (p * grad_p).sum(-1, keepdim=True))


def _build_softmax_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def softmax_kernel(X, Y, stride_x, stride_y, n_cols, BLOCK: tl.constexpr):
        # One program per row; the row is read in BLOCK-wide pieces, so it need not fit on chip.
        row = tl.program_id(0)
        m = tl.full((1,), float("-inf"), tl.float32)
        l = tl.zeros((1,), tl.float32)
        for start in range(0, n_cols, BLOCK):
            cols = start + tl.arange(0, BLOCK)
            x = tl.load(X + row * stride_x + cols, mask=cols < n_cols, other=float("-inf")).to(tl.float32)
            m_new = tl.maximum(m, tl.max(x, axis=0))
            # A fully masked prefix keeps m = -inf; exp(-inf - -inf) would be NaN, so shift by 0 instead.
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            l = l * tl.exp(m - m_safe) + tl.sum(tl.exp(x - m_safe), axis=0)
            m = m_new
        for start in range(0, n_cols, BLOCK):
            cols = start + tl.arange(0, BLOCK)
            x = tl.load(X + row * stride_x + cols, mask=cols < n_cols, other=float("-inf")).to(tl.float32)
            y = tl.exp(x - m) / l
            tl.store(Y + row * stride_y + cols, y.to(Y.dtype.element_ty), mask=cols < n_cols)

    return softmax_kernel


def triton_softmax(x, block=16):
    """Softmax over the last dimension of a 2-D tensor. Set TRITON_INTERPRET=1 to run on CPU."""
    if x.dim() != 2 or min(x.shape) < 1 or x.dtype not in (torch.float16,torch.bfloat16,torch.float32):
        raise ValueError("nonempty 2-D FP16/BF16/FP32 tensor required")
    if type(block) is not int or block < 1 or block & (block - 1):
        raise ValueError('Triton tile width must be a positive power of two')
    if torch.isnan(x).any() or torch.isposinf(x).any() or not torch.isfinite(x).any(-1).all():
        raise ValueError('each softmax row needs a finite candidate, without NaN or positive infinity')
    x = x.contiguous()
    y = torch.empty_like(x)
    _build_softmax_kernel()[(x.shape[0],)](x, y, x.stride(0), y.stride(0), x.shape[1], BLOCK=block)
    return y


if __name__ == "__main__":
    torch.manual_seed(1)
    q, k, v = [torch.randn(37, 16, dtype=torch.float64) for _ in range(3)]
    for causal in (False, True):
        error = (online_attention(q, k, v, 16, causal) - dense_attention(q, k, v, causal)).abs().max().item()
        print(f"online attention, causal={causal}: error {error:.2e}")
    s = torch.randn(5, 37, dtype=torch.float64, requires_grad=True)
    upstream = torch.randn(5, 37, dtype=torch.float64)
    p = s.softmax(-1)
    (reference,) = torch.autograd.grad((p * upstream).sum(), s)
    print(f"softmax backward identity: error {(softmax_backward(p.detach(), upstream) - reference).abs().max().item():.2e}")
    if os.environ.get("TRITON_INTERPRET") == "1":
        x = torch.randn(7, 37) * 30
        print("triton softmax (interpreter) error:", (triton_softmax(x) - x.softmax(-1)).abs().max().item())
