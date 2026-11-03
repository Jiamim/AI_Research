"""Policy objectives used by recent reasoning-model RL recipes, on the same tensor contract as posttrain.py.

All losses take per-token log-probs [B,T], detached old log-probs [B,T], per-sequence
advantages [B] and a 0/1 response mask [B,T], and return a scalar to minimize.

* dapo_loss: decoupled clip (eps_low, eps_high) and token-level aggregation over the
  whole batch (DAPO, arXiv 2503.14476).
* cispo_loss: clip the importance weight itself and stop its gradient; exceeding
  the upper bound alone no longer zeros the REINFORCE term (MiniMax-M1, arXiv 2506.13585).
* dynamic_sampling_mask: drop prompt groups whose rewards are all equal.
* overlong_penalty: DAPO's soft length penalty for truncated responses.
* clipped_token_fraction: share of tokens whose gradient a PPO-style clip removes.
"""
import math
import torch
from .posttrain import masked_log_ratio, finite_ratio, check_advantage


def token_level_mean(values, valid):
    """Sum over every valid token in the batch, divided by the total valid-token count."""
    return values.masked_fill(~valid, 0).sum() / valid.sum()


def clipping_bounds(eps_low, eps_high):
    if not math.isfinite(eps_low) or not math.isfinite(eps_high) or not 0 <= eps_low < 1 or eps_high < 0:
        raise ValueError("finite 0 <= eps_low < 1 and eps_high >= 0 required")
    return 1 - eps_low, 1 + eps_high


def dapo_loss(logp, old_logp, advantage, mask, eps_low=0.2, eps_high=0.28):
    log_ratio, valid = masked_log_ratio(logp, old_logp, mask)
    check_advantage(advantage, len(logp), max(eps_low, 0))
    low, high = clipping_bounds(eps_low, eps_high)
    ratio = finite_ratio(log_ratio)
    a = advantage.detach()[:, None]
    objective = torch.minimum(ratio * a, ratio.clamp(low, high) * a)
    return -token_level_mean(objective, valid)


def cispo_loss(logp, old_logp, advantage, mask, eps_high=2.0, eps_low=None):
    """sg(clip(r)) * A * log pi; upper clipping alone does not zero the term's gradient."""
    log_ratio, valid = masked_log_ratio(logp, old_logp, mask)
    check_advantage(advantage, len(logp), 0)
    ratio = finite_ratio(log_ratio)
    low, high = clipping_bounds(0.0 if eps_low is None else eps_low, eps_high)
    low = 0.0 if eps_low is None else low
    weight = ratio.clamp(low, high).detach()
    current = logp.masked_fill(~valid, 0)
    return -token_level_mean(weight * advantage.detach()[:, None] * current, valid)


def clipped_token_fraction(logp, old_logp, advantage, mask, eps_low=0.2, eps_high=0.2):
    """Tokens where the clipped branch is active and therefore receives zero gradient."""
    low, high = clipping_bounds(eps_low, eps_high)
    check_advantage(advantage, len(logp), eps_low)
    log_ratio, valid = masked_log_ratio(logp, old_logp, mask)
    ratio = finite_ratio(log_ratio)
    a = advantage[:, None]
    clipped = ((a > 0) & (ratio > high)) | ((a < 0) & (ratio < low))
    return (clipped & valid).sum().item() / valid.sum().item()


def dynamic_sampling_mask(rewards):
    """[prompts, group] rewards -> bool [prompts]: keep groups that contain different rewards."""
    if rewards.ndim != 2 or min(rewards.shape) < 1 or not torch.isfinite(rewards).all():
        raise ValueError("finite nonempty [prompts, group] rewards required")
    return rewards.max(-1).values != rewards.min(-1).values


def overlong_penalty(lengths, max_length, cache):
    """0 up to max_length - cache, then a linear ramp to -1 at max_length, and -1 beyond."""
    if not math.isfinite(cache) or not math.isfinite(max_length) or not 0 < cache <= max_length:
        raise ValueError("require 0 < cache <= max_length")
    lengths = torch.as_tensor(lengths, dtype=torch.float32)
    if not torch.isfinite(lengths).all() or torch.any(lengths < 0):
        raise ValueError("finite nonnegative lengths required")
    start = max_length - cache
    ramp = (start - lengths) / cache
    return torch.where(lengths <= start, torch.zeros_like(lengths), torch.where(lengths <= max_length, ramp, -torch.ones_like(lengths)))


def demo():
    torch.manual_seed(0)
    old = torch.log_softmax(torch.randn(4, 6), -1)
    # A drifted policy; the third token's ratio is e^0.22 = 1.246, between 1.2 and 1.28.
    logp = old + torch.tensor([[1.2, -0.1, 0.22, 0.0, -1.0, 0.3]] * 4)
    mask = torch.ones(4, 6)
    advantage = torch.tensor([1.0, -1.0, 1.0, -1.0])
    rewards = torch.tensor([[1., 1, 1, 1], [0, 0, 0, 0], [1, 0, 0, 1], [0, 1, 0, 0]])
    return dict(
        clipped_fraction_symmetric_0_2=clipped_token_fraction(logp, old, advantage, mask, 0.2, 0.2),
        clipped_fraction_dapo_0_2_0_28=clipped_token_fraction(logp, old, advantage, mask, 0.2, 0.28),
        groups_kept_by_dynamic_sampling=dynamic_sampling_mask(rewards).tolist(),
        overlong_penalty=overlong_penalty([14000, 16384, 18000, 20480, 21000], 20480, 4096).tolist())


if __name__ == "__main__":
    import json
    print(json.dumps(demo(), indent=2))
