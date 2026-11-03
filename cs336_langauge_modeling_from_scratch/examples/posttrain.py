"""Post-training objectives (DPO, SimPO, GRPO family, GSPO) and a complete GRPO loop on a tiny LM.

Teaching reference with its own conventions, chosen for explanation; they do not match the
2026 A5 handout. Do not submit or adapt it as assignment code; write your own from the handout
(coursework restrictions are explained in chapter 28).
"""
import json
import math
import torch
from torch.nn import functional as F
from .objectives import accumulation_dtype, response_mask, sequence_values, token_logps


def masked_sequence_logps(logits, labels, response_mask):
    # logits and labels are already shifted/aligned; [B,T,V], [B,T], [B,T].
    return token_logps(logits, labels, response_mask).sum(-1)


def token_entropy(logits, mask):
    """Full-vocabulary conditional entropy; invalid padding contributes exactly zero."""
    if logits.ndim!=3 or not logits.shape[-1] or not logits.is_floating_point() or logits.device != mask.device:
        raise ValueError("expected [batch,time,vocabulary] logits")
    valid=response_mask(mask,logits.shape[:2])
    work=logits.masked_fill(~valid[...,None],0).to(accumulation_dtype(logits))
    if not torch.isfinite(work).all():
        raise ValueError("valid logits must be finite")
    logp=F.log_softmax(work,-1)
    return -(logp.exp()*logp).sum(-1).masked_fill(~valid,0)


def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1):
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    policy_chosen, policy_rejected, ref_chosen, ref_rejected = sequence_values(
        policy_chosen, policy_rejected, ref_chosen, ref_rejected)
    margin = beta * ((policy_chosen - ref_chosen.detach()) - (policy_rejected - ref_rejected.detach()))
    return -F.logsigmoid(margin).mean()


def simpo_loss(chosen_logps, rejected_logps, chosen_lengths, rejected_lengths, beta=2.0, margin=1.0):
    """Reference-free, length-normalized preference objective for supplied sequence logps."""
    if not math.isfinite(beta) or not math.isfinite(margin) or beta<=0 or margin<0:
        raise ValueError("positive beta and nonnegative margin required")
    if any(x.shape!=chosen_logps.shape for x in [rejected_logps,chosen_lengths,rejected_lengths]):
        raise ValueError("all per-sequence inputs must have matching shapes")
    if torch.any(chosen_lengths<=0) or torch.any(rejected_lengths<=0):
        raise ValueError("responses must have positive effective lengths")
    chosen_logps, rejected_logps = sequence_values(chosen_logps, rejected_logps)
    if any(x.device != chosen_logps.device or not torch.isfinite(x).all() for x in (chosen_lengths, rejected_lengths)):
        raise ValueError("finite effective lengths on the log-probability device required")
    gap=beta*(chosen_logps/chosen_lengths-rejected_logps/rejected_lengths)-margin
    return -F.logsigmoid(gap).mean()


def advantages(rewards, method="grpo", eps=1e-6):
    # [prompts, group_size], population standard deviation (correction=0).
    if not rewards.is_floating_point():
        rewards = rewards.to(torch.get_default_dtype())  # integer/bool 0-1 rewards are common
    if rewards.ndim != 2 or min(rewards.shape) < 1 or not math.isfinite(eps) or eps <= 0 or not torch.isfinite(rewards).all():
        raise ValueError("finite, nonempty [prompts,group] rewards and positive eps required")
    if method in ("maxrl", "rft") and not torch.all((rewards == 0) | (rewards == 1)):
        raise ValueError("this MaxRL/RFT teaching variant assumes binary rewards")
    rewards = rewards.detach().to(accumulation_dtype(rewards))
    mean = rewards.mean(-1, keepdim=True)
    centered = rewards - mean
    if method == "grpo":
        return centered / (rewards.std(-1, keepdim=True, correction=0) + eps)
    if method == "dr_grpo":
        return centered
    if method == "maxrl":
        return centered / mean.clamp_min(eps)
    if method == "rft":
        return rewards
    if method == "rloo":
        n = rewards.shape[-1]
        if n < 2:
            raise ValueError("leave-one-out needs at least 2 responses")
        return centered * n / (n - 1)
    raise ValueError(method)


@torch.no_grad()
def generalized_advantages(rewards, values, next_values, terminated, truncated, gamma=1., lam=.95):
    """GAE for a flat vector of real transitions; no padding or advantage normalization.

    values[t] = V(s_t), next_values[t] = V(s_{t+1}) before any environment reset.
    A true terminal disables bootstrap and the trace. A sampling timeout keeps
    the final-state bootstrap but stops the trace before the next episode. Both
    outputs (advantages, lambda-return value targets) are detached. The final
    recorded transition stops the trace even if neither boundary flag is set.
    """
    if rewards.ndim != 1 or not rewards.numel() or any(
        x.shape != rewards.shape or x.device != rewards.device or not x.is_floating_point()
        for x in (rewards, values, next_values)
    ):
        raise ValueError("matching nonempty floating [T] transition vectors on one device required")
    if any(x.shape != rewards.shape or x.device != rewards.device or x.dtype != torch.bool
           for x in (terminated, truncated)):
        raise ValueError("aligned boolean terminated/truncated flags required")
    if not math.isfinite(gamma) or not math.isfinite(lam) or not 0 <= gamma <= 1 or not 0 <= lam <= 1:
        raise ValueError("finite gamma and lambda in [0,1] required")
    dtype = torch.float64 if any(x.dtype == torch.float64 for x in (rewards, values, next_values)) else torch.float32
    reward, value = rewards.to(dtype), values.to(dtype)
    following = next_values.to(dtype).masked_fill(terminated, 0)
    if any(not torch.isfinite(x).all() for x in (reward, value, following)):
        raise ValueError("finite rewards, values and nonterminal bootstrap values required")
    residual = reward + gamma * following - value
    result = torch.empty_like(residual)
    carry = residual.new_zeros(())
    continuing = ~(terminated | truncated)
    for t in range(len(residual) - 1, -1, -1):
        carry = residual[t] + gamma * lam * torch.where(continuing[t], carry, torch.zeros_like(carry))
        result[t] = carry
    return result, result + value


def pg_estimator_variance(probs, rewards, scores, baseline):
    """Exact per-sample variance of (r - b) * score for a small discrete policy, by enumeration.

    probs, rewards and scores give one entry per action; score is d log pi(a) / d theta along
    one parameter direction. The mean, sum_a pi(a) r(a) score(a), does not depend on b.
    """
    z = [(r - baseline) * g for r, g in zip(rewards, scores)]
    mean = sum(p * x for p, x in zip(probs, z))
    return sum(p * x * x for p, x in zip(probs, z)) - mean ** 2


def optimal_baseline(probs, rewards, scores):
    """The variance-minimizing constant baseline b* = E[r s^2] / E[s^2] (chapter 21.3)."""
    return (sum(p * r * g * g for p, r, g in zip(probs, rewards, scores))
            / sum(p * g * g for p, g in zip(probs, scores)))


def masked_log_ratio(logp, old_logp, mask):
    if logp.shape != old_logp.shape or not logp.is_floating_point() or not old_logp.is_floating_point():
        raise ValueError("current and old log probabilities must have matching shapes")
    if logp.device != old_logp.device or mask.device != logp.device:
        raise ValueError("log probabilities and response mask must share a device")
    valid = response_mask(mask, logp.shape)
    current = logp.masked_fill(~valid, 0).to(accumulation_dtype(logp))
    old = old_logp.detach().masked_fill(~valid, 0).to(current.dtype)
    if not torch.isfinite(current).all() or not torch.isfinite(old).all():
        raise ValueError("nonfinite log probability at a valid response token")
    return current - old, valid


def finite_ratio(log_ratio):
    ratio = log_ratio.exp()
    if not torch.isfinite(ratio).all():
        raise ValueError("importance ratio overflow on valid tokens; inspect policy lag")
    return ratio


def check_advantage(advantage, batch, eps):
    if advantage.shape != (batch,) or not torch.isfinite(advantage).all() or not 0 <= eps < 1:
        raise ValueError("finite [batch] advantage and 0 <= clipping epsilon < 1 required")


def on_policy_loss(logp, advantage, mask, normalization="sequence", max_length=None):
    valid = response_mask(mask, logp.shape)
    clean = logp.masked_fill(~valid, 0).to(accumulation_dtype(logp))
    check_advantage(advantage, len(logp), 0)
    if not torch.isfinite(clean).all():
        raise ValueError("nonfinite log probability at a valid response token")
    objective = clean * advantage.detach()[:, None]
    return aggregate_policy_objective(objective, valid, normalization, max_length)


def aggregate_policy_objective(objective, mask, normalization, max_length):
    if normalization == "sequence":
        return -(objective.sum(-1) / mask.sum(-1)).mean()
    if normalization == "constant":
        if max_length is None or max_length < 1 or torch.any(mask.sum(-1) > max_length):
            raise ValueError("fixed max_length must bound all valid response lengths")
        return -objective.sum() / (len(objective) * max_length)
    raise ValueError(normalization)


def clipped_policy_loss(logp, old_logp, advantage, mask, eps=0.2, normalization="sequence", max_length=None):
    # [B,T] token log probabilities; advantage [B].
    log_ratio, valid = masked_log_ratio(logp, old_logp, mask)
    check_advantage(advantage, len(logp), eps)
    ratio = finite_ratio(log_ratio)
    a = advantage.detach()[:, None]
    objective = torch.minimum(ratio * a, ratio.clamp(1-eps, 1+eps) * a).masked_fill(~valid, 0)
    return aggregate_policy_objective(objective, valid, normalization, max_length)


def gspo_loss(logp, old_logp, advantage, mask, eps=0.01):
    delta, valid = masked_log_ratio(logp, old_logp, mask)
    check_advantage(advantage, len(logp), eps)
    log_ratio = delta.sum(-1) / valid.sum(-1)
    ratio = finite_ratio(log_ratio)
    a = advantage.detach()
    return -torch.minimum(ratio * a, ratio.clamp(1-eps, 1+eps) * a).mean()


@torch.no_grad()
def sampled_kl_metric(logp, ref_logp, mask):
    """Nonnegative k3 diagnostic under samples from CURRENT policy, not an RL loss.

    On a fixed action sample, differentiating this scalar is not generally the
    gradient of the expected KL. Old-policy samples also change its expectation.
    """
    delta, valid = masked_log_ratio(ref_logp, logp, mask)
    values = torch.expm1(delta) - delta
    if not torch.isfinite(values).all():
        raise ValueError("nonfinite KL diagnostic on a valid token")
    return values.masked_fill(~valid, 0).sum() / valid.sum()


ARITHMETIC_FACTS = [(a, b) for a in range(10) for b in range(10)]


def arithmetic_reward(a, b, response):
    """(total, format): format needs an ASCII digit then newline; total also needs the right digit.

    str.isdigit() would also accept "²" (U+00B2) or Arabic-Indic digits, which int() rejects or misreads.
    """
    text = bytes(response).decode("utf-8", errors="replace")
    formatted = len(text) == 2 and text[0] in "0123456789" and text[1] == "\n"
    return float(formatted and int(text[0]) == (a + b) % 10), float(formatted)


@torch.no_grad()
def greedy_accuracy(model, facts):
    from .model import generate
    prompts = torch.tensor([list(f"{a}+{b}=".encode()) for a, b in facts])
    responses = generate(model, prompts, 2, temperature=0)[:, 4:]
    return sum(arithmetic_reward(a, b, r.tolist())[0] for (a, b), r in zip(facts, responses)) / len(facts)


def grpo_arithmetic_demo(steps=200, prompts_per_step=32, group=8, lr=1e-3, epochs_per_rollout=2,
                         microbatches=4, method="grpo", clip=0.2, seed=0, pretrain_steps=120, verbose=True):
    """A complete on-policy RL loop on a real (tiny) language model.

    Policy: TinyLM pretrained for `pretrain_steps` on synthetic arithmetic text (it knows
    the format but mostly guesses the answer). Each step: sample prompts, roll out a
    group of 2-byte answers per prompt, score them, compute group advantages, then run
    `epochs_per_rollout` passes of the clipped surrogate with microbatch accumulation.
    The first pass is on-policy (ratio = 1); later passes reuse stale rollouts.
    20 facts never appear as RL prompts (they do occur in the pretraining text), so their
    accuracy shows whether RL sharpened shared skill or only rewarded prompts.
    """
    import contextlib
    import io
    import random
    import tempfile
    from .model import Config, TinyLM, generate
    from .optim import AdamW, clip_grad_norm
    from .train import run
    if any(type(value) is not int or value < 1 for value in
           (steps, prompts_per_step, group, epochs_per_rollout, microbatches, pretrain_steps)):
        raise ValueError("positive integer step, group, epoch and microbatch counts required")
    if not math.isfinite(lr) or lr <= 0 or not 0 <= clip < 1 or method not in ('grpo', 'dr_grpo', 'maxrl', 'rloo', 'rft'):
        raise ValueError("positive finite learning rate, valid clip and supported advantage required")
    torch.manual_seed(seed)
    with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
        run(steps=pretrain_steps, out=folder)
        saved = torch.load(f"{folder}/checkpoint.pt", weights_only=True)
    policy = TinyLM(Config(**saved["config"]))
    policy.load_state_dict(saved["model"])
    facts = ARITHMETIC_FACTS[:]
    random.Random(seed).shuffle(facts)
    train_facts, heldout = facts[:80], facts[80:]
    opt = AdamW(policy.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    sampler, rng = torch.Generator().manual_seed(seed), random.Random(seed + 1)
    log = [dict(step=0, train_accuracy=greedy_accuracy(policy, train_facts),
                heldout_accuracy=greedy_accuracy(policy, heldout))]
    interval_norms = []  # pre-clip norm of EVERY optimizer update since the last log entry
    for step in range(1, steps + 1):
        chosen = [rng.choice(train_facts) for _ in range(prompts_per_step)]
        prompts = torch.tensor([list(f"{a}+{b}=".encode()) for a, b in chosen]).repeat_interleave(group, 0)
        sequences = generate(policy, prompts, 2, temperature=1.0, generator=sampler)
        responses = sequences[:, 4:]
        scored = [arithmetic_reward(a, b, r.tolist()) for (a, b), r in
                  zip([f for f in chosen for _ in range(group)], responses)]
        rewards = torch.tensor([r for r, _ in scored]).view(prompts_per_step, group)
        adv = advantages(rewards, method).reshape(-1)
        mask = torch.ones_like(responses, dtype=torch.float32)
        policy.train()

        def response_logits():
            # Logits at positions 3 and 4 predict the two response bytes.
            return policy(sequences[:, :-1])[:, 3:]

        with torch.no_grad():
            old_logp = token_logps(response_logits(), responses, mask)
        for _ in range(epochs_per_rollout):
            opt.zero_grad(set_to_none=True)
            for rows in torch.arange(len(sequences)).chunk(microbatches):
                logits = policy(sequences[rows, :-1])[:, 3:]
                logp = token_logps(logits, responses[rows], mask[rows])
                loss = clipped_policy_loss(logp, old_logp[rows], adv[rows], mask[rows], eps=clip)
                # Sequence-normalized loss: weight each microbatch by its share of sequences.
                (loss * len(rows) / len(sequences)).backward()
            grad_norm = clip_grad_norm(policy.parameters(), 1.0).item()
            interval_norms.append(grad_norm)
            opt.step()
        if step % 10 == 0 or step == steps:
            with torch.no_grad():
                entropy = token_entropy(response_logits(), mask).mean().item()
            log.append(dict(step=step, train_reward=rewards.mean().item(),
                            format_reward=sum(f for _, f in scored) / len(scored),
                            mixed_groups=((rewards.min(1).values == 0) & (rewards.max(1).values == 1)).float().mean().item(),
                            entropy=entropy, grad_norm=grad_norm,
                            max_grad_norm=max(interval_norms), clipped_updates=sum(n > 1.0 for n in interval_norms),
                            updates=len(interval_norms),
                            train_accuracy=greedy_accuracy(policy, train_facts),
                            heldout_accuracy=greedy_accuracy(policy, heldout)))
            interval_norms.clear()
            if verbose:
                print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in log[-1].items()}), flush=True)
    return dict(method=method, steps=steps, rollouts_per_step=prompts_per_step * group, log=log,
                scope="2-byte answers from a 125K-parameter model; shows the RL loop, not reasoning")


if __name__ == "__main__":
    print(json.dumps(grpo_arithmetic_demo(), indent=2))
