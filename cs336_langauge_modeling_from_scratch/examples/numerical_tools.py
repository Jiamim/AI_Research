"""Controlled numerical experiments for the 2026-10 review update.

No official assignment implementations or real-model performance claims.
"""
import math
import numpy as np
import torch
from torch.nn import functional as F


def dpo_shared_parameter_demo(step=0.1):
    theta = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    def logps(t):
        common = F.logsigmoid(-2*t)
        return common+F.logsigmoid(t), common+F.logsigmoid(-t)
    chosen, rejected = logps(theta)
    loss = -F.logsigmoid((chosen-chosen.detach())-(rejected-rejected.detach()))
    gradient, = torch.autograd.grad(loss, theta)
    updated = theta.detach()-step*gradient
    a, b = logps(updated)
    return dict(theta_after=updated.item(), chosen_before=chosen.exp().item(),
                rejected_before=rejected.exp().item(), chosen_after=a.exp().item(),
                rejected_after=b.exp().item(), remaining_after=torch.sigmoid(2*updated).item(),
                ratio_after=(a-b).exp().item(), scope="one-parameter normalized toy sequence model")


def corrected_kl(logits, reference, sampling):
    """Exact finite-sum IS k3 check at a FIXED prefix; sampling/ref have full support."""
    if logits.ndim != 1 or reference.shape != logits.shape or sampling.shape != logits.shape:
        raise ValueError("matching vectors required")
    for distribution in (reference, sampling):
        if not torch.isfinite(distribution).all() or torch.any(distribution<=0) or not torch.isclose(distribution.sum(),distribution.new_tensor(1.)):
            raise ValueError("finite normalized distributions with positive support required")
    if not torch.isfinite(logits).all():
        raise ValueError("finite logits required")
    logp = logits.log_softmax(-1)
    p = logp.exp()
    q, old = reference.detach(), sampling.detach()
    # Algebraically identical to the fully differentiable IS-k3 sum. Cancel p
    # before division to avoid 0 * inf when a tiny probability underflows.
    return (q - p + p * (logp - q.log())).sum()


def entropy_logit_differential(logits, delta):
    """First-order entropy change at a fixed prefix for a specified logit perturbation."""
    if logits.ndim != 1 or delta.shape != logits.shape or not torch.isfinite(logits).all() or not torch.isfinite(delta).all():
        raise ValueError("matching finite vectors required")
    logp = logits.log_softmax(-1)
    p = logp.exp()
    entropy = -(p * logp).sum()
    return -(p * (logp + entropy) * delta).sum()


def chapter_review_demo():
    """Small counterexamples for chapter-by-chapter review; no new LM training."""
    from .model import Config, TinyLM
    from .posttrain import clipped_policy_loss
    difference = {}
    with torch.random.fork_rng():
        for depth in (1, 2):
            torch.manual_seed(0)
            model = TinyLM(Config(vocab=32, width=16, layers=depth, heads=2,
                                 kv_heads=2, hidden=32, position="none")).eval().double()
            with torch.no_grad():
                a, b = torch.tensor([[3, 5, 9]]), torch.tensor([[5, 3, 9]])
                difference[str(depth)] = (model(a)[0, -1] - model(b)[0, -1]).abs().max().item()
    # Each positive-advantage sample shares this single parameter. One finite step
    # crosses the clip boundary, which is not a projection constraint.
    theta = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    logp = F.logsigmoid(theta).reshape(1, 1)
    loss = clipped_policy_loss(logp, logp.detach(), torch.ones(1), torch.ones(1, 1))
    gradient, = torch.autograd.grad(loss, theta)
    updated = theta.detach() - 2 * gradient
    ratio_after = (F.logsigmoid(updated) - F.logsigmoid(theta.detach())).exp()
    z = torch.tensor([.3, -.5, .8], dtype=torch.float64, requires_grad=True)
    delta = torch.tensor([.1, -.4, .2], dtype=torch.float64)
    p, lp = z.softmax(-1), z.log_softmax(-1)
    h = -(p * lp).sum()
    derivative, = torch.autograd.grad(h, z)
    flops = 2 * 70e9 * 15e12
    gpu_hours = flops / (989e12 * .4) / 3600
    return dict(
        nope_last_logit_difference=difference,
        hybrid_state_ratios={str(t): .25 + .25 / (t / 8192) for t in (8192, 131072)},
        ppo=dict(ratio_after=ratio_after.item(), clip_boundary=1.2, parameter_step=updated.item()),
        entropy=dict(analytic=entropy_logit_differential(z, delta).item(), autograd=(derivative * delta).sum().item()),
        annotation_cost=dict(flops=flops, gpu_hours=gpu_hours, ratio_to_6000_hours=gpu_hours / 6000),
        scope="CPU counterexamples, fixed-prefix derivatives and ideal cost accounting; no GPU performance or new LM training",
    )


def chapter_invariants_demo():
    """Small references for the guide's normalization, stability and evaluation examples.

    These compare algebraic identities and counterexamples, not trained-model quality.
    """
    from .extensions import clip_loss
    from .long_context import rope_inverse_frequencies, yarn_inverse_frequencies
    from .optim import AdamW

    logits = torch.tensor([1000., 998., -20., 999.], dtype=torch.float64, requires_grad=True)
    values = torch.tensor([[1., -1.], [2., 3.], [4., 0.], [-2., 5.]], dtype=torch.float64, requires_grad=True)
    normalizers = torch.stack([part.logsumexp(0) for part in logits.split(2)])
    local_outputs = torch.stack([part.softmax(0) @ value for part, value in zip(logits.split(2), values.split(2))])
    merged = normalizers.softmax(0) @ local_outputs
    dense = logits.softmax(0) @ values
    actual = torch.autograd.grad(merged.square().sum(), (logits, values))
    expected = torch.autograd.grad(dense.square().sum(), (logits, values))

    key = torch.tensor([2., 0.], dtype=torch.float64)
    state = torch.tensor([[.2, .3], [.4, -.1]], dtype=torch.float64)
    target = torch.tensor([1., -1.], dtype=torch.float64)
    residual = target - state.T @ key
    contractions = {}
    for rate in (.125, .25, .75):
        updated = state + rate * key[:, None] * residual[None, :]
        ratio = ((target - updated.T @ key).square().sum() / residual.square().sum()).item()
        contractions[str(rate)] = ratio

    images = torch.eye(2, dtype=torch.float64, requires_grad=True)
    texts = torch.tensor([[1., .3], [.4, 1.]], dtype=torch.float64)
    full = clip_loss(images, texts, temperature=1.)
    micro = sum(clip_loss(images[i:i+1], texts[i:i+1], temperature=1.) for i in range(2)) / 2

    probability = torch.tensor([.7, .3, 0., 0.], dtype=torch.float64)
    experts = {"uniform": 4 * (1 - (1 - .25) ** 3),
               "concentrated": (1 - (1 - probability) ** 3).sum().item()}
    gradient_old, gradient_new = torch.tensor([1., 1.]), torch.tensor([1., -.5])
    preconditioner = torch.diag(torch.tensor([.1, 10.]))

    parameter = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float64))
    optimizer = AdamW([parameter], lr=.1, weight_decay=0.)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    initial = parameter.detach().item()
    parameter.grad = None
    optimizer.step()
    absent = parameter.detach().item()
    parameter.grad = torch.zeros_like(parameter)
    optimizer.step()

    base = rope_inverse_frequencies(64, 50000.)
    scaled, _ = yarn_inverse_frequencies(64, 50000., 32, 4096, beta_fast=1, beta_slow=1)
    return dict(
        scope="CPU algebraic identities and counterexamples; no new large-model or GPU measurements",
        softmax_merge=dict(output_error=(merged-dense).abs().max().item(),
                           gradient_error=max((a-b).abs().max().item() for a,b in zip(actual,expected))),
        delta_rule=dict(key_squared_norm=key.square().sum().item(), squared_error_ratios=contractions),
        clip_microbatches=dict(full_loss=full.item(), separate_loss=micro.item(),
                              full_gradient_norm=torch.autograd.grad(full,images)[0].norm().item()),
        expert_union=experts,
        optimizer_zero_vs_none=dict(no_gradient_change=absent-initial,
                                    zero_gradient_change=parameter.detach().item()-absent),
        preconditioned_transfer=dict(raw_inner_product=(gradient_old@gradient_new).item(),
                                    weighted_inner_product=(gradient_old@preconditioner@gradient_new).item()),
        retained_token_mix=[1/11, 10/11],
        incomplete_k3=dict(expected_k3=math.log(2)-.5, true_kl=math.log(2)),
        normalized_recovery=dict(mean_of_ratios=.55, ratio_of_sums=2/11),
        padded_attention_pair_ratio=2*8192**2/(1024**2+8192**2),
        kimi_yarn=dict(unscaled_pairs=int((scaled==base).sum()), interpolated_pairs=int((scaled==base/32).sum())),
    )


def packing_equivalence_demo():
    """A one-head network: packed independent documents vs separate forwards.

    No positional encoding or padding query is needed in this small construction.
    Both paths share embedding/projection weights and score the same five targets.
    """
    from .objectives import packed_causal_mask, packed_next_token_mask, sft_loss
    rng = torch.Generator().manual_seed(40336)
    embedding = torch.randn(9, 6, dtype=torch.float64, generator=rng).requires_grad_()
    query, key, value = [torch.randn(6, 6, dtype=torch.float64, generator=rng).requires_grad_() for _ in range(3)]
    head = torch.randn(6, 9, dtype=torch.float64, generator=rng).requires_grad_()
    parameters = (embedding, query, key, value, head)
    # Reuse BOS/EOS IDs; distinct document IDs identify the two independent segments.
    ids = torch.tensor([[1, 3, 4, 2, 1, 5, 2]])
    documents = torch.tensor([[0, 0, 0, 0, 1, 1, 1]])
    allowed = packed_causal_mask(documents[:, :-1])[0]
    mask = packed_next_token_mask(documents)

    def logits(tokens, visibility=None):
        x = embedding[tokens]
        scores = (x @ query) @ (x @ key).T / math.sqrt(x.shape[-1])
        if visibility is None:
            visibility = torch.ones(len(tokens), len(tokens), dtype=torch.bool).tril()
        return (scores.masked_fill(~visibility, -torch.inf).softmax(-1) @ (x @ value)) @ head

    packed_logits = logits(ids[0, :-1], allowed)
    packed = sft_loss(packed_logits[None], ids[:, 1:], mask)
    separate = sum(F.cross_entropy(logits(part[:-1]), part[1:], reduction='sum')
                   for part in (ids[0, :4], ids[0, 4:])) / mask.sum()
    packed_gradient = torch.autograd.grad(packed, parameters)
    separate_gradient = torch.autograd.grad(separate, parameters)
    perturbed = ids[0, :-1].clone()
    perturbed[1:3] = torch.tensor([7, 8])
    isolated = logits(perturbed, allowed)[4:]
    return dict(targets=int(mask.sum()), cross_document_targets_suppressed=int((~mask).sum()),
                loss_error=abs(packed.item()-separate.item()),
                gradient_error=max((a-b).abs().max().item() for a,b in zip(packed_gradient,separate_gradient)),
                other_document_logit_error=(isolated-packed_logits[4:]).abs().max().item())


def chapter_interfaces_demo():
    """Small executable checks for data/model/evaluation interfaces, without training."""
    from .evaluation import paired_bootstrap, paired_cluster_bootstrap
    from .extensions import linear_attention_recurrent, linear_attention_explicit
    from .long_context import ntk_aware, position_interpolation, rope_inverse_frequencies, yarn_inverse_frequencies
    from .posttrain import generalized_advantages
    from .tokenizer import Tokenizer

    tokenizer = Tokenizer({i:bytes([i]) for i in range(256)}, [])
    ids = list('中🙂'.encode('utf-8'))
    attention = {}
    for dtype in (torch.float16, torch.float32, torch.float64):
        q = torch.ones(2, 16, dtype=dtype)
        k = torch.full_like(q, -20.)
        v = torch.ones(2, 3, dtype=dtype)
        recurrent, dense = linear_attention_recurrent(q,k,v), linear_attention_explicit(q,k,v)
        attention[str(dtype)] = dict(finite=bool(recurrent.isfinite().all() and dense.isfinite().all()),
                                    reference_error=(recurrent.double()-1).abs().max().item(),
                                    dense_error=(recurrent-dense).abs().max().item())
    a, b, clusters = [0.]*10, [1.]+[0.]*9, ['small']+['large']*9
    rewards = torch.tensor([0., 0., 1.], dtype=torch.float64)
    values = torch.tensor([.2, .4, .7], dtype=torch.float64)
    following = torch.tensor([.4, .7, 0.], dtype=torch.float64)
    last = torch.tensor([False, False, True])
    no_boundary = torch.zeros_like(last)
    terminal_adv, _ = generalized_advantages(rewards,values,following,last,no_boundary,lam=1.)
    following[-1] = .6
    truncated_adv, _ = generalized_advantages(rewards,values,following,no_boundary,last,lam=1.)
    posterior = torch.tensor([.9, .1], dtype=torch.float64)
    scores = torch.tensor([[2., -1.], [-1., 2.]], dtype=torch.float64)
    base = rope_inverse_frequencies(128, 10000.)
    yarn, _ = yarn_inverse_frequencies(128, 10000., 32., 4096)
    return dict(
        scope='Small CPU interface examples, exact counterexamples and cluster resampling; no model training',
        packing=packing_equivalence_demo(),
        utf8=dict(whole=tokenizer.decode(ids), streamed=''.join(tokenizer.decode_iterable(iter(ids))),
                  incorrect_tokenwise=''.join(tokenizer.decode([i]) for i in ids)),
        linear_attention=attention,
        initialization=dict(uniform_nll=math.log(2), symmetric_expected_nll=(-scores.log_softmax(-1)[:,0]).mean().item(),
                            biased_independent_nll=-math.log(.9)),
        calibration=dict(constant_score=.5, true_positive_fraction=posterior.mean().item(),
                         true_density_ratios=(posterior/(1-posterior)).tolist(), score_based_ratio=1.),
        bootstrap=dict(item=paired_cluster_bootstrap(a,b,clusters),
                       cluster=paired_cluster_bootstrap(a,b,clusters,weighting='cluster'),
                       wrongly_independent=paired_bootstrap(a,b)),
        gae=dict(lambda_one_terminal=terminal_adv.tolist(), lambda_one_timeout=truncated_adv.tolist()),
        rope_scaling=dict(setting=dict(dim=128, base=10000., factor=32., original_context=4096),
                          pi=(base/position_interpolation(128,10000.,32.)).tolist(),
                          ntk=(base/ntk_aware(128,10000.,32.)).tolist(), yarn=(base/yarn).tolist()),
    )


def mismatch_stats(train_logps, rollout_logps, mask):
    if train_logps.shape != rollout_logps.shape or mask.shape != train_logps.shape or not torch.any(mask):
        raise ValueError("matching tensors and a nonempty mask required")
    if not train_logps.is_floating_point() or not rollout_logps.is_floating_point() \
            or train_logps.device != rollout_logps.device or mask.device != train_logps.device \
            or not torch.all((mask == 0) | (mask == 1)):
        raise ValueError('floating log probabilities and a binary mask on one device required')
    current = train_logps[mask.bool()].detach().double()
    old = rollout_logps[mask.bool()].detach().double()
    if not torch.isfinite(current).all() or not torch.isfinite(old).all():
        raise ValueError("valid log probabilities must be finite")
    delta = current - old
    if not torch.isfinite(delta).all():
        raise ValueError('log-probability difference exceeds the FP64 range')
    absolute = delta.abs()
    return dict(mean=delta.mean().item(), absolute_p50=absolute.quantile(.5).item(),
                absolute_p95=absolute.quantile(.95).item(), absolute_p99=absolute.quantile(.99).item(),
                absolute_max=absolute.max().item())


def bootstrap_power_prediction(compute, losses, target, repeats=500, seed=19):
    """Two-parameter power model L=a*C^b; independent-row bootstrap + residual draw.

    Illustrates intervals, not an LM scaling law or a calibrated far-extrapolation bound.
    """
    c, y = np.asarray(compute, dtype=float), np.asarray(losses, dtype=float)
    if c.ndim!=1 or c.shape!=y.shape or len(c)<4 or np.any(c<=0) or np.any(y<=0) \
            or not math.isfinite(target) or target<=0 or type(repeats) is not int or repeats<10:
        raise ValueError("positive matching samples, target and at least 10 repeats required")
    if not np.isfinite(c).all() or not np.isfinite(y).all():
        raise ValueError("finite samples required")
    x, logy = np.log(c), np.log(y)
    design = np.column_stack((np.ones(len(c)), x))
    if np.linalg.matrix_rank(design) < 2:
        raise ValueError("distinct compute budgets are needed to identify a power-law exponent")
    coef = np.linalg.lstsq(design, logy, rcond=None)[0]
    residual = logy-design@coef
    residual = residual-residual.mean()
    rng = np.random.default_rng(seed)
    fitted, future = [], []
    for _ in range(repeats):
        indices = rng.integers(len(c), size=len(c))
        matrix = design[indices]
        if np.linalg.matrix_rank(matrix)<2:
            continue
        fit = np.linalg.lstsq(matrix, logy[indices], rcond=None)[0]
        value = np.array([1., math.log(target)])@fit
        fit_residual = logy[indices]-matrix@fit
        # exp(log-space trend) is a median, not the arithmetic conditional mean.
        # Empirical residual smearing estimates the mean on the original scale.
        fitted.append(math.exp(value)*np.exp(fit_residual).mean())
        future.append(math.exp(value+rng.choice(residual)))
    if len(fitted) < 2:
        raise ValueError("too few identifiable bootstrap resamples; use more distinct budgets or repeats")
    return dict(fitted_mean_interval=np.quantile(fitted,[.025,.975]).tolist(),
                new_run_prediction_interval=np.quantile(future,[.025,.975]).tolist(),
                prediction=float(np.exp(np.array([1.,math.log(target)])@coef)*np.exp(residual).mean()),
                scope="independent synthetic-row bootstrap; model misspecification not covered")


def fake_quantize_ste(x, scale=0.5, qmax=7):
    """Fixed scale, full identity STE (including clipped values) for teaching."""
    if not math.isfinite(scale) or scale<=0 or qmax<1:
        raise ValueError("positive finite scale and qmax required")
    quantized = (x/scale).round().clamp(-qmax,qmax)*scale
    return x+(quantized-x).detach()


def two_of_four_mask(weight):
    if weight.ndim!=2 or weight.shape[-1]%4 or not torch.isfinite(weight).all():
        raise ValueError("finite matrix with input dimension divisible by four required")
    groups = weight.reshape(weight.shape[0],-1,4)
    selected = groups.abs().topk(2,dim=-1).indices
    return torch.zeros_like(groups,dtype=torch.bool).scatter(-1,selected,True).reshape(weight.shape)


def sae_encode_decode(h, encoder, decoder):
    """Bias-free illustrative SAE, h [B,d], encoder [d,k], decoder [k,d]."""
    z = (h@encoder).relu()
    return z, z@decoder


def steer_feature_with_residual(h, z, decoder, feature, shift):
    modified = z.clone()
    modified[:,feature] += shift
    return h+(modified-z)@decoder


def muon_shape_rms(matrix):
    """SVD partial polar factor; zero singular directions stay zero, as in NS."""
    if matrix.ndim!=2 or min(matrix.shape)<1 or not torch.isfinite(matrix).all():
        raise ValueError("finite nonempty matrix required")
    u, s, vh = torch.linalg.svd(matrix,full_matrices=False)
    active = s > s.max()*max(matrix.shape)*torch.finfo(matrix.dtype).eps
    polar = (u*active)@vh
    scaled = .2*math.sqrt(max(matrix.shape))*polar
    return dict(unscaled_rms=polar.square().mean().sqrt().item(),
                scaled_rms=scaled.square().mean().sqrt().item())


def demo():
    rng = np.random.default_rng(11)
    c = np.geomspace(1,100,50)
    loss = 3*c**(-.2)*np.exp(rng.normal(0,.08,len(c)))
    return dict(dpo=dpo_shared_parameter_demo(),
                prediction=bootstrap_power_prediction(c,loss,150),
                fp4_bits=dict(mxfp4=4+8/32,nvfp4_local=4+8/16),
                muon=muon_shape_rms(torch.tensor([[1.,2.,3.],[2.,-1.,4.]],dtype=torch.float64)))


if __name__ == "__main__":
    import json
    print(json.dumps(demo(),indent=2))
