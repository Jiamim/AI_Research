"""Small, executable mechanisms used by the Chinese principles tutorial.

These are mathematical/CPU teaching experiments, not paper checkpoints or GPU
benchmarks. Run `python -m examples.principle_lab` to save their numeric results.
"""
from __future__ import annotations

from functools import lru_cache
from itertools import product
from pathlib import Path
import json
import math

import numpy as np
import torch

from .speculative_sampling import nonnegative_int, probability_vector, token_ids
from .tiny_transformer import TinyTransformer, token_tensor


def processed_distribution(logits, temperature=1.0, top_k=0, top_p=1.0, allowed=None):
    """Teaching order: grammar mask -> temperature -> top-k -> top-p.

    The boundary token that crosses top_p is retained; token-ID breaks ties.
    Real engines may have a different processor order. Always use their actual q.
    """
    z = np.asarray(logits, dtype=np.float64).copy()
    if z.ndim != 1 or z.size == 0 or not np.isfinite(z).all():
        raise ValueError('Need a finite, nonempty logits vector')
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Temperature must be finite and positive')
    top_k = nonnegative_int(top_k, 'top_k')
    if not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError('top_p must be in (0,1]')
    mask = np.ones(z.size, dtype=bool) if allowed is None else np.asarray(allowed, dtype=bool)
    if mask.shape != z.shape or not mask.any():
        raise ValueError('At least one vocabulary token must remain allowed')
    z[~mask] = -np.inf
    # Subtract before division, avoiding overflow from a large positive logit.
    with np.errstate(over='ignore'):
        z = (z - z.max()) / temperature
    order = np.argsort(-z, kind='stable')
    if top_k:
        z[order[min(top_k, len(z)):]] = -np.inf
    p = np.exp(z)
    p /= p.sum()
    if top_p < 1:
        order = np.argsort(-p, kind='stable')
        # Keep a token if the mass strictly before it is below the threshold.
        keep = np.cumsum(p[order]) - p[order] < top_p
        p[order[~keep]] = 0
        p /= p.sum()
    return p


def coupling_table(p, q):
    """Rows Y~q, columns X~p. Diagonal accepts, off-diagonal repairs."""
    p, q = probability_vector(p), probability_vector(q)
    if p.shape != q.shape:
        raise ValueError('Same vocabulary required')
    common = np.minimum(p, q)
    surplus = np.maximum(q-p, 0)
    deficit = np.maximum(p-q, 0)
    joint = np.diag(common)
    if deficit.sum() > 0:
        joint += np.outer(surplus, deficit / deficit.sum())
    return joint


def tree_layout(parents):
    """One topologically ordered root (-1), boolean ancestry mask and depth."""
    if not parents or parents[0] != -1:
        raise ValueError('Root must be node 0 with parent -1')
    mask = np.zeros((len(parents), len(parents)), dtype=bool)
    depths = np.zeros(len(parents), dtype=int)
    for i, parent in enumerate(parents):
        if i:
            parent = nonnegative_int(parent, 'parent')
            if parent >= i:
                raise ValueError('Each parent must precede its child')
            mask[i] = mask[parent]
            depths[i] = depths[parent] + 1
        mask[i, i] = True
    return mask, depths


def ancestor_path(parents, node):
    tree_layout(parents)
    if nonnegative_int(node, 'node') >= len(parents):
        raise ValueError('Node outside tree')
    path = []
    while node >= 0:
        path.append(node)
        node = parents[node]
    return path[::-1]


@torch.inference_mode()
def tree_forward(model, prompt, parents, nodes):
    """Actual multi-layer tree attention using TinyTransformer's frozen weights.

    nodes[0] must be prompt[-1]. Returns logits/cache in flattened tree order.
    Cache rows are [common prompt[:-1], nodes], NOT a valid linear KV sequence.
    """
    prompt, nodes = token_ids(prompt, model.vocab), token_ids(nodes, model.vocab)
    visible, depth = tree_layout(parents)
    if not prompt or len(nodes) != len(parents) or nodes[0] != prompt[-1]:
        raise ValueError('Tree root must equal the last prompt token')
    prefix_length = len(prompt)-1
    if prefix_length + int(depth.max()) >= model.position.num_embeddings:
        raise ValueError('Tree depth exceeds position capacity')
    cache = None
    if prefix_length:
        _, cache = model(token_tensor(prompt[:-1], model))
    device = next(model.parameters()).device
    ids = token_tensor(nodes, model)
    positions = torch.as_tensor(prefix_length + depth, device=device)
    x = model.embedding(ids) + model.position(positions)
    mask = torch.as_tensor(np.column_stack([np.ones((len(nodes), prefix_length), bool), visible]), device=device)
    result_cache = []
    for i, layer in enumerate(model.blocks):
        batch, count, width = x.shape
        qkv = layer.qkv(layer.norm1(x)).view(batch, count, 3, layer.heads, width//layer.heads)
        q, k, v = qkv.permute(2,0,3,1,4).unbind(0)
        if cache is not None:
            k = torch.cat([cache[i][0], k], dim=-2)
            v = torch.cat([cache[i][1], v], dim=-2)
        scores = q @ k.transpose(-2,-1) / math.sqrt(width//layer.heads)
        attention = scores.masked_fill(~mask, -torch.inf).softmax(-1)
        values = (attention @ v).transpose(1,2).reshape(batch,count,width)
        x = x + layer.proj(values)
        x = x + layer.mlp(layer.norm2(x))
        result_cache.append((k,v))
    return model.head(model.norm(x)), result_cache


def gather_tree_cache(cache, prompt_length, parents, node):
    """Select common history + one root-to-node path, in autoregressive order."""
    path = ancestor_path(parents, node)
    indices = list(range(prompt_length-1)) + [prompt_length-1+i for i in path]
    return [(k.index_select(-2, torch.tensor(indices, device=k.device)),
             v.index_select(-2, torch.tensor(indices, device=v.device))) for k,v in cache]


@torch.inference_mode()
def split_layers_forward(model, ids, early_layers):
    """Process lower layers serially, reuse their activations in batched upper layers.

    Fixed known tokens, no early-exit training or self-speculative sampler. Shows
    exactly which activations can be reused when early computations are identical.
    """
    ids = token_ids(ids, model.vocab)
    if not ids or not 0 < early_layers < len(model.blocks):
        raise ValueError('Need nonempty tokens and a split inside the model')
    if len(ids) > model.position.num_embeddings:
        raise ValueError('Sequence exceeds position capacity')
    caches = [None]*early_layers
    activations = []
    device = next(model.parameters()).device
    for pos, token in enumerate(ids):
        x = model.embedding(token_tensor([token], model)) + model.position(torch.tensor([pos], device=device))
        for i, layer in enumerate(model.blocks[:early_layers]):
            x, caches[i] = layer(x, caches[i])
        activations.append(x)
    x = torch.cat(activations, dim=1)
    for layer in model.blocks[early_layers:]:
        x, _ = layer(x, None)
    return model.head(model.norm(x))


def block_context_mask(context_length, anchors, block_size):
    """Queries are [anchor, masks...] in separate blocks; keys include target features.

    A block anchored at absolute token a can see target features at positions <a,
    plus all *masked-input* slots of its own block. Actual future tokens are absent.
    """
    context_length = nonnegative_int(context_length, 'context_length')
    block_size = nonnegative_int(block_size, 'block_size')
    anchors = token_ids(anchors)
    if block_size < 2 or not anchors or any(a > context_length for a in anchors):
        raise ValueError('Need valid anchors and block_size >= 2')
    rows = len(anchors)*block_size
    mask = np.zeros((rows, context_length+rows), dtype=bool)
    for b, anchor in enumerate(anchors):
        start, end = b*block_size, (b+1)*block_size
        mask[start:end, :anchor] = True
        mask[start:end, context_length+start:context_length+end] = True
    return mask


def factorized_joint(first, second):
    return np.outer(probability_vector(first), probability_vector(second))


def conditional_joint(first, next_given_first):
    first = probability_vector(first)
    rows = np.stack([probability_vector(row) for row in next_given_first])
    if rows.shape[0] != len(first):
        raise ValueError('One conditional distribution per first token required')
    return first[:,None]*rows


def joint_divergences(target, proposal):
    p, q = probability_vector(np.asarray(target).ravel()), probability_vector(np.asarray(proposal).ravel())
    return dict(overlap=float(np.minimum(p,q).sum()), tv=float(np.abs(p-q).sum()/2))


def expected_progress(conditional_acceptance):
    a = np.asarray(conditional_acceptance, dtype=float)
    if a.ndim != 1 or not np.isfinite(a).all() or ((a<0)|(a>1)).any():
        raise ValueError('Expected conditional acceptance probabilities in [0,1]')
    survival = np.cumprod(a)
    return survival, float(1+survival.sum())


def allocate_budget(acceptance_rows, cost_by_total_tokens):
    """Exhaustive tiny scheduler with estimates known BEFORE candidate sampling.

    Reference optimum for the given synthetic cost table; not a causal online
    scheduler if callers compute scores retrospectively from current candidates.
    """
    rows = [expected_progress(row)[0] for row in acceptance_rows]
    if not rows:
        raise ValueError('Need at least one active request')
    costs = np.asarray(cost_by_total_tokens, dtype=float)
    max_tokens = len(rows) + sum(len(row) for row in rows)
    if costs.ndim != 1 or len(costs) <= max_tokens or not np.isfinite(costs).all() or (costs<=0).any():
        raise ValueError('Positive cost for every total-token count required')
    best = None
    for lengths in product(*(range(len(row)+1) for row in rows)):
        total = len(rows)+sum(lengths)
        emitted = sum(1+row[:length].sum() for row,length in zip(rows,lengths))
        rate = float(emitted/costs[total])
        if best is None or rate > best['tokens_per_cost']:
            best = dict(lengths=list(lengths), expected_emitted=float(emitted),
                        total_verify_tokens=total, cost=float(costs[total]), tokens_per_cost=rate)
    return best


def optimal_tree_score(child_probabilities, node_budget, max_depth):
    """Tiny Sequoia-inspired DP, positional acceptance assumption, root included.

    p[k] is the probability of following child k (mutually exclusive), not a
    conditional acceptance test after rejecting its siblings. Toy fixed p only.
    """
    p = np.asarray(child_probabilities, dtype=float)
    node_budget = nonnegative_int(node_budget, 'node_budget')
    max_depth = nonnegative_int(max_depth, 'max_depth')
    if not node_budget or p.ndim != 1 or not len(p) or not np.isfinite(p).all() or (p<0).any() or p.sum()>1+1e-12:
        raise ValueError('Need root budget and nonnegative child masses summing to <=1')
    @lru_cache(None)
    def solve(n,d):
        if n==1 or d==0:
            return 1.0
        values = [0.0]*n
        for mass in p:
            new = values.copy()
            for budget in range(1,n):
                new[budget] = max(values[budget-size]+mass*solve(size,d-1) for size in range(1,budget+1))
                new[budget] = max(new[budget],values[budget])
            values = new
        return 1+values[n-1]
    return solve(node_budget,max_depth)


def dpace_loss(logits, labels, smoothing=0.1):
    """One training block, last dim vocabulary; weights detached, CE unsmoothed."""
    if not 0 <= smoothing <= 1:
        raise ValueError('smoothing must be in [0,1]')
    if logits.ndim != 2 or labels.shape != logits.shape[:1]:
        raise ValueError('Expected [slots,vocabulary] logits and [slots] labels')
    log_q = logits.log_softmax(-1).gather(-1, labels[:,None]).squeeze(-1)
    with torch.no_grad():
        smoothed = (1-smoothing)*log_q.exp()+smoothing
        prefix = smoothed.cumprod(0)
        weights = prefix.flip(0).cumsum(0).flip(0)
    return -(weights*log_q).sum(), weights


def loss_gradient_demo():
    p = torch.tensor([.8,.15,.05],dtype=torch.float64)
    output = {}
    for name in ['kl','tv','hybrid']:
        z = torch.zeros(3,dtype=torch.float64,requires_grad=True)
        q = z.softmax(-1)
        kl = (p*(p.log()-z.log_softmax(-1))).sum()
        tv = (p-q).abs().sum()/2
        loss = {'kl':kl,'tv':tv,'hybrid':.5*(kl+tv)}[name]
        loss.backward()
        output[name] = dict(value=float(loss.detach()), gradient=z.grad.tolist())
    labels = torch.tensor([0,1,2])
    z = torch.tensor([[2.,0.,0.],[0.,.4,0.],[0.,0.,-.2]],dtype=torch.float64,requires_grad=True)
    loss, weights = dpace_loss(z,labels)
    loss.backward()
    output['dpace'] = dict(weights=weights.tolist(), loss=float(loss.detach()), gradient=z.grad.tolist())
    return output


def run_all():
    torch.set_num_threads(1)
    p,q = np.array([.5,.3,.2]),np.array([.2,.5,.3])
    parents,nodes,prompt = [-1,0,0,1,1,2],[3,4,5,6,7,8],[1,2,3]
    torch.manual_seed(2026)
    model=TinyTransformer(width=16,heads=2,layers=3).double().eval()
    logits,cache=tree_forward(model,prompt,parents,nodes)
    errors=[]
    with torch.inference_mode():
        for node in range(len(nodes)):
            path=ancestor_path(parents,node)
            serial,_=model(token_tensor(prompt[:-1]+[nodes[i] for i in path],model))
            errors.append(float((logits[0,node]-serial[0,-1]).abs().max()))
        selected=4
        path=ancestor_path(parents,selected)
        gathered=gather_tree_cache(cache,len(prompt),parents,selected)
        continued,_=model(token_tensor([9],model),gathered)
        serial,_=model(token_tensor(prompt[:-1]+[nodes[i] for i in path]+[9],model))
        gather_error=float((continued[0,-1]-serial[0,-1]).abs().max())
        full,_=model(token_tensor([1,2,3,4],model))
        split=split_layers_forward(model,[1,2,3,4],1)
        split_error=float((full-split).abs().max())
    target=np.diag([.5,.5]);independent=factorized_joint([.5,.5],[.5,.5])
    conditional=conditional_joint([.5,.5],[[.95,.05],[.05,.95]])
    curves=[]
    for gamma in range(9):
        _,length=expected_progress([.8]*gamma)
        cost=1 if gamma==0 else 1+gamma*.135
        curves.append(dict(gamma=gamma, expected=length, cost=cost, speed=length/cost))
    costs=[1+.1*i for i in range(9)]
    return dict(
        note='Exact math and random-weight CPU mechanisms; no pretrained model or GPU benchmark.',
        coupling=dict(target=p.tolist(),proposal=q.tolist(),joint=coupling_table(p,q).tolist()),
        processors={key:processed_distribution([2.,1.,0.],**kw).tolist() for key,kw in
                    [('raw',{}),('cold',{'temperature':.5}),('top_k',{'top_k':2}),('top_p',{'top_p':.8}),('grammar',{'allowed':[False,True,True]})]},
        tree=dict(parents=parents,nodes=nodes,prompt=prompt,mask=tree_layout(parents)[0].astype(int).tolist(),
                  positions=(len(prompt)-1+tree_layout(parents)[1]).tolist(),max_logits_error=max(errors),gather_error=gather_error),
        layer_reuse=dict(max_logits_error=split_error),
        block_mask=block_context_mask(5,[2,4],3).astype(int).tolist(),
        dependence=dict(target=target.tolist(),independent=independent.tolist(),conditional=conditional.tolist(),
                        independent_overlap=joint_divergences(target,independent)['overlap'],conditional_overlap=joint_divergences(target,conditional)['overlap']),
        training=loss_gradient_demo(),
        scheduler=dict(allocation=allocate_budget([[.9,.9,.9],[.6,.4,.2]],costs),costs=costs,
                       selection_bias=dict(target_A=.7, biased_A=.5+.5*.7)),
        tree_budget=[dict(nodes=n,expected=optimal_tree_score([.6,.3],n,4)) for n in range(1,13)],
        speed_curve=curves)


if __name__=='__main__':
    result=run_all()
    output=Path(__file__).resolve().parents[1]/'data/principle_results.json'
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ['tree','layer_reuse','scheduler']},indent=2))
    print(f'Saved {output}')
