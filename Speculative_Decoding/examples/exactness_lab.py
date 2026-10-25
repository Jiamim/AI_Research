"""Exact finite probability experiments; no Monte Carlo or model downloads.

The functions enumerate complete output laws, including context dependence,
predictable drafting budgets, and EOS. Complexity is exponential: use tiny
vocabularies and horizons only. This is a mathematical oracle, not a fast decoder.
Run: python -m examples.exactness_lab
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction as F
from itertools import product
import json
from pathlib import Path


def distribution(values):
    row = tuple(F(value) for value in values)
    if not row or min(row) < 0 or sum(row) != 1:
        raise ValueError("Expected an exactly normalized nonnegative distribution")
    return row


def tv(left, right):
    """Total variation for finite laws with possibly different support keys."""
    return sum(abs(left.get(key, 0) - right.get(key, 0))
               for key in left.keys() | right.keys()) / 2


def verification_kernel(p, assumed_q):
    """K[y][x] is the output law conditional on proposed token y.

    For assumed_q[y] == 0 the event is null under the assumed proposal; define
    that row to be p. This completes a stochastic kernel for the TV proof, not
    permission to pass impossible sampled tokens to the production verifier.
    """
    p, q = distribution(p), distribution(assumed_q)
    if len(p) != len(q):
        raise ValueError("Vocabulary mismatch")
    residual = tuple(max(a-b, F(0)) for a, b in zip(p, q))
    mass = sum(residual)
    rows = []
    for y, qy in enumerate(q):
        if not qy:
            rows.append(p)
            continue
        acceptance = min(F(1), p[y] / qy)
        row = [F(0)] * len(p)
        row[y] = acceptance
        if acceptance < 1:
            assert mass > 0
            for x in range(len(p)):
                row[x] += (1-acceptance) * residual[x] / mass
        rows.append(distribution(row))
    return tuple(rows)


def one_step_law(p, actual_q, assumed_q):
    actual_q = distribution(actual_q)
    kernel = verification_kernel(p, assumed_q)
    if len(actual_q) != len(kernel):
        raise ValueError("Vocabulary mismatch")
    return tuple(sum(actual_q[y] * kernel[y][x] for y in range(len(actual_q)))
                 for x in range(len(actual_q)))


def mixture_conditional(components, weights, history):
    """Bayes conditionals of an iid-slot mixture with one Z shared by a block."""
    components = tuple(distribution(row) for row in components)
    weights = distribution(weights)
    if len(components) != len(weights) or len({len(row) for row in components}) != 1:
        raise ValueError("Mixture shape mismatch")
    posterior = list(weights)
    for token in history:
        for z, row in enumerate(components):
            posterior[z] *= row[token]
    evidence = sum(posterior)
    if not evidence:
        raise ValueError("Conditioning history has zero proposal probability")
    posterior = tuple(weight/evidence for weight in posterior)
    conditional = tuple(sum(weight*row[v] for weight, row in zip(posterior, components))
                        for v in range(len(components[0])))
    return posterior, conditional


def verify_events(candidates, qs, ps, eos_id=None):
    """Enumerate (emitted tuple, accepted count, conditional event mass).

    Every rejected path ends at its correction. Accepted EOS ends immediately;
    the bonus distribution is used only if the whole nonterminal draft survives.
    """
    alive = F(1)
    accepted = ()
    for i, y in enumerate(candidates):
        p, q = ps[i], qs[i]
        if q[y] <= 0:
            raise ValueError("Sampled candidate has zero assumed probability")
        a = min(F(1), p[y]/q[y])
        rejected_mass = alive * (1-a)
        if rejected_mass:
            residual = tuple(max(pv-qv, F(0)) for pv, qv in zip(p, q))
            total = sum(residual)
            assert total > 0
            for x, weight in enumerate(residual):
                if weight:
                    yield accepted + (x,), i, rejected_mass*weight/total
        alive *= a
        if not alive:
            return
        accepted += (y,)
        if y == eos_id:
            yield accepted, i+1, alive
            return
    for x, weight in enumerate(ps[-1]):
        if weight:
            yield accepted + (x,), len(candidates), alive*weight


def latent_block_experiment(mode):
    """Complete each variable-length round to a three-token target sample."""
    if mode not in {"component", "posterior", "wrong_marginal"}:
        raise ValueError("Unknown mixture correction mode")
    components = ((F(9, 10), F(1, 10)), (F(1, 10), F(9, 10)))
    weights, p = (F(1, 2), F(1, 2)), (F(1, 2), F(1, 2))
    law = defaultdict(F)
    expected_accepted = F(0)
    for z, q in enumerate(components):
        for candidates in product(range(2), repeat=2):
            draft_mass = weights[z] * q[candidates[0]] * q[candidates[1]]
            if mode == "component":
                qs = (q, q)
            elif mode == "posterior":
                qs = tuple(mixture_conditional(components, weights, candidates[:i])[1]
                           for i in range(2))
            else:
                qs = (p, p)  # Deliberately wrong: unconditional slot marginals.
            for emitted, accepted, event_mass in verify_events(candidates, qs, (p, p, p)):
                mass = draft_mass * event_mass
                expected_accepted += mass * accepted
                for tail in product(range(2), repeat=3-len(emitted)):
                    law[emitted + tail] += mass * F(1, 2)**len(tail)
    assert sum(law.values()) == 1
    return dict(law), expected_accepted


def draft_paths(draft, prefix, budget, *, eos_id=None, continue_before_draw=None):
    """Decide admission from the prefix BEFORE drawing the next candidate.

    continue_before_draw(committed_prefix, accepted_candidate_prefix) may inspect
    earlier draft tokens. It cannot inspect the next random sample or later ones.
    """
    def visit(candidates, qs, mass):
        terminal = candidates and candidates[-1] == eos_id
        stop = len(candidates) == budget or terminal
        if not stop and continue_before_draw is not None:
            stop = not continue_before_draw(prefix, candidates)
        if stop:
            yield candidates, qs, mass
            return
        q = distribution(draft(prefix + candidates))
        for y, weight in enumerate(q):
            if weight:
                yield from visit(candidates + (y,), qs + (q,), mass*weight)
    yield from visit((), (), F(1))


def enumerate_sequences(target, draft, prompt, max_new_tokens, gamma, *,
                        eos_id=None, continue_before_draw=None):
    """Enumerate ALL rounds, so variable emitted lengths do not bias the test."""
    if type(max_new_tokens) is not int or max_new_tokens < 0 or type(gamma) is not int or gamma < 0:
        raise ValueError("Use nonnegative integer horizon and gamma")
    prompt = tuple(prompt)
    if not prompt:
        raise ValueError("Need a nonempty prompt")
    law = defaultdict(F)

    def finish(prefix, mass):
        emitted = prefix[len(prompt):]
        remaining = max_new_tokens-len(emitted)
        if not remaining or (emitted and emitted[-1] == eos_id):
            law[emitted] += mass
            return
        budget = min(gamma, remaining-1)
        for candidates, qs, candidate_mass in draft_paths(
                draft, prefix, budget, eos_id=eos_id,
                continue_before_draw=continue_before_draw):
            ps = tuple(distribution(target(prefix + candidates[:i]))
                       for i in range(len(candidates)+1))
            for new, _, event_mass in verify_events(candidates, qs, ps, eos_id=eos_id):
                finish(prefix + new, mass*candidate_mass*event_mass)
    finish(prompt, F(1))
    assert sum(law.values()) == 1
    return dict(law)


def target_sequence_law(target, prompt, max_new_tokens, eos_id=None):
    """Independent autoregressive oracle, without any speculative functions."""
    prompt = tuple(prompt)
    law = defaultdict(F)

    def visit(prefix, mass):
        output = prefix[len(prompt):]
        if len(output) == max_new_tokens or (output and output[-1] == eos_id):
            law[output] += mass
            return
        for token, probability in enumerate(distribution(target(prefix))):
            if probability:
                visit(prefix + (token,), mass*probability)
    visit(prompt, F(1))
    return dict(law)


def acceptance_survival(target, draft, prompt, gamma):
    """Fixed-length, nonterminal chain: propagate unnormalized accepted mass."""
    frontier = {tuple(prompt): F(1)}
    survival = [F(1)]
    for _ in range(gamma):
        next_frontier = {}
        for prefix, mass in frontier.items():
            p, q = distribution(target(prefix)), distribution(draft(prefix))
            for token, overlap in enumerate(map(min, p, q)):
                if overlap:
                    next_frontier[prefix + (token,)] = mass*overlap
        frontier = next_frontier
        survival.append(sum(frontier.values(), F(0)))
    return tuple(survival)


def summarize_law(law):
    return {','.join(map(str, key)): str(value) for key, value in sorted(law.items())}


def run():
    target_fair = {key: F(1, 8) for key in product(range(2), repeat=3)}
    mixtures = {}
    for mode in ("component", "posterior", "wrong_marginal"):
        law, accepted = latent_block_experiment(mode)
        mixtures[mode] = dict(tv=str(tv(law, target_fair)),
                              expected_accepted=str(accepted), law=summarize_law(law))
    assert mixtures['component']['tv'] == mixtures['posterior']['tv'] == '0'
    assert mixtures['wrong_marginal']['tv'] == '8/25'
    p_rows = ((F(1, 2), F(1, 3), F(1, 6)),
              (F(1, 4), F(1, 4), F(1, 2)),
              (F(1, 3), F(1, 3), F(1, 3)))
    q_rows = ((F(1, 4), F(1, 2), F(1, 4)),
              (F(1, 2), F(1, 4), F(1, 4)),
              (F(1, 3), F(1, 3), F(1, 3)))
    target = lambda prefix: p_rows[prefix[-1]]
    draft = lambda prefix: q_rows[prefix[-1]]
    # Keep the first sampled token; use its value only to admit the NEXT draw.
    continue_before_draw = lambda prefix, candidates: not candidates or candidates[-1] != 0
    joint_checks = []
    for eos in (None, 2):
        expected = target_sequence_law(target, (0,), 4, eos)
        for gamma in (0, 1, 2, 3):
            for dynamic in (False, True):
                actual = enumerate_sequences(target, draft, (0,), 4, gamma, eos_id=eos,
                                             continue_before_draw=continue_before_draw if dynamic else None)
                assert actual == expected
                joint_checks.append(dict(eos=eos, gamma=gamma, dynamic=dynamic,
                                         support_size=len(actual), tv=str(tv(actual, expected))))
    binary = lambda prefix: (F(1, 2), F(1, 2))
    binary_draft = lambda prefix: (F(4, 5), F(1, 5))
    eos_law = enumerate_sequences(binary, binary_draft, (0,), 3, 2, eos_id=1)
    assumed = (F(1, 2), F(1, 2))
    actual = (F(3, 4), F(1, 4))
    perturbed = one_step_law(assumed, actual, assumed)
    assert perturbed == actual
    target_A, proposal_A = F(7, 10), F(1, 2)
    biased_A = proposal_A + (1-proposal_A)*target_A
    # The distribution of a surviving prefix differs from target on-policy data.
    conditional_target = lambda s: ((F(1, 2), F(1, 2)) if len(s) == 1 or s[-1] == 1
                                    else (F(1, 10), F(9, 10)))
    conditional_draft = lambda s: ((F(1, 2), F(1, 2)) if len(s) > 1 and s[-1] == 1
                                   else (F(9, 10), F(1, 10)))
    survival = acceptance_survival(conditional_target, conditional_draft, (0,), 2)
    assert survival == (F(1), F(3, 5), F(1, 5))
    return dict(
        scope="Finite rational arithmetic; no model, GPU, or asymptotic benchmark.",
        mixtures=mixtures, context_and_stopping_checks=joint_checks,
        eos_law=summarize_law(eos_law),
        proposal_mismatch=dict(target=list(map(str, assumed)),
                               actual_q=list(map(str, actual)),
                               output=list(map(str, perturbed)), tv='1/4'),
        biased_admission=dict(target_A=str(target_A), output_A=str(biased_A),
                              tv=str(abs(biased_A-target_A))),
        survival_measure=dict(survival=list(map(str, survival)),
                              conditional_acceptance=['3/5', '1/3'],
                              expected_emitted=str(sum(survival)),
                              wrong_target_prefix_estimate='49/25'),
        sequence_error_bound=[dict(horizon=n, epsilon='1/100',
                                   formula=f'1-(99/100)^{n}',
                                   upper_bound=float(1-F(99, 100)**n))
                              for n in (1, 8, 32, 128, 512)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parents[1]/'data/exactness_results.json')
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
