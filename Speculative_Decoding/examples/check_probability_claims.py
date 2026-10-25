"""Independent exact checks of conditional acceptance and cost claims in chapters 20-22.

Small finite examples only; does not execute downloaded model code or implement
EDR training. Run: python -m examples.check_probability_claims
"""
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import json


def conditioned_block_law(use_actual_component):
    """Two latent components, two independent slot draws conditional on Z.

    Complete the variable-length speculative round to three target tokens.
    Comparing only emitted prefixes of varying lengths would be the wrong test.
    """
    p = [F(1, 2), F(1, 2)]
    components = [[F(9, 10), F(1, 10)], [F(1, 10), F(9, 10)]]
    law = {seq: F(0) for seq in product(range(2), repeat=3)}
    for q_actual in components:
        q_verify = q_actual if use_actual_component else p
        for draft in product(range(2), repeat=2):
            draft_mass = F(1, 2) * q_actual[draft[0]] * q_actual[draft[1]]
            alive = F(1)
            for rejected_at in range(3):
                if rejected_at < 2:
                    a = min(F(1), p[draft[rejected_at]] / q_verify[draft[rejected_at]])
                    event = alive * (1-a)
                    alive *= a
                    weights = [max(x-y, F(0)) for x, y in zip(p, q_verify)]
                else:
                    event, weights = alive, p
                if event == 0:
                    continue
                total = sum(weights)
                assert total > 0
                weights = [w/total for w in weights]
                for repair in range(2):
                    prefix = draft[:rejected_at] + (repair,)
                    for tail in product(range(2), repeat=3-len(prefix)):
                        law[prefix+tail] += draft_mass * event * weights[repair] * F(1, 2)**len(tail)
    assert sum(law.values()) == 1
    return law


def check():
    p, q = [F(1, 2), F(3, 10), F(1, 5)], [F(1, 5), F(1, 2), F(3, 10)]
    joint_accept = [min(a,b) for a,b in zip(p,q)]
    forward = [a/b for a,b in zip(joint_accept,q)]
    reverse = [a/b for a,b in zip(joint_accept,p)]
    assert forward == [1, F(3,5), F(2,3)]
    assert reverse == [F(2,5), 1, 1]
    correct = conditioned_block_law(True)
    wrong = conditioned_block_law(False)
    target = {seq: F(1,8) for seq in correct}
    assert correct == target
    wrong_tv = sum(abs(wrong[s]-target[s]) for s in target)/2
    assert wrong_tv == F(8,25)
    speed_gain = F(13,10)
    latency_reduction = 1-1/speed_gain
    assert latency_reduction == F(3,13)
    # Same one-response marginals do not imply independent response groups.
    correlated = {(0,0):F(1,2),(0,1):F(0),(1,0):F(0),(1,1):F(1,2)}
    group_tv = sum(abs(v-F(1,4)) for v in correlated.values())/2
    assert group_tv == F(1,2)
    # Sum of round times is exactly R * arithmetic mean round time in ONE run.
    emitted, durations = [1, 3, 2], [F(2), F(5), F(1)]
    n, rounds = sum(emitted), len(emitted)
    assert n/(F(n,rounds))*sum(durations)/rounds == sum(durations)
    return {
        'scope': 'Exact finite examples and unit checks; no model-code or GPU validation.',
        'conditional_acceptance': {'given_draft': list(map(str,forward)),
                                   'given_output_and_fixed_draft_state': list(map(str,reverse)),
                                   'target_only_bonus': 'accepted-draft probability is 0, not q/p'},
        'latent_proposal': {'actual_component_preserves_fixed_length_target': True,
                           'wrong_marginal_correction_tv': str(wrong_tv),
                           'correct_law': {''.join(map(str,k)):str(v) for k,v in correct.items()}},
        'speed_vs_latency': {'speed_factor':'13/10','latency_reduction':str(latency_reduction)},
        'group_marginals_not_joint': {'joint_tv': str(group_tv)},
        'single_run_time_identity': True,
    }


if __name__ == '__main__':
    result=check()
    path=Path(__file__).resolve().parents[1]/'data/claim_checks.json'
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))
