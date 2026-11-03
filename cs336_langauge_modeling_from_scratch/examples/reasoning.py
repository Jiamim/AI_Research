"""Synthetic candidate generation and verifier selection: no real LM benchmark."""
import json
import math
import numpy as np


def iid_pass_probability(success_probability, k):
    if not 0 <= success_probability <= 1 or not isinstance(k, int) or k < 1:
        raise ValueError("probability in [0,1] and positive integer k required")
    if success_probability == 1:
        return 1.0
    # The direct form is exact for moderate p (e.g. 1-(1-0.25)**1 == 0.25); expm1/log1p keeps
    # precision only where it is needed, when p is so small that 1-p rounds toward 1.
    if success_probability < 1e-8:
        return -math.expm1(k * math.log1p(-success_probability))
    return 1 - (1 - success_probability) ** k


def select_candidate(scores):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or len(scores) == 0 or not np.isfinite(scores).all():
        raise ValueError("finite nonempty score vector required")
    return int(np.argmax(scores))


def verifier_experiment(seed=23, trials=20000, success_probability=.25, false_high_probability=.03, ks=(1, 4, 16, 64)):
    if type(trials) is not int or trials < 1 or not ks or not 0 < success_probability < 1 or not 0 <= false_high_probability <= 1 \
            or any(not isinstance(k, int) or k < 1 for k in ks):
        raise ValueError("invalid experiment settings")
    rng = np.random.default_rng(seed)
    maximum = max(ks)
    correct = rng.random((trials, maximum)) < success_probability
    false_high = (~correct) & (rng.random((trials, maximum)) < false_high_probability)
    # Correct answers score 1, verifier-fooling wrong answers 2, other wrong answers 0; tiny jitter breaks ties.
    scores = np.where(false_high, 2., np.where(correct, 1., 0.)) + rng.random((trials, maximum)) * 1e-4
    false_high_rate = (1 - success_probability) * false_high_probability
    records = []
    for k in ks:
        picked = scores[:, :k].argmax(-1)
        records.append(dict(k=k, oracle_coverage=float(correct[:, :k].any(-1).mean()),
                            selected_accuracy=float(correct[np.arange(trials), picked].mean()),
                            theoretical_iid_coverage=iid_pass_probability(success_probability, k),
                            theoretical_selected_accuracy=(1 - false_high_rate) ** k
                            - (1 - success_probability - false_high_rate) ** k))
    return dict(synthetic=True, seed=seed, trials=trials, success_probability=success_probability,
                false_high_probability_given_incorrect=false_high_probability, records=records)


if __name__ == "__main__":
    print(json.dumps(verifier_experiment(), indent=2))
