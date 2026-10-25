"""Exact arithmetic checks for the survey, not a general decoding implementation.

Enumerate a two-token, context-independent example from Block Verification
(https://arxiv.org/abs/2403.10444). Complete each emitted prefix to three tokens
using the target distribution; both verifiers must recover its full joint law.
"""
from fractions import Fraction as F
import itertools
import json
from pathlib import Path


def check():
    p, q = [F(1, 3), F(2, 3)], [F(2, 3), F(1, 3)]
    results = {}
    for method in ("token", "block"):
        law = {sequence: F(0) for sequence in itertools.product(range(2), repeat=3)}
        expected_accepted = F(0)
        for candidates in itertools.product(range(2), repeat=2):
            proposal_mass = q[candidates[0]] * q[candidates[1]]
            if method == "token":
                acceptance = [min(F(1), p[y] / q[y]) for y in candidates]
                stop = [1 - acceptance[0], acceptance[0] * (1 - acceptance[1]),
                        acceptance[0] * acceptance[1]]
                residuals = [[max(x - y, 0) for x, y in zip(p, q)]] * 2
            else:
                rho = [F(1)]
                for y in candidates:
                    rho.append(min(F(1), rho[-1] * p[y] / q[y]))
                mass = sum(max(rho[1] * x - y, 0) for x, y in zip(p, q))
                denominator = mass + 1 - rho[1]
                # This example has no degenerate intermediate h. A general
                # implementation must handle zero-mass events separately.
                assert denominator > 0
                h1 = mass / denominator
                # Longest passing prefix, with independent prefix tests.
                stop = [(1 - rho[2]) * (1 - h1), (1 - rho[2]) * h1, rho[2]]
                residuals = [[max(rho[i] * x - y, 0) for x, y in zip(p, q)]
                             for i in range(2)]
            assert sum(stop) == 1
            for accepted, event_probability in enumerate(stop):
                if event_probability == 0:
                    continue
                expected_accepted += proposal_mass * event_probability * accepted
                weights = p if accepted == 2 else residuals[accepted]
                assert sum(weights) > 0
                weights = [weight / sum(weights) for weight in weights]
                for repair in range(2):
                    for tail in itertools.product(range(2), repeat=2 - accepted):
                        mass = proposal_mass * event_probability * weights[repair]
                        for token in tail:
                            mass *= p[token]
                        law[candidates[:accepted] + (repair,) + tail] += mass
        target_law = {sequence: p[sequence[0]] * p[sequence[1]] * p[sequence[2]]
                      for sequence in law}
        assert law == target_law
        assert expected_accepted == (F(10, 9) if method == "token" else F(11, 9))
        results[method] = dict(
            expected_accepted=str(expected_accepted),
            expected_emitted=str(expected_accepted + 1),
            exact_joint_distribution_matches=True,
            law={"".join(map(str, sequence)): str(value) for sequence, value in law.items()},
        )
    results["fastmtp_arithmetic"] = {
        "2.03_vs_1.21": 2.03 / 1.21 - 1,
        "2.03_vs_1.67": 2.03 / 1.67 - 1,
        "2.03_vs_1.81": 2.03 / 1.81 - 1,
    }
    return results


if __name__ == "__main__":
    output = Path(__file__).resolve().parents[1] / "data" / "block_derivation_checks.json"
    result = check()
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
