"""Independent finite joint-law and perturbation checks for the advanced tutorial."""
from fractions import Fraction as F
from itertools import product
import unittest

from examples.exactness_lab import (
    acceptance_survival, enumerate_sequences, latent_block_experiment, mixture_conditional,
    one_step_law, run, target_sequence_law, tv, verification_kernel,
)


class ExactnessLabTests(unittest.TestCase):
    def test_shared_latent_requires_bayes_conditioning(self):
        components = ((F(9, 10), F(1, 10)), (F(1, 10), F(9, 10)))
        posterior, conditional = mixture_conditional(components, (F(1, 2),)*2, (0,))
        self.assertEqual(posterior, (F(9, 10), F(1, 10)))
        self.assertEqual(conditional, (F(41, 50), F(9, 50)))
        posterior, conditional = mixture_conditional(components, (F(1, 2),)*2, (0, 1))
        self.assertEqual(posterior, (F(1, 2),)*2)
        self.assertEqual(conditional, (F(1, 2),)*2)

    def test_component_and_marginal_conditionals_are_exact_but_cost_different(self):
        target = {key: F(1, 8) for key in product(range(2), repeat=3)}
        component, ac = latent_block_experiment('component')
        posterior, ap = latent_block_experiment('posterior')
        wrong, aw = latent_block_experiment('wrong_marginal')
        self.assertEqual(component, target)
        self.assertEqual(posterior, target)
        self.assertEqual((ac, ap, aw), (F(24, 25), F(42, 25), F(2)))
        self.assertEqual(tv(wrong, target), F(8, 25))
        # All one-position histograms pass despite the joint-law error.
        for i in range(3):
            self.assertEqual(sum(p for seq, p in wrong.items() if seq[i] == 0), F(1, 2))

    def test_every_context_eos_and_predictable_length_configuration(self):
        checks = run()['context_and_stopping_checks']
        self.assertEqual(len(checks), 16)
        self.assertEqual({row['tv'] for row in checks}, {'0'})
        self.assertEqual({row['support_size'] for row in checks}, {31, 81})

    def test_eos_has_variable_length_law_with_no_post_terminal_token(self):
        target = lambda prefix: (F(1, 2), F(1, 2))
        draft = lambda prefix: (F(4, 5), F(1, 5))
        expected = {(1,): F(1, 2), (0, 1): F(1, 4),
                    (0, 0, 1): F(1, 8), (0, 0, 0): F(1, 8)}
        actual = enumerate_sequences(target, draft, (0,), 3, 2, eos_id=1)
        self.assertEqual(actual, expected)
        self.assertEqual(actual, target_sequence_law(target, (0,), 3, eos_id=1))

    def test_zero_horizon_never_calls_either_model(self):
        def forbidden(_):
            raise AssertionError('Model called with zero horizon')
        self.assertEqual(enumerate_sequences(forbidden, forbidden, (0,), 0, 4), {(): F(1)})

    def test_support_mismatch_is_repaired_in_complete_sequence_law(self):
        target = lambda prefix: (F(1, 2), F(1, 2))
        draft = lambda prefix: (F(1), F(0))
        expected = {key: F(1, 16) for key in product(range(2), repeat=4)}
        self.assertEqual(enumerate_sequences(target, draft, (0,), 4, 3), expected)

    def test_markov_kernel_contracts_proposal_tv_including_zero_support(self):
        grid = [(F(i, 8), 1-F(i, 8)) for i in range(9)]
        for p, assumed in product(grid, repeat=2):
            kernel = verification_kernel(p, assumed)
            self.assertTrue(all(sum(row) == 1 for row in kernel))
            self.assertEqual(one_step_law(p, assumed, assumed), p)
            for actual in grid:
                output = one_step_law(p, actual, assumed)
                output_tv = sum(abs(a-b) for a, b in zip(output, p))/2
                input_tv = sum(abs(a-b) for a, b in zip(actual, assumed))/2
                self.assertLessEqual(output_tv, input_tv)

    def test_sequence_error_bound_can_be_tight(self):
        # Target always emits zero. Perturbed target independently emits one
        # with probability epsilon, so any occurrence is a distinguishable event.
        epsilon, horizon = F(1, 10), 5
        target = lambda prefix: (F(1), F(0))
        approximate = lambda prefix: (1-epsilon, epsilon)
        exact = target_sequence_law(target, (0,), horizon)
        perturbed = target_sequence_law(approximate, (0,), horizon)
        self.assertEqual(tv(exact, perturbed), 1-(1-epsilon)**horizon)

    def test_survival_uses_reweighted_prefix_law(self):
        target = lambda s: ((F(1, 2), F(1, 2)) if len(s) == 1 or s[-1] == 1
                            else (F(1, 10), F(9, 10)))
        draft = lambda s: ((F(1, 2), F(1, 2)) if len(s) > 1 and s[-1] == 1
                           else (F(9, 10), F(1, 10)))
        survival = acceptance_survival(target, draft, (0,), 2)
        self.assertEqual(survival, (F(1), F(3, 5), F(1, 5)))
        self.assertEqual(survival[2]/survival[1], F(1, 3))
        self.assertEqual(sum(survival), F(9, 5))


if __name__ == '__main__':
    unittest.main()
