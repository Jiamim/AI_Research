"""Probability assumptions and single-source forecast metadata must remain explicit."""
import copy
from fractions import Fraction as F
import unittest

from examples.forecast_contract import load_forecasts, month_label, render_table, validate_forecasts
from examples.check_probability_claims import check, conditioned_block_law


class ClaimReviewTests(unittest.TestCase):
    def test_claim_checks_cover_reverse_conditioning_and_units(self):
        result=check()
        self.assertEqual(result['conditional_acceptance']['given_output_and_fixed_draft_state'],['2/5','1','1'])
        self.assertEqual(result['speed_vs_latency']['latency_reduction'],'3/13')
        self.assertTrue(result['single_run_time_identity'])

    def test_wrong_marginals_hide_a_joint_distribution_error(self):
        right=conditioned_block_law(True)
        wrong=conditioned_block_law(False)
        self.assertEqual(set(right.values()),{F(1,8)})
        for position in range(3):
            self.assertEqual(sum(v for k,v in wrong.items() if k[position]==0),F(1,2))
        self.assertNotEqual(right,wrong)


class ForecastContractTests(unittest.TestCase):
    def test_actual_ledger_has_consistent_dates_and_rendered_labels(self):
        data=load_forecasts()
        table=render_table(data)
        for row in data['forecasts']:
            self.assertIn(row['prediction'],table)
            self.assertIn(row['horizon'],table)
        self.assertEqual(month_label('2026-10-20',6),'2027-04')
        self.assertEqual(month_label('2026-10-20',18),'2028-04')

    def test_stale_horizon_and_missing_criteria_rejected(self):
        for mutation in ['horizon','criteria','duplicate']:
            data=copy.deepcopy(load_forecasts())
            if mutation=='horizon':data['forecasts'][0]['horizon']='2028-01 至 2029-01'
            elif mutation=='criteria':data['forecasts'][0]['success_criteria']=[]
            else:data['forecasts'][1]['id']=data['forecasts'][0]['id']
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                validate_forecasts(data)
