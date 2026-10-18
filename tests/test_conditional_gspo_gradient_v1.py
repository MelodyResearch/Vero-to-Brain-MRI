"""CPU-only conditional B3 TDD; no models or historical weight files read."""
import unittest
import numpy as np
from mri_rl import conditional_gspo_gradient_v1 as p


class ConditionalGSPOCPU(unittest.TestCase):
    def chunks(self, sensitivities, advantages, label_fraction=1.):
        responses = [-a/4*np.asarray(s, dtype=float) for a, s in zip(advantages, sensitivities)]
        return [{'name': 'CPU_TOY', 'group': sum(responses), 'responses': responses,
                 'label_responses': [v*label_fraction for v in responses]}]

    def test_actual_descent_direction_raises_correct_and_lowers_wrong(self):
        advantages = [.5, .5, .5, -1.5]
        result = p.directional_summary(self.chunks([[1., 0.]]*3+[[-1., 0.]], advantages), advantages, [True]*3+[False])
        np.testing.assert_allclose([r['mean_logp_directional_derivative'] for r in result['responses']], [.75]*3+[-.75])
        self.assertTrue(all(r['direction_matches_strict_saved_correctness'] for r in result['responses']))

    def test_nonzero_group_norm_is_not_directional_correctness(self):
        advantages = [.5, .5, .5, -1.5]
        result = p.directional_summary(self.chunks([[1., 0.]]*3+[[.1, 0.]], advantages), advantages, [True]*3+[False])
        self.assertGreater(result['group_l2'], 0)
        self.assertGreater(result['responses'][-1]['mean_logp_directional_derivative'], 0)
        self.assertFalse(result['responses'][-1]['direction_matches_strict_saved_correctness'])

    def test_signed_credit_can_cancel_and_label_plus_remainder_is_full(self):
        advantages = [.5, .5, .5, -1.5]
        result = p.directional_summary(self.chunks([[1., 0.]]*4, advantages, .25), advantages, [True]*3+[False])
        self.assertEqual(result['group_l2'], 0.)
        self.assertEqual(result['positive_negative_cosine'], -1.)
        self.assertTrue(all(r['mean_logp_directional_derivative'] == 0 for r in result['responses']))
        self.assertEqual(result['label_plus_remainder_closure_max_abs'], 0.)

    def test_strict_nonclass_label_is_not_salvaged(self):
        response = r'<think>Evidence.</think><answer>\boxed{contrast-enhanced\ T1}</answer>'
        value = p.strict_correctness('mri_sequence', response, 'FLAIR')
        self.assertFalse(value['valid_class']); self.assertFalse(value['correct'])
        self.assertEqual(value['format'], 1.)
        valid = p.strict_correctness('mri_visit_correspondence', r'<think>Evidence.</think><answer>\boxed{same}</answer>', 'same')
        self.assertTrue(valid['valid_class'] and valid['correct'])

    def test_conditional_worker_budget_is_24_backwards_and_16_own_old(self):
        result = p.budget(2)
        self.assertEqual(result['backward_calls'], 24)
        self.assertEqual(result['detached_old_forwards'], 16)
        self.assertEqual(result['full_scoring_forwards'], 40)
        self.assertEqual(result['generation_calls'], result['optimizer_updates'])
        with self.assertRaises(ValueError): p.budget(3)

    def test_own_old_current_alignment_is_checked_without_another_forward(self):
        self.assertEqual(p.old_current_alignment([-1., -2.], [-1., -2.])['max_abs_error'], 0.)
        with self.assertRaises(ValueError): p.old_current_alignment([-1., -2.00002], [-1., -2.])
        with self.assertRaises(ValueError): p.old_current_alignment([float('nan')], [-1.])

    def test_float32_cancellation_closure_uses_roundoff_bound_not_relative_group_norm(self):
        values = [np.array([1.], np.float32), np.array([1e-8], np.float32), np.array([-1.], np.float32), np.array([0.], np.float32)]
        accumulated = values[0]
        for value in values[1:]: accumulated = (accumulated+value).astype(np.float32)
        result = p.directional_summary([{'group': accumulated, 'responses': values, 'label_responses': values}], [.5, .5, .5, -1.5], [True]*3+[False])
        self.assertGreater(result['group_response_sum_closure_max_abs'], 1e-10)
        self.assertLessEqual(result['group_response_sum_closure_max_abs'], result['maximum_float32_sum_roundoff_bound'])
        self.assertEqual(result['remainder_definition'], 'REMAINDER_DEFINED_BY_DIFFERENCE')


if __name__ == '__main__': unittest.main()
