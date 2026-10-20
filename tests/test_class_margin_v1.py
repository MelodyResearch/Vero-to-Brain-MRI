"""CPU TDD; no model imports, training, or saved-gradient regeneration."""
import unittest
import numpy as np
from mri_rl import class_margin_v1 as p


class MarginCPU(unittest.TestCase):
    def test_mean_sum_first_have_distinct_length_semantics(self):
        result = p.margin_scores([-1., -3.], [-2.]*5)
        self.assertEqual(result['mean'], 0.)
        self.assertEqual(result['sum'], 6.)
        self.assertEqual(result['first'], 1.)

    def test_derivative_is_negative_dot_not_advantage_conversion(self):
        result = p.margin_directions(-2., -1., 2, 5, first=-.25)
        self.assertEqual(result['mean'], -1.)
        self.assertEqual(result['sum'], 1.)
        self.assertEqual(result['first'], -.25)
        # Same sign correct-class change need not imply favorable margin.
        self.assertLess(p.margin_directions(.1, .4, 1, 1)['mean'], 0.)

    def test_single_token_metrics_share_their_class_gradients(self):
        result = p.margin_directions(.4, -.2, 1, 1)
        self.assertEqual(result['mean'], result['sum'])
        self.assertEqual(result['mean'], result['first'])
        with self.assertRaises(ValueError): p.margin_directions(.4, -.2, 2, 5)

    def test_prefix_is_exact_saved_reasoning_history_and_strict_class(self):
        text = r'<think>Visible asymmetries.</think><answer>\boxed{different}</answer>'
        self.assertEqual(p.saved_prefix(text, 'different'), r'<think>Visible asymmetries.</think><answer>\boxed{')
        with self.assertRaises(ValueError): p.saved_prefix(text, 'same')
        with self.assertRaises(ValueError): p.saved_prefix(r'<think>x</think><answer>\boxed{contrast-enhanced\ T1}</answer>', 'contrast-enhanced T1')

    def test_same_prefix_numeric_gates_do_not_claim_gradient_parity(self):
        self.assertTrue(p.score_alignment([-.1, -2.], [-.1, -2.])['within_tolerance'])
        with self.assertRaises(ValueError): p.score_alignment([-.1], [-.103])
        with self.assertRaises(ValueError): p.score_alignment([-.1], [-.1, -2.])

    def test_budget_is_fourteen_new_margin_backwards_plus_one_Dallas_qualification(self):
        result = p.budget()
        self.assertEqual(result['margin_backward_calls'], 14)
        self.assertEqual(result['language_model_calls'], 33)
        self.assertEqual(result['vision_embedding_calls'], 16)
        self.assertEqual(result['generation_calls'], 0)
        self.assertEqual(result['optimizer_updates'], 0)

    def test_qualification_uses_existing_distribution_metric_and_fixed_tolerance(self):
        self.assertTrue(p.qualification_gate({'max_full_vocab_normalized_logp_delta': .001}, .001, True))
        with self.assertRaises(ValueError): p.qualification_gate({'max_full_vocab_normalized_logp_delta': .003}, 0., True)
        with self.assertRaises(ValueError): p.qualification_gate({'max_full_vocab_normalized_logp_delta': 0.}, 0., False)

    def test_frozen_feature_fingerprint_and_distribution_interfaces(self):
        from mri_rl import native_probability_resolution_v1 as q
        from types import SimpleNamespace
        utility=SimpleNamespace(_fingerprints=lambda b,a:{k:(str(v.dtype),list(v.shape),v.tobytes().hex()) for k,v in a.items()})
        features={'inputs_embeds':np.ones((1,2,3),dtype=np.float32),'deepstack_visual_embeds':[np.ones((2,3),dtype=np.float32)],'position_ids':np.ones((3,1,2),dtype=np.int64)}
        value=q.feature_fingerprints(None,utility,features)
        self.assertEqual(len(value['arrays']),3)
        comparison=q.distribution_comparison(np.array([[1.,2.,3.]]),np.array([[1.,2.,3.]]),[1])
        self.assertTrue(p.qualification_gate(comparison,0.,True))
        self.assertEqual(comparison['max_selected_normalized_logp_delta'],0.)

    def test_token_boundaries_and_context_deduplication(self):
        from mri_rl.answer_likelihood import PREFIX
        group={'row': {'ability':'mri_sequence','extra_info':{'internal_split':'train'},'reward_model':{'ground_truth':'FLAIR'}},
               'raw': {'responses':[{'text': PREFIX+'FLAIR}</answer>'}]}}
        plan=p.token_plan(group,lambda s:list(map(ord,s)))
        self.assertEqual(len(plan['contexts']),1)
        self.assertEqual(len(plan['contexts'][0]['aliases']),2)
        with self.assertRaises(ValueError): p.token_plan(group,lambda s:[1])

    def test_saved_gradient_stream_dot_uses_all_tensors(self):
        result = p.directional_dot([('a', [2], np.array([1., 2.]), np.array([3., -1.]))])
        self.assertEqual(result['derivative_along_saved_descent'], -1.)
        self.assertEqual(result['tensors'], 1)


if __name__ == '__main__': unittest.main()
