"""CPU mathematical and parser counterexamples; no MRI files or native imports."""
import math
import unittest
from mri_rl.root_cause_reward_replay_v1 import analyze_group, sample_advantages, ReadAudit

class ReplayTests(unittest.TestCase):
    def group(self, answers, lengths=None, calibration=None):
        return analyze_group(answers, lengths or [20]*len(answers), 'same', 'mri_visit_correspondence', calibration)

    def test_sample_sd_epsilon_matches_native(self):
        a=sample_advantages([.2,1.,.2,1.])
        expected=.4/(math.sqrt(.64/3)+1e-6)
        self.assertAlmostEqual(a[1],expected)
        self.assertEqual(sample_advantages([.1]*4),[0.]*4)

    def test_overlap_and_ordered_gate_are_different(self):
        g=self.group(['bad']*4)
        self.assertEqual(g['first_gate'],'format_not_all_full')
        self.assertEqual(g['failed_gates'],['format_not_all_full','task_accuracy_constant'])

    def test_correctness_varies_but_format_rejects(self):
        g=self.group([r'<think>x</think><answer>same</answer>',r'<think>x</think><answer>\boxed{different}</answer>']*2)
        self.assertTrue(g['task_accuracy_varies'])
        self.assertFalse(g['accepted'])
        self.assertEqual(g['first_gate'],'format_not_all_full')

    def test_format_only_variation_does_not_become_informative(self):
        g=self.group([r'<think>x</think><answer>different</answer>',r'<think>x</think><answer>\boxed{different}</answer>']*2)
        self.assertTrue(g['format_only_total_variation'])
        self.assertFalse(g['accepted'])
        self.assertTrue(g['total_reward_varies'])

    def test_binary_calibration_advantages_epsilon_free(self):
        texts=[r'<think>x</think><answer>\boxed{same}</answer>',r'<think>x</think><answer>\boxed{different}</answer>']*2
        g=self.group(texts,calibration={'p50':.25,'iqr':1.})
        self.assertTrue(g['accepted'])
        self.assertLess(g['raw_calibrated_epsilon_free_max_difference'],1e-14)
        self.assertGreater(g['raw_calibrated_max_difference'],0)

    def test_cap_is_not_length_penalty_threshold(self):
        g=self.group([r'<think>x</think><answer>\boxed{same}</answer>',r'<think>x</think><answer>\boxed{different}</answer>']*2,[256,20,256,20])
        self.assertEqual(g['overlong'],[0.]*4)
        self.assertTrue(g['accepted'])

    def test_forbidden_reads_fail_before_open(self):
        audit=ReadAudit()
        for path in ('/private/tmp/final100/receipt.json','/private/tmp/run/model.safetensors','/private/tmp/internal_dev/receipt.json'):
            with self.assertRaises(ValueError):audit.path(path)
        self.assertEqual(audit.files,{})

    def test_format_invalid_zero_is_not_valid_class_variation(self):
        g=self.group([r'<think>x</think><answer>\boxed{same}</answer>','bad']*2)
        self.assertTrue(g['task_accuracy_varies'])
        self.assertFalse(g['positive_format_valid_class_mix'])
        self.assertFalse(g['full_format_valid_class_mix'])
        self.assertEqual(g['native_accuracy_variation_source'],'only_format_invalid_zero_vs_valid_correct')

    def test_unboxed_valid_wrong_is_real_class_variation(self):
        g=self.group([r'<think>x</think><answer>\boxed{same}</answer>',r'<think>x</think><answer>different</answer>']*2)
        self.assertTrue(g['positive_format_valid_class_mix'])
        self.assertFalse(g['full_format_valid_class_mix'])
        self.assertEqual(g['native_accuracy_variation_source'],'valid_class_correct_and_wrong')

    def test_nonclass_tagged_answer_remains_unknown(self):
        g=self.group([r'<think>x</think><answer>\boxed{same}</answer>',r'<think>x</think><answer>uncertain</answer>']*2)
        self.assertTrue(g['task_accuracy_varies'])
        self.assertFalse(g['positive_format_valid_class_mix'])
        self.assertEqual(g['native_accuracy_variation_source'],'unknown_or_nonclass_zero_vs_valid_correct')

    def test_historical_gate_can_admit_correct_vs_full_format_nonclass(self):
        g=self.group([r'<think>x</think><answer>\boxed{same}</answer>',r'<think>x</think><answer>\boxed{uncertain}</answer>']*2)
        self.assertTrue(g['accepted'])
        self.assertFalse(g['positive_format_valid_class_mix'])
        self.assertEqual(g['native_accuracy_variation_source'],'unknown_or_nonclass_zero_vs_valid_correct')

    def test_bare_correct_box_is_diagnostic_only(self):
        g=self.group([r'\boxed{same}',r'\boxed{different}']*2)
        self.assertEqual(g['raw_accuracy'],[0.]*4)
        self.assertTrue(g['single_box_diagnostic_accuracy_varies'])
        self.assertFalse(g['accepted'])

if __name__=='__main__':unittest.main()
