import unittest
import numpy as np
from mri_rl.paired_signal_saved_answers_v1 import score_answer_context

class SavedAnswersTests(unittest.TestCase):
    def test_response_context_and_unequal_length_upper_bound(self):
        logits=np.array([[0.,-7.,8.,-10.],[0.,0.,5.,0.]])
        r=score_answer_context(logits,0,[2,2],[1,3,3,3,3])
        self.assertLess(r['correct_minus_wrong_mean_upper_bound'],0)
        self.assertIsNone(r['correct_minus_wrong_complete_label_margin'])
        self.assertEqual(r['first_token_margin'],-15)

    def test_single_token_labels_are_exact_in_actual_context(self):
        r=score_answer_context(np.array([[2.,-1.,0.]]),0,[0],[1])
        self.assertEqual(r['correct_minus_wrong_complete_label_margin'],-3)
        self.assertFalse(r['common_canonical_prefix_measured'])

    def test_invalid_or_unavailable_actual_continuation_fails(self):
        for values in [(np.array([[float('nan'),0]]),0,[0],[1]),(np.zeros((1,2)),0,[0,0],[1])]:
            with self.assertRaises(ValueError):score_answer_context(*values)

if __name__=='__main__':unittest.main()
