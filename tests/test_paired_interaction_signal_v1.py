import math
import unittest
from mri_rl.paired_interaction_signal_v1 import pair_loss, score_derivatives, attach_pairs, mean_label_score, differentiable_pair_loss, operation_budget, auxiliary_objective


class PairSignalTests(unittest.TestCase):
    def test_zero_correct_rollouts_do_not_remove_supervised_signal(self):
        s=[[-.1,-3.],[-.2,-2.]]  # both predict FLAIR, T1 image is wrong
        d=score_derivatives(s)
        self.assertGreater(abs(d[1][1]),0)
        self.assertLess(pair_loss([[s[i][j]-.01*d[i][j] for j in range(2)] for i in range(2)]),pair_loss(s))

    def test_common_class_prior_cancels(self):
        s=[[-1.,-3.],[-2.,-.5]]
        self.assertAlmostEqual(pair_loss(s),pair_loss([[r[0]+100,r[1]-90] for r in s]),places=12)
        g=score_derivatives(s)
        self.assertEqual(g[0][0]+g[1][0],0)
        self.assertEqual(g[0][1]+g[1][1],0)

    def test_analytic_derivative_matches_finite_difference(self):
        s=[[-.7,-3.],[-2.,-.4]];g=score_derivatives(s);eps=1e-5
        for i in range(2):
            for j in range(2):
                a=[r[:] for r in s];b=[r[:] for r in s];a[i][j]+=eps;b[i][j]-=eps
                self.assertAlmostEqual(g[i][j],(pair_loss(a)-pair_loss(b))/(2*eps),places=8)

    def test_image_insensitive_jacobians_cancel(self):
        g=score_derivatives([[-1.,-2.],[-1.,-2.]])
        self.assertEqual(sum(g[i][j]*([2.,3.][j]) for i in range(2) for j in range(2)),0)

    def test_separation_does_not_guarantee_t1_margin_or_accuracy(self):
        # mF=10+2theta,mT=-9-theta: descending pair loss increases theta;
        # mean margin improves while T1 gets worse and both still predict FLAIR.
        a=[[10.,0.],[9.,0.]];b=[[10.2,0.],[9.1,0.]]
        self.assertLess(pair_loss(b),pair_loss(a))
        self.assertLess(b[1][1]-b[1][0],a[1][1]-a[1][0])
        self.assertGreater(b[1][0],b[1][1])

    def test_only_label_span_is_scored(self):
        self.assertEqual(mean_label_score([-999,-2,-4,1000],1,3),-3)
        for args in [([],0,0),([float('nan')],0,1),([1],0,2)]:
            with self.assertRaises(ValueError):mean_label_score(*args)

    def test_pair_lookup_preserves_main_schedule_and_rejects_bad_pair(self):
        f={'id':'f','patient':'p','level':45,'label':'FLAIR','image_sha256':'a','question':'q'}
        t={**f,'id':'t','label':'contrast-enhanced T1','image_sha256':'b'}
        slots=[{'task':'sequence','record_id':'f'},{'task':'Dallas','record_id':'x'}]
        result=attach_pairs(slots,[f,t]);self.assertEqual(result[0]['main'],slots[0]);self.assertEqual(result[0]['pair_ids'],['f','t']);self.assertIsNone(result[1]['pair_ids'])
        with self.assertRaises(ValueError):attach_pairs(slots,[f,{**t,'image_sha256':'a'}])
        with self.assertRaises(ValueError):attach_pairs(slots,[f,{**t,'question':'label leak'}])

    def test_reject_nonfinite_and_wrong_shape(self):
        for s in [[],[[1,2]],[[1,float('inf')],[1,2]]]:
            with self.assertRaises(ValueError):pair_loss(s)

    def test_backend_formula_and_separate_auxiliary_cost(self):
        import numpy as np
        s=[[-1.,-3.],[-2.,-.5]]
        self.assertAlmostEqual(pair_loss(s),float(differentiable_pair_loss(s,np)))
        self.assertEqual(operation_budget(2,2)['GSPO_reserved_tokens'],9040)
        self.assertEqual(operation_budget(2,2)['combined_accounted_token_positions'],13560)
        self.assertEqual(operation_budget(18,14)['combined_accounted_token_positions'],111464)
        self.assertEqual(operation_budget(18,14)['auxiliary_backward_calls'],36)

    def test_inactive_gspo_is_not_relabelled_as_an_active_reward_group(self):
        import numpy as np
        s=[[-.1,-3.],[-.2,-2.]]
        total=auxiliary_objective(0.,s,np)
        self.assertAlmostEqual(total,.1*pair_loss(s))
        self.assertAlmostEqual(auxiliary_objective(2.,s,np)-total,2.)

if __name__=='__main__':unittest.main()
