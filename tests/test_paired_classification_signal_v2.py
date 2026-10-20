import math
import unittest
import numpy as np
from mri_rl.paired_classification_signal_v2 import classification_loss,score_derivatives,differentiable_classification_loss,auxiliary_objective,margins,update_kind
from mri_rl.paired_interaction_signal_v1 import pair_loss

class ClassMarginTests(unittest.TestCase):
    def test_actual_adam60_scores_expose_confident_T1_error(self):
        s=[[-.0005300811026209828,-1.4055701681752908],[-.000748788763299757,-1.3223209990657778]]
        m=margins(s);self.assertAlmostEqual(m[0],1.4050400870726698);self.assertAlmostEqual(m[1],-1.321572210302478)
        self.assertGreater(classification_loss(s),pair_loss(s))
        g=score_derivatives(s);self.assertGreater(abs(g[1][1]),abs(g[0][0]))
        changed=[[s[i][j]-.01*g[i][j] for j in range(2)] for i in range(2)]
        self.assertGreater(margins(changed)[0],m[0]);self.assertGreater(margins(changed)[1],m[1])

    def test_same_paired_separation_can_hide_wrong_T1(self):
        balanced=[[-1,-3],[-3,-1]];biased=[[-1,-9],[-1,-5]]
        self.assertAlmostEqual(pair_loss(balanced),pair_loss(biased))
        self.assertGreater(classification_loss(biased),classification_loss(balanced))

    def test_v2_detects_v1_counterexample(self):
        a=[[-1,-11],[-1,-10]];b=[[-1,-11.2],[-1,-10.1]]
        self.assertLess(pair_loss(b),pair_loss(a))
        self.assertGreater(classification_loss(b),classification_loss(a))

    def test_derivatives_match_finite_differences(self):
        s=[[-.5,-2],[-.8,-2.3]];g=score_derivatives(s);h=1e-5
        for i in range(2):
            for j in range(2):
                a=[r[:] for r in s];b=[r[:] for r in s];a[i][j]+=h;b[i][j]-=h
                self.assertAlmostEqual(g[i][j],(classification_loss(a)-classification_loss(b))/(2*h),places=8)

    def test_shared_prior_is_penalized_but_no_image_model_cannot_classify(self):
        biased=[[-1,-5],[-1,-5]];neutral=[[-1,-1],[-1,-1]]
        self.assertEqual(pair_loss(biased),pair_loss(neutral))
        self.assertGreater(classification_loss(biased),classification_loss(neutral))
        g=score_derivatives(biased)
        self.assertGreater(g[0][0]+g[1][0],0) # descent decreases a common FLAIR score bias
        self.assertEqual(classification_loss(neutral),math.log(2))
        self.assertEqual(sum(v for r in score_derivatives(neutral) for v in r),0)

    def test_shared_parameter_can_still_hurt_a_class(self):
        s=[[-.5,-1.905],[-.5,-1.821]];g=score_derivatives(s)
        # d mF /dtheta=100, d mT /dtheta=-1; combined descent can hurt T1.
        dtheta=-(g[0][0]*100+g[1][1]*-1)
        self.assertLess(-dtheta,0)

    def test_backend_expression_is_exact_and_does_not_invent_gspo_loss(self):
        s=[[-1,-3],[-.5,-2]]
        self.assertAlmostEqual(classification_loss(s),float(differentiable_classification_loss(s,np)))
        self.assertAlmostEqual(auxiliary_objective(0,s,np),.1*classification_loss(s))
        self.assertAlmostEqual(auxiliary_objective(2,s,np)-auxiliary_objective(0,s,np),2)

    def test_stable_extremes_and_invalid_inputs(self):
        self.assertTrue(math.isfinite(classification_loss([[-1,-10001],[-1,-10001]])))
        for s in [[],[[1,2]],[[1,float('nan')],[2,3]]]:
            with self.assertRaises(ValueError):classification_loss(s)

    def test_updates_are_counted_as_one_joint_stage_event(self):
        self.assertEqual(update_kind(True,True),'JOINT')
        self.assertEqual(update_kind(False,True),'AUXILIARY_ONLY')
        self.assertEqual(update_kind(True,False),'GSPO_ONLY')
        self.assertEqual(update_kind(False,False),'NO_UPDATE')

if __name__=='__main__':unittest.main()
