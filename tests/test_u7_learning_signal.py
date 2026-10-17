"""Focused CPU checks for descriptive U7 artifact statistics, not training."""
import importlib.util
from pathlib import Path
import pytest


def module():
    p=Path(__file__).parents[1]/'scripts/diagnose_u7_learning_signal.py'
    assert p.is_file(),'U7 learning-signal artifact diagnostic missing'
    s=importlib.util.spec_from_file_location('u7_diag',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def test_cluster_randomization_and_paired_class_error_counts():
    m=module();x=m.cluster_uncertainty({'a':[-1.],'b':[-1.],'c':[-1.],'d':[-1.]})
    assert x['units']==4 and x['paired_delta']==-1 and x['exact_sign_flip_two_sided_p']==.125
    assert x['bootstrap_95_percentile_interval']==[-1.,-1.]
    z=m.cluster_uncertainty({'a':[0.],'b':[0.],'c':[0.],'d':[0.]})
    assert z['exact_sign_flip_two_sided_p']==1 and z['paired_delta']==0
    assert m.paired_errors([1,1,0,0],[1,0,1,0])=={'both_correct':1,'lost_correct':1,'gained_correct':1,'both_wrong':1}


def test_binary_affine_advantages_equal_without_epsilon_and_formula_with_epsilon():
    m=module();raw=[.2,.2,1.,1.]
    for scale in (.1,.5,2.):
        transformed=[scale*x+.3 for x in raw]
        assert m.standardize(raw,0)==pytest.approx(m.standardize(transformed,0),abs=1e-14)
        a=m.standardize(raw);b=m.standardize(transformed)
        factor=(m.sample_std(raw)+1e-6)/(m.sample_std(raw)+1e-6/scale)
        assert b==pytest.approx([x*factor for x in a],abs=1e-14)


def test_global_quartile_allocation_is_outcome_independent():
    m=module();q=[m.quartile(i,125) for i in range(125)]
    assert [q.count(i) for i in range(4)]==[32,31,31,31]
    assert q[0]==0 and q[-1]==3


def test_unique_source_counts_do_not_count_repeated_occurrences_as_new_images():
    m=module();g={'task':'mri_sequence','grades':[{'accuracy':0.,'format':1.,'total':.2},{'accuracy':1.,'format':1.,'total':1.}],
        'people':{'PERSON'},'query':'PERSON','image':'SAME_PAYLOAD','source_id':'SAME_SOURCE','updated':True,'informative':True,
        'variance':.32,'accuracy_variance':.5,'predictions':['flair','contrast-enhanced t1'],'advantages':[-1.,1.],
        'input_tokens':426,'generated_tokens':2,'tokens':428}
    out=m.summarize([g,g])
    assert out['attempted_groups']==2 and out['unique_source_rows_attempted']==1
    assert out['unique_source_rows_contributing_updates']==out['unique_image_payloads_contributing_updates']==1
    assert out['accepted_positive_advantage_answer_counts']=={'contrast-enhanced t1':2}


def test_actual_calibration_uses_literal_factor3_frozen_reward_formula():
    m=module();from mri_rl.reward_replay import _logistic
    frozen={'mri_sequence':{'p50':1.,'iqr':.8125},'mri_visit_correspondence':{'p50':.25,'iqr':1.}}
    expected={'mri_sequence':(0.02430880061411401,.5),'mri_visit_correspondence':(.320821300824607,.9046505351008906)}
    groups=[{'task':task,'informative':True,'grades':[{'accuracy':a,'format':1.,'overlong':0.,'total':.8*a+.2} for a in (0.,0.,0.,1.)]} for task in frozen]
    x=m.calibration_invariance(groups,frozen,'CPU_FIXTURE')
    for task,(lo,hi) in expected.items():
        p=frozen[task]
        assert _logistic(0.,p['p50'],p['iqr'])==pytest.approx(lo,abs=1e-15)
        assert _logistic(1.,p['p50'],p['iqr'])==pytest.approx(hi,abs=1e-15)
        assert x['actual_frozen_statistics'][task]['positive_reward_scale']==pytest.approx(hi-lo,abs=1e-15)
        raw=[.2,.2,.2,1.];transformed=[.8*lo+.2]*3+[.8*hi+.2]
        expected_epsilon=max(abs(a-b) for a,b in zip(m.standardize(raw),m.standardize(transformed)))
        assert x['actual_frozen_statistics'][task]['maximum_abs_advantage_change_at_actual_epsilon']==pytest.approx(expected_epsilon,abs=1e-14)
