import pytest
from mri_rl import paired_z045_evaluation_v1 as p

def response(label):
    return '<think>compare contrast</think><answer>\\boxed{'+label+'}</answer>'

def test_strict_format_is_not_relaxed_to_salvage_truncation():
    assert p.strict_prediction(response('FLAIR'), 'mri_sequence') == 0
    assert p.strict_prediction('\\boxed{FLAIR}', 'mri_sequence') == -1
    assert p.strict_prediction('<think>unfinished', 'mri_sequence') == -1

def test_candidate_tie_is_invalid_and_margin_is_length_readout_specific():
    assert p.candidate_prediction([1., 1.]) == -1
    assert p.candidate_prediction([1., 2.]) == 1
    with pytest.raises(ValueError): p.candidate_prediction([float('nan'), 2.])

def test_go_requires_actual_t1_candidate_gain_and_dallas_retention():
    a=p.decision_vector(.5,.5,.5,.625,.5,.5,.5)
    b=p.decision_vector(.6,.5,.7,.7,.5,.5,.5)
    assert p.decide(a,b)['decision']=='ENGINEERING_GO_REQUIRES_NEW_AUTHORIZATION'
    b['Dallas_accuracy']=.4
    assert p.decide(a,b)['decision']=='STOP_SAME_CONFIGURATION'
    b['Dallas_accuracy']=.5; b['sequence_candidate_balanced_accuracy']=.625
    assert p.decide(a,b)['decision']=='STOP_SAME_CONFIGURATION'

def test_prediction_change_alone_is_not_successful_image_swap():
    x=p.swap_summary([0,0,-1,1],[1,1,1,1],[0,1,0,1])
    assert x['changed_predictions']==3
    assert x['valid_opposite_class_changes']==2
    assert x['both_images_correct']==1
    assert x['comparisons']==4

def test_protocol_never_uses_formal100_and_separates_precision():
    d=p.protocol()
    assert d['model_training_authorized'] is False
    assert d['holdout_status']=='NOT_ESTABLISHED'
    assert d['decoding']=={'temperature':0.,'max_tokens':256,'prompt':'UNCHANGED_ORIGINAL_VERO_SYSTEM_AND_TASK_QUESTION'}
    assert d['numeric_contract']['baseline_and_child_same_contract'] is True
    assert d['formal100_access'] is False
    assert d['development']['greedy_calls_per_checkpoint']==40
    assert d['development']['candidate_queries_per_checkpoint']==40
    assert d['development']['swap_new_model_calls']==0

def test_swap_is_an_alias_of_exact_opposite_real_input():
    import importlib.util
    from pathlib import Path
    spec=importlib.util.spec_from_file_location('prep',Path(__file__).parents[1]/'scripts/prepare_paired_z045_evaluation_v1.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    def row(i,label):return {'id':i,'images':[i+'.png'],'ability':'mri_sequence','extra_info':{'patient_key':'synthetic','level':45,'question':'Which modality?'},'reward_model':{'ground_truth':label}}
    pair=[row('a','FLAIR'),row('b','contrast-enhanced T1')]
    alias=m.swap_aliases(pair)
    assert alias[0]['reuse_real_id_PRIVATE']=='b'
    assert alias[0]['row_PRIVATE']['reward_model']['ground_truth']=='contrast-enhanced T1'
    assert alias[0]['row_PRIVATE']['images']==['b.png']

def test_missing_metric_never_becomes_zero_or_go():
    x=p.decision_vector(.5,.5,.5,.625,.5,.5,.5)
    del x['T1_recall']
    with pytest.raises(ValueError): p.decide(x,x)

def test_actual_cost_is_not_estimated_when_no_update_is_accepted():
    assert p.cost_per_update(707,42.,0)['tokens_per_update'] is None
    assert p.cost_per_update(707,42.,2)['seconds_per_update']==21.
    with pytest.raises(ValueError):p.cost_per_update(707,float('nan'),2)
