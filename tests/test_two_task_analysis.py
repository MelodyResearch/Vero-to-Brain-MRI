"""Synthetic scoring fixtures only; no MRI model result or patient inference."""
import copy
import importlib.util
import json
import pytest


def mod():
    assert importlib.util.find_spec('mri_rl.two_task_analysis') is not None,'Missing two-task aggregate helper'
    from mri_rl import two_task_analysis
    return two_task_analysis


def answer(label):return '<think>Observation.</think><answer>\\boxed{'+label+'}</answer>'


def rows_fixture():
    rows=[]
    # Three questions for p1 versus one for p2: row mean .75, patient mean .5.
    for key,p,label in [('s1','p1','FLAIR'),('s2','p1','FLAIR'),('s3','p1','FLAIR'),('s4','p2','contrast-enhanced T1')]:
        rows.append({'id':key,'data_source':'OpenNeuro_ds007045','ability':'mri_sequence',
                     'reward_model':{'ground_truth':label},'extra_info':{'reward_type':'mri_sequence','patient_key':p,'internal_split':'internal_dev','patient_lineage':'BIDS_SUBJECT_VERIFIED'}})
    # Two cycles, two queries and both same/different references per query.
    for cycle,people in [('c1',('a','b')),('c2',('c','d'))]:
        for p in people:
            for c in people:
                rows.append({'id':f'd{p}{c}','data_source':'OpenNeuro_ds004856','ability':'mri_visit_correspondence',
                    'reward_model':{'ground_truth':'same' if p==c else 'different'},
                    'extra_info':{'reward_type':'mri_visit_correspondence','patient_key':p,'candidate_patient_key':c,
                                  'statistical_cluster_key':cycle,'internal_split':'internal_dev','patient_lineage':'BIDS_SUBJECT_VERIFIED'}})
    return rows


def predictions(rows):return {r['id']:{'response':answer(r['reward_model']['ground_truth'])} for r in rows}


def test_unbalanced_questions_average_patients_cycles_and_invalid_missing_zero():
    m=mod();rows=rows_fixture();p=predictions(rows)
    p['s4']['response']='\\boxed{contrast-enhanced T1}' # malformed native tags => accuracy zero
    for key in ('daa','dab','dba','dbb'):p[key]['response']='bad'
    del p['s3']
    result=m.score_predictions(rows,p)
    sequence=result['public']['tasks']['mri_sequence'];dallas=result['public']['tasks']['mri_visit_correspondence']
    assert sequence['questions']==4 and sequence['patients']==2
    assert sequence['native_task_accuracy']==pytest.approx(1/3) # mean(2/3,0)
    assert sequence['question_native_task_accuracy']==.5 and sequence['missing_predictions']==1
    assert dallas['patients']==4 and dallas['independent_units']==2 and dallas['native_task_accuracy']==.5
    assert result['public']['equal_task_macro_native_accuracy']==pytest.approx(5/12)
    public=json.dumps(result['public'])
    assert 'response' not in public and 'patient_key' not in public and 'record_id' not in public and 'Observation.' not in public
    assert result['public']['independent_patient_generalization_claimed'] is False


def test_native_half_format_accuracy_is_not_silently_redefined():
    m=mod();rows=rows_fixture();p=predictions(rows);p['s1']['response']='<think>Observation.</think><answer>FLAIR</answer>'
    scored=m.score_predictions(rows,p)
    assert scored['private']['cases']['s1']['native_task_accuracy']==1
    assert scored['private']['cases']['s1']['format']==.5
    assert scored['private']['cases']['s1']['full_format_accuracy']==0


@pytest.mark.parametrize('bad',['split','target','source','cycle','output_gt','unknown_id'])
def test_corrupted_reference_output_or_cluster_fails_closed(bad):
    m=mod();rows=rows_fixture();p=predictions(rows)
    if bad=='split':rows[0]['extra_info']['internal_split']='train'
    if bad=='target':rows[4]['reward_model']['ground_truth']='different'
    if bad=='source':rows[0]['data_source']='OpenNeuro_ds004856'
    if bad=='cycle':rows[5]['extra_info']['statistical_cluster_key']='foreign'
    if bad=='output_gt':p['s1']['ground_truth']='contrast-enhanced T1'
    if bad=='unknown_id':p['unknown']={'response':answer('FLAIR')}
    with pytest.raises(ValueError):m.score_predictions(rows,p)


def test_paired_matrix_keeps_seeds_and_offdiagonal_delta_and_small_n():
    m=mod();rows=rows_fixture();base=predictions(rows);bad=copy.deepcopy(base)
    for k in ('daa','dab','dba','dbb'):bad[k]['response']='bad'
    scores={('B28',7):m.score_predictions(rows,base),('S_sequence',7):m.score_predictions(rows,bad)}
    matrix=m.build_matrix(scores,baseline_arm='B28',training_task_by_arm={'B28':None,'S_sequence':'mri_sequence'},n_bootstrap=50)
    cells=matrix['cells'];off=next(x for x in cells if x['arm']=='S_sequence' and x['evaluation_task']=='mri_visit_correspondence')
    assert off['seed']==7 and off['off_diagonal'] is True and off['paired_delta_from_common_start']==-.5
    assert off['independent_units']==2 and off['paired_interval']['subjects']==2
    assert off['descriptive_only'] is True
    assert next(x for x in matrix['arm_seed_summary'] if x['arm']=='S_sequence')['worst_task_delta_from_common_start']==-.5
    altered=copy.deepcopy(scores[('S_sequence',7)]);altered['private']['cases']['s1']['target']='contrast-enhanced T1'
    with pytest.raises(ValueError,match='target'):m.compare_scores(scores[('B28',7)],altered,n_bootstrap=50)


def test_controls_need_exact_explicit_transformed_target_reference():
    m=mod();rows=rows_fixture();p=predictions(rows);original=m.score_predictions(rows,p)
    replacement=copy.deepcopy(rows);replacement[0]['reward_model']['ground_truth']='contrast-enhanced T1'
    p2=predictions(replacement);control=m.score_predictions(replacement,p2)
    with pytest.raises(ValueError,match='target'):m.compare_scores(original,control,n_bootstrap=50)
    references={r['id']:r['reward_model']['ground_truth'] for r in replacement}
    contrast=m.compare_scores(original,control,n_bootstrap=50,explicit_control_targets=references)
    assert contrast['equal_task_macro_delta']==0


def test_dallas_unequal_questions_do_not_overweight_one_query_person():
    m=mod();rows=rows_fixture()
    for i in range(4):
        extra=copy.deepcopy(rows[4]);extra['id']=f'repeated-query-view-{i}';rows.append(extra)
    p=predictions(rows)
    for row in rows:
        if row['ability']=='mri_visit_correspondence' and row['extra_info']['patient_key']!='a':p[row['id']]['response']='invalid'
    result=m.score_predictions(rows,p)['public']['tasks']['mri_visit_correspondence']
    assert result['questions']==12 and result['patients']==4 and result['independent_units']==2
    assert result['native_task_accuracy']==.25 # cycle1=(1+0)/2; cycle2=(0+0)/2
    assert result['question_native_task_accuracy']==.5


def test_one_patient_or_cycle_does_not_invent_a_bootstrap_interval():
    m=mod();rows=[r for r in rows_fixture() if (r['ability']=='mri_sequence' and r['extra_info']['patient_key']=='p1') or (r['ability']=='mri_visit_correspondence' and r['extra_info']['statistical_cluster_key']=='c1')]
    scored=m.score_predictions(rows,predictions(rows));contrast=m.compare_scores(scored,scored,n_bootstrap=20)
    for task in contrast['tasks'].values():
        assert task['independent_units']==1 and task['paired_interval']['confidence_interval'] is None
        assert task['paired_interval']['status']=='NOT_ESTIMATED_TOO_FEW_UNITS'


def test_change_in_worst_task_accuracy_differs_from_worst_individual_task_drop():
    m=mod();rows=rows_fixture();base=predictions(rows);candidate=copy.deepcopy(base)
    # Dallas stays the worst task at .5. Sequence falls1->.5, so the
    # minimum-across-tasks accuracy stays .5 despite one task dropping .5.
    for key in ('daa','dab','dba','dbb'):
        base[key]['response']='invalid';candidate[key]['response']='invalid'
    candidate['s4']['response']='invalid'
    a=m.score_predictions(rows,base);b=m.score_predictions(rows,candidate)
    assert a['public']['worst_task_native_accuracy']==b['public']['worst_task_native_accuracy']==.5
    contrast=m.compare_scores(a,b,n_bootstrap=20)
    assert contrast['worst_task_delta']==0.
    assert contrast['worst_individual_task_delta']==-.5
    matrix=m.build_matrix({('B28',7):a,('U_raw',7):b},training_task_by_arm={'B28':None,'U_raw':None},n_bootstrap=20)
    summary=next(x for x in matrix['arm_seed_summary'] if x['arm']=='U_raw')
    assert summary['worst_task_delta_from_common_start']==0.
    assert summary['worst_individual_task_delta_from_common_start']==-.5
