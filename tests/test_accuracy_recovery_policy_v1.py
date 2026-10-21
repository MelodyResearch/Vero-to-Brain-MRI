"""Frozen conservative GO policy (Root spec hash 3107aa44…) and the saved-only score summary. Synthetic vectors/rows only; no model, no real outputs."""
import importlib.util
import json
from pathlib import Path
import pytest
from mri_rl import accuracy_recovery_eval_v1 as ae
from mri_rl import paired_z045_evaluation_v1 as ev


def load(name):
    spec=importlib.util.spec_from_file_location(name[:-3],Path(__file__).with_name(name));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


crit=load('test_accuracy_recovery_criterion_v1.py')
e=crit.e
reset=crit.reset


def v(f,t,c=.5,s=.5,d=.5):
    return ev.decision_vector((f+t)/2,f,t,c,(s+d)/2,s,d)


# ---- the five independent Codex counterexamples, assertions unchanged (copied from the reviewer's private fixture)
@pytest.mark.parametrize('reason,parent,a,b',[
    ('candidate_B_minus_A_negative',v(.5,.5,.5),v(.5,.5,.75),v(.5,.75,.625)),
    ('candidate_B_minus_Parent_negative',v(.5,.5,.875),v(.5,.5,.5),v(.5,.75,.75)),
    ('strict_sequence_and_T1_B_minus_Parent_negative',v(.5,1.,.875),v(.5,.5,.75),v(.5,.75,1.)),
    ('Dallas_same_recall_drops_despite_better_overall',v(.5,.5,.5,.75,.25),v(.5,.5,.625,.75,.25),v(.5,.75,.75,.5,.75)),
    ('FLAIR_B_minus_A_negative_despite_Parent_retained',v(.5,.5,.5),v(.75,.25,.625),v(.5,.75,.75)),
])
def test_frozen_protection_failures_cannot_receive_GO(reason,parent,a,b):
    result=ae.verdict(parent,a,b)
    assert result['decision']=='STOP_CANDIDATE', (reason,result)


# ---- the declared policy, exhaustively
BASE=v(.5,.5,.5,.5,.5)
GOOD=v(.5,.75,.75,.5,.5)                       # strict gains on seq BA, T1 recall, candidate BA; FLAIR/Dallas all unchanged


def test_all_conditions_holding_is_the_only_GO():
    out=ae.verdict(BASE,BASE,GOOD);assert out['decision']=='CONTINUE_CANDIDATE_REQUIRES_NEW_AUTHORIZATION' and out['benefit'] is True and not any(out['degrade'].values())
    assert set(out['B_minus_Parent'])==set(ev.KEYS) and out['training_authorized'] is False


@pytest.mark.parametrize('key',ae.BENEFIT_KEYS)
@pytest.mark.parametrize('reference',['A','parent'])
def test_each_benefit_metric_must_strictly_increase_against_both_A_and_the_Parent(key,reference):
    parent=BASE if reference=='A' else {**BASE,key:GOOD[key]-0.0}
    a=BASE if reference=='parent' else {**BASE,key:GOOD[key]}
    b=GOOD
    out=ae.verdict(parent,a,b);assert out['decision']=='STOP_CANDIDATE' and out['benefit'] is False,(key,reference,out['benefit_detail'])


@pytest.mark.parametrize('key',ae.PROTECTED_KEYS)
@pytest.mark.parametrize('reference',['A','parent'])
def test_each_protected_metric_must_not_fall_against_either_A_or_the_Parent(key,reference):
    higher={**BASE,key:min(1.,BASE[key]+.25)}
    parent=higher if reference=='parent' else BASE;a=higher if reference=='A' else BASE
    # keep derived overall fields internally consistent only where the vector validator requires finiteness/range
    out=ae.verdict(parent,a,GOOD);assert out['decision']=='STOP_CANDIDATE' and out['degrade'][f'{key}_below_{reference}'] is True


def test_equal_protected_metrics_do_not_block_and_counts_remain_diagnostic_only():
    out=ae.verdict(BASE,BASE,GOOD,None,{'admitted_groups':{'A':2,'B':1},'cosine':.0})
    assert out['decision']=='CONTINUE_CANDIDATE_REQUIRES_NEW_AUTHORIZATION' and out['training_signal_diagnostics']['diagnostic_only_never_a_criterion'] is True


def test_prospective_STOP_RULE_states_exactly_this_policy():
    text=json.dumps(ae.STOP_RULE)
    for needle in ('B-A AND B-Parent','STRICTLY increase','mean-candidate balanced accuracy','FLAIR recall, Dallas overall accuracy, Dallas same recall and Dallas different recall','NOT fall against A or against the Parent','3107aa44a6f3a00eaf715065522fc4f7d83faf89ce62ccb0dae21573df4e30d7'):
        assert needle in text
    assert tuple(ae.BENEFIT_KEYS)==('sequence_balanced_accuracy','T1_recall','sequence_candidate_balanced_accuracy')
    doc=(Path(ae.__file__).resolve().parents[2]/'research/VERO_TO_MRI_ACCURACY_RECOVERY_V1.md').read_text()
    assert 'B−Parent' in doc and 'Dallas same recall' in doc and '3107aa44' in doc


# ---- saved-only score summary
def kernel_rows(flair=12,t1=3,dallas=6):
    rows=[]
    for i,row in enumerate(e.rows_for(flair,t1,dallas)):
        answer=row['greedy']['text'].split('boxed{')[1].split('}')[0]
        rows.append(crit.kernel_row(row['task'],row['truth'],answer,[x['mean'] for x in row['candidates']['per_label']],f'r{i}'))
    return rows


def test_summary_has_mean_sum_first_class_recalls_margins_invalids_and_costs_from_saved_rows():
    rows=kernel_rows();rows[0]['candidates']['per_label'][0]['first']=-.5;rows[0]['candidates']['per_label'][1]['first']=-.5       # a 'first' tie -> INVALID
    rows[1]['candidates']['per_label'][0]['first']=-.5;rows[1]['candidates']['per_label'][1]['first']=-.5
    rows[-1]['greedy']['text']='not in the strict format'
    s=ae.score_summary(rows);seq=s['tasks']['mri_sequence'];dal=s['tasks']['mri_visit_correspondence']
    assert s['primary_candidate_readout']=='mean' and s['secondary_candidate_readouts']==['sum','first']
    assert seq['greedy']['class_recall']=={'FLAIR':1.,'contrast-enhanced T1':.25} and seq['greedy']['balanced_accuracy']==.625 and seq['greedy']['invalid']==0
    assert seq['candidate']['mean']['balanced_accuracy']==.625 and seq['candidate']['mean']['invalid']==0 and seq['candidate']['sum']['balanced_accuracy']==.625
    assert seq['candidate']['first']['invalid']==2 and seq['candidate']['first']['class_recall']['FLAIR']==pytest.approx(10/12)          # ties are INVALID and count against recall
    margin=seq['margin_correct_minus_wrong']['mean'];assert margin['FLAIR']==pytest.approx(.8) and margin['contrast-enhanced T1']==pytest.approx(-.4) and margin['aggregate']==pytest.approx(.2)
    assert dal['greedy']['invalid']==1 and s['costs']=={'greedy_calls':40,'candidate_score_API_calls':120,'generated_tokens':80,'teacherforced_input_positions':400}


def test_summary_requires_exact_40_rows_and_does_not_choose_a_best_readout():
    with pytest.raises(ValueError):ae.score_summary(kernel_rows()[:-1])
    s=ae.score_summary(kernel_rows());assert 'best' not in json.dumps(s).lower()


def test_numeric_delta_recurses_and_skips_non_numeric_leaves():
    left={'a':{'x':.5,'y':None,'z':'s'},'b':3,'flag':True};right={'a':{'x':.25,'y':1.,'z':'t'},'b':1,'flag':False}
    assert ae.numeric_delta(left,right)=={'a':{'x':.25},'b':2}


def test_analyze_reports_parent_A_B_summaries_deltas_costs_and_alias_has_no_new_cost(tmp_path):
    root=tmp_path/'eval'
    for label,args in (('parent',(12,3,6)),('B',(12,6,6))):
        directory=root/f'{label}_PRIVATE/rows_PRIVATE';directory.mkdir(parents=True)
        for i,payload in enumerate(kernel_rows(*args)):(directory/f'r{i}_PRIVATE.json').write_text(json.dumps(payload))
    analysis=ae.analyze(root,{'A':{'alias_of_parent':True},'B':{'alias_of_parent':False}})
    assert set(analysis['summaries'])=={'parent','A','B'} and analysis['summaries']['A']['costs']=='ALIAS_OF_PARENT_NO_NEW_CALLS' and analysis['summaries']['B']['costs']['greedy_calls']==40
    assert analysis['deltas']['A_minus_Parent']['tasks']['mri_sequence']['greedy']['balanced_accuracy']==0 and analysis['deltas']['B_minus_Parent']['tasks']['mri_sequence']['greedy']['class_recall']['contrast-enhanced T1']==pytest.approx(.25)
    assert 'costs' not in json.dumps(analysis['deltas']) and analysis['verdict']['decision']=='CONTINUE_CANDIDATE_REQUIRES_NEW_AUTHORIZATION' and 'no clinical' in analysis['limits']
