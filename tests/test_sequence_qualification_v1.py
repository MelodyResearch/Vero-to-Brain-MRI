"""Bounded CPU data qualification tests; no model imports."""
import copy
import pytest
from mri_rl.sequence_qualification_v1 import blind_pairs, source_sequence_render, lopo_nearest_neighbor


def pair_rows():
    return [{'id':f'person{p}_z{z:03}_{m}_sequence','ability':'mri_sequence',
             'images':[f'/source/person{p}_{m}.png'],
             'extra_info':{'patient_key':f'person{p}','internal_split':'train'},
             'reward_model':{'ground_truth':label}}
            for p in range(3) for z in (45,91,136)
            for m,label in [('flair','FLAIR'),('t1ce','contrast-enhanced T1')]]


def test_blind_pairs_are_complete_randomized_and_semantically_opaque():
    a=blind_pairs(pair_rows(),seed=932187);b=blind_pairs(pair_rows(),seed=932187)
    assert a==b and len(a)==9
    assert len({r['case_id'] for r in a})==9
    assert all(len(r['case_id'])==16 for r in a)
    assert {r['left']['reward_model']['ground_truth'] for r in a}=={'FLAIR','contrast-enhanced T1'}
    assert {r['left']['id'] for r in a}|{r['right']['id'] for r in a}=={r['id'] for r in pair_rows()}


def test_blinding_rejects_nontrain_or_missing_modality():
    rows=pair_rows();rows[0]['extra_info']['internal_split']='internal_dev'
    with pytest.raises(ValueError):blind_pairs(rows,seed=7)
    with pytest.raises(ValueError):blind_pairs(pair_rows()[:-1],seed=7)


def test_lopo_never_trains_on_same_person_and_returns_all_predictions():
    import numpy as np
    x=np.array([[0.],[1.],[0.1],[.9],[.2],[.8]])
    y=np.array([0,1,0,1,0,1]);p=['a','a','b','b','c','c']
    result=lopo_nearest_neighbor(x,y,p)
    assert result['correct']==6 and result['rows']==6
    assert result['same_patient_fit_overlap']==0
    assert len(result['prediction_PRIVATE'])==6
    assert all(p[i]!=p[j] for i,j in enumerate(result['neighbor_indices_PRIVATE']))
    assert result['fitted_parameters']==0


def test_source_renderer_exact_orientation_window_and_pixel_convention():
    import numpy as np
    from PIL import Image
    volume=np.arange(1,5*7*3+1,dtype=np.float32).reshape((5,7,3))
    rendered,window=source_sequence_render(volume,1,(10,14))
    lo,hi=np.percentile(volume[volume!=0][::16],[.5,99.5])
    expected=Image.fromarray(np.uint8(np.round(np.clip((np.rot90(volume[:,:,1])-lo)/(hi-lo),0,1)*255))).resize((10,14),Image.Resampling.BILINEAR).convert('RGB')
    assert rendered.tobytes()==expected.tobytes()
    assert window==[float(lo),float(hi)]


def test_unblind_keeps_abstention_and_rejects_changed_case_inventory():
    from mri_rl import sequence_qualification_v1 as q
    pairs=[{'case_id':'a','sides_PRIVATE':{'left':{'source_row_PRIVATE':{'reward_model':{'ground_truth':'FLAIR'}}},'right':{'source_row_PRIVATE':{'reward_model':{'ground_truth':'contrast-enhanced T1'}}}}},
           {'case_id':'b','sides_PRIVATE':{'left':{'source_row_PRIVATE':{'reward_model':{'ground_truth':'contrast-enhanced T1'}}},'right':{'source_row_PRIVATE':{'reward_model':{'ground_truth':'FLAIR'}}}}}]
    jud=[{'case_id':'a','left_modality_guess':'FLAIR','right_modality_guess':'T1ce','confidence':'high','coverage':'adequate','eligible_visual_review':True},
         {'case_id':'b','left_modality_guess':'ABSTAIN','right_modality_guess':'ABSTAIN','confidence':'low','coverage':'tiny','eligible_visual_review':False}]
    r=q.unblind_judgments(pairs,jud)
    assert (r['pairs'],r['answered_pairs'],r['correct_answered_pairs'],r['abstained_pairs'])==(2,1,1,1)
    assert r['clinical_scientific_admission'] is False
    with pytest.raises(ValueError):q.unblind_judgments(pairs,jud[:-1])
    wrong=copy.deepcopy(jud);wrong[0].update(left_modality_guess='T1ce',right_modality_guess='FLAIR')
    scored=q.unblind_judgments(pairs,wrong)
    assert scored['correct_answered_pairs']==0 and scored['incorrect_answered_pairs']==1
