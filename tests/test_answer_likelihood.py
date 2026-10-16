import math
import pytest
from mri_rl.answer_likelihood import partition_target, summarize_tokens, compare_candidates

def test_manifest_rejects_nontraining_and_mismatched_image_label():
    from mri_rl.answer_likelihood import training_pairs
    row={'id':'p_z045_flair_sequence','ability':'mri_sequence','data_source':'OpenNeuro_ds007045',
         'images':['p_z045_flair.png'],'reward_model':{'ground_truth':'FLAIR'},
         'extra_info':{'internal_split':'test','patient_lineage':'BIDS_SUBJECT_VERIFIED','patient_key':'p','question':'q'}}
    with pytest.raises(ValueError,match='training'):training_pairs([row],{'p'},45)
    row['extra_info']['internal_split']='train';row['reward_model']['ground_truth']='contrast-enhanced T1'
    with pytest.raises(ValueError,match='label'):training_pairs([row],{'p'},45)

def test_patient_pair_completeness_and_order_are_fixed():
    from mri_rl.answer_likelihood import training_pairs
    rows=[]
    for p in ['b','a']:
        for modality,label in [('flair','FLAIR'),('t1ce','contrast-enhanced T1')]:
            rows.append({'id':f'{p}_z045_{modality}_sequence','ability':'mri_sequence','data_source':'OpenNeuro_ds007045',
                'images':[f'{p}_z045_{modality}.png'],'reward_model':{'ground_truth':label},
                'extra_info':{'internal_split':'train','patient_lineage':'BIDS_SUBJECT_VERIFIED','patient_key':p,'question':'q'}})
    a=training_pairs(rows,{'a','b'},45)
    assert a==training_pairs(list(reversed(rows)),{'a','b'},45)
    assert len(a)==4 and a[0]['extra_info']['patient_key']==a[1]['extra_info']['patient_key']
    with pytest.raises(ValueError,match='pair'):training_pairs(rows[:-1],{'a','b'},45)


class CharacterTokenizer:
    def encode(self, s, add_special_tokens=False):
        return list(s.encode())


def test_exact_label_span_and_causal_positions():
    x = partition_target(CharacterTokenizer(), 'FLAIR', prompt_length=7)
    assert x['label_ids'] == list(b'FLAIR')
    assert x['label_logit_positions'][0] == 7 + len(x['prefix_ids']) - 1
    assert x['label_logit_positions'][-1] == 7 + len(x['prefix_ids']) + 4 - 1
    assert x['target_ids'] == x['prefix_ids'] + x['label_ids'] + x['suffix_ids']


def test_cross_boundary_token_is_rejected():
    class MergeTokenizer(CharacterTokenizer):
        def encode(self, s, add_special_tokens=False):
            return super().encode(s.replace('{F', 'Z'))
    with pytest.raises(ValueError, match='boundary'):
        partition_target(MergeTokenizer(), 'FLAIR', prompt_length=7)


def test_partitions_reconcile_and_invalid_logp_fails():
    x = {'prefix_ids': [1,2], 'label_ids':[3,4], 'suffix_ids':[5], 'target_ids':[1,2,3,4,5]}
    s = summarize_tokens([-0.1,-0.2,-1,-2,-0.3], x)
    assert s['label_logp_sum'] == -3
    assert s['label_logp_mean'] == -1.5
    assert s['fixed_nll_mean'] == pytest.approx(.2)
    assert s['whole_target_nll_mean'] == pytest.approx(.72)
    with pytest.raises(ValueError):summarize_tokens([float('nan')]*5,x)
    with pytest.raises(ValueError):summarize_tokens([.1]*5,x)
    with pytest.raises(ValueError):summarize_tokens([-1]*4,x)


def test_prespecified_length_aware_score_can_reverse_raw_preference():
    a={'FLAIR': {'label_logp_sum':-2.,'label_logp_mean':-1.},
       'contrast-enhanced T1':{'label_logp_sum':-3.,'label_logp_mean':-.6}}
    x=compare_candidates(a)
    assert x['raw_preferred']=='FLAIR'
    assert x['mean_preferred']=='contrast-enhanced T1'
    assert x['mean_t1_minus_flair']==pytest.approx(.4)
    assert x['raw_t1_minus_flair']==-1
    assert sum(x['mean_candidate_preferences'])==pytest.approx(1)
    assert 0<x['mean_candidate_entropy_nats']<=math.log(2)


def test_tie_is_not_implicitly_a_correct_class():
    x=compare_candidates({label:{'label_logp_sum':0.,'label_logp_mean':0.} for label in ('FLAIR','contrast-enhanced T1')})
    assert x['mean_preferred']=='TIE'
