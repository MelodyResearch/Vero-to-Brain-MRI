
import copy
from mri_rl.c40_visual_controls import build_pairs, paired_greedy_summary, select_probe_rows, bound_candidate_functions
from mri_rl import answer_likelihood as al
import pytest

def fixtures():
 rows=[]
 for person in ('p0','p1'):
  for level in (45,91,136):
   for modality,target in (('flair','FLAIR'),('t1ce','contrast-enhanced T1')):
    rid=f'{person}_z{level:03d}_{modality}_sequence'
    rows.append({'id':rid,'ability':'mri_sequence','data_source':'OpenNeuro_ds007045','images':[rid+'.png'],
     'reward_model':{'ground_truth':target},'extra_info':{'patient_key':person,'internal_split':'internal_dev','patient_lineage':'BIDS_SUBJECT_VERIFIED','reward_type':'mri_sequence','question':'sequence?'}})
 for query in ('a','b'):
  for candidate in ('a','b'):
   rid=f'{query}-{candidate}'
   rows.append({'id':rid,'ability':'mri_visit_correspondence','data_source':'OpenNeuro_ds004856','images':[rid+'.png'],
    'reward_model':{'ground_truth':'same' if query==candidate else 'different'},
    'extra_info':{'patient_key':query,'candidate_patient_key':candidate,'statistical_cluster_key':'cycle0','internal_split':'internal_dev','patient_lineage':'BIDS_SUBJECT_VERIFIED','reward_type':'mri_visit_correspondence','question':'correspondence?','query_source_sha256':query+'1','candidate_source_sha256':candidate+'0'}})
 return rows

def response(label):
 return {'response':'<think>Evidence.</think><answer>\\boxed{'+label+'}</answer>'}

def test_pairs_derive_actual_displayed_gold_and_dallas_order_never_flips_truth():
 rows=fixtures();pairs=build_pairs(rows)
 assert len(pairs['mri_sequence'])==6 and len(pairs['mri_visit_correspondence'])==2
 for row in rows:
  if row['ability']=='mri_visit_correspondence':
   e=row['extra_info']
   assert row['reward_model']['ground_truth']==('same' if e['candidate_patient_key']==e['patient_key'] else 'different')
 assert pairs['mri_visit_correspondence'][0]['negative']['reward_model']['ground_truth']=='different'
 bad=copy.deepcopy(rows);bad[-1]['reward_model']['ground_truth']='different'
 with pytest.raises(ValueError):build_pairs(bad)

def test_no_image_dependence_when_answers_constant_and_invalid_pairs_not_dropped():
 rows=fixtures();pred={r['id']:response('FLAIR' if r['ability']=='mri_sequence' else 'different') for r in rows}
 result=paired_greedy_summary(rows,pred)
 assert result['mri_sequence']['pair_count']==6 and result['mri_sequence']['independent_units']==2
 assert result['mri_sequence']['correctly_separated_pairs']==0 and result['mri_sequence']['same_valid_answer_pairs']==6
 assert result['mri_visit_correspondence']['pair_count']==2 and result['mri_visit_correspondence']['independent_units']==1
 pred[rows[0]['id']]={'response':'broken'}
 result=paired_greedy_summary(rows,pred)
 assert result['mri_sequence']['pair_count']==6 and result['mri_sequence']['pairs_with_invalid']==1
 assert result['mri_sequence']['unit_averaged_direction_score']==0

def test_perfect_and_reversed_predictions_have_opposite_pair_separation():
 rows=fixtures();truth={r['id']:response(r['reward_model']['ground_truth']) for r in rows}
 result=paired_greedy_summary(rows,truth)
 assert all(v['unit_averaged_direction_score']==1 for v in result.values())
 reverse={'FLAIR':'contrast-enhanced T1','contrast-enhanced T1':'FLAIR','same':'different','different':'same'}
 result=paired_greedy_summary(rows,{r['id']:response(reverse[r['reward_model']['ground_truth']]) for r in rows})
 assert all(v['unit_averaged_direction_score']==-1 for v in result.values())

def test_subset_is_score_independent_balanced_fixed_level_and_complete_cycle():
 rows=fixtures();selected=select_probe_rows(rows)
 assert selected==select_probe_rows(list(reversed(rows)))
 assert len(selected)==8
 assert sum(r['ability']=='mri_sequence' for r in selected)==4
 assert all('_z091_' in r['id'] for r in selected if r['ability']=='mri_sequence')
 assert sum(r['reward_model']['ground_truth']=='same' for r in selected)==2
 bad=copy.deepcopy(rows);bad[0]['extra_info']['internal_split']='test'
 with pytest.raises(ValueError):select_probe_rows(bad)

def test_shared_likelihood_code_bound_to_dallas_labels_without_global_mutation():
 class Tok:
  def encode(self,text,add_special_tokens=False):return list(text.encode())
 part,compare=bound_candidate_functions(('different','same'))
 partition=part(Tok(),'same',prompt_length=8)
 assert partition['label_ids']==list(b'same')
 scores={label:al.summarize_tokens([-1.]*len(part(Tok(),label,prompt_length=8)['target_ids']),part(Tok(),label,prompt_length=8)) for label in ('different','same')}
 assert compare(scores)['mean_t1_minus_flair']==0
 assert al.LABELS==('FLAIR','contrast-enhanced T1')
 with pytest.raises(ValueError):part(Tok(),'FLAIR',prompt_length=8)
