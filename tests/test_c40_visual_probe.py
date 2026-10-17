
import copy,json
import pytest
from mri_rl import answer_likelihood as al
from mri_rl import paced_answer_likelihood as old
from mri_rl.c40_visual_probe import numeric_validator, require_training_done, validate_record_route, task_comparisons, validate_numeric_route
from mri_rl.c40_visual_controls import bound_candidate_functions,LABELS
class Tok:
 def encode(self,text,add_special_tokens=False):return list(text.encode())
def numeric(task):
 labels=LABELS[task];part,compare=bound_candidate_functions(labels);scores={}
 for label in labels:
  p=part(Tok(),label,prompt_length=5);values=[-1.]*len(p['target_ids'])
  scores[label]={**al.summarize_tokens(values,p),'partition':p,'target_token_logps':values,'first_label_candidate_logps':[-1.,-1.]}
 return {'category':task,'prompt_ids':[1]*5,'candidate_scores':scores,'prefix_oracle_first_logps':[-1.,-1.],'prefix_max_disagreement':0.,'comparisons':compare(scores)}
def test_exact_existing_numeric_validator_code_reused_for_both_tasks():
 for task in LABELS:
  fn=numeric_validator(task);assert fn.__code__ is old.validate_numeric_record.__code__
  fn(numeric(task),Tok())
 assert al.LABELS==LABELS['mri_sequence']
def test_wrong_task_labels_alignment_and_changed_scores_rejected():
 x=numeric('mri_visit_correspondence')
 with pytest.raises(ValueError):numeric_validator('mri_sequence')(x,Tok())
 for field in ('positions','mean','oracle'):
  y=copy.deepcopy(x)
  if field=='positions':y['candidate_scores']['same']['partition']['label_logit_positions'][0]+=1
  elif field=='mean':y['candidate_scores']['same']['label_logp_mean']+=.01
  else:y['prefix_oracle_first_logps'][0]+=.01
  with pytest.raises(ValueError):numeric_validator('mri_visit_correspondence')(y,Tok())
def test_completion_requires_all_original21_terminal_and_no_halt():
 ids=[f'rl:{i}' for i in range(20)]
 c={'stages':{i:{'state':'DONE','summary':{'full_budget_terminal_verified':True}} for i in ids}}
 c['stages']['rl:U_raw:7']={'state':'DONE','summary':{'full_budget_terminal_verified':True}}
 require_training_done(c,ids)
 for bad in ('missing','running','halted','bounded'):
  d=copy.deepcopy(c)
  if bad=='missing':d['stages'].pop(ids[0])
  elif bad=='running':d['stages'][ids[0]]['state']='RUNNING'
  elif bad=='halted':d['halted']='failure'
  else:d['stages'][ids[0]]['summary']['full_budget_terminal_verified']=False
  with pytest.raises(ValueError):require_training_done(d,ids)
def test_route_does_not_treat_noimage_as_a_labeled_accuracy_case():
 r={'category':'mri_visit_correspondence','candidate_labels':['different','same'],'condition':'no_image','image':None,'image_sha256':None,'visible_target_PRIVATE':None}
 validate_record_route(r)
 for update in ({'candidate_labels':['same','different']},{'visible_target_PRIVATE':'same'},{'image':'a.png'},{'category':'new_task'}):
  bad={**r,**update}
  with pytest.raises(ValueError):validate_record_route(bad)


def test_task_named_comparison_fields_roundtrip_without_reversing_sign():
 _,compare=bound_candidate_functions(LABELS['mri_visit_correspondence'])
 scores={'different':{'label_logp_sum':-3.,'label_logp_mean':-2.},'same':{'label_logp_sum':-1.,'label_logp_mean':-1.}}
 original=compare(scores);named=task_comparisons('mri_visit_correspondence',original)
 assert named['mean_same_minus_different']==1 and named['raw_same_minus_different']==2
 assert not any('t1' in k or 'flair' in k for k in named)
 assert task_comparisons('mri_visit_correspondence',named,to_legacy=True)==original
 assert task_comparisons('mri_sequence',original)==original

def test_shared_durable_prefix_rejects_torn_or_changed_route(tmp_path):
 from types import FunctionType
 base={'record_id':'no_image','category':'mri_visit_correspondence','candidate_labels':['different','same'],
       'condition':'no_image','image':None,'image_sha256':None,'visible_target_PRIVATE':None}
 row={**numeric('mri_visit_correspondence'),**base,'response':'saved','run_identity':'frozen'}
 row['comparisons']=task_comparisons('mri_visit_correspondence',row['comparisons'])
 path=tmp_path/'outputs.jsonl';path.write_text(json.dumps(row)+'\n')
 prefix=old.completed_prefix;fn=FunctionType(prefix.__code__,{**prefix.__globals__,'validate_numeric_record':validate_numeric_route},prefix.__name__,prefix.__defaults__,prefix.__closure__)
 assert fn(path,[base],'frozen')==1
 with pytest.raises(ValueError):fn(path,[{**base,'candidate_labels':['same','different']}],'frozen')
 path.write_text(json.dumps(row))
 with pytest.raises(ValueError,match='Torn output'):fn(path,[base],'frozen')


def test_execute_blocks_before_review_or_native_import_while_training_incomplete(monkeypatch):
 import mri_rl.c40_visual_probe as probe
 monkeypatch.setattr(probe,'prepare_scope',lambda *a:({}, {'original21training_ready':False}))
 def forbidden(**kwargs):raise AssertionError('review/native path reached')
 monkeypatch.setattr(probe,'_check_review',forbidden)
 with pytest.raises(ValueError,match='Original21training must finish'):
  probe.execute(None,None,None,None,None,None,None)

def test_completed_probe_does_not_reload_model_or_lock_queue(monkeypatch,tmp_path):
 import mri_rl.c40_visual_probe as probe
 output=tmp_path/'complete';output.write_text('fixture')
 monkeypatch.setattr(probe,'prepare_scope',lambda *a:({}, {'original21training_ready':True,'completed_records':10}))
 monkeypatch.setattr(probe,'_check_review',lambda **kwargs:None)
 def forbidden(*args):raise AssertionError('training or model reload reached')
 monkeypatch.setattr(probe,'authenticate_training',forbidden)
 result=probe.execute(None,'B28',output,tmp_path/'missing/cursor',None,tmp_path/'fixture_review',None)
 assert result['status']=='ACTUAL_PROBE_ALREADY_COMPLETE' and result['completed']==10
