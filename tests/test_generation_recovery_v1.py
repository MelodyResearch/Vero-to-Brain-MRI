import importlib.util,copy,json
from pathlib import Path
import pytest
W=Path(__file__).parents[1]
def helper():
 p=W/'src/mri_rl/generation_recovery_v1.py';assert p.exists(),'Generation recovery helper missing';s=importlib.util.spec_from_file_location('generation_recovery_test',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def records():
 g=[{'record_id':str(i),'category':'mri_sequence' if i<24 else 'mri_visit_correspondence','image':'pixel','image_sha256':'fixed','visible_target_PRIVATE':'FLAIR' if i<24 else 'same','prompt':[{'role':'system','content':'FROZEN SYSTEM'},{'role':'user','content':'FROZEN QUESTION?'}]} for i in range(40)]
 c=[dict(x,condition='real_image',candidate_labels=['FLAIR','contrast-enhanced T1'],source_record_id=x['record_id']) for x in g[:24]];c.append(dict(c[0],record_id='NULL',condition='no_image',image=None,image_sha256=None,visible_target_PRIVATE=None));return {'greedy_records_PRIVATE':g,'candidate_records_PRIVATE':c}

def test_two_single_factor_controls_preserve_system_question_targets():
 m=helper();e=records();old=copy.deepcopy(e);x=m.control_records(e)
 assert set(x)=={'original_512','short_reason_256'} and e==old
 assert x['original_512']['max_tokens']==512 and x['short_reason_256']['max_tokens']==256
 assert x['original_512']['greedy'][0]['prompt']==e['greedy_records_PRIVATE'][0]['prompt']
 for r in x['short_reason_256']['greedy']+x['short_reason_256']['candidates']:
  assert r['prompt'][0]=={'role':'system','content':'FROZEN SYSTEM'}
  assert r['prompt'][1]['content']=='FROZEN QUESTION?\n\n'+m.SHORT_INSTRUCTION
 assert len(x['short_reason_256']['greedy'])==40 and len(x['short_reason_256']['candidates'])==25
 assert x['original_512']['candidate_mode']=='baseline_reference'

def test_resume_rejects_changed_prompt_cap_duplicate_and_nonfinite(tmp_path):
 m=helper();record=helper().control_records(records())['short_reason_256']['greedy'][0];row={**record,'run_identity':'run','weight_updates':0,'response':'x','token_ids':[1],'generation_tokens':1,'finish_reason':'stop','hit_max_tokens':False}
 p=tmp_path/'out';p.write_text(json.dumps(row)+'\n');assert len(m.durable_prefix(p,[record],'run',lambda r:m.validate_greedy(r,256)))==1
 for changed in ({**row,'prompt':[]},{**row,'max_new_tokens':512},{**row,'generation_tokens':float('nan')}):
  p.write_text(json.dumps(changed)+'\n')
  with pytest.raises(ValueError):m.durable_prefix(p,[record],'run',lambda r:m.validate_greedy(r,256))
 p.write_text(json.dumps(row)+'\n'+json.dumps(row)+'\n')
 with pytest.raises(ValueError):m.durable_prefix(p,[record],'run',lambda r:m.validate_greedy(r,256))

def test_completion_authenticates_all_outputs_and_release(tmp_path):
 m=helper();paths={k:tmp_path/k for k in ('original_greedy','original_null','short_greedy','short_candidates','short_null')}
 for k,p in paths.items():p.write_text('{}\n'*{'original_greedy':40,'original_null':1,'short_greedy':40,'short_candidates':25,'short_null':1}[k])
 scope={'run_identity':'R','adapter':{'sha256':'A'},'baseline_evidence':{'scope':{'sha256':'B'}},'evaluation_contract_sha256':'E'}
 receipt={'status':'ACTUAL_GENERATION_RECOVERY_CONTROLS_COMPLETE','run_identity':'R','adapter_sha256':'A','baseline_scope_sha256':'B','evaluation_contract_sha256':'E','native_release_verified':True,'weight_updates':0,'counts':{'original_512_greedy':40,'original_512_null':1,'short_reason_256_greedy':40,'short_reason_256_candidates':25,'short_reason_256_null':1},'outputs_sha256':{k:m.sha(v) for k,v in paths.items()}}
 m.validate_completion(receipt,scope,paths)
 for key,val in [('native_release_verified',False),('adapter_sha256','wrong'),('weight_updates',1)]:
  with pytest.raises(ValueError):m.validate_completion({**receipt,key:val},scope,paths)
 paths['short_candidates'].write_text('changed')
 with pytest.raises(ValueError):m.validate_completion(receipt,scope,paths)

def test_claimed_completion_cannot_hide_partial_file_coverage(tmp_path):
 m=helper();paths={k:tmp_path/k for k in ('original_greedy','original_null','short_greedy','short_candidates','short_null')}
 for p in paths.values():p.write_text('{}\n')
 scope={'run_identity':'R','adapter':{'sha256':'A'},'baseline_evidence':{'scope':{'sha256':'B'}},'evaluation_contract_sha256':'E'}
 receipt={'status':'ACTUAL_GENERATION_RECOVERY_CONTROLS_COMPLETE','run_identity':'R','adapter_sha256':'A','baseline_scope_sha256':'B','evaluation_contract_sha256':'E','native_release_verified':True,'weight_updates':0,'counts':m.COUNTS,'outputs_sha256':{k:m.sha(v) for k,v in paths.items()}}
 with pytest.raises(ValueError):m.validate_completion(receipt,scope,paths)
