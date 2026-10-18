"""Synthetic CPU evidence only; no model imports."""
import importlib.util,json,random
from pathlib import Path
import numpy as np
import pytest
from mri_rl.mlx_gspo_smoke import task_group_is_informative

def load(name):
 p=Path(__file__).parents[1]/'src/mri_rl'/f'{name}.py'
 assert p.exists(), 'Versioned adaptive continuation missing'
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_only_supported_mixed_format_exclusion_is_removed():
 m=load('adaptive_accuracy_gate_v1')
 grades=[{'accuracy':1.,'format':1.,'overlong':0.,'total':1.},{'accuracy':0.,'format':1.,'overlong':0.,'total':.2}, {'accuracy':0.,'format':0.,'overlong':0.,'total':0.}]
 assert not task_group_is_informative(grades)
 assert m.accuracy_group_is_informative(grades)
 # One valid correct + one invalid wrong is a formatting artifact, not valid task variation.
 assert not m.accuracy_group_is_informative([grades[0],grades[2]])
 for changed in ([dict(g,accuracy=0.) for g in grades], [dict(g,total=1.) for g in grades], [grades[0],grades[1],dict(grades[2],overlong=-.1)]):
  assert not m.accuracy_group_is_informative(changed)
 assert m.accuracy_group_is_informative([dict(g,format=1.) for g in grades])
 with pytest.raises(ValueError):m.accuracy_group_is_informative([dict(g,total=float('nan')) for g in grades])

def backend_class():
 p=Path(__file__).with_name('test_continuous_rl_driver.py');s=importlib.util.spec_from_file_location('prior_fixture',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
 class Backend(m.NumericBackend):
  def save_adapter(self,path):
   super().save_adapter(path);Path(path).with_name("adapter_config.json").write_text("{}")
  def rollout(self,row,index,workdir):
   result=super().rollout(row,index,workdir)
   self.absolute_index=index
   return result
 return Backend

def fixtures(tmp_path):
 m=load('adaptive_accuracy_driver_v1');Backend=backend_class();parent=tmp_path/'parent';parent.mkdir();work=parent/'.pending-initial';work.mkdir()
 b=Backend(None,{});b.state['step']=np.array(52,dtype=np.uint64);b.state['weight']={'m':np.array([.4,-.3]),'v':np.array([.2,.1])}
 original=load('continuous_rl');r=original._publish(parent,work,b,{'fixture':'original-parent'},114,52,'prior-hash',{'updated':False})
 image=tmp_path/'image.png';image.write_bytes(b'SYNTHETIC')
 rows=[{'id':f'future-{i}','ability':'mri_sequence','data_source':'OpenNeuro_ds007045','images':[str(image)],'reward_model':{'ground_truth':'FLAIR'},'extra_info':{'reward_type':'mri_sequence','internal_split':'train','patient_lineage':'BIDS_SUBJECT_VERIFIED','patient_key':'fixture','question':'Fixture'}} for i in range(3)]
 return m,Backend,r['path'],original.digest(r['path']/'receipt.json'),rows

def invoke(m,B,parent,sha,rows,out,groups,resume=False):
 return m.run(output=out,schedule=rows,bindings={'fixture':'child'},backend_factory=B,authorize=lambda c:None,max_new_groups=groups,resume=resume,parent_checkpoint=parent,parent_receipt_sha256=sha,max_optimizer_updates=4)

def test_absolute_adam52_and_rng_survive_relative_journal_resume(tmp_path):
 m,B,parent,sha,rows=fixtures(tmp_path)
 full=invoke(m,B,parent,sha,rows,tmp_path/'full',3);a=B.instances[-1]
 invoke(m,B,parent,sha,rows,tmp_path/'split',1);split=invoke(m,B,parent,sha,rows,tmp_path/'split',2,True);b=B.instances[-1]
 assert full['optimizer_updates']==split['optimizer_updates']==3
 assert full['absolute_optimizer_updates']==55
 assert int(a.state['step'])==int(b.state['step'])==55
 np.testing.assert_array_equal(a.weight,b.weight)
 for k in ['m','v']:np.testing.assert_array_equal(a.state['weight'][k],b.state['weight'][k])
 assert a.rng.random()==b.rng.random()
 assert b.absolute_index==116
 first=json.loads((tmp_path/'split/boundary_0000/receipt.json').read_text());last=json.loads((tmp_path/'split/boundary_0003/receipt.json').read_text())
 assert first['optimizer_updates']==0 and first['absolute_optimizer_updates']==52
 assert last['completed_groups']==3 and last['absolute_completed_groups']==117
 manifest=json.loads((tmp_path/'split/boundary_0003/optimizer/manifest.json').read_text())
 assert manifest['optimizer_updates']==55 and manifest['completed_groups']==117

def test_parent_tamper_rejected_before_backend(tmp_path):
 m,B,parent,sha,rows=fixtures(tmp_path);n=len(B.instances)
 (parent/'receipt.json').write_text('{}')
 with pytest.raises(ValueError,match='parent|Parent'):invoke(m,B,parent,sha,rows,tmp_path/'child',1)
 assert len(B.instances)==n

def test_skip_keeps_absolute_state_and_budget_stops_before_extra_rollout(tmp_path):
 m,B,parent,sha,rows=fixtures(tmp_path)
 class Mixed(B):
  def rollout(self,row,index,workdir):
   result=super().rollout(row,index,workdir)
   if index==115:result['responses'][0]['text']='wrong too'
   return result
 r=m.run(output=tmp_path/'cap',schedule=rows,bindings={'fixture':'cap'},backend_factory=Mixed,authorize=lambda c:None,max_new_groups=3,parent_checkpoint=parent,parent_receipt_sha256=sha,max_optimizer_updates=1)
 assert r['optimizer_updates']==1 and r['completed_groups']==1 and int(Mixed.instances[-1].state['step'])==53
 # A different config cannot resume the same journal; the original max is an authorization envelope, not an Adam reset.
 with pytest.raises(ValueError):
  m.run(output=tmp_path/'bad',schedule=rows*6,bindings={'fixture':'bad'},backend_factory=B,authorize=lambda c:None,max_new_groups=16,parent_checkpoint=parent,parent_receipt_sha256=sha,max_optimizer_updates=4)

def test_entry_missing():
 load('adaptive_accuracy_entry_v1')

def test_partial_resume_preserves_absolute_rollout_offset_and_cost(tmp_path):
 m=load('adaptive_accuracy_driver_v1');work=tmp_path/'pending';work.mkdir();row={'id':'row'}
 context={'binding_sha256':'binding','boundary_groups':{'parent':0},'schedule':[row], 'image_sha256':{'row':'pixel'},'group_size':4,'max_tokens':256,'parent_group_offset':114}
 reservation={'schema':'attempt-reservation-v1','run_binding_sha256':'binding','parent_receipt_sha256':'parent','index':0,'record_id':'row','image_sha256':'pixel','input_tokens':213,'group_size':4,'max_tokens':256}
 (work/'attempt_reservation.json').write_text(json.dumps(reservation))
 raw={'record_id':'row','group_index':114,'image_sha256':'pixel','responses':[{'text':'partial','token_ids':[1,2]}]}
 (work/'rollout_progress_PRIVATE.json').write_text(json.dumps(raw))
 assert m._partial_cost(work,{'row':row},lambda r:213,context=context)==(215,469)
 raw['group_index']=0;(work/'rollout_progress_PRIVATE.json').write_text(json.dumps(raw))
 with pytest.raises(ValueError,match='index'):m._partial_cost(work,{'row':row},lambda r:213,context=context)

def test_interrupted_group_debt_is_charged_once_across_resume(tmp_path):
 from mri_rl.two_task_training import TokenLedger
 m,B,parent,sha,rows=fixtures(tmp_path)
 class Crash(B):
  def rollout(self,row,index,workdir):
   raw=super().rollout(row,index,workdir);raw['group_index']=index;raw['responses']=raw['responses'][:1]
   (Path(workdir)/'rollout_progress_PRIVATE.json').write_text(json.dumps(raw))
   raise RuntimeError('interrupted response')
 lp={'group_size':2,'max_tokens':256,'budget':{'maximum_attempted_groups':3,'text_visual_token_cap':10000,'input_tokens_min':213,'input_tokens_max':213}}
 def go(cls,resume):
  ledger=TokenLedger(lp)
  result=m.run(output=tmp_path/'debt',schedule=rows,bindings={'fixture':'debt'},backend_factory=cls,authorize=lambda c:None,max_new_groups=1,resume=resume,parent_checkpoint=parent,parent_receipt_sha256=sha,max_optimizer_updates=4,ledger=ledger,input_tokens=lambda r:213)
  return result,ledger
 with pytest.raises(RuntimeError,match='interrupted'):go(Crash,False)
 _,first=go(B,True);assert first.used==214+469+428 and first.imputed==469
 _,second=go(B,True);assert second.used==214+469+428+428 and second.imputed==469
 r=json.loads((tmp_path/'debt/boundary_0002/receipt.json').read_text())
 assert len(r['ledger']['partial_dirs_included'])==1 and r['absolute_optimizer_updates']==54
