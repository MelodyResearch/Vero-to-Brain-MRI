"""Synthetic CPU fixtures only; native continuation remains separately reviewed."""
import importlib.util,json
from pathlib import Path
import numpy as np
import pytest

def load(name):
 p=Path(__file__).parents[1]/'src/mri_rl'/f'{name}.py';s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def fixture(tmp_path):
 old=load('adaptive_accuracy_driver_v1')
 p=Path(__file__).with_name('test_adaptive_accuracy_gate_v1.py');s=importlib.util.spec_from_file_location('prior_synthetic',p);f=importlib.util.module_from_spec(s);s.loader.exec_module(f);B=f.backend_class();backend=B(None,{});backend.state['step']=np.array(56,dtype=np.uint64);backend.state['weight']={'m':np.array([.4,-.3]),'v':np.array([.2,.1])};out=tmp_path/'old-child';out.mkdir();work=out/'.pending';work.mkdir();binding={'experiment':{'adaptive_parent':{'groups':114,'updates':52}}};r=old._publish(out,work,backend,binding,7,4,'old-chain',{'updated':True});image=tmp_path/'image.png';image.write_bytes(b'SYNTHETIC');rows=[{'id':f'uniform-{i}','ability':'mri_sequence','data_source':'OpenNeuro_ds007045','images':[str(image)],'reward_model':{'ground_truth':'FLAIR'},'extra_info':{'reward_type':'mri_sequence','internal_split':'train','patient_lineage':'BIDS_SUBJECT_VERIFIED','patient_key':'fixture','question':'Fixture'}} for i in range(3)];return B,r['path'],old.digest(r['path']/'receipt.json'),rows

def test_nested_parent_absolute56_rng_and_resume(tmp_path):
 m=load('uniform_balance_continuation_driver_v1');B,parent,sha,rows=fixture(tmp_path)
 def run(output,groups,resume=False):return m.run(output=output,schedule=rows,bindings={'fixture':'new-uniform-stage'},backend_factory=B,authorize=lambda c:None,max_new_groups=groups,resume=resume,parent_checkpoint=parent,parent_receipt_sha256=sha,max_optimizer_updates=4)
 full=run(tmp_path/'full',3);a=B.instances[-1];run(tmp_path/'split',1);split=run(tmp_path/'split',2,True);b=B.instances[-1];assert full['absolute_optimizer_updates']==split['absolute_optimizer_updates']==59 and full['absolute_completed_groups']==124;np.testing.assert_array_equal(a.weight,b.weight)
 for k in ('m','v'):np.testing.assert_array_equal(a.state['weight'][k],b.state['weight'][k])
 assert a.rng.random()==b.rng.random() and b.absolute_index==123
 initial=json.loads((tmp_path/'split/boundary_0000/receipt.json').read_text());assert initial['optimizer_updates']==0 and initial['absolute_optimizer_updates']==56 and initial['absolute_completed_groups']==121
 manifest=json.loads((tmp_path/'split/boundary_0003/optimizer/manifest.json').read_text());assert manifest['optimizer_updates']==59 and manifest['completed_groups']==124

def test_nested_parent_metadata_and_native_manifest_mismatch_rejected(tmp_path):
 m=load('uniform_balance_continuation_gate_v1');B,parent,sha,rows=fixture(tmp_path);a=m.parent_anchor(parent,sha);assert a['groups']==121 and a['updates']==56 and a['local_groups']==7 and a['local_updates']==4
 p=parent/'receipt.json';r=json.loads(p.read_text());r['absolute_optimizer_updates']=4;p.write_text(json.dumps(r));from mri_rl.continuous_rl import digest
 with pytest.raises(ValueError):m.parent_anchor(parent,digest(p))

def test_only_declared_sampling_binding_changes():
 m=load('uniform_balance_continuation_entry_v1');base={'arm':{'name':'P_calibrated','sampling':'proportional','reward':'medgrpo_eq3'},'schedule_binding_sha256':'old','common':{'same':'weights'},'seed':13};new={**base,'arm':{'name':'U_calibrated','sampling':'uniform','reward':'medgrpo_eq3'},'schedule_binding_sha256':'new'};m.validate_sampler_only_change(base,new)
 with pytest.raises(ValueError):m.validate_sampler_only_change(base,{**new,'seed':7})
 with pytest.raises(ValueError):m.validate_sampler_only_change(base,{**new,'arm':{**new['arm'],'reward':'direct'}})

def test_nested_partial_offset_and_one_time_debt_accounting(tmp_path):
 from mri_rl.two_task_training import TokenLedger
 m=load('uniform_balance_continuation_driver_v1');B,parent,sha,rows=fixture(tmp_path)
 class Crash(B):
  def rollout(self,row,index,workdir):
   raw=super().rollout(row,index,workdir);raw['group_index']=index;raw['responses']=raw['responses'][:1];(Path(workdir)/'rollout_progress_PRIVATE.json').write_text(json.dumps(raw));raise RuntimeError('bounded interrupted fixture')
 lp={'group_size':2,'max_tokens':256,'budget':{'maximum_attempted_groups':3,'text_visual_token_cap':10000,'input_tokens_min':213,'input_tokens_max':213}}
 def go(cls,resume):
  ledger=TokenLedger(lp);r=m.run(output=tmp_path/'debt',schedule=rows,bindings={'fixture':'new-uniform-debt'},backend_factory=cls,authorize=lambda c:None,max_new_groups=1,resume=resume,parent_checkpoint=parent,parent_receipt_sha256=sha,max_optimizer_updates=4,ledger=ledger,input_tokens=lambda row:213);return r,ledger
 with pytest.raises(RuntimeError,match='interrupted'):go(Crash,False)
 _,first=go(B,True);assert first.used==214+469+428 and first.imputed==469
 _,second=go(B,True);assert second.used==214+469+428+428 and second.imputed==469
 r=json.loads((tmp_path/'debt/boundary_0002/receipt.json').read_text());assert r['absolute_completed_groups']==123 and r['absolute_optimizer_updates']==58 and len(r['ledger']['partial_dirs_included'])==1
