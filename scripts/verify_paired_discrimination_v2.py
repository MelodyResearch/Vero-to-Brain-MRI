"""Conditional CPU proof for actual paired-pilot boundaries0/1/2. Never loads a model.
Reuses ONLY pinned existing CPU mmap/Adam math by AST; no new backward claim.
"""
import argparse,ast,hashlib,importlib.abc,json,math,os,random,struct,sys
from pathlib import Path
import numpy as np
MATH_SHA='0c3408940bc36a646dfc97eb37c27c5964c0eba331f8435b636dce3e714df048'
MATH_NAMES={'TensorFile','decode','verify_adam_tensor','moments','optimizer_layout','arrays_equal'}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
canonical=lambda v:json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)
objsha=lambda v:hashlib.sha256(canonical(v).encode()).hexdigest()
load=lambda p:json.loads(Path(p).read_text())
def load_approved_math(path):
 assert sha(path)==MATH_SHA,'Reviewed CPU math source changed'
 tree=ast.parse(Path(path).read_text());nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in MATH_NAMES];assert {n.name for n in nodes}==MATH_NAMES
 ns={'np':np,'struct':struct,'json':json};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
 return ns, {n.name:hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest() for n in nodes}
def validate_declared_plan(plan):
 rows=plan['rows_PRIVATE'];assert plan['maximum_updates']==2 and plan['schedule']==[[0,1,2,3],[4,5,6,7]] and len(rows)==len({r['id'] for r in rows})==8
 for i in range(0,8,2):
  a,b=rows[i:i+2];assert a['ability']==b['ability']=='mri_sequence' and a['extra_info']['internal_split']==b['extra_info']['internal_split']=='train'
  assert (a['extra_info']['patient_key'],a['extra_info']['level'])==(b['extra_info']['patient_key'],b['extra_info']['level'])
  assert [a['reward_model']['ground_truth'],b['reward_model']['ground_truth']]==['FLAIR','contrast-enhanced T1']
 assert set(plan['input_tokens'])==set(plan['target_token_cost'])=={r['id'] for r in rows}
 assert all(type(v) is int and v==213 for v in plan['input_tokens'].values()) and all(type(v) is int and v==35 for v in plan['target_token_cost'].values())
 return [sum(2*plan['input_tokens'][rows[i]['id']]+plan['target_token_cost'][rows[i]['id']] for i in step) for step in plan['schedule']]
def validate_native_terminal(terminal,receipt_sha):
 assert terminal['status']=='ACTUAL_TWO_UPDATE_PAIRED_PILOT_COMPLETE_PENDING_ACCURACY' and terminal['checkpoint_receipt_sha256']==receipt_sha,'Actual full pilot terminal/checkpoint missing'
class NoNative(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'mlx','mlx_vlm','torch','transformers'}:raise RuntimeError('Native imports forbidden by CPU verifier')
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for k in ('primary-root','scope','review','exit-witness','native-log','output'):p.add_argument('--'+k,type=Path,required=True)
 for k in ('scope','review','exit-witness'):p.add_argument('--expected-'+k+'-sha256',required=True)
 p.add_argument('--reviewer-id',required=True);a=p.parse_args();assert a.reviewer_id.strip();sys.meta_path.insert(0,NoNative())
 R=a.primary_root.resolve();W=Path(__file__).resolve().parents[1];assert a.output.resolve().is_relative_to(W/'local_data') and not a.output.exists(),'New private proof path required'
 assert sha(a.scope)==a.expected_scope_sha256 and sha(a.review)==a.expected_review_sha256 and sha(a.exit_witness)==a.expected_exit_witness_sha256
 scope=load(a.scope);review=load(a.review);witness=load(a.exit_witness)
 assert review['verdict'] in ('PASS','PASS_WITH_LIMITATIONS') and review['reviewer_kind'] in ('isolated_codex','claude_code') and review['reviewer_id'] and review['reviewer_findings'] and all(review[k]==v for k,v in scope.items())
 assert scope['kind']=='paired_discrimination_pilot_train_v2' and scope['ready'] is True and scope['seed']==7 and scope['max_new_steps']==2 and 1<=scope['maximum_new_updates']<=2
 assert witness['exit_code']==0 and witness['review_sha256']==a.expected_review_sha256 and witness['raw_log_sha256']==sha(a.native_log),'External actual exit/log/approval differs'
 assert sha(W/'scripts/run_paired_discrimination_pilot_v2.py')==scope['entry_sha256'] and sha(W/'src/mri_rl/paired_discrimination_pilot_v2.py')==scope['paired_source_sha256']
 packpath=W/'local_data/v2_cpu_representation_v1/frozen_inputs_PRIVATE.json';assert sha(packpath)=='446808a0203ba1e6dd9f45e7f052a7667420b016ffff5a5b88f295d728fc79ab';frozen={r['id']:r for r in load(packpath)['train_PRIVATE']}
 planpath=Path(scope['pilot_plan_path']);assert sha(planpath)==scope['pilot_plan_sha256'];plan=load(planpath);costs=validate_declared_plan(plan);assert costs==[1844,1844]
 assert plan['source_pack_sha256']==sha(packpath) and all(r==frozen[r['id']] for r in plan['rows_PRIVATE']),'Pilot rows differ from original frozen train-only pack'
 assert plan['objective_source_sha256']==scope['paired_source_sha256'] and plan['sources']==scope['sources']
 for name,h in plan['sources'].items():assert sha(R/'src/mri_rl'/name)==h
 for name,h in scope['native_sources'].items():
  path=R/'../../work/mlx-venv/lib/python3.13/site-packages/mlx_vlm'/name.removeprefix('installed_mlx_vlm/') if name.startswith('installed_mlx_vlm/') else R/'scripts/run_c40_qwen_visual_probe.py' if name=='entry_script' else R/'src/mri_rl'/name
  assert sha(path)==h
 feature=scope['feature_evidence'];assert feature['ready'] and sha(feature['receipt_path'])==feature['receipt_sha256'];assert sha(feature['plan_path'])==feature['plan_sha256']
 measured=plan['measurement_evidence'];assert sha(measured['path'])==measured['sha256'] and load(measured['path'])['input_counts']==plan['input_tokens']
 for row in plan['rows_PRIVATE']:assert sha(row['images'][0])==plan['image_sha256'][row['id']]
 # Token partitions are revalidated with the frozen tokenizer only; no processor/model.
 try:
  from tokenizers import Tokenizer
 except ModuleNotFoundError:
  sys.path.append(str((R/'../../work/mlx-venv/lib/python3.13/site-packages').resolve()));from tokenizers import Tokenizer
 original=load(R/'local_data/c_two_task_pool_v1/u7_visual_controls_v1/probe_plan_PRIVATE.json');tokenizer_path=Path(original['model_path'])/'tokenizer.json';assert sha(tokenizer_path)==plan['tokenizer_sha256'];tok=Tokenizer.from_file(str(tokenizer_path))
 prefix='<think>Visible MRI evidence.</think><answer>\\boxed{';suffix='}</answer>';candidate_ids={l:tok.encode(prefix+l+suffix,add_special_tokens=False).ids for l in ('FLAIR','contrast-enhanced T1')}
 for row in plan['rows_PRIVATE']:assert plan['target_ids'][row['id']]==candidate_ids and sum(map(len,candidate_ids.values()))==35
 rows=plan['rows_PRIVATE'];steps=[]
 for index,chunk in enumerate(plan['schedule']):
  ex=[rows[i] for i in chunk];steps.append({'id':f'sft-step-{index:04d}','ability':'sft_step','data_source':'C_pool_sft','images':[ex[0]['images'][0]],'extra_info':{'reward_type':'sft_step','source_ids':[e['id'] for e in ex]},'examples':ex})
 optimizer={'name':'Adam','learning_rate':1e-5,'betas':[.9,.999],'eps':1e-8,'bias_correction':False};ledger_plan={'budget':{'text_visual_token_cap':3688,'maximum_attempted_groups':2}}
 experiment={'objective':'paired_mean_label_discrimination_v2','pilot_plan_sha256':objsha(plan),'feature_proof_sha256':feature['receipt_sha256'],'example_image_sha256':objsha(plan['image_sha256']),'ledger_plan_sha256':objsha(ledger_plan)}
 binding={'experiment':experiment,'configuration':{'profile':'tier_c_sft_v1','examples_per_step':4,'target_cap':64,'optimizer':optimizer},'schedule_sha256':objsha(steps),'schedule_record_ids':[s['id'] for s in steps],'image_sha256':{s['id']:sha(s['images'][0]) for s in steps},'driver_source_sha256':sha(R/'src/mri_rl/continuous_rl.py')}
 run=Path(scope['run_directory']);paths=sorted(run.glob('boundary_*'));assert [x.name for x in paths]==['boundary_0000','boundary_0001','boundary_0002']
 receipts=[load(x/'receipt.json') for x in paths];parent=None;verified_files=0;losses=[]
 for index,(path,r) in enumerate(zip(paths,receipts)):
  assert r['schema']=='continuous-rl-boundary-v1' and r['binding']==binding and r['completed_groups']==r['optimizer_updates']==index and r['parent_receipt_sha256']==parent and r['updated']==(index>0)
  assert all(r['state_validation'][k] is True for k in ('parameters_finite','optimizer_finite','second_moments_nonnegative'))
  assert sha(path/'adapter_config.json')==scope['common_B28']['adapter_config_sha256']
  for name,h in r['files_sha256'].items():assert (path/name).is_file() and not (path/name).is_symlink() and sha(path/name)==h;verified_files+=1
  if index==0:assert sha(path/'adapter.safetensors')==scope['common_B28']['sha256'] and r['ledger']['used_tokens']==0
  else:
   step=steps[index-1];rollout=load(path/'rollout_PRIVATE.json');assert r['record_id']==r['schedule_record_id']==step['id'] and rollout['source_ids']==step['extra_info']['source_ids'];assert rollout['prompt_tokens']==[426]*4 and rollout['target_tokens']==[35]*4
   reservation=load(path/'attempt_reservation.json');assert reservation['parent_receipt_sha256']==parent and reservation['run_binding_sha256']==objsha(binding) and reservation['index']==index-1 and reservation['example_ids']==step['extra_info']['source_ids'] and reservation['example_costs']==[461]*4 and reservation['reserved_cost']==1844
   assert len(r['loss_terms'])==4 and all(math.isfinite(v) and v>=0 for v in r['loss_terms']) and math.isfinite(r['gradient_l2']) and r['gradient_l2']>0;losses.extend(r['loss_terms'])
  assert r['ledger']['cap']==3688 and r['ledger']['used_tokens']==sum(costs[:index]) and r['ledger']['imputed_unknown_tokens']==0 and not r['ledger']['partial_dirs_included'] and r['ledger']['charged_groups']==index
  parent=sha(path/'receipt.json')
 terminal=None
 for line in reversed(a.native_log.read_text().splitlines()):
  try:item=json.loads(line)
  except json.JSONDecodeError:continue
  if isinstance(item,dict) and item.get('status')=='ACTUAL_TWO_UPDATE_PAIRED_PILOT_COMPLETE_PENDING_ACCURACY':terminal=item;break
 assert terminal is not None;validate_native_terminal(terminal,parent)
 mathpath=R/'local_data/c_two_task_pool_v1/tier_c_v1/independent_remaining_RL_v1/verify_completed_RL_PRIVATE.py';lib,math_ast=load_approved_math(mathpath);TensorFile=lib['TensorFile'];decode=lib['decode'];moments=lib['moments'];arrays_equal=lib['arrays_equal'];optimizer_layout=lib['optimizer_layout'];recon=[]
 for index,(path,r) in enumerate(zip(paths,receipts)):
  manifest=load(path/'optimizer/manifest.json');assert manifest['bindings']['run']==binding and manifest['bindings']['adapter_sha256']==r['files_sha256']['adapter.safetensors'] and manifest['completed_groups']==manifest['optimizer_updates']==index and sha(path/'optimizer/state.safetensors')==manifest['tensors_sha256']
  with TensorFile(path/'optimizer/state.safetensors') as state:
   tree=decode(manifest['layout'],state);opt=tree['optimizer'];mm=moments(opt);assert int(opt['step'])==index and abs(float(opt['learning_rate'])-1e-5)<1e-12;random.Random().setstate(tree['rng']['python']);rng=tree['rng']['mlx'];assert len(rng)==1 and rng[0].shape==(2,) and rng[0].dtype==np.uint32;assert all(np.isfinite(v).all() for v in state.values())
   if index==0:assert not mm;continue
   previous=paths[index-1];oldmanifest=load(previous/'optimizer/manifest.json')
   with TensorFile(previous/'optimizer/state.safetensors') as prior,TensorFile(path/'optimizer_before.safetensors') as before,TensorFile(path/'optimizer_after.safetensors') as after,TensorFile(previous/'adapter.safetensors') as w0,TensorFile(path/'adapter.safetensors') as w1:
    oldopt=decode(oldmanifest['layout'],prior)['optimizer'];oldmm=moments(oldopt);assert int(oldopt['step'])==index-1;assert set(mm)==set(w0)==set(w1) and len(mm)==504
    assert sha(path/'optimizer_before.safetensors') and hashlib.sha256((canonical(optimizer_layout(oldmanifest))+sha(path/'optimizer_before.safetensors')).encode()).hexdigest()==r['optimizer_before_fingerprint'];assert hashlib.sha256((canonical(optimizer_layout(manifest))+sha(path/'optimizer_after.safetensors')).encode()).hexdigest()==r['optimizer_after_fingerprint'];arrays_equal(before,{k:prior[k] for k in before});arrays_equal(after,{k:state[k] for k in after})
    assert len(before)==2+2*504*int(bool(oldmm)) and len(after)==1010
    delta2=grad2=0.;changed=elements=0;maxw=maxv=0.
    for name,left in w0.items():
     check=lib['verify_adam_tensor'](left,w1[name],oldmm[name]['m'] if oldmm else 0.,oldmm[name]['v'] if oldmm else 0.,mm[name]['m'],mm[name]['v']);delta2+=check['delta_squared'];grad2+=check['gradient_squared'];changed+=int(check['changed_elements']>0);elements+=check['changed_elements'];maxw=max(maxw,check['w_error']);maxv=max(maxv,check['v_error'])
    norm=math.sqrt(grad2);assert changed>0 and delta2>0 and math.isclose(norm,r['gradient_l2'],rel_tol=1e-5,abs_tol=1e-7)
    recon.append({'step_before':index-1,'step_after':index,'adapter_tensors':504,'changed_tensors':changed,'changed_elements':elements,'delta_l2':math.sqrt(delta2),'gradient_l2_reconstructed':norm,'gradient_l2_saved':r['gradient_l2'],'max_weight_error':maxw,'max_second_moment_error':maxv})
 assert len(losses)==8 and len(recon)==2 and paths==sorted(run.glob('boundary_*')) and sha(a.native_log)==witness['raw_log_sha256'] and not any(n in sys.modules for n in ('mlx','mlx.core','mlx_vlm','torch','transformers'))
 report={'schema':'paired-discrimination-two-update-CPU-proof-v2','verdict':'PASS_WITH_LIMITATIONS','reviewer':a.reviewer_id,'reviewer_kind':'isolated_codex','actual_completed_updates':2,'training_images':8,'training_pairs':4,'training_people':len({r['extra_info']['patient_key'] for r in rows}),'loss_terms':losses,'loss_vector_native_forward_recomputed':False,'loss_interpretation':'Actual persisted finite nonnegative CE scalars; native candidate logits/backward not independently recomputed.','used_tokens':3688,'token_cost_per_update':costs,'candidate_teacherforced_sequences':16,'persisted_optimizer_RNG_states_verified':3,'RNG_must_change':False,'verified_files':verified_files,'all_Adam_updates':recon,'evidence_sha256':{'scope':sha(a.scope),'review':sha(a.review),'external_exit':sha(a.exit_witness),'closed_native_log':sha(a.native_log),'plan':sha(planpath),'feature_completion':feature['receipt_sha256'],'boundary_receipts':[sha(x/'receipt.json') for x in paths],'initial_adapter':receipts[0]['files_sha256']['adapter.safetensors'],'final_adapter':receipts[2]['files_sha256']['adapter.safetensors'],'reviewed_CPU_math_source':MATH_SHA,'reused_math_function_AST':math_ast,'this_verifier':sha(__file__)},'limitations':['Saved data/label/source and persistent Adam/moment/weight consistency only; actual native backward and CE logits are NOT independently recomputed.','No fresh native checkpoint reload or model inference is performed by this verifier.','Two-update optimization evidence is not MRI accuracy improvement. Frozen real/null/swap class-specific development evaluation is separate.','Only two training people/four pairs; no independent clinical, seed stability or anatomical reasoning claim.']}
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('x') as f:json.dump(report,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps({'status':'ACTUAL_TWO_UPDATE_CPU_PROOF_COMPLETE','verdict':report['verdict'],'proof_sha256':sha(a.output),'updates':2,'used_tokens':3688,'changed_tensors_per_step':[v['changed_tensors'] for v in recon]},sort_keys=True))
if __name__=='__main__':main()
