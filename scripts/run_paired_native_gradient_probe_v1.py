"""Prepare or run one independently reviewed, read-only native pair-gradient scope."""
import argparse,fcntl,gc,hashlib,importlib.util,json,sys
from pathlib import Path
W=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
load=lambda p:json.loads(Path(p).read_text())
def require(ok,message):
 if not ok:raise ValueError(message)
def module(path,name,expected=None):
 if expected:require(sha(path)==expected,'Bound source changed')
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def prepare(a):
 R=a.primary_root.resolve();V=R/'.worktrees/vero-mri-v2-controls';sys.path.insert(0,str(R/'src'));helperpath=W/'src/mri_rl/paired_native_gradient_probe_v1.py';helper=module(helperpath,'pair_gradient_helper');entry=module(V/'scripts/run_paired_discrimination_pilot_v2.py','pair_gradient_reviewed_entry','d0202667c4a580c4ea34cdd22a31a2238fd2de8ad81f26277600f351b5dd152c');early=module(V/'scripts/run_c40_early_visual_probe_v1.py','pair_gradient_loader','8cafacbf3a9086160c8e257673c6f933070df208bffc5c32dec38bc0e1e82e06');api=early.primary_api(R);original=api.validate_plan(R/'local_data/c_two_task_pool_v1/u7_visual_controls_v1/probe_plan_PRIVATE.json')
 childscope=W/'local_data/adaptive_post_Pcal13_eval_v1/scope_READY_REQUIRES_REVIEW_PRIVATE.json';require(sha(childscope)=='022c97281506fce016d8dc74dcec983c0aa52797f34c92baf02c4189717a84f8','Expected selected-child scope');cs=load(childscope);child=cs['adaptive_child'];require(child['absolute_optimizer_updates']==56 and child['relative_completed_groups']==7 and child['relative_optimizer_updates']==4,'Exact actual seven-group child required');require(sha(child['proof_path'])==child['proof_sha256']=='39d62aa3d9949b3ec39a52fb9b64fe82eb1233fdafdd63158fb7e0d558a87967','Actual independent optimizer proof changed');proof=load(child['proof_path']);require(proof['preserved_initial_parent_state'] is True and proof['absolute_optimizer_updates']==56,'Inherited state proof missing');boundary=Path(child['receipt_path']).parent;require(sha(child['receipt_path'])==child['receipt_sha256']=='547ec55a19490b401cf140772a087d1656e0deabf2831cf1923697b4059e8bf0','Actual child boundary changed');br=load(child['receipt_path']);require(all(sha(boundary/n)==br['files_sha256'][n] for n in ('optimizer/manifest.json','optimizer/state.safetensors')),'Child optimizer/RNG artifact changed');adapter=cs['adapter'];require(sha(adapter['path'])==adapter['sha256']=='23b6ce522f770bf39acbadd9b66ccf00529de7822a5bb4173e7c4b96cd8618a7' and sha(boundary/'adapter_config.json')==adapter['adapter_config_sha256'],'Child model/config changed')
 planpath=V/'local_data/paired_discrimination_pilot_v2/plan_v2_MEASURED_PENDING_FEATURE_AND_REVIEW_PRIVATE.json';require(sha(planpath)=='94aac6d202b050e231c6765449adfe80c856f9ab43606c43c80a13cf0009d748','Frozen pilot image/target plan changed');plan=load(planpath);rows=plan['rows_PRIVATE'][:2];helper.validate_pair(rows);require(all(sha(r['images'][0])==plan['image_sha256'][r['id']] for r in rows),'Prespecified train image changed')
 pairedpath=V/'src/mri_rl/paired_discrimination_pilot_v2.py';paired=module(pairedpath,'pair_gradient_existing_objective','79963d311b4797e8010a5b6c8de81c21f2f0adb1f5088613f2e867afcdfd021e');paired.primary_modules(R)
 from mri_rl import answer_likelihood as al
 sources=api.source_hashes();require(sources==cs['native_sources'],'Existing numerical/native source closure changed');native=(R/'../../work/mlx-venv/lib/python3.13/site-packages').resolve();extra={'mlx_nn_utils':{'path':str(native/'mlx/nn/utils.py'),'sha256':'a5ab60f6731a7a8d3821b7e44ad96961ef848ae0315090f47a895627ede102a8'},'mlx_core_API_contract':{'path':str(native/'mlx/core/__init__.pyi'),'sha256':sha(native/'mlx/core/__init__.pyi')},'qwen_vision':{'path':str(native/'mlx_vlm/models/qwen3_vl/vision.py'),'sha256':'9885d83cf85e037a88f8364817007094a82b5941d49e909d2d9fe11b5e03bf6c'}}
 for v in extra.values():require(sha(v['path'])==v['sha256'],'Installed native backward/vision source changed')
 utilitypath=V/'src/mri_rl/early_native_conditioning_probe_v1.py';utility=module(utilitypath,'pair_gradient_existing_fingerprints','9a01c95314564afc8ef6e0fa83089dbef2ccaaccedb83ebf095ac4d7b424660c');out=a.output_directory.resolve();require(out.is_relative_to(W/'local_data') and not out.exists(),'Fresh private output required; preserve any partial run')
 scope={'kind':'paired_native_readonly_gradient_probe_v1','execution_tier':'B','primary_root':str(R),'entry_sha256':sha(__file__),'helper_sha256':sha(helperpath),'existing_objective_sha256':sha(pairedpath),'existing_evaluator_sha256':entry.sha(V/'scripts/run_paired_discrimination_pilot_v2.py'),'utility_sha256':sha(utilitypath),'sources':sources,'installed_extra':extra,'child_scope_sha256':sha(childscope),'child_checkpoint':child,'adapter':adapter,'optimizer_manifest_sha256':sha(boundary/'optimizer/manifest.json'),'optimizer_state_sha256':sha(boundary/'optimizer/state.safetensors'),'pilot_plan_path':str(planpath),'pilot_plan_sha256':sha(planpath),'selected_pair_rows_PRIVATE':rows,'selected_image_sha256_PRIVATE':[plan['image_sha256'][r['id']] for r in rows],'selection':'FIRST ordered same-person/same-level FLAIR,T1 pair of existing frozen8pilot images; outcome-independent training subset','budget':helper.BUDGET,'conditions':list(helper.CONDITIONS),'output_directory':str(out),'derived_cache':original['derived_cache'],'contract_sha256':sha(original['prompt_contract_path']),'model_path':original['model_path'],'global_lock':str(Path(api.MANIFEST).parent/'remaining20.lock'),'runtime_versions':{'mlx':'0.32.4','mlx-vlm':'0.7.8'},'new_training_updates':0,'sealed_test_accessed':False,'claim':'Native image-conditioned CE-gradient path diagnostic; no accuracy recovery or anatomical understanding claim'}
 return helper,entry,early,api,original,plan,paired,al,utility,scope,boundary,br

def execute(a,values):
 helper,entry,early,api,original,plan,paired,al,utility,scope,boundary,br=values;early.verify_review(a.review,a.expected_review_sha256,scope);out=Path(scope['output_directory'])
 with Path(scope['global_lock']).open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  from importlib.metadata import version
  require({k:version(k) for k in scope['runtime_versions']}==scope['runtime_versions'],'Native runtime version changed')
  from mri_rl.paced_resources import PacedGate
  from mri_rl.mlx_continuous_gspo import NativeGSPOBackend,is_loading_unit
  from mri_rl.experiment_contract import load_contract
  from mri_rl.likelihood_cache import verify_cache_identity
  from mri_rl.optimizer_checkpoint import load_optimizer_checkpoint
  out.mkdir(parents=True,exist_ok=False);gate=PacedGate(out/'resources_PRIVATE.jsonl');cache={'dir':scope['derived_cache']['cache_dir']}
  def before(name):gate.before_unit(name,loading=is_loading_unit(name,cache))
  before('paired_gradient_before_native');certpath=a.primary_root.resolve()/'results/EXACT_QUANT_CACHE_CERTIFICATE.json';cert=load(certpath);model=Path(scope['model_path']);artifacts={str(model/n):sha(model/n) for n in set(cert['non_weight_files_identical'])|{'config.json'}};require(verify_cache_identity(cache['dir'],certpath,scope['derived_cache']['certificate_sha256'],artifacts)==scope['derived_cache'],'Effective base cache changed');backend=None;result=None;state=rng=None
  try:
   backend=NativeGSPOBackend(scope['adapter']['path'],{'group_size':4,'max_tokens':256},model_path=model,contract=load_contract(Path(original['prompt_contract_path'])),seed=13,before_unit=before,cache=cache);before('paired_gradient_restore_child56');state,rng,_=load_optimizer_checkpoint(boundary/'optimizer',expected_bindings={'run':br['binding'],'adapter_sha256':scope['adapter']['sha256'],'trainable_signature':backend.trainable_signature()},load_arrays=backend.load_arrays);backend.restore(state,rng);require(int(backend.optimizer_state()['step'].item())==56,'Child Adam56 restore failed');result=helper.run_native(backend,scope['selected_pair_rows_PRIVATE'],plan,paired,al,utility,out,before)
  finally:
   del state,rng;gc.collect()
   if backend is not None:backend.close();gc.collect();backend.mx.clear_cache();active=int(backend.mx.get_active_memory());require(backend.model is None and active<128*1024**2,'Native release failed')
  require(result is not None,'No completed measurement');result.update(status='ACTUAL_THREE_PAIR_GRADIENT_CONDITIONS_COMPLETE_AND_RELEASED',scope_sha256=entry.canon(scope),scope_source_sha256=scope['entry_sha256'],native_release_verified=True,active_MLX_bytes_after_release=active,adapter_sha256=scope['adapter']['sha256']);entry.save(out/'complete_SAFE.json',result);print(json.dumps({'status':result['status'],'safe_receipt_sha256':sha(out/'complete_SAFE.json'),'weight_updates':0},sort_keys=True))

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--primary-root',type=Path,required=True);p.add_argument('--output-directory',type=Path,required=True);p.add_argument('--scope-output',type=Path);p.add_argument('--review',type=Path);p.add_argument('--expected-review-sha256');p.add_argument('--execute',action='store_true');a=p.parse_args();values=prepare(a)
 if not a.execute:require(a.scope_output and a.scope_output.resolve().is_relative_to(W/'local_data'),'Private scope required');values[1].save(a.scope_output,values[-3]);print(json.dumps({'status':'SCOPE_ONLY_NO_NATIVE','scope_sha256':sha(a.scope_output),'budget':values[-3]['budget']}));return
 execute(a,values)
if __name__=='__main__':main()
