"""One reviewed, read-only resident Qwen vision stage: exactly24train+24dev PNGs.
Preview is CPU-only. No language forward, generation, backward, optimizer update,
quadrant pooling, Dallas extraction, data changes or automatic model launch.
"""
import argparse,fcntl,gc,hashlib,json,os,sys
from pathlib import Path
import numpy as np
W=Path(__file__).resolve().parents[1];sys.path.insert(0,str(W/'src'))
from mri_rl.v2_qwen_feature_probe import select_sequence_records,pool_merged,dual_ridge,sequence_metrics,paired_sequence
from mri_rl.two_task_pool import write_exact
INPUT_SHA='446808a0203ba1e6dd9f45e7f052a7667420b016ffff5a5b88f295d728fc79ab';OLDPLAN_SHA='732260275ca977ccce1b59ea5856701f77a54b0150899acfc17aee2a6484b0ab'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
canon=lambda v:hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
load=lambda p:json.loads(Path(p).read_text())
def save(path,value):write_exact(path,(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode())
def verify_source_closure(R,sources):
 for name,digest in sources.items():
  path=(R/'../../work/mlx-venv/lib/python3.13/site-packages/mlx_vlm'/name.removeprefix('installed_mlx_vlm/')) if name.startswith('installed_mlx_vlm/') else R/'scripts/run_c40_qwen_visual_probe.py' if name=='entry_script' else R/'src/mri_rl'/name
  assert sha(path)==digest,'Existing reviewed native source changed'
  if not name.startswith('installed_mlx_vlm/') and name!='entry_script':assert sha(W/'src/mri_rl'/name)==digest,'Worktree imported native source differs from primary binding'
def bind_direct_dependencies(R,sources):
 sources=dict(sources)
 for name in ('experiments.py','two_task_pool.py','vero_mri_reward.py'):sources[name]=sha(R/'src/mri_rl'/name)
 verify_source_closure(R,sources);return sources
def work_budget(completed,release_complete):
 assert type(completed) is int and 0<=completed<=48 and (not release_complete or completed==48),'Invalid completion/release state'
 return {'new_vision_forwards':48-completed,'max_model_loads':int(not release_complete),'mode':'ALREADY_RELEASED_CPU_ANALYSIS_ONLY' if release_complete else 'INTEGRITY_FINALIZATION_ONLY' if completed==48 else 'EXTRACT_REMAINING_AND_FINALIZE'}
def release_backend(backend):
 backend.close();gc.collect();backend.mx.clear_cache();active=int(backend.mx.get_active_memory());model_none=backend.model is None
 assert model_none and active<128*1024**2,'Native model or active allocations were not released'
 return {'native_release_verified':True,'model_is_none':model_none,'active_MLX_bytes':active,'active_limit_bytes':128*1024**2}
def validate_completion(receipt,identity,plan_sha,record_hashes):
 assert len(record_hashes)==48 and receipt['run_identity']==identity and receipt['plan_sha256']==plan_sha and receipt['record_sha256']==record_hashes and receipt['adapter_tensors_compared_after']==504 and receipt['optimizer_step']==receipt['weight_updates']==0,'Native final integrity binding differs'
 released=receipt['release'];assert released['native_release_verified'] is True and released['model_is_none'] is True and released['active_limit_bytes']==128*1024**2 and 0<=released['active_MLX_bytes']<128*1024**2,'Native completion lacks verified post-close release'
def preview(a):
 R=a.primary_root.resolve();out=a.output_directory.resolve();assert out.is_relative_to(W/'local_data') and a.frozen_inputs.resolve().is_relative_to(W/'local_data');assert sha(a.frozen_inputs)==INPUT_SHA
 frozen=load(a.frozen_inputs);rows=select_sequence_records(frozen['train_PRIVATE'],frozen['dev_PRIVATE']);images=frozen['images_PRIVATE'];assert len(rows)==48
 oldpath=R/'local_data/c_two_task_pool_v1/u7_visual_controls_v1/probe_plan_PRIVATE.json';assert sha(oldpath)==OLDPLAN_SHA;old=load(oldpath);adapter=old['checkpoints']['B28'];assert sha(adapter['path'])==adapter['sha256'] and sha(Path(adapter['path']).with_name('adapter_config.json'))==adapter['adapter_config_sha256']
 for row in rows:assert sha(row['images'][0])==images[row['id']]
 # Bind unchanged certified native code plus the new narrow source closure.
 prior=load(W/'local_data/c40_early_visual_v1/B28_scope_RECOVERY_V6_READY_FOR_REVIEW_PRIVATE.json');sources=dict(prior['sources'])
 sources=bind_direct_dependencies(R,sources)
 sources.update({'V2_feature_helper':sha(W/'src/mri_rl/v2_qwen_feature_probe.py'),'V2_CPU_helper':sha(W/'src/mri_rl/v2_cpu_representation.py'),'V2_feature_entry':sha(__file__)})
 native=(R/'../../work/mlx-venv/lib/python3.13/site-packages/mlx_vlm').resolve();assert sha(native/'models/qwen3_vl/vision.py')=='9885d83cf85e037a88f8364817007094a82b5941d49e909d2d9fe11b5e03bf6c';sources['installed_vision']=sha(native/'models/qwen3_vl/vision.py')
 contract=load(old['prompt_contract_path']);plan={'schema':'frozen-Qwen-merged-mean48-v4','input_sha256':INPUT_SHA,'original_prompt_plan_sha256':OLDPLAN_SHA,'rows_PRIVATE':rows,'image_sha256_PRIVATE':{r['id']:images[r['id']] for r in rows},'train_rows':24,'dev_rows':24,'train_people':4,'dev_people':4,'selection':'Lexically first4 frozen training patient keys, all3 paired levels; all4 fixed dev people. No score argument.','model_path':old['model_path'],'adapter':adapter,'derived_cache':old['derived_cache'],'contract_path':old['prompt_contract_path'],'contract_sha256':sha(old['prompt_contract_path']),'system_instruction':contract['system_instruction'],'pooling':'Merged vision tokens castF32; arithmetic mean over all tokens;4096features. Ignore deepstack outputs.','spatial_pooling':'NOT_VERIFIED_NOT_RUN; no claim about complete encoder adequacy from mean-only probe','vision_call':'model.vision_tower(pixel_values.astype(model.vision_tower.patch_embed.proj.weight.dtype), image_grid_thw)','runtime_versions':{'mlx':'0.32.4','mlx-vlm':'0.7.8'},'forward_budget':48,'language_forwards':0,'generations':0,'backwards':0,'weight_updates':0,'head':'Fixed alpha1 train-standardized dual ridge; solve24x24, never4096x4096; same head for matched CPU pixel/hist features','sources':sources,'sealed_final100_used':False}
 out.mkdir(parents=True,exist_ok=True);save(out/'plan_PRIVATE.json',plan);identity=canon(plan);completed=[]
 for i,path in enumerate(sorted(out.glob('record_*.json'))):
  assert path.name==f'record_{i:04d}.json' and i<48;r=load(path);expect=rows[i];assert r['index']==i and r['run_identity']==identity and r['record_id']==expect['id'] and r['image_sha256']==images[expect['id']] and r['forward_count']==1
  vector=np.array(r['mean_feature']);assert vector.shape==(4096,) and np.isfinite(vector).all() and r['feature_sha256']==canon(r['mean_feature']);completed.append(sha(path))
 assert not list(out.glob('*.partial')),'Preserve incomplete evidence and reconcile before resume'
 completion=out/'native_complete_SAFE.json';released=completion.is_file()
 if released:validate_completion(load(completion),identity,sha(out/'plan_PRIVATE.json'),completed)
 budget=work_budget(len(completed),released)
 scope={'execution_tier':'B','kind':'QWEN_VISION_MEAN48_READ_ONLY_V1','primary_root':str(R),'plan_path':str(out/'plan_PRIVATE.json'),'plan_sha256':sha(out/'plan_PRIVATE.json'),'run_identity':identity,'source_sha256':sources,'adapter':adapter,'derived_cache':old['derived_cache'],'output_directory':str(out),'completed_records':len(completed),'completed_record_sha256':completed,**budget,'existing_completion_sha256':sha(completion) if released else None,'language_forwards':0,'generations':0,'backwards':0,'weight_updates':0,'patient_split':'4train people/4dev people;24+24paired sequences only','global_lock':str(R/'local_data/c_two_task_pool_v1/tier_c_v1/remaining20_bundle_v1/remaining20.lock')}
 return plan,scope

def analyze(a,plan,scope):
 # CPU heads fit only after all48 genuine private vectors are durable.
 from mri_rl.v2_cpu_representation import pixel_features
 from PIL import Image
 from mri_rl.vero_mri_reward import _answer_and_format
 out=a.output_directory.resolve();rows=plan['rows_PRIVATE'];raw=[load(out/f'record_{i:04d}.json') for i in range(48)];dev=rows[24:];y=np.array([int(r['reward_model']['ground_truth']=='contrast-enhanced T1') for r in rows[:24]]);features={'Qwen_merged_mean':np.array([r['mean_feature'] for r in raw]),'pixel_thumbnail16':[],'intensity_histogram16':[]};private={};predictions={}
 for row in rows:
  assert sha(row['images'][0])==plan['image_sha256_PRIVATE'][row['id']]
  with Image.open(row['images'][0]) as image:v,h=pixel_features(image)
  features['pixel_thumbnail16'].append(v);features['intensity_histogram16'].append(h)
 for name,value in features.items():
  value=np.array(value);p,proof=dual_ridge(value[:24],y,value[24:]);predictions[name]=p;private[name]=proof
 predictions['null_training_prior']=np.full(24,int(y.mean()>.5));R=a.primary_root.resolve()
 for name,path,key,expected in [('B28_saved_native',R/'local_data/c_two_task_pool_v1/evaluation_v1/C40_B28_baseline_independent_scoring_PRIVATE.json','normalized_response_components_PRIVATE','32ce2cd88a9d316e94f75177eb8c6c36e33bcd351fe4028ed9e88aa3c26dda8f'),('U7_saved_native',R/'local_data/c_two_task_pool_v1/tier_c_v1/U_raw_seed7_final_C40_release_v1/final_C40_scoring_PRIVATE.json','normalized_new_outputs_PRIVATE','de73178d59d5894d7ddfe4494f05dd785aa35211905b9457e9a1757e1f8b059e')]:
  assert sha(path)==expected;n=load(path)[key];p=[]
  for row in dev:
   answer,fmt=_answer_and_format(n[row['id']]['response']);labels=('flair','contrast-enhanced t1');p.append(labels.index(answer.lower()) if fmt>0 and answer.lower() in labels else -1)
  predictions[name]=np.array(p)
 save(out/'linear_fits_predictions_PRIVATE.json',{'fits_PRIVATE':private,'predictions_PRIVATE':{k:v.tolist() for k,v in predictions.items()},'dev_record_ids_PRIVATE':[r['id'] for r in dev]})
 report={'schema':'Qwen-frozen-merged-mean48-result-v1','status':'ACTUAL_FEATURE_EXTRACTION_AND_CPU_HEAD_FIT_COMPLETE','source_author':'Codex /root/two_task_admission','execution_attribution':'Actual launcher and exit witness required; not inferred from source authorship','vision_forwards':48,'head_training_images':24,'dev_images':24,'train_people':4,'dev_people':4,'Qwen_mean_dimensions':4096,'pooling':plan['pooling'],'spatial_pooling':plan['spatial_pooling'],'metrics':{k:sequence_metrics(dev,v) for k,v in predictions.items()},'Qwen_head_paired_delta':{k:paired_sequence(dev,v,predictions['Qwen_merged_mean']) for k,v in predictions.items() if k!='Qwen_merged_mean'},'evidence_sha256':{'plan':sha(out/'plan_PRIVATE.json'),'native_record_files':[sha(out/f'record_{i:04d}.json') for i in range(48)],'native_completion':sha(out/'native_complete_SAFE.json'),'fitted_private_parameters':sha(out/'linear_fits_predictions_PRIVATE.json')},'comparability':'All ridge heads use exactly same4train people24images and4dev people24images; original28-person CPU logistic remains separate. Native B28/U7 responses differ in supervision/model and are reference only.','limitations':['Repeated internal dev4people only; no independent clinical or training-seed generalization.','Merged mean discards token positions and excludes deepstack features; failure does not establish entire frozen visual encoder inadequacy.','Acquisition/intensity shortcuts remain; successful sequence classification does not prove anatomical or lesion reasoning.','Dallas native features NOT_RUN; full216+40 extraction NOT_RUN; no VLM language forward/generation/backward/update or sealed final100 access.']}
 save(out/'analysis_SAFE.json',report);return {'status':report['status'],'safe_report_sha256':sha(out/'analysis_SAFE.json'),'output_directory':str(out)}
def execute(a):
 from mri_rl.experiment_contract import _check_review,load_contract
 plan,scope=preview(a);_check_review(review_verdict_path=a.review,expected_review_sha256=a.expected_review_sha256,scope=scope)
 out=a.output_directory.resolve();R=a.primary_root.resolve();lockpath=Path(scope['global_lock']);assert lockpath.is_file(),'Existing global lock required'
 if scope['mode']=='ALREADY_RELEASED_CPU_ANALYSIS_ONLY':return analyze(a,plan,scope)
 with lockpath.open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  from importlib.metadata import version
  assert {name:version(name) for name in plan['runtime_versions']}==plan['runtime_versions'],'Exact native runtime versions required'
  from mri_rl.paced_resources import PacedGate
  from mri_rl.mlx_continuous_gspo import NativeGSPOBackend,is_loading_unit
  from mri_rl.likelihood_cache import verify_cache_identity
  gate=PacedGate(out/'resources_PRIVATE.jsonl');cache={'dir':plan['derived_cache']['cache_dir']}
  def before(name):return gate.before_unit(name,loading=is_loading_unit(name,cache))
  gate.before_unit('feature_before_native_import',loading=True)
  certificate=R/'results/EXACT_QUANT_CACHE_CERTIFICATE.json';cert=load(certificate);model=Path(plan['model_path']);artifacts={str(model/name):sha(model/name) for name in set(cert['non_weight_files_identical'])|{'config.json'}}
  assert verify_cache_identity(cache['dir'],certificate,plan['derived_cache']['certificate_sha256'],artifacts)==plan['derived_cache'];backend=None;integrity=None;released=None;disk=current=prepared=ids=grid=pixels=merged=deep=mean=None
  try:
   backend=NativeGSPOBackend(plan['adapter']['path'],{'group_size':2,'max_tokens':256},model_path=model,contract=load_contract(Path(plan['contract_path'])),seed=7,before_unit=before,cache=cache);mx=backend.mx;config=backend.model.config.vision_config;assert config.out_hidden_size==4096 and config.spatial_merge_size==2;backend.model.eval()
   for i,row in enumerate(plan['rows_PRIVATE'][scope['completed_records']:],scope['completed_records']):
    before(f'feature_prepare_{i}');assert sha(row['images'][0])==plan['image_sha256_PRIVATE'][row['id']];messages=[{'role':'system','content':plan['system_instruction']},{'role':'user','content':row['extra_info']['question']}];prompt=backend.template(backend.processor,backend.model.config,messages,num_images=1,enable_thinking=False,add_generation_prompt=True);assert prompt.count('<|image_pad|>')==1 and prompt.endswith('<|im_start|>assistant\n');prepared=backend.prepare_inputs(backend.processor,images=[row['images'][0]],prompts=prompt,image_token_index=backend.model.config.image_token_index,add_special_tokens=backend.special_tokens(backend.model.config.model_type,backend.processor));ids=prepared['input_ids'];assert ids.shape[0]==1 and 0<ids.shape[1]<=2048 and bool(mx.all(prepared['attention_mask']==1).item());grid=prepared['image_grid_thw'];gridlist=grid.tolist();tokens=ids[0].tolist();visual=tokens.count(backend.model.config.image_token_index);assert visual==sum(t*h*w//4 for t,h,w in gridlist) and visual>0
    before(f'feature_vision_forward_{i}');pixels=prepared['pixel_values'];dtype=backend.model.vision_tower.patch_embed.proj.weight.dtype;merged,deep=backend.model.vision_tower(pixels.astype(dtype),grid);mx.eval(merged);shape=list(merged.shape);mean=pool_merged(np.asarray(merged.astype(mx.float32)),gridlist,2,4096);assert sha(row['images'][0])==plan['image_sha256_PRIVATE'][row['id']]
    record={'index':i,'run_identity':scope['run_identity'],'record_id':row['id'],'image_sha256':plan['image_sha256_PRIVATE'][row['id']],'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'prompt_ids':tokens,'prompt_tokens':len(tokens),'image_grid_thw':gridlist,'image_token_count':visual,'merged_shape':shape,'vision_native_dtype':str(dtype),'mean_feature':mean.tolist(),'feature_sha256':canon(mean.tolist()),'forward_count':1,'weight_updates':0}
    save(out/f'record_{i:04d}.json',record);del prepared,ids,grid,pixels,merged,deep,mean;gc.collect();mx.clear_cache();before(f'feature_saved_{i}');print(json.dumps({'completed':i+1,'total':48,'weight_updates':0}),flush=True)
   before('feature_readonly_adapter_check');disk=mx.load(plan['adapter']['path']);current=dict(backend.tree_flatten(backend.model.trainable_parameters()));assert len(current)==504 and set(current)==set(disk) and all(bool(mx.array_equal(v,current[k]).item()) for k,v in disk.items());assert int(backend.optimizer_state()['step'].item())==0
   integrity={'status':'ACTUAL48_VISION_FORWARDS_READONLY_COMPLETE_AND_RELEASED','plan_sha256':sha(out/'plan_PRIVATE.json'),'run_identity':scope['run_identity'],'adapter_sha256':plan['adapter']['sha256'],'base_parameter_constructor_checks':1438,'adapter_tensors_compared_after':504,'optimizer_step':0,'record_sha256':[sha(out/f'record_{i:04d}.json') for i in range(48)],'weight_updates':0,'finalization_only_no_new_forwards':scope['mode']=='INTEGRITY_FINALIZATION_ONLY'}
  finally:
   # These local dictionaries/tensors otherwise keep MLX arrays alive after close.
   disk=current=prepared=ids=grid=pixels=merged=deep=mean=None
   if backend is not None:released=release_backend(backend)
  assert integrity is not None and released is not None,'No successful final integrity/release evidence'
  integrity['release']=released;validate_completion(integrity,scope['run_identity'],scope['plan_sha256'],integrity['record_sha256']);save(out/'native_complete_SAFE.json',integrity)
 return analyze(a,plan,scope)
def main():
 p=argparse.ArgumentParser()
 for k in ('primary-root','frozen-inputs','output-directory'):p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--scope-output',type=Path);p.add_argument('--execute',action='store_true');p.add_argument('--review',type=Path);p.add_argument('--expected-review-sha256');a=p.parse_args()
 assert all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'))
 if a.execute:result=execute(a)
 else:
  plan,scope=preview(a);assert a.scope_output and a.scope_output.resolve().is_relative_to(W/'local_data');save(a.scope_output,scope);result={'status':'SCOPE_READY_NOT_RUN','plan_sha256':scope['plan_sha256'],'scope_sha256':sha(a.scope_output),'pending_vision_forwards':scope['new_vision_forwards']}
 print(json.dumps(result,sort_keys=True))
if __name__=='__main__':main()
