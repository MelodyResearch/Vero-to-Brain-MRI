"""Training-only V2 paired label discrimination; source/CPU preparation, NOT_RUN.

Two Adam updates maximum; four balanced same-person/level pairs, two pairs per
update. Uses existing SFT atomic adapter/Adam/RNG journal and NativeGSPO loader.
The new objective is CE over two MEAN label-token log scores. Prefix/suffix
log-probability terms are excluded. No new prompt text, vision unfreezing or RL.
"""
from __future__ import annotations
import fcntl,gc,hashlib,json,math,sys
from collections import defaultdict
from pathlib import Path

LABELS=('FLAIR','contrast-enhanced T1')
PACK_SHA='446808a0203ba1e6dd9f45e7f052a7667420b016ffff5a5b88f295d728fc79ab'
MAX_UPDATES=2

def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)
def identity(x):return hashlib.sha256(canonical(x).encode()).hexdigest()
def discrimination_loss(chosen,partitions,truth,ops):
 """Stable differentiable binary CE, using ONLY mean label token logps."""
 require(len(chosen)==len(partitions)==2 and truth in (0,1),'Exactly two candidates and binary truth required')
 scores=[]
 for vector,part in zip(chosen,partitions):
  start=len(part['prefix_ids']);n=len(part['label_ids']);require(n>0,'Empty label mask')
  scores.append(ops.mean(vector[start:start+n]))
 return ops.logaddexp(scores[1-truth]-scores[truth],0.)

def validate_candidate_alignment(tokenizer,prompt,input_ids,target_text,part,image_token_id,visual_tokens):
 """Verify raw text boundary separately from processor-expanded image prefix."""
 enc=lambda text:list(tokenizer.encode(text,add_special_tokens=False))
 raw=enc(prompt);require(type(visual_tokens) is int and visual_tokens>0 and raw.count(image_token_id)==1,'Exactly one raw image placeholder and measured visual tokens required')
 expanded=[token for value in raw for token in ([value]*visual_tokens if value==image_token_id else [value])]
 require(expanded==list(input_ids) and input_ids.count(image_token_id)==visual_tokens,'Native expanded image prefix mismatch')
 require(enc(target_text)==part['target_ids'] and enc(prompt+target_text)==raw+part['target_ids'],'Raw prompt/target tokenizer boundary mismatch')
 start=len(input_ids)+len(part['prefix_ids'])-1
 require(part['label_logit_positions']==list(range(start,start+len(part['label_ids']))),'Expanded causal label positions mismatch')

def select_four_pairs(rows):
 groups=defaultdict(dict)
 for row in rows:
  if row['ability']!='mri_sequence':continue
  e=row['extra_info'];require(e['internal_split']=='train','Training-only pairs required')
  label=row['reward_model']['ground_truth'];require(label in LABELS and len(row['images'])==1,'Invalid sequence training label/image')
  key=(e['patient_key'],e['level']);require(label not in groups[key],'Duplicate paired class');groups[key][label]=row
 require(len(groups)>=4 and all(set(g)==set(LABELS) for g in groups.values()),'Incomplete opposite-class same-person/level pair')
 selected=[]
 for key in sorted(groups)[:4]:
  pair=[groups[key][label] for label in LABELS]
  require(pair[0]['extra_info']['question']==pair[1]['extra_info']['question'],'Paired prompt differs')
  selected.extend(pair)
 return selected

def primary_modules(primary):
 primary=Path(primary).resolve();sys.path.insert(0,str(primary/'src'))
 from mri_rl import tier_c_sft as sft,answer_likelihood as al,tier_c_entry as entry
 for module in (sft,al,entry):require(Path(module.__file__).resolve().is_relative_to(primary/'src'),'Wrong primary source imported')
 return sft,al,entry

def compile_pilot(primary,pack_path,tokenizer_json,*,input_counts=None,measurement_evidence=None):
 """CPU compile; missing actual measurements deliberately blocks native entry."""
 require(sha(pack_path)==PACK_SHA,'Frozen train-only source pack changed');pack=json.loads(Path(pack_path).read_text());sft,al,entry=primary_modules(primary)
 require(sum(r['ability']=='mri_sequence' for r in pack['train_PRIVATE'])==168,'Exact existing168 training-image inventory required')
 rows=select_four_pairs(pack['train_PRIVATE'])
 images={r['id']:sha(r['images'][0]) for r in rows};require(all(pack['images_PRIVATE'][k]==v for k,v in images.items()),'Frozen training pixels changed')
 from tokenizers import Tokenizer
 tok=Tokenizer.from_file(str(tokenizer_json))
 class Raw:
  def encode(self,text,add_special_tokens=False):return tok.encode(text,add_special_tokens=add_special_tokens).ids
 parts={label:al.partition_target(Raw(),label,prompt_length=1) for label in LABELS}
 targets={r['id']:{label:parts[label]['target_ids'] for label in LABELS} for r in rows}
 target_cost={r['id']:sum(len(parts[label]['target_ids']) for label in LABELS) for r in rows}
 require(all(n<=sft.TARGET_CAP for n in target_cost.values()),'Combined candidate target cost exceeds inherited SFT target cap')
 measured=input_counts is not None
 if measured:
  require(set(input_counts)=={r['id'] for r in rows} and all(type(x) is int and 0<x<=2048 for x in input_counts.values()),'Exact selected native input counts required')
  require(measurement_evidence and sha(measurement_evidence['path'])==measurement_evidence['sha256'],'Externally bound actual measurement evidence required')
  measurement=json.loads(Path(measurement_evidence['path']).read_text())
  require(measurement['schema']=='paired-pilot-native-processor-inputs-v1' and measurement['model_weights_loaded'] is False and measurement['input_counts']==input_counts and measurement['tokenizer_sha256']==sha(tokenizer_json),'Actual processor measurement values/tokenizer differ')
  measured_rows={r['record_id']:r for r in measurement['rows_PRIVATE']}
  require(set(measured_rows)==set(images) and all(measured_rows[k]['image_sha256']==v for k,v in images.items()),'Actual selected image measurements differ')
 sources={name:sha(Path(primary)/'src/mri_rl'/name) for name in (*entry.CRITICAL_SOURCES,'tier_c_sft.py')}
 return {'schema':'paired-mean-label-discrimination-pilot-v2','source_pack_sha256':PACK_SHA,'source_manifest_sha256':pack['source_sha256']['full_sequence_train'],
  'objective':'mean-label two-candidate cross-entropy; no prefix/suffix loss; average four images (two balanced pairs) per update',
  'selection':'First four (patient_key,level) keys in frozen train pack; class order FLAIR,T1; no outcome/dev selection',
  'rows_PRIVATE':rows,'image_sha256':images,'target_ids':targets,'tokenizer_sha256':sha(tokenizer_json),'input_tokens':input_counts,
  'measurement_evidence':measurement_evidence,'target_token_cost':target_cost,'schedule':[[0,1,2,3],[4,5,6,7]],'maximum_updates':MAX_UPDATES,
  'maximum_image_examples':8,'maximum_teacher_forced_candidate_sequences':16,'new_generation_tokens':0,'seed':7,
  'optimizer':{'name':'Adam','learning_rate':1e-5,'betas':[.9,.999],'eps':1e-8,'bias_correction':False},
  'trainable_parameters':'Existing common-B28 language LoRA only, exact504 tensors','vision_unfreeze':False,
  'sources':sources,'objective_source_sha256':sha(__file__),'native_admission':'NOT_VERIFIED_PENDING_FEATURE_PROOF_AND_SCOPED_REVIEW' if measured else 'NOT_VERIFIED_PENDING_ACTUAL_INPUT_MEASUREMENTS_AND_FEATURE_PROOF',
  'native_training_executed':False,'independent_patient_generalization_claimed':False}

def native_backend_class(primary):
 primary_modules(primary)
 from mri_rl.mlx_continuous_gspo import NativeGSPOBackend
 from mri_rl import answer_likelihood as al
 class NativePairedDiscriminationBackend(NativeGSPOBackend):
  def sft_step(self,examples,work):
   mx,nn=self.mx,self.nn;self.model.eval();grad_sum=None;losses=[];prompt_costs=[];target_costs=[]
   require(len(examples)==4,'Two balanced pairs per update required')
   require(len(self.tree_flatten(self.model.trainable_parameters()))==504,'Unexpected trainable inventory')
   def loss_fn(model,prompt_ids,pixels,kwargs,parts,truth):
    chosen=[];P=int(prompt_ids.shape[-1])
    for part in parts:
     target=mx.array(part['target_ids'],dtype=prompt_ids.dtype);ids=mx.concatenate((prompt_ids,target[None,:]),axis=1);inputs=ids[:,:-1]
     logits=model(inputs,pixels,mx.ones_like(inputs),**kwargs).logits
     require(int(logits.shape[1])==int(inputs.shape[1]),'Causal logits length mismatch')
     lp=nn.log_softmax(logits[:,P-1:,:].astype(mx.float32),axis=-1)
     chosen.append(mx.take_along_axis(lp,target[None,:,None],axis=-1)[0,:,0])
    return discrimination_loss(chosen,parts,truth,mx)
   vg=nn.value_and_grad(self.model,loss_fn)
   for row in examples:
    self.before_unit('paired_discrimination_example');require(sha(row['images'][0])==self.expected_images[row['id']],'Frozen image changed')
    messages=[{'role':'system','content':self.contract.system_instruction},{'role':'user','content':row['extra_info']['question']}]
    prompt=self.template(self.processor,self.model.config,messages,add_generation_prompt=True,num_images=1,enable_thinking=self.contract.enable_thinking)
    require(prompt.count('<|image_pad|>')==1 and prompt.endswith(self.contract.assistant_prefix),'Native prompt contract mismatch')
    prepared=self.prepare_inputs(self.processor,images=row['images'],prompts=prompt,image_token_index=getattr(self.model.config,'image_token_index',None),add_special_tokens=self.special_tokens(self.model.config.model_type,self.processor))
    ids=prepared['input_ids'];P=int(ids.shape[-1]);require(P==self.pilot_plan['input_tokens'][row['id']] and bool(mx.all(prepared['attention_mask']==1).item()),'Native measured prompt/mask mismatch')
    parts=[al.partition_target(self.processor.tokenizer,label,prompt_length=P) for label in LABELS]
    for label,part in zip(LABELS,parts):
     require(part['target_ids']==self.expected_target_ids[row['id']][label],'Frozen candidate target token IDs mismatch')
     grid=prepared['image_grid_thw'].tolist();merge=int(self.processor.image_processor.merge_size)
     require(merge>0 and prepared.get('pixel_values') is not None,'Missing native visual conditioning')
     visual=sum(t*h*w//(merge*merge) for t,h,w in grid)
     validate_candidate_alignment(self.processor.tokenizer,prompt,ids[0].tolist(),al.PREFIX+label+al.SUFFIX,part,self.model.config.image_token_index,visual)
    kwargs={k:v for k,v in prepared.items() if k not in {'input_ids','attention_mask','pixel_values'}}
    loss,grad=vg(self.model,ids,prepared.get('pixel_values'),kwargs,parts,LABELS.index(row['reward_model']['ground_truth']));mx.eval(loss,grad)
    value=float(loss.item());require(math.isfinite(value),'Nonfinite discriminative loss')
    scaled=self.tree_map(lambda g:g/4,grad);grad_sum=scaled if grad_sum is None else self.tree_map(lambda a,b:a+b,grad_sum,scaled);mx.eval(grad_sum)
    losses.append(value);prompt_costs.append(2*P);target_costs.append(sum(len(p['target_ids']) for p in parts))
    del prepared,ids,kwargs,grad,scaled,loss;gc.collect();mx.clear_cache()
   norm=math.sqrt(sum(float(mx.sum(g.astype(mx.float32)**2).item()) for _,g in self.tree_flatten(grad_sum)))
   require(math.isfinite(norm) and norm>0,'Finite nonzero discriminative gradient required')
   self.before_unit('optimizer_update');self.optimizer.update(self.model,grad_sum);mx.eval(self.model.trainable_parameters(),self.optimizer.state);mx.clear_cache()
   return {'loss_terms':losses,'gradient_l2':norm,'prompt_tokens':prompt_costs,'target_tokens':target_costs}
 return NativePairedDiscriminationBackend

def run_pilot(*,primary,plan,output,adapter,model_path,contract,cache,before_unit,authorize,feature_proof,feature_proof_sha256,resume=False,max_new_steps=2):
 """Executable native path; future independent scope/feature evidence mandatory.

Uses SAME existing SFT atomic checkpoint/optimizer/RNG/resume journal. Its inherited
profile name is an engineering wrapper, while run bindings explicitly name this
new objective. Exact cost counts BOTH candidate forwards per image, unlike old CE.
 """
 sft,al,entry=primary_modules(primary)
 require(sha(adapter)=='b316db3d5eadf2e1fc0cbe80ccc93c7f7405858769f7d32b39a137dea3d6c572' and sha(Path(adapter).with_name('adapter_config.json'))=='032becabb7cbb6d2af6f3451eb127e5f944551d42eb51eaee3e8d75b839b2f6a','Only exact common-B28 starting weights/config are allowed')
 from mri_rl.experiment_contract import load_contract
 from mri_rl.likelihood_cache import verify_cache_identity
 baseline_path=Path(primary)/'local_data/c_two_task_pool_v1/u7_visual_controls_v1/probe_plan_PRIVATE.json'
 require(sha(baseline_path)=='732260275ca977ccce1b59ea5856701f77a54b0150899acfc17aee2a6484b0ab','Common model/prompt metadata changed')
 baseline=json.loads(baseline_path.read_text());require(str(model_path)==baseline['model_path'] and contract.identity()==load_contract(Path(baseline['prompt_contract_path'])).identity(),'Common base model/native prompt differs')
 certificate=Path(primary)/'results/EXACT_QUANT_CACHE_CERTIFICATE.json';cert=json.loads(certificate.read_text())
 artifacts={str(Path(model_path)/n):sha(Path(model_path)/n) for n in set(cert['non_weight_files_identical'])|{'config.json'}}
 require(cache and verify_cache_identity(cache['dir'],certificate,'8fcd2b0bd1122d78153ed8e2114204cdbe7fb095bb7a6273005731f907928160',artifacts)==baseline['derived_cache'],'Common certified effective weights differ')
 require(plan['objective_source_sha256']==sha(__file__) and all(sha(Path(primary)/'src/mri_rl'/n)==v for n,v in plan['sources'].items()),'Bound source changed')
 require(plan['input_tokens'] is not None and feature_proof and sha(feature_proof)==feature_proof_sha256,'Actual feature and input evidence required before any native load')
 require(plan['maximum_updates']==2 and max_new_steps in (1,2) and callable(authorize) and callable(before_unit),'Bounded independent scope and resource authorizer required')
 require(all(sha(r['images'][0])==plan['image_sha256'][r['id']] for r in plan['rows_PRIVATE']),'Bound pixels changed')
 from mri_rl.two_task_training import TokenLedger
 rows=plan['rows_PRIVATE'];steps=sft.step_rows(rows,plan['schedule']);inputs=lambda r:2*plan['input_tokens'][r['id']];targets=plan['target_token_cost']
 cap=sum(sft.step_cost(step,inputs,targets) for step in steps);ledger=TokenLedger({'budget':{'text_visual_token_cap':cap,'maximum_attempted_groups':2}})
 Native=native_backend_class(primary)
 def factory(starting,config):
  backend=Native(starting,config,model_path=model_path,contract=contract,seed=7,before_unit=before_unit,cache=cache);backend.pilot_plan=plan;return backend
 queue_lock=Path(primary)/'local_data/c_two_task_pool_v1/tier_c_v1/remaining20_bundle_v1/remaining20.lock'
 with queue_lock.open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  return sft.run_sft(output=output,steps=steps,bindings={'objective':'paired_mean_label_discrimination_v2','pilot_plan_sha256':identity(plan),'feature_proof_sha256':feature_proof_sha256},backend_factory=factory,authorize=authorize,ledger=ledger,input_tokens=inputs,target_tokens=targets,resume=resume,starting_adapter=adapter,max_new_steps=max_new_steps,expected_target_ids=plan['target_ids'])
