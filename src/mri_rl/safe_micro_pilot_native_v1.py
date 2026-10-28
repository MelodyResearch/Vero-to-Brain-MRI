"""Bounded genuine model stages; importing this module initializes no MLX.

Uses historical F32 generation/scorer, strict Vero parser, persistent Adam and
micro transaction. No auxiliary loss or sampler change. Separate child per stage.
"""
from __future__ import annotations
import gc,hashlib,json,math,time
from pathlib import Path
from types import SimpleNamespace
from mri_rl import shared_rollout_gspo_core_v1 as shared
require=shared.require


class CallCounter:
 def __init__(self,limits,path):self.limits=dict(limits);self.used={k:0 for k in limits};self.path=Path(path)
 def charge(self,key,n=1):
  require(key in self.used and type(n) is int and n>=0 and self.used[key]+n<=self.limits[key],'Frozen native-call budget exceeded')
  self.used[key]+=n;shared.replace_atomic(self.path,{'limits':self.limits,'used':self.used})


def forward_verdict(errors,*,different_pixels,same_target_delta):
 require(len(errors)==2 and all(math.isfinite(x) and 0<=x<=.002 for x in errors),'Actual F32 generation/scorer error exceeds .002')
 require(different_pixels and math.isfinite(same_target_delta),'Distinct actual image tensors and finite scores required')
 return 'PASS' if abs(same_target_delta)>1e-7 else 'PARTIAL'


def greedy_metrics(rows):
 from mri_rl.paired_z045_evaluation_v1 import strict_prediction,LABELS
 labels=LABELS['mri_sequence'];counts=[0,0];hits=[0,0];invalid=0
 for row in rows:
  y=labels.index(row['truth']);pred=strict_prediction(row['text'],'mri_sequence');counts[y]+=1;hits[y]+=pred==y;invalid+=pred<0
 require(all(counts),'Both sequence classes required')
 recalls=[hits[i]/counts[i] for i in (0,1)]
 return {'cases':len(rows),'correct':sum(hits),'accuracy':sum(hits)/len(rows),'FLAIR_recall':recalls[0],'T1_recall':recalls[1],'balanced_accuracy':sum(recalls)/2,'invalid':invalid,'class_counts':counts,'class_correct':hits}


def pilot_decision(metrics):
 b=metrics['B'];checks={base: {'balanced_accuracy_increases':b['balanced_accuracy']>metrics[base]['balanced_accuracy'],'T1_recall_increases':b['T1_recall']>metrics[base]['T1_recall'],'FLAIR_no_drop':b['FLAIR_recall']>=metrics[base]['FLAIR_recall']} for base in ('Parent','A')}
 return {'decision':'POSITIVE_ENGINEERING_SIGNAL_NO_AUTOMATIC_SCALE' if all(all(x.values()) for x in checks.values()) else 'NO_POSITIVE_ADAPTATION_SIGNAL','checks':checks,'generalization_proven':False,'candidate_and_Dallas_retention':'NOT_MEASURED_IN_GREEDY_SEQUENCE_ONLY_SCOPE','additional_training_authorized':False}


def verify_terminal_receipt(receipt,expected):
 require(receipt.get('receipt_sha256')==expected and shared.identity({k:v for k,v in receipt.items() if k!='receipt_sha256'})==expected,'Terminal receipt differs from completed training record')
 return True


def terminal_choice(receipt,parent_policy,directory):
 require(receipt['parent_policy_sha256']==parent_policy,'Different A/B parent')
 if not receipt['updated']:
  require(receipt['alias_of_parent'] is True and receipt['policy_sha_after']==parent_policy and receipt['adam_step_after']==52 and receipt['checkpoint'] is None and receipt['files_sha256']=={},'Zero-update alias lacks exact unchanged parent identity')
  return {'alias_of_parent':True,'updates':0,'policy_sha256':parent_policy}
 require(receipt['alias_of_parent'] is False and receipt['adam_step_after']==53 and receipt['checkpoint']=='checkpoint_PRIVATE' and receipt['policy_sha_after']!=parent_policy,'Single real Adam53 terminal required')
 path=Path(directory)/receipt['checkpoint']
 require(set(receipt['files_sha256'])=={'adapter.safetensors','optimizer/manifest.json','optimizer/state.safetensors'},'Complete terminal files required')
 for name,h in receipt['files_sha256'].items():require(shared.file_sha256(path/name)==h,'Terminal payload changed')
 return {'alias_of_parent':False,'updates':1,'path':str(path),'policy_sha256':receipt['policy_sha_after']}


def state_identity(backend):return {'policy':backend.policy_sha(),'adam':backend.adam_digest(),'rng':backend.rng_digest()}


def generate_one(backend,row,image_sha,*,greedy,counter,before):
 from mlx_vlm.generate.dispatch import stream_generate
 from mri_rl.paired_z045_f32_backend_v1 import F32GenerationModelView,ObservedFP32Processor,capture_stream,reset_effective_eos
 from mri_rl.paired_z045_matched_f32_eval_v1 import prepared_inputs,greedy_settings
 from mri_rl.native_gspo_path_probe_v1 import strict_tensor_identity
 from mri_rl import early_native_conditioning_probe_v1 as utility
 from mri_rl.paired_z045_continuation_entry_v1 import verify_images
 mx=backend.mx;expected={row['id']:image_sha};before('pilot_greedy' if greedy else 'pilot_F32_onpolicy');verify_images([row],expected)
 prompt,inputs=prepared_inputs(backend,row);P=int(inputs['input_ids'].shape[-1]);require(P<=213,'Frozen sequence prompt bound exceeded')
 observed=[];evidence=[]
 def observe(ids,pixels,mask,kwargs):
  extra={k:kwargs[k] for k in inputs if k not in ('input_ids','pixel_values','attention_mask')}
  observed.append(strict_tensor_identity(backend,{'input_ids':ids,'pixel_values':pixels,'attention_mask':mask,**extra},inputs,utility))
 def guard(name):
  before(name)
  if name=='f32_generation_LM_call':counter.charge('LM')
  if name=='f32_generation_vision_embeddings':counter.charge('vision');verify_images([row],expected)
 def processor(tokens,logits):counter.charge('sampler');return ObservedFP32Processor(mx)(tokens,logits)
 view=F32GenerationModelView(backend.model,mx,guard,evidence,observe);eos=reset_effective_eos(backend.processor.tokenizer,backend.model.config.eos_token_id)
 counter.charge('generations');settings=greedy_settings();settings['temperature']=0. if greedy else 1.
 response=capture_stream(stream_generate(view,backend.processor,prompt,image=row['images'],**settings,logits_processors=[processor]),prompt_tokens=P,max_tokens=256,eos_ids=eos,before=before)
 counter.charge('generated_tokens',len(response['token_ids']));require(len(observed)==1 and evidence,'Actual image preparation/F32 trace missing')
 clean=getattr(backend.processor,'clean_output',None)
 if callable(clean):response['text']=clean(response['text'])
 result={'response':response,'image_sha256':image_sha,'input_fingerprints_PRIVATE':observed[0],'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'F32_evidence_PRIVATE':evidence,'prompt_length':P}
 del inputs,view,observed,evidence;gc.collect();mx.clear_cache()
 return result


def score_one(backend,row,image_sha,tokens,*,counter,before):
 from mri_rl.paired_z045_f32_backend_v1 import f32_response_logps_observed
 from mri_rl.paired_z045_matched_f32_eval_v1 import prepared_inputs
 from mri_rl.paired_z045_continuation_entry_v1 import verify_images
 before('pilot_readonly_scorer');verify_images([row],{row['id']:image_sha});_,inputs=prepared_inputs(backend,row);P=int(inputs['input_ids'].shape[-1])
 require(0<len(tokens)<=256 and P<=213,'Bounded exact TF positions required');counter.charge('scorer');counter.charge('TF',P+len(tokens)-1)
 value=f32_response_logps_observed(backend,tokens,inputs);backend.mx.eval(value);result=value.tolist();require(all(math.isfinite(x) for x in result),'Finite selected probabilities required')
 del value,inputs;gc.collect();backend.mx.clear_cache();return result


def _readonly(backend,run,out):
 """Full1942 parameter + Adam integrity BEFORE restoring the generation RNG."""
 from mri_rl import early_native_conditioning_probe_v1 as utility
 from mri_rl.paired_z045_f32_qualification_v1 import _optimizer
 initial=backend.snapshot();identity=state_identity(backend)
 before=utility._fingerprints(backend,dict(backend.tree_flatten(backend.model.parameters())));adam=_optimizer(backend,utility)
 try:return run()
 finally:
  after=utility._fingerprints(backend,dict(backend.tree_flatten(backend.model.parameters())));adama=_optimizer(backend,utility)
  shared.publish_once(Path(out)/'readonly_integrity_PRIVATE.json',{'parameters_before':before,'parameters_after':after,'Adam_before':adam,'Adam_after':adama,'unchanged':before==after and adam==adama})
  require(len(before)==1942 and before==after and adam==adama,'Readonly operation mutated full parameters or Adam')
  backend.restore(initial);require(state_identity(backend)==identity,'Full readonly parent/RNG restore failed');initial=None;backend._unit_snapshot=None


def _stage2(backend,micro,dev,out,before):
 from mri_rl.paired_z045_f32_backend_v1 import generation_alignment
 units=micro['units_PRIVATE'];rows=[c['row'] for c in units[0]['contexts']];pins={x['record_id_PRIVATE']:x['image_binding_PRIVATE']['sha256'] for x in micro['planned_image_bindings_PRIVATE']}
 counter=CallCounter({'generations':2,'generated_tokens':512,'LM':520,'vision':2,'sampler':514,'scorer':6,'TF':2808},Path(out)/'calls_PRIVATE.json')
 def run():
  samples=[generate_one(backend,row,pins[row['id']],greedy=False,counter=counter,before=before) for row in rows]
  errors=[];details=[]
  for i,row in enumerate(rows):
   response=samples[i]['response'];own=score_one(backend,row,pins[row['id']],response['token_ids'],counter=counter,before=before)
   errors.append(generation_alignment(response['raw_selected_logps'],own)['max_abs_error'])
   other=rows[1-i];swapped=score_one(backend,other,pins[other['id']],response['token_ids'],counter=counter,before=before)
   details.append({'own_logps':own,'swapped_logps':swapped})
  tokens=dev['token_plans_PRIVATE']['mri_sequence'];target=tokens['prefix']+tokens['labels'][0]
  scores=[score_one(backend,row,pins[row['id']],target,counter=counter,before=before) for row in rows]
  n=len(tokens['labels'][0]);delta=sum(scores[0][-n:])/n-sum(scores[1][-n:])/n
  status=forward_verdict(errors,different_pixels=samples[0]['image_sha256']!=samples[1]['image_sha256'] and samples[0]['input_fingerprints_PRIVATE']!=samples[1]['input_fingerprints_PRIVATE'],same_target_delta=delta)
  shared.publish_once(Path(out)/'forward_actual_PRIVATE.json',{'samples':samples,'own_swapped':details,'canonical_fixed_first_label_scores':scores,'same_target_image_delta':delta,'parity_errors':errors,'status':status,'interpretation':'Same fixed token target image sensitivity; not two-class margin or anatomical grounding.'})
  return {'status':status,'parity_max_abs':max(errors),'same_target_image_delta':delta,'calls':counter.used,'optimizer_updates':0}
 return _readonly(backend,run,out)


def _configuration(runtime):
 from mri_rl.paired_z045_continuation_driver_v1 import PROFILE
 return {**PROFILE,'group_size':4,'max_tokens':256}


def run_phase(stage,*,spec,scope,authorization,resource_gate):
 """Called ONLY by a reviewed child holding the global flock after admission."""
 from mri_rl.safe_micro_pilot_entry_v1 import Authorization,PHASES
 require(isinstance(authorization,Authorization) and authorization.parent is not None and authorization.scope==scope and scope['execution_stage']==PHASES[stage],'Exact genuine child-stage authority required')
 require(resource_gate.authorization is authorization and resource_gate.scope==scope,'Matching resource gate required')
 authorization.check('native_import')
 from mri_rl.safe_micro_native_v1 import MicroNativeBackend
 from mri_rl.accuracy_stage_native_v1 import restore_parent
 from mri_rl import accuracy_recovery_v1 as recovery
 from mri_rl.experiment_contract import load_contract
 from mri_rl.two_task_training import make_grader
 from mri_rl.paired_z045_continuation_entry_v1 import run_with_release,verify_images
 from mri_rl import safe_micro_training_v1 as microcode
 runtime=json.loads(Path(spec['runtime_spec']).read_text());micro=json.loads(Path(spec['micro_input']).read_text());dev=json.loads(Path(spec['dev24']).read_text());out=Path(scope['output']);out.mkdir(exist_ok=False)
 parent=Path(runtime['parent_checkpoint']);disk_before=recovery.parent_disk_snapshot(parent);anchor=recovery.adam52_parent_anchor(parent,runtime['parent_receipt_sha256'])
 rows=[c['row'] for u in micro['units_PRIVATE'] for c in u['contexts']]
 images={x['record_id_PRIVATE']:x['image_binding_PRIVATE']['sha256'] for x in micro['planned_image_bindings_PRIVATE']}
 verify_images(rows,images)
 configuration=_configuration(runtime);calibration=json.loads(Path(runtime['calibration']).read_text());reward_sha=shared.file_sha256(runtime['calibration'])
 def construct(directory,resident,checkpoint=None):
  resource_gate('native_model_load')
  backend=MicroNativeBackend.__new__(MicroNativeBackend);resident[0]=backend
  backend.__init__((Path(checkpoint) if checkpoint else parent)/'adapter.safetensors',configuration,model_path=Path(runtime['model']),contract=load_contract(Path(runtime['contract'])),seed=13,before_unit=resource_gate,cache={'dir':runtime['cache_dir']})
  backend.configure_stage(grader=make_grader(calibration),expected_images=images,prompt_lengths=micro['prompt_lengths'],work=directory/'rollout_work_PRIVATE',binding={'scope_sha256':authorization.sha,'parent_receipt_sha256':anchor['receipt_sha256']})
  if checkpoint is None:restore_parent(backend,parent,anchor)
  else:
   manifest=json.loads((Path(checkpoint)/'optimizer/manifest.json').read_text());backend.stage_binding=manifest['bindings']['run'];backend.load_checkpoint(checkpoint)
  return backend
 def one(directory,operation,checkpoint=None):
  directory.mkdir(exist_ok=False);resident=[None]
  def call():
   construct(directory,resident,checkpoint);return operation(resident[0])
  result=run_with_release(call,resident,directory)
  release=json.loads((directory/'release_SAFE.json').read_text());require(release['release_verified'],'Actual allocator release required');resident[0]=None;gc.collect()
  return result
 if stage==2:
  result=one(out/'parent_load_PRIVATE',lambda backend:_stage2(backend,micro,dev,out,resource_gate))
 elif stage==3:
  def train(backend):
   expected=state_identity(backend)
   return microcode.execute_micro(backend=backend,units=micro['units_PRIVATE'],prompt_lengths=micro['prompt_lengths'],reward_policy_sha256=reward_sha,output=out/'training_PRIVATE',before=resource_gate,expected_parent=expected,checkpoint_receipt_sha256=anchor['receipt_sha256'],verify_parent_disk=lambda:recovery.require_parent_unchanged(disk_before,parent))
  actual=one(out/'training_model_PRIVATE',train);result={'status':'PASS','training':actual,'optimizer_updates':actual['optimizer_updates']}
 elif stage==4:
  trainout=Path(spec['output'])/'stage3_PRIVATE/training_PRIVATE'
  receipts={arm:json.loads((trainout/('arm_'+arm)/'receipt_PRIVATE.json').read_text()) for arm in ('A','B')}
  completed=json.loads((Path(spec['output'])/'stage3_PRIVATE/terminal_PRIVATE.json').read_text())
  for arm in ('A','B'):verify_terminal_receipt(receipts[arm],completed['training']['arms'][arm]['receipt_sha256'])
  parent_policy=receipts['A']['parent_policy_sha256'];choices={k:terminal_choice(v,parent_policy,trainout/('arm_'+k)) for k,v in receipts.items()}
  counter=CallCounter({'generations':72,'generated_tokens':18432,'LM':18720,'vision':72,'sampler':18504},out/'calls_PRIVATE.json');metrics={};outputs={}
  for label in ('Parent','A','B'):
   if label!='Parent' and choices[label]['alias_of_parent']:
    metrics[label]={**metrics['Parent'],'alias_of':'Parent','actual_updates':0};continue
   directory=out/(label+'_PRIVATE');directory.mkdir();checkpoint=None if label=='Parent' else choices[label]['path']
   def evaluate(backend):
    backend.expected_image_sha256={x['row_PRIVATE']['id']:x['image_sha256'] for x in dev['records_PRIVATE']}
    if label!='Parent':require(backend.policy_sha()==choices[label]['policy_sha256'],'Loaded terminal policy differs from authenticated receipt')
    def run():
     results=[]
     for i,record in enumerate(dev['records_PRIVATE']):
      row=record['row_PRIVATE'];start=time.monotonic();actual=generate_one(backend,row,record['image_sha256'],greedy=True,counter=counter,before=resource_gate)
      item={'id':record['id'],'truth':row['reward_model']['ground_truth'],'text':actual['response']['text'],'actual_PRIVATE':actual,'wall_seconds':time.monotonic()-start}
      shared.publish_once(directory/('row_%02d_PRIVATE.json'%i),item);results.append(item)
     require(len(results)==24,'All24 real DEV required');return greedy_metrics(results)
    return _readonly(backend,run,directory)
   metrics[label]=one(directory/'resident_PRIVATE',evaluate,checkpoint);metrics[label]['actual_updates']=0 if label=='Parent' else choices[label]['updates']
  result={'status':'PASS','metrics':metrics,'comparison':pilot_decision(metrics),'calls':counter.used,'candidate_calls':0,'interpretation':'Explored4patient DEV24 engineering pilot only; Dallas retention NOT_MEASURED.'}
 else:raise ValueError('Only2/3/4 native stages permitted')
 recovery.require_parent_unchanged(disk_before,parent);authorization.check('stage_complete')
 result.update(release_verified=True,parent_disk_unchanged=True,stage=stage,resources=resource_gate.peaks())
 shared.publish_once(out/'native_complete_PRIVATE.json',result);return result
