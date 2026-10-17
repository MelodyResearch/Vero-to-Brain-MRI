
"""Finite optional diagnostic on an ALREADY resident native backend. No loader/CLI.
No numerical-closeness acceptance threshold and no optimizer call.
"""
from __future__ import annotations
import ast,contextlib,gc,hashlib,importlib.metadata,inspect,json,math,os,struct,textwrap
from pathlib import Path

CONTRACT={'version':'early-native-conditioning-probe-v1','selection':'FIRST real_image in original immutable732 plan order, same record at B28/U7',
 'new_trajectories':1,'max_new_tokens':32,'teacherforced_forwards':2,'backward_calls':2,'optimizer_updates':0,'additional_model_loads':0,
 'sampling':{'temperature':1.,'top_p':1.,'top_k':0,'min_p':0.,'repetition_penalty':None,'presence_penalty':None,'frequency_penalty':None,'logit_bias':None},
 'gradient_reference':'Exact existing NativeGSPOBackend.update.loss_fn AST; advantage+1; each conditioning has OWN detached teacherforced old logps.',
 'zero_control':'Same-shape zeroed pixel_values only; keep prompt/input_ids/mask/image_grid fixed. OOD diagnostic, not a no-image prompt or real patient.',
 'closeness_threshold':None,'reference_tolerances_only':{'existing_old_current_gate':1e-5,'existing_candidate_prefix_oracle':.002},
 'runtime_versions':{'mlx':'0.32.4','mlx-vlm':'0.7.8'},
 'native_backend_sha256':'aceda9c4e212667d512e747447f3ef5533b91565f6cebc5a928b8a805c0b96e3'}
NATIVE_SOURCES={'generate/dispatch.py':'f7dcec326a4d2aff1286b1abd4419f8ec533189d1d26b0ac69002513ba45dab8',
 'generate/ar.py':'c38f86a5d94924daf819f4bfd206a7c874526bd15e8dd3447921621ee7850050',
 'sample_utils.py':'56b3d5043fe9d9c3e95e184412f694d62669d5865dca6c6d11f5897beac12f7e',
 'utils.py':'d38eabab88d5d6b835d4ab703cdac2e30924fc04463be8134d12ae5406e50b0c',
 'tokenizer_utils.py':'7fb64819e0577e7e960c8b8eb247644b4a70046c117fa5eb4a00d30c0f9c645f',
 'prompt_utils.py':'8d3469f3d34d58dab84d207d820e0ec3c9c50f2573c47e9a893cc5162469756a',
 'models/qwen3_vl/qwen3_vl.py':'12f0393421918eb1cbaa6daa447d93a405762e997f688676688e5c71d987d3ee',
 'models/qwen3_vl/language.py':'2ebeacac837db9417e683780b94d24cdee563fa340a9638b0965a6c0bfcde056',
 'models/qwen3_vl/processing_qwen3_vl.py':'9c6bc521ea6b45c3346967502f2d0b69697371327d7b340c23650e76de91f5d2'}

def require(ok,message):
 if not ok:raise ValueError(message)
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def durable(path,value):
 path=Path(path)
 with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 fd=os.open(path.parent,os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
def validate_logprob_shape(shape,token):
 require(len(shape)==2 and shape[0]==1 and type(token) is int and 0<=token<shape[1],'Missing/wrong-shape selected raw logprob')
def capture_stream(iterator,*,selected_logp,before_unit,eos_ids,prompt_tokens,max_tokens):
 values={};final=None
 try:
  while True:
   before_unit('conditioning_stream_next')
   try:step=next(iterator)
   except StopIteration:break
   n=getattr(step,'generation_tokens',None);token=getattr(step,'token',None)
   require(type(n) is int and 1<=n<=max_tokens and type(token) is int,'Shifted/missing stream token index')
   require(step.prompt_tokens==prompt_tokens,'Generation/scorer prompt token count differs')
   value=selected_logp(step.logprobs,token);require(math.isfinite(value) and value<=0,'Nonfinite/positive raw logprob')
   if n in values:require(values[n]==(token,value),'Duplicate final stream token/logprob differs')
   else:require(n==len(values)+1,'Shifted stream token position');values[n]=(token,value)
   finish=getattr(step,'finish_reason',None)
   if finish is not None:
    require(finish in {'stop','length'},'Unknown finite stream termination')
    ids=[values[i][0] for i in range(1,len(values)+1)]
    require(getattr(step,'token_ids',None)==ids,'Final returned token IDs differ from captured per-step IDs')
    require((finish=='stop' and ids[-1] in eos_ids) or (finish=='length' and not any(t in eos_ids for t in ids)),'EOS/finish evidence differs')
    require(not any(t in eos_ids for t in ids[:-1]),'Token after stopping EOS')
    final={'token_ids':ids,'raw_selected_logps':[values[i][1] for i in range(1,len(values)+1)],'finish_reason':finish,'prompt_tokens':prompt_tokens,
      'eos_observed':finish=='stop','EOS_coverage':'OBSERVED_AND_SCORED' if finish=='stop' else 'NOT_OBSERVED_LENGTH_LIMIT'}
    break
 finally:
  close=getattr(iterator,'close',None)
  if callable(close):close()
 require(final is not None,'No complete final stream evidence')
 return final
def logprob_comparison(raw,teacher):
 require(len(raw)==len(teacher)>0 and all(math.isfinite(x) for x in [*raw,*teacher]),'Nonfinite/missing/shifted logprob vector')
 deltas=[b-a for a,b in zip(raw,teacher)]
 return {'tokens':len(raw),'per_token':[{'index':i,'stream_raw_logp':a,'teacherforced_logp':b,'teacherforced_minus_stream':b-a} for i,(a,b) in enumerate(zip(raw,teacher))],
  'max_abs_delta':max(abs(x) for x in deltas),'mean_abs_delta':math.fsum(abs(x) for x in deltas)/len(deltas),'mean_signed_delta':math.fsum(deltas)/len(deltas),
  'numerical_closeness_verdict':'NOT_VERIFIED','acceptance_threshold':None}
def validate_condition_fingerprints(real,zero):
 require(set(real)==set(zero) and 'pixel_values' in real,'Condition tensor inventory differs')
 for name in real:
  if name!='pixel_values':require(real[name]==zero[name],'Zero condition changed prompt/mask/grid/metadata')
  else:
   require(real[name]['shape']==zero[name]['shape'] and real[name]['dtype']==zero[name]['dtype'],'Zero pixel shape/dtype differs')
   require(zero[name].get('all_zero') is True and real[name].get('all_zero') is False,'Real/zero conditioning not demonstrated')
def gradient_summary(chunks):
 rr=zz=dot=dd=0.;maximum=0.;count=tensors=0
 for name,shape,left,right in chunks:
  left=list(map(float,left));right=list(map(float,right));n=math.prod(shape)
  require(len(left)==len(right)==n and all(math.isfinite(x) for x in [*left,*right]),'Gradient shape/nonfinite mismatch')
  rr+=math.fsum(x*x for x in left);zz+=math.fsum(x*x for x in right);dot+=math.fsum(a*b for a,b in zip(left,right))
  dd+=math.fsum((a-b)**2 for a,b in zip(left,right));maximum=max(maximum,max((abs(a-b) for a,b in zip(left,right)),default=0.));count+=n;tensors+=1
 require(tensors>0,'No actual gradient vectors')
 return {'tensors':tensors,'elements':count,'real_l2':math.sqrt(rr),'zero_l2':math.sqrt(zz),'difference_l2':math.sqrt(dd),
  'max_abs_component_difference':maximum,'cosine_similarity':dot/math.sqrt(rr*zz) if rr>0 and zz>0 else None,
  'conditioning_interpretation':'NOT_VERIFIED','sensitivity_is_not_brain_grounding':True}
@contextlib.contextmanager
def preserve_active(backend):
 old=backend.active
 try:yield
 finally:backend.active=old
def check_weights(before,after):
 require(before==after,'Read-only diagnostic changed adapter parameters')
 return True

def _runtime_binding(backend):
 require(sha(inspect.getsourcefile(type(backend)))==CONTRACT['native_backend_sha256'],'Native backend source differs')
 stream=backend.generate.__globals__.get('stream_generate');require(callable(stream),'Raw stream interface not exposed')
 root=Path(inspect.getsourcefile(stream)).parent.parent
 for name,digest in NATIVE_SOURCES.items():require(sha(root/name)==digest,'Installed native source changed: '+name)
 require(sha(inspect.getsourcefile(backend.nn.value_and_grad))=='a5ab60f6731a7a8d3821b7e44ad96961ef848ae0315090f47a895627ede102a8','Native value_and_grad wrapper changed')
 for name,version in CONTRACT['runtime_versions'].items():require(importlib.metadata.version(name)==version,'Native runtime version changed')
 ar=stream.__globals__['generate_step'].__globals__
 for name in ('make_logits_processors','make_sampler'):
  effective=ar['_generate_module_override'](name,ar[name])
  require(Path(inspect.getsourcefile(effective))==root/'sample_utils.py','Runtime sampler/processor override differs')
 require(ar['_generate_module_override']('make_logits_processors',ar['make_logits_processors'])(None,None,20,None,20,None,20)==[],'Neutral sampling still applies logit processors')
 return stream,{'native_backend':CONTRACT['native_backend_sha256'],'installed':NATIVE_SOURCES,'versions':CONTRACT['runtime_versions'],
  'raw_logprob_source':'Pinned generate/ar.py _step: logits - logsumexp after empty processors, before sampler. No raw/scaled-score substitution.'}
def _fingerprints(backend,inputs):
 import numpy as np
 mx=backend.mx;result={}
 for name,value in inputs.items():
  if hasattr(value,'shape'):
   mx.eval(value);array=np.asarray(value.view(mx.uint16) if value.dtype==mx.bfloat16 else value)
   result[name]={'shape':list(value.shape),'dtype':str(value.dtype),'sha256':hashlib.sha256(array.tobytes()).hexdigest()}
   if name=='pixel_values':result[name]['all_zero']=bool(np.all(array==0))
  else:result[name]={'metadata_sha256':canonical(value)}
 return result
def _weights(backend):
 values=dict(backend.tree_flatten(backend.model.trainable_parameters()))
 require(len(values)==504 and all('lora' in name.lower() and 'vision' not in name.lower() for name in values),'Expected504languageLoRA parameters')
 return _fingerprints(backend,values)
def _existing_loss(backend):
 tree=ast.parse(textwrap.dedent(inspect.getsource(type(backend).update)))
 node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='loss_fn')
 namespace={'mx':backend.mx,'self':backend}
 exec(compile(ast.Module(body=[node],type_ignores=[]),inspect.getsourcefile(type(backend)),'exec'),namespace)
 return namespace['loss_fn'],hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
def _gradient_chunks(real,zero):
 import numpy as np
 def header(path):
  with Path(path).open('rb') as f:
   n=struct.unpack('<Q',f.read(8))[0];return json.loads(f.read(n)),8+n
 rh,ro=header(real);zh,zo=header(zero);rh.pop('__metadata__',None);zh.pop('__metadata__',None)
 require(set(rh)==set(zh) and len(rh)==504,'Gradient vector inventory differs')
 with Path(real).open('rb') as rf,Path(zero).open('rb') as zf:
  for name in sorted(rh):
   a,b=rh[name],zh[name];require(a['dtype']==b['dtype']=='F32' and a['shape']==b['shape'],'Gradient dtype/shape differs')
   rf.seek(ro+a['data_offsets'][0]);zf.seek(zo+b['data_offsets'][0])
   left=np.frombuffer(rf.read(a['data_offsets'][1]-a['data_offsets'][0]),dtype='<f4')
   right=np.frombuffer(zf.read(b['data_offsets'][1]-b['data_offsets'][0]),dtype='<f4')
   yield name,a['shape'],left,right
def _fsync(path):
 with Path(path).open('rb') as f:os.fsync(f.fileno())

def run_conditioning_probe(backend,record,output_dir,before_unit):
 """No model load. Caller binds helper source, exact first732 real record/scope."""
 out=Path(output_dir);out.mkdir(parents=True,exist_ok=False)
 mx=backend.mx;criteria=None;prior_eos=None;private={};before_weights=None
 try:
  require(record.get('condition')=='real_image' and record.get('image') and sha(record['image'])==record['image_sha256'],'Frozen real image required')
  require(callable(before_unit),'Existing resource gate callback required')
  stream,binding=_runtime_binding(backend)
  before_unit('conditioning_weights_before');before_weights=_weights(backend);backend.model.eval()
  with preserve_active(backend):
   before_unit('conditioning_prepare_same_inputs')
   prompt=backend.template(backend.processor,backend.model.config,record['prompt'],num_images=1,enable_thinking=False,add_generation_prompt=True)
   require(prompt.count('<|image_pad|>')==1 and prompt.endswith('<|im_start|>assistant\n'),'Native prompt/prefix mismatch')
   inputs=backend.prepare_inputs(backend.processor,images=[record['image']],prompts=prompt,image_token_index=getattr(backend.model.config,'image_token_index',None),add_special_tokens=backend.special_tokens(backend.model.config.model_type,backend.processor))
   P=int(inputs['input_ids'].shape[-1]);require(inputs['input_ids'].shape[0]==1 and 0<P<=2048 and inputs.get('pixel_values') is not None and bool(mx.all(inputs['attention_mask']==1).item()),'Prepared input/padding mismatch')
   require('image_grid_thw' in inputs,'No image grid conditioning evidence')
   before_unit('conditioning_real_tensor_identity');real_fp=_fingerprints(backend,inputs)
   visual=inputs['input_ids'][0].tolist().count(151655);grid=inputs['image_grid_thw'].tolist()
   require(visual>0 and visual==sum(t*h*w//4 for t,h,w in grid),'Visual/grid token identity mismatch')
   tokenizer=backend.processor.tokenizer;criteria=tokenizer.stopping_criteria;prior_eos=list(criteria.eos_token_ids)
   criteria.reset(backend.model.config.eos_token_id);eos=set(criteria.eos_token_ids)
   require(eos and all(type(t) is int for t in eos),'Native stopping IDs unavailable')
   extras={k:v for k,v in inputs.items() if k not in ('input_ids','pixel_values','attention_mask')}
   iterator=stream(backend.model,backend.processor,prompt,image=None,input_ids=inputs['input_ids'],pixel_values=inputs['pixel_values'],mask=inputs['attention_mask'],
     max_tokens=32,temperature=1.,top_p=1.,top_k=0,min_p=0.,repetition_penalty=None,presence_penalty=None,frequency_penalty=None,logit_bias=None,logits_processors=None,sampler=None,
     prompt_cache=None,vision_cache=None,prompt_cache_state=None,apc_manager=None,verbose=False,**extras)
   def selected(values,token):
    validate_logprob_shape(values.shape,token);value=values[0,token];mx.eval(value);return float(value.item())
   trajectory=capture_stream(iterator,selected_logp=selected,before_unit=before_unit,eos_ids=eos,prompt_tokens=P,max_tokens=32)
   before_unit('conditioning_generation_input_identity');require(_fingerprints(backend,inputs)==real_fp,'Stream mutated caller-prepared tensors')
   require(sha(record['image'])==record['image_sha256'],'Real source image changed during trajectory')
   tokens=trajectory['token_ids'];durable(out/'trajectory_PRIVATE.json',{'record_id':record['record_id'],'image_sha256':record['image_sha256'],'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'input_tensor_fingerprints':real_fp,'trajectory':trajectory,'effective_EOS':sorted(eos)})
   active={'record_id':record['record_id'],'inputs':inputs,'prompt_length':P};backend.active=active
   before_unit('conditioning_teacherforced_real');real=mx.stop_gradient(backend._logps(tokens));mx.eval(real);real_values=real.tolist()
   parity=logprob_comparison(trajectory['raw_selected_logps'],real_values)
   durable(out/'teacherforced_real_PRIVATE.json',{'logps':real_values,'parity':parity})
   before_unit('conditioning_prepare_zero_pixels');zero_inputs={**inputs,'pixel_values':mx.zeros_like(inputs['pixel_values'])};mx.eval(zero_inputs['pixel_values'])
   before_unit('conditioning_zero_tensor_identity');zero_fp=_fingerprints(backend,zero_inputs);validate_condition_fingerprints(real_fp,zero_fp)
   backend.active={**active,'inputs':zero_inputs}
   before_unit('conditioning_teacherforced_zero');zero=mx.stop_gradient(backend._logps(tokens));mx.eval(zero);zero_values=zero.tolist()
   require(len(zero_values)==len(tokens) and all(math.isfinite(x) for x in zero_values),'Zero-conditioning score invalid')
   durable(out/'teacherforced_zero_PRIVATE.json',{'logps':zero_values,'input_tensor_fingerprints':zero_fp})
   loss_fn,loss_ast=_existing_loss(backend);loss_grad=backend.nn.value_and_grad(backend.model,loss_fn);losses={}
   for label,condition,previous in (('real',inputs,real),('zero',zero_inputs,zero)):
    backend.active={**active,'inputs':condition};before_unit('conditioning_backward_'+label)
    loss,gradient=loss_grad(backend.model,tokens,previous,1.);mx.eval(loss,gradient)
    value=float(loss.item());require(math.isfinite(value),'Nonfinite native gradient loss')
    arrays={name:g.astype(mx.float32) for name,g in backend.tree_flatten(gradient)}
    require(set(arrays)==set(before_weights) and all(bool(mx.all(mx.isfinite(g)).item()) for g in arrays.values()),'Nonfinite/missing native LoRA gradients')
    path=out/('gradient_'+label+'_PRIVATE.safetensors');backend.save_arrays(path,arrays);_fsync(path);losses[label]=value
    del loss,gradient,arrays;gc.collect();mx.clear_cache()
   gradients=gradient_summary(_gradient_chunks(out/'gradient_real_PRIVATE.safetensors',out/'gradient_zero_PRIVATE.safetensors'))
   before_unit('conditioning_weights_after');after_weights=_weights(backend);check_weights(before_weights,after_weights)
   private={'contract':CONTRACT,'record_id':record['record_id'],'prompt_ids':inputs['input_ids'][0].tolist(),'real_inputs':real_fp,'zero_inputs':zero_fp,'trajectory':trajectory,
     'teacherforced_real':real_values,'teacherforced_zero':zero_values,'weights_before':before_weights,'weights_after':after_weights,
     'gradient_files_sha256':{label:sha(out/('gradient_'+label+'_PRIVATE.safetensors')) for label in ('real','zero')}}
   durable(out/'receipt_PRIVATE.json',private)
   safe={'schema':'early-native-conditioning-probe-v1','status':'MEASUREMENTS_COMPLETED_PENDING_INTERPRETATION','verdict':'NOT_VERIFIED','contract':CONTRACT,
    'source_binding':binding,'helper_source_sha256':sha(__file__),'trajectory_tokens':len(tokens),'prompt_tokens':P,'visual_tokens':visual,'EOS_coverage':trajectory['EOS_coverage'],
    'generation_scorer_same_input_tensor_fingerprints':True,'finalEOS_scored_if_observed':trajectory['eos_observed'],
    'rollout_vs_teacherforced_real':parity,'gradient_condition_comparison':gradients,'native_loss_terms':losses,'loss_fn_AST_sha256':loss_ast,
    'own_condition_detached_old_reference':True,'advantage':1.,'native_fresh_old_equality_inside_backward':'NOT_INDEPENDENTLY_CAPTURED; own TF reference and identical condition, no intervening updates.',
    'adapter_tensors_checked_before_after':504,'adapter_before_sha256':canonical(before_weights),'adapter_after_sha256':canonical(after_weights),'weight_updates':0,
    'private_receipt_sha256':sha(out/'receipt_PRIVATE.json'),'gradient_files_sha256':private['gradient_files_sha256'],
    'limitations':['Finite generation/teacherforcing numeric differences are measured without an invented acceptance tolerance. Interpretive comparison remains unverified.',
      'Zero pixels are OOD with image/grid present; gradient sensitivity is not brain-grounding, MRI understanding, reward learning or generalization.',
      'Exact existing loss AST/autograd path reused; independent finite-difference/native-backward formula verification NOT_VERIFIED.']}
   durable(out/'receipt_SAFE.json',safe)
  return {'status':safe['status'],'verdict':safe['verdict'],'safe_receipt_path':str(out/'receipt_SAFE.json'),'safe_receipt_sha256':sha(out/'receipt_SAFE.json'),
   'private_receipt_path':str(out/'receipt_PRIVATE.json'),'private_receipt_sha256':sha(out/'receipt_PRIVATE.json'),'weight_updates':0}
 except BaseException as error:
  if not (out/'failure_SAFE.json').exists():durable(out/'failure_SAFE.json',{'status':'FAILED_OR_INCOMPLETE_PRESERVE_EVIDENCE','verdict':'FAIL','error_type':type(error).__name__,'weight_updates_called':False,'completion_claimed':False})
  raise
 finally:
  if criteria is not None and prior_eos is not None:criteria.eos_token_ids=prior_eos
  gc.collect();mx.clear_cache()
