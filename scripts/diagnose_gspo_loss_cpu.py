"""Evaluate AST-extracted native/official GSPO loss with tiny NumPy forward-mode AD.

No MLX, Torch, model, patient data or training mutations. Source fixtures preserve
stop-gradient semantics; backend-specific nonsmooth tie conventions are separate.
"""
from pathlib import Path
import ast,hashlib,importlib.abc,json,math,subprocess,sys,types
class NoModel(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'mlx','mlx_vlm','torch','transformers'}:raise RuntimeError('No model/native framework imports in CPU fixture')
sys.meta_path.insert(0,NoModel())
import numpy as np
R=Path(__file__).resolve().parents[1];OFF=R/'../../work/vero-official/vero-rl/verl/trainer/ppo/core_algos.py';NATIVE=R/'src/mri_rl/mlx_continuous_gspo.py'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
class D:
 dim=0;__array_priority__=1000
 def __init__(self,v,j=None):
  self.v=np.asarray(v,dtype=float);self.j=np.zeros(self.v.shape+(D.dim,)) if j is None else j
 @staticmethod
 def cast(x):return x if isinstance(x,D) else D(x)
 def bin(self,x,f,da,db):
  x=D.cast(x);shape=np.broadcast_shapes(self.v.shape,x.v.shape);a=np.broadcast_to(self.v,shape);b=np.broadcast_to(x.v,shape)
  ja=np.broadcast_to(self.j,shape+(D.dim,));jb=np.broadcast_to(x.j,shape+(D.dim,))
  return D(f(a,b),da(a,b)[...,None]*ja+db(a,b)[...,None]*jb)
 def __add__(self,x):return self.bin(x,lambda a,b:a+b,lambda a,b:np.ones_like(a),lambda a,b:np.ones_like(b))
 __radd__=__add__
 def __neg__(self):return D(-self.v,-self.j)
 def __sub__(self,x):return self+-D.cast(x)
 def __rsub__(self,x):return D.cast(x)+-self
 def __mul__(self,x):return self.bin(x,lambda a,b:a*b,lambda a,b:b,lambda a,b:a)
 __rmul__=__mul__
 def __truediv__(self,x):return self.bin(x,lambda a,b:a/b,lambda a,b:1/b,lambda a,b:-a/(b*b))
 def __getitem__(self,index):return D(self.v[index],self.j[index])
 def detach(self):return D(self.v)
 def unsqueeze(self,dim):
  dim=dim if dim>=0 else self.v.ndim+1+dim;return D(np.expand_dims(self.v,dim),np.expand_dims(self.j,dim))
 def clamp(self,min=None,max=None):return API.clamp(self,min=min,max=max)
class API:
 tie=.5;clip_boundary=1.;cap_boundary=1.
 @staticmethod
 def stop_gradient(x):return D.cast(x).detach()
 @staticmethod
 def exp(x):
  x=D.cast(x);v=np.exp(x.v);return D(v,v[...,None]*x.j)
 @staticmethod
 def sum(x,dim=None,axis=None):
  x=D.cast(x);dim=axis if axis is not None else dim;axes=tuple(range(x.v.ndim)) if dim is None else dim%x.v.ndim
  return D(np.sum(x.v,axis=dim),np.sum(x.j,axis=axes))
 @staticmethod
 def mean(x,dim=None):
  x=D.cast(x);n=x.v.size if dim is None else x.v.shape[dim];return API.sum(x,dim)/n
 @staticmethod
 def minimum(a,b):
  return D.cast(a).bin(b,np.minimum,lambda a,b:np.where(a<b,1.,np.where(a>b,0.,API.cap_boundary)),lambda a,b:np.where(a<b,0.,np.where(a>b,1.,1-API.cap_boundary)))
 @staticmethod
 def maximum(a,b):
  return D.cast(a).bin(b,np.maximum,lambda a,b:np.where(a>b,1.,np.where(a<b,0.,API.tie)),lambda a,b:np.where(a>b,0.,np.where(a<b,1.,1-API.tie)))
 @staticmethod
 def clip(x,lo,hi):return API.clamp(x,min=lo,max=hi)
 @staticmethod
 def clamp(x,min=None,max=None):
  x=D.cast(x);lo=-np.inf if min is None else min;hi=np.inf if max is None else max
  slope=np.where((x.v>lo)&(x.v<hi),1.,np.where((x.v==lo)|(x.v==hi),API.clip_boundary,0.))
  return D(np.clip(x.v,lo,hi),slope[...,None]*x.j)

def extracted():
 native_tree=ast.parse(NATIVE.read_text());cls=next(v for v in native_tree.body if isinstance(v,ast.ClassDef) and v.name=='NativeGSPOBackend')
 update=next(v for v in cls.body if isinstance(v,ast.FunctionDef) and v.name=='update');loss=next(v for v in update.body if isinstance(v,ast.FunctionDef) and v.name=='loss_fn')
 ns={'mx':API};exec(compile(ast.Module(body=[loss],type_ignores=[]),'<actual_native_loss_AST>','exec'),ns)
 tree=ast.parse(OFF.read_text());official=next(v for v in tree.body if isinstance(v,ast.FunctionDef) and v.name=='compute_policy_loss_gspo')
 names={'negative_approx_kl','seq_lengths','negative_approx_kl_seq','log_seq_importance_ratio','seq_importance_ratio','pg_losses1','pg_losses2','pg_losses','pg_loss'}
 body=[v for v in official.body if isinstance(v,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in v.targets)]
 # Keep the exact pinned aggregation branch; remove only unrelated modes/metrics/types.
 agg=next(v for v in tree.body if isinstance(v,ast.FunctionDef) and v.name=='agg_loss');branch=agg.body[1]
 while branch.test.comparators[0].value!='seq-mean-token-mean':branch=branch.orelse[0]
 aggfn=ast.FunctionDef(name='agg_loss',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg=x) for x in ('loss_mat','loss_mask','loss_agg_mode')],kwonlyargs=[],kw_defaults=[],defaults=[]),body=branch.body+[ast.Return(value=ast.Name(id='loss',ctx=ast.Load()))],decorator_list=[])
 fn=ast.FunctionDef(name='official_loss',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg=x) for x in ('old_log_prob','log_prob','advantages','response_mask','clip_ratio_low','clip_ratio_high')],kwonlyargs=[],kw_defaults=[],defaults=[]),body=body+[ast.Return(value=ast.Name(id='pg_loss',ctx=ast.Load()))],decorator_list=[])
 module=ast.fix_missing_locations(ast.Module(body=[aggfn,fn],type_ignores=[]));ons={'torch':API};exec(compile(module,'<pinned_official_loss_AST>','exec'),ons)
 return ns,ons['official_loss'],{'native_loss_AST_sha256':hashlib.sha256(ast.dump(loss,include_attributes=False).encode()).hexdigest(),'official_loss_AST_subset_sha256':hashlib.sha256(ast.dump(module,include_attributes=False).encode()).hexdigest()}

def fixture(ns,official,lengths,mus,advantages):
 B=len(lengths);L=max(lengths);N=B*L;D.dim=N;mask=np.zeros((B,L));old=np.full((B,L),-30.);fresh=old.copy()
 for i,n in enumerate(lengths):mask[i,:n]=1;fresh[i,:n]+=mus[i]+np.linspace(-.03,.03,n)-(np.linspace(-.03,.03,n).mean())
 var=D(fresh,np.eye(N).reshape(B,L,N));reference=official(D(old),var,np.array(advantages)[:,None],mask,.0003,.0004)
 native=D(0.)
 for i,n in enumerate(lengths):
  ns['self']=types.SimpleNamespace(_logps=lambda tokens,i=i,n=n:var[i,:n]);native+=ns['loss_fn'](None,None,D(old[i,:n]),advantages[i])/B
 analytic=np.zeros((B,L));values=[];boundary=False
 for i,(n,mu,A) in enumerate(zip(lengths,mus,advantages)):
  ratio=math.exp(min(float(np.mean(fresh[i,:n]-old[i,:n])),10));values.append(max(-A*ratio,-A*np.clip(ratio,1-.0003,1+.0004)))
  at_boundary=abs(ratio-(1-.0003))<1e-13 or abs(ratio-(1+.0004))<1e-13 or abs(mu-10)<1e-13;boundary|=at_boundary
  clipped=(A>0 and ratio>1+.0004) or (A<0 and ratio<1-.0003) or mu>10
  if not clipped:analytic[i,:n]=-A*ratio/(B*n)
 value_err=abs(float(native.v-reference.v));gradient_err=float(np.max(np.abs(native.j-reference.j)))
 assert value_err<1e-12 and gradient_err<1e-10,(value_err,gradient_err)
 assert abs(float(native.v)-np.mean(values))<1e-12
 if not boundary:assert np.max(np.abs(native.j.reshape(B,L)-analytic))<1e-10
 assert np.max(np.abs(native.j.reshape(B,L)[mask==0]),initial=0)==0
 return {'lengths':lengths,'mean_log_ratios':mus,'advantages':advantages,'native_vs_official_value_abs_error':value_err,'native_vs_official_gradient_max_abs_error':gradient_err,
         'analytic_gradient_max_abs_error':None if boundary else float(np.max(np.abs(native.j.reshape(B,L)-analytic))),'nonsmooth_boundary':boundary,'fixture_loss':float(native.v)}

def causal_alignment():
 # Run the actual frozen _logps AST against a tiny identity-logit model.
 tree=ast.parse(NATIVE.read_text());cls=next(v for v in tree.body if isinstance(v,ast.ClassDef) and v.name=='NativeGSPOBackend')
 method=next(v for v in cls.body if isinstance(v,ast.FunctionDef) and v.name=='_logps')
 ns={'require':lambda ok,why:None if ok else (_ for _ in ()).throw(AssertionError(why))}
 exec(compile(ast.Module(body=[method],type_ignores=[]),'<actual_logps_AST>','exec'),ns)
 P,T,V=5,4,32;ids=np.arange(P+T)[None,:];logits=np.full((1,P+T-1,V),-100.)
 for pos,label in enumerate(ids[0,1:]):logits[0,pos,label]=0.
 calls=[]
 def model(actual,pixels,mask,**kwargs):
  assert np.array_equal(actual,ids[:,:-1]) and np.all(mask==1) and kwargs['image_grid_thw']==[[1,2,2]]
  calls.append(True);return types.SimpleNamespace(logits=logits)
 def log_softmax(x,axis):return x-np.log(np.sum(np.exp(x-np.max(x,axis=axis,keepdims=True)),axis=axis,keepdims=True))-np.max(x,axis=axis,keepdims=True)
 self=types.SimpleNamespace(mx=np,model=model,nn=types.SimpleNamespace(log_softmax=log_softmax),active={'inputs':{'input_ids':ids[:,:P],'pixel_values':np.zeros((1,1)),'image_grid_thw':[[1,2,2]]},'prompt_length':P})
 response=ns['_logps'](self,ids[0,P:].tolist());assert len(response)==T and np.max(np.abs(response))<1e-6 and calls
 labels=ids[:,1:];wrong=np.take_along_axis(logits,ids[:,:-1,None],axis=-1)[0,P-1:,0];assert np.all(wrong==-100)
 return {'actual_logps_AST_sha256':hashlib.sha256(ast.dump(method,include_attributes=False).encode()).hexdigest(),'prompt_tokens':P,'response_tokens':T,'predicted_response_labels':labels[0,P-1:].tolist(),'correct_response_start_logit':P-1,'off_by_one_negative_control_detected':True,'visual_kwargs_preserved':True}

def prompt_source_check():
 contract_path=R/'local_data/ds007045_bounded/continuous_gspo_frozen_flair_candidate_v1/contract.json'
 contract=json.loads(contract_path.read_text());train_path=R/'local_data/c_two_task_pool_v1/train.verl.jsonl';dev_path=R/'local_data/c_two_task_pool_v1/evaluation_v1/internal_dev_native_system_matched.jsonl'
 train=[json.loads(x) for x in train_path.read_text().splitlines() if x.strip()];dev=[json.loads(x) for x in dev_path.read_text().splitlines() if x.strip()]
 system=contract['system_instruction'];matched=all(v['prompt'][0]=={'role':'system','content':system} for v in dev)
 tq={task:{v['extra_info']['question'] for v in train if v['ability']==task} for task in ('mri_sequence','mri_visit_correspondence')}
 dq={task:{v['prompt'][1]['content'] for v in dev if v['category']==task} for task in tq}
 assert matched and tq==dq and contract['enable_thinking'] is False and contract['assistant_prefix']=='<|im_start|>assistant\n'
 processor=R/'../../work/mlx-venv/lib/python3.13/site-packages/mlx_vlm/prompt_utils.py';tree=ast.parse(processor.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='apply_chat_template')
 defaults=dict(zip([a.arg for a in fn.args.args][-len(fn.args.defaults):],fn.args.defaults));assert isinstance(defaults['add_generation_prompt'],ast.Constant) and defaults['add_generation_prompt'].value is True
 return {'frozen_contract_sha256':sha(contract_path),'train_manifest_sha256':sha(train_path),'dev_projection_sha256':sha(dev_path),'native_chat_template_helper_sha256':sha(processor),'train_rows':len(train),'dev_projection_rows':len(dev),'all_dev_systems_match_native_training_contract':matched,'task_question_sets_identical':tq==dq,'enable_thinking':False,'assistant_prefix_same':True,'eval_implicit_add_generation_prompt_default':True,'training_temperature':1.,'evaluation_temperature':0.,'decoding_difference':'Intentional stochastic rollout versus greedy evaluation; not a prompt mismatch.','actual_generate_prepared_ID_equality':'Only generation prompt length is checked by native rollout; this CPU diagnostic does not newly intercept model-generation internals.'}

def main():
 assert subprocess.check_output(['git','-C',str(OFF.parents[4]),'rev-parse','HEAD'],text=True).strip()=='025ca3c800fc95224c86323c1f7fd33dceed3090'
 subprocess.run(['git','-C',str(OFF.parents[4]),'diff','--quiet','HEAD','--','vero-rl/verl/trainer/ppo/core_algos.py'],check=True)
 ns,official,hashes=extracted();cases=[];lengths=[1,2,5,9]
 for A in (1.,-1.,0.):
  for mu in (0.,math.log(1-.0003)-.002,math.log(1-.0003)+.0001,math.log(1+.0004)-.0001,math.log(1+.0004)+.002,-12.,11.,math.log(1-.0003),math.log(1+.0004),10.):
   cases.append(fixture(ns,official,lengths,[mu]*4,[A]*4))
 cases.append(fixture(ns,official,lengths,[0.,.001,-.001,.0001],[-1.2,.8,1.1,-.7]))
 result={'schema':'independent-GSPO-AST-loss-gradient-diagnostic-v1','verdict':'NOT_FOUND_FOR_INTERIOR_LOSS_GRADIENT_OR_CAUSAL_TOKEN_OFFSET',
  'reviewer_kind':'isolated_codex','reviewer_id':'/root/full_arm_execution_contract','research_tag':'PRIMARY_VERO_TRANSFER','evidence_tier':'REPRODUCED_CPU_AST_FORWARD_AUTODIFF_AND_ANALYTIC',
  'cases':cases,'case_count':len(cases),'max_native_official_value_error':max(c['native_vs_official_value_abs_error'] for c in cases),
  'max_native_official_gradient_error':max(c['native_vs_official_gradient_max_abs_error'] for c in cases),'causal_token_fixture':causal_alignment(),
  'prompt_source_check':prompt_source_check(),
  'source_sha256':{'fixture':sha(__file__),'native_backend':sha(NATIVE),'pinned_official_core_algos':sha(OFF),**hashes},'official_commit':'025ca3c800fc95224c86323c1f7fd33dceed3090',
  'findings':{'sequence_ratio_stop_gradient_clipped_surrogate':'NOT_FOUND: exact native AST and pinned official AST agree under stated CPU AD conventions.',
  'unequal_length_masked_sequence_mean':'NOT_FOUND: native mean per response then mean K agrees with pinned seq-mean-token-mean; padding gradients0.',
  'causal_response_offset':'NOT_FOUND: response labels start at shifted-logit P-1; generated EOS is included when present.',
  'actual_MLX_boundary_subgradient':'UNRESOLVED: this CPU fixture does not execute MLX; exact clipping/cap ties can select backend-specific valid subgradients.',
  'old_policy_collection':'DOCUMENTED_DESIGN: old logps are detached current pre-update logps; all K gradients accumulate before one Adam update, with model.eval and identical captured input.',
  'on_policy_clipping_activity':'DOCUMENTED_DESIGN: ratios at actual one-step group updates are approximately1; clipping is normally inactive. Official dp_actor also has an on-policy old=current.detach branch.',
  'negative_U7_cause':'UNRESOLVED: formula correctness does not identify why internal-dev accuracy declined; reward/exposure/grounding/uncertainty remain separate.'},
  'limitations':['NumPy forward-mode AD validates exact source expressions and independent analytic derivatives; it does not execute native MLX visual backward or test model logits.',
  'At exact nonsmooth ties, the fixture chooses explicit maximum tie.5 and clip/cap boundary1; native backend boundary conventions remain unverified. Actual frozen on-policy updates are near ratio1, away from tiny clip boundaries.',
  'Finite local LoRA one-update groups are not a full distributed veRL training parity claim. No source changes, model launch, queue intervention, or experimental repair follows from this fixture.']}
 out=R/'results/GSPO_LOSS_AST_CPU_INDEPENDENT_DIAGNOSTIC.json';out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps({'verdict':result['verdict'],'cases':len(cases),'value_error':result['max_native_official_value_error'],'gradient_error':result['max_native_official_gradient_error'],'receipt_sha256':sha(out)}))
if __name__=='__main__':main()
