"""Focused CPU proof of parent Adam52/RNG continuity and <=4 child updates.
No model/native imports; no repeated proof of the parent's114 groups.
"""
import argparse,ast,hashlib,importlib.abc,json,math,os,random,struct,sys
from pathlib import Path
import numpy as np
W=Path(__file__).resolve().parents[1]
MATH_SHA='0c3408940bc36a646dfc97eb37c27c5964c0eba331f8435b636dce3e714df048'
PARENT_PROOF_SHA='a5d5cd1faf58492b173177668175e84cc2ffc94b13e15dff4705d6bf5d7f59cb'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
load=lambda p:json.loads(Path(p).read_text())
canonical=lambda v:json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)
objsha=lambda v:hashlib.sha256(canonical(v).encode()).hexdigest()
def require(ok,message):
 if not ok:raise ValueError(message)
class NoNative(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'mlx','mlx_vlm','torch','transformers'}:raise RuntimeError('Native import forbidden')
def approved_math(path):
 require(sha(path)==MATH_SHA,'Reviewed CPU math changed');names={'TensorFile','decode','verify_adam_tensor','moments','optimizer_layout','arrays_equal'};nodes=[n for n in ast.parse(Path(path).read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names];require({n.name for n in nodes}==names,'Missing approved math');ns={'np':np,'struct':struct,'json':json};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns);return ns

def same_tree(a,b):
 if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a,b)
 if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(same_tree(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return type(a) is type(b) and len(a)==len(b) and all(same_tree(x,y) for x,y in zip(a,b))
 return type(a) is type(b) and a==b

def terminal_selection(terminal,receipt,final_path):
 require(terminal['status'] in {'SCHEDULE_COMPLETE','UPDATE_BUDGET_REACHED','TOKEN_BUDGET_EXHAUSTED'},'Actual finite terminal required')
 n=receipt['completed_groups'];k=receipt['optimizer_updates'];require(type(n) is type(k) is int and 1<=n<=16 and 0<=k<=4 and k<=n,'Child finite counters invalid')
 require(terminal['completed_groups']==n and terminal['optimizer_updates']==k and Path(terminal['boundary']).resolve()==final_path.resolve(),'Select exact terminal checkpoint, never best-dev')
 require(receipt['absolute_completed_groups']==terminal['absolute_completed_groups']==114+n and receipt['absolute_optimizer_updates']==terminal['absolute_optimizer_updates']==52+k,'Absolute parent offsets invalid')
 require(n==16 or k==4,'Finite stage ended before16groups or4updates; zero-debt full-worstcase budget cannot be prematurely exhausted')
 require(terminal['status']!='SCHEDULE_COMPLETE' or n==16,'Incomplete schedule called complete');require(terminal['status']!='UPDATE_BUDGET_REACHED' or k==4,'Update budget not reached');return n,k

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('primary-root','scope','review','exit-witness','native-log','native-complete','output'):p.add_argument('--'+name,type=Path,required=True)
 for name in ('scope','review','exit-witness'):p.add_argument('--expected-'+name+'-sha256',required=True)
 p.add_argument('--reviewer-id',required=True);a=p.parse_args();require(a.reviewer_id.strip(),'Truthful reviewer required');sys.meta_path.insert(0,NoNative());R=a.primary_root.resolve();require(a.output.resolve().is_relative_to(W/'local_data') and not a.output.exists(),'Fresh private proof path required')
 for name in ('scope','review','exit_witness'):require(sha(getattr(a,name))==getattr(a,'expected_'+name+'_sha256'),'External approval/exit SHA differs')
 s=load(a.scope);review=load(a.review);exitw=load(a.exit_witness);terminal=load(a.native_complete)
 require(s['kind']=='adaptive_accuracy_gate_v1' and review['verdict'] in ('PASS','PASS_WITH_LIMITATIONS') and all(review[k]==v for k,v in s.items()),'Exact independent adaptive approval required')
 require(exitw['exit_code']==0 and exitw['raw_log_sha256']==sha(a.native_log) and exitw.get('scope_file_sha256')==sha(a.scope) and exitw.get('review_sha256')==sha(a.review),'Actual closed exit/scope/approval binding required')
 require(terminal['scope_sha256']==objsha(s),'Native terminal scope mismatch');specpath=W/'local_data/adaptive_accuracy_gate_v1/spec_NOT_APPROVED_PRIVATE.json';spec=load(specpath);require(objsha(spec)==s['spec_sha256'],'Frozen spec changed')
 for key,h in spec['file_sha256'].items():require(sha(spec[key])==h,'Frozen input changed')
 for name,h in s['source_sha256'].items():require(sha(W/'src/mri_rl'/name)==h,'Frozen runtime source changed')
 gate_ns={'math':math};gate_tree=ast.parse((W/'src/mri_rl/adaptive_accuracy_gate_v1.py').read_text());gate_fn=next(v for v in gate_tree.body if isinstance(v,ast.FunctionDef) and v.name=='accuracy_group_is_informative');exec(compile(ast.Module(body=[gate_fn],type_ignores=[]),'bound_admission','exec'),gate_ns)
 require(sha(spec['parent_proof'])==PARENT_PROOF_SHA,'Independent parent proof changed');parent=Path(spec['parent_checkpoint']);pr=load(parent/'receipt.json');require(sha(parent/'receipt.json')==spec['parent_receipt_sha256'] and pr['completed_groups']==114 and pr['optimizer_updates']==52,'Parent anchor changed')
 run=Path(s['output']);paths=sorted(run.glob('boundary_*'));require(paths and paths[0].name=='boundary_0000','Bootstrap missing');rs=[load(q/'receipt.json') for q in paths];n,k=terminal_selection(terminal,rs[-1],paths[-1]);require([q.name for q in paths]==[f'boundary_{i:04d}' for i in range(n+1)],'Child chain incomplete')
 binding=s['run_binding'];anchor=binding['experiment']['adaptive_parent'];require(anchor['groups']==114 and anchor['updates']==52 and anchor['receipt_sha256']==sha(parent/'receipt.json') and anchor['original_binding']==pr['binding'],'Parent binding changed')
 require(rs[0]['files_sha256']['adapter.safetensors']==pr['files_sha256']['adapter.safetensors'] and sha(paths[0]/'adapter.safetensors')==sha(parent/'adapter.safetensors'),'Child bootstrap weights reset')
 lib=approved_math(R/'local_data/c_two_task_pool_v1/tier_c_v1/independent_remaining_RL_v1/verify_completed_RL_PRIVATE.py');TF=lib['TensorFile'];decode=lib['decode'];moments=lib['moments'];ae=lib['arrays_equal'];layout=lib['optimizer_layout']
 # Compare actual full optimizer AND RNG trees across parent->child publication.
 pm=load(parent/'optimizer/manifest.json');bm=load(paths[0]/'optimizer/manifest.json')
 with TF(parent/'optimizer/state.safetensors') as old,TF(paths[0]/'optimizer/state.safetensors') as new:
  require(sha(parent/'optimizer/state.safetensors')==pm['tensors_sha256'] and sha(paths[0]/'optimizer/state.safetensors')==bm['tensors_sha256'],'Bootstrap state hash differs');require(same_tree(decode(pm['layout'],old),decode(bm['layout'],new)),'Parent Adam or RNG reset during bootstrap')
 previous=None;previous_updates=0;recon=[];verified_files=0;inactive=0;completed_tokens=0;maxalign=0.
 for i,(q,r) in enumerate(zip(paths,rs)):
  require(r['binding']==binding and r['completed_groups']==i and r['absolute_completed_groups']==114+i and r['absolute_optimizer_updates']==52+r['optimizer_updates'] and r['parent_receipt_sha256']==previous,'Child binding/parent/offset mismatch')
  require(r['optimizer_updates']==previous_updates+int(r['updated']) and r['updated']==(i>0 and r['updated']),'Relative optimizer cursor mismatch')
  for name,h in r['files_sha256'].items():require((q/name).is_file() and not (q/name).is_symlink() and sha(q/name)==h,'Committed file changed');verified_files+=1
  require(all(r['state_validation'][name] is True for name in ('parameters_finite','optimizer_finite','second_moments_nonnegative')),'State validation failed')
  m=load(q/'optimizer/manifest.json');require(m['bindings']['run']==binding and m['bindings']['adapter_sha256']==r['files_sha256']['adapter.safetensors'] and m['completed_groups']==114+i and m['optimizer_updates']==52+r['optimizer_updates'] and sha(q/'optimizer/state.safetensors')==m['tensors_sha256'],'Absolute manifest counters/hash invalid')
  with TF(q/'optimizer/state.safetensors') as st:
   tree=decode(m['layout'],st);opt=tree['optimizer'];mm=moments(opt);require(int(opt['step'])==52+r['optimizer_updates'] and len(mm)==504,'Raw Adam step/moments reset');random.Random().setstate(tree['rng']['python']);rng=tree['rng']['mlx'];require(len(rng)==1 and rng[0].shape==(2,) and rng[0].dtype==np.uint32 and all(np.isfinite(v).all() for v in st.values()),'Invalid saved state/RNG')
   if i:
    prev=paths[i-1];oldm=load(prev/'optimizer/manifest.json');reservation=load(q/'attempt_reservation.json');raw=load(q/'rollout_PRIVATE.json');require(reservation['index']==i-1 and reservation['parent_receipt_sha256']==previous and reservation['run_binding_sha256']==objsha(binding) and raw['group_index']==114+i-1 and raw['record_id']==binding['schedule_record_ids'][i-1],'Schedule/cursor changed');require(raw['image_sha256']==binding['image_sha256'][raw['record_id']],'Image identity changed');require(len(raw['responses'])==4,'Group size changed');require(r['updated']==gate_ns['accuracy_group_is_informative'](r['grades']),'Saved admission decision mismatch');rewards=[g['total'] for g in r['grades']];mean=sum(rewards)/4;std=math.sqrt(sum((v-mean)**2 for v in rewards)/3);require(np.allclose(r['advantages'],[(v-mean)/(std+1e-6) for v in rewards],rtol=0,atol=1e-12),'Saved group advantages mismatch');completed_tokens+=4*reservation['input_tokens']+sum(len(v['token_ids']) for v in raw['responses'])
    with TF(prev/'optimizer/state.safetensors') as prior,TF(q/'optimizer_before.safetensors') as before,TF(q/'optimizer_after.safetensors') as after,TF(prev/'adapter.safetensors') as w0,TF(q/'adapter.safetensors') as w1:
     oldtree=decode(oldm['layout'],prior);oldmm=moments(oldtree['optimizer']);require(set(mm)==set(oldmm)==set(w0)==set(w1) and len(w0)==504,'Trainable inventory changed');ae(before,{name:prior[name] for name in before});ae(after,{name:st[name] for name in after});require(hashlib.sha256((canonical(layout(oldm))+sha(q/'optimizer_before.safetensors')).encode()).hexdigest()==r['optimizer_before_fingerprint'] and hashlib.sha256((canonical(layout(m))+sha(q/'optimizer_after.safetensors')).encode()).hexdigest()==r['optimizer_after_fingerprint'],'Moment fingerprints changed')
     if not r['updated']:
      require(same_tree(oldtree['optimizer'],opt) and all(np.array_equal(w0[name],w1[name]) for name in w0),'Inactive group changed optimizer/weights');inactive+=1
     else:
      stats=[lib['verify_adam_tensor'](left,w1[name],oldmm[name]['m'],oldmm[name]['v'],mm[name]['m'],mm[name]['v']) for name,left in w0.items()];norm=math.sqrt(sum(v['gradient_squared'] for v in stats));ev=r['update_evidence'];require(math.isclose(norm,ev['gradient_l2'],rel_tol=1e-5,abs_tol=1e-7),'Gradient norm mismatch');require(len(ev['loss_terms'])==len(ev['old_current_logp_max_abs_errors'])==4 and all(math.isfinite(v) for v in ev['loss_terms']) and all(math.isfinite(v) and 0<=v<=1e-5 for v in ev['old_current_logp_max_abs_errors']),'Invalid saved update evidence');maxalign=max(maxalign,max(ev['old_current_logp_max_abs_errors']));changed=sum(v['changed_elements']>0 for v in stats);require(changed>0,'Active update has no changed weights');recon.append({'absolute_step_before':52+previous_updates,'absolute_step_after':52+r['optimizer_updates'],'changed_tensors':changed,'max_weight_error':max(v['w_error'] for v in stats),'max_second_moment_error':max(v['v_error'] for v in stats),'gradient_l2_reconstructed':norm,'gradient_l2_saved':ev['gradient_l2']})
  previous=sha(q/'receipt.json');previous_updates=r['optimizer_updates']
 ledger=rs[-1]['ledger'];require(ledger['cap']==s['new_token_cap'] and ledger['used_tokens']<=ledger['cap'] and terminal['new_stage_budget']==ledger['cap'],'Terminal token ledger changed');debt=ledger['used_tokens']-completed_tokens;require(debt==ledger['imputed_unknown_tokens']==0 and not ledger['partial_dirs_included'] and ledger['charged_groups']==n and len(recon)==k,'Partial debt requires separate reconciliation or token/update counts inconsistent')
 require(not any(name in sys.modules for name in ('mlx','mlx.core','mlx_vlm','torch','transformers')),'Native runtime imported')
 report={'schema':'adaptive-continuation-focused-CPU-proof-v1','verdict':'PASS_WITH_LIMITATIONS','reviewer':a.reviewer_id,'reviewer_kind':'isolated_codex','preserved_initial_parent_state':True,'parent_native_step':52,'completed_groups':n,'new_optimizer_updates':k,'absolute_completed_groups':114+n,'absolute_optimizer_updates':52+k,'terminal_boundary_receipt_sha256':previous,'final_adapter_sha256':sha(paths[-1]/'adapter.safetensors'),'all_Adam_reconstructions':recon,'inactive_groups_with_unchanged_weights_and_optimizer':inactive,'optimizer_RNG_states_verified':n+1,'verified_files':verified_files,'completed_rollout_tokens':completed_tokens,'used_tokens':ledger['used_tokens'],'partial_work_debt_tokens':debt,'ledger':ledger,'maximum_saved_onpolicy_alignment_error':maxalign,'evidence_sha256':{'scope':sha(a.scope),'review':sha(a.review),'external_exit':sha(a.exit_witness),'closed_native_log':sha(a.native_log),'native_complete':sha(a.native_complete),'parent_proof':PARENT_PROOF_SHA,'parent_receipt':sha(parent/'receipt.json'),'boundary_receipts':[sha(q/'receipt.json') for q in paths],'this_verifier':sha(__file__),'reviewed_math_source':MATH_SHA},'limitations':['Focused parent-state and saved Adam/weight consistency proof only; parent114 groups are not re-executed.','Native backward/loss/logprob and raw-response task grades are not independently recomputed; saved-grade admission and sample-std group advantages are checked.','RNG preservation across initial restore is checked exactly; later RNG must be valid and saved, not necessarily changed.','New-stage token ledger is separate from exhausted parent cap; any partial-work debt requires separate reconciliation if nonzero.','No MRI accuracy, independent-patient generalization or admission-rule benefit follows from more updates.']}
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('x') as f:json.dump(report,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps({'status':'ACTUAL_ADAPTIVE_CPU_PROOF_COMPLETE','verdict':report['verdict'],'proof_sha256':sha(a.output),'new_groups':n,'new_updates':k,'absolute_Adam_step':52+k}))
if __name__=='__main__':main()
