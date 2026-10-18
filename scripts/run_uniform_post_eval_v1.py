"""Exact finite uniform-stage terminal evaluation; completed Adam56 baseline is authenticated and never regenerated."""
import argparse,copy,hashlib,importlib.util,json
from pathlib import Path
from types import SimpleNamespace
W=Path(__file__).resolve().parents[1]
ENTRY_SHA='d0202667c4a580c4ea34cdd22a31a2238fd2de8ad81f26277600f351b5dd152c'
EVAL_SHA='8c52a40f572393f4a01c6fbd2eb84d044bc14ecd151ae00ef56d9e7600a2539b'
BASELINE_EVIDENCE_SHA256={'scope': '022c97281506fce016d8dc74dcec983c0aa52797f34c92baf02c4189717a84f8', 'eval_receipt': '2fde141b0fcff71a02aed4b7e214c4082f867a4a8c5d21b35a59b8b1a0d8f517', 'exit_receipt': 'fb767854489dc2f006a0c20a97e1bfbb35e179b86c778113329f63ad933f2596', 'closed_log': '5781ffd5df55a1a3c302c9fb02ad7c320156a6cf8b20a15ad5460f0cf4a0c5c6', 'greedy_output': '852bbcf33361a0b7788d3b404a1d04b6a14d97099cb8753e2472802439b48b0c', 'candidate_output': 'd62a9c1e8b98fb3334f8c90b445af0e537e4c5aca916e204f767fcc42158ee36'}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
load=lambda p:json.loads(Path(p).read_text())
def require(ok,message):
 if not ok:raise ValueError(message)
def module(path,name,expected):
 require(sha(path)==expected,'Reviewed evaluator changed');s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def validate_child(proof,terminal,receipt,receipt_sha,adapter_sha):
 require(proof['schema']=='uniform-continuation-focused-CPU-proof-v1' and proof['verdict']=='PASS_WITH_LIMITATIONS' and proof['preserved_initial_parent_state'] is True and proof['parent_native_step']==56 and proof['parent_absolute_groups']==121,'Verified Adam56/full-parent-state continuation required')
 n=proof['completed_groups'];k=proof['new_optimizer_updates'];require(type(n) is type(k) is int and 1<=n<=16 and 0<=k<=min(4,n),'Uniform finite budget exceeded')
 require(receipt['completed_groups']==terminal['completed_groups']==n and receipt['optimizer_updates']==terminal['optimizer_updates']==k and receipt['absolute_completed_groups']==terminal['absolute_completed_groups']==proof['absolute_completed_groups']==121+n and receipt['absolute_optimizer_updates']==terminal['absolute_optimizer_updates']==proof['absolute_optimizer_updates']==56+k,'Relative/absolute uniform counters differ')
 require(terminal['status'] in {'SCHEDULE_COMPLETE','UPDATE_BUDGET_REACHED','TOKEN_BUDGET_EXHAUSTED'} and proof['terminal_boundary_receipt_sha256']==receipt_sha and proof['final_adapter_sha256']==adapter_sha,'Only exact finite-terminal child is eligible')
 require((n==16 or k==4) and (terminal['status']!='SCHEDULE_COMPLETE' or n==16) and (terminal['status']!='UPDATE_BUDGET_REACHED' or k==4),'Actual16group or4update finite termination required')

def select_child_scope(entry,baseline,child,output,evidence):
 s=copy.deepcopy(baseline);s.update(kind='uniform_category_post_read_only_evaluation_v1',checkpoint='uniform_category_continuation',adapter=child['adapter'],uniform_child=child,baseline_evaluation=evidence,output_directory=str(output.resolve()),starting_journal=None,resume=False,maximum_new_updates=0,max_new_steps=0,candidate_forwards=75,new_greedy_generations=41,uniform_adapter_selector_sha256=sha(__file__),scientific_claim='Uniform-category continuation from actual Adam56 trained parent; before/after is not a matched proportional-continuation causal experiment.')
 s['run_identity']=entry.scope_identity(s);s['starting_outputs_sha256']={n:(sha(output/n) if (output/n).exists() else None) for n in ('greedy_PRIVATE.jsonl','candidates_PRIVATE.jsonl','eval_complete_SAFE.json')};return s

def baseline_scope(entry):
 root=W/'local_data/adaptive_post_Pcal13_eval_v1';paths={'scope':root/'scope_READY_REQUIRES_REVIEW_PRIVATE.json','eval_receipt':root/'actual/eval_complete_SAFE.json','exit_receipt':root/'actual/actual_exit_PRIVATE.json','closed_log':root/'actual/native_stdout_PRIVATE.log','greedy_output':root/'actual/greedy_PRIVATE.jsonl','candidate_output':root/'actual/candidates_PRIVATE.jsonl'}
 for name,path in paths.items():require(sha(path)==BASELINE_EVIDENCE_SHA256[name],'Completed Adam56 baseline changed')
 s=load(paths['scope']);r=load(paths['eval_receipt']);exitw=load(paths['exit_receipt']);require(s['checkpoint']=='P_calibrated13_adaptive' and s['kind']=='adaptive_post_read_only_evaluation_v1' and s['entry_sha256']==ENTRY_SHA and s['evaluation_contract_sha256']==EVAL_SHA and s['run_identity']==entry.scope_identity(s),'Saved Adam56 baseline contract differs')
 require(exitw['exit_code']==0 and exitw['raw_log_sha256']==sha(paths['closed_log']) and exitw['scope_file_sha256']==sha(paths['scope']),'Saved baseline exit/log differs');entry.validate_completion(r,s,load(s['evaluation_contract_path']),paths['greedy_output'],paths['candidate_output'])
 return s,{name:{'path':str(path),'sha256':sha(path)} for name,path in paths.items()}

def main():
 p=argparse.ArgumentParser();p.add_argument('--primary-root',type=Path,required=True);p.add_argument('--proof',type=Path,required=True);p.add_argument('--expected-proof-sha256');p.add_argument('--output-directory',type=Path,required=True);p.add_argument('--scope-output',type=Path);p.add_argument('--review',type=Path);p.add_argument('--expected-review-sha256');p.add_argument('--execute',action='store_true');a=p.parse_args();R=a.primary_root.resolve();V=R/'.worktrees/vero-mri-v2-controls';require(a.output_directory.resolve().is_relative_to(W/'local_data'),'Private uniform evaluation required')
 if not a.proof.exists():
  require(not a.execute,'Actual uniform child proof required');print(json.dumps({'status':'NOT_READY_NO_ACTUAL_UNIFORM_PROOF','native_execution':False}));return
 require(a.expected_proof_sha256 and sha(a.proof)==a.expected_proof_sha256,'External uniform proof digest required');proof=load(a.proof);entry=module(V/'scripts/run_paired_discrimination_pilot_v2.py','uniform_reviewed_evaluator',ENTRY_SHA);base,evidence=baseline_scope(entry)
 phase=W/'local_data/uniform_balance_continuation_v1';terminalpath=phase/'native_complete_SAFE.json';terminal=load(terminalpath);q=Path(terminal['boundary']);run=phase/'run';require(q.parent.resolve()==run.resolve() and q.name==f"boundary_{terminal['completed_groups']:04d}" and q==max(run.glob('boundary_*')),'Only last uniform boundary eligible');r=load(q/'receipt.json');validate_child(proof,terminal,r,sha(q/'receipt.json'),sha(q/'adapter.safetensors'))
 require(proof['evidence_sha256']['native_complete']==sha(terminalpath),'Uniform native completion changed');exitpath=phase/'actual_exit_PRIVATE.json';require(sha(exitpath)==proof['evidence_sha256']['external_exit'] and load(exitpath)['exit_code']==0,'Uniform actual exit changed');require(sha(q/'adapter_config.json')==base['adapter']['adapter_config_sha256'],'Adapter configuration changed')
 f=base['feature_evidence'];b=SimpleNamespace(primary_root=R,pilot_plan=Path(base['pilot_plan_path']),run_directory=Path(base['run_directory']),eval_contract=Path(base['evaluation_contract_path']),output_directory=V/'local_data/paired_discrimination_pilot_v2/parent_bootstrap_UNUSED',feature_plan=Path(f['plan_path']),feature_plan_sha256=f['plan_sha256'],feature_completion=Path(f['receipt_path']),feature_completion_sha256=f['receipt_sha256'],checkpoint='post2',mode='eval',resume=False,review=a.review,expected_review_sha256=a.expected_review_sha256)
 require(not b.output_directory.exists() and sha(b.eval_contract)==EVAL_SHA,'Frozen evaluation bootstrap changed');values=entry.prepare(b)
 child={'adapter':{'path':str(q/'adapter.safetensors'),'sha256':sha(q/'adapter.safetensors'),'adapter_config_sha256':sha(q/'adapter_config.json')},'receipt_path':str(q/'receipt.json'),'receipt_sha256':sha(q/'receipt.json'),'proof_path':str(a.proof.resolve()),'proof_sha256':sha(a.proof),'native_complete_path':str(terminalpath),'native_complete_sha256':sha(terminalpath),'native_exit_path':str(exitpath),'native_exit_sha256':sha(exitpath),'relative_completed_groups':r['completed_groups'],'relative_optimizer_updates':r['optimizer_updates'],'absolute_completed_groups':r['absolute_completed_groups'],'absolute_optimizer_updates':r['absolute_optimizer_updates']}
 scope=select_child_scope(entry,base,child,a.output_directory,evidence);b.output_directory=a.output_directory.resolve();b.checkpoint='uniform_category_continuation'
 if not a.execute:
  require(a.scope_output and a.scope_output.resolve().is_relative_to(W/'local_data'),'Private scope required');entry.save(a.scope_output,scope);print(json.dumps({'status':'UNIFORM_TERMINAL_EVAL_SCOPE_ONLY','scope_sha256':sha(a.scope_output),'weight_updates':0,'baseline_regenerated':False}));return
 entry.execute(b,*values[:-1],scope)
if __name__=='__main__':main()
