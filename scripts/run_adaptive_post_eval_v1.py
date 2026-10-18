"""Conditional terminal-child selector; unchanged reviewed40+25 evaluator runtime."""
import argparse,copy,hashlib,importlib.util,json
from pathlib import Path
from types import SimpleNamespace
W=Path(__file__).resolve().parents[1]
PARENT_SELECTOR_SHA='9ac7bcfa0784c4e3b275b261b9e1760122422f800bcd6714fb1725f0db63e17d'
PARENT_SCOPE_SHA='cc57a97ec4e5825d5cd86a691ec35b13ffac18cf520e187b39a2c8765d1c1d31'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
load=lambda p:json.loads(Path(p).read_text())
def require(ok,message):
 if not ok:raise ValueError(message)
def module(path,name,expected):
 require(sha(path)==expected,'Reviewed selector/runtime changed');s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def validate_child(proof,terminal,receipt,receipt_sha,adapter_sha):
 require(proof['schema']=='adaptive-continuation-focused-CPU-proof-v1' and proof['verdict']=='PASS_WITH_LIMITATIONS' and proof['preserved_initial_parent_state'] is True and proof['parent_native_step']==52,'Actual independent parent-preservation proof required')
 n=proof['completed_groups'];k=proof['new_optimizer_updates'];require(type(n) is type(k) is int and 1<=n<=16 and 0<=k<=min(4,n),'Child finite budget exceeded')
 require(receipt['completed_groups']==terminal['completed_groups']==n and receipt['optimizer_updates']==terminal['optimizer_updates']==k and receipt['absolute_completed_groups']==terminal['absolute_completed_groups']==proof['absolute_completed_groups']==114+n and receipt['absolute_optimizer_updates']==terminal['absolute_optimizer_updates']==proof['absolute_optimizer_updates']==52+k,'Relative/absolute counters differ')
 require(terminal['status'] in {'SCHEDULE_COMPLETE','UPDATE_BUDGET_REACHED','TOKEN_BUDGET_EXHAUSTED'} and proof['terminal_boundary_receipt_sha256']==receipt_sha and proof['final_adapter_sha256']==adapter_sha,'Exact terminal checkpoint/proof required; no best-dev selection')
 require((n==16 or k==4) and (terminal['status']!='SCHEDULE_COMPLETE' or n==16) and (terminal['status']!='UPDATE_BUDGET_REACHED' or k==4),'Actual16group or4update finite termination required')

def select_child_scope(entry,parent_scope,child,output):
 s=copy.deepcopy(parent_scope);s.update(kind='adaptive_post_read_only_evaluation_v1',checkpoint='P_calibrated13_adaptive',adapter=child['adapter'],adaptive_child=child,output_directory=str(output.resolve()),starting_journal=None,maximum_new_updates=0,max_new_steps=0,candidate_forwards=75,new_greedy_generations=41,post_adapter_selector_sha256=sha(__file__),scientific_claim='Exact finite-terminal adaptive checkpoint vs its own Pcal13 parent; repeatedly examined internal dev only.')
 s['run_identity']=entry.scope_identity(s);s['starting_outputs_sha256']={n:(sha(output/n) if (output/n).exists() else None) for n in ('greedy_PRIVATE.jsonl','candidates_PRIVATE.jsonl','eval_complete_SAFE.json')};return s

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--primary-root',type=Path,required=True);p.add_argument('--proof',type=Path,required=True);p.add_argument('--expected-proof-sha256');p.add_argument('--output-directory',type=Path,required=True);p.add_argument('--scope-output',type=Path);p.add_argument('--review',type=Path);p.add_argument('--expected-review-sha256');p.add_argument('--execute',action='store_true');a=p.parse_args();require(a.output_directory.resolve().is_relative_to(W/'local_data'),'Private adaptive destination required')
 if not a.proof.exists():
  require(not a.execute,'No independent child proof');print(json.dumps({'status':'NOT_READY_NO_ACTUAL_COMPLETED_CHILD_PROOF','ready':False,'native_execution':False}));return
 require(a.expected_proof_sha256 and sha(a.proof)==a.expected_proof_sha256,'External independent proof digest required');proof=load(a.proof);R=a.primary_root.resolve();V=R/'.worktrees/vero-mri-v2-controls';parent=module(W/'scripts/run_adaptive_parent_eval_v1.py','adaptive_parent_selector',PARENT_SELECTOR_SHA);entry=module(V/'scripts/run_paired_discrimination_pilot_v2.py','adaptive_post_reviewed_evaluator',parent.ENTRY_SHA)
 parentpath=W/'local_data/adaptive_parent_Pcal13_eval_v1/scope_READY_REQUIRES_REVIEW_PRIVATE.json';require(sha(parentpath)==PARENT_SCOPE_SHA,'Frozen actual parent baseline scope changed');ps=load(parentpath)
 spec=load(W/'local_data/adaptive_accuracy_gate_v1/spec_NOT_APPROVED_PRIVATE.json');run=Path(spec['output']);terminalpath=run.parent/'native_complete_SAFE.json';terminal=load(terminalpath);q=Path(terminal['boundary']);require(q.parent.resolve()==run.resolve() and q.name==f"boundary_{terminal['completed_groups']:04d}" and q==max(run.glob('boundary_*')),'Only last actual terminal boundary eligible');r=load(q/'receipt.json');validate_child(proof,terminal,r,sha(q/'receipt.json'),sha(q/'adapter.safetensors'))
 require(proof['evidence_sha256']['native_complete']==sha(terminalpath),'Actual terminal changed after independent verification');exitpath=run.parent/'actual_exit_PRIVATE.json';require(exitpath.exists() and sha(exitpath)==proof['evidence_sha256']['external_exit'] and load(exitpath)['exit_code']==0,'Actual child exit witness changed')
 require(sha(q/'adapter_config.json')==ps['adapter']['adapter_config_sha256'],'Frozen adapter configuration changed')
 f=ps['feature_evidence'];b=SimpleNamespace(primary_root=R,pilot_plan=Path(ps['pilot_plan_path']),run_directory=Path(ps['run_directory']),eval_contract=Path(ps['evaluation_contract_path']),output_directory=V/'local_data/paired_discrimination_pilot_v2/parent_bootstrap_UNUSED',feature_plan=Path(f['plan_path']),feature_plan_sha256=f['plan_sha256'],feature_completion=Path(f['receipt_path']),feature_completion_sha256=f['receipt_sha256'],checkpoint='post2',mode='eval',resume=False,review=a.review,expected_review_sha256=a.expected_review_sha256)
 require(not b.output_directory.exists() and sha(b.eval_contract)==parent.EVAL_SHA,'Frozen evaluator bootstrap changed');values=entry.prepare(b)
 child={'adapter':{'path':str(q/'adapter.safetensors'),'sha256':sha(q/'adapter.safetensors'),'adapter_config_sha256':sha(q/'adapter_config.json')},'receipt_path':str(q/'receipt.json'),'receipt_sha256':sha(q/'receipt.json'),'proof_path':str(a.proof.resolve()),'proof_sha256':sha(a.proof),'native_complete_path':str(terminalpath),'native_complete_sha256':sha(terminalpath),'native_exit_path':str(exitpath),'native_exit_sha256':sha(exitpath),'relative_completed_groups':r['completed_groups'],'relative_optimizer_updates':r['optimizer_updates'],'absolute_completed_groups':r['absolute_completed_groups'],'absolute_optimizer_updates':r['absolute_optimizer_updates']}
 scope=select_child_scope(entry,ps,child,a.output_directory);b.output_directory=a.output_directory.resolve();b.checkpoint='P_calibrated13_adaptive'
 if not a.execute:
  require(a.scope_output and a.scope_output.resolve().is_relative_to(W/'local_data'),'Private scope destination required');entry.save(a.scope_output,scope);print(json.dumps({'status':'EXACT_TERMINAL_CHILD_EVAL_SCOPE_NOT_RUN','ready':True,'scope_sha256':sha(a.scope_output),'greedy_generations':41,'candidate_forwards':75,'new_updates':0}));return
 entry.execute(b,*values[:-1],scope)
if __name__=='__main__':main()
