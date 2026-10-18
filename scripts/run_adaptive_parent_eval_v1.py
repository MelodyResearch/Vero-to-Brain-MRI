"""Read-only parent checkpoint selector over the unchanged reviewed V2 evaluator."""
import argparse,copy,hashlib,importlib.util,json
from pathlib import Path
from types import SimpleNamespace
W=Path(__file__).resolve().parents[1]
ENTRY_SHA='d0202667c4a580c4ea34cdd22a31a2238fd2de8ad81f26277600f351b5dd152c'
EVAL_SHA='8c52a40f572393f4a01c6fbd2eb84d044bc14ecd151ae00ef56d9e7600a2539b'
PARENT_SHA='c70b426b58191446545ba6333f7595e96f4fdb4cc689ddbaf5206e276b329db6'
RECEIPT_SHA='e0696a8186d413b2bb35034b5c5eb2568ddc0b05e1b719778de42fa491cc882a'
PROOF_SHA='a5d5cd1faf58492b173177668175e84cc2ffc94b13e15dff4705d6bf5d7f59cb'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
load=lambda p:json.loads(Path(p).read_text())
def require(ok,message):
 if not ok:raise ValueError(message)
def validate_parent(receipt,proof,adapter_sha):
 require(adapter_sha==PARENT_SHA and proof['verdict']=='PASS_WITH_LIMITATIONS' and proof['stage_id']=='rl:P_calibrated:13' and proof['full_arm_completed'] is True,'Exact independently verified parent required')
 require(receipt['completed_groups']==proof['completed_groups']==114 and receipt['optimizer_updates']==proof['optimizer_updates']==52,'Parent counters differ')
 require(proof['final_adapter_sha256']==PARENT_SHA and proof['terminal_boundary_receipt_sha256']==RECEIPT_SHA,'Parent proof/checkpoint differs')
def select_scope(entry,scope,parent,output):
 s=copy.deepcopy(scope);s.update(kind='adaptive_parent_read_only_evaluation_v1',checkpoint='P_calibrated13',adapter=parent['adapter'],parent_checkpoint=parent,output_directory=str(output.resolve()),resume=False,starting_journal=None,maximum_new_updates=0,max_new_steps=0,candidate_forwards=75,new_greedy_generations=41,adapter_selector_sha256=sha(__file__),scientific_claim='Read-only parent baseline for a separately versioned adaptive continuation; repeated internal dev only.')
 s['run_identity']=entry.scope_identity(s);s['starting_outputs_sha256']={n:(sha(output/n) if (output/n).exists() else None) for n in ('greedy_PRIVATE.jsonl','candidates_PRIVATE.jsonl','eval_complete_SAFE.json')};return s

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--primary-root',type=Path,required=True);p.add_argument('--output-directory',type=Path,required=True);p.add_argument('--scope-output',type=Path);p.add_argument('--review',type=Path);p.add_argument('--expected-review-sha256');p.add_argument('--execute',action='store_true');a=p.parse_args();R=a.primary_root.resolve();V=R/'.worktrees/vero-mri-v2-controls';output=a.output_directory.resolve();require(output.is_relative_to(W/'local_data'),'Private adaptive output required')
 entrypath=V/'scripts/run_paired_discrimination_pilot_v2.py';require(sha(entrypath)==ENTRY_SHA,'Reviewed evaluator changed');spec=importlib.util.spec_from_file_location('adaptive_parent_reviewed_evaluator',entrypath);entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
 template=load(V/'local_data/paired_discrimination_pilot_v2/entry_v2/post2_evaluation_AFTER_TWO_UPDATES_READY_FOR_REVIEW_PRIVATE.json');feature=template['feature_evidence']
 b=SimpleNamespace(primary_root=R,pilot_plan=Path(template['pilot_plan_path']),run_directory=Path(template['run_directory']),eval_contract=Path(template['evaluation_contract_path']),output_directory=V/'local_data/paired_discrimination_pilot_v2/parent_bootstrap_UNUSED',feature_plan=Path(feature['plan_path']),feature_plan_sha256=feature['plan_sha256'],feature_completion=Path(feature['receipt_path']),feature_completion_sha256=feature['receipt_sha256'],checkpoint='post2',mode='eval',resume=False,review=a.review,expected_review_sha256=a.expected_review_sha256)
 require(sha(b.eval_contract)==EVAL_SHA and not b.output_directory.exists(),'Exact frozen evaluation/unused bootstrap required');values=entry.prepare(b);require(values[-1]['candidate_forwards']==75 and values[-1]['new_greedy_generations']==41,'Existing full evaluation route differs')
 boundary=R/'local_data/c_two_task_pool_v1/tier_c_v1/calibrated_bundle_v1/P_calibrated_seed13/run/boundary_0114';proofpath=R/'local_data/c_two_task_pool_v1/tier_c_v1/independent_remaining_RL_v1/rl_P_calibrated_13/receipt_SAFE.json';require(sha(boundary/'receipt.json')==RECEIPT_SHA and sha(proofpath)==PROOF_SHA,'Parent terminal or independent proof changed');receipt=load(boundary/'receipt.json');proof=load(proofpath);validate_parent(receipt,proof,sha(boundary/'adapter.safetensors'))
 require(sha(boundary/'adapter_config.json')==template['common_B28']['adapter_config_sha256'],'Parent adapter config differs')
 parent={'adapter':{'path':str(boundary/'adapter.safetensors'),'sha256':PARENT_SHA,'adapter_config_sha256':sha(boundary/'adapter_config.json')},'receipt_path':str(boundary/'receipt.json'),'receipt_sha256':RECEIPT_SHA,'proof_path':str(proofpath),'proof_sha256':PROOF_SHA,'completed_groups':114,'optimizer_updates':52}
 scope=select_scope(entry,values[-1],parent,output);require(scope['entry_sha256']==ENTRY_SHA and scope['evaluation_contract_sha256']==EVAL_SHA,'Evaluator contract changed');b.output_directory=output;b.checkpoint='P_calibrated13'
 if not a.execute:
  require(a.scope_output and a.scope_output.resolve().is_relative_to(W/'local_data'),'Private scope output required');entry.save(a.scope_output,scope);print(json.dumps({'status':'PARENT_EVAL_SCOPE_ONLY_NO_NATIVE','scope_sha256':sha(a.scope_output),'ready':scope['ready'],'greedy_generations':41,'candidate_forwards':75,'weight_updates':0}));return
 entry.execute(b,*values[:-1],scope)
if __name__=='__main__':main()
