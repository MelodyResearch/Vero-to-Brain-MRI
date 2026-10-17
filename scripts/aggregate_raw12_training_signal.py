"""Aggregate only twelve verified sanitized raw-arm training proofs.

No raw rollouts, checkpoints, images, development outputs or native imports.
All counts are descriptive realized training exposure, not MRI performance.
"""
from pathlib import Path
import argparse,hashlib,json
ARMS=('S_sequence','S_correspondence','P_raw','U_raw');SEEDS=(7,11,13)
TASKS=('mri_sequence','mri_visit_correspondence')
U7_SHA='39b1a6bd0d9064323a4b62222de1f3539f7bdf1b8d70d0dc8555bc9589ab7b89'
def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rate(a,b):return a/b if b else None
def integer(x,label):require(type(x) is int and x>=0,'Invalid count: '+label);return x
def path_for(root,arm,seed):
 return root/('U_RAW_SEED7_FINAL_NUMERICAL_INDEPENDENT_VERIFICATION.json' if (arm,seed)==('U_raw',7) else f'COMPLETED_rl_{arm}_{seed}_NUMERICAL_INDEPENDENT_VERIFICATION.json')
def normalize(proof,arm,seed,file_hash):
 require(proof['verdict'] in {'PASS','PASS_WITH_LIMITATIONS'} and proof['full_arm_completed'] is True,'Arm proof is not independently verified complete')
 if (arm,seed)==('U_raw',7):
  require(file_hash==U7_SHA and proof['schema']=='u-raw7-final-independent-cpu-verification-v1','Unsupported legacy U7 proof')
  provenance='Legacy proof lacks arm/seed/stage_id fields; exact known U7 proof SHA identifies these values.'
 else:
  require(proof['schema']=='completed-RL-independent-numerical-verification-v1' and proof['stage_id']==f'rl:{arm}:{seed}' and proof['seed']==seed and proof['arm']['name']==arm and proof['arm']['reward']=='raw','Arm/seed/raw reward differs')
  provenance='Explicit arm/seed/stage_id from proof.'
 groups=integer(proof['completed_groups'],'groups');updates=integer(proof['optimizer_updates'],'updates');completed=integer(proof['completed_rollout_tokens'],'completed tokens');used=integer(proof['used_tokens_recomputed'],'charged tokens')
 measured=integer(proof['measured_crashed_work_tokens'],'measured crash tokens');imputed=integer(proof['imputed_unknown_tokens'],'imputed crash tokens');cap=integer(proof['cap'],'cap')
 require(cap==169216 and completed+measured+imputed==used<=cap,'Token budget/debt decomposition differs')
 require(sum(proof['attempted_groups_by_task'].values())==groups and sum(proof['accepted_update_groups_by_task'].values())==updates,'Task counts do not reconcile')
 result={'arm':arm,'seed':seed,'completed_attempted_groups':groups,'accepted_optimizer_updates':updates,'completed_rollout_tokens':completed,'measured_crashed_work_tokens':measured,'imputed_unknown_tokens':imputed,'charged_tokens':used,'cap':cap,'unused_tokens':cap-used,'tasks':{},'missing_or_limited_fields':['Group-level all-full-format counts/joint reason histograms are not published; marginal full-format rollouts do not uniquely identify why a group was rejected.','Crash debt is retained at arm level; no task assignment inferred.'],'identity_provenance':provenance}
 for task in TASKS:
  n=integer(proof['attempted_groups_by_task'].get(task,0),task+'attempts');accepted=integer(proof['accepted_update_groups_by_task'].get(task,0),task+'updates');variance=proof['variance_diagnostics'];zero=integer(variance.get(task+':zero_total_variance',0),task+'zero variance');vary=integer(variance.get(task+':task_varies',0),task+'task varies');nonzero=n-zero
  require(0<=accepted<=vary<=nonzero<=n,'Raw informative/variance subset relation violated')
  counts=proof['raw_response_counts_and_correctness_by_task'].get(task,{})
  if n:require(all(k in variance for k in (task+':zero_total_variance',task+':task_varies')) and all(k in counts for k in ('responses','full_format')) and all(k in proof['token_decomposition'] for k in (task+':input',task+':generated')),'Exposed task diagnostics missing; cannot substitute zeros')
  responses=integer(counts.get('responses',0),task+'responses');full=integer(counts.get('full_format',0),task+'full format');require(responses==4*n and full<=responses,'Rollout/format marginal mismatch')
  inp=integer(proof['token_decomposition'].get(task+':input',0),task+'input tokens');gen=integer(proof['token_decomposition'].get(task+':generated',0),task+'generated tokens')
  result['tasks'][task]={'attempted_groups':n,'accepted_updates':accepted,'attempt_share':n/groups,'accepted_update_share':rate(accepted,updates),'accepted_per_attempt':rate(accepted,n),'groups_with_task_accuracy_variation':vary,'accepted_per_task_varying_group':rate(accepted,vary),'task_varying_groups_not_accepted':vary-accepted,'zero_total_reward_variance_groups':zero,'nonzero_total_reward_variance_groups':nonzero,'nonzero_total_variance_without_task_variation':nonzero-vary,'nonzero_total_variance_groups_not_accepted':nonzero-accepted,'rollouts':responses,'full_format_rollouts':full,'not_full_format_rollouts':responses-full,'full_format_rollout_fraction':rate(full,responses),'input_tokens':inp,'generated_tokens':gen,'completed_rollout_tokens':inp+gen,'completed_rollout_token_share':rate(inp+gen,completed)}
 require(sum(t['completed_rollout_tokens'] for t in result['tasks'].values())==completed,'Task completed token sum differs')
 return result

def aggregate(rows):
 result={}
 fields=('attempted_groups','accepted_updates','groups_with_task_accuracy_variation','task_varying_groups_not_accepted','zero_total_reward_variance_groups','nonzero_total_reward_variance_groups','nonzero_total_variance_without_task_variation','nonzero_total_variance_groups_not_accepted','rollouts','full_format_rollouts','not_full_format_rollouts','input_tokens','generated_tokens','completed_rollout_tokens')
 for arm in ARMS:
  selected=[r for r in rows if r['arm']==arm];require([r['seed'] for r in selected]==list(SEEDS),'Missing arm seeds')
  out={'seeds':list(SEEDS),'completed_attempted_groups':sum(r['completed_attempted_groups'] for r in selected),'accepted_optimizer_updates':sum(r['accepted_optimizer_updates'] for r in selected),'charged_tokens':sum(r['charged_tokens'] for r in selected),'completed_rollout_tokens':sum(r['completed_rollout_tokens'] for r in selected),'measured_crashed_work_tokens':sum(r['measured_crashed_work_tokens'] for r in selected),'imputed_unknown_tokens':sum(r['imputed_unknown_tokens'] for r in selected),'tasks':{}}
  for task in TASKS:
   t={f:sum(r['tasks'][task][f] for r in selected) for f in fields};t.update(attempt_share=rate(t['attempted_groups'],out['completed_attempted_groups']),accepted_update_share=rate(t['accepted_updates'],out['accepted_optimizer_updates']),accepted_per_attempt=rate(t['accepted_updates'],t['attempted_groups']),accepted_per_task_varying_group=rate(t['accepted_updates'],t['groups_with_task_accuracy_variation']),full_format_rollout_fraction=rate(t['full_format_rollouts'],t['rollouts']),completed_rollout_token_share=rate(t['completed_rollout_tokens'],out['completed_rollout_tokens']),accepted_updates_range_across_seeds=[min(r['tasks'][task]['accepted_updates'] for r in selected),max(r['tasks'][task]['accepted_updates'] for r in selected)])
   out['tasks'][task]=t
  result[arm]=out
 return result

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--proof-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--reviewer-id',required=True);a=p.parse_args();require(a.reviewer_id.strip(),'Actual aggregation operator required')
 rows=[];inputs={}
 for arm in ARMS:
  for seed in SEEDS:
   path=path_for(a.proof_root,arm,seed);require(path.is_file(),'WAITING_VERIFIED_PUBLIC_PROOF: '+path.name);digest=sha(path);x=json.loads(path.read_text());rows.append(normalize(x,arm,seed,digest));inputs[path.name]={'sha256':digest,'source_reviewer':x.get('reviewer'),'source_reviewer_kind':x.get('reviewer_kind'),'terminal_boundary_receipt_sha256':x['terminal_boundary_receipt_sha256']}
 require(len(inputs)==12,'Exactly twelve primary proofs required')
 result={'schema':'raw12-realized-training-signal-synthesis-v1','status':'COMPLETED_RAW12_PRIMARY_PROOFS_AGGREGATED','evidence_tier':'REPRODUCED_CPU_FROM_VERIFIED_SANITIZED_TRAINING_PROOFS','research_tag':'PRIMARY_VERO_TRANSFER','aggregation_operator':a.reviewer_id,'script_author':'/root/full_arm_execution_contract','script_sha256':sha(__file__),'input_proofs':inputs,'arm_seed_rows':rows,'pooled_three_seed_arm_totals':aggregate(rows),'definitions':{'attempted_groups':'Committed complete groups only; charged incomplete/crashed work appears separately as measured/imputed tokens.','task_accuracy_variation':'At least two distinct raw accuracy rewards in the group; alone this is not the accepted update predicate.','accepted_update':'Existing predicate requires every response full-format, task accuracy variation and identical overlong penalties; verified native step increments count actual accepted groups.','nonzero_total_variance':'A nonconstant combined task+format+overlong reward group; not equivalent to task accuracy variation or update acceptance.','not_full_format_rollouts':'Responses minus full_format, a marginal response count, NOT the number of rejected groups.','pooled_rates':'Ratios of summed counts across three seeds; no confidence interval, power or independence claim.','zero_exposure':'An untrained specialist task has zero counts and undefined conditional rates (null).'},'scientific_synthesis':['Uniform category sampling balances attempted categories approximately; accepted updates and token expenditure are separately measured realized exposures, not guaranteed balanced by that sampler.','Differences between task-varying and accepted counts show the existing admission predicate selects a subset; published marginals cannot assign a unique reason or causal failure mechanism.','Training response/reward statistics cannot establish task improvement, MRI grounding or independent-patient generalization.'],'limitations':['No development, final-evaluation or sealed-test outputs used.','No raw generations/checkpoints/images reread; synthesis inherits the scope and numerical-verifier limitations of the twelve exact source proofs.','Repeated groups/seeds are not new independent patients; per-task participant counts are not summed across arms.','Matched charged-token caps do not guarantee equal optimizer steps, task tokens, FLOPs or effective training exposure.','No unique policy-collapse, reward, task-mixing or kernel causal diagnosis assigned from these descriptive counts.'],'native_models_launched':0,'source_training_protocol_changed':False,'heldout_performance_claimed':False}
 for filename,e in inputs.items():require(sha(a.proof_root/filename)==e['sha256'],'Input proof changed during aggregation')
 require(not a.output.exists(),'Preserve previous result');a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n');print(json.dumps({'status':result['status'],'proofs':12,'output_sha256':sha(a.output)}))
if __name__=='__main__':main()
