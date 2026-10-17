"""Aggregate only immutable sanitized independent completed-arm proof JSON.

No model/native imports, private raw responses, MRI images, checkpoints or test
content are opened. This secondary summary is not a new Adam or decoder audit.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import mean,pstdev

ROOT=Path(__file__).resolve().parents[1]
TASKS=('mri_sequence','mri_visit_correspondence')
ALLOWED={'mri_sequence':('flair','contrast-enhanced t1'),'mri_visit_correspondence':('different','same')}
WITHHELD='WITHHELD_NONCATEGORICAL_OR_UNPARSED'
REQUIRED_CHECKS=('actual_prompt_native_input_token_alignment','all_active_Adam_continuity_and_changed_weights',
    'complete_official_sampler_and_source_image_binding','final_actual_optimizer_step_LR_RNG','full_atomic_parent_chain_cursor',
    'inactive_optimizer_and_weights_unchanged','independent_ByteLevel_response_token_decoding',
    'task_conditioned_reward_and_sample_std_advantages','unique_occurrence_ids_not_people')

def require(ok,msg):
    if not ok:raise ValueError(msg)
def file_sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def is_hash(x):return isinstance(x,str) and len(x)==64 and all(c in '0123456789abcdef' for c in x)
def integer(x):return type(x) is int and x>=0

def safe_answer_counts(raw,task):
    result={label:0 for label in ALLOWED[task]};result[WITHHELD]=raw.get('withheld_non_categorical_answer_count',0)
    for key,value in raw.items():
        if not key.startswith('answer:'):continue
        require(integer(value),'Answer-frequency count is not a nonnegative integer')
        label=key[7:];result[label if label in ALLOWED[task] else WITHHELD]+=value
    require(all(integer(v) for v in result.values()) and sum(result.values())==raw['responses'],'Categorical/withheld frequencies do not reconcile with responses')
    return result


def allocations(p):
    groups=p['completed_groups'];updates=p['optimizer_updates'];out={}
    for task in TASKS:
        n=p['attempted_groups_by_task'].get(task,0);u=p['accepted_update_groups_by_task'].get(task,0)
        out[task]={'attempted_groups':n,'accepted_updates':u,'attempt_share':n/groups,
            'accepted_update_rate':u/n if n else None,'accepted_update_share':u/updates if updates else None}
    return out


def load_verified_proof(path,spec,shared):
    require(Path(path).is_file() and file_sha(path)==spec['sha256'],'Completed proof digest missing or changed')
    p=load(path)
    require(p.get('schema')==spec['schema'] and p.get('verdict') in {'PASS','PASS_WITH_LIMITATIONS'} and
        p.get('reviewer_kind') in {'isolated_codex','claude_code'} and p.get('reviewer') and
        p.get('evidence_tier')=='REPRODUCED_CPU_FROM_ACTUAL_NATIVE_CHECKPOINTS' and p.get('full_arm_completed') is True,
        'Expected independent completed-native-proof schema/verdict/provenance absent')
    if spec['schema']=='completed-RL-independent-numerical-verification-v1':
        require(p.get('stage_id')==f"rl:{spec['arm']}:{spec['seed']}" and p.get('arm',{}).get('name')==spec['arm'] and
            p.get('seed')==spec['seed'] and p['arm']['reward']=='raw','Generic proof arm/seed/reward differs')
    else:require(spec['schema']=='u-raw7-final-independent-cpu-verification-v1' and spec['arm']=='U_raw' and spec['seed']==7,'Unrecognized special proof')
    ev=p.get('evidence_sha256',{});checks=p.get('checks',{})
    require(all(ev.get(k)==v for k,v in shared.items()) and ev.get('verifier_source')==spec['verifier_source_sha256'],
        'Proof common-start/data/plan/source evidence differs from fixed input plan')
    require(all(checks.get(k) is True for k in REQUIRED_CHECKS) and checks.get('critical_source_hashes')==19 and
        integer(checks.get('all_saved_file_hashes_verified')) and checks['all_saved_file_hashes_verified']>0,'Required independent interface/source checks absent')
    n=p['completed_groups'];u=p['optimizer_updates'];require(integer(n) and n>0 and integer(u) and u<=n and
        p['actual_saved_optimizer_RNG_states_verified']==n+1 and p['native_terminal_status'] in {'TOKEN_BUDGET_EXHAUSTED','SCHEDULE_COMPLETE'},'Invalid completed cursor/update/native terminal')
    require(sum(p['attempted_groups_by_task'].values())==n and sum(p['accepted_update_groups_by_task'].values())==u and
        set(p['attempted_groups_by_task'])<=set(TASKS) and set(p['accepted_update_groups_by_task'])<=set(TASKS),'Task attempt/update totals differ')
    require(p['cap']==169216 and 0<=p['used_tokens_recomputed']<=169216 and p['unspent_tokens']==169216-p['used_tokens_recomputed'] and
        p['completed_rollout_tokens']+p['measured_crashed_work_tokens']+p['imputed_unknown_tokens']==p['used_tokens_recomputed'],'Completed/crashed/imputed token charges differ')
    require(len(ev.get('receipt_chain',[]))==n+1 and ev['receipt_chain'][-1]==p['terminal_boundary_receipt_sha256'] and
        all(is_hash(x) for x in ev['receipt_chain']) and all(is_hash(ev.get(k)) for k in ('exit_witness','native_stdout','release_approval','verifier_source')),'Incomplete terminal/source evidence hash inventory')
    require(checks['independent_reward_response_recomputations']==4*n,'Native reward/response verification count differs')
    total=0
    for task,attempts in p['attempted_groups_by_task'].items():
        require(integer(attempts) and 0<=p['accepted_update_groups_by_task'].get(task,0)<=attempts,'Invalid task update rate')
        raw=p['raw_response_counts_and_correctness_by_task'][task];require(raw['responses']==4*attempts and
            0<=raw['full_format_correct']<=raw['native_correct']<=raw['responses'] and 0<=raw['full_format']<=raw['responses'] and
            0<=raw['invalid']<=raw['responses'],'Response correctness/format denominators differ')
        safe_answer_counts(raw,task)
        zero=p['variance_diagnostics'].get(task+':zero_total_variance',0);varies=p['variance_diagnostics'].get(task+':task_varies',0)
        require(integer(zero) and zero<=attempts-p['accepted_update_groups_by_task'].get(task,0) and integer(varies) and
            p['accepted_update_groups_by_task'].get(task,0)<=varies<=attempts,'Variance/update group counts differ')
        L=213 if task=='mri_sequence' else 405;inp=p['token_decomposition'][task+':input'];gen=p['token_decomposition'][task+':generated']
        require(inp==4*attempts*L and integer(gen),'Measured task input/generated exposure differs')
        total+=inp+gen
    require(total==p['completed_rollout_tokens'],'Task token decomposition differs from completed rollout exposure')
    return p


def summarize_proof(p,spec):
    tasks=allocations(p);den=p['completed_rollout_tokens']
    for task,out in tasks.items():
        n=out['attempted_groups'];raw=p['raw_response_counts_and_correctness_by_task'].get(task)
        inp=p['token_decomposition'].get(task+':input',0);gen=p['token_decomposition'].get(task+':generated',0)
        zero=p['variance_diagnostics'].get(task+':zero_total_variance',0)
        out.update({'input_tokens':inp,'generated_tokens':gen,'total_completed_rollout_tokens':inp+gen,
            'completed_rollout_token_share':(inp+gen)/den,'zero_total_reward_variance_groups':zero,
            'nonzero_total_variance_groups':n-zero,'nonzero_total_variance_groups_without_update':n-zero-out['accepted_updates'],
            'groups_with_task_reward_variation':p['variance_diagnostics'].get(task+':task_varies',0),
            'unique_source_rows_attempted':p['unique_source_rows_by_task'].get(task,0),
            'source_people_attempted':p['distinct_source_people_by_task'].get(task,0)})
        if raw:
            out.update({'rollouts':raw['responses'],'correct_training_rollouts':raw['native_correct'],
                'training_rollout_correct_fraction':raw['native_correct']/raw['responses'],
                'full_format_rollouts':raw['full_format'],'invalid_answer_rollouts':raw['invalid'],
                'categorical_answer_counts':safe_answer_counts(raw,task),
                'categorical_answer_fractions':{k:v/raw['responses'] for k,v in safe_answer_counts(raw,task).items()}})
        else:out.update({'rollouts':0,'correct_training_rollouts':0,'training_rollout_correct_fraction':None,'full_format_rollouts':0,
            'invalid_answer_rollouts':0,'categorical_answer_counts':{k:0 for k in (*ALLOWED[task],WITHHELD)},'categorical_answer_fractions':None})
    return {'arm':spec['arm'],'seed':spec['seed'],'completed_groups':p['completed_groups'],'optimizer_updates':p['optimizer_updates'],
        'cap':169216,'used_tokens':p['used_tokens_recomputed'],'completed_rollout_tokens':den,
        'measured_crashed_work_tokens':p['measured_crashed_work_tokens'],'imputed_unknown_tokens':p['imputed_unknown_tokens'],
        'tasks':tasks,'source_proof_sha256':spec['sha256'],'source_verifier_sha256':spec['verifier_source_sha256'],
        'source_reviewer':p['reviewer'],'source_reviewer_kind':p['reviewer_kind']}


def seed_variation(rows):
    grouped=defaultdict(list)
    for row in rows:grouped[row['arm']].append(row)
    result={}
    for arm,group in grouped.items():
        group=sorted(group,key=lambda r:r['seed']);values=[r['optimizer_updates'] for r in group]
        result[arm]={'completed_seeds':[r['seed'] for r in group],'seed_count':len(group),'seed_rows':group,
            'optimizer_updates':{'mean':mean(values),'minimum':min(values),'maximum':max(values),'population_sd_descriptive':pstdev(values)},
            'accuracy_or_causal_improvement_claimed':False,'interpretation':'Observed seed realizations only; no inferential variance estimate, held-out accuracy, causal task-balance or optimization-effect claim.'}
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--inputs',type=Path,required=True);ap.add_argument('--expected-inputs-sha256',required=True)
    ap.add_argument('--output',type=Path,default=ROOT/'results/COMPLETED_RAW9_TRAINING_SIGNAL_COMPARISON.json');a=ap.parse_args()
    require(file_sha(a.inputs)==a.expected_inputs_sha256,'Externally supplied fixed input plan digest differs');plan=load(a.inputs)
    require(plan['schema']=='completed-raw9-signal-inputs-v1' and len(plan['proofs'])==9 and len({(s['arm'],s['seed']) for s in plan['proofs']})==9,'Exact nine-arm snapshot required')
    rows=[]
    for spec in plan['proofs']:
        path=ROOT/spec['path'];require(path.resolve().is_relative_to(ROOT/'results'),'Only sanitized public result proof paths permitted')
        rows.append(summarize_proof(load_verified_proof(path,spec,plan['shared_evidence_sha256']),spec))
    result={'schema':'completed-raw9-training-signal-comparison-v1','status':'AGGREGATED_EXISTING_INDEPENDENT_SANITIZED_COMPLETED_PROOFS',
        'author':'Codex /root/likelihood_finish_reviewer','new_aggregate_independent_review':'Coordinator review recorded separately in the root-cause report; script execution is not independent approval. Source proofs retain their actual reviewer identities.',
        'evidence_tier':'SECONDARY_CPU_AGGREGATION_OF_VERIFIED_NATIVE_SUMMARIES','research_tag':'PRIMARY_VERO_TRANSFER','completed_RL_arms_in_snapshot':9,
        'input_plan_sha256':a.expected_inputs_sha256,'shared_common_start_evidence':plan['shared_evidence_sha256'],'arm_seed_rows':rows,'within_arm_seed_variation':seed_variation(rows),
        'metric_definitions':{'attempted':'Complete saved rollout groups, including zero-variance and rejected groups','accepted_update':'Actual verified optimizer updates from source proof, not task sampling allocations',
            'update_rate':'Accepted updates / attempted groups within task','attempt_share':'Task attempted groups / all attempted groups','accepted_update_share':'Task accepted updates / all accepted updates; null if no updates',
            'tokens':'Measured native input (including visual positions) plus generated tokens; crashed measured and imputed charges separate, not optimizer FLOPs',
            'nonzero_variance_without_update':'Residual count only; public proof does not identify rejection reason or true-label-conditioned contributions',
            'categorical_answers':'Exact allowed task strings only; every other answer key and sanitized withheld count grouped without exporting text'},
        'observations_not_causes':['Equal169216-token caps do not imply equal accepted optimizer updates, number of complete groups, sampled labels, response diversity or effective task exposure.',
            'Uniform task sampling can yield different accepted-update task shares across seeds because the realized response/group filter and evolving policy differ.',
            'A low update count or constant categorical answers is not proof of weak or strong MRI visual grounding, GSPO ineffectiveness, a code defect or held-out capability.'],
        'minimum_justified_V2_diagnostic':{'status':'PROPOSED_NOT_RUN_SEPARATE_FROM_ORIGINAL18','question':'How much active-group acceptance variation persists when prompt/image/true-label case mix and initial weights are held fixed?',
            'minimal_design':'After frozen original arms and existing visual probe, consider a separately reviewed no-update common-B28 diagnostic on a prospectively hash-selected fixed8 training prompts (two per task×binary-label cell), seeds7/11/13,K4,temp1,max256;96 responses. Report within-prompt acceptance/format/label distributions, not evaluation accuracy.',
            'necessity_limit':'First use the completed frozen seed matrix and already planned image controls. This proposal is not automatically authorized, launched or required to keep the original queue running.',
            'interpretation_and_KILL':'This three-seed/eight-prompt pilot is descriptive and cannot exclude sampling noise when counts look stable. Do not promote it to a corrective training run or claim a causal root cause from its acceptance counts. Require the complete frozen seed comparison and image controls to identify a reproducible failure before selecting a separately reviewed V2 intervention.',
            'original18_protocol_changes':False},
        'limitations':['Nine completed raw RL summaries are a time-frozen snapshot, not a full18-arm or21-training-stage completion.',
            'Training correctness and categorical frequencies are stochastic on-policy exposure, not held-out accuracy, calibrated probabilities, independent patients or improved MRI understanding.',
            'Multiple seed outputs are not additional patients. No p values or population generalization inference.',
            'Source proof verifier versions include independently corrected decoder/public-sanitization evidence; pinned proof hashes preserve provenance. No optimizer/decoder/native audit is repeated here.',
            'True-label-conditioned update mechanisms, gradient influence and causal learning explanations are not identifiable from these sanitized task-level summaries.'],
        'native_models_launched':0,'private_raw_or_images_read':False,'optimizer_arrays_reaudited':False,'sealed_test_used':False,'training_protocol_changed':False,'source_sha256':file_sha(__file__)}
    require(a.output.resolve().is_relative_to(ROOT/'results'),'Safe aggregate output must stay in results')
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({'output_sha256':file_sha(a.output),'completed_arms':len(rows),'updates':{f"{r['arm']}:{r['seed']}":r['optimizer_updates'] for r in rows},'native_models_launched':0}))

if __name__=='__main__':main()
