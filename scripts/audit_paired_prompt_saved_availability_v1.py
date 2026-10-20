#!/usr/bin/env python3
"""Bounded CPU metadata audit; never imports model/runtime or opens image/state bytes.

No exhaustive filesystem search/absence claim. --self-test runs synthetic contract
fixtures only. Default audit pins the enumerated historical TRAIN evidence and
writes a new, non-executable prospective input selection, not native permission.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import copy
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sys

W=Path(__file__).resolve().parents[1]
PINS={
'original76':'455efc650444f2820f26832bc29b042ce2ec3babc0c9fe745491705a3fdf7e2c',
'original18_audit':'0bb560f50ec4237fe097fb3d2842c8e5a386eddd871c6e0de21118e5390656ba',
'original18_review':'d80c8c53e5e6130169847ba15a870cfaca6a4bfc6f93f9a63171ea39bf537a03',
'candidate_binding':'527196310047649f2439634a9f79564edf6e2cd147b90d815c2b88b3d0d487f9',
'sequence56':'c257ef5481c0e97a907fdc75bfcba6345744bd3769b3e6391f0563147a17a770',
'membership':'bf92f7e560ce3ed3b3629427f590d8c6f6cf4c4c3bae6ab04dc14af42ad4ce99',
'candidate_independent_review':'1466f1c8715bd1cd85d38baa3d4ab9df3edb704bcfbe65f1be12bb16753a1322',
'original_image_inventory':'fb78af69995bae9d6191fb5c6143c14b894710b54f0f878acf727fd599c4df7c',
'measurements':'0839e1a9c10e3a7d2b7d45d8f7bceeda46cdcb94146adb3b2233abc82ebec5e9',
'pool_receipt':'2279dab26b1fb8fa36f2a8499b3582197fad6d1300d5e50ee83563cab1f7b19b',
'f32_scope':'865aeda5a772c9a917370bacddfd84309ac51ac225cd52732e0ee0e2f56f3511',
'f32_complete':'ea8c354b99f17409e8c2fdeaa9f572f2188995159da02d26302c62029387d69d',
'f32_sequence_rollout':'ccf56ed1e027293c31fc7b59e3af9dd4eea0b1d2ec4300aa9d562bb142b355c4',
'f32_Dallas_rollout':'adb50f5184950d653fdd3b921a967ea0eaf640ec8e5d22bdd4d68d648a90eb81',
'f32_actual_review':'b61f9966b591082416e2da3a3b4b39f9ef1d26b841781bf01fec831c083af673',
'canonical_probe_complete':'252d44fb879ef8926b4dda301168265be6702641d37c44554fe5f306a6f1a911',
'canonical_probe_review':'51f8c7eca305f00f364313b29e358ff6592cd752e49492aa574f9f6a82ebfd25',
'parent_runtime_spec':'ddd53814a73ac13f5e819f9f3835f579f5bed712d352cf5f4c7b5680028b8993',
'parent_receipt':'3c2567723e7bc90a65464c8c3645f49ad91c38a0ad886961c508bc1f8a0e3879',
'margin_probe_actual':'41b3dc15b1484734558ce15133e6bdf0406c8aca6a90385574c1e23cb226e101',
'margin_probe_review':'daf7829404381f02e06e51754eb7b50cb3b2d459fa17138f48887419e2dba087'}
SEQ='mri_sequence';DALLAS='mri_visit_correspondence';LABELS=['FLAIR','contrast-enhanced T1']


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(obj,*,ensure_ascii=False):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=ensure_ascii,allow_nan=False).encode()).hexdigest()


def prompt_contract(row):
    """Preserve text/roles/image slots;image path is data routing,not prompt text."""
    prompt=copy.deepcopy(row['prompt'])
    for message in prompt:
        require(isinstance(message['content'],list),'Expected structured native image prompt')
        for part in message['content']:
            if part.get('type')=='image':part['image']='<IMAGE_SOURCE_ROUTING>'
    return prompt


def eligible_blocks(groups):
    """Metadata necessary conditions only;never reconstruct responses or advantages."""
    by=defaultdict(list)
    for g in groups:
        if g['split']=='train' and g['level']==45 and g['real_onpolicy'] and g['responses']==4:
            by[(g['patient'],g['parent_binding'],g['policy'],g['prompt_sha'],g['token_contract'])].append(g)
    out=[]
    for key,items in by.items():
        f=[g for g in items if g['modality']=='FLAIR'];t=[g for g in items if g['modality']=='contrast-enhanced T1']
        if len(f)==len(t)==1 and f[0]['image_sha']!=t[0]['image_sha']:out.append((key,f[0],t[0]))
    return out


def prospective_semantics_v2(plan):
    """Offline replay identity is distinct from future on-policy trajectories."""
    plan=copy.deepcopy(plan);plan['schema']='paired-prompt-prospective-fixed-TRAIN-inputs-v2'
    plan.pop('within_sequence_pair_same_checkpoint_before_any_update_required',None)
    plan['offline_replay_all8_samecheckpoint_required']=True
    plan['future_training_each_K4_one_unchanged_policy_required']=True
    plan['runner_update_placement']='A:F4->optional authorized update->T4->optional authorized update; B:F2T2->optional authorized update->F2T2->optional authorized update. Each K4 collected under its own unchanged current policy;later group responses follow the resulting intrinsic trajectory,not one frozen8-response block.'
    plan['future_arms_same_realized_responses_required']=False
    return plan


def self_test():
    a={'split':'train','level':45,'real_onpolicy':True,'responses':4,'patient':'fixture','parent_binding':'checkpoint',
       'policy':'F32','prompt_sha':'prompt','token_contract':'contract','modality':'FLAIR','image_sha':'imageF'}
    b={**a,'modality':'contrast-enhanced T1','image_sha':'imageT'}
    assert len(eligible_blocks([a,b]))==1
    changes=[('parent_binding','other'),('policy','BF16'),('prompt_sha','other'),('token_contract','other'),('responses',3),
             ('level',65),('split','dev'),('patient','other'),('real_onpolicy',False),('image_sha','imageF')]
    for key,value in changes:assert eligible_blocks([a,{**b,key:value}])==[]
    assert eligible_blocks([a,a])==[] and eligible_blocks([a,b,b])==[]
    # Historical scope serialization uses ASCII escaping;row manifests use UTF8.
    assert canonical({'x':'\u2264'},ensure_ascii=True)!=canonical({'x':'\u2264'})
    prompt=lambda image,text:{'prompt':[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':text}]}]}
    assert prompt_contract(prompt('pathF','question'))==prompt_contract(prompt('pathT','question'))
    assert prompt_contract(prompt('pathF','question'))!=prompt_contract(prompt('pathT','differentquestion'))
    fixture={'schema':'v1','rows_PRIVATE':[{'fixture':'same'}],'within_sequence_pair_same_checkpoint_before_any_update_required':True}
    updated=prospective_semantics_v2(fixture)
    assert updated['offline_replay_all8_samecheckpoint_required'] and updated['future_training_each_K4_one_unchanged_policy_required']
    assert updated['rows_PRIVATE']==fixture['rows_PRIVATE'] and not updated['future_arms_same_realized_responses_required'] and 'within_sequence_pair_same_checkpoint_before_any_update_required' not in updated
    return 18


def sources(R):
    D=R/'.worktrees/vero-mri-paired-z045/local_data';data=D/'paired_z045_data_v1/candidate_v2';q=D/'paired_z045_f32_qualification_v1'
    P=R/'.worktrees/vero-mri-paired-signal/local_data/paired_signal_candidate_probe_v1'
    return {'original76':R/'local_data/c_two_task_pool_v1/train.verl.jsonl',
        'original18_audit':W/'results/COMPLETED_RL18_LENGTH_CONTRACT_ACTUAL_CPU_AUDIT_V1.json',
        'original18_review':W/'results/COMPLETED_RL18_INDEPENDENT_AGGREGATE_REVIEW_V1.json',
        'candidate_binding':data/'binding_PRIVATE.json','sequence56':data/'paired_z045_sequence56_PRIVATE.verl.jsonl',
        'membership':data/'source_and_pair_membership_PRIVATE.json','candidate_independent_review':W/'results/PAIRED_Z045_DATA_INDEPENDENT_REVIEW_V1.json',
        'original_image_inventory':R/'local_data/c_two_task_pool_v1/row_inventory_PRIVATE.json',
        'measurements':R/'local_data/c_two_task_pool_v1/native_tokens_v2/measurements_PRIVATE.json',
        'pool_receipt':W/'results/C_TWO_TASK_POOL_AND_NATIVE_TOKEN_MEASUREMENTS.json',
        'f32_scope':q/'scope_IMAGE_BOUND_FINAL_PRIVATE.json','f32_complete':q/'actual_once_v1_PRIVATE/complete_PRIVATE.json',
        'f32_sequence_rollout':q/'actual_once_v1_PRIVATE/group_0_PRIVATE/rollout_PRIVATE.json',
        'f32_Dallas_rollout':q/'actual_once_v1_PRIVATE/group_1_PRIVATE/rollout_PRIVATE.json',
        'f32_actual_review':W/'results/PAIRED_Z045_F32_QUALIFICATION_ACTUAL_INDEPENDENT_REVIEW_V1.json',
        'canonical_probe_complete':P/'run_v4_PRIVATE/complete_PRIVATE.json',
        'canonical_probe_review':W/'results/PAIRED_SIGNAL_CANONICAL_CANDIDATE_ACTUAL_INDEPENDENT_REVIEW_V1.json',
        'parent_runtime_spec':D/'paired_z045_continuation_v4/spec_PREPARATION_ONLY_PRIVATE.json',
        'parent_receipt':R/'.worktrees/mri-adaptive-accuracy-gate/local_data/uniform_balance_continuation_v1/run/boundary_0008/receipt.json',
        'margin_probe_actual':W/'results/CLASS_MARGIN_V2_ADAM60_ACTUAL_DIRECTIONS_V1.json',
        'margin_probe_review':W/'results/CLASS_MARGIN_V2_ACTUAL_INDEPENDENT_REVIEW_V1.json'}


def audit(R):
    paths=sources(R);ledger={};d={}
    for name,path in paths.items():
        payload=path.read_bytes();digest=hashlib.sha256(payload).hexdigest();require(digest==PINS[name],'Pinned source bytes differ: '+name)
        ledger[name]={'path_PRIVATE':str(path),'bytes':len(payload),'sha256':digest}
        d[name]=[json.loads(line) for line in payload.decode().splitlines() if line] if name in ['original76','sequence56'] else json.loads(payload)
    old=d['original76'];new=d['sequence56'];oldseq=[r for r in old if r['ability']==SEQ];oldDallas=[r for r in old if r['ability']==DALLAS]
    require(len(old)==76 and len(oldseq)==28 and len(oldDallas)==48 and len(new)==56,'TRAIN source coverage differs')
    require(all(r['extra_info']['internal_split']=='train' for r in old+new),'NonTRAIN row')
    require(len({r['id'] for r in old})==76 and len({r['id'] for r in new})==56,'Duplicate IDs')
    people=defaultdict(list)
    for r in oldseq:people[r['extra_info']['patient_key']].append(r['reward_model']['ground_truth'])
    require(len(people)==28 and all(len(x)==1 for x in people.values()),'Historical one-modality/person premise differs')
    require(d['original18_audit']['train_manifest_sha256']==PINS['original76'] and d['original18_audit']['original_arms']==18,'Original18 source manifest binding differs')
    require(d['original18_audit']['completed_groups']==2286 and d['original18_audit']['task_totals']['mri_sequence']['groups']==1139,'Original18 count differs')
    require(d['original18_review']['actual_completed_original_RL_proofs']==18 and d['original18_review']['task_totals']['mri_sequence']['attempted_groups']==1139,'Independent18 proof differs')
    for name,key in [('sequence56','paired_z045_sequence56_PRIVATE.verl.jsonl'),('membership','source_and_pair_membership_PRIVATE.json')]:
        require(d['candidate_binding']['artifact_sha256'][key]==PINS[name],'Reviewed candidate artifact chain differs')
    pairs=defaultdict(dict);members={x['original_record_id_PRIVATE']:x for x in d['membership']}
    for r in new:
        label=r['reward_model']['ground_truth'];patient=r['extra_info']['patient_key'];source=r['paired_z045_data_v1']['source_record_id_PRIVATE']
        require(label in LABELS and label not in pairs[patient] and r['paired_z045_data_v1']['level']==45,'Candidate pair ambiguous')
        m=members[source];require(m['patient_PRIVATE']==patient and m['level']==45 and m['size']==[364,436] and m['pixel_check_PRIVATE']['pixel_exact'],'Existing source/pixel qualification differs')
        require(m['modality_PRIVATE']==('flair' if label==LABELS[0] else 't1ce'),'Candidate source modality mismatch')
        pairs[patient][label]=(r,m)
    require(set(pairs)==set(people) and all(set(v)==set(LABELS) for v in pairs.values()),'Reviewed56 pair membership differs')
    scope=d['f32_scope'];complete=d['f32_complete'];review=d['f32_actual_review'];groups=[];group_private=[]
    require(complete['scope_sha256']==canonical(scope,ensure_ascii=True)==review['bindings']['scope_canonical_sha256'],'F32 actual canonical scope mismatch')
    require(review['bindings']['native_complete_sha256']==PINS['f32_complete'] and review['actual']['responses']==8 and complete['optimizer_updates']==0,'F32 saved actual binding differs')
    for index,name in enumerate(['f32_sequence_rollout','f32_Dallas_rollout']):
        raw=d[name];row=scope['rows_PRIVATE'][index];responses=raw['responses'];measure=complete['groups'][index]
        require(raw['record_id']==row['id'] and len(responses)==len(measure['grades'])==4,'Actual source/K4/grade membership differs')
        require(raw['numerical_policy']==scope['policy']=='F32_ACTIVATION_GENERATION_SCORER_GSPO_V1','Actual numerical policy differs')
        require(hashlib.sha256(raw['rendered_prompt'].encode()).hexdigest()==raw['prompt_sha256'],'Actual prompt SHA differs')
        require(raw['image_sha256']==scope['expected_image_sha256'][row['id']],'Actual image receipt differs')
        require(all(len(x['token_ids'])==len(x['raw_selected_logps'])>0 for x in responses),'Per-response own token/logp mismatch')
        input_ids=raw['prompt_token_ids'];P=len(input_ids[0]);require(len(input_ids)==1 and P==(213 if index==0 else 405),'Actual measured prompt length differs')
        if index==0:
            g={'split':'train','level':row['paired_z045_data_v1']['level'],'real_onpolicy':True,'responses':4,
                'patient':row['extra_info']['patient_key'],'parent_binding':scope['parent']['receipt_sha256'],'policy':raw['numerical_policy'],
                'prompt_sha':raw['prompt_sha256'],'token_contract':d['parent_runtime_spec']['file_sha256']['contract'],
                'modality':row['reward_model']['ground_truth'],'image_sha':raw['image_sha256']};groups.append(g)
        group_private.append({'task':row['ability'],'source_record_id_PRIVATE':row['id'],'rollout_sha256':PINS[name],
            'prompt_sha256':raw['prompt_sha256'],'image_sha256':raw['image_sha256'],'input_tokens':P,'responses':4,
            'returned_tokens':sum(len(x['token_ids']) for x in responses),'own_response_receipts_sha256':[canonical(x) for x in responses],
            'own_grade_receipts_sha256':[canonical(x) for x in measure['grades']]})
    eligible=eligible_blocks(groups);require(not eligible,'Previously unavailable real matched block appeared;review scope before use')
    require(d['canonical_probe_complete']['generation_calls']==d['canonical_probe_complete']['optimizer_updates']==0,'Canonical probe gained generation/updates')
    require(d['margin_probe_actual']['generation_calls']==d['margin_probe_review']['generation_calls']==0,'Margin probe gained generation')
    # Selection has no dependency on rollout answers, scores, grades, losses or margins.
    patient=sorted(pairs)[0];selected=[pairs[patient][label] for label in LABELS]
    require(prompt_contract(selected[0][0])==prompt_contract(selected[1][0]) and selected[0][0]['extra_info']['question']==selected[1][0]['extra_info']['question'],'Pair text/image-slot prompts differ')
    dallas=[next(r for r in oldDallas if r['reward_model']['ground_truth']==label) for label in ['same','different']]
    cost={x['record_sha256']:x for x in d['measurements']['rows']};require(d['measurements']['manifest_sha256']==PINS['original76'] and len(cost)==76,'Historical measured source differs')
    oldmatch=next(r for r in oldseq if r['extra_info']['patient_key']==patient);seqcost=cost[canonical(oldmatch)]
    require(seqcost['input_tokens']==213 and seqcost['visual_tokens']==154,'Historical sequence cost differs')
    selected_private=[]
    for row,m in selected:
        selected_private.append({'role':'SEQUENCE_F' if row['reward_model']['ground_truth']==LABELS[0] else 'SEQUENCE_T',
            'record_PRIVATE':row,'record_sha256':canonical(row),'expected_image_sha256':m['PNG_sha256'],
            'expected_image_hash_provenance':'Pinned reviewed56 source_and_pair_membership;no image bytes read in this audit',
            'source_membership_receipt_sha256':canonical(m),'historical_expected_input_tokens':213,'historical_expected_visual_tokens':154,
            'input_cost_status':'SHAPE_AND_IDENTICAL_PROMPT_DERIVED_FROM_EXISTING_NATIVE_MEASUREMENT;NEW_INPUT_REMEASUREMENT_REQUIRED',
            'historical_measurement_row_sha256':canonical(oldmatch),'prompt_contract_sha256':canonical(prompt_contract(row)),
            'prompt_same_as_historically_measured_row':prompt_contract(row)==prompt_contract(oldmatch)})
    for row in dallas:
        c=cost[canonical(row)];require(c['input_tokens']==405 and c['visual_tokens']==288,'Dallas measured cost differs')
        selected_private.append({'role':'DALLAS_'+row['reward_model']['ground_truth'].upper(),'record_PRIVATE':row,'record_sha256':canonical(row),
            'expected_image_sha256':d['original_image_inventory']['image_sha256_by_record'][row['id']],
            'expected_image_hash_provenance':'Pinned original76 row_inventory;no image bytes read in this audit',
            'historical_expected_input_tokens':405,'historical_expected_visual_tokens':288,'input_cost_status':'EXACT_ORIGINAL_ROW_HISTORICAL_NATIVE_MEASURED;CURRENT_REMEASUREMENT_REQUIRED'})
    require(all(x.get('prompt_same_as_historically_measured_row',True) for x in selected_private),'Prospective sequence prompt differs from measured contract')
    runtime=d['parent_runtime_spec'];parent=d['parent_receipt'];p=scope['parent']
    require(runtime['parent_receipt_sha256']==PINS['parent_receipt']==p['receipt_sha256'] and parent['absolute_optimizer_updates']==p['updates']==60 and parent['absolute_completed_groups']==p['groups']==129,'Exact Adam60 parent metadata differs')
    require(parent['files_sha256']['adapter.safetensors']==p['adapter_sha256'] and parent['files_sha256']['optimizer/state.safetensors']==p['optimizer_state_sha256'],'Parent artifact hash receipts disagree')
    B={'groups_per_arm':4,'responses_per_group':4,'responses_per_arm':16,'sequence_groups':2,'Dallas_groups':2,'maximum_updates_all_branches':4,
       'historical_sequence_input_tokens':213,'historical_Dallas_input_tokens':405,'prompt_positions_per_arm':4*(2*213+2*405),
       'returned_response_token_cap_per_arm':16*256,'charged_cap_per_arm':4*(2*213+2*405)+16*256,
       'two_arm_charged_cap':2*(4*(2*213+2*405)+16*256),'planned_attempts':4,'minimum_updates_required':0,
       'cap_includes_only_prompt_plus_returned_reservation':True,'internal_TF_backward_lookahead_cost':'NOT_INCLUDED_MUST_BE_SEPARATELY_FROZEN',
       'current_input_lengths':'NOT_REMEASURED;future_exact_runner_must_measure_and_STOP_if_exceeds_bound'}
    require(B['prompt_positions_per_arm']==4944 and B['charged_cap_per_arm']==9040,'Prospective arithmetic error')
    plan={'schema':'paired-prompt-prospective-fixed-TRAIN-inputs-v1','status':'METADATA_SELECTION_FROZEN_NATIVE_NOT_RUN','selection_UTC':datetime.now(timezone.utc).isoformat(),
       'selection_rule':'Lexicographically first patient_key among pinned reviewed56 TRAIN pairs;FLAIR thenT1. First original76 source-order Dallas same then different. No outputs/scores used in selection.',
       'rows_PRIVATE':selected_private,'same_inputs_both_arms':True,'parent':{'checkpoint_path_PRIVATE':runtime['parent_checkpoint'],
         'receipt_sha256':PINS['parent_receipt'],'adapter_sha256_from_receipt':p['adapter_sha256'],'optimizer_state_sha256_from_receipt':p['optimizer_state_sha256'],
         'optimizer_manifest_sha256_from_receipt':p['optimizer_manifest_sha256'],'Adam_step':60,'absolute_groups':129,'payload_bytes_read_or_verified_here':False},
       'budget':B,'within_sequence_pair_same_checkpoint_before_any_update_required':True,
       'runner_update_placement':'NOT_SPECIFIED_BY_THIS_INPUT_PLAN;exact runner must preserve same checkpoint for all8pair responses',
       'source_sha256':PINS,'native_approved':False,'training_authorized':False,'model_calls':0}
    plan=prospective_semantics_v2(plan)
    details={'schema':'paired-prompt-saved-availability-private-v1','status':'NOT_AVAILABLE_IN_ENUMERATED_AUTHENTICATED_SOURCES',
       'read_ledger_PRIVATE':ledger,'actual_F32_group_metadata_PRIVATE':group_private,'sequence_group_matching_fields_PRIVATE':groups,
       'eligible_eight_response_blocks':len(eligible),'raw_response_tokens_text_logits_or_arrays_exported':False,'prospective_plan':plan}
    safe={'schema':'paired-prompt-saved-matched-availability-v1','status':'NOT_AVAILABLE_IN_ENUMERATED_AUTHENTICATED_SOURCES','author':'Codex /root/rc_data',
      'evidence_tier':'OBSERVED_SAVED_TRAIN_METADATA_AND_SOURCE_MEMBERSHIP','script_sha256':sha(__file__),'source_proof_sha256':PINS,
      'coverage':{'original18_completed_runs':18,'original18_saved_completed_groups':2286,'original18_sequence_groups':1139,'original18_sequence_responses':4556,
        'original_pool_sequence_rows':28,'original_pool_sequence_people':28,'modalities_per_original_sequence_person':1,'original_pool_same_patient_dual_modality_pairs':0,
        'reviewed_candidate_sequence_rows':56,'reviewed_candidate_sequence_people':28,'reviewed_candidate_pairs':28,
        'F32_actual_saved_groups':2,'F32_sequence_groups':1,'F32_Dallas_groups':1,'F32_responses_per_group':4,'F32_total_saved_responses':8,
        'F32_saved_sequence_opposite_modality_group':0,'eligible_matched_eight_response_blocks':0,
        'later_canonical_probe_generation_calls':0,'later_margin_probe_generation_calls':0},
      'precise_missing_evidence':'No matching opposite-modality K4 real on-policy group for the single authenticated F32 TRAIN sequence context. Historical original18 source pool has no same-person opposite-modality image at all.',
      'replay_allowed':False,'cross_checkpoint_or_F32_BF16_or_prompt_splicing_permitted':False,
      'prospective_inputs':{'status':'METADATA_ONLY_FROZEN_NOT_NATIVE_READY','sequence_pairs':1,'sequence_images':2,'Dallas_same':1,'Dallas_different':1,
        'selection_depends_on_outputs':False,'same_inputs_both_arms':True,'all_rows_train':True,'expected_image_hashes_from_prior_pinned_receipts':True,
        'current_input_length_measurement':'NOT_RUN;historical213/405bounds require actual remeasurement','budget':B},
      'CPU_contract_fixtures':self_test(),'new_model_loads':0,'new_generation':0,'new_backward':0,'optimizer_updates':0,'image_or_weight_or_optimizer_payload_reads':0,
      'mission_alignment':{'tag':'ENGINEERING_UNBLOCK','link':'Determine whether an exact paired-group Vero-style TRAIN contrast can reuse authenticated real rollouts before any new scope is proposed.',
         'falsifier':'A source-bound same-checkpoint/same-policy/same-prompt F/TK4 block would refute this scoped availability result.'},
      'limitations':['This is a bounded audit of explicitly pinned source families, not an exhaustive filesystem or universal absence claim.',
         'Original18 coverage is inherited from saved independent aggregate/length audit and source manifest;their2286rollout bodies were not reread.',
         'ActualF32 rollout JSON bytes were read only to bind source,own token/logp/grade receipt identities and count responses;no vocabulary arrays or native state payloads opened.',
         'All result identities,record/image paths,hidden targets and actual response receipt identities remain PRIVATE;no held-out or final outcomes accessed.',
         'Canonical score/VJP evaluations and supervised target records are not stochastic on-policy responses and cannot replace the missing group.',
         'Future input plan is PROPOSED only;does not qualify active F32GSPO backward,native cost,state transactional continuity or training.',
         '9040charges prompt+returned-token reservation only;TF/autograd/lookahead/internal compute require separate exact scope.',
         'No clinical,independent-patient generalization,accuracy or causal learning claim.']}
    return safe,details,plan


def amend_semantics_v2():
    """Preserve selection/inputs exactly;read only our frozen v1 audit outputs."""
    private=W/'local_data/paired_prompt_saved_availability_v1';snapshot=private/'v1_frozen_snapshot_PRIVATE'
    expected={'prospective_fixed_inputs_PRIVATE.json':'5fe98288a08ab6e6c620e7418603206ac8b486acf64074793a22500b84ed91de',
      'audit_PRIVATE.json':'51f5e1eb4f8925d13c7b9aa7ecfc60bac62dd56f6d07b72fabcbea6360395a59',
      'PAIRED_PROMPT_SAVED_MATCHED_AVAILABILITY_V1.json':'af769aa74ebf83c4002c8bcfd18ff65ecf9c9166b6138c6014e416c9a52563da',
      'audit_paired_prompt_saved_availability_v1.py':'4de98e4f0ab7de20104a9052e6491753c790ac2eb699f8684d607403ac0c1da3'}
    require(all(sha(snapshot/name)==digest for name,digest in expected.items()),'Preserved v1 snapshot changed')
    oldplan=json.loads((snapshot/'prospective_fixed_inputs_PRIVATE.json').read_text());plan=prospective_semantics_v2(oldplan)
    for key in ['rows_PRIVATE','selection_UTC','parent','budget','source_sha256']:require(plan[key]==oldplan[key],'Semantics amendment changed frozen inputs/parent/cap')
    detail=json.loads((snapshot/'audit_PRIVATE.json').read_text());safe=json.loads((snapshot/'PAIRED_PROMPT_SAVED_MATCHED_AVAILABILITY_V1.json').read_text())
    amendment={'UTC':datetime.now(timezone.utc).isoformat(),'reason':'Same-checkpointall8 is the exactoffline-replay condition;futureA/B uses per-K4 on-policy collection and per-group update trajectories.',
      'data_reselection':False,'new_source_payload_reads':0,'v1_snapshot_sha256':expected,'current_script_sha256':sha(__file__)}
    plan['semantics_amendment']=amendment;detail['schema']='paired-prompt-saved-availability-private-v2';detail['prospective_plan']=plan;detail['semantics_amendment']=amendment
    safe['schema']='paired-prompt-saved-matched-availability-v2';safe['script_sha256']=sha(__file__);safe['CPU_contract_fixtures']=self_test()
    safe['semantics_amendment']={k:v for k,v in amendment.items() if k!='v1_snapshot_sha256'}
    safe['prospective_inputs'].update({'offline_replay_all8_samecheckpoint_required':True,'future_training_each_K4_one_unchanged_policy_required':True,
        'future_arms_same_realized_responses_required':False,'future_update_placement':'A:F4->update->T4->update;B:F2T2->update->F2T2->update;actual updates conditional on future authorized gates,not mandatory.',
        'same_parent_equal_input_exposure_different_intrinsic_policy_trajectories':True})
    planpath=private/'prospective_fixed_inputs_v2_PRIVATE.json';detailpath=private/'audit_v2_PRIVATE.json';public=W/'results/PAIRED_PROMPT_SAVED_MATCHED_AVAILABILITY_V2.json'
    require(not any(path.exists() for path in [planpath,detailpath,public]),'Frozen v2 outputs already exist')
    def write(path,value):path.write_text(json.dumps(value,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n')
    write(planpath,plan);write(detailpath,detail);safe['private_plan_sha256']=sha(planpath);safe['private_audit_sha256']=sha(detailpath);write(public,safe)
    print(json.dumps({'status':safe['status'],'private_plan_v2_sha256':sha(planpath),'private_audit_v2_sha256':sha(detailpath),'public_v2_sha256':sha(public),'script_sha256':sha(__file__),'data_reselection':False,'new_native_calls':0}))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--primary-root',type=Path);ap.add_argument('--self-test',action='store_true');ap.add_argument('--amend-semantics-v2',action='store_true');a=ap.parse_args()
    if a.self_test:print(json.dumps({'CPU_contract_fixtures_passed':self_test(),'model_imports':0}));return
    if a.amend_semantics_v2:
        require(all(os.environ.get(k)=='1' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']),'CPU thread caps1 required')
        amend_semantics_v2();return
    require(a.primary_root is not None,'Explicit primary root required')
    require(all(os.environ.get(k)=='1' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']),'CPU thread caps1 required')
    private=W/'local_data/paired_prompt_saved_availability_v1';public=W/'results/PAIRED_PROMPT_SAVED_MATCHED_AVAILABILITY_V1.json'
    require(not private.exists() and not public.exists(),'Frozen outputs already exist;use a new version instead of overwrite')
    safe,detail,plan=audit(a.primary_root.resolve());private.mkdir(parents=True)
    def write(path,value):path.write_text(json.dumps(value,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n')
    planpath=private/'prospective_fixed_inputs_PRIVATE.json';detailpath=private/'audit_PRIVATE.json';write(planpath,plan);write(detailpath,detail)
    safe['private_plan_sha256']=sha(planpath);safe['private_audit_sha256']=sha(detailpath)
    require(not any(x=='mlx' or x.startswith('mlx.') or x=='mlx_vlm' or x.startswith('mlx_vlm.') for x in sys.modules),'Native module unexpectedly imported')
    write(public,safe);print(json.dumps({'status':safe['status'],'eligible_matched_blocks':safe['coverage']['eligible_matched_eight_response_blocks'],
       'private_plan_sha256':sha(planpath),'private_audit_sha256':sha(detailpath),'public_sha256':sha(public),'script_sha256':sha(__file__)}))

if __name__=='__main__':main()
