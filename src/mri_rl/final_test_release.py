"""Minimal final-test release/projection and authenticated matrix bridge (CPU only).

This does not decide that training completed. An independently supplied, exact
review digest must attest original receipt chains and final budgets; original
receipts remain hashed files, not re-labelled as a new training schema. Missing,
BLOCKED, pilot-only or zero-IQR arms cannot silently become a full comparison.
No sealed registry, target or image is read before validate_release succeeds.
The released-row loader remains responsible for the separately reviewed exact
sequence/Dallas renderer; this module never substitutes a new renderer.
"""
from __future__ import annotations
import json
import inspect
import re
from pathlib import Path

from mri_rl.two_task_pool import canonical_sha, require, sha, write_exact
from mri_rl.two_task_evaluation import project_records
from mri_rl.two_task_analysis import _reference, score_predictions, build_matrix

ARMS=('S_sequence','S_correspondence','P_raw','U_raw','P_calibrated','U_calibrated')
BINDINGS=('plan_sha256','sequence_registry_sha256','dallas_registry_sha256','dallas_metadata_plan_sha256',
          'sequence_renderer_source_sha256','dallas_renderer_source_sha256','prompt_contract_sha256',
          'scorer_source_sha256','raw_adapter_source_sha256','statistics_source_sha256','projection_source_sha256','release_source_sha256','released_row_loader_source_sha256')


def required_checkpoints(plan):
    require(plan.get('schema')=='two-task-finite-arm-plan-v1' and plan.get('seeds')==[7,11,13], 'Frozen finite plan/seeds differ')
    require([a['name'] for a in plan['rl_arms']]==list(ARMS),'Full finite primary arm inventory differs')
    return ('B28',)+tuple(f'rl:{arm}:{seed}' for arm in ARMS for seed in plan['seeds'])+tuple(f'sft:{seed}' for seed in plan['seeds'])


def required_matrix_keys(plan):
    required_checkpoints(plan)
    return {(arm,seed) for seed in plan['seeds'] for arm in ('B28',*ARMS,'same_C_SFT')}


def _bound_file(descriptor):
    require(isinstance(descriptor,dict) and set(descriptor)=={'path','sha256'},'Exact file descriptor required')
    require(isinstance(descriptor['sha256'],str) and re.fullmatch('[0-9a-f]{64}',descriptor['sha256']), 'Malformed file digest')
    require(Path(descriptor['path']).is_file() and sha(descriptor['path'])==descriptor['sha256'],'Reviewed checkpoint/evidence file differs')


def validate_release(plan,inventory,bindings,approval,*,expected_approval_sha256):
    """Validate an externally authenticated scoped review, then bound local files.

    expected_approval_sha256 must come from the independent reviewer/parent,
    never be computed from an untrusted approval and treated as authorization.
    A file hash proves identity, not training correctness; that substantive
    verification is explicitly required in the independently approved scope.
    """
    require(canonical_sha(approval)==expected_approval_sha256,'Independent release approval digest differs')
    require(approval.get('verdict') in {'PASS','PASS_WITH_LIMITATIONS'} and approval.get('reviewer'),'Independent release verdict required')
    require(set(bindings)==set(BINDINGS) and all(isinstance(v,str) and re.fullmatch('[0-9a-f]{64}',v) for v in bindings.values()),'Exact final scoring/renderer/source bindings required')
    require(bindings['plan_sha256']==canonical_sha(plan),'Finite plan differs from release binding')
    expected_scope={'kind':'final-two-task-test-release-v1','bindings':bindings,
        'checkpoint_inventory_sha256':canonical_sha(inventory),'training_completion_verified':True,
        'checkpoint_selection':'final_budget_boundary','test_outcomes_used':False,
        'all_final_receipt_chains_and_budgets_verified':True}
    require(approval.get('scope')==expected_scope,'Final receipt/budget/checkpoint/scoring review scope differs')
    source_files={'dallas_renderer_source_sha256':'visit_correspondence.py','scorer_source_sha256':'vero_mri_reward.py','raw_adapter_source_sha256':'two_task_native_results.py',
        'statistics_source_sha256':'two_task_analysis.py','projection_source_sha256':'two_task_evaluation.py','release_source_sha256':'final_test_release.py'}
    require(all(bindings[k]==sha(Path(__file__).with_name(name)) for k,name in source_files.items()),'Scoring/projection/release source changed after independent review')
    needed=required_checkpoints(plan)
    require(len(inventory)==len(needed) and {x['checkpoint_id'] for x in inventory}==set(needed),'Final checkpoint coverage incomplete or duplicated')
    for item in inventory:
        require(set(item)=={'checkpoint_id','adapter','adapter_config','completion_evidence'},'Unexpected final inventory fields')
        _bound_file(item['adapter']);_bound_file(item['adapter_config'])
        evidence=item['completion_evidence']
        require(isinstance(evidence,list) and (item['checkpoint_id']=='B28' or bool(evidence)),'Final training completion evidence missing')
        for descriptor in evidence:_bound_file(descriptor)
        if item['checkpoint_id']=='B28':
            require(item['adapter']['sha256']==plan['common_bindings']['starting_adapter_sha256'],'Common B28 initializer differs')
    return {'status':'RELEASE_SCOPE_AND_FILE_IDENTITIES_VERIFIED','approval_sha256':expected_approval_sha256,
        'checkpoint_inventory_sha256':canonical_sha(inventory),'bindings':bindings,'distinct_checkpoints':len(needed),
        'matrix_cells':len(required_matrix_keys(plan))*2,'actual_test_execution':False,
        'training_correctness_source':'Independent exact receipt-chain/budget review; file existence is not completion evidence.'}


def _test_counts(rows):
    _reference(rows,'test')
    sequence=[r for r in rows if r['ability']=='mri_sequence'];dallas=[r for r in rows if r['ability']=='mri_visit_correspondence']
    require(len(sequence)==84 and len(dallas)==16,'Frozen final test case counts differ')
    require(len({r['extra_info']['patient_key'] for r in sequence})==14,'Prospective sequence patient inventory differs')
    require(len({r['extra_info']['patient_key'] for r in dallas})==8 and len({r['extra_info']['statistical_cluster_key'] for r in dallas})==4,'Dallas patient/cycle inventory differs')
    require({r['reward_model']['ground_truth']:sum(x['reward_model']['ground_truth']==r['reward_model']['ground_truth'] for x in sequence) for r in sequence}=={'FLAIR':42,'contrast-enhanced T1':42},'Sequence modality inventory differs')
    for patient in {r['extra_info']['patient_key'] for r in sequence}:
        person=[r for r in sequence if r['extra_info']['patient_key']==patient]
        require(len(person)==6 and all(sum(r['reward_model']['ground_truth']==t for r in person)==3 for t in ('FLAIR','contrast-enhanced T1')),'Sequence patient/level inventory differs')


def release_projection(plan,inventory,bindings,approval,*,expected_approval_sha256,load_released_rows,contract_path,destination):
    """Only invoke a reviewed renderer/typed-row loader after all release checks.

    Loader must verify the exact registry hashes and renderer bound in bindings,
    then return the complete unchanged typed test rows (no case selection).
    Its implementation is separately source-bound in the independent approval.
    No loader is called on an incomplete or unauthenticated release.
    """
    proof=validate_release(plan,inventory,bindings,approval,expected_approval_sha256=expected_approval_sha256)
    require(sha(contract_path)==bindings['prompt_contract_sha256'],'Prompt contract differs')
    contract=json.loads(Path(contract_path).read_text())
    loader_source=inspect.getsourcefile(load_released_rows)
    require(loader_source and sha(loader_source)==bindings['released_row_loader_source_sha256'],'Released row-loader source differs from independent review')
    rows=load_released_rows();_test_counts(rows)
    records=project_records(rows,contract);destination=Path(destination)
    image_hashes={r['id']:sha(r['images'][0]) for r in rows}
    source=destination/'final_test.verl.jsonl';projected=destination/'final_test_native_system_matched.jsonl'
    for path,items in ((source,rows),(projected,records)):
        write_exact(path,(''.join(json.dumps(r,sort_keys=True)+'\n' for r in items)).encode())
    receipt={**proof,'status':'RELEASED_FINAL_TEST_PROJECTION_FROZEN','cases':100,
        'source_verl_manifest_sha256':sha(source),'eval_manifest_sha256':sha(projected),
        'prompt_contract_sha256':bindings['prompt_contract_sha256'],'sequence_people':14,'dallas_people':8,'dallas_cycles':4,
        'projection_kind':'training-system-matched: system message plus unchanged question string',
        'actual_model_execution':False,'image_inventory_sha256':canonical_sha(image_hashes),'cross_source_global_independence_verified':False,
        'whole_head_correspondence_not_brain_only':True}
    write_exact(destination/'projection_receipt_SAFE.json',(json.dumps(receipt,indent=2,sort_keys=True)+'\n').encode())
    return source,projected,receipt


def adapt_final_matrix(plan,matrix_adapters,release_bundle,*,n_bootstrap=2000):
    """Authenticate actual CLI files using the existing strict adapter; no inference.

    Each matrix value is an unchanged kwargs mapping for adapt_cli_output with
    externally authenticated scope/projection/output hashes. One immutable B28
    output may be reused across seeds after exact bindings; it is not three runs.
    """
    require(set(matrix_adapters)==required_matrix_keys(plan),'Final matrix coverage incomplete or extra')
    inventory=release_bundle['inventory'];bindings=release_bundle['bindings']
    validate_release(plan,inventory,bindings,release_bundle['approval'],expected_approval_sha256=release_bundle['expected_approval_sha256'])
    checkpoints={x['checkpoint_id']:x for x in inventory};runs={};proofs={};source_sha=None
    from mri_rl.two_task_native_results import adapt_cli_output
    for (arm,seed),kwargs in sorted(matrix_adapters.items()):
        key='B28' if arm=='B28' else f'sft:{seed}' if arm=='same_C_SFT' else f'rl:{arm}:{seed}'
        item=checkpoints[key];config=kwargs['expected_config']
        require(config['adapter_checkpoint_sha256']==item['adapter']['sha256'] and config['adapter_config_sha256']==item['adapter_config']['sha256'],'Matrix checkpoint differs from approved final inventory')
        require(sha(kwargs['contract_path'])==bindings['prompt_contract_sha256'],'Matrix prompt contract differs')
        actual=sha(kwargs['source_manifest']);source_sha=actual if source_sha is None else source_sha
        require(actual==source_sha,'Final matrix reference manifests differ')
        projection=json.loads(Path(kwargs['projection_receipt']).read_text())
        require(projection.get('status')=='RELEASED_FINAL_TEST_PROJECTION_FROZEN'
                and projection.get('approval_sha256')==release_bundle['expected_approval_sha256']
                and projection.get('bindings')==bindings,'Output projection is not the independently released final test')
        normalized,proof=adapt_cli_output(**kwargs)
        rows=[json.loads(line) for line in Path(kwargs['source_manifest']).read_text().splitlines() if line]
        _test_counts(rows);require(set(normalized)=={r['id'] for r in rows},'Final matrix prediction coverage incomplete')
        runs[(arm,seed)]=score_predictions(rows,normalized,expected_split='test');proofs[f'{arm}:{seed}']=proof
    training={a:None for a in ('B28',*ARMS,'same_C_SFT')}
    training.update(S_sequence='mri_sequence',S_correspondence='mri_visit_correspondence')
    return {'matrix':build_matrix(runs,training_task_by_arm=training,n_bootstrap=n_bootstrap),
            'file_binding_proofs':proofs,'common_baseline_reuse_is_not_extra_runs':True,
            'source_verl_manifest_sha256':source_sha,'actual_test_execution':False,
            'interpretation':'Sequence14 patients and Dallas4 two-person cycles analyzed separately; no global cross-source independence or brain-only/clinical claim.'}
