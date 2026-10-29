"""CPU preparation of an Adam60 paired-z045 continuation; training launch BLOCKED.

The separate F32 backend is CPU implemented, never native qualified. Narrow
read-only proofs cannot unlock it or the historical BF16 backend. This module
imports no MLX; actual train entry requires authenticated complete qualification,
new human authorization and an independent exact continuation review.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from mri_rl import paired_z045_continuation_driver_v1 as driver
from mri_rl.continuous_rl import canonical, require

ROOT=Path(__file__).resolve().parents[2]
IMPLEMENTATION_STATUS='IMPLEMENTED_CPU_NOT_NATIVE_QUALIFIED'


def digest(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def numeric_prerequisites():
    return {'implementation_status':IMPLEMENTATION_STATUS,'narrow_readonly_proofs_authorize_training':False,
        'required_actual_evidence':{
            'onpolicy_generation_logps':'Implemented native F32 activation/cache/logits rollout captures actual selected logps, sampled IDs, full causal context and stopping receipt; same policy as scorer, no substituted old BF16 generate.',
            'KV_full_on_generated_IDS':'Both declared task shapes and all complete generated response positions, selected/F64-normalized full-vocabulary comparisons under frozen .002 threshold; explicit source/runtime/dtype/cache bindings.',
            'gradient_path':'Same functional F32 policy used by unchanged native GSPO autograd, finite504 gradients, own-old/current<=1e-5 and actual ratio evidence before any optimizer update; no CE surrogate.',
            'parent_Adam60_state':'Trained boundary8/global129/Adam60 restored adapter504+full Adam moments/counter+Python/MLX RNG, no state reset; source/runtime compatibility verified.',
            'complete_response_EOS':'Maximum256 response tokens including observed EOS or explicit length termination; no post-EOS tokens; prompt/input preparation exact across generation/scorer.',
            'safety_review':'Exact finite qualification scope and actual independent review first; separate future human continuation authorization and independently reviewed exact continuation scope; source/hash changes invalidate.'},
        'existing_evidence_limit':'Sequence17 and Dallas17 fixed-response F32 probes and saved B3/margins qualify narrow read-only functions, not complete stochastic generation/scorer/backward training implementation.',
        'native_fallback':'FORBIDDEN; no fallback to historical BF16 backend if F32 missing or fails'}


def parent_metadata(path,expected_sha256):
    """Inspect small receipt/manifest only. Array bytes require future execution verification."""
    path=Path(path); receipt_path=path/'receipt.json'
    require(path.is_dir() and not path.is_symlink() and digest(receipt_path)==expected_sha256,'Parent receipt changed')
    receipt=json.loads(receipt_path.read_text());manifest_path=path/'optimizer/manifest.json'
    require(digest(manifest_path)==receipt['files_sha256']['optimizer/manifest.json'],'Parent optimizer manifest changed')
    manifest=json.loads(manifest_path.read_text())
    require(receipt['schema']=='continuous-rl-boundary-v1' and path.name=='boundary_0008' and
            receipt['completed_groups']==8 and receipt['optimizer_updates']==4 and
            receipt['absolute_completed_groups']==129 and receipt['absolute_optimizer_updates']==60,
            'Exact trained boundary8/global129/Adam60 required')
    previous=receipt['binding']['experiment']['adaptive_parent']
    require(previous['groups']+8==129 and previous['updates']+4==60 and
            manifest['completed_groups']==129 and manifest['optimizer_updates']==60 and
            manifest['bindings']['run']==receipt['binding'] and
            manifest['bindings']['adapter_sha256']==receipt['files_sha256']['adapter.safetensors'],'Parent state/counter binding mismatch')
    require(all((path/name).is_file() and not (path/name).is_symlink() for name in ('adapter.safetensors','optimizer/state.safetensors','adapter_config.json')),'Missing parent state files')
    return {'receipt_sha256':expected_sha256,'groups':129,'updates':60,'local_groups':8,'local_updates':4,
            'original_binding':receipt['binding'],'adapter_sha256':receipt['files_sha256']['adapter.safetensors'],
            'optimizer_manifest_sha256':receipt['files_sha256']['optimizer/manifest.json'],
            'optimizer_state_sha256':receipt['files_sha256']['optimizer/state.safetensors'],
            'array_bytes_verified':False,'array_verification':'DEFERRED_NO_CPU_WEIGHT_OR_ADAM_PAYLOAD_READ'}


def scope_digest(scope):return hashlib.sha256(canonical(scope).encode()).hexdigest()


def validate_launch_evidence(scope,numeric,human,review):
    """Prospective exact-evidence shape, independently of absent implementation.

    This validator never grants authorization itself. Actual human/reviewer
    artifacts must be authenticated by the future launcher; prep emits none.
    """
    require(scope.get('scope_kind')=='FUTURE_EXACT_TRAIN_LAUNCH_SCOPE','Preparation artifact is not a train launch scope')
    require(numeric.get('policy')=='F32_ACTIVATION_GENERATION_SCORER_GSPO_V1' and
            numeric.get('status') in ('PASS','PASS_WITH_LIMITATIONS') and
            numeric.get('contract_sha256')==scope.get('numeric_contract_sha256') and
            numeric.get('source_sha256')==scope.get('source_sha256'),'Exact implemented F32 numeric source/contract evidence required')
    require(all(numeric.get('actual_evidence',{}).get(k) is True for k in numeric_prerequisites()['required_actual_evidence']), 'Complete actual F32 rollout/scorer/GSPO evidence required; narrow probes insufficient')
    binding=scope_digest(scope)
    require(human.get('authorized') is True and human.get('action')=='finite_train' and human.get('scope_sha256')==binding and human.get('human_instruction_sha256'),'Fresh human authorization for exact finite train scope required')
    require(review.get('verdict') in ('PASS','PASS_WITH_LIMITATIONS') and review.get('scope_sha256')==binding and review.get('reviewer_id') and review.get('reviewer_kind') in ('isolated_codex','claude_code'),'Independent exact train-scope review required')
    require(all(type(scope['budget'].get(k)) is int and 1<=scope['budget'][k]<=limit for k,limit in (('maximum_attempted_groups',32),('maximum_optimizer_updates',8))),'Finite paired-stage envelope exceeded')
    return True


def training_preflight(scope,numeric_contract,human_authorization,independent_review):
    # Mere JSON booleans and old narrow proofs cannot qualify this implementation.
    require(numeric_contract and numeric_contract.get('actual_proof_path') and
            numeric_contract.get('actual_proof_sha256') and numeric_contract.get('actual_review_path') and
            numeric_contract.get('actual_review_sha256'),'F32_NATIVE_QUALIFICATION_REQUIRED')
    actual_path=Path(numeric_contract['actual_proof_path']);review_path=Path(numeric_contract['actual_review_path'])
    require(actual_path.is_file() and review_path.is_file() and digest(actual_path)==numeric_contract['actual_proof_sha256'] and
            digest(review_path)==numeric_contract['actual_review_sha256'],'F32_NATIVE_QUALIFICATION_REQUIRED: authenticated actual artifacts absent/changed')
    actual=json.loads(actual_path.read_text());review=json.loads(review_path.read_text())
    require(actual.get('status')=='ACTUAL_F32_ONPOLICY_GSPO_READONLY_QUALIFIED' and actual.get('optimizer_updates')==0 and
            review.get('verdict') in ('PASS','PASS_WITH_LIMITATIONS') and review.get('actual_proof_sha256')==digest(actual_path) and
            actual.get('source_sha256')==scope.get('source_sha256') and
            review.get('reviewer_id') and review.get('reviewer_kind') in ('isolated_codex','claude_code'), 'F32_NATIVE_QUALIFICATION_REQUIRED: incomplete actual policy/gradient review')
    validate_launch_evidence(scope,numeric_contract,human_authorization,independent_review)
    return True


def stage_spec_binding(spec):return scope_digest({k:v for k,v in spec.items() if k!='resume'})


def run_with_release(run,resident,out):
    """Preserve first failure, clear unwound native frames, then audit release."""
    import gc,traceback
    from mri_rl.native_gspo_path_probe_v1 import clear_unwound_exception_frames
    out=Path(out);pending=None
    try:return run()
    except BaseException as error:
        pending=error
        driver.write_json(out/'entry_STOP_PRIVATE.json',{'error_type':type(error).__name__,'error':str(error),'traceback':''.join(traceback.format_exception(error))})
        clear_unwound_exception_frames(error);raise
    finally:
        if resident[0] is not None:
            if pending:clear_unwound_exception_frames(pending)
            backend=resident[0]
            try:
                backend.close();gc.collect();backend.mx.clear_cache();active=int(backend.mx.get_active_memory())
                released=backend.model is None and active<128*1024**2
                driver.write_json(out/'release_SAFE.json',{'model_is_none':backend.model is None,'active_MLX_bytes':active,'release_verified':released})
                if pending is None:require(released,'Native model release failed')
            except BaseException as error:
                driver.write_json(out/'release_failure_PRIVATE.json',{'error':str(error),'original_failure_preserved':pending is not None})
                clear_unwound_exception_frames(error)
                if pending is None:raise


def validate_prepared_budget(scope,prepared):
    require(scope.get('budget')==prepared.get('budget'),'Exact future launch budget must equal prepared32/8/token/wall budget')
    return True


def cache_metadata(spec):
    """Expected certified binding using only JSON/nonweight bytes + shard stat.

    This intentionally DOES NOT verify payload hashes. The authorized execute
    path still calls full cache_binding_for inside the exclusive lock, followed
    by inherited loaded-parameter parity before attaching the trained adapter.
    """
    cache=Path(spec['cache_dir']);model=Path(spec['model']);certpath=Path(spec['cache_certificate'])
    require(digest(certpath)==spec['cache_certificate_sha256'],'Cache certificate digest changed')
    cert=json.loads(certpath.read_text())
    require(cert.get('tensors_checked')==1438 and cert.get('payload_bytes_equal_to_verified_units') is True,'Full1438 certified cache required')
    indexpath=cache/'model.safetensors.index.json';index=json.loads(indexpath.read_text())['weight_map']
    require(len(index)==1438 and set(index.values())==set(cert['shards']),'Cache tensor/shard index differs')
    for name,meta in cert['shards'].items():
        path=cache/name
        require(Path(name).name==name and path.is_file() and not path.is_symlink() and path.stat().st_size==meta['bytes'],'Cache shard metadata differs')
    require(digest(cache/'config.json')==cert['config_cached_sha256'] and digest(cache/'config_source.json')==cert['config_source_sha256'] and
            digest(model/'config.json')==cert['config_source_sha256'],'Cache original/cached config differs')
    source=json.loads((cache/'config_source.json').read_text());new=json.loads((cache/'config.json').read_text())
    require({k for k in set(source)|set(new) if source.get(k)!=new.get(k)}=={'quantization','quantization_config'} and new['quantization']==new['quantization_config'],'Cache quantization config delta differs')
    for name,expected in cert['non_weight_files_identical'].items():
        require(digest(cache/name)==expected and (not (model/name).is_file() or digest(model/name)==expected),'Cache nonweight source file differs')
    expected={'certificate_sha256':spec['cache_certificate_sha256'],'plan_sha256':cert['plan_sha256'],'shards':cert['shards'],
        'config_cached_sha256':cert['config_cached_sha256'],'config_source_sha256':cert['config_source_sha256'],
        'tensors':1438,'quantized_modules':cert['quantized_modules'],'cache_dir':str(cache.resolve())}
    return {'expected_binding':expected,'index_sha256':digest(indexpath),'shard_sizes_verified':True,
            'payload_verification':'DEFERRED_TO_AUTHORIZED_EXECUTE_FULL_SHA_AND_LOADED_PARAMETER_PARITY'}


def frozen_image_map(spec,schedule):
    """Join authenticated historical image evidence, never hash-current-as-truth."""
    for key in ('data_membership','original_image_inventory','original_pool_receipt'):
        require(key in spec and key in spec['file_sha256'] and digest(spec[key])==spec['file_sha256'][key],'Frozen image provenance artifact changed: '+key)
    membership=json.loads(Path(spec['data_membership']).read_text())
    by_sequence={row['record_id_PRIVATE']:row['PNG_sha256'] for row in membership}
    require(len(by_sequence)==len(membership),'Duplicate sequence membership key')
    inventory=json.loads(Path(spec['original_image_inventory']).read_text())
    receipt=json.loads(Path(spec['original_pool_receipt']).read_text())
    require(receipt['private_image_inventory_sha256']==digest(spec['original_image_inventory']),'Original Dallas image inventory is not receipt-authenticated')
    expected={}
    for row in schedule:
        source=row['source_record_id']
        if row['ability']=='mri_sequence':
            sha=by_sequence.get(source)
            require(sha and row.get('rc_data_v1',{}).get('image_sha256')==sha,'Sequence image SHA differs from authenticated membership')
        else:
            require(row['ability']=='mri_visit_correspondence','Unknown frozen image route')
            sha=inventory['image_sha256_by_record'].get(source)
        require(isinstance(sha,str) and len(sha)==64,'Frozen image source key missing')
        expected[row['id']]=sha
    return expected


def verify_images(rows,expected):
    require(all(row['id'] in expected and len(row['images'])==1 and digest(row['images'][0])==expected[row['id']] for row in rows),'Approved frozen image bytes changed')
    return True


def source_closure(spec):
    from mri_rl.tier_c_entry import CRITICAL_SOURCES
    names=set(CRITICAL_SOURCES)|{
        'paired_z045_continuation_driver_v1.py','paired_z045_continuation_entry_v1.py','paired_z045_f32_backend_v1.py','paired_z045_f32_qualification_v1.py',
        'uniform_balance_continuation_gate_v1.py','uniform_balance_continuation_driver_v1.py',
        'conditional_gspo_gradient_v2.py','native_probability_resolution_v1.py','rollout_precision_v2.py',
        'native_gspo_path_probe_v1.py','early_native_conditioning_probe_v1.py'}
    result={str(ROOT/'src/mri_rl'/name):digest(ROOT/'src/mri_rl'/name) for name in names}
    runtime=spec.get('runtime_source_sha256',{})
    suffixes=('mlx_vlm/generate/ar.py','mlx_vlm/generate/dispatch.py','mlx_vlm/generate/common.py',
              'mlx_vlm/sample_utils.py','mlx_vlm/models/qwen3_vl/language.py','mlx_vlm/models/qwen3_vl/qwen3_vl.py',
              'mlx_vlm/models/base.py','mlx_vlm/models/cache.py','mlx/nn/layers/activations.py',
              'mlx/nn/layers/quantized.py','mlx/nn/layers/normalization.py','mlx_vlm/trainer/peft/lora_layers.py')
    require(all(any(path.endswith(suffix) for path in runtime) for suffix in suffixes),'Complete exact installed F32 generation/scorer/runtime closure required')
    for path,expected in runtime.items():require(digest(path)==expected,'Installed runtime source differs: '+path)
    result.update(runtime)
    return result


def prepare(spec):
    """Bind prepared data/schedule and historical profile; inspect, never load arrays."""
    anchor=parent_metadata(spec['parent_checkpoint'],spec['parent_receipt_sha256'])
    cache_preparation=cache_metadata(spec)
    for key,expected in spec['file_sha256'].items(): require(digest(spec[key])==expected,'Prepared input digest differs: '+key)
    reference=json.loads(Path(spec['reference_schedule']).read_text());schedule=json.loads(Path(spec['candidate_schedule']).read_text())
    if isinstance(reference,dict):reference=reference['schedule']
    if isinstance(schedule,dict):schedule=schedule['schedule']
    require(len(reference)==len(schedule)==32 and len({r['id'] for r in schedule})==32,'Frozen32 unique occurrences required')
    for previous,current in zip(reference,schedule):
        require(previous['ability']==current['ability'],'Historical uniform category slots changed')
        if previous['ability']=='mri_visit_correspondence': require(previous==current,'Historical Dallas occurrence changed')
    expected_images=frozen_image_map(spec,schedule);verify_images(schedule,expected_images)
    require(set(r['ability'] for r in schedule)=={'mri_sequence','mri_visit_correspondence'},'Two-task schedule required')
    from mri_rl.two_task_training import validate_training_row
    for row in schedule:validate_training_row(row)
    config={**driver.PROFILE,'group_size':4,'max_tokens':256}
    require(config==anchor['original_binding']['configuration'],'Historical optimizer/reward/admission/profile differs')
    calibration=json.loads(Path(spec['calibration']).read_text())
    original=anchor['original_binding']['experiment']['parent_experiment']
    require(original['arm']=={'name':'U_calibrated','reward':'medgrpo_eq3','sampling':'uniform'} and original['seed']==13 and
            scope_digest(calibration)==original['calibration_sha256'],'Inherited arm/calibration changed')
    costs=json.loads(Path(spec['input_costs']).read_text())
    require(set(costs)=={r['id'] for r in schedule} and all(type(v) is int and 0<v<=2048 for v in costs.values()),'Exact frozen input-token costs required')
    counts={task:sum(r['ability']==task for r in schedule) for task in ('mri_sequence','mri_visit_correspondence')}
    require(counts=={'mri_sequence':18,'mri_visit_correspondence':14} and all(costs[r['id']]==(213 if r['ability']=='mri_sequence' else 405) for r in schedule),'Exact nextslots8:40 category/input budget differs')
    cap=sum(4*(costs[r['id']]+256) for r in schedule)
    require(spec['text_visual_token_cap']==cap==70784,'Full worst-case32K4 token reservation cap differs')
    require(type(spec['maximum_seconds']) is int and spec['maximum_seconds']==3000 and
            0<=spec['minimum_group_window_seconds']<spec['maximum_seconds'],'Finite boundary-safe wall budget required')
    output=Path(spec['output']).resolve();require(output.is_relative_to(ROOT/'local_data'),'Ignored private output required')
    bindings={'stage':'paired_z045_continuation_v1','parent_purpose':'Literal52→56→60 continuity, not test-selected best checkpoint',
        'adaptive_parent':anchor,'source_pool_change':'sequence28→56z045; Dallas48 unchanged',
        'sampler_stage':'Historical uniformseed13 task/Dallas nextslots8:40 fixed after authenticating prior8; existingfull198 sampler appliedold76/new104, filteredsequence substream skipsprior2 draws thennext18 mappedtofixedslots. Oldcontrol source contentrestore exact; no newrowqueue/forcedpair/quota.',
        'schedule_sha256':hashlib.sha256(canonical(schedule).encode()).hexdigest(),'spec_sha256':stage_spec_binding(spec)}
    # Inspection must not call driver's _binding/parent_anchor: those verify array
    # bytes at actual execution. Scope records their exact deferred requirements.
    return {'schema':'paired-z045-continuation-preparation-v1','execution_tier':'C','status':'CPU_PREPARED_TRAIN_LAUNCH_BLOCKED',
        'training_authorized':False,'model_loads':0,'optimizer_updates':0,'parent':anchor,'bindings':bindings,'configuration':config,
        'budget':{**driver.ENVELOPE,'group_size':4,'max_tokens':256,'text_visual_token_cap':cap,'maximum_seconds':spec['maximum_seconds'],'minimum_group_window_seconds':spec['minimum_group_window_seconds']},
        'scope_kind':'PREPARATION_ONLY_NOT_LAUNCH_SCOPE','expected_image_sha256':expected_images,'derived_cache':cache_preparation['expected_binding'],'cache_preparation_verification':cache_preparation,'schedule':schedule,'input_costs':costs,'output':str(output),
        'numeric_prerequisites':numeric_prerequisites(),'file_sha256':spec['file_sha256'],
        'source_sha256':source_closure(spec),
        'resume_contract':'Inherited Adam60/fullRNG at boundary0, new frozen32 occurrence stagecursor0 representing oldsampler slots8:40; exact same bound schedule onresume. Immutable absolute3000s deadline journal binds experiment and counts idle/holds across resumes; never resets. Driver persists reservation before stochastic work and charges measured/imputed pending debt once. No wall kill midgroup; actual atomic group/release canoverrun.',
        'limitations':['F32 backend CPU implemented NOT_NATIVE_QUALIFIED; prep cannot be promoted by editing JSON evidence.','No full56 pool/pair coverage guarantee within32 category slots.','F32 vs historicalBF16 is a new numerical stage; matched parent/child F32 readouts needed, no accuracy/clinical eligibility claim.'],
        'mission_alignment':{'tag':'C_TASK_ELIGIBILITY','evidence_tier':'PROPOSED_NOT_RUN','question':'Prepare controlled sequence exposure continuation while preserving taskslots/Dallas/reward/GSPO/optimizer continuity.','decision':'HOLD_NATIVE_PENDING_QUALIFICATION_AUTHORIZATION_REVIEW'}}


def execute_train(args):
    """Future executable entry; every approval gate precedes native imports."""
    import fcntl
    required=('spec','scope','expected_scope_sha256','numeric_contract','expected_numeric_sha256',
              'human_authorization','expected_human_sha256','review','expected_review_sha256')
    require(all(getattr(args,k,None) for k in required),'F32_NATIVE_QUALIFICATION_REQUIRED: exact future scope/numeric/human/review files and hashes required')
    for path,expected in ((args.scope,args.expected_scope_sha256),(args.numeric_contract,args.expected_numeric_sha256),
                          (args.human_authorization,args.expected_human_sha256),(args.review,args.expected_review_sha256)):
        require(digest(path)==expected,'Future launch evidence file digest changed')
    spec=json.loads(args.spec.read_text());prepared=prepare(spec);scope=json.loads(args.scope.read_text())
    numeric=json.loads(args.numeric_contract.read_text());human=json.loads(args.human_authorization.read_text());review=json.loads(args.review.read_text())
    require(scope.get('preparation_sha256')==scope_digest(prepared) and scope.get('numeric_contract_file_sha256')==digest(args.numeric_contract), 'Exact preparation/numeric proof scope binding differs')
    validate_prepared_budget(scope,prepared)
    training_preflight(scope,numeric,human,review)
    def verify_sources():
        require(scope.get('source_sha256')==prepared['source_sha256'],'Future launch source closure differs')
        for path,expected in scope['source_sha256'].items():require(digest(path)==expected,'Future launch source changed: '+path)
    verify_sources()
    from mri_rl.paced_resources import PacedGate
    from mri_rl.two_task_training import TokenLedger,make_grader,validate_training_row
    from mri_rl.experiment_contract import load_contract
    from mri_rl.continuous_rl_pilot import cache_binding_for
    from types import SimpleNamespace
    output=Path(spec['output']);gate=PacedGate(output.with_name(output.name+'_resources_PRIVATE.jsonl'))
    ledger_plan={'group_size':4,'max_tokens':256,'budget':{'maximum_attempted_groups':32,'text_visual_token_cap':70784,'input_tokens_min':213,'input_tokens_max':405}}
    with Path(spec['global_lock']).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(prepare(spec)==prepared,'Preparation changed while obtaining global lock');verify_sources()
        cache_binding=cache_binding_for(SimpleNamespace(cache_dir=Path(spec['cache_dir']),cache_certificate=Path(spec['cache_certificate']),
            cache_certificate_sha256=spec['cache_certificate_sha256'],model=Path(spec['model'])))
        require(cache_binding==scope.get('derived_cache'),'Future launch cache identity differs')

        # Importing this CPU module is safe; constructing inherited native loader
        # happens only in the authorized driver callback after all gates above.
        from mri_rl.paired_z045_f32_backend_v1 import F32NativeGSPOBackend
        from mri_rl.mlx_continuous_gspo import is_loading_unit
        cache={'dir':spec['cache_dir']}
        def before(name):
            sample=gate.before_unit(name,loading=is_loading_unit(name,cache))
            if name=='native_model_load':verify_images(prepared['schedule'],prepared['expected_image_sha256'])
            return sample
        resident=[None]
        evidence=Path(scope['launch_evidence_directory']).resolve();require(evidence.is_relative_to(ROOT/'local_data') and not evidence.exists(),'Fresh private per-launch evidence directory required');evidence.mkdir(parents=True)
        def factory(adapter,configuration):
            verify_images(prepared['schedule'],prepared['expected_image_sha256'])
            resident[0]=F32NativeGSPOBackend(adapter,configuration,model_path=Path(spec['model']),contract=load_contract(Path(spec['contract'])),seed=13,before_unit=before,cache=cache)
            resident[0].expected_image_sha256=prepared['expected_image_sha256']
            return resident[0]
        def authorize(context):
            before('authorize_'+context['phase'])
            verify_images(prepared['schedule'] if context['phase']=='load' else [prepared['schedule'][context['cursor']]],prepared['expected_image_sha256'])
        def run():return driver.run(output=output,schedule=prepared['schedule'],bindings=prepared['bindings'],config=prepared['configuration'],
            backend_factory=factory,authorize=authorize,max_new_groups=32,max_groups_cap=32,max_optimizer_updates=8,
            parent_checkpoint=spec['parent_checkpoint'],parent_receipt_sha256=spec['parent_receipt_sha256'],resume=bool(spec.get('resume')),
            grader=make_grader(json.loads(Path(spec['calibration']).read_text())),row_validator=validate_training_row,
            ledger=TokenLedger(ledger_plan),input_tokens=lambda row:prepared['input_costs'][row['id']],maximum_seconds=3000,minimum_group_window_seconds=spec['minimum_group_window_seconds'])
        result=run_with_release(run,resident,evidence)
        verify_sources()
        result={**result,'scope_sha256':scope_digest(scope),'numerical_policy':'F32_ACTIVATION_GENERATION_SCORER_GSPO_V1','science_claim':False}
        driver.write_json(output.parent/'native_complete_SAFE.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('inspect','preflight','train'))
    for name in ('spec','output','scope','numeric-contract','human-authorization','review'):parser.add_argument('--'+name,type=Path)
    for name in ('expected-scope-sha256','expected-numeric-sha256','expected-human-sha256','expected-review-sha256'):parser.add_argument('--'+name)
    args=parser.parse_args()
    if args.action=='train':
        result=execute_train(args);print(canonical(result));return
    if args.spec is None:result={'numeric_prerequisites':numeric_prerequisites(),'training_authorized':False,'model_loads':0}
    else:result=prepare(json.loads(args.spec.read_text()))
    if args.output:
        require(args.output.resolve().is_relative_to(ROOT/'local_data'),'Private CPU output required');args.output.parent.mkdir(parents=True,exist_ok=True);driver.write_json(args.output,result)
    print(canonical({'status':result.get('status','CPU_INSPECTED_NATIVE_BLOCKED'),'training_authorized':False,'model_loads':0,'numeric_backend':IMPLEMENTATION_STATUS}))

if __name__=='__main__':main()
