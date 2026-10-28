"""Backend-neutral crash-consistent driver for VERO_TO_MRI_ACCURACY_RECOVERY_V1. IMPLEMENTED_CPU_NOT_NATIVE_QUALIFIED.

Versioned copy of shared_rollout_gspo_driver_v1 (history preserved; the original is untouched) with these deltas only:
the parent is the AUTHENTIC original Adam52/boundary114 (core.adam52_parent_anchor), ONE combined scope replaces the
qualify/train/adopt split, an integrated read-only parent restore preflight precedes the rollout, the full actual
per-arm gradient is archived BEFORE each optimizer step, own-old parity (.002) is re-asserted before any step, and the
parent directory is hashed before and after the run. Order: restore parent -> preflight -> ONE F4+T4 rollout ->
parent restored -> own-old scoring/parity once -> arm A then B: restore identical parent, gradient, archive, AT MOST
ONE update, atomic boundary, once-only marker, restore parent. Authorization is a token issued by
core.authorize_launch and re-checked (reviewed source closure re-hashed) before backend construction and every phase;
there is no authorize=lambda hook. Reservations without commits are never retried automatically.
"""
from __future__ import annotations
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from mri_rl import paired_z045_continuation_driver_v1 as old
from mri_rl import accuracy_recovery_v1 as core
from mri_rl.optimizer_checkpoint import load_optimizer_checkpoint,pack_tree

require=core.require
STATUS=core.STATUS
ARM_ORDER=('A','B')


def _json(path):return json.loads(Path(path).read_text())


def _modality_rows(schedule):
    require(isinstance(schedule,list) and len(schedule)==2,'Exactly one FLAIR and one T1 TRAIN row required')
    require([r['reward_model']['ground_truth'] for r in schedule]==['FLAIR','contrast-enhanced T1'],'Schedule order must be FLAIR then contrast-enhanced T1')
    rows={'F':schedule[0],'T':schedule[1]};f,t=rows['F'],rows['T']
    require(f['ability']==t['ability']=='mri_sequence' and f['extra_info'].get('patient_key') and f['extra_info']['patient_key']==t['extra_info'].get('patient_key') and
            f['extra_info'].get('question') and f['extra_info']['question']==t['extra_info'].get('question') and f['id']!=t['id'],
            'Same-person generic-question sequence pair required')
    # Certified z045 source lineage (paired_z045_data_v1.select_z045); absent/None metadata never "matches".
    for row,modality in ((f,'flair'),(t,'t1ce')):
        meta=row.get('paired_z045_data_v1')
        require(isinstance(meta,dict) and meta.get('version')==1 and meta.get('level')==45 and meta.get('acquisition_modality_source_eligible') is True and
                meta.get('modality_PRIVATE')==modality and meta.get('training_authorized') is False,'Certified paired z045 source lineage required for '+modality)
    require(f['paired_z045_data_v1']['level']==t['paired_z045_data_v1']['level']==45,'Same z045 level required')
    return rows


def pair_identity(rows):
    extra=rows['F']['extra_info'];return core.identity({'patient_key':extra['patient_key'],'level':extra.get('level')})


def committed_boundary(output,arm,arm_binding):
    """The validated arm terminal if the atomic boundary_0001 rename happened (the commit point), else None."""
    arm_dir=Path(output)/f'arm_{arm}'
    if not (arm_dir/'boundary_0001').exists():return None
    last=old.inspect_boundaries(arm_dir,arm_binding)
    return last if last is not None and last['completed_groups']==1 else None


def _arm_status(output,arm,arm_binding):
    """COMMITTED / ABSENT. Any uncommitted attempt is a hard stop, never an automatic retry."""
    output=Path(output);marker=output/f'arm_{arm}_COMMITTED.json';arm_dir=output/f'arm_{arm}'
    reservation=output/f'arm_{arm}_reservation_PRIVATE.json'
    if marker.exists():
        record=_json(marker)
        last=old.inspect_boundaries(arm_dir,arm_binding)
        require(last is not None and last['completed_groups']==1 and old.digest(last['path']/'receipt.json')==record['boundary_receipt_sha256'],'Committed arm terminal differs from marker')
        return 'COMMITTED'
    last=committed_boundary(output,arm,arm_binding)
    if last is not None:
        # The boundary_0001 rename IS the commit; only the marker is missing. Record it, never replay (leftover pending dirs are harmless).
        core.publish_once(marker,{'arm':arm,'updated':bool(last['updated']),'boundary_receipt_sha256':old.digest(last['path']/'receipt.json'),'recovered_without_replay':True})
        return 'COMMITTED'
    if arm_dir.exists() or reservation.exists():raise RuntimeError(f'INCOMPLETE_ARM_{arm}: uncommitted attempt preserved; no automatic retry')
    return 'ABSENT'


def run(*,output,schedule,bindings,backend_factory,authorization,phase_gate,parent_checkpoint,parent_receipt_sha256,grader,
        reward_policy_sha256,budget,resume=False,config=None,row_validator=None,frozen=core.FROZEN_PARENT,
        clock=time.monotonic,wall_clock=time.time):
    """Execute (or resume a committed prefix of) the single combined experiment. Fixture backends are explicit.

    `authorization` must be the core.Authorization issued for exactly this scope/source/input/budget/parent; it is re-verified
    before the backend and every phase. `phase_gate(context)` is the ADDITIONAL resource/power/deadline gate; it can add refusals
    but cannot replace the authorization check. `frozen` is a fixture seam; production callers never pass it.
    """
    require(isinstance(authorization,core.Authorization) and callable(phase_gate),'A scope-bound Authorization token and a resource phase gate are required')
    require(isinstance(bindings,dict) and isinstance(bindings.get('source_sha256'),dict) and bindings['source_sha256'] and core.sha(bindings.get('input_sha256')) and core.sha(bindings.get('scope_sha256')),
            'Run must bind the actual reviewed source closure, input hash and scope digest')
    require(core.sha(reward_policy_sha256),'Exact task-routed reward calibration identity required')
    rows=_modality_rows(schedule)
    require(isinstance(budget,dict) and budget==core.derive_budget(budget.get('prompt_lengths')),'Budget is not the exact derived accuracy-recovery budget')
    maximum_seconds=budget['maximum_seconds'];started=clock();absolute=wall_clock()
    anchor=core.adam52_parent_anchor(parent_checkpoint,parent_receipt_sha256,frozen=frozen)
    require(authorization.matches(scope_sha256=bindings['scope_sha256'],source_sha256=bindings['source_sha256'],input_sha256=bindings['input_sha256'],budget=budget,parent=anchor),
            'Authorization was issued for a different scope/source/input/budget/parent')
    def authorize(context):
        authorization.check(context['phase']);phase_gate(context)
    configuration=json.loads(old.canonical(config or {**old.PROFILE,'group_size':core.GROUP_SIZE,'max_tokens':core.MAX_TOKENS}))
    for key,value in old.PROFILE.items():
        if key not in ('group_size','max_tokens'):require(configuration.get(key)==value,'Historical optimizer/reward/LoRA/LR profile changed')
    require(configuration['group_size']==core.GROUP_SIZE and configuration['max_tokens']==core.MAX_TOKENS,'Frozen K4/256 required')
    experiment={**bindings,'adaptive_parent':anchor,'experiment':core.EXPERIMENT,'execution_subtype':core.EXECUTION_SUBTYPE,'recipe':core.RECIPE_LABEL,
                'budget_sha256':core.identity(budget),'reward_policy_sha256':reward_policy_sha256}
    binding=old._binding(schedule,experiment,configuration,row_validator)
    binding_sha=hashlib.sha256(old.canonical(binding).encode()).hexdigest()
    arm_bindings={arm:{**binding,'experiment':{**binding['experiment'],'arm':arm}} for arm in ARM_ORDER}
    output=Path(output)
    if output.exists() and not resume:raise FileExistsError('Existing run requires explicit resume; preserve evidence')
    require(not resume or output.is_dir(),'Missing run for resume')
    output.mkdir(parents=True,exist_ok=True)
    with (output/'.writer.lock').open('a') as writer:
        try:fcntl.flock(writer,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another accuracy-recovery writer is active') from None
        deadline_path=output/'operation_deadline_PRIVATE.json'
        if deadline_path.exists():
            deadline=_json(deadline_path)
            require(deadline['binding_sha256']==binding_sha and deadline['maximum_seconds']==maximum_seconds and
                    deadline['deadline']==deadline['start']+maximum_seconds,'Original absolute deadline cannot reset')
        else:
            require(not resume,'Missing original deadline')
            deadline={'binding_sha256':binding_sha,'maximum_seconds':maximum_seconds,'start':absolute,'deadline':absolute+maximum_seconds}
            old.write_json(deadline_path,deadline);old.sync_dir(output)
        expired=lambda:clock()-started>=maximum_seconds or wall_clock()>=deadline['deadline']
        pair=pair_identity(rows)
        identity_record={'binding_sha256':binding_sha,'source_sha256':bindings['source_sha256'],'input_sha256':bindings['input_sha256'],'scope_sha256':bindings['scope_sha256'],
                         'parent':{'receipt_sha256':anchor['receipt_sha256'],'adapter_sha256':anchor['adapter_sha256']},'budget_sha256':core.identity(budget)}
        identity_path=output/'run_identity_PRIVATE.json'
        if identity_path.exists():require(_json(identity_path)==identity_record,'Run identity (source/input/scope/parent/budget) differs from the original run')
        else:core.publish_once(identity_path,identity_record)
        def validated(bundle_value):
            loaded=core.load_bundle(bundle_value)
            core.validate_bundle_against_run(loaded,anchor=anchor,rows=rows,expected_image_sha256=binding['image_sha256'],pair_id=pair,reward_policy_sha256=reward_policy_sha256)
            return loaded
        # ----- on-disk state, BEFORE any backend exists
        bundle_marker=output/'bundle_COMMITTED.json';bundle=None
        if bundle_marker.exists():
            marker=_json(bundle_marker);path=output/'bundle_PRIVATE.json'
            require(path.is_file() and old.digest(path)==marker['file_sha256'],'Committed bundle bytes changed')
            bundle=validated(_json(path));require(bundle.sha256==marker['bundle_sha256'] and marker['binding_sha256']==binding_sha,'Bundle belongs to a different run')
        elif (output/'rollout_reservation_PRIVATE.json').exists():
            raise RuntimeError('INCOMPLETE_ROLLOUT: reserved generation without committed bundle; no automatic generation retry')
        status={arm:_arm_status(output,arm,arm_bindings[arm]) for arm in ARM_ORDER}

        def summary(name,constructed=True):
            updates=sum(bool(_json(output/f'arm_{a}_COMMITTED.json').get('updated')) for a in ARM_ORDER if status[a]=='COMMITTED')
            used=_json(output/'counters_PRIVATE.json')['used'] if (output/'counters_PRIVATE.json').exists() else None
            return {'status':name,'experiment':core.EXPERIMENT,'arms':dict(status),'bundle_sha256':bundle.sha256 if bundle else None,'native_backend_constructed':constructed,
                    'optimizer_updates':updates,'counters':used,'counters_scope':'ALL_COUNTERS_OF_THIS_COMBINED_RUN','recipe':core.RECIPE_LABEL,'science_claim':False}
        if bundle is not None and all(v=='COMMITTED' for v in status.values()):return summary('SCHEDULE_COMPLETE',False)
        if expired():return summary('WALL_BUDGET_REACHED',False)
        # ----- counters (persisted on every charge, restored on resume; nothing resets)
        limits=core.counter_limits(budget);counters_path=output/'counters_PRIVATE.json'
        def persist(used):core.replace_atomic(counters_path,{'binding_sha256':binding_sha,'limits':limits,'used':used})
        if counters_path.exists():
            saved=_json(counters_path);require(saved['binding_sha256']==binding_sha and saved['limits']==limits,'Counter journal belongs to a different scope')
            counters=core.Counters(limits,saved['used'],persist)
        else:
            counters=core.Counters(limits,None,persist);persist(counters.snapshot())
        authorize({'phase':'load','binding':binding,'budget':budget})
        require(not expired(),'Deadline exhausted during backend resource hold')
        parent_disk=core.parent_disk_snapshot(parent_checkpoint)       # parent disk is immutable: hashed now, compared at the end (and best-effort in STOP records)
        adapter=Path(parent_checkpoint)/'adapter.safetensors'
        backend=backend_factory(adapter,configuration)
        try:
            expected={'run':anchor['original_binding'],'adapter_sha256':old.digest(adapter),'trainable_signature':backend.trainable_signature()}
            state,rng,_=load_optimizer_checkpoint(Path(parent_checkpoint)/'optimizer',expected_bindings=expected,load_arrays=backend.load_arrays)
            backend.restore(state,rng);state=rng=None   # restored after setup, immediately before the first stochastic action
            require(old._step(backend)==anchor['updates'],'Adam52 counter continuity lost')
            backend.configure_shared(counters=counters,rows=rows,parent=anchor,binding_sha256=binding_sha,expected_image_sha256=binding['image_sha256'])
            if bundle is not None:backend.bind_bundle(bundle)
            parent_snapshot=backend.transaction_snapshot();parent_identity=core.snapshot_identity(parent_snapshot)
            parent_rng=backend.rng_digest();parent_optimizer=backend.optimizer_digest()

            def restore_parent(label):
                backend.transaction_restore(parent_snapshot)
                require(backend.rng_digest()==parent_rng and backend.optimizer_digest()==parent_optimizer,'Parent Adam/RNG restore not exact after '+label)

            def audit(label,*,adam=True,rng=True):
                """Pre-restore check: the (parameter-only) transaction validator cannot see Adam/RNG, and restoring would hide a fault."""
                require(not adam or backend.optimizer_digest()==parent_optimizer,f'{label}: Adam state changed (pre-restore audit)')
                require(not rng or backend.rng_digest()==parent_rng,f'{label}: RNG changed (pre-restore audit)')

            def stop(name,error,**extra):
                try:core.require_parent_unchanged(parent_disk,parent_checkpoint);disk=True
                except BaseException:disk=False
                core.replace_atomic(output/f'{name}_{time.time_ns()}_STOP_PRIVATE.json',{'error_type':type(error).__name__,'error':str(error),'counters':counters.snapshot(),'parent_disk_unchanged':disk,**extra})
            # ----- integrated minimal parent restore + read-only fixed preflight (no forward, generation, backward or update)
            authorize({'phase':'preflight','binding':binding})
            try:
                restore_parent('preflight');audit('preflight')
                require(core.snapshot_identity(backend.transaction_snapshot())==parent_identity,'Restored full parameter+Adam+RNG identity differs from the parent snapshot')
                readback_dir=Path(tempfile.mkdtemp(prefix='.preflight-',dir=output))
                try:
                    backend.save_adapter(readback_dir/'adapter.safetensors');readback=old.digest(readback_dir/'adapter.safetensors')
                finally:shutil.rmtree(readback_dir,ignore_errors=True)
                audit('preflight_readback')
                preflight={'binding_sha256':binding_sha,'kind':'STATE_ONLY_RESTORE_PREFLIGHT_NO_NUMERICAL_PROBABILITY_CHECK','adam_step':old._step(backend),
                    'trainable_tensors':len(backend.trainable_signature()),'parent_identity':parent_identity,'full_state_identity_equals_parent':True,
                    'rng_digest':parent_rng,'optimizer_digest':parent_optimizer,'parent_adapter_sha256':anchor['adapter_sha256'],'adapter_readback_sha256':readback,
                    'adapter_readback_equals_parent_adapter':readback==anchor['adapter_sha256'],
                    'readback_note':'Informational only: native full parameters+Adam/RNG equality is the criterion; safetensors serialization may legitimately differ byte-wise.',
                    'forwards':0,'generation':0,'backward':0,'optimizer_updates':0,'model_numerical_PASS':False,
                    'numerical_check':'INTEGRATED_SHARED8_ONPOLICY_OWN_OLD_PARITY_.002_BEFORE_ANY_STEP (no separate pre-rollout probability probe: it would need extra F32 forwards and add no information the on-policy parity does not)',
                    'scope':'INTEGRATED_PREFLIGHT_NOT_A_QUALIFICATION_PHASE_NOT_A_MODEL_PASS'}
                require(preflight['adam_step']==anchor['updates'],'Preflight Adam counter differs from the frozen parent')
                preflight_path=output/'preflight_PRIVATE.json'
                if preflight_path.exists():require(_json(preflight_path)==preflight,'Preflight differs from the original run preflight')
                else:core.publish_once(preflight_path,preflight)
            except BaseException as error:
                stop('preflight',error);raise
            # ----- ONE rollout of F4+T4
            if bundle is None:
                require(not expired(),'Deadline before rollout');authorize({'phase':'rollout','binding':binding})
                core.publish_once(output/'rollout_reservation_PRIVATE.json',{'binding_sha256':binding_sha,'parent_identity':parent_identity,'budget_sha256':core.identity(budget)})
                try:
                    work=output/'rollout_work_PRIVATE';work.mkdir()
                    captured=backend.generate_shared_rollout(rows,work)
                    require(backend.optimizer_digest()==parent_optimizer,'Rollout changed Adam')
                    require(backend.rng_digest()!=parent_rng,'Rollout did not consume RNG: post-generation R1 evidence would equal R0')
                    backend.transaction_validate(parent_snapshot,False)
                    # Post-generation RNG R1 is persisted as evidence; BOTH arms deliberately restart from the exact parent R0.
                    r1_state=backend.rng_state();layout,arrays=pack_tree(r1_state);r1_file=output/'rng_R1_post_generation_PRIVATE.safetensors'
                    backend.save_arrays(r1_file,arrays)
                    with r1_file.open('rb') as handle:os.fsync(handle.fileno())
                    r1_digest=backend.rng_digest()
                    core.publish_once(output/'rng_R1_PRIVATE.json',{'layout':layout,'tensors_file':r1_file.name,'tensors_sha256':old.digest(r1_file),'R0_rng_digest':parent_rng,
                        'R1_rng_digest':r1_digest,'R1_differs_from_R0':r1_digest!=parent_rng,'arm_start_policy':'BOTH_ARMS_RESTORE_R0_NOT_R1'})
                    items=[]
                    for modality,row in rows.items():
                        require(old.digest(row['images'][0])==captured['contexts'][modality]['image_sha256'],'Rollout image differs from frozen input')
                        raw=captured['raw'][modality]
                        for index,(response,grade) in enumerate(zip(raw['responses'],old._grades(row,raw,configuration,grader))):
                            items.append({'response_id':f'{modality}{index}','text':response['text'],'token_ids':response['token_ids'],'raw_selected_logps':response['raw_selected_logps'],
                                'finish_reason':response['finish_reason'],'EOS_coverage':response['EOS_coverage'],'generation_tokens':response['generation_tokens'],'grade':grade})
                    contexts={m:{**captured['contexts'][m],'row_id':rows[m]['id'],'pair_id':pair,'checkpoint_sha256':anchor['adapter_sha256'],
                        'policy_sha256':core.identity(core.POLICY),'reward_policy_sha256':reward_policy_sha256} for m in ('F','T')}
                    frozen_bundle=core.freeze_bundle(parent={'receipt_sha256':anchor['receipt_sha256'],'adapter_sha256':anchor['adapter_sha256']},contexts=contexts,
                        responses=sorted(items,key=lambda i:core.RESPONSE_IDS.index(i['response_id'])),
                        effective_eos_ids={m:captured['raw'][m]['responses'][0]['effective_eos_ids_PRIVATE'] for m in ('F','T')})
                    bundle=validated(frozen_bundle);backend.bind_bundle(bundle)
                    core.publish_once(output/'bundle_PRIVATE.json',frozen_bundle)
                    core.publish_once(bundle_marker,{'bundle_sha256':bundle.sha256,'binding_sha256':binding_sha,'file_sha256':old.digest(output/'bundle_PRIVATE.json')})
                    authorize({'phase':'post_rollout','binding':binding})
                    backend.transaction_validate(parent_snapshot,False);restore_parent('rollout')
                except BaseException as error:
                    stop('rollout',error);raise
            # ----- own-old scoring / parity once (committed; reused on resume, never recharged)
            old_path=output/'old_PRIVATE.json'
            if old_path.exists():
                saved=_json(old_path);require(saved['bundle_sha256']==bundle.sha256 and saved['parent_identity']==parent_identity,'Committed own-old belongs to a different bundle/parent')
                own_old=saved['old'];parity=core.parity_report(bundle,own_old);core.check_old_shapes(bundle,own_old)     # re-derived on CPU, never trusted from the file's own claims
            else:
                require(not expired(),'Deadline before own-old scoring');authorize({'phase':'own_old','binding':binding})
                try:
                    own_old=backend.score_old(bundle);audit('own_old');core.check_old_shapes(bundle,own_old);parity=core.parity_report(bundle,own_old)
                    backend.transaction_validate(parent_snapshot,False);restore_parent('own_old')
                    core.publish_once(old_path,{'bundle_sha256':bundle.sha256,'parent_identity':parent_identity,'old':own_old,'rollout_vs_own_context_parity':parity})
                except BaseException as error:
                    stop('own_old',error);raise
            parity_max=max(v['max_abs_error'] for v in parity.values())
            # ----- PHASE 1: both gradients, archived and compared; NO optimizer step may occur before both are durable
            stopped=None
            gradient_dir=output/'gradients_PRIVATE'
            metas={arm:gradient_dir/f'arm_{arm}_gradient_meta_PRIVATE.json' for arm in ARM_ORDER}
            if any(v!='COMMITTED' for v in status.values()):
                for arm in ARM_ORDER:
                    if metas[arm].exists():continue                           # the archive metadata link is the gradient commit marker
                    reservation=output/f'grad_{arm}_reservation_PRIVATE.json'
                    if reservation.exists():raise RuntimeError(f'INCOMPLETE_GRADIENT_{arm}: uncommitted gradient attempt preserved; no automatic retry')
                    if expired():stopped='WALL_BUDGET_REACHED';break
                    planned=core.arm_plan(bundle,arm)
                    authorize({'phase':'gradient','arm':arm,'binding':binding})
                    core.publish_once(reservation,{'binding_sha256':binding_sha,'arm':arm,'bundle_sha256':bundle.sha256,'parent_identity':parent_identity})
                    try:
                        restore_parent('gradient_'+arm+'_start')
                        gradient,gradient_evidence=backend.arm_gradient(planned,own_old)
                        # BEFORE any restore: Adam/RNG exactness AND the full trainable-parameter transaction check (a backward that secretly edits weights is a fault)
                        audit('arm_gradient_'+arm)
                        integrity=backend.transaction_validate(parent_snapshot,False)
                        norm=float(gradient_evidence.get('gradient_l2',0.)) if gradient is not None else 0.
                        decision=core.update_decision(planned,norm)
                        require(gradient is not None or not decision['update'],'Update decision without a gradient')
                        require(gradient_evidence.get('backward_calls',0)==planned.backward_calls_needed,'Backward calls differ from admitted responses')
                        require(parity_max<=core.ROLLOUT_TOLERANCE,'Own-old parity (.002) must hold before any gradient is trusted')
                        core.archive_gradient(backend=backend,directory=gradient_dir,arm=arm,planned=planned,gradient=gradient,evidence=gradient_evidence,bundle_sha256=bundle.sha256,
                            parent_identity=parent_identity,anchor=anchor,parent_path=parent_checkpoint,profile=configuration,decision=decision,
                            audit={'transaction_validate_unchanged_parameters':True,'transaction_validate':integrity,'adam_digest_equals_parent':True,'rng_digest_equals_parent':True,'before_restore':True})
                        gradient=None;restore_parent('gradient_'+arm+'_end')
                    except BaseException as error:
                        rollback=None
                        try:restore_parent('gradient_'+arm+'_failure')
                        except BaseException as restore_error:rollback=str(restore_error)
                        stop(f'gradient_{arm}',error,rolled_back_model_Adam_RNG=rollback is None,rollback_error=rollback,optimizer_updates_so_far=0)
                        raise
                if stopped is None:
                    comparison_path=output/'comparison_PRIVATE.json'
                    authorize({'phase':'comparison','binding':binding})
                    comparison=core.comparison_record(metas,backend.load_arrays)
                    if comparison_path.exists():require(_json(comparison_path)==comparison,'Stored gradient comparison differs from the re-derived comparison')
                    else:core.publish_once(comparison_path,comparison)
            # ----- PHASE 2: only now, one serial step per arm from the identical restored parent and the gradient READ BACK from disk
            for arm in ARM_ORDER:
                if stopped:break
                if status[arm]=='COMMITTED':continue
                if expired():stopped='WALL_BUDGET_REACHED';break
                planned=core.arm_plan(bundle,arm);arm_dir=output/f'arm_{arm}';arm_binding=arm_bindings[arm]
                core.require_parent_unchanged(parent_disk,parent_checkpoint)       # nothing may touch the immutable parent between arms
                authorize({'phase':'arm','arm':arm,'binding':binding})
                core.publish_once(output/f'arm_{arm}_reservation_PRIVATE.json',{'binding_sha256':binding_sha,'arm':arm,'bundle_sha256':bundle.sha256,'parent_identity':parent_identity})
                try:
                    restore_parent('arm_'+arm+'_start')
                    gate={'effective_groups':planned.admitted_groups,'admitted_response_ids':list(planned.admitted_response_ids)}
                    record,gradient=core.load_gradient_tree(metas[arm],backend.load_arrays)
                    require(record['bundle_sha256']==bundle.sha256 and record['parent_identity']==parent_identity and record['arm']==arm,'Archived gradient belongs to a different bundle/parent/arm')
                    norm=float(record['gradient_l2_reported']) if gradient is not None else 0.
                    decision=core.update_decision(planned,norm)
                    require(decision==record['decision'] and (gradient is not None or not decision['update']),'Update decision differs from the archived decision')
                    gradient_evidence={'gradient_l2':norm,'weighted_loss':record['weighted_loss'],'backward_calls':record['backward_calls']}
                    require(parity_max<=core.ROLLOUT_TOLERANCE,'Own-old parity (.002) must hold before any optimizer step')
                    authorize({'phase':'pre_update','arm':arm,'binding':binding})
                    before_step=old._step(backend);updated=False
                    arm_dir.mkdir()
                    current=old._publish(arm_dir,Path(tempfile.mkdtemp(prefix='.pending-',dir=arm_dir)),backend,arm_binding,0,0,None,
                                         {'updated':False,'record_id':None,'schedule_record_id':None})
                    work=Path(tempfile.mkdtemp(prefix='.pending-',dir=arm_dir))
                    before_state=old.state_fingerprint(backend,backend.optimizer_state(),work/'optimizer_before.safetensors')
                    update_evidence={}
                    if decision['update']:
                        counters.charge('optimizer_updates')
                        update_evidence=backend.apply_update(gradient);updated=True
                    require(old._step(backend)==before_step+int(updated),'More than one Adam call or hidden update')
                    audit('optimizer_update_'+arm,adam=not updated)            # one Adam step may change moments, never the parent RNG
                    require(not updated or backend.optimizer_digest()!=parent_optimizer,'Updated arm left Adam unchanged')
                    after_state=old.state_fingerprint(backend,backend.optimizer_state(),work/'optimizer_after.safetensors')
                    if not updated:require(before_state==after_state,'No-update arm changed Adam')
                    check=work/'verify_adapter.safetensors';backend.save_adapter(check)
                    changed=old.digest(check)!=current['files_sha256']['adapter.safetensors'];check.unlink()
                    require(changed==updated,'Adapter/actual Adam transition mismatch')
                    integrity=backend.transaction_validate(parent_snapshot,updated)
                    (work/'rollout_PRIVATE.json').write_text(old.canonical(bundle.raw)+'\n')
                    current=old._publish(arm_dir,work,backend,arm_binding,1,int(updated),old.digest(current['path']/'receipt.json'),
                        {'updated':updated,'record_id':rows['F']['id'],'schedule_record_id':rows['F']['id'],'arm':arm,
                         'attempted_K4_groups':2,'bundle_sha256':bundle.sha256,'gate':gate,'update_decision':decision,
                         'gradient_evidence':gradient_evidence,'update_evidence':update_evidence,
                         'gradient_archive_meta_sha256':old.digest(metas[arm]),'gradient_archive_tensors_sha256':record['tensors_sha256'],
                         'adam_recurrence':{'step_before':before_step,'step_after':old._step(backend),'optimizer_profile':configuration['optimizer'],
                             'parent_optimizer_state_sha256':anchor['optimizer_state_sha256'],'parent_optimizer_manifest_sha256':anchor['optimizer_manifest_sha256'],
                             'recompute':'adam_next_state(parent optimizer/ state, gradient archive) must equal this boundary optimizer/ state; delta equals adapter change'},
                         'own_old_parity_max_abs':parity_max,'optimizer_before_fingerprint':before_state,'optimizer_after_fingerprint':after_state,'integrity':integrity,
                         'rng_policy':'ARM_STARTS_FROM_EXACT_PARENT_R0_NOT_POST_GENERATION_R1','arm_start_rng_digest':parent_rng})
                    core.publish_once(output/f'arm_{arm}_COMMITTED.json',{'arm':arm,'updated':updated,'boundary_receipt_sha256':old.digest(current['path']/'receipt.json'),'bundle_sha256':bundle.sha256})
                    status[arm]='COMMITTED';gradient=None
                    restore_parent('arm_'+arm+'_end')
                except BaseException as error:
                    committed=committed_boundary(output,arm,arm_binding)   # the boundary_0001 rename is the commit, whatever failed afterwards
                    if committed is not None and not (output/f'arm_{arm}_COMMITTED.json').exists():
                        try:core.publish_once(output/f'arm_{arm}_COMMITTED.json',{'arm':arm,'updated':bool(committed['updated']),'boundary_receipt_sha256':old.digest(committed['path']/'receipt.json'),'recovered_without_replay':True})
                        except BaseException:pass
                    visible=committed is not None or (output/f'arm_{arm}_COMMITTED.json').exists();rollback=None
                    if not visible:
                        try:restore_parent('arm_'+arm+'_failure')
                        except BaseException as restore_error:rollback=str(restore_error)
                    stop(f'arm_{arm}',error,committed=visible,rolled_back_model_Adam_RNG=(not visible and rollback is None),rollback_error=rollback)
                    raise
            # ----- end-of-run exact original state + immutable parent disk checks
            require(core.snapshot_identity(backend.transaction_snapshot())==parent_identity,'Original parent weights/Adam/Python+MLX RNG not restored at end')
            core.require_parent_unchanged(parent_disk,parent_checkpoint)
            done=stopped or ('SCHEDULE_COMPLETE' if all(v=='COMMITTED' for v in status.values()) else 'BOUNDED_STAGE_COMPLETE')
            result=summary(done)
            core.replace_atomic(output/'complete_SAFE.json',{**result,'final_state_equals_parent':True,'parent_disk_unchanged':True,'subtype':core.EXECUTION_SUBTYPE})
            return result
        finally:backend.close()
