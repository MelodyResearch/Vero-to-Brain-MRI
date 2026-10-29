"""Backend-neutral crash-consistent driver for the shared-rollout one-step GSPO control.

IMPLEMENTED_CPU_NOT_NATIVE_QUALIFIED. One resident backend/optimizer. Order: restore
exact Adam60 parent -> ONE F4+T4 rollout (reservation first, immutable bundle committed) ->
parent restored -> own-old scoring/parity once (committed) -> for arm A then B: restore
identical full parent state, compute that arm's per-response gradients, AT MOST ONE
optimizer update, publish through the existing atomic boundary bundle, once-only commit
marker, restore parent again. Qualification mode computes both arms' gradients with zero
updates. Reservations without commits are never retried automatically.
Reuses (does not modify) paired_z045_continuation_driver_v1 publication/inspection.
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
from mri_rl import shared_rollout_gspo_core_v1 as core
from mri_rl.optimizer_checkpoint import load_optimizer_checkpoint,pack_tree
from mri_rl.uniform_balance_continuation_gate_v1 import parent_anchor

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
        if 'boundary_receipt_sha256' in record:
            last=old.inspect_boundaries(arm_dir,arm_binding)
            require(last is not None and last['completed_groups']==1 and old.digest(last['path']/'receipt.json')==record['boundary_receipt_sha256'],'Committed arm terminal differs from marker')
        else:
            evidence=output/f'qualification_arm_{arm}_PRIVATE.json'
            require(evidence.is_file() and old.digest(evidence)==record['evidence_sha256'],'Committed qualification arm evidence changed')
        return 'COMMITTED'
    last=committed_boundary(output,arm,arm_binding)
    if last is not None:
        # The boundary_0001 rename IS the commit; only the marker is missing. Record it, never replay (leftover pending dirs are harmless).
        core.publish_once(marker,{'arm':arm,'updated':bool(last['updated']),'boundary_receipt_sha256':old.digest(last['path']/'receipt.json'),'recovered_without_replay':True})
        return 'COMMITTED'
    if arm_dir.exists() or reservation.exists():raise RuntimeError(f'INCOMPLETE_ARM_{arm}: uncommitted attempt preserved; no automatic retry')
    return 'ABSENT'


def run(*,output,mode,schedule,bindings,backend_factory,authorize,parent_checkpoint,parent_receipt_sha256,grader,
        reward_policy_sha256,budget,resume=False,config=None,row_validator=None,
        clock=time.monotonic,wall_clock=time.time,adopt=None):
    """Execute (or resume a committed prefix of) the shared-rollout experiment. Fixture backends are explicit.

    authorize(context) must independently validate the exact admission scope and resources; it is called
    before backend construction and before every phase. `adopt` ({'run':qualification run directory,'proof':
    reviewed qualification proof}) lets a separately authorized TRAIN run reuse the qualification's committed
    F4+T4 bundle and own-old logps without any new generation; new counters then exclude that inherited work.
    """
    require(callable(authorize) and mode in core.MODES,'Independent authorizer and known mode required')
    require(adopt is None or mode=='train','Bundle adoption is a train-mode action')
    require(isinstance(bindings,dict) and isinstance(bindings.get('source_sha256'),dict) and bindings['source_sha256'] and core.sha(bindings.get('input_sha256')),
            'Run must bind the actual reviewed source closure and input hash')
    require(core.sha(reward_policy_sha256),'Exact task-routed reward calibration identity required')
    rows=_modality_rows(schedule)
    require(isinstance(budget,dict) and budget==core.derive_budget(budget.get('prompt_lengths'),mode),'Budget is not the exact derived shared-rollout budget')
    maximum_seconds=budget['maximum_seconds'];started=clock();absolute=wall_clock()
    anchor=parent_anchor(parent_checkpoint,parent_receipt_sha256)
    configuration=json.loads(old.canonical(config or {**old.PROFILE,'group_size':core.GROUP_SIZE,'max_tokens':core.MAX_TOKENS}))
    for key,value in old.PROFILE.items():
        if key not in ('group_size','max_tokens'):require(configuration.get(key)==value,'Historical optimizer/reward/LoRA/LR profile changed')
    require(configuration['group_size']==core.GROUP_SIZE and configuration['max_tokens']==core.MAX_TOKENS,'Frozen K4/256 required')
    experiment={**bindings,'adaptive_parent':anchor,'experiment':core.EXPERIMENT,'execution_subtype':core.EXECUTION_SUBTYPE,'mode':mode,
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
        except BlockingIOError:raise RuntimeError('Another shared-rollout writer is active') from None
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
        identity_record={'binding_sha256':binding_sha,'mode':mode,'source_sha256':bindings['source_sha256'],'input_sha256':bindings['input_sha256'],
                         'parent':{'receipt_sha256':anchor['receipt_sha256'],'adapter_sha256':anchor['adapter_sha256']},'budget_sha256':core.identity(budget)}
        identity_path=output/'run_identity_PRIVATE.json'
        if identity_path.exists():require(_json(identity_path)==identity_record,'Run identity (source/input/parent/budget) differs from the original run')
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
        elif (output/'adoption_reservation_PRIVATE.json').exists():
            raise RuntimeError('INCOMPLETE_ADOPTION: reserved bundle adoption without commit; no automatic retry')
        adopted=(output/'adoption_PRIVATE.json').exists() or adopt is not None
        status={arm:_arm_status(output,arm,arm_bindings[arm]) for arm in ARM_ORDER}

        def summary(name,constructed=True):
            updates=sum(bool(_json(output/f'arm_{a}_COMMITTED.json').get('updated')) for a in ARM_ORDER if status[a]=='COMMITTED')
            used=_json(output/'counters_PRIVATE.json')['used'] if (output/'counters_PRIVATE.json').exists() else None
            inherited=_json(output/'adoption_PRIVATE.json')['inherited_counters'] if (output/'adoption_PRIVATE.json').exists() else None
            return {'status':name,'mode':mode,'arms':dict(status),'bundle_sha256':bundle.sha256 if bundle else None,'native_backend_constructed':constructed,
                    'optimizer_updates':updates,'counters':used,'counters_scope':'NEW_COUNTERS_OF_THIS_RUN_ONLY','inherited_counters':inherited,'science_claim':False}
        if bundle is not None and all(v=='COMMITTED' for v in status.values()):return summary('SCHEDULE_COMPLETE',False)
        if expired():return summary('WALL_BUDGET_REACHED',False)
        # ----- counters (persisted on every charge, restored on resume; nothing resets)
        limits=core.adopted_limits(budget) if adopted else core.counter_limits(budget);counters_path=output/'counters_PRIVATE.json'
        def persist(used):core.replace_atomic(counters_path,{'binding_sha256':binding_sha,'limits':limits,'used':used})
        if counters_path.exists():
            saved=_json(counters_path);require(saved['binding_sha256']==binding_sha and saved['limits']==limits,'Counter journal belongs to a different scope')
            counters=core.Counters(limits,saved['used'],persist)
        else:
            counters=core.Counters(limits,None,persist);persist(counters.snapshot())
        authorize({'phase':'load','mode':mode,'binding':binding,'budget':budget})
        require(not expired(),'Deadline exhausted during backend resource hold')
        adapter=Path(parent_checkpoint)/'adapter.safetensors'
        backend=backend_factory(adapter,configuration)
        try:
            expected={'run':anchor['original_binding'],'adapter_sha256':old.digest(adapter),'trainable_signature':backend.trainable_signature()}
            state,rng,_=load_optimizer_checkpoint(Path(parent_checkpoint)/'optimizer',expected_bindings=expected,load_arrays=backend.load_arrays)
            backend.restore(state,rng);state=rng=None   # restored after setup, immediately before the first stochastic action
            require(old._step(backend)==anchor['updates'],'Adam60 counter continuity lost')
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
                core.replace_atomic(output/f'{name}_{time.time_ns()}_STOP_PRIVATE.json',{'error_type':type(error).__name__,'error':str(error),'counters':counters.snapshot(),**extra})
            # ----- adoption of a reviewed qualification's committed bundle + own-old (no generation, no rescoring)
            if bundle is None and adopt is not None:
                qrun=Path(adopt['run']);proof=adopt['proof']
                require(isinstance(proof,dict) and proof.get('status')=='ACTUAL_SHARED_ROLLOUT_GSPO_READONLY_QUALIFIED' and not proof.get('blockers'),'Adoption needs the effective qualification proof')
                qidentity=qrun/'run_identity_PRIVATE.json';qmarker=_json(qrun/'bundle_COMMITTED.json');qold=qrun/'old_PRIVATE.json';qr1=qrun/'rng_R1_PRIVATE.json'
                require(qidentity.is_file() and old.digest(qidentity)==proof['run_identity_sha256'],'Qualification run identity differs from the reviewed proof')
                recorded=_json(qidentity)
                require(recorded['mode']=='qualify' and recorded['source_sha256']==bindings['source_sha256'] and recorded['input_sha256']==bindings['input_sha256'] and
                        recorded['parent']==identity_record['parent'],'Qualification run source/input/parent differs from this train run')
                require(qmarker['bundle_sha256']==proof['bundle_sha256'] and old.digest(qrun/'bundle_PRIVATE.json')==qmarker['file_sha256'] and
                        qold.is_file() and old.digest(qold)==proof['old_sha256'] and qr1.is_file(),'Adopted bundle/own-old/R1 differ from the reviewed proof')
                require(old.digest(qr1)==proof.get('rng_R1_sha256'),'Adopted R1 metadata differs from the reviewed proof')
                r1=_json(qr1);r1_name=r1.get('tensors_file')
                require(isinstance(r1_name,str) and Path(r1_name).name==r1_name and (qrun/r1_name).is_file() and
                        old.digest(qrun/r1_name)==r1.get('tensors_sha256'),'Adopted R1 tensor payload differs from its frozen metadata')
                candidate=validated(_json(qrun/'bundle_PRIVATE.json'));saved_old=_json(qold)
                require(saved_old['bundle_sha256']==candidate.sha256 and saved_old['parent_identity']==parent_identity,'Qualification own-old belongs to a different bundle/parent state')
                # Completion/coverage/parity are RE-DERIVED from the qualification run's own files, never from the proof's or caller's claims.
                require(core.qualification_blockers(qrun)==[],'Qualification run lacks an effective finite nonzero backward path in both arms')
                core.check_old_shapes(candidate,saved_old['old']);core.parity_report(candidate,saved_old['old'])
                inherited=_json(qrun/'counters_PRIVATE.json')['used']
                core.publish_once(output/'adoption_reservation_PRIVATE.json',{'binding_sha256':binding_sha,'bundle_sha256':candidate.sha256})
                shutil.copyfile(qrun/r1_name,output/r1_name)
                core.publish_once(output/'rng_R1_PRIVATE.json',r1)
                core.publish_once(output/'old_PRIVATE.json',saved_old)
                core.publish_once(output/'bundle_PRIVATE.json',_json(qrun/'bundle_PRIVATE.json'))
                core.publish_once(output/'adoption_PRIVATE.json',{'qualification_run_identity_sha256':proof['run_identity_sha256'],'bundle_sha256':candidate.sha256,'old_sha256':proof['old_sha256'],
                    'release_witness_sha256':proof['release_witness_sha256'],'inherited_counters':inherited,'generation_in_this_run':False,'new_generation_charged':0})
                core.publish_once(bundle_marker,{'bundle_sha256':candidate.sha256,'binding_sha256':binding_sha,'file_sha256':old.digest(output/'bundle_PRIVATE.json'),'adopted':True})
                bundle=candidate;backend.bind_bundle(bundle)
            # ----- ONE rollout of F4+T4
            if bundle is None:
                require(not expired(),'Deadline before rollout');authorize({'phase':'rollout','mode':mode,'binding':binding})
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
                    frozen=core.freeze_bundle(parent={'receipt_sha256':anchor['receipt_sha256'],'adapter_sha256':anchor['adapter_sha256']},contexts=contexts,
                        responses=sorted(items,key=lambda i:core.RESPONSE_IDS.index(i['response_id'])),
                        effective_eos_ids={m:captured['raw'][m]['responses'][0]['effective_eos_ids_PRIVATE'] for m in ('F','T')})
                    bundle=validated(frozen);backend.bind_bundle(bundle)
                    core.publish_once(output/'bundle_PRIVATE.json',frozen)
                    core.publish_once(bundle_marker,{'bundle_sha256':bundle.sha256,'binding_sha256':binding_sha,'file_sha256':old.digest(output/'bundle_PRIVATE.json')})
                    authorize({'phase':'post_rollout','mode':mode,'binding':binding})
                    backend.transaction_validate(parent_snapshot,False);restore_parent('rollout')
                except BaseException as error:
                    stop('rollout',error);raise
            # ----- own-old scoring / parity once (committed; reused on resume, never recharged)
            old_path=output/'old_PRIVATE.json'
            if old_path.exists():
                saved=_json(old_path);require(saved['bundle_sha256']==bundle.sha256 and saved['parent_identity']==parent_identity,'Committed own-old belongs to a different bundle/parent')
                own_old=saved['old'];core.parity_report(bundle,own_old);core.check_old_shapes(bundle,own_old)     # re-derived on CPU, never trusted from the file's own claims
            else:
                require(not expired(),'Deadline before own-old scoring');authorize({'phase':'own_old','mode':mode,'binding':binding})
                try:
                    own_old=backend.score_old(bundle);audit('own_old');core.check_old_shapes(bundle,own_old);parity=core.parity_report(bundle,own_old)
                    backend.transaction_validate(parent_snapshot,False);restore_parent('own_old')
                    core.publish_once(old_path,{'bundle_sha256':bundle.sha256,'parent_identity':parent_identity,'old':own_old,'rollout_vs_own_context_parity':parity})
                except BaseException as error:
                    stop('own_old',error);raise
            # ----- arms: identical parent each time, at most one update each
            stopped=None
            for arm in ARM_ORDER:
                if status[arm]=='COMMITTED':continue
                if expired():stopped='WALL_BUDGET_REACHED';break
                planned=core.arm_plan(bundle,arm);arm_dir=output/f'arm_{arm}';arm_binding=arm_bindings[arm]
                authorize({'phase':'arm','arm':arm,'mode':mode,'binding':binding})
                core.publish_once(output/f'arm_{arm}_reservation_PRIVATE.json',{'binding_sha256':binding_sha,'arm':arm,'bundle_sha256':bundle.sha256,'parent_identity':parent_identity})
                try:
                    restore_parent('arm_'+arm+'_start')
                    gate={'effective_groups':planned.admitted_groups,'admitted_response_ids':list(planned.admitted_response_ids)}
                    gradient,gradient_evidence=backend.arm_gradient(planned,own_old)
                    audit('arm_gradient_'+arm)             # deterministic backward must leave Adam moments and RNG exactly at the parent
                    norm=float(gradient_evidence.get('gradient_l2',0.)) if gradient is not None else 0.
                    decision=core.update_decision(planned,norm)
                    require(gradient is not None or not decision['update'],'Update decision without a gradient')
                    require(gradient_evidence.get('backward_calls',0)==planned.backward_calls_needed,'Backward calls differ from admitted responses')
                    authorize({'phase':'pre_update','arm':arm,'mode':mode,'binding':binding})
                    before_step=old._step(backend);updated=False
                    if mode=='train':
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
                             'optimizer_before_fingerprint':before_state,'optimizer_after_fingerprint':after_state,'integrity':integrity,
                             'rng_policy':'ARM_STARTS_FROM_EXACT_PARENT_R0_NOT_POST_GENERATION_R1','arm_start_rng_digest':parent_rng})
                        core.publish_once(output/f'arm_{arm}_COMMITTED.json',{'arm':arm,'updated':updated,'boundary_receipt_sha256':old.digest(current['path']/'receipt.json'),'bundle_sha256':bundle.sha256})
                    else:
                        require(before_step==anchor['updates'],'Qualification must not update')
                        evidence_path=output/f'qualification_arm_{arm}_PRIVATE.json'
                        core.publish_once(evidence_path,{'arm':arm,'gate':gate,'update_decision':decision,'gradient_evidence':gradient_evidence,'optimizer_updates':0,
                            'rng_policy':'ARM_STARTS_FROM_EXACT_PARENT_R0_NOT_POST_GENERATION_R1','arm_start_rng_digest':parent_rng})
                        require(old._step(backend)==before_step,'Qualification changed the optimizer counter')
                        backend.transaction_validate(parent_snapshot,False);audit('qualification_arm_'+arm)
                        core.publish_once(output/f'arm_{arm}_COMMITTED.json',{'arm':arm,'updated':False,'evidence_sha256':old.digest(evidence_path)})
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
            # ----- end-of-run exact original state check
            require(core.snapshot_identity(backend.transaction_snapshot())==parent_identity,'Original parent weights/Adam/Python+MLX RNG not restored at end')
            done=stopped or ('SCHEDULE_COMPLETE' if all(v=='COMMITTED' for v in status.values()) else 'BOUNDED_STAGE_COMPLETE')
            result=summary(done)
            core.replace_atomic(output/'complete_SAFE.json',{**result,'final_state_equals_parent':True,'subtype':core.EXECUTION_SUBTYPE})
            return result
        finally:backend.close()
