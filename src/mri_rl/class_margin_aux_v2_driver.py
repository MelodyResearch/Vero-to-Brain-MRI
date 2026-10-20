"""New auxiliary transaction loop; reuses historical atomic publication/state APIs."""
from __future__ import annotations
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from mri_rl import paired_z045_continuation_driver_v1 as old
from mri_rl.optimizer_checkpoint import save_optimizer_checkpoint,load_optimizer_checkpoint
from mri_rl.continuous_rl import require
from mri_rl.class_margin_aux_v2 import require_update_budget,ComputeBudget


def _replace(path,value):
    temporary=path.with_name(path.name+'.partial')
    with temporary.open('w') as f:f.write(old.canonical(value)+'\n');f.flush();os.fsync(f.fileno())
    os.replace(temporary,path);old.sync_dir(path.parent)


def run(*,output,schedule,bindings,backend_factory,authorize,parent_checkpoint,parent_receipt_sha256,
        resume=False,max_new_groups=4,maximum_seconds=1200,config=None,row_validator=None,grader=None,
        ledger=None,input_tokens=None,clock=time.monotonic,wall_clock=time.time):
    """Four fixed slots, two TOTAL steps. No replay of an atomically committed update.

    A complete uncommitted rollout + postgeneration Adam/RNG is reusable; an
    interrupted incomplete rollout blocks this version (no automatic generation
    retry). Durable reservations/counters preserve spent work on every failure.
    """
    require(callable(authorize) and len(schedule)==4 and type(max_new_groups) is int and 1<=max_new_groups<=4,'Exact four-slot stage/authorizer required')
    require(math.isfinite(maximum_seconds) and 0<maximum_seconds<=1200,'Finite operation-start budget required')
    started=clock();absolute=wall_clock();anchor=old.parent_anchor(parent_checkpoint,parent_receipt_sha256)
    configuration=json.loads(old.canonical(config or {**old.PROFILE,'group_size':4,'max_tokens':256}))
    for key,value in old.PROFILE.items():
        if key not in ('group_size','max_tokens'):require(configuration.get(key)==value,'Historical optimizer/reward/LoRA profile changed')
    binding=old._binding(schedule,{**bindings,'adaptive_parent':anchor},configuration,row_validator,ledger.plan if ledger else None)
    output=Path(output);require(resume or not output.exists(),'Existing run requires explicit exact resume')
    require(not resume or output.is_dir(),'Missing resume output');output.mkdir(parents=True,exist_ok=True)
    with (output/'.writer.lock').open('a') as writer:
        fcntl.flock(writer,fcntl.LOCK_EX|fcntl.LOCK_NB)
        current=old.inspect_boundaries(output,binding)
        deadline_path=output/'operation_deadline_PRIVATE.json';binding_sha=hashlib.sha256(old.canonical(binding).encode()).hexdigest()
        if deadline_path.exists():
            deadline=json.loads(deadline_path.read_text());require(deadline['binding_sha256']==binding_sha and deadline['maximum_seconds']==maximum_seconds and
                deadline['deadline']==deadline['start']+maximum_seconds,'Original absolute deadline cannot reset')
        else:
            require(not resume,'Missing original deadline')
            deadline={'binding_sha256':binding_sha,'maximum_seconds':maximum_seconds,'start':absolute,'deadline':absolute+maximum_seconds}
            old.write_json(deadline_path,deadline);old.sync_dir(output)
        expired=lambda:clock()-started>=maximum_seconds or wall_clock()>=deadline['deadline']
        def result(status,native=True):
            return {'status':status,'completed_groups':current['completed_groups'] if current else 0,
                    'optimizer_updates':current['optimizer_updates'] if current else 0,'boundary':str(current['path']) if current else None,
                    'absolute_optimizer_updates':anchor['updates']+(current['optimizer_updates'] if current else 0),
                    'native_backend_constructed':native,'science_claim':False}
        if expired():return result('WALL_BUDGET_REACHED',False)
        require(not resume or current is not None,'No complete boundary to restore')
        cursor=current['completed_groups'] if current else 0
        if cursor==4:return result('SCHEDULE_COMPLETE',False)
        if current and current['optimizer_updates']>=2:return result('TOTAL_UPDATE_BUDGET_REACHED',False)
        authorize({'phase':'load','cursor':cursor,'binding':binding});require(not expired(),'Deadline exhausted during load resource hold')
        adapter=current['path']/'adapter.safetensors' if current else Path(parent_checkpoint)/'adapter.safetensors'
        backend=backend_factory(adapter,configuration)
        try:
            state_binding={'run':binding if current else anchor['original_binding'],'adapter_sha256':old.digest(adapter),'trainable_signature':backend.trainable_signature()}
            state,rng,_=load_optimizer_checkpoint((current['path'] if current else Path(parent_checkpoint))/'optimizer',expected_bindings=state_binding,load_arrays=backend.load_arrays)
            backend.restore(state,rng);require(old._step(backend)==anchor['updates']+(current['optimizer_updates'] if current else 0),'Adam60 counter continuity lost')
            state=rng=None
            if current is None:
                work=Path(tempfile.mkdtemp(prefix='.pending-',dir=output))
                current=old._publish(output,work,backend,binding,0,0,None,{'updated':False,'record_id':None,'schedule_record_id':None})
            if ledger:
                prior=current.get('ledger',{});ledger.used=prior.get('used_tokens',0);ledger.charged_groups=prior.get('charged_groups',0);ledger.imputed=prior.get('imputed_unknown_tokens',0)
            evidence_root=output/'attempt_evidence_PRIVATE';evidence_root.mkdir(exist_ok=True)
            for index in range(cursor,min(cursor+max_new_groups,4)):
                if expired():return result('WALL_BUDGET_REACHED')
                if current['optimizer_updates']>=2:return result('TOTAL_UPDATE_BUDGET_REACHED')
                row=schedule[index];authorize({'phase':'group','cursor':index,'binding':binding})
                if expired():return result('WALL_BUDGET_REACHED')
                require(old.digest(row['images'][0])==binding['image_sha256'][row['id']],'Image changed after resource gate')
                parent_sha=old.digest(current['path']/'receipt.json');pending=[]
                for directory in sorted(output.glob('.pending-*')):
                    reservation=directory/'attempt_reservation.json'
                    if not reservation.exists():continue
                    record=json.loads(reservation.read_text())
                    require(record.get('run_binding_sha256')==binding_sha,'Foreign pending reservation')
                    if record['parent_receipt_sha256']==parent_sha:
                        require(record['index']==index and record['record_id']==row['id'],'Pending cursor changed');pending.append(directory)
                require(len(pending)<=1,'Multiple attempts for the same uncommitted group; stop for review')
                reuse=bool(pending)
                if not reuse and ledger and not ledger.can_start(input_tokens(row)):return result('TOKEN_BUDGET_EXHAUSTED')
                work=pending[0] if reuse else Path(tempfile.mkdtemp(prefix='.pending-',dir=output))
                auxdir=evidence_root/work.name;auxdir.mkdir(exist_ok=True)
                journal_path=auxdir/'journal_PRIVATE.json'
                snapshot=backend.transaction_snapshot();before_step=old._step(backend);committed=False
                require(before_step==anchor['updates']+current['optimizer_updates'],'Optimizer continuity before group lost')
                try:
                    if reuse:
                        require(journal_path.exists() and (work/'rollout_PRIVATE.json').exists() and (auxdir/'post_rollout/manifest.json').exists(),
                                'INCOMPLETE_PENDING_ROLLOUT: preserved; no automatic retry')
                        journal=json.loads(journal_path.read_text());require(journal['binding_sha256']==binding_sha and journal['index']==index,'Pending evidence binding changed')
                        # Keep failed unpublished optimizer/adapter bytes; rebuild only the atomic staging bundle.
                        staged=auxdir/('failed_staging_'+str(time.time_ns()))
                        for name in ('adapter.safetensors','adapter_config.json','optimizer','optimizer_after.safetensors','receipt.json'):
                            source=work/name
                            if source.exists():staged.mkdir(exist_ok=True);os.rename(source,staged/name)
                        raw=json.loads((work/'rollout_PRIVATE.json').read_text());require(old.digest(work/'rollout_PRIVATE.json')==journal['rollout_sha256'],'Pending rollout bytes changed')
                        post_binding={'run':binding,'adapter_sha256':current['files_sha256']['adapter.safetensors'],'trainable_signature':backend.trainable_signature()}
                        state,rng,_=load_optimizer_checkpoint(auxdir/'post_rollout',expected_bindings=post_binding,load_arrays=backend.load_arrays)
                        backend.restore(state,rng);backend.activate_saved_rollout(row,raw);state=rng=None
                        if ledger:
                            ledger.used=journal['ledger_after_rollout']['used_tokens'];ledger.charged_groups=journal['ledger_after_rollout']['charged_groups'];ledger.imputed=journal['ledger_after_rollout']['imputed_unknown_tokens']
                    else:
                        old.write_json(work/'attempt_reservation.json',{'schema':'attempt-reservation-v1','run_binding_sha256':binding_sha,
                            'parent_receipt_sha256':parent_sha,'record_id':row['id'],'index':index,'image_sha256':binding['image_sha256'][row['id']],
                            'input_tokens':input_tokens(row) if input_tokens else None,'group_size':configuration['group_size'],'max_tokens':configuration['max_tokens']})
                        old.sync_dir(work);old.sync_dir(output)
                        old.state_fingerprint(backend,backend.optimizer_state(),work/'optimizer_before.safetensors')
                        raw=backend.rollout(row,anchor['groups']+index,work);old.write_json(work/'rollout_PRIVATE.json',raw)
                        require(raw['image_sha256']==binding['image_sha256'][row['id']] and old.digest(row['images'][0])==raw['image_sha256'],'Rollout image changed')
                        if ledger:ledger.charge(input_tokens(row),[len(r['token_ids']) for r in raw['responses']])
                        save_optimizer_checkpoint(auxdir/'post_rollout',optimizer_state=backend.optimizer_state(),rng_state=backend.rng_state(),
                            bindings={'run':binding,'adapter_sha256':current['files_sha256']['adapter.safetensors'],'trainable_signature':backend.trainable_signature()},
                            completed_groups=anchor['groups']+index,optimizer_updates=before_step,save_arrays=backend.save_arrays)
                        journal={'binding_sha256':binding_sha,'index':index,'rollout_sha256':old.digest(work/'rollout_PRIVATE.json'),
                                 'ledger_after_rollout':ledger.state() if ledger else None,'compute_used':current.get('compute_used',{'full_TF_forwards':0,'backward_calls':0,'teacherforced_positions':0})}
                        _replace(journal_path,journal)
                    def persist_compute():journal['compute_used']=dict(backend.compute.used);_replace(journal_path,journal)
                    if hasattr(backend,'compute'):
                        backend.compute=ComputeBudget(used=journal['compute_used']);backend.compute.persist=persist_compute
                    authorize({'phase':'post_rollout','cursor':index,'binding':binding});require(not expired(),'Deadline after generation; pending preserved for separately approved resume')
                    grades=old._grades(row,raw,configuration,grader);advantages=old.advantages_from_rewards([g['total'] for g in grades]);active=old.task_group_is_informative(grades)
                    before_state=old.state_fingerprint(backend,backend.optimizer_state(),auxdir/'pre_transition.safetensors')
                    update=(backend.transition(row,raw,advantages,active,baseline=snapshot) if hasattr(backend,'transaction_validate') else backend.transition(row,raw,advantages,active));updated=update.get('updated')
                    require(type(updated) is bool and updated==(update['mode']!='NO_UPDATE'),'Transition must record actual Adam branch status');require_update_budget(current['optimizer_updates'],2,update['mode'])
                    require(old._step(backend)==before_step+int(updated),'More than one Adam call or hidden update')
                    after_state=old.state_fingerprint(backend,backend.optimizer_state(),work/'optimizer_after.safetensors')
                    # Original atomic bundle requires before-state witness at fixed path.
                    require((work/'optimizer_before.safetensors').exists(),'Original pre-group optimizer witness missing')
                    if not updated:require(before_state==after_state,'NO_UPDATE changed optimizer')
                    check=work/'verify_adapter.safetensors';backend.save_adapter(check);changed=old.digest(check)!=current['files_sha256']['adapter.safetensors'];check.unlink()
                    require(changed==updated,'Adapter/actual Adam transition mismatch')
                    if hasattr(backend,'transaction_validate'):update['integrity']=backend.transaction_validate(snapshot,updated)
                    prior_accepted=current.get('accepted_GSPO_groups',0)
                    current=old._publish(output,work,backend,binding,index+1,current['optimizer_updates']+int(updated),parent_sha,
                        {'updated':updated,'record_id':row['id'],'schedule_record_id':row['id'],'grades':grades,'advantages':advantages,
                         'accepted_GSPO_groups':prior_accepted+int(active),'accepted_GSPO':active,'update_kind':update['mode'],
                         'completed_stream_runtime_API_counts':raw.get('runtime_API_counts'),
                         'optimizer_before_fingerprint':before_state,'optimizer_after_fingerprint':after_state,'update_evidence':update,
                         'compute_used':journal['compute_used'],**({'ledger':ledger.state()} if ledger else {})})
                    committed=True
                except BaseException as error:
                    # Rename is the commit point even if fsync/return subsequently raises.
                    visible=old.inspect_boundaries(output,binding)
                    committed=visible is not None and visible['completed_groups']>=index+1
                    rollback_error=None
                    if not committed:
                        try:backend.transaction_restore(snapshot)
                        except BaseException as restore_error:rollback_error=str(restore_error)
                    _replace(auxdir/'STOP_PRIVATE.json',{'error_type':type(error).__name__,'error':str(error),'committed':committed,
                        'rolled_back_model_Adam_RNG':not committed and rollback_error is None,'rollback_error':rollback_error,'pending_rollout_reuse_possible':(auxdir/'post_rollout/manifest.json').exists()})
                    raise
                finally:snapshot=None
            return result('TOTAL_UPDATE_BUDGET_REACHED' if current['optimizer_updates']>=2 else 'SCHEDULE_COMPLETE' if current['completed_groups']==4 else 'BOUNDED_STAGE_COMPLETE')
        finally:backend.close()
