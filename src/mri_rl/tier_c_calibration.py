"""Common-B28 training-only calibration rollouts and frozen MedGRPO Eq.3 statistics (Claude-authored; new file).

Statistics unit: the per-prompt mean raw accuracy over K rollouts of the COMMON B28 start on the admitted TRAINING rows of
each task (no updates, no development/test rows). p50 = median, IQR = q75 - q25 over prompts. Zero (or non-finite) IQR
blocks the calibrated configuration for that task; no floor is invented. Rollouts are durable (fsynced jsonl, resumable,
never repeated) and their token cost is charged to a shared calibration ledger, reported separately from the arms.
"""
from __future__ import annotations
import json
import math
import os
import hashlib
import fcntl
import tempfile
from pathlib import Path
import numpy as np

from mri_rl.continuous_rl import digest, require, canonical, write_json, sync_dir
from mri_rl.optimizer_checkpoint import save_optimizer_checkpoint, load_optimizer_checkpoint
from mri_rl.two_task_pool import canonical_sha, validate_model_boundary
from mri_rl.two_task_training import ROUTES, make_grader, validate_training_row


def stats_from_accuracies(by_task):
    """{task: [per-prompt mean accuracy]} -> {task: {p50, iqr, n, blocked, levels}}."""
    out = {}
    for task in ROUTES:
        v = by_task.get(task, [])
        require(len(v) >= 2 and all(isinstance(x, (int, float)) and math.isfinite(x) and 0 <= x <= 1 for x in v),
                'Calibration needs finite per-prompt accuracies for every task')
        q25, p50, q75 = (float(x) for x in np.percentile(np.array(v, dtype=float), [25, 50, 75]))
        iqr = q75 - q25
        out[task] = {'p50': p50, 'iqr': iqr, 'n': len(v), 'blocked': not iqr > 0, 'distinct_levels': len(set(v))}
    return out


def calibration_for_arms(stats):
    """Frozen statistics usable by make_grader; raises for any blocked task (zero IQR)."""
    require(set(stats)==set(ROUTES), 'Complete calibration statistics required for both tasks')
    require(all(math.isfinite(s['p50']) and 0 <= s['p50'] <= 1 and math.isfinite(s['iqr']) and
                s['iqr'] >= 0 and s['blocked'] == (s['iqr'] <= 0) for s in stats.values()), 'Invalid calibration statistics')
    blocked = [t for t, s in stats.items() if s['blocked']]
    if blocked:
        raise ValueError('Calibrated configuration blocked by zero IQR: ' + ','.join(blocked))
    return {t: {'p50': s['p50'], 'iqr': s['iqr']} for t, s in stats.items()}


def _hash(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _raw_record(raw,row,index,binding,prompts,*,partial=False):
    expected=prompts[row['id']];k=binding['group_size'];limit=binding['max_tokens']
    require(raw.get('record_id')==row['id'] and raw.get('group_index')==index and
            raw.get('image_sha256')==binding['images'][row['id']], 'Calibration source/image/index mismatch')
    ids=raw.get('prompt_token_ids');text=raw.get('rendered_prompt')
    require(isinstance(text,str) and hashlib.sha256(text.encode()).hexdigest()==raw.get('prompt_sha256')==expected['rendered_prompt_sha256']
            and ids==expected['input_token_ids'] and len(ids)==1 and len(ids[0])==binding['input_tokens'][row['id']],
            'Calibration native prompt/token IDs differ from frozen measurement')
    responses=raw.get('responses',[])
    require(isinstance(responses,list) and (0<=len(responses)<=k if partial else len(responses)==k), 'Incomplete or oversized native calibration group')
    for response in responses:
        tokens=response.get('token_ids')
        require(isinstance(response.get('text'),str) and isinstance(tokens,list) and 0<len(tokens)<=limit and
                all(type(t) is int and t>=0 for t in tokens), 'Native calibration response token bounds violated')
    grade=make_grader(None)
    scores=[grade(row,r['text'],{}) for r in responses]
    require(all(math.isfinite(g['accuracy']) and 0<=g['accuracy']<=1 and math.isfinite(g['format']) and 0<=g['format']<=1 for g in scores), 'Invalid raw calibration score')
    lengths=[len(r['token_ids']) for r in responses];cost=len(responses)*binding['input_tokens'][row['id']]+sum(lengths)
    return {'id':row['id'],'task':row['ability'],'accuracy':[g['accuracy'] for g in scores],
            'mean_accuracy':sum(g['accuracy'] for g in scores)/len(scores) if scores else None,
            'format':[g['format'] for g in scores], 'input_tokens':binding['input_tokens'][row['id']],
            'generated_tokens':lengths,'cost_tokens':cost}


def _state_binding(binding,backend,anchor):
    return {'run':binding,'adapter_sha256':binding['caller']['starting_adapter_sha256'],
            'trainable_signature':backend.trainable_signature(),'parameter_fingerprint':anchor}


def _publish(output,work,backend,binding,anchor,index,parent,ledger,record=None):
    require(int(backend.optimizer_state()['step'].item())==0 and backend.parameter_fingerprint()==anchor, 'Calibration changed weights or optimizer')
    validity=backend.validate_state()
    require(all(validity.get(k) is True for k in ('parameters_finite','optimizer_finite','second_moments_nonnegative')), 'Invalid calibration state')
    sb=_state_binding(binding,backend,anchor)
    save_optimizer_checkpoint(work/'state',optimizer_state=backend.optimizer_state(),rng_state=backend.rng_state(),
        bindings=sb,completed_groups=index+1,optimizer_updates=0,save_arrays=backend.save_arrays)
    files={str(p.relative_to(work)):digest(p) for p in work.rglob('*') if p.is_file()}
    allowed={'raw_PRIVATE.json','state/manifest.json','state/state.safetensors','attempt_reservation.json','rollout_progress_PRIVATE.json'}
    require(set(files)<=allowed,'Unexpected calibration journal payload')
    receipt={'schema':'calibration-row-journal-v1','binding_sha256':_hash(binding),'index':index,'parent_receipt_sha256':parent,
             'files_sha256':files,'snapshot_bindings':sb,'parameter_fingerprint':anchor,'optimizer_updates':0,
             'record':record,'ledger':ledger,'state_validation':validity}
    write_json(work/'receipt.json',receipt)
    for p in work.rglob('*'):
        if p.is_file():
            with p.open('rb') as f:os.fsync(f.fileno())
    sync_dir(work);destination=output/('initial_state' if index<0 else f'row_{index:04d}')
    require(not destination.exists(),'Calibration row already committed; no replacement allowed')
    os.rename(work,destination);sync_dir(output)
    return receipt,destination


def _charge(output,pending,binding,rows,prompts,parents):
    reservation=pending/'attempt_reservation.json'
    require(reservation.is_file(),'Pending calibration group has no reservation')
    r=json.loads(reservation.read_text());index=r.get('index');parent=r.get('parent_receipt_sha256')
    require(r.get('schema')=='calibration-attempt-v1' and r.get('binding_sha256')==_hash(binding) and
            r.get('directory')==pending.name and type(index) is int and 0<=index<len(rows) and
            parents.get(parent)==index, 'Pending calibration group is not bound to its row/cursor')
    row=rows[index];L=binding['input_tokens'][row['id']];reserve=binding['group_size']*(L+binding['max_tokens'])
    require(r.get('record_id')==row['id'] and r.get('reserved_cost')==reserve,'Reservation differs from frozen calibration protocol')
    rawpath=pending/'raw_PRIVATE.json'
    if not rawpath.exists():rawpath=pending/'rollout_progress_PRIVATE.json'
    prefix=0;measured=0
    if rawpath.exists():
        raw=json.loads(rawpath.read_text());rec=_raw_record(raw,row,index,binding,prompts,partial=True)
        prefix=len(raw['responses']);measured=rec['cost_tokens']
    charge={'binding_sha256':_hash(binding),'directory':pending.name,'parent_receipt_sha256':parent,
            'reservation_sha256':digest(reservation),'raw_prefix_sha256':digest(rawpath) if rawpath.exists() else None,
            'measured_tokens':measured,'imputed_unknown_tokens':(binding['group_size']-prefix)*(L+binding['max_tokens'])}
    charges=output/'abandoned_charges';charges.mkdir(exist_ok=True);path=charges/(pending.name+'.json')
    if path.exists():require(json.loads(path.read_text())==charge,'Abandoned calibration evidence changed')
    else:write_json(path,charge);sync_dir(charges);sync_dir(output)
    return charge,digest(path)


def _inspect(output,binding,rows,prompts):
    initial=output/'initial_state';row_paths=sorted(output.glob('row_*'))
    require(not row_paths or initial.is_dir(),'Completed rows lack their initial RNG snapshot')
    paths=([initial] if initial.exists() else [])+row_paths
    records=[];previous=None;anchor=None;ledger={'measured_tokens':0,'imputed_unknown_tokens':0,'abandoned_charge_sha256':{}}
    parents={}
    for n,p in enumerate(paths):
        index=n-1;require(p.name==('initial_state' if index<0 else f'row_{index:04d}') and not p.is_symlink(), 'Calibration journal gap or invalid path')
        receipt=json.loads((p/'receipt.json').read_text())
        require(receipt['schema']=='calibration-row-journal-v1' and receipt['binding_sha256']==_hash(binding) and
                receipt['index']==index and receipt['parent_receipt_sha256']==previous and receipt['optimizer_updates']==0,
                'Calibration journal identity/cursor mismatch')
        for name,h in receipt['files_sha256'].items():
            require(name in {'raw_PRIVATE.json','state/manifest.json','state/state.safetensors','attempt_reservation.json','rollout_progress_PRIVATE.json'} and
                    (p/name).is_file() and not (p/name).is_symlink() and digest(p/name)==h,'Calibration journal file hash mismatch')
        require({'state/manifest.json','state/state.safetensors'}<=set(receipt['files_sha256']),'Missing RNG checkpoint')
        sb=receipt['snapshot_bindings'];manifest=json.loads((p/'state/manifest.json').read_text())
        require(sb['run']==binding and manifest['bindings']==sb and manifest['completed_groups']==index+1 and
                manifest['optimizer_updates']==0 and manifest['tensors_sha256']==digest(p/'state/state.safetensors'), 'Saved calibration RNG/state binding differs')
        if anchor is None:anchor=receipt['parameter_fingerprint']
        require(receipt['parameter_fingerprint']==anchor==sb['parameter_fingerprint'],'Calibration weights changed between rows')
        newledger=receipt['ledger'];require(set(ledger['abandoned_charge_sha256'])<=set(newledger['abandoned_charge_sha256']),'Lost abandoned work charge')
        measured,imputed=ledger['measured_tokens'],ledger['imputed_unknown_tokens']
        for dirname,h in newledger['abandoned_charge_sha256'].items():
            q=output/'abandoned_charges'/(dirname+'.json');require(q.is_file() and digest(q)==h,'Missing immutable abandoned-work evidence')
            if dirname in ledger['abandoned_charge_sha256']:require(ledger['abandoned_charge_sha256'][dirname]==h,'Changed abandoned-work charge')
            else:
                ch=json.loads(q.read_text());require(ch['binding_sha256']==_hash(binding) and ch['directory']==dirname,'Abandoned charge belongs to another run')
                measured+=ch['measured_tokens'];imputed+=ch['imputed_unknown_tokens']
        if index>=0:
            require(index<len(rows) and 'raw_PRIVATE.json' in receipt['files_sha256'],'Unexpected calibration row')
            raw=json.loads((p/'raw_PRIVATE.json').read_text());rec=_raw_record(raw,rows[index],index,binding,prompts)
            require(receipt['record']==rec,'Calibration scores differ from frozen raw group')
            measured+=rec['cost_tokens'];records.append(rec)
        else:require(receipt['record'] is None,'Initial state must not invent a result')
        require(newledger['measured_tokens']==measured and newledger['imputed_unknown_tokens']==imputed,'Calibration cost ledger differs from raw evidence')
        ledger=newledger;previous=digest(p/'receipt.json');parents[previous]=index+1
    return records,ledger,previous,anchor,parents


def _sync_log(output,records):
    path=output/'calibration_rows_PRIVATE.jsonl';raw=path.read_bytes() if path.exists() else b''
    if raw and not raw.endswith(b'\n'):
        at=raw.rfind(b'\n')+1;tail=raw[at:];backup=output/('torn_log_'+hashlib.sha256(tail).hexdigest()+'.PRIVATE')
        if not backup.exists():
            with backup.open('xb') as f:f.write(tail);f.flush();os.fsync(f.fileno())
        with path.open('wb') as f:f.write(raw[:at]);f.flush();os.fsync(f.fileno())
        raw=raw[:at]
    lines=[json.loads(x) for x in raw.splitlines()]
    require(len(lines)<=len(records) and lines==records[:len(lines)],'Calibration log has unknown/duplicate/changed rows')
    if len(lines)<len(records):
        with path.open('a') as f:
            for rec in records[len(lines):]:f.write(canonical(rec)+'\n');f.flush();os.fsync(f.fileno())
        sync_dir(output)


def run_calibration(*,rows,backend_factory,authorize,output,group_size,input_tokens,resume=False,
                    bindings,prompt_signatures,max_tokens=256):
    """One resident, no-update K4 calibration; row-atomic RNG snapshots support exact ordered resume.

    Only incomplete row groups are retried. Their known prefix cost and conservative
    unknown remainder remain separately recorded; completed rows are never repeated.
    """
    require(callable(authorize) and isinstance(bindings,dict) and bindings.get('starting_adapter_sha256'), 'Exact common initializer binding required')
    require(type(group_size) is int and group_size==4 and type(max_tokens) is int and max_tokens==256,'Frozen native K4/max256 calibration required')
    require(rows and len({r['id'] for r in rows})==len(rows) and set(prompt_signatures)=={r['id'] for r in rows},'Unique complete calibration source/prompt coverage required')
    for row in rows:validate_training_row(row);validate_model_boundary(row,'train')
    costs={r['id']:input_tokens(r) for r in rows};require(all(type(n) is int and 0<n<=2048 for n in costs.values()),'Invalid measured prompt cost')
    cap=group_size*sum(costs.values())+len(rows)*group_size*max_tokens
    binding={'caller':bindings,'row_sha256':[canonical_sha(r) for r in rows],'row_ids':[r['id'] for r in rows],
             'images':{r['id']:digest(r['images'][0]) for r in rows},'prompts_sha256':_hash(prompt_signatures),
             'input_tokens':costs,'group_size':group_size,'max_tokens':max_tokens,'token_cap':cap,
             'helper_sha256':digest(__file__),'weight_updates':0}
    output=Path(output)
    if output.exists() and not resume:raise FileExistsError('Existing calibration requires explicit resume')
    require(not resume or output.is_dir(),'Calibration resume directory missing')
    output.mkdir(parents=True,exist_ok=True)
    with (output/'.writer.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another calibration writer is active') from None
        header=output/'binding_PRIVATE.json'
        if header.exists():require(json.loads(header.read_text())==binding,'Calibration source/model/prompt/config binding changed')
        else:
            require(not [p for p in output.iterdir() if p.name!='.writer.lock'],'Unbound legacy calibration output cannot be resumed')
            write_json(header,binding);sync_dir(output)
        records,ledger,parent,anchor,parents=_inspect(output,binding,rows,prompt_signatures);_sync_log(output,records)
        for pending in sorted(output.glob('.pending-*')):
            ch,h=_charge(output,pending,binding,rows,prompt_signatures,parents)
            if pending.name in ledger['abandoned_charge_sha256']:require(ledger['abandoned_charge_sha256'][pending.name]==h,'Changed included abandoned charge')
            else:
                ledger={**ledger,'measured_tokens':ledger['measured_tokens']+ch['measured_tokens'],
                        'imputed_unknown_tokens':ledger['imputed_unknown_tokens']+ch['imputed_unknown_tokens'],
                        'abandoned_charge_sha256':{**ledger['abandoned_charge_sha256'],pending.name:h}}
        cursor=len(records);backend=None
        def fits(index):return ledger['measured_tokens']+ledger['imputed_unknown_tokens']+group_size*(costs[rows[index]['id']]+max_tokens)<=cap
        try:
            if cursor<len(rows) and fits(cursor):
                authorize({'phase':'load','pending':len(rows)-cursor});backend=backend_factory()
                current_anchor=backend.parameter_fingerprint()
                require(anchor is None or current_anchor==anchor,'Common-B28 weights changed on resume')
                anchor=current_anchor
                if parent is not None:
                    path=output/('initial_state' if cursor==0 else f'row_{cursor-1:04d}')
                    state,rng,_=load_optimizer_checkpoint(path/'state',expected_bindings=_state_binding(binding,backend,anchor),load_arrays=backend.load_arrays)
                    backend.restore(state,rng)
                else:
                    work=Path(tempfile.mkdtemp(prefix='.initial-',dir=output))
                    _,path=_publish(output,work,backend,binding,anchor,-1,None,ledger)
                    parent=digest(path/'receipt.json');parents[parent]=0
                require(int(backend.optimizer_state()['step'].item())==0,'Calibration optimizer must remain at zero')
                for index in range(cursor,len(rows)):
                    if not fits(index):break
                    row=rows[index];authorize({'phase':'row','index':index})
                    require(digest(row['images'][0])==binding['images'][row['id']],'Frozen image changed before calibration row')
                    work=Path(tempfile.mkdtemp(prefix='.pending-',dir=output))
                    write_json(work/'attempt_reservation.json',{'schema':'calibration-attempt-v1','binding_sha256':_hash(binding),
                        'directory':work.name,'parent_receipt_sha256':parent,'index':index,'record_id':row['id'],
                        'reserved_cost':group_size*(costs[row['id']]+max_tokens)})
                    sync_dir(work);sync_dir(output)
                    raw=backend.rollout(row,index,work);write_json(work/'raw_PRIVATE.json',raw)
                    require(digest(row['images'][0])==binding['images'][row['id']],'Frozen image changed during calibration')
                    rec=_raw_record(raw,row,index,binding,prompt_signatures)
                    ledger={**ledger,'measured_tokens':ledger['measured_tokens']+rec['cost_tokens']}
                    _,path=_publish(output,work,backend,binding,anchor,index,parent,ledger,rec)
                    parent=digest(path/'receipt.json');parents[parent]=index+1;records.append(rec);_sync_log(output,records)
        finally:
            if backend is not None:backend.close()
        complete=len(records)==len(rows)
        by_task={t:[r['mean_accuracy'] for r in records if r['task']==t] for t in ROUTES}
        stats=stats_from_accuracies(by_task) if complete else None
        return {'status':'CALIBRATION_COMPLETE' if complete else 'TOKEN_BUDGET_EXHAUSTED_INCOMPLETE',
                'rows':len(records),'expected_rows':len(rows),'complete':complete,'stats':stats,
                'calibrated_arms_unblocked':complete and not any(s['blocked'] for s in stats.values()),
                'measured_tokens':ledger['measured_tokens'],'imputed_unknown_tokens':ledger['imputed_unknown_tokens'],
                'cost_tokens':ledger['measured_tokens']+ledger['imputed_unknown_tokens'],'token_cap':cap,'weight_updates':0,
                'accuracy_support_levels':{t:sorted(set(v)) for t,v in by_task.items()}}
