"""Model-independent, independently approved resident likelihood chunks.

Owns no model or resource baseline. The caller keeps one model and the SAME gate
alive, fsyncs each output row, and releases the model in its finally block. Every
next <=8-record chunk needs a fresh request-bound review; waiting performs no
model calls. This module neither changes scoring nor grants launch approval.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

PRIVATE_ROOT=Path(__file__).resolve().parents[2]/'local_data'
POLL_SECONDS=15
WAIT_SECONDS=180
COOLDOWN_SECONDS=15
CHANGING_SCOPE={'existing_output_sha256','completed_records','new_record_ids','max_new_records'}


def require(value,message):
    if not value:raise ValueError(message)


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def _sync_dir(path):
    fd=os.open(path,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def _immutable_json(path,value):
    with Path(path).open('x') as f:
        f.write(canonical(value)+'\n');f.flush();os.fsync(f.fileno())
    _sync_dir(Path(path).parent)


class ResidentSession:
    def __init__(self,*,records,initial_pending,initial_scope,approval_path,approval_sha256,
                 session_dir,plan_next,gate,verify_adapter,clock=time.monotonic,sleep=time.sleep,
                 verify_approval=None):
        self.records=json.loads(canonical(records));self.initial_pending=json.loads(canonical(initial_pending))
        self.initial_scope=json.loads(canonical(initial_scope));self.session_dir=Path(session_dir).resolve()
        self.output=Path(initial_scope['output']).resolve();self.approval_path=Path(approval_path).resolve()
        require(all(p.is_relative_to(PRIVATE_ROOT.resolve()) for p in (self.session_dir,self.output,self.approval_path)),
                'Resident session, approvals and outputs must remain private')
        require(len(self.records)==57 and len({r['record_id'] for r in self.records})==57,'Fixed unique57-input manifest required')
        require(initial_scope.get('resident') is True,'Resident execution must be explicit in approved scope')
        require(callable(plan_next) and callable(verify_adapter) and callable(getattr(gate,'before_unit',None)),
                'Existing planner, adapter checker and process resource gate required')
        if verify_approval is None:
            from mri_rl.paced_answer_likelihood import verify_review
            verify_approval=verify_review
        self.plan_next=plan_next;self.gate=gate;self.verify_adapter=verify_adapter;self.verify_approval=verify_approval
        self.clock=clock;self.sleep=sleep;self.approval_sha256=approval_sha256
        self.stable_scope={k:v for k,v in self.initial_scope.items() if k not in CHANGING_SCOPE}
        self.status='NOT_STARTED';self.completed_records=initial_scope['completed_records'];self.started=False
        self._baseline=getattr(gate,'initial',None);self._history=getattr(gate,'history',None)
        self._history_length=len(self._history) if self._history is not None else None
        self.session_digest=hashlib.sha256(canonical({'nonce':str(uuid.uuid4()),'initial_scope':initial_scope,
            'manifest_sha256':hashlib.sha256(canonical(self.records).encode()).hexdigest(),
            'helper_source_sha256':sha(__file__)}).encode()).hexdigest()

    def _gate(self,name):
        # Never recreate a battery gate, erase history or reset its start sample.
        require(getattr(self.gate,'initial',None) is self._baseline and getattr(self.gate,'history',None) is self._history,
                'Process-wide resource baseline/history was reset')
        if self._history is not None:require(len(self._history)>=self._history_length,'Resource history was truncated')
        self.gate.before_unit(name)
        require(getattr(self.gate,'initial',None) is self._baseline and getattr(self.gate,'history',None) is self._history,
                'Process-wide resource baseline/history changed')
        if self._history is not None:
            require(len(self._history)>self._history_length,'Resource history was cleared/truncated during gate')
            self._history_length=len(self._history)

    def _prefix(self):
        if not self.output.exists():return b'',None,[]
        raw=self.output.read_bytes()
        require(not raw or raw.endswith(b'\n'),'Torn resident output; preserve partial evidence')
        try:rows=[json.loads(line) for line in raw.splitlines()]
        except (ValueError,TypeError):raise ValueError('Malformed resident output prefix') from None
        require(len(rows)<=57 and [r.get('record_id') for r in rows]==[r['record_id'] for r in self.records[:len(rows)]],
                'Resident output is not the exact frozen completed prefix')
        return raw,hashlib.sha256(raw).hexdigest(),rows

    def _check_plan(self,pending,scope):
        require({k:v for k,v in scope.items() if k not in CHANGING_SCOPE}==self.stable_scope,
                'Resident model/configuration/run identity changed')
        limit=scope.get('max_new_records');cursor=scope.get('completed_records')
        require(type(limit) is int and 1<=limit<=8 and type(cursor) is int and 0<=cursor<=57,
                'Resident chunk exceeds finite1–8-record budget')
        require(pending==self.records[cursor:cursor+limit] and scope.get('new_record_ids')==[r['record_id'] for r in pending],
                'Pending resident chunk differs from frozen manifest')
        raw,digest,rows=self._prefix()
        require(cursor==len(rows)==self.completed_records and scope.get('existing_output_sha256')==digest,
                'Completed output prefix/hash changed from exact scope')
        return raw,digest

    def _wait(self,scope):
        cursor=scope['completed_records'];request=self.session_dir/f'request_{cursor:03d}.json'
        response=self.session_dir/f'response_{cursor:03d}.json'
        require(not response.exists(),'Stale resident response already exists')
        body={'kind':'likelihood_resident_request_v1','session_digest':self.session_digest,'scope':scope}
        request_digest=hashlib.sha256(canonical(body).encode()).hexdigest()
        _immutable_json(request,{**body,'request_digest':request_digest})
        request_hash=sha(request);deadline=self.clock()+WAIT_SECONDS
        while self.clock()<deadline:
            self._gate(f'resident_wait:{cursor}')
            # A resource HOLD itself can consume the entire waiting window.
            if self.clock()>=deadline:return None
            require(sha(request)==request_hash,'Immutable resident request changed')
            _,digest,_=self._prefix()
            require(digest==scope['existing_output_sha256'],'Output changed while waiting for approval')
            if response.exists():
                response_hash=sha(response)
                try:answer=json.loads(response.read_text())
                except (ValueError,TypeError):raise ValueError('Invalid resident approval response') from None
                require(answer.get('request_digest')==request_digest,'Approval response belongs to another request/session')
                approval=Path(answer.get('approval_path','')).resolve();approval_digest=answer.get('approval_sha256')
                require(approval.is_relative_to(PRIVATE_ROOT.resolve()) and approval.is_file(), 'Private independent approval missing')
                self.verify_approval(approval,approval_digest,scope)
                self._gate(f'resident_approved:{cursor}')
                if self.clock()>=deadline:return None
                require(sha(response)==response_hash and sha(request)==request_hash,'Resident approval/request mutated')
                self.verify_approval(approval,approval_digest,scope)
                _,digest,_=self._prefix()
                require(digest==scope['existing_output_sha256'],'Output changed after exact approval')
                if self.clock()>=deadline:return None
                return {'approval_path':str(approval),'approval_sha256':approval_digest,
                        'request_digest':request_digest,'deadline':deadline}
            self.sleep(min(POLL_SECONDS,max(0.,deadline-self.clock())))
        return None

    def __iter__(self):
        require(not self.started,'Resident session is one-shot; never replay a consumed chunk')
        self.started=True;self.session_dir.mkdir(parents=True,exist_ok=True)
        _immutable_json(self.session_dir/'session.json',{'kind':'likelihood_resident_session_v1',
            'session_digest':self.session_digest,'initial_scope':self.initial_scope,'manifest_records':57,
            'wait_seconds':WAIT_SECONDS,'poll_seconds':POLL_SECONDS,'cooldown_seconds':COOLDOWN_SECONDS})
        scope=self.initial_scope;pending=self.initial_pending;release_deadline=None
        try:
            before,_=self._check_plan(pending,scope)
            self.verify_approval(self.approval_path,self.approval_sha256,scope)
            self._gate('resident_initial_chunk')
            # Gate callbacks may wait; revalidate the precise approved prefix.
            self._check_plan(pending,scope)
            self.verify_approval(self.approval_path,self.approval_sha256,scope)
            while pending:
                chunk={'pending':json.loads(canonical(pending)),'scope':json.loads(canonical(scope))}
                if release_deadline is not None and self.clock()>=release_deadline:
                    self.status='APPROVAL_TIMEOUT';return
                self.status='CHUNK_RUNNING'
                yield chunk
                raw,_,rows=self._prefix()
                require(raw.startswith(before) and len(rows)==scope['completed_records']+len(pending),
                        'Resident completed chunk is partial, replaced or over budget')
                with self.output.open('rb') as f:os.fsync(f.fileno())
                _sync_dir(self.output.parent)
                self.completed_records=len(rows)
                self.verify_adapter()  # Only here, never during idle approval wait.
                self._gate(f'resident_chunk_complete:{self.completed_records}')
                self.sleep(COOLDOWN_SECONDS)
                self._gate(f'resident_after_cooldown:{self.completed_records}')
                if self.completed_records==57:
                    self.status='COMPLETED';return
                self.status='WAITING_FOR_APPROVAL'
                planned=self.plan_next();require(isinstance(planned,tuple) and len(planned)==2,'Planner must return pending,scope')
                pending,scope=planned
                pending=json.loads(canonical(pending));scope=json.loads(canonical(scope))
                before,_=self._check_plan(pending,scope)
                approval=self._wait(scope)
                if approval is None:
                    self.status='APPROVAL_TIMEOUT';return
                self._check_plan(pending,scope)
                release_deadline=approval['deadline']
            self.status='COMPLETED'
        except BaseException:
            if self.status not in ('COMPLETED','APPROVAL_TIMEOUT'):self.status='STOPPED_PRESERVE_COMPLETED_PREFIX'
            raise
        finally:
            # The caller owns model release in its finally block. No model work here.
            if self.status not in ('COMPLETED','APPROVAL_TIMEOUT','STOPPED_PRESERVE_COMPLETED_PREFIX'):
                self.status='STOPPED_PRESERVE_COMPLETED_PREFIX'
