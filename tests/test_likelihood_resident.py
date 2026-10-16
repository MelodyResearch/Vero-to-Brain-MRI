"""Synthetic resident-control tests: fake clock, no models or native imports."""
import importlib.util
import json
from pathlib import Path
import pytest


def module():
    path=Path(__file__).parents[1]/'src/mri_rl/likelihood_resident.py'
    assert path.exists(),'Resident continuation helper missing'
    spec=importlib.util.spec_from_file_location('resident_test',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


class Clock:
    def __init__(self):self.value=0.;self.hook=None;self.sleeps=[]
    def now(self):return self.value
    def sleep(self,seconds):
        self.sleeps.append(seconds);self.value+=seconds
        if self.hook:self.hook()


class Gate:
    def __init__(self,clock):self.clock=clock;self.calls=[];self.initial=object();self.history=[];self.hook=None
    def before_unit(self,name):
        self.calls.append(name);self.history.append(self.clock.now())
        if self.hook:self.hook(name)


@pytest.fixture
def setup(tmp_path,monkeypatch):
    m=module();monkeypatch.setattr(m,'PRIVATE_ROOT',tmp_path)
    clock=Clock();gate=Gate(clock);output=tmp_path/'outputs.jsonl';records=[{'record_id':f'fixture-{i}'} for i in range(57)]
    def plan():
        completed=len(output.read_text().splitlines()) if output.exists() else 0
        pending=records[completed:completed+2]
        scope={'resident':True,'run_identity':'frozen-identity','checkpoint':'A0','output':str(output),
          'completed_records':completed,'new_record_ids':[r['record_id'] for r in pending],
          'max_new_records':2,'existing_output_sha256':m.sha(output) if output.exists() else None}
        return pending,scope
    pending,scope=plan();approval=tmp_path/'initial.json'
    approval.write_text(json.dumps({'kind':'answer_likelihood_v1','verdict':'PASS_WITH_LIMITATIONS','reviewer':'independent-fixture','scope':scope}))
    checks=[]
    session=m.ResidentSession(records=records,initial_pending=pending,initial_scope=scope,
      approval_path=approval,approval_sha256=m.sha(approval),session_dir=tmp_path/'session',
      plan_next=plan,gate=gate,verify_adapter=lambda:checks.append('verified'),clock=clock.now,sleep=clock.sleep)
    def complete(chunk):
        with output.open('a') as f:
            for row in chunk['pending']:f.write(json.dumps(row)+'\n')
    def approve_request(*,bad=False):
        requests=sorted((tmp_path/'session').glob('request_*.json'))
        if not requests:return
        request=requests[-1];value=json.loads(request.read_text())
        response=request.with_name(request.name.replace('request_','response_'))
        if response.exists():return
        p=tmp_path/('approval-'+str(value['scope']['completed_records'])+'.json')
        p.write_text(json.dumps({'kind':'answer_likelihood_v1','verdict':'PASS_WITH_LIMITATIONS','reviewer':'independent-fixture','scope':value['scope']}))
        response.write_text(json.dumps({'request_digest':'wrong' if bad else value['request_digest'],
          'approval_path':str(p),'approval_sha256':m.sha(p)}))
    return m,session,clock,gate,output,records,complete,approve_request,checks


def test_no_second_yield_before_independent_approval_and_same_gate(setup):
    m,s,c,g,out,records,complete,approve,checks=setup;baseline=g.initial
    iterator=iter(s);first=next(iterator);complete(first)
    def delayed():
        assert s.completed_records==2
        if c.value>=45:approve()
    c.hook=delayed
    second=next(iterator)
    assert second['pending']==records[2:4] and c.value>=45
    assert checks==['verified'] and g.initial is baseline and len(g.history)>=3
    assert c.sleeps[0]==15
    iterator.close()


def test_timeout_preserves_prefix_without_next_model_work(setup):
    m,s,c,g,out,records,complete,approve,checks=setup
    iterator=iter(s);complete(next(iterator));before=out.read_bytes()
    with pytest.raises(StopIteration):next(iterator)
    assert s.status=='APPROVAL_TIMEOUT' and out.read_bytes()==before and checks==['verified']
    assert c.value==195 and max(c.sleeps)<=15


def test_late_approval_after_slow_resource_gate_never_yields(setup):
    m,s,c,g,out,records,complete,approve,checks=setup
    iterator=iter(s);complete(next(iterator))
    def delayed_gate(name):
        if name.startswith('resident_wait'):
            c.value+=181;approve()
    g.hook=delayed_gate
    with pytest.raises(StopIteration):next(iterator)
    assert s.status=='APPROVAL_TIMEOUT'


@pytest.mark.parametrize('damage',['wrong_request','output_mutation','wrong_approval'])
def test_wrong_or_stale_evidence_fails_closed(setup,damage):
    m,s,c,g,out,records,complete,approve,checks=setup
    iterator=iter(s);complete(next(iterator))
    def inject():
        if not list(s.session_dir.glob('request_*.json')):return
        approve(bad=damage=='wrong_request')
        if damage=='output_mutation':out.write_text(out.read_text().replace('fixture-0','mutated-0'))
        if damage=='wrong_approval':
            p=next(s.session_dir.parent.glob('approval-*.json'));p.write_text('{}')
    c.hook=inject
    with pytest.raises(ValueError):next(iterator)
    assert s.completed_records==2


def test_incomplete_chunk_and_budget_overflow_reject(setup):
    m,s,c,g,out,records,complete,approve,checks=setup
    iterator=iter(s);chunk=next(iterator)
    with out.open('w') as f:f.write(json.dumps(chunk['pending'][0])+'\n')
    with pytest.raises(ValueError,match='completed|chunk'):next(iterator)


def test_finite_full_manifest_and_no_repeated_chunk(setup):
    m,s,c,g,out,records,complete,approve,checks=setup;c.hook=approve
    executed=[]
    for chunk in s:
        assert len(chunk['pending'])<=8
        executed.extend(row['record_id'] for row in chunk['pending']);complete(chunk)
    assert executed==[r['record_id'] for r in records] and s.completed_records==57 and s.status=='COMPLETED'
    assert len(checks)==29
    with pytest.raises(ValueError,match='one-shot'):next(iter(s))


def test_deadline_after_final_approval_verification_blocks_yield(setup):
    m,s,c,g,out,records,complete,approve,checks=setup
    original=s.verify_approval;calls=0
    def delayed(path,digest,scope):
        nonlocal calls
        original(path,digest,scope)
        if scope['completed_records']==2:
            calls+=1
            if calls==2:c.value+=181
    s.verify_approval=delayed;c.hook=approve
    iterator=iter(s);complete(next(iterator))
    with pytest.raises(StopIteration):next(iterator)
    assert s.status=='APPROVAL_TIMEOUT'


def test_initial_approval_mutation_inside_resource_gate_rejects(setup):
    m,s,c,g,out,records,complete,approve,checks=setup
    def mutate(name):
        if name=='resident_initial_chunk':s.approval_path.write_text('{}')
    g.hook=mutate
    with pytest.raises(ValueError,match='approval'):next(iter(s))
    assert not out.exists()


def test_resource_history_cannot_be_cleared_in_place(setup):
    m,s,c,g,out,records,complete,approve,checks=setup;c.hook=approve
    iterator=iter(s);complete(next(iterator))
    def erase(name):
        if name.startswith('resident_wait'):g.history.clear()
    g.hook=erase
    with pytest.raises(ValueError,match='history'):next(iterator)


def test_oversized_next_chunk_rejected_before_request(setup):
    m,s,c,g,out,records,complete,approve,checks=setup
    original=s.plan_next
    def too_many():
        _,scope=original();pending=records[2:11]
        return pending,{**scope,'max_new_records':9,'new_record_ids':[r['record_id'] for r in pending]}
    s.plan_next=too_many
    iterator=iter(s);complete(next(iterator))
    with pytest.raises(ValueError,match='budget'):next(iterator)
    assert not list(s.session_dir.glob('request_*.json'))
