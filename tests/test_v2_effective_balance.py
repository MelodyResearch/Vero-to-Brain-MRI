"""New quota control fixtures; these are not native training experiments."""
import copy
import importlib.util
from pathlib import Path
import pytest


def module():
    p=Path(__file__).parents[1]/'src/mri_rl/v2_effective_balance.py'
    assert p.exists(), 'Independent V2 controller has not been implemented'
    spec=importlib.util.spec_from_file_location('v2_control_test',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def data(m):
    rows=[{'id':t+str(i),'ability':t,'extra_info':{'internal_split':'train'}} for t in m.TASKS for i in range(3)]
    return rows,{r['id']:10 for r in rows}


def grades(active=True,fmt=1.):
    return [{'accuracy':float(i==3) if active else 0.,'format':fmt,'overlong':0.,'total':(.8*float(i==3) if active else 0.)+.2*fmt} for i in range(4)]


def test_rejected_groups_are_charged_and_quota_balances_actual_updates():
    m=module();rows,tokens=data(m);c=m.QuotaControl(rows,tokens,seed=7,target=3,cap=40000);original=copy.deepcopy(rows)
    attempts={t:0 for t in m.TASKS}
    while c.phase=='READY':
        d=c.offer();
        if d is None:break
        t=d['task'];attempts[t]+=1
        active=t==m.TASKS[0] or attempts[t]%3==0
        a=c.assess(d,grades(active),[1]*4);c.commit(a)
    assert c.accepted=={t:3 for t in m.TASKS}
    assert attempts[m.TASKS[0]]==3 and attempts[m.TASKS[1]]==9
    assert c.used==12*44 and c.phase=='QUOTAS_COMPLETE' and rows==original


def test_remaining_quota_probabilities_not_attempt_counts():
    m=module();rows,tokens=data(m);c=m.QuotaControl(rows,tokens,seed=11,target=2,cap=20000)
    first=c.offer();c.commit(c.assess(first,grades(),[1]*4))
    p=c.probabilities();assert p[first['task']]==pytest.approx(1/3)
    assert p[next(t for t in m.TASKS if t!=first['task'])]==pytest.approx(2/3)


def test_format_rejection_zero_variance_and_budget_reservation():
    m=module();rows,tokens=data(m);c=m.QuotaControl(rows,tokens,seed=1,target=2,cap=2200)
    d=c.offer();c.commit(c.assess(d,grades(fmt=0.),[256]*4))
    assert sum(c.accepted.values())==0 and c.used==1064
    d=c.offer();c.commit(c.assess(d,grades(False),[256]*4))
    assert c.offer() is None and c.phase=='TOKEN_CAP' and c.used==2128


def test_double_commit_row_substitution_and_overspend_rejected():
    m=module();rows,tokens=data(m);c=m.QuotaControl(rows,tokens,seed=1,target=2,cap=10000);d=c.offer()
    bad=copy.deepcopy(d);bad['row']['id']='changed'
    with pytest.raises(ValueError):c.assess(bad,grades(),[1]*4)
    with pytest.raises(ValueError):c.assess(d,grades(),[257]*4)
    a=c.assess(d,grades(),[1]*4);c.commit(a)
    with pytest.raises(ValueError):c.commit(a)


def test_exact_rng_pool_and_controller_resume_and_tamper_rejection():
    m=module();rows,tokens=data(m);c=m.QuotaControl(rows,tokens,seed=13,target=4,cap=100000)
    for _ in range(7):
        d=c.offer();c.commit(c.assess(d,grades(False),[2]*4))
    state=c.snapshot();b=m.QuotaControl.restore(rows,tokens,state,m.digest(state))
    for _ in range(10):
        d,e=c.offer(),b.offer();assert d==e
        for x,z in ((c,d),(b,e)):x.commit(x.assess(z,grades(False),[3]*4))
    assert c.snapshot()==b.snapshot()
    state['used_tokens']+=1
    with pytest.raises(ValueError):m.QuotaControl.restore(rows,tokens,state,'0'*64)


def test_no_dev_rows_or_duplicate_sources_and_pending_requires_reconciliation():
    m=module();rows,tokens=data(m);bad=copy.deepcopy(rows);bad[0]['extra_info']['internal_split']='dev'
    with pytest.raises(ValueError):m.QuotaControl(bad,tokens,seed=1)
    with pytest.raises(ValueError):m.QuotaControl(rows+[rows[0]],tokens,seed=1)
    c=m.QuotaControl(rows,tokens,seed=1);d=c.offer();s=c.snapshot();b=m.QuotaControl.restore(rows,tokens,s,m.digest(s))
    assert b.pending==d
    with pytest.raises(ValueError):b.offer()


def test_file_state_is_authenticated_and_roundtrips(tmp_path):
    m=module();rows,tokens=data(m);c=m.QuotaControl(rows,tokens,seed=9);d=c.offer();c.commit(c.assess(d,grades(),[2]*4))
    p=tmp_path/'state.json';h=c.save(p);b=m.QuotaControl.load(p,h,rows,tokens)
    assert b.offer()==c.offer()
    p.write_text(p.read_text()+' ')
    with pytest.raises(ValueError):m.QuotaControl.load(p,h,rows,tokens)


def test_native_callbacks_route_only_accepted_groups_without_claiming_native_run(tmp_path):
    m=module();rows,tokens=data(m);c=m.QuotaControl(rows,tokens,seed=1,target=1,cap=10000);calls=[]
    class Backend:
        step=0
        def rollout(self,row,index,out):
            calls.append('rollout');return {'record_id':row['id'],'responses':[{'text':'fixture','token_ids':[1]} for _ in range(4)]}
        def after_rollout(self):calls.append('after_rollout')
        def optimizer_state(self):return {'step':self.step}
        def update(self,row,raw,advantages):calls.append('update');self.step+=1;return {'gradient_l2':1.}
    backend=Backend()
    def persist(phase,*args):calls.append(phase)
    def grader(row,raw):return grades(True)
    r=m.run_attempt(c,backend,tmp_path,persist,grader=grader)
    assert r['accepted'] and backend.step==1
    assert calls==['reserved','rollout','after_rollout','update','committed']


def test_failed_update_halts_and_charges_completed_rollout(tmp_path):
    m=module();rows,tokens=data(m);c=m.QuotaControl(rows,tokens,seed=1,target=1,cap=10000)
    class Backend:
        def rollout(self,row,index,out):return {'record_id':row['id'],'responses':[{'text':'fixture','token_ids':[1]} for _ in range(4)]}
        def after_rollout(self):pass
        def optimizer_state(self):return {'step':0}
        def update(self,*args):raise RuntimeError('fixture update failure')
    with pytest.raises(RuntimeError):m.run_attempt(c,Backend(),tmp_path,lambda *args:None,grader=lambda *args:grades())
    assert c.phase=='FAILED' and c.used==44 and sum(c.accepted.values())==0
