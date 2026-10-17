"""Separate accepted-group quota control, authored independently by Codex.

This is an exposure ablation, not a change to frozen Vero training. Native
checkpoint integration requires separate review; CPU fixtures are not GSPO.
"""
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import random

from mri_rl.mlx_gspo_smoke import task_group_is_informative, advantages_from_rewards
from mri_rl.continuous_rl import _grades, PROFILE

TASKS=('mri_sequence','mri_visit_correspondence')
FULL={'target':32,'cap':8*169216}
PILOT={'target':4,'cap':169216}


def require(ok,message):
    if not ok:raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def tuples(value):
    return tuple(tuples(x) for x in value) if isinstance(value,(list,tuple)) else value


class QuotaControl:
    def __init__(self,rows,input_tokens,*,seed,target=32,cap=8*169216):
        require(type(target) is int and 1<=target<=32 and type(cap) is int and 0<cap<=8*169216,'Invalid finite quota/cap')
        require(type(seed) is int and rows and len({r['id'] for r in rows})==len(rows),'Invalid seed or duplicate source')
        require(set(input_tokens)=={r['id'] for r in rows} and all(type(n) is int and 0<n<=2048 for n in input_tokens.values()),'Exact measured input-token inventory required')
        require(all(r['ability'] in TASKS and r['extra_info'].get('internal_split')=='train' for r in rows),'Only training rows from two admitted tasks permitted')
        self.rows={r['id']:deepcopy(r) for r in rows};self.tokens=dict(input_tokens)
        self.data_sha=digest(rows);self.seed=seed;self.target=target;self.cap=cap
        self.rng=random.Random(seed);self.order={t:[r['id'] for r in rows if r['ability']==t] for t in TASKS}
        require(all(self.order.values()),'Both tasks need admitted training rows')
        for pool in self.order.values():self.rng.shuffle(pool)
        self.cursor={t:0 for t in TASKS};self.accepted={t:0 for t in TASKS}
        self.history=[];self.used=0;self.phase='READY';self.pending=None;self.failure=None

    @classmethod
    def from_manifest(cls,path,expected_sha256,input_tokens,**kwargs):
        raw=Path(path).read_bytes();require(hashlib.sha256(raw).hexdigest()==expected_sha256,'Training manifest changed')
        return cls([json.loads(x) for x in raw.splitlines() if x.strip()],input_tokens,**kwargs)

    def probabilities(self):
        remaining={t:self.target-self.accepted[t] for t in TASKS};n=sum(remaining.values())
        return {t:(remaining[t]/n if n else 0.) for t in TASKS}

    def offer(self):
        require(self.pending is None,'Pending attempt requires evidence/debt reconciliation, not a new rollout')
        if self.phase!='READY':return None
        remaining={t:self.target-self.accepted[t] for t in TASKS};n=sum(remaining.values())
        if n==0:self.phase='QUOTAS_COMPLETE';return None
        k=self.rng.randrange(n);task=TASKS[0] if k<remaining[TASKS[0]] else TASKS[1]
        if self.cursor[task]==len(self.order[task]):self.rng.shuffle(self.order[task]);self.cursor[task]=0
        rid=self.order[task][self.cursor[task]];inp=self.tokens[rid];reserve=4*(inp+256)
        if self.used+reserve>self.cap:self.phase='TOKEN_CAP';return None
        self.cursor[task]+=1
        self.pending={'attempt':len(self.history),'task':task,'row':deepcopy(self.rows[rid]),'input_tokens':inp,'reservation':reserve}
        return deepcopy(self.pending)

    def assess(self,decision,grades,lengths):
        require(self.phase=='READY' and self.pending is not None and digest(decision)==digest(self.pending),'Pending row/cursor differs')
        require(len(grades)==len(lengths)==4 and all(type(n) is int and 0<n<=256 for n in lengths),'Complete bounded group required')
        for g in grades:
            require(all(k in g and isinstance(g[k],(int,float)) and math.isfinite(g[k]) for k in ('accuracy','format','overlong','total')),'Finite decomposed reward required')
            require(g['accuracy'] in (0.,1.) and g['format'] in (0.,.5,1.) and g['overlong']==0. and abs(g['total']-(.8*g['accuracy']+.2*g['format']))<1e-12,'Frozen raw task/format/length reward required')
        cost=4*decision['input_tokens']+sum(lengths);require(cost<=decision['reservation'] and self.used+cost<=self.cap,'Attempt exceeds reserved/capped cost')
        return {'decision':deepcopy(decision),'grades':deepcopy(grades),'lengths':list(lengths),'cost':cost,'accepted':task_group_is_informative(grades),
                'advantages':advantages_from_rewards([g['total'] for g in grades])}

    def commit(self,assessment):
        actual=self.assess(assessment['decision'],assessment['grades'],assessment['lengths'])
        require(digest(actual)==digest(assessment),'Assessment changed before commit')
        t=self.pending['task'];self.used+=actual['cost'];self.accepted[t]+=int(actual['accepted'])
        self.history.append({'attempt':len(self.history),'task':t,'row_id':self.pending['row']['id'],'cost':actual['cost'],'accepted':actual['accepted']})
        self.pending=None
        if all(n==self.target for n in self.accepted.values()):self.phase='QUOTAS_COMPLETE'

    def fail(self,assessment=None):
        require(self.phase=='READY' and self.failure is None and self.pending is not None,'Failure must reconcile a fresh pending attempt exactly once')
        if assessment is not None:
            require(isinstance(assessment,dict) and all(k in assessment for k in ('decision','grades','lengths','cost')),'Complete authenticated assessment required')
            actual=self.assess(assessment['decision'],assessment['grades'],assessment['lengths'])
            require(digest(actual)==digest(assessment),'Failed-attempt assessment changed')
            cost=actual['cost']
        else:cost=self.pending['reservation']
        require(self.used+cost<=self.cap,'Failure debt exceeds reservation')
        self.used+=cost;self.phase='FAILED';self.failure={'charged_tokens':cost,'imputed':assessment is None,'native_state_requires_reconciliation':True}

    def snapshot(self):
        return deepcopy({'schema':'accepted-group-quota-v2','data_sha256':self.data_sha,'token_inventory':self.tokens,'seed':self.seed,'target':self.target,'cap':self.cap,
            'pool_order':self.order,'pool_cursor':self.cursor,'rng':self.rng.getstate(),'accepted':self.accepted,'history':self.history,
            'used_tokens':self.used,'phase':self.phase,'pending':self.pending,'failure':self.failure})

    @classmethod
    def restore(cls,rows,input_tokens,state,expected_sha256):
        require(digest(state)==expected_sha256 and state['schema']=='accepted-group-quota-v2','Externally authenticated controller state required')
        obj=cls(rows,input_tokens,seed=state['seed'],target=state['target'],cap=state['cap'])
        require(state['data_sha256']==obj.data_sha and state['token_inventory']==input_tokens,'Training data/token binding changed')
        for t in TASKS:
            require(sorted(state['pool_order'][t])==sorted(obj.order[t]) and type(state['pool_cursor'][t]) is int and 0<=state['pool_cursor'][t]<=len(obj.order[t]),'Invalid pool/RNG cursor')
            require(type(state['accepted'][t]) is int and 0<=state['accepted'][t]<=obj.target,'Invalid accepted quota')
        history=state['history'];require(all(h['attempt']==i and h['task'] in TASKS and h['row_id'] in obj.rows and obj.rows[h['row_id']]['ability']==h['task'] and type(h['cost']) is int and h['cost']>0 and type(h['accepted']) is bool for i,h in enumerate(history)),'Invalid completed attempts')
        require({t:sum(int(h['accepted']) for h in history if h['task']==t) for t in TASKS}==state['accepted'],'Accepted count differs from committed attempts')
        debt=state['failure']['charged_tokens'] if state['failure'] else 0
        require(sum(h['cost'] for h in history)+debt==state['used_tokens'] and 0<=state['used_tokens']<=obj.cap,'Restored ledger differs')
        require(state['phase'] in {'READY','QUOTAS_COMPLETE','TOKEN_CAP','FAILED'} and (state['phase']=='FAILED')==bool(state['failure']),'Invalid terminal/failure state')
        if state['pending'] is not None:
            d=state['pending'];rid=d['row']['id'];require(d['attempt']==len(history) and rid in obj.rows and d['row']==obj.rows[rid] and d['task']==obj.rows[rid]['ability'] and d['input_tokens']==obj.tokens[rid] and d['reservation']==4*(obj.tokens[rid]+256),'Invalid pending evidence')
        obj.order=deepcopy(state['pool_order']);obj.cursor=dict(state['pool_cursor']);obj.rng.setstate(tuples(state['rng']));obj.accepted=dict(state['accepted']);obj.history=deepcopy(history)
        obj.used=state['used_tokens'];obj.phase=state['phase'];obj.pending=deepcopy(state['pending']);obj.failure=deepcopy(state['failure']);return obj

    def save(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.partial')
        state=self.snapshot()
        with tmp.open('w') as f:json.dump(state,f,sort_keys=True,allow_nan=False);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path);return hashlib.sha256(path.read_bytes()).hexdigest()

    @classmethod
    def load(cls,path,expected_file_sha256,rows,input_tokens):
        raw=Path(path).read_bytes();require(hashlib.sha256(raw).hexdigest()==expected_file_sha256,'Controller checkpoint file changed')
        state=json.loads(raw);return cls.restore(rows,input_tokens,state,digest(state))


def _step(backend):
    n=backend.optimizer_state()['step'];n=n.item() if hasattr(n,'item') else n
    require(type(n) is int and n>=0,'Invalid actual optimizer step');return n


def run_attempt(control,backend,workdir,persist,*,grader=None):
    """One real-backend-compatible attempt; persistence callback remains mandatory.

    persist(phase, controller_state, backend, evidence) MUST save adapter, full
    optimizer/RNG and controller state together. This adapter is NOT a reviewed
    native checkpoint implementation and cannot authorize a native V2 launch.
    """
    require(callable(persist),'Durable native/controller persistence callback required')
    d=control.offer()
    if d is None:return {'phase':control.phase,'accepted':False}
    assessment=None
    try:
        persist('reserved',control.snapshot(),backend,None)
        raw=backend.rollout(deepcopy(d['row']),d['attempt'],Path(workdir));backend.after_rollout()
        gs=grader(d['row'],raw) if grader else _grades(d['row'],raw,{**PROFILE,'group_size':4,'max_tokens':256})
        require(raw.get('record_id')==d['row']['id'] and len(raw['responses'])==4,'Actual rollout inventory/route changed')
        assessment=control.assess(d,gs,[len(r['token_ids']) for r in raw['responses']]);before=_step(backend);evidence=None
        if assessment['accepted']:evidence=backend.update(d['row'],raw,assessment['advantages'])
        require(_step(backend)==before+int(assessment['accepted']),'Actual optimizer update count disagrees with accepted group')
        control.commit(assessment);persist('committed',control.snapshot(),backend,{'raw':raw,'assessment':assessment,'update_evidence':evidence})
        return {'phase':control.phase,'accepted':assessment['accepted'],'cost':assessment['cost'],'native_checkpoint_verification':'NOT_VERIFIED_BY_CONTROLLER'}
    except BaseException:
        if control.pending is not None:control.fail(assessment)
        else:
            control.phase='FAILED';control.failure={'charged_tokens':0,'imputed':False,'native_state_requires_reconciliation':True}
        try:persist('failed',control.snapshot(),backend,{'assessment':assessment})
        except BaseException:pass
        raise
