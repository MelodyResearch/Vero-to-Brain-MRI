"""SHARED_ROLLOUT_BATCHED_ONE_STEP_GSPO_MECHANISM_CONTROL: backend-neutral CPU core.

IMPLEMENTED_CPU_NOT_NATIVE_QUALIFIED. No MLX import, model load, generation or
optimizer here. One same-person F/T TRAIN pair is rolled out ONCE (F4+T4) under the
unchanged Adam60 parent; the resulting immutable 8-response bundle is consumed by
both arms (A: F4,T4; B: F0F1T0T1,F2F3T2T3). Each admitted response carries
weight 1/8 and each arm performs AT MOST ONE actual optimizer update from the
identical full parent state. This is a different execution subtype from a future
SEQUENTIAL_ON_POLICY_GSPO (which updates after every K4). A typed bundle does not
authenticate real model outputs; the native producer and independent review do.
"""
from __future__ import annotations
from dataclasses import asdict,dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from mri_rl.paired_prompt_groups_v1 import (PromptContext,Grade,Response,make_plan,group_diagnostics,identity,canonical,
                                            inherited_Adam60_gate,STATUS as GROUPS_STATUS)

STATUS='IMPLEMENTED_CPU_NOT_NATIVE_QUALIFIED'
EXPERIMENT='SHARED_ROLLOUT_BATCHED_ONE_STEP_GSPO_MECHANISM_CONTROL'
EXECUTION_SUBTYPE='SHARED_ROLLOUT_NOT_SEQUENTIAL_ON_POLICY'
POLICY='F32_ACTIVATION_GENERATION_SCORER_GSPO_V1'
BUNDLE_SCHEMA='shared-rollout-gspo-bundle-v1'
ARMS={'A':'same_prompt','B':'paired_prompt'}
MODES=('qualify','train')
MAX_TOKENS=256
GROUP_SIZE=4
RESPONSES=8
WEIGHT=1/8
ROLLOUT_TOLERANCE=.002
OWN_OLD_TOLERANCE=1e-5
RESPONSE_IDS=tuple(f'{task}{i}' for task in ('F','T') for i in range(GROUP_SIZE))
CHARGEABLE=('generated_response_tokens','sampler_draws','generation_language_model_calls','generation_vision_embedding_calls',
            'detached_old_forwards','gradient_forwards','backward_calls','scorer_vision_embedding_calls',
            'scorer_language_model_calls','teacherforced_input_positions','optimizer_updates')
LABEL_MODALITY={'FLAIR':'F','contrast-enhanced T1':'T'}
ACTIONS={'qualify':'shared_rollout_gspo_readonly_qualification','train':'shared_rollout_gspo_one_step_train'}
SCOPE_KINDS={'qualify':'FUTURE_EXACT_SHARED_ROLLOUT_GSPO_READONLY_QUALIFICATION_SCOPE','train':'FUTURE_EXACT_SHARED_ROLLOUT_GSPO_ONE_STEP_TRAIN_SCOPE'}
REVIEWER_KINDS=('isolated_codex','claude_code')


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(value):return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)
def file_sha256(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()
def sync_dir(path):
    fd=os.open(path,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


# ---------------------------------------------------------------- budgets / counters
def derive_budget(prompt_lengths,mode):
    """Exact worst-case scope for ONE generation of 8 responses and both arms.

    Rollout once (8x256). Own-old/rollout-parity forwards are computed ONCE on the
    shared parent (8). Each arm backward includes its own fresh forward (<=8 each,
    only for gate-admitted groups): 16 backward, 16 gradient forwards. No arm
    generates; unused budget never transfers; no retry. Vision features are never
    cached across calls, so every scorer forward charges one vision embedding.
    """
    require(mode in MODES,'Unknown shared-rollout mode')
    require(isinstance(prompt_lengths,dict) and set(prompt_lengths)=={'F','T'} and
            all(type(v) is int and 0<v<=2048 for v in prompt_lengths.values()),'Exact native F/T prompt lengths required')
    old,gradient=RESPONSES,2*RESPONSES
    # per context: 4 own-old forwards + at most 4 responses x 2 arms of gradient forwards
    positions=sum((GROUP_SIZE+2*GROUP_SIZE)*(prompt_lengths[task]+MAX_TOKENS-1) for task in ('F','T'))
    return {'experiment':EXPERIMENT,'execution_subtype':EXECUTION_SUBTYPE,'mode':mode,'prompt_lengths':dict(prompt_lengths),
        'contexts':2,'responses':RESPONSES,'responses_per_context':GROUP_SIZE,'max_tokens':MAX_TOKENS,'new_generation_per_arm':0,
        'generated_response_tokens':RESPONSES*MAX_TOKENS,'sampler_draws':RESPONSES*(MAX_TOKENS+1),
        'generation_language_model_calls':RESPONSES*(MAX_TOKENS+4),'generation_vision_embedding_calls':RESPONSES,
        'detached_old_forwards':old,'gradient_forwards':gradient,'full_scorer_forwards':old+gradient,'backward_calls':gradient,
        'scorer_vision_embedding_calls':old+gradient,'scorer_language_model_calls':old+gradient,
        'vision_embedding_calls':RESPONSES+old+gradient,'language_model_calls':RESPONSES*(MAX_TOKENS+4)+old+gradient,
        'teacherforced_input_positions':positions,'optimizer_updates':2 if mode=='train' else 0,
        'optimizer_updates_per_arm':1 if mode=='train' else 0,'maximum_seconds':1800,
        'simultaneous_models':1,'simultaneous_image_autograd_graphs':1,'automatic_retry':False,'unused_budget_transfer':False,
        'counting':'Completed API endpoints, not kernel/hardware counts; checkpoint internal recomputation and one unreturned AR lookahead per response are additional work disclosed separately.'}


def counter_limits(budget):
    require(isinstance(budget,dict) and budget.get('mode') in MODES,'Derived budget required')
    require(all(type(budget.get(key)) is int and budget[key]>=0 for key in CHARGEABLE),'Finite integer chargeable budget required')
    return {key:budget[key] for key in CHARGEABLE}


class Counters:
    """Strict finite integer counters. A charge is atomic: refused charges mutate nothing."""
    def __init__(self,limits,used=None,persist=None):
        require(isinstance(limits,dict) and set(limits)==set(CHARGEABLE) and all(type(v) is int and v>=0 for v in limits.values()),'Exact counter inventory required')
        self.limits=dict(limits)
        if used is None:used={key:0 for key in CHARGEABLE}
        require(isinstance(used,dict) and set(used)==set(CHARGEABLE) and all(type(v) is int and 0<=v<=self.limits[k] for k,v in used.items()),
                'Restored counters outside exact finite limits')
        self.used=dict(used);self.persist=persist
    def charge(self,name,amount=1):self.charge_many({name:amount})
    def charge_many(self,amounts):
        require(isinstance(amounts,dict) and amounts and set(amounts)<=set(CHARGEABLE),'Unknown counter')
        require(all(type(v) is int and v>=0 for v in amounts.values()),'Counter charge must be a nonnegative integer')
        require(all(self.used[k]+v<=self.limits[k] for k,v in amounts.items()),'Frozen native call/token budget exhausted before operation')
        previous=dict(self.used)
        for key,value in amounts.items():self.used[key]+=value
        if self.persist is not None:
            try:self.persist(dict(self.used))
            except BaseException:
                self.used=previous;raise
    def snapshot(self):return dict(self.used)


# ---------------------------------------------------------------- durable once-only files
def publish_once(path,value):
    """Write temp, fsync, link to final name: a final name appears complete or never, and never twice."""
    path=Path(path);payload=(canonical(value)+'\n').encode()
    fd,name=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as handle:handle.write(payload);handle.flush();os.fsync(handle.fileno())
        os.link(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)
    sync_dir(path.parent);return path


def replace_atomic(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.partial')
    with temporary.open('w') as handle:handle.write(canonical(value)+'\n');handle.flush();os.fsync(handle.fileno())
    os.replace(temporary,path);sync_dir(path.parent)


# ---------------------------------------------------------------- immutable shared bundle
def _finite(values):return all(type(v) in (int,float) and math.isfinite(v) for v in values)


def _validate_stream(item,eos_ids):
    tokens=item['token_ids'];logps=item['raw_selected_logps']
    require(isinstance(tokens,list) and 0<len(tokens)<=MAX_TOKENS and all(type(t) is int and t>=0 for t in tokens),'Bounded response token IDs required')
    require(isinstance(logps,list) and len(logps)==len(tokens) and _finite(logps) and all(v<=0 for v in logps),'Finite nonpositive selected rollout logp per token required')
    require(item.get('generation_tokens')==len(tokens),'Generation token count differs')
    finish=item.get('finish_reason');eos=set(eos_ids)
    require(eos and all(type(t) is int and t>=0 for t in eos),'Effective EOS set required')
    require(not any(t in eos for t in tokens[:-1]),'Token after EOS')
    if finish=='stop':
        require(tokens[-1] in eos and item.get('EOS_coverage')=='OBSERVED_AND_SCORED','Stop termination requires scored EOS')
    else:
        require(finish=='length' and len(tokens)==MAX_TOKENS and tokens[-1] not in eos and item.get('EOS_coverage')=='NOT_OBSERVED_LENGTH_LIMIT','Length termination receipt differs')


def _grade_dict(grade):
    require(isinstance(grade,dict) and all(k in grade for k in ('accuracy','format','total','overlong','raw_accuracy')),'Complete original reward receipt required')
    return {k:grade[k] for k in sorted(grade)}


def reward_receipt(response_id,context_sha256,token_sha256,text_sha256,logps_sha256,grade):
    return identity({'response_id':response_id,'context_sha256':context_sha256,'token_sha256':token_sha256,'text_sha256':text_sha256,
                     'selected_logps_sha256':logps_sha256,'grade':grade})


def prompt_context(modality,record):
    return PromptContext(record['row_id'],record['pair_id'],modality,record['prompt_text_sha256'],record['image_sha256'],
                         record['checkpoint_sha256'],record['policy_sha256'],record['reward_policy_sha256'])


def freeze_bundle(*,parent,contexts,responses,effective_eos_ids):
    """Build the one immutable JSON bundle from the SINGLE F/T rollout. Never mutated afterwards.

    contexts: {'F':record,'T':record}; responses: eight dicts in order F0..F3,T0..T3 containing the
    native capture (text, token_ids, raw_selected_logps, finish_reason, EOS_coverage, generation_tokens)
    plus the unchanged original grader result as 'grade'.
    """
    require(isinstance(parent,dict) and sha(parent.get('receipt_sha256')) and sha(parent.get('adapter_sha256')),'Exact parent receipt/adapter identity required')
    require(isinstance(contexts,dict) and set(contexts)=={'F','T'},'Exactly one F and one T context required')
    prepared={}
    for modality,record in contexts.items():
        require(all(sha(record.get(k)) for k in ('prompt_text_sha256','image_sha256','checkpoint_sha256','policy_sha256','reward_policy_sha256','rendered_prompt_token_ids_sha256','input_fingerprints_sha256')),
                'Context prompt/image/token/input fingerprint hashes required')
        require(record['checkpoint_sha256']==parent['adapter_sha256'] and record['policy_sha256']==identity(POLICY),'Context must bind the exact parent adapter and numerical policy')
        require(type(record.get('prompt_length')) is int and 0<record['prompt_length']<=2048 and record.get('row_id') and record.get('pair_id'),'Context identity incomplete')
        context=prompt_context(modality,record);prepared[modality]={**record,'context_sha256':context.sha256}
    require(isinstance(responses,list) and len(responses)==RESPONSES,'Exactly eight responses required')
    items=[]
    for expected,item in zip(RESPONSE_IDS,responses):
        require(item.get('response_id')==expected,'Response order/identity must be F0..F3,T0..T3')
        modality=expected[0];_validate_stream(item,effective_eos_ids[modality])
        grade=_grade_dict(item['grade']);Grade(accuracy=grade['accuracy'],format=grade['format'],total=grade['total'],overlong=grade['overlong'],raw_accuracy=grade['raw_accuracy'])
        token_sha=identity(tuple(item['token_ids']));logps_sha=identity(item['raw_selected_logps']);text_sha=hashlib.sha256(item['text'].encode()).hexdigest()
        receipt=reward_receipt(expected,prepared[modality]['context_sha256'],token_sha,text_sha,logps_sha,grade)
        items.append({'response_id':expected,'modality':modality,'index':int(expected[1]),'text':item['text'],'text_sha256':text_sha,
            'token_ids':list(item['token_ids']),'token_sha256':token_sha,'raw_selected_logps':list(item['raw_selected_logps']),'selected_logps_sha256':logps_sha,
            'finish_reason':item['finish_reason'],'EOS_coverage':item['EOS_coverage'],'generation_tokens':item['generation_tokens'],
            'effective_eos_ids':sorted(effective_eos_ids[modality]),'grade':grade,'reward_receipt_sha256':receipt})
    body={'schema':BUNDLE_SCHEMA,'experiment':EXPERIMENT,'execution_subtype':EXECUTION_SUBTYPE,'numerical_policy':POLICY,'parent':dict(parent),
          'contexts':prepared,'responses':items}
    return {**body,'bundle_sha256':identity(body)}


@dataclass(frozen=True)
class Bundle:
    raw:dict
    contexts:dict
    responses:tuple
    sha256:str


def load_bundle(value):
    """Re-validate a stored bundle (all identities, EOS, receipts) before anything consumes it."""
    require(isinstance(value,dict) and value.get('schema')==BUNDLE_SCHEMA and value.get('experiment')==EXPERIMENT,'Unknown shared bundle')
    body={k:v for k,v in value.items() if k!='bundle_sha256'}
    require(sha(value.get('bundle_sha256')) and identity(body)==value['bundle_sha256'],'Bundle hash differs')
    raw=json.loads(canonical(value));parent=raw['parent']
    contexts={m:prompt_context(m,raw['contexts'][m]) for m in ('F','T')}
    for m,context in contexts.items():
        require(raw['contexts'][m]['context_sha256']==context.sha256 and context.checkpoint_sha256==parent['adapter_sha256'] and context.policy_sha256==identity(POLICY),'Context identity differs')
    require(len(raw['responses'])==RESPONSES and tuple(r['response_id'] for r in raw['responses'])==RESPONSE_IDS,'Exact eight-response inventory required')
    responses=[]
    for item in raw['responses']:
        m=item['modality'];_validate_stream(item,item['effective_eos_ids'])
        require(item['token_sha256']==identity(tuple(item['token_ids'])) and item['selected_logps_sha256']==identity(item['raw_selected_logps']) and
                item['text_sha256']==hashlib.sha256(item['text'].encode()).hexdigest(),'Token/logp/text binding differs')
        require(item['reward_receipt_sha256']==reward_receipt(item['response_id'],contexts[m].sha256,item['token_sha256'],item['text_sha256'],item['selected_logps_sha256'],item['grade']),'Immutable reward receipt binding differs')
        g=item['grade'];responses.append(Response(item['response_id'],contexts[m],tuple(item['token_ids']),tuple(float(v) for v in item['raw_selected_logps']),
            Grade(accuracy=g['accuracy'],format=g['format'],total=g['total'],overlong=g['overlong'],raw_accuracy=g['raw_accuracy']),item['reward_receipt_sha256']))
    return Bundle(raw,contexts,tuple(responses),raw['bundle_sha256'])


# ---------------------------------------------------------------- arm plans
@dataclass(frozen=True)
class ArmPlan:
    arm:str
    plan:object
    admitted_groups:int
    admitted_response_ids:tuple
    backward_calls_needed:int


def arm_plan(bundle,arm):
    """Same eight Response objects in both arms; only group membership differs. Weight 1/8 once per admitted response."""
    require(isinstance(bundle,Bundle) and arm in ARMS,'Typed bundle and arm A/B required')
    plan=make_plan(bundle.responses,ARMS[arm])
    require(len(plan.items)==RESPONSES and {i.response.response_id for i in plan.items}==set(RESPONSE_IDS),'Plan must contain exactly the shared eight responses')
    for item in plan.items:
        require(item.weight==(WEIGHT if item.admitted else 0.),'Admitted response weight must be exactly 1/8')
        require(item.response is bundle.responses[RESPONSE_IDS.index(item.response.response_id)],'Plan response differs from immutable bundle object')
    admitted=tuple(sorted((i.response.response_id for i in plan.items if i.weight>0),key=RESPONSE_IDS.index))
    return ArmPlan(arm,plan,plan.effective_groups,admitted,len(admitted))


def gate_report(bundle):
    """Descriptive group gate evidence for both arms (inherited Adam60 valid-format-subset gate, identical in both)."""
    report={}
    for arm,grouping in ARMS.items():
        planned=arm_plan(bundle,arm)
        report[arm]={'grouping':grouping,'gate':'uniform_balance_continuation_gate_v1.accuracy_group_is_informative (inherited Adam60)',
            'effective_groups':planned.admitted_groups,'admitted_response_ids':list(planned.admitted_response_ids),
            'weights':{i.response.response_id:i.weight for i in planned.plan.items},'advantages':{i.response.response_id:i.advantage for i in planned.plan.items},
            'group_membership':[[i.response.response_id for i in planned.plan.items if i.group_index==g] for g in (0,1)],
            'diagnostics':group_diagnostics(bundle.responses,grouping)}
    return report


def order_key(item):return (item.response.context.modality,item.response.response_id)


def weighted_gradient_sum(plan,response_gradient,scale,add):
    """Per-response autograd gradients summed with their 1/8 weight (serial; one graph at a time).

    response_gradient(item)->{'loss','gradient','old_current_max_abs'} supplies an actual unchanged-loss
    gradient. scale(tree,weight) and add(left,right) are the backend tree maps. Returns (total or None, evidence).
    """
    total=None;records=[];weighted_loss=0.
    for item in sorted((i for i in plan.items if i.weight>0),key=order_key):
        result=response_gradient(item)
        loss=float(result['loss']);error=float(result['old_current_max_abs'])
        require(math.isfinite(loss) and math.isfinite(error) and 0<=error<=OWN_OLD_TOLERANCE,'Own-old/current 1e-5 or finite-loss gate failed')
        scaled=scale(result['gradient'],item.weight)
        total=scaled if total is None else add(total,scaled)
        weighted_loss+=item.weight*loss
        records.append({'response_id':item.response.response_id,'advantage':item.advantage,'weight':item.weight,'loss':loss,'old_current_max_abs':error})
    return total,{'responses':records,'weighted_loss':weighted_loss,'backward_calls':len(records)}


def update_decision(planned,gradient_l2):
    """Zero admitted groups or a zero gradient is NEGATIVE evidence: 0 updates, no resample, no momentum-only step."""
    if planned.admitted_groups==0:return {'update':False,'reason':'NO_ADMITTED_GROUP'}
    require(type(gradient_l2) in (int,float) and math.isfinite(gradient_l2) and gradient_l2>=0,'Finite gradient norm required')
    if gradient_l2==0:return {'update':False,'reason':'ZERO_GRADIENT_NO_MOMENTUM_STEP'}
    return {'update':True,'reason':'ADMITTED_FINITE_NONZERO_GRADIENT'}


def snapshot_identity(snapshot):
    """Full parameter + Adam + Python/MLX RNG identity of an exact transaction snapshot."""
    return identity({'parameters':snapshot['parameter_fingerprints'],'state':snapshot['state']})


# ---------------------------------------------------------------- parity
def selected_logp_parity(sampled,scored):
    """Own-context rollout selected logps vs same-context scorer (.002)."""
    require(len(sampled)==len(scored)>0 and _finite([*sampled,*scored]),'Finite complete rollout/scorer vectors required')
    error=max(abs(float(a)-float(b)) for a,b in zip(sampled,scored))
    require(error<=ROLLOUT_TOLERANCE,'Rollout/own-context scorer selected logp mismatch exceeds .002')
    return {'max_abs_error':error,'tolerance':ROLLOUT_TOLERANCE,'positions':len(scored)}


def parity_report(bundle,old):
    require(isinstance(old,dict) and set(old)==set(RESPONSE_IDS),'Own-old inventory must be exactly eight responses')
    return {item['response_id']:selected_logp_parity(item['raw_selected_logps'],old[item['response_id']]) for item in bundle.raw['responses']}


# ---------------------------------------------------------------- fail-closed authorization
def scope_digest(scope):return identity(scope)


def authorize_launch(mode,scope,human,review,*,qualification=None,source_sha256=None,consumed=None):
    """Existing admission-contract shape: exact scope + FRESH human action + independent exact review.

    Local booleans (approved/authorized flags in the scope) never authorize. Qualification
    execution and training use different human actions; training additionally needs an
    independently reviewed actual read-only qualification proof for the same source closure.
    `consumed` is a callable once-only ledger so a human instruction cannot be reused.
    """
    require(mode in MODES,'Unknown mode')
    require(isinstance(scope,dict) and scope.get('scope_kind')==SCOPE_KINDS[mode],'Preparation is not an exact launch scope')
    require(not any(scope.get(k) is True for k in ('approved','authorized','training_authorized','native_approved','READY_FOR_TRAINING')),'A local approval flag cannot authorize')
    require(scope.get('experiment')==EXPERIMENT and scope.get('execution_subtype')==EXECUTION_SUBTYPE and scope.get('numerical_policy')==POLICY,'Experiment/subtype/policy differs')
    budget=scope.get('budget');require(isinstance(budget,dict) and budget==derive_budget(budget.get('prompt_lengths'),mode),'Scope budget is not the exact derived budget')
    require(source_sha256 is None or scope.get('source_sha256')==source_sha256,'Reviewed source closure differs from current source')
    binding=scope_digest(scope)
    require(isinstance(human,dict) and human.get('authorized') is True and human.get('action')==ACTIONS[mode] and human.get('scope_sha256')==binding and
            sha(human.get('human_instruction_sha256')),'Fresh exact human authorization for this action and scope required')
    require(isinstance(review,dict) and review.get('scope_sha256')==binding and review.get('verdict') in ('PASS','PASS_WITH_LIMITATIONS') and
            review.get('reviewer_id') and review.get('reviewer_kind') in REVIEWER_KINDS and review.get('source_sha256')==scope.get('source_sha256') and
            review.get('input_sha256')==scope.get('input_sha256'),'Independent exact source/input/scope review required')
    if mode=='train':
        proof,proof_review=(qualification or {}).get('proof'),(qualification or {}).get('review')
        require(isinstance(proof,dict) and proof.get('status')=='ACTUAL_SHARED_ROLLOUT_GSPO_READONLY_QUALIFIED' and proof.get('optimizer_updates')==0 and
                proof.get('source_sha256')==scope.get('source_sha256') and proof.get('input_sha256')==scope.get('input_sha256'),'Actual same-source read-only qualification required before training')
        require(isinstance(proof_review,dict) and proof_review.get('verdict') in ('PASS','PASS_WITH_LIMITATIONS') and proof_review.get('actual_proof_sha256')==identity(proof) and
                proof_review.get('reviewer_id') and proof_review.get('reviewer_kind') in REVIEWER_KINDS,'Independent review of the actual qualification proof required')
    if consumed is not None:consumed(human['human_instruction_sha256'])
    return True


def consume_once(directory):
    """Once-only human-instruction ledger inside the private output tree."""
    directory=Path(directory)
    def consume(instruction_sha256):
        directory.mkdir(parents=True,exist_ok=True)
        try:publish_once(directory/f'authorization_consumed_{instruction_sha256}.json',{'consumed':instruction_sha256})
        except FileExistsError:raise ValueError('Human authorization instruction was already consumed; a fresh instruction is required') from None
    return consume
