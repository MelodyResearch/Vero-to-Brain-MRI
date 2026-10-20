"""CPU fixtures only: routing, accounting, transactions and fail-closed gates. Never native or MRI evidence."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import numpy as np
import pytest
from mri_rl import shared_rollout_gspo_core_v1 as core
from mri_rl import shared_rollout_gspo_driver_v1 as drv
from mri_rl import paired_z045_continuation_driver_v1 as old
from mri_rl.optimizer_checkpoint import load_optimizer_checkpoint
from mri_rl.uniform_balance_continuation_gate_v1 import parent_anchor

LENGTHS={'F':213,'T':213}
ACCURACY={'F':[1,0,1,0],'T':[0,1,0,0]}      # all four K4/pair groups informative
EOS=99


def sha(text):return hashlib.sha256(text.encode()).hexdigest()


# ------------------------------------------------------------------ pure bundle fixtures
def grade(accuracy,fmt=1.,overlong=0.):
    return {'accuracy':float(accuracy),'raw_accuracy':float(accuracy),'format':fmt,'overlong':overlong,'score':.8*accuracy+.2*fmt,'total':.8*accuracy+.2*fmt+overlong}


def contexts_record(parent_adapter,pair='pair'):
    base={'pair_id':pair,'prompt_text_sha256':sha('generic question'),'checkpoint_sha256':parent_adapter,'policy_sha256':core.identity(core.POLICY),
          'reward_policy_sha256':sha('calibration'),'prompt_length':213}
    return {m:{**base,'row_id':'row-'+m,'image_sha256':sha('image'+m),'rendered_prompt_token_ids_sha256':core.identity([1]),'input_fingerprints_sha256':sha('fp'+m)} for m in ('F','T')}


def response_items(accuracy=ACCURACY,fmt=None):
    items=[]
    for m in ('F','T'):
        for i in range(4):
            tokens=[10+i,20+i,EOS];logps=[-1.-.1*i,-2.,-.5]
            items.append({'response_id':f'{m}{i}','text':f'<think>{m}{i}</think>','token_ids':tokens,'raw_selected_logps':logps,'finish_reason':'stop',
                          'EOS_coverage':'OBSERVED_AND_SCORED','generation_tokens':3,'grade':grade(accuracy[m][i],(fmt or {}).get(f'{m}{i}',1.))})
    return items


PARENT={'receipt_sha256':sha('parent receipt'),'adapter_sha256':sha('parent adapter')}


def frozen(accuracy=ACCURACY,fmt=None):
    return core.freeze_bundle(parent=PARENT,contexts=contexts_record(PARENT['adapter_sha256']),responses=response_items(accuracy,fmt),effective_eos_ids={'F':[EOS],'T':[EOS]})


def bundle(accuracy=ACCURACY,fmt=None):return core.load_bundle(frozen(accuracy,fmt))


def test_bundle_binds_prompt_image_tokens_logps_eos_and_reward_immutably():
    raw=frozen();loaded=core.load_bundle(raw)
    assert len(loaded.responses)==8 and [r.response_id for r in loaded.responses]==list(core.RESPONSE_IDS)
    assert {r.context.modality for r in loaded.responses[:4]}=={'F'} and loaded.responses[0].context.image_sha256!=loaded.responses[4].context.image_sha256
    assert loaded.responses[0].context.prompt_sha256==loaded.responses[4].context.prompt_sha256
    for edit in (lambda b:b['responses'][0]['token_ids'].__setitem__(0,11),lambda b:b['responses'][1]['grade'].__setitem__('accuracy',1.),
                 lambda b:b['contexts']['T'].__setitem__('image_sha256',sha('swapped')),lambda b:b['responses'][2]['raw_selected_logps'].__setitem__(0,-9.),
                 lambda b:b['responses'][3].__setitem__('text','changed')):
        tampered=copy.deepcopy(raw);edit(tampered)
        with pytest.raises(ValueError):core.load_bundle(tampered)
    # a self-consistent re-hash still cannot change a response's receipt binding
    tampered=copy.deepcopy(raw);tampered['responses'][0]['grade']['accuracy']=0.;body={k:v for k,v in tampered.items() if k!='bundle_sha256'};tampered['bundle_sha256']=core.identity(body)
    with pytest.raises(ValueError,match='reward receipt'):core.load_bundle(tampered)


def test_eos_and_length_receipts_and_order_are_enforced():
    items=response_items();items[0]['token_ids']=[EOS,5,EOS]
    with pytest.raises(ValueError,match='after EOS'):core.freeze_bundle(parent=PARENT,contexts=contexts_record(PARENT['adapter_sha256']),responses=items,effective_eos_ids={'F':[EOS],'T':[EOS]})
    items=response_items();items[0]['finish_reason']='length'
    with pytest.raises(ValueError,match='Length termination'):core.freeze_bundle(parent=PARENT,contexts=contexts_record(PARENT['adapter_sha256']),responses=items,effective_eos_ids={'F':[EOS],'T':[EOS]})
    items=response_items()[::-1]
    with pytest.raises(ValueError,match='order'):core.freeze_bundle(parent=PARENT,contexts=contexts_record(PARENT['adapter_sha256']),responses=items,effective_eos_ids={'F':[EOS],'T':[EOS]})
    other=contexts_record(sha('other adapter'))
    with pytest.raises(ValueError,match='exact parent adapter'):core.freeze_bundle(parent=PARENT,contexts=other,responses=response_items(),effective_eos_ids={'F':[EOS],'T':[EOS]})


def test_both_arms_use_the_same_eight_response_objects_and_groups_differ_only():
    b=bundle();a=core.arm_plan(b,'A');p=core.arm_plan(b,'B')
    by_id=lambda planned:sorted((i.response for i in planned.plan.items),key=lambda r:core.RESPONSE_IDS.index(r.response_id))
    assert by_id(a)==by_id(p)==list(b.responses) and all(x is y for x,y in zip(by_id(p),b.responses))
    groups=lambda planned:[[i.response.response_id for i in planned.plan.items if i.group_index==g] for g in (0,1)]
    assert groups(a)==[['F0','F1','F2','F3'],['T0','T1','T2','T3']] and groups(p)==[['F0','F1','T0','T1'],['F2','F3','T2','T3']]
    assert a.admitted_groups==p.admitted_groups==2 and len(a.admitted_response_ids)==len(p.admitted_response_ids)==8
    assert all(i.weight==1/8 for i in a.plan.items+p.plan.items if i.admitted)
    assert core.WEIGHT==.125 and a.backward_calls_needed==p.backward_calls_needed==8


def test_weight_is_one_eighth_once_and_unadmitted_groups_carry_zero_weight():
    b=bundle({'F':[1,1,1,1],'T':[0,1,0,0]})              # F group has no within-group accuracy variation
    a=core.arm_plan(b,'A')
    assert a.admitted_groups==1 and a.admitted_response_ids==('T0','T1','T2','T3') and a.backward_calls_needed==4
    assert [i.weight for i in a.plan.items]==[0.]*4+[1/8]*4
    p=core.arm_plan(b,'B')
    assert all(i.weight in (0.,1/8) for i in p.plan.items)


def test_inherited_adam60_valid_subset_gate_is_used_in_both_arms_not_the_historical18_gate():
    fmt={'F2':.5}                                           # half-format response inside an otherwise informative group
    b=bundle({'F':[1,0,1,0],'T':[0,1,0,0]},fmt)
    report=core.gate_report(b)
    assert report['A']['gate']==report['B']['gate'] and 'inherited Adam60' in report['A']['gate']
    from mri_rl.mlx_gspo_smoke import task_group_is_informative as historical18
    grades=[{**r.grade.__dict__} for r in b.responses[:4]]
    assert not historical18(grades) and report['A']['effective_groups']>=1   # historical gate would reject; inherited admits


def test_inactive_and_zero_gradient_are_negative_evidence_without_update():
    b=bundle({'F':[1,1,1,1],'T':[0,0,0,0]})
    a=core.arm_plan(b,'A');p=core.arm_plan(b,'B')
    assert a.admitted_groups==0 and core.update_decision(a,0.)=={'update':False,'reason':'NO_ADMITTED_GROUP'}
    assert p.admitted_groups==2 and core.update_decision(p,0.)['reason']=='ZERO_GRADIENT_NO_MOMENTUM_STEP' and core.update_decision(p,1e-9)['update']
    with pytest.raises(ValueError):core.update_decision(p,float('nan'))


def test_derived_budget_is_exact_and_mode_dependent():
    t=core.derive_budget(LENGTHS,'train');q=core.derive_budget(LENGTHS,'qualify')
    assert (t['generated_response_tokens'],t['sampler_draws'],t['generation_language_model_calls'],t['generation_vision_embedding_calls'])==(2048,2056,2080,8)
    assert (t['detached_old_forwards'],t['gradient_forwards'],t['backward_calls'],t['full_scorer_forwards'],t['vision_embedding_calls'])==(8,16,16,24,32)
    assert t['teacherforced_input_positions']==24*(213+255)==11232 and t['new_generation_per_arm']==0
    assert t['optimizer_updates']==2 and t['optimizer_updates_per_arm']==1 and q['optimizer_updates']==0
    assert t['automatic_retry'] is False and t['unused_budget_transfer'] is False and t['simultaneous_models']==1
    for bad in ({'F':213},{'F':213,'T':0},{'F':213.,'T':213}):
        with pytest.raises(ValueError):core.derive_budget(bad,'train')


def test_counters_atomic_at_exact_limits_and_restored_state_validated():
    limits=core.counter_limits(core.derive_budget(LENGTHS,'train'));saved=[]
    c=core.Counters(limits,persist=lambda used:saved.append(dict(used)))
    c.charge('backward_calls',16);assert c.used['backward_calls']==16 and saved[-1]['backward_calls']==16
    before=c.snapshot()
    with pytest.raises(ValueError):c.charge('backward_calls')
    with pytest.raises(ValueError):c.charge_many({'detached_old_forwards':1,'backward_calls':1})   # whole charge refused
    assert c.snapshot()==before
    for bad in (True,-1,1.5):
        with pytest.raises(ValueError):c.charge('detached_old_forwards',bad)
    with pytest.raises(ValueError):c.charge('unknown_counter')
    def failing(used):raise OSError('journal write failed')
    d=core.Counters(limits,persist=failing)
    with pytest.raises(OSError):d.charge('sampler_draws',5)
    assert d.used['sampler_draws']==0
    with pytest.raises(ValueError):core.Counters(limits,{**c.snapshot(),'backward_calls':17})
    with pytest.raises(ValueError):core.Counters(limits,{k:v for k,v in c.snapshot().items() if k!='backward_calls'})
    assert core.Counters(limits,c.snapshot()).used==c.snapshot()


def test_weighted_gradient_sum_applies_one_eighth_per_admitted_response_and_gates_alignment():
    b=bundle()
    def make(plan_arm):
        planned=core.arm_plan(b,plan_arm)
        index=lambda item:core.RESPONSE_IDS.index(item.response.response_id)+1
        gradient=lambda item:{'loss':1.,'gradient':np.array([item.advantage*index(item),1.]),'old_current_max_abs':0.}
        total,evidence=core.weighted_gradient_sum(planned.plan,gradient,lambda tree,w:tree*w,lambda x,y:x+y)
        expected=sum(i.weight*np.array([i.advantage*index(i),1.]) for i in planned.plan.items)
        np.testing.assert_allclose(total,expected,rtol=1e-14,atol=1e-15);return total,evidence,planned
    ta,ea,pa=make('A');tb,eb,pb=make('B')
    assert ea['backward_calls']==eb['backward_calls']==8 and ea['weighted_loss']==pytest.approx(8*(1/8)) and [r['weight'] for r in ea['responses']]==[1/8]*8
    assert [r['response_id'] for r in ea['responses']]==list(core.RESPONSE_IDS)
    assert not np.allclose(ta,tb)               # same responses, different group baselines
    bad=lambda item:{'loss':1.,'gradient':np.array([1.]),'old_current_max_abs':2e-5}
    with pytest.raises(ValueError,match='1e-5'):core.weighted_gradient_sum(pa.plan,bad,lambda t,w:t*w,lambda x,y:x+y)
    none=core.weighted_gradient_sum(core.arm_plan(bundle({'F':[1]*4,'T':[0]*4}),'A').plan,bad,lambda t,w:t*w,lambda x,y:x+y)
    assert none[0] is None and none[1]['backward_calls']==0


def test_selected_logp_parity_gate_is_point_zero_zero_two():
    b=bundle();old_logps={r['response_id']:[v+.001 for v in r['raw_selected_logps']] for r in b.raw['responses']}
    assert all(v['max_abs_error']<=.002 for v in core.parity_report(b,old_logps).values())
    old_logps['T3'][1]+=.01
    with pytest.raises(ValueError,match='exceeds .002'):core.parity_report(b,old_logps)
    with pytest.raises(ValueError):core.parity_report(b,{k:v for k,v in old_logps.items() if k!='T3'})


def test_publish_once_never_replaces_and_leaves_no_partial_name(tmp_path):
    path=tmp_path/'marker.json';core.publish_once(path,{'a':1})
    with pytest.raises(FileExistsError):core.publish_once(path,{'a':2})
    assert json.loads(path.read_text())=={'a':1} and not [p for p in tmp_path.iterdir() if p.name!='marker.json']


# ------------------------------------------------------------------ fail-closed authorization
def launch_pack(mode,tmp_path=None):
    budget=core.derive_budget(LENGTHS,mode)
    scope={'scope_kind':core.SCOPE_KINDS[mode],'experiment':core.EXPERIMENT,'execution_subtype':core.EXECUTION_SUBTYPE,'numerical_policy':core.POLICY,
           'budget':budget,'source_sha256':{'/src/a.py':sha('a')},'input_sha256':sha('inputs')}
    binding=core.scope_digest(scope)
    human={'authorized':True,'action':core.ACTIONS[mode],'scope_sha256':binding,'human_instruction_sha256':sha('instruction-'+mode)}
    review={'verdict':'PASS_WITH_LIMITATIONS','scope_sha256':binding,'reviewer_id':'isolated','reviewer_kind':'isolated_codex','source_sha256':scope['source_sha256'],'input_sha256':scope['input_sha256']}
    proof={'status':'ACTUAL_SHARED_ROLLOUT_GSPO_READONLY_QUALIFIED','optimizer_updates':0,'blockers':[],'source_sha256':scope['source_sha256'],'input_sha256':scope['input_sha256'],
           **{k:sha(k) for k in ('bundle_sha256','old_sha256','release_witness_sha256','run_identity_sha256')}}
    qualification={'proof':proof,'review':{'verdict':'PASS','actual_proof_sha256':core.identity(proof),'reviewer_id':'isolated','reviewer_kind':'isolated_codex'}}
    return scope,human,review,qualification


def test_authorization_is_fail_closed_and_has_no_local_approved_certificate():
    before=set(sys.modules)
    scope,human,review,qual=launch_pack('qualify')
    assert core.authorize_launch('qualify',scope,human,review,source_sha256=scope['source_sha256'])
    for flag in ('approved','authorized','training_authorized','native_approved','READY_FOR_TRAINING'):
        with pytest.raises(ValueError,match='local approval flag'):core.authorize_launch('qualify',{**scope,flag:True},human,review)
    with pytest.raises(ValueError):core.authorize_launch('qualify',scope,{**human,'authorized':'yes'},review)
    with pytest.raises(ValueError):core.authorize_launch('qualify',scope,{**human,'action':core.ACTIONS['train']},review)
    with pytest.raises(ValueError):core.authorize_launch('qualify',scope,{**human,'scope_sha256':sha('other')},review)
    with pytest.raises(ValueError):core.authorize_launch('qualify',scope,human,{**review,'verdict':'FAIL'})
    with pytest.raises(ValueError):core.authorize_launch('qualify',scope,human,{**review,'reviewer_kind':'self'})
    with pytest.raises(ValueError):core.authorize_launch('qualify',scope,human,{**review,'source_sha256':{'/src/a.py':sha('edited')}})
    with pytest.raises(ValueError,match='Reviewed source closure'):core.authorize_launch('qualify',scope,human,review,source_sha256={'/src/a.py':sha('edited')})
    with pytest.raises(ValueError,match='derived budget'):core.authorize_launch('qualify',{**scope,'budget':{**scope['budget'],'backward_calls':17}},human,review)
    with pytest.raises(ValueError,match='exact launch scope'):core.authorize_launch('qualify',{**scope,'scope_kind':'PREPARATION_ONLY_NOT_LAUNCH_SCOPE'},human,review)
    assert not any(n=='mlx' or n.startswith('mlx.') for n in set(sys.modules)-before)


def test_training_needs_its_own_human_action_and_independently_reviewed_qualification():
    scope,human,review,qual=launch_pack('train')
    assert core.authorize_launch('train',scope,human,review,qualification=qual)
    with pytest.raises(ValueError,match='qualification'):core.authorize_launch('train',scope,human,review)
    with pytest.raises(ValueError,match='qualification'):core.authorize_launch('train',scope,human,review,qualification={'proof':{**qual['proof'],'optimizer_updates':1},'review':qual['review']})
    with pytest.raises(ValueError):core.authorize_launch('train',scope,human,review,qualification={'proof':qual['proof'],'review':{**qual['review'],'actual_proof_sha256':sha('x')}})
    qscope,qhuman,qreview,_=launch_pack('qualify')            # a qualification authorization cannot train
    with pytest.raises(ValueError):core.authorize_launch('train',scope,qhuman,review,qualification=qual)


def test_human_instruction_is_consumed_once(tmp_path):
    scope,human,review,_=launch_pack('qualify');consume=core.consume_once(tmp_path/'ledger')
    assert core.authorize_launch('qualify',scope,human,review,consumed=consume)
    with pytest.raises(ValueError,match='already consumed'):core.authorize_launch('qualify',scope,human,review,consumed=consume)


def test_entry_refuses_before_any_native_import():
    from types import SimpleNamespace
    from mri_rl import shared_rollout_gspo_entry_v1 as e
    before=set(sys.modules)
    for mode in core.MODES:
        with pytest.raises(ValueError,match='NOT_AUTHORIZED'):e.execute(mode,SimpleNamespace())
    assert not any(n=='mlx' or n.startswith('mlx.') for n in set(sys.modules)-before)


def test_modules_import_without_mlx_and_historical_paths_stay_disabled():
    import mri_rl.shared_rollout_gspo_backend_v1 as b
    assert not any(n=='mlx' or n.startswith('mlx.') for n in sys.modules)
    backend=b.SharedRolloutBackend.__new__(b.SharedRolloutBackend)
    for call in (lambda:backend.rollout(),lambda:backend.update(),lambda:backend._logps([1])):
        with pytest.raises(ValueError,match='DISABLED'):call()
    with pytest.raises(ValueError,match='NOT_QUALIFIED'):backend.greedy_generate()     # BF16-inherited evaluation unreachable


# ------------------------------------------------------------------ CPU fake backend + parent fixture
def load(name):
    path=Path(__file__).with_name(name);spec=importlib.util.spec_from_file_location(name[:-3],path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


NUMERIC=load('test_continuous_rl_driver.py').NumericBackend


class Fake(NUMERIC):
    """Genuine Adam/RNG/serialization (inherited fixture) + the shared-rollout backend API on numpy."""
    instances=[];log=None;rollouts=0;accuracy=ACCURACY
    fail_update_arm=None
    def __init__(self,adapter,config):
        super().__init__(adapter,config);self.arm_updates=0;self.events=[];Fake.instances.append(self);self.rows=None;self.bound=None
    def save_adapter(self,path):
        super().save_adapter(path);Path(path).with_name('adapter_config.json').write_text('{}')
    def configure_shared(self,*,counters,rows,parent,binding_sha256,expected_image_sha256):
        self.counters=counters;self.rows=rows;self.expected=expected_image_sha256
    def bind_bundle(self,bundle):self.bound=bundle.sha256
    def _params(self):return hashlib.sha256(self.weight.tobytes()).hexdigest()
    def _state(self):
        arrays=[self.state['step'],self.state['learning_rate'],*[v for k,v in sorted(self.state.get('weight',{}).items())]]
        return hashlib.sha256(b''.join(np.asarray(a).tobytes() for a in arrays)).hexdigest()
    def rng_digest(self):return hashlib.sha256(repr(self.rng.getstate()).encode()).hexdigest()
    def optimizer_digest(self):return self._state()
    def transaction_snapshot(self):
        return {'weight':self.weight.copy(),'state':copy.deepcopy(self.state),'rng':self.rng.getstate(),
                'parameter_fingerprints':{'weight':self._params()},'state_fp':None,
                'state':{'optimizer':self._state(),'rng':self.rng_digest()},'_full':copy.deepcopy((self.weight,self.state,self.rng.getstate()))}
    def transaction_validate(self,snapshot,updated):
        changed=self._params()!=snapshot['parameter_fingerprints']['weight']
        assert changed==updated,'weights changed without an update (or vice versa)';return {'parameters1942_checked':True,'fixture':True}
    def transaction_restore(self,snapshot):
        weight,state,rng=copy.deepcopy(snapshot['_full']);self.weight=weight;self.state=state;self.rng.setstate(rng);self.arm_updates=0;self.events.append('restore')
    def generate_shared_rollout(self,rows,workdir):
        Fake.rollouts+=1;self.events.append('rollout');captured={'raw':{},'contexts':{}}
        if Fake.log=='interrupt':
            self.rng.random();raise RuntimeError('rollout interrupted')
        for m in ('F','T'):
            responses=[]
            for i in range(4):
                draw=self.rng.random()                       # the rollout consumes RNG; restore must undo it
                tokens=[10+i,20+i,EOS];self.counters.charge('generated_response_tokens',3);self.counters.charge('sampler_draws',4)
                responses.append({'text':f'<think>ACC={Fake.accuracy[m][i]}</think>','token_ids':tokens,'raw_selected_logps':[-1.-.1*i,-2.,-.5-0*draw],'finish_reason':'stop',
                    'EOS_coverage':'OBSERVED_AND_SCORED','generation_tokens':3,'effective_eos_ids_PRIVATE':[EOS]})
            image=Path(rows[m]['images'][0]).read_bytes()
            captured['raw'][m]={'record_id':rows[m]['id'],'responses':responses}
            captured['contexts'][m]={'prompt_text_sha256':sha('generic question'),'image_sha256':hashlib.sha256(image).hexdigest(),'rendered_prompt_token_ids_sha256':sha('ids'),
                                     'input_fingerprints_sha256':sha('fp'+m),'prompt_length':213}
        return captured
    def score_old(self,bundle):
        self.events.append('score_old');out={}
        for item in bundle.raw['responses']:
            self.counters.charge_many({'detached_old_forwards':1,'scorer_vision_embedding_calls':1,'scorer_language_model_calls':1,'teacherforced_input_positions':213+2})
            out[item['response_id']]=[v+.0005 for v in item['raw_selected_logps']]
        return out
    def arm_gradient(self,planned,own_old):
        self.events.append('gradient_'+planned.arm);self.arm_identity=(self._params(),self._state(),self.rng_digest());Fake.identities.append(self.arm_identity)
        def one(item):
            self.counters.charge_many({'gradient_forwards':1,'scorer_vision_embedding_calls':1,'scorer_language_model_calls':1,'teacherforced_input_positions':215,'backward_calls':1})
            idx=core.RESPONSE_IDS.index(item.response.response_id)
            return {'loss':.1,'gradient':np.array([item.advantage*(idx+1),-item.advantage*.5*(idx%3-1)]),'old_current_max_abs':0.}
        total,evidence=core.weighted_gradient_sum(planned.plan,one,lambda t,w:t*w,lambda x,y:x+y)
        evidence['gradient_l2']=float(np.linalg.norm(total)) if total is not None else 0.;return total,evidence
    def apply_update(self,gradient):
        assert self.arm_updates==0,'second optimizer call in one arm';self.arm_updates+=1;self.events.append('update')
        if Fake.fail_update_arm and self.events.count('update')==Fake.fail_update_arm:raise RuntimeError('simulated failure inside optimizer update')
        moments=self.state.setdefault('weight',{'m':np.zeros(2),'v':np.zeros(2)})
        moments['m']=.9*moments['m']+.1*gradient;moments['v']=.999*moments['v']+.001*gradient*gradient;self.state['step']=self.state['step']+1
        self.weight=self.weight-self.state['learning_rate']*moments['m']/(np.sqrt(moments['v'])+1e-8);return {'optimizer_calls':1}
Fake.identities=[]


def grader(row,text,config):
    accuracy=float(text.split('ACC=')[1].split('<')[0]);return grade(accuracy)


def parent_fixture(tmp_path):
    b=Fake(None,{});b.state['step']=np.array(60,dtype=np.uint64);b.state['weight']={'m':np.array([.4,-.3]),'v':np.array([.2,.1])}
    out=tmp_path/'adam60';out.mkdir();pending=out/'.pending';pending.mkdir()
    r=old._publish(out,pending,b,{'experiment':{'adaptive_parent':{'groups':121,'updates':56}}},8,4,'prior',{'updated':True})
    Fake.instances.remove(b);rows=[]
    for i,(label,m) in enumerate((('FLAIR','F'),('contrast-enhanced T1','T'))):
        image=tmp_path/f'{m}.png';image.write_bytes(b'SYNTHETIC IMAGE '+m.encode())
        rows.append({'id':f'row-{m}','ability':'mri_sequence','data_source':'OpenNeuro_ds007045','images':[str(image)],'reward_model':{'ground_truth':label},
                     'extra_info':{'reward_type':'mri_sequence','internal_split':'train','patient_lineage':'BIDS_SUBJECT_VERIFIED','patient_key':'p1','question':'Generic?'},
                     'paired_z045_data_v1':{'version':1,'level':45,'modality_PRIVATE':'flair' if m=='F' else 't1ce','acquisition_modality_source_eligible':True,'training_authorized':False}})
    return r['path'],old.digest(r['path']/'receipt.json'),rows


def launch(tmp_path,out,mode='train',cls=Fake,resume=False,authorize=None,bindings_input=None,**kwargs):
    parent,sha_,rows=parent_fixture(tmp_path) if not (tmp_path/'adam60').exists() else (tmp_path/'adam60'/'boundary_0008',old.digest(tmp_path/'adam60'/'boundary_0008'/'receipt.json'),json.loads((tmp_path/'rows.json').read_text()))
    if not (tmp_path/'rows.json').exists():(tmp_path/'rows.json').write_text(json.dumps(rows))
    return drv.run(output=out,mode=mode,schedule=rows,bindings={'stage':'fixture','source_sha256':{'/s.py':sha('s')},'input_sha256':bindings_input or sha('i')},backend_factory=cls,authorize=authorize or (lambda context:None),
        parent_checkpoint=parent,parent_receipt_sha256=sha_,grader=grader,reward_policy_sha256=sha('calibration'),budget=core.derive_budget(LENGTHS,mode),resume=resume,**kwargs)


@pytest.fixture(autouse=True)
def reset():
    Fake.instances=[];Fake.rollouts=0;Fake.log=None;Fake.accuracy=ACCURACY;Fake.fail_update_arm=None;Fake.identities=[]


def test_train_run_one_rollout_two_arms_one_update_each_from_identical_parent(tmp_path):
    result=launch(tmp_path,tmp_path/'run');b=Fake.instances[-1]
    assert result['status']=='SCHEDULE_COMPLETE' and result['optimizer_updates']==2 and Fake.rollouts==1 and len(Fake.instances)==1
    assert b.events.count('update')==2 and b.events.count('score_old')==1 and b.events.index('rollout')<b.events.index('score_old')<b.events.index('gradient_A')<b.events.index('gradient_B')
    assert Fake.identities[0]==Fake.identities[1]                 # identical full parent weights + Adam + RNG at both arm starts
    parent_anchor_=parent_anchor(tmp_path/'adam60'/'boundary_0008',old.digest(tmp_path/'adam60'/'boundary_0008'/'receipt.json'))
    for arm in ('A','B'):
        terminal=tmp_path/'run'/f'arm_{arm}'/'boundary_0001';receipt=json.loads((terminal/'receipt.json').read_text())
        assert receipt['updated'] and receipt['optimizer_updates']==1 and receipt['absolute_optimizer_updates']==61 and receipt['absolute_completed_groups']==130
        assert receipt['attempted_K4_groups']==2 and receipt['update_evidence']=={'optimizer_calls':1} and receipt['gradient_evidence']['backward_calls']==8
        assert (tmp_path/'run'/f'arm_{arm}_COMMITTED.json').exists()
        anchor=parent_anchor(terminal,old.digest(terminal/'receipt.json'));assert anchor['updates']==61
    weights={arm:np.load(tmp_path/'run'/f'arm_{arm}'/'boundary_0001'/'adapter.safetensors')['weight'] for arm in 'AB'}
    parent_weight=np.load(tmp_path/'adam60'/'boundary_0008'/'adapter.safetensors')['weight']
    assert not np.array_equal(weights['A'],parent_weight) and not np.array_equal(weights['A'],weights['B'])
    np.testing.assert_array_equal(b.weight,parent_weight)          # resident model restored to the parent at the end
    assert json.loads((tmp_path/'run'/'complete_SAFE.json').read_text())['final_state_equals_parent'] is True


def test_full_adam_moments_step_and_rng_persist_and_match_manual_update(tmp_path):
    launch(tmp_path,tmp_path/'run');b=Fake.instances[-1]
    parent_state,parent_rng,_=load_optimizer_checkpoint(tmp_path/'adam60'/'boundary_0008'/'optimizer',expected_bindings=json.loads((tmp_path/'adam60'/'boundary_0008'/'optimizer'/'manifest.json').read_text())['bindings'],load_arrays=b.load_arrays)
    for arm in 'AB':
        terminal=tmp_path/'run'/f'arm_{arm}'/'boundary_0001';manifest=json.loads((terminal/'optimizer/manifest.json').read_text())
        state,rng,_=load_optimizer_checkpoint(terminal/'optimizer',expected_bindings=manifest['bindings'],load_arrays=b.load_arrays)
        assert int(state['step'])==61 and manifest['optimizer_updates']==61
        evidence=json.loads((terminal/'receipt.json').read_text())
        assert rng['python']==parent_rng['python']                     # RNG consumed by the rollout is NOT in either arm: parent RNG restored
        assert not np.array_equal(state['weight']['m'],parent_state['weight']['m']) and np.all(state['weight']['v']>=0)
        assert evidence['optimizer_before_fingerprint']!=evidence['optimizer_after_fingerprint']
        gradient_l2=evidence['gradient_evidence']['gradient_l2']
        assert gradient_l2>0


def test_arm_updates_use_each_arms_own_gradient_with_weight_one_eighth(tmp_path):
    launch(tmp_path,tmp_path/'run');planned={a:core.arm_plan(bundle_from_run(tmp_path/'run'),a) for a in 'AB'}
    for arm in 'AB':
        expected=sum(i.weight*np.array([i.advantage*(core.RESPONSE_IDS.index(i.response.response_id)+1),-i.advantage*.5*(core.RESPONSE_IDS.index(i.response.response_id)%3-1)]) for i in planned[arm].plan.items)
        terminal=tmp_path/'run'/f'arm_{arm}'/'boundary_0001';receipt=json.loads((terminal/'receipt.json').read_text())
        assert receipt['gradient_evidence']['gradient_l2']==pytest.approx(float(np.linalg.norm(expected)),rel=1e-12)
        assert [r['weight'] for r in receipt['gradient_evidence']['responses']]==[1/8]*8


def bundle_from_run(run):return core.load_bundle(json.loads((run/'bundle_PRIVATE.json').read_text()))


def test_qualification_computes_both_arm_gradients_with_zero_updates(tmp_path):
    result=launch(tmp_path,tmp_path/'run',mode='qualify');b=Fake.instances[-1]
    assert result['status']=='SCHEDULE_COMPLETE' and result['optimizer_updates']==0 and 'update' not in b.events and result['counters']['backward_calls']==16
    assert result['counters']['optimizer_updates']==0 and result['counters']['gradient_forwards']==16 and result['counters']['detached_old_forwards']==8
    assert not list((tmp_path/'run').glob('arm_*/boundary_*'))
    for arm in 'AB':assert json.loads((tmp_path/'run'/f'qualification_arm_{arm}_PRIVATE.json').read_text())['optimizer_updates']==0
    with pytest.raises(ValueError,match='Release witness'):                      # no externally written release witness -> no proof
        from mri_rl import shared_rollout_gspo_entry_v1 as e
        (tmp_path/'noev').mkdir();e.qualification_proof(tmp_path/'run',tmp_path/'noev',PREPARED)


def test_inactive_arm_makes_zero_updates_and_records_negative_evidence_without_resample(tmp_path):
    Fake.accuracy={'F':[1,1,1,1],'T':[0,0,0,0]}                       # A: both K4 groups uninformative; B: both mixed groups informative
    result=launch(tmp_path,tmp_path/'run');b=Fake.instances[-1]
    assert result['optimizer_updates']==1 and Fake.rollouts==1 and b.events.count('update')==1
    a=json.loads((tmp_path/'run/arm_A/boundary_0001/receipt.json').read_text())
    assert a['updated'] is False and a['update_decision']['reason']=='NO_ADMITTED_GROUP' and a['optimizer_updates']==0 and a['absolute_optimizer_updates']==60
    assert a['files_sha256']['adapter.safetensors']==json.loads((tmp_path/'adam60/boundary_0008/receipt.json').read_text())['files_sha256']['adapter.safetensors']
    assert json.loads((tmp_path/'run/arm_B/boundary_0001/receipt.json').read_text())['updated'] is True


def test_both_arms_inactive_cancel_with_zero_updates(tmp_path):
    Fake.accuracy={'F':[1,1,1,1],'T':[1,1,1,1]}
    result=launch(tmp_path,tmp_path/'run');b=Fake.instances[-1]
    assert result['optimizer_updates']==0 and 'update' not in b.events and Fake.rollouts==1
    for arm in 'AB':assert json.loads((tmp_path/f'run/arm_{arm}/boundary_0001/receipt.json').read_text())['update_decision']['reason']=='NO_ADMITTED_GROUP'


def test_failure_before_commit_rolls_back_to_parent_and_is_never_auto_retried(tmp_path):
    Fake.fail_update_arm=1
    with pytest.raises(RuntimeError,match='simulated failure'):launch(tmp_path,tmp_path/'run')
    b=Fake.instances[-1];run=tmp_path/'run'
    assert not (run/'arm_A_COMMITTED.json').exists() and not (run/'arm_A/boundary_0001').exists()
    stop=list(run.glob('arm_A_*_STOP_PRIVATE.json'));assert stop and json.loads(stop[0].read_text())['rolled_back_model_Adam_RNG'] is True
    np.testing.assert_array_equal(b.weight,np.load(tmp_path/'adam60/boundary_0008/adapter.safetensors')['weight'])
    assert int(b.state['step'])==60
    count=Fake.rollouts;Fake.fail_update_arm=None
    with pytest.raises(RuntimeError,match='INCOMPLETE_ARM_A'):launch(tmp_path,run,resume=True)
    assert Fake.rollouts==count==1 and all('update' not in x.events for x in Fake.instances[1:])      # no regeneration, no partial retry
    assert (run/'bundle_COMMITTED.json').exists()


def test_failure_after_atomic_commit_is_not_replayed_on_resume(tmp_path,monkeypatch):
    publish=drv.old._publish
    def crash(output,work,backend,binding,completed,updates,*args,**kwargs):
        result=publish(output,work,backend,binding,completed,updates,*args,**kwargs)
        if completed==1 and binding['experiment']['arm']=='A':raise RuntimeError('after atomic rename')
        return result
    monkeypatch.setattr(drv.old,'_publish',crash)
    with pytest.raises(RuntimeError,match='after atomic rename'):launch(tmp_path,tmp_path/'run')
    run=tmp_path/'run';assert (run/'arm_A/boundary_0001/receipt.json').exists()
    stop=json.loads(next(run.glob('arm_A_*_STOP_PRIVATE.json')).read_text());assert stop['committed'] is True and stop['rolled_back_model_Adam_RNG'] is False      # reported committed, not rolled back
    monkeypatch.setattr(drv.old,'_publish',publish)
    result=launch(tmp_path,run,resume=True);b=Fake.instances[-1]
    assert result['status']=='SCHEDULE_COMPLETE' and Fake.rollouts==1 and b.events.count('update')==1 and 'gradient_A' not in b.events and 'gradient_B' in b.events
    assert json.loads((run/'arm_A_COMMITTED.json').read_text())['recovered_without_replay'] is True
    assert result['optimizer_updates']==2


def test_rename_is_the_commit_even_if_marker_write_fails_and_pending_leftovers_are_harmless(tmp_path,monkeypatch):
    real=core.publish_once;armed=[True]
    def flaky(path,value):
        if Path(path).name=='arm_A_COMMITTED.json' and armed[0]:raise OSError('fsync after rename failed')
        return real(path,value)
    monkeypatch.setattr(core,'publish_once',flaky)
    with pytest.raises(OSError,match='fsync after rename'):launch(tmp_path,tmp_path/'run')
    run=tmp_path/'run';assert (run/'arm_A/boundary_0001/receipt.json').exists() and not (run/'arm_A_COMMITTED.json').exists()
    assert json.loads(next(run.glob('arm_A_*_STOP_PRIVATE.json')).read_text())['committed'] is True
    (run/'arm_A/.pending-leftover').mkdir();armed[0]=False
    result=launch(tmp_path,run,resume=True);b=Fake.instances[-1]
    assert result['status']=='SCHEDULE_COMPLETE' and 'gradient_A' not in b.events and Fake.rollouts==1 and (run/'arm_A_COMMITTED.json').exists()


PREPARED={'source_sha256':{'/s.py':sha('s')},'input_sha256':sha('i')}


def qualify(tmp_path,accuracy=ACCURACY,release=True,name='qrun'):
    from types import SimpleNamespace as NS
    from mri_rl import paired_z045_continuation_entry_v1 as inherited
    Fake.accuracy=accuracy;run=tmp_path/name;evidence=tmp_path/(name+'_evidence');evidence.mkdir()
    resident=[NS(close=lambda:None,mx=NS(clear_cache=lambda:None,get_active_memory=lambda:0),model=None)]
    work=lambda:launch(tmp_path,run,mode='qualify')
    result=inherited.run_with_release(work,resident,evidence) if release else work()
    core.publish_once(evidence/'complete_PRIVATE.json',result);return run,evidence


def test_qualification_proof_binds_actual_run_identity_and_external_release_witness(tmp_path):
    from mri_rl import shared_rollout_gspo_entry_v1 as e
    run,evidence=qualify(tmp_path)
    proof=e.qualification_proof(run,evidence,PREPARED)
    assert proof['status']=='ACTUAL_SHARED_ROLLOUT_GSPO_READONLY_QUALIFIED' and proof['blockers']==[] and proof['optimizer_updates']==0
    assert proof['release_witness_sha256']==core.file_sha256(evidence/'release_SAFE.json') and proof['run_identity_sha256']==core.file_sha256(run/'run_identity_PRIVATE.json')
    assert proof['old_sha256']==core.file_sha256(run/'old_PRIVATE.json') and proof['source_sha256']==PREPARED['source_sha256']
    with pytest.raises(ValueError,match='differ from the current reviewed'):e.qualification_proof(run,evidence,{**PREPARED,'input_sha256':sha('other')})   # caller cannot supply arbitrary hashes
    release=evidence/'release_SAFE.json';release.write_text(release.read_text().replace('true','false'))
    with pytest.raises(ValueError,match='Release witness'):e.qualification_proof(run,evidence,PREPARED)


def test_qualification_without_effective_backward_path_is_partial_and_training_is_denied(tmp_path):
    from mri_rl import shared_rollout_gspo_entry_v1 as e
    run,evidence=qualify(tmp_path,{'F':[1,1,1,1],'T':[0,0,0,0]})            # arm A has no admitted group -> no backward
    proof=e.qualification_proof(run,evidence,PREPARED)
    assert proof['status'].startswith('PARTIAL') and proof['blockers']==['ARM_A_NO_EFFECTIVE_FINITE_NONZERO_BACKWARD_PATH']
    scope,human,review,qual=launch_pack('train')
    with pytest.raises(ValueError,match='qualification'):core.authorize_launch('train',scope,human,review,qualification={'proof':proof,'review':qual['review']})
    run2,evidence2=qualify(tmp_path,{'F':[1,1,1,1],'T':[1,1,1,1]},name='q2')
    assert len(e.qualification_proof(run2,evidence2,PREPARED)['blockers'])==2


def test_proof_needs_a_release_witness_the_run_wrote(tmp_path):
    from mri_rl import shared_rollout_gspo_entry_v1 as e
    run,evidence=qualify(tmp_path,release=False)
    with pytest.raises(ValueError,match='Release witness'):e.qualification_proof(run,evidence,PREPARED)


def test_rng_post_generation_state_is_persisted_and_both_arms_restart_from_R0(tmp_path):
    launch(tmp_path,tmp_path/'run');run=tmp_path/'run'
    r1=json.loads((run/'rng_R1_PRIVATE.json').read_text())
    assert r1['R1_differs_from_R0'] is True and r1['R0_rng_digest']!=r1['R1_rng_digest'] and r1['arm_start_policy']=='BOTH_ARMS_RESTORE_R0_NOT_R1'
    assert (run/r1['tensors_file']).is_file() and core.file_sha256(run/r1['tensors_file'])==r1['tensors_sha256'] and r1['layout']['kind']=='dict'
    for arm in 'AB':
        receipt=json.loads((run/f'arm_{arm}/boundary_0001/receipt.json').read_text())
        assert receipt['arm_start_rng_digest']==r1['R0_rng_digest'] and receipt['rng_policy'].startswith('ARM_STARTS_FROM_EXACT_PARENT_R0')


def test_qualify_to_train_adoption_reuses_bundle_without_regeneration_or_recharge(tmp_path):
    from mri_rl import shared_rollout_gspo_entry_v1 as e
    run,evidence=qualify(tmp_path);proof=e.qualification_proof(run,evidence,PREPARED);assert Fake.rollouts==1
    result=launch(tmp_path,tmp_path/'train',adopt={'run':run,'proof':proof})
    assert Fake.rollouts==1 and result['status']=='SCHEDULE_COMPLETE' and result['optimizer_updates']==2 and result['bundle_sha256']==proof['bundle_sha256']
    new=result['counters'];inherited=result['inherited_counters']
    assert new['generated_response_tokens']==0 and new['sampler_draws']==0 and new['detached_old_forwards']==0 and new['backward_calls']==16 and new['optimizer_updates']==2
    assert inherited['generated_response_tokens']==24 and inherited['detached_old_forwards']==8 and inherited['optimizer_updates']==0 and result['counters_scope']=='NEW_COUNTERS_OF_THIS_RUN_ONLY'
    adoption=json.loads((tmp_path/'train/adoption_PRIVATE.json').read_text());assert adoption['generation_in_this_run'] is False and adoption['release_witness_sha256']==proof['release_witness_sha256']
    assert json.loads((tmp_path/'train/counters_PRIVATE.json').read_text())['limits']['generated_response_tokens']==0
    resumed=launch(tmp_path,tmp_path/'train',resume=True);assert resumed['native_backend_constructed'] is False and resumed['inherited_counters']==inherited


def test_adoption_is_refused_for_foreign_partial_or_tampered_evidence_and_never_automatic(tmp_path):
    from mri_rl import shared_rollout_gspo_entry_v1 as e
    run,evidence=qualify(tmp_path);proof=e.qualification_proof(run,evidence,PREPARED)
    with pytest.raises(ValueError):launch(tmp_path,tmp_path/'t1',adopt={'run':run,'proof':{**proof,'status':'PARTIAL_SHARED_ROLLOUT_GSPO_QUALIFICATION','blockers':['x']}})
    with pytest.raises(ValueError,match='differ from the reviewed proof'):launch(tmp_path,tmp_path/'t2',adopt={'run':run,'proof':{**proof,'bundle_sha256':sha('other')}})
    with pytest.raises(ValueError,match='differ from the reviewed proof'):launch(tmp_path,tmp_path/'t3',adopt={'run':run,'proof':{**proof,'old_sha256':sha('other')}})
    with pytest.raises(ValueError,match='source/input/parent'):launch(tmp_path,tmp_path/'t4',adopt={'run':run,'proof':proof},bindings_input=sha('different input'))
    with pytest.raises(ValueError,match='train-mode'):launch(tmp_path,tmp_path/'t5',mode='qualify',adopt={'run':run,'proof':proof})
    assert Fake.rollouts==1 and not any((tmp_path/n/'bundle_COMMITTED.json').exists() for n in ('t1','t2','t3','t4','t5'))
    before=Fake.rollouts;launch(tmp_path,tmp_path/'plain')      # without an explicit adopt argument a train run generates its own bundle
    assert Fake.rollouts==before+1


def test_interrupted_adoption_is_not_retried(tmp_path,monkeypatch):
    from mri_rl import shared_rollout_gspo_entry_v1 as e
    run,evidence=qualify(tmp_path);proof=e.qualification_proof(run,evidence,PREPARED);real=core.publish_once
    def flaky(path,value):
        if Path(path).name=='adoption_PRIVATE.json':raise OSError('crash during adoption')
        return real(path,value)
    monkeypatch.setattr(core,'publish_once',flaky)
    with pytest.raises(OSError):launch(tmp_path,tmp_path/'train',adopt={'run':run,'proof':proof})
    monkeypatch.setattr(core,'publish_once',real)
    with pytest.raises(RuntimeError,match='INCOMPLETE_ADOPTION'):launch(tmp_path,tmp_path/'train',resume=True,adopt={'run':run,'proof':proof})


def test_modality_rows_require_certified_z045_lineage_not_none_equality():
    rows=json.loads(json.dumps([{'id':m,'ability':'mri_sequence','reward_model':{'ground_truth':g},'extra_info':{'patient_key':'p','question':'q'},
        'paired_z045_data_v1':{'version':1,'level':45,'modality_PRIVATE':z,'acquisition_modality_source_eligible':True,'training_authorized':False}}
        for m,g,z in (('F','FLAIR','flair'),('T','contrast-enhanced T1','t1ce'))]))
    assert drv._modality_rows(rows)['T']['id']=='T'
    for edit in (lambda r:r[0].pop('paired_z045_data_v1'),lambda r:r[1]['paired_z045_data_v1'].__setitem__('level',91),lambda r:r[1]['paired_z045_data_v1'].__setitem__('modality_PRIVATE','flair'),
                 lambda r:r[0]['paired_z045_data_v1'].__setitem__('acquisition_modality_source_eligible',False),lambda r:r[1]['extra_info'].__setitem__('patient_key','other'),
                 lambda r:[x['extra_info'].pop('patient_key') for x in r],lambda r:[x['extra_info'].pop('question') for x in r],lambda r:r.reverse()):
        bad=copy.deepcopy(rows);edit(bad)
        with pytest.raises((ValueError,KeyError)):drv._modality_rows(bad)


def test_consumer_entry_revalidates_ids_and_external_anchor_rows_and_images():
    raw=frozen();anchor={'receipt_sha256':PARENT['receipt_sha256'],'adapter_sha256':PARENT['adapter_sha256']}
    rows={m:{'id':'row-'+m} for m in 'FT'};images={'row-F':sha('imageF'),'row-T':sha('imageT')}
    good=dict(anchor=anchor,rows=rows,expected_image_sha256=images,pair_id='pair',reward_policy_sha256=sha('calibration'))
    assert core.validate_bundle_against_run(core.load_bundle(raw),**good)
    for key,value in (('anchor',{**anchor,'receipt_sha256':sha('foreign')}),('anchor',{**anchor,'adapter_sha256':sha('foreign')}),('rows',{'F':{'id':'row-T'},'T':{'id':'row-F'}}),
                      ('expected_image_sha256',{**images,'row-F':sha('swapped')}),('pair_id','other'),('reward_policy_sha256',sha('other calibration'))):
        with pytest.raises(ValueError,match='external anchor|own row'):core.validate_bundle_against_run(core.load_bundle(raw),**{**good,key:value})
    for edit in (lambda b:b['responses'][1].__setitem__('index',2),lambda b:b['responses'][5].__setitem__('modality','F'),lambda b:b['responses'][0].__setitem__('response_id','T0')):
        tampered=copy.deepcopy(raw);edit(tampered);body={k:v for k,v in tampered.items() if k!='bundle_sha256'};tampered['bundle_sha256']=core.identity(body)
        with pytest.raises(ValueError):core.load_bundle(tampered)


def test_interrupted_rollout_is_preserved_and_never_regenerated(tmp_path):
    Fake.log='interrupt'
    with pytest.raises(RuntimeError,match='rollout interrupted'):launch(tmp_path,tmp_path/'run')
    run=tmp_path/'run';assert (run/'rollout_reservation_PRIVATE.json').exists() and not (run/'bundle_COMMITTED.json').exists()
    count=Fake.rollouts;Fake.log=None
    with pytest.raises(RuntimeError,match='INCOMPLETE_ROLLOUT'):launch(tmp_path,run,resume=True)
    assert Fake.rollouts==count and len(Fake.instances)==1             # the resumed process never even constructs a backend


def test_completed_run_resumes_without_backend_and_existing_output_needs_explicit_resume(tmp_path):
    launch(tmp_path,tmp_path/'run');n=len(Fake.instances)
    result=launch(tmp_path,tmp_path/'run',resume=True)
    assert result['status']=='SCHEDULE_COMPLETE' and len(Fake.instances)==n and result['native_backend_constructed'] is False
    with pytest.raises(FileExistsError):launch(tmp_path,tmp_path/'run')


def test_authorization_precedes_backend_construction(tmp_path):
    def deny(context):raise PermissionError('resource/admission gate denied')
    with pytest.raises(PermissionError):launch(tmp_path,tmp_path/'run',authorize=deny)
    assert Fake.instances==[]


def test_counter_limit_stops_a_run_and_spent_counters_survive_resume(tmp_path):
    class Greedy(Fake):
        def score_old(self,bundle):
            for _ in range(9):self.counters.charge('detached_old_forwards')
            return super().score_old(bundle)
    with pytest.raises(ValueError,match='budget exhausted'):launch(tmp_path,tmp_path/'run',cls=Greedy)
    saved=json.loads((tmp_path/'run/counters_PRIVATE.json').read_text())
    assert saved['used']['detached_old_forwards']==8 and saved['used']['generated_response_tokens']==24
    with pytest.raises(ValueError,match='budget exhausted'):launch(tmp_path,tmp_path/'run',resume=True)   # spent budget is not refunded by a restart
    assert Fake.rollouts==1
