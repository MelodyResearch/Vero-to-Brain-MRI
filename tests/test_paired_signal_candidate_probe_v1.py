"""CPU preparation/call/state gates; no native model imports or generation."""
import copy
import pytest
from mri_rl import paired_signal_candidate_probe_v1 as p


def fixed():
    def row(id,truth,image,task='mri_sequence'):
        return {'id':id,'source_record_id':id,'ability':task,'images':[image],'extra_info':{'patient_key':'person','question':'exact question'},'reward_model':{'ground_truth':truth},'rc_data_v1':{'image_sha256':id*64},'paired_z045_data_v1':{'level':45}}
    seq=row('a','contrast-enhanced T1','/a.png');seq['id']='historical-seq';seq['source_record_id']='a'
    other=row('b','FLAIR','/b.png');Dallas=row('d','same','/d.png','mri_visit_correspondence')
    scope={'rows_PRIVATE':[seq,Dallas],'expected_image_sha256':{'historical-seq':'a'*64,'d':'d'*64}}
    return scope,[copy.deepcopy(seq)|{'id':'a'},other]


def test_fixed_four_contexts_include_genuine_null_and_sameperson_opposite():
    scope,pool=fixed();contexts=p.fixed_contexts(scope,pool)
    assert [c['name'] for c in contexts]==['SEQUENCE_REAL','SEQUENCE_OPPOSITE_SAME_PERSON_Z045','SEQUENCE_NO_IMAGE_NULL','DALLAS_REAL']
    assert contexts[1]['row_PRIVATE']['reward_model']['ground_truth']=='FLAIR'
    assert contexts[2]['row_PRIVATE']['images']==[] and contexts[2]['image_sha256'] is None
    assert contexts[2]['source_truth'] is None
    assert all(c['row_PRIVATE']['extra_info']['question']=='exact question' for c in contexts[:3])
    pool[1]['extra_info']['patient_key']='different person'
    with pytest.raises(ValueError):p.fixed_contexts(scope,pool)


def test_null_is_not_zero_pixels_or_image_grid():
    result=p.normalize_inputs({'input_ids':'ids','attention_mask':'mask'},no_image=True)
    assert result['pixel_values'] is None and 'image_grid_thw' not in result
    with pytest.raises(ValueError):p.normalize_inputs({'input_ids':'i','pixel_values':0},no_image=True)
    with pytest.raises(ValueError):p.normalize_inputs({'input_ids':'i','image_grid_thw':[1,2,3]},no_image=True)


def test_exact_twelve_forward_and_5100_position_caps_no_other_operations():
    assert p.budget()['full_TF_forwards']==12 and p.budget()['teacherforced_positions']==5100
    assert p.budget()['generation_calls']==p.budget()['backward_calls']==p.budget()['optimizer_updates']==0
    c=p.Counter()
    for _ in range(12):c.charge(425)
    assert c.values=={'full_TF_forwards':12,'teacherforced_positions':5100}
    with pytest.raises(ValueError):c.charge(1)


def test_no_image_shape_bound_and_image_cost_are_failclosed():
    assert p.prompt_guard('mri_sequence',213,False)==213
    assert p.prompt_guard('mri_visit_correspondence',405,False)==405
    assert p.prompt_guard('mri_sequence',59,True)==59
    for task,length,null in [('mri_sequence',214,False),('mri_visit_correspondence',406,False),('mri_sequence',213,True)]:
        with pytest.raises(ValueError):p.prompt_guard(task,length,null)


def test_direction_free_canonical_margins_and_null_do_not_invent_correctness():
    values=[{'mean':-1.,'sum':-2.,'first':-.5},{'mean':-2.,'sum':-10.,'first':-1.}]
    r=p.margins(values,['FLAIR','contrast-enhanced T1'],'contrast-enhanced T1')
    assert r['canonical_label0_minus_label1']['mean']==1 and r['source_truth_minus_alternative']['mean']==-1
    assert p.margins(values,['FLAIR','contrast-enhanced T1'],None)['source_truth_minus_alternative'] is None


def test_new_human_permission_and_exact_review_required_no_old_one_shot_reuse():
    scope={'training_authorized':False,'status':'CPU_PREPARED_NOT_APPROVED','x':'fixed'}
    with pytest.raises(ValueError):p.authorize(scope,{'authorized':True,'action':'old_qualification'},{'verdict':'PASS'})
    sha=p.canonical(scope);h={'authorized':True,'action':'readonly_paired_signal_candidate_probe','scope_sha256':sha}
    r={'verdict':'PASS_WITH_LIMITATIONS','scope_sha256':sha,'reviewer_id':'isolated','reviewer_kind':'isolated_codex'}
    assert p.authorize(scope,h,r)
    with pytest.raises(ValueError):p.authorize({**scope,'x':'changed'},h,r)


def test_weight_Adam_RNG_mismatch_stops_before_restore():
    assert p.state_unchanged({'all':'same','Adam':60,'rng':'x'},{'all':'same','Adam':60,'rng':'x'})
    with pytest.raises(ValueError):p.state_unchanged({'Adam':60,'rng':'x'},{'Adam':60,'rng':'changed'})


def test_actual_probe_worker_calls_only_three_full_scores_per_context_and_fails_numeric_gate():
    import numpy as np
    from types import SimpleNamespace as NS
    raw=np.array([-5.,-1.,-2.,-3.,-4.,-5.,-6.,-7.,-8.,-9.],dtype=np.float32)
    def logsoftmax(value,axis=-1):
        maximum=np.max(value,axis=axis,keepdims=True);return value-maximum-np.log(np.exp(value-maximum).sum(axis=axis,keepdims=True))
    mx=NS(eval=lambda *args:None,clear_cache=lambda:None,float32=np.float32,all=np.all,isfinite=np.isfinite)
    backend=NS(mx=mx,nn=NS(log_softmax=logsoftmax))
    context={'name':'NULL','no_image':True,'image_sha256':None,'source_truth':None,'row_PRIVATE':{'images':[],'ability':'mri_sequence'}}
    inputs={'input_ids':np.ones((1,59),dtype=np.int64),'pixel_values':None}
    plans={'prefix':[9,8],'labels':[[1,3],[2,4,5,6,7]],'label_order':['FLAIR','contrast-enhanced T1']}
    called=[]
    def scorer(b,tokens,actual,*,evidence,vocab_observer):
        called.append(list(tokens));evidence.append({'logits_dtype':'float32'})
        logits=np.tile(raw,(1,len(tokens),1))
        if vocab_observer:vocab_observer(logits)
        return logsoftmax(logits)[0,np.arange(len(tokens)),tokens]
    counter=p.Counter()
    for _ in range(4):
        result=p.score_context(backend,context,inputs,plans,counter,lambda name:None,scorer=scorer)
        assert result['prefix_oracle']['max_abs_error']==0 and result['margins']['source_truth_minus_alternative'] is None
    assert len(called)==counter.values['full_TF_forwards']==12 and counter.values['teacherforced_positions']<=5100
    calls=[]
    def bad(b,tokens,actual,**kwargs):
        value=scorer(b,tokens,actual,**kwargs);calls.append(True)
        if len(calls)==2:value[0]+=.004
        return value
    with pytest.raises(ValueError,match='prefix'):
        p.score_context(backend,context,inputs,plans,p.Counter(),lambda name:None,scorer=bad)
    assert len(calls)==2


def test_same_length_changed_runtime_contract_is_rejected_by_actual_prepare(tmp_path,monkeypatch):
    import json,sys
    from types import SimpleNamespace
    contract=tmp_path/'contract.json';contract.write_text('{"system_instruction":"fixed-A"}')
    original=p.entry.digest(contract);original_size=contract.stat().st_size
    runtime=tmp_path/'runtime.json';runtime.write_text(json.dumps({'contract':str(contract),'file_sha256':{'contract':original}}))
    prior=tmp_path/'prior.json';prior.write_text('{}')
    spec={'prior_scope':str(prior),'runtime_spec':str(runtime),'file_sha256':{'prior_scope':p.entry.digest(prior),'runtime_spec':p.entry.digest(runtime)}}
    monkeypatch.setattr(p,'PRIOR_SCOPE_SHA',p.entry.digest(prior))
    # This test must fail at the contract gate before tokenizer/model preparation.
    monkeypatch.setitem(sys.modules,'tokenizers',SimpleNamespace(Tokenizer=object))
    contract.write_text('{"system_instruction":"fixed-B"}');assert contract.stat().st_size==original_size
    with pytest.raises(ValueError,match='runtime.*contract'):p.prepare(spec)


def test_all_runtime_file_hashes_are_carried_to_artifact_map(tmp_path):
    contract=tmp_path/'contract.json';contract.write_text('fixed');data=tmp_path/'data.json';data.write_text('bound')
    runtime={'contract':str(contract),'data':str(data),'file_sha256':{'contract':p.entry.digest(contract),'data':p.entry.digest(data)}}
    assert p.runtime_artifact_bindings(runtime)=={str(contract):p.entry.digest(contract),str(data):p.entry.digest(data)}
    data.write_text('other')
    with pytest.raises(ValueError):p.runtime_artifact_bindings(runtime)


@pytest.mark.parametrize('power,battery,expected',[
    ('BATTERY',None,'STOP'),('BATTERY',59.9,'STOP'),
    ('BATTERY',60,'CONTINUE'),('AC',96,'CONTINUE'),
    ('AC',59,'STOP'),('AC',None,'STOP'),
    ('SENSOR_UNAVAILABLE',96,'STOP'),('BATTERY',float('nan'),'STOP'),
])
def test_local_battery_gate_threshold_and_truthful_durable_sample(tmp_path,power,battery,expected):
    import json
    from mri_rl.resource_supervisor import ResourceSample
    sample=ResourceSample(monotonic_seconds=1,available_gib=12,pressure_free_percent=70,
                          thermal_state='NOMINAL',power_source=power,battery_percent=battery,other_heavy_processes=0)
    path=tmp_path/'resources.jsonl';gate=p.ProbeBatteryPacedGate(path,sampler=lambda:sample,sleep=lambda seconds:None)
    if expected=='STOP':
        with pytest.raises(RuntimeError,match='stopped'):gate.before_unit('candidate',loading=True)
    else:assert gate.before_unit('candidate',loading=True) is sample
    event=json.loads(path.read_text());assert event['action']==expected
    assert event['sample']['power_source']==power
    assert event['power_exception']['minimum_battery_percent']==60
    assert gate.policy._last is sample


def test_local_battery_gate_keeps_memory_and_other_resource_protections(tmp_path):
    import json
    from dataclasses import replace
    from mri_rl.resource_supervisor import ResourceSample
    sample=ResourceSample(monotonic_seconds=1,available_gib=1,pressure_free_percent=1,
                          thermal_state='NOMINAL',power_source='BATTERY',battery_percent=90,other_heavy_processes=0)
    path=tmp_path/'critical.jsonl';gate=p.ProbeBatteryPacedGate(path,sampler=lambda:sample,sleep=lambda seconds:None)
    with pytest.raises(RuntimeError,match='severe_resource_pressure'):gate.before_unit('candidate')
    assert json.loads(path.read_text())['sample']['power_source']=='BATTERY'
    busy=replace(sample,available_gib=12,pressure_free_percent=70,other_heavy_processes=1)
    action,reasons=p.probe_power_decisions(busy,loading=True)
    assert action=='HOLD' and 'concurrent_process_inventory_unknown_or_occupied' in reasons
    assert p.budget()['maximum_seconds']==600
