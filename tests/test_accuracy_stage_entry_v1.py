"""CPU admission failures and exact scope binding; no native execution or MRI."""
import copy,importlib,json,sys
from pathlib import Path
from types import SimpleNamespace
import pytest
from mri_rl import accuracy_recovery_v1 as ar
from mri_rl import exposure_rounds_v1 as ex, multistep_gspo_core_v1 as ms
from test_exposure_rounds_v1 import pool56,dallas48


def entry():
    # A missing implementation is an explicit failing behavior, not a skipped test.
    assert importlib.util.find_spec('mri_rl.accuracy_stage_entry_v1'), 'CPU stage entry not implemented'
    return importlib.import_module('mri_rl.accuracy_stage_entry_v1')


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path


def fixture(tmp_path,monkeypatch):
    e=entry();root=tmp_path/'repo';(root/'.git').mkdir(parents=True);d=root/'local_data/accuracy_stage_v1';d.mkdir(parents=True)
    cp=root/'parent/boundary_0114';cp.mkdir(parents=True);receipt=put(cp/'receipt.json',{'schema':'continuous-rl-boundary-v1','completed_groups':114,'optimizer_updates':52})
    frozen={**ar.FROZEN_PARENT,'receipt_sha256':ar.file_sha256(receipt)};monkeypatch.setattr(e,'FROZEN_PARENT',frozen)
    seq,sites=pool56();dallas=dallas48()
    for row in seq+dallas:row['prompt']=[{'role':'user','content':[{'type':'image','image':row['images'][0]},{'type':'text','text':'q'}]}]
    plan=ex.build_round_plan(seq,dallas,sites,rounds=4);units=ms.make_units(plan)
    capsules={}
    for name,rows in [('DATA56_INPUT_PRIVATE.json',seq),('DALLAS48_INPUT_v2_PRIVATE.json',dallas)]:
        wrapped=[]
        for row in rows:
            item={'record_PRIVATE':row,'original_source_row_sha256':ar.identity(row),'current_image_file_bindings_PRIVATE':[{'path':row['images'][0],'sha256':'f'*64,'bytes':12}]}
            if row['ability']=='mri_sequence':item.update(patient_key_PRIVATE=row['extra_info']['patient_key'],site_PRIVATE=sites[row['extra_info']['patient_key']],image_path=row['images'][0],expected_png_sha256='f'*64,PNGbytes=12)
            wrapped.append(item)
        p=put(d/name,{'rows':wrapped});capsules[name]={'path':str(p),'sha256':ar.file_sha256(p)}
    devrows=[]
    for row in seq[:8]:
        for level in (45,91,136):
            r=copy.deepcopy(row);r['id']+=str(level);r['extra_info']['internal_split']='internal_dev'
            devrows.append({'record_PRIVATE':r,'original_source_row_sha256':ar.identity(r),'current_image_file_bindings_PRIVATE':[{'path':r['images'][0],'sha256':'f'*64,'bytes':12}],'level_for_fixture':level})
    for row in dallas[:16]:
        r=copy.deepcopy(row);r['extra_info']['internal_split']='internal_dev';devrows.append({'record_PRIVATE':r,'original_source_row_sha256':ar.identity(r),'current_image_file_bindings_PRIVATE':[{'path':r['images'][0],'sha256':'f'*64,'bytes':12}]})
    devp=put(d/'DEV40_INPUT_PRIVATE.json',{'rows':devrows});capsules[devp.name]={'path':str(devp),'sha256':ar.file_sha256(devp)}
    from mri_rl.answer_likelihood import PREFIX
    from mri_rl.paired_z045_evaluation_v1 import LABELS
    real=[]
    for i,item in enumerate(devrows):
        r=copy.deepcopy(item['record_PRIVATE'])
        if r['ability']=='mri_sequence':r['extra_info']['level']=item['level_for_fixture']
        real.append({'id':'real'+str(i),'condition':'real_image','row_PRIVATE':r,'image_sha256':'f'*64,'original_source_row_sha256':item['original_source_row_sha256']})
    z=[r for r in real if r['row_PRIVATE']['extra_info'].get('level')==45];aliases=[]
    for i,r in enumerate(z):
        other=next(q for q in z if q['row_PRIVATE']['extra_info']['patient_key']==r['row_PRIVATE']['extra_info']['patient_key'] and q['row_PRIVATE']['reward_model']['ground_truth']!=r['row_PRIVATE']['reward_model']['ground_truth'])
        alias=copy.deepcopy(other['row_PRIVATE']);alias['id']='aliasrow'+str(i)
        aliases.append({'id':'alias'+str(i),'condition':'same_patient_same_level_opposite_modality','row_PRIVATE':alias,'image_sha256':'f'*64,'original_real_id_PRIVATE':r['id'],'reuse_real_id_PRIVATE':other['id']})
    evaldata=put(d/'EVAL40.json',{'records_PRIVATE':real+aliases,'candidate_prefix':PREFIX,'label_order':LABELS,'token_plans_PRIVATE':{task:{'prefix':[1],'labels':[[2],[3]],'label_order':list(labels)} for task,labels in LABELS.items()}})
    stage=put(d/'stage.json',{'schema':'accuracy-stage-four-round-input-v1','parent52_metadata_PRIVATE':{'Adam_step':52,'groups':114,'receipt_sha256':frozen['receipt_sha256'],'adapter_sha256_from_existing_metadata':frozen['adapter_sha256'],'checkpoint_path':str(cp)},'round_plan_PRIVATE':plan,'units_PRIVATE':units,'prompt_lengths':{'mri_sequence':213,'mri_visit_correspondence':405},'planned_image_bindings_PRIVATE':[{'record_id_PRIVATE':r['id'],'image_binding_PRIVATE':{'path':r['images'][0],'sha256':'f'*64,'bytes':12}} for u in units for c in u['contexts'] for r in [c['row']]],'input_capsules_PRIVATE':capsules})
    model=root/'model';model.mkdir();cache=root/'cache';cache.mkdir();cal=put(root/'cal.json',{'fixed':True});contract=put(root/'resource.json',{'fixed':True})
    bridge=root/'accuracy_stage_native_v1.py';bridge.write_text('def run_stage(spec,scope,authorization,resource_gate):\n return None\ndef make_resource_gate(spec,scope,authorization):\n return None\n')
    evalsrc=root/'paired_z045_matched_f32_eval_v1.py';evalsrc.write_text('def run_checkpoint():\n pass\n')
    runtime=put(root/'runtime.json',{'parent_checkpoint':str(cp),'parent_receipt_sha256':frozen['receipt_sha256'],'calibration':str(cal),'model':str(model),'cache_dir':str(cache),'contract':str(contract),'global_lock':str(root/'one_model.lock'),'runtime_source_sha256':{str(bridge):ar.file_sha256(bridge)},'file_sha256':{'calibration':ar.file_sha256(cal),'contract':ar.file_sha256(contract)}})
    native=put(root/'native.json',{'schema':'accuracy-stage-native-source-contract-v1','status':'SOURCE_READY_NOT_NATIVE_VALIDATED','module':'mri_rl.accuracy_stage_native_v1','entrypoint':'run_stage','resource_gate_entrypoint':'make_resource_gate','wired_hooks':['cache','train','evaluate','resource_gate'],'source_sha256':{str(bridge):ar.file_sha256(bridge)}})
    cachecontract=put(root/'cache_scope.json',{'scope_kind':'FRESH_EXACT_HISTORICAL_GPU_CACHE_V4_CONTINUATION_SCOPE','schema':'accuracy-recovery-historical-gpu-cache-v4-scope-v1','expected':{'final_sha256':'a'*64},'limits':{'maximum_seconds':1800},'source_sha256':{str(bridge):ar.file_sha256(bridge)},'input_sha256':'b'*64})
    evaluation=put(root/'eval.json',{'status':'SOURCE_READY_NOT_NATIVE_VALIDATED','entrypoint':'run_checkpoint','real_rows':40,'swap_aliases':8,'source_sha256':{str(evalsrc):ar.file_sha256(evalsrc)}})
    spec={'schema':'accuracy-stage-spec-v1','stage_input':str(stage),'runtime_spec':str(runtime),'native_contract':str(native),'cache_contract':str(cachecontract),'evaluation_contract':str(evaluation),'evaluation':str(evaldata),'output':str(d/'run_PRIVATE')}
    spec['file_sha256']={k:ar.file_sha256(Path(spec[k])) for k in ['stage_input','runtime_spec','native_contract','cache_contract','evaluation_contract','evaluation']}
    sp=put(d/'spec.json',spec);return e,root,sp,spec


def test_pending_inputs_never_yield_executable_scope(tmp_path):
    e=entry();sp=put(tmp_path/'missing.json',{'schema':'accuracy-stage-spec-v1'})
    result=e.prepare_stage_spec(sp)
    assert result['status']=='BLOCKED' and result['blockers'] and 'executable_scope' not in result
    assert not any(n=='mlx' or n.startswith('mlx.') for n in sys.modules)


def test_ready_scope_derives_four_round_budget_and_pins_conditional_cache(tmp_path,monkeypatch):
    e,root,sp,spec=fixture(tmp_path,monkeypatch);p=e.prepare_stage_spec(sp,start=root)
    assert not p['blockers'];scope=p['executable_scope']
    assert scope['budget']['per_arm']['optimizer_updates']==8
    assert scope['budget']['per_arm']['limits']['teacherforced_input_positions']==68448
    assert scope['budget']['evaluation']['maximum_generated_tokens']==30720
    assert scope['cache_prerequisite']['status']=='CONDITIONAL_EXECUTE_FIRST'
    assert scope['resource_policy']['battery_percent_min']==20 and scope['resource_policy']['this_round_only']
    assert scope['input_sha256'] and scope['source_sha256'] and p['model_loads']==0
    assert 'authorized' not in scope and 'training_authorized' not in scope


@pytest.mark.parametrize('change',['input','source','runtime','budget','hook'])
def test_changed_or_unwired_binding_is_refused_before_admission(tmp_path,monkeypatch,change):
    e,root,sp,spec=fixture(tmp_path,monkeypatch)
    if change=='input':Path(spec['stage_input']).write_text('{}')
    elif change=='source':(root/'accuracy_stage_native_v1.py').write_text('changed')
    elif change=='runtime':Path(spec['runtime_spec']).write_text('{}')
    elif change=='budget':
        x=json.loads(Path(spec['stage_input']).read_text());x['units_PRIVATE']=x['units_PRIVATE'][:2];put(Path(spec['stage_input']),x);spec['file_sha256']['stage_input']=ar.file_sha256(Path(spec['stage_input']));put(sp,spec)
    else:
        x=json.loads(Path(spec['native_contract']).read_text());x['wired_hooks'].remove('evaluate');put(Path(spec['native_contract']),x);spec['file_sha256']['native_contract']=ar.file_sha256(Path(spec['native_contract']));put(sp,spec)
    try:p=e.prepare_stage_spec(sp,start=root)
    except ValueError:return
    assert p['blockers'] and 'executable_scope' not in p


def sign(e,scope):
    from datetime import datetime,timedelta,timezone
    now=datetime.now(timezone.utc)
    digest=e.scope_digest(scope)
    return ({'authorized':True,'action':e.ACTION,'scope_sha256':digest,'human_instruction_sha256':'7'*64,'approved_by':'researcher','included_actions':['cache','training','evaluation'],'cache_scope_sha256':scope['cache_prerequisite']['scope_sha256'],'issued_at_utc':(now-timedelta(seconds=10)).isoformat(),'expires_at_utc':(now+timedelta(hours=4)).isoformat()}, {'verdict':'PASS','reviewer_kind':'isolated_codex','reviewer_id':'independent-agent','scope_sha256':digest,'source_sha256':scope['source_sha256'],'input_sha256':scope['input_sha256']})


def test_admission_requires_exact_separate_review_human_and_consumes_once(tmp_path,monkeypatch):
    e,root,sp,spec=fixture(tmp_path,monkeypatch);scope=e.prepare_stage_spec(sp,start=root)['executable_scope'];h,r=sign(e,scope)
    with pytest.raises(ValueError):e.authorize_stage(scope,{**h,'action':'old_action'},r,consumed=ar.consume_once(root/'ledger'))
    with pytest.raises(ValueError):e.authorize_stage(scope,h,{**r,'reviewer_id':e.IMPLEMENTED_BY},consumed=ar.consume_once(root/'ledger'))
    token=e.authorize_stage(scope,h,r,consumed=ar.consume_once(root/'ledger'));assert token.check('test')
    with pytest.raises(ValueError,match='consumed'):e.authorize_stage(scope,h,r,consumed=ar.consume_once(root/'ledger'))
    (root/'accuracy_stage_native_v1.py').write_text('changed')
    with pytest.raises(ValueError):token.check('changed source')


def test_execute_requires_pinned_artifact_hashes_before_native_loader(tmp_path,monkeypatch):
    e,root,sp,spec=fixture(tmp_path,monkeypatch)
    # Missing authority artifacts must fail before native import or runtime dispatch.
    with pytest.raises(ValueError,match='NOT_AUTHORIZED'):
        e.execute_stage(SimpleNamespace(spec=sp),start=root)


def test_cli_prepare_prints_only_aggregate_and_does_not_write_scope_when_blocked(tmp_path,capsys):
    e=entry();sp=put(tmp_path/'spec.json',{'schema':'accuracy-stage-spec-v1'});out=tmp_path/'scope.json'
    result=e.main(['--spec',str(sp),'--scope-output',str(out)])
    printed=json.loads(capsys.readouterr().out);assert printed['status']=='BLOCKED' and not out.exists()
    assert str(tmp_path) not in json.dumps(printed) and 'executable_scope' not in printed


def test_scope_requires_frozen_eval_records_and_capsule_membership(tmp_path,monkeypatch):
    e,root,sp,spec=fixture(tmp_path,monkeypatch)
    spec.pop('evaluation',None);spec['file_sha256'].pop('evaluation',None);put(sp,spec)
    p=e.prepare_stage_spec(sp,start=root)
    assert p['blockers'] and 'executable_scope' not in p


def test_scope_cannot_self_attest_an_arbitrary_same_shape_training_plan(tmp_path,monkeypatch):
    e,root,sp,spec=fixture(tmp_path,monkeypatch);stage=json.loads(Path(spec['stage_input']).read_text())
    # Alter the actual question but preserve the sizes/caps. Membership must catch it.
    stage['round_plan_PRIVATE']['rounds'][0]['blocks'][0]['rows'][0]['prompt'][0]['content'][1]['text']='foreign question'
    stage['units_PRIVATE']=ms.make_units(stage['round_plan_PRIVATE']);put(Path(spec['stage_input']),stage)
    spec['file_sha256']['stage_input']=ar.file_sha256(Path(spec['stage_input']));put(sp,spec)
    with pytest.raises(ValueError):e.prepare_stage_spec(sp,start=root)


def test_static_local_dependency_omission_is_included_in_exact_closure(tmp_path,monkeypatch):
    e,root,sp,spec=fixture(tmp_path,monkeypatch);p=e.prepare_stage_spec(sp,start=root)
    paths=p['executable_scope']['source_sha256']
    assert any(path.endswith('/mri_rl/optimizer_checkpoint.py') for path in paths)
    assert any(path.endswith('/mri_rl/paired_prompt_groups_v1.py') for path in paths)


@pytest.mark.parametrize('case',['expired','future','too_long','old_start'])
def test_human_time_window_is_fresh_bounded_and_precedes_consumption(tmp_path,monkeypatch,case):
    from datetime import datetime,timedelta,timezone
    e,root,sp,spec=fixture(tmp_path,monkeypatch);scope=e.prepare_stage_spec(sp,start=root)['executable_scope'];h,r=sign(e,scope)
    now=datetime.now(timezone.utc)
    issued=now-timedelta(seconds=10);expires=now+timedelta(hours=1)
    if case=='expired':expires=now-timedelta(seconds=1)
    elif case=='future':issued=now+timedelta(seconds=1)
    elif case=='too_long':expires=issued+timedelta(hours=25)
    else:issued=now-timedelta(seconds=1801)
    h.update(issued_at_utc=issued.isoformat(),expires_at_utc=expires.isoformat())
    with pytest.raises(ValueError,match='authorization time'):e.authorize_stage(scope,h,r,consumed=ar.consume_once(root/'ledger'))
    assert not (root/'ledger').exists()


def test_token_stat_cache_rehashes_on_change_and_at_explicit_boundaries(tmp_path,monkeypatch):
    e,root,sp,spec=fixture(tmp_path,monkeypatch);scope=e.prepare_stage_spec(sp,start=root)['executable_scope'];h,r=sign(e,scope)
    token=e.authorize_stage(scope,h,r,consumed=ar.consume_once(root/'ledger'));original=e._metadata_hash;reads=[]
    def measured(path):reads.append(str(path));return original(path)
    monkeypatch.setattr(e,'_metadata_hash',measured)
    token.check('f32_generation_language_model');assert reads==[]
    token.check('native_model_load');assert len(reads)>=len(scope['source_sha256'])
    reads.clear();p=root/'accuracy_stage_native_v1.py';p.write_text(p.read_text()+'# changed\n')
    with pytest.raises(ValueError):token.check('f32_generation_language_model')
    assert str(p) in reads


def test_ready_cli_scope_file_stays_inside_private_canonical_root(tmp_path,monkeypatch):
    e,root,sp,spec=fixture(tmp_path,monkeypatch)
    # Output guard must also apply to the scope artifact, not just model outputs.
    monkeypatch.setattr(e.authority,'canonical_project_root',lambda start=None:root)
    with pytest.raises(ValueError,match='canonical accuracy_stage'):
        e.main(['--spec',str(sp),'--scope-output',str(root/'public_scope.json')])
    assert not (root/'public_scope.json').exists()
