import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

path=Path(__file__).resolve().parents[1]/'src/mri_rl/paced_frozen_eval.py'
s=importlib.util.spec_from_file_location('paced_eval',path);m=importlib.util.module_from_spec(s)
if path.exists():s.loader.exec_module(m)


def test_partial_b_cannot_enter_matched28_evaluation():
    assert hasattr(m,'validate_completion_claim')
    with pytest.raises(ValueError):m.validate_completion_claim({'optimizer_steps':27,'completed_control_28steps':False})
    with pytest.raises(ValueError):m.validate_completion_claim({'optimizer_steps':28,'completed_control_28steps':True,'status':'SFT_RELOAD_PENDING'})


def test_generation_metadata_uses_actual_backend_finish_reason_not_token_guess():
    assert hasattr(m,'generation_metadata')
    r=SimpleNamespace(generation_tokens=256,finish_reason='stop',peak_memory=11.2,prompt_tokens=300,total_tokens=556)
    d=m.generation_metadata(r)
    assert d['generation_tokens']==256 and d['finish_reason']=='stop' and d['truncated'] is False
    r.finish_reason='length';assert m.generation_metadata(r)['truncated'] is True
    r.finish_reason=None;assert m.generation_metadata(r)['truncated'] is None


def test_generation_metadata_rejects_missing_or_nonfinite_measurements():
    assert hasattr(m,'generation_metadata')
    with pytest.raises(ValueError):m.generation_metadata(SimpleNamespace())
    with pytest.raises(ValueError):m.generation_metadata(SimpleNamespace(generation_tokens=2,finish_reason='stop',peak_memory=float('nan'),prompt_tokens=3,total_tokens=5))


def test_bounded_pending_plan_keeps_frozen_full_manifest_order_and_never_repeats():
    assert hasattr(m,'bounded_pending')
    rows=[{'record_id':str(i)} for i in range(54)]
    assert [x['record_id'] for x in m.bounded_pending(rows,{'0','2'},4)]==['1','3','4','5']
    with pytest.raises(ValueError):m.bounded_pending(rows,set(),0)
    with pytest.raises(ValueError):m.bounded_pending(rows,set(),5)
    with pytest.raises(ValueError):m.bounded_pending(rows,{'foreign'},4)


def test_resume_rejects_changed_config_and_duplicate_complete_rows(tmp_path):
    assert hasattr(m,'resume_completed')
    image=tmp_path/'i.png';image.write_bytes(b'pixels')
    manifest=[{'record_id':'a','prompt':'question','image':str(image),'category':'mri_sequence'}]
    expected={'manifest_sha256':'frozen','max_tokens':256}
    out=tmp_path/'predictions.jsonl'
    row={'record_id':'a','prompt':'question','image_sha256':m.sha256(image),'response':'answer',**expected}
    out.write_text(json.dumps(row)+'\n')
    assert m.resume_completed(out,manifest,expected,resume=True)=={'a'}
    with pytest.raises(FileExistsError):m.resume_completed(out,manifest,expected,resume=False)
    with pytest.raises(ValueError):m.resume_completed(out,manifest,dict(expected,max_tokens=64),resume=True)
    out.write_text((json.dumps(row)+'\n')*2)
    with pytest.raises(ValueError):m.resume_completed(out,manifest,expected,resume=True)


def test_scoped_approval_rejects_new_record_budget_or_source_change(tmp_path):
    assert hasattr(m,'verify_approval')
    scope={'new_record_ids':['a'],'max_new_records':1,'source_sha256':'source'}
    path=tmp_path/'approval.json';path.write_text(json.dumps({'kind':'paced_frozen_eval_v1','verdict':'PASS_WITH_LIMITATIONS','reviewer':'independent-codex','scope':scope}))
    digest=m.sha256(path);m.verify_approval(path,digest,scope)
    with pytest.raises(ValueError):m.verify_approval(path,digest,dict(scope,max_new_records=4))
    with pytest.raises(ValueError):m.verify_approval(path,'0'*64,scope)


def completed_receipt():
    return {'optimizer_steps':28,'completed_control_28steps':True,'status':'SFT_ENGINEERING_CONTROL',
            'checkpoint_reloaded':True,'schedule_policy':'balanced_sequence_v2','schedule_sha256':m.B_SCHEDULE_SHA256,
            'executed_prefix_examples':112,'executed_prefix_sha256':m.B_SCHEDULE_SHA256,
            'original_starting_adapter_sha256':m.ORIGINAL_ADAPTER_SHA256,'parameters_finite':True,
            'adapter_tensor_count':504,'changed_tensors':504,'examples':[{}]*112,'loss_terms':[1.0]*112,'gradient_l2_by_step':[0.1]*28}


def test_claim_validation_requires_complete_finite28_update_history():
    receipt=completed_receipt();m.validate_completion_claim(receipt)
    receipt['gradient_l2_by_step'][-1]=float('nan')
    with pytest.raises(ValueError):m.validate_completion_claim(receipt)
    receipt=completed_receipt();receipt['loss_terms'].pop()
    with pytest.raises(ValueError):m.validate_completion_claim(receipt)


def test_torn_response_is_preserved_and_only_unfinished_record_remains(tmp_path):
    rows=[{'record_id':k,'prompt':'q','image':None} for k in ('first','second')]
    result={'record_id':'first','prompt':'q','image_sha256':None,'response':'answer','manifest_sha256':'all54'}
    out=tmp_path/'results.jsonl';out.write_bytes((json.dumps(result)+'\n{"record_id":"second"').encode())
    done=m.resume_completed(out,rows,{'manifest_sha256':'all54'},resume=True)
    assert done=={'first'} and m.bounded_pending(rows,done,4)==[rows[1]]
    assert (tmp_path/'results.jsonl.partial').read_bytes()==b'{"record_id":"second"'
    assert len(out.read_text().splitlines())==1


def test_import_is_cpu_only_without_mlx_initialization():
    import os
    import subprocess
    import sys
    env=dict(os.environ,PYTHONPATH=str(Path(__file__).resolve().parents[1]/'src'))
    result=subprocess.run([sys.executable,'-c',"import sys; import mri_rl.paced_frozen_eval; assert 'mlx.core' not in sys.modules; assert 'mlx_vlm' not in sys.modules"],env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr


@pytest.fixture
def matched_plan_fixture(tmp_path,monkeypatch):
    """Exercise real prepare_plan fairness logic; unrelated B28 validation is stubbed.

    Synthetic baseline bytes let tests mutate pixels/prompts without touching MRI.
    """
    root=tmp_path/'repo';source=root/'src/mri_rl';source.mkdir(parents=True)
    for name in ('paced_frozen_eval','frozen_eval','paced_resources','resource_supervisor','experiment_contract'):
        (source/f'{name}.py').write_text('# fixture source identity\n')
    monkeypatch.setattr(m,'__file__',str(source/'paced_frozen_eval.py'))
    private=root/'local_data';private.mkdir();model=private/'model';model.mkdir()
    file_names={'model_config':'config.json','processor_config':'preprocessor_config.json','tokenizer_config':'tokenizer_config.json','chat_template':'chat_template.json'}
    paths={}
    for key,name in file_names.items():paths[key]=model/name;paths[key].write_text('{}')
    contract=SimpleNamespace(system_instruction=m.SYSTEM_PROMPT,generation={'temperature':0.0,'max_tokens':256},identity=lambda:'contract',
        **{key+'_sha256':m.sha256(path) for key,path in paths.items()})
    monkeypatch.setattr(m,'load_contract',lambda p:contract);monkeypatch.setattr(m,'resolve_artifact_paths',lambda p:paths)
    image=private/'image.png';image.write_bytes(b'frozen MRI pixels')
    rows=[]
    for category,count in m.CATEGORY_COUNTS.items():
        rows += [{'record_id':f'{category}-{i}','category':category,'image':str(image),'prompt':[{'role':'system','content':m.SYSTEM_PROMPT},{'role':'user','content':'question'}]} for i in range(count)]
    manifest=private/'manifest.jsonl';manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows));monkeypatch.setattr(m,'MATCHED_MANIFEST_SHA256',m.sha256(manifest))
    configsha='032becabb7cbb6d2af6f3451eb127e5f944551d42eb51eaee3e8d75b839b2f6a'
    specs={};baseline_files=[]
    for name,adaptersha in [('sft0_matched_native_outputs_PRIVATE.jsonl',m.ORIGINAL_ADAPTER_SHA256),('sft28_matched_native_outputs_PRIVATE.jsonl','244c26fd5ce2e3b02b46703d4f46a90608f2ef1a868eeee682ee4fe48a58ed73')]:
        path=private/name;output=[]
        for row in rows:
            output.append({**row,'image_sha256':m.sha256(image),'manifest_sha256':m.sha256(manifest),'response':'baseline',
                'model_path':str(model.resolve()),'model_config_sha256':contract.model_config_sha256,'model_template_sha256':contract.chat_template_sha256,
                'template_override_sha256':None,'max_tokens':256,'temperature':0,'image_placeholder_count':1,
                'adapter_verified_parameter_tensors':504,'adapter_checkpoint_sha256':adaptersha,'adapter_config_sha256':configsha})
        path.write_text(''.join(json.dumps(r)+'\n' for r in output));baseline_files.append(path)
        specs[name]={'sha256':m.sha256(path),'adapter_checkpoint_sha256':adaptersha}
    monkeypatch.setattr(m,'BASELINES',specs,raising=False)
    monkeypatch.setattr(m,'validate_b28',lambda *args:(private/'after.safetensors',{'adapter_checkpoint_sha256':'b28','adapter_config_sha256':configsha,'b28_receipt_sha256':'b28-receipt','b28_state_manifest_sha256':'state'}))
    args=(model,manifest,private/'new_outputs.jsonl',private/'b28',private/'contract.json')
    return {'args':args,'image':image,'baselines':baseline_files,'manifest':manifest,'rows':rows,'specs':specs}


def test_prepare_plan_rejects_pixel_drift_from_immutable_historical_results(matched_plan_fixture):
    fixture=matched_plan_fixture
    # Same manifest pathname and bytes, but its image now differs from both baselines.
    fixture['image'].write_bytes(b'changed MRI pixels')
    with pytest.raises(ValueError,match='(?i)baseline|historical|pixel'):
        m.prepare_plan(*fixture['args'])


def test_plan_binds_both_immutable_baseline_output_digests(matched_plan_fixture):
    fixture=matched_plan_fixture
    _,_,expected,scope,_=m.prepare_plan(*fixture['args'])
    assert expected['historical_baseline_sha256']=={name:spec['sha256'] for name,spec in fixture['specs'].items()}
    assert scope['expected']['historical_baseline_sha256']==expected['historical_baseline_sha256']


def test_plan_rejects_baseline_file_digest_drift(matched_plan_fixture):
    fixture=matched_plan_fixture
    with fixture['baselines'][0].open('a') as f:f.write('\n')
    with pytest.raises(ValueError,match='(?i)baseline|historical'):
        m.prepare_plan(*fixture['args'])


@pytest.mark.parametrize('field,value',[
    ('prompt',[{'role':'user','content':'different question'}]),('category','wrong-category'),
    ('record_id','unknown-id'),('model_path','different-model'),('model_config_sha256','wrong'),
    ('model_template_sha256','wrong'),('template_override_sha256','override'),('max_tokens',64),
    ('temperature',0.5),('image_placeholder_count',0),('adapter_verified_parameter_tensors',1),
    ('adapter_checkpoint_sha256','wrong-adapter'),('adapter_config_sha256','wrong-config')])
def test_baseline_semantic_profile_must_match_even_in_a_newly_bound_fixture(matched_plan_fixture,field,value):
    fixture=matched_plan_fixture;path=fixture['baselines'][1]
    rows=[json.loads(line) for line in path.read_text().splitlines()];rows[0][field]=value
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    fixture['specs'][path.name]['sha256']=m.sha256(path)
    with pytest.raises(ValueError,match='(?i)baseline|historical'):
        m.prepare_plan(*fixture['args'])


def test_baseline_rejects_duplicate_ids_with_correct_row_count(matched_plan_fixture):
    fixture=matched_plan_fixture;path=fixture['baselines'][0]
    rows=[json.loads(line) for line in path.read_text().splitlines()];rows[-1]=rows[0]
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows));fixture['specs'][path.name]['sha256']=m.sha256(path)
    with pytest.raises(ValueError,match='(?i)baseline|historical'):
        m.prepare_plan(*fixture['args'])
