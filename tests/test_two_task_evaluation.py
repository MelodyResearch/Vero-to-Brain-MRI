"""Projection/binding tests use synthetic text and bytes, never model results."""
import copy
import importlib.util
import json
from pathlib import Path
import pytest


def mod():
    assert importlib.util.find_spec('mri_rl.two_task_evaluation') is not None,'Missing bound C evaluation projection'
    from mri_rl import two_task_evaluation
    return two_task_evaluation


def test_projection_passes_fixed_system_and_question_string_no_hidden_gold():
    m=mod();rows=[{'id':'anonymous','ability':'mri_sequence','images':['/private/opaque.png'],
                 'reward_model':{'ground_truth':'SECRET GOLD'},
                 'extra_info':{'question':'What sequence?','patient_key':'PRIVATE PATIENT'}}]
    records=m.project_records(rows,{'system_instruction':'Native system','assistant_prefix':'<|im_start|>assistant\n','enable_thinking':False})
    assert records==[{'record_id':'anonymous','category':'mri_sequence','image':'/private/opaque.png',
                     'prompt':[{'role':'system','content':'Native system'},{'role':'user','content':'What sequence?'}]}]
    assert 'SECRET GOLD' not in json.dumps(records) and 'PRIVATE PATIENT' not in json.dumps(records)
    assert isinstance(records[0]['prompt'][1]['content'],str)


def native_adapter_fixture(tmp_path):
    import hashlib
    from mri_rl.two_task_pool import canonical_sha
    image=tmp_path/'opaque.png';image.write_bytes(b'fixture-only pixels')
    contract={'system_instruction':'Native system','assistant_prefix':'<|im_start|>assistant\n','enable_thinking':False,'generation':{'max_tokens':256},'model_config_sha256':'1'*64,'chat_template_sha256':'2'*64}
    cp=tmp_path/'contract.json';cp.write_text(json.dumps(contract))
    row={'id':'r','ability':'mri_sequence','images':[str(image)],'extra_info':{'question':'Question'},'reward_model':{'ground_truth':'FLAIR'}}
    source=tmp_path/'source.jsonl';source.write_text(json.dumps(row)+'\n')
    rec=mod().project_records([row],contract)[0];manifest=tmp_path/'eval.jsonl';manifest.write_text(json.dumps(rec)+'\n')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    projection={'source_verl_manifest_sha256':sha(source),'eval_manifest_sha256':sha(manifest),'prompt_contract_sha256':sha(cp)}
    pp=tmp_path/'projection.json';pp.write_text(json.dumps(projection))
    expected={'model_path':str(tmp_path/'model'),'model_config_sha256':'1'*64,'model_template_sha256':'2'*64,'template_override_sha256':None,
              'manifest_sha256':sha(manifest),'max_tokens':256,'temperature':0,'adapter_checkpoint_sha256':'3'*64,'adapter_config_sha256':'4'*64}
    cache={'certificate_sha256':'5'*64,'config_source_sha256':'1'*64,'tensors':1438}
    output=tmp_path/'output.jsonl'
    scope={'execution_tier':'B','kind':'continuous_rl_dev_eval_v1','label':'fixture','contract_id':canonical_sha(contract),
           'manifest_sha256':sha(manifest),'subset_record_ids':['r'],'max_tokens':256,'temperature':0,
           'adapter_checkpoint_sha256':'3'*64,'adapter_config_sha256':'4'*64,'output':str(output),'weight_updates':0,
           'derived_cache':cache,'image_sha256':{'r':sha(image)},'model_inventory':{'model_dir':str(tmp_path/'model'),'files':[{'path':'config.json','sha256':'1'*64}]},
           'source_sha256':{'continuous_rl_dev_eval.py':'6'*64,'mlx_continuous_gspo.py':'7'*64,'likelihood_cache.py':'8'*64}}
    prediction={'record_id':'r','category':'mri_sequence','prompt':rec['prompt'],'image_sha256':sha(image),'image_placeholder_count':1,
                'engine':'continuous_rl_dev_eval_native_serial_v1','response':'fixture raw output','token_ids':[1],**expected}
    output.write_text(json.dumps(prediction)+'\n')
    kwargs={'source_manifest':source,'eval_manifest':manifest,'projection_receipt':pp,'contract_path':cp,'output_path':output,
            'expected_config':expected,'approved_scope':scope,'expected_scope_sha256':canonical_sha(scope),
            'expected_projection_sha256':sha(pp),'expected_output_sha256':sha(output),'expected_cache_binding':cache}
    return kwargs,prediction,sha


def adapter():
    assert importlib.util.find_spec('mri_rl.two_task_native_results') is not None,'Missing strict native raw-output adapter'
    from mri_rl import two_task_native_results
    return two_task_native_results


def test_real_cli_schema_adapts_only_after_exact_external_scope_and_file_bindings(tmp_path):
    m=adapter();kwargs,_,_=native_adapter_fixture(tmp_path)
    normalized,proof=m.adapt_cli_output(**kwargs)
    assert normalized=={'r':{'response':'fixture raw output'}}
    assert proof['rows']==1 and proof['scope_supplies_omitted_contract_cache_precision_binding'] is True


@pytest.mark.parametrize('field',['adapter_checkpoint_sha256','image_sha256','prompt','manifest_sha256','model_config_sha256','temperature','engine','cache','scope_sha','output_sha','missing'])
def test_wrong_checkpoint_image_prompt_model_precision_or_missing_binding_rejected(tmp_path,field):
    m=adapter();kwargs,row,sha=native_adapter_fixture(tmp_path)
    if field=='cache':kwargs['expected_cache_binding']={'certificate_sha256':'9'*64}
    elif field=='scope_sha':kwargs['expected_scope_sha256']='0'*64
    elif field=='output_sha':kwargs['expected_output_sha256']='0'*64
    elif field=='missing':row.pop('adapter_config_sha256')
    elif field=='prompt':row['prompt']=[{'role':'user','content':'Question'}]
    elif field=='temperature':row[field]=1
    elif field=='engine':row[field]='unreviewed-engine'
    else:row[field]='0'*64
    kwargs['output_path'].write_text(json.dumps(row)+'\n')
    if field!='output_sha':kwargs['expected_output_sha256']=sha(kwargs['output_path'])
    with pytest.raises(ValueError):m.adapt_cli_output(**kwargs)
