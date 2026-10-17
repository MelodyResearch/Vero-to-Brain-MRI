"""CPU protocol fixtures for common-start calibration. Not native approvals."""
import importlib.util
import json
from pathlib import Path
import sys
import pytest
from mri_rl.two_task_pool import canonical_sha
from tests.test_two_task_training import ROWS


def entry():
    p=Path(__file__).parents[1]/'src/mri_rl/tier_c_calibration_entry.py'
    assert p.is_file(), 'Executable calibration entry missing'
    spec=importlib.util.spec_from_file_location('cal_entry_test',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


@pytest.fixture
def case(tmp_path):
    e=entry();train=tmp_path/'train.jsonl';train.write_text(''.join(json.dumps(r)+'\n' for r in ROWS))
    adapter=tmp_path/'a.safetensors';adapter.write_bytes(b'CPU_COMMON_START');(tmp_path/'adapter_config.json').write_text('{}')
    measure={'schema':'frozen-training-token-measurements-v1','split':'train','manifest_sha256':e.digest(train),
        'processor_sha256':'a'*64,'prompt_contract_sha256':'b'*64,
        'rows':[{'record_sha256':canonical_sha(r),'source':r['data_source'],'route':r['ability'],
                 'input_tokens':213 if r['ability']=='mri_sequence' else 405,'visual_tokens':154 if r['ability']=='mri_sequence' else 288} for r in ROWS]}
    native=[{'record_id':r['id'],'image_sha256':e.digest(r['images'][0]),'rendered_prompt_sha256':'c'*64,
             'input_token_ids':[[0]*(213 if r['ability']=='mri_sequence' else 405)],
             'input_token_ids_sha256':canonical_sha([[0]*(213 if r['ability']=='mri_sequence' else 405)]),
             'image_grid_thw':[[1,2,3]]} for r in ROWS]
    semantics={'system_instruction':'CPU fixture','assistant_prefix':'<|im_start|>assistant\n',
               'enable_thinking':False,'generation':{'max_tokens':256,'temperature':1.}}
    p=e.compile_calibration_plan(train,measure,native,adapter,'d'*64,semantics)
    return e,p,train,measure,native,adapter,semantics


def test_protocol_is_frozen_before_native_sampling(case):
    e,p,t,m,n,a,s=case
    assert p['seed']==7 and p['group_size']==4 and p['max_tokens']==256 and p['temperature']==1.
    assert p['row_ids']==[r['id'] for r in ROWS] and p['task_counts']=={'mri_sequence':28,'mri_visit_correspondence':48}
    assert p['input_token_sum']==25404 and p['token_cap']==179440
    assert p['statistics']['unit']=='per-prompt mean raw accuracy' and p['statistics']['zero_IQR']=='BLOCK_NO_FLOOR'
    assert 'tier_c_calibration.py' in e.source_hashes() and 'tier_c_calibration_entry.py' in e.source_hashes()
    assert e.compile_calibration_plan(t,m,n,a,'d'*64,s)==p


@pytest.mark.parametrize('changed',['missing_native','duplicated_native','changed_prompt_ids','changed_protocol','changed_image'])
def test_prepare_rejects_changed_coverage_inputs_or_protocol(case,changed):
    e,p,t,m,n,a,s=case;image=None;old=None
    if changed=='missing_native':n=n[:-1]
    elif changed=='duplicated_native':n=n+[n[0]]
    elif changed=='changed_prompt_ids':n=[{**n[0],'input_token_ids':[[1]*213]}]+n[1:]
    elif changed=='changed_protocol':p={**p,'group_size':8}
    else:image=Path(ROWS[0]['images'][0]);old=image.read_bytes();image.write_bytes(b'CHANGED')
    try:
        with pytest.raises(ValueError):e.prepare_calibration(p,t,m,n)
    finally:
        if image is not None:image.write_bytes(old)


def test_non76_or_reordered_manifest_cannot_inherit_measurement_receipt(case):
    e,p,t,m,n,a,s=case;t.write_text(''.join(json.dumps(r)+'\n' for r in ROWS[:-1]))
    with pytest.raises(ValueError):e.prepare_calibration(p,t,m,n)


def test_entry_import_does_not_construct_model_or_import_mlx():
    entry();assert not any(x in sys.modules for x in ('mlx.core','mlx_vlm','torch'))
