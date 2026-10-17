"""Conditional calibrated configs only; no native runtime, approvals or training."""
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest
from mri_rl.tier_c_calibration_entry import compile_calibration_plan
from mri_rl.two_task_pool import canonical_sha
from tests.test_two_task_training import ROWS


def module():
    p=Path(__file__).parents[1]/'src/mri_rl/tier_c_calibrated_bundle.py'
    assert p.is_file(), 'Conditional calibrated-arm builder is missing'
    spec=importlib.util.spec_from_file_location('cal_bundle_test',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_conditional_builder_not_an_approval_or_native_runner():
    m=module();p=m.conditions()
    assert p['status']=='WAITING_FOR_ACTUAL_CALIBRATION_AND_INDEPENDENT_PROOF'
    assert p['stages']==[] and p['models_launched']==0
    assert p['prospective_stage_order']==[f'rl:{a}:{s}' for s in (7,11,13) for a in ('P_calibrated','U_calibrated')]
    assert p['zero_IQR']=='BLOCK_NO_FLOOR'


def test_missing_or_changed_independent_proof_blocks_before_outputs(tmp_path):
    m=module();p=tmp_path/'proof.json';p.write_text('{}')
    with pytest.raises(ValueError,match='proof'):
        m.authenticate_calibration(tmp_path/'missing',tmp_path/'plan',p,'0'*64,{'common_bindings':{}})
    assert list(tmp_path.iterdir())==[p]


@pytest.fixture
def arm_case(tmp_path):
    m=module();a=tmp_path/'adapter.safetensors';a.write_bytes(b'CPU');(tmp_path/'adapter_config.json').write_text('{}')
    t=tmp_path/'train.jsonl';t.write_text(''.join(json.dumps(r)+'\n' for r in ROWS))
    measure={'schema':'frozen-training-token-measurements-v1','split':'train','manifest_sha256':m.digest(t),'processor_sha256':'a'*64,'prompt_contract_sha256':'b'*64,
        'rows':[{'record_sha256':canonical_sha(r),'source':r['data_source'],'route':r['ability'],'input_tokens':213 if r['ability']=='mri_sequence' else 405,'visual_tokens':154 if r['ability']=='mri_sequence' else 288} for r in ROWS]}
    from mri_rl.full_arm_protocol import compile_plan
    from mri_rl.tier_c_entry import common_bindings
    plan=compile_plan(measure,common_bindings(measure,a,'d'*64))
    return m,plan,measure,t


def test_six_configs_have_exact_same_sampler_schedule_as_raw_arms(arm_case,tmp_path):
    m,p,meas,t=arm_case
    cal={r:{'p50':.5,'iqr':.25} for r in ('mri_sequence','mri_visit_correspondence')}
    configs=m.stage_configs(p,ROWS,meas,cal,tmp_path/'bundle',{'CPU_FIXTURE':True})
    assert len(configs)==6
    for spec in configs:
        assert spec['scope']['calibration_sha256']==m.sha(cal)
        assert spec['scope']['budget']==p['budget'] and spec['scope']['group_size']==4
        assert spec['scope']['max_attempted_groups']==198
        assert spec['scope']['common']['starting_adapter_sha256']==p['common_bindings']['starting_adapter_sha256']
        raw=m.te.prepare_arm(p,spec['arm'].replace('calibrated','raw'),spec['seed'],ROWS,meas['rows'])
        assert spec['schedule_ids']==[r['id'] for r in raw['schedule']]
    assert len({s['output'] for s in configs})==6


def test_zero_IQR_never_emits_arm_configs(arm_case,tmp_path):
    m,p,meas,t=arm_case
    with pytest.raises(ValueError):m.stage_configs(p,ROWS,meas,{'mri_sequence':{'p50':1.,'iqr':0.},'mri_visit_correspondence':{'p50':.5,'iqr':.25}},tmp_path/'out',{})
    assert not (tmp_path/'out').exists()


def test_launch_argv_has_explicit_calibration_and_preserves_common_start():
    m=module();base=['/lexical/venv/bin/python','-u','-m','mri_rl.tier_c_entry','train','--arm','P_raw','--seed','7','--output','/old','--adapter','/same','--plan','/plan']
    out=m.launch_argv(base,{'arm':'U_calibrated','seed':13,'output':'/new'},Path('/frozen/calibration.json'))
    assert out[out.index('--calibration')+1]=='/frozen/calibration.json'
    assert out[out.index('--arm')+1]=='U_calibrated' and out[out.index('--seed')+1]=='13'
    assert out[out.index('--adapter')+1]=='/same' and out[out.index('--plan')+1]=='/plan'
    assert '--review-verdict' not in out

@pytest.fixture
def journal_case(arm_case,tmp_path):
    m,rawplan,measure,t=arm_case
    from tests.test_tier_c_calibration_journal import JournalBackend,PROMPT
    class Backend(JournalBackend):
        def rollout(self,row,index,directory):
            raw=super().rollout(row,index,directory)
            raw['prompt_token_ids']=[[0]*(213 if row['ability']=='mri_sequence' else 405)]
            return raw
    sigs={r['id']:{'rendered_prompt_sha256':hashlib.sha256(PROMPT.encode()).hexdigest(),
        'input_token_ids':[[0]*(213 if r['ability']=='mri_sequence' else 405)]} for r in ROWS}
    plan={'schema':'common-B28-training-calibration-v1','source_sha256':m.calibration_sources(),
        'common_bindings':rawplan['common_bindings'],'manifest_sha256':rawplan['manifest_sha256'],
        'processor_sha256':rawplan['processor_sha256'],'measurements_sha256':rawplan['measurements_sha256'],
        'row_ids':[r['id'] for r in ROWS],'row_sha256':[canonical_sha(r) for r in ROWS],
        'image_sha256':{r['id']:m.digest(r['images'][0]) for r in ROWS},'prompt_signatures':sigs}
    pp=tmp_path/'calplan.json';pp.write_text(json.dumps(plan));run=tmp_path/'calrun'
    result=m.cal.run_calibration(rows=ROWS,backend_factory=Backend,authorize=lambda x:None,output=run,group_size=4,
        input_tokens=lambda r:213 if r['ability']=='mri_sequence' else 405,
        bindings={'plan_sha256':m.sha(plan),'starting_adapter_sha256':rawplan['common_bindings']['starting_adapter_sha256'],'seed':7},prompt_signatures=sigs)
    assert result['complete'] and result['calibrated_arms_unblocked']
    calibration=m.cal.calibration_for_arms(result['stats']);cp=run/'calibration_for_arms_PRIVATE.json';cp.write_text(json.dumps(calibration))
    result.update({'plan_sha256':m.sha(plan),'journal_binding_sha256':m.digest(run/'binding_PRIVATE.json'),
        'final_row_receipt_sha256':m.digest(run/'row_0075/receipt.json'),'calibration_file_sha256':m.digest(cp)})
    complete=run/'calibration_complete_SAFE.json';complete.write_text(json.dumps(result))
    proof={'schema':m.PROOF_SCHEMA,'verdict':'PASS_WITH_LIMITATIONS','reviewer_kind':'isolated_codex','reviewer_id':'CPU_FIXTURE_NOT_AUTHORIZATION',
        'ordered_raw_76_verified':True,'raw_quantiles_independently_recomputed':True,'calibration_plan_sha256':m.sha(plan),
        'journal_binding_sha256':m.digest(run/'binding_PRIVATE.json'),'final_row_receipt_sha256':m.digest(run/'row_0075/receipt.json'),
        'statistics_sha256':m.sha(result['stats']),'complete_receipt_sha256':m.digest(complete)}
    proofpath=tmp_path/'proof.json';proofpath.write_text(json.dumps(proof))
    return m,rawplan,run,pp,proofpath


def test_authenticated_complete_journal_required_before_usable_config(journal_case):
    m,p,run,plan,proof=journal_case
    value=m.authenticate_calibration(run,plan,proof,m.digest(proof),p,ROWS)
    assert not value['blocked'] and value['file_sha256']==m.digest(run/'calibration_for_arms_PRIVATE.json')
    assert set(value['value'])=={'mri_sequence','mri_visit_correspondence'}
    (run/'calibration_for_arms_PRIVATE.json').write_text('{}')
    with pytest.raises(ValueError,match='Frozen calibration file'):m.authenticate_calibration(run,plan,proof,m.digest(proof),p,ROWS)


def test_proof_for_other_raw_quantiles_or_incomplete_run_cannot_release(journal_case):
    m,p,run,plan,proof=journal_case
    x=json.loads(proof.read_text());x['statistics_sha256']='0'*64;proof.write_text(json.dumps(x))
    with pytest.raises(ValueError,match='another calibration'):m.authenticate_calibration(run,plan,proof,m.digest(proof),p,ROWS)
    (run/'row_0075/raw_PRIVATE.json').write_text('{}')
    with pytest.raises(ValueError,match='file hash'):m.authenticate_calibration(run,plan,proof,m.digest(proof),p,ROWS)
