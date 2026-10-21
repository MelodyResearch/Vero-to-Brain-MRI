"""Synthetic CPU phase separation/admission/metadata tests; no native execution."""
import copy,importlib,json
from datetime import datetime,timedelta,timezone
from pathlib import Path
import pytest
from mri_rl import accuracy_recovery_v1 as ar


def entry():
    assert importlib.util.find_spec('mri_rl.safe_micro_entry_v1'), 'safe micro entry missing'
    return importlib.import_module('mri_rl.safe_micro_entry_v1')


def put(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x));return p


def cache_fixture(tmp_path,monkeypatch):
    e=entry();root=tmp_path/'repo';(root/'.git').mkdir(parents=True);D=root/'local_data/safe_micro_v1';D.mkdir(parents=True)
    launcher=root/'launch_safe_micro_v1.py';launcher.write_text('# external witness source fixture\n');monkeypatch.setattr(e,'EXTERNAL_LAUNCHER',launcher)
    cachecode=root/'cache_source.py';cachecode.write_text('def execute(**kwargs):\n pass\n')
    cache=put(D/'cache_scope.json',{'schema':'accuracy-recovery-historical-gpu-cache-v4-scope-v1','scope_kind':'FRESH_EXACT_HISTORICAL_GPU_CACHE_V4_CONTINUATION_SCOPE','source_sha256':{str(cachecode):ar.file_sha256(cachecode)},'input_sha256':'a'*64,'limits':{'operation_start_window_seconds':1800},'expected':{'units':750},'roots':{'global_lock':str(root/'one.lock')}})
    monkeypatch.setattr(e,'CACHE_SCOPE_FILE_SHA256',ar.file_sha256(cache));revoked=put(D/'revoked.json',{'status':'REVOKED_NOT_EXECUTABLE'})
    spec={'schema':'safe-micro-spec-v1','phase':'A','stage':1,'cache_contract':{'path':str(cache),'sha256':ar.file_sha256(cache)},'revoked_full_scope':{'path':str(revoked),'sha256':ar.file_sha256(revoked)},'output':str(D/'phase_A_attempt_PRIVATE')}
    sp=put(D/'A_spec.json',spec);return e,root,D,sp,spec


def sign(e,scope):
    now=datetime.now(timezone.utc);binding=e.scope_digest(scope)
    return ({'authorized':True,'action':scope['action'],'phase':scope['phase'],'scope_sha256':binding,'human_instruction_sha256':'6'*64,'approved_by':'researcher','issued_at_utc':(now-timedelta(seconds=5)).isoformat(),'expires_at_utc':(now+timedelta(hours=1)).isoformat(),'cache_scope_sha256':scope.get('cache_scope_sha256')}, {'verdict':'PASS','reviewer_id':'independent','reviewer_kind':'isolated_codex','scope_sha256':binding,'source_sha256':scope['source_sha256'],'input_sha256':scope['input_sha256']})


def test_A_is_cache_only_and_cannot_admit_B_or_auto_progress(tmp_path,monkeypatch):
    e,root,D,sp,spec=cache_fixture(tmp_path,monkeypatch);p=e.prepare(sp,start=root);s=p['executable_scope']
    assert s['phase']=='A' and s['automatic_phase_B'] is False and s['model_loads']==0
    assert s['budget']['training_updates']==0 and s['budget']['generation']==0
    h,r=sign(e,s);token=e.authorize(s,h,r,consumed=ar.consume_once(D/'ledger'));assert token.phase=='A'
    with pytest.raises(ValueError):e.authorize(s,{**h,'phase':'B'},r,consumed=ar.consume_once(D/'other'))
    with pytest.raises(ValueError,match='consumed'):e.authorize(s,h,r,consumed=ar.consume_once(D/'ledger'))


def test_missing_actual_cache_success_blocks_B_and_emits_no_launch_scope(tmp_path,monkeypatch):
    e,root,D,sp,spec=cache_fixture(tmp_path,monkeypatch);spec.update(phase='B',stage=2);put(sp,spec)
    p=e.prepare(sp,start=root);assert p['status']=='BLOCKED' and 'executable_scope' not in p
    assert 'STAGE_2_NOT_IMPLEMENTED_NOT_AUTHORIZED' in p['blockers']


def test_exact_fixed_budget_counts_common_once_and_joint_gradient_once_per_arm():
    b=entry().micro_budget()
    assert b['training']['generated_response_tokens']==4096
    assert b['training']['backward_calls']==32 and b['training']['teacherforced_input_positions']==27072
    assert b['training']['optimizer_updates']==2
    assert 'evaluation' not in b and 'total' not in b
    ev=entry().eval24_budget()
    assert ev['greedy_generations']==72 and ev['candidate_score_API_calls']==216
    assert ev['maximum_generated_tokens']==18432 and ev['teacherforced_input_positions']==91800
    assert b['batching']['weight_per_response']==1/16 and b['batching']['sequence_gradient_weight']==.5 and b['batching']['Dallas_gradient_weight']==.5


def test_source_or_metadata_tamper_is_refused(tmp_path,monkeypatch):
    e,root,D,sp,spec=cache_fixture(tmp_path,monkeypatch);p=e.prepare(sp,start=root);s=p['executable_scope'];h,r=sign(e,s)
    token=e.authorize(s,h,r,consumed=ar.consume_once(D/'ledger'));(root/'cache_source.py').write_text('changed')
    with pytest.raises(ValueError):token.check('native_boundary')
    with pytest.raises(ValueError):e.prepare(sp,start=root)


def test_stale_or_self_reviewed_A_does_not_consume_permission(tmp_path,monkeypatch):
    e,root,D,sp,spec=cache_fixture(tmp_path,monkeypatch);s=e.prepare(sp,start=root)['executable_scope'];h,r=sign(e,s)
    with pytest.raises(ValueError):e.authorize(s,h,{**r,'reviewer_id':e.IMPLEMENTED_BY},consumed=ar.consume_once(D/'ledger'))
    h['expires_at_utc']=(datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat()
    with pytest.raises(ValueError):e.authorize(s,h,r,consumed=ar.consume_once(D/'ledger'))
    assert not (D/'ledger').exists()


def test_cli_cpu_prepare_does_not_invoke_cache_or_native(tmp_path,monkeypatch,capsys):
    e,root,D,sp,spec=cache_fixture(tmp_path,monkeypatch);monkeypatch.setattr(e.authority,'canonical_project_root',lambda start=None:root)
    r=e.main(['--spec',str(sp),'--scope-output',str(D/'A_scope.json')]);printed=json.loads(capsys.readouterr().out)
    assert r['phase']=='A' and printed['model_loads']==0 and (D/'A_scope.json').exists()
    assert str(root) not in json.dumps(printed)


def test_execution_requires_all_new_artifact_hashes_before_cache_dispatch(tmp_path):
    from types import SimpleNamespace
    e=entry()
    with pytest.raises(ValueError,match='NOT_AUTHORIZED'):e.execute(SimpleNamespace(spec=tmp_path/'unused'))


def test_A_scope_cannot_be_changed_to_future_train_permission(tmp_path,monkeypatch):
    e,root,D,sp,spec=cache_fixture(tmp_path,monkeypatch);s=e.prepare(sp,start=root)['executable_scope']
    changed={**s,'stage':3};h,r=sign(e,changed)
    with pytest.raises(ValueError):e.authorize(changed,h,r,consumed=ar.consume_once(D/'ledger'))
    assert not (D/'ledger').exists()


def test_phase_A_current_scope_and_cli_do_not_import_old_full_entry(tmp_path,monkeypatch):
    import sys
    e,root,D,sp,spec=cache_fixture(tmp_path,monkeypatch)
    before=set(sys.modules);e.prepare(sp,start=root)
    new=set(sys.modules)-before
    assert not any(k.split('.')[0] in ('mlx','mlx_vlm','torch','transformers') for k in new)
    assert 'mri_rl.accuracy_stage_entry_v1' not in new
    assert 'mri_rl.accuracy_stage_native_v1' not in new


def test_absent_external_exit_witness_blocks_the_scope(tmp_path,monkeypatch):
    e,root,D,sp,spec=cache_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(e,'EXTERNAL_LAUNCHER',root/'absent_launcher.py',raising=False)
    p=e.prepare(sp,start=root)
    assert p['status']=='BLOCKED' and 'EXTERNAL_EXIT_WITNESS_SOURCE_NOT_PRESENT' in p['blockers']
    assert 'executable_scope' not in p


def test_stage1_binds_external_exit_record_without_authorizing_next_stage(tmp_path,monkeypatch):
    e,root,D,sp,spec=cache_fixture(tmp_path,monkeypatch);s=e.prepare(sp,start=root)['executable_scope']
    assert s['external_witness']['required_before_stage2'] is True
    assert s['external_witness']['path'].endswith('_external_witness_PRIVATE/actual_exit_PRIVATE.json')
    assert s['automatic_next_stage'] is False
