"""Tier C entry glue on synthetic rows with the CPU NumericBackend (no model)."""
import json
import pytest
from mri_rl import tier_c_entry as te, two_task_training as tt
from mri_rl.two_task_pool import canonical_sha
from tests.test_two_task_training import ROWS
from tests.test_continuous_rl_driver import NumericBackend

COMMON = {'starting_adapter_sha256': 'a' * 64, 'cache_certificate_sha256': 'b' * 64, 'prompt_contract_sha256': 'c' * 64,
          'native_backend_sha256': 'd' * 64, 'reward_source_sha256': 'e' * 64}


def plan(arms=None, max_groups=4, cap=10**6):
    return {'seeds': [7, 11, 13], 'group_size': 2, 'max_tokens': 256, 'processor_sha256': 'p' * 64, 'measurements_sha256': 'm' * 64, 'manifest_sha256': 'f' * 64, 'common_bindings': dict(COMMON),
            'rl_arms': arms or [{'name': 'U_raw', 'sampling': 'uniform', 'reward': 'raw'},
                                {'name': 'U_calibrated', 'sampling': 'uniform', 'reward': 'medgrpo_eq3'},
                                {'name': 'S_sequence', 'sampling': 'specialist', 'task': 'mri_sequence', 'reward': 'raw'}],
            'budget': {'input_tokens_min': 213, 'input_tokens_max': 405, 'text_visual_token_cap': cap, 'maximum_attempted_groups': max_groups}}


FULL = {**COMMON, 'processor_sha256': 'p' * 64, 'manifest_sha256': 'f' * 64}
MEAS = [{'record_sha256': canonical_sha(r), 'input_tokens': 213 if r['ability'] == 'mri_sequence' else 405} for r in ROWS]


def test_prepare_binds_plan_arm_seed_schedule_ledger_and_hashes():
    p = te.prepare_arm(plan(), 'U_raw', 7, ROWS, MEAS)
    assert len(p['schedule']) == 4 and p['bindings']['arm']['name'] == 'U_raw' and p['bindings']['seed'] == 7
    assert p['bindings']['common'] == FULL and p['bindings']['schedule_binding_sha256']
    assert te.prepare_arm(plan(), 'U_raw', 7, ROWS, MEAS)['bindings'] == p['bindings']          # deterministic
    assert te.prepare_arm(plan(), 'U_raw', 11, ROWS, MEAS)['bindings'] != p['bindings']


def test_prepare_rejects_unplanned_seed_arm_missing_common_calibration_and_uncovered_rows():
    with pytest.raises(ValueError): te.prepare_arm(plan(), 'U_raw', 99, ROWS, MEAS)
    with pytest.raises(ValueError): te.prepare_arm(plan(), 'X', 7, ROWS, MEAS)
    bad = plan(); bad['common_bindings'] = {**COMMON, 'starting_adapter_sha256': ''}
    with pytest.raises(ValueError): te.prepare_arm(bad, 'U_raw', 7, ROWS, MEAS)
    with pytest.raises(tt.CalibrationBlocked): te.prepare_arm(plan(), 'U_calibrated', 7, ROWS, MEAS)
    with pytest.raises(ValueError): te.prepare_arm(plan(), 'U_raw', 7, ROWS, MEAS[:-1])
    cal = {'mri_sequence': {'p50': .5, 'iqr': .5}, 'mri_visit_correspondence': {'p50': .5, 'iqr': .5}}
    assert te.prepare_arm(plan(), 'U_calibrated', 7, ROWS, MEAS, calibration=cal)['bindings']['calibration_sha256']


def test_train_arm_runs_the_whole_resident_budget_with_ledger_and_resumes(tmp_path):
    NumericBackend.instances = []
    p = te.prepare_arm(plan(), 'U_raw', 7, ROWS, MEAS)
    r = te.train_arm(p, output=tmp_path / 'o', backend_factory=NumericBackend, authorize=lambda c: None)
    assert r['completed_groups'] == 4 and len(NumericBackend.instances) == 1          # one resident backend for all groups
    assert r['ledger']['charged_groups'] == 4 and r['ledger']['used_tokens'] > 0
    again = te.train_arm(te.prepare_arm(plan(), 'U_raw', 7, ROWS, MEAS), output=tmp_path / 'o',
                         backend_factory=NumericBackend, authorize=lambda c: None, resume=True)
    assert again['status'] == 'SCHEDULE_COMPLETE' and again['completed_groups'] == 4


def test_train_arm_stops_when_the_next_whole_group_does_not_fit_the_token_cap(tmp_path):
    p = te.prepare_arm(plan(max_groups=6, cap=2 * (405 + 256) * 2 + 5), 'S_sequence', 7, ROWS, MEAS)
    r = te.train_arm(p, output=tmp_path / 'o', backend_factory=NumericBackend, authorize=lambda c: None)
    assert r['status'] == 'TOKEN_BUDGET_EXHAUSTED' and r['completed_groups'] < 6


def test_review_scope_binds_plan_arm_seed_budget_sources_and_common_start(tmp_path):
    p = te.prepare_arm(plan(), 'U_raw', 7, ROWS, MEAS)
    sc = te.review_scope(p, output=tmp_path / 'o')
    assert sc['execution_tier'] == 'C' and sc['arm']['name'] == 'U_raw' and sc['seed'] == 7 and sc['common'] == FULL
    assert sc['max_attempted_groups'] == 4 and 'continuous_rl.py' in sc['source_sha256'] and 'vero_category_sampler.py' in sc['source_sha256']
    assert te.review_scope(te.prepare_arm(plan(), 'U_raw', 11, ROWS, MEAS), output=tmp_path / 'o') != sc


def test_common_bindings_come_from_files_and_sources_not_free_text(tmp_path):
    adapter = tmp_path / 'a.safetensors'; adapter.write_bytes(b'weights')
    cb = te.common_bindings({'prompt_contract_sha256': 'c' * 64}, adapter, 'b' * 64)
    import hashlib
    assert cb['starting_adapter_sha256'] == hashlib.sha256(b'weights').hexdigest() and set(cb) == set(__import__('mri_rl.full_arm_protocol', fromlist=['x']).BINDINGS)


def test_runtime_contract_may_differ_in_file_hash_only_with_identical_prompt_semantics(tmp_path):
    import hashlib
    from types import SimpleNamespace
    def mk(name, system='S', extra=''):
        p = tmp_path / name; p.write_text(json.dumps({'system_instruction': system, 'assistant_prefix': 'A', 'enable_thinking': False,
                                                      'generation': {'max_tokens': 256, 'temperature': 1.0}, 'split_manifest': extra})); return p
    load = lambda p: SimpleNamespace(**json.loads(p.read_text()))
    measured, same, other = mk('m.json', extra='old split'), mk('r.json', extra='new split'), mk('o.json', system='DIFFERENT')
    adapter = tmp_path / 'a.safetensors'; adapter.write_bytes(b'w')
    meas = {'prompt_contract_sha256': hashlib.sha256(measured.read_bytes()).hexdigest()}
    pl = {'measurements_sha256': te.measurements_sha256(meas), 'common_bindings': {'starting_adapter_sha256': hashlib.sha256(b'w').hexdigest()}}
    assert te.verify_runtime_artifacts(pl, meas, adapter, same, load, measured)['runtime_contract_sha256'] != meas['prompt_contract_sha256']
    with pytest.raises(ValueError): te.verify_runtime_artifacts(pl, meas, adapter, same, load)               # differing file, no measured contract
    with pytest.raises(ValueError): te.verify_runtime_artifacts(pl, meas, adapter, other, load, measured)    # different prompt semantics
    (tmp_path / 'adapter_config.json').write_text('{}')
    pl['adapter_config_sha256'] = '0' * 64
    with pytest.raises(ValueError): te.verify_runtime_artifacts(pl, meas, adapter, measured, load)           # persisted quant map/config differs
