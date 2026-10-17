"""Tier C entry glue on synthetic rows with the CPU NumericBackend (no model)."""
import json
import pytest
from mri_rl import tier_c_entry as te, two_task_training as tt
from mri_rl.two_task_pool import canonical_sha
from tests.test_two_task_training import ROWS
from tests.test_continuous_rl_driver import NumericBackend

COMMON = {'adapter_sha256': 'a' * 64, 'cache_certificate_sha256': 'b' * 64, 'processor_sha256': 'p' * 64, 'prompt_contract_sha256': 'c' * 64}


def plan(arms=None, max_groups=4, cap=10**6):
    return {'seeds': [7, 11, 13], 'group_size': 2, 'max_tokens': 256, 'processor_sha256': 'p' * 64, 'measurements_sha256': 'm' * 64,
            'rl_arms': arms or [{'name': 'U_raw', 'sampling': 'uniform', 'reward': 'raw'},
                                {'name': 'U_calibrated', 'sampling': 'uniform', 'reward': 'medgrpo_eq3'},
                                {'name': 'S_sequence', 'sampling': 'specialist', 'task': 'mri_sequence', 'reward': 'raw'}],
            'budget': {'input_tokens_min': 213, 'input_tokens_max': 405, 'text_visual_token_cap': cap, 'maximum_attempted_groups': max_groups}}


MEAS = [{'record_sha256': canonical_sha(r), 'input_tokens': 213 if r['ability'] == 'mri_sequence' else 405} for r in ROWS]


def test_prepare_binds_plan_arm_seed_schedule_ledger_and_hashes():
    p = te.prepare_arm(plan(), 'U_raw', 7, ROWS, MEAS, COMMON)
    assert len(p['schedule']) == 4 and p['bindings']['arm']['name'] == 'U_raw' and p['bindings']['seed'] == 7
    assert p['bindings']['common'] == COMMON and p['bindings']['schedule_binding_sha256']
    assert te.prepare_arm(plan(), 'U_raw', 7, ROWS, MEAS, COMMON)['bindings'] == p['bindings']          # deterministic
    assert te.prepare_arm(plan(), 'U_raw', 11, ROWS, MEAS, COMMON)['bindings'] != p['bindings']


def test_prepare_rejects_unplanned_seed_arm_missing_common_calibration_and_uncovered_rows():
    with pytest.raises(ValueError): te.prepare_arm(plan(), 'U_raw', 99, ROWS, MEAS, COMMON)
    with pytest.raises(ValueError): te.prepare_arm(plan(), 'X', 7, ROWS, MEAS, COMMON)
    with pytest.raises(ValueError): te.prepare_arm(plan(), 'U_raw', 7, ROWS, MEAS, {**COMMON, 'adapter_sha256': ''})
    with pytest.raises(ValueError): te.prepare_arm(plan(), 'U_raw', 7, ROWS, MEAS, {**COMMON, 'processor_sha256': 'z' * 64})
    with pytest.raises(tt.CalibrationBlocked): te.prepare_arm(plan(), 'U_calibrated', 7, ROWS, MEAS, COMMON)
    with pytest.raises(ValueError): te.prepare_arm(plan(), 'U_raw', 7, ROWS, MEAS[:-1], COMMON)
    cal = {'mri_sequence': {'p50': .5, 'iqr': .5}, 'mri_visit_correspondence': {'p50': .5, 'iqr': .5}}
    assert te.prepare_arm(plan(), 'U_calibrated', 7, ROWS, MEAS, COMMON, calibration=cal)['bindings']['calibration_sha256']


def test_train_arm_runs_the_whole_resident_budget_with_ledger_and_resumes(tmp_path):
    NumericBackend.instances = []
    p = te.prepare_arm(plan(), 'U_raw', 7, ROWS, MEAS, COMMON)
    r = te.train_arm(p, output=tmp_path / 'o', backend_factory=NumericBackend, authorize=lambda c: None)
    assert r['completed_groups'] == 4 and len(NumericBackend.instances) == 1          # one resident backend for all groups
    assert r['ledger']['charged_groups'] == 4 and r['ledger']['used_tokens'] > 0
    again = te.train_arm(te.prepare_arm(plan(), 'U_raw', 7, ROWS, MEAS, COMMON), output=tmp_path / 'o',
                         backend_factory=NumericBackend, authorize=lambda c: None, resume=True)
    assert again['status'] == 'SCHEDULE_COMPLETE' and again['completed_groups'] == 4


def test_train_arm_stops_when_the_next_whole_group_does_not_fit_the_token_cap(tmp_path):
    p = te.prepare_arm(plan(max_groups=6, cap=2 * (405 + 256) * 2 + 5), 'S_sequence', 7, ROWS, MEAS, COMMON)
    r = te.train_arm(p, output=tmp_path / 'o', backend_factory=NumericBackend, authorize=lambda c: None)
    assert r['status'] == 'TOKEN_BUDGET_EXHAUSTED' and r['completed_groups'] < 6
