"""Tier C arm entry: binds the compiled plan, frozen occurrence schedule, token ledger, grader and common start into
ONE run configuration for continuous_rl.run. Claude-authored glue; sampler, plan compiler, Dallas scorer and analysis are
Codex-authored helpers reused unchanged. No model imports; the native backend factory/gate are supplied by the caller.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

from mri_rl.continuous_rl import run, canonical
from mri_rl.two_task_training import (TokenLedger, build_schedule, input_tokens_by_source, make_grader,
                                      realized_report, validate_training_row, CalibrationBlocked)

REQUIRED_COMMON = ('adapter_sha256', 'cache_certificate_sha256', 'processor_sha256', 'prompt_contract_sha256')


def _sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def prepare_arm(plan, arm_name, seed, source_rows, measurement_rows, common, calibration=None):
    """Return everything run() needs. Raises on any binding mismatch; never invents calibration."""
    if seed not in plan['seeds']:
        raise ValueError('Seed outside the compiled plan')
    arms = {a['name']: a for a in plan['rl_arms']}
    if arm_name not in arms:
        raise ValueError('Arm outside the compiled plan')
    arm = arms[arm_name]
    missing = [k for k in REQUIRED_COMMON if not common.get(k)]
    if missing:
        raise ValueError('Common-start bindings missing: ' + ','.join(missing))
    if common['processor_sha256'] != plan['processor_sha256']:
        raise ValueError('Plan/common processor mismatch')
    for row in source_rows:
        validate_training_row(row)
    maximum = plan['budget']['maximum_attempted_groups']
    schedule, schedule_binding = build_schedule(source_rows, arm, seed, maximum)
    if arm['reward'] == 'medgrpo_eq3' and calibration is None:
        raise CalibrationBlocked('Calibrated arm needs frozen common-initializer training-only statistics')
    grader = make_grader(calibration if arm['reward'] == 'medgrpo_eq3' else None)
    lookup = input_tokens_by_source(measurement_rows, source_rows)
    ledger_plan = {'group_size': plan['group_size'], 'max_tokens': plan['max_tokens'], 'budget': plan['budget']}
    bindings = {'plan_sha256': _sha(plan), 'arm': arm, 'seed': seed, 'schedule_binding_sha256': _sha(schedule_binding),
                'measurements_sha256': plan['measurements_sha256'], 'common': common,
                'calibration_sha256': _sha(calibration) if calibration is not None else None}
    return {'arm': arm, 'seed': seed, 'schedule': schedule, 'schedule_binding': schedule_binding, 'bindings': bindings,
            'ledger_plan': ledger_plan, 'grader': grader, 'input_tokens': lookup, 'max_groups': maximum,
            'realized': realized_report(schedule)}


def train_arm(prepared, *, output, backend_factory, authorize, resume=False, starting_adapter=None):
    """One resident run over the whole compiled budget (groups are atomic checkpoints; no per-batch unload/approval)."""
    config = {'profile': 'continuous_mlx_diagnostic_v1', 'optimizer': {'name': 'Adam', 'learning_rate': 1e-5, 'betas': [0.9, 0.999],
              'eps': 1e-8, 'bias_correction': False}, 'scheduler': {'type': 'constant'}, 'reward_profile': 'direct_vero',
              'temperature': 1.0, 'group_size': prepared['ledger_plan']['group_size'], 'max_tokens': prepared['ledger_plan']['max_tokens'],
              'clip_low': .0003, 'clip_high': .0004, 'log_ratio_max': 10., 'precision': '4bit-base-native-LoRA', 'lora_rank': 4,
              'lora_alpha': 8, 'active_group_rule': 'all-full-format_task-varies_equal-overlong'}
    ledger = TokenLedger(prepared['ledger_plan'])
    result = run(output=output, schedule=prepared['schedule'], bindings=prepared['bindings'], backend_factory=backend_factory,
                 authorize=authorize, max_new_groups=prepared['max_groups'], resume=resume, starting_adapter=starting_adapter,
                 config=config, grader=prepared['grader'], row_validator=validate_training_row, ledger=ledger,
                 input_tokens=prepared['input_tokens'], max_groups_cap=prepared['max_groups'])
    return {**result, 'ledger': ledger.state()}
