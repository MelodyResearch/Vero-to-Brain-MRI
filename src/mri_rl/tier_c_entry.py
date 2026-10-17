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

REQUIRED_COMMON = ('starting_adapter_sha256', 'cache_certificate_sha256', 'processor_sha256', 'prompt_contract_sha256',
                   'native_backend_sha256', 'reward_source_sha256')


def _sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def prepare_arm(plan, arm_name, seed, source_rows, measurement_rows, calibration=None):
    """Return everything run() needs. Raises on any binding mismatch; never invents calibration."""
    if seed not in plan['seeds']:
        raise ValueError('Seed outside the compiled plan')
    arms = {a['name']: a for a in plan['rl_arms']}
    if arm_name not in arms:
        raise ValueError('Arm outside the compiled plan')
    arm = arms[arm_name]
    common = {**plan['common_bindings'], 'processor_sha256': plan['processor_sha256'], 'manifest_sha256': plan['manifest_sha256']}
    missing = [k for k in REQUIRED_COMMON if not common.get(k)]
    if missing:
        raise ValueError('Common-start bindings missing: ' + ','.join(missing))
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


ROOT = Path(__file__).resolve().parents[2]
CRITICAL_SOURCES = ('tier_c_entry.py', 'two_task_training.py', 'continuous_rl.py', 'mlx_continuous_gspo.py', 'continuous_rl_pilot.py',
                    'optimizer_checkpoint.py', 'vero_mri_reward.py', 'visit_correspondence.py', 'vero_category_sampler.py',
                    'full_arm_protocol.py', 'likelihood_cache.py', 'likelihood_loading.py', 'paced_resources.py',
                    'resource_supervisor.py', 'reward_replay.py', 'two_task_pool.py')


def source_hashes():
    return {n: hashlib.sha256((ROOT / 'src' / 'mri_rl' / n).read_bytes()).hexdigest() for n in CRITICAL_SOURCES}


def common_bindings(measurements, adapter, cache_certificate_sha256):
    """Derive the compiler's five exact bindings from local files/sources (no free-form hashes)."""
    h = source_hashes()
    reward = {k: h[k] for k in ('vero_mri_reward.py', 'visit_correspondence.py', 'two_task_training.py', 'reward_replay.py')}
    return {'cache_certificate_sha256': cache_certificate_sha256,
            'starting_adapter_sha256': hashlib.sha256(Path(adapter).read_bytes()).hexdigest(),
            'prompt_contract_sha256': measurements['prompt_contract_sha256'],
            'native_backend_sha256': h['mlx_continuous_gspo.py'],
            'reward_source_sha256': hashlib.sha256(canonical(reward).encode()).hexdigest()}


def review_scope(prepared, *, output):
    """Exact scope an independent reviewer must bind (Tier C: substantial training, whole compiled budget)."""
    b = prepared['bindings']
    return {'kind': 'tier_c_arm_v1', 'execution_tier': 'C', 'phase': 'continuation', 'arm': prepared['arm'], 'seed': prepared['seed'],
            'plan_sha256': b['plan_sha256'], 'schedule_binding_sha256': b['schedule_binding_sha256'],
            'schedule_ids_sha256': prepared['realized']['schedule_sha256'], 'common': b['common'],
            'calibration_sha256': b['calibration_sha256'], 'budget': prepared['ledger_plan']['budget'],
            'group_size': prepared['ledger_plan']['group_size'], 'max_attempted_groups': prepared['max_groups'],
            'output': str(Path(output).resolve()), 'source_sha256': source_hashes()}


def main():
    import argparse
    from mri_rl.full_arm_protocol import compile_plan
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=('compile', 'scope', 'train'))
    ap.add_argument('--measurements', type=Path, required=True)
    ap.add_argument('--train', type=Path, required=True)
    ap.add_argument('--adapter', type=Path, required=True)
    ap.add_argument('--cache-certificate-sha256', required=True)
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--arm'); ap.add_argument('--seed', type=int)
    ap.add_argument('--calibration', type=Path)
    ap.add_argument('--output', type=Path)
    ap.add_argument('--model', type=Path); ap.add_argument('--contract', type=Path)
    ap.add_argument('--cache-dir', type=Path); ap.add_argument('--cache-certificate', type=Path)
    ap.add_argument('--review-verdict', type=Path); ap.add_argument('--expected-review-sha256')
    ap.add_argument('--resume', action='store_true')
    a = ap.parse_args()
    meas = json.loads(a.measurements.read_text())
    if a.command == 'compile':
        plan = compile_plan(meas, common_bindings(meas, a.adapter, a.cache_certificate_sha256))
        plan['common_bindings'] = common_bindings(meas, a.adapter, a.cache_certificate_sha256)
        with a.plan.open('x') as f:
            json.dump(plan, f, indent=2, sort_keys=True, allow_nan=False)
        print(json.dumps({'plan_sha256': _sha(plan), 'token_cap': plan['budget']['text_visual_token_cap'],
                          'max_groups': plan['budget']['maximum_attempted_groups'], 'arms': len(plan['rl_arms'])}))
        return
    plan = json.loads(a.plan.read_text())
    rows = [json.loads(l) for l in a.train.read_text().splitlines() if l.strip()]
    calibration = json.loads(a.calibration.read_text()) if a.calibration else None
    prepared = prepare_arm(plan, a.arm, a.seed, rows, meas['rows'], calibration)
    scope = review_scope(prepared, output=a.output)
    if a.command == 'scope':
        print(json.dumps({'scope_sha256': _sha(scope), 'schedule_sha256': prepared['realized']['schedule_sha256'],
                          'category_counts': prepared['realized']['category_counts'], 'plan_sha256': prepared['bindings']['plan_sha256']}))
        return
    from mri_rl.experiment_contract import _check_review, load_contract
    require_local = a.output.resolve().is_relative_to(ROOT / 'local_data')
    if not require_local:
        raise ValueError('Outputs must stay under ignored local_data')
    _check_review(review_verdict_path=a.review_verdict, expected_review_sha256=a.expected_review_sha256, scope=scope)
    from mri_rl.continuous_rl_pilot import cache_binding_for
    from mri_rl.mlx_continuous_gspo import backend_factory, is_loading_unit
    from mri_rl.paced_resources import PacedGate
    from types import SimpleNamespace
    binding = cache_binding_for(SimpleNamespace(cache_dir=a.cache_dir.resolve(), cache_certificate=a.cache_certificate.resolve(),
                                                cache_certificate_sha256=a.cache_certificate_sha256, model=a.model.resolve()))
    if binding['certificate_sha256'] != plan['common_bindings']['cache_certificate_sha256']:
        raise ValueError('Cache certificate differs from the compiled plan')
    contract = load_contract(a.contract); cache = {'dir': str(a.cache_dir.resolve())}
    gate = PacedGate(a.output.with_name(a.output.name + '_resource_events.jsonl'))
    def before_unit(name):
        gate.before_unit(name, loading=is_loading_unit(name, cache))
    out = train_arm(prepared, output=a.output, resume=a.resume, starting_adapter=a.adapter.resolve(),
                    backend_factory=backend_factory(model_path=a.model.resolve(), contract=contract, seed=a.seed,
                                                    before_unit=before_unit, cache=cache),
                    authorize=lambda ctx: before_unit('authorize_' + ctx['phase']))
    print(json.dumps({k: v for k, v in out.items() if k != 'boundary'}))


if __name__ == '__main__':
    main()
