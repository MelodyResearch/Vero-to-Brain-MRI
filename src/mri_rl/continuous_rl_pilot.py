"""Finite real-MRI continuous-GSPO pilot launcher (Tier B diagnostic, <=1 optimizer update).

Two subcommands share one reviewed scope: `train` runs continuous_rl.run through the native
Qwen backend; `reload` rebuilds the model from the committed boundary adapter and runs one
greedy generation (the backend constructor verifies exact adapter tensors). Nothing here
relaxes the reward, the group-activity rule or any resource gate. Outputs stay in local_data.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from mri_rl.continuous_rl import PROFILE, digest, inspect_boundaries, require, run, _binding
from mri_rl.experiment_contract import load_contract, require_launch_preflight, sha256_file
from mri_rl.paced_resources import PacedGate

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ('continuous_rl.py', 'mlx_continuous_gspo.py', 'continuous_rl_pilot.py',
           'optimizer_checkpoint.py', 'vero_mri_reward.py', 'paced_resources.py',
           'resource_supervisor.py', 'mlx_gspo_smoke.py')


def source_hashes():
    return {name: sha256_file(ROOT / 'src' / 'mri_rl' / name) for name in SOURCES}


def load_schedule(train, record_ids):
    """Ordered rows by explicit id; training-split eligibility is enforced again by the driver."""
    rows = {}
    for line in Path(train).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row['id']] = row
    require(len(set(record_ids)) == len(record_ids) and all(i in rows for i in record_ids),
            'Every requested schedule record must exist exactly once in train')
    return [rows[i] for i in record_ids]


def launch_configuration(*, output, seed, record_ids, group_size, mode):
    return {'runner': 'continuous_rl_pilot', 'mode': mode, 'seed': seed,
            'output': str(Path(output).resolve()), 'schedule_record_ids': list(record_ids),
            'profile': PROFILE['profile'], 'group_size': group_size,
            'source_sha256': source_hashes()}


def preflight(args, record_ids):
    contract = load_contract(args.contract)
    require_launch_preflight(
        contract_path=args.contract, train=args.train, dev=args.dev, model_path=args.model,
        dynamic_receipt_path=None, review_verdict_path=args.review_verdict,
        expected_system=contract.system_instruction,
        expected_max_tokens=contract.generation['max_tokens'], expected_reward_profile='direct_vero',
        expected_starting_adapter=args.adapter, execution_tier='B', phase='diagnostic',
        launch_budget={'optimizer_steps': 1, 'max_groups': len(record_ids),
                       'rollouts_per_group': args.group_size, 'max_tokens': contract.generation['max_tokens']},
        expected_review_sha256=args.expected_review_sha256, expected_temperature=1.0,
        launch_configuration=launch_configuration(output=args.output, seed=args.seed,
            record_ids=record_ids, group_size=args.group_size, mode=args.command))
    return contract


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=('train', 'reload'))
    for name in ('model', 'train', 'dev', 'contract', 'adapter', 'output', 'review-verdict'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--expected-review-sha256', required=True)
    p.add_argument('--record-ids', nargs='+', required=True)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--group-size', type=int, default=2)
    args = p.parse_args()
    for name in ('model', 'train', 'dev', 'contract', 'adapter', 'output', 'review_verdict'):
        setattr(args, name, getattr(args, name).resolve())
    require(args.output.is_relative_to(ROOT / 'local_data'), 'Outputs must stay under ignored local_data')
    contract = preflight(args, args.record_ids)
    schedule = load_schedule(args.train, args.record_ids)
    config = {**PROFILE, 'group_size': args.group_size, 'max_tokens': contract.generation['max_tokens']}
    gate = PacedGate(args.output.with_name(args.output.name + '_resource_events.jsonl'))
    def before_unit(name):
        gate.before_unit(name, loading=name in {'native_model_load', 'quantized_model_ready'})
    from mri_rl.mlx_continuous_gspo import backend_factory, NativeGSPOBackend
    bindings = {'adapter_sha256': sha256_file(args.adapter), 'contract_id': contract.identity(),
                'seed': args.seed, 'source_sha256': source_hashes(), 'model_dir': str(args.model)}
    if args.command == 'train':
        result = run(output=args.output, schedule=schedule, bindings=bindings,
            backend_factory=backend_factory(model_path=args.model, contract=contract, seed=args.seed,
                                            before_unit=before_unit),
            authorize=lambda context: before_unit('authorize_' + context['phase']),
            max_new_groups=len(schedule), starting_adapter=args.adapter, config=config)
        print(json.dumps({k: v for k, v in result.items() if k != 'boundary'}))
        return
    binding = _binding(schedule, bindings, config)
    last = inspect_boundaries(args.output, binding)
    require(last is not None and last['optimizer_updates'] >= 1, 'Reload check needs a committed update')
    adapter = last['path'] / 'adapter.safetensors'
    require(digest(adapter) != bindings['adapter_sha256'], 'Updated adapter equals starting adapter')
    backend = NativeGSPOBackend(adapter, config, model_path=args.model, contract=contract,
                                seed=args.seed, before_unit=before_unit)
    try:  # constructor already asserts exact loaded-vs-saved adapter tensor equality
        out = backend.greedy_generate(schedule[0])
    finally:
        backend.close()
    receipt = {'kind': 'continuous_rl_reload_v1', 'boundary_receipt_sha256': digest(last['path'] / 'receipt.json'),
               'starting_adapter_sha256': bindings['adapter_sha256'], 'reloaded_adapter_sha256': digest(adapter),
               'exact_reloaded_adapter_tensors': True, 'generation_token_count': len(out['token_ids']),
               'prompt_sha256': out['prompt_sha256']}
    (args.output / 'reload_receipt_PRIVATE.json').write_text(json.dumps(receipt, sort_keys=True, indent=1))
    print(json.dumps({k: v for k, v in receipt.items() if k != 'prompt_sha256'}))


if __name__ == '__main__':
    main()
