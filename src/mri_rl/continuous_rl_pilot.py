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

MAX_UPDATES = 1  # Tier B diagnostic ceiling declared to preflight AND enforced by the driver
ROOT = Path(__file__).resolve().parents[2]
SOURCES = ('continuous_rl.py', 'mlx_continuous_gspo.py', 'continuous_rl_pilot.py',
           'optimizer_checkpoint.py', 'vero_mri_reward.py', 'paced_resources.py',
           'resource_supervisor.py', 'mlx_gspo_smoke.py', 'likelihood_loading.py')


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


def launch_configuration(*, output, seed, record_ids, group_size, mode, boundary_receipt_sha256=None):
    return {'runner': 'continuous_rl_pilot', 'mode': mode, 'seed': seed,
            'output': str(Path(output).resolve()), 'schedule_record_ids': list(record_ids),
            'profile': PROFILE['profile'], 'group_size': group_size,
            'max_optimizer_updates': MAX_UPDATES, 'boundary_receipt_sha256': boundary_receipt_sha256,
            'source_sha256': source_hashes()}


def boundary_receipt(output):
    """Digest of the latest committed boundary so a reload approval binds the actual updated state."""
    boundaries = sorted(Path(output).glob('boundary_*'))
    require(boundaries, 'No committed boundary to bind')
    return digest(boundaries[-1] / 'receipt.json')


def preflight(args, record_ids):
    contract = load_contract(args.contract)
    require_launch_preflight(
        contract_path=args.contract, train=args.train, dev=args.dev, model_path=args.model,
        dynamic_receipt_path=None, review_verdict_path=args.review_verdict,
        expected_system=contract.system_instruction,
        expected_max_tokens=contract.generation['max_tokens'], expected_reward_profile='direct_vero',
        expected_starting_adapter=args.adapter, execution_tier='B', phase='diagnostic',
        launch_budget={'optimizer_steps': MAX_UPDATES, 'max_groups': len(record_ids),
                       'rollouts_per_group': args.group_size, 'max_tokens': contract.generation['max_tokens']},
        expected_review_sha256=args.expected_review_sha256, expected_temperature=1.0,
        launch_configuration=launch_configuration(output=args.output, seed=args.seed,
            record_ids=record_ids, group_size=args.group_size, mode=args.command,
            boundary_receipt_sha256=args.approved_boundary))
    return contract


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=('train', 'reload'))
    for name in ('model', 'train', 'dev', 'contract', 'adapter', 'output', 'review-verdict'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--expected-review-sha256', required=True)
    p.add_argument('--record-ids', nargs='+', required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--group-size', type=int, default=2)
    args = p.parse_args()
    for name in ('model', 'train', 'dev', 'contract', 'adapter', 'output', 'review_verdict'):
        setattr(args, name, getattr(args, name).resolve())
    require(args.output.is_relative_to(ROOT / 'local_data'), 'Outputs must stay under ignored local_data')
    # The boundary digest is captured ONCE; the approved value is what must be loaded.
    args.approved_boundary = boundary_receipt(args.output) if args.command == 'reload' else None
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
            max_new_groups=min(len(schedule), 3), resume=args.resume,
            starting_adapter=args.adapter, config=config, max_optimizer_updates=MAX_UPDATES)
        print(json.dumps({k: v for k, v in result.items() if k != 'boundary'}))
        return
    binding = _binding(schedule, bindings, config)
    last = inspect_boundaries(args.output, binding)
    require(last is not None and last['optimizer_updates'] >= 1, 'Reload check needs a committed update')
    require(digest(last['path'] / 'receipt.json') == args.approved_boundary,
            'Latest boundary differs from the approved boundary; refusing to load a different state')
    adapter = last['path'] / 'adapter.safetensors'
    require(digest(adapter) != bindings['adapter_sha256'], 'Updated adapter equals starting adapter')
    backend = NativeGSPOBackend(adapter, config, model_path=args.model, contract=contract,
                                seed=args.seed, before_unit=before_unit)
    expected_image = binding['image_sha256'][schedule[0]['id']]
    require(digest(schedule[0]['images'][0]) == expected_image, 'Frozen image changed before reload generation')
    try:  # constructor already asserts exact loaded-vs-saved adapter tensor equality
        out = backend.greedy_generate(schedule[0], expected_image)
    finally:
        backend.close()
    require(digest(schedule[0]['images'][0]) == expected_image, 'Frozen image changed during reload generation')
    # Full text/token ids stay private so the answer can be scored later; stdout gets counts only.
    (args.output / 'reload_generation_PRIVATE.json').write_text(json.dumps(out, sort_keys=True))
    receipt = {'kind': 'continuous_rl_reload_v1', 'boundary_receipt_sha256': digest(last['path'] / 'receipt.json'),
               'starting_adapter_sha256': bindings['adapter_sha256'], 'reloaded_adapter_sha256': digest(adapter),
               'exact_reloaded_adapter_tensors': True, 'generation_token_count': len(out['token_ids']),
               'generation_sha256': digest(args.output / 'reload_generation_PRIVATE.json'),
               'prompt_sha256': out['prompt_sha256']}
    (args.output / 'reload_receipt_PRIVATE.json').write_text(json.dumps(receipt, sort_keys=True, indent=1))
    print(json.dumps({k: v for k, v in receipt.items() if k != 'prompt_sha256'}))


if __name__ == '__main__':
    main()
