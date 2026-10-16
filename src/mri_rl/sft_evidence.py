"""Validate historical supervised evidence without pretending it is RL evidence."""
from __future__ import annotations
import json
import math
from pathlib import Path
from mri_rl.experiment_contract import sha256_file, canonical_hash


def bound_file(binding: dict, private_root: Path) -> Path:
    path=Path(binding.get('path','')).resolve()
    if not path.is_relative_to(private_root.resolve()):
        raise ValueError('Supervised evidence must remain private')
    if not path.is_file() or sha256_file(path)!=binding.get('sha256'):
        raise ValueError('Supervised evidence artifact missing or changed')
    return path


def verify_history(source: dict, schedule: list[list[dict]]) -> None:
    expected=[(i,row['ability'],row['extra_info']['patient_key'])
              for i,step in enumerate(schedule) for row in step]
    observed=[(r.get('step'),r.get('category'),r.get('patient_key'))
              for r in source.get('examples',[])]
    if source.get('optimizer_steps')!=len(schedule) or expected!=observed:
        raise ValueError('Historical supervised schedule mismatch')
    losses=source.get('loss_terms',[]);grads=source.get('gradient_l2_by_step',[])
    if (len(losses)!=len(expected) or len(grads)!=len(schedule) or
        any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in losses+grads) or
        any(v<=0 for v in grads)):
        raise ValueError('Historical losses/gradients lack finite optimization evidence')


def verify_prefix_binding(inventory: list[dict], configuration: dict, steps: int) -> None:
    prefix=inventory[:4*steps]
    if (configuration.get('executed_prefix_examples')!=len(prefix) or
        configuration.get('executed_prefix_sha256')!=canonical_hash(prefix)):
        raise ValueError('Executed SFT prefix differs from reviewed bounded schedule')


def verify_supervised_evidence(dynamic: dict, *, contract, train: Path,
                               starting_adapter: Path, configuration: dict,
                               budget: dict, private_root: Path) -> None:
    from mri_rl.mlx_sft_warmstart import _schedule_train_rows, schedule_inventory
    from mri_rl.ds007045_internal_dev_eval import score_outputs
    if (dynamic.get('kind')!='supervised_control_v1' or
        dynamic.get('contract_id')!=contract.identity() or
        configuration.get('runner')!='mlx_sft_warmstart' or
        configuration.get('schedule_policy')!='balanced_sequence_v2' or
        configuration.get('seed')!=7 or budget.get('optimizer_steps') not in (1,2) or
        budget.get('max_groups')!=budget.get('optimizer_steps') or budget.get('rollouts_per_group')!=4):
        raise ValueError('Supervised evidence is scoped to the matched balanced SFT28 control')
    if dynamic.get('training_records_sha256')!=sha256_file(train):
        raise ValueError('Supervised training manifest mismatch')
    paths={key:bound_file(dynamic[key],private_root) for key in
           ('historical_receipt','before_adapter','after_adapter','reload_outputs',
            'reload_manifest','reload_gold')}
    before,after=sha256_file(paths['before_adapter']),sha256_file(paths['after_adapter'])
    source=json.loads(paths['historical_receipt'].read_text())
    if (source.get('status')!='SFT_ENGINEERING_CONTROL' or before==after or
        before!=sha256_file(starting_adapter) or source.get('before_adapter_sha256')!=before or
        source.get('after_adapter_sha256')!=after or source.get('learning_rate')!=1e-5 or
        source.get('distinct_patients_exposed')!=28 or source.get('heldout_test_used') is not False):
        raise ValueError('Historical supervised adapter or optimization mismatch')
    legacy=_schedule_train_rows(train,28,7,'legacy_lexical_v1')
    balanced=_schedule_train_rows(train,28,7,'balanced_sequence_v2')
    verify_history(source,legacy)
    if canonical_hash(schedule_inventory(balanced))!=configuration.get('schedule_sha256'):
        raise ValueError('Frozen selected supervised image schedule changed')
    verify_prefix_binding(schedule_inventory(balanced),configuration,budget['optimizer_steps'])
    for old,new in zip(legacy,balanced):
        if old[1:]!=new[1:] or old[0]['extra_info']['patient_key']!=new[0]['extra_info']['patient_key']:
            raise ValueError('Supervised comparison changes more than sequence examples')
    # Recompute the actual historical evaluation including prompt/image/label linkage.
    score_outputs(paths['reload_manifest'],paths['reload_gold'],paths['reload_outputs'])
    outputs=[json.loads(l) for l in paths['reload_outputs'].read_text().splitlines()]
    if len(outputs)!=54 or any(r.get('adapter_checkpoint_sha256')!=after or
        r.get('adapter_verified_parameter_tensors')!=504 or
        r.get('model_config_sha256')!=contract.model_config_sha256 or
        r.get('model_template_sha256')!=contract.chat_template_sha256 or
        r.get('max_tokens')!=256 or r.get('temperature')!=0 or
        not any(m.get('role')=='system' and m.get('content')==contract.system_instruction
                for m in r.get('prompt',[])) for r in outputs):
        raise ValueError('Historical supervised checkpoint reload evidence mismatch')
