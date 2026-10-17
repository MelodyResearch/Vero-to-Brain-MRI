"""Codex SFT entry/glue; Claude's SFT objective and optimizer remain unchanged.

A one-step diagnostic and later continuation use the SAME full frozen schedule,
plan and run binding. Stage limits and proof receipts are approval-scope fields,
never new training trajectories. This module imports no native model runtime.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata as md
import json
from pathlib import Path

from mri_rl import tier_c_entry as common, tier_c_sft as sft
from mri_rl.continuous_rl import canonical, digest, inspect_boundaries, require
from mri_rl.experiment_contract import _check_review, load_contract
from mri_rl.full_arm_protocol import compile_plan
from mri_rl.two_task_pool import canonical_sha, read_rows, validate_model_boundary
from mri_rl.two_task_training import TokenLedger, input_tokens_by_source

ROOT = Path(__file__).resolve().parents[2]
SOURCES = tuple(dict.fromkeys((*common.CRITICAL_SOURCES, 'tier_c_sft.py', 'tier_c_sft_entry.py')))


def sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def source_hashes():
    return {name: digest(ROOT / 'src/mri_rl' / name) for name in SOURCES}


def _rows(train, measurements):
    require(digest(train) == measurements['manifest_sha256'], 'Training manifest differs from native measurements')
    rows = [x['row'] for x in read_rows(train)]
    for row in rows:
        validate_model_boundary(row, 'train')
    measured = {x['record_sha256']: x for x in measurements['rows']}
    require(len(measured) == len(measurements['rows']) == len(rows) and
            set(measured) == {canonical_sha(r) for r in rows}, 'Native measurements must cover exactly the frozen source rows')
    require(all(measured[canonical_sha(r)]['source'] == r['data_source'] and
                measured[canonical_sha(r)]['route'] == r['ability'] for r in rows), 'Measurement route/source mismatch')
    return rows


def _targets(rows, tokenizer):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(tokenizer))
    values = {r['id']: {'text': sft.sft_target(r), 'ids': tok.encode(sft.sft_target(r), add_special_tokens=False).ids} for r in rows}
    require(all(0 < len(v['ids']) <= sft.TARGET_CAP for v in values.values()), 'Target token cap exceeded')
    return values


def compile_sft_plan(train, measurements, tokenizer, adapter, certificate_sha256):
    rows = _rows(Path(train), measurements)
    cb = common.common_bindings(measurements, adapter, certificate_sha256)
    plan = compile_plan(measurements, cb)
    plan['schema'] = 'same-C-data-SFT-plan-v1'
    plan['adapter_config_sha256'] = digest(Path(adapter).parent / 'adapter_config.json')
    targets = _targets(rows, tokenizer); counts = {k: len(v['ids']) for k, v in targets.items()}
    lookup = input_tokens_by_source(measurements['rows'], rows)
    schedules, costs = {}, {}
    for seed in plan['seeds']:
        schedules[str(seed)], costs[str(seed)] = sft.build_steps(rows, seed, plan['budget']['text_visual_token_cap'], lookup, counts)
    plan['sft'] = {'sampling': 'full-pool permutations, natural-proportional', 'examples_per_step': sft.EXAMPLES_PER_STEP,
                   'target_cap': sft.TARGET_CAP, 'tokenizer_sha256': digest(tokenizer), 'targets': targets,
                   'target_ids': {k: v['ids'] for k, v in targets.items()},
                   'image_sha256': {r['id']: digest(r['images'][0]) for r in rows},
                   'schedules': schedules, 'schedule_costs': costs, 'exact_FLOP_match_claimed': False}
    plan['source_sha256'] = source_hashes()
    return plan


def prepare_sft(plan, train, measurements, tokenizer, seed):
    require(plan.get('schema') == 'same-C-data-SFT-plan-v1' and seed in plan['seeds'], 'Unplanned SFT seed/schema')
    require(plan['source_sha256'] == source_hashes(), 'SFT/core source changed after compilation')
    require(common.measurements_sha256(measurements) == plan['measurements_sha256'], 'Measurements differ from compiled plan')
    require(plan['budget'] == compile_plan(measurements, plan['common_bindings'])['budget'], 'SFT budget differs from frozen measured compiler')
    rows = _rows(Path(train), measurements)
    require(digest(train) == plan['manifest_sha256'] and digest(tokenizer) == plan['sft']['tokenizer_sha256'], 'Manifest/tokenizer changed')
    targets = _targets(rows, tokenizer)
    require(targets == plan['sft']['targets'] and {k:v['ids'] for k,v in targets.items()} == plan['sft']['target_ids'], 'SFT target text/token IDs changed')
    images = {r['id']: digest(r['images'][0]) for r in rows}
    require(images == plan['sft']['image_sha256'], 'Approved training image bytes changed')
    lookup = input_tokens_by_source(measurements['rows'], rows); counts = {k:len(v['ids']) for k,v in targets.items()}
    indices, cost = sft.build_steps(rows, seed, plan['budget']['text_visual_token_cap'], lookup, counts)
    require(indices == plan['sft']['schedules'][str(seed)] and cost == plan['sft']['schedule_costs'][str(seed)], 'Frozen full SFT schedule differs')
    steps = sft.step_rows(rows, indices)
    bindings = {'sft_plan_sha256': sha(plan), 'seed': seed, 'schedule_sha256': sha(steps),
                'manifest_sha256': plan['manifest_sha256'], 'common': plan['common_bindings'],
                'source_sha256': source_hashes(), 'target_tokens_sha256': sha(targets)}
    return {'plan':plan, 'steps':steps, 'bindings':bindings, 'input_tokens':lookup, 'target_tokens':counts,
            'ledger_plan':{'group_size':4, 'max_tokens':256, 'budget':plan['budget']}}


def _output_state(output, resume):
    output = Path(output)
    if not resume:
        require(not output.exists(), 'Fresh SFT output already exists')
        return None
    require(output.is_dir(), 'Resume output is missing')
    receipts = sorted(output.glob('boundary_*/receipt.json'))
    require(receipts, 'No complete SFT boundary to resume')
    return {'last_receipt_sha256':digest(receipts[-1]),
            'pending_files_sha256':{str(p.relative_to(output)):digest(p) for p in sorted(output.glob('.pending-*/*')) if p.is_file()}}


def review_scope(prepared, *, output, runtime, resume, max_new_steps, tier):
    require(tier in {'B','C'}, 'SFT launch tier must be B or C')
    require(max_new_steps is None or (type(max_new_steps) is int and max_new_steps > 0), 'Invalid SFT stage bound')
    if tier == 'B':
        require(max_new_steps == 1 and not resume, 'Tier B SFT is one fresh diagnostic step')
    steps = prepared['steps']; state = _output_state(output, resume)
    cursor = 0
    if state:
        last = sorted(Path(output).glob('boundary_*/receipt.json'))[-1]
        cursor = json.loads(last.read_text())['completed_groups']
    require(type(cursor) is int and 0 <= cursor <= len(steps), 'Invalid SFT resume cursor')
    limit = len(steps)-cursor if max_new_steps is None else min(max_new_steps,len(steps)-cursor)
    cost = sum(sft.step_cost(x,prepared['input_tokens'],prepared['target_tokens']) for x in steps[cursor:cursor+limit])
    return {'kind':'same_C_SFT_stage_v1', 'execution_tier':tier, 'phase':'diagnostic' if tier=='B' else 'continuation',
            'sft_plan_sha256':prepared['bindings']['sft_plan_sha256'], 'seed':prepared['bindings']['seed'],
            'full_schedule_sha256':prepared['bindings']['schedule_sha256'], 'full_schedule_steps':len(steps),
            'full_processed_token_cap':prepared['plan']['budget']['text_visual_token_cap'],
            'max_new_steps':max_new_steps, 'resume':resume, 'starting_output_state':state,
            'weight_updates_at_most':limit, 'stage_known_input_target_tokens_at_most':cost,
            'examples_per_step':4, 'target_cap':64, 'output':str(Path(output).resolve()),
            'runtime':runtime, 'source_sha256':source_hashes()}


def _checkpoint_proof(boundary_path, restore_path, plan, *, sft_proof=False):
    require(boundary_path is not None and restore_path is not None, 'Actual update and native restore proofs required')
    path = Path(boundary_path); r = json.loads(path.read_text()); restore = json.loads(Path(restore_path).read_text())
    last = inspect_boundaries(path.parent.parent, r['binding'])
    require(last['path'] == path.parent and r['updated'] and r['optimizer_updates'] >= 1, 'Update proof is not the latest committed update')
    experiment = r['binding']['experiment']
    if sft_proof:
        require(r['binding']['configuration']['profile'] == 'tier_c_sft_v1' and
                experiment['sft_plan_sha256'] == sha(plan), 'SFT update proof belongs to another plan')
    else:
        require(experiment['adapter_sha256'] == plan['common_bindings']['starting_adapter_sha256'] and
                experiment['derived_cache_certificate_sha256'] == plan['common_bindings']['cache_certificate_sha256'],
                'GSPO dynamic proof has a different common initializer/cache')
    norm = r.get('gradient_l2', r.get('update_evidence',{}).get('gradient_l2'))
    require(isinstance(norm,(int,float)) and __import__('math').isfinite(norm) and norm > 0, 'No finite positive update proof')
    require(restore.get('status') == 'NATIVE_RESTORE_AND_GREEDY_COMPLETED' and
            restore.get('boundary_receipt_sha256') == digest(path) and restore.get('new_weight_updates') == 0 and
            restore.get('restored_optimizer_step') == r['optimizer_updates'] and
            all(restore.get(k) is True for k in ('restored_adam_exact','restored_python_and_MLX_RNG_exact','read_only_weights_and_adam_unchanged')),
            'Native checkpoint restore proof is incomplete or mismatched')
    require(restore['derived_cache_certificate_sha256'] == plan['common_bindings']['cache_certificate_sha256'], 'Restore cache differs')
    require(all(restore.get('source_sha256',{}).get(k) == digest(ROOT/'src/mri_rl'/k)
                for k in ('mlx_continuous_gspo.py','optimizer_checkpoint.py')),
            'Shared native restore/optimizer I/O changed since the executed restore proof')
    return {'boundary_receipt_sha256':digest(path), 'restore_receipt_sha256':digest(restore_path),
            'updated_adapter_sha256':r['files_sha256']['adapter.safetensors']}


def sft_update_proof(boundary_path, plan):
    """Bind the actual SFT update; shared native restore is independently proven by the G8 receipt.

    This is checkpoint validation, never a claim that another native SFT reload ran.
    The continuation itself restores full native Adam/RNG before further work.
    """
    require(boundary_path is not None, 'Actual first SFT update receipt required')
    path=Path(boundary_path);r=json.loads(path.read_text())
    inspect_boundaries(path.parent.parent,r['binding'])  # complete immutable chain, including this historical boundary
    experiment=r['binding']['experiment']
    require(r['updated'] and r['optimizer_updates']>=1 and
            r['binding']['configuration']['profile']=='tier_c_sft_v1' and
            experiment['sft_plan_sha256']==sha(plan) and experiment['source_sha256']==source_hashes(),
            'SFT update proof belongs to a different plan/source')
    norm=r.get('gradient_l2');losses=r.get('loss_terms',[])
    require(isinstance(norm,(int,float)) and __import__('math').isfinite(norm) and norm>0 and len(losses)==4 and
            all(__import__('math').isfinite(x) for x in losses), 'No finite actual SFT gradient/loss proof')
    require(r['files_sha256']['adapter.safetensors']!=plan['common_bindings']['starting_adapter_sha256'] and
            all(r['state_validation'].get(k) is True for k in ('parameters_finite','optimizer_finite','second_moments_nonnegative')),
            'SFT updated weights/state are not verified')
    return {'boundary_receipt_sha256':digest(path),'updated_adapter_sha256':r['files_sha256']['adapter.safetensors'],
            'optimizer_updates':r['optimizer_updates'],'native_SFT_reload':'NOT_SEPARATELY_RUN',
            'continuation_requires_shared_native_restore':True}


def runtime_bindings(a, plan, measurements, *, tier):
    require(digest(a.train) == plan['manifest_sha256'], 'Runtime training manifest differs')
    runtime = common.verify_runtime_artifacts(plan, measurements, a.adapter, a.contract, load_contract, a.measurement_contract)
    require(all(runtime['installed_versions'].values()), 'Native package metadata unavailable')
    artifacts = measurements['processor_artifacts']; current = {}
    for key,value in artifacts.items():
        if key.startswith('runtime_'):
            current[key] = md.version({'runtime_transformers':'transformers','runtime_mlx_vlm':'mlx-vlm'}[key])
        elif key == 'native_processor_patch_sha256':
            current[key] = digest(md.distribution('mlx-vlm').locate_file('mlx_vlm/models/qwen3_vl/processing_qwen3_vl.py'))
        else:
            require(Path(key).name == key, 'Processor artifact path must be a model-relative filename')
            current[key] = digest(a.model / key)
    require(current == artifacts and canonical_sha(current) == measurements['processor_sha256'], 'Measured processor/tokenizer/runtime changed')
    require(a.tokenizer.resolve() == (a.model/'tokenizer.json').resolve(), 'Target tokenizer is not the native model tokenizer')
    from mri_rl.continuous_rl_pilot import cache_binding_for
    cache = cache_binding_for(a)
    require(cache and cache['certificate_sha256'] == plan['common_bindings']['cache_certificate_sha256'], 'Compiled cache differs')
    runtime.update({'derived_cache':cache, 'model_path':str(a.model), 'starting_adapter_path':str(a.adapter),
                    'processor_artifacts':current, 'train_manifest_sha256':digest(a.train),
                    'target_tokenizer_sha256':digest(a.tokenizer),
                    'GSPO_update_and_restore':_checkpoint_proof(a.gspo_boundary,a.gspo_restore,plan)})
    if tier == 'C':
        runtime['SFT_update_receipt'] = sft_update_proof(a.sft_boundary,plan)
    from mri_rl.paced_resources import PACED_POLICY
    runtime['resource_policy'] = dict(PACED_POLICY)
    return runtime


def train_prepared(prepared, *, output, adapter, backend_factory, authorize, resume=False, max_new_steps=None):
    return sft.run_sft(output=output,steps=prepared['steps'],bindings=prepared['bindings'],
        backend_factory=backend_factory,authorize=authorize,ledger=TokenLedger(prepared['ledger_plan']),
        input_tokens=prepared['input_tokens'],target_tokens=prepared['target_tokens'],
        resume=resume,starting_adapter=adapter,max_new_steps=max_new_steps,
        expected_target_ids=prepared['plan']['sft']['target_ids'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('compile','scope','train'))
    for name in ('measurements','train','adapter','tokenizer','plan'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--cache-certificate-sha256',required=True)
    for name in ('model','contract','measurement-contract','cache-dir','cache-certificate','output','scope-output',
                 'review-verdict','gspo-boundary','gspo-restore','sft-boundary'):
        p.add_argument('--'+name,type=Path)
    p.add_argument('--seed',type=int);p.add_argument('--execution-tier',choices=('B','C'))
    p.add_argument('--max-new-steps',type=int);p.add_argument('--resume',action='store_true')
    p.add_argument('--expected-review-sha256')
    a = p.parse_args()
    for name,value in vars(a).items():
        if isinstance(value,Path):setattr(a,name,value.resolve())
    measurements = json.loads(a.measurements.read_text())
    require(a.plan.is_relative_to(ROOT/'local_data'), 'Compiled SFT plan must stay private')
    if a.command == 'compile':
        plan = compile_sft_plan(a.train,measurements,a.tokenizer,a.adapter,a.cache_certificate_sha256)
        a.plan.parent.mkdir(parents=True,exist_ok=True)
        with a.plan.open('x') as f:json.dump(plan,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
        print(json.dumps({'status':'COMPILED_NOT_AUTHORIZED','plan_sha256':sha(plan),'seeds':plan['seeds'],
                          'steps_per_seed':{k:len(v) for k,v in plan['sft']['schedules'].items()},'cap':plan['budget']['text_visual_token_cap']}));return
    require(a.execution_tier is not None and a.output is not None and a.output.is_relative_to(ROOT/'local_data'), 'Exact private output and tier required')
    require(all(getattr(a,n) is not None for n in ('model','contract','cache_dir','cache_certificate')), 'Complete native runtime paths required')
    plan = json.loads(a.plan.read_text()); prepared = prepare_sft(plan,a.train,measurements,a.tokenizer,a.seed)
    runtime = runtime_bindings(a,plan,measurements,tier=a.execution_tier)
    scope = review_scope(prepared,output=a.output,runtime=runtime,resume=a.resume,max_new_steps=a.max_new_steps,tier=a.execution_tier)
    if a.command == 'scope':
        require(a.scope_output is not None and a.scope_output.is_relative_to(ROOT/'local_data'), 'Private scope output required')
        with a.scope_output.open('x') as f:json.dump(scope,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
        print(json.dumps({'status':'SCOPE_NOT_AUTHORIZED','scope_sha256':sha(scope),'full_steps':len(prepared['steps']),
                          'max_new_steps':a.max_new_steps,'stage_cost':scope['stage_known_input_target_tokens_at_most']}));return
    _check_review(review_verdict_path=a.review_verdict,expected_review_sha256=a.expected_review_sha256,scope=scope)
    from mri_rl.paced_resources import PacedGate
    from mri_rl.mlx_continuous_gspo import is_loading_unit
    cache = {'dir':str(a.cache_dir)};gate = PacedGate(a.output.with_name(a.output.name+'_resource_events.jsonl'))
    def before(name):return gate.before_unit(name,loading=is_loading_unit(name,cache))
    cls = sft.native_backend_class();contract = load_contract(a.contract)
    result = train_prepared(prepared,output=a.output,adapter=a.adapter,resume=a.resume,max_new_steps=a.max_new_steps,
        backend_factory=lambda adapter,config:cls(adapter,config,model_path=a.model,contract=contract,seed=a.seed,before_unit=before,cache=cache),
        authorize=lambda ctx:before('authorize_'+ctx['phase']))
    print(json.dumps(result,sort_keys=True))


if __name__ == '__main__':
    main()
