"""Fail-closed entry for VERO_TO_MRI_ACCURACY_RECOVERY_V1 (authentic Adam52/boundary114 parent). CPU `prepare` imports no MLX.

`prepare` reads metadata/receipts only: no model, no weight/Adam payload arrays, no images unless `--verify-images`. Inputs that
are not yet bound (runtime spec, parent directory, certified cache, runtime-source closure, expected parent arm) are reported as
BLOCKERS and no launch scope is emitted; nothing is fabricated and no model PASS is ever printed. Every actual output and private payload lives ONLY under the canonical project root's local_data/accuracy_recovery_v1 (git common dir), never the worktree.
`run` requires the exact launch
scope, a fresh human authorization and an independent review by a different agent kind, digest-bound on the command line; all gates
precede any native import. There is no qualification phase, no mode and no approval flag. IMPLEMENTED_CPU_NOT_NATIVE_QUALIFIED; NOT_RUN.
Inputs forbidden by the researcher (old final100 outputs, new16 images/labels/outputs) are refused by name.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
from mri_rl import accuracy_recovery_v1 as core
from mri_rl import accuracy_recovery_driver_v1 as driver
from mri_rl import paired_z045_continuation_entry_v1 as inherited

ROOT=Path(__file__).resolve().parents[2]
require=core.require
IMPLEMENTED_BY='claude_code'
FORBIDDEN_TOKENS=('final100','new16','holdout16','HOLDOUT16')
RUNTIME_KEYS=('parent_checkpoint','parent_receipt_sha256','calibration','cache_dir','cache_certificate','cache_certificate_sha256','model','contract','global_lock','runtime_source_sha256')
PER_INVOCATION_KEYS=('output','resume')
NOT_COMPARED_PROFILE_KEYS=('group_size','max_tokens','active_group_rule')


def _json(path):return json.loads(Path(path).read_text())


def _strings(value):
    if isinstance(value,dict):
        for k,v in value.items():yield str(k);yield from _strings(v)
    elif isinstance(value,(list,tuple)):
        for v in value:yield from _strings(v)
    elif isinstance(value,str):yield value


def refuse_forbidden(spec):
    for text in _strings(spec):
        require(not any(token in text for token in FORBIDDEN_TOKENS),'Forbidden input (old final100 outputs / new16 images, labels, outputs) is not part of this experiment: '+text[:80])


def immutable_spec_binding(spec):
    """Scientific/runtime inputs only; output path and resume flag are per-invocation metadata bound by the launch scope digest."""
    return inherited.scope_digest({k:v for k,v in spec.items() if k not in PER_INVOCATION_KEYS})


def pending_inputs(spec):
    """Bound-input-pending report from metadata presence only. Returns (blockers, runtime or None)."""
    blockers=[];runtime=None;path=spec.get('runtime_spec')
    if not path or not Path(path).is_file():return ['RUNTIME_SPEC_NOT_PRESENT'],None
    for key,expected in spec.get('file_sha256',{}).items():
        require(Path(spec[key]).is_file() and inherited.digest(spec[key])==expected,'Frozen specification input changed: '+key)
    runtime=_json(path)
    missing=[k for k in RUNTIME_KEYS if k not in runtime]
    if missing:blockers.append('RUNTIME_KEYS_NOT_BOUND:'+','.join(missing))
    if not missing:
        for key,expected in runtime.get('file_sha256',{}).items():require(Path(runtime[key]).is_file() and inherited.digest(runtime[key])==expected,'Frozen runtime input changed: '+key)
        if not Path(runtime['parent_checkpoint']).is_dir():blockers.append('PARENT_CHECKPOINT_NOT_PRESENT')
        if not runtime['runtime_source_sha256']:blockers.append('RUNTIME_SOURCE_CLOSURE_NOT_BOUND')
        if not (Path(runtime['cache_certificate']).is_file() and Path(runtime['cache_dir']).is_dir() and Path(runtime['model']).exists()):blockers.append('CERTIFIED_BASE_CACHE_OR_MODEL_NOT_PRESENT')
    for key in ('schedule','expected_image_sha256','output'):
        if key not in spec:blockers.append(f'SPEC_KEY_NOT_BOUND:{key}')
    return blockers,runtime


def prepare(spec,*,verify_images=False,start=None):
    """Bind exact source/input/scope. Never loads a model; never reads weights/Adam payload arrays."""
    refuse_forbidden(spec)
    blockers,runtime=pending_inputs(spec)
    lengths=spec.get('prompt_lengths',{'F':213,'T':213});require(lengths=={'F':213,'T':213},'Fixed historical sequence prompt cost 213 required')
    base={'schema':'accuracy-recovery-preparation-v1','implementation_status':core.STATUS,'experiment':core.EXPERIMENT,'execution_subtype':core.EXECUTION_SUBTYPE,
          'recipe_label':core.RECIPE_LABEL,'scope_kind':'PREPARATION_ONLY_NOT_LAUNCH_SCOPE','training_authorized':False,'native_approved':False,'model_loads_this_preparation':0,
          'optimizer_updates_this_preparation':0,'model_PASS':False,'frozen_parent':{k:core.FROZEN_PARENT[k] for k in ('receipt_sha256','adapter_sha256','groups','updates','boundary')},
          'budget':core.derive_budget(lengths),'implemented_by':IMPLEMENTED_BY,'blockers':list(blockers),
          'required_next_steps':['Codex binds real private runtime/schedule/expected image hashes/expected parent arm in the private spec','independent Codex review of exact sources/inputs/scope',
                                 'genuine fresh human authorization of the emitted scope digest for action '+core.ACTION]}
    if blockers:return base
    parent_path=runtime['parent_checkpoint'];anchor=core.adam52_parent_anchor(parent_path,runtime['parent_receipt_sha256'])
    schedule=_json(spec['schedule']);schedule=schedule['schedule'] if isinstance(schedule,dict) else schedule
    rows=driver._modality_rows(schedule)
    from mri_rl.two_task_training import validate_training_row
    for row in schedule:validate_training_row(row)
    expected=spec['expected_image_sha256']
    require(set(expected)=={row['id'] for row in schedule} and all(core.sha(v) for v in expected.values()),'Exact frozen per-row image SHA required')
    if verify_images:inherited.verify_images(schedule,expected)
    calibration=_json(runtime['calibration']);original=anchor['original_binding']['experiment']
    require(spec.get('expected_parent_arm',core.PARENT_ARM)==core.PARENT_ARM,'Spec parent arm differs from the verified original P_calibrated arm')
    require(original.get('arm')==core.PARENT_ARM and original.get('seed')==core.PARENT_SEED and
            inherited.scope_digest(calibration)==original.get('calibration_sha256')==core.PARENT_CALIBRATION_SHA256,'Original parent arm/seed/task-routed calibration differs from the verified original metadata')
    from mri_rl import paired_z045_continuation_driver_v1 as old
    parent_configuration=anchor['original_binding']['configuration']
    configuration={**old.PROFILE,'group_size':core.GROUP_SIZE,'max_tokens':core.MAX_TOKENS}
    require(parent_configuration.get('active_group_rule')==core.PARENT_ACTIVE_GROUP_RULE,'Original parent active group rule differs from the verified all-full-format rule')
    require(all(parent_configuration.get(k)==v for k,v in configuration.items() if k not in NOT_COMPARED_PROFILE_KEYS) and
            parent_configuration.get('group_size')==core.GROUP_SIZE and parent_configuration.get('max_tokens')==core.MAX_TOKENS,'Inherited optimizer/reward/LR/LoRA/K4 configuration changed')
    sources=inherited.source_closure(runtime)
    for name in core.OWN_SOURCES:sources[str(ROOT/name)]=inherited.digest(ROOT/name)
    output=core.require_canonical_output(spec['output'],start=start)
    input_sha=core.identity({'schedule_sha256':inherited.scope_digest(schedule),'expected_image_sha256':expected,'parent_receipt_sha256':anchor['receipt_sha256'],'parent_adapter_sha256':anchor['adapter_sha256'],
        'calibration_sha256':original['calibration_sha256'],'immutable_spec_sha256':immutable_spec_binding(spec),'runtime_spec_sha256':inherited.digest(spec['runtime_spec']),'recipe':core.RECIPE_LABEL})
    return {**base,'parent':anchor,'parent_original_active_group_rule':parent_configuration.get('active_group_rule'),
        'stage_change_vs_parent':{'parent_rule':core.PARENT_ACTIVE_GROUP_RULE,'both_arms_rule':'narrow informative gate (accuracy varies in the valid-format subset, equal overlong, nonzero total)',
            'interpretation':'Explicit stage change versus the parent, identical in A and B, so it is not an A/B confound; reward/LR/LoRA/optimizer profile are unchanged.'},'schedule_PRIVATE':schedule,'expected_image_sha256':expected,
        'configuration':configuration,'reward_policy_sha256':original['calibration_sha256'],'source_sha256':sources,'input_sha256':input_sha,
        'derived_cache':inherited.cache_metadata(runtime)['expected_binding'],'output':str(output),
        'invocation':{'output':str(output),'resume':bool(spec.get('resume')),'binding':'PER-INVOCATION metadata: bound by the launch scope digest, EXCLUDED from the immutable input_sha256'},
        'arm_definition':{'A':'F0F1F2F3 | T0T1T2T3','B':'F0F1T0T1 | F2F3T2T3','weight':'1/8 per gate-admitted response','gate':'explicit LOCAL narrow informative group gate, identical in both arms (not complete official Vero)',
                          'updates':'at most one actual Adam update per arm from the identical restored Adam52 parent; no resample; inactive arm = 0 updates and an explicit None gradient'},
        'limitations':['Native mixed-context F32 rollout/scorer/backward NOT measured; CPU fixtures prove routing, accounting, archives and transactions only.',
                       'LOCAL Vero-style recipe: the parent chain used a different (older) group rule; the explicit narrow gate here is shared by both arms and is not the complete official Vero.',
                       'Typed receipts do not authenticate real model outputs; the combined run and independent review do. No accuracy or training-gain claim.']}


def launch_scope(prepared):
    require(not prepared.get('blockers'),'Launch scope is unavailable while bound inputs are pending: '+','.join(prepared.get('blockers',[])))
    return {'invocation':prepared['invocation'],'scope_kind':core.SCOPE_KIND,'experiment':core.EXPERIMENT,'execution_subtype':core.EXECUTION_SUBTYPE,'numerical_policy':core.POLICY,
            'recipe_label':core.RECIPE_LABEL,'implemented_by':IMPLEMENTED_BY,'budget':prepared['budget'],'source_sha256':prepared['source_sha256'],'input_sha256':prepared['input_sha256'],
            'parent':{'receipt_sha256':prepared['parent']['receipt_sha256'],'adapter_sha256':prepared['parent']['adapter_sha256']},
            'preparation_sha256':core.identity(prepared),'output':prepared['output']}


def _authorized(args):
    required=['spec','scope','expected_scope_sha256','human','expected_human_sha256','review','expected_review_sha256']
    require(all(getattr(args,key,None) for key in required),'NOT_AUTHORIZED: exact scope/human/independent-review artifacts and hashes required before any model import')
    for key,expected in (('scope','expected_scope_sha256'),('human','expected_human_sha256'),('review','expected_review_sha256')):
        require(inherited.digest(getattr(args,key))==getattr(args,expected),'Exact launch artifact changed: '+key)
    spec=_json(args.spec);prepared=prepare(spec);scope=_json(args.scope)
    require(scope==launch_scope(prepared),'Scope differs from exact current source/input preparation')
    output=Path(prepared['output'])
    token=core.authorize_launch(scope,_json(args.human),_json(args.review),consumed=core.consume_once(output.with_name(output.name+'_authorizations_PRIVATE')))
    return spec,prepared,scope,token


def execute(args):
    """All admission gates first; only then global lock, resource gate and lazy native imports."""
    spec,prepared,scope,token=_authorized(args)
    import fcntl
    from types import SimpleNamespace
    from mri_rl.paced_resources import PacedGate
    from mri_rl.two_task_training import make_grader
    from mri_rl.mlx_continuous_gspo import is_loading_unit
    from mri_rl.continuous_rl_pilot import cache_binding_for
    from mri_rl.experiment_contract import load_contract
    runtime=_json(spec['runtime_spec']);out=Path(prepared['output'])
    gate=PacedGate(out.with_name(out.name+'_resources_PRIVATE.jsonl'));resident=[None]
    def before(name):
        deadline=out/'operation_deadline_PRIVATE.json'
        def check():
            if deadline.exists():require(time.time()<_json(deadline)['deadline'],'Absolute operation-start deadline exhausted')
        check();gate.before_unit(name,loading=is_loading_unit(name,{'dir':runtime['cache_dir']}));check()
        inherited.verify_images(prepared['schedule_PRIVATE'],prepared['expected_image_sha256'])
    with Path(runtime['global_lock']).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)   # global one-model flock
        require(prepare(spec)==prepared,'Preparation changed while obtaining global lock');token.check('lock')
        evidence=out.with_name(out.name+'_launch_evidence_PRIVATE');require(not evidence.exists(),'Fresh per-invocation evidence output required');evidence.mkdir(parents=True)
        def factory(adapter,configuration):
            before('certified_cache_binding')
            derived=cache_binding_for(SimpleNamespace(cache_dir=Path(runtime['cache_dir']),cache_certificate=Path(runtime['cache_certificate']),
                cache_certificate_sha256=runtime['cache_certificate_sha256'],model=Path(runtime['model'])))
            require(derived==prepared['derived_cache'],'Certified original base cache changed')
            from mri_rl.shared_rollout_gspo_backend_v1 import backend_factory
            resident[0]=backend_factory(model_path=Path(runtime['model']),contract=load_contract(Path(runtime['contract'])),seed=13,before_unit=before,
                                        cache={'dir':runtime['cache_dir']})(adapter,configuration)
            require(len(resident[0].trainable_signature())==504,'Exact504LoRA inventory required')
            return resident[0]
        def run():
            return driver.run(output=out,schedule=prepared['schedule_PRIVATE'],bindings={'stage':'accuracy_recovery_v1','scope_sha256':core.scope_digest(scope),
                'source_sha256':prepared['source_sha256'],'input_sha256':prepared['input_sha256']},backend_factory=factory,authorization=token,
                phase_gate=lambda context:before('phase_'+context['phase']),parent_checkpoint=runtime['parent_checkpoint'],parent_receipt_sha256=runtime['parent_receipt_sha256'],
                grader=make_grader(_json(runtime['calibration'])),reward_policy_sha256=prepared['reward_policy_sha256'],budget=prepared['budget'],resume=bool(spec.get('resume')),config=prepared['configuration'])
        result=inherited.run_with_release(run,resident,evidence)
        require(prepare(spec)==prepared,'Frozen sources/artifacts changed during execution');token.check('complete')
        core.publish_once(evidence/'complete_PRIVATE.json',{**result,'scope_sha256':core.scope_digest(scope)})
        return result


def main(argv=None,*,start=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','run'))
    for name in ('spec','output','scope','human','review'):parser.add_argument('--'+name,type=Path)
    for name in ('scope','human','review'):parser.add_argument('--expected-'+name+'-sha256')
    parser.add_argument('--verify-images',action='store_true')
    args=parser.parse_args(argv);require(args.spec,'Specification required')
    if args.action=='run':
        print(json.dumps(execute(args),sort_keys=True));return
    prepared=prepare(_json(args.spec),verify_images=args.verify_images,start=start)
    require(args.output and not args.output.exists(),'Fresh private CPU preparation output required');core.require_canonical_output(args.output,start=start)
    args.output.parent.mkdir(parents=True,exist_ok=True);core.publish_once(args.output,prepared)
    if not prepared['blockers']:core.publish_once(args.output.with_name(args.output.stem+'_scope.json'),launch_scope(prepared))
    print(json.dumps({'status':'PREPARED_BLOCKED_PENDING_INPUTS' if prepared['blockers'] else prepared['implementation_status'],'blockers':prepared['blockers'],
                      'preparation_sha256':core.identity(prepared),'model_loads':0,'optimizer_updates':0,'training_authorized':False,'model_PASS':False}))


if __name__=='__main__':main()
