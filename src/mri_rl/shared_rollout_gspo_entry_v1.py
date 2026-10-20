"""Fail-closed entry for the shared-rollout one-step GSPO mechanism control.

`prepare` is CPU only: metadata/receipts, no model, no payload arrays, no image reads unless
`verify_images` is requested. `qualify` and `train` are future modes: each requires the exact
launch scope, a fresh human authorization for that action, and an independent source/input/scope
review, all digest-bound on the command line (existing admission-contract shape; a local
approved=true file cannot authorize). Training additionally needs an independently reviewed actual
read-only qualification proof of the same source closure. All gates precede any native import.
IMPLEMENTED_CPU_NOT_NATIVE_QUALIFIED; NOT_RUN.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
from mri_rl import shared_rollout_gspo_core_v1 as core
from mri_rl import shared_rollout_gspo_driver_v1 as driver
from mri_rl import paired_z045_continuation_entry_v1 as inherited

ROOT=Path(__file__).resolve().parents[2]
require=core.require
OWN_SOURCES=('src/mri_rl/shared_rollout_gspo_core_v1.py','src/mri_rl/shared_rollout_gspo_driver_v1.py','src/mri_rl/shared_rollout_gspo_backend_v1.py',
             'src/mri_rl/shared_rollout_gspo_entry_v1.py','src/mri_rl/shared_rollout_gspo_eval_v1.py','scripts/run_shared_rollout_gspo_v1.py',
             'src/mri_rl/paired_prompt_groups_v1.py','src/mri_rl/class_margin_aux_v2.py','src/mri_rl/paired_z045_matched_f32_eval_v1.py')


def _json(path):return json.loads(Path(path).read_text())


def inputs(spec):
    for key,expected in spec['file_sha256'].items():require(inherited.digest(spec[key])==expected,'Frozen specification input changed: '+key)
    runtime=_json(spec['runtime_spec'])
    for key,expected in runtime['file_sha256'].items():require(inherited.digest(runtime[key])==expected,'Frozen runtime input changed: '+key)
    return runtime


def prepare(spec,*,verify_images=False):
    """Bind exact source/input/scope. Never loads a model; never reads weights/Adam payload arrays."""
    runtime=inputs(spec);parent=inherited.parent_metadata(runtime['parent_checkpoint'],runtime['parent_receipt_sha256'])
    schedule=_json(spec['schedule']);schedule=schedule['schedule'] if isinstance(schedule,dict) else schedule
    rows=driver._modality_rows(schedule)
    from mri_rl.two_task_training import validate_training_row
    for row in schedule:validate_training_row(row)
    expected=spec['expected_image_sha256']
    require(set(expected)=={row['id'] for row in schedule} and all(core.sha(v) for v in expected.values()),'Exact frozen per-row image SHA required')
    if verify_images:inherited.verify_images(schedule,expected)
    calibration=_json(runtime['calibration']);original=parent['original_binding']['experiment']['parent_experiment']
    require(original['arm']=={'name':'U_calibrated','reward':'medgrpo_eq3','sampling':'uniform'} and original['seed']==13 and
            inherited.scope_digest(calibration)==original['calibration_sha256'],'Original task-routed calibration differs')
    from mri_rl import paired_z045_continuation_driver_v1 as old
    configuration={**old.PROFILE,'group_size':core.GROUP_SIZE,'max_tokens':core.MAX_TOKENS}
    require(configuration==parent['original_binding']['configuration'],'Inherited reward/LR/LoRA/K4 configuration changed')
    lengths=spec.get('prompt_lengths',{'F':213,'T':213});require(lengths=={'F':213,'T':213},'Fixed historical sequence prompt cost 213 required')
    sources=inherited.source_closure(runtime)
    for name in OWN_SOURCES:sources[str(ROOT/name)]=inherited.digest(ROOT/name)
    output=Path(spec['output']).resolve();require(output.is_relative_to(ROOT/'local_data/shared_rollout_gspo'),'Fresh isolated private output required')
    input_sha=core.identity({'schedule_sha256':inherited.scope_digest(schedule),'expected_image_sha256':expected,'parent_receipt_sha256':parent['receipt_sha256'],
        'calibration_sha256':original['calibration_sha256'],'spec_sha256':inherited.stage_spec_binding(spec),'runtime_spec_sha256':inherited.digest(spec['runtime_spec'])})
    prepared={'schema':'shared-rollout-gspo-preparation-v1','implementation_status':core.STATUS,'experiment':core.EXPERIMENT,'execution_subtype':core.EXECUTION_SUBTYPE,
        'scope_kind':'PREPARATION_ONLY_NOT_LAUNCH_SCOPE','training_authorized':False,'native_approved':False,'model_loads_this_preparation':0,'optimizer_updates_this_preparation':0,
        'parent':parent,'schedule_PRIVATE':schedule,'expected_image_sha256':expected,'configuration':configuration,'reward_policy_sha256':original['calibration_sha256'],
        'budgets':{mode:core.derive_budget(lengths,mode) for mode in core.MODES},'source_sha256':sources,'input_sha256':input_sha,
        'derived_cache':inherited.cache_metadata(runtime)['expected_binding'],'output':str(output),
        'arm_definition':{'A':'F0F1F2F3 | T0T1T2T3','B':'F0F1T0T1 | F2F3T2T3','weight':'1/8 per gate-admitted response','gate':'inherited Adam60 valid-format-subset gate in both arms',
                          'updates':'at most one actual Adam update per arm from identical full parent state; no sequential stale second-group update; inactive arm = 0 updates, no resample'},
        'limitations':['Native mixed-context F32 rollout/scorer/backward NOT measured; CPU fixtures prove routing, accounting and transactions only.',
                       'Different execution subtype from future SEQUENTIAL_ON_POLICY_GSPO; shared batch gradient is not a per-K4 update trajectory.',
                       'Typed receipts do not authenticate real model outputs; qualification and independent review do.','No accuracy or training-gain claim.']}
    return prepared


def launch_scope(prepared,mode):
    require(mode in core.MODES,'Unknown mode')
    return {'scope_kind':core.SCOPE_KINDS[mode],'experiment':core.EXPERIMENT,'execution_subtype':core.EXECUTION_SUBTYPE,'numerical_policy':core.POLICY,
            'budget':prepared['budgets'][mode],'source_sha256':prepared['source_sha256'],'input_sha256':prepared['input_sha256'],
            'preparation_sha256':core.identity(prepared),'output':prepared['output']}


def qualification_proof(output,prepared):
    """Aggregate proof of a completed read-only qualification run, for independent review."""
    output=Path(output);done=_json(output/'complete_SAFE.json');require(done['status']=='SCHEDULE_COMPLETE' and done['mode']=='qualify' and done['optimizer_updates']==0,'Completed zero-update qualification required')
    old=_json(output/'old_PRIVATE.json')
    arms={arm:_json(output/f'qualification_arm_{arm}_PRIVATE.json') for arm in driver.ARM_ORDER}
    return {'status':'ACTUAL_SHARED_ROLLOUT_GSPO_READONLY_QUALIFIED','optimizer_updates':0,'source_sha256':prepared['source_sha256'],'input_sha256':prepared['input_sha256'],
            'bundle_sha256':done['bundle_sha256'],'counters':done['counters'],
            'rollout_vs_own_context_parity_max_abs':{k:v['max_abs_error'] for k,v in old['rollout_vs_own_context_parity'].items()},
            'arms':{a:{'effective_groups':v['gate']['effective_groups'],'backward_calls':v['gradient_evidence']['backward_calls'],'gradient_l2':v['gradient_evidence']['gradient_l2'],
                       'old_current_max_abs':[r['old_current_max_abs'] for r in v['gradient_evidence']['responses']],'update_decision':v['update_decision']} for a,v in arms.items()},
            'science_claim':False}


def _authorized(mode,args):
    required=['spec','scope','expected_scope_sha256','human','expected_human_sha256','review','expected_review_sha256']
    if mode=='train':required+=['qualification','expected_qualification_sha256','qualification_review','expected_qualification_review_sha256']
    require(all(getattr(args,key,None) for key in required),'NOT_AUTHORIZED: exact scope/human/independent-review'+('/qualification' if mode=='train' else '')+' artifacts and hashes required before any model import')
    pairs=[('scope','expected_scope_sha256'),('human','expected_human_sha256'),('review','expected_review_sha256')]
    if mode=='train':pairs+=[('qualification','expected_qualification_sha256'),('qualification_review','expected_qualification_review_sha256')]
    for key,expected in pairs:require(inherited.digest(getattr(args,key))==getattr(args,expected),'Exact launch artifact changed: '+key)
    spec=_json(args.spec);prepared=prepare(spec);scope=_json(args.scope)
    require(scope==launch_scope(prepared,mode),'Scope differs from exact current source/input preparation')
    output=Path(prepared['output'])
    qualification=None
    if mode=='train':qualification={'proof':_json(args.qualification),'review':_json(args.qualification_review)}
    core.authorize_launch(mode,scope,_json(args.human),_json(args.review),qualification=qualification,source_sha256=prepared['source_sha256'],
                          consumed=core.consume_once(output.with_name(output.name+'_authorizations_PRIVATE')))
    return spec,prepared,scope


def execute(mode,args):
    """All admission gates first; only then global lock, resource gate and lazy native imports."""
    spec,prepared,scope=_authorized(mode,args)
    import fcntl
    from types import SimpleNamespace
    from mri_rl.paced_resources import PacedGate
    from mri_rl.two_task_training import make_grader
    from mri_rl.mlx_continuous_gspo import is_loading_unit
    from mri_rl.continuous_rl_pilot import cache_binding_for
    from mri_rl.experiment_contract import load_contract
    runtime=inputs(spec);out=Path(prepared['output']);gate=PacedGate(out.with_name(out.name+'_resources_PRIVATE.jsonl'));resident=[None]
    def check_sources():
        for path,expected in prepared['source_sha256'].items():require(inherited.digest(path)==expected,'Reviewed source changed: '+path)
    def before(name):
        deadline=out/'operation_deadline_PRIVATE.json'
        def check():
            if deadline.exists():require(time.time()<_json(deadline)['deadline'],'Absolute operation-start deadline exhausted')
        check();gate.before_unit(name,loading=is_loading_unit(name,{'dir':runtime['cache_dir']}));check()
        inherited.verify_images(prepared['schedule_PRIVATE'],prepared['expected_image_sha256'])
    with Path(runtime['global_lock']).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)   # global one-model flock
        require(prepare(spec)==prepared,'Preparation changed while obtaining global lock');check_sources()
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
            return driver.run(output=out,mode=mode,schedule=prepared['schedule_PRIVATE'],bindings={'stage':'shared_rollout_gspo_v1','scope_sha256':core.scope_digest(scope)},
                backend_factory=factory,authorize=lambda context:before('authorize_'+context['phase']),parent_checkpoint=runtime['parent_checkpoint'],
                parent_receipt_sha256=runtime['parent_receipt_sha256'],grader=make_grader(_json(runtime['calibration'])),reward_policy_sha256=prepared['reward_policy_sha256'],
                budget=prepared['budgets'][mode],resume=bool(spec.get('resume')),config=prepared['configuration'])
        result=inherited.run_with_release(run,resident,evidence)
        require(prepare(spec)==prepared,'Frozen sources/artifacts changed during execution');check_sources()
        core.publish_once(evidence/'complete_PRIVATE.json',{**result,'scope_sha256':core.scope_digest(scope)})
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','qualify','train','qualification-proof'))
    for name in ('spec','output','scope','human','review','qualification','qualification-review','run-directory'):parser.add_argument('--'+name,type=Path)
    for name in ('scope','human','review','qualification','qualification-review'):parser.add_argument('--expected-'+name+'-sha256')
    parser.add_argument('--verify-images',action='store_true')
    args=parser.parse_args();require(args.spec,'Specification required')
    if args.action in ('qualify','train'):
        print(json.dumps(execute(args.action,args),sort_keys=True));return
    prepared=prepare(_json(args.spec),verify_images=args.verify_images)
    if args.action=='qualification-proof':
        require(args.run_directory and args.output and not args.output.exists(),'Completed qualification run directory and fresh output required')
        core.publish_once(args.output,qualification_proof(args.run_directory,prepared));print(json.dumps({'status':'QUALIFICATION_PROOF_WRITTEN_FOR_INDEPENDENT_REVIEW'}));return
    require(args.output and not args.output.exists() and args.output.resolve().is_relative_to(ROOT/'local_data'),'Fresh private CPU preparation output under ignored local_data required')
    args.output.parent.mkdir(parents=True,exist_ok=True);core.publish_once(args.output,prepared)
    for mode in core.MODES:core.publish_once(args.output.with_name(args.output.stem+f'_{mode}_scope.json'),launch_scope(prepared,mode))
    print(json.dumps({'status':prepared['implementation_status'],'preparation_sha256':core.identity(prepared),'model_loads':0,'optimizer_updates':0,'training_authorized':False}))


if __name__=='__main__':main()
