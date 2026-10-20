"""Matched F32 parent / arm A / arm B evaluation entry; IMPLEMENTED_CPU_NOT_NATIVE_QUALIFIED, NOT_RUN.

Reuses the existing separate F32 evaluator (paired_z045_matched_f32_eval_v1): the same frozen
internal-40 records, F32 generation/scorer, temperature 0, max 256 tokens, strict parser, candidate
mean/sum/first, swap alias reuse. Only checkpoint discovery, the 3-checkpoint budget and the admission
action are new. The historical BF16 inherited evaluation methods are never called (the F32 backend
class raises on them). A non-updated arm is identical to the parent and is recorded as an alias, not
re-evaluated. Sealed new16 outputs are never read.
"""
from __future__ import annotations
import gc
import json
from pathlib import Path
import time
from mri_rl import paired_z045_continuation_entry_v1 as entry
from mri_rl import paired_z045_matched_f32_eval_v1 as matched
from mri_rl import shared_rollout_gspo_core_v1 as core

require=core.require
ACTION='shared_rollout_readonly_matched_f32_evaluation'
LABELS=('parent','A','B')


def eval_budget(checkpoints=3):
    """Matched evaluator per-checkpoint maxima scaled to parent+A+B."""
    base=matched.budget();per={k:v//2 for k,v in base.items() if k in ('greedy_generations','candidate_score_API_calls','maximum_generated_tokens',
        'teacherforced_input_positions','generation_language_model_calls','vision_embedding_calls')}
    require(checkpoints in (1,2,3),'Parent only (both arms aliased), or parent plus one or two updated arm checkpoints')
    return {**{k:v*checkpoints for k,v in per.items()},'checkpoints':checkpoints,'optimizer_updates':0,'maximum_seconds':1200*checkpoints,'simultaneous_models':1,'simultaneous_image_graphs':1}


class EvalCounter(matched.Counter):
    def __init__(self,checkpoints):super().__init__();self.limits=eval_budget(checkpoints)
    def charge(self,name,value=1):
        self.values[name]=self.values.get(name,0)+value
        require(type(value) is int and value>=0 and self.values[name]<=self.limits[name],'Frozen evaluation budget exceeded: '+name)


def arm_checkpoint(run_directory,arm,parent):
    """Metadata of a COMMITTED arm terminal; no tensor reads. A non-updated arm aliases the parent."""
    run_directory=Path(run_directory);marker=run_directory/f'arm_{arm}_COMMITTED.json'
    require(marker.is_file(),'Arm terminal not committed; nothing to evaluate')
    record=json.loads(marker.read_text());path=run_directory/f'arm_{arm}'/'boundary_0001'
    require(entry.digest(path/'receipt.json')==record['boundary_receipt_sha256'],'Arm terminal receipt differs from commit marker')
    receipt=json.loads((path/'receipt.json').read_text());manifest=json.loads((path/'optimizer/manifest.json').read_text())
    experiment=receipt['binding']['experiment']
    require(receipt['schema']=='continuous-rl-boundary-v1' and receipt['completed_groups']==1 and receipt['updated']==record['updated'] and receipt['optimizer_updates']==int(record['updated']) and
            receipt['absolute_completed_groups']==parent['groups']+1 and receipt['absolute_optimizer_updates']==parent['updates']+int(record['updated']) and
            experiment['experiment']==core.EXPERIMENT and experiment['arm']==arm and experiment['adaptive_parent']['receipt_sha256']==parent['receipt_sha256'] and
            manifest['optimizer_updates']==receipt['absolute_optimizer_updates'] and manifest['bindings']['run']==receipt['binding'] and
            manifest['bindings']['adapter_sha256']==receipt['files_sha256']['adapter.safetensors'],'Arm terminal parent/Adam/arm binding differs')
    unchanged=receipt['files_sha256']['adapter.safetensors']==parent['adapter_sha256']
    require(unchanged==(not record['updated']),'Adapter change disagrees with the update record')
    return {'path':str(path.resolve()),'receipt_sha256':record['boundary_receipt_sha256'],'adapter_sha256':receipt['files_sha256']['adapter.safetensors'],
            'updated':bool(record['updated']),'alias_of_parent':not record['updated'],'groups':receipt['absolute_completed_groups'],'updates':receipt['absolute_optimizer_updates'],'array_bytes_verified':False}


def prepare(spec):
    """Reuse the matched preparation for the frozen records/tokens/cache; bind parent + committed arms."""
    base=matched.prepare({**spec,'terminal_checkpoint':None})
    arms={arm:arm_checkpoint(spec['shared_run'],arm,base['parent']) for arm in ('A','B')}
    distinct=1+sum(not v['alias_of_parent'] for v in arms.values())
    root=Path(core.__file__).resolve().parents[2]
    direct=('src/mri_rl/paired_z045_evaluation_v1.py','src/mri_rl/answer_likelihood.py','src/mri_rl/vero_mri_reward.py','src/mri_rl/paired_z045_f32_backend_v1.py',
            'src/mri_rl/shared_rollout_gspo_core_v1.py','src/mri_rl/shared_rollout_gspo_entry_v1.py','src/mri_rl/shared_rollout_gspo_eval_v1.py','scripts/run_shared_rollout_gspo_eval_v1.py')
    from mri_rl import shared_rollout_gspo_entry_v1 as shared_entry
    sources={**base['source_sha256'],**{str(root/name):entry.digest(root/name) for name in (*direct,*shared_entry.OWN_SOURCES)}}
    return {**base,'schema':'shared-rollout-gspo-matched-F32-evaluator-v1','status':core.STATUS,'arms':arms,'terminal':None,'terminal_binding_status':'ARM_TERMINALS_BOUND',
            'source_sha256':sources,'budget':eval_budget(distinct),'evaluated_checkpoints':['parent']+[a for a,v in arms.items() if not v['alias_of_parent']],
            'shared_bundle_sha256':json.loads((Path(spec['shared_run'])/'bundle_COMMITTED.json').read_text())['bundle_sha256'],
            'scope_limitation':'Parent and committed shared-rollout arm terminals only; engineering diagnostic, no independent-generalization/clinical/training-gain claim.'}


def authorize_eval(args,scope):
    """Every gate, before any native import. The numeric prerequisite is the NEW shared-rollout read-only
    qualification (effective, reviewed, same source) - not the old ACTUAL_F32 status, which this path cannot produce."""
    required=('review','expected_review_sha256','human_authorization','expected_human_sha256','numeric_contract','expected_numeric_sha256')
    require(all(getattr(args,k,None) for k in required),'NOT_AUTHORIZED: exact new evaluation review/human/shared-qualification artifacts required')
    for path,expected in ((args.review,args.expected_review_sha256),(args.human_authorization,args.expected_human_sha256),(args.numeric_contract,args.expected_numeric_sha256)):
        require(entry.digest(path)==expected,'Eval authorization evidence changed')
    require(scope.get('status')==core.STATUS and scope.get('numerical_contract_sha256')==args.expected_numeric_sha256,'Eval scope status/numerical contract SHA differs')
    review=json.loads(args.review.read_text());human=json.loads(args.human_authorization.read_text());numeric=json.loads(args.numeric_contract.read_text());binding=entry.scope_digest(scope)
    require(review.get('verdict') in ('PASS','PASS_WITH_LIMITATIONS') and review.get('scope_sha256')==binding and review.get('reviewer_id') and review.get('reviewer_kind') in core.REVIEWER_KINDS,'Independent exact eval review missing')
    require(human.get('authorized') is True and human.get('action')==ACTION and human.get('scope_sha256')==binding and core.sha(human.get('human_instruction_sha256')),'New human shared-rollout evaluation authorization missing')
    require(numeric.get('actual_proof_path') and numeric.get('actual_review_path'),'Shared-rollout qualification prerequisite missing')
    for key in ('actual_proof','actual_review'):require(entry.digest(numeric[key+'_path'])==numeric[key+'_sha256'],'Qualification artifact changed')
    actual=json.loads(Path(numeric['actual_proof_path']).read_text());actual_review=json.loads(Path(numeric['actual_review_path']).read_text())
    from mri_rl import shared_rollout_gspo_entry_v1 as shared_entry
    own={str(Path(core.__file__).resolve().parents[2]/name) for name in shared_entry.OWN_SOURCES}
    require(actual.get('status')=='ACTUAL_SHARED_ROLLOUT_GSPO_READONLY_QUALIFIED' and not actual.get('blockers') and actual.get('optimizer_updates')==0 and
            actual_review.get('verdict') in ('PASS','PASS_WITH_LIMITATIONS') and actual_review.get('actual_proof_sha256')==core.identity(actual) and
            actual_review.get('reviewer_id') and actual_review.get('reviewer_kind') in core.REVIEWER_KINDS and
            own<=set(actual['source_sha256']) and own<=set(scope['source_sha256']) and all(scope['source_sha256'][p]==actual['source_sha256'][p] for p in own) and
            all(scope['source_sha256'].get(p,d)==d for p,d in actual['source_sha256'].items()),'Effective same-source shared-rollout qualification prerequisite not qualified')
    return {'binding':binding,'review':review,'human':human}


def execute(args,spec,scope):
    import fcntl
    from mri_rl import early_native_conditioning_probe_v1 as utility
    binding=authorize_eval(args,scope)['binding']
    cs=json.loads(Path(spec['continuation_spec']).read_text());out=Path(scope['output']);require(not out.exists(),'Fresh private matched evaluation output required')
    def verify():
        for path,expected in scope['source_sha256'].items():require(entry.digest(path)==expected,'Matched eval source changed')
        require(prepare(spec)==scope,'Matched eval data/checkpoint/source metadata changed')
    verify()
    from mri_rl.paced_resources import PacedGate
    from mri_rl.continuous_rl_pilot import cache_binding_for
    from mri_rl.optimizer_checkpoint import load_optimizer_checkpoint
    from mri_rl.uniform_balance_continuation_gate_v1 import parent_anchor
    from mri_rl.experiment_contract import load_contract
    from mri_rl.paired_z045_f32_backend_v1 import F32NativeGSPOBackend
    from mri_rl.mlx_continuous_gspo import is_loading_unit
    from types import SimpleNamespace
    started=time.monotonic();limit=scope['budget']['maximum_seconds'];counter=EvalCounter(scope['budget']['checkpoints']);results=[]
    checkpoints=[('parent',scope['parent'])]+[(a,scope['arms'][a]) for a in ('A','B') if not scope['arms'][a]['alias_of_parent']]
    with Path(cs['global_lock']).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);out.mkdir(parents=True);gate=PacedGate(out/'resources_PRIVATE.jsonl');cache={'dir':cs['cache_dir']}
        derived=cache_binding_for(SimpleNamespace(cache_dir=Path(cs['cache_dir']),cache_certificate=Path(cs['cache_certificate']),cache_certificate_sha256=cs['cache_certificate_sha256'],model=Path(cs['model'])))
        require(derived==scope['derived_cache'],'Matched evaluator certified cache changed')
        def before(name):
            require(time.monotonic()-started<=limit,'Evaluation operation-start deadline exhausted');gate.before_unit(name,loading=is_loading_unit(name,cache));require(time.monotonic()-started<=limit,'Evaluation deadline exhausted during HOLD')
            if name=='native_model_load':entry.verify_images([r['row_PRIVATE'] for r in scope['records_PRIVATE']],scope['expected_image_sha256'])
        for label,checkpoint in checkpoints:
            directory=out/(label+'_PRIVATE');directory.mkdir();resident=[None]
            def run():
                anchor=parent_anchor(checkpoint['path'] if label!='parent' else scope['parent']['path'],checkpoint['receipt_sha256'])
                resident[0]=F32NativeGSPOBackend(Path(anchor_path(scope,label,checkpoint))/'adapter.safetensors',{**entry.driver.PROFILE,'group_size':4,'max_tokens':256},
                    model_path=Path(cs['model']),contract=load_contract(Path(cs['contract'])),seed=13,before_unit=before,cache=cache)
                backend=resident[0]
                state,rng,_=load_optimizer_checkpoint(Path(anchor_path(scope,label,checkpoint))/'optimizer',expected_bindings={'run':anchor['original_binding'],'adapter_sha256':anchor['adapter_sha256'],
                    'trainable_signature':backend.trainable_signature()},load_arrays=backend.load_arrays)
                backend.restore(state,rng);state=rng=None
                return matched.run_checkpoint(backend,scope,label,checkpoint,directory/'rows_PRIVATE',counter,before)
            results.append(entry.run_with_release(run,resident,directory));utility.durable(directory/'complete_PRIVATE.json',results[-1]);resident[0]=None;gc.collect();verify()
    utility.durable(out/'complete_PRIVATE.json',{'status':'ACTUAL_SHARED_ROLLOUT_MATCHED_F32_EVALUATION_COMPLETE','scope_sha256':binding,'checkpoints':results,
        'aliases_of_parent':[a for a in ('A','B') if scope['arms'][a]['alias_of_parent']],'actual_calls':counter.values,'optimizer_updates':0})


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--spec',type=Path,required=True);parser.add_argument('--scope-output',type=Path,required=True);parser.add_argument('--execute',action='store_true')
    for name in ('review','human-authorization','numeric-contract'):parser.add_argument('--'+name,type=Path)
    for name in ('expected-review-sha256','expected-human-sha256','expected-numeric-sha256'):parser.add_argument('--'+name)
    args=parser.parse_args(argv);spec=json.loads(args.spec.read_text());scope=prepare(spec);args.scope_output.parent.mkdir(parents=True,exist_ok=True)
    if args.scope_output.exists():require(json.loads(args.scope_output.read_text())==scope,'Frozen evaluator scope differs')
    else:core.publish_once(args.scope_output,scope)
    if args.execute:execute(args,spec,scope);return {'status':'EXECUTED'}
    result={'status':scope['status'],'scope_sha256':entry.scope_digest(scope),'evaluated_checkpoints':scope['evaluated_checkpoints'],'model_loads':0}
    print(json.dumps(result,sort_keys=True));return result


def anchor_path(scope,label,checkpoint):return scope['parent']['path'] if label=='parent' else checkpoint['path']
