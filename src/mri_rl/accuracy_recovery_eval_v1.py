"""VERO_TO_MRI_ACCURACY_RECOVERY_V1 matched read-only F32 evaluation: Parent52 / committed A / committed B. IMPLEMENTED_CPU_NOT_NATIVE_QUALIFIED.

Reuses the existing evaluation kernels unchanged (paired_z045_matched_f32_eval_v1.run_checkpoint/run_row/token_plan/compile_records:
frozen internal-dev 40 real + 8 swap aliases, strict temperature-0 greedy with the F32 final processor, strict canonical label spans with
mean/sum/first, prefix oracle) and the existing eval budget scaling. New here: checkpoint discovery that authenticates THIS experiment's
receipts and the ORIGINAL Adam52 parent (no Adam60 validator, no consumed qualification proof), a fail-closed Authorization for the
evaluation scope, a testable serial runner (at most three models, each released, a no-update arm is an alias of the Parent and is never
re-evaluated), a canonical-root output guard, and a CPU analyzer with the pre-declared stop rule. No optimizer is constructed or called.
Sealed new16 and old final100 outputs are never read; the evaluation records must be the internal dev records only.
"""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import time
from mri_rl import accuracy_recovery_v1 as core
from mri_rl import accuracy_recovery_entry_v1 as train_entry
from mri_rl import paired_z045_continuation_entry_v1 as inherited
from mri_rl import paired_z045_continuation_driver_v1 as old
from mri_rl import paired_z045_evaluation_v1 as ev
from mri_rl import paired_z045_matched_f32_eval_v1 as matched
from mri_rl import shared_rollout_gspo_eval_v1 as shared_eval
from mri_rl.answer_likelihood import PREFIX

require=core.require
ROOT=Path(__file__).resolve().parents[2]
LABELS_ORDER=('parent','A','B')
EVAL_RUNTIME_KEYS=('parent_checkpoint','parent_receipt_sha256','cache_dir','cache_certificate','cache_certificate_sha256','model','contract','global_lock','runtime_source_sha256')
eval_budget=shared_eval.eval_budget
EvalCounter=shared_eval.EvalCounter


# ---------------------------------------------------------------- checkpoint discovery (this experiment only)
def arm_checkpoint(run_directory,arm,anchor):
    """Metadata of a COMMITTED arm terminal of the combined run. Validates the new experiment id, the original Adam52 parent, absolute counters
    114->115 / 52->52|53 and the gradient archive; a non-updated arm aliases the Parent. No tensor reads."""
    run_directory=Path(run_directory);marker_path=run_directory/f'arm_{arm}_COMMITTED.json'
    require(marker_path.is_file(),'Arm terminal not committed; nothing to evaluate')
    done=json.loads((run_directory/'complete_SAFE.json').read_text()) if (run_directory/'complete_SAFE.json').is_file() else None
    require(done is not None and done.get('status')=='SCHEDULE_COMPLETE' and done.get('experiment')==core.EXPERIMENT and done.get('final_state_equals_parent') is True and
            done.get('parent_disk_unchanged') is True,'Completed combined run (complete_SAFE) required')
    identity=json.loads((run_directory/'run_identity_PRIVATE.json').read_text())
    require(identity['parent']=={'receipt_sha256':anchor['receipt_sha256'],'adapter_sha256':anchor['adapter_sha256']},'Run was not started from the frozen Adam52 parent')
    record=json.loads(marker_path.read_text());path=run_directory/f'arm_{arm}'/'boundary_0001'
    require(old.digest(path/'receipt.json')==record['boundary_receipt_sha256'],'Arm terminal receipt differs from commit marker')
    receipt=json.loads((path/'receipt.json').read_text());manifest=json.loads((path/'optimizer/manifest.json').read_text());experiment=receipt['binding']['experiment']
    updated=bool(record['updated'])
    require(receipt['schema']=='continuous-rl-boundary-v1' and receipt['completed_groups']==1 and receipt['updated']==updated and receipt['optimizer_updates']==int(updated) and
            receipt['absolute_completed_groups']==anchor['groups']+1 and receipt['absolute_optimizer_updates']==anchor['updates']+int(updated) and
            experiment['experiment']==core.EXPERIMENT and experiment['arm']==arm and experiment['adaptive_parent']['receipt_sha256']==anchor['receipt_sha256'] and
            experiment['adaptive_parent']['adapter_sha256']==anchor['adapter_sha256'] and experiment['adaptive_parent']['groups']==anchor['groups'] and
            manifest['optimizer_updates']==receipt['absolute_optimizer_updates'] and manifest['completed_groups']==receipt['absolute_completed_groups'] and
            manifest['bindings']['run']==receipt['binding'] and manifest['bindings']['adapter_sha256']==receipt['files_sha256']['adapter.safetensors'],'Arm terminal parent/Adam/arm/experiment binding differs')
    meta=run_directory/'gradients_PRIVATE'/f'arm_{arm}_gradient_meta_PRIVATE.json'
    require(meta.is_file() and old.digest(meta)==receipt['gradient_archive_meta_sha256'],'Arm terminal is not bound to its pre-step gradient archive')
    unchanged=receipt['files_sha256']['adapter.safetensors']==anchor['adapter_sha256']
    require(unchanged==(not updated),'Adapter change disagrees with the update record')
    return {'path':str(path.resolve()),'receipt_sha256':record['boundary_receipt_sha256'],'adapter_sha256':receipt['files_sha256']['adapter.safetensors'],'updated':updated,
            'alias_of_parent':not updated,'groups':receipt['absolute_completed_groups'],'updates':receipt['absolute_optimizer_updates'],'array_bytes_verified':False,
            'bundle_sha256':record['bundle_sha256']}


def evaluated_labels(arms):
    return ['parent']+[a for a in ('A','B') if not arms[a]['alias_of_parent']]


# ---------------------------------------------------------------- CPU preparation
def pending_inputs(spec):
    blockers=[];runtime=None
    for key in ('runtime_spec','evaluation','training_run'):
        if key not in spec:blockers.append(f'SPEC_KEY_NOT_BOUND:{key}')
    if blockers:return blockers,None
    if not Path(spec['runtime_spec']).is_file():return ['RUNTIME_SPEC_NOT_PRESENT'],None
    runtime=json.loads(Path(spec['runtime_spec']).read_text());missing=[k for k in EVAL_RUNTIME_KEYS if k not in runtime]
    if missing:blockers.append('RUNTIME_KEYS_NOT_BOUND:'+','.join(missing))
    else:
        if not Path(runtime['parent_checkpoint']).is_dir():blockers.append('PARENT_CHECKPOINT_NOT_PRESENT')
        if not (Path(runtime['cache_certificate']).is_file() and Path(runtime['cache_dir']).is_dir() and Path(runtime['model']).exists()):blockers.append('CERTIFIED_BASE_CACHE_OR_MODEL_NOT_PRESENT')
        if not runtime['runtime_source_sha256']:blockers.append('RUNTIME_SOURCE_CLOSURE_NOT_BOUND')
    if not Path(spec['evaluation']).is_file():blockers.append('EVALUATION_SPEC_NOT_PRESENT')
    if not (Path(spec['training_run'])/'complete_SAFE.json').is_file():blockers.append('TRAINING_RUN_NOT_COMPLETE')
    return blockers,runtime


def _encode(runtime,encode):
    if encode is not None:return encode,[]
    try:
        from tokenizers import Tokenizer
    except ImportError:
        return None,['TOKENIZER_PACKAGE_NOT_AVAILABLE']
    tok=Tokenizer.from_file(str(Path(runtime['model'])/'tokenizer.json'));return (lambda text:tok.encode(text,add_special_tokens=False).ids),[]


def prepare(spec,*,encode=None,verify_images=False,start=None):
    """Bind exact checkpoints/records/tokens/sources. Never loads a model or weights; optional image hash verification only."""
    train_entry.refuse_forbidden(spec)
    blockers,runtime=pending_inputs(spec)
    base={'schema':'accuracy-recovery-evaluation-preparation-v1','implementation_status':core.STATUS,'experiment':core.EXPERIMENT,'scope_kind':'PREPARATION_ONLY_NOT_LAUNCH_SCOPE',
          'model_loads_this_preparation':0,'optimizer_updates':0,'model_PASS':False,'blockers':list(blockers),'implemented_by':train_entry.IMPLEMENTED_BY}
    if blockers:return base
    encode,missing=_encode(runtime,encode)
    if missing:return {**base,'blockers':missing}
    anchor=core.adam52_parent_anchor(runtime['parent_checkpoint'],runtime['parent_receipt_sha256'])
    evaluation_sha=inherited.digest(spec['evaluation']);require(spec.get('evaluation_sha256',evaluation_sha)==evaluation_sha,'Frozen evaluation records changed')
    evaluation=json.loads(Path(spec['evaluation']).read_text());plan=matched.compile_records(evaluation)
    for record in plan['real']:
        require(not any(t in str(record['row_PRIVATE'].get('images')) for t in train_entry.FORBIDDEN_TOKENS),'Evaluation records must be the internal-dev records, never final100/new16')
    tokens={task:matched.token_plan(task,PREFIX,encode) for task in ev.LABELS}
    expected={r['id']:r['image_sha256'] for r in plan['real']}
    if verify_images:inherited.verify_images([r['row_PRIVATE'] for r in plan['real']],expected)
    arms={arm:arm_checkpoint(spec['training_run'],arm,anchor) for arm in ('A','B')}
    require(arms['A']['bundle_sha256']==arms['B']['bundle_sha256'],'Arms were trained on different bundles')
    labels=evaluated_labels(arms);output=core.require_canonical_output(spec['output'],start=start)
    sources=inherited.source_closure(runtime)
    for name in (*core.OWN_SOURCES,*core.EVAL_OWN_SOURCES):sources[str(ROOT/name)]=inherited.digest(ROOT/name)
    parent={**{k:anchor[k] for k in ('receipt_sha256','adapter_sha256','optimizer_manifest_sha256','optimizer_state_sha256','groups','updates')},'path':str(Path(runtime['parent_checkpoint']).resolve())}
    run_identity=Path(spec['training_run'])/'run_identity_PRIVATE.json'
    input_sha=core.identity({'evaluation_sha256':evaluation_sha,'expected_image_sha256':expected,'tokens':tokens,'parent':parent['receipt_sha256'],'arms':{a:v['receipt_sha256'] for a,v in arms.items()},
                             'bundle_sha256':arms['A']['bundle_sha256'],'runtime_spec_sha256':inherited.digest(spec['runtime_spec'])})
    return {**base,'parent':parent,'arms':arms,'evaluated_checkpoints':labels,'records_PRIVATE':plan['real'],'aliases_PRIVATE':plan['aliases'],'token_plans_PRIVATE':tokens,
            'expected_image_sha256':expected,'source_sha256':sources,'input_sha256':input_sha,'numerical_contract_sha256':'NOT_REQUIRED_NO_OLD_QUALIFICATION_PROOF',
            'derived_cache':inherited.cache_metadata(runtime)['expected_binding'],'budget':eval_budget(len(labels)),'output':str(output),'policy':core.POLICY,'decoding':matched.greedy_settings(),
            'training_run':{'path':str(Path(spec['training_run']).resolve()),'run_identity_sha256':old.digest(run_identity),'complete_sha256':old.digest(Path(spec['training_run'])/'complete_SAFE.json')},
            'stop_rule':STOP_RULE,'scope_limitation':'Engineering diagnostic on the internal-dev records (4 people / 24 sequence rows / 16 Dallas rows); parent and committed arms only; no independent-generalization or clinical claim.'}


def launch_scope(prepared):
    require(not prepared.get('blockers'),'Evaluation scope is unavailable while bound inputs are pending: '+','.join(prepared.get('blockers',[])))
    return {'scope_kind':core.EVAL_SCOPE_KIND,'experiment':core.EXPERIMENT,'execution_subtype':core.EXECUTION_SUBTYPE,'numerical_policy':core.POLICY,'recipe_label':core.RECIPE_LABEL,
            'implemented_by':prepared['implemented_by'],'budget':prepared['budget'],'source_sha256':prepared['source_sha256'],'input_sha256':prepared['input_sha256'],
            'parent':{'receipt_sha256':prepared['parent']['receipt_sha256'],'adapter_sha256':prepared['parent']['adapter_sha256']},'output':prepared['output'],
            'arms':prepared['arms'],'evaluated_checkpoints':prepared['evaluated_checkpoints'],'training_run':prepared['training_run'],'preparation_sha256':core.identity(prepared)}


# ---------------------------------------------------------------- serial three-model runner (testable; native pieces are injected)
def run_checkpoints(prepared,token,*,backend_factory,runner,before,out,counter,releaser,load_state):
    """Evaluate Parent52 then each UPDATED committed arm, strictly serially. Authorization (source closure) is re-checked before each model;
    the model is released after each checkpoint; a no-update arm is never re-evaluated (alias of Parent)."""
    out=Path(out);out.mkdir(parents=True,exist_ok=False);results=[]
    checkpoints=[('parent',prepared['parent'])]+[(a,prepared['arms'][a]) for a in ('A','B') if not prepared['arms'][a]['alias_of_parent']]
    require(len(checkpoints)<=3 and [c[0] for c in checkpoints]==prepared['evaluated_checkpoints'],'At most three serial models; checkpoints differ from the authorized scope')
    for label,checkpoint in checkpoints:
        token.check('eval_'+label);directory=out/(label+'_PRIVATE');directory.mkdir();resident=[None]
        def run():
            resident[0]=backend_factory(label,checkpoint);backend=resident[0];load_state(backend,label,checkpoint)
            return runner(backend,prepared,label,checkpoint,directory/'rows_PRIVATE',counter,before)
        results.append(releaser(run,resident,directory));resident[0]=None;gc.collect()
    token.check('eval_complete');return results


STOP_RULE={'primary_benefit':'B must beat A on strict-greedy sequence balanced accuracy AND T1 recall (B-A>0 on both)',
           'degrade':'Stop if B FLAIR recall or Dallas accuracy is below the Parent, or Dallas accuracy is below A (retention); a no-update arm equals the Parent',
           'decision':'CONTINUE_CANDIDATE_REQUIRES_NEW_AUTHORIZATION only if benefit and no degrade, else STOP_CANDIDATE; never a training authorization',
           'resolution':'24 sequence rows = 12 FLAIR + 12 T1 from 4 people: one row moves a recall by 1/12; engineering diagnostic, not statistical evidence'}


# ---------------------------------------------------------------- CPU analyzer
def endpoint_vector(rows):
    """Endpoint vector from saved real-row outputs only (strict greedy; candidate mean; ties INVALID)."""
    def prediction(row):return ev.strict_prediction(row['greedy']['text'],row['task'])
    def candidate(row):return ev.candidate_prediction([item['mean'] for item in row['candidates']['per_label']])
    def recall(subset,label,predict):
        group=[r for r in subset if r['truth']==label];require(group,'Missing class rows: '+label)
        return sum(predict(r)==ev.LABELS[r['task']].index(label) for r in group)/len(group)
    sequence=[r for r in rows if r['task']=='mri_sequence'];dallas=[r for r in rows if r['task']=='mri_visit_correspondence']
    require(len(sequence)==24 and len(dallas)==16,'Exact 24 sequence and 16 Dallas real rows required')
    flair,t1=recall(sequence,'FLAIR',prediction),recall(sequence,'contrast-enhanced T1',prediction)
    candidate_ba=(recall(sequence,'FLAIR',candidate)+recall(sequence,'contrast-enhanced T1',candidate))/2
    dallas_accuracy=sum(prediction(r)==ev.LABELS[r['task']].index(r['truth']) for r in dallas)/len(dallas)
    return ev.decision_vector((flair+t1)/2,flair,t1,candidate_ba,dallas_accuracy,recall(dallas,'same',prediction),recall(dallas,'different',prediction))


def swap_summaries(rows,aliases):
    """Image-dependence diagnostic from the 8 swap aliases: the reused opposite-modality real output is the swapped-image prediction (0 new calls)."""
    by_id={r['id']:r for r in rows};real=[];swapped=[];truth=[]
    for alias in aliases.values():
        original,reuse=by_id[alias['original_real_id_PRIVATE']],by_id[alias['reuse_real_id_PRIVATE']]
        real.append(ev.strict_prediction(original['greedy']['text'],original['task']));swapped.append(ev.strict_prediction(reuse['greedy']['text'],reuse['task']))
        truth.append(ev.LABELS[original['task']].index(original['truth']))
    return ev.swap_summary(real,swapped,truth)


def verdict(parent,a,b,swaps=None):
    """Pre-declared stop rule on saved endpoint vectors (see STOP_RULE). Alias arms must be passed as the Parent vector/summary."""
    for vector in (parent,a,b):ev._validate_vector(vector)
    delta={k:b[k]-a[k] for k in ev.KEYS}
    benefit=delta['sequence_balanced_accuracy']>0 and delta['T1_recall']>0
    degrade={'FLAIR_recall_below_parent':b['FLAIR_recall']<parent['FLAIR_recall'],'Dallas_accuracy_below_parent':b['Dallas_accuracy']<parent['Dallas_accuracy'],
             'Dallas_accuracy_below_A':b['Dallas_accuracy']<a['Dallas_accuracy']}
    if swaps is not None:degrade['swap_valid_opposite_class_changes_below_A']=swaps['B']['valid_opposite_class_changes']<swaps['A']['valid_opposite_class_changes']
    go=benefit and not any(degrade.values())
    return {'decision':'CONTINUE_CANDIDATE_REQUIRES_NEW_AUTHORIZATION' if go else 'STOP_CANDIDATE','B_minus_A':delta,'benefit':benefit,'degrade':degrade,'swap_summaries':swaps,
            'A_vs_parent':ev.decide(parent,a)['decision'],'B_vs_parent':ev.decide(parent,b)['decision'],'training_authorized':False,'generalization_proven':False,'stop_rule':STOP_RULE}


def load_rows(checkpoint_dir):
    rows=[]
    for path in sorted((Path(checkpoint_dir)/'rows_PRIVATE').glob('*_PRIVATE.json')):
        value=json.loads(path.read_text())
        if 'greedy' in value and 'candidates' in value:rows.append(value)
    return rows


def analyze(evaluation_output,arms,aliases=None):
    """Vectors for Parent/A/B (an alias arm reuses the Parent rows, no duplicate evaluation) and the verdict."""
    root=Path(evaluation_output);rows={'parent':load_rows(root/'parent_PRIVATE')}
    for arm in ('A','B'):rows[arm]=rows['parent'] if arms[arm]['alias_of_parent'] else load_rows(root/f'{arm}_PRIVATE')
    vectors={label:endpoint_vector(value) for label,value in rows.items()}
    swaps={label:swap_summaries(value,aliases) for label,value in rows.items() if label in ('A','B')} if aliases else None
    return {'schema':'accuracy-recovery-evaluation-analysis-v1','vectors':vectors,'aliases_of_parent':[a for a in ('A','B') if arms[a]['alias_of_parent']],
            'verdict':verdict(vectors['parent'],vectors['A'],vectors['B'],swaps)}


# ---------------------------------------------------------------- guarded native execution
def _authorized(args):
    required=['spec','scope','expected_scope_sha256','human','expected_human_sha256','review','expected_review_sha256']
    require(all(getattr(args,key,None) for key in required),'NOT_AUTHORIZED: exact evaluation scope/human/independent-review artifacts and hashes required before any model import')
    for key,expected in (('scope','expected_scope_sha256'),('human','expected_human_sha256'),('review','expected_review_sha256')):
        require(inherited.digest(getattr(args,key))==getattr(args,expected),'Exact launch artifact changed: '+key)
    spec=json.loads(Path(args.spec).read_text());prepared=prepare(spec);scope=json.loads(Path(args.scope).read_text())
    require(scope==launch_scope(prepared),'Scope differs from exact current checkpoint/record/source preparation')
    out=Path(prepared['output'])
    token=core.authorize_evaluation(scope,json.loads(Path(args.human).read_text()),json.loads(Path(args.review).read_text()),consumed=core.consume_once(out.with_name(out.name+'_eval_authorizations_PRIVATE')))
    return spec,prepared,scope,token


def execute(args):
    spec,prepared,scope,token=_authorized(args)
    import fcntl
    from types import SimpleNamespace
    from mri_rl import early_native_conditioning_probe_v1 as utility
    from mri_rl.paced_resources import PacedGate
    from mri_rl.continuous_rl_pilot import cache_binding_for
    from mri_rl.optimizer_checkpoint import load_optimizer_checkpoint
    from mri_rl.experiment_contract import load_contract
    from mri_rl.paired_z045_f32_backend_v1 import F32NativeGSPOBackend
    from mri_rl.mlx_continuous_gspo import is_loading_unit
    runtime=json.loads(Path(spec['runtime_spec']).read_text());out=Path(prepared['output']);started=time.monotonic();cache={'dir':runtime['cache_dir']}
    counter=EvalCounter(len(prepared['evaluated_checkpoints']))
    with Path(runtime['global_lock']).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(prepare(spec)==prepared,'Evaluation preparation changed while obtaining global lock');token.check('eval_lock')
        out.parent.mkdir(parents=True,exist_ok=True);gate=PacedGate(out.with_name(out.name+'_resources_PRIVATE.jsonl'))
        derived=cache_binding_for(SimpleNamespace(cache_dir=Path(runtime['cache_dir']),cache_certificate=Path(runtime['cache_certificate']),cache_certificate_sha256=runtime['cache_certificate_sha256'],model=Path(runtime['model'])))
        require(derived==prepared['derived_cache'],'Certified base cache changed')
        limit=prepared['budget']['maximum_seconds']
        def before(name):
            require(time.monotonic()-started<=limit,'Evaluation operation-start deadline exhausted');gate.before_unit(name,loading=is_loading_unit(name,cache));require(time.monotonic()-started<=limit,'Evaluation deadline exhausted during HOLD')
            if name=='native_model_load':inherited.verify_images([r['row_PRIVATE'] for r in prepared['records_PRIVATE']],prepared['expected_image_sha256'])
        def factory(label,checkpoint):
            return F32NativeGSPOBackend(Path(checkpoint['path'])/'adapter.safetensors',{**old.PROFILE,'group_size':4,'max_tokens':256},model_path=Path(runtime['model']),
                contract=load_contract(Path(runtime['contract'])),seed=13,before_unit=before,cache=cache)
        def load_state(backend,label,checkpoint):
            path=Path(checkpoint['path']);binding=json.loads((path/'receipt.json').read_text())['binding']
            state,rng,_=load_optimizer_checkpoint(path/'optimizer',expected_bindings={'run':binding,'adapter_sha256':old.digest(path/'adapter.safetensors'),'trainable_signature':backend.trainable_signature()},load_arrays=backend.load_arrays)
            backend.restore(state,rng)
        def releaser(run,resident,directory):
            result=inherited.run_with_release(run,resident,directory);utility.durable(Path(directory)/'complete_PRIVATE.json',result);return result
        results=run_checkpoints(prepared,token,backend_factory=factory,runner=lambda backend,p,label,checkpoint,rows,c,b:matched.run_checkpoint(backend,p,label,checkpoint,rows,c,b),
                                before=before,out=out,counter=counter,releaser=releaser,load_state=load_state)
        require(prepare(spec)==prepared,'Frozen sources/artifacts changed during evaluation')
        analysis=analyze(out,prepared['arms'],prepared['aliases_PRIVATE'])
        utility.durable(out/'complete_PRIVATE.json',{'status':'ACTUAL_ACCURACY_RECOVERY_V1_MATCHED_F32_EVALUATION_COMPLETE','scope_sha256':core.scope_digest(scope),'checkpoints':results,
                                                    'aliases_of_parent':analysis['aliases_of_parent'],'actual_calls':counter.values,'optimizer_updates':0,'analysis':analysis})
        return analysis


def main(argv=None,*,start=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','run','analyze'))
    for name in ('spec','output','scope','human','review','evaluation-output'):parser.add_argument('--'+name,type=Path)
    for name in ('scope','human','review'):parser.add_argument('--expected-'+name+'-sha256')
    args=parser.parse_args(argv)
    if args.action=='run':
        print(json.dumps(execute(args),sort_keys=True));return
    require(args.spec,'Specification required');spec=json.loads(args.spec.read_text())
    if args.action=='analyze':
        require(args.evaluation_output,'Evaluation output directory required');prepared=prepare(spec,start=start)
        print(json.dumps(analyze(args.evaluation_output,prepared['arms'],prepared['aliases_PRIVATE']),sort_keys=True));return
    prepared=prepare(spec,start=start)
    require(args.output and not args.output.exists(),'Fresh private CPU preparation output required');core.require_canonical_output(args.output,start=start)
    args.output.parent.mkdir(parents=True,exist_ok=True);core.publish_once(args.output,prepared)
    if not prepared['blockers']:core.publish_once(args.output.with_name(args.output.stem+'_scope.json'),launch_scope(prepared))
    print(json.dumps({'status':'PREPARED_BLOCKED_PENDING_INPUTS' if prepared['blockers'] else prepared['implementation_status'],'blockers':prepared['blockers'],'model_loads':0,'optimizer_updates':0,'model_PASS':False}))


if __name__=='__main__':main()
