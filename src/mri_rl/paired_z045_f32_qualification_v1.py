"""Future bounded F32 on-policy/GSPO read-only qualification, NOT_RUN.

Prepare is CPU only. Execute requires fresh human and independent exact-scope
artifacts. Never calls backend.update or optimizer.update. Actual zero-variance
or inactive groups produce PARTIAL, with no synthetic advantages/reselection.
"""
from __future__ import annotations
import argparse
import ast
import gc
import hashlib
import inspect
import json
from pathlib import Path
import textwrap
import time
import traceback
import sys
from mri_rl import paired_z045_continuation_entry_v1 as entry
from mri_rl.mlx_continuous_gspo import NativeGSPOBackend
from mri_rl.uniform_balance_continuation_gate_v1 import accuracy_group_is_informative
from mri_rl.continuous_rl import require,canonical


def budget():
    return {'groups':2,'responses_per_group':4,'model_loads':1,'generated_response_tokens':2048,
        'maximum_sampler_draws_including_installed_AR_lookahead':2056,'language_model_calls':2080,
        'vision_embedding_calls':24,'full_scorer_calls':16,'backward_calls':8,'optimizer_updates':0,
        'maximum_seconds':1200,'simultaneous_image_autograd_graphs':1}


def fixed_rows(schedule):
    return [next(row for row in schedule if row['ability']==task) for task in ('mri_sequence','mri_visit_correspondence')]


def frozen_loss_node():
    tree=ast.parse(textwrap.dedent(inspect.getsource(NativeGSPOBackend.update)))
    return next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='loss_fn')


def qualification_active(grades):return accuracy_group_is_informative(grades)


def _state(backend,utility):
    from mri_rl.optimizer_checkpoint import pack_tree
    layout,arrays=pack_tree({'optimizer':backend.optimizer_state(),'rng':backend.rng_state()})
    return {'layout':layout,'arrays':utility._fingerprints(backend,arrays)}


def _optimizer(backend,utility):
    from mri_rl.optimizer_checkpoint import pack_tree
    layout,arrays=pack_tree(backend.optimizer_state())
    return {'layout':layout,'arrays':utility._fingerprints(backend,arrays)}


def compare_full_vocab(response,full_logits,output):
    import numpy as np
    from mri_rl.native_probability_resolution_v1 import distribution_comparison
    receipt=response.get('generation_full_vocab_PRIVATE')
    require(receipt is not None,'PARTIAL_SELECTED_ONLY: actual generated full vocabulary unavailable')
    path=Path(receipt['path_PRIVATE']);require(entry.digest(path)==receipt['sha256'],'Actual generation vocabulary bytes changed')
    actual=np.memmap(path,dtype='<f4',mode='r',shape=(receipt['rows_including_unreturned_lookahead'],receipt['vocabulary']))
    full=np.asarray(full_logits,dtype=np.float32)
    require(full.shape==(receipt['returned_rows'],receipt['vocabulary']),'TF/generated vocabulary context shape differs')
    np.save(output,full,allow_pickle=False)
    result=distribution_comparison(actual[:receipt['returned_rows']],full,response['token_ids'])
    from mri_rl import early_native_conditioning_probe_v1 as utility
    utility.durable(Path(str(output)+'.comparison_PRIVATE.json'),result)
    require(result['max_full_vocab_normalized_logp_delta']<=.002 and result['max_selected_normalized_logp_delta']<=.002 and result['max_TV']<=.002,
            'F64 generated/full full-vocabulary or TV .002 gate failed')
    return {k:v for k,v in result.items() if not k.endswith('_PRIVATE')}


def run_readonly(backend,rows,out,utility,before,charge,grader):
    from mri_rl import paired_z045_continuation_driver_v1 as driver
    from mri_rl.mlx_gspo_smoke import advantages_from_rewards
    from mri_rl.paired_z045_f32_backend_v1 import generation_alignment
    from mri_rl.conditional_gspo_gradient_v2 import old_current_alignment
    from types import SimpleNamespace
    mx=backend.mx;out=Path(out);initial_state=backend.optimizer_state();initial_rng=backend.rng_state()
    state_before=_state(backend,utility);optimizer_before=_optimizer(backend,utility)
    parameters_before=utility._fingerprints(backend,dict(backend.tree_flatten(backend.model.parameters())))
    require(len(parameters_before)==1942,'Full1942 parameter inventory required')
    results=[];pending=None;backend.qualification_vocab_capture=True
    try:
        for index,row in enumerate(rows):
            group_dir=out/f'group_{index}_PRIVATE';group_dir.mkdir()
            before('qualification_group_'+str(index));raw=backend.rollout(row,129+index,group_dir)
            utility.durable(group_dir/'rollout_PRIVATE.json',raw)
            # Generation consumes real RNG, including installed lookahead draws.
            # Record it before deterministic checks; never restore to hide them.
            after_generation=_state(backend,utility)
            grades=driver._grades(row,raw,backend.config,grader)
            advantages=advantages_from_rewards([g['total'] for g in grades])
            active=qualification_active(grades);losses=[];alignments=[];own_old=[];old=[]
            full_vocab_comparisons=[]
            for response_index,response in enumerate(raw['responses']):
                capture=[];backend.active['vocab_observer']=lambda logits:capture.append(logits)
                before('qualification_detached_old');charge('full_scorer_calls');charge('language_model_calls');charge('vision_embedding_calls')
                values=mx.stop_gradient(backend._logps(response['token_ids']));mx.eval(values,capture)
                backend.active.pop('vocab_observer',None)
                require(len(capture)==1,'One full vocabulary capture from same F32 scorer forward required')
                full_vocab_comparisons.append(compare_full_vocab(response,capture[0][0],group_dir/('response_'+str(response_index)+'_full_TF_vocab_PRIVATE.npy')))
                del capture
                alignments.append(generation_alignment(response['raw_selected_logps'],values.tolist()));old.append(values)
            require(_state(backend,utility)==after_generation,'F32 scorer consumed RNG or changed Adam after actual generation')
            if active:
                require(max(advantages)-min(advantages)>1e-12,'Actual active group has zero advantages')
                for i,(response,previous,advantage) in enumerate(zip(raw['responses'],old,advantages)):
                    captured=[]
                    def score(tokens):
                        value=backend._logps(tokens);captured.append(mx.stop_gradient(value));return value
                    node=frozen_loss_node();namespace={'mx':mx,'self':SimpleNamespace(_logps=score)}
                    exec(compile(ast.Module(body=[node],type_ignores=[]),'<existing-native-GSPO-loss-AST>','exec'),namespace)
                    vg=backend.nn.value_and_grad(backend.model,namespace['loss_fn'])
                    before('qualification_readonly_gradient');charge('backward_calls');charge('full_scorer_calls');charge('language_model_calls');charge('vision_embedding_calls')
                    loss,gradient=vg(backend.model,response['token_ids'],previous,advantage);mx.eval(loss,gradient,captured)
                    require(len(captured)==1,'One fresh full scorer inside original GSPO gradient required')
                    own_old.append(old_current_alignment(captured[0].tolist(),previous.tolist()))
                    flat=dict(backend.tree_flatten(gradient));require(len(flat)==504 and all(bool(mx.all(mx.isfinite(v)).item()) for v in flat.values()) and bool(mx.isfinite(loss).item()),'Finite504 original GSPO gradient required')
                    path=group_dir/f'response_{i}_original_GSPO_gradient_PRIVATE.safetensors'
                    backend.save_arrays(path,{k:v.astype(mx.float32)/4 for k,v in flat.items()});utility._fsync(path)
                    losses.append(float(loss.item())/4)
                    del loss,gradient,flat,captured,vg,namespace;gc.collect();mx.clear_cache()
                    require(_state(backend,utility)==after_generation,'Original GSPO backward mutated Adam/RNG')
            group_result={'task':row['ability'],'active_original_gate':active,'grades':grades,'advantages':advantages,
                'generation_vs_full':alignments,'full_vocabulary_comparisons':full_vocab_comparisons,'own_old_current':own_old,'losses':losses,
                'status':'ACTUAL_QUALIFIED_GROUP' if active else 'PARTIAL_ORIGINAL_GATE_INACTIVE_NO_BACKWARD_NO_FALLBACK',
                'state_after_generation_PRIVATE':after_generation,'unchanged_after_deterministic_scoring_backward':True}
            utility.durable(group_dir/'measurements_PRIVATE.json',group_result);results.append(group_result)
            del old;gc.collect();mx.clear_cache()
        return {'status':'ACTUAL_F32_ONPOLICY_GSPO_READONLY_QUALIFIED' if len(results)==2 and all(r['active_original_gate'] for r in results) else 'PARTIAL_ACTUAL_GROUP_UNINFORMATIVE',
            'groups':results,'optimizer_updates':0,'maximum_response_tokens':256,'native_loss_AST_sha256':hashlib.sha256(ast.dump(frozen_loss_node()).encode()).hexdigest()}
    except BaseException as error:
        pending=error;utility.durable(out/'worker_STOP_PRIVATE.json',{'error':str(error),'traceback':''.join(traceback.format_exception(error)),'optimizer_updates':0});raise
    finally:
        after_parameters=utility._fingerprints(backend,dict(backend.tree_flatten(backend.model.parameters())))
        before_restore=_state(backend,utility);optimizer_after=_optimizer(backend,utility)
        unchanged=parameters_before==after_parameters and optimizer_before==optimizer_after
        utility.durable(out/'before_restore_integrity_PRIVATE.json',{'parameters_before':parameters_before,'parameters_after':after_parameters,
            'optimizer_before':optimizer_before,'optimizer_after':optimizer_after,'state_initial':state_before,'state_after_actual_generation_before_restore':before_restore,
            'parameters_Adam_unchanged':unchanged,'RNG_change':'Actual stochastic generation deliberately consumes RNG; recorded before final restoration. Deterministic scorer/backward checked against each postgeneration state.'})
        require(unchanged,'Qualification changed model/Adam; refuse restoration that could hide mutation')
        backend.restore(initial_state,initial_rng)
        after_restore=_state(backend,utility);utility.durable(out/'restored_state_PRIVATE.json',{'initial':state_before,'restored':after_restore,'unchanged':state_before==after_restore})
        require(state_before==after_restore,'Qualification failed to restore original Python/MLX RNG and Adam')
        backend.qualification_vocab_capture=False


def prepare(spec):
    prepared=entry.prepare(spec);rows=fixed_rows(prepared['schedule'])
    derived=prepared['derived_cache']
    sources={**prepared['source_sha256'],str(Path(__file__)):entry.digest(__file__)}
    return {'schema':'paired-z045-F32-readonly-qualification-v1','status':'CPU_PREPARED_NOT_APPROVED_NOT_RUN','execution_tier':'B',
        'scope_kind':'NEW_READONLY_NUMERIC_QUALIFICATION','preparation_sha256':entry.scope_digest(prepared),'source_sha256':sources,
        'parent':prepared['parent'],'expected_image_sha256':prepared['expected_image_sha256'],'derived_cache':derived,'cache_preparation_verification':prepared['cache_preparation_verification'],'rows_PRIVATE':rows,'budget':budget(),'policy':'F32_ACTIVATION_GENERATION_SCORER_GSPO_V1',
        'selection':'First sequence and first Dallas occurrence of frozen candidate nextslots8:40, no outcome-based reselection',
        'criteria':{'sampled_KV_selected_vs_full_centered_max_abs':.002,'own_old_current_max_abs':1e-5,'finite_parameters':1942,'finite_gradient_tensors':504,
            'actual_rewards_only':'Existing frozen calibrated grader+original informative gate; inactive/zero-adv group PARTIAL, no synthetic fallback/no further samples',
            'F64_normalized_full_vocabulary_max_abs':.002,'F64_full_vocabulary_TV_max':.002,'full_vocab_storage':'One CPU row streamed during AR; ≤257×vocabF32 per response including1lookahead; TF one response matrix private; no parallel graphs. Selected-only is incomplete and cannot qualify training.'},
        'RNG_contract':'Initial Adam60 Python/MLX RNG preserved as snapshot; real generation advancement captured; deterministic scorer/backward RNG unchanged from postgeneration; restore initial only AFTER full parameter+Adam comparison and durable before-restore receipts.',
        'resource_contract':'Existing remaining20.lock/PacedGate AC/nootherheavy,1200s checked before/after every gate, one model/serialimage; save STOP; no retry; fullparameter/Adam audit+explicitRNGrestore+release<128MiB.',
        'termination':'Installedstream stoppingcriteria reset resolves tokenizer+additionalEOS; capture finalEOS selectedlogp and exacttext; installedAR extra lookahead is retained and counted, not response tokens.',
        'no_training':True,'optimizer_updates':0,'model_loads_this_preparation':0,'training_authorized':False,
        'limitations':['CPU implementation is not native qualification.','Qualification changes numerical policy from historicalBF16; no performance/accuracy/clinical claim.','Only these2fixedcontexts/K4actualresponses≤256 qualified if both actual activegroups pass; no universal context guarantee.','No optimizer step, no new checkpoint weights.']}


def execute(args,spec,scope):
    import fcntl
    require(args.review and args.expected_review_sha256 and args.human_authorization and args.expected_human_sha256,'New exact human+independent read-only review required')
    require(entry.digest(args.review)==args.expected_review_sha256 and entry.digest(args.human_authorization)==args.expected_human_sha256,'Read-only approval artifacts changed')
    review=json.loads(args.review.read_text());human=json.loads(args.human_authorization.read_text());binding=entry.scope_digest(scope)
    require(review.get('scope_sha256')==binding and review.get('verdict') in ('PASS','PASS_WITH_LIMITATIONS') and review.get('reviewer_id') and review.get('reviewer_kind') in ('isolated_codex','claude_code'),'Exact independent qualification review missing')
    require(human.get('authorized') is True and human.get('action')=='readonly_numeric_qualification' and human.get('scope_sha256')==binding,'New human read-only qualification authorization missing')
    out=Path(args.native_output).resolve();require(out.is_relative_to(entry.ROOT/'local_data') and not out.exists(),'Fresh private qualification output required')
    for path,expected in scope['source_sha256'].items():require(entry.digest(path)==expected,'Qualification source closure changed')
    from mri_rl.paced_resources import PacedGate
    from mri_rl import early_native_conditioning_probe_v1 as utility
    from mri_rl.native_gspo_path_probe_v1 import clear_unwound_exception_frames
    from mri_rl.uniform_balance_continuation_gate_v1 import parent_anchor
    from mri_rl.optimizer_checkpoint import load_optimizer_checkpoint
    from mri_rl.experiment_contract import load_contract
    from mri_rl.two_task_training import make_grader
    from mri_rl.paired_z045_f32_backend_v1 import F32NativeGSPOBackend
    from mri_rl.mlx_continuous_gspo import is_loading_unit
    started=time.monotonic();counts={};limits=budget();backend=None;pending=None
    with Path(spec['global_lock']).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);out.mkdir(parents=True)
        gate=PacedGate(out/'resources_PRIVATE.jsonl');cache={'dir':spec['cache_dir']}
        def charge(name):counts[name]=counts.get(name,0)+1;require(counts[name]<=limits[name],'Qualification finite call budget exceeded: '+name)
        def before(name):
            require(time.monotonic()-started<=1200,'Qualification operation-start deadline exhausted')
            gate.before_unit(name,loading=is_loading_unit(name,cache))
            require(time.monotonic()-started<=1200,'Qualification deadline exhausted during resource HOLD')
            if name=='native_model_load':entry.verify_images(scope['rows_PRIVATE'],scope['expected_image_sha256'])
            if name=='f32_generation_LM_call':charge('language_model_calls')
            if name=='f32_generation_vision_embeddings':charge('vision_embedding_calls')
        try:
            from mri_rl.continuous_rl_pilot import cache_binding_for
            from types import SimpleNamespace
            derived=cache_binding_for(SimpleNamespace(cache_dir=Path(spec['cache_dir']),cache_certificate=Path(spec['cache_certificate']),cache_certificate_sha256=spec['cache_certificate_sha256'],model=Path(spec['model'])))
            require(derived==scope['derived_cache'],'Qualification certified cache identity changed')
            entry.verify_images(scope['rows_PRIVATE'],scope['expected_image_sha256'])
            anchor=parent_anchor(spec['parent_checkpoint'],spec['parent_receipt_sha256'])
            backend=F32NativeGSPOBackend(Path(spec['parent_checkpoint'])/'adapter.safetensors',{**entry.driver.PROFILE,'group_size':4,'max_tokens':256},model_path=Path(spec['model']),contract=load_contract(Path(spec['contract'])),seed=13,before_unit=before,cache=cache)
            state,rng,_=load_optimizer_checkpoint(Path(spec['parent_checkpoint'])/'optimizer',expected_bindings={'run':anchor['original_binding'],'adapter_sha256':anchor['adapter_sha256'],'trainable_signature':backend.trainable_signature()},load_arrays=backend.load_arrays)
            backend.restore(state,rng);state=rng=None
            backend.expected_image_sha256=scope['expected_image_sha256']
            result=run_readonly(backend,scope['rows_PRIVATE'],out,utility,before,charge,make_grader(json.loads(Path(spec['calibration']).read_text())))
        except BaseException as error:
            pending=error;utility.durable(out/'STOP_PRIVATE.json',{'error':str(error),'traceback':''.join(traceback.format_exception(error)),'actual_calls':counts,'optimizer_updates':0});clear_unwound_exception_frames(error);raise
        finally:
            if backend is not None:
                if pending:clear_unwound_exception_frames(pending)
                state=rng=None
                try:
                    backend.close();gc.collect();backend.mx.clear_cache();active=int(backend.mx.get_active_memory())
                    released=backend.model is None and active<128*1024**2
                    utility.durable(out/'release_SAFE.json',{'model_is_none':backend.model is None,'active_MLX_bytes':active,'release_verified':released})
                    if pending is None:require(released,'Qualification native release failed')
                except BaseException as error:
                    utility.durable(out/'release_failure_PRIVATE.json',{'error':str(error),'original_failure_preserved':pending is not None});clear_unwound_exception_frames(error)
                    if pending is None:raise
    for path,expected in scope['source_sha256'].items():require(entry.digest(path)==expected,'Qualification source changed during execution')
    result.update({'source_sha256':scope['source_sha256'],'scope_sha256':binding,'actual_calls':counts})
    utility.durable(out/'complete_PRIVATE.json',result)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--spec',type=Path,required=True);parser.add_argument('--scope-output',type=Path,required=True)
    parser.add_argument('--execute',action='store_true');parser.add_argument('--native-output',type=Path)
    parser.add_argument('--review',type=Path);parser.add_argument('--expected-review-sha256');parser.add_argument('--human-authorization',type=Path);parser.add_argument('--expected-human-sha256');args=parser.parse_args()
    spec=json.loads(args.spec.read_text());scope=prepare(spec);args.scope_output.parent.mkdir(parents=True,exist_ok=True)
    if args.scope_output.exists():require(json.loads(args.scope_output.read_text())==scope,'Frozen qualification scope differs')
    else:entry.driver.write_json(args.scope_output,scope)
    if args.execute:require(args.native_output is not None,'Fresh native output required');execute(args,spec,scope)
    else:print(canonical({'status':scope['status'],'scope_sha256':entry.scope_digest(scope),'budget':budget(),'model_loads':0}))

if __name__=='__main__':main()
