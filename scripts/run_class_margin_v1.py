"""Prepare / exact-review-gated saved-GSPO class-margin diagnostic, no update."""
from __future__ import annotations
import argparse
import fcntl
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback

W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W/'src'))
from mri_rl import class_margin_v1 as p
from mri_rl.native_gspo_path_probe_v1 import canonical, sha, clear_unwound_exception_frames, strict_tensor_identity
require = p.require


def load(path): return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
    if path.exists(): require(path.read_bytes() == data, 'Preserve prior scope/output')
    else:
        with path.open('xb') as handle: handle.write(data); handle.flush(); os.fsync(handle.fileno())


def verify(scope):
    for path, expected in {**scope['bound_artifacts'], **scope['source_sha256']}.items():
        require(sha(path) == expected, 'Reviewed source/artifact changed: '+path)


def prepare(a):
    from tokenizers import Tokenizer
    oldpath=a.historical_scope.resolve(); old=load(oldpath)
    require(old['schema']=='conditional-f32-fixed-sample-gspo-v2' and len(old['groups_PRIVATE'])==2,'Original completed two TRAIN groups required')
    oldroot=Path(old['worktree']); saved=Path(old['output_directory'])
    proofpath=oldroot/'results/CONDITIONAL_GSPO_GRADIENT_ACTUAL_INDEPENDENT_REVIEW_V2.json'
    proof=load(proofpath)
    require(proof['verdict']=='PASS_WITH_LIMITATIONS', 'Actual B3 independent acceptance required')
    require(proof['scope_file_sha256']==sha(oldpath) and proof['scope_sha256']==canonical(old), 'Actual reviewed B3 scope differs')
    require((saved/'complete_SAFE.json').exists() and load(saved/'release_SAFE.json')['release_verified'],'Completed/released saved B3 required')
    # Bind the entire existing reviewed closure using authenticated digests. No
    # CPU preparation weight reads; independent review/execute rehash actual bytes.
    bound={**old['bound_artifacts'],**old['source_sha256'],str(oldpath):sha(oldpath),str(proofpath):sha(proofpath)}
    for path in saved.rglob('*'):
        if path.is_file(): bound[str(path)]=sha(path)
    for path in (oldpath.parent/'actual_exit_v3_PRIVATE.json',oldpath.parent/'native_v3_stdout_PRIVATE.log'):
        bound[str(path)]=sha(path)
    require(sha(oldpath.parent/'actual_exit_v3_PRIVATE.json')==proof['actual_exit_sha256'] and sha(oldpath.parent/'native_v3_stdout_PRIVATE.log')==proof['closed_stdout_sha256'],'Reviewed B3 exit/stdout differ')
    require(load(oldpath.parent/'actual_exit_v3_PRIVATE.json')['actual_exit_code']==0,'Actual B3 exit must be successful')
    tokenizer_path=Path(old['model_path'])/'tokenizer.json'; tok=Tokenizer.from_file(str(tokenizer_path))
    encode=lambda s:tok.encode(s,add_special_tokens=False).ids
    groups=old['groups_PRIVATE']; plans=[p.token_plan(g,encode) for g in groups]
    require([g['row']['ability'] for g in groups]==['mri_sequence','mri_visit_correspondence'],'No reselection/order changes')
    require([[len(c['labels'][r]['token_ids']) for r in ('correct','wrong')] for c in plans[0]['contexts']]==[[2,5]] and
            all([len(c['labels'][r]['token_ids']) for r in ('correct','wrong')]==[1,1] for c in plans[1]['contexts']) and len(plans[1]['contexts'])==2,'Prospective tokenizer call budget differs')
    for i,(group,plan) in enumerate(zip(groups,plans)):
        text=group['raw']['responses'][0]['text']; tokens=group['raw']['responses'][0]['token_ids']
        visible=tokens[:-1] if tokens and tokens[-1] in (151643,151645) else tokens
        require(tok.decode(visible,skip_special_tokens=False)==text,'Actual saved response0 roundtrip differs')
        plan['saved_gradient_directory_PRIVATE']=str(saved/f'group_{i}_PRIVATE')
    out=a.output_directory.resolve();require(out.is_relative_to(W/'local_data') and not out.exists(),'Fresh private output required')
    sources=dict(old['source_sha256'])
    for path,expected in old['source_sha256'].items():
        candidate=Path(path)
        if candidate.is_relative_to(oldroot) and candidate.name != 'AGENTS.md':
            local=W/candidate.relative_to(oldroot)
            require(local.is_file() and sha(local)==expected, 'New worktree frozen imported source differs: '+str(local))
            sources[str(local)]=expected
    for path in (Path(__file__),W/'src/mri_rl/class_margin_v1.py',W/'tests/test_class_margin_v1.py',W/'src/mri_rl/answer_likelihood.py',W/'AGENTS.md'):
        sources[str(path)]=sha(path)
    return {**{k:old[k] for k in ('runtime_versions','model_path','contract_path','derived_cache','global_lock')},
        'schema':'saved-gspo-class-margin-v1','worktree':str(W),'primary_root':str(a.primary_root.resolve()),
        'output_directory':str(out),'groups_PRIVATE':groups,'plans_PRIVATE':plans,'bound_artifacts':bound,'source_sha256':sources,
        'budget':p.budget(),'operation_start_deadline_limitation':'1700 seconds checked before and after resource gate. Atomic final call/release may overrun; no hard wall guarantee.',
        'schedule':'Dallas savedresponse0 qualification FIRST (18 KV LM+1full,2vision), then sequence margins6 and Dallas margins8 (14vision/LM/backward). Restore each authentic historical Adam/RNG before its group. One resident identical adapter.',
        'normalization':'NEW F32 inputs_embeds/deepstack + actual native centered nn.log_softmax. No weights cast. Historical BF16 generated fixed samples; saved directions are F32 fixed-sample surrogate gradients.',
        'metrics':{'primary':'M_mean=mean(true label logps)-mean(wrong label logps), length-normalized score NOT class-event log probability',
            'secondary_sum':'M_sum=sum(true label logps)-sum(wrong label logps), joint canonical label-span log probability margin',
            'secondary_first':'M_first=first true token logp-first wrong token logp, shared next-token contrast',
            'derivative':'For each new gradient h and each saved REAL/ZERO group g, D=-dot(g,h); sum reuses n*class-mean gradient; mult-token first uses separate shared-prefix contrast gradient.',
            'reporting':'Report all mean/sum/first values/directions and sign discordance; no posthoc primary switch or selective harm claim.'},
        'gates':{'Dallas_saved_response0_F32_KV_full_full_vocab_and_selected':.002,'saved_B3_same_prefix_label_logps':.002,'shared_first_vs_each_candidate_first':.002,
                 'full_parameters':1942,'gradient_tensors':504,'nonfinite':'STOP','state':'Full parameters+Adam/Python/MLX RNG exact before/after each parent; compare before next restoration'},
        'tokens':'Only strict canonical label span scored. Prefix/suffix/EOS excluded. Candidate teacherforce inputs are prompt+prefix+label[:-1]; first contrast inputs prompt+prefix.',
        'contexts':'COMMON fixed C40 answer prefix is primary. SAVED_RESPONSE0 exact historical reasoning prefix secondary; sequence identical prefix deduplicated. REAL/ZERO retain same reasoning history.',
        'counter_definition':'LM calls count public language forward invocations, not internal gradient-checkpoint recomputation. Each margin backward has one fresh vision forward; one serial image graph.',
        'resource_contract':'Existing remaining20 lock/PacedGate/AC/no-other-heavy. One load, no retry after numeric/resource failure. Preserve original STOP and clear unwound tensor frames before release <128MiB.',
        'limitations':['No generation/update/training; local derivatives do not establish finite learning gain or greedy classification.',
            'New F32 numerical policy differs from historical BF16; not historical on-policy gradient reconstruction.',
            'Dallas one saved context qualifies only itself; new candidate context shapes additionally gated by saved label and shared-first agreement, not universal KV parity.',
            'ZERO normalized pixels are OOD; historical reasoning prefix remains fixed under image intervention, not question-only classification.',
            'Saved answer-supported gradient keeps original full-response T denominator; compare T17-rescaled reference only, never assume new-gradient equivalence from score agreement.',
            'No label/credit loss modification without actual margin harm evidence and separate CPU design. No training authorization.'],
        'mission_alignment':{'tag':'ENGINEERING_UNBLOCK','evidence_tier':'PROPOSED_NOT_RUN','question':'Determine whether existing GSPO descent harms canonical correct-vs-wrong margin rather than only whole-response likelihood.',
            'fixed':'Original two TRAIN parents/images/saved directions; strict prospective prefixes/classes; no final100.','falsifier':'Nonfinite/context/numeric/state/resource gates stop; signs for all three margins reported without selective endpoint.'},
        'tokenizer_preflight':{'native_framework_imports':0,'model_loads':0,'strict_roundtrip_boundary_verified':True,'tokenizer_sha256':sha(tokenizer_path)},
        'status':'CPU_PREPARED_REQUIRES_EXACT_REVIEW','generation_calls':0,'optimizer_updates':0}

def execute(a, scope):
    require(a.review and a.expected_review_sha256 and sha(a.review) == a.expected_review_sha256, 'Exact independent margin review required')
    review = load(a.review)
    require(review.get('verdict') in ('PASS', 'PASS_WITH_LIMITATIONS') and review.get('scope_sha256') == canonical(scope) and
            review.get('reviewer_kind') in ('isolated_codex', 'claude_code') and review.get('reviewer_id') and review.get('reviewer_findings'), 'Margin source/scope not independently approved')
    verify(scope); out = Path(scope['output_directory']); out.mkdir(parents=True, exist_ok=False)
    from mri_rl import early_native_conditioning_probe_v1 as utility
    from mri_rl.paced_resources import PacedGate
    from mri_rl.mlx_continuous_gspo import NativeGSPOBackend, is_loading_unit
    from mri_rl.experiment_contract import load_contract
    from mri_rl.likelihood_cache import verify_cache_identity
    from mri_rl.optimizer_checkpoint import load_optimizer_checkpoint
    started = time.monotonic(); cache = {'dir': scope['derived_cache']['cache_dir']}; gate = PacedGate(out/'resources_PRIVATE.jsonl')
    def before(name):
        require(time.monotonic()-started <= 1700, 'Margin operation-start deadline exhausted')
        sample = gate.before_unit(name, loading=is_loading_unit(name, cache))
        require(time.monotonic()-started <= 1700, 'Margin deadline exhausted during resource hold')
        return sample
    backend = None; state = rng = disk = actual = prepared = None; pending_error = None; results = []; full_before = None
    with Path(scope['global_lock']).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            before('native_model_load'); require({k: importlib.metadata.version(k) for k in scope['runtime_versions']} == scope['runtime_versions'], 'Runtime version changed')
            certpath = Path(scope['primary_root'])/'results/EXACT_QUANT_CACHE_CERTIFICATE.json'; cert = load(certpath); modelpath = Path(scope['model_path'])
            artifacts = {str(modelpath/name): sha(modelpath/name) for name in set(cert['non_weight_files_identical']) | {'config.json'}}
            require(verify_cache_identity(cache['dir'], certpath, scope['derived_cache']['certificate_sha256'], artifacts) == scope['derived_cache'], 'Base cache changed')
            first = scope['groups_PRIVATE'][0]
            backend = NativeGSPOBackend(first['adapter_path'], {'group_size': 4, 'max_tokens': 256}, model_path=modelpath,
                contract=load_contract(Path(scope['contract_path'])), seed=7, before_unit=before, cache=cache)
            full_before = utility._fingerprints(backend, dict(backend.tree_flatten(backend.model.parameters())))
            require(len(full_before) == 1942, 'All model parameter inventory required')
            counts={}
            def charge(name):
                counts[name]=counts.get(name,0)+1; require(counts[name]<=scope['budget'][name], 'Margin call budget exceeded: '+name)
            from mlx_vlm.models.cache import make_prompt_cache
            # Qualify Dallas FIRST, then its margins; sequence follows.
            for index in (1,0):
                group=scope['groups_PRIVATE'][index]
                verify(scope); before('B3_restore_parent_'+str(index))
                disk = backend.load_arrays(group['adapter_path']); actual = dict(backend.tree_flatten(backend.model.trainable_parameters()))
                require(len(actual) == 504 and set(disk) == set(actual) and all(bool(backend.mx.array_equal(v, actual[k]).item()) for k, v in disk.items()), 'Corresponding pre-rollout adapter differs')
                disk = actual = None
                state, rng, _ = load_optimizer_checkpoint(Path(group['parent_directory'])/'optimizer',
                    expected_bindings={'run': group['binding'], 'adapter_sha256': group['adapter_sha256'], 'trainable_signature': backend.trainable_signature()}, load_arrays=backend.load_arrays)
                backend.restore(state, rng); require(int(backend.optimizer_state()['step'].item()) == group['parent_optimizer_updates'] == 0, 'Actual pre-rollout Adam0 required')
                row = group['row']; require(sha(row['images'][0]) == group['raw']['image_sha256'], 'Bound TRAIN image changed')
                prompt = backend.template(backend.processor, backend.model.config,
                    [{'role': 'system', 'content': backend.contract.system_instruction}, {'role': 'user', 'content': row['extra_info']['question']}],
                    num_images=1, add_generation_prompt=True, enable_thinking=backend.contract.enable_thinking)
                require(prompt == group['raw']['rendered_prompt'], 'Historical prompt differs')
                from mlx_vlm.generate.dispatch import _prepare_generation_inputs
                before('B3_prepare_native_inputs_'+str(index))
                ids, pixels, mask, extra = _prepare_generation_inputs(backend.model, backend.processor, prompt, row['images'], None, None,
                    {'max_tokens': 256, 'temperature': 1., 'top_p': 1., 'top_k': 0, 'min_p': 0., 'repetition_penalty': None})
                prepared = {'input_ids': ids, 'pixel_values': pixels, 'attention_mask': mask, **extra}
                scoring = backend.prepare_inputs(backend.processor, images=row['images'], prompts=prompt, image_token_index=backend.model.config.image_token_index,
                    add_special_tokens=backend.special_tokens(backend.model.config.model_type, backend.processor))
                require(ids.tolist() == group['raw']['prompt_token_ids'] and bool(backend.mx.all(mask == 1).item()), 'Historical prompt IDs/mask differ')
                strict_tensor_identity(backend, prepared, scoring, utility)
                results.append(p.run_group(backend,group,scope['plans_PRIVATE'][index],prepared,
                    scope['plans_PRIVATE'][index]['saved_gradient_directory_PRIVATE'],out/f'group_{index}_PRIVATE',utility,before,charge,
                    qualify=index==1,make_cache=make_prompt_cache))
                state = rng = prepared = scoring = ids = pixels = mask = extra = None
                gc.collect(); backend.mx.clear_cache()
            verify(scope)
        except BaseException as error:
            pending_error = error; utility.durable(out/'STOP_PRIVATE.json', {'error_type': type(error).__name__, 'error': str(error),
                'traceback': ''.join(traceback.format_exception(error)), 'completed_groups': len(results), 'optimizer_updates': 0})
            clear_unwound_exception_frames(error); raise
        finally:
            if backend is not None:
                try:
                    after = utility._fingerprints(backend, dict(backend.tree_flatten(backend.model.parameters())))
                    utility.durable(out/'all_parameter_integrity_PRIVATE.json', {'before': full_before, 'after': after, 'unchanged': full_before == after})
                    require(full_before == after, 'Full parameter integrity failure'); verify(scope)
                except BaseException as error:
                    utility.durable(out/'exit_integrity_failure_PRIVATE.json', {'error': str(error), 'original_failure_preserved': pending_error is not None})
                    clear_unwound_exception_frames(error)
                    if pending_error is None: raise
                finally:
                    original = pending_error or sys.exception()
                    if original is not None: clear_unwound_exception_frames(original)
                    state = rng = disk = actual = prepared = scoring = ids = pixels = mask = extra = None
                    try:
                        backend.close(); gc.collect(); backend.mx.clear_cache(); active = int(backend.mx.get_active_memory())
                        released = backend.model is None and active < 128*1024**2
                        utility.durable(out/'release_SAFE.json', {'model_is_none': backend.model is None, 'active_MLX_bytes': active, 'release_verified': released})
                        if original is None: require(released, 'Native release failure')
                    except BaseException as error:
                        utility.durable(out/'release_failure_PRIVATE.json', {'error': str(error), 'original_failure_preserved': original is not None})
                        clear_unwound_exception_frames(error)
                        if original is None: raise
    require(counts == {k:scope['budget'][k] for k in ('margin_backward_calls','language_model_calls','vision_embedding_calls')}, 'Final exact operation inventory differs')
    result = {'schema': scope['schema'], 'status': 'ACTUAL_SAVED_GSPO_CLASS_MARGINS_COMPLETE_RELEASED', 'groups': results,
        'scope_sha256': canonical(scope), 'all1942_model_parameters_unchanged': True, 'optimizer_updates': 0, 'generation_calls': 0,
        'budget': scope['budget'], 'actual_calls':counts, 'limitations': scope['limitations']}
    save(out/'complete_SAFE.json', result); print(json.dumps({'status': result['status'], 'receipt_sha256': sha(out/'complete_SAFE.json')}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('primary-root', 'historical-scope', 'output-directory', 'scope-output'): parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--review', type=Path); parser.add_argument('--expected-review-sha256'); parser.add_argument('--execute', action='store_true')
    a = parser.parse_args(); scope = prepare(a)
    require(a.scope_output.resolve().is_relative_to(W/'local_data'), 'Ignored private scope required'); save(a.scope_output, scope)
    if not a.execute:
        print(json.dumps({'status': 'CPU_PREPARED_NO_NATIVE', 'scope_sha256': canonical(scope), 'scope_file_sha256': sha(a.scope_output), 'budget': scope['budget']})); return
    execute(a, scope)


if __name__ == '__main__': main()
