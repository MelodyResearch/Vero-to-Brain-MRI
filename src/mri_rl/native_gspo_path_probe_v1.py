"""Bounded read-only saved-trajectory GSPO path qualification.

Import is CPU-only. Native calls require the separately reviewed CLI. No rollout,
sampler, optimizer.update, model save or training entry point is used here.
"""
from __future__ import annotations
import ast
import gc
import hashlib
import inspect
import json
import math
import re
import textwrap
import time
import traceback
from pathlib import Path

CONDITIONS = ('REAL', 'ZERO_NORMALIZED_PIXELS')
SUPPORTS = ('ALL_RESPONSE_TOKENS', 'LABEL_OVERLAP_TOKENS')
LOGP_MAX_ABS_TOLERANCE = .002
MAX_SECONDS = 1800
HISTORICAL_EOS_IDS = (151645, 151643)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def clear_unwound_exception_frames(error):
    """Drop tensors in completed frames after durable evidence/integrity capture.

    Keep the exception and its traceback locations for the original failure;
    traceback.clear_frames leaves currently executing frames alone. Walk chains
    because a secondary diagnostic failure can retain the primary graph too.
    """
    pending, seen = [error], set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current)); traceback.clear_frames(current.__traceback__)
        pending.extend((current.__cause__, current.__context__))


def select_first_groups(ledger):
    """Frozen before any native result: first informative accepted per task.

    Stable ledger order is the historical arm/seed/group order. No validation or
    final100 outcome, CE result or gradient is used to select examples.
    """
    selected = []
    for task in ('mri_sequence', 'mri_visit_correspondence'):
        row = next((r for r in ledger if r['task'] == task and r['accepted'] and r['informative_advantages']), None)
        require(row is not None, 'No accepted informative TRAIN group for ' + task)
        require(len(row['advantages']) == len(row['lengths']) == 4 and
                any(a > 0 for a in row['advantages']) and any(a < 0 for a in row['advantages']), 'K4 signed credit required')
        selected.append(row)
    return selected


def finite_logp_comparison(left, right):
    require(len(left) == len(right) > 0 and all(math.isfinite(x) and x <= 1e-6 for x in [*left, *right]),
            'Nonfinite/positive/misaligned conditional log probabilities')
    delta = [b-a for a, b in zip(left, right)]
    return {'tokens': len(delta), 'max_abs_delta': max(map(abs, delta)),
            'mean_abs_delta': math.fsum(map(abs, delta))/len(delta), 'delta': delta,
            'within_frozen_tolerance': max(map(abs, delta)) <= LOGP_MAX_ABS_TOLERANCE,
            'tolerance': LOGP_MAX_ABS_TOLERANCE}


def label_support(tokens, decode, task):
    """Character-overlap mask, including any tokenizer-boundary overlap.

    This is a derivative projection of GSPO, not a second target/CE objective.
    It locates actual saved categorical text even if the historical parser gave
    zero reward (e.g. escaped space in contrast-enhanced T1).
    """
    pieces = [decode(tokens[:i]) for i in range(len(tokens)+1)]
    text = pieces[-1]
    require(all(text.startswith(p) for p in pieces) and all(len(a) <= len(b) for a, b in zip(pieces, pieces[1:])),
            'Tokenizer prefix decoding is not monotonic; label support not certified')
    from mri_rl.vero_mri_reward import _answer_and_format
    parsed, formatting = _answer_and_format(text)
    classes = {'mri_sequence': {'flair', 'contrast-enhanced t1'}, 'mri_visit_correspondence': {'same', 'different'}}
    require(task in classes, 'Unknown categorical task')
    if formatting != 1 or parsed.lower() not in classes[task]:
        return {'mask': [0]*len(tokens), 'status': 'NOT_SEPARABLE_STRICT_NONCLASS', 'label_tokens': 0,
                'response_tokens': len(tokens), 'label_character_span': None}
    answer = text.find('<answer>')
    require(answer >= 0, 'Saved answer span missing')
    matches = list(re.finditer(re.escape(parsed), text[answer:], flags=re.IGNORECASE))
    require(len(matches) == 1, 'Ambiguous categorical label span')
    start, end = matches[0].span(); start += answer; end += answer
    mask = [int(len(pieces[i]) < end and len(pieces[i+1]) > start) for i in range(len(tokens))]
    require(any(mask), 'Label token support empty')
    return {'mask': mask, 'status': 'STRICT_CLASS_CHARACTER_OVERLAP', 'label_character_span': [start, end], 'label_text': text[start:end],
            'token_boundary_overlap': True, 'label_tokens': sum(mask), 'response_tokens': len(tokens)}


def native_loss_ast(backend, scorer):
    """Extract the actual frozen native loss; replace only its scorer binding.

    The scorer returns identical forward logps while allowing a label derivative
    projection. The ratio/clipping/averaging expression is not reimplemented.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(type(backend).update)))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'loss_fn')
    from types import SimpleNamespace
    namespace = {'mx': backend.mx, 'self': SimpleNamespace(_logps=scorer)}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<existing-native-GSPO-loss-AST>', 'exec'), namespace)
    return namespace['loss_fn'], hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def budget(groups):
    lengths = [n for group in groups for n in group['lengths']]
    require(1 <= len(groups) <= 2 and len(lengths) == 4*len(groups) and all(0 < n <= 256 for n in lengths), 'Saved K4 budget invalid')
    return {'groups': len(groups), 'model_loads': 1, 'new_generated_tokens': 0, 'optimizer_updates': 0,
            'KV_prefills': len(lengths), 'KV_token_forwards': sum(n-1 for n in lengths),
            'B2_full_teacherforce_forwards': len(lengths),
            'B3_detached_old_forwards': len(lengths),  # REAL reuses B2 full; ZERO has own baseline
            'B3_backward_calls': len(lengths)*3,
            'B3_loss_forwards': len(lengths)*3,
            'simultaneous_image_autograd_graphs': 1, 'maximum_seconds': MAX_SECONDS,
            'logp_max_abs_tolerance': LOGP_MAX_ABS_TOLERANCE,
            'native_generation_or_sampler_calls': 0}


def strict_tensor_identity(backend, left, right, utility):
    require(set(left) == set(right) and {'input_ids', 'pixel_values', 'attention_mask', 'image_grid_thw'} <= set(left),
            'Generation/scorer tensor inventory differs')
    require(not {'cache', 'image_features', 'cached_image_features', 'inputs_embeds'} & set(left), 'Cached image feature route forbidden')
    a, b = utility._fingerprints(backend, left), utility._fingerprints(backend, right)
    require(a == b, 'Generation/scorer IDs/pixels/grid/mask/metadata differ')
    return a


def forced_kv_logps(backend, inputs, tokens, before, charge):
    """Pinned generation embedding + KV LM path, forced saved IDs; never sample.

    No generation distribution receipt is invented. There is one fresh cache per
    response and no speculative decoding, APC, quantized KV or shared image cache.
    """
    from mlx_vlm.models.cache import make_prompt_cache
    mx = backend.mx
    cache = make_prompt_cache(backend.model.language_model, max_kv_size=None)
    kwargs = {k: v for k, v in inputs.items() if k not in ('input_ids', 'pixel_values', 'attention_mask')}
    before('B2_KV_embedding'); charge('KV_prefills')
    embedding = backend.model.get_input_embeddings(inputs['input_ids'], inputs['pixel_values'],
                                                   mask=inputs['attention_mask'], **kwargs)
    kwargs.update({k: v for k, v in embedding.to_dict().items() if k != 'inputs_embeds' and v is not None})
    y, embeds = inputs['input_ids'], embedding.inputs_embeds
    native_values, stable_values, dtypes = [], [], []
    for index, token in enumerate(tokens):
        before('B2_KV_saved_token_' + str(index))
        if index: charge('KV_token_forwards')
        output = backend.model.language_model(y, inputs_embeds=embeds, cache=cache, logits_to_keep=1, **kwargs)
        logits = output.logits[:, -1, :]
        # Primary follows installed ar._step exactly: native dtype, then direct
        # logsumexp/subtraction. F32 normalization is a separate same-logit
        # diagnostic and adds no model forward.
        native = (logits - mx.logsumexp(logits, axis=-1, keepdims=True))[0, token]
        stable_logits = logits.astype(mx.float32)
        stable = (stable_logits - mx.logsumexp(stable_logits, axis=-1, keepdims=True))[0, token]
        mx.eval(native, stable, [c.state for c in cache])
        native_values.append(float(native.item())); stable_values.append(float(stable.item()))
        dtypes.append({'logits': str(logits.dtype), 'native_logp': str(native.dtype), 'stable_logp': str(stable.dtype)})
        # Mirrors generation ar._step: Qwen VL has neither cross-attention nor encoder state.
        require(output.cross_attention_states is None and output.encoder_outputs is None, 'Unexpected conditional cache route')
        kwargs = {}; embeds = None; y = mx.array([[token]], dtype=inputs['input_ids'].dtype)
        del output, logits, native, stable_logits, stable
    del embedding, cache, y, embeds, kwargs
    gc.collect(); mx.clear_cache()
    return {'native_dtype_selected_logps': native_values, 'fp32_stable_selected_logps': stable_values, 'per_token_dtypes': dtypes}


def run_group(backend, group, out, utility, before):
    """One historical TRAIN K4 group; persist per-unit evidence before comparison.

    Caller restores the exact pre-rollout Adam/Python/MLX RNG and verifies all 504
    adapter tensors against that parent's disk checkpoint before entering.
    """
    from mri_rl.optimizer_checkpoint import pack_tree
    out = Path(out); out.mkdir(exist_ok=False)
    mx = backend.mx; model = backend.model; model.eval()
    counts = {}; limits = budget([{'lengths': [len(r['token_ids']) for r in group['raw']['responses']]}])
    started = time.monotonic()
    def charge(name):
        counts[name] = counts.get(name, 0)+1
        require(counts[name] <= limits[name], 'Frozen operation budget exceeded: ' + name)
        require(time.monotonic()-started <= MAX_SECONDS, 'Frozen wall time exhausted')
    def state_fingerprint():
        layout, arrays = pack_tree({'optimizer': backend.optimizer_state(), 'rng': backend.rng_state()})
        return {'layout': layout, 'arrays': utility._fingerprints(backend, arrays)}
    before_weights = utility._weights(backend); before_state = state_fingerprint()
    saved_state, saved_rng = backend.optimizer_state(), backend.rng_state()
    prior_active = backend.active
    def integrity(stage):
        before(stage)
        require(utility._weights(backend) == before_weights, 'Read-only GSPO changed adapter tensors')
        require(state_fingerprint() == before_state, 'Read-only GSPO changed optimizer/Python/MLX RNG')
    try:
        raw = group['raw']; row = group['row']
        require(row['extra_info']['internal_split'] == 'train' and sha(row['images'][0]) == raw['image_sha256'], 'Actual TRAIN image changed')
        messages = [{'role': 'system', 'content': backend.contract.system_instruction}, {'role': 'user', 'content': row['extra_info']['question']}]
        prompt = backend.template(backend.processor, model.config, messages, num_images=1,
                                  add_generation_prompt=True, enable_thinking=backend.contract.enable_thinking)
        require(prompt == raw['rendered_prompt'] and hashlib.sha256(prompt.encode()).hexdigest() == raw['prompt_sha256'], 'Historical rendered prompt changed')
        # Execute exactly the installed generation preparation function, without generation.
        from mlx_vlm.generate.dispatch import _prepare_generation_inputs
        before('B2_generation_prepare_inputs')
        ids, pixels, mask, metadata = _prepare_generation_inputs(model, backend.processor, prompt, row['images'], None, None,
            {'max_tokens': 256, 'temperature': 1., 'top_p': 1., 'top_k': 0, 'min_p': 0., 'repetition_penalty': None})
        generation = {'input_ids': ids, 'pixel_values': pixels, 'attention_mask': mask, **metadata}
        before('B2_historical_scorer_prepare_inputs')
        scoring = backend.prepare_inputs(backend.processor, images=row['images'], prompts=prompt,
            image_token_index=model.config.image_token_index,
            add_special_tokens=backend.special_tokens(model.config.model_type, backend.processor))
        fingerprints = strict_tensor_identity(backend, generation, scoring, utility)
        require(ids.tolist() == raw['prompt_token_ids'] and bool(mx.all(mask == 1).item()), 'Historical full prompt IDs/mask differ')
        require(not fingerprints['pixel_values']['all_zero'], 'Empty real image pixels')
        utility.durable(out/'input_identity_PRIVATE.json', {'generation': fingerprints, 'scorer': utility._fingerprints(backend, scoring),
            'historical_prompt_ids_equal': True, 'historical_pixels_grid_mask_fingerprints': 'NOT_SAVED_PARTIAL'})
        P = int(ids.shape[-1]); real_old = []; token_records = []
        def visible_decode(tokens):
            require(not any(t in HISTORICAL_EOS_IDS for t in tokens[:-1]), 'Saved tokens continue after stopping EOS')
            # The historical native detokenizer omitted terminating EOS from response text.
            visible = tokens[:-1] if tokens and tokens[-1] in HISTORICAL_EOS_IDS else tokens
            return backend.processor.tokenizer.decode(visible)
        require(all(visible_decode(r['token_ids']) == r['text'] for r in raw['responses']), 'Native tokenizer differs from saved response text')
        supports = [label_support(r['token_ids'], visible_decode, row['ability']) for r in raw['responses']]
        backend.active = {'record_id': row['id'], 'inputs': scoring, 'prompt_length': P}
        for i, response in enumerate(raw['responses']):
            tokens = response['token_ids']
            kv = forced_kv_logps(backend, generation, tokens, before, charge)
            before('B2_full_teacherforce'); charge('B2_full_teacherforce_forwards')
            teacher = mx.stop_gradient(backend._logps(tokens)); mx.eval(teacher)
            values = teacher.tolist()
            comparison = finite_logp_comparison(kv['native_dtype_selected_logps'], values)
            stable_comparison = finite_logp_comparison(kv['fp32_stable_selected_logps'], values)
            dtype_comparison = finite_logp_comparison(kv['native_dtype_selected_logps'], kv['fp32_stable_selected_logps'])
            history = response.get('raw_selected_logps')
            historical = finite_logp_comparison(history, values) if history is not None else None
            record = {'response_index': i, 'token_ids': tokens, 'incremental_KV_conditional_logps': kv,
                'full_teacherforce_conditional_logps': values, 'teacherforce_logp_dtype': str(teacher.dtype),
                'comparison': comparison, 'primary_comparison': 'NATIVE_DTYPE_KV_vs_FP32_FULL_TEACHERFORCE',
                'stable_KV_vs_full_teacherforce': stable_comparison, 'native_KV_vs_stable_KV': dtype_comparison,
                'saved_generation_logp_comparison': historical,
                'historical_generation_distribution': 'PARTIAL_NOT_SAVED' if history is None else 'SAVED_VECTOR_COMPARED',
                'label_support': supports[i]}
            utility.durable(out/f'B2_response_{i}_PRIVATE.json', record)
            require(comparison['within_frozen_tolerance'] and stable_comparison['within_frozen_tolerance'] and
                    (historical is None or historical['within_frozen_tolerance']), 'B2 numerical parity failed; stop without threshold change')
            real_old.append(teacher); token_records.append(record); gc.collect(); mx.clear_cache()
        integrity('B2_integrity')
        condition_results = {}
        loss_ast = None
        for condition in CONDITIONS:
            inputs = scoring if condition == 'REAL' else {**scoring, 'pixel_values': mx.zeros_like(scoring['pixel_values'])}
            mx.eval(inputs)
            backend.active = {'record_id': row['id'], 'inputs': inputs, 'prompt_length': P}
            previous = real_old
            if condition != 'REAL':
                previous = []
                for response in raw['responses']:
                    before('B3_zero_detached_old'); charge('B3_detached_old_forwards')
                    value = mx.stop_gradient(backend._logps(response['token_ids'])); mx.eval(value)
                    require(all(math.isfinite(v) for v in value.tolist()), 'Nonfinite detached counterfactual old logp')
                    previous.append(value); mx.clear_cache()
            old_records = []
            for i, old in enumerate(previous):
                values = old.tolist()
                mean_delta_from_real = math.fsum(v-r for v, r in zip(values, real_old[i].tolist()))/len(values)
                ratio = math.exp(min(mean_delta_from_real, 10.))
                advantage = group['advantages'][i]
                old_records.append({'response_index': i, 'own_detached_old_logps': values,
                    'mean_log_ratio_relative_to_REAL_old': mean_delta_from_real, 'ratio_relative_to_REAL_old': ratio,
                    'hypothetical_REAL_old_clipped': ((advantage > 0 and ratio > 1.0004) or (advantage < 0 and ratio < .9997))})
            utility.durable(out/f'{condition}_old_reference_PRIVATE.json', old_records)
            support_results = {}
            for support in (SUPPORTS if condition == 'REAL' else SUPPORTS[:1]):
                summed = positive = negative = None; response_norms = []; loss_values = []
                for i, (response, old, advantage) in enumerate(zip(raw['responses'], previous, group['advantages'])):
                    mask_values = supports[i]['mask'] if support == SUPPORTS[1] else [1]*len(response['token_ids'])
                    derivative_mask = mx.array(mask_values, dtype=mx.float32)
                    def projected_scorer(tokens):
                        fresh = backend._logps(tokens)
                        stopped = mx.stop_gradient(fresh)
                        return stopped + derivative_mask*(fresh-stopped)
                    loss_fn, current_ast = native_loss_ast(backend, projected_scorer)
                    if loss_ast is None: loss_ast = current_ast
                    require(current_ast == loss_ast, 'Native GSPO loss AST changed during run')
                    vg = backend.nn.value_and_grad(model, loss_fn)
                    before('B3_serial_response_gradient'); charge('B3_backward_calls'); charge('B3_loss_forwards')
                    loss, gradient = vg(model, response['token_ids'], old, advantage)
                    mx.eval(loss, gradient); value = float(loss.item()); require(math.isfinite(value), 'Nonfinite native GSPO loss')
                    flat = dict(backend.tree_flatten(gradient))
                    require(set(flat) == set(before_weights) and all(bool(mx.all(mx.isfinite(v)).item()) for v in flat.values()), '504 finite gradient inventory required')
                    scaled = backend.tree_map(lambda g: g.astype(mx.float32)/4, gradient)
                    norm = math.sqrt(math.fsum(float(mx.sum(g*g).item()) for _, g in backend.tree_flatten(scaled)))
                    response_norms.append(norm); loss_values.append(value/4)
                    summed = scaled if summed is None else backend.tree_map(lambda a, b: a+b, summed, scaled)
                    if advantage > 0:
                        positive = scaled if positive is None else backend.tree_map(lambda a, b: a+b, positive, scaled)
                    else:
                        negative = scaled if negative is None else backend.tree_map(lambda a, b: a+b, negative, scaled)
                    mx.eval(summed, positive, negative)
                    # Each primitive is private, and makes signed cancellation auditable.
                    path = out/f'{condition}_{support}_response_{i}_gradient_PRIVATE.safetensors'
                    backend.save_arrays(path, dict(backend.tree_flatten(scaled))); utility._fsync(path)
                    del gradient, flat, scaled, loss, vg, loss_fn; gc.collect(); mx.clear_cache()
                paths = {}
                for name, gradient in [('group', summed), ('positive_advantage', positive), ('negative_advantage', negative)]:
                    require(gradient is not None, 'Signed group partition missing')
                    path = out/f'{condition}_{support}_{name}_gradient_PRIVATE.safetensors'
                    backend.save_arrays(path, dict(backend.tree_flatten(gradient))); utility._fsync(path); paths[name] = {'path_PRIVATE': str(path), 'sha256': sha(path)}
                group_l2 = math.sqrt(math.fsum(float(mx.sum(g*g).item()) for _, g in backend.tree_flatten(summed)))
                support_results[support] = {'group_loss': math.fsum(loss_values), 'per_response_scaled_loss': loss_values,
                    'per_response_scaled_gradient_l2': response_norms, 'group_gradient_l2': group_l2,
                    'group_to_sum_response_norms': group_l2/math.fsum(response_norms) if math.fsum(response_norms) else None,
                    'gradient_files_PRIVATE': paths, 'native_loss_ast_sha256': loss_ast}
                del summed, positive, negative, gradient; gc.collect(); mx.clear_cache()
                integrity('B3_condition_support_integrity')
            condition_results[condition] = support_results
            utility.durable(out/f'{condition}_PRIVATE.json', support_results)
            del previous, inputs
        comparisons = {}
        from mri_rl.paired_native_gradient_probe_v1 import vector_comparison
        for support in SUPPORTS[:1]:
            left = condition_results['REAL'][support]['gradient_files_PRIVATE']['group']['path_PRIVATE']
            right = condition_results['ZERO_NORMALIZED_PIXELS'][support]['gradient_files_PRIVATE']['group']['path_PRIVATE']
            comparisons[support+'_REAL_vs_ZERO'] = vector_comparison(utility._gradient_chunks(left, right))
        for condition in CONDITIONS[:1]:
            left = condition_results[condition][SUPPORTS[0]]['gradient_files_PRIVATE']['group']['path_PRIVATE']
            right = condition_results[condition][SUPPORTS[1]]['gradient_files_PRIVATE']['group']['path_PRIVATE']
            comparisons[condition+'_FULL_vs_LABEL_DERIVATIVE'] = vector_comparison(utility._gradient_chunks(left, right))
        for condition in CONDITIONS:
            for support in (SUPPORTS if condition == 'REAL' else SUPPORTS[:1]):
                files = condition_results[condition][support]['gradient_files_PRIVATE']
                comparisons[condition+'_'+support+'_POSITIVE_vs_NEGATIVE'] = vector_comparison(utility._gradient_chunks(
                    files['positive_advantage']['path_PRIVATE'], files['negative_advantage']['path_PRIVATE']))
        integrity('native_gspo_path_final_integrity')
        utility.durable(out/'integrity_PRIVATE.json', {'weights_before': before_weights, 'weights_after': utility._weights(backend),
            'state_before': before_state, 'state_after': state_fingerprint()})
        result = {'status': 'READONLY_GSPO_GROUP_COMPLETED', 'budget_actual': counts, 'comparisons': comparisons,
            'condition_results_PRIVATE': condition_results, 'native_loss_ast_sha256': loss_ast,
            'historical_generation_distribution': 'PARTIAL_NOT_SAVED' if any(r['saved_generation_logp_comparison'] is None for r in token_records) else 'COMPARED',
            'historical_full_input_fingerprints': 'PARTIAL_NOT_SAVED', 'adapter_tensors_checked': 504,
            'optimizer_Python_MLX_RNG_preserved': True, 'optimizer_updates': 0,
            'counterfactual_old_reference': 'OWN_DETACHED_PREUPDATE_CONDITION_LOGPS; same fixed historical signed advantages',
            'label_support': 'Read-only derivative projection of identical native GSPO forward; response-length denominator retained. Only strict valid categorical responses contribute; nonclass responses have zero projection and remain in unpartitioned remainder.',
            'label_support_status': 'PARTIAL_NONCLASS_RESPONSES' if any(s['status'] != 'STRICT_CLASS_CHARACTER_OVERLAP' for s in supports) else 'ALL_RESPONSES_STRICT_LABEL_SUPPORT'}
        utility.durable(out/'complete_PRIVATE.json', result)
        return result
    except BaseException as error:
        utility.durable(out/'STOP_PRIVATE.json', {'status': 'STOP_NO_RETRY', 'error_type': type(error).__name__, 'error': str(error),
            'budget_actual': counts, 'elapsed_seconds': time.monotonic()-started, 'optimizer_updates': 0})
        raise
    finally:
        # Inspect actual state before any restore; never hide a mutation by restoring it.
        try:
            actual_weights, actual_state = utility._weights(backend), state_fingerprint()
            utility.durable(out/'exit_integrity_PRIVATE.json', {'weights_before': before_weights, 'weights_after': actual_weights,
                'state_before': before_state, 'state_after': actual_state,
                'weights_unchanged': actual_weights == before_weights, 'optimizer_Python_MLX_RNG_unchanged': actual_state == before_state})
            require(actual_weights == before_weights and actual_state == before_state,
                    'Final failure-path integrity violation')
        finally:
            backend.active = prior_active
            del saved_state, saved_rng
            gc.collect(); mx.clear_cache()
