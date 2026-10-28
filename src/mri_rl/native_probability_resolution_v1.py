"""Saved-TRAIN full-vocabulary numerical resolution, CPU import only.

No sampler, generation, backward or optimizer call. Native-facing functions
operate only on an explicitly supplied resident backend and reviewed callbacks.
"""
from __future__ import annotations
import gc
import hashlib
import json
import math
from pathlib import Path
import struct

TOLERANCE = .002
MAX_SECONDS = 1800
ROUTES = ('LEGACY_WHOLE_PROMPT_KV', 'INSTALLED_CHUNKED_KV', 'F32_SHARED_EXPLICIT_CHUNKED_KV')


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as handle: return hashlib.file_digest(handle, 'sha256').hexdigest()


def budget(tokens):
    require(tokens == 17, 'Exactly the already saved first 17-token TRAIN response is authorized')
    return {'vision_embedding_calls': 2, 'language_model_calls': 55, 'text_embedding_identity_checks': 1,
            'generation_calls': 0, 'backward_calls': 0, 'optimizer_updates': 0,
            'maximum_seconds': MAX_SECONDS, 'response_tokens': tokens,
            'routes': {'legacy_whole_KV': 17, 'installed_chunked_KV': 18,
                       'F32_shared_explicit_chunked_KV': 18, 'baseline_full': 1, 'F32_shared_full': 1}}


def causal_schedule(prompt_length, response_length):
    require(type(prompt_length) is int and prompt_length > 0 and 0 < response_length <= 17, 'Invalid causal schedule')
    return [{'response_index': i, 'input_start': 0 if i == 0 else prompt_length+i-1,
             'input_end': prompt_length if i == 0 else prompt_length+i,
             'score_position': prompt_length+i-1} for i in range(response_length)]


def prefix_features(features, length):
    require(length > 0, 'Nonempty feature prefix required')
    result = dict(features)
    result['inputs_embeds'] = features['inputs_embeds'][:, :length]
    for name in ('position_ids', 'visual_pos_masks'):
        if features.get(name) is not None: result[name] = features[name][..., :length]
    return result


def activation_control(features, dtype):
    result = dict(features)
    result['inputs_embeds'] = features['inputs_embeds'].astype(dtype)
    if features.get('deepstack_visual_embeds') is not None:
        result['deepstack_visual_embeds'] = [v.astype(dtype) for v in features['deepstack_visual_embeds']]
    return result


def distribution_comparison(left, right, selected):
    """Treat each input as categorical scores and normalize in CPU float64.

    For native BF16 logps this removes a shared additive normalization error,
    exposing any rounding-induced categorical difference. Token IDs/topk and
    per-position statistics remain PRIVATE in the caller's receipt.
    """
    import numpy as np
    left, right = np.asarray(left), np.asarray(right)
    require(left.shape == right.shape and left.ndim == 2 and left.shape[0] == len(selected) and left.shape[1] > 1,
            'Full vocabulary shape/selected alignment mismatch')
    require(np.isfinite(left).all() and np.isfinite(right).all() and all(type(t) is int and 0 <= t < left.shape[1] for t in selected),
            'Nonfinite distribution scores or invalid selected token')
    rows = []
    def normalized(scores):
        scores = scores.astype(np.float64); maximum = scores.max()
        return scores-(maximum+np.log(np.exp(scores-maximum).sum()))
    for index, token in enumerate(selected):
        a, b = left[index].astype(np.float64), right[index].astype(np.float64)
        la, lb = normalized(a), normalized(b); pa, pb = np.exp(la), np.exp(lb)
        middle = (pa+pb)/2
        # logaddexp remains finite even for vocabulary entries below exp underflow.
        lm = np.logaddexp(la, lb)-math.log(2.)
        delta = a-b; centered = delta-delta.mean()
        topn = min(5, len(a)); ta = np.argsort(-a, kind='stable')[:topn]; tb = np.argsort(-b, kind='stable')[:topn]
        row = {'position': index, 'raw_max_abs': float(np.max(np.abs(delta))),
            'common_offset': float(delta.mean()), 'centered_max_abs': float(np.max(np.abs(centered))),
            'centered_probability_weighted_RMS': float(np.sqrt(np.sum(middle*centered**2))),
            'full_vocab_normalized_logp_max_abs': float(np.max(np.abs(la-lb))),
            'selected_normalized_logp_abs': float(abs(la[token]-lb[token])),
            'KL_left_right': max(0., float(np.sum(pa*(la-lb)))),
            'KL_right_left': max(0., float(np.sum(pb*(lb-la)))),
            'JS': max(0., float((np.sum(pa*(la-lm))+np.sum(pb*(lb-lm)))/2)),
            'TV': float(np.abs(pa-pb).sum()/2), 'top1_equal': bool(ta[0] == tb[0]),
            'top5_overlap': len(set(ta.tolist()) & set(tb.tolist()))/topn,
            'top5_ids_left_PRIVATE': ta.tolist(), 'top5_ids_right_PRIVATE': tb.tolist(),
            'top2_logit_margin_abs_delta': float(abs((a[ta[0]]-a[ta[1]])-(b[tb[0]]-b[tb[1]])))}
        require(all(math.isfinite(v) for k, v in row.items() if type(v) is float), 'Nonfinite distribution statistic')
        rows.append(row)
    return {'positions': len(rows), 'vocabulary': left.shape[1],
        'max_raw_logit_delta': max(r['raw_max_abs'] for r in rows),
        'max_centered_logit_delta': max(r['centered_max_abs'] for r in rows),
        'max_centered_probability_weighted_RMS': max(r['centered_probability_weighted_RMS'] for r in rows),
        'max_full_vocab_normalized_logp_delta': max(r['full_vocab_normalized_logp_max_abs'] for r in rows),
        'max_selected_normalized_logp_delta': max(r['selected_normalized_logp_abs'] for r in rows),
        'max_KL_left_right': max(r['KL_left_right'] for r in rows), 'max_KL_right_left': max(r['KL_right_left'] for r in rows),
        'max_JS': max(r['JS'] for r in rows), 'max_TV': max(r['TV'] for r in rows),
        'top1_equal_fraction': sum(r['top1_equal'] for r in rows)/len(rows),
        'mean_top5_overlap': sum(r['top5_overlap'] for r in rows)/len(rows),
        'max_top2_margin_delta': max(r['top2_logit_margin_abs_delta'] for r in rows), 'per_position_PRIVATE': rows}


def acceptance(comparison, *, dtype_observed, context_equal, legacy_reproduction=None, direct_stable=None):
    numeric = (comparison['max_full_vocab_normalized_logp_delta'] <= TOLERANCE and
               comparison['max_selected_normalized_logp_delta'] <= TOLERANCE)
    reproduced = (legacy_reproduction is not None and legacy_reproduction['native_selected_max_abs_delta'] <= 1e-5 and
                  legacy_reproduction['stable_selected_max_abs_delta'] <= 1e-5 and
                  legacy_reproduction['frozen_scorer_selected_max_abs_delta'] <= 1e-5)
    stable = (direct_stable is not None and direct_stable['max_full_vocab_logp_delta'] <= TOLERANCE and
              direct_stable['max_selected_logp_delta'] <= TOLERANCE)
    return {'B2_numeric_acceptable': bool(numeric and stable and reproduced and dtype_observed and context_equal),
            'full_vocab_logp_max_abs_tolerance': TOLERANCE, 'selected_logp_max_abs_tolerance': TOLERANCE,
            'direct_AR_F32_vs_frozen_centered_scorer_gate': bool(stable), 'legacy_reproduction_tolerance': 1e-5,
            'legacy_reproduction': 'EXACT_WITHIN_FROZEN_TOLERANCE' if reproduced else 'NOT_EXACT_OR_NOT_VERIFIED',
            'F32_cache_and_output_observed': bool(dtype_observed), 'exact_context_alignment': bool(context_equal),
            'B3_authorized': False, 'B3_requirement': 'Separate exact scope and independent review of actual B2 evidence; no automatic B3 execution.'}


def load_f32_arrays(path):
    """Read private diagnostic arrays on CPU; no native safetensors runtime."""
    import numpy as np
    with Path(path).open('rb') as handle:
        size = struct.unpack('<Q', handle.read(8))[0]; require(0 < size < 8*1024**2, 'Invalid private array header')
        header = json.loads(handle.read(size)); header.pop('__metadata__', None); start = 8+size
        result = {}
        for key, spec in header.items():
            require(spec['dtype'] == 'F32', 'Diagnostic vectors must be stored as F32')
            handle.seek(start+spec['data_offsets'][0]); data = handle.read(spec['data_offsets'][1]-spec['data_offsets'][0])
            result[key] = np.frombuffer(data, dtype='<f4').reshape(spec['shape'])
        return result


def feature_fingerprints(backend, utility, features):
    from mri_rl.optimizer_checkpoint import pack_tree
    layout, arrays = pack_tree(features)
    return {'layout': layout, 'arrays': utility._fingerprints(backend, arrays)}


def cache_observation(cache, expected_offset):
    rows = []
    for index, layer in enumerate(cache):
        offset = getattr(layer, '_idx', layer.offset)
        if hasattr(offset, 'item'): offset = int(offset.item())
        require(offset == expected_offset, 'Cache offset differs from actual causal prefix')
        keys, values = layer.state
        require(keys is not None and values is not None and int(keys.shape[-2]) == expected_offset and int(values.shape[-2]) == expected_offset,
                'Cache key/value context length differs')
        rows.append({'layer': index, 'class': type(layer).__name__, 'offset': offset,
                     'keys_dtype': str(keys.dtype), 'values_dtype': str(values.dtype),
                     'keys_shape': list(keys.shape), 'values_shape': list(values.shape)})
    return rows


def _save_distribution(backend, utility, path, raw, native, stable):
    mx = backend.mx
    arrays = {'raw_logits_F32': mx.stack(raw).astype(mx.float32), 'native_logps_as_F32': mx.stack(native).astype(mx.float32),
              'stable_logps_F32': mx.stack(stable).astype(mx.float32)}
    mx.eval(arrays); backend.save_arrays(path, arrays); utility._fsync(path)


def _reset_language_metadata(language):
    language._rope_deltas = None; language._position_ids = None


def _kv_route(backend, utility, inputs, prompt_features, full_features, tokens, route, out, before, charge, make_cache):
    """Saved-ID forced route; exact installed chunk order where declared."""
    mx = backend.mx; language = backend.model.language_model
    P = int(inputs['input_ids'].shape[-1]); schedule = causal_schedule(P, len(tokens))
    cache = make_cache(language, max_kv_size=None); _reset_language_metadata(language)
    require(not hasattr(backend.model, 'no_chunked_prefill') and not hasattr(language, 'no_chunked_prefill') and
            not hasattr(backend.model, 'chunked_prefill_policy') and not hasattr(language, 'chunked_prefill_policy'),
            'Unexpected runtime chunked-prefill policy; source-derived schedule not applicable')
    image_kwargs = {k: v for k, v in inputs.items() if k not in ('input_ids', 'pixel_values', 'attention_mask')}
    kwargs = {**image_kwargs, **{k: v for k, v in prompt_features.items() if k != 'inputs_embeds' and v is not None}}
    embeds = prompt_features['inputs_embeds']; y = inputs['input_ids']
    raw, native, stable, dtype_rows, trace = [], [], [], [], []
    if route != ROUTES[0]:
        before(route+'_chunk212'); charge('language_model_calls')
        output = language(inputs=y[:, :-1], inputs_embeds=embeds[:, :-1], cache=cache,
                          n_to_process=P-1, logits_to_keep=1, **kwargs)
        mx.eval([c.state for c in cache]); del output
        trace.append({'kind': 'discarded_chunk_prefill', 'input_start': 0, 'input_end': P-1,
                      'cache': cache_observation(cache, P-1)})
        y, embeds = y[:, -1:], embeds[:, -1:]
        gc.collect(); mx.clear_cache()
    for index, (token, item) in enumerate(zip(tokens, schedule)):
        before(route+'_score_'+str(index)); charge('language_model_calls')
        offset = 0 if route == ROUTES[0] and index == 0 else P-1 if index == 0 else P+index-1
        if index:
            # Check installed implicit decode position formula against the full
            # authoritative matrix before optionally passing that explicit slice.
            delta = language._rope_deltas
            require(delta is not None and bool(mx.array_equal(delta, full_features['rope_deltas']).item()), 'Decode RoPE delta changed')
            derived = mx.broadcast_to((mx.array(offset)+delta)[None, ...], (3, 1, 1))
            authoritative = full_features['position_ids'][..., offset:offset+1]
            require(bool(mx.array_equal(derived, authoritative).item()), 'Implicit cache/RoPE context differs from explicit full positions')
            if route == ROUTES[2]:
                embeds = full_features['inputs_embeds'][:, offset:offset+1]
                kwargs = {'position_ids': authoritative, 'rope_deltas': full_features['rope_deltas']}
        output = language(y, inputs_embeds=embeds, cache=cache, logits_to_keep=1, **kwargs)
        logits = output.logits[:, -1, :]
        raw_lp = logits-mx.logsumexp(logits, axis=-1, keepdims=True)
        converted = logits.astype(mx.float32)
        stable_lp = converted-mx.logsumexp(converted, axis=-1, keepdims=True)
        mx.eval(logits, raw_lp, stable_lp, [c.state for c in cache])
        require(bool(mx.all(mx.isfinite(logits)).item()) and bool(mx.all(mx.isfinite(raw_lp)).item()) and
                bool(mx.all(mx.isfinite(stable_lp)).item()), 'Nonfinite full-vocabulary KV evidence')
        raw.append(mx.stop_gradient(logits[0])); native.append(mx.stop_gradient(raw_lp[0])); stable.append(mx.stop_gradient(stable_lp[0]))
        dtype_rows.append({'position': index, 'input_embeds_dtype': str(embeds.dtype) if embeds is not None else 'IMPLICIT_EMBEDDING_WEIGHT_DTYPE',
                           'logits_dtype': str(logits.dtype), 'native_logp_dtype': str(raw_lp.dtype), 'stable_logp_dtype': str(stable_lp.dtype)})
        cache_rows = cache_observation(cache, P+index)
        trace.append({'kind': 'scored_saved_token', **item, 'actual_input_start': offset, 'cache_offset_after': P+index,
                      'cache': cache_rows if index in (0, len(tokens)-1) else None,
                      'decode_positions_equal_full_matrix': True if index else 'PREFILL_POSITION_MATRIX_SLICED_BY_NATIVE_LANGUAGE'})
        require(output.cross_attention_states is None and output.encoder_outputs is None, 'Unexpected cross/encoder cache route')
        kwargs = {}; embeds = None; y = mx.array([[token]], dtype=inputs['input_ids'].dtype)
        del output, logits, raw_lp, converted, stable_lp; gc.collect(); mx.clear_cache()
    path = out/(route+'_vocab_PRIVATE.safetensors'); _save_distribution(backend, utility, path, raw, native, stable)
    summary = {'route': route, 'array_sha256': sha(path), 'per_position_dtype_PRIVATE': dtype_rows, 'causal_cache_trace_PRIVATE': trace,
               'Q_dtype': 'NOT_DIRECTLY_OBSERVED; no projection replay or attention monkeypatch',
               'K_V_dtype': 'Observed actual per-layer cache arrays; no claim about earlier intermediate projection dtype',
               'native_sampling': 'NOT_RUN_FORCED_SAVED_IDS_ONLY'}
    utility.durable(out/(route+'_trace_PRIVATE.json'), summary)
    del cache, raw, native, stable, kwargs, y, embeds; gc.collect(); mx.clear_cache()
    return summary


def _full_route(backend, utility, full_ids, features, pixels, image_kwargs, tokens, prompt_length, name, out, before, charge):
    mx = backend.mx; language = backend.model.language_model; _reset_language_metadata(language)
    kwargs = {**image_kwargs, **{k: v for k, v in features.items() if v is not None}}
    before(name); charge('language_model_calls')
    # Mirrors installed Model.__call__: exact prepared embeddings, reset with
    # pixel_values, then LanguageModel(...,mask=ones); native wrapper builds causal mask.
    output = language(full_ids, mask=mx.ones_like(full_ids), pixel_values=pixels, **kwargs)
    logits = output.logits[0, prompt_length-1:prompt_length-1+len(tokens), :]
    require(logits.shape[0] == len(tokens), 'Causal full teacherforce slice shifted')
    native = logits-mx.logsumexp(logits, axis=-1, keepdims=True)
    fp32 = logits.astype(mx.float32); stable = fp32-mx.logsumexp(fp32, axis=-1, keepdims=True)
    frozen_scorer = backend.nn.log_softmax(fp32, axis=-1)
    mx.eval(logits, native, stable, frozen_scorer)
    require(all(bool(mx.all(mx.isfinite(v)).item()) for v in (logits, native, stable, frozen_scorer)), 'Nonfinite full teacherforce vocabulary')
    path = out/(name+'_vocab_PRIVATE.safetensors')
    arrays = {'raw_logits_F32': logits.astype(mx.float32), 'native_logps_as_F32': native.astype(mx.float32),
              'stable_logps_F32': stable, 'frozen_scorer_logps_F32': frozen_scorer}
    backend.save_arrays(path, arrays); utility._fsync(path)
    result = {'route': name, 'array_sha256': sha(path), 'logits_dtype': str(logits.dtype),
              'native_logp_dtype': str(native.dtype), 'stable_logp_dtype': str(stable.dtype),
              'frozen_scorer_logp_dtype': str(frozen_scorer.dtype),
              'normalizers': {'stable_logps_F32': 'DIRECT_AR_FP32_logit_minus_logsumexp',
                             'frozen_scorer_logps_F32': 'ACTUAL_backend.nn.log_softmax_FP32_max_centered'},
              'inputs_embeds_dtype': str(features['inputs_embeds'].dtype), 'cache': 'NONE_FULL_CAUSAL'}
    utility.durable(out/(name+'_trace_PRIVATE.json'), result)
    del output, logits, native, fp32, stable, frozen_scorer, arrays, kwargs; gc.collect(); mx.clear_cache()
    return result


def run_resolution(backend, group, old_record, out, utility, before, make_cache):
    """All numerical branches are declared before execution; B3 is absent."""
    from mri_rl.optimizer_checkpoint import pack_tree
    from mri_rl.native_gspo_path_probe_v1 import strict_tensor_identity
    import numpy as np
    out = Path(out); out.mkdir(exist_ok=False); mx = backend.mx; model = backend.model; model.eval()
    tokens = group['raw']['responses'][0]['token_ids']; limits = budget(len(tokens)); counts = {}
    def charge(name):
        counts[name] = counts.get(name, 0)+1; require(counts[name] <= limits[name], 'Frozen operation budget exceeded')
    def state_fp():
        layout, arrays = pack_tree({'optimizer': backend.optimizer_state(), 'rng': backend.rng_state()})
        return {'layout': layout, 'arrays': utility._fingerprints(backend, arrays)}
    weights_before = utility._weights(backend); state_before = state_fp()
    language = model.language_model; metadata_before = (language._rope_deltas, language._position_ids)
    traces = {}; raw = group['raw']; row = group['row']
    def integrity():
        require(utility._weights(backend) == weights_before and state_fp() == state_before, 'Diagnostic mutated adapter/Adam/Python/MLX RNG')
    try:
        require(row['extra_info']['internal_split'] == 'train' and sha(row['images'][0]) == raw['image_sha256'], 'Exact TRAIN image changed')
        messages = [{'role': 'system', 'content': backend.contract.system_instruction}, {'role': 'user', 'content': row['extra_info']['question']}]
        prompt = backend.template(backend.processor, model.config, messages, num_images=1, add_generation_prompt=True,
                                  enable_thinking=backend.contract.enable_thinking)
        require(prompt == raw['rendered_prompt'], 'Saved rendered prompt changed')
        from mlx_vlm.generate.dispatch import _prepare_generation_inputs
        before('generation_and_scorer_actual_prepare_inputs')
        ids, pixels, mask, meta = _prepare_generation_inputs(model, backend.processor, prompt, row['images'], None, None,
            {'max_tokens': 256, 'temperature': 1., 'top_p': 1., 'top_k': 0, 'min_p': 0., 'repetition_penalty': None})
        inputs = {'input_ids': ids, 'pixel_values': pixels, 'attention_mask': mask, **meta}
        scoring = backend.prepare_inputs(backend.processor, images=row['images'], prompts=prompt,
            image_token_index=model.config.image_token_index,
            add_special_tokens=backend.special_tokens(model.config.model_type, backend.processor))
        input_fp = strict_tensor_identity(backend, inputs, scoring, utility)
        require(ids.tolist() == raw['prompt_token_ids'] and bool(mx.all(mask == 1).item()), 'Saved full prompt IDs/mask differ')
        P = int(ids.shape[-1]); require(P == 213 and len(tokens) == 17 and tokens == old_record['token_ids'], 'Same first TRAIN response only')
        full_ids = mx.concatenate((ids, mx.array(tokens, dtype=ids.dtype)[None, :]), axis=1)[:, :-1]
        image_kwargs = {k: v for k, v in inputs.items() if k not in ('input_ids', 'pixel_values', 'attention_mask')}
        before('baseline_prompt_visual_embeddings'); charge('vision_embedding_calls')
        prompt_features = model.get_input_embeddings(ids, pixels, mask=mask, **image_kwargs).to_dict()
        prompt_features = {k: v for k, v in prompt_features.items() if v is not None}
        mx.eval(prompt_features)
        before('baseline_full_visual_embeddings'); charge('vision_embedding_calls')
        # Exact installed Model.__call__ does not forward its mask into this helper.
        full_features = model.get_input_embeddings(full_ids, pixels, **image_kwargs).to_dict()
        full_features = {k: v for k, v in full_features.items() if v is not None}; mx.eval(full_features)
        prompt_fp = feature_fingerprints(backend, utility, prompt_features)
        full_prefix_fp = feature_fingerprints(backend, utility, prefix_features(full_features, P))
        require(prompt_fp == full_prefix_fp, 'Prompt/full image embeddings, deepstack, positions or RoPE prefix differ')
        before('saved_response_embedding_identity'); charge('text_embedding_identity_checks')
        decoded_embeds = language.model.embed_tokens(mx.array(tokens[:-1], dtype=ids.dtype)[None, :])
        require(bool(mx.array_equal(decoded_embeds, full_features['inputs_embeds'][:, P:]).item()), 'Response input embeddings differ')
        del decoded_embeds
        utility.durable(out/'input_context_PRIVATE.json', {'prepared_inputs': input_fp, 'prompt_features': prompt_fp,
            'full_prefix_features': full_prefix_fp, 'response_embeddings_exact': True, 'tokens_PRIVATE': tokens,
            'full_position_matrix_PRIVATE': full_features['position_ids'].tolist(), 'rope_delta_PRIVATE': full_features['rope_deltas'].tolist()})
        for route in ROUTES[:2]:
            traces[route] = _kv_route(backend, utility, inputs, prompt_features, full_features, tokens, route, out, before, charge, make_cache)
            integrity()
        traces['BASELINE_FULL_TF'] = _full_route(backend, utility, full_ids, full_features, pixels, image_kwargs, tokens, P,
            'BASELINE_FULL_TF', out, before, charge); integrity()
        controlled = activation_control(full_features, mx.float32)
        preservation = {'inputs_embeds_equal_upcast': bool(mx.array_equal(controlled['inputs_embeds'], full_features['inputs_embeds'].astype(mx.float32)).item()),
            'inputs_embeds_equal_roundtrip': bool(mx.array_equal(controlled['inputs_embeds'].astype(full_features['inputs_embeds'].dtype), full_features['inputs_embeds']).item()),
            'deepstack_each_equal_upcast': [bool(mx.array_equal(a, b.astype(mx.float32)).item()) for a, b in
                zip(controlled.get('deepstack_visual_embeds', []), full_features.get('deepstack_visual_embeds', []))],
            'metadata_unchanged': feature_fingerprints(backend, utility, {k: v for k, v in full_features.items() if k not in ('inputs_embeds', 'deepstack_visual_embeds')}) ==
                feature_fingerprints(backend, utility, {k: v for k, v in controlled.items() if k not in ('inputs_embeds', 'deepstack_visual_embeds')})}
        utility.durable(out/'F32_activation_value_preservation_PRIVATE.json', preservation)
        require(preservation['inputs_embeds_equal_upcast'] and preservation['inputs_embeds_equal_roundtrip'] and
                all(preservation['deepstack_each_equal_upcast']) and preservation['metadata_unchanged'], 'F32 activation upcast changed values/metadata')
        traces[ROUTES[2]] = _kv_route(backend, utility, inputs, prefix_features(controlled, P), controlled,
            tokens, ROUTES[2], out, before, charge, make_cache); integrity()
        traces['F32_SHARED_FULL_TF'] = _full_route(backend, utility, full_ids, controlled, pixels, image_kwargs, tokens, P,
            'F32_SHARED_FULL_TF', out, before, charge); integrity()
        arrays = {name: load_f32_arrays(out/(name+'_vocab_PRIVATE.safetensors')) for name in traces}
        comparisons = {}
        for name in ROUTES:
            comparisons[name+'_F32_vs_BASELINE_FULL'] = distribution_comparison(arrays[name]['raw_logits_F32'], arrays['BASELINE_FULL_TF']['raw_logits_F32'], tokens)
            comparisons[name+'_NATIVE_RENORMALIZED_vs_SAME_LOGITS_F32'] = distribution_comparison(
                arrays[name]['native_logps_as_F32'], arrays[name]['raw_logits_F32'], tokens)
        comparisons['WHOLE_vs_INSTALLED_CHUNKED'] = distribution_comparison(arrays[ROUTES[0]]['raw_logits_F32'], arrays[ROUTES[1]]['raw_logits_F32'], tokens)
        comparisons['F32_SHARED_KV_vs_F32_SHARED_FULL'] = distribution_comparison(
            arrays[ROUTES[2]]['raw_logits_F32'], arrays['F32_SHARED_FULL_TF']['raw_logits_F32'], tokens)
        legacy = arrays[ROUTES[0]]
        selected = lambda matrix: [float(matrix[i, token]) for i, token in enumerate(tokens)]
        old_native = old_record['incremental_KV_conditional_logps']['native_dtype_selected_logps']
        old_stable = old_record['incremental_KV_conditional_logps']['fp32_stable_selected_logps']
        reproduction = {'native_selected_max_abs_delta': max(abs(a-b) for a, b in zip(selected(legacy['native_logps_as_F32']), old_native)),
                        'stable_selected_max_abs_delta': max(abs(a-b) for a, b in zip(selected(legacy['stable_logps_F32']), old_stable)),
                        'frozen_scorer_selected_max_abs_delta': max(abs(a-b) for a, b in zip(selected(arrays['BASELINE_FULL_TF']['frozen_scorer_logps_F32']), old_record['full_teacherforce_conditional_logps']))}
        left_stable = arrays[ROUTES[2]]['stable_logps_F32']; right_stable = arrays['F32_SHARED_FULL_TF']['frozen_scorer_logps_F32']
        direct_stable = {'max_full_vocab_logp_delta': float(np.max(np.abs(left_stable.astype(np.float64)-right_stable.astype(np.float64)))),
            'max_selected_logp_delta': max(abs(a-b) for a, b in zip(selected(left_stable), selected(right_stable))),
            'left_normalizer': 'DIRECT_AR_FP32', 'right_normalizer': 'ACTUAL_FROZEN_SCORER_CENTERED_FP32'}
        normalizer_controls = {}
        for name in ('BASELINE_FULL_TF', 'F32_SHARED_FULL_TF'):
            delta = arrays[name]['stable_logps_F32'].astype(np.float64)-arrays[name]['frozen_scorer_logps_F32'].astype(np.float64)
            normalizer_controls[name] = {'max_full_vocab_logp_delta': float(np.max(np.abs(delta))),
                'max_selected_logp_delta': max(abs(float(delta[i, token])) for i, token in enumerate(tokens))}
        rows = traces[ROUTES[2]]['per_position_dtype_PRIVATE']
        cache_rows = [c for event in traces[ROUTES[2]]['causal_cache_trace_PRIVATE'] for c in (event.get('cache') or [])]
        f32_observed = (all(r['logits_dtype'] == str(mx.float32) for r in rows) and
                        traces['F32_SHARED_FULL_TF']['logits_dtype'] == str(mx.float32) and
                        all(r['keys_dtype'] == r['values_dtype'] == str(mx.float32) for r in cache_rows))
        accepted = acceptance(comparisons['F32_SHARED_KV_vs_F32_SHARED_FULL'], dtype_observed=f32_observed, context_equal=True,
                              legacy_reproduction=reproduction, direct_stable=direct_stable)
        utility.durable(out/'comparisons_PRIVATE.json', {'comparisons': comparisons, 'legacy_reproduction': reproduction,
                        'direct_AR_vs_frozen_scorer_F32_comparison': direct_stable, 'normalizer_controls': normalizer_controls, 'acceptance': accepted})
        safe_comparisons = {name: {k: v for k, v in value.items() if not k.endswith('_PRIVATE')} for name, value in comparisons.items()}
        native_masses = {}
        for name, values in arrays.items():
            lp = values['native_logps_as_F32'].astype(np.float64); maximum = lp.max(axis=-1)
            logmass = maximum+np.log(np.exp(lp-maximum[:, None]).sum(axis=-1))
            native_masses[name] = {'minimum_sum_exp_native_logps': float(np.exp(logmass).min()), 'maximum_sum_exp_native_logps': float(np.exp(logmass).max())}
        result = {'schema': 'native-probability-resolution-v1', 'status': 'B2_ACCEPTABLE_PENDING_INDEPENDENT_REVIEW' if accepted['B2_numeric_acceptable'] else 'B2_RESIDUAL_NOT_RESOLVED',
            'comparisons': safe_comparisons, 'legacy_reproduction': reproduction, 'native_logp_mass': native_masses,
            'direct_AR_vs_frozen_scorer_F32_comparison': direct_stable, 'full_same_logits_normalizer_controls': normalizer_controls,
            'acceptance': accepted, 'actual_calls': counts, 'all_adapter_Adam_Python_MLX_RNG_unchanged': True,
            'B3_status': 'NOT_RUN_SEPARATE_REVIEW_REQUIRED', 'Q_dtype': 'NOT_DIRECTLY_OBSERVED',
            'causal_interpretation': 'Whole-vs-installed-chunked isolates the prefill schedule. F32/shared/explicit-position control is a bundled numerical control; a passing association does not identify a specific attention/RoPE kernel bug.',
            'historical_generation_distribution': 'NOT_SAVED; present source-derived route comparison does not recover historical sampled distribution.'}
        utility.durable(out/'measurements_SAFE.json', result)
        return result
    except BaseException as error:
        utility.durable(out/'STOP_PRIVATE.json', {'error_type': type(error).__name__, 'error': str(error), 'actual_calls': counts,
                                                'generation_calls': 0, 'backward_calls': 0, 'optimizer_updates': 0})
        raise
    finally:
        try:
            weights_after, state_after = utility._weights(backend), state_fp()
            utility.durable(out/'exit_integrity_PRIVATE.json', {'weights_before': weights_before, 'weights_after': weights_after,
                'state_before': state_before, 'state_after': state_after, 'unchanged': weights_before == weights_after and state_before == state_after})
            require(weights_before == weights_after and state_before == state_after, 'Exit adapter/Adam/RNG integrity failed')
        finally:
            language._rope_deltas, language._position_ids = metadata_before
            gc.collect(); mx.clear_cache()
