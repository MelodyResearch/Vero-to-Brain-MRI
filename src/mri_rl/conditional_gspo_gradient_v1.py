"""Conditional B3 CPU utilities/resident worker; no model loader or CLI.

This is a NEW F32-activation numerical-policy diagnostic, not an assertion that
historical BF16 generation/scoring passed. A caller must first bind accepted B2
evidence and independent exact B3 review. No optimizer or generation entry.
"""
from __future__ import annotations
import gc
import math
from pathlib import Path


def require(ok, message):
    if not ok: raise ValueError(message)


def budget(groups):
    require(type(groups) is int and 1 <= groups <= 2, 'At most two prebound historical TRAIN groups')
    return {'groups': groups, 'responses_per_group': 4, 'backward_calls': groups*12,
            'detached_old_forwards': groups*8, 'full_scoring_forwards': groups*20,
            'vision_embedding_calls': groups*20, 'language_model_calls': groups*20,
            'generation_calls': 0, 'optimizer_updates': 0, 'maximum_seconds': 1800,
            'simultaneous_image_autograd_graphs': 1}


def strict_correctness(task, response, truth):
    from mri_rl.vero_mri_reward import _answer_and_format
    classes = {'mri_sequence': {'flair', 'contrast-enhanced t1'}, 'mri_visit_correspondence': {'same', 'different'}}
    require(task in classes and truth.lower() in classes[task], 'Strict categorical route/truth required')
    answer, formatting = _answer_and_format(response)
    valid = formatting == 1 and answer.lower() in classes[task]
    return {'valid_class': valid, 'correct': bool(valid and answer.lower() == truth.lower()), 'format': formatting,
            'class_status': 'STRICT_CLASS' if valid else 'STRICT_NONCLASS_OR_FORMAT_INVALID'}


def old_current_alignment(fresh, old):
    require(len(fresh) == len(old) > 0 and all(math.isfinite(v) for v in [*fresh, *old]), 'Nonfinite/misaligned own-old/current vectors')
    error = max(abs(a-b) for a, b in zip(fresh, old))
    require(error <= 1e-5, 'Own-old/current alignment exceeds fixed 1e-5; stop')
    return {'max_abs_error': error, 'tolerance': 1e-5}


def directional_summary(chunks, advantages, correct):
    """CPU dot products from ACTUAL native gradients, not CPU gradients.

    Saved g_i = -(a_i/K) grad(mean response logp). Under descent direction -g,
    d meanlogp_i[-g] = (K/a_i) dot(g,g_i). Label projection keeps the full response
    length denominator, so label/remainder derivatives are additive components.
    """
    import numpy as np
    require(len(advantages) == len(correct) == 4 and all(math.isfinite(a) and a != 0 for a in advantages), 'K4 nonzero fixed advantages required')
    group2 = positive2 = negative2 = signed_dot = label2 = remainder2 = label_remainder_dot = 0.
    dots = [0.]*4; label_dots = [0.]*4; response2 = [0.]*4; label_response2 = [0.]*4
    max_closure = max_allowed_closure = max_label_reconstruction = 0.; tensors = 0
    unit_roundoff = float(np.finfo(np.float32).eps)/2
    gamma3 = 3*unit_roundoff/(1-3*unit_roundoff)
    for chunk in chunks:
        g = np.asarray(chunk['group'], dtype=np.float64).ravel()
        responses = [np.asarray(v, dtype=np.float64).ravel() for v in chunk['responses']]
        labels = [np.asarray(v, dtype=np.float64).ravel() for v in chunk['label_responses']]
        require(len(responses) == len(labels) == 4 and all(v.shape == g.shape for v in [*responses, *labels]) and
                all(np.isfinite(v).all() for v in [g, *responses, *labels]), 'Gradient chunk inventory/nonfinite failure')
        total = sum(responses); diff = float(np.max(np.abs(g-total), initial=0.))
        max_closure = max(max_closure, diff)
        # Three materialized F32 additions accumulate the four scaled vectors.
        # Bound roundoff by absolute component magnitudes, including cancellation;
        # relative tolerance against a near-zero final group is inappropriate.
        allowed = gamma3*sum(np.abs(v) for v in responses)+3*float(np.finfo(np.float32).smallest_subnormal)
        max_allowed_closure = max(max_allowed_closure, float(np.max(allowed, initial=0.)))
        require(bool(np.all(np.abs(g-total) <= allowed)), 'Saved group sum exceeds the declared IEEE F32 three-addition error bound')
        pos = sum((v for a, v in zip(advantages, responses) if a > 0), np.zeros_like(g))
        neg = sum((v for a, v in zip(advantages, responses) if a < 0), np.zeros_like(g))
        label = sum(labels); remainder = g-label
        max_label_reconstruction = max(max_label_reconstruction, float(np.max(np.abs(g-(label+remainder)), initial=0.)))
        group2 += float(g@g); positive2 += float(pos@pos); negative2 += float(neg@neg); signed_dot += float(pos@neg)
        label2 += float(label@label); remainder2 += float(remainder@remainder); label_remainder_dot += float(label@remainder)
        for i, (response, projected) in enumerate(zip(responses, labels)):
            dots[i] += float(g@response); label_dots[i] += float(g@projected)
            response2[i] += float(response@response); label_response2[i] += float(projected@projected)
        tensors += 1
    require(tensors > 0, 'No actual gradient chunks')
    directions = []
    for i, a in enumerate(advantages):
        value = 4/a*dots[i]; label_value = 4/a*label_dots[i]
        matches = value > 0 if correct[i] else value < 0
        directions.append({'response_index': i, 'advantage': a, 'strict_saved_correct': bool(correct[i]),
            'scaled_response_gradient_l2': math.sqrt(response2[i]), 'scaled_label_gradient_l2': math.sqrt(label_response2[i]),
            'group_dot_scaled_response_gradient': dots[i], 'group_dot_scaled_label_gradient': label_dots[i],
            'mean_logp_directional_derivative': value, 'label_supported_directional_derivative': label_value,
            'remainder_directional_derivative': value-label_value, 'direction_matches_strict_saved_correctness': bool(matches),
            'flat': value == 0.})
    return {'tensors': tensors, 'group_l2': math.sqrt(group2), 'positive_l2': math.sqrt(positive2), 'negative_l2': math.sqrt(negative2),
        'positive_negative_cosine': signed_dot/math.sqrt(positive2*negative2) if positive2*negative2 > 0 else None,
        'group_to_sum_individual_norms': math.sqrt(group2)/sum(math.sqrt(x) for x in response2) if sum(response2) > 0 else None,
        'label_group_l2': math.sqrt(label2), 'remainder_group_l2': math.sqrt(remainder2),
        'label_remainder_cosine': label_remainder_dot/math.sqrt(label2*remainder2) if label2*remainder2 > 0 else None,
        'group_response_sum_closure_max_abs': max_closure, 'maximum_float32_sum_roundoff_bound': max_allowed_closure,
        'group_sum_closure_model': 'Componentwise gamma3*sum(abs(scaled_response_components))+3*minimum_F32_subnormal; three materialized F32 additions.',
        'label_plus_remainder_closure_max_abs': max_label_reconstruction,
        'remainder_definition': 'REMAINDER_DEFINED_BY_DIFFERENCE',
        'remainder_limitation': 'Includes every non-strict-label derivative and the entire nonclass response; not an independently measured format-only gradient.',
        'responses': directions,
        'direction_definition': 'Derivative under one infinitesimal descent direction -group_gradient; no optimizer step or finite improvement inferred.'}


def f32_activation_response_logps(backend, tokens, inputs, *, evidence=None):
    """Functional Task1 F32 full-TF path; model/weight dtypes stay unchanged.

    Recompute image features inside each call; no cached-image-feature shortcut.
    As in accepted Task1 full route, normalize selected response-logit rows with
    actual backend.nn.log_softmax. Different saved response lengths are disclosed
    in the B3 scope; only the first 17-token B2 instance was KV/full qualified.
    """
    from mri_rl.native_probability_resolution_v1 import activation_control
    mx = backend.mx; prompt = inputs['input_ids']; P = int(prompt.shape[-1])
    ids = mx.concatenate((prompt, mx.array(tokens, dtype=prompt.dtype)[None, :]), axis=1)[:, :-1]
    extra = {k: v for k, v in inputs.items() if k not in ('input_ids', 'pixel_values', 'attention_mask')}
    require(not {'cache', 'cached_image_features', 'image_features', 'inputs_embeds'} & set(extra), 'Unreviewed feature/cache injection')
    features = backend.model.get_input_embeddings(ids, inputs['pixel_values'], **extra).to_dict()
    features = activation_control({k: v for k, v in features.items() if v is not None}, mx.float32)
    output = backend.model.language_model(ids, mask=mx.ones_like(ids), pixel_values=inputs['pixel_values'], **{**extra, **features})
    require(output.logits.dtype == mx.float32 and features['inputs_embeds'].dtype == mx.float32 and
            all(v.dtype == mx.float32 for v in features.get('deepstack_visual_embeds', [])), 'F32 full scorer endpoint/input dtype differs')
    require(features['position_ids'].shape[-1] == ids.shape[-1], 'Full scorer position context length differs')
    if evidence is not None:
        evidence.append({'prompt_tokens': P, 'response_tokens': len(tokens), 'full_input_tokens': int(ids.shape[-1]),
            'inputs_embeds_dtype': str(features['inputs_embeds'].dtype), 'logits_dtype': str(output.logits.dtype),
            'deepstack_dtypes': [str(v.dtype) for v in features.get('deepstack_visual_embeds', [])], 'cache': 'NONE_FULL_CAUSAL'})
    logits = output.logits[:, P-1:P-1+len(tokens), :].astype(mx.float32)
    result = mx.take_along_axis(backend.nn.log_softmax(logits, axis=-1), mx.array(tokens)[None, :, None], axis=-1)[0, :, 0]
    require(result.shape[0] == len(tokens), 'F32 response scorer alignment differs')
    return result


def _direction_chunks(utility, group_path, response_paths, label_paths):
    iterators = [utility._gradient_chunks(group_path, path) for path in [*response_paths, *label_paths]]
    for blocks in zip(*iterators):
        name, shape, group, _ = blocks[0]
        require(all(b[0] == name and b[1] == shape for b in blocks), 'Directional tensor inventory differs')
        yield {'name': name, 'group': group, 'responses': [b[3] for b in blocks[:4]], 'label_responses': [b[3] for b in blocks[4:]]}


def run_resident_group(backend, group, prepared_inputs, out, utility, before, *, scorer,
                       accepted_B2_binding, exact_B3_review_binding):
    """No loader, no automatic invocation: caller supplies reviewed state/gates."""
    require(accepted_B2_binding and exact_B3_review_binding, 'Accepted actual B2 and separate exact B3 review bindings required')
    require(scorer is f32_activation_response_logps, 'Only separately declared new F32 numerical policy scorer is supported')
    from mri_rl.native_gspo_path_probe_v1 import label_support, native_loss_ast
    from mri_rl.optimizer_checkpoint import pack_tree
    from mri_rl.paired_native_gradient_probe_v1 import vector_comparison
    mx = backend.mx; out = Path(out); out.mkdir(exist_ok=False); raw = group['raw']; row = group['row']; advantages = group['advantages']
    require(len(raw['responses']) == len(advantages) == 4 and row['extra_info']['internal_split'] == 'train', 'Actual historical TRAIN K4 required')
    require(all(a != 0 and math.isfinite(a) for a in advantages), 'Fixed informative nonzero saved advantages required')
    backend.model.eval(); limits = budget(1); counts = {'backward_calls': 0, 'detached_old_forwards': 0, 'full_scoring_forwards': 0}
    def fingerprint_state():
        layout, arrays = pack_tree({'optimizer': backend.optimizer_state(), 'rng': backend.rng_state()})
        return {'layout': layout, 'arrays': utility._fingerprints(backend, arrays)}
    weights_before = utility._weights(backend); state_before = fingerprint_state()
    all_before = utility._fingerprints(backend, dict(backend.tree_flatten(backend.model.parameters())))
    require(len(all_before) == 1942, 'All 1942 model parameters required')
    language = backend.model.language_model; runtime_before = (language._rope_deltas, language._position_ids)
    def integrity():
        require(utility._weights(backend) == weights_before and fingerprint_state() == state_before, 'B3 mutated adapter/Adam/Python/MLX RNG')
    score_evidence = []
    def score(tokens):
        counts['full_scoring_forwards'] += 1
        require(counts['full_scoring_forwards'] <= limits['full_scoring_forwards'], 'B3 full-forward budget exceeded')
        return scorer(backend, tokens, condition_inputs, evidence=score_evidence)
    def visible_decode(tokens):
        require(not any(t in (151643, 151645) for t in tokens[:-1]), 'Saved response after EOS')
        return backend.processor.tokenizer.decode(tokens[:-1] if tokens[-1] in (151643, 151645) else tokens)
    supports = [label_support(r['token_ids'], visible_decode, row['ability']) for r in raw['responses']]
    correctness = [strict_correctness(row['ability'], r['text'], row['reward_model']['ground_truth']) for r in raw['responses']]
    require(all(visible_decode(r['token_ids']) == r['text'] for r in raw['responses']), 'Saved response decoding differs')
    conditions = {}; loss_ast = None
    try:
        for condition in ('REAL', 'ZERO_NORMALIZED_PIXELS'):
            condition_inputs = prepared_inputs if condition == 'REAL' else {**prepared_inputs, 'pixel_values': mx.zeros_like(prepared_inputs['pixel_values'])}
            mx.eval(condition_inputs)
            utility.validate_condition_fingerprints(utility._fingerprints(backend, prepared_inputs), utility._fingerprints(backend, condition_inputs)) if condition != 'REAL' else None
            old = []
            for response in raw['responses']:
                before('B3_'+condition+'_own_detached_old'); counts['detached_old_forwards'] += 1
                require(counts['detached_old_forwards'] <= limits['detached_old_forwards'], 'B3 own-old budget exceeded')
                value = mx.stop_gradient(score(response['token_ids'])); mx.eval(value)
                require(bool(mx.all(mx.isfinite(value)).item()), 'Nonfinite condition-own old logp')
                old.append(value); gc.collect(); mx.clear_cache()
            utility.durable(out/(condition+'_old_PRIVATE.json'), [v.tolist() for v in old])
            condition_result = {}
            for support in (('FULL', 'STRICT_LABEL_DERIVATIVE') if condition == 'REAL' else ('FULL',)):
                summed = positive = negative = None; paths = []; losses = []; errors = []
                for i, (response, previous, a) in enumerate(zip(raw['responses'], old, advantages)):
                    derivative_mask = mx.array(supports[i]['mask'] if support != 'FULL' else [1]*len(response['token_ids']), dtype=mx.float32)
                    fresh_capture = []
                    def projected(tokens):
                        fresh = score(tokens)
                        fresh_capture.append(mx.stop_gradient(fresh))
                        # Same forward objective; derivative restricted only for strict-label projection.
                        stopped = mx.stop_gradient(fresh)
                        return stopped+derivative_mask*(fresh-stopped)
                    objective, ast_sha = native_loss_ast(backend, projected)
                    if loss_ast is None: loss_ast = ast_sha
                    require(ast_sha == loss_ast, 'Native GSPO loss AST differs')
                    vg = backend.nn.value_and_grad(backend.model, objective)
                    before('B3_'+condition+'_'+support+'_response_'+str(i)); counts['backward_calls'] += 1
                    require(counts['backward_calls'] <= limits['backward_calls'], 'B3 backward budget exceeded')
                    value, grad = vg(backend.model, response['token_ids'], previous, a); mx.eval(value, grad)
                    require(math.isfinite(float(value.item())), 'Nonfinite GSPO loss')
                    require(len(fresh_capture) == 1, 'Expected one scorer forward in each GSPO loss')
                    mx.eval(fresh_capture)
                    alignment = old_current_alignment(fresh_capture[0].tolist(), previous.tolist()); errors.append(alignment['max_abs_error'])
                    leaves = dict(backend.tree_flatten(grad))
                    require(set(leaves) == set(weights_before) and all(bool(mx.all(mx.isfinite(v)).item()) for v in leaves.values()), '504 finite native gradient tensors required')
                    scaled = backend.tree_map(lambda g: g.astype(mx.float32)/4, grad)
                    summed = scaled if summed is None else backend.tree_map(lambda x, y: x+y, summed, scaled)
                    if a > 0: positive = scaled if positive is None else backend.tree_map(lambda x, y: x+y, positive, scaled)
                    else: negative = scaled if negative is None else backend.tree_map(lambda x, y: x+y, negative, scaled)
                    mx.eval(summed, positive, negative); path = out/f'{condition}_{support}_response_{i}_gradient_PRIVATE.safetensors'
                    backend.save_arrays(path, dict(backend.tree_flatten(scaled))); utility._fsync(path); paths.append(str(path)); losses.append(float(value.item())/4)
                    del value, grad, leaves, scaled, vg, objective, fresh_capture; gc.collect(); mx.clear_cache()
                files = {}
                for name, gradient in [('group', summed), ('positive', positive), ('negative', negative)]:
                    path = out/f'{condition}_{support}_{name}_gradient_PRIVATE.safetensors'
                    backend.save_arrays(path, dict(backend.tree_flatten(gradient))); utility._fsync(path); files[name] = str(path)
                condition_result[support] = {'response_paths_PRIVATE': paths, 'gradient_files_PRIVATE': files, 'losses': losses,
                                             'old_current_logp_max_abs_errors': errors, 'native_loss_AST_sha256': loss_ast}
                del summed, positive, negative, gradient; gc.collect(); mx.clear_cache(); integrity()
            conditions[condition] = condition_result
            utility.durable(out/(condition+'_PRIVATE.json'), condition_result)
            del old, condition_inputs
        real = conditions['REAL']['FULL']; zero = conditions['ZERO_NORMALIZED_PIXELS']['FULL']; label = conditions['REAL']['STRICT_LABEL_DERIVATIVE']
        directions = directional_summary(_direction_chunks(utility, real['gradient_files_PRIVATE']['group'], real['response_paths_PRIVATE'], label['response_paths_PRIVATE']),
                                         advantages, [r['correct'] for r in correctness])
        require(directions['tensors'] == 504, 'Complete directional gradient inventory required')
        comparison = vector_comparison(utility._gradient_chunks(real['gradient_files_PRIVATE']['group'], zero['gradient_files_PRIVATE']['group']))
        zero_signed = vector_comparison(utility._gradient_chunks(zero['gradient_files_PRIVATE']['positive'], zero['gradient_files_PRIVATE']['negative']))
        utility.durable(out/'scorer_context_dtype_PRIVATE.json', score_evidence)
        result = {'status': 'ACTUAL_NEW_F32_POLICY_READONLY_GSPO_GROUP_MEASURED', 'actual_calls': counts, 'directions': directions,
            'real_vs_zero_group_gradient': comparison, 'zero_positive_vs_negative_gradient': zero_signed, 'strict_correctness': correctness,
            'strict_label_support': [s['status'] for s in supports], 'native_loss_AST_sha256': loss_ast,
            'old_current_logp_max_abs_errors': {c: {s: v['old_current_logp_max_abs_errors'] for s, v in d.items()} for c, d in conditions.items()},
            'B2_binding': accepted_B2_binding, 'B3_review_binding': exact_B3_review_binding,
            'sample_reference': 'FIXED_SAMPLE_SURROGATE_REFERENCE: historical BF16 responses, new F32 condition-own detached old logps; no F32 on-policy rollout',
            'numerical_policy': 'NEW_F32_ACTIVATION_ACTUAL_CENTERED_SCORER; not historical BF16 parity',
            'original_BF16_gradient_impact': 'NOT_MEASURED', 'Dallas_F32_KV_parity': 'NOT_MEASURED_SAME_SCORER_CODE_ONLY',
            'context_generalization_beyond_two_groups': 'NOT_VERIFIED', 'optimizer_updates': 0}
        utility.durable(out/'measurements_SAFE.json', result); return result
    except BaseException as error:
        utility.durable(out/'STOP_PRIVATE.json', {'error_type': type(error).__name__, 'error': str(error), 'actual_calls': counts, 'optimizer_updates': 0})
        raise
    finally:
        try:
            integrity(); all_after = utility._fingerprints(backend, dict(backend.tree_flatten(backend.model.parameters())))
            utility.durable(out/'exit_integrity_PRIVATE.json', {'all_parameters_before': all_before, 'all_parameters_after': all_after,
                'state_before': state_before, 'state_after': fingerprint_state(), 'unchanged': all_before == all_after and fingerprint_state() == state_before})
            require(all_before == all_after, 'B3 changed frozen/base model parameter')
        finally:
            language._rope_deltas, language._position_ids = runtime_before
            gc.collect(); mx.clear_cache()
