"""NATIVE_GROUP_SIGNAL_VJP_V1 -- CPU core of the bounded read-only A/B/C VJP experiment on Parent Adam52 + Common32.

Author: Claude Code (claude-opus-5-5), 2026-10-22. STATUS: IMPLEMENTED_CPU_TESTED_SYNTHETIC_NOT_NATIVE_RUN.
Importing this module imports no MLX/torch/JAX/transformers and loads nothing. It provides:
  * response jobs = the derived union of Common32 responses with nonzero weight in A (Direct), B (Paired) or C (frozen
    answer_matched_baseline_v1.c_plan), with identical exposure across plans (segment 0, common parent only);
  * the ACTUAL GSPO derivative per response from observed current vs SAVED own-old logps (ratio / clip / log-ratio cap),
    d loss / d mean-logp = kappa = -A*r on the active unclipped branch, else 0 (the unchanged native loss_fn semantics,
    cross-checked against paired_prompt_groups_v1.response_surrogate); ambiguous clip/cap edges are refused, never guessed;
  * plan gradients g_X = sum_units .25 * sum_i (1/8 * kappa_i) * u_i (two-level float32, Dallas included; joint gradient-norm
    clipping NONE as in the original update path), streamed one response vector at a time;
  * A/B reconstruction against the saved joint gradients (relative L2 <= 1e-4, plus max-abs / per-leaf / cosine / inventory);
  * canonical margins on the repository prefix answer_likelihood.PREFIX/SUFFIX (identical literal prefix for both labels, full
    prefix+label+suffix tokenizer boundary verified), mean/sum/first kept distinct; efficient call plan (MRI: 2 label-mean VJPs
    + 1 shared-prefix first-token contrast VJP; single-token Dallas: 1 VJP);
  * read-only Adam52 directions from the saved m52/v52/lr/betas/eps (bias_correction=False, step 52): formula delta and the
    ACTUAL float32 parameter displacement float32(theta52 + delta) - theta52 (primary), momentum-only and incremental parts;
    no optimizer call, no parameter/Adam write;
  * the frozen empirical engineering screen tau = max(1e-6, 5*e), e = max|h.(d_rec - d_saved)| over A and B (NOT a confidence
    interval or formal error bound) and the frozen per-context classification (primary MEAN margin, Adam52-rounded direction);
  * prepare() over the authenticated V2 run (CPU only), analyze() of committed native outputs (CPU, after model release),
    and a sanitized public aggregate.
"""
from __future__ import annotations
import copy
import json
import math
from pathlib import Path

from mri_rl import shared_rollout_gspo_core_v1 as shared
from mri_rl import paired_prompt_groups_v1 as ppg
from mri_rl import group_relative_visual_signal_v1 as grv
from mri_rl.answer_likelihood import PREFIX as CANONICAL_PREFIX, SUFFIX as CANONICAL_SUFFIX
from mri_rl.paired_prompt_groups_v1 import identity

require = shared.require
digest = identity
sha256_file = shared.file_sha256
SCHEMA = 'native-group-signal-vjp-scope-v1'
KIND = 'FRESH_EXACT_NATIVE_GROUP_SIGNAL_VJP_PARENT52_COMMON32_SCOPE'
ACTION = 'native_group_signal_vjp_v1_parent52_common32_ABC_readonly_once'
IMPLEMENTED_BY = 'Claude Code (claude-opus-5-5)'
STATUS = 'IMPLEMENTED_CPU_TESTED_SYNTHETIC_NOT_NATIVE_RUN'
PLANS = ('A', 'B', 'C')
METRICS = ('mean', 'sum', 'first')
PRIMARY = {'metric': 'mean', 'direction': 'adam52_rounded'}
SCREEN_FAMILIES = ('negative_gradient', 'adam52_rounded')
FAMILIES = ('negative_gradient', 'adam52_formula', 'adam52_rounded', 'momentum_only_formula', 'momentum_only_rounded',
            'adam52_incremental_formula', 'adam52_incremental_rounded')
DECISIONS = ('PASS_FOR_SMALL_PILOT', 'FAIL_REJECT_C', 'INCONCLUSIVE')
UNIT_WEIGHT = .25
PARENT_STEP = 52
JOINT_GRADIENT_NORM_CLIPPING = 'NONE (shared.update_decision only gates; apply_update calls optimizer.update directly)'
ADAM_PROFILE = {'betas': (.9, .999), 'eps': 1e-8, 'bias_correction': False, 'step': PARENT_STEP,
                'formula': 'installed MLX Adam.apply_single, bias_correction=False: m=b1*m+(1-b1)g; v=b2*v+(1-b2)g^2; delta=-lr*m/(sqrt(v)+eps)'}
TASK_LABELS = {'mri_sequence': ('FLAIR', 'contrast-enhanced T1'), 'mri_visit_correspondence': ('different', 'same')}
TOLERANCES = {'own_old_max_abs': 1e-5, 'reconstruction_relative_l2_max': 1e-4, 'tau_floor': 1e-6, 'tau_error_multiplier': 5.,
              'pair_multiplier': 2., 'clip_edge_relative_guard': 1e-6, 'native_log_ratio_crosscheck_max_abs': 1e-5,
              'first_token_alignment_max_abs': .002}
REQUIRED_CONTEXT_COUNTS = {'MRI_FLAIR': 2, 'MRI_T1': 2, 'DALLAS': 4}
TOLERANCE_BASIS = {
    'own_old_max_abs': 'historical training own-old/current gate (shared.OWN_OLD_TOLERANCE 1e-5); NOT the .002 KV probability gate',
    'reconstruction_relative_l2_max': ('engineering extension of the existing diagnosis gradient_norm_relative 1e-4 to the full vector; '
                                       'not previously demonstrated vector parity'),
    'tau': ('empirical engineering screen: tau = max(1e-6, 5*e), e = max(|h.(dA_rec-dA_saved)|, |h.(dB_rec-dB_saved)|) per context/metric/'
            'direction family; 1e-6 operational first-order floor, 5 heuristic buffer; NOT a confidence interval or formal error bound'),
    'clip_edge_relative_guard': 'ratio within 1e-6 of a clip bound or log-ratio within 1e-6 of the cap is AMBIGUOUS (F32 vs F64), never guessed',
    'native_log_ratio_crosscheck_max_abs': ('the F32 sequence log-ratio captured inside the native response VJP must agree with the independent CPU '
                                            'float64 recomputation from the same observed logps within 1e-5 (= own-old bound); else AMBIGUOUS'),
    'first_token_alignment_max_abs': ('existing frozen .002 F32 scorer alignment: EACH shared-prefix oracle first-token logp must match the first '
                                      'label token of its own label-mean forward; separate from the 1e-5 own-old gate; mismatch stops C qualification')}
FROZEN_C_SOURCE_SHA256 = 'a1a54b9a75a327f0de4815ae450e0371e534e5904082b0f6e4d2183cc7be58ac'          # committed final (docstring-only edit)
EXECUTED_C_SOURCE_SHA256 = 'd3998ac46caa23ce9659e5d6c8c5d475dba4772a955354f74c631227cc60467d'         # executed for the CPU C result
C_DOCSTRING_RECEIPT = 'results/GROUP_RELATIVE_C_DOCSTRING_PRECISION_V1.json'
INDEPENDENT_BUDGET_FACTS = 'results/NATIVE_GROUP_SIGNAL_VJP_INPUT_BUDGET_INDEPENDENT_V1.json'
FROZEN_GRV_SOURCE_SHA256 = '920a73b2f27b2fbbdca9fc4a740348d8334de9acda9ac8f40b9a3990f03242d2'
RUN_LIMITS = {'operation_window_seconds_including_waits': 3600, 'model_loads_max': 1, 'automatic_retry': False, 'automatic_recoveries': 0,
              'resume': False, 'incomplete_operation': 'INDETERMINATE_NO_REPLAY', 'one_fresh_child': True,
              'planning_estimate': ('~90 s load/release + 44 F32 full-TF forwards with VJP at ~10-15 s each (historical V2 gradient units); '
                                    'PLANNING ONLY, not a wall-time guarantee; the hard limit is the operation-start window')}
CLAIMS = {'accuracy': 'NO_NEW_ACCURACY_RESULT', 'margins': 'DIAGNOSTIC_ONLY_NOT_A_LOSS',
          'first_order': 'LOCAL_FIRST_ORDER_DIRECTIONAL_EVIDENCE_ONLY; no finite update evaluated, no training',
          'grounding': 'NOT_CLAIMED', 'existing_accuracy_context': 'Parent Adam52 18/24, A54 16/24, B54 16/24 (DEV24, unchanged)'}


# ================================================================ response jobs (Common32 union of nonzero weights)
def response_jobs(plans):
    require(set(plans) == set(PLANS), 'Exactly the A, B and C plans required')
    exposure = grv.matched_exposure(plans['A'], plans['B'])            # segment 0 common parent only; identical exposure
    grv.matched_exposure(plans['A'], plans['C'])
    rows = {x: {(r['unit'], r['key']): r for r in plans[x]} for x in PLANS}
    jobs = []
    for r in plans['A']:
        k = (r['unit'], r['key'])
        if not any(rows[x][k]['w'] > 0 for x in PLANS):
            continue
        jobs.append({'index': len(jobs), 'unit': r['unit'], 'key': r['key'], 'task': r['task'], 'image': r['image'], 'tokens': r['tokens'],
                     'weights': {x: float(rows[x][k]['w']) for x in PLANS}, 'advantages': {x: float(rows[x][k]['A']) for x in PLANS}})
    require(jobs, 'No response carries nonzero weight in any plan')
    return jobs


def job_summary(jobs):
    by = {'MRI': 0, 'DALLAS': 0}
    for j in jobs:
        by[j['task']] = by.get(j['task'], 0) + 1
    return {'responses': len(jobs), 'by_task': by, 'nonzero_weight_by_plan': {x: sum(j['weights'][x] > 0 for j in jobs) for x in PLANS}}


# ================================================================ actual GSPO derivative
def gspo_coefficient(current, old, advantage, *, native=None, guard=TOLERANCES['clip_edge_relative_guard'],
                     crosscheck=TOLERANCES['native_log_ratio_crosscheck_max_abs']):
    """d loss / d mean-logp of the unchanged native loss_fn at the OBSERVED current logps vs the SAVED own-old logps.

    native = {'log_ratio', 'ratio'} captured in F32 INSIDE the native response VJP (stop_gradient(mean(v-old)) and
    exp(min(.,10))); when supplied it is the coefficient source and the independent float64 recomputation is a cross-check
    (|difference| > 1e-5 -> AMBIGUOUS). Without it (synthetic tests) the float64 recomputation is used. Never old == current."""
    require(isinstance(current, (list, tuple)) and isinstance(old, (list, tuple)) and len(current) == len(old) > 0, 'Current/old length mismatch')
    require(all(math.isfinite(float(v)) for v in [*current, *old, advantage]), 'Nonfinite current/old logps or advantage')
    delta64 = math.fsum(float(a) - float(b) for a, b in zip(current, old)) / len(current)
    out = {'cpu_float64_log_ratio': delta64, 'max_abs_current_minus_old': max(abs(float(a) - float(b)) for a, b in zip(current, old))}
    if native is not None:
        lr, r = float(native['log_ratio']), float(native['ratio'])
        require(math.isfinite(lr) and math.isfinite(r) and r > 0, 'Nonfinite native F32 ratio capture')
        out.update(native_f32_log_ratio=lr, native_f32_ratio=r, native_minus_cpu_log_ratio=lr - delta64)
        if abs(lr - delta64) > crosscheck or abs(r - math.exp(min(lr, ppg.LOG_RATIO_MAX))) > guard * max(r, 1.):
            return {**out, 'sequence_log_ratio': lr, 'sequence_ratio': r, 'branch': 'AMBIGUOUS_NATIVE_CPU_DISCREPANCY', 'd_loss_d_mean_logp': None}
        delta, ratio, source = lr, r, 'NATIVE_F32_CAPTURE'
    else:
        delta, source = delta64, 'CPU_FLOAT64_FROM_OBSERVED_LOGPS'
        ratio = math.exp(min(delta, ppg.LOG_RATIO_MAX))
    out.update(sequence_log_ratio=delta, sequence_ratio=ratio, ratio_source=source)
    if advantage == 0:
        return {**out, 'branch': 'ZERO_ADVANTAGE', 'd_loss_d_mean_logp': 0.}
    edges = (1 - ppg.CLIP_LOW, 1 + ppg.CLIP_HIGH)
    if abs(delta - ppg.LOG_RATIO_MAX) <= guard or any(abs(ratio - e) <= guard * e for e in edges):
        return {**out, 'branch': 'AMBIGUOUS_EDGE', 'd_loss_d_mean_logp': None}
    active = delta < ppg.LOG_RATIO_MAX and ((advantage > 0 and ratio < 1 + ppg.CLIP_HIGH) or (advantage < 0 and ratio > 1 - ppg.CLIP_LOW))
    if native is None:
        require(active == ppg.response_surrogate(list(current), list(old), advantage)['clip_branch_active'], 'Branch differs from the reviewed CPU surrogate')
    branch = 'UNCLIPPED' if active else ('LOG_RATIO_CAP' if delta >= ppg.LOG_RATIO_MAX else 'CLIPPED')
    return {**out, 'branch': branch, 'd_loss_d_mean_logp': -advantage * ratio if active else 0.}


def _native(native, i):
    return None if native is None else native[i]


def plan_coefficients(jobs, *, current, old, native=None):
    """{plan: {job index: joint weight * kappa}}. Raises on an ambiguous clip/cap edge or native/CPU discrepancy (INCONCLUSIVE)."""
    out = {x: {} for x in PLANS}
    for j in jobs:
        i = j['index']
        for x in PLANS:
            w = j['weights'][x]
            if w == 0:
                out[x][i] = 0.
                continue
            c = gspo_coefficient(current[i], old[i], j['advantages'][x], native=_native(native, i))
            require(c['d_loss_d_mean_logp'] is not None, f"AMBIGUOUS coefficient ({c['branch']}) for response job {i} plan {x}")
            out[x][i] = w * c['d_loss_d_mean_logp']
    return out


def coefficient_report(jobs, *, current, old, native=None):
    rows = []
    for j in jobs:
        for x in PLANS:
            if j['weights'][x] > 0:
                c = gspo_coefficient(current[j['index']], old[j['index']], j['advantages'][x], native=_native(native, j['index']))
                rows.append({'plan': x, 'task': j['task'], 'branch': c['branch'], 'max_abs_current_minus_old': c['max_abs_current_minus_old'],
                             'native_minus_cpu': c.get('native_minus_cpu_log_ratio')})
    branches = {}
    for r in rows:
        k = r['plan'] + '|' + r['task'] + '|' + r['branch']
        branches[k] = branches.get(k, 0) + 1
    disc = [abs(r['native_minus_cpu']) for r in rows if r['native_minus_cpu'] is not None]
    return {'branch_counts': branches, 'max_abs_current_minus_old': max((r['max_abs_current_minus_old'] for r in rows), default=None),
            'ratio_source': 'NATIVE_F32_CAPTURE' if native is not None else 'CPU_FLOAT64_FROM_OBSERVED_LOGPS',
            'max_abs_native_minus_cpu_log_ratio': max(disc) if disc else None}


def native_ratios(records):
    """Captured F32 ratios from response records; None unless EVERY record carries them (synthetic fixtures may not)."""
    if all('f32_sequence_log_ratio' in r and 'f32_sequence_ratio' in r for r in records):
        return {r['index']: {'log_ratio': r['f32_sequence_log_ratio'], 'ratio': r['f32_sequence_ratio']} for r in records}
    require(not any('f32_sequence_log_ratio' in r for r in records), 'Partial native ratio capture is refused')
    return None


def joint_gradients(jobs, coeffs, *, load, names):
    """Two-level float32 accumulation (per unit: sum_i (1/8*kappa_i) u_i; joint: sum_units .25 * unit), each vector loaded once."""
    import numpy as np
    f32 = np.float32
    out = {x: {} for x in PLANS}
    units = []
    for j in jobs:
        if not units or units[-1][0] != j['unit']:
            units.append((j['unit'], []))
        units[-1][1].append(j)
    require(len({u for u, _ in units}) == len(units), 'Response jobs must be grouped by unit')
    shapes = {}
    for _, ujobs in units:
        acc = {x: {} for x in PLANS}
        for j in ujobs:
            vector = load(j['index'])
            for n in names:
                shapes.setdefault(n, np.shape(vector[n]))
            for x in PLANS:
                c = coeffs[x][j['index']]
                if c == 0:
                    continue
                s = f32(c / UNIT_WEIGHT)
                for n in names:
                    term = np.asarray(vector[n], dtype=np.float32) * s
                    acc[x][n] = term if n not in acc[x] else acc[x][n] + term
            del vector
        for x in PLANS:
            for n, a in acc[x].items():
                scaled = a * f32(UNIT_WEIGHT)
                out[x][n] = scaled if n not in out[x] else out[x][n] + scaled
    for x in PLANS:
        for n in names:
            if n not in out[x]:
                out[x][n] = np.zeros(shapes[n], dtype=np.float32)
    return out


def reconstruction_check(reconstructed, saved, *, relative_l2_max=TOLERANCES['reconstruction_relative_l2_max']):
    import numpy as np
    require(set(reconstructed) == set(saved), 'Reconstructed/saved gradient inventory differs')
    dd = ss = rr = rs = 0.
    per_leaf, finite, max_abs = {}, True, 0.
    for n in sorted(saved):
        a, b = np.asarray(reconstructed[n], dtype=np.float64), np.asarray(saved[n], dtype=np.float64)
        require(a.shape == b.shape, 'Reconstructed/saved gradient inventory (shape) differs: ' + n)
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            finite = False
            continue
        d = a - b
        dd += float(np.vdot(d, d)); ss += float(np.vdot(b, b)); rr += float(np.vdot(a, a)); rs += float(np.vdot(a, b))
        m = float(np.abs(d).max()) if d.size else 0.
        per_leaf[n] = m
        max_abs = max(max_abs, m)
    rel = math.sqrt(dd) / max(math.sqrt(ss), 1e-12) if finite else None
    cos = rs / math.sqrt(rr * ss) if finite and rr > 0 and ss > 0 else None
    return {'leaves': len(saved), 'finite': finite, 'relative_l2': rel, 'cos': cos, 'max_abs_error': max_abs if finite else None,
            'saved_l2': math.sqrt(ss), 'reconstructed_l2': math.sqrt(rr), 'relative_l2_max': relative_l2_max,
            'PASS': bool(finite and rel is not None and rel <= relative_l2_max), 'per_leaf_max_abs_PRIVATE': per_leaf}


# ================================================================ canonical margins: token plans, positions, calls, budget
def canonical_token_plan(task, truth, encode):
    require(task in TASK_LABELS and truth in TASK_LABELS[task], 'Fixed strict TRAIN class mapping required')
    wrong = next(c for c in TASK_LABELS[task] if c != truth)
    prefix_ids, suffix_ids = list(encode(CANONICAL_PREFIX)), list(encode(CANONICAL_SUFFIX))
    require(prefix_ids and suffix_ids, 'Canonical prefix/suffix must tokenize')
    labels = {}
    for role, text in (('correct', truth), ('wrong', wrong)):
        ids = list(encode(text))
        require(ids and list(encode(CANONICAL_PREFIX + text + CANONICAL_SUFFIX)) == prefix_ids + ids + suffix_ids,
                'Tokenizer boundary overlaps prefix/label/suffix: ' + text)
        labels[role] = {'text': text, 'token_ids': ids}
    require(labels['correct']['token_ids'][0] != labels['wrong']['token_ids'][0], 'First-token contrast not identifiable')
    return {'task': task, 'truth': truth, 'wrong': wrong, 'prefix': CANONICAL_PREFIX, 'prefix_ids': prefix_ids, 'suffix_ids': suffix_ids,
            'labels': labels, 'first_identifiable': True,
            'single_token_labels': len(labels['correct']['token_ids']) == len(labels['wrong']['token_ids']) == 1}


def margin_positions(prompt_length, plan):
    p = len(plan['prefix_ids'])
    return {'label_mean': {role: prompt_length + p + len(plan['labels'][role]['token_ids']) - 1 for role in ('correct', 'wrong')},
            'first': prompt_length + p}


def margin_calls(plan):
    if plan['single_token_labels']:
        return {'label_mean_vjps': 0, 'first_contrast_vjps': 1, 'forwards': 1}
    return {'label_mean_vjps': 2, 'first_contrast_vjps': 1, 'forwards': 3}


PASSTHROUGH = ('unit', 'context_key', 'task', 'truth', 'kind', 'row_id_sha256', 'recorded_context_sha256')


def margin_jobs_for_contexts(contexts):
    jobs = []
    for c in contexts:
        plan, P = c['plan'], c['prompt_length']
        pos = margin_positions(P, plan)
        base = {'context_id': c['context_id'], 'prefix_ids': plan['prefix_ids'], 'prompt_length': P, **{k: c[k] for k in PASSTHROUGH if k in c}}
        first_pair = [plan['labels']['correct']['token_ids'][0], plan['labels']['wrong']['token_ids'][0]]
        if not plan['single_token_labels']:
            for role in ('correct', 'wrong'):
                jobs.append({**base, 'index': len(jobs), 'objective': 'LABEL_MEAN', 'role': role, 'label_text': plan['labels'][role]['text'],
                             'label_ids': plan['labels'][role]['token_ids'], 'positions': pos['label_mean'][role]})
        jobs.append({**base, 'index': len(jobs), 'objective': 'FIRST_CONTRAST', 'role': 'correct_minus_wrong', 'first_pair': first_pair,
                     'single_token_labels': plan['single_token_labels'], 'positions': pos['first']})
    return jobs


def scope_budget(response_jobs, margin_jobs):
    tf = sum(j['tf_positions'] for j in response_jobs) + sum(m['positions'] for m in margin_jobs)
    n = len(response_jobs) + len(margin_jobs)
    return {'model_loads': 1, 'response_vjps': len(response_jobs), 'margin_vjps': len(margin_jobs), 'forwards': n, 'vjps': n,
            'teacherforced_positions': tf, 'generation': 0, 'optimizer_updates': 0, 'parameter_writes': 0, 'checkpoint_writes': 0,
            'simultaneous_image_autograd_graphs': 1}


def budget_for(responses, contexts):
    rj = [{**r, 'tf_positions': r.get('tf_positions', r['prompt_length'] + r['tokens'] - 1)} for r in responses]
    return scope_budget(rj, margin_jobs_for_contexts(contexts))


def margin_gradients(label_grads, *, lengths, truth, first_contrast):
    wrong = next(k for k in label_grads if k != truth)
    gc, gw, nc, nw = label_grads[truth], label_grads[wrong], lengths[truth], lengths[wrong]
    require(nc > 0 and nw > 0, 'Positive label lengths required')
    return {'mean': {n: gc[n] - gw[n] for n in gc}, 'sum': {n: nc * gc[n] - nw * gw[n] for n in gc}, 'first': first_contrast}


def margin_values(label_logps, *, truth):
    wrong = next(k for k in label_logps if k != truth)
    c, w = [float(v) for v in label_logps[truth]], [float(v) for v in label_logps[wrong]]
    require(c and w and all(math.isfinite(v) for v in [*c, *w]), 'Finite nonempty label spans required')
    return {'mean': sum(c) / len(c) - sum(w) / len(w), 'sum': sum(c) - sum(w), 'first': c[0] - w[0]}


def single_token_margin(gradient, *, value):
    return {'gradients': {'mean': gradient, 'sum': gradient, 'first': gradient}, 'values': {'mean': value, 'sum': value, 'first': value}}


def first_token_alignment(oracle_pair, correct_label_logps, wrong_label_logps, *, tolerance=TOLERANCES['first_token_alignment_max_abs']):
    """EACH shared-prefix oracle first-token logp (from the FIRST_CONTRAST forward that also produced the first-token VJP) must match the
    first label token of its own label-mean forward (same image, same canonical prefix) within the frozen .002 F32 scorer alignment."""
    require(isinstance(oracle_pair, (list, tuple)) and len(oracle_pair) == 2 and correct_label_logps and wrong_label_logps, 'Oracle pair and both label spans required')
    vals = [*oracle_pair, correct_label_logps[0], wrong_label_logps[0]]
    require(all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals), 'Finite first-token values required')
    ec, ew = abs(oracle_pair[0] - correct_label_logps[0]), abs(oracle_pair[1] - wrong_label_logps[0])
    return {'correct_abs_error': ec, 'wrong_abs_error': ew, 'max_abs': max(ec, ew), 'tolerance': tolerance, 'PASS': max(ec, ew) <= tolerance}


def context_margin_values(correct_label_logps, wrong_label_logps, oracle_pair):
    """mean/sum from the two label-mean forwards; PRIMARY first = the oracle contrast whose gradient is the first-token VJP."""
    values = margin_values({'c': correct_label_logps, 'w': wrong_label_logps}, truth='c')
    first_from_labels = values['first']
    values['first'] = float(oracle_pair[0]) - float(oracle_pair[1])
    values['first_from_label_forwards'] = first_from_labels
    return values


# ================================================================ read-only Adam52 directions (no optimizer, no write)
def _check_profile(profile):
    require(profile.get('bias_correction') is False and tuple(profile.get('betas', ())) == ADAM_PROFILE['betas'] and profile.get('eps') == ADAM_PROFILE['eps']
            and profile.get('step') == PARENT_STEP, 'Only the original Adam52 profile (bias_correction=False, step 52) is supported')
    lr = float(profile['learning_rate'])
    require(math.isfinite(lr) and lr > 0, 'Finite positive learning rate required')
    return lr


def adam_leaf(g, m, v, theta, *, lr, ops):
    """The installed MLX Adam.apply_single expression on read-only copies; returns all direction families for one tensor."""
    b1, b2 = ADAM_PROFILE['betas']
    eps = ADAM_PROFILE['eps']
    finite = lambda x: bool(ops.all(ops.isfinite(x)).item())
    require(g.shape == m.shape == v.shape == theta.shape, 'Adam preview shape mismatch')
    require(all(finite(x) for x in (g, m, v, theta)) and bool(ops.all(v >= 0).item()), 'Adam preview nonfinite input or negative second moment')
    lr32 = ops.array(lr, dtype=g.dtype)
    mp = b1 * m + (1 - b1) * g
    vp = b2 * v + (1 - b2) * ops.square(g)
    formula = -lr32 * mp / (ops.sqrt(vp) + eps)
    momentum = -lr32 * (b1 * m) / (ops.sqrt(b2 * v) + eps)
    rounded = (theta + formula).astype(theta.dtype) - theta
    mom_rounded = (theta + momentum).astype(theta.dtype) - theta
    out = {'negative_gradient': -g, 'adam52_formula': formula, 'adam52_rounded': rounded, 'momentum_only_formula': momentum,
           'momentum_only_rounded': mom_rounded, 'adam52_incremental_formula': formula - momentum, 'adam52_incremental_rounded': rounded - mom_rounded}
    require(all(finite(x) for x in out.values()), 'Adam preview nonfinite direction')
    return out


def adam_directions(g, m, v, profile, *, ops, theta):
    lr = _check_profile(profile)
    require(set(g) == set(m) == set(v) == set(theta), 'Adam preview inventory mismatch')
    out = {f: {} for f in FAMILIES}
    for n in g:
        leaf = adam_leaf(g[n], m[n], v[n], theta[n], lr=lr, ops=ops)
        for f in FAMILIES:
            out[f][n] = leaf[f]
    return out


def directional_effects(margin_grads, directions):
    import numpy as np
    flat = lambda a: np.asarray(a, dtype=np.float64).ravel()
    effects, norms, cross, hnorm = {}, {}, {}, {}
    names = None
    for ctx, metrics in margin_grads.items():
        for met, h in metrics.items():
            names = names or sorted(h)
            hnorm.setdefault(ctx, {})[met] = math.sqrt(sum(float(np.vdot(flat(h[n]), flat(h[n]))) for n in h))
            for x, fams in directions.items():
                for fam, d in fams.items():
                    effects.setdefault(x, {}).setdefault(fam, {}).setdefault(met, {})[ctx] = sum(float(np.vdot(flat(h[n]), flat(d[n]))) for n in h)
    plans = sorted(directions)
    for x in plans:
        for fam, d in directions[x].items():
            norms.setdefault(x, {})[fam] = math.sqrt(sum(float(np.vdot(flat(d[n]), flat(d[n]))) for n in d))
    for fam in sorted({f for x in plans for f in directions[x]}):
        have = [x for x in plans if fam in directions[x]]
        for i, p in enumerate(have):
            for q in have[i + 1:]:
                dot = sum(float(np.vdot(flat(directions[p][fam][n]), flat(directions[q][fam][n]))) for n in directions[p][fam])
                a, b = norms[p][fam], norms[q][fam]
                cross.setdefault(fam, {})[p + '|' + q] = dot / (a * b) if a > 0 and b > 0 else None
    return {'effects': effects, 'direction_norms': norms, 'direction_cosines': cross, 'margin_gradient_norms': hnorm}


# ================================================================ frozen screen and classification
def require_frozen_tolerances(tolerances):
    require(tolerances == TOLERANCES, 'Tolerances differ from the frozen values (no loosening/tightening after preparation)')
    return True


def empirical_tau(e_a, e_b, tolerances=TOLERANCES):
    require(all(isinstance(v, (int, float)) and math.isfinite(v) for v in (e_a, e_b)), 'Finite reconstruction errors required')
    return max(tolerances['tau_floor'], tolerances['tau_error_multiplier'] * max(abs(e_a), abs(e_b)))


def empirical_taus(error_effects, tolerances=TOLERANCES):
    """error_effects[A|B][family][metric][ctx] = h.(d_reconstructed - d_saved) -> taus[family][metric][ctx]."""
    out = {}
    for fam, mets in error_effects['A'].items():
        for met, ctxs in mets.items():
            for ctx, ea in ctxs.items():
                out.setdefault(fam, {}).setdefault(met, {})[ctx] = empirical_tau(ea, error_effects['B'][fam][met][ctx], tolerances)
    return out


def _tau(taus, fam, met, ctx):
    v = taus.get(ctx) if isinstance(taus.get(ctx), (int, float)) else taus[fam][met][ctx]
    require(isinstance(v, (int, float)) and math.isfinite(v) and v > 0, 'Finite positive tau required')
    return v


def _verdict(kind, c, a, b, tau, pair):
    if kind == 'MRI_T1':
        if c > tau and c - a > pair * tau and c - b > pair * tau:
            return 'PASS', None
        if c < -tau:
            return 'FAIL', 'T1_not_positive'
        if c - a < -pair * tau or c - b < -pair * tau:
            return 'FAIL', 'T1_not_better_than_A_B'
        return 'AMBIGUOUS', None
    if kind == 'MRI_FLAIR':
        return ('PASS', None) if (c >= -tau and c - a >= -pair * tau and c - b >= -pair * tau) else ('FAIL', 'FLAIR_harmed')
    require(kind == 'DALLAS', 'Unknown context kind')
    return ('PASS', None) if (c - a >= -pair * tau and c - b >= -pair * tau) else ('FAIL', 'DALLAS_worse_than_A_or_B')


def _evaluate(effects, contexts, taus, fam, met, pair):
    out = {}
    for ctx, kind in contexts.items():
        vals = [effects[x][fam][met][ctx] for x in ('C', 'A', 'B')]
        require(all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals), 'Missing/nonfinite effect: ' + ctx)
        out[ctx] = (*_verdict(kind, *vals, _tau(taus, fam, met, ctx), pair), vals)
    return out


def _inventory_complete(effects, contexts, taus):
    """Every plan x screen family x metric x context effect and tau must exist and be finite (no fake 'unavailable' secondaries)."""
    try:
        for fam in SCREEN_FAMILIES:
            for met in METRICS:
                for ctx in contexts:
                    _tau(taus, fam, met, ctx)
                    for x in PLANS:
                        v = effects[x][fam][met][ctx]
                        if not (isinstance(v, (int, float)) and math.isfinite(v)):
                            return False
    except (KeyError, TypeError, ValueError):
        return False
    return True


def classify(effects, contexts, taus, *, reconstruction, parity, expected_counts, extra_preconditions=None, tolerances=TOLERANCES):
    pair = tolerances['pair_multiplier']
    reasons = []
    counts = {k: 0 for k in expected_counts}
    for kind in contexts.values():
        counts[kind] = counts.get(kind, 0) + 1
    if not contexts or counts != dict(expected_counts) or any(k not in ('MRI_FLAIR', 'MRI_T1', 'DALLAS') for k in counts):
        reasons.append('context_inventory_mismatch')
    if not all(reconstruction.get(x, {}).get('PASS') is True for x in ('A', 'B')):
        reasons.append('A_B_reconstruction_not_PASS')
    if parity.get('PASS') is not True:
        reasons.append('own_old_parity_not_PASS')
    for k, ok in (extra_preconditions or {}).items():
        if ok is not True:
            reasons.append(k)
    if not _inventory_complete(effects, contexts, taus):
        reasons.append('effects_or_tau_inventory_incomplete_or_nonfinite')
    base = {'primary': dict(PRIMARY), 'secondary_metrics_do_not_rescue': True, 'flags': [], 'failures': {}, 'ambiguous': []}
    if reasons:
        return {**base, 'decision': 'INCONCLUSIVE', 'preconditions_failed': reasons}
    primary = _evaluate(effects, contexts, taus, PRIMARY['direction'], PRIMARY['metric'], pair)
    failures, ambiguous = {}, []
    for ctx, (verdict, reason, _) in primary.items():
        if verdict == 'FAIL':
            failures.setdefault(reason, []).append(ctx)
        elif verdict == 'AMBIGUOUS':
            ambiguous.append(ctx)
    flags = []
    t1 = [c for c, k in contexts.items() if k == 'MRI_T1']
    try:
        neg = _evaluate(effects, {c: contexts[c] for c in t1}, taus, 'negative_gradient', PRIMARY['metric'], pair)
        if all(v[0] == 'PASS' for v in neg.values()) and not all(primary[c][0] == 'PASS' for c in t1):
            flags.append('NEGATIVE_GRADIENT_ONLY')
    except (KeyError, TypeError, ValueError):
        pass
    for ctx in contexts:                                   # inventory already verified complete and finite
        m = effects['C'][PRIMARY['direction']]['mean'][ctx]
        tm = _tau(taus, PRIMARY['direction'], 'mean', ctx)
        for met in ('sum', 'first'):
            o = effects['C'][PRIMARY['direction']][met][ctx]
            to = _tau(taus, PRIMARY['direction'], met, ctx)
            if abs(m) > tm and abs(o) > to and (m > 0) != (o > 0):
                flags.append(f'METRIC_SIGN_DISAGREEMENT:{ctx}:{met}')
    decision = 'FAIL_REJECT_C' if failures else ('INCONCLUSIVE' if ambiguous else 'PASS_FOR_SMALL_PILOT')
    return {**base, 'decision': decision, 'failures': failures, 'ambiguous': ambiguous, 'flags': flags, 'preconditions_failed': [],
            'per_context_PRIVATE': {c: {'verdict': v[0], 'reason': v[1], 'C_A_B': v[2]} for c, v in primary.items()},
            'meaning': {'PASS_FOR_SMALL_PILOT': 'local first-order evidence only; a separate training scope and fresh approval are still required',
                        'FAIL_REJECT_C': 'robust T1/FLAIR/Dallas failure under the primary rule: stop C',
                        'INCONCLUSIVE': 'numerical ambiguity inside the frozen screen: no claim'}[decision]}


def public_decision(decision):
    """Class-level view of a classification: decision, failure COUNTS by reason, ambiguous count, flag KINDS. No context IDs."""
    return {'decision': decision['decision'], 'primary': decision['primary'], 'secondary_metrics_do_not_rescue': True,
            'failure_counts': {reason: len(ids) for reason, ids in decision.get('failures', {}).items()},
            'ambiguous_count': len(decision.get('ambiguous', [])), 'flag_kinds': sorted({f.split(':')[0] for f in decision.get('flags', [])}),
            'flag_count': len(decision.get('flags', [])), 'preconditions_failed': sorted({r.split(':')[0] for r in decision.get('preconditions_failed', [])}),
            'meaning': decision.get('meaning')}


def kind_summary(kinds, effects, taus, values):
    """Per class (MRI_FLAIR / MRI_T1 / DALLAS): counts and min/max/mean only; per-image tables stay PRIVATE."""
    stat = lambda xs: {'n': len(xs), 'min': min(xs), 'max': max(xs), 'mean': sum(xs) / len(xs)} if xs else None
    out = {}
    for kind in ('MRI_FLAIR', 'MRI_T1', 'DALLAS'):
        ids = [c for c, k in kinds.items() if k == kind]
        fams = {}
        for fam in SCREEN_FAMILIES + ('momentum_only_rounded', 'adam52_incremental_rounded'):
            fams[fam] = {}
            for met in METRICS:
                cell = {x: stat([effects[x][fam][met][c] for c in ids]) for x in PLANS}
                cell['C_minus_A'] = stat([effects['C'][fam][met][c] - effects['A'][fam][met][c] for c in ids])
                cell['C_minus_B'] = stat([effects['C'][fam][met][c] - effects['B'][fam][met][c] for c in ids])
                if fam in SCREEN_FAMILIES:
                    cell['tau'] = stat([_tau(taus, fam, met, c) for c in ids])
                fams[fam][met] = cell
        out[kind] = {'count': len(ids), 'parent_margin_value': {met: stat([values[c][met] for c in ids]) for met in METRICS},
                     'first_order_change': fams}
    return out


# ================================================================ scope assembly
def assemble_scope(*, response_jobs, margin_jobs, run_output, sources, files, payload, runtime, preparation, bindings, tolerances=TOLERANCES):
    budget = scope_budget(response_jobs, margin_jobs)
    from mri_rl import safe_micro_dynamic_resources_v1 as resources
    from mri_rl.safe_micro_watchdog_v2 import POLICY as GUARDIAN_POLICY
    inputs = {'files': files, 'payload': payload, 'response_jobs': identity(response_jobs), 'margin_jobs': identity(margin_jobs), 'bindings': identity(bindings)}
    return {'schema': SCHEMA, 'scope_kind': KIND, 'action': ACTION, 'implemented_by': IMPLEMENTED_BY,
            'status_at_preparation': 'PREPARED_PENDING_EXACT_REVIEW_AND_FRESH_HUMAN', 'source_sha256': dict(sources), 'file_sha256': dict(files),
            'payload_sha256': dict(payload), 'input_sha256': identity(inputs), 'run_output': str(run_output),
            'response_jobs_PRIVATE': response_jobs, 'margin_jobs_PRIVATE': margin_jobs, 'bindings_PRIVATE': bindings, 'budget': budget,
            'tolerances': dict(tolerances), 'tolerance_basis': dict(TOLERANCE_BASIS), 'adam_profile': {k: list(v) if isinstance(v, tuple) else v for k, v in ADAM_PROFILE.items()},
            'joint_gradient_norm_clipping': JOINT_GRADIENT_NORM_CLIPPING, 'primary': dict(PRIMARY), 'run_limits': dict(RUN_LIMITS),
            'resource_policy': dict(resources.POLICY), 'external_guardian_policy': GUARDIAN_POLICY, 'runtime_PRIVATE': dict(runtime),
            'preparation_PRIVATE': dict(preparation), 'automatic_retry': False, 'training': False, 'synthetic': bool(bindings.get('synthetic_test_fixture')),
            'forbidden_inputs': ['DEV24', 'final100', 'sealed new16 outputs'], 'claims': dict(CLAIMS)}


# ================================================================ CPU preparation over the authenticated V2 run
def _context_kind(task, truth):
    return 'DALLAS' if task == 'mri_visit_correspondence' else ('MRI_FLAIR' if truth == 'FLAIR' else 'MRI_T1')


def prepare(*, root, worktree, out_dir, run_dir=None, results_path=None, expected_v2_scope_sha256=None):
    """CPU only, writes nothing: authenticate V2, derive A/B/C plans on Common32, bind every input/source/payload, freeze jobs/budget."""
    from mri_rl import v2_native_readonly_v1 as nr
    from mri_rl import v2_regression_diagnosis_v1 as diag
    from mri_rl import accuracy_pilot_v2_core as core
    from mri_rl import multistep_gspo_core_v1 as ms
    from mri_rl import answer_matched_baseline_v1 as amb
    root, worktree = Path(root).resolve(), Path(worktree).resolve()
    expected_v2_scope_sha256 = expected_v2_scope_sha256 or nr.EXPECTED['v2_scope_file_sha256']
    run_dir = Path(run_dir or root / 'local_data/accuracy_pilot_v2/pilot_v2_attempt1_PRIVATE').resolve()
    results_path = Path(results_path or worktree / 'results/ACCURACY_PILOT_V2_ACTUAL_RESULTS_V1.json').resolve()
    out_dir = Path(out_dir).resolve()
    base = (root / 'local_data/native_group_signal_vjp_v1').resolve()
    require(out_dir.is_relative_to(base) and out_dir != base and not out_dir.is_relative_to(run_dir), 'Preparation output must be a new attempt directory under local_data/native_group_signal_vjp_v1')
    with nr.cpu_only_imports():
        pins = {'v2_results_sha256': diag.require_file_sha(results_path, nr.EXPECTED['v2_results_sha256'], 'committed V2 results'),
                'diagnosis_module_sha256': diag.require_file_sha(worktree / 'src/mri_rl/v2_regression_diagnosis_v1.py', nr.EXPECTED['diagnosis_module_sha256'], 'diagnosis module'),
                'frozen_C_source_sha256': diag.require_file_sha(worktree / 'src/mri_rl/answer_matched_baseline_v1.py', FROZEN_C_SOURCE_SHA256, 'frozen C source'),
                'frozen_grv_source_sha256': diag.require_file_sha(worktree / 'src/mri_rl/group_relative_visual_signal_v1.py', FROZEN_GRV_SOURCE_SHA256, 'frozen diagnosis source')}
        receipt = json.loads((worktree / C_DOCSTRING_RECEIPT).read_text())     # executed d3998.. -> committed a1a54.. is docstring-only
        require(receipt.get('executed_source_sha256') == EXECUTED_C_SOURCE_SHA256 and receipt.get('final_source_sha256') == FROZEN_C_SOURCE_SHA256 and
                receipt.get('executable_AST_without_module_docstring_identical') is True and receipt.get('new_computation') == 0,
                'C docstring-only precision receipt does not bind the executed C source to the committed final source')
        pins['C_docstring_receipt_sha256'] = shared.file_sha256(worktree / C_DOCSTRING_RECEIPT)
        ctx = diag.authenticate(root=root, worktree=worktree, run_dir=run_dir, results_path=results_path, expected_scope_sha256=expected_v2_scope_sha256)
        require(ctx['condition'] == 'concise', 'V2 frozen prompt condition must be concise')
        spec, runtime, v2scope = ctx['spec'], ctx['runtime'], ctx['scope']
        lengths = core.prompt_lengths_for('concise', spec['concise_sequence_prompt_length'])
        units4 = ctx['units'][:4]
        require([u['index'] for u in units4] == list(core.SEGMENTS[0]), 'Common units 0-3 only (segment 0)')
        raw_path = ctx['common_dir'] / 'raw_rollout_and_own_old_PRIVATE.json'
        raw = json.loads(raw_path.read_text())
        ps = ctx['common']['parent_state']
        blocks = diag.rebuild_common_blocks(units4, raw['raws'], parent_policy=ps['policy'], reward_sha=ctx['reward_sha'], parent_receipt=ctx['parent_receipt'])
        require([b.sha256 for b in blocks] == ctx['common']['bundle_sha256'], 'Rebuilt Common bundles differ from the common record')
        for block, old in zip(blocks, raw['olds']):
            ms.check_old(block, old)
        arms = {arm: diag.rederive_arm(ctx['units'], ctx['common_dir'], run_dir / f'stage{arm}_PRIVATE' / 'arm_PRIVATE', arm,
                                       reward_sha=ctx['reward_sha'], parent_receipt=ctx['parent_receipt']) for arm in ('A', 'B')}
        for arm in ('A', 'B'):
            require(arms[arm][0]['saved_evidence_all_match'] and arms[arm][0]['rollout_origin'] == 'COMMON_PARENT', f'{arm}0 plan/evidence mismatch')
            require(ctx['receipts'][(arm, 0)]['adam_step_before'] == PARENT_STEP, 'Segment 0 must start at Parent Adam52')
        plans = {arm: grv.response_table(arms[arm][0]['units'], segment=0, arm=arm, origin=arms[arm][0]['rollout_origin']) for arm in ('A', 'B')}
        plans['C'] = amb.c_plan(plans['A'])
        jobs = response_jobs(plans)
        plan_maps = {arm: {(r['unit'], x['response_id']): x for r in arms[arm][0]['units'] for x in r['responses']} for arm in ('A', 'B')}
        model = Path(runtime['model'])
        tokenizer_files = {'vocab': model / 'vocab.json', 'tokenizer_json': model / 'tokenizer.json'}
        decoder = diag.TokenDecoder.from_files(tokenizer_files['vocab'], tokenizer_files['tokenizer_json'])
        tf = {(j['unit'], j['response_id']): j for j in nr.derive_tf_jobs(units4, raw, blocks, plan_maps, lengths, decoder)}
        for j in jobs:
            t = tf[(j['unit'], j['key'])]
            require(t['tokens'] == j['tokens'], 'Saved token count differs from the plan table')
            j.update(context_key=t['context_key'], token_ids=t['token_ids'], token_sha256=t['token_sha256'], prompt_length=t['prompt_length'],
                     tf_positions=t['tf_positions'], old_logps_PRIVATE=t['old_logps_PRIVATE'], old_logps_sha256=t['old_logps_sha256'],
                     row_id_sha256=t['row_id_sha256'], recorded_context_sha256=t['recorded_context_sha256'], block_sha256=t['block_sha256'],
                     reward_receipt_sha256=t['reward_receipt_sha256'])
        from tokenizers import Tokenizer                    # tokenizer library only (no transformers/MLX/torch)
        tokenizer = Tokenizer.from_file(str(tokenizer_files['tokenizer_json']))
        encode = lambda s: tokenizer.encode(s, add_special_tokens=False).ids
        contexts = []
        for u, rollout in zip(units4, raw['raws']):
            keys = [c['key'] for c in u['contexts']]
            require(len(keys) == 2 and len(set(keys)) == 2 and set(keys) == set(rollout['contexts']), 'Each original Common unit must have exactly its 2 recorded contexts')
            if u['kind'] == ms.SEQ_KIND:
                require(set(keys) == {'F', 'T'}, 'SEQ_PAIR unit must hold exactly the F and T contexts')
            for c in u['contexts']:
                row, key = c['row'], c['key']
                truth, task = row['reward_model']['ground_truth'], row['ability']
                rec = rollout['contexts'][key]
                contexts.append({'context_id': f"u{u['index']}_{key}", 'unit': u['index'], 'context_key': key, 'task': task, 'truth': truth,
                                 'kind': _context_kind(task, truth), 'plan': canonical_token_plan(task, truth, encode), 'prompt_length': rec['prompt_length'],
                                 'row_id_sha256': identity({'row_id': row['id']}), 'recorded_context_sha256': identity(rec)})
        kinds = [c['kind'] for c in contexts]
        require(kinds.count('MRI_FLAIR') == 2 and kinds.count('MRI_T1') == 2 and kinds.count('DALLAS') == 4, 'All 8 Common TRAIN contexts: 2 FLAIR, 2 T1, 4 Dallas')
        for c in contexts:
            require(c['prompt_length'] == lengths[c['task']], 'Recorded prompt length differs from the frozen concise ledger')
        mjobs = margin_jobs_for_contexts(contexts)
        saved, payload = {}, {}
        for arm in ('A', 'B'):
            seg = run_dir / f'stage{arm}_PRIVATE' / 'arm_PRIVATE' / 'segment_0'
            grad = (seg / 'joint_gradient_float32_PRIVATE.safetensors').resolve()
            commit = json.loads((run_dir / f'stage{arm}_PRIVATE' / 'v2_journal_PRIVATE' / 'segment0' / 'commit_PRIVATE.json').read_text())
            require(commit['files'].get(str(grad)) == shared.file_sha256(grad), f'{arm}0 saved joint gradient differs from its journal commit')
            saved[arm] = {'path': str(grad), 'sha256': commit['files'][str(grad)], 'receipt_joint_gradient_l2': ctx['receipts'][(arm, 0)]['joint_gradient_l2']}
            payload[str(grad)] = saved[arm]['sha256']
        historical = {}
        for label in ('A53', 'B53'):
            entry = nr.checkpoint_entry(run_dir, ctx['receipts'][(nr.ARM_OF[label], 0)], label)
            historical[label] = {'checkpoint_dir': entry['checkpoint_dir'], 'adapter_sha256': ctx['receipts'][(nr.ARM_OF[label], 0)]['files_sha256']['adapter.safetensors']}
            payload.update(entry['payload_sha256'])
        from mri_rl import accuracy_recovery_v1 as recovery
        parent_dir = Path(runtime['parent_checkpoint']).resolve()
        anchor = recovery.adam52_parent_anchor(parent_dir, runtime['parent_receipt_sha256'])
        for name, item in recovery.parent_disk_snapshot(parent_dir).items():
            payload[str(parent_dir / name)] = item['sha256']
        # exact base cache / model non-weight files / certificate bound as immutable payload BEFORE any model construction (no GPU cache recovery)
        from mri_rl.likelihood_cache import verify_cache_identity
        from mri_rl.safe_micro_recovery_entry_v1 import SHARDS
        cert = Path(runtime['cache_certificate']).resolve()
        originals = {str(p): shared.file_sha256(p) for p in sorted(model.iterdir())
                     if p.is_file() and p.suffix != '.safetensors' and p.name != 'model.safetensors.index.json'}
        cache_binding = verify_cache_identity(Path(runtime['cache_dir']), cert, shared.file_sha256(cert), originals)
        require({n: v['sha256'] for n, v in cache_binding['shards'].items()} == SHARDS, 'Cache shards differ from the historical certified SHARDS')
        for p in sorted(Path(runtime['cache_dir']).resolve().iterdir()):
            if p.is_file():
                payload[str(p)] = shared.file_sha256(p)
        cache_files = {str(cert): shared.file_sha256(cert), **originals}
        stage_input = json.loads(Path(spec['stage_input']).read_text())
        train_images = {x['record_id_PRIVATE']: x['image_binding_PRIVATE']['sha256'] for x in stage_input['planned_image_bindings_PRIVATE']}
        expected_images = {c['row']['id']: train_images[c['row']['id']] for u in units4 for c in u['contexts']}
        spec_path = root / 'local_data/accuracy_pilot_v2/PILOT_V2_SPEC_PRIVATE.json'
        files = {str(Path(p)): h for p, h in v2scope['file_sha256'].items()}
        for p, h in files.items():
            require(shared.file_sha256(p) == h, 'V2-bound metadata changed: ' + Path(p).name)
        for p in (spec_path, results_path, raw_path, ctx['common_dir'] / 'common_record_PRIVATE.json', tokenizer_files['vocab'], tokenizer_files['tokenizer_json'],
                  Path(spec['runtime_spec']), Path(spec['stage_input']), Path(runtime['calibration']), worktree / 'results/GROUP_RELATIVE_C_MATCHED_ACTUAL_CPU_V1.json',
                  worktree / C_DOCSTRING_RECEIPT, worktree / INDEPENDENT_BUDGET_FACTS):
            files[str(Path(p).resolve())] = shared.file_sha256(p)
        files.update(cache_files)
        for arm in ('A', 'B'):
            seg = run_dir / f'stage{arm}_PRIVATE' / 'arm_PRIVATE' / 'segment_0'
            for n in ('gradient_evidence_PRIVATE.json', 'receipt_PRIVATE.json'):
                files[str((seg / n).resolve())] = shared.file_sha256(seg / n)
        from mri_rl.safe_micro_entry_v1 import _closure
        from mri_rl.safe_micro_recovery_entry_v1 import bind_runtime_sources
        own = ('src/mri_rl/native_group_signal_vjp_v1.py', 'src/mri_rl/native_group_signal_vjp_entry_v1.py', 'src/mri_rl/native_group_signal_vjp_backend_v1.py',
               'scripts/run_native_group_signal_vjp_v1.py', 'src/mri_rl/answer_matched_baseline_v1.py', 'src/mri_rl/group_relative_visual_signal_v1.py',
               'src/mri_rl/v2_regression_diagnosis_v1.py', 'src/mri_rl/v2_native_readonly_v1.py', 'src/mri_rl/accuracy_pilot_v2_native.py',
               'src/mri_rl/class_margin_v1.py', 'src/mri_rl/accuracy_stage_native_v1.py', 'src/mri_rl/safe_micro_recovery_native_v1.py',
               'src/mri_rl/safe_micro_watchdog_v2.py', 'src/mri_rl/safe_micro_dynamic_resources_v1.py', 'src/mri_rl/likelihood_cache.py',
               'tests/test_native_group_signal_vjp_v1.py', 'research/NATIVE_GROUP_SIGNAL_VJP_DESIGN_V1.md')   # mutable experiment log NOT bound
        sources = _closure({str(worktree / p): shared.file_sha256(worktree / p) for p in own})
        for p, h in v2scope['source_sha256'].items():                  # every executed V2 source unchanged
            require(shared.file_sha256(p) == h and sources.get(p, h) == h, 'Executed V2 source changed: ' + Path(p).name)
            sources[p] = h
        sources = bind_runtime_sources(sources, runtime)
        bindings = {'units': {str(u['index']): u for u in units4}, 'recorded_contexts': {str(u['index']): r['contexts'] for u, r in zip(units4, raw['raws'])},
                    'margin_contexts': contexts, 'lengths': dict(lengths), 'expected_images': expected_images, 'saved_joint_gradients': saved,
                    'historical_53': historical, 'parent': {'dir': str(parent_dir), 'receipt_sha256': anchor['receipt_sha256'], 'adapter_sha256': anchor['adapter_sha256'],
                                                            'common_parent_state': ps, 'optimizer_step': PARENT_STEP},
                    'tokenizer': {k: str(v) for k, v in tokenizer_files.items()}, 'pins': {**pins, 'v2_scope_file_sha256': expected_v2_scope_sha256,
                                                                                          'v2_journal_commits_verified': len(ctx['record']['journal_commits'])},
                    'plans_summary': job_summary(jobs)}
        scope = assemble_scope(response_jobs=jobs, margin_jobs=mjobs, run_output=out_dir / 'run_PRIVATE', sources=sources, files=files, payload=payload,
                               runtime={'runtime_spec': spec['runtime_spec'], 'global_lock': runtime['global_lock'], 'recovery_spec': spec['recovery_spec'],
                                        'v2_spec': str(spec_path)},
                               preparation={'root': str(root), 'worktree': str(worktree), 'out_dir': str(out_dir), 'run_dir': str(run_dir),
                                            'results_path': str(results_path), 'expected_v2_scope_sha256': expected_v2_scope_sha256},
                               bindings=bindings)
        facts = json.loads((worktree / INDEPENDENT_BUDGET_FACTS).read_text())['budget']
        b = scope['budget']
        crosscheck = {'response_VJPs': (b['response_vjps'], facts['response_VJPs']),
                      'response_TF_positions': (sum(j['tf_positions'] for j in jobs), facts['response_TF_positions']),
                      'margin_TF_positions': (sum(m['positions'] for m in mjobs), facts['margin_TF_positions']),
                      'total_TF_forwards': (b['forwards'], facts['total_TF_forwards']), 'total_backward_VJPs': (b['vjps'], facts['total_backward_VJPs']),
                      'total_TF_positions': (b['teacherforced_positions'], facts['total_TF_positions']), 'model_loads': (b['model_loads'], facts['model_loads'])}
        require(all(mine == theirs for mine, theirs in crosscheck.values()),
                'Derived budget disagrees with the independent budget facts (cross-check only; investigate, never override)')
    summary = {'independent_budget_crosscheck': {k: {'derived': v[0], 'independent': v[1], 'match': True} for k, v in crosscheck.items()},
               'response_jobs': job_summary(jobs), 'margin_contexts': {k: kinds.count(k) for k in ('MRI_FLAIR', 'MRI_T1', 'DALLAS')},
               'margin_jobs': len(mjobs), 'dallas_single_token_labels': all(c['plan']['single_token_labels'] for c in contexts if c['kind'] == 'DALLAS'),
               'budget': scope['budget'], 'tolerances': dict(TOLERANCES), 'run_limits': dict(RUN_LIMITS), 'heavy_modules_loaded': nr.heavy_modules_loaded()}
    return {'status': 'PREPARED_PENDING_EXACT_REVIEW_AND_FRESH_HUMAN', 'executable_scope': scope, 'scope_sha256': digest(scope), 'native_calls': 0,
            'public_summary': summary}


# ================================================================ in-child reconstruction gate (CPU arithmetic on saved private vectors)
def reconstruction_gate(scope, records, *, load_vector, load_saved):
    """Before ANY margin call: own-old parity (1e-5), unambiguous clip branches and A/B vector reconstruction (<=1e-4)."""
    jobs = scope['response_jobs_PRIVATE']
    require([r['index'] for r in records] == [j['index'] for j in jobs], 'Response records must cover every job in order')
    current = {r['index']: r['selected_logps'] for r in records}
    old = {j['index']: j['old_logps_PRIVATE'] for j in jobs}
    worst = max(max(abs(float(a) - float(b)) for a, b in zip(current[i], old[i])) for i in old)
    parity = {'max_abs_current_minus_old': worst, 'bound': scope['tolerances']['own_old_max_abs'], 'PASS': worst <= scope['tolerances']['own_old_max_abs']}
    try:
        native = native_ratios(records)
        coeffs = plan_coefficients(jobs, current=current, old=old, native=native)
    except ValueError as error:
        return {'PASS': False, 'parity': parity, 'reason': 'coefficient: ' + str(error).split(' for response')[0]}
    saved = {x: load_saved(x) for x in ('A', 'B')}
    names = sorted(saved['A'])
    g = joint_gradients(jobs, coeffs, load=load_vector, names=names)
    rec = {x: reconstruction_check({n: g[x][n] for n in names}, saved[x], relative_l2_max=scope['tolerances']['reconstruction_relative_l2_max']) for x in ('A', 'B')}
    return {'PASS': parity['PASS'] and rec['A']['PASS'] and rec['B']['PASS'], 'parity': parity,
            'reconstruction': {x: {k: v for k, v in r.items() if not k.endswith('_PRIVATE')} for x, r in rec.items()},
            'coefficients': coefficient_report(jobs, current=current, old=old, native=native)}


# ================================================================ CPU analysis after the native child has exited (model released)
class _Lazy:
    def __init__(self, tensor_file):
        self.f = tensor_file

    def __getitem__(self, name):
        return self.f.get(name)

    def keys(self):
        return self.f.keys()


def analyze(scope, *, out=None):
    import numpy as np
    from mri_rl import v2_regression_diagnosis_v1 as diag
    from mri_rl.safe_micro_journal_v1 import Journal
    base = Path(scope['run_output'])
    out = Path(out or base.parent / 'analysis_PRIVATE')
    require(not out.exists(), 'Analysis output already exists (no overwrite)')
    b = scope['bindings_PRIVATE']
    with Journal(base / 'journal_PRIVATE', {'scope': digest(scope), 'input': scope['input_sha256']}) as journal:
        complete = journal.get('native_complete')
        require(complete is not None and complete['release_verified'] is True, 'Committed native terminal with verified release required')
        records = [journal.get('resp_%02d' % j['index']) for j in scope['response_jobs_PRIVATE']]
        margins = [journal.get('margin_%02d' % m['index']) for m in scope['margin_jobs_PRIVATE']] if complete['status'] == 'PASS' else None
    for r in records + (margins or []):
        require(r is not None and sha256_file(r['vector_path']) == r['vector_sha256'], 'Committed vector missing/changed')
    vec = lambda r: _Lazy(diag.TensorFile(r['vector_path']))
    saved = {x: diag.TensorFile(b['saved_joint_gradients'][x]['path']) for x in ('A', 'B')}
    for x in ('A', 'B'):
        require(sha256_file(b['saved_joint_gradients'][x]['path']) == b['saved_joint_gradients'][x]['sha256'], 'Saved joint gradient changed')
    gate = reconstruction_gate(scope, records, load_vector=lambda i: {n: vec(records[i])[n] for n in saved['A'].keys()},
                               load_saved=lambda x: {n: saved[x].get(n) for n in saved[x].keys()})
    report = {'schema': SCHEMA + '-analysis', 'native_complete_PRIVATE': complete, 'reconstruction_gate': gate, 'budget': scope['budget'],
              'tolerances': scope['tolerances'], 'claims': dict(CLAIMS)}
    forbidden = private_identifiers(scope)
    if complete['status'] != 'PASS' or not gate['PASS']:
        report['decision'] = {'decision': 'INCONCLUSIVE', 'preconditions_failed': ['A_B_reconstruction_or_parity_not_PASS'], 'C_science': 'NOT_EVALUATED'}
        return _write_analysis(out, report, forbidden)
    jobs = scope['response_jobs_PRIVATE']
    coeffs = plan_coefficients(jobs, current={r['index']: r['selected_logps'] for r in records}, old={j['index']: j['old_logps_PRIVATE'] for j in jobs},
                               native=native_ratios(records))
    names = sorted(saved['A'].keys())
    G = joint_gradients(jobs, coeffs, load=lambda i: {n: vec(records[i])[n] for n in names}, names=names)
    parent = diag.Checkpoint(b['parent']['dir'], label='Parent_Adam52', adapter_sha256=b['parent']['adapter_sha256'])
    require(parent.step() == PARENT_STEP, 'Parent must be Adam52')
    lr = float(parent.lr())
    hist = {k: diag.Checkpoint(v['checkpoint_dir'], label=k, adapter_sha256=v['adapter_sha256']) for k, v in b['historical_53'].items()}
    contexts = {c['context_id']: c for c in b['margin_contexts']}
    mj = {}
    for m, r in zip(scope['margin_jobs_PRIVATE'], margins):
        mj.setdefault(m['context_id'], {})[m['objective'] + ':' + m['role']] = (m, r)
    values, alignment = {}, {}
    for cid, c in contexts.items():
        oracle = mj[cid]['FIRST_CONTRAST:correct_minus_wrong'][1]['label_logps']        # [lp(correct first), lp(wrong first)], shared prefix
        if c['plan']['single_token_labels']:                                             # Dallas: value AND gradient alias mean=sum=first
            v = oracle[0] - oracle[1]
            values[cid] = {'mean': v, 'sum': v, 'first': v}
        else:
            lc, lw = mj[cid]['LABEL_MEAN:correct'][1]['label_logps'], mj[cid]['LABEL_MEAN:wrong'][1]['label_logps']
            alignment[cid] = first_token_alignment(oracle, lc, lw, tolerance=scope['tolerances']['first_token_alignment_max_abs'])
            values[cid] = context_margin_values(lc, lw, oracle)
    files = {cid: {k: diag.TensorFile(r['vector_path']) for k, (_, r) in jobs_.items()} for cid, jobs_ in mj.items()}
    eff, err, norms, cross, hn = {}, {'A': {}, 'B': {}}, {}, {}, {}
    hist_check = {k: diag.Check(k) for k in ('A', 'B')}
    add = lambda d, keys, v: d.setdefault(keys[0], {}).setdefault(keys[1], {}).setdefault(keys[2], {}).__setitem__(keys[3], d[keys[0]][keys[1]][keys[2]].get(keys[3], 0.) + v)
    for n in names:
        theta = parent.param(n)
        m52, v52 = parent.moments(n)
        dirs = {x: adam_leaf(G[x][n].astype(np.float32), m52, v52, theta, lr=lr, ops=np) for x in PLANS}
        ref = {x: adam_leaf(saved[x].get(n), m52, v52, theta, lr=lr, ops=np) for x in ('A', 'B')}
        for x, label in (('A', 'A53'), ('B', 'B53')):
            actual = hist[label].param(n).astype(np.float64) - theta.astype(np.float64)
            pred = ref[x]['adam52_rounded'].astype(np.float64)
            bound = 2 * np.spacing(np.abs(hist[label].param(n))).astype(np.float64) + 8 * 2. ** -23 * np.abs(ref[x]['adam52_formula'].astype(np.float64))
            hist_check[x].add(pred, actual, bound)
        hs = {}
        for cid, c in contexts.items():
            fc = files[cid]['FIRST_CONTRAST:correct_minus_wrong'].get(n).astype(np.float64)
            if c['plan']['single_token_labels']:
                hs[cid] = {'mean': fc, 'sum': fc, 'first': fc}
            else:
                gc = files[cid]['LABEL_MEAN:correct'].get(n).astype(np.float64)
                gw = files[cid]['LABEL_MEAN:wrong'].get(n).astype(np.float64)
                nc, nw = len(c['plan']['labels']['correct']['token_ids']), len(c['plan']['labels']['wrong']['token_ids'])
                hs[cid] = {'mean': gc - gw, 'sum': nc * gc - nw * gw, 'first': fc}
        for cid, mets in hs.items():
            for met, h in mets.items():
                hf = h.ravel()
                hn.setdefault(cid, {})[met] = hn.get(cid, {}).get(met, 0.) + float(hf @ hf)
                for x in PLANS:
                    for fam in FAMILIES:
                        add(eff, (x, fam, met, cid), float(hf @ dirs[x][fam].astype(np.float64).ravel()))
                for x in ('A', 'B'):
                    for fam in SCREEN_FAMILIES:
                        diff = dirs[x][fam].astype(np.float64) - ref[x][fam].astype(np.float64)
                        add(err, (x, fam, met, cid), float(hf @ diff.ravel()))
        for fam in FAMILIES:
            fl = {x: dirs[x][fam].astype(np.float64).ravel() for x in PLANS}
            for x in PLANS:
                norms.setdefault(x, {})[fam] = norms.get(x, {}).get(fam, 0.) + float(fl[x] @ fl[x])
            for p, q in (('A', 'B'), ('A', 'C'), ('B', 'C')):
                cross.setdefault(fam, {})[p + '|' + q] = cross.get(fam, {}).get(p + '|' + q, 0.) + float(fl[p] @ fl[q])
    err = {x: err[x] for x in ('A', 'B')}
    taus = empirical_taus(err)
    norms = {x: {f: math.sqrt(v) for f, v in fs.items()} for x, fs in norms.items()}
    cosines = {f: {k: (v / (norms[k[0]][f] * norms[k[2]][f]) if norms[k[0]][f] > 0 and norms[k[2]][f] > 0 else None) for k, v in d.items()} for f, d in cross.items()}
    hist_report = {x: c.report() for x, c in hist_check.items()}
    kinds = {cid: c['kind'] for cid, c in contexts.items()}
    decision = classify(eff, kinds, taus, reconstruction=gate['reconstruction'], parity=gate['parity'], expected_counts=REQUIRED_CONTEXT_COUNTS,
                        extra_preconditions={'saved_reference_adam_reproduces_historical_53': all(r['PASS'] for r in hist_report.values()),
                                             'first_token_oracle_alignment': bool(alignment) and all(a['PASS'] for a in alignment.values()),
                                             'margin_values_finite': all(math.isfinite(v) for d in values.values() for v in d.values())})
    report.update({'effects_PRIVATE': eff, 'reconstruction_error_effects_PRIVATE': err, 'taus_PRIVATE': taus, 'direction_norms': norms, 'direction_cosines': cosines,
                   'margin_gradient_norms_PRIVATE': {cid: {m: math.sqrt(v) for m, v in d.items()} for cid, d in hn.items()},
                   'margin_values_PRIVATE': values, 'first_token_alignment_PRIVATE': alignment,
                   'first_token_alignment': {'count': len(alignment), 'max_abs': max((a['max_abs'] for a in alignment.values()), default=None),
                                             'tolerance': scope['tolerances']['first_token_alignment_max_abs'], 'all_PASS': all(a['PASS'] for a in alignment.values())},
                   'historical_53_rounded_adam_consistency': hist_report, 'decision_PRIVATE': decision, 'decision': public_decision(decision),
                   'kind_summary': kind_summary(kinds, eff, taus, values) if not decision.get('preconditions_failed') else None,
                   'per_context_table_PRIVATE': _context_table(kinds, eff, taus, values)})
    return _write_analysis(out, report, forbidden)


def _context_table(kinds, eff, taus, values):
    """PRIVATE per-image table (inspect locally); never part of the public aggregate."""
    out = {}
    for cid in sorted(kinds):
        out[cid] = {'kind': kinds[cid], 'margin_value_parent': values[cid],
                    'first_order_change': {fam: {met: {x: eff[x][fam][met][cid] for x in PLANS} for met in METRICS} for fam in FAMILIES},
                    'tau': {fam: {met: taus[fam][met][cid] for met in METRICS} for fam in SCREEN_FAMILIES}}
    return out


def _write_analysis(out, report, forbidden=()):
    from mri_rl.safe_micro_journal_v1 import durable_mkdir
    public = public_aggregate(report, forbidden_values=forbidden)                # refuse BEFORE writing anything
    durable_mkdir(out)
    shared.publish_once(out / 'analysis_PRIVATE.json', report)
    shared.publish_once(out / 'AGGREGATE_PUBLIC_CANDIDATE.json', public)
    return {'decision': report['decision']['decision'], 'public': public}


# ================================================================ public aggregate
def _strip_private(node):
    if isinstance(node, dict):
        return {k: _strip_private(v) for k, v in node.items() if not str(k).endswith('_PRIVATE')}
    if isinstance(node, list):
        return [_strip_private(v) for v in node]
    return node


def public_aggregate(report, *, forbidden_values=()):
    """Strip every *_PRIVATE key, then refuse absolute paths, private field names, case/group keys or per-record lists (recursive
    contract of the prior phase) and any literal private identifier (context IDs, row/receipt/token hashes) supplied by the caller."""
    pub = _strip_private(copy.deepcopy(report))
    pub['claims'] = dict(CLAIMS)
    text = json.dumps(pub, allow_nan=False)
    require('/Users/' not in text and '/home/' not in text and '_PRIVATE' not in text, 'Public aggregate carries private paths or fields')
    hits = sorted({v for v in forbidden_values if isinstance(v, str) and len(v) >= 4 and v in text})
    require(not hits, f'Public aggregate carries {len(hits)} private identifier value(s)')
    pub['public_case_keys'] = grv.public_case_key_check(pub)
    pub['private_identifier_screen'] = {'checked_values': len(forbidden_values), 'hits': 0}
    return pub


def private_identifiers(scope):
    vals = set()
    for c in scope['bindings_PRIVATE'].get('margin_contexts', []):
        vals.update(str(c.get(k)) for k in ('context_id', 'row_id_sha256', 'recorded_context_sha256') if c.get(k))
    for j in scope['response_jobs_PRIVATE']:
        vals.update(str(j.get(k)) for k in ('token_sha256', 'row_id_sha256', 'recorded_context_sha256', 'block_sha256', 'reward_receipt_sha256',
                                            'old_logps_sha256') if j.get(k))
    vals.update(str(v) for v in scope['bindings_PRIVATE'].get('expected_images', {}))
    vals.update(str(v) for v in scope['bindings_PRIVATE'].get('expected_images', {}).values())
    return sorted(vals)
