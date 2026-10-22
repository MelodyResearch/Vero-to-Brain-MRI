"""TESTS-FIRST (RED until implementation) for the bounded native read-only A/B/C VJP experiment on Parent Adam52 + Common32.

Everything here is SYNTHETIC (NumPy arrays through injected ops). No MLX/torch/transformers import, no model, no private asset.
Modules under test (not yet implemented):
  mri_rl.native_group_signal_vjp_v1        CPU data/math/prepare/classification/public aggregate
  mri_rl.native_group_signal_vjp_entry_v1  authorization + one-shot journaled read-only controller (ops injected)
  mri_rl.native_group_signal_vjp_backend_v1 lazy native ops (import must stay MLX-free)
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_k, '1')
import copy
import json
import math
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mri_rl import group_relative_visual_signal_v1 as grv          # noqa: E402
from mri_rl import answer_matched_baseline_v1 as amb               # noqa: E402
from mri_rl import paired_prompt_groups_v1 as ppg                  # noqa: E402
from mri_rl.answer_likelihood import PREFIX, SUFFIX               # noqa: E402
from mri_rl import native_group_signal_vjp_v1 as ngs               # noqa: E402
from mri_rl import native_group_signal_vjp_entry_v1 as entry       # noqa: E402
from test_answer_matched_baseline_v1 import pair, dallas, C_, Fw, Fi, Tc, Tw, Ti   # noqa: E402

grv.install_import_guard()
NAMES = ('layers.0.q.lora_a', 'layers.0.q.lora_b')
SHAPES = {'layers.0.q.lora_a': (3, 2), 'layers.0.q.lora_b': (2, 3)}
FL, T1 = 'FLAIR', 'contrast-enhanced T1'


# ================================================================ SYNTHETIC fixtures
def three_plans():
    """Two SEQ_PAIR units + one Dallas unit, identical responses in every plan. A=Direct, B=Paired, C=amb.c_plan(A)."""
    F1, T1_ = [C_(), C_(), C_(), Fw()], [Tw()] * 4             # Direct admits F only
    F2, T2 = [C_(), C_(), Fi(), C_()], [Tc(), Tw(), Tw(), Tw()]
    a = pair(F1, T1_, unit=0, arm='A') + pair(F2, T2, unit=1, arm='A') + dallas(9, 'A')
    b = pair(F1, T1_, unit=0, arm='B') + pair(F2, T2, unit=1, arm='B') + dallas(9, 'B')
    return {'A': a, 'B': b, 'C': amb.c_plan(a)}


def vec(seed):
    r = np.random.default_rng(seed)
    return {n: r.normal(0, 1, SHAPES[n]).astype(np.float32) for n in NAMES}


def good_encode(s):
    """Synthetic tokenizer: greedy longest match over a tiny vocabulary (boundary-respecting by construction)."""
    vocab = {PREFIX: [1, 2, 3], SUFFIX: [9], 'FLAIR': [10, 11], 'contrast-enhanced T1': [20, 21, 22, 23, 24], 'same': [30], 'different': [31]}
    out, i = [], 0
    while i < len(s):
        piece = max((p for p in vocab if s.startswith(p, i)), key=len, default=None)
        assert piece is not None, 'unknown synthetic text'
        out += vocab[piece]
        i += len(piece)
    return out


# ================================================================ Common32 response jobs (union of nonzero weights)
def test_response_jobs_are_the_derived_union_of_nonzero_weights_including_dallas():
    plans = three_plans()
    jobs = ngs.response_jobs(plans)
    expected = {(r['unit'], r['key']) for p in plans.values() for r in p if r['w'] > 0}
    assert {(j['unit'], j['key']) for j in jobs} == expected                        # derived, never forced to 32
    assert all(set(j['weights']) == set(j['advantages']) == {'A', 'B', 'C'} for j in jobs)
    assert any(j['task'] == 'DALLAS' for j in jobs) and any(j['task'] == 'MRI' for j in jobs)
    for j in jobs:
        for x in 'ABC':
            row = next(r for r in plans[x] if (r['unit'], r['key']) == (j['unit'], j['key']))
            assert j['weights'][x] == row['w'] and j['advantages'][x] == row['A']
    summary = ngs.job_summary(jobs)
    assert summary['responses'] == len(expected) and summary['by_task']['MRI'] + summary['by_task']['DALLAS'] == len(expected)


def test_response_jobs_refuse_any_exposure_difference_between_plans():
    plans = three_plans()
    plans['B'][3] = dict(plans['B'][3], tokens=plans['B'][3]['tokens'] + 1)
    with pytest.raises(ValueError, match='exposure'):
        ngs.response_jobs(plans)


def test_plans_come_only_from_segment0_common_parent(tmp_path):
    plans = three_plans()
    plans['A'] = [dict(r, segment=1, origin='OWN_POLICY') for r in plans['A']]
    with pytest.raises(ValueError, match='segment 0'):
        ngs.response_jobs(plans)


# ================================================================ actual GSPO derivative (ratio / clip / cap), not old==current
@pytest.mark.parametrize('shift,adv,branch,active', [
    (0., .7, 'UNCLIPPED', True), (0., -.7, 'UNCLIPPED', True),
    (.001, .7, 'CLIPPED', False), (.001, -.7, 'UNCLIPPED', True),          # ratio e^.001 > 1+.0004
    (-.001, .7, 'UNCLIPPED', True), (-.001, -.7, 'CLIPPED', False),         # ratio e^-.001 < 1-.0003
    (11., .7, 'LOG_RATIO_CAP', False)])
def test_gspo_coefficient_uses_actual_current_vs_saved_old_ratio_and_clip_branch(shift, adv, branch, active):
    old = [-.5, -1.25, -.75, -2.]
    current = [v + shift for v in old]
    c = ngs.gspo_coefficient(current, old, adv)
    ratio = math.exp(min(shift, ppg.LOG_RATIO_MAX))
    assert c['branch'] == branch and c['sequence_ratio'] == pytest.approx(ratio, rel=1e-12)
    assert c['d_loss_d_mean_logp'] == pytest.approx(-adv * ratio if active else 0., abs=1e-15)
    surrogate = ppg.response_surrogate(current, old, adv)                       # the reviewed CPU derivative, per token
    assert c['d_loss_d_mean_logp'] == pytest.approx(surrogate['d_loss_d_token_logps'][0] * len(old), abs=1e-15)
    assert c['max_abs_current_minus_old'] == pytest.approx(abs(shift), abs=1e-12)


def test_gspo_coefficient_refuses_nonfinite_or_length_mismatch():
    with pytest.raises(ValueError):
        ngs.gspo_coefficient([-.1, float('nan')], [-.1, -.2], .5)
    with pytest.raises(ValueError):
        ngs.gspo_coefficient([-.1], [-.1, -.2], .5)


def test_joint_gradients_stream_each_vector_once_and_include_dallas():
    plans = three_plans()
    jobs = ngs.response_jobs(plans)
    r = np.random.default_rng(3)
    old = {j['index']: list(r.uniform(-2, -.1, 5)) for j in jobs}
    cur = {j['index']: [v + float(r.normal(0, 2e-4)) for v in old[j['index']]] for j in jobs}   # NOT old == current
    coeffs = ngs.plan_coefficients(jobs, current=cur, old=old)
    vectors = {j['index']: vec(100 + j['index']) for j in jobs}
    loads = []
    g = ngs.joint_gradients(jobs, coeffs, load=lambda i: (loads.append(i), vectors[i])[1], names=NAMES)
    assert sorted(loads) == sorted(j['index'] for j in jobs)                   # one load per response, all three plans at once
    for x in 'ABC':
        for n in NAMES:
            manual = sum(coeffs[x][j['index']] * vectors[j['index']][n].astype(np.float64) for j in jobs)
            assert np.allclose(g[x][n], manual, rtol=0, atol=1e-6)
    for j in jobs:
        assert coeffs['A'][j['index']] == pytest.approx(j['weights']['A'] * ngs.gspo_coefficient(cur[j['index']], old[j['index']], j['advantages']['A'])['d_loss_d_mean_logp'])
    dallas_jobs = [j for j in jobs if j['task'] == 'DALLAS']
    assert dallas_jobs and all(coeffs['A'][j['index']] == coeffs['B'][j['index']] == coeffs['C'][j['index']] for j in dallas_jobs)
    for n in NAMES:                                                            # B - A carries MRI differences only
        mri_only = sum((coeffs['B'][j['index']] - coeffs['A'][j['index']]) * vectors[j['index']][n].astype(np.float64) for j in jobs if j['task'] == 'MRI')
        assert np.allclose(g['B'][n] - g['A'][n], mri_only, atol=1e-6)


def test_reconstruction_check_against_saved_joint_gradient():
    saved = vec(7)
    ok = ngs.reconstruction_check({n: saved[n] * (1 + 1e-7) for n in NAMES}, saved, relative_l2_max=1e-4)
    assert ok['PASS'] is True and ok['relative_l2'] < 1e-6 and ok['cos'] == pytest.approx(1, abs=1e-9)
    bad = ngs.reconstruction_check({n: saved[n] * 1.01 for n in NAMES}, saved, relative_l2_max=1e-4)
    assert bad['PASS'] is False and bad['relative_l2'] == pytest.approx(.01, rel=1e-3)
    nan = {n: saved[n].copy() for n in NAMES}
    nan[NAMES[0]][0, 0] = np.nan
    assert ngs.reconstruction_check(nan, saved, relative_l2_max=1e-4)['PASS'] is False
    with pytest.raises(ValueError, match='inventory'):
        ngs.reconstruction_check({NAMES[0]: saved[NAMES[0]]}, saved, relative_l2_max=1e-4)


# ================================================================ canonical margin token plans / positions / budget
def test_canonical_token_plan_uses_the_repository_prefix_and_verifies_the_full_boundary():
    assert ngs.CANONICAL_PREFIX == PREFIX and ngs.CANONICAL_SUFFIX == SUFFIX
    plan = ngs.canonical_token_plan('mri_sequence', 'contrast-enhanced T1', good_encode)
    assert plan['prefix_ids'] == [1, 2, 3] and plan['labels']['correct']['token_ids'] == [20, 21, 22, 23, 24]
    assert plan['labels']['wrong']['token_ids'] == [10, 11] and plan['single_token_labels'] is False and plan['first_identifiable'] is True
    d = ngs.canonical_token_plan('mri_visit_correspondence', 'same', good_encode)
    assert d['single_token_labels'] is True and d['labels']['wrong']['token_ids'] == [31]
    bad = lambda s: [1, 2, 3, 99, 11, 9] if s == PREFIX + 'FLAIR' + SUFFIX else good_encode(s)
    with pytest.raises(ValueError, match='boundary'):
        ngs.canonical_token_plan('mri_sequence', 'FLAIR', bad)
    with pytest.raises(ValueError):
        ngs.canonical_token_plan('mri_sequence', 'T2', good_encode)


def test_margin_positions_and_efficient_call_plan():
    mri = ngs.canonical_token_plan('mri_sequence', 'FLAIR', good_encode)
    dal = ngs.canonical_token_plan('mri_visit_correspondence', 'different', good_encode)
    pos = ngs.margin_positions(213, mri)
    assert pos['label_mean'] == {'correct': 213 + 3 + 2 - 1, 'wrong': 213 + 3 + 5 - 1} and pos['first'] == 213 + 3
    calls = ngs.margin_calls(mri)
    assert calls == {'label_mean_vjps': 2, 'first_contrast_vjps': 1, 'forwards': 3}
    assert ngs.margin_calls(dal) == {'label_mean_vjps': 0, 'first_contrast_vjps': 1, 'forwards': 1}   # mean = sum = first: one VJP
    contexts = [{'context_id': f'm{i}', 'task': 'mri_sequence', 'plan': mri, 'prompt_length': 213} for i in range(4)] + \
               [{'context_id': f'd{i}', 'task': 'mri_visit_correspondence', 'plan': dal, 'prompt_length': 405} for i in range(4)]
    responses = [{'index': i, 'prompt_length': 213, 'tokens': 30} for i in range(5)]
    b = ngs.budget_for(responses, contexts)
    assert b['model_loads'] == 1 and b['generation'] == 0 and b['optimizer_updates'] == 0 and b['parameter_writes'] == 0
    assert b['response_vjps'] == 5 and b['margin_vjps'] == 4 * 3 + 4 * 1 and b['forwards'] == 5 + 16 and b['vjps'] == 5 + 16
    assert b['teacherforced_positions'] == 5 * (213 + 30 - 1) + 4 * ((213 + 4) + (213 + 7) + (213 + 3)) + 4 * (405 + 3)


def test_margin_mean_sum_first_are_distinct_for_unequal_label_lengths():
    gF, gT, gfirst = vec(1), vec(2), vec(3)
    m = ngs.margin_gradients({FL: gF, T1: gT}, lengths={FL: 2, T1: 5}, truth=FL, first_contrast=gfirst)
    for n in NAMES:
        assert np.allclose(m['mean'][n], gF[n] - gT[n]) and np.allclose(m['sum'][n], 2 * gF[n] - 5 * gT[n]) and np.allclose(m['first'][n], gfirst[n])
    t = ngs.margin_gradients({FL: gF, T1: gT}, lengths={FL: 2, T1: 5}, truth=T1, first_contrast=gfirst)
    assert np.allclose(t['mean'][NAMES[0]], gT[NAMES[0]] - gF[NAMES[0]]) and np.allclose(t['sum'][NAMES[0]], 5 * gT[NAMES[0]] - 2 * gF[NAMES[0]])
    v = ngs.margin_values({FL: [-.2, -.4], T1: [-1., -2., -3., -1., -.5]}, truth=FL)
    assert v == pytest.approx({'mean': -.3 - (-7.5 / 5), 'sum': -.6 + 7.5, 'first': -.2 + 1.})
    single = ngs.single_token_margin(gfirst, value=.25)
    assert single['gradients']['mean'] is single['gradients']['sum'] is single['gradients']['first'] and single['values'] == {'mean': .25, 'sum': .25, 'first': .25}


# ================================================================ read-only Adam52 displacement
def test_readonly_adam52_direction_formula_and_actual_float32_rounded_displacement():
    g, m, v = vec(4), vec(5), {n: np.abs(vec(6)[n]) * 1e-4 for n in NAMES}
    theta = {n: (vec(11)[n] * np.float32(.02)).astype(np.float32) for n in NAMES}
    m0, v0, t0 = copy.deepcopy(m), copy.deepcopy(v), copy.deepcopy(theta)
    profile = {'learning_rate': 1e-5, 'betas': (.9, .999), 'eps': 1e-8, 'bias_correction': False, 'step': 52}
    d = ngs.adam_directions(g, m, v, profile, ops=np, theta=theta)
    for n in NAMES:
        mn = .9 * m[n] + .1 * g[n]
        vn = .999 * v[n] + .001 * np.square(g[n])
        formula = -1e-5 * mn / (np.sqrt(vn) + 1e-8)
        assert np.allclose(d['adam52_formula'][n], formula, rtol=1e-6)
        rounded = (theta[n] + d['adam52_formula'][n]).astype(np.float32) - theta[n]      # actual F32 parameter displacement
        assert d['adam52_rounded'][n].dtype == np.float32 and np.array_equal(d['adam52_rounded'][n], rounded)
        mom = -1e-5 * (.9 * m[n]) / (np.sqrt(.999 * v[n]) + 1e-8)
        assert np.allclose(d['momentum_only_formula'][n], mom, rtol=1e-6)
        assert np.array_equal(d['momentum_only_rounded'][n], (theta[n] + d['momentum_only_formula'][n]).astype(np.float32) - theta[n])
        assert np.array_equal(d['adam52_incremental_rounded'][n], d['adam52_rounded'][n] - d['momentum_only_rounded'][n])
        assert np.allclose(d['adam52_incremental_formula'][n], d['adam52_formula'][n] - d['momentum_only_formula'][n])
        assert np.allclose(d['negative_gradient'][n], -g[n])
        assert np.array_equal(m[n], m0[n]) and np.array_equal(v[n], v0[n]) and np.array_equal(theta[n], t0[n])   # nothing written
    with pytest.raises(ValueError):
        ngs.adam_directions(g, m, v, {**profile, 'bias_correction': True}, ops=np, theta=theta)
    with pytest.raises(ValueError):
        ngs.adam_directions(g, m, {n: -v[n] for n in NAMES}, profile, ops=np, theta=theta)
    with pytest.raises(ValueError):
        ngs.adam_directions(g, m, v, {**profile, 'step': 53}, ops=np, theta=theta)


def test_directional_effects_report_dot_norm_and_cosine_separately():
    grad = vec(8)
    dirs = {'A': {'negative_gradient': vec(9)}, 'C': {'negative_gradient': vec(10)}}
    e = ngs.directional_effects({'m0': {'mean': grad}}, dirs)
    flat = lambda d: np.concatenate([d[n].astype(np.float64).ravel() for n in NAMES])
    for x in 'AC':
        cell = e['effects'][x]['negative_gradient']['mean']['m0']
        assert cell == pytest.approx(float(flat(grad) @ flat(dirs[x]['negative_gradient'])), rel=1e-9)
    assert e['direction_norms']['C']['negative_gradient'] == pytest.approx(float(np.linalg.norm(flat(dirs['C']['negative_gradient']))), rel=1e-9)
    a, c = flat(dirs['A']['negative_gradient']), flat(dirs['C']['negative_gradient'])
    assert e['direction_cosines']['negative_gradient']['A|C'] == pytest.approx(float(a @ c / np.linalg.norm(a) / np.linalg.norm(c)), rel=1e-9)


# ================================================================ predeclared classification (per image, primary MEAN, Adam52 direction)
CONTEXTS = {'F0': 'MRI_FLAIR', 'F1': 'MRI_FLAIR', 'T0': 'MRI_T1', 'T1': 'MRI_T1', 'D0': 'DALLAS', 'D1': 'DALLAS'}
BAND = {c: 1e-3 for c in CONTEXTS}


def effects(c_adam, a_adam=None, b_adam=None, c_neg=None, sum_c=None):
    a_adam = a_adam or {c: 0. for c in CONTEXTS}
    b_adam = b_adam or {c: 0. for c in CONTEXTS}
    def block(adam, neg, s):
        return {'adam52_rounded': {'mean': dict(adam), 'sum': dict(s or adam), 'first': dict(adam)},
                'negative_gradient': {'mean': dict(neg or adam), 'sum': dict(neg or adam), 'first': dict(neg or adam)}}
    return {'A': block(a_adam, None, None), 'B': block(b_adam, None, None), 'C': block(c_adam, c_neg, sum_c)}


GOOD = {'F0': .01, 'F1': .002, 'T0': .02, 'T1': .03, 'D0': 0., 'D1': .001}
RECON_OK = {'A': {'PASS': True}, 'B': {'PASS': True}}


COUNTS = {'MRI_FLAIR': 2, 'MRI_T1': 2, 'DALLAS': 2}           # explicit for this synthetic fixture; production uses REQUIRED_CONTEXT_COUNTS (2/2/4)


def decide(e, recon=RECON_OK, parity=True, band=BAND, contexts=CONTEXTS, counts=COUNTS, extra=None):
    return ngs.classify(e, contexts, band, reconstruction=recon, parity={'PASS': parity}, expected_counts=counts, extra_preconditions=extra)


def test_classification_pass_requires_every_image_and_context():
    assert decide(effects(GOOD))['decision'] == 'PASS_FOR_SMALL_PILOT'


@pytest.mark.parametrize('change,expected', [
    ({'T0': -.02}, 'FAIL_REJECT_C'),                       # T1 margin materially decreases
    ({'T0': .0005}, 'INCONCLUSIVE'),                       # inside numeric band: cannot establish a positive change
    ({'F1': -.05}, 'FAIL_REJECT_C'),                       # one FLAIR image harmed even though F0 is up (no averaging)
    ({'D1': -.05}, 'FAIL_REJECT_C'),                       # Dallas worse than both A and B beyond band
])
def test_classification_per_image_failures_are_not_averaged_away(change, expected):
    e = effects({**GOOD, **change})
    assert decide(e)['decision'] == expected


def test_T1_must_beat_both_A_and_B_in_the_adam_direction():
    e = effects(GOOD, b_adam={**{c: 0. for c in CONTEXTS}, 'T1': .06})
    out = decide(e)
    assert out['decision'] == 'FAIL_REJECT_C' and 'T1' in out['failures']['T1_not_better_than_A_B']


def test_better_only_under_negative_gradient_is_not_qualification():
    e = effects({**GOOD, 'T0': -.02}, c_neg={**GOOD, 'T0': .05})
    out = decide(e)
    assert out['decision'] == 'FAIL_REJECT_C' and 'NEGATIVE_GRADIENT_ONLY' in out['flags']


def test_preconditions_force_inconclusive_never_pass():
    assert decide(effects(GOOD), recon={'A': {'PASS': True}, 'B': {'PASS': False}})['decision'] == 'INCONCLUSIVE'
    assert decide(effects(GOOD), parity=False)['decision'] == 'INCONCLUSIVE'
    missing = effects(GOOD)
    del missing['C']['adam52_rounded']['mean']['T1']
    assert decide(missing)['decision'] == 'INCONCLUSIVE'
    nan = effects({**GOOD, 'F0': float('nan')})
    assert decide(nan)['decision'] == 'INCONCLUSIVE'


def test_metric_sign_disagreement_is_flagged_without_switching_the_primary_endpoint():
    out = decide(effects(GOOD, sum_c={**GOOD, 'T0': -.02}))
    assert out['decision'] == 'PASS_FOR_SMALL_PILOT' and out['primary'] == {'metric': 'mean', 'direction': 'adam52_rounded'}
    assert any('T0' in f for f in out['flags'] if f.startswith('METRIC_SIGN_DISAGREEMENT'))


def test_classification_uses_pair_band_against_each_of_A_and_B_with_nested_taus():
    taus = {fam: {met: {c: 1e-3 for c in CONTEXTS} for met in ('mean', 'sum', 'first')} for fam in ('adam52_rounded', 'negative_gradient')}
    within = effects({**GOOD, 'T0': .0045}, a_adam={**{c: 0. for c in CONTEXTS}, 'T0': .003})       # C-A = .0015 < 2*tau
    assert decide(within, band=taus)['decision'] == 'INCONCLUSIVE'
    flair = effects({**GOOD, 'F0': .0}, b_adam={**{c: 0. for c in CONTEXTS}, 'F0': .0025})           # C-B = -.0025 < -2*tau
    assert decide(flair, band=taus)['decision'] == 'FAIL_REJECT_C'


def test_classification_fails_closed_on_context_or_metric_inventory():
    assert decide(effects(GOOD), contexts={}, counts={})['decision'] == 'INCONCLUSIVE'                 # empty inventory
    out = decide(effects(GOOD), counts=ngs.REQUIRED_CONTEXT_COUNTS)                                    # 2 Dallas given, 4 required
    assert out['decision'] == 'INCONCLUSIVE' and 'context_inventory_mismatch' in out['preconditions_failed']
    no_sum = effects(GOOD)
    del no_sum['B']['negative_gradient']['sum']['D0']                                                  # a secondary is missing: no fake 'unavailable'
    assert decide(no_sum)['decision'] == 'INCONCLUSIVE'
    assert decide(effects(GOOD), extra={'first_token_oracle_alignment': False})['decision'] == 'INCONCLUSIVE'


def test_first_token_oracle_must_match_each_label_mean_forward_and_is_the_primary_first_value():
    ok = ngs.first_token_alignment([-.20, -1.90], [-.2005, -.4], [-1.9010, -1., -2., -3., -.5])
    assert ok['PASS'] is True and ok['max_abs'] == pytest.approx(.001)
    bad = ngs.first_token_alignment([-.20, -1.90], [-.2005, -.4], [-1.95, -1., -2., -3., -.5])       # wrong label off by .05 > .002
    assert bad['PASS'] is False and bad['wrong_abs_error'] == pytest.approx(.05)
    v = ngs.context_margin_values([-.2005, -.4], [-1.9010, -1., -2., -3., -.5], [-.20, -1.90])
    assert v['first'] == pytest.approx(-.20 + 1.90) and v['first_from_label_forwards'] == pytest.approx(-.2005 + 1.9010)
    assert v['mean'] == pytest.approx(-.30025 - (-8.401 / 5)) and v['sum'] == pytest.approx(-.6005 + 8.401)
    assert ngs.TOLERANCES['first_token_alignment_max_abs'] == .002 and ngs.TOLERANCES['own_old_max_abs'] == 1e-5


def test_native_F32_ratio_capture_is_the_coefficient_source_with_cpu_crosscheck():
    old = [-.5, -1.25, -.75, -2.]
    cur = [v + 1e-4 for v in old]
    delta = math.fsum(a - b for a, b in zip(cur, old)) / 4
    nat = {'log_ratio': delta + 3e-7, 'ratio': math.exp(delta + 3e-7)}                     # F32 capture differs by rounding only
    c = ngs.gspo_coefficient(cur, old, .7, native=nat)
    assert c['ratio_source'] == 'NATIVE_F32_CAPTURE' and c['branch'] == 'UNCLIPPED'
    assert c['d_loss_d_mean_logp'] == pytest.approx(-.7 * nat['ratio'], abs=1e-15) and c['native_minus_cpu_log_ratio'] == pytest.approx(3e-7, abs=1e-12)
    far = ngs.gspo_coefficient(cur, old, .7, native={'log_ratio': delta + 1e-3, 'ratio': math.exp(delta + 1e-3)})
    assert far['branch'] == 'AMBIGUOUS_NATIVE_CPU_DISCREPANCY' and far['d_loss_d_mean_logp'] is None
    jobs = [{'index': 0, 'unit': 0, 'task': 'MRI', 'weights': {x: W8 for x in 'ABC'}, 'advantages': {x: .7 for x in 'ABC'}}]
    with pytest.raises(ValueError, match='AMBIGUOUS'):
        ngs.plan_coefficients(jobs, current={0: cur}, old={0: old}, native={0: {'log_ratio': delta + 1e-3, 'ratio': math.exp(delta + 1e-3)}})
    recs = [{'index': 0, 'f32_sequence_log_ratio': nat['log_ratio'], 'f32_sequence_ratio': nat['ratio']}, {'index': 1}]
    with pytest.raises(ValueError, match='Partial'):
        ngs.native_ratios(recs)


W8 = .25 / 8


def test_frozen_tolerances_and_empirical_tau():
    assert ngs.TOLERANCES['own_old_max_abs'] == 1e-5 and ngs.TOLERANCES['reconstruction_relative_l2_max'] == 1e-4
    assert ngs.TOLERANCES['tau_floor'] == 1e-6 and ngs.TOLERANCES['tau_error_multiplier'] == 5. and ngs.TOLERANCES['pair_multiplier'] == 2.
    ngs.require_frozen_tolerances(dict(ngs.TOLERANCES))
    with pytest.raises(ValueError, match='frozen'):
        ngs.require_frozen_tolerances({**ngs.TOLERANCES, 'tau_error_multiplier': 2.})
    assert ngs.empirical_tau(1e-7, -1e-7) == 1e-6                                          # floor
    assert ngs.empirical_tau(-3e-6, 1e-6) == pytest.approx(1.5e-5)                          # 5 * max |error|
    err = {'A': {'adam52_rounded': {'mean': {'T0': 4e-6}}}, 'B': {'adam52_rounded': {'mean': {'T0': -6e-6}}}}
    assert ngs.empirical_taus(err)['adam52_rounded']['mean']['T0'] == pytest.approx(3e-5)
    with pytest.raises(ValueError):
        ngs.empirical_tau(float('nan'), 0.)


# ================================================================ authorization / one-shot controller with injected ops
def scope_fixture(tmp_path, responses=3, margins=2):
    """TEST HELPER (not production): a generic scope assembled from synthetic jobs via the production assembler."""
    rjobs = [{'index': i, 'unit': 0, 'key': f'F{i}', 'task': 'MRI', 'tokens': 4 + i, 'prompt_length': 213, 'tf_positions': 213 + 4 + i - 1}
             for i in range(responses)]
    mjobs = [{'index': k, 'context_id': f'c{k}', 'objective': 'FIRST_CONTRAST', 'positions': 216 + k, 'synthetic_label_logps': [-.3, -1.2]}
             for k in range(margins)]
    return ngs.assemble_scope(response_jobs=rjobs, margin_jobs=mjobs, run_output=tmp_path / 'run_PRIVATE', sources={}, files={}, payload={},
                              runtime={}, preparation={}, bindings={'synthetic_test_fixture': True})


def credentials(scope, *, reviewer='isolated_codex_reviewer', kind='isolated_codex', minutes_ago=1, sha=None):
    now = datetime.now(timezone.utc)
    human = {'authorized': True, 'action': entry.ACTION, 'scope_sha256': sha or ngs.digest(scope), 'human_instruction_sha256': 'a' * 64,
             'approved_by': 'researcher', 'issued_at_utc': (now - timedelta(minutes=minutes_ago)).isoformat(),
             'expires_at_utc': (now + timedelta(hours=2)).isoformat()}
    review = {'verdict': 'PASS', 'scope_sha256': ngs.digest(scope), 'reviewer_id': reviewer, 'reviewer_kind': kind,
              'source_sha256': scope['source_sha256'], 'input_sha256': scope['input_sha256']}
    return human, review


class FakeOps:
    def __init__(self, scope, *, fail_at=None, release_bytes=0, drift=False, extra_positions=0, reconstruction_pass=True):
        self.scope, self.fail_at, self.release_bytes, self.drift, self.extra = scope, fail_at, release_bytes, drift, extra_positions
        self.reconstruction_pass = reconstruction_pass
        self.constructed = False
        self.released = False
        self.calls = []

    def reconstruction_gate(self, scope, records):
        assert len(records) == len(scope['response_jobs_PRIVATE'])
        return {'PASS': self.reconstruction_pass, 'reason': 'synthetic'}

    def construct(self, scope, gate):
        self.constructed = True
        return object()

    def load_parent(self, backend, scope):
        return {'parameters': 'p', 'optimizer': 'o', 'rng': 'r', 'policy': 'P', 'adam': 'A', 'optimizer_step': 52}

    def full_identity(self, backend):
        base = self.load_parent(backend, None)
        return {**base, 'rng': 'changed'} if self.drift and len(self.calls) > 0 else base

    def response_vjp(self, backend, job):
        self.calls.append(('response', job['index']))
        if self.fail_at == len(self.calls):
            raise RuntimeError('synthetic native failure')
        return {'selected_logps': [-.5] * job['tokens'], 'positions': job['tf_positions'] + self.extra, 'vector': vec(job['index'])}

    def margin_vjp(self, backend, mjob):
        self.calls.append(('margin', mjob['index']))
        if self.fail_at == len(self.calls):
            raise RuntimeError('synthetic native failure')
        return {'label_logps': mjob['synthetic_label_logps'], 'positions': mjob['positions'], 'vector': vec(50 + mjob['index'])}

    def save_vector(self, path, vector):
        np.savez(path, **{k.replace('.', '__'): v for k, v in vector.items()})
        return ngs.sha256_file(path)

    def release(self, backend, directory):
        self.released = True
        return {'model_is_none': True, 'active_MLX_bytes': self.release_bytes, 'release_verified': self.release_bytes < 128 * 1024 ** 2}


class Gate:
    def __init__(self):
        self.phases, self.loads = [], 0

    def __call__(self, phase):
        self.phases.append(phase)

    def claim_model_load(self):
        self.loads += 1
        assert self.loads <= 1


def run(scope, ops, human, review, consumed=None):
    token = entry.authorize(scope, human, review, consumed=consumed or (lambda h: None))
    return entry.run_readonly(scope, token, Gate(), ops)


def test_authorization_is_checked_before_any_ops_construction(tmp_path):
    scope = scope_fixture(tmp_path)
    for kw in ({'sha': '0' * 64}, {'minutes_ago': 120}, {'reviewer': 'claude-opus-5-5', 'kind': 'claude_code'}):
        ops = FakeOps(scope)
        human, review = credentials(scope, **kw)
        with pytest.raises(ValueError):
            run(scope, ops, human, review)
        assert ops.constructed is False and not Path(scope['run_output']).exists()
    loosened = {**scope, 'tolerances': {**scope['tolerances'], 'reconstruction_relative_l2_max': 1e-2}}
    human, review = credentials(loosened)
    ops = FakeOps(loosened)
    with pytest.raises(ValueError, match='frozen'):
        run(loosened, ops, human, review)
    assert ops.constructed is False


def test_successful_readonly_run_is_journaled_counted_released_and_one_shot(tmp_path):
    scope = scope_fixture(tmp_path)
    human, review = credentials(scope)
    ops = FakeOps(scope)
    result = run(scope, ops, human, review)
    assert result['status'] == 'PASS' and result['optimizer_updates'] == 0 and result['generation'] == 0 and result['model_loads'] == 1
    assert result['counters'] == {k: scope['budget'][k] for k in ('forwards', 'vjps', 'teacherforced_positions')}
    assert result['identity_before_equals_after'] is True and result['release_verified'] is True
    assert [c[0] for c in ops.calls] == ['response'] * 3 + ['margin'] * 2                  # serial, responses then margins
    vectors = list(Path(scope['run_output']).rglob('*_vector_PRIVATE.npz'))
    assert len(vectors) == 5
    human, review = credentials(scope)
    with pytest.raises(ValueError, match='exist|replay'):
        run(scope, FakeOps(scope), human, review)                                       # never overwritten or replayed


@pytest.mark.parametrize('ops_kw,match', [
    ({'fail_at': 2}, 'synthetic native failure'),
    ({'release_bytes': 512 * 1024 ** 2}, 'release'),
    ({'drift': True}, 'identity'),
    ({'extra_positions': 7}, 'budget|positions'),
])
def test_failures_leave_no_complete_record_and_block_replay(tmp_path, ops_kw, match):
    scope = scope_fixture(tmp_path)
    human, review = credentials(scope)
    with pytest.raises(Exception, match=match):
        run(scope, FakeOps(scope, **ops_kw), human, review)
    out = Path(scope['run_output'])
    assert not (out / 'native_complete_PRIVATE.json').exists()
    human, review = credentials(scope)
    ops = FakeOps(scope)
    with pytest.raises(Exception):
        run(scope, ops, human, review)                                                  # partial/indeterminate: no automatic retry
    assert ops.calls == []


class Payload:
    """Stands in for a resident model / autograd graph (weak-referenceable)."""
    def __init__(self):
        self.data = np.zeros(1000)


class PartialConstructOps(FakeOps):
    """__init__ fails AFTER a resident object was allocated and registered with ops (like PilotV2Backend.__init__ mid-way)."""
    def construct(self, scope, gate):
        import weakref
        self.constructed = True
        self.resident = Payload()
        self.resident_ref = weakref.ref(self.resident)
        raise RuntimeError('synthetic __init__ failure after allocation')

    def release(self, backend, directory):
        import gc
        target = backend if backend is not None else self.resident
        self.release_target_was_partial = backend is None and target is not None
        self.resident = target = None
        gc.collect()
        self.released = True
        return {'model_is_none': True, 'active_MLX_bytes': 0, 'release_verified': self.resident_ref() is None}


class GraphRetainingOps(FakeOps):
    """The failing native call holds a 'graph' in its frame; release must observe it already freed (frames cleared first)."""
    def response_vjp(self, backend, job):
        import weakref
        graph = Payload()
        self.graph_ref = weakref.ref(graph)
        self.calls.append(('response', job['index']))
        raise RuntimeError('synthetic VJP failure holding a graph')

    def release(self, backend, directory):
        import gc
        gc.collect()
        self.graph_alive_at_release = self.graph_ref() is not None
        return super().release(backend, directory)


def test_partially_constructed_resident_is_released_and_original_error_preserved(tmp_path):
    scope = scope_fixture(tmp_path)
    human, review = credentials(scope)
    ops = PartialConstructOps(scope)
    with pytest.raises(RuntimeError, match='__init__ failure'):
        run(scope, ops, human, review)
    assert ops.released and ops.release_target_was_partial and ops.resident_ref() is None
    failure = json.loads((Path(scope['run_output']) / 'failure_PRIVATE.json').read_text())
    assert failure['error_type'] == 'RuntimeError' and failure['release']['release_verified'] is True and failure['automatic_retry'] is False
    assert not (Path(scope['run_output']) / 'native_complete_PRIVATE.json').exists()


def test_exception_frames_are_cleared_before_release_and_spent_budget_is_not_refunded(tmp_path):
    scope = scope_fixture(tmp_path)
    human, review = credentials(scope)
    ops = GraphRetainingOps(scope)
    with pytest.raises(RuntimeError, match='holding a graph'):
        run(scope, ops, human, review)
    assert ops.graph_alive_at_release is False                                  # traceback no longer pins the failed call's graph
    out = Path(scope['run_output'])
    failure = json.loads((out / 'failure_PRIVATE.json').read_text())
    assert failure['spent_including_indeterminate']['forwards'] == 1             # charged BEFORE the call, kept after failure
    assert json.loads((out / 'counters_PRIVATE.json').read_text())['used']['forwards'] == 1
    assert 'holding a graph' in failure['traceback']


class WrongLengthMarginOps(FakeOps):
    def margin_vjp(self, backend, mjob):
        r = super().margin_vjp(backend, mjob)
        return {**r, 'label_logps': r['label_logps'] + [-.1]}                     # 3 values for a FIRST_CONTRAST pair


def test_margin_logp_length_must_match_the_frozen_objective(tmp_path):
    scope = scope_fixture(tmp_path)
    human, review = credentials(scope)
    with pytest.raises(ValueError, match='exact length'):
        run(scope, WrongLengthMarginOps(scope), human, review)


def test_failed_A_B_reconstruction_stops_before_any_margin_call_and_commits_PARTIAL(tmp_path):
    scope = scope_fixture(tmp_path)
    human, review = credentials(scope)
    ops = FakeOps(scope, reconstruction_pass=False)
    result = run(scope, ops, human, review)
    assert result['status'] == 'PARTIAL_RECONSTRUCTION_FAILED' and result['C_science'] == 'NOT_EVALUATED'
    assert [c[0] for c in ops.calls] == ['response'] * 3 and ops.released is True and result['release_verified'] is True
    assert (Path(scope['run_output']) / 'native_complete_PRIVATE.json').exists()             # terminal, not indeterminate
    assert result['counters']['forwards'] == 3 and result['optimizer_updates'] == 0


def test_human_permission_is_consumed_once(tmp_path):
    scope = scope_fixture(tmp_path)
    human, review = credentials(scope)
    used = set()
    def consume(h):
        if h in used:
            raise ValueError('permission already consumed')
        used.add(h)
    entry.authorize(scope, human, review, consumed=consume)
    with pytest.raises(ValueError, match='consumed'):
        entry.authorize(scope, human, review, consumed=consume)


# ================================================================ public aggregate and isolation
def test_public_view_of_an_actual_style_decision_carries_no_context_ids(tmp_path):
    ids = {'u0_F': 'MRI_FLAIR', 'u0_T': 'MRI_T1', 'u1_F': 'MRI_FLAIR', 'u1_T': 'MRI_T1', 'u2_D0': 'DALLAS', 'u2_D1': 'DALLAS', 'u3_D0': 'DALLAS', 'u3_D1': 'DALLAS'}
    good = {'u0_F': .01, 'u1_F': .002, 'u0_T': .02, 'u1_T': -.03, 'u2_D0': 0., 'u2_D1': 0., 'u3_D0': -.05, 'u3_D1': 0.}
    zero = {c: 0. for c in ids}
    blk = lambda v: {fam: {met: dict(v) for met in ('mean', 'sum', 'first')} for fam in ('adam52_rounded', 'negative_gradient')}
    e = {'A': blk(zero), 'B': blk(zero), 'C': blk(good)}
    d = ngs.classify(e, ids, {c: 1e-3 for c in ids}, reconstruction=RECON_OK, parity={'PASS': True}, expected_counts=ngs.REQUIRED_CONTEXT_COUNTS)
    assert d['decision'] == 'FAIL_REJECT_C' and 'u1_T' in d['failures']['T1_not_positive'] and 'u3_D0' in d['failures']['DALLAS_worse_than_A_or_B']
    report = {'decision_PRIVATE': d, 'decision': ngs.public_decision(d), 'per_context_table_PRIVATE': {c: {'x': 1} for c in ids}}
    pub = ngs.public_aggregate(report, forbidden_values=list(ids))
    text = json.dumps(pub)
    assert not any(c in text for c in ids) and pub['decision']['failure_counts'] == {'T1_not_positive': 1, 'DALLAS_worse_than_A_or_B': 1}
    with pytest.raises(ValueError, match='identifier'):
        ngs.public_aggregate({'decision': d}, forbidden_values=list(ids))       # a raw decision with IDs is refused, not silently published


def test_public_aggregate_carries_no_vectors_paths_or_case_keys(tmp_path):
    report = {'decision': {'decision': 'INCONCLUSIVE', 'flags': []}, 'effects_PRIVATE': {'C': {'x': 1}},
              'summary': {'by_kind': {'MRI_T1': {'C_adam_mean_min': .1}}}, 'vectors_PRIVATE': [str(tmp_path / 'v.npz')]}
    pub = ngs.public_aggregate(report)
    text = json.dumps(pub)
    assert '_PRIVATE' not in text and str(tmp_path) not in text and pub['public_case_keys']['status'] == 'PASS'
    assert pub['claims']['accuracy'] == 'NO_NEW_ACCURACY_RESULT' and pub['claims']['margins'] == 'DIAGNOSTIC_ONLY_NOT_A_LOSS'


def test_cpu_modules_and_lazy_backend_import_no_heavy_runtime():
    code = ('import sys\n'
            'import mri_rl.native_group_signal_vjp_v1, mri_rl.native_group_signal_vjp_entry_v1, mri_rl.native_group_signal_vjp_backend_v1\n'
            'bad = [m for m in sys.modules if m.split(".")[0] in ("mlx", "mlx_vlm", "mlx_lm", "torch", "jax", "transformers")]\n'
            'assert not bad, bad\nprint("OK")\n')
    out = subprocess.run([sys.executable, '-c', code], env={**os.environ, 'PYTHONPATH': str(SRC)}, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0 and out.stdout.strip() == 'OK', out.stderr
