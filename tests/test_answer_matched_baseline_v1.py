"""Synthetic CPU tests for candidate C = ANSWER-MATCHED CROSS-IMAGE BASELINE (counterfactual advantages; no model/MRI data).

Every table here is SYNTHETIC and labelled as such; it falsifies or checks mathematical claims only, never MRI learning.
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_k, '1')
import json
import math
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mri_rl import answer_matched_baseline_v1 as amb           # noqa: E402
from mri_rl import group_relative_visual_signal_v1 as grv      # noqa: E402
from mri_rl import v2_regression_diagnosis_v1 as diag          # noqa: E402
from mri_rl.mlx_gspo_smoke import advantages_from_rewards       # noqa: E402
from mri_rl.uniform_balance_continuation_gate_v1 import accuracy_group_is_informative   # noqa: E402

grv.install_import_guard()
CAL = {1: .5, 0: .0243088}
FL, T1, INV = 'FLAIR', 'contrast-enhanced T1', 'INVALID_FORMAT'
W = .25 / 8


# ================================================================ SYNTHETIC pair builder: Direct (A) and Paired (B) plans of the SAME 8 responses
def r_(image, correct, fmt, answer, tokens=30):
    acc = CAL[correct]
    return {'image': image, 'correct': correct, 'format': fmt, 'valid': fmt == 1, 'acc_cal': acc, 'overlong': 0., 'tokens': tokens,
            'answer': answer, 'r': .8 * acc + .2 * fmt, 'r_raw': .8 * correct + .2 * fmt}


def pair(F, T, *, unit=0, arm='A', segment=0):
    """F, T: four responses each. arm A = same-image groups [F0..F3],[T0..T3]; arm B = [F0,F1,T0,T1],[F2,F3,T2,T3]."""
    items = [dict(x, key=f'F{i}') for i, x in enumerate(F)] + [dict(x, key=f'T{i}') for i, x in enumerate(T)]
    groups = ([[0, 1, 2, 3], [4, 5, 6, 7]] if arm == 'A' else [[0, 1, 4, 5], [2, 3, 6, 7]])
    rows = [None] * 8
    for g, idx in enumerate(groups):
        members = [items[i] for i in idx]
        adv = advantages_from_rewards([m['r'] for m in members])
        ok = accuracy_group_is_informative([{'accuracy': m['acc_cal'], 'format': m['format'], 'total': m['r'], 'overlong': 0.} for m in members])
        for i, m, a in zip(idx, members, adv):
            rows[i] = {**m, 'segment': segment, 'arm': arm, 'origin': 'COMMON_PARENT' if segment == 0 else 'OWN_POLICY', 'unit': unit, 'task': 'MRI',
                       'group': g, 'admitted': ok, 'w': W if ok else 0., 'A': a, 'regrade_match': None}
    return rows


def dallas(unit=9, arm='A'):
    out = []
    for g in range(2):
        rewards = [.6, .2, .6, .2] if g == 0 else [.6] * 4
        adv = advantages_from_rewards(rewards)
        for i, (rw, a) in enumerate(zip(rewards, adv)):
            out.append({'segment': 0, 'arm': arm, 'origin': 'COMMON_PARENT', 'unit': unit, 'task': 'DALLAS', 'key': f'D{g}_{i}', 'group': g,
                        'image': f'D:D{g}', 'correct': int(rw > .5), 'format': 1., 'valid': True, 'acc_cal': (rw - .2) / .8, 'overlong': 0.,
                        'r': rw, 'r_raw': rw, 'tokens': 20, 'answer': None, 'admitted': g == 0, 'w': W if g == 0 else 0., 'A': a, 'regrade_match': None})
    return out


C_ = lambda: r_('F', 1, 1., FL)          # FLAIR image answered FLAIR (correct)
Fw = lambda: r_('F', 0, 1., T1)          # FLAIR image answered T1 (wrong, valid)
Fi = lambda: r_('F', 0, 0., INV)         # FLAIR image, invalid format
Tc = lambda: r_('T', 1, 1., T1)          # T1 image answered T1 (correct)
Tw = lambda: r_('T', 0, 1., FL)          # T1 image answered FLAIR (wrong, valid)
Ti = lambda: r_('T', 0, 0., INV)


def by_key(rows):
    return {r['key']: r for r in rows}


# ================================================================ definition / equivalence boundaries
def test_constant_answer_pair_is_exactly_one_paired_group_of_eight():
    """All 8 answer FLAIR (the published counterexample shape): one answer-matched set = the whole pair -> C == Paired-8."""
    rows = amb.c_plan(pair([C_()] * 4, [Tw()] * 4))
    eight = advantages_from_rewards([r['r'] for r in rows])
    assert all(r['baseline_kind'] == 'ANSWER_MATCHED' for r in rows)
    assert [r['A'] for r in rows] == pytest.approx(eight, abs=1e-12)
    assert all(r['A'] > 0 for r in rows[:4]) and all(r['A'] < 0 for r in rows[4:]) and all(r['admitted'] for r in rows)
    assert all(r['arm'] == 'C' and r['origin_label'] == amb.LABEL for r in rows)


def test_perfectly_discriminating_saturated_pair_gives_no_signal():
    rows = amb.c_plan(pair([C_()] * 4, [Tc()] * 4))
    assert all(r['baseline_kind'] == 'SAME_IMAGE_FALLBACK' for r in rows)
    assert all(abs(r['A']) < 1e-12 and r['w'] == 0 and not r['admitted'] for r in rows)


def test_no_shared_answer_reduces_to_direct_centering_with_pair_scale():
    """When no valid answer occurs on both images, C is Direct same-image centering rescaled by the pair SD (equivalence declared)."""
    F, T = [C_(), C_(), Fi(), C_()], [Tc(), Ti(), Tc(), Tc()]
    rows = amb.c_plan(pair(F, T))
    sd = amb.sample_sd([r['r'] for r in rows]) + amb.ADV_EPS
    for img in ('F', 'T'):
        sub = [r for r in rows if r['image'] == img]
        m = sum(r['r'] for r in sub) / 4
        assert [r['A'] for r in sub] == pytest.approx([(r['r'] - m) / sd for r in sub], abs=1e-12)
    assert all(r['baseline_kind'] == 'SAME_IMAGE_FALLBACK' for r in rows)


def test_action_dependent_baseline_changes_the_expected_score_gradient():
    """Independent reviewer counterexample (SYNTHETIC): label-conditioned population baselines are policy-dependent, not the midpoint,
    so C is a changed action-dependent surrogate/estimator, not an unbiased baseline."""
    rw = .2194470404912912
    out = amb.action_conditioned_population_bias(.8, .6, .6, rw)
    assert out['b_FLAIR'] == pytest.approx(.4369058744962677, abs=1e-12) and out['b_T1'] == pytest.approx(.4731490134970971, abs=1e-12)
    assert out['midpoint'] == pytest.approx(.4097235202456456, abs=1e-12) and out['b_FLAIR'] != out['b_T1']
    assert out['true_reward_beta_gradient'] == pytest.approx(-.015222118380348353, abs=1e-12)
    assert out['baseline_added_term'] == pytest.approx(.007248627800165874, abs=1e-12)
    assert out['resulting_beta_gradient'] == pytest.approx(-.007973490580182478, abs=1e-12)
    doc = amb.__doc__
    assert 'ACTION-DEPENDENT' in doc and 'no expected bias' not in doc and 'not the expected objective' not in doc
    assert amb.INTERVENTION['isolated_baseline_only_causal_test'] is False and 'ACTION_DEPENDENT' in amb.INTERVENTION['estimator']


def test_global_centering_is_lost_when_same_image_fallback_overlaps_a_matched_set():
    """F: two FLAIR-correct (matched with T's FLAIR-wrong), one T1-wrong and one invalid on F use the F-image mean, which contains the
    matched F responses. Each matched set sums to zero; the pair does not. Reported, deliberately not recentered."""
    rows = amb.c_plan(pair([C_(), C_(), Fw(), Fi()], [Tw()] * 4))
    kinds = [r['baseline_kind'] for r in rows]
    assert kinds.count('ANSWER_MATCHED') == 6 and kinds.count('SAME_IMAGE_FALLBACK') == 2
    m = amb.centering_metrics(rows)
    assert m['answer_matched_sets'] == 1 and m['answer_matched_set_max_abs_sum'] < 1e-12
    manual = sum(r['w'] * r['A'] for r in rows if r['w'] > 0)
    assert m['signed_weighted_advantage_MRI_total'] == pytest.approx(manual, abs=1e-15) and abs(manual) > 1e-6
    assert m['global_centering_lost'] is True
    assert m['signed_weighted_advantage_by_image']['F'] + m['signed_weighted_advantage_by_image']['T'] == pytest.approx(manual, abs=1e-15)
    direct = pair([C_(), C_(), Fw(), Fi()], [Tw()] * 4)
    assert abs(sum(r['A'] for r in direct if r['image'] == 'F')) < 1e-12                 # Direct K4 group stays centered
    q = amb.quality(rows, amb.split_parts(rows), direct_rows=direct)
    assert q['centering']['global_centering_lost'] is True and 'BY CONSTRUCTION' in q['sign_note']


def test_answer_matched_sign_equals_correctness_by_construction_and_each_matched_set_sums_to_zero():
    """Sign agreement is BY CONSTRUCTION (a shared binary label is correct on exactly one image): a definition check, not learning evidence."""
    rows = amb.c_plan(pair([C_(), Fw(), C_(), Fi()], [Tw(), Tc(), Tw(), Tw()]))
    for r in rows:
        if r['baseline_kind'] == 'ANSWER_MATCHED':
            assert (r['A'] > 0) == (r['correct'] == 1) and abs(r['A']) > 1e-9
    for ans in (FL, T1):
        s = [r['A'] for r in rows if r['baseline_kind'] == 'ANSWER_MATCHED' and r['answer'] == ans]
        assert s and abs(sum(s)) < 1e-12                                   # zero answer-INDICATOR projection per set (indicator only)


def test_format_confound_of_paired_split_does_not_cross_images_under_C():
    """B1-like: Paired half-split makes a wrong valid T answer positive through format; C compares it with same-answer F responses."""
    F, T = [C_(), Fi(), C_(), C_()], [Tw(), Ti(), Tw(), Ti()]
    b = by_key(pair(F, T, arm='B'))
    assert b['T0']['A'] > 0 and b['T0']['correct'] == 0 and b['T0']['valid']        # historical Paired: WRONG_VALID_POSITIVE
    c = by_key(amb.c_plan(pair(F, T)))
    assert c['T0']['A'] < 0 and c['T2']['A'] < 0 and c['T0']['baseline_kind'] == 'ANSWER_MATCHED'
    q = amb.quality(list(c.values()), amb.split_parts(list(c.values())), direct_rows=pair(F, T))
    assert q['sign']['WRONG_VALID_POSITIVE'] == 0 and q['sign']['CORRECT_NEGATIVE'] == 0


def test_hard_modality_coverage_where_direct_is_inactive():
    F, T = [C_(), C_(), C_(), Fw()], [Tw()] * 4                                       # T saturated wrong: Direct T group inactive
    a = pair(F, T)
    assert not by_key(a)['T0']['admitted']
    c = amb.c_plan(a)
    q = amb.quality(c, amb.split_parts(c), direct_rows=a)
    cov = q['hard_modality_coverage']
    assert cov['direct_inactive_images'] == 1 and cov['images_receiving_nonzero_C_credit'] == 1
    assert cov['valid_credited_responses'] == 4 and cov['valid_credited_sign_correct'] == 4
    assert by_key(c)['F3']['baseline_kind'] == 'SAME_IMAGE_FALLBACK' and by_key(c)['F3']['A'] < 0      # T1 never said on T: Direct fallback


def test_C_is_invariant_to_response_order_within_an_image_but_paired_split_is_not():
    F, T = [C_(), C_(), Fw(), Fi()], [Tw(), Tc(), Tw(), Ti()]
    F2, T2 = [C_(), Fw(), C_(), Fi()], T                                    # same F multiset, different half-split membership
    sig = lambda rows: sorted((r['image'], r['correct'], r['format'], r['answer'], round(r['A'], 12)) for r in rows)
    assert sig(amb.c_plan(pair(F, T))) == sig(amb.c_plan(pair(F2, T2)))
    assert sig(pair(F, T, arm='B')) != sig(pair(F2, T2, arm='B'))


def test_exposure_weights_and_dallas_are_unchanged():
    src = pair([C_(), Fw(), C_(), Fi()], [Tw(), Tc(), Tw(), Ti()]) + dallas()
    c = amb.c_plan(src)
    assert [(r['key'], r['tokens'], r['r'], r['image'], r['answer']) for r in c] == [(r['key'], r['tokens'], r['r'], r['image'], r['answer']) for r in src]
    for r, s in zip(c, src):
        if r['task'] == 'DALLAS':
            assert (r['A'], r['w'], r['admitted'], r['group']) == (s['A'], s['w'], s['admitted'], s['group'])
        else:
            assert r['w'] in (0., W) and r['group'] == 0
    assert amb.c_plan(src)[0]['A'] == c[0]['A']                                  # deterministic


def test_baseline_split_is_exact_for_A_B_and_C():
    F, T = [C_(), Fw(), C_(), Fi()], [Tw(), Tc(), Tw(), Ti()]
    for rows in (pair(F, T), pair(F, T, arm='B'), amb.c_plan(pair(F, T))):
        parts = amb.split_parts(rows)
        for r, p in zip(rows, parts):
            assert abs(p['accuracy'] + p['format'] + p['overlong'] + p['unexplained'] - r['A']) < 1e-9
            assert abs(p['unexplained']) < 1e-9


def test_c_plan_refuses_incomplete_pairs():
    rows = pair([C_()] * 4, [Tw()] * 4)[:7]
    with pytest.raises(ValueError, match='four F and four T'):
        amb.c_plan(rows)


# ================================================================ predeclared decision rule
def _metrics(wvp=0, cn=0, fmt=.1, pb=.9, inactive=1, credited=1, ok=4, n=4):
    return {'sign': {'WRONG_VALID_POSITIVE': wvp, 'CORRECT_NEGATIVE': cn}, 'format_share_of_abs_baseline_split_mass': fmt,
            'point_biserial_valid_admitted': pb,
            'hard_modality_coverage': {'direct_inactive_images': inactive, 'images_receiving_nonzero_C_credit': credited,
                                       'valid_credited_responses': n, 'valid_credited_sign_correct': ok}}


def test_decision_rule_rejects_on_any_failed_criterion_and_never_says_go():
    B = _metrics(fmt=.2, pb=.8)
    assert amb.decide(_metrics(), B)['decision'] == 'INCONCLUSIVE_CPU_COEFFICIENT_ONLY'
    for bad in (_metrics(wvp=1), _metrics(cn=1), _metrics(fmt=.2), _metrics(pb=.7), _metrics(credited=0), _metrics(ok=3), _metrics(pb=None)):
        assert amb.decide(bad, B)['decision'] == 'REJECT_C_BEFORE_GPU'
    assert all('GO' not in d.replace('NO_GO', '') for d in amb.DECISIONS)


# ================================================================ end to end on the existing synthetic V2 run
@pytest.fixture
def synthetic_arms(tmp_path):
    from test_accuracy_pilot_v2 import units4, lengths_for, make_common, run_arm, PARENT_RECEIPT
    from test_multistep_gspo_v1 import REWARD
    units, lengths = units4('concise'), lengths_for('concise')
    _, _, common = make_common(tmp_path, units, lengths)
    for arm in ('A', 'B'):
        run_arm(tmp_path, arm, units, lengths, common)
    return {arm: diag.rederive_arm(units, tmp_path / 'common', tmp_path / ('arm_' + arm), arm, reward_sha=REWARD, parent_receipt=PARENT_RECEIPT)
            for arm in ('A', 'B')}


def test_end_to_end_matched_common32_and_observational_segment1(synthetic_arms):
    report = amb.analyze(synthetic_arms)
    json.dumps(report, allow_nan=False)
    m = report['common32_matched_A_B_C']
    assert m['exposure']['identical'] is True and m['exposure']['responses'] == 32
    assert set(m['arms']) == {'A_direct_actual_plan', 'B_paired_actual_plan', 'C_answer_matched_COUNTERFACTUAL'}
    assert m['decision']['decision'] in amb.DECISIONS
    assert m['arms']['C_answer_matched_COUNTERFACTUAL']['compute']['new_generations'] == 0
    obs = report['segment1_observational_counterfactual']
    assert obs['label'].startswith('OBSERVATIONAL') and 'decision' not in obs
    pub = amb.public_aggregate(report)
    text = json.dumps(pub)
    assert '"key"' not in text and '_PRIVATE' not in text and '/Users/' not in text
    assert pub['public_case_keys']['status'] == 'PASS' and pub['public_schema']['status'] == 'PASS'
    assert pub['claims']['accuracy'] == 'NO_NEW_ACCURACY_RESULT' and grv.forbidden_modules_loaded() == []
