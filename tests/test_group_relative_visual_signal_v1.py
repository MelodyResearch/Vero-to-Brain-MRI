"""Synthetic CPU tests for the group-relative visual learning-signal mechanism analysis (no MLX/model/MRI data/real run).

Table-level fixtures are explicitly SYNTHETIC; the end-to-end test reuses the existing synthetic V2 run fixture
(test_accuracy_pilot_v2) through the unchanged v2_regression_diagnosis_v1.rederive_arm planners.
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_k, '1')
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mri_rl import group_relative_visual_signal_v1 as grv     # noqa: E402
from mri_rl import v2_regression_diagnosis_v1 as diag          # noqa: E402
from mri_rl.mlx_gspo_smoke import advantages_from_rewards       # noqa: E402

grv.install_import_guard()
CAL = {1: .5, 0: .0243088}                                      # calibrated-accuracy shape of the real MRI receipts
FL, T1, INV = 'FLAIR', 'contrast-enhanced T1', 'INVALID_FORMAT'


# ================================================================ SYNTHETIC table builders (labelled; never MRI evidence)
def resp(image, correct, fmt, answer, tokens=40, overlong=0.):
    acc = CAL[correct]
    return {'image': image, 'correct': correct, 'format': fmt, 'acc_cal': acc, 'overlong': overlong, 'tokens': tokens, 'answer': answer,
            'r': .8 * acc + .2 * fmt + overlong}


def group(members, *, unit=0, g=0, admitted=True, task='MRI', segment=0, arm='A', start=0):
    adv = advantages_from_rewards([m['r'] for m in members])
    w = .25 / 8 if admitted else 0.
    return [{'segment': segment, 'arm': arm, 'origin': 'COMMON_PARENT', 'unit': unit, 'task': task, 'key': f'u{unit}g{g}r{i + start}', 'group': g,
             'admitted': admitted, 'w': w, 'A': a, 'valid': m['format'] == 1, 'r_raw': .8 * m['correct'] + .2 * m['format'] + m['overlong'], **m}
            for i, (m, a) in enumerate(zip(members, adv))]


# ================================================================ exact decompositions
def test_advantage_decomposition_is_exact_and_splits_within_image_from_between_image():
    rows = group([resp('F', 1, 1., FL), resp('F', 0, 1., T1), resp('T', 0, 1., FL), resp('T', 0, 0., INV)])
    comps = grv.advantage_components(rows)
    for c, r in zip(comps, rows):
        total = c['within_accuracy'] + c['within_format'] + c['within_overlong'] + c['between_image']
        assert abs(total - r['A']) < 1e-9 and abs(c['residual']) < 1e-9
    assert comps[0]['within_accuracy'] > 0 and comps[1]['within_accuracy'] < 0            # F: genuine same-image correct/wrong
    assert comps[2]['within_format'] > 0 and comps[3]['within_format'] < 0                 # T: only a format contrast inside the image
    assert comps[0]['between_image'] == pytest.approx(comps[1]['between_image'])          # one offset per image
    assert comps[0]['between_image'] > 0                                                   # F image mean above group mean


def test_between_image_only_group_is_admitted_and_is_answer_constant_image_conditioned():
    """All-F-correct / all-T-wrong answering FLAIR everywhere: Paired admits it; its signed push is purely image x answer."""
    rows = group([resp('F', 1, 1., FL), resp('F', 1, 1., FL), resp('T', 0, 1., FL), resp('T', 0, 1., FL)])
    g = grv.group_records(rows)[0]
    assert g['contrast'] == 'BETWEEN_IMAGE_ONLY' and g['format_valid_within'] is False and g['gate_admitted'] is True
    comps = grv.advantage_components(rows)
    assert all(abs(c['within_accuracy']) < 1e-12 and abs(c['within_format']) < 1e-12 for c in comps)
    proj = grv.allocation(rows)['label_indicator_projection']
    assert abs(proj['sequence']['answer_indicator_projection_FLAIR_minus_T1']) < 1e-12      # zero ANSWER-INDICATOR projection only
    assert proj['sequence']['image_x_answer_indicator_projection'] > 0                       # +FLAIR|F, -FLAIR|T (indicator level)
    assert proj['label'].startswith('RESPONSE_LABEL_INDICATOR_COEFFICIENT_PROJECTION') and 'NOT parameter-gradient' in proj['label']
    assert not any('prior' in k or k == 'image_conditioned' for k in proj['sequence'])     # no parameter-part naming


def test_zero_answer_indicator_projection_does_not_mean_zero_score_gradient():
    """Reproduces results/GROUP_RELATIVE_SYNTHETIC_SCORE_NORMALIZER_COUNTEREXAMPLE_V1.json (SYNTHETIC, not MRI evidence).

    P(FLAIR|x) = sigmoid(beta + gamma*x), x_F=+1, x_T=-1, p_F=.8, p_T=.6; 2 F-correct + 2 T-wrong, all answer FLAIR, equal L,
    sample-SD advantages +/- sqrt(3)/2 (epsilon 0). The label-indicator projection is 0, yet the actual score-function gradient
    (normalizer -E_policy[feature] included) moves BOTH the bias and the image feature, and raises both correct log-odds."""
    rows = group([resp('F', 1, 1., FL), resp('F', 1, 1., FL), resp('T', 0, 1., FL), resp('T', 0, 1., FL)])
    proj = grv.allocation(rows)['label_indicator_projection']['sequence']
    assert abs(proj['answer_indicator_projection_FLAIR_minus_T1']) < 1e-12
    p, x = {'F': .8, 'T': .6}, {'F': 1., 'T': -1.}
    adv = {'F': math.sqrt(3) / 2, 'T': -math.sqrt(3) / 2}                       # epsilon 0, as in the published counterexample
    beta = gamma = 0.
    for r in rows:                                                              # d log P(FLAIR|x) = (1-p)[1, x]
        beta += adv[r['image']] * (1 - p[r['image']])
        gamma += adv[r['image']] * (1 - p[r['image']]) * x[r['image']]
    assert beta == pytest.approx(-0.34641016151377557, abs=1e-12) and gamma == pytest.approx(1.0392304845413263, abs=1e-12)
    assert beta + gamma == pytest.approx(0.6928203230275507, abs=1e-12)                 # FLAIR log-odds on F rises
    assert -(beta - gamma) == pytest.approx(1.3856406460551018, abs=1e-12)              # T1 log-odds on T rises
    assert 'normalizer' in grv.INDICATOR_LABEL


def test_public_export_drops_group_records_and_rejects_case_keys_recursively():
    rows = (group([resp('F', 1, 1., FL), resp('F', 0, 1., T1), resp('F', 1, 1., FL), resp('F', 1, 1., FL)], g=0)
            + group([resp('T', 0, 1., FL), resp('T', 1, 1., T1), resp('T', 0, 1., FL), resp('T', 0, 1., FL)], g=1, start=4))
    for r in rows:
        r['regrade_match'] = None
    view = grv.segment_view(rows)
    assert view['groups']['group_stats']                                                # retained PRIVATELY
    pub = grv.public_aggregate({'schema': 'x', 'view': view, 'rows_PRIVATE': {'A0': rows}})
    text = json.dumps(pub)
    assert 'group_stats' not in text and '"unit"' not in text and '"group"' not in text and 'correct_by_image' not in text
    assert pub['public_case_keys']['status'] == 'PASS'
    for bad in ({'a': {'unit': 3}}, {'a': {'b': {'group': 1}}}, {'a': [{'n': 1}]}, {'a': {'index': 0}}, {'a': {'id': 'x'}}, {'a': {'context': 'F'}}):
        with pytest.raises(ValueError, match='case/group'):
            grv.public_case_key_check(bad)


def test_same_image_contrast_in_direct_group_has_nonzero_answer_indicator_projection():
    """Direct F group with 3 FLAIR-correct + 1 T1-wrong: within-image contrast whose ANSWER-indicator projection is nonzero
    (an indicator statement only; not a claim about the bias-parameter gradient)."""
    rows = group([resp('F', 1, 1., FL)] * 3 + [resp('F', 0, 1., T1)])
    comps = grv.advantage_components(rows)
    assert all(abs(c['between_image']) < 1e-12 for c in comps)
    proj = grv.allocation(rows)['label_indicator_projection']['sequence']
    assert proj['answer_indicator_projection_FLAIR_minus_T1'] > 0 and proj['image_x_answer_indicator_projection'] > 0


def test_anova_and_reward_component_sums_are_exact():
    rows = (group([resp('F', 1, 1., FL), resp('F', 0, 1., T1), resp('F', 1, 1., FL), resp('F', 0, 0., INV)], g=0)
            + group([resp('T', 0, 1., FL), resp('T', 1, 1., T1), resp('T', 0, 0., INV), resp('T', 0, 1., FL)], g=1, start=4))
    for value in ('r', 'r_raw', 'correct', 'format'):
        a = grv.anova(rows, value)
        assert a['total_ss'] == pytest.approx(a['within_image_ss'] + a['between_image_ss'], abs=1e-12)
    comp = grv.reward_component_ss(rows)
    for part in ('within_image', 'between_image', 'total'):
        assert sum(comp[part]['terms'].values()) == pytest.approx(comp[part]['reward_ss'], abs=1e-12)
    valid = grv.anova([r for r in rows if r['valid']], 'correct')
    assert valid['n'] == 6 and valid['within_image_ss'] > 0


def test_image_stats_flag_same_image_format_valid_contrast_separately_from_format_contrast():
    rows = (group([resp('F', 1, 1., FL), resp('F', 1, 1., FL), resp('F', 0, 0., INV), resp('F', 1, 1., FL)], g=0)
            + group([resp('T', 0, 1., FL), resp('T', 1, 1., T1), resp('T', 0, 1., FL), resp('T', 1, 1., T1)], g=1, start=4))
    s = grv.image_stats(rows)
    by = {x['image']: x for x in s['images']}
    assert by['F']['same_image_format_valid_contrast'] is False and by['F']['same_image_any_contrast'] is True
    assert by['T']['same_image_format_valid_contrast'] is True and by['T']['valid_correct'] == 2 == by['T']['valid_wrong']
    assert by['F']['reward_sample_sd'] > 0 and s['counts']['images_with_format_valid_contrast'] == 1


def test_length_changes_per_token_coefficients_but_not_the_zero_sum_sequence_weights():
    rows = group([resp('F', 1, 1., FL, tokens=20), resp('F', 0, 1., T1, tokens=200), resp('F', 1, 1., FL, tokens=20), resp('F', 0, 1., T1, tokens=200)])
    a = grv.allocation(rows)
    assert abs(a['total']['signed_weighted_advantage']) < 1e-12                                       # zero-sum at sequence level
    assert a['total']['signed_per_token_coefficient'] > 0                                            # length factor breaks zero sum
    assert a['by_correctness']['MRI|VALID_CORRECT']['abs_per_token_coefficient'] == pytest.approx(
        10 * a['by_correctness']['MRI|VALID_WRONG']['abs_per_token_coefficient'])
    assert 'NOT_A_GRADIENT' in a['label']


def test_advantage_sign_agreement_with_correctness_and_format_driven_wrong_positive():
    rows = group([resp('F', 1, 1., FL), resp('F', 0, 0., INV), resp('T', 0, 1., FL), resp('T', 0, 0., INV)])
    a = grv.allocation(rows)
    assert a['sign_vs_correctness']['WRONG_VALID_POSITIVE'] == 1                    # T image answered FLAIR, up-weighted via format
    assert a['sign_vs_correctness']['CORRECT_POSITIVE'] == 1
    assert 0 < a['sign_vs_correctness']['agreement_fraction'] < 1


def test_inactive_groups_contribute_no_weight_and_are_counted_separately():
    rows = group([resp('F', 1, 1., FL)] * 4, admitted=False)
    g = grv.group_records(rows)[0]
    assert g['contrast'] == 'NO_REWARD_VARIANCE' and g['admitted'] is False
    assert grv.allocation(rows)['total']['admitted_responses'] == 0


# ================================================================ matched / observational separation
def test_segment0_matched_requires_identical_exposure_between_groupings():
    a = group([resp('F', 1, 1., FL), resp('F', 0, 1., T1), resp('F', 1, 1., FL), resp('F', 1, 1., FL)], arm='A')
    b = [dict(r, arm='B') for r in a]
    assert grv.matched_exposure(a, b)['identical'] is True
    b[1] = dict(b[1], tokens=b[1]['tokens'] + 1)
    with pytest.raises(ValueError, match='exposure'):
        grv.matched_exposure(a, b)
    seg1 = [dict(r, segment=1, origin='OWN_POLICY') for r in a]
    with pytest.raises(ValueError, match='segment 0'):
        grv.matched_exposure(seg1, [dict(r, arm='B') for r in seg1])


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


def test_end_to_end_report_on_synthetic_run_is_consistent_and_sanitized(synthetic_arms):
    report = grv.analyze(synthetic_arms)
    json.dumps(report, allow_nan=False)
    m = report['segment0_common32_matched']
    assert m['exposure']['identical'] is True and m['exposure']['responses'] == 32
    for arm in ('A', 'B'):
        seg = m['arms'][arm]
        planned = sum(u['effective_groups'] for u in synthetic_arms[arm][0]['units'])
        assert seg['groups']['admitted_total'] == planned
        assert seg['groups']['decomposition_max_abs_residual'] < 1e-9
        assert sum(seg['groups']['admitted_by_task_and_contrast'].values()) == planned
    assert m['shared_block_anova']['r']['total_ss'] == pytest.approx(
        m['shared_block_anova']['r']['within_image_ss'] + m['shared_block_anova']['r']['between_image_ss'])
    obs = report['segment1_own_policy_observational']
    assert obs['label'].startswith('OBSERVATIONAL') and set(obs['arms']) == {'A', 'B'} and 'matched' not in obs
    public = grv.public_aggregate(report)
    text = json.dumps(public)
    assert '_PRIVATE' not in text and '/Users/' not in text and '"key"' not in text and 'group_stats' not in text
    assert public['public_schema']['status'] == 'PASS' and public['public_case_keys']['status'] == 'PASS'
    assert public['boundaries']['model_loads'] == 0 and public['claims']['accuracy'] == 'NO_NEW_ACCURACY_RESULT'


def test_module_imports_no_heavy_runtime():
    code = ('import sys\nimport mri_rl.group_relative_visual_signal_v1 as m\nm.install_import_guard()\n'
            'for name in ("mlx.core", "torch", "jax", "transformers"):\n'
            ' try:\n  __import__(name)\n except ImportError as error:\n  assert "blocked" in str(error), error\n'
            ' else:\n  raise SystemExit("not blocked: " + name)\nprint("OK")\n')
    out = subprocess.run([sys.executable, '-c', code], env={**os.environ, 'PYTHONPATH': str(SRC)}, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0 and out.stdout.strip() == 'OK', out.stderr
