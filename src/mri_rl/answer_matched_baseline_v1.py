"""ANSWER_MATCHED_BASELINE_V1 -- candidate C: answer-matched cross-image group-relative baseline for paired MRI images.

Author: Claude Code (claude-opus-5-5), 2026-10-22. STATUS: IMPLEMENTED_CPU_PENDING_ROOT_EXECUTION. PROPOSED VERO-STYLE ADAPTATION,
not official same-prompt GSPO and not an established contribution. CPU counterfactual advantages only: no model, generation,
teacher forcing, backward, optimizer or new reward. Reward, GSPO loss/ratio/clip, LR, LoRA, optimizer, concise prompt, unit weight .25
and per-response weight 1/8 are UNCHANGED. The single declared intervention is ONE versioned MRI ADVANTAGE PLAN that changes, together,
(i) the baseline set (answer-matched or same-image fallback), (ii) the scale (pair-of-8 SD instead of B's two K4 group SDs) and (iii) the
admission unit (one pair-level gate instead of two K4 gates). It is NOT an isolated baseline-only causal test.

Definition (one SEQ_PAIR unit u = same patient/level FLAIR image F and contrast-enhanced T1 image T, K=4 responses each):
  r_i         the unchanged calibrated verifier reward; a_i the reward parser's answer if format-valid and in {FLAIR, T1}, else none.
  S(a)        format-valid responses in u whose parsed canonical class label is a.  If S(a) contains responses on BOTH images it is an
              answer-matched set: its members share the parsed class LABEL (their full text, reasoning prefix and length still differ),
              the label is held fixed while verifier correctness can differ across images; this is not full-response or visual-causal control.
  b_i         mean_{j in S(a_i)} r_j          if a_i is defined and S(a_i) spans both images        (ANSWER_MATCHED)
              mean_{j on image(i)} r_j        otherwise: invalid/other answers, or an answer seen on one image only (SAME_IMAGE_FALLBACK = Direct)
  s_u         sample SD of the 8 rewards of u (one pair scale, as one Paired group of 8)
  A^C_i       (r_i - b_i) / (s_u + 1e-6);  pair admitted iff the inherited gate passes on its 8 grades and some A^C_i != 0.
Retained cross-image information: ONLY comparisons across images between responses with the same parsed class label. Discarded relative
to Paired: cross-image comparisons between DIFFERENT parsed labels, whose reward gaps mix image difficulty, the label, and peers' format.
Exact boundaries (tested): all-same-label pair == one Paired group of 8; no shared label == Direct centering rescaled by s_u;
perfectly discriminating saturated pair -> no signal. For an answer-matched valid response sign(A^C) = correctness BY CONSTRUCTION
(a shared binary label is correct on exactly one image): sign agreement is therefore NOT independent learning evidence.
Answer-indicator projections per matched set are zero: an INDICATOR property only; the real score normalizer -E_policy[feature] persists.

C IS AN ACTION-DEPENDENT, CHANGED SURROGATE/ESTIMATOR. The baseline depends on the sampled response's own label, and the population
label-conditioned baselines E[r | label] depend on the policy (cohort counts). They are NOT the midpoint (r_c + r_w)/2 in general, so
subtracting them changes the expected score-function gradient (see action_conditioned_population_bias: uniform images, P(FLAIR|F)=.8,
P(FLAIR|T)=.6 gives b(FLAIR)=.43691, b(T1)=.47315, true reward beta-gradient -.015222, baseline term +.007249, resulting -.007973).
Global centering is also lost: matched sets are each centered, but same-image fallback baselines overlap them, so the pair's summed
advantage is generally NONZERO (reported as signed-sum metrics; deliberately NOT recentered).
Closest conceptual ancestry (bounded primary notes, results/GROUP_RELATIVE_PRIMARY_SOURCE_NOTES_V1.json): mDPO/CoPO compares the SAME full
chosen response under a preferred vs less-compatible image (stronger control than matching a parsed label across independently sampled
responses); SignBalance keeps verifier signs and rescales composition-dependent magnitudes. No novelty claim.
"""
from __future__ import annotations
import math
from pathlib import Path

from mri_rl import group_relative_visual_signal_v1 as grv
from mri_rl import shared_rollout_gspo_core_v1 as shared
from mri_rl import v2_regression_diagnosis_v1 as diag
from mri_rl.mlx_gspo_smoke import advantages_from_rewards
from mri_rl.uniform_balance_continuation_gate_v1 import accuracy_group_is_informative

SCHEMA = 'answer-matched-baseline-v1'
STATUS = 'IMPLEMENTED_CPU_PENDING_ROOT_EXECUTION'
NAME = 'ANSWER_MATCHED_CROSS_IMAGE_BASELINE'
LABEL = 'COUNTERFACTUAL_CPU_ADVANTAGES_NOT_HISTORICAL_TRAINING'
require = shared.require
ADV_EPS = grv.ADV_EPS
RESPONSE_WEIGHT = .25 / 8                   # unchanged joint weight per admitted response (UNIT_WEIGHT * 1/8)
VALID_ANSWERS = (grv.FL, grv.T1)
ZERO = grv.ZERO
DECISIONS = ('REJECT_C_BEFORE_GPU', 'INCONCLUSIVE_CPU_COEFFICIENT_ONLY')
INTERVENTION = {'kind': 'ONE_VERSIONED_MRI_ADVANTAGE_PLAN', 'isolated_baseline_only_causal_test': False,
                'changed_jointly_vs_B': ['baseline set: parsed-label-matched across images, else same-image fallback',
                                         'scale: SD of the pair\'s 8 rewards (B: SD of each K4 half-split group)',
                                         'admission: one inherited gate on the pair\'s 8 grades (B: one gate per K4 group)'],
                'estimator': 'ACTION_DEPENDENT_CHANGED_SURROGATE: baseline depends on the sampled label; expected score gradient changes',
                'global_centering': 'NOT_GUARANTEED: matched sets centered, overlapping same-image fallbacks are not; signed sums reported, not recentered',
                'label_matching': 'parsed canonical class label only; full text/reasoning prefix/length NOT held fixed; no visual causality identified',
                'unchanged': ['verifier reward', 'GSPO sequence ratio/clip/length mean', 'unit weight .25 x 1/8', 'Dallas plans', 'LR', 'LoRA',
                              'optimizer', 'concise prompt']}


def action_conditioned_population_bias(p_flair_given_F, p_flair_given_T, r_correct, r_wrong):
    """SYNTHETIC population check (uniform F/T images, logistic P(FLAIR|x)=sigmoid(beta+gamma*x)): the beta-component of the true reward
    score gradient, the extra term contributed by subtracting the label-conditioned population baselines b(a)=E[r|a], and their sum.
    Shows the action-dependent baseline is NOT unbiased when cohort counts depend on the policy."""
    pF, pT, rc, rw = p_flair_given_F, p_flair_given_T, r_correct, r_wrong
    b_flair = (pF * rc + pT * rw) / (pF + pT)                       # FLAIR is correct on F, wrong on T
    b_t1 = ((1 - pF) * rw + (1 - pT) * rc) / ((1 - pF) + (1 - pT))
    s = {'F': pF * (1 - pF), 'T': pT * (1 - pT)}                    # d P(FLAIR|x) / d beta
    true_beta = .5 * s['F'] * (rc - rw) + .5 * s['T'] * (rw - rc)
    baseline_term = -sum(.5 * s[x] * (b_flair - b_t1) for x in ('F', 'T'))
    return {'b_FLAIR': b_flair, 'b_T1': b_t1, 'midpoint': (rc + rw) / 2, 'true_reward_beta_gradient': true_beta,
            'baseline_added_term': baseline_term, 'resulting_beta_gradient': true_beta + baseline_term, 'label': 'SYNTHETIC_NOT_MRI_EVIDENCE'}


def centering_metrics(rows):
    """Signed sums of C advantages: each answer-matched set is centered, the pair as a whole generally is not (explicit limitation)."""
    mri = [r for r in rows if r['task'] == 'MRI']
    adm = [r for r in mri if r['w'] > 0]
    sets = {}
    for r in mri:
        if r.get('baseline_kind') == 'ANSWER_MATCHED':
            sets.setdefault((r['segment'], r['unit'], r['answer']), []).append(r['A'])
    total = sum(r['w'] * r['A'] for r in adm)
    return {'signed_weighted_advantage_MRI_total': total,
            'signed_weighted_advantage_by_image': {img: sum(r['w'] * r['A'] for r in adm if r['image'] == img) for img in ('F', 'T')},
            'signed_unweighted_advantage_MRI_admitted_total': sum(r['A'] for r in adm),
            'answer_matched_sets': len(sets), 'answer_matched_set_max_abs_sum': max((abs(sum(v)) for v in sets.values()), default=None),
            'global_centering_lost': abs(total) > ZERO,
            'limitation': ('Matched cohorts are individually centered; same-image fallback baselines overlap them, so the summed advantage is '
                           'generally nonzero. Deliberately NOT recentered (that would be another candidate).')}
CRITERIA = {
    'declared_before_actual_C_results': True,
    'Q1_sign': 'C: zero WRONG_VALID_POSITIVE and zero CORRECT_NEGATIVE among admitted MRI responses with nonzero advantage',
    'Q2_format': 'C format share of absolute baseline-split mass strictly below the actual Paired (B) plan on the same responses',
    'Q3_hard_modality': ('at least one MRI image whose Direct same-image group is inactive receives nonzero C credit, and every '
                         'format-valid credited response there is sign-correct'),
    'Q4_correctness_alignment': 'C point-biserial(correct, A) among admitted format-valid MRI >= the actual Paired (B) plan (None fails)',
    'rule': 'Any failure -> REJECT_C_BEFORE_GPU. All pass -> INCONCLUSIVE_CPU_COEFFICIENT_ONLY (never GO: margins/VJPs/accuracy unmeasured).',
    'not_criteria': 'More admitted groups/responses is NOT a success criterion; DEV/final100/new16 are never consulted.'}


def sample_sd(xs):
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _grade(r):
    return {'accuracy': r['acc_cal'], 'format': r['format'], 'total': r['r'], 'overlong': r['overlong']}


def _answer(r):
    return r['answer'] if r['valid'] and r['answer'] in VALID_ANSWERS else None


# ================================================================ C plan (counterfactual)
def c_plan(rows):
    """Return NEW rows (same order, same responses/tokens/rewards) with C advantages for MRI pairs; Dallas rows copied unchanged."""
    out = [dict(r) for r in rows]
    units = {}
    for i, r in enumerate(rows):
        if r['task'] == 'MRI':
            units.setdefault((r['segment'], r['unit']), []).append(i)
    for idx in units.values():
        sub = [rows[i] for i in idx]
        require(len(sub) == 8 and sorted(r['image'] for r in sub) == ['F'] * 4 + ['T'] * 4,
                'C requires complete SEQ_PAIR units: exactly four F and four T responses')
        scale = sample_sd([r['r'] for r in sub]) + ADV_EPS
        sets = {}
        for k, r in enumerate(sub):
            a = _answer(r)
            if a is not None:
                sets.setdefault(a, []).append(k)
        matched = {a: ks for a, ks in sets.items() if len({sub[k]['image'] for k in ks}) == 2}
        advs, kinds, members = [], [], []
        for k, r in enumerate(sub):
            a = _answer(r)
            if a in matched:
                ks, kind = matched[a], 'ANSWER_MATCHED'
            else:
                ks, kind = [j for j, s in enumerate(sub) if s['image'] == r['image']], 'SAME_IMAGE_FALLBACK'
            b = sum(sub[j]['r'] for j in ks) / len(ks)
            advs.append((r['r'] - b) / scale)
            kinds.append(kind)
            members.append(tuple(sub[j]['key'] for j in ks))
        admitted = accuracy_group_is_informative([_grade(r) for r in sub]) and max(abs(a) for a in advs) > ZERO
        for k, i in enumerate(idx):
            out[i].update(arm='C', group=0, A=advs[k], admitted=admitted, w=RESPONSE_WEIGHT if admitted else 0., baseline_kind=kinds[k],
                          baseline_keys_PRIVATE=members[k], scale=scale, origin_label=LABEL)
    for r in out:
        if r['task'] != 'MRI':
            r.update(arm='C', origin_label=LABEL + ' (Dallas plan copied unchanged from the source plan)')
    return out


def direct_plan(rows):
    """Direct same-image GSPO plan of the SAME MRI responses (unchanged estimator + inherited gate); used to locate inactive images."""
    out = [dict(r) for r in rows]
    groups = {}
    for i, r in enumerate(rows):
        if r['task'] == 'MRI':
            groups.setdefault((r['segment'], r['unit'], r['image']), []).append(i)
    for idx in groups.values():
        sub = [rows[i] for i in idx]
        adv = advantages_from_rewards([r['r'] for r in sub])
        ok = accuracy_group_is_informative([_grade(r) for r in sub])
        for i, a in zip(idx, adv):
            out[i].update(arm='A_direct_recomputed', group={'F': 0, 'T': 1}[rows[i]['image']], A=a, admitted=ok, w=RESPONSE_WEIGHT if ok else 0.)
    return out


# ================================================================ consistent baseline split (A, B and C alike)
def split_parts(rows):
    """A_i * d = r_i - mean_{baseline set}(r), split exactly into accuracy / format / overlong / unexplained parts.
    Baseline set: C rows -> their declared set and pair scale; other rows -> their own (segment, arm, unit, group) and its SD."""
    by_key = {(r['segment'], r['unit'], r['key']): r for r in rows}
    groups = {}
    for r in rows:
        groups.setdefault((r['segment'], r['arm'], r['unit'], r['group']), []).append(r)
    parts = []
    for r in rows:
        if 'baseline_keys_PRIVATE' in r:
            base = [by_key[(r['segment'], r['unit'], k)] for k in r['baseline_keys_PRIVATE']]
            d = r['scale']
        else:
            base = groups[(r['segment'], r['arm'], r['unit'], r['group'])]
            d = sample_sd([b['r'] for b in base]) + ADV_EPS
        m = {f: sum(b[f] for b in base) / len(base) for f in ('acc_cal', 'format', 'overlong')}
        p = {'accuracy': grv.ACC_W * (r['acc_cal'] - m['acc_cal']) / d, 'format': grv.FMT_W * (r['format'] - m['format']) / d,
             'overlong': (r['overlong'] - m['overlong']) / d}
        p['unexplained'] = r['A'] - p['accuracy'] - p['format'] - p['overlong']
        parts.append(p)
    return parts


# ================================================================ quality metrics and the predeclared decision
def quality(rows, parts, *, direct_rows):
    mri = [(r, p) for r, p in zip(rows, parts) if r['task'] == 'MRI']
    adm = [(r, p) for r, p in mri if r['w'] > 0]
    sign = grv._sign_table([r for r, _ in adm if abs(r['A']) > ZERO])
    valid = [r for r, _ in adm if r['valid']]
    total = sum(r['w'] * (abs(p['accuracy']) + abs(p['format']) + abs(p['overlong']) + abs(p['unexplained'])) for r, p in adm)
    fmt = sum(r['w'] * abs(p['format']) for r, p in adm)
    inactive = {(r['unit'], r['image']) for r in direct_rows if r['task'] == 'MRI' and not r['admitted']}
    credited = {(r['unit'], r['image']) for r, _ in adm if (r['unit'], r['image']) in inactive and abs(r['A']) > ZERO}
    cov_valid = [r for r, _ in adm if (r['unit'], r['image']) in inactive and abs(r['A']) > ZERO and r['valid']]
    coeff = [r['w'] * abs(r['A']) / r['tokens'] for r, _ in adm if abs(r['A']) > ZERO]
    by_image = {img: {'admitted_responses': sum(r['image'] == img for r, _ in adm),
                      'abs_weighted_advantage': sum(r['w'] * abs(r['A']) for r, _ in adm if r['image'] == img),
                      'signed_weighted_advantage': sum(r['w'] * r['A'] for r, _ in adm if r['image'] == img),
                      'abs_per_token_coefficient': sum(r['w'] * abs(r['A']) / r['tokens'] for r, _ in adm if r['image'] == img)} for img in ('F', 'T')}
    alloc = grv.allocation(rows)
    return {'MRI_admitted_responses': len(adm), 'sign': sign, 'point_biserial_valid_admitted': grv._corr([float(r['correct']) for r in valid], [r['A'] for r in valid]),
            'format_share_of_abs_baseline_split_mass': None if total <= ZERO else fmt / total,
            'max_abs_unexplained': max((abs(p['unexplained']) for _, p in mri), default=0.),
            'by_image': by_image, 'per_token_coefficient_max_over_min': None if not coeff else max(coeff) / min(coeff),
            'hard_modality_coverage': {'direct_inactive_images': len(inactive), 'images_receiving_nonzero_C_credit': len(credited),
                                       'valid_credited_responses': len(cov_valid),
                                       'valid_credited_sign_correct': sum((r['correct'] == 1) == (r['A'] > 0) for r in cov_valid)},
            'label_indicator_projection': {'sequence': grv._push([r for r, _ in adm], False), 'label': grv.INDICATOR_LABEL},
            'task_shares': alloc['task_shares'], 'baseline_kind_counts_admitted': _counts(r.get('baseline_kind', 'GROUP_MEAN') for r, _ in adm),
            'centering': centering_metrics(rows),
            'sign_note': 'For C answer-matched responses sign = correctness BY CONSTRUCTION; not independent learning evidence',
            'definitions': {'format_share': 'sum w|format part| / sum w(|acc|+|format|+|overlong|+|unexplained|) relative to each response\'s OWN '
                                            'baseline set; a coefficient-mass descriptor, NOT a variance or gradient share'}}


def _counts(values):
    out = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out


def decide(c, b):
    cov = c['hard_modality_coverage']
    q = {'Q1_sign': c['sign']['WRONG_VALID_POSITIVE'] == 0 and c['sign']['CORRECT_NEGATIVE'] == 0,
         'Q2_format': c['format_share_of_abs_baseline_split_mass'] is not None and b['format_share_of_abs_baseline_split_mass'] is not None
                      and c['format_share_of_abs_baseline_split_mass'] < b['format_share_of_abs_baseline_split_mass'],
         'Q3_hard_modality': cov['images_receiving_nonzero_C_credit'] >= 1 and cov['valid_credited_sign_correct'] == cov['valid_credited_responses'],
         'Q4_correctness_alignment': c['point_biserial_valid_admitted'] is not None and b['point_biserial_valid_admitted'] is not None
                                     and c['point_biserial_valid_admitted'] >= b['point_biserial_valid_admitted']}
    decision = DECISIONS[1] if all(q.values()) else DECISIONS[0]
    return {'criteria': CRITERIA, 'results': q, 'decision': decision,
            'meaning': ('REJECT: stop C, no GPU scope.' if decision == DECISIONS[0] else
                        'C passes the predeclared scalar coefficient descriptors on the exact Common32 responses only (Q1 largely by '
                        'construction; C is an action-dependent changed estimator without global centering). Gradient direction, canonical class margins '
                        'and accuracy remain NOT_MEASURED; the most that may follow is ONE separately scoped read-only VJP check after '
                        'independent review and fresh human permission. Not training, not GO replication.')}


def _compute(rows):
    adm = [r for r in rows if r['w'] > 0]
    return {'new_generations': 0, 'new_rewards': 0, 'per_response_backward_calls_if_trained': len(adm), 'MRI_admitted_responses': sum(r['task'] == 'MRI' for r in adm),
            'admitted_response_tokens': sum(r['tokens'] for r in adm), 'optimizer_updates_if_trained': 1 if adm else 0,
            'note': 'Counts a native GSPO step WOULD need (one backward per admitted response); nothing here is executed.'}


# ================================================================ analysis
def analyze(arms):
    tables = {(arm, seg): grv.response_table(arms[arm][seg]['units'], segment=seg, arm=arm, origin=arms[arm][seg]['rollout_origin'])
              for arm in ('A', 'B') for seg in (0, 1)}
    exposure = grv.matched_exposure(tables[('A', 0)], tables[('B', 0)])
    a0, b0 = tables[('A', 0)], tables[('B', 0)]
    d0 = direct_plan(a0)
    drift = max((abs(x['A'] - y['A']) for x, y in zip(d0, a0) if x['task'] == 'MRI'), default=0.)
    require(drift < 1e-12 and all(x['admitted'] == y['admitted'] for x, y in zip(d0, a0) if x['task'] == 'MRI'),
            'Recomputed Direct plan differs from the actual A segment-0 plan')
    c0 = c_plan(a0)
    plans = {'A_direct_actual_plan': a0, 'B_paired_actual_plan': b0, 'C_answer_matched_COUNTERFACTUAL': c0}
    q = {k: quality(v, split_parts(v), direct_rows=a0) for k, v in plans.items()}
    for k, v in plans.items():
        q[k]['compute'] = _compute(v)
    obs = {}
    for arm in ('A', 'B'):
        t = tables[(arm, 1)]
        d1, c1 = direct_plan(t), c_plan(t)
        obs[arm] = {'actual_plan': quality(t, split_parts(t), direct_rows=d1), 'C_COUNTERFACTUAL': quality(c1, split_parts(c1), direct_rows=d1)}
    return {'schema': SCHEMA, 'status': 'EXECUTED_CPU_READ_ONLY', 'candidate': NAME, 'label': LABEL, 'intervention': INTERVENTION,
            'boundaries': dict(grv.BOUNDARIES), 'claims': dict(grv.CLAIMS),
            'common32_matched_A_B_C': {'label': 'EXACT_MATCHED: Parent Adam52, identical 32 responses/tokens/images/rewards/exposure; only the advantage plan differs',
                                       'exposure': exposure, 'arms': q, 'decision': decide(q['C_answer_matched_COUNTERFACTUAL'], q['B_paired_actual_plan'])},
            'segment1_observational_counterfactual': {'label': ('OBSERVATIONAL: each arm\'s own post-update rollouts re-planned under C; A and B segment-1 '
                                                                'responses come from DIFFERENT policies, are never mixed, and are not common-parent evidence'),
                                                      'arms': obs},
            'rows_PRIVATE': {'C0': c0}}


def public_aggregate(report):
    pub = grv._strip({k: v for k, v in report.items() if k != 'rows_PRIVATE'})
    pub['schema'] = SCHEMA + '-aggregate'
    pub['public_case_keys'] = grv.public_case_key_check(pub)
    pub['public_schema'] = diag.public_schema_check(pub)
    return pub


def run(*, root, worktree, run_dir, out_dir, results_path, expected_scope_sha256=None):
    grv.install_import_guard()
    ctx = diag.authenticate(root=root, worktree=worktree, run_dir=run_dir, results_path=results_path, expected_scope_sha256=expected_scope_sha256)
    run_dir, root, out_dir = ctx['run_dir'], Path(root).resolve(), Path(out_dir).resolve()
    require(out_dir.is_relative_to(root / 'local_data') and not out_dir.is_relative_to(run_dir), 'Output must be under local_data, outside the immutable run')
    targets = [out_dir / n for n in ('answer_matched_baseline_PRIVATE.json', 'answer_matched_baseline_AGGREGATE_PUBLIC_CANDIDATE.json')]
    require(not any(p.exists() for p in targets), 'Refusing to overwrite an existing C output')
    for arm in ('A', 'B'):
        require(ctx['receipts'][(arm, 0)]['adam_step_before'] == 52, 'Segment 0 must start at Parent Adam52')
    arms = {arm: diag.rederive_arm(ctx['units'], ctx['common_dir'], run_dir / f'stage{arm}_PRIVATE' / 'arm_PRIVATE', arm,
                                   reward_sha=ctx['reward_sha'], parent_receipt=ctx['parent_receipt']) for arm in ('A', 'B')}
    for arm in ('A', 'B'):
        for seg in (0, 1):
            require(arms[arm][seg]['saved_evidence_all_match'], f'{arm}{seg}: re-derived plan differs from saved evidence')
    report = analyze(arms)
    report['authentication'] = {k: ctx['record'][k] for k in ('scope_file_sha256', 'parent_adapter_sha256', 'reward_calibration_sha256', 'prompt_condition',
                                                              'segment_receipts')}
    public = public_aggregate(report)
    public['privacy'] = report['privacy'] = diag.privacy_check(public, diag._forbidden_values(ctx))
    require(not grv.forbidden_modules_loaded(), 'Forbidden module imported during analysis')
    reauth = diag.authenticate(root=root, worktree=worktree, run_dir=run_dir, results_path=results_path, expected_scope_sha256=expected_scope_sha256)
    report['inputs_unchanged_after_analysis'] = public['inputs_unchanged_after_analysis'] = reauth['record'] == ctx['record']
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {p.name: diag.write_exclusive(p, v) for p, v in zip(targets, (report, public))}
    return {'written_sha256': written, 'public': public}
