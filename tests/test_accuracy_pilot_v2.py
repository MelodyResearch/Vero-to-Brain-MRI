"""Synthetic CPU tests for Accuracy Pilot V2 (core, native glue statics, entry authority). No MLX, model, MRI data or native evidence."""
import ast
import hashlib
import inspect
import json
import subprocess
import sys
import textwrap
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import pytest
from test_multistep_gspo_v1 import make_plan, LENGTHS, REWARD, sha
from test_safe_micro_training_v1 import MicroFake, MIXED, DALLAS_INACTIVE
from mri_rl import accuracy_pilot_v2_core as core
from mri_rl import multistep_gspo_core_v1 as ms
from mri_rl import shared_rollout_gspo_core_v1 as shared

SRC = str(Path(__file__).resolve().parents[1] / 'src')
PARENT_RECEIPT = sha('parent receipt')
CONCISE_LEN = 233


def units4(condition='original'):
    return core.condition_units(ms.make_units(make_plan(4)), condition)


def lengths_for(condition):
    return core.prompt_lengths_for(condition, CONCISE_LEN if condition == 'concise' else None)


def read(path):
    return json.loads(Path(path).read_text())


def make_common(tmp_path, units, lengths, backend=None):
    backend = backend or MicroFake()
    counters = shared.Counters(core.common_budget(units[:4], lengths), None, None)
    record = core.make_common(backend=backend, units=units[:4], lengths=lengths, reward_policy_sha256=REWARD, output=tmp_path / 'common',
                              before=lambda n: None, parent_receipt_sha256=PARENT_RECEIPT, counters=counters)
    common = core.load_common(tmp_path / 'common', units[:4], reward_policy_sha256=REWARD, parent_receipt_sha256=PARENT_RECEIPT)
    return backend, record, common


def run_arm(tmp_path, arm, units, lengths, common, backend=None, name=None):
    backend = backend or MicroFake()
    counters = shared.Counters(core.arm_budget(units, lengths)['limits'], None, None)
    summary = core.run_arm(arm=arm, backend=backend, units=units, lengths=lengths, reward_policy_sha256=REWARD, common=common, counters=counters,
                           output=tmp_path / (name or 'arm_' + arm), before=lambda n: None)
    return backend, summary


# ================================================================ Experiment 1: one predeclared prompt
def test_concise_instruction_is_the_verbatim_previously_tested_sentence():
    from mri_rl.generation_recovery_v1 import SHORT_INSTRUCTION
    assert core.CONCISE_INSTRUCTION == SHORT_INSTRUCTION == core.CONCISE_INSTRUCTION_TEXT
    assert core.CONCISE_INSTRUCTION_SHA256 == hashlib.sha256(SHORT_INSTRUCTION.encode()).hexdigest()


def test_condition_row_changes_only_the_sequence_question():
    units = ms.make_units(make_plan(1))
    seq, dallas = units[0]['contexts'][0]['row'], units[1]['contexts'][0]['row']
    concise = core.condition_row(seq, 'concise')
    assert concise['extra_info']['question'] == seq['extra_info']['question'] + '\n\n' + core.CONCISE_INSTRUCTION
    assert {k: v for k, v in concise.items() if k != 'extra_info'} == {k: v for k, v in seq.items() if k != 'extra_info'}
    assert {k: v for k, v in concise['extra_info'].items() if k != 'question'} == {k: v for k, v in seq['extra_info'].items() if k != 'question'}
    assert core.condition_row(dallas, 'concise') == dallas and core.condition_row(seq, 'original') == seq
    assert core.condition_row(seq, 'original') is not seq
    with pytest.raises(ValueError):
        core.condition_row(concise, 'concise')
    with pytest.raises(ValueError):
        core.condition_row(seq, 'short_v2')


def test_prompt_lengths_keep_the_original_ledger_and_require_a_measured_concise_length():
    assert core.prompt_lengths_for('original', None) == {'mri_sequence': 213, 'mri_visit_correspondence': 405}
    assert core.prompt_lengths_for('concise', 233) == {'mri_sequence': 233, 'mri_visit_correspondence': 405}
    for bad in (None, 213, 200, 233.0, 999):
        with pytest.raises(ValueError):
            core.prompt_lengths_for('concise', bad)


def answer(label):
    return r'<think>x</think><answer>\boxed{%s}</answer>' % label


def items(f_preds, t_preds, tokens=40):
    out = []
    for truth, preds in (('FLAIR', f_preds), ('contrast-enhanced T1', t_preds)):
        for p in preds:
            text = 'Visible evidence? No. ' * 30 if p == 'INV' else answer({'F': 'FLAIR', 'T': 'contrast-enhanced T1'}[p])
            out.append({'truth': truth, 'text': text, 'tokens': 256 if p == 'INV' else tokens, 'wall_seconds': 1.})
    return out


def test_greedy_summary_counts_invalid_caps_predictions_and_tokens():
    s = core.greedy_summary(items(['F', 'F', 'INV', 'INV'], ['T', 'F', 'INV', 'T']))
    assert s['cases'] == 8 and s['correct'] == 4 and s['invalid'] == 3 and s['class_correct'] == [2, 2] and s['cap_hits'] == 3
    assert s['predicted_counts'] == {'FLAIR': 3, 'contrast-enhanced T1': 2, 'INVALID': 3} and s['returned_tokens'] == 5 * 40 + 3 * 256


@pytest.mark.parametrize('original,concise,expected', [
    ((['F', 'F', 'INV', 'INV'], ['T', 'F', 'INV', 'T']), (['F', 'F', 'F', 'T'], ['T', 'F', 'T', 'T']), 'ADOPT_CONCISE'),   # fewer invalid, both classes gain
    ((['F', 'F', 'INV', 'T'], ['T', 'F', 'INV', 'T']), (['F', 'F', 'INV', 'T'], ['T', 'F', 'INV', 'T']), 'KEEP_ORIGINAL'),  # identical: no benefit
    ((['F', 'F', 'F', 'INV'], ['T', 'F', 'F', 'F']), (['F', 'F', 'T', 'T'], ['T', 'T', 'T', 'T']), 'KEEP_ORIGINAL'),        # FLAIR loses: prior shift guard
    ((['F', 'F', 'T', 'T'], ['T', 'F', 'F', 'F']), (['F', 'F', 'F', 'INV'], ['T', 'T', 'F', 'INV']), 'KEEP_ORIGINAL'),      # invalid rises
])
def test_prompt_decision_rule_is_frozen_and_conservative(original, concise, expected):
    d = core.prompt_decision(core.greedy_summary(items(*original)), core.greedy_summary(items(*concise)))
    assert d['decision'] == expected and d['condition'] == ('concise' if expected == 'ADOPT_CONCISE' else 'original')
    assert d['label'] == 'GENERATION_PROMPT_EFFECT_NOT_VERO_RL' and d['token_cap_changed'] is False and d['verifier_changed'] is False


def test_prompt_decision_requires_the_same_balanced_eight_images():
    with pytest.raises(ValueError):
        core.prompt_decision(core.greedy_summary(items(['F'] * 4, ['T'] * 4)), core.greedy_summary(items(['F'] * 4, ['T'] * 3)))


# ================================================================ Experiment 2: schedule / budget
def test_frozen_four_round_schedule_validates_four_distinct_dev_disjoint_pairs():
    units = units4()
    report = core.validate_schedule(units, {'sub-99'})
    assert report['patients'] == 4 and report['sequence_images'] == 8 and report['dallas_rows'] == 8 and report['patients_per_update'] == 2
    first = units[0]['contexts'][0]['row']['extra_info']['patient_key']
    with pytest.raises(ValueError, match='disjoint'):
        core.validate_schedule(units, {first})
    with pytest.raises(ValueError):
        core.validate_schedule(units[:6], set())
    dup = json.loads(json.dumps(units))
    for c in dup[2]['contexts']:
        c['row']['extra_info']['patient_key'] = first
    with pytest.raises(ValueError, match='Four distinct'):
        core.validate_schedule(dup, set())


def test_arm_budget_two_updates_four_patients_and_reports_granularity():
    units = units4()
    b = core.arm_budget(units, LENGTHS)
    s0, s1, total = b['segment0'], b['segment1'], b['limits']
    assert s0['generated_response_tokens'] == 0 and s0['detached_old_forwards'] == 0 and s0['backward_calls'] == 32 and s0['optimizer_updates'] == 1
    assert s1['generated_response_tokens'] == 4 * 8 * 256 and s1['detached_old_forwards'] == 32 and s1['backward_calls'] == 32 and s1['optimizer_updates'] == 1
    assert total['backward_calls'] == 64 and total['optimizer_updates'] == 2 and b['optimizer_updates_max'] == 2
    assert b['weight_per_admitted_response_in_joint'] == 1 / 32 and b['v1_weight_per_admitted_response_in_joint'] == 1 / 16
    c = core.common_budget(units[:4], LENGTHS)
    assert c['generated_response_tokens'] == 8192 and c['detached_old_forwards'] == 32 and c['backward_calls'] == 0 and c['optimizer_updates'] == 0
    concise = core.arm_budget(units4('concise'), lengths_for('concise'))['limits']
    assert concise['teacherforced_input_positions'] > total['teacherforced_input_positions']


# ================================================================ Experiment 2: common once, own policy later, matched arms
def test_common_rollout_is_generated_once_and_both_arms_start_from_its_parent(tmp_path):
    units, lengths = units4(), LENGTHS
    gen, record, common = make_common(tmp_path, units, lengths)
    assert gen.generate_calls == 4 and record['responses'] == 32 and record['shared_by_arms'] == ['A', 'B']
    with pytest.raises(ValueError, match='once'):
        make_common(tmp_path, units, lengths)
    a_backend, a = run_arm(tmp_path, 'A', units, lengths, common)
    b_backend, b = run_arm(tmp_path, 'B', units, lengths, common)
    for summary, backend in ((a, a_backend), (b, b_backend)):
        assert summary['patients_exposed'] == 4 and summary['segments'] == 2 and summary['responses'] == 64
        assert summary['common_rollout_segments'] == 1 and summary['own_policy_segments'] == 1
        assert backend.generate_calls == 4                       # each arm generates only its own segment-1 units
    ra = [read(tmp_path / 'arm_A' / f'segment_{i}/receipt_PRIVATE.json') for i in (0, 1)]
    rb = [read(tmp_path / 'arm_B' / f'segment_{i}/receipt_PRIVATE.json') for i in (0, 1)]
    assert ra[0]['bundle_sha256'] == rb[0]['bundle_sha256'] == record['bundle_sha256']
    assert ra[0]['policy_sha_before'] == rb[0]['policy_sha_before'] == record['parent_state']['policy']
    for r in (ra, rb):
        assert r[1]['prev_receipt_sha256'] == r[0]['receipt_sha256'] and r[1]['policy_sha_before'] == r[0]['policy_sha_after']
        assert r[0]['rng_digest_after'] == record['rng_after_common_digest'] and r[1]['rollout_origin'] == 'OWN_POLICY'
        assert r[0]['patients'] != r[1]['patients'] and len(set(r[0]['patients'] + r[1]['patients'])) == 4
    assert ra[0]['grouping'] == 'same_prompt' and rb[0]['grouping'] == 'paired_prompt'


def test_arm_must_start_from_the_full_common_parent_state(tmp_path):
    units = units4()
    _, _, common = make_common(tmp_path, units, LENGTHS)
    moved = MicroFake(); moved.rng.random()                 # same weights/Adam, different R0
    with pytest.raises(ValueError, match='common parent'):
        run_arm(tmp_path, 'A', units, LENGTHS, common, backend=moved)


def test_cross_arm_or_stale_rollout_is_rejected_in_the_own_policy_segment(tmp_path):
    units = units4()
    _, _, common = make_common(tmp_path, units, LENGTHS)
    backend = MicroFake()
    counters = shared.Counters(core.arm_budget(units, LENGTHS)['limits'], None, None)
    out = tmp_path / 'arm_B'; out.mkdir()
    r0 = core.run_segment(arm='B', index=0, backend=backend, units=units[:4], lengths=LENGTHS, reward_policy_sha256=REWARD, counters=counters,
                          output=out, before=lambda n: None, common=common)
    class Foreign(MicroFake):
        def generate(self, unit, lengths):
            raw = super().generate(unit, lengths); raw['policy_sha256'] = sha('the other arm policy'); return raw
    foreign = Foreign(); foreign.load_checkpoint(out / 'segment_0/checkpoint_PRIVATE')
    with pytest.raises(ValueError, match='off-policy|cross-arm'):
        core.run_segment(arm='B', index=1, backend=foreign, units=units[4:], lengths=LENGTHS, reward_policy_sha256=REWARD, counters=counters,
                         output=out, before=lambda n: None, previous_receipt=r0['receipt_sha256'])
    assert not (out / 'segment_1').exists()


def expected_joint(evidence, p=.5):
    total = np.zeros(4)
    for unit in evidence:
        for item in unit['items']:
            if not item['admitted']:
                continue
            rid = item['response_id']; key = rid[0] if rid[0] in 'FT' else rid.split('_')[0]
            assert item['weight'] == 1 / 8
            y = item['reward'] == 1.0
            total += core.UNIT_WEIGHT * item['weight'] * (-item['advantage']) * ((1 - p) if y else -p) * MicroFake().phi(key, None)
    return total


def test_first_joint_update_is_the_quarter_weighted_sum_of_unit_gradients(tmp_path):
    units = units4()
    _, _, common = make_common(tmp_path, units, LENGTHS)
    for arm in 'AB':
        run_arm(tmp_path, arm, units, LENGTHS, common)
        seg = tmp_path / f'arm_{arm}/segment_0'
        evidence = read(seg / 'gradient_evidence_PRIVATE.json')
        assert [e['unit'] for e in evidence] == [0, 1, 2, 3]
        receipt = read(seg / 'receipt_PRIVATE.json')
        want = expected_joint(evidence)
        if receipt['updated']:
            got = np.load(seg / 'joint_gradient_float32_PRIVATE.safetensors')
            assert got.dtype == np.float32 and np.allclose(got, want, atol=1e-6) and receipt['adam_step_after'] == 53
        else:
            assert np.allclose(want, 0) or receipt['effective_groups'] == 0


def test_inactive_units_contribute_zero_and_are_not_reweighted(tmp_path):
    units = units4()
    _, _, common = make_common(tmp_path, units, LENGTHS, backend=MicroFake(DALLAS_INACTIVE))
    run_arm(tmp_path, 'A', units, LENGTHS, common, backend=MicroFake(DALLAS_INACTIVE))
    seg = tmp_path / 'arm_A/segment_0'
    evidence = read(seg / 'gradient_evidence_PRIVATE.json')
    assert [e['effective_groups'] for e in evidence if e['kind'] == ms.DALLAS_KIND] == [0, 0]
    got = np.load(seg / 'joint_gradient_float32_PRIVATE.safetensors')
    want = expected_joint(evidence)
    assert np.allclose(got, want, atol=1e-6) and not np.allclose(got, 2 * want, atol=1e-6)


def test_no_signal_means_no_update_no_resample_but_four_patients_still_exposed(tmp_path):
    units = units4()
    _, _, common = make_common(tmp_path, units, LENGTHS, backend=MicroFake(always=1))
    backend, summary = run_arm(tmp_path, 'A', units, LENGTHS, common, backend=MicroFake(always=1))
    assert summary['optimizer_updates'] == 0 and summary['adam_step_after'] == 52 and summary['patients_exposed'] == 4
    assert summary['update_reasons'] == ['NO_ADMITTED_GROUP', 'NO_ADMITTED_GROUP'] and backend.generate_calls == 4
    r0 = read(tmp_path / 'arm_A/segment_0/receipt_PRIVATE.json')
    assert r0['rng_digest_after'] == common.rng1 and not (tmp_path / 'arm_A/segment_0/joint_gradient_float32_PRIVATE.safetensors').exists()


def test_committed_segment_is_never_replayed_and_resume_matches_uninterrupted(tmp_path):
    units = units4()
    _, _, common = make_common(tmp_path, units, LENGTHS)
    _, full = run_arm(tmp_path, 'A', units, LENGTHS, common, name='uninterrupted')
    first = MicroFake(); out = tmp_path / 'resumed'; out.mkdir()
    counters = shared.Counters(core.arm_budget(units, LENGTHS)['limits'], None, None)
    core.run_segment(arm='A', index=0, backend=first, units=units[:4], lengths=LENGTHS, reward_policy_sha256=REWARD, counters=counters,
                     output=out, before=lambda n: None, common=common)
    with pytest.raises(ValueError, match='never replayed'):
        core.run_segment(arm='A', index=0, backend=MicroFake(), units=units[:4], lengths=LENGTHS, reward_policy_sha256=REWARD, counters=counters,
                         output=out, before=lambda n: None, common=common)
    fresh = MicroFake()                                       # a new process: parent state in memory, committed segment 0 on disk
    resumed_counters = shared.Counters(counters.limits, counters.snapshot(), None)
    summary = core.run_arm(arm='A', backend=fresh, units=units, lengths=LENGTHS, reward_policy_sha256=REWARD, common=common, counters=resumed_counters,
                           output=out, before=lambda n: None, load_checkpoint=fresh.load_checkpoint)
    assert summary['policy_sha_after'] == full['policy_sha_after'] and summary['adam_step_after'] == full['adam_step_after']
    duplicate = MicroFake(); loads = []; before_state = core.state(duplicate)
    with pytest.raises(ValueError, match='never replayed'):
        core.run_arm(arm='A', backend=duplicate, units=units, lengths=LENGTHS, reward_policy_sha256=REWARD, common=common, counters=resumed_counters,
                     output=out, before=lambda n: None, load_checkpoint=loads.append)
    assert loads == [] and core.state(duplicate) == before_state and duplicate.generate_calls == 0   # rejected before any load/mutation


def test_failure_before_commit_leaves_no_segment_and_spent_work_is_not_refunded(tmp_path):
    units = units4()
    _, _, common = make_common(tmp_path, units, LENGTHS, backend=MicroFake(MIXED))      # scripted: admitted groups guaranteed
    backend = MicroFake(MIXED); backend.apply_fail = True
    counters = shared.Counters(core.arm_budget(units, LENGTHS)['limits'], None, None)
    out = tmp_path / 'arm_A'; out.mkdir()
    with pytest.raises(RuntimeError, match='optimizer'):
        core.run_segment(arm='A', index=0, backend=backend, units=units[:4], lengths=LENGTHS, reward_policy_sha256=REWARD, counters=counters,
                         output=out, before=lambda n: None, common=common)
    assert not (out / 'segment_0').exists() and counters.snapshot()['backward_calls'] > 0


def test_concise_condition_flows_into_training_rollouts(tmp_path):
    units, lengths = units4('concise'), lengths_for('concise')
    gen, record, common = make_common(tmp_path, units, lengths)
    seq_questions = {c['row']['extra_info']['question'] for u in units if u['kind'] == ms.SEQ_KIND for c in u['contexts']}
    assert all(q.endswith(core.CONCISE_INSTRUCTION) for q in seq_questions)
    assert all(not c['row']['extra_info']['question'].endswith(core.CONCISE_INSTRUCTION) for u in units if u['kind'] == ms.DALLAS_KIND for c in u['contexts'])
    raw = read(tmp_path / 'common/raw_rollout_and_own_old_PRIVATE.json')['raws'][0]
    assert {v['prompt_length'] for v in raw['contexts'].values()} == {CONCISE_LEN}


# ================================================================ Experiment 3: paired image substitution
def dev_fixture(nonpixel_break=None):
    records, rows = [], []
    for patient in ('p1', 'p2'):
        for level in (40, 45):
            for truth in ('FLAIR', 'contrast-enhanced T1'):
                rid = f'{patient}_{level}_{truth[0]}'
                records.append({'id': rid, 'image_sha256': sha('img' + rid), 'row_PRIVATE': {'id': rid, 'extra_info': {'patient_key': patient, 'level': level},
                                                                                             'reward_model': {'ground_truth': truth}}})
                ids = sha('ids-broken') if nonpixel_break == rid else sha('ids')
                rows.append({'id': rid, 'truth': truth, 'text': '', 'tokens': 30,
                             'actual_PRIVATE': {'prompt_sha256': sha('prompt'), 'prompt_length': 213, 'image_sha256': sha('img' + rid),
                                                'input_fingerprints_PRIVATE': {'input_ids': {'sha256': ids}, 'attention_mask': {'sha256': sha('m')},
                                                                               'image_grid_thw': {'sha256': sha('g')}, 'pixel_values': {'sha256': sha('px' + rid)}}}})
    return records, rows


def with_preds(rows, preds):
    return [{**r, 'text': answer({'F': 'FLAIR', 'T': 'contrast-enhanced T1'}[p]) if p != 'I' else 'loop loop'} for r, p in zip(rows, preds)]


def test_pairs_are_same_patient_same_level_and_exact_substitution_is_detected():
    records, rows = dev_fixture()
    pairs = core.dev_pairs(records)
    assert len(pairs) == 4 and all(records[f]['row_PRIVATE']['reward_model']['ground_truth'] == 'FLAIR' for f, _ in pairs)
    eq = core.substitution_equivalence(pairs, rows)
    assert eq['all_exact'] and eq['exact_substitution_pairs'] == 4 and eq['label'] == core.SUBSTITUTION_LABEL and eq['independent_intervention'] is False
    records2, rows2 = dev_fixture(nonpixel_break='p2_45_c')
    eq2 = core.substitution_equivalence(core.dev_pairs(records2), rows2)
    assert not eq2['all_exact'] and eq2['exact_substitution_pairs'] == 3 and eq2['label'] == 'NOT_EQUIVALENT_PREDECLARE_SEPARATE_SWAP'
    broken = json.loads(json.dumps(records)); del broken[0]['row_PRIVATE']['extra_info']['level']
    with pytest.raises(ValueError, match='Pair key'):
        core.dev_pairs(broken)


def test_image_conditioned_gain_is_distinguished_from_a_class_prior_shift():
    records, rows = dev_fixture(); pairs = core.dev_pairs(records)
    parent = with_preds(rows, 'FFFFFFFF')                    # every pair constant FLAIR: 4/8, T1 recall 0
    image_gain = with_preds(rows, 'FTFTFTFF')                # three pairs now discriminate; no FLAIR lost
    prior = with_preds(rows, 'TTTTTTTT')                     # every pair constant T1: still 4/8
    g = core.compare_to_reference(pairs, parent, image_gain)
    assert (g['discrimination_gains'], g['discrimination_losses'], g['constant_answer_shift_to_T1']) == (3, 0, 0)
    assert g['reading'] == 'IMAGE_CONDITIONED_DISCRIMINATION_GAIN' and g['case_wrong_to_correct'] == 3 and g['case_correct_to_wrong'] == 0
    assert g['sign_test_two_sided_p_DESCRIPTIVE'] == pytest.approx(.25) and g['anatomical_grounding_claim'] is False
    s = core.compare_to_reference(pairs, parent, prior)
    assert s['discrimination_gains'] == 0 and s['constant_answer_shift_to_T1'] == 4 and s['reading'] == 'CLASS_PRIOR_SHIFT_PLAUSIBLE'
    assert s['predicted_T1_change'] == 8 and core.greedy_summary(prior)['accuracy'] == core.greedy_summary(parent)['accuracy']
    inv = core.pair_profile(pairs, with_preds(rows, 'FIFTTTFF'))
    assert inv['counts']['INVALID_INVOLVED'] == 1 and inv['valid_pairs'] == 3


def test_exact_sign_test_values():
    assert core.binomial_two_sided(0, 0) == 1. and core.binomial_two_sided(3, 0) == pytest.approx(.25)
    assert core.binomial_two_sided(5, 0) == pytest.approx(.0625) and core.binomial_two_sided(2, 2) == 1.


def m(bal, correct, t1, flair, invalid):
    return {'balanced_accuracy': bal, 'correct': correct, 'T1_recall': t1, 'FLAIR_recall': flair, 'invalid': invalid}


def b_vs_a(a_preds, b_preds):
    records, rows = dev_fixture(); pairs = core.dev_pairs(records)
    return core.compare_to_reference(pairs, with_preds(rows, a_preds), with_preds(rows, b_preds))


def test_frozen_scale_decision_go_no_go_inconclusive():
    gain = {'reading': 'IMAGE_CONDITIONED_DISCRIMINATION_GAIN'}; valid = {'child_correct': 10, 'reference_correct': 8}
    ba_gain = b_vs_a('FFFFFFFF', 'FTFTFTFF')
    go = core.scale_decision({'Parent': m(.5, 12, .5, .5, 2), 'A': m(.5, 12, .5, .5, 2), 'B': m(.6, 14, .66, .5, 2)}, gain, valid, ba_gain)
    assert go['decision'] == 'GO_PROPOSE_ROUND2' and go['automatic_round2'] is False
    assert go['method_specific_image_evidence']['status'] == 'B_VS_A_PAIRED_GAIN_DESCRIPTIVE'
    assert go['method_specific_image_evidence']['paired_prompt_image_grounding_demonstrated'] is False
    one_case = core.scale_decision({'Parent': m(.5, 12, .5, .5, 2), 'A': m(.5, 13, .5, .5, 2), 'B': m(.55, 14, .58, .5, 2)}, gain, valid, ba_gain)
    assert one_case['decision'] == 'INCONCLUSIVE' and not one_case['checks']['B_beats_both_by_2_cases']
    prior = core.scale_decision({'Parent': m(.5, 12, .5, .5, 2), 'A': m(.5, 12, .5, .5, 2), 'B': m(.6, 14, .66, .5, 2)}, {'reading': 'CLASS_PRIOR_SHIFT_PLAUSIBLE'}, valid, ba_gain)
    assert prior['decision'] == 'INCONCLUSIVE'
    no = core.scale_decision({'Parent': m(.5, 12, .5, .5, 2), 'A': m(.55, 13, .5, .6, 2), 'B': m(.5, 12, .58, .41, 2)}, gain, valid, ba_gain)
    assert no['decision'] == 'NO_GO'
    flair_drop = core.scale_decision({'Parent': m(.5, 12, .5, .5, 2), 'A': m(.5, 12, .5, .5, 2), 'B': m(.6, 15, .83, .33, 2)}, gain, valid, ba_gain)
    assert flair_drop['decision'] == 'NO_GO'


def test_b_vs_parent_gain_alone_is_inconclusive_without_b_vs_a_paired_gain():
    """Frozen before any V2 output: B-vs-Parent passing everything is NOT enough when B and A have the same pair profile."""
    gain = {'reading': 'IMAGE_CONDITIONED_DISCRIMINATION_GAIN'}; valid = {'child_correct': 10, 'reference_correct': 8}
    metrics = {'Parent': m(.5, 12, .5, .5, 2), 'A': m(.5, 12, .5, .5, 2), 'B': m(.6, 14, .66, .5, 2)}
    identical = core.scale_decision(metrics, gain, valid, b_vs_a('FTFTFTFF', 'FTFTFTFF'))
    assert identical['decision'] == 'INCONCLUSIVE' and not identical['checks']['B_vs_A_paired_image_gain']
    assert identical['method_specific_image_evidence']['status'] == 'NOT_DEMONSTRATED_IDENTICAL_PAIR_PROFILES'
    assert identical['method_specific_image_evidence']['B_minus_A_net_pair_gain'] == 0
    one_pair = core.scale_decision(metrics, gain, valid, b_vs_a('FTFFFFFF', 'FTFTFFFF'))       # net +1 < 2
    assert one_pair['decision'] == 'INCONCLUSIVE' and one_pair['method_specific_image_evidence']['status'] == 'INCONCLUSIVE'
    assert one_pair['method_specific_image_evidence']['pairs_denominator'] == 4                  # all pairs, invalid included
    assert 'B-vs-A over ALL 12 DEV pairs' in core.SCALE_RULE and core.B_VS_A_MIN_NET_PAIR_GAIN == 2


def test_effect_decomposition_never_credits_prompt_gain_to_rl():
    base = {'accuracy': .4, 'balanced_accuracy': .4, 'FLAIR_recall': .4, 'T1_recall': .4, 'invalid_rate': .375}
    prompted = {**base, 'accuracy': .6, 'balanced_accuracy': .6, 'invalid_rate': .05}
    e = core.effect_decomposition(base, prompted, prompted, {**prompted, 'accuracy': .65})
    assert e['prompt_effect_on_Parent']['accuracy'] == pytest.approx(.2) and e['RL_effect_A_vs_Parent_same_prompt']['accuracy'] == 0
    assert e['RL_effect_B_vs_Parent_same_prompt']['accuracy'] == pytest.approx(.05)


# ================================================================ native glue statics (no MLX)
def test_v2_modules_import_no_mlx():
    code = textwrap.dedent('''
        import sys
        sys.path.insert(0, sys.argv[1])
        import mri_rl.accuracy_pilot_v2_core, mri_rl.accuracy_pilot_v2_native, mri_rl.accuracy_pilot_v2_entry
        bad = sorted(m for m in sys.modules if m == 'mlx' or m.startswith('mlx.') or m.startswith('mlx_vlm'))
        assert not bad, bad
        ''')
    result = subprocess.run([sys.executable, '-I', '-c', code, SRC], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_context_preparation_is_the_unchanged_ast_with_only_the_length_ledger_rebound():
    from mri_rl import accuracy_pilot_v2_native as native
    from mri_rl import shared_rollout_gspo_backend_v1 as module
    backend = native.PilotV2Backend.__new__(native.PilotV2Backend)
    backend.configure_v2(condition='concise', prompt_lengths=lengths_for('concise'))
    original = ast.parse(textwrap.dedent(inspect.getsource(module.SharedRolloutBackend._prepare_context))).body[0]
    assert backend.v2_context_ast_sha256 == hashlib.sha256(ast.dump(original).encode()).hexdigest()
    globs = backend._v2_prepare.__func__.__globals__
    assert globs is not vars(module) and globs['require_prompt_cost'] is not module.require_prompt_cost
    assert all(globs[k] is v for k, v in vars(module).items() if k not in ('require_prompt_cost', '_prepare_context'))
    cost = globs['require_prompt_cost']
    assert cost('mri_sequence', CONCISE_LEN) == CONCISE_LEN and cost('mri_visit_correspondence', 405) == 405
    for task, n in (('mri_sequence', 213), ('mri_sequence', 234), ('mri_visit_correspondence', 404)):
        with pytest.raises(ValueError):
            cost(task, n)
    assert module.require_prompt_cost('mri_sequence', 213) == 213          # historical module global untouched
    with pytest.raises(ValueError):
        native.PilotV2Backend.__new__(native.PilotV2Backend)._prepare_context('F')


def test_greedy_path_has_no_hardcoded_original_length_and_keeps_256_cap():
    from mri_rl import accuracy_pilot_v2_native as native
    for fn in (native.prepared_inputs_v2, native.generate_one_v2):
        source = inspect.getsource(fn)
        assert '213' not in source
    assert 'max_tokens=core.MAX_TOKENS' in inspect.getsource(native.generate_one_v2) and core.MAX_TOKENS == 256
    assert native.call_limits(24) == {'generations': 24, 'generated_tokens': 6144, 'LM': 6240, 'vision': 24, 'sampler': 6168}


# ================================================================ entry authority (no model)
def entry():
    import mri_rl.accuracy_pilot_v2_entry as module
    return module


def test_windows_cover_worst_case_from_actual_v1_costs_and_policy_is_unchanged():
    e = entry()
    from mri_rl import safe_micro_dynamic_resources_v1 as resources
    assert e.MODEL_POLICY == resources.POLICY and e.STAGES == core.STAGES
    for stage, b in e.BUDGETS.items():
        assert b['window_seconds'] >= 1.15 * b['worst_case_seconds'], stage
    assert sum(b['model_loads'] for b in e.BUDGETS.values()) == e.RUN_LIMITS['planned_model_loads'] == 7 <= e.RUN_LIMITS['model_loads_max'] == 9
    assert sum(b['window_seconds'] for b in e.BUDGETS.values()) <= e.RUN_LIMITS['global_operation_start_window_seconds']
    assert e.BUDGETS['A'] == e.BUDGETS['B'] and e.BUDGETS['A']['optimizer_updates'] == 2
    assert e.BUDGETS['EP']['window_seconds'] > 2384.31 / 3 * 2
    blob = json.dumps([e.BUDGETS, e.RUN_LIMITS, e.CONDITIONAL_FLOW, e.COST_MODEL])
    from mri_rl.accuracy_recovery_v1 import APPROVAL_KEYS
    assert not any('"' + k + '"' in blob for k in APPROVAL_KEYS)


def fixed_scope(tmp_path):
    e = entry()
    f = tmp_path / 'bound.json'; f.write_text('{}')
    h = shared.file_sha256(f)
    return {'schema': e.SCHEMA, 'scope_kind': e.KIND, 'action': e.ACTION, 'implemented_by': e.IMPLEMENTED_BY, 'resource_policy': e.MODEL_POLICY,
            'external_guardian_policy': e.GUARDIAN_POLICY, 'budgets': e.BUDGETS, 'run_limits': e.RUN_LIMITS, 'parent': e.PARENT, 'stage_order': list(e.STAGES),
            'automatic_retry': False, 'automatic_round2': False, 'source_sha256': {str(f): h}, 'file_sha256': {str(f): h}, 'input_sha256': sha('inputs'),
            'output': str(tmp_path / 'out')}


def grant(scope, **over):
    e = entry(); now = datetime.now(timezone.utc)
    human = {'authorized': True, 'action': e.ACTION, 'scope_sha256': e.digest(scope), 'included_stages': list(e.STAGES), 'human_instruction_sha256': sha('human'),
             'approved_by': 'researcher', 'issued_at_utc': (now - timedelta(minutes=1)).isoformat(), 'expires_at_utc': (now + timedelta(hours=2)).isoformat()}
    human.update(over); return human


def review(scope, **over):
    e = entry()
    r = {'verdict': 'PASS_WITH_LIMITATIONS', 'scope_sha256': e.digest(scope), 'source_sha256': scope['source_sha256'], 'input_sha256': scope['input_sha256'],
         'reviewer_kind': 'isolated_codex', 'reviewer_id': 'Codex/root/v2_exact_review'}
    r.update(over); return r


def test_authorization_requires_fresh_all_stage_grant_and_non_author_family_review(tmp_path):
    e = entry(); scope = fixed_scope(tmp_path); spent = []
    token = e.authorize(scope, grant(scope), review(scope), consumed=spent.append)
    assert spent == [sha('human')] and token.for_phase('A').scope['execution_stage'] == 'ARM_TRAIN'
    bad = [(grant(scope, included_stages=list(e.STAGES[:-1])), review(scope)),
           (grant(scope, authorized=False), review(scope)),
           (grant(scope, issued_at_utc=(datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()), review(scope)),
           (grant(scope), review(scope, reviewer_kind='claude_code')),
           (grant(scope), review(scope, reviewer_id='Claude Code second session')),
           (grant(scope), review(scope, reviewer_id=e.IMPLEMENTED_BY)),
           (grant(scope), review(scope, source_sha256={}))]
    for human, rev in bad:
        with pytest.raises(ValueError):
            e.authorize(scope, human, rev, consumed=spent.append)
    changed = {**scope, 'budgets': {**e.BUDGETS, 'A': {**e.BUDGETS['A'], 'optimizer_updates': 4}}}
    with pytest.raises(ValueError):
        e.authorize(changed, grant(changed), review(changed), consumed=spent.append)


def test_phase_scopes_bind_stage_execution_and_output(tmp_path):
    e = entry(); scope = fixed_scope(tmp_path)
    for stage in e.STAGES:
        ps = e.phase_scope(scope, stage)
        assert ps['execution_stage'] == core.EXECUTION[stage] and ps['output'].endswith('stage' + stage + '_PRIVATE') and ps['budget'] == e.BUDGETS[stage]
    with pytest.raises(ValueError):
        e.phase_scope(scope, 'E')


def test_v2_model_load_ledger_is_exactly_nine(tmp_path):
    e = entry()
    gate = object.__new__(e.V2ResourceGate)
    gate.authorization = types.SimpleNamespace(check=lambda name: True)
    gate.spec = {'output': str(tmp_path)}; gate.scope = {'run_limits': e.RUN_LIMITS, 'compound_scope_sha256': sha('compound')}
    for _ in range(9):
        gate.claim_model_load()
    with pytest.raises(ValueError):
        gate.claim_model_load()
    assert read(tmp_path / 'model_load_ledger_PRIVATE.json')['claimed'] == 9


# ================================================================ schedule level provenance (bound source, not row field)
def z045_fixture():
    units = units4()
    rows = [c['row'] for u in units if u['kind'] == ms.SEQ_KIND for c in u['contexts']]
    images = {r['id']: sha('png' + r['id']) for r in rows}
    def entry(r):                                                      # actual authenticated DATA56 entry schema
        patient, modality = r['extra_info']['patient_key'], 'flair' if r['reward_model']['ground_truth'] == 'FLAIR' else 't1ce'
        return {'level': 45, 'patient_key_PRIVATE': patient, 'original_source_record_id_PRIVATE': '%s_z045_%s_sequence' % (patient, modality),
                'modality_PRIVATE': modality, 'expected_png_sha256': images[r['id']], 'record_PRIVATE': json.loads(json.dumps(r))}
    return units, rows, [entry(r) for r in rows], images


def test_schedule_never_reports_missing_row_level_as_proven_and_z045_comes_from_bound_source():
    units, rows, pool, images = z045_fixture()
    assert core.validate_schedule(units, set())['row_level_field'] == 'PRESENT_IN_ROWS_NOT_USED_AS_PROOF'      # synthetic rows carry one
    real_like = json.loads(json.dumps(units))                         # actual frozen rows: no extra_info.level at all
    for u in real_like:
        for c in u['contexts']:
            c['row']['extra_info'].pop('level', None)
    report = core.validate_schedule(real_like, set())
    assert 'levels' not in report and report['row_level_field'] == 'ABSENT_IN_MODEL_ROWS_BY_DESIGN'
    z = core.z045_provenance(rows, pool, images)
    assert z['level'] == 45 and z['unique_image_sha256'] == 8 and z['complete_patient_pairs'] == 4 and z['row_level_field_used_as_proof'] is False
    def mutate(index, change):
        bad = json.loads(json.dumps(pool)); change(bad[index]); return bad
    sid = 'original_source_record_id_PRIVATE'
    legacy = {k: v for k, v in pool[2].items() if k == 'record_PRIVATE'}
    legacy['record_PRIVATE'] = {**legacy['record_PRIVATE'], 'paired_z045_data_v1': {'level': 45, 'source_record_id_PRIVATE': pool[2][sid]}}
    bads = [mutate(0, lambda e: e.update({sid: e[sid].replace('z045', 'z091')})),
            mutate(0, lambda e: e.update({sid: e[sid].replace('flair', 't1ce')})),
            mutate(0, lambda e: e.pop(sid)),
            mutate(1, lambda e: e.update(level=91)),
            mutate(1, lambda e: e.update(level='45')),
            mutate(1, lambda e: e.pop('level')),
            mutate(3, lambda e: e.update(modality_PRIVATE='flair' if e['modality_PRIVATE'] == 't1ce' else 't1ce')),
            mutate(4, lambda e: e.update(patient_key_PRIVATE=e['patient_key_PRIVATE'] + 'x')),
            mutate(5, lambda e: e.update(expected_png_sha256=sha('other'))),
            mutate(5, lambda e: e.pop('expected_png_sha256')),
            mutate(6, lambda e: e['record_PRIVATE']['images'].append('extra') if isinstance(e['record_PRIVATE'].get('images'), list)
                   else e['record_PRIVATE'].update(images=['extra'])),
            mutate(7, lambda e: e['record_PRIVATE']['reward_model'].update(ground_truth='OTHER')),
            [legacy if i == 2 else e for i, e in enumerate(pool)]]        # invented legacy schema is not accepted as a fallback
    for bad in bads:
        with pytest.raises(ValueError):
            core.z045_provenance(rows, bad, images)
    with pytest.raises(ValueError, match='expected PNG SHA-256 must equal the planned'):
        core.z045_provenance(rows, pool, {**images, rows[1]['id']: images[rows[0]['id']]})
    with pytest.raises(ValueError, match='unique planned MRI image'):     # source and plan agree, but duplicated
        core.z045_provenance(rows, mutate(1, lambda e: e.update(expected_png_sha256=images[rows[0]['id']])),
                             {**images, rows[1]['id']: images[rows[0]['id']]})


# ================================================================ journal: immutable per-segment counters snapshot (real Journal)
def journal_segment(out, cpath, name, counters, charge, legacy=False):
    from mri_rl import accuracy_pilot_v2_native as native
    def op(opdir):
        counters.charge('backward_calls', charge)
        payload = out / (name + '_PRIVATE.json'); shared.publish_once(payload, {'segment': name})
        files = {str(payload.resolve()): shared.file_sha256(payload)}
        files.update({str(cpath.resolve()): shared.file_sha256(cpath)} if legacy else native.counters_snapshot(opdir, counters))
        return {'value': {'counters_after': counters.snapshot()}, 'files': files}
    return op


def test_two_segment_journal_commits_survive_later_ledger_charges(tmp_path):
    from mri_rl.safe_micro_journal_v1 import Journal
    from mri_rl import accuracy_pilot_v2_native as native
    limits = core.arm_budget(units4(), LENGTHS)['limits']
    out = tmp_path / 'stageA_PRIVATE'; out.mkdir(); cpath = out / 'arm_counters_PRIVATE.json'
    with Journal(out / 'v2_journal_PRIVATE', {'stage': 'A'}) as journal:
        c = native.resume_counters(cpath, limits, [])
        r0 = journal.run('segment0', journal_segment(out, cpath, 'segment0', c, 32))
        c = native.resume_counters(cpath, limits, [r0['counters_after']])     # resumed process: spent ledger restored
        journal.run('segment1', journal_segment(out, cpath, 'segment1', c, 32))
        assert journal.assert_recoverable() and journal.get('segment0')['counters_after']['backward_calls'] == 32
        assert read(cpath)['used']['backward_calls'] == 64
        assert read(out / 'v2_journal_PRIVATE/segment0/counters_after_PRIVATE.json')['used']['backward_calls'] == 32
    # reproduction of the confirmed blocker: committing the MUTABLE ledger invalidates segment 0 once segment 1 charges it
    out = tmp_path / 'legacy'; out.mkdir(); cpath = out / 'arm_counters_PRIVATE.json'
    with Journal(out / 'v2_journal_PRIVATE', {'stage': 'A'}) as journal:
        c = native.resume_counters(cpath, limits, [])
        journal.run('segment0', journal_segment(out, cpath, 'segment0', c, 32, legacy=True))
        journal.run('segment1', journal_segment(out, cpath, 'segment1', c, 32, legacy=True))
        with pytest.raises(ValueError, match='Committed artifact missing/changed'):
            journal.get('segment0')


def test_restored_ledger_cannot_be_below_a_committed_snapshot(tmp_path):
    from mri_rl import accuracy_pilot_v2_native as native
    limits = core.arm_budget(units4(), LENGTHS)['limits']; cpath = tmp_path / 'ledger.json'
    committed = {k: 0 for k in shared.CHARGEABLE}; committed['backward_calls'] = 32
    assert native.resume_counters(cpath, limits, [committed]).snapshot() == committed            # absent ledger: committed spend, no refund
    shared.replace_atomic(cpath, {'limits': limits, 'used': {**committed, 'backward_calls': 5}})
    with pytest.raises(ValueError, match='below a committed'):
        native.resume_counters(cpath, limits, [committed])


def test_native_arm_stage_commits_snapshots_not_the_mutable_ledger():
    from mri_rl import accuracy_pilot_v2_native as native
    source = inspect.getsource(native.run_phase)
    assert 'shared.file_sha256(cpath)' not in source and 'str(path.resolve()): shared.file_sha256(path)' not in source
    assert source.count('counters_snapshot(opdir') == 3


# ================================================================ entry: every completed / skipped stage is revalidated
SCOPE_FILE = sha('scope file')


def finish_stage(e, scope, stage, **extra):
    base = Path(scope['output']); out = base / ('stage' + stage + '_PRIVATE'); out.mkdir(parents=True, exist_ok=True)
    terminal = out / 'terminal_PRIVATE.json'
    terminal.write_text(json.dumps({'status': 'PASS', 'release_verified': True, 'scope_sha256': e.digest(e.phase_scope(scope, stage)), 'stage': stage, **extra}))
    wdir = base / ('stage%s_attempt0_exit_PRIVATE' % stage); wdir.mkdir()
    logs = {}
    for name in ('stdout', 'stderr'):
        (wdir / (name + '_PRIVATE.log')).write_text(name + ' closed\n'); logs[name + '_sha256'] = shared.file_sha256(wdir / (name + '_PRIVATE.log'))
    witness = wdir / 'actual_exit_PRIVATE.json'
    witness.write_text(json.dumps({'native_exit_code': 0, 'child_reaped': True, 'post_child_global_lock': {'no_open_holders': True}, **logs}))
    (base / ('stage' + stage + '_verified_PRIVATE.json')).write_text(json.dumps({'scope_file_sha256': SCOPE_FILE, 'terminal_sha256': shared.file_sha256(terminal),
                                                                               'exit_path': str(witness), 'exit_sha256': shared.file_sha256(witness)}))


def chain_through_ep(tmp_path, decision='KEEP_ORIGINAL', ep_reason='KEEP_ORIGINAL_REUSE_HASH_BOUND_V1_PARENT_DEV24'):
    e = entry(); tmp_path.mkdir(parents=True, exist_ok=True); scope = fixed_scope(tmp_path); spec = {'output': scope['output']}
    finish_stage(e, scope, 'P', prompt_decision={'decision': decision, 'condition': 'original' if decision == 'KEEP_ORIGINAL' else 'concise'})
    for stage in ('C', 'A', 'B'):
        finish_stage(e, scope, stage)
    (Path(scope['output']) / 'stageEP_skipped_PRIVATE.json').write_text(json.dumps({'stage': 'EP', 'reason': ep_reason, 'model_loads': 0, 'native_calls': 0}))
    return e, scope, spec


def test_resume_revalidates_every_completed_stage_not_only_the_immediate_predecessor(tmp_path):
    e, scope, spec = chain_through_ep(tmp_path)
    e._prior(scope, spec, 'EA', SCOPE_FILE)
    assert [e._completed(scope, spec, s, SCOPE_FILE) for s in e.STAGES] == ['VERIFIED'] * 4 + ['SKIPPED', None, None]
    with pytest.raises(ValueError, match='scope mismatch'):
        e._prior(scope, spec, 'EA', sha('other scope file'))
    p = Path(scope['output']) / 'stageP_PRIVATE/terminal_PRIVATE.json'                 # an EARLY terminal decision changes after commit
    original = p.read_text(); p.write_text(original.replace('KEEP_ORIGINAL', 'ADOPT_CONCISE'))
    with pytest.raises(ValueError, match='terminal changed'):
        e._prior(scope, spec, 'EA', SCOPE_FILE)
    p.write_text(original); e._prior(scope, spec, 'EA', SCOPE_FILE)
    log = Path(scope['output']) / 'stageC_attempt0_exit_PRIVATE/stdout_PRIVATE.log'
    log.write_text('rewritten\n')
    with pytest.raises(ValueError, match='Closed log changed'):
        e._prior(scope, spec, 'EA', SCOPE_FILE)


def test_skip_marker_must_match_reason_derived_from_bound_evidence(tmp_path):
    e, scope, spec = chain_through_ep(tmp_path / 'forged', decision='ADOPT_CONCISE')     # EP must RUN under concise; a skip is forged
    with pytest.raises(ValueError, match='Skip marker'):
        e._prior(scope, spec, 'EA', SCOPE_FILE)
    e, scope, spec = chain_through_ep(tmp_path / 'reason', ep_reason='ZERO_UPDATE_ARM_IS_PARENT_ALIAS')
    with pytest.raises(ValueError, match='Skip marker'):
        e._completed(scope, spec, 'EP', SCOPE_FILE)
    e, scope, spec = chain_through_ep(tmp_path / 'artifacts')
    (Path(scope['output']) / 'stageEP_clock_PRIVATE.json').write_text('{}')
    with pytest.raises(ValueError, match='native artifacts'):
        e._completed(scope, spec, 'EP', SCOPE_FILE)
    e, scope, spec = chain_through_ep(tmp_path / 'both')
    finish_stage(e, scope, 'EP')
    with pytest.raises(ValueError, match='both verified and skipped'):
        e._completed(scope, spec, 'EP', SCOPE_FILE)


def test_execute_chain_and_child_never_trust_completion_markers_by_existence():
    e = entry()
    chain, child = inspect.getsource(e.execute_chain), inspect.getsource(e.execute_child)
    assert 'if verified.exists() or skipped.exists()' not in chain and '_completed(scope, spec, stage, args.expected_scope_sha256) is not None' in chain
    assert "'Stage incomplete before analysis: '" in chain and '_prior(scope, spec, stage, args.expected_scope_sha256)' in child
