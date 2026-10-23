"""TESTS-FIRST CPU tests for heterogeneous_mri_pilot_v1 (sequence FLAIR/T1ce + VQA-RAD midline-shift). SYNTHETIC only.

No MLX/torch/transformers import, no model, no private MRI payload. The fake backends are the existing numpy stand-ins
(test_multistep_gspo_v1.FakeBackend / test_safe_micro_training_v1.MicroFake) extended only with scripted greedy answers.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from test_multistep_gspo_v1 import make_plan, REWARD                         # noqa: E402
from test_safe_micro_training_v1 import MicroFake, MIXED, DALLAS_INACTIVE    # noqa: E402
from mri_rl import multistep_gspo_core_v1 as ms                             # noqa: E402
from mri_rl import heterogeneous_mri_pilot_v1 as hp                         # noqa: E402

MID = hp.MIDLINE_ABILITY
LENGTHS = {'mri_sequence': 213, MID: 300}


# ================================================================ SYNTHETIC fixtures
def src_row(qid, image, case, question, answer, *, organ='HEAD', evaluation='evaluated', qtype='PRES', atype='CLOSED'):
    return {'qid': qid, 'image_name': image, 'image_case_url': f'https://medpix.nlm.nih.gov/case?id={case}', 'image_organ': organ,
            'question': question, 'answer': answer, 'answer_type': atype, 'question_type': qtype, 'evaluation': evaluation,
            'phrase_type': 'freeform'}


def modality(qid, image, case, answer):
    return src_row(qid, image, case, 'What modality is this?', answer, qtype='MODALITY', atype='OPEN')


def census_rows():
    rows = []
    for i in range(5):                                                       # 5 MRI midline items: 3 yes, 2 no, distinct cases
        rows += [src_row(i, f'm{i}.jpg', f'c{i}', 'Is there midline shift?' if i % 2 else 'Are structures shifted across the midline?', 'yes' if i < 3 else 'no'),
                 modality(100 + i, f'm{i}.jpg', f'c{i}', 'MRI - T2 weighted')]
    rows += [src_row(10, 'ct.jpg', 'cct', 'Is there midline shift?', 'yes'), modality(110, 'ct.jpg', 'cct', 'CT noncontrast')]   # CT
    rows += [src_row(11, 'u.jpg', 'cu', 'Is there midlight shift?', 'no')]                                                        # no modality evidence
    rows += [src_row(12, 'n.jpg', 'cn', 'Is there midline shift?', 'no', evaluation='not evaluated'), modality(112, 'n.jpg', 'cn', 'MRI')]
    rows += [src_row(13, 'mass.jpg', 'cm', 'Is there mass effect causing shift?', 'yes'), modality(113, 'mass.jpg', 'cm', 'MRI')]
    rows += [src_row(14, 'm0b.jpg', 'c0', 'Is there midline shift?', 'yes'), modality(114, 'm0b.jpg', 'c0', 'MRI')]               # same case as m0
    rows += [src_row(15, 'chest.jpg', 'cc', 'Is there midline shift?', 'no', organ='CHEST')]
    rows += [src_row(16, 'm0.jpg', 'c0', 'Is there mass effect?', 'yes'),                               # mass effect without shift: not counted
             src_row(17, 'm1.jpg', 'c1', 'Is there shifting of structures across the middle?', 'yes')]  # related paraphrase only
    return rows


def test_census_mri_evidence_includes_affirmed_sequence_question_or_mri_wording_but_not_negated():
    rows = [src_row(1, 'a.jpg', 'ca', 'Is there midline shift?', 'yes'), src_row(2, 'a.jpg', 'ca', 'Is this a T2 weighted image?', 'yes', qtype='MODALITY'),
            src_row(3, 'b.jpg', 'cb', 'Is there midline shift on this MRI?', 'no'),
            src_row(4, 'c.jpg', 'cc', 'Is there midline shift?', 'no'), src_row(5, 'c.jpg', 'cc', 'Is this an MRI?', 'no')]
    c = hp.census(rows)
    assert c['by_modality_evidence'] == {'MRI': 2, 'UNCERTAIN': 1}
    assert c['eligible_case_components'] == {'yes': 1, 'no': 1, 'conflict': 0}


def sequence_unit():
    return ms.make_units(make_plan(1))[0]


def midline_rows():
    out = []
    for qid, answer, split in ((52, 'no', 'train'), (69, 'yes', 'train'), (1037, 'yes', 'eval'), (2236, 'no', 'eval')):
        out.append(hp.midline_row({'source_qid': str(qid), 'split': split, 'question': f'q{qid} midline shift?', 'ground_truth': answer,
                                   'source_image_sha256': f'{qid:064d}', 'model_input': {'image': f'/x/img{qid}.jpg', 'question': f'q{qid} midline shift?'}}))
    return out


def midline_unit():
    return hp.midline_unit([r for r in midline_rows() if r['extra_info']['split'] == 'train'])


def train_rows():
    return [r for r in midline_rows() if r['extra_info']['split'] == 'train']


def control_record(train=None):
    train = train or train_rows()
    return {'schema': 'heterogeneous-central-occlusion-controls-v1', 'model_outputs_used': False,
            'controls': [{'source_qid': r['extra_info']['source_qid'], 'control_path': f'/x/occ{i}.png', 'control_sha256': f'{900 + i:064d}',
                          'source_sha256': r['image_sha256'], 'control_kind': hp.BORDER_KIND, 'question_original_unchanged': r['extra_info']['original_question']}
                         for i, r in enumerate(train)]}


def borders():
    return hp.border_controls(control_record(), train_rows())


class HFake(MicroFake):
    """MicroFake (52-step parent, scripted rollout grades) + scripted greedy answers for gate/eval controls."""
    def __init__(self, script=None, greedy=None, **kw):
        super().__init__(script, **kw)
        self.greedy_answers = greedy or {}
        self.greedy_calls = []

    def greedy(self, row, image_sha256):
        self.greedy_calls.append((row['id'], image_sha256))
        answer = self.greedy_answers.get((row['extra_info']['question'], image_sha256), 'yes')
        return {'text': f'<think>x</think><answer>\\boxed{{{answer}}}</answer>', 'token_ids': [1, 2], 'positions': 0}


def parent_state(b):
    return {'policy': b.policy_sha(), 'adam': b.adam_digest(), 'rng': b.rng_digest()}


# ================================================================ census
def test_census_counts_mri_evaluated_case_deduplicated_items_and_flags_mass_shift_separately():
    c = hp.census(census_rows())
    assert c['midline_yesno_head_questions'] == 9                       # 5 MRI + CT + uncertain + not-evaluated + same-case duplicate
    assert c['mass_shift_ambiguous_questions'] == 1                       # never merged into the endpoint
    assert c['related_paraphrase_questions'] == 1                         # reported separately, adds no component
    assert c['by_modality_evidence'] == {'MRI': 7, 'CT': 1, 'UNCERTAIN': 1}
    assert c['evaluated'] == 8 and c['not_evaluated'] == 1
    assert c['eligible_case_components'] == {'yes': 3, 'no': 2, 'conflict': 0}   # m0 and m0b collapse to one case
    assert c['decision'] == 'DATA_TOO_SMALL_FOR_LEARNING_CLAIM' and c['threshold'] == {'train': {'yes': 2, 'no': 2}, 'eval': {'yes': 2, 'no': 2}}
    assert 'midlight' in c['matched_spellings']


def test_census_threshold_reached_only_with_four_yes_and_four_no_independent_cases():
    rows = []
    for i in range(8):
        rows += [src_row(i, f'm{i}.jpg', f'c{i}', 'Is there midline shift?', 'yes' if i < 4 else 'no'), modality(100 + i, f'm{i}.jpg', f'c{i}', 'MRI')]
    assert hp.census(rows)['decision'] == 'SUFFICIENT_FOR_4_TRAIN_4_EVAL'
    rows.append(src_row(50, 'm0x.jpg', 'c0', 'Is there midline shift?', 'no'))                   # same case, opposite answer -> conflict
    rows.append(modality(150, 'm0x.jpg', 'c0', 'MRI'))
    c = hp.census(rows)
    assert c['eligible_case_components']['conflict'] == 1 and c['decision'] == 'DATA_TOO_SMALL_FOR_LEARNING_CLAIM'


# ================================================================ rows, grading, units
def test_midline_rows_keep_the_original_question_and_route_to_the_existing_yes_no_verifier():
    rows = midline_rows()
    r = rows[0]
    assert r['ability'] == MID and r['reward_model']['ground_truth'] == 'no'
    assert r['extra_info']['question'] == 'q52 midline shift?\n\nAnswer yes or no. ' + hp.SHORT_INSTRUCTION          # frozen once, appended only
    assert hp.MIDLINE_SUFFIX_SHA256 == __import__('hashlib').sha256(hp.MIDLINE_SUFFIX.encode()).hexdigest()
    assert r['extra_info']['original_question'] == 'q52 midline shift?' and r['images'] == ['/x/img52.jpg']
    seq_calls = []
    grader = hp.routing_grader(lambda row, text, config: seq_calls.append(row['id']) or {'score': .6, 'accuracy': .5, 'format': 1., 'raw_accuracy': 1.})
    g = grader(r, '<think>x</think><answer>\\boxed{no}</answer>', {})
    assert g['accuracy'] == 1. and g['raw_accuracy'] == 1. and g['format'] == 1. and g['score'] == pytest.approx(1.)
    w = grader(r, '<think>x</think><answer>\\boxed{yes}</answer>', {})
    assert w['accuracy'] == 0. and w['score'] == pytest.approx(.2)
    seq_row = sequence_unit()['contexts'][0]['row']
    grader(seq_row, 'x', {})
    assert seq_calls == [seq_row['id']]                                   # sequence keeps its own unchanged (calibrated) grader


def test_midline_unit_uses_the_unchanged_two_context_same_prompt_block_ordered_by_qid_not_label():
    u = midline_unit()
    assert u['kind'] == ms.DALLAS_KIND and [c['key'] for c in u['contexts']] == ['D1', 'D2']
    assert [c['row']['extra_info']['source_qid'] for c in u['contexts']] == ['52', '69']
    with pytest.raises(ValueError, match='balanced'):
        hp.midline_unit([midline_rows()[0], midline_rows()[0]])


# ================================================================ pre-update image-dependence gate (TRAIN only, 6 controls)
def test_gate_controls_are_true_swap_and_bound_border_controls_with_the_same_question():
    train = train_rows()
    controls = hp.gate_controls(train, borders())
    assert [c['name'] for c in controls] == list(hp.GATE_NAMES) and [c['kind'] for c in controls] == ['TRUE'] * 2 + ['SWAP'] * 2 + ['BORDER'] * 2
    swap, border = controls[2], controls[4]
    assert swap['row']['extra_info']['question'] == train[0]['extra_info']['question'] and swap['image_sha256'] == train[1]['image_sha256']
    assert swap['row']['images'] == train[1]['images'] and swap['image_label'] == 'yes'
    assert border['row']['extra_info']['question'] == train[0]['extra_info']['question'] and border['row']['images'] == ['/x/occ0.png']
    assert border['image_sha256'] == f'{900:064d}' and border['image_label'] is None and len({c['row']['id'] for c in controls}) == 6 - 0
    assert all(c['row']['extra_info']['split'] == 'train' for c in controls)
    assert hp.NO_IMAGE_CONTROL['status'] == 'NOT_RUN_EXISTING_PATH_REQUIRES_EXACTLY_ONE_IMAGE'


@pytest.mark.parametrize('mutate', [lambda c: c.update(source_sha256='1' * 64), lambda c: c.update(control_kind='BRAIN_MASK'),
                                    lambda c: c.update(question_original_unchanged='edited?'), lambda c: c.update(control_sha256=f'{52:064d}')])
def test_border_controls_must_bind_their_train_item_source_image_and_unchanged_question(mutate):
    record = control_record()
    mutate(record['controls'][0])
    with pytest.raises(ValueError, match='Border control'):
        hp.border_controls(record, train_rows())
    with pytest.raises(ValueError, match='frozen border-control'):
        hp.border_controls({**control_record(), 'model_outputs_used': True}, train_rows())


@pytest.mark.parametrize('answers,verdict,reason', [
    (('no', 'yes', 'yes', 'no', 'yes', 'yes'), 'PASS', 'IMAGE_FOLLOWING'),
    (('yes', 'no', 'no', 'yes', 'no', 'no'), 'PASS', 'IMAGE_FOLLOWING'),                    # inverted pattern: dependence, not correctness
    (('no', 'yes', 'yes', 'no', 'no', 'yes'), 'NO_SIGNAL', 'BORDER_REPRODUCES_TRUE'),       # borders alone reproduce TRUE -> blocked
    (('yes', 'yes', 'yes', 'yes', 'no', 'no'), 'NO_SIGNAL', 'CONSTANT_ANSWER'),
    (('no', 'yes', 'no', 'yes', 'yes', 'yes'), 'NO_SIGNAL', 'QUESTION_FOLLOWING'),
    (('no', 'yes', 'yes', 'no', 'yes', None), 'NO_SIGNAL', 'INVALID_CONTROL'),
    (('no', 'yes', 'yes', 'yes', 'yes', 'yes'), 'NO_SIGNAL', 'MIXED'),
])
def test_image_dependence_gate_is_conservative_and_never_uses_correctness_or_eval(answers, verdict, reason):
    out = hp.image_dependence_gate(dict(zip(hp.GATE_NAMES, answers)))
    assert out['verdict'] == verdict and out['reason'] == reason and out['controls'] == 6
    assert out['grounding_claim'] == 'NOT_CLAIMED' and out['uses_eval_or_dev'] is False and out['border_control'] == 'LIMITED_DIAGNOSTIC_NOT_A_BRAIN_MASK'
    with pytest.raises(ValueError, match='six'):
        hp.image_dependence_gate(dict(zip(hp.GATE_NAMES[:4], answers)))


# ================================================================ the one joint update (or NO_SIGNAL)
def run_segment(tmp_path, script, name='seg'):
    backend = HFake(script)
    expected = parent_state(backend)
    result = hp.train_segment(backend=backend, seq_unit=sequence_unit(), mid_unit=midline_unit(), lengths=LENGTHS, reward_policy_sha256=REWARD,
                              output=tmp_path / name, before=lambda n: None, expected_parent=expected, parent_receipt_sha256=REWARD)
    return backend, expected, result


def test_one_joint_update_with_explicit_equal_task_weight(tmp_path):
    backend, expected, r = run_segment(tmp_path, MIXED)
    assert r['decision']['update'] is True and r['updated'] is True and backend.apply_calls == 1 and backend.step == 53
    assert r['task_weight'] == {'mri_sequence': .5, MID: .5} and r['per_response_weight_within_task'] == 1 / 8
    assert set(r['admitted_groups']) == {'mri_sequence', MID} and all(v > 0 for v in r['admitted_groups'].values())
    assert (tmp_path / 'seg' / 'checkpoint_PRIVATE').is_dir() and json.loads((tmp_path / 'seg' / 'receipt_PRIVATE.json').read_text())['updated'] is True


def test_joint_gradient_is_half_sequence_plus_half_midline_regardless_of_admitted_counts(tmp_path):
    captured = {}

    class Capture(HFake):
        def apply_update(self, gradient, lr=.05):
            captured['joint'] = np.array(gradient, dtype=np.float64)
            return super().apply_update(gradient, lr)

        def gradient(self, planned, own_old, unit):
            tree, info = super().gradient(planned, own_old, unit)
            captured.setdefault('task', []).append(None if tree is None else np.array(tree, dtype=np.float64))
            return tree, info
    backend = Capture(MIXED)
    hp.train_segment(backend=backend, seq_unit=sequence_unit(), mid_unit=midline_unit(), lengths=LENGTHS, reward_policy_sha256=REWARD,
                     output=tmp_path / 'seg', before=lambda n: None, expected_parent=parent_state(backend), parent_receipt_sha256=REWARD)
    g_seq, g_mid = captured['task']
    assert np.allclose(captured['joint'], .5 * g_seq + .5 * g_mid, atol=1e-6)


SEQ_INACTIVE = {**MIXED, **{f'F{i}': 1 for i in range(4)}, **{f'T{i}': 0 for i in range(4)}}
BOTH_INACTIVE = {**DALLAS_INACTIVE, **{f'F{i}': 1 for i in range(4)}, **{f'T{i}': 0 for i in range(4)}}


@pytest.mark.parametrize('script,missing', [(DALLAS_INACTIVE, [MID]), (SEQ_INACTIVE, ['mri_sequence']), (BOTH_INACTIVE, ['mri_sequence', MID])])
def test_no_signal_in_either_task_means_no_backward_no_update_and_unchanged_parameters(tmp_path, script, missing):
    gradients = []

    class Spy(HFake):
        def gradient(self, planned, own_old, unit):
            gradients.append(unit['index'])
            return super().gradient(planned, own_old, unit)
    backend = Spy(script)
    expected = parent_state(backend)
    r = hp.train_segment(backend=backend, seq_unit=sequence_unit(), mid_unit=midline_unit(), lengths=LENGTHS, reward_policy_sha256=REWARD,
                         output=tmp_path / 'seg', before=lambda n: None, expected_parent=expected, parent_receipt_sha256=REWARD)
    assert r['decision'] == {'update': False, 'reason': 'NO_SIGNAL', 'tasks_without_valid_contrast': missing}
    assert gradients == [] and backend.apply_calls == 0 and backend.step == 52 and r['updated'] is False and r['backward_calls'] == 0
    assert backend.policy_sha() == expected['policy'] and backend.adam_digest() == expected['adam']
    assert not (tmp_path / 'seg' / 'checkpoint_PRIVATE').exists() and r['alias_of_parent'] is True     # terminal = Parent alias
    assert r['dallas_data'] is False and 'TRANSPORT ONLY' in r['block_transport'][MID]


class Halt(BaseException):
    pass


@pytest.mark.parametrize('boundary', ['hetero_gradient_mri_sequence', 'hetero_gradient_' + MID, 'hetero_before_update', 'hetero_checkpoint'])
def test_resource_stop_records_STOP_without_in_process_rollback_or_further_work(tmp_path, boundary):
    from mri_rl.safe_micro_dynamic_resources_v1 import ResourcePause
    restores, grads = [], []

    class Guarded(HFake):
        def restore(self, snap):
            restores.append(1)
            return super().restore(snap)

        def gradient(self, planned, own_old, unit):
            grads.append(unit['index'])
            return super().gradient(planned, own_old, unit)

    def before(name):
        if name == boundary:
            raise ResourcePause('RED')
    backend = Guarded(MIXED)
    with pytest.raises(ResourcePause):
        hp.train_segment(backend=backend, seq_unit=sequence_unit(), mid_unit=midline_unit(), lengths=LENGTHS, reward_policy_sha256=REWARD,
                         output=tmp_path / 'seg', before=before, expected_parent=parent_state(backend), parent_receipt_sha256=REWARD)
    stops = list((tmp_path / 'seg').glob('STOP_*_PRIVATE.json'))
    assert restores == [] and len(stops) == 1 and json.loads(stops[0].read_text())['in_process_rollback'] is False
    assert len(grads) == {'hetero_gradient_mri_sequence': 0, 'hetero_gradient_' + MID: 1}.get(boundary, 2)
    assert backend.apply_calls == (1 if boundary == 'hetero_checkpoint' else 0)            # a spent update is never refunded or redone
    assert not (tmp_path / 'seg' / 'receipt_PRIVATE.json').exists()
    with pytest.raises(ValueError, match='never replayed'):                                # no replay of the incomplete operation
        hp.train_segment(backend=HFake(MIXED), seq_unit=sequence_unit(), mid_unit=midline_unit(), lengths=LENGTHS, reward_policy_sha256=REWARD,
                         output=tmp_path / 'seg', before=lambda n: None, expected_parent=parent_state(HFake(MIXED)), parent_receipt_sha256=REWARD)


def test_incomplete_journal_operation_is_indeterminate_and_never_replayed(tmp_path):
    from mri_rl.safe_micro_journal_v1 import Journal, Indeterminate
    calls = []

    def crash(opdir):
        calls.append(1)
        raise Halt()
    with Journal(tmp_path / 'j', {'scope': 'x'}) as journal:
        with pytest.raises(Halt):
            journal.run('parent_phase', crash, reservation={'optimizer_updates': 1})
    with Journal(tmp_path / 'j', {'scope': 'x'}) as journal:
        with pytest.raises(Indeterminate):
            journal.run('parent_phase', crash)
    assert calls == [1]


def test_segment_is_one_shot_and_refuses_a_non_parent_start(tmp_path):
    run_segment(tmp_path, MIXED)
    with pytest.raises(ValueError, match='never replayed'):
        run_segment(tmp_path, MIXED)
    backend = HFake(MIXED)
    backend.step = 53
    with pytest.raises(ValueError, match='Adam52'):
        hp.train_segment(backend=backend, seq_unit=sequence_unit(), mid_unit=midline_unit(), lengths=LENGTHS, reward_policy_sha256=REWARD,
                         output=tmp_path / 'other', before=lambda n: None, expected_parent=parent_state(backend), parent_receipt_sha256=REWARD)


# ================================================================ evaluation summary and predeclared outcome
def boxed(answer):
    return f'<think>x</think><answer>\\boxed{{{answer}}}</answer>'


def seq_eval(flair_ok, t1_ok, invalid=0):
    rows = []
    for i in range(12):
        rows.append({'task': 'mri_sequence', 'id': f'f{i}', 'truth': 'FLAIR', 'tokens': 9, 'text': boxed('FLAIR' if i < flair_ok else 'contrast-enhanced T1')})
    for i in range(12):
        bad = i >= 12 - invalid
        rows.append({'task': 'mri_sequence', 'id': f't{i}', 'truth': 'contrast-enhanced T1', 'tokens': 9,
                     'text': 'no box' if bad else boxed('contrast-enhanced T1' if i < t1_ok else 'FLAIR')})
    return rows


def mid_eval(answers):
    return [{'task': MID, 'id': f'm{t}', 'truth': t, 'text': boxed(a) if a else 'none'} for t, a in zip(('yes', 'no'), answers)]


def test_evaluation_summary_reports_per_task_equal_worst_recalls_invalids_with_strict_full_format():
    s = hp.evaluation_summary(seq_eval(10, 8) + mid_eval(('yes', 'yes')))
    seq, mid = s['tasks']['mri_sequence'], s['tasks'][MID]
    assert seq['FLAIR_recall'] == pytest.approx(10 / 12) and seq['T1_recall'] == pytest.approx(8 / 12) and seq['balanced_accuracy'] == pytest.approx(18 / 24)
    assert mid['strict_accuracy'] == .5 and mid['items'] == 2 and mid['invalid'] == 0
    assert s['equal_task'] == pytest.approx((18 / 24 + .5) / 2) and s['worst_task'] == .5
    answer_only = [{**r, 'text': '<answer>\\boxed{FLAIR}</answer>'} if r['id'] == 'f0' else r for r in seq_eval(10, 8)]
    assert hp.evaluation_summary(answer_only + mid_eval(('yes', 'yes')))['tasks']['mri_sequence']['invalid'] == 1      # parser NOT relaxed
    with pytest.raises(ValueError, match='inventory'):
        hp.evaluation_summary(seq_eval(10, 8)[:-1] + mid_eval(('yes', 'no')))


def summary(flair, t1, mid, invalid=0):
    return hp.evaluation_summary(seq_eval(flair, t1, invalid) + mid_eval(mid))


@pytest.mark.parametrize('terminal,label', [
    (lambda: summary(10, 8, ('yes', 'no')), 'FEASIBLE'),
    (lambda: summary(10, 7, ('yes', 'no')), 'NEGATIVE_TRANSFER'),            # midline up, T1 down: cross-task negative transfer
    (lambda: summary(11, 8, ('no', 'yes')), 'NEGATIVE_TRANSFER'),            # sequence up, midline balanced down: never hidden
    (lambda: summary(10, 8, ('yes', None)), 'FAIL_FORMAT_COLLAPSE'),         # invalid increase is a format failure, not transfer
    (lambda: summary(10, 8, ('yes', 'no'), invalid=1), 'FAIL_FORMAT_COLLAPSE'),
])
def test_predeclared_outcome_rule(terminal, label):
    parent = summary(10, 8, ('yes', 'yes'))
    out = hp.outcome(parent, terminal(), updated=True)
    assert out['outcome'] == label and out['accuracy_claim'] == 'NONE_FEASIBILITY_ONLY' and out['data_scale'] == 'DATA_TOO_SMALL_FOR_LEARNING_CLAIM'
    assert set(out['item_flips']) == {'mri_sequence', MID}
    assert hp.outcome(parent, parent, updated=False)['outcome'] == 'NO_SIGNAL'


def test_item_flips_report_correct_to_wrong_per_task():
    out = hp.outcome(summary(10, 8, ('yes', 'yes')), summary(11, 7, ('yes', 'no')), updated=True)
    assert out['item_flips'] == {'mri_sequence': {'correct_to_wrong': 1, 'wrong_to_correct': 1}, MID: {'correct_to_wrong': 0, 'wrong_to_correct': 1}}


# ================================================================ budget (exact formula; refuses unmeasured prompt lengths)
GATE_LENGTHS = {'TRUE_0': 300, 'TRUE_1': 310, 'SWAP_0': 305, 'SWAP_1': 306, 'BORDER_0': 301, 'BORDER_1': 311}


def test_budget_is_exact_from_frozen_prompt_lengths_and_refuses_unmeasured_midline_lengths():
    lengths = {'sequence': 233, 'midline_train': {'52': 300, '69': 310}, 'midline_gate': dict(GATE_LENGTHS), 'midline_eval': {'1037': 320, '2236': 330}}
    b = hp.budget(lengths, dev_rows=24)
    assert b['model_loads_max'] == 2 and b['optimizer_updates_max'] == 1 and b['sampled_rollout_responses'] == 16
    assert b['greedy_calls_max'] == 6 + 2 + 24 + 2 == 34 and b['generations_max'] == 50 and b['generated_tokens_max'] == 12800
    assert b['own_old_TF_positions_max'] == 8 * (233 + 255) + 4 * (300 + 255) + 4 * (310 + 255)
    assert b['backward_calls_max'] == 16 and b['scorer_forwards_max'] == 32 and b['gradient_TF_positions_max'] == b['own_old_TF_positions_max']
    with pytest.raises(ValueError, match='PENDING_PROCESSOR_MEASUREMENT'):
        hp.budget({**lengths, 'midline_eval': {'1037': None, '2236': 330}}, dev_rows=24)
    with pytest.raises(ValueError, match='6 gate controls'):
        hp.budget({**lengths, 'midline_gate': {k: v for k, v in GATE_LENGTHS.items() if not k.startswith('BORDER')}}, dev_rows=24)


# ================================================================ controller: journaled, one-shot, gate before any rollout
def parent_phase(tmp_path, backend, name='parent', ev=None):
    return hp.run_parent_phase(backend=backend, seq_unit=sequence_unit(), midline_train=train_rows(),
                               midline_eval=ev if ev is not None else [r for r in midline_rows() if r['extra_info']['split'] == 'eval'],
                               borders=borders(), lengths=LENGTHS, reward_policy_sha256=REWARD, output=tmp_path / name, before=lambda n: None,
                               expected_parent=parent_state(backend), parent_receipt_sha256=REWARD)


def following(border_follows):
    (q0, q1), (s0, s1) = ((r['extra_info']['question'] for r in train_rows()), (r['image_sha256'] for r in train_rows()))
    o0, o1 = f'{900:064d}', f'{901:064d}'
    greedy = {(q0, s0): 'no', (q1, s1): 'yes', (q0, s1): 'yes', (q1, s0): 'no'}
    greedy.update({(q0, o0): 'no', (q1, o1): 'yes'} if border_follows else {(q0, o0): 'yes', (q1, o1): 'yes'})
    return greedy


def test_parent_phase_stops_at_a_failed_image_gate_before_any_rollout(tmp_path):
    backend = HFake(MIXED)                                                    # greedy default 'yes' everywhere -> CONSTANT_ANSWER
    out = parent_phase(tmp_path, backend)
    assert out['status'] == 'NO_SIGNAL_IMAGE_GATE' and backend.generate_calls == 0 and backend.apply_calls == 0
    assert len(backend.greedy_calls) == 6 and out['optimizer_updates'] == 0
    with pytest.raises(ValueError, match='exist'):
        parent_phase(tmp_path, HFake(MIXED))


def test_border_retained_reproduction_of_true_blocks_the_update(tmp_path):
    backend = HFake(MIXED, greedy=following(border_follows=True))
    out = parent_phase(tmp_path, backend)
    assert out['status'] == 'NO_SIGNAL_IMAGE_GATE' and out['gate']['reason'] == 'BORDER_REPRODUCES_TRUE'
    assert backend.generate_calls == 0 and backend.apply_calls == 0 and len(backend.greedy_calls) == 6


def test_parent_phase_with_image_following_gate_evaluates_parent_then_one_update(tmp_path):
    ev = [r for r in midline_rows() if r['extra_info']['split'] == 'eval']
    backend = HFake(MIXED, greedy=following(border_follows=False))
    out = parent_phase(tmp_path, backend)
    assert out['status'] == 'UPDATED' and out['gate']['verdict'] == 'PASS' and backend.apply_calls == 1
    assert [c[0] for c in backend.greedy_calls][6:] == [r['id'] for r in ev]          # Parent midline EVAL before any rollout/update
    assert out['parent_midline_eval_count'] == 2 and out['optimizer_updates'] == 1 and out['greedy_calls'] == 8


# ================================================================ native boundary (no MLX): exact per-row prompt length hook
def test_native_prompt_cost_is_exact_per_row_for_midline_and_per_task_for_sequence():
    from mri_rl import heterogeneous_mri_pilot_native_v1 as native

    class Stub:
        _hp_row_id = None
    stub = Stub()
    cost = native.per_row_prompt_cost(stub, 233, {'vqarad_52': 300, 'vqarad_69': 310})
    assert cost('mri_sequence', 233) == 233
    with pytest.raises(ValueError):
        cost('mri_sequence', 234)
    stub._hp_row_id = 'vqarad_69'
    assert cost(MID, 310) == 310
    with pytest.raises(ValueError, match='per-row'):
        cost(MID, 300)                                                     # another row's length is refused
    stub._hp_row_id = None
    with pytest.raises(ValueError):
        cost(MID, 300)


def test_pilot_input_is_authenticated_by_canonical_identity_after_a_real_durable_publish(tmp_path):
    from mri_rl.shared_rollout_gspo_core_v1 import publish_once, file_sha256, identity
    value = {'midline_train_PRIVATE': train_rows(), 'n': 1}
    p = publish_once(tmp_path / 'pilot_input_PRIVATE.json', value)
    assert file_sha256(p) != identity(value)                                   # the old native check would always STOP
    assert hp.authenticated_input(p, identity(value)) == json.loads(json.dumps(value))
    with pytest.raises(ValueError, match='canonical identity'):
        hp.authenticated_input(p, identity({**value, 'n': 2}))


# ================================================================ entry: measurement binding, fresh grant, EP-row authentication
def entry_mod():
    from mri_rl import heterogeneous_mri_pilot_entry_v1 as entry
    return entry


def test_prepare_refuses_scope_without_measured_midline_lengths_and_binds_measurement_to_rows(tmp_path):
    entry = entry_mod()
    train, ev = train_rows(), [r for r in midline_rows() if r['extra_info']['split'] == 'eval']
    gate = hp.gate_controls(train, borders())
    lengths, status = entry.measured_lengths(None, train, gate, ev)
    assert status == 'PENDING_PROCESSOR_MEASUREMENT' and lengths['sequence'] == 233 and len(lengths['midline_gate']) == 6 and len(lengths['by_row']) == 8
    with pytest.raises(ValueError, match='PENDING_PROCESSOR_MEASUREMENT'):
        hp.budget({k: lengths[k] for k in ('sequence', 'midline_train', 'midline_gate', 'midline_eval')})
    with pytest.raises(ValueError, match='Measured budget'):
        entry.phase_scope({'budget': None, 'output': '/x'}, 'parent')
    rows = entry.measured_rows(train, gate, ev)
    m = {'schema': entry.MEASUREMENT_SCHEMA, 'model_calls': 0, 'mlx_imported': False, 'rows_identity': entry.rows_identity(rows),
         'midline_suffix_sha256': hp.MIDLINE_SUFFIX_SHA256, 'by_row': {i: 300 + n for n, i in enumerate(sorted(rows))},
         'sequence_concise_check': {'expected': 233, 'actual': [233, 233]}}
    good = tmp_path / 'm.json'; good.write_text(json.dumps(m))
    measured, status = entry.measured_lengths(good, train, gate, ev)
    assert status == 'MEASURED' and measured['midline_gate']['TRUE_0'] == measured['midline_train'][train[0]['id']]
    for bad in ({**m, 'rows_identity': '0' * 64}, {**m, 'sequence_concise_check': {'expected': 233, 'actual': [233, 234]}}, {**m, 'mlx_imported': True}):
        good.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            entry.measured_lengths(good, train, gate, ev)


def human(entry, sha, issued, expires, scope):
    return {'authorized': True, 'action': entry.ACTION, 'scope_sha256': entry.digest(scope), 'included_phases': list(entry.PHASES),
            'human_instruction_sha256': sha, 'approved_by': 'researcher', 'issued_at_utc': issued.isoformat(), 'expires_at_utc': expires.isoformat()}


def test_fresh_grant_must_differ_from_consumed_history_cover_the_flow_and_be_consumed_once(tmp_path, monkeypatch):
    from datetime import datetime, timedelta, timezone
    from mri_rl.shared_rollout_gspo_core_v1 import consume_once
    entry = entry_mod()
    scope = {'schema': entry.SCHEMA, 'scope_kind': entry.KIND, 'action': entry.ACTION, 'implemented_by': entry.IMPLEMENTED_BY,
             'resource_policy': entry.MODEL_POLICY, 'external_guardian_policy': entry.GUARDIAN_POLICY, 'run_limits': entry.RUN_LIMITS,
             'windows_seconds': entry.WINDOWS, 'budget': {'backward_calls_max': 16}, 'automatic_retry': False, 'source_sha256': {}, 'file_sha256': {},
             'input_sha256': '1' * 64}
    review = {'verdict': 'PASS', 'scope_sha256': entry.digest(scope), 'source_sha256': {}, 'input_sha256': '1' * 64, 'reviewer_kind': 'isolated_codex', 'reviewer_id': 'codex'}
    now = datetime.now(timezone.utc)
    ok = human(entry, 'a' * 64, now - timedelta(minutes=1), now + timedelta(hours=3), scope)
    ledger = consume_once(tmp_path / 'ledger')
    with pytest.raises(ValueError, match='earlier scope'):
        entry.authorize(scope, ok, review, consumed=ledger, history={'a' * 64})
    with pytest.raises(ValueError, match='whole bounded flow'):
        entry.authorize(scope, {**ok, 'expires_at_utc': (now + timedelta(minutes=30)).isoformat()}, review, consumed=ledger, history=set())
    with pytest.raises(ValueError, match='whole bounded flow'):
        entry.authorize(scope, {**ok, 'issued_at_utc': (now - timedelta(minutes=31)).isoformat()}, review, consumed=ledger, history=set())
    with pytest.raises(ValueError, match='non-Claude'):
        entry.authorize(scope, ok, {**review, 'reviewer_id': 'claude-subagent'}, consumed=ledger, history=set())
    token = entry.authorize(scope, ok, review, consumed=ledger, history=set())
    assert token.sha == entry.digest(scope)
    with pytest.raises(ValueError, match='already consumed'):
        entry.authorize(scope, ok, review, consumed=ledger, history=set())
    (tmp_path / 'local' / 'x' / 'y').mkdir(parents=True)
    (tmp_path / 'local' / 'x' / 'y' / ('authorization_consumed_' + 'b' * 64 + '.json')).write_text('{}')
    assert entry.historical_grants(tmp_path / 'local') == {'b' * 64}


def test_phase_window_is_enforced_including_waits(tmp_path):
    from datetime import datetime, timedelta, timezone
    entry = entry_mod()
    scope = {'budget': {'backward_calls_max': 16}, 'output': str(tmp_path / 'run'), 'source_sha256': {}, 'file_sha256': {}}
    compound = entry.Authorization(entry._ISSUER, scope, datetime.now(timezone.utc) + timedelta(hours=2))
    (tmp_path / 'run').mkdir()
    (tmp_path / 'run' / 'parent_clock_PRIVATE.json').write_text(json.dumps({'started_monotonic': 1000.}))
    phase = compound.for_phase('parent')
    compound.start = 0.
    assert phase.check('x', clock=lambda: 1000. + entry.WINDOWS['parent'] - 1)
    with pytest.raises(ValueError, match='Phase window'):
        phase.check('x', clock=lambda: 1000. + entry.WINDOWS['parent'] + 1)
    with pytest.raises(ValueError, match='Global window'):
        compound.check('x', clock=lambda: entry.RUN_LIMITS['global_window_seconds'] + 1)
    waits = []
    compound.start = __import__('time').monotonic()
    with pytest.raises(ValueError, match='Phase window exhausted while waiting'):
        entry.wait_nonresident(compound, 0., 10., lambda: waits.append(1) or {'action': 'WAIT', 'reasons': []}, sleep=lambda s: None,
                               clock=iter([0., 5., 11.] + [11.] * 10).__next__)
    assert len(waits) == 2                                                       # WAIT samples consumed the window; no model attempt


def test_copied_parent_dev24_rows_are_authenticated_through_committed_journal_records(tmp_path):
    from mri_rl.safe_micro_journal_v1 import Journal
    from mri_rl.shared_rollout_gspo_core_v1 import publish_once, file_sha256
    entry = entry_mod()
    ep = tmp_path / 'stageEP_PRIVATE'; (ep / 'rows_PRIVATE').mkdir(parents=True)
    binding = {'scope': 'c' * 64, 'stage': 'EP', 'input': 'd' * 64}
    ids = ['r0', 'r1']
    with Journal(ep / 'v2_journal_PRIVATE', binding) as journal:
        for i, rid in enumerate(ids):
            key = 'Parent_row_%02d' % i

            def op(opdir, key=key, rid=rid):
                item = {'key': key, 'id': rid, 'condition': 'concise', 'truth': 'FLAIR', 'text': boxed('FLAIR'), 'tokens': 9}
                p = publish_once(ep / 'rows_PRIVATE' / (key + '_PRIVATE.json'), item)
                return {'value': {'item': item}, 'files': {str(p.resolve()): file_sha256(p)}}
            journal.run(key, op)
    rows, files = entry.authenticate_parent_rows(ep, binding, ids)
    assert [r['id'] for r in rows] == ids and all(r['task'] == 'mri_sequence' for r in rows) and len(files) == 6
    with pytest.raises(ValueError, match='binding'):
        entry.authenticate_parent_rows(ep, {**binding, 'scope': 'e' * 64}, ids)
    with pytest.raises(ValueError, match='r9|differs'):
        entry.authenticate_parent_rows(ep, binding, ['r0', 'r9'])
    row = ep / 'rows_PRIVATE' / 'Parent_row_01_PRIVATE.json'
    row.chmod(0o644); row.write_text(row.read_text().replace('FLAIR', 'contrast-enhanced T1', 1))
    with pytest.raises(ValueError, match='changed'):
        entry.authenticate_parent_rows(ep, binding, ids)


SOURCE = Path(__file__).resolve().parents[3] / 'local_data/two_path_clinical_dataset_v1/vqarad_original_PUBLIC_SOURCE_PRIVATE.json'


@pytest.mark.skipif(not SOURCE.is_file(), reason='public VQA-RAD source JSON not present in this checkout')
def test_actual_source_census_matches_the_independent_oracle_counts():
    c = hp.census(json.loads(SOURCE.read_text()))
    assert (c['midline_yesno_head_questions'], c['evaluated'], c['mass_shift_ambiguous_questions']) == (14, 11, 2)
    assert c['eligible_case_components'] == {'yes': 3, 'no': 2, 'conflict': 0} and c['decision'] == 'DATA_TOO_SMALL_FOR_LEARNING_CLAIM'
    assert c['MRI_components_including_not_evaluated_NOT_ADMITTED'] == {'yes': 4, 'no': 4, 'conflict': 0}     # reported, never promoted


def test_census_cli_prints_counts_only(tmp_path):
    path = tmp_path / 'qa.json'
    path.write_text(json.dumps(census_rows()))
    out = subprocess.run([sys.executable, str(SRC.parent / 'scripts/run_heterogeneous_mri_pilot_v1.py'), '--census', str(path)],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    printed = json.loads(out.stdout)
    assert printed['decision'] == 'DATA_TOO_SMALL_FOR_LEARNING_CLAIM'
    assert 'midline shift' not in out.stdout.lower() and 'm0.jpg' not in out.stdout                  # no question text / image names


# ================================================================ isolation
def test_cpu_and_native_modules_import_no_heavy_runtime():
    code = ('import sys\nimport mri_rl.heterogeneous_mri_pilot_v1, mri_rl.heterogeneous_mri_pilot_native_v1, mri_rl.heterogeneous_mri_pilot_entry_v1\n'
            'bad=[m for m in sys.modules if m.split(".")[0] in ("mlx","mlx_vlm","mlx_lm","torch","jax","transformers")]\n'
            'assert not bad, bad\nprint("OK")\n')
    out = subprocess.run([sys.executable, '-c', code], env={**os.environ, 'PYTHONPATH': str(SRC)}, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0 and out.stdout.strip() == 'OK', out.stderr


def test_segment_teacher_forced_cap_uses_exact_frozen_per_row_lengths():
    units = [sequence_unit(), midline_unit()]
    rows = {c['row']['id']: 233 for c in units[0]['contexts']}
    rows.update({'vqarad_52': 405, 'vqarad_69': 460})
    limits = hp.segment_limits(units, {'mri_sequence': 233, MID: 460}, rows)
    assert limits['teacherforced_input_positions'] == 2 * (8 * (233 + 255) + 4 * (405 + 255) + 4 * (460 + 255)) and limits['optimizer_updates'] == 1
    assert limits['backward_calls'] == 16 and limits['detached_old_forwards'] == 16
    with pytest.raises(ValueError, match='per-row'):
        hp.segment_limits(units, {'mri_sequence': 233, MID: 460}, {'vqarad_52': 405})


# ================================================================ supplement: traces, success-only read-only integrity, runtime admission
class TraceFake(HFake):
    def greedy(self, row, image_sha256):
        out = super().greedy(row, image_sha256)
        return {**out, 'actual_PRIVATE': {'image_sha256': image_sha256, 'prompt_sha256': 'p' * 64, 'F32_evidence_PRIVATE': ['f32'],
                                          'input_fingerprints_PRIVATE': {'row': row['id']}}}


def test_gate_and_parent_eval_records_keep_the_actual_f32_image_prompt_trace_without_extra_calls(tmp_path):
    backend = TraceFake(MIXED, greedy=following(border_follows=False))
    parent_phase(tmp_path, backend)
    gate = json.loads((tmp_path / 'parent' / 'gate_PRIVATE.json').read_text())['controls']
    ev = json.loads((tmp_path / 'parent' / 'parent_midline_eval_PRIVATE.json').read_text())['items']
    assert all(r['actual_PRIVATE']['image_sha256'] == r['image_sha256'] and r['actual_PRIVATE']['F32_evidence_PRIVATE'] for r in gate)
    assert all(r['actual_PRIVATE']['prompt_sha256'] == 'p' * 64 for r in ev) and len(backend.greedy_calls) == 8
    assert hp.evaluation_summary(seq_eval(10, 8) + ev)['tasks'][MID]['items'] == 2          # scored records unchanged by the trace


def test_native_greedy_returns_the_generate_one_trace():
    from mri_rl import heterogeneous_mri_pilot_native_v1 as native
    actual = {'response': {'text': 't', 'token_ids': [1]}, 'image_sha256': 's', 'prompt_sha256': 'p', 'prompt_length': 5,
              'input_fingerprints_PRIVATE': {'a': 1}, 'F32_evidence_PRIVATE': ['e']}
    out = native.greedy_record(actual)
    assert out['text'] == 't' and out['token_ids'] == [1] and out['actual_PRIVATE'] == {k: v for k, v in actual.items() if k != 'response'}


class RO:
    def __init__(self):
        self.events, self.state = [], 'R0'

    def save_rng(self, d):
        self.events.append('save_rng'); (Path(d) / 'rng.json').write_text(self.state)

    def load_rng(self, d):
        self.events.append('load_rng'); self.state = (Path(d) / 'rng.json').read_text()

    def rng_digest(self):
        return self.state

    def restore(self, *a):
        self.events.append('restore')


def test_guarded_readonly_checks_integrity_and_restores_rng_only_on_success(tmp_path):
    from mri_rl import heterogeneous_mri_pilot_native_v1 as native
    b = RO()
    integrity = lambda backend: backend.events.append('fingerprint') or {'params': 1, 'adam': 2}

    def run():
        b.state = 'R1'; return ['rows']
    assert native.guarded_readonly(b, run, tmp_path / 'ok', integrity=integrity) == ['rows']
    assert b.events == ['fingerprint', 'save_rng', 'fingerprint', 'load_rng'] and b.state == 'R0'
    assert json.loads((tmp_path / 'ok' / 'readonly_integrity_PRIVATE.json').read_text())['unchanged'] is True


@pytest.mark.parametrize('error', ['pause', 'fatal', 'other'])
def test_guarded_readonly_after_emergency_records_tainted_stop_with_no_restore_or_fingerprint(tmp_path, error):
    from mri_rl import heterogeneous_mri_pilot_native_v1 as native
    from mri_rl.safe_micro_dynamic_resources_v1 import ResourcePause, ResourceFatal
    exc = {'pause': ResourcePause('RED'), 'fatal': ResourceFatal('STOP'), 'other': RuntimeError('x')}[error]
    b = RO()
    integrity = lambda backend: backend.events.append('fingerprint') or {}
    runs = []

    def run():
        runs.append(1); raise exc
    with pytest.raises(type(exc)):
        native.guarded_readonly(b, run, tmp_path / 'e', integrity=integrity)
    assert b.events == ['fingerprint', 'save_rng'] and runs == [1]                          # nothing GPU-side after the failure, no retry
    stop = json.loads((tmp_path / 'e' / 'readonly_STOP_PRIVATE.json').read_text())
    assert stop['resident_state'] == 'TAINTED_RELEASE_PROCESS' and stop['in_process_restore'] is False and stop['integrity_after'] == 'NOT_RUN'


def test_pre_native_runtime_check_reuses_installed_fingerprint_and_certified_cache_shards(tmp_path):
    entry = entry_mod()
    from mri_rl.safe_micro_recovery_entry_v1 import SHARDS
    (tmp_path / 'cert.json').write_text('{}')
    (tmp_path / 'cache_scope.json').write_text(json.dumps({'installed': {'mlx_version': '1'}}))
    v2 = {'recovery_spec': str(tmp_path / 'rspec.json'), 'runtime_spec': str(tmp_path / 'rt.json')}
    (tmp_path / 'rspec.json').write_text(json.dumps({'cache_scope': str(tmp_path / 'cache_scope.json')}))
    (tmp_path / 'rt.json').write_text(json.dumps({'cache_dir': str(tmp_path), 'cache_certificate': str(tmp_path / 'cert.json'), 'model': str(tmp_path)}))
    good = {'shards': {n: {'sha256': h} for n, h in SHARDS.items()}}
    out = entry.pre_native_runtime_check(v2, installed=lambda: {'mlx_version': '1'}, binding=lambda ns: good)
    assert out['installed_matches_cache_scope'] is True and out['shards'] == SHARDS and out['quantization_or_cache_rebuild'] is False
    with pytest.raises(ValueError, match='runtime'):
        entry.pre_native_runtime_check(v2, installed=lambda: {'mlx_version': '2'}, binding=lambda ns: good)
    with pytest.raises(ValueError, match='Cache payload'):
        entry.pre_native_runtime_check(v2, installed=lambda: {'mlx_version': '1'},
                                       binding=lambda ns: {'shards': {n: {'sha256': '0' * 64} for n in SHARDS}})


def test_readonly_red_after_last_response_blocks_postrun_gpu_fingerprints(tmp_path):
    from mri_rl import heterogeneous_mri_pilot_native_v1 as native
    from mri_rl.safe_micro_dynamic_resources_v1 import ResourceFatal
    b = RO()
    def guard(name):
        if name == 'readonly_after_run_before_integrity':
            raise ResourceFatal('RED after final response')
    def integrity(backend):
        backend.events.append('fingerprint'); return {}
    with pytest.raises(ResourceFatal, match='RED'):
        native.guarded_readonly(b, lambda: ['last row'], tmp_path / 'red_after', integrity=integrity, before=guard)
    assert b.events == ['fingerprint', 'save_rng']


def test_parent_red_after_last_eval_blocks_rng_reload_and_training(tmp_path):
    from mri_rl.safe_micro_dynamic_resources_v1 import ResourceFatal
    b = TraceFake(MIXED, greedy=following(border_follows=False))
    b.reloads = 0
    old = b.load_rng
    def reload(path):
        b.reloads += 1; return old(path)
    b.load_rng = reload
    def guard(name):
        if name == 'hetero_before_rollout_rng_reload':
            raise ResourceFatal('RED after Parent EVAL')
    with pytest.raises(ResourceFatal, match='RED'):
        hp.run_parent_phase(backend=b, seq_unit=sequence_unit(), midline_train=[r for r in midline_rows() if r['extra_info']['split']=='train'],
                            midline_eval=[r for r in midline_rows() if r['extra_info']['split']=='eval'], borders=borders(), lengths=LENGTHS,
                            reward_policy_sha256=REWARD, output=tmp_path/'parent', before=guard, expected_parent=parent_state(b),
                            parent_receipt_sha256=REWARD)
    assert b.reloads == 0 and b.apply_calls == 0 and not (tmp_path / 'parent' / 'segment').exists()
