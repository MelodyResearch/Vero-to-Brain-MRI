"""CPU tests for heterogeneous_mri_pilot_v2. SYNTHETIC only: no MLX/torch/transformers, no model, no private MRI payload."""
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from test_multistep_gspo_v1 import make_plan, REWARD                          # noqa: E402
from test_safe_micro_training_v1 import MicroFake, MIXED, DALLAS_INACTIVE     # noqa: E402
from mri_rl import multistep_gspo_core_v1 as ms                              # noqa: E402
from mri_rl import heterogeneous_mri_pilot_v1 as v1                          # noqa: E402
from mri_rl import heterogeneous_mri_pilot_v2 as h2                          # noqa: E402

MID, SEQ = h2.MIDLINE_ABILITY, h2.SEQ_TASK
LENGTHS = {SEQ: 233, MID: 470}
OTHER = {'FLAIR': 'contrast-enhanced T1', 'contrast-enhanced T1': 'FLAIR', 'yes': 'no', 'no': 'yes'}


def boxed(answer):
    return f'<think>x</think><answer>\\boxed{{{answer}}}</answer>'


def records():
    out = []
    for qid, answer, split in ((52, 'no', 'train'), (69, 'yes', 'train'), (1037, 'yes', 'eval'), (2236, 'no', 'eval')):
        q = f'q{qid} midline shift?'
        out.append({'source_qid': str(qid), 'split': split, 'question': q, 'ground_truth': answer, 'source_image_sha256': f'{qid:064d}',
                    'case_proxy_sha256': f'{qid + 7:064d}', 'model_input': {'image': f'/x/img{qid}.jpg', 'question': q}})
    return out


def rows(split):
    return [h2.midline_row(r) for r in records() if r['split'] == split]


def borders():
    train = rows('train')
    rec = {'schema': 'heterogeneous-central-occlusion-controls-v1', 'model_outputs_used': False,
           'controls': [{'source_qid': r['extra_info']['source_qid'], 'control_path': f'/x/occ{i}.png', 'control_sha256': f'{900 + i:064d}',
                         'source_sha256': r['image_sha256'], 'control_kind': v1.BORDER_KIND, 'question_original_unchanged': r['extra_info']['original_question']}
                        for i, r in enumerate(train)]}
    return v1.border_controls(rec, train)


def seq_unit():
    return ms.make_units(make_plan(1))[0]


def mid_unit():
    return v1.midline_unit(rows('train'))


class Fake(MicroFake):
    """Scripted grades AND matching strict texts (unless `texts` overrides a response). Greedy answers are scripted by (question, image)."""
    def __init__(self, script=MIXED, greedy=None, texts=None, gate_text=None, nonfinite=False):
        super().__init__(script)
        self.greedy_answers, self.texts, self.gate_text, self.nonfinite = greedy or {}, texts or {}, gate_text, nonfinite
        self.greedy_calls, self.restores, self.gradient_calls = [], [], []

    def generate(self, unit, prompt_lengths):
        raw = super().generate(unit, prompt_lengths)
        truth = {c['key']: c['row']['reward_model']['ground_truth'] for c in unit['contexts']}
        for r in raw['responses']:
            t = truth[r['context_key']]
            r['text'] = self.texts.get(r['response_id'], boxed(t if self.script[r['response_id']] else OTHER[t]))
            r['finish_reason'] = 'stop'
        return raw

    def gradient(self, planned, own_old, unit):
        self.gradient_calls.append(unit['index'])
        tree, info = super().gradient(planned, own_old, unit)
        if self.nonfinite and tree is not None:
            tree = tree * np.float32('nan')
        return tree, info

    def restore(self, snap):
        self.restores.append(1)
        return super().restore(snap)

    def greedy(self, row, image_sha256):
        self.greedy_calls.append((row['id'], image_sha256))
        q = row['extra_info']['question']
        text = self.gate_text(row) if self.gate_text else boxed(self.greedy_answers.get((q, image_sha256), 'yes'))
        return {'text': text, 'token_ids': [5, 6, 99], 'finish_reason': 'stop', 'EOS_coverage': 'OBSERVED_AND_SCORED',
                'actual_PRIVATE': {'F32_evidence_PRIVATE': ['f32'], 'input_fingerprints_PRIVATE': {'id': row['id']}, 'prompt_sha256': 'p' * 64}}


def parent_state(b):
    return {'policy': b.policy_sha(), 'adam': b.adam_digest(), 'rng': b.rng_digest()}


def train(tmp_path, backend, name='train', before=None):
    return h2.run_train_phase(backend=backend, seq_unit=seq_unit(), midline_train=rows('train'), borders=borders(), lengths=LENGTHS,
                              reward_policy_sha256=REWARD, output=tmp_path / name, before=before or (lambda n: None),
                              expected_parent=parent_state(backend), parent_receipt_sha256=REWARD)


# ================================================================ frozen suffix / rows
def test_v2_suffix_is_one_frozen_symmetric_boxed_full_grammar_with_literal_backslash():
    s = h2.MIDLINE_SUFFIX
    assert '\b' not in s and s.count('\\boxed{yes}') == 1 and s.count('\\boxed{no}') == 1
    assert s.count('<think>') == s.count('</think>') == s.count('<answer>') == s.count('</answer>') == 2
    assert 'image' in s and 'one short sentence' in s and h2.MIDLINE_SUFFIX != v1.MIDLINE_SUFFIX
    r = rows('train')[0]
    assert r['extra_info']['question'] == 'q52 midline shift?\n\n' + s and r['extra_info']['original_question'] == 'q52 midline shift?'
    assert v1.midline_row(records()[0])['extra_info']['question'].endswith(v1.MIDLINE_SUFFIX)          # v1 unchanged


@pytest.mark.parametrize('text,truth,expected', [
    (boxed('yes'), 'yes', 'CORRECT'), (boxed('no'), 'yes', 'WRONG'), (boxed('No'), 'no', 'CORRECT'),
    ('<think>x</think><answer>yes</answer>', 'yes', 'INVALID'),                         # unboxed: v1 failure, never repaired
    ('<think>x</think><answer>\\boxed{yes} \\boxed{no}</answer>', 'yes', 'INVALID'),   # multiple labels
    ('<think>x</think><answer>\\boxed{maybe}</answer>', 'no', 'INVALID'),
    (boxed('FLAIR'), 'FLAIR', 'CORRECT'), (boxed('FLAIR'), 'contrast-enhanced T1', 'WRONG'), ('<think>x</think><answer>FLAIR</answer>', 'FLAIR', 'INVALID'),
])
def test_strict_outcome_uses_the_unchanged_parsers_without_repair(text, truth, expected):
    task = SEQ if truth in ('FLAIR', 'contrast-enhanced T1') else MID
    assert h2.strict_outcome(task, text, truth) == expected


def test_suffix_examples_in_the_prompt_are_not_parsed_as_an_answer():
    assert h2.strict_outcome(MID, h2.MIDLINE_SUFFIX, 'yes') == 'INVALID'


# ================================================================ Phase 1: TRUE format gate; SWAP/BORDER never veto
def test_true_format_failure_is_format_blocked_before_any_on_policy_work(tmp_path):
    b = Fake(gate_text=lambda row: '<think>x</think><answer>yes</answer>')               # exactly the v1 failure shape
    out = train(tmp_path, b)
    assert out['status'] == 'FORMAT_BLOCKED' and out['format_gate']['invalid_true_controls'] == ['TRUE_0', 'TRUE_1']
    assert b.generate_calls == 0 and b.gradient_calls == [] and b.apply_calls == 0 and len(b.greedy_calls) == 6
    assert out['on_policy_responses'] == 0 and out['group_variance'] == 'NOT_APPLICABLE_NO_SAMPLED_GROUP'


def test_greedy_question_following_is_recorded_but_is_not_a_hard_gate(tmp_path):
    b = Fake()                                                                            # every control answers boxed 'yes'
    out = train(tmp_path, b)
    diag = out['format_gate']['diagnostic_v1_image_dependence']
    assert diag['verdict'] == 'NO_SIGNAL' and diag['use'] == 'DIAGNOSTIC_ONLY_NEVER_A_VETO'
    assert out['status'] == 'UPDATED' and b.generate_calls == 2 and b.apply_calls == 1


def test_invalid_border_and_swap_controls_never_veto(tmp_path):
    b = Fake(gate_text=lambda row: boxed('no') if row['id'].startswith('vqarad_') and '__' not in row['id'] else 'garbage')
    out = train(tmp_path, b)
    assert out['format_gate']['answers']['BORDER_0'] is None and out['format_gate']['answers']['SWAP_1'] is None
    assert out['format_gate']['status'] == 'FORMAT_QUALIFIED' and out['status'] == 'UPDATED'


def test_controls_persist_token_ids_finish_reason_and_trace(tmp_path):
    train(tmp_path, Fake())
    recs = json.loads((tmp_path / 'train' / 'controls_pre_PRIVATE.json').read_text())['controls']
    assert [r['name'] for r in recs] == list(h2.GATE_NAMES)
    assert all(r['token_ids'] == [5, 6, 99] and r['finish_reason'] == 'stop' and r['actual_PRIVATE']['F32_evidence_PRIVATE'] for r in recs)


# ================================================================ Phase 2: class-contrast two-task gate
def plan_record(ids_by_group, admitted):
    return {'groups': ids_by_group, 'items': [{'response_id': i, 'group': g, 'admitted': admitted[g]} for g, ids in enumerate(ids_by_group) for i in ids]}


def test_class_contrast_requires_admitted_groups_with_strict_valid_correct_and_wrong():
    rec = plan_record([['a0', 'a1', 'a2', 'a3'], ['b0', 'b1', 'b2', 'b3']], [True, True])
    ok = h2.class_contrast(rec, {'a0': 'CORRECT', 'a1': 'WRONG', 'a2': 'INVALID', 'a3': 'WRONG', 'b0': 'CORRECT', 'b1': 'CORRECT', 'b2': 'INVALID', 'b3': 'CORRECT'})
    assert ok['admitted_groups'] == 2 and ok['groups_with_class_contrast'] == 1 and ok['groups'][0]['invalid'] == 1
    format_only = h2.class_contrast(rec, {i: ('WRONG' if i.endswith(('0', '1')) else 'INVALID') for g in rec['groups'] for i in g})
    assert format_only['groups_with_class_contrast'] == 0                                          # reward varies by format only
    not_admitted = h2.class_contrast(plan_record(rec['groups'], [False, False]), {i: 'CORRECT' if i.endswith('0') else 'WRONG' for g in rec['groups'] for i in g})
    assert not_admitted['admitted_groups'] == 0 and not_admitted['groups_with_class_contrast'] == 0


def run_segment(tmp_path, backend, before=None):
    return h2.train_segment(backend=backend, seq_unit=seq_unit(), mid_unit=mid_unit(), lengths=LENGTHS, reward_policy_sha256=REWARD,
                            output=tmp_path / 'seg', before=before or (lambda n: None), expected_parent=parent_state(backend), parent_receipt_sha256=REWARD)


def test_both_tasks_with_class_contrast_make_exactly_one_half_half_joint_step_and_keep_all_16(tmp_path):
    captured = {}

    class Capture(Fake):
        def apply_update(self, gradient, lr=.05):
            captured['joint'] = np.array(gradient, dtype=np.float64)
            return super().apply_update(gradient, lr)

        def gradient(self, planned, own_old, unit):
            tree, info = super().gradient(planned, own_old, unit)
            captured.setdefault('task', []).append(np.array(tree, dtype=np.float64))
            return tree, info
    b = Capture(texts={'T3': 'unboxed T1'})                                              # an invalid response stays in the bundle
    r = run_segment(tmp_path, b)
    assert r['updated'] is True and b.apply_calls == 1 and b.step == 53 and r['responses'] == {SEQ: 8, MID: 8}
    assert r['strict_outcome_counts'][SEQ]['INVALID'] == 1 and r['task_weight'] == {SEQ: .5, MID: .5}
    assert np.allclose(captured['joint'], .5 * captured['task'][0] + .5 * captured['task'][1], atol=1e-6)
    raw = json.loads((tmp_path / 'seg' / f'rollout_raw_{SEQ}_PRIVATE.json').read_text())
    assert len(raw['responses']) == 8 and all('token_ids' in x and x['finish_reason'] == 'stop' and 'text' in x for x in raw['responses'])
    plans = json.loads((tmp_path / 'seg' / 'inputs_PRIVATE.json').read_text())['plans']
    assert plans[SEQ]['groups'] == [['F0', 'F1', 'F2', 'F3'], ['T0', 'T1', 'T2', 'T3']] and plans[MID]['groups'][0][0] == 'D1_0'     # same-prompt K4


SEQ_INACTIVE = {**MIXED, **{f'F{i}': 1 for i in range(4)}, **{f'T{i}': 0 for i in range(4)}}
MID_FORMAT_ONLY = {f'D{k}_{i}': 'unboxed' for k in (1, 2) for i in range(4) if MIXED[f'D{k}_{i}'] == 0}       # wrong answers unboxed


@pytest.mark.parametrize('script,texts,missing', [(DALLAS_INACTIVE, {}, [MID]), (SEQ_INACTIVE, {}, [SEQ]),
                                                  ({**DALLAS_INACTIVE, **{f'F{i}': 1 for i in range(4)}, **{f'T{i}': 0 for i in range(4)}}, {}, [SEQ, MID]),
                                                  (MIXED, MID_FORMAT_ONLY, [MID])])
def test_missing_class_contrast_in_either_task_is_no_gspo_signal_before_any_backward(tmp_path, script, texts, missing):
    b = Fake(script, texts=texts)
    r = run_segment(tmp_path, b)
    assert r['decision'] == {'update': False, 'reason': 'NO_GSPO_SIGNAL', 'tasks_without_class_contrast': missing}
    assert b.gradient_calls == [] and b.apply_calls == 0 and b.step == 52 and r['backward_calls'] == 0 and not (tmp_path / 'seg' / 'checkpoint_PRIVATE').exists()


def test_nonfinite_gradient_is_no_gspo_signal_with_zero_updates(tmp_path):
    b = Fake(nonfinite=True)
    r = run_segment(tmp_path, b)
    assert r['decision']['reason'] == 'NO_GSPO_SIGNAL' and r['decision']['gradient_problem'] == 'NONFINITE'
    assert b.apply_calls == 0 and b.step == 52 and r['updated'] is False


@pytest.mark.parametrize('boundary', ['hetero_v2_gradient_' + SEQ, 'hetero_v2_gradient_' + MID, 'hetero_v2_before_update', 'hetero_v2_checkpoint'])
def test_resource_error_stops_without_rollback_extra_backward_or_update(tmp_path, boundary):
    from mri_rl.safe_micro_dynamic_resources_v1 import ResourcePause

    def before(name):
        if name == boundary:
            raise ResourcePause('RED')
    b = Fake()
    with pytest.raises(ResourcePause):
        run_segment(tmp_path, b, before)
    stops = list((tmp_path / 'seg').glob('STOP_*_PRIVATE.json'))
    assert b.restores == [] and len(stops) == 1 and json.loads(stops[0].read_text())['in_process_rollback'] is False
    assert len(b.gradient_calls) == {'hetero_v2_gradient_' + SEQ: 0, 'hetero_v2_gradient_' + MID: 1}.get(boundary, 2)
    assert b.apply_calls == (1 if boundary == 'hetero_v2_checkpoint' else 0) and not (tmp_path / 'seg' / 'receipt_PRIVATE.json').exists()
    with pytest.raises(ValueError, match='never replayed'):
        run_segment(tmp_path, Fake())


def test_train_phase_is_one_shot(tmp_path):
    train(tmp_path, Fake())
    with pytest.raises(ValueError, match='never replayed'):
        train(tmp_path, Fake())


# ================================================================ Phase 3: same prompts after, outcome rule
def test_child_jobs_reuse_the_identical_six_control_prompts_and_images():
    controls = v1.gate_controls(rows('train'), borders())
    dev = [{'id': f'd{i}', 'image_sha256': f'{i:064d}', 'reward_model': {'ground_truth': 'FLAIR'}} for i in range(24)]
    jobs = h2.child_jobs(dev, rows('eval'), controls)
    assert len(jobs) == 32 and [j[0] for j in jobs[-6:]] == list(h2.GATE_NAMES)
    assert all(j[2] is c['row'] and j[3] == c['image_sha256'] for j, c in zip(jobs[-6:], controls))
    pre = [{'name': c['name'], 'id': c['row']['id'], 'image_sha256': c['image_sha256'], 'text': boxed('yes')} for c in controls]
    post = [{**p, 'text': boxed('no')} for p in pre]
    cmp = h2.control_comparison(pre, post)
    assert cmp['TRUE_0'] == {'pre': 'yes', 'post': 'no'} and cmp['use'] == 'DIAGNOSTIC_ONLY'
    with pytest.raises(ValueError, match='identical'):
        h2.control_comparison(pre, [{**post[0], 'image_sha256': '1' * 64}] + post[1:])


def seq_eval(flair_ok, t1_ok, invalid=0):
    out = [{'task': SEQ, 'id': f'f{i}', 'truth': 'FLAIR', 'tokens': 9, 'text': boxed('FLAIR' if i < flair_ok else 'contrast-enhanced T1')} for i in range(12)]
    out += [{'task': SEQ, 'id': f't{i}', 'truth': 'contrast-enhanced T1', 'tokens': 9,
             'text': 'bad' if i >= 12 - invalid else boxed('contrast-enhanced T1' if i < t1_ok else 'FLAIR')} for i in range(12)]
    return out


def summary(flair, t1, mid, invalid=0):
    items = seq_eval(flair, t1, invalid) + [{'task': MID, 'id': f'm{t}', 'truth': t, 'text': boxed(a) if a else 'none'} for t, a in zip(('yes', 'no'), mid)]
    return v1.evaluation_summary(items)


@pytest.mark.parametrize('child,result', [(lambda: summary(10, 8, ('yes', 'no')), 'FEASIBLE'), (lambda: summary(10, 8, ('yes', 'yes')), 'NO_ACCURACY_GAIN'),
                                          (lambda: summary(11, 7, ('yes', 'no')), 'NEGATIVE_TRANSFER'), (lambda: summary(10, 8, ('yes', None)), 'NEGATIVE_TRANSFER'),
                                          (lambda: summary(10, 8, ('yes', 'no'), invalid=1), 'NEGATIVE_TRANSFER')])
def test_outcome_after_a_committed_update(child, result):
    out = h2.outcome(summary(10, 8, ('yes', 'yes')), child(), status='UPDATED')
    assert out['outcome'] == 'REAL_PILOT_COMPLETED' and out['result'] == result and out['efficacy_or_generalization_claim'] is False


@pytest.mark.parametrize('status', ['FORMAT_BLOCKED', 'NO_GSPO_SIGNAL'])
def test_no_update_is_never_completed_learning(status):
    out = h2.outcome(None, None, status=status)
    assert out['outcome'] == status and out['completed_learning'] is False and out['evaluation'] == 'NOT_RUN_NO_COMMITTED_UPDATE'


# ================================================================ budget
def lengths_measured():
    return {'sequence': 233, 'midline_train': {'vqarad_52': 470, 'vqarad_69': 525}, 'midline_eval': {'vqarad_1037': 352, 'vqarad_2236': 208},
            'midline_gate': {'TRUE_0': 470, 'TRUE_1': 525, 'SWAP_0': 521, 'SWAP_1': 474, 'BORDER_0': 470, 'BORDER_1': 525}}


def test_budget_is_three_serial_loads_56_generations_14336_tokens_and_exact_tf():
    b = h2.budget(lengths_measured())
    assert b['model_loads_max'] == 3 and b['greedy_calls_max'] == 40 and b['generations_max'] == 56 and b['generated_tokens_max'] == 14336
    assert b['greedy_by_phase'] == {'train': 6, 'parent_eval': 2, 'child_eval': 32} and b['optimizer_updates_max'] == 1 and b['backward_calls_max'] == 16
    assert b['own_old_TF_positions_max'] == 8 * (233 + 255) + 4 * (470 + 255) + 4 * (525 + 255) and b['teacherforced_positions_total_max'] == 2 * b['own_old_TF_positions_max']
    with pytest.raises(ValueError, match='PENDING'):
        h2.budget({**lengths_measured(), 'midline_eval': {'vqarad_1037': None, 'vqarad_2236': 208}})
    units = [seq_unit(), mid_unit()]
    rl = {**{c['row']['id']: 233 for c in units[0]['contexts']}, 'vqarad_52': 470, 'vqarad_69': 525}
    assert v1.segment_limits(units, {SEQ: 233, MID: 525}, rl)['teacherforced_input_positions'] == b['teacherforced_positions_total_max']


# ================================================================ native / entry (CPU boundary)
def test_native_greedy_record_keeps_full_response_metadata():
    from mri_rl import heterogeneous_mri_pilot_native_v2 as n2
    actual = {'response': {'text': 't', 'token_ids': [1, 99], 'raw_selected_logps': [-.1, -.2], 'generation_tokens': 2, 'finish_reason': 'stop',
                           'EOS_coverage': 'OBSERVED_AND_SCORED'}, 'image_sha256': 's', 'prompt_sha256': 'p', 'prompt_length': 5,
              'input_fingerprints_PRIVATE': {'a': 1}, 'F32_evidence_PRIVATE': ['e']}
    out = n2.greedy_record_full(actual)
    assert out['token_ids'] == [1, 99] and out['finish_reason'] == 'stop' and out['EOS_coverage'] == 'OBSERVED_AND_SCORED'
    assert out['actual_PRIVATE']['F32_evidence_PRIVATE'] == ['e'] and out['actual_PRIVATE']['response']['raw_selected_logps'] == [-.1, -.2]


def entry():
    from mri_rl import heterogeneous_mri_pilot_entry_v2 as e2
    return e2


def test_measurement_binds_v2_rows_and_suffix(tmp_path):
    e2 = entry()
    tr, ev = rows('train'), rows('eval')
    gate = v1.gate_controls(tr, borders())
    lengths, status = e2.measured_lengths(None, tr, gate, ev)
    assert status == 'PENDING_PROCESSOR_MEASUREMENT' and len(lengths['by_row']) == 8
    rws = e2.measured_rows(tr, gate, ev)
    m = {'schema': e2.MEASUREMENT_SCHEMA, 'model_calls': 0, 'mlx_imported': False, 'rows_identity': e2.rows_identity(rws),
         'midline_suffix_sha256': h2.MIDLINE_SUFFIX_SHA256, 'by_row': {i: 400 + n for n, i in enumerate(sorted(rws))},
         'sequence_concise_check': {'expected': 233, 'actual': [233, 233]}, 'visible_prompt_leaks': 0}
    p = tmp_path / 'm.json'; p.write_text(json.dumps(m))
    assert e2.measured_lengths(p, tr, gate, ev)[1] == 'MEASURED'
    for bad in ({**m, 'midline_suffix_sha256': v1.MIDLINE_SUFFIX_SHA256}, {**m, 'visible_prompt_leaks': 1}, {**m, 'rows_identity': '0' * 64}):
        p.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            e2.measured_lengths(p, tr, gate, ev)


def test_visible_prompt_leak_check_flags_ids_paths_and_sites():
    e2 = entry()
    r = rows('train')[0]
    assert e2.prompt_leaks('system\nq52 midline shift?\n\n' + h2.MIDLINE_SUFFIX, r) == []
    assert e2.prompt_leaks('see /x/img52.jpg', r) and e2.prompt_leaks('row vqarad_52', r) and e2.prompt_leaks('img52', r)


def test_phase_schedule_is_conditional_and_loads_are_capped_at_three():
    e2 = entry()
    assert e2.next_phases('FORMAT_BLOCKED') == [] and e2.next_phases('NO_GSPO_SIGNAL') == [] and e2.next_phases('UPDATED') == ['parent_eval', 'child_eval']
    assert e2.RUN_LIMITS['model_loads_max'] == 3 and set(e2.WINDOWS) == set(h2.PHASES)
    assert all(e2.WINDOWS[p] > e2.WORST_CASE_SECONDS[p] for p in h2.PHASES) and e2.RUN_LIMITS['global_window_seconds'] >= sum(e2.WINDOWS.values())


def test_fresh_grant_no_borrow_no_replay_and_canonical_identity(tmp_path):
    from datetime import datetime, timedelta, timezone
    from mri_rl.shared_rollout_gspo_core_v1 import consume_once, publish_once, file_sha256, identity
    e2 = entry()
    scope = {'schema': e2.SCHEMA, 'scope_kind': e2.KIND, 'action': e2.ACTION, 'implemented_by': e2.IMPLEMENTED_BY, 'resource_policy': e2.MODEL_POLICY,
             'external_guardian_policy': e2.GUARDIAN_POLICY, 'run_limits': e2.RUN_LIMITS, 'windows_seconds': e2.WINDOWS, 'budget': {'backward_calls_max': 16},
             'automatic_retry': False, 'source_sha256': {}, 'file_sha256': {}, 'input_sha256': '1' * 64}
    review = {'verdict': 'PASS', 'scope_sha256': e2.digest(scope), 'source_sha256': {}, 'input_sha256': '1' * 64, 'reviewer_kind': 'isolated_codex', 'reviewer_id': 'codex'}
    now = datetime.now(timezone.utc)
    human = {'authorized': True, 'action': e2.ACTION, 'scope_sha256': e2.digest(scope), 'included_phases': list(h2.PHASES), 'human_instruction_sha256': 'a' * 64,
             'approved_by': 'researcher', 'issued_at_utc': (now - timedelta(minutes=1)).isoformat(), 'expires_at_utc': (now + timedelta(hours=3)).isoformat()}
    ledger = consume_once(tmp_path / 'ledger')
    with pytest.raises(ValueError, match='earlier scope'):
        e2.authorize(scope, human, review, consumed=ledger, history={'a' * 64})
    with pytest.raises(ValueError, match='Fresh explicit human'):
        e2.authorize(scope, {**human, 'action': 'heterogeneous_mri_pilot_v1_gate6_parent_midline2_rollout16_one_joint_update_terminal_DEV24_midline2_once'},
                     review, consumed=ledger, history=set())
    assert e2.authorize(scope, human, review, consumed=ledger, history=set()).sha == e2.digest(scope)
    with pytest.raises(ValueError, match='already consumed'):
        e2.authorize(scope, human, review, consumed=ledger, history=set())
    value = {'x': [1, 2]}
    p = publish_once(tmp_path / 'pilot_input_PRIVATE.json', value)
    assert file_sha256(p) != identity(value) and v1.authenticated_input(p, identity(value)) == value


def test_resource_gate_claims_at_most_three_loads(tmp_path):
    e2 = entry()
    scope = {'compound_scope_sha256': 'c' * 64}

    class Auth:
        def check(self, phase):
            return True
        parent = type('P', (), {'scope': {'output': str(tmp_path)}})()
    gate = e2.PilotResourceGateV2.__new__(e2.PilotResourceGateV2)
    gate.authorization, gate.scope = Auth(), scope
    for _ in range(3):
        gate.claim_model_load()
    with pytest.raises(ValueError, match='budget'):
        gate.claim_model_load()


def test_v2_modules_import_no_heavy_runtime():
    code = ('import sys\nimport mri_rl.heterogeneous_mri_pilot_v2, mri_rl.heterogeneous_mri_pilot_native_v2, mri_rl.heterogeneous_mri_pilot_entry_v2\n'
            'bad=[m for m in sys.modules if m.split(".")[0] in ("mlx","mlx_vlm","mlx_lm","torch","jax","transformers")]\nassert not bad, bad\nprint("OK")\n')
    out = subprocess.run([sys.executable, '-c', code], env={**os.environ, 'PYTHONPATH': str(SRC)}, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0 and out.stdout.strip() == 'OK', out.stderr


@pytest.mark.parametrize('boundary', ['hetero_segment_state_preflight', 'hetero_segment_snapshot'])
def test_red_blocks_segment_state_operations(tmp_path, boundary):
    from mri_rl.safe_micro_dynamic_resources_v1 import ResourceFatal
    b = Fake()
    def guard(name):
        if name == boundary:
            raise ResourceFatal('RED before state operation')
    with pytest.raises(ResourceFatal, match='RED'):
        run_segment(tmp_path, b, guard)
    assert not b.gradient_calls and b.apply_calls == 0


def test_red_after_controls_blocks_rng_reload(tmp_path):
    from mri_rl.safe_micro_dynamic_resources_v1 import ResourceFatal
    b = Fake()
    reloads = []
    original = b.load_rng
    def reload(path):
        reloads.append(1)
        return original(path)
    b.load_rng = reload
    def guard(name):
        if name == 'hetero_before_rollout_rng_reload':
            raise ResourceFatal('RED after final control')
    with pytest.raises(ResourceFatal, match='RED'):
        train(tmp_path, b, before=guard)
    assert not reloads and not b.gradient_calls and b.apply_calls == 0


def test_phase3_rejects_inner_receipt_without_committed_train(tmp_path):
    from mri_rl import heterogeneous_mri_pilot_native_v2 as n
    from mri_rl.shared_rollout_gspo_core_v1 import identity
    scope = {'output': str(tmp_path)}
    segment = tmp_path / 'train_PRIVATE/train_phase_PRIVATE/segment'
    segment.mkdir(parents=True)
    receipt = {'updated': True}
    receipt['receipt_sha256'] = identity(receipt)
    (segment / 'receipt_PRIVATE.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='completed|committed|terminal'):
        n.authenticate_evaluation_prerequisites(scope, 'parent_eval', scope_file_sha256='f' * 64)


def test_raw_capture_refuses_unknown_cleaner_and_marks_observed_no_cleaning():
    from types import SimpleNamespace
    from mri_rl import heterogeneous_mri_pilot_native_v2 as n
    response = {'text': boxed('yes'), 'token_ids': [1, 2]}
    n.mark_uncleaned_response(SimpleNamespace(processor=SimpleNamespace()), response)
    assert response['preclean_text_PRIVATE'] == boxed('yes') and response['clean_called'] is False
    with pytest.raises(ValueError, match='clean_output'):
        n.mark_uncleaned_response(SimpleNamespace(processor=SimpleNamespace(clean_output=lambda x: x)), response)


def test_native_readonly_call_carries_resource_guard():
    import ast
    tree = ast.parse((SRC / 'mri_rl/heterogeneous_mri_pilot_native_v2.py').read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'guarded_readonly']
    assert len(calls) == 1 and any(k.arg == 'before' and isinstance(k.value, ast.Name) and k.value.id == 'resource_gate' for k in calls[0].keywords)


def test_phase3_accepts_only_complete_journal_and_rejects_checkpoint_tamper(tmp_path, monkeypatch):
    from mri_rl import heterogeneous_mri_pilot_native_v2 as n
    from mri_rl.safe_micro_journal_v1 import Journal
    from mri_rl.shared_rollout_gspo_core_v1 import identity, file_sha256, publish_once
    e = entry()
    monkeypatch.setattr(e, 'phase_scope', lambda scope, phase: {'phase': phase})
    scope = {'output': str(tmp_path)}
    out = tmp_path / 'train_PRIVATE'
    segment = out / 'train_phase_PRIVATE/segment'
    segment.mkdir(parents=True)
    checkpoint = segment / 'adapter.fixture'
    checkpoint.write_bytes(b'saved CPU fixture')
    receipt = {'updated': True, 'adam_step_after': 53}
    receipt['receipt_sha256'] = identity(receipt)
    rp = segment / 'receipt_PRIVATE.json'
    publish_once(rp, receipt)
    phase_sha = e.digest({'phase': 'train'})
    training = {'status': 'UPDATED', 'optimizer_updates': 1, 'segment_receipt_sha256': receipt['receipt_sha256']}
    native = {**training, 'phase': 'train', 'release_verified': True}
    with Journal(out / 'journal_PRIVATE', {'scope': phase_sha, 'phase': 'train'}) as j:
        j.run('train_phase', lambda p: {'value': training, 'files': {str(rp): file_sha256(rp), str(checkpoint): file_sha256(checkpoint)}})
        j.run('native_complete', lambda p: {'value': native, 'files': {}})
    publish_once(out / 'terminal_PRIVATE.json', {**native, 'scope_sha256': phase_sha})
    exit_dir = tmp_path / 'train_exit_PRIVATE'
    exit_dir.mkdir()
    publish_once(exit_dir / 'actual_exit_PRIVATE.json', {'native_exit_code': 0, 'post_child_inspection_complete': True,
                 'SIGTERM_sent': False, 'SIGKILL_after_grace': False, 'monitor_log_errors': [], 'post_child_global_lock': {'no_open_holders': True},
                 'child_reaped': True, 'guardian_stop': False, 'scope_file_sha256': 'f' * 64})
    assert n.authenticate_evaluation_prerequisites(scope, 'parent_eval', scope_file_sha256='f' * 64) == receipt
    with pytest.raises(ValueError, match='terminal'):
        n.authenticate_evaluation_prerequisites(scope, 'child_eval', scope_file_sha256='f' * 64)
    wp = exit_dir / 'actual_exit_PRIVATE.json'
    witness = json.loads(wp.read_text())
    for key, bad in (('child_reaped', False), ('guardian_stop', True), ('scope_file_sha256', '0' * 64)):
        wp.write_text(json.dumps({**witness, key: bad}))
        with pytest.raises(ValueError, match='exit/release'):
            n.authenticate_evaluation_prerequisites(scope, 'parent_eval', scope_file_sha256='f' * 64)
    wp.write_text(json.dumps(witness))
    checkpoint.write_bytes(b'changed after commit')
    with pytest.raises(ValueError, match='Committed artifact'):
        n.authenticate_evaluation_prerequisites(scope, 'parent_eval', scope_file_sha256='f' * 64)


def test_real_backend_nonfinite_leaf_error_stops_instead_of_reporting_completed_signal(tmp_path):
    class NativeFailure(Fake):
        def gradient(self, *args):
            raise ValueError('Nonfinite native gradient leaf')
    b = NativeFailure()
    with pytest.raises(ValueError, match='Nonfinite native'):
        run_segment(tmp_path, b)
    assert b.apply_calls == 0 and list((tmp_path / 'seg').glob('STOP_*_PRIVATE.json'))
