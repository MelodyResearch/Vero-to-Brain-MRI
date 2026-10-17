"""Synthetic-row tests for the shared two-task training surface (no model, no MRI)."""
from collections import Counter
import tempfile
from pathlib import Path
import pytest
from mri_rl import two_task_training as tt
from mri_rl import visit_correspondence as vc


TMP = Path(tempfile.mkdtemp(prefix='ttt_'))


def _img(name):
    p = TMP / name
    p.write_bytes(b'PNG')
    return str(p)


def seq_row(i):
    img = _img(f's{i}.png')
    return {'id': f'seq{i}', 'ability': 'mri_sequence', 'data_source': tt.SEQ_SOURCE, 'images': [img],
            'prompt': [{'role': 'user', 'content': [{'type': 'image', 'image': img}, {'type': 'text', 'text': 'q'}]}],
            'reward_model': {'ground_truth': 'FLAIR' if i % 2 else 'contrast-enhanced T1'},
            'extra_info': {'reward_type': 'mri_sequence', 'internal_split': 'train', 'patient_lineage': 'BIDS_SUBJECT_VERIFIED',
                           'patient_key': f'p{i}', 'question': 'q'}}


def dal_row(i):
    img = _img(f'd{i}.png')
    h = lambda c: c * 64
    same = i % 2 == 0
    key = lambda n: f'{n:024x}'
    e = {'reward_type': 'mri_visit_correspondence', 'source_version': vc.SOURCE_VERSION, 'internal_split': 'train',
         'patient_lineage': 'BIDS_SUBJECT_VERIFIED', 'patient_key': key(i), 'candidate_patient_key': key(i) if same else key(i + 1000),
         'statistical_cluster_key': key(i + 5000), 'query_source_sha256': h('a'), 'candidate_source_sha256': h('b'),
         'metadata_plan_sha256': h('c'), 'render_contract_sha256': h('d'), 'question': vc.QUESTION}
    return {'id': f'dal{i}', 'ability': 'mri_visit_correspondence', 'data_source': vc.SOURCE, 'images': [img],
            'reward_model': {'ground_truth': 'same' if same else 'different'}, 'extra_info': e,
            'prompt': [{'role': 'user', 'content': [{'type': 'image', 'image': img}, {'type': 'text', 'text': vc.QUESTION}]}]}


ROWS = [seq_row(i) for i in range(28)] + [dal_row(i) for i in range(48)]


def test_weights_follow_the_admitted_manifest():
    assert tt.category_weights({'sampling': 'proportional'}, ROWS) == {'mri_sequence': 28 / 76, 'mri_visit_correspondence': 48 / 76}
    assert tt.category_weights({'sampling': 'uniform'}, ROWS) == {'mri_sequence': .5, 'mri_visit_correspondence': .5}
    assert tt.category_weights({'sampling': 'specialist', 'task': 'mri_sequence'}, ROWS) == {'mri_sequence': 1.0, 'mri_visit_correspondence': 0.0}


def test_schedule_wraps_the_pinned_sampler_with_opaque_ids_and_source_record_ids():
    arm = {'sampling': 'uniform'}
    a, binding = tt.build_schedule(ROWS, arm, 7, 198)
    b, _ = tt.build_schedule(ROWS, arm, 7, 198)
    assert [r['id'] for r in a] == [r['id'] for r in b] and len(a) == 198 and len({r['id'] for r in a}) == 198
    assert all(r['id'] != r['source_record_id'] and r['source_record_id'] in {x['id'] for x in ROWS} for r in a)
    assert [r['id'] for r in a] != [r['id'] for r in tt.build_schedule(ROWS, arm, 11, 198)[0]]
    assert binding['official_vero_commit'].startswith('025ca3') and binding['mode'] == 'uniform'
    rep = tt.realized_report(a)
    assert rep['unique_source_rows']['mri_sequence'] == 28 and rep['category_counts']['mri_sequence'] > 28   # pool exhausted/reshuffled
    # model-visible fields unchanged by the wrapper
    src = {x['id']: x for x in ROWS}
    assert all(r['prompt'] == src[r['source_record_id']].get('prompt') and r['reward_model'] == src[r['source_record_id']]['reward_model'] for r in a)


def test_specialist_and_proportional_schedules_follow_the_arm_definition():
    s, _ = tt.build_schedule(ROWS, {'sampling': 'specialist', 'task': 'mri_visit_correspondence'}, 7, 64)
    assert {r['ability'] for r in s} == {'mri_visit_correspondence'}
    p, bind = tt.build_schedule(ROWS, {'sampling': 'proportional'}, 7, 198)
    assert bind['logical_batch_category_counts']['mri_sequence'] in (94, 95) and Counter(r['ability'] for r in p)['mri_sequence'] > 0


def test_input_token_lookup_uses_the_source_record_id_and_requires_full_coverage():
    from mri_rl.two_task_pool import canonical_sha
    sched, _ = tt.build_schedule(ROWS, {'sampling': 'uniform'}, 7, 20)
    meas = [{'record_sha256': canonical_sha(r), 'input_tokens': 213 if r['ability'] == 'mri_sequence' else 405} for r in ROWS]
    look = tt.input_tokens_by_source(meas, ROWS)
    assert all(look(r) == (213 if r['ability'] == 'mri_sequence' else 405) for r in sched)
    with pytest.raises(ValueError): tt.input_tokens_by_source(meas[:-1], ROWS)


def test_row_validation_rejects_wrong_source_split_or_prompt():
    tt.validate_training_row(ROWS[0]); tt.validate_training_row(ROWS[-1])
    bad = {**ROWS[-1], 'data_source': tt.SEQ_SOURCE}
    with pytest.raises(ValueError): tt.validate_training_row(bad)
    dev = {**ROWS[0], 'extra_info': {**ROWS[0]['extra_info'], 'internal_split': 'internal_dev'}}
    with pytest.raises(ValueError): tt.validate_training_row(dev)
    leak = {**ROWS[-1], 'prompt': [{'role': 'user', 'content': 'patient id'}]}
    with pytest.raises(ValueError): tt.validate_training_row(leak)


BOX = '<think>x</think><answer>\\boxed{%s}</answer>'


def test_routed_grading_uses_the_right_scorer_per_source_and_never_a_fake_007045_correspondence():
    g = tt.make_grader()
    assert g(seq_row(1), BOX % 'FLAIR', {})['accuracy'] == 1.0 and g(seq_row(1), BOX % 'contrast-enhanced T1', {})['accuracy'] == 0.0
    d = dal_row(0)                                                               # ground truth 'same'
    assert g(d, BOX % 'same', {})['accuracy'] == 1.0 and g(d, BOX % 'different', {})['accuracy'] == 0.0
    with pytest.raises(ValueError):
        g({**d, 'data_source': tt.SEQ_SOURCE}, BOX % 'same', {})                  # relabeled source is rejected by the scorers


def test_eq3_calibration_uses_frozen_statistics_and_zero_iqr_blocks():
    cal = {'mri_sequence': {'p50': 0.5, 'iqr': 0.5}, 'mri_visit_correspondence': {'p50': 0.0, 'iqr': 0.0}}
    g = tt.make_grader(cal)
    hi, lo = g(seq_row(1), BOX % 'FLAIR', {}), g(seq_row(1), BOX % 'contrast-enhanced T1', {})
    assert hi['raw_accuracy'] == 1.0 and 0.5 < hi['accuracy'] < 1 and 0 < lo['accuracy'] < 0.5
    assert hi['score'] == pytest.approx(0.8 * hi['accuracy'] + 0.2 * hi['format'])
    with pytest.raises(tt.CalibrationBlocked): g(dal_row(0), BOX % 'same', {})


def test_ledger_reserves_whole_groups_and_charges_actual_cost():
    plan = {'group_size': 4, 'max_tokens': 256, 'budget': {'input_tokens_min': 213, 'input_tokens_max': 405, 'text_visual_token_cap': 169216}}
    led = tt.TokenLedger(plan)
    assert led.can_start(405) and led.charge(213, [17, 17, 17, 17]) == 4 * 213 + 68
    led.used = 169216 - 4 * (405 + 256) + 1
    assert not led.can_start(405)                                                # cannot reserve the whole group
    assert led.state()['charged_groups'] == 1


def test_import_and_scoring_work_where_nibabel_is_absent_but_rendering_still_fails_loudly():
    import subprocess, sys, textwrap
    code = textwrap.dedent("""
        import sys
        sys.modules['nibabel'] = None            # simulate the MLX venv: import nibabel -> ImportError
        sys.path.insert(0, 'src')
        from mri_rl import two_task_training as tt
        assert tt.DALLAS_SOURCE == 'OpenNeuro_ds004856'
        g = tt.make_grader()
        from mri_rl import visit_correspondence as vc
        assert 'nibabel' in sys.modules and sys.modules['nibabel'] is None      # stub was removed again
        try:
            vc.render_volume('x')
        except Exception as exc:
            print(type(exc).__name__)
    """)
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, cwd=str(Path(__file__).parents[1]))
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() != ''
