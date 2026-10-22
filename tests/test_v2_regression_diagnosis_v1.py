"""Synthetic CPU tests for the V2 regression diagnosis (no MLX, model, MRI data or real run evidence)."""
import json
import math
import os
import subprocess
import sys
import textwrap
from pathlib import Path
import numpy as np
import pytest
from safetensors.numpy import save_file
from mri_rl import v2_regression_diagnosis_v1 as diag
from mri_rl.optimizer_checkpoint import pack_tree
from mri_rl.uniform_balance_continuation_gate_v1 import accuracy_group_is_informative
from mri_rl.paired_prompt_groups_v1 import identity

SRC = str(Path(__file__).resolve().parents[1] / 'src')
ACC = {1: .5, 0: .0243088}          # calibrated-accuracy shape of the real MRI receipts (correct .6 / valid wrong ~.2194 totals)


# ================================================================ reward / advantage semantics
def grades(spec):
    return [{'accuracy': ACC[raw], 'format': fmt, 'overlong': 0., 'total': .8 * ACC[raw] + .2 * fmt, 'raw_accuracy': raw} for raw, fmt in spec]


def records(spec, *, task='MRI', admitted=True):
    """spec: [(image_class, raw, fmt, predicted)] for ONE group; mirrors plan_records fields."""
    gs = grades([(raw, fmt) for _, raw, fmt, _ in spec])
    comps = diag.advantage_components(gs)
    w = .125 if admitted else 0.
    return [{'task': task, 'image_class': img, 'predicted': pred, 'raw_accuracy': raw, 'format': fmt, 'weight': w, 'joint_weight': .25 * w,
             'advantage': c['advantage'], 'advantage_accuracy_part': c['accuracy_part'], 'advantage_format_part': c['format_part'],
             'advantage_overlong_part': c['overlong_part'], 'reward_identity_holds': c['reward_identity_holds'], 'sign_class': diag.sign_class(raw, c['advantage'])}
            for (img, raw, fmt, pred), c in zip(spec, comps)]


def test_advantage_split_is_exact_and_matches_the_unchanged_estimator():
    from mri_rl.mlx_gspo_smoke import advantages_from_rewards
    gs = grades([(1, 1.), (0, 1.), (0, 1.), (0, 0.)])
    comps = diag.advantage_components(gs)
    assert [c['advantage'] for c in comps] == advantages_from_rewards([g['total'] for g in gs])
    assert all(c['reward_identity_holds'] and abs(c['split_residual']) < 1e-12 for c in comps)
    assert comps[0]['accuracy_part'] > 0 and comps[3]['format_part'] < 0          # correct gains via accuracy; invalid loses via format
    bad = [dict(g) for g in gs]
    bad[0]['total'] += .05                                                        # total not .8acc+.2fmt+overlong -> split refused, never guessed
    assert diag.advantage_components(bad)[0]['accuracy_part'] is None


def test_group_contrast_distinguishes_class_format_and_between_image_contrasts():
    mk = lambda ctx, raw, fmt: {'context': ctx, 'raw_accuracy': raw, 'accuracy': ACC[raw], 'format': fmt, 'total': .8 * ACC[raw] + .2 * fmt}
    assert diag.group_contrast([mk('F', 0, 1.), mk('F', 0, 1.), mk('F', 0, 0.), mk('F', 0, 1.)]) == 'FORMAT_OR_VALIDITY_ONLY'
    assert diag.group_contrast([mk('F', 1, 1.), mk('F', 0, 0.), mk('F', 1, 1.), mk('F', 0, 0.)]) == 'FORMAT_OR_VALIDITY_ONLY'   # invalid != wrong class
    assert diag.group_contrast([mk('F', 1, 1.), mk('F', 0, 1.), mk('F', 1, 1.), mk('F', 1, 1.)]) == 'WITHIN_IMAGE_CLASS'
    assert diag.group_contrast([mk('F', 1, 1.), mk('F', 0, 0.), mk('F', 1, 1.), mk('F', 0, 1.)]) == 'WITHIN_IMAGE_CLASS_WITH_FORMAT'
    assert diag.group_contrast([mk('F', 1, 1.), mk('F', 1, 1.), mk('T', 0, 1.), mk('T', 0, 1.)]) == 'BETWEEN_IMAGE_ONLY'
    assert diag.group_contrast([mk('F', 1, 1.), mk('F', 0, 0.), mk('T', 0, 1.), mk('T', 0, 0.)]) == 'BETWEEN_IMAGE_ONLY_WITH_FORMAT'
    assert diag.group_contrast([mk('T', 0, 1.)] * 4) == 'NO_REWARD_VARIANCE'


def test_within_image_flags_report_all_response_and_format_valid_readings_separately():
    mk = lambda ctx, raw, fmt: {'context': ctx, 'raw_accuracy': raw, 'format': fmt}
    invalid_only = [mk('F', 1, 1.), mk('F', 0, 0.), mk('T', 0, 1.), mk('T', 0, 1.)]       # F wrong only via invalid format
    assert diag.within_image_flags(invalid_only) == {'all_response_within': True, 'format_valid_within': False}
    genuine = [mk('F', 1, 1.), mk('F', 0, 1.), mk('T', 0, 1.), mk('T', 0, 1.)]
    assert diag.within_image_flags(genuine) == {'all_response_within': True, 'format_valid_within': True}
    between = [mk('F', 1, 1.), mk('F', 1, 1.), mk('T', 0, 1.), mk('T', 0, 1.)]             # intended Paired-Prompt contrast, not flagged invalid
    assert diag.within_image_flags(between) == {'all_response_within': False, 'format_valid_within': False}
    assert diag.group_contrast([dict(m, accuracy=ACC[m['raw_accuracy']], total=.8 * ACC[m['raw_accuracy']] + .2 * m['format']) for m in between]) == 'BETWEEN_IMAGE_ONLY'


def test_raw_reward_counterfactual_flips_a_calibrated_wrong_valid_answer_with_fixed_plan():
    spec = [('FLAIR_IMAGE', 1, 1., 'FLAIR'), ('FLAIR_IMAGE', 0, 0., 'INVALID_FORMAT'), ('T1_IMAGE', 0, 1., 'FLAIR'), ('T1_IMAGE', 0, 0., 'INVALID_FORMAT')]
    recs = [dict(r, unit=5, group=0, response_id=f'x_{i}', overlong=0.) for i, r in enumerate(records(spec))]
    assert recs[2]['advantage'] > 0                                                         # actual calibrated reward: wrong-valid is up-weighted
    cf = diag.raw_reward_counterfactual([{'unit': 5, 'responses': recs}])['by_task']['MRI']
    assert cf['advantage_sign_changes_actual_to_raw'] == 1 and cf['sign_changes_by_kind']['WRONG_FORMAT_VALID'] == 1
    assert cf['raw_correct_positive_under_raw'] == cf['raw_correct_total'] == 1 and cf['raw_wrong_negative_under_raw'] == cf['raw_wrong_total'] == 3
    assert 0 < cf['actual_wrong_positive_share_of_abs_joint_weighted_advantage_mass'] < 1


def test_format_only_variance_gives_correct_answers_negative_advantage_but_is_gate_rejected():
    spec = [('FLAIR_IMAGE', 1, 1., 'FLAIR')] * 3 + [('FLAIR_IMAGE', 1, .5, 'FLAIR')]
    gs = [{'accuracy': ACC[1], 'format': f, 'overlong': 0., 'total': .8 * ACC[1] + .2 * f} for f in (1., 1., 1., .5)]
    assert not accuracy_group_is_informative(gs)
    counts = diag.sign_counts(records(spec, admitted=False))
    assert counts['all']['CORRECT_NEGATIVE'] == counts['all']['CORRECT_NEGATIVE_FORMAT_DRIVEN'] == 1
    assert counts['admitted']['CORRECT_NEGATIVE'] == 0


def test_paired_group_rewards_a_valid_wrong_T1_answer_only_through_format():
    spec = [('FLAIR_IMAGE', 1, 1., 'FLAIR'), ('FLAIR_IMAGE', 0, 0., 'INVALID_FORMAT'), ('T1_IMAGE', 0, 1., 'FLAIR'), ('T1_IMAGE', 0, 0., 'INVALID_FORMAT')]
    recs = records(spec)
    wrong_valid = recs[2]
    assert wrong_valid['advantage'] > 0 and wrong_valid['advantage_accuracy_part'] < 0 < wrong_valid['advantage_format_part']
    counts = diag.sign_counts(recs)['admitted']
    assert counts['WRONG_POSITIVE'] == counts['WRONG_POSITIVE_FORMAT_DRIVEN'] == 1


def test_single_class_admitted_group_is_a_flair_answer_push_at_sequence_level():
    f_group = records([('FLAIR_IMAGE', 1, 1., 'FLAIR')] * 3 + [('FLAIR_IMAGE', 0, 1., 'contrast-enhanced T1')])
    t_group = records([('T1_IMAGE', 0, 1., 'FLAIR')] * 3 + [('T1_IMAGE', 1, 1., 'contrast-enhanced T1')])
    balanced = diag.allocation_summary(f_group + t_group)
    assert abs(balanced['sequence_level_push_toward_FLAIR_answers_regardless_of_image']) < 1e-9
    assert balanced['sequence_level_image_conditioned_push'] > 0
    only_f = diag.allocation_summary(f_group + records([('T1_IMAGE', 0, 1., 'FLAIR')] * 4, admitted=False))
    assert only_f['sequence_level_push_toward_FLAIR_answers_regardless_of_image'] > 0
    assert only_f['admitted_responses'] == 4 and only_f['MRI_share_of_abs_joint_weighted_advantage'] == 1


def test_regrade_flags_a_reward_that_text_does_not_support():
    saved = {'accuracy': .5, 'format': 1., 'raw_accuracy': 1., 'overlong': 0., 'total': .6}
    ok = lambda row, text, config: {'accuracy': .5, 'format': 1., 'raw_accuracy': 1., 'score': .6}
    bad = lambda row, text, config: {'accuracy': ACC[0], 'format': 1., 'raw_accuracy': 0., 'score': .8 * ACC[0] + .2}
    assert diag.regrade_one(ok, {}, 'x', 20, saved)['match']
    assert not diag.regrade_one(bad, {}, 'x', 20, saved)['match']


# ================================================================ real planners on a synthetic V2 run (shared fake backend fixtures)
@pytest.fixture
def synthetic_run(tmp_path):
    from test_accuracy_pilot_v2 import units4, lengths_for, make_common, run_arm
    units, lengths = units4('concise'), lengths_for('concise')
    _, _, common = make_common(tmp_path, units, lengths)
    for arm in ('A', 'B'):
        run_arm(tmp_path, arm, units, lengths, common)
    return tmp_path, units


def _rederive(tmp, units, arm):
    from test_accuracy_pilot_v2 import PARENT_RECEIPT
    from test_multistep_gspo_v1 import REWARD
    return diag.rederive_arm(units, tmp / 'common', tmp / ('arm_' + arm), arm, reward_sha=REWARD, parent_receipt=PARENT_RECEIPT)


def test_rederived_groups_advantages_and_weights_match_saved_evidence(synthetic_run):
    tmp, units = synthetic_run
    out = {arm: _rederive(tmp, units, arm) for arm in ('A', 'B')}
    for arm in ('A', 'B'):
        for seg in (0, 1):
            assert out[arm][seg]['saved_evidence_all_match'], out[arm][seg]['saved_evidence_comparison']
            for unit in out[arm][seg]['units']:
                for r in unit['responses']:
                    assert r['joint_weight'] == .25 * r['weight']
    shared0 = diag.shared_segment0_comparison({arm: out[arm][0]['units'] for arm in ('A', 'B')})
    assert shared0['responses'] == 32 and shared0['Dallas_units_identical_plans']
    json.dumps(out, allow_nan=False)


def test_saved_evidence_tampering_is_reported_not_trusted(synthetic_run):
    tmp, units = synthetic_run
    path = tmp / 'arm_A' / 'segment_0' / 'gradient_evidence_PRIVATE.json'
    evidence = json.loads(path.read_text())
    evidence[0]['items'][0]['advantage'] += 1e-3
    path.write_text(json.dumps(evidence))
    out = _rederive(tmp, units, 'A')
    assert not out[0]['saved_evidence_all_match'] and 'advantage' in out[0]['saved_evidence_comparison'][0]['problems']


def test_receipt_bundle_and_rollout_mutations_are_rejected(synthetic_run):
    tmp, units = synthetic_run
    bundle = tmp / 'arm_A' / 'segment_1' / 'bundle_unit5_PRIVATE.json'
    original = bundle.read_text()
    value = json.loads(original)
    value['responses'][0]['grade']['total'] += .1
    bundle.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='hash differs'):
        _rederive(tmp, units, 'A')
    body = {k: v for k, v in value.items() if k != 'bundle_sha256'}
    value['bundle_sha256'] = identity(body)                               # forger also re-hashes the body
    bundle.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='reward receipt'):
        _rederive(tmp, units, 'A')
    bundle.write_text(original)
    receipt = tmp / 'arm_B' / 'segment_0' / 'receipt_PRIVATE.json'
    r = json.loads(receipt.read_text())
    r['effective_groups'] += 1
    receipt.write_text(json.dumps(r))
    with pytest.raises(ValueError, match='receipt changed'):
        _rederive(tmp, units, 'B')
    raw = tmp / 'common' / 'raw_rollout_and_own_old_PRIVATE.json'
    v = json.loads(raw.read_text())
    v['raws'][0]['responses'][0]['grade']['total'] += .1
    raw.write_text(json.dumps(v))
    with pytest.raises(ValueError):
        _rederive(tmp, units, 'A')


def _real_journal_commit(tmp_path, key='op', payload='{"a":1}'):
    """A commit written by the unchanged production Journal (immutable format, not re-implemented)."""
    from mri_rl.safe_micro_journal_v1 import Journal
    f = tmp_path / f'{key}_payload.json'
    with Journal(tmp_path / 'journal', {'scope': 'synthetic'}) as journal:
        def op(_):
            f.write_text(payload)
            return {'value': {'v': 1}, 'files': {str(f.resolve()): diag.sha256(f)}}
        journal.run(key, op)
    return f, tmp_path / 'journal' / key / 'commit_PRIVATE.json'


def test_self_hash_file_hash_and_journal_commit_bindings(tmp_path):
    record = {'x': 1}
    record['receipt_sha256'] = identity({'x': 1})
    assert diag.verify_self_hash(record)
    with pytest.raises(ValueError, match='self-hash'):
        diag.verify_self_hash({**record, 'x': 2})
    f, commit = _real_journal_commit(tmp_path)
    ok = diag.verify_journal_commit(commit, [f], value={'v': 1})
    assert ok['committed_files_verified'] == 1 and ok['record_sha256'] == json.loads(commit.read_text())['record_sha256']
    with pytest.raises(ValueError, match='value differs'):
        diag.verify_journal_commit(commit, [f], value={'v': 2})
    other = tmp_path / 'other.json'
    other.write_text('{}')
    with pytest.raises(ValueError, match='absent'):
        diag.verify_journal_commit(commit, [other])
    original = commit.read_text()
    rec = json.loads(original)
    rec['value'] = {'v': 9}                                                   # commit edited without re-signing
    commit.write_text(json.dumps(rec))
    with pytest.raises(ValueError, match='record_sha256'):
        diag.verify_journal_commit(commit, [f])
    commit.write_text(original)
    claim = commit.parent / 'reservation_PRIVATE.json'
    claim_text = claim.read_text()
    claim.write_text(claim_text.replace('"op"', '"other"'))                 # reservation swapped
    with pytest.raises(ValueError, match='reservation'):
        diag.verify_journal_commit(commit, [f])
    claim.write_text(claim_text)
    f.write_text('{"a":2}')                                                   # committed artifact mutated (also when not "required")
    with pytest.raises(ValueError, match='SHA-256 changed'):
        diag.verify_journal_commit(commit, [])


def test_coherent_local_rehash_passes_the_journal_but_not_the_committed_stage_anchor(tmp_path):
    f, commit = _real_journal_commit(tmp_path)
    run = tmp_path / 'run'
    results = {'scope_file_sha256': 's' * 64, 'evidence': {'all_stage_hashes': {}}}
    for stage in diag.STAGES:
        d = run / f'stage{stage}_PRIVATE'
        d.mkdir(parents=True)
        (d / 'terminal_PRIVATE.json').write_text(json.dumps({'status': 'PASS', 'stage': stage}))
        w = run / f'stage{stage}_attempt0_exit_PRIVATE'
        w.mkdir()
        (w / 'actual_exit_PRIVATE.json').write_text('{"native_exit_code":0}')
        h = {'terminal_sha256': diag.sha256(d / 'terminal_PRIVATE.json'), 'exit_sha256': diag.sha256(w / 'actual_exit_PRIVATE.json')}
        results['evidence']['all_stage_hashes'][stage] = h
        (run / f'stage{stage}_verified_PRIVATE.json').write_text(json.dumps({'scope_file_sha256': 's' * 64, 'exit_path': str(w / 'actual_exit_PRIVATE.json'), **h}))
    assert set(diag.verify_stage_anchors(run, results)) == set(diag.STAGES)
    # forger mutates a payload and coherently re-signs the journal commit: the local journal check alone cannot see it ...
    f.write_text('{"a":666}')
    rec = json.loads(commit.read_text())
    rec['files'] = {str(f.resolve()): diag.sha256(f)}
    rec['record_sha256'] = identity({k: v for k, v in rec.items() if k != 'record_sha256'})
    commit.write_text(json.dumps(rec))
    assert diag.verify_journal_commit(commit, [f])
    # ... but a coherently rewritten stage terminal is rejected by the external committed-result anchor
    t = run / 'stageA_PRIVATE' / 'terminal_PRIVATE.json'
    t.write_text(json.dumps({'status': 'PASS', 'stage': 'A', 'forged': True}))
    with pytest.raises(ValueError, match='committed-result anchor'):
        diag.verify_stage_anchors(run, results)
    t.write_text(json.dumps({'status': 'PASS', 'stage': 'A'}))
    exit_a = run / 'stageA_attempt0_exit_PRIVATE' / 'actual_exit_PRIVATE.json'
    exit_a.write_text('{"native_exit_code":1}')
    with pytest.raises(ValueError, match='committed-result anchor'):
        diag.verify_stage_anchors(run, results)


def test_terminal_links_require_receipts_and_dev_metrics_to_match_the_anchored_terminals():
    from mri_rl import accuracy_pilot_v2_core as core
    rows = {k: [{'truth': lab, 'text': '<think>r</think><answer>\\boxed{FLAIR}</answer>', 'tokens': 10, 'wall_seconds': 1.}
                for lab in ('FLAIR', 'contrast-enhanced T1')] for k in ('Parent', 'A', 'B')}
    receipts = {(a, s): {'receipt_sha256': f'{a}{s}' * 32} for a in ('A', 'B') for s in (0, 1)}
    terminals = {a: {'terminal': {'receipts_sha256': [f'{a}0' * 32, f'{a}1' * 32]}, 'arm': {'receipt_sha256': [f'{a}0' * 32, f'{a}1' * 32]}} for a in ('A', 'B')}
    terminals.update({s: {'metrics': json.loads(json.dumps(core.greedy_summary(rows[l])))} for s, l in (('EP', 'Parent'), ('EA', 'A'), ('EB', 'B'))})
    assert diag.link_terminals(terminals, receipts, rows)['A'] == 'SEGMENT_RECEIPTS_MATCH_ANCHORED_TERMINAL'
    receipts[('B', 1)] = {'receipt_sha256': 'f' * 64}
    with pytest.raises(ValueError, match='segment receipts differ'):
        diag.link_terminals(terminals, receipts, rows)


def test_source_pins_reject_drift_from_the_native_run_scope():
    from mri_rl import multistep_gspo_core_v1 as ms
    root = Path(ms.__file__).resolve().parents[2]          # the tree the planners were actually imported from
    assert all(v['status'] == 'NOT_IN_SCOPE' for v in diag.source_pins(root, {'source_sha256': {}}, [ms]).values())
    target = str((root / 'src/mri_rl/paired_prompt_groups_v1.py').resolve())
    with pytest.raises(ValueError, match='drifted'):
        diag.source_pins(root, {'source_sha256': {target: '0' * 64}}, [ms])


# ================================================================ synthetic Adam checkpoints (installed-formula fixtures)
NAMES = ('language_model.model.layers.0.self_attn.q_proj.lora_a', 'language_model.model.layers.0.self_attn.q_proj.lora_b',
         'language_model.model.layers.1.mlp.down_proj.lora_a', 'language_model.model.layers.1.mlp.down_proj.lora_b')
SHAPES = {NAMES[0]: (6, 2), NAMES[1]: (2, 5), NAMES[2]: (5, 2), NAMES[3]: (2, 6)}
LR = np.array(1e-5, dtype=np.float32)


def nest(flat):
    tree = {}
    for name, value in flat.items():
        node = tree
        parts = name.split('.')
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    model = tree['language_model']['model']
    model['layers'] = [model['layers'][str(i)] for i in range(len(model['layers']))]
    return tree


def write_ckpt(d, theta, m, v, step):
    (d / 'optimizer').mkdir(parents=True)
    save_file({k: np.require(x, requirements='C') for k, x in theta.items()}, str(d / 'adapter.safetensors'))
    state = {'step': np.array(step, dtype=np.uint64), 'learning_rate': LR.copy(), **nest({k: {'m': m[k], 'v': v[k]} for k in theta})}
    layout, arrays = pack_tree({'optimizer': state, 'rng': {'mlx_fixture': [np.zeros(2, dtype=np.uint32)]}})
    save_file({k: np.require(x, requirements='C') for k, x in arrays.items()}, str(d / 'optimizer' / 'state.safetensors'))
    (d / 'optimizer' / 'manifest.json').write_text(json.dumps({'schema': 'mri-optimizer-boundary-v1', 'layout': layout, 'optimizer_updates': step,
                                                               'completed_groups': step, 'bindings': {},
                                                               'tensors_sha256': diag.sha256(d / 'optimizer' / 'state.safetensors')}))
    return d


def mlx_adam(theta, m, v, g):
    """Independent restatement of the installed formula (float32): never an optimizer object."""
    f = np.float32
    out = {}
    for k in theta:
        m1 = f(.9) * m[k] + f(1 - .9) * g[k]
        v1 = f(.999) * v[k] + f(1 - .999) * np.square(g[k])
        out[k] = (theta[k] - (LR * m1) / (np.sqrt(v1) + f(1e-8)), m1, v1)
    return {k: x[0] for k, x in out.items()}, {k: x[1] for k, x in out.items()}, {k: x[2] for k, x in out.items()}


def rand(seed, scale):
    r = np.random.default_rng(seed)
    return {k: (scale * r.standard_normal(SHAPES[k])).astype(np.float32) for k in NAMES}


def write_grad(path, g):
    save_file({k: np.require(x, requirements='C') for k, x in g.items()}, str(path))
    return diag.TensorFile(path)


def path_of(tmp, label, theta, m, v, gs, step=52):
    dirs = [write_ckpt(tmp / f'{label}{step}', theta, m, v, step)]
    files = []
    for i, g in enumerate(gs):
        theta, m, v = mlx_adam(theta, m, v, g)
        dirs.append(write_ckpt(tmp / f'{label}{step + i + 1}', theta, m, v, step + i + 1))
        files.append(write_grad(tmp / f'{label}_g{i}.safetensors', g))
    return [diag.Checkpoint(d, label=d.name) for d in dirs], files


def test_adam_recomputation_passes_and_identifies_the_applied_gradient(tmp_path):
    theta, m, v = rand(1, .02), rand(2, .003), {k: np.abs(x) + np.float32(1e-6) for k, x in rand(3, 1e-5).items()}
    g = rand(4, .004)
    (p, n), (gf,) = path_of(tmp_path, 'X', theta, m, v, [g])
    rep = diag.adam_step_report(p, n, gf)
    assert all(c['PASS'] for c in rep['formula_checks'].values())
    assert rep['saved_gradient_is_the_applied_gradient']
    assert rep['beta_fit_from_saved_moments_and_saved_joint_gradient']['b1']['abs_diff'] < 1e-5
    split = rep['metrics']['actual_denominator_split']
    assert abs(split['momentum_share_along_delta'] + split['gradient_share_along_delta'] - 1) < 1e-2 and split['residual_l2_over_delta_l2'] < 1e-2
    json.dumps(rep, allow_nan=False)


def test_counterexample_old_momentum_reverses_the_euclidean_descent_direction(tmp_path):
    ones = {k: np.ones(SHAPES[k], dtype=np.float32) for k in NAMES}
    theta = {k: .01 * x for k, x in ones.items()}
    m = {k: np.float32(-.5) * x for k, x in ones.items()}                 # inherited momentum opposite to the new gradient
    v = {k: np.float32(1e-4) * x for k, x in ones.items()}
    g = {k: np.float32(.01) * x for k, x in ones.items()}
    (p, n), (gf,) = path_of(tmp_path, 'R', theta, m, v, [g])
    metrics = diag.adam_step_report(p, n, gf)['metrics']
    assert metrics['cos_descent_vs_actual_delta'] < -.999                    # the actual step ASCENDS the current surrogate
    assert metrics['first_order_surrogate_change_g_dot_delta'] > 0
    assert metrics['cos_actual_delta_vs_momentum_only'] > .999               # it is the g=0 momentum step
    split = metrics['actual_denominator_split']
    assert split['gradient_share_along_delta'] < 0 < 1 < split['momentum_share_along_delta']
    assert split['cos_gradient_term_vs_delta'] < 0                           # the preconditioned gradient term itself is descent-aligned


def test_adam_mutations_wrong_gradient_and_step_count_are_detected(tmp_path):
    theta, m, v, g = rand(5, .02), rand(6, .003), {k: np.abs(x) + np.float32(1e-6) for k, x in rand(7, 1e-5).items()}, rand(8, .004)
    (p, n), _ = path_of(tmp_path, 'M', theta, m, v, [g])
    wrong = write_grad(tmp_path / 'wrong.safetensors', {k: np.float32(2) * x for k, x in g.items()})
    rep = diag.adam_step_report(p, n, wrong)
    assert not rep['saved_gradient_is_the_applied_gradient'] and not rep['formula_checks']['m']['PASS']
    th1, m1, v1 = mlx_adam(theta, m, v, g)
    th1 = dict(th1)
    th1[NAMES[0]] = th1[NAMES[0]].copy()
    th1[NAMES[0]][0, 0] += np.float32(1e-6)                                   # far beyond float32 spacing at |theta|~.02
    bumped = diag.Checkpoint(write_ckpt(tmp_path / 'bumped', th1, m1, v1, 53), label='bumped')
    gf = write_grad(tmp_path / 'g.safetensors', g)
    assert diag.adam_step_report(p, bumped, gf)['formula_checks']['theta']['violations'] >= 1
    same_step = diag.Checkpoint(write_ckpt(tmp_path / 'same', th1, m1, v1, 52), label='same')
    with pytest.raises(ValueError, match='Exactly one optimizer step'):
        diag.adam_step_report(p, same_step, gf)
    manifest = tmp_path / 'same' / 'optimizer' / 'manifest.json'
    value = json.loads(manifest.read_text())
    value['tensors_sha256'] = '0' * 64
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='SHA-256 changed'):
        diag.Checkpoint(tmp_path / 'same', label='same')


@pytest.mark.parametrize('pred,saved,bound', [(np.nan, 0., 0.), (np.inf, 0., np.inf), (0., 0., np.nan), (0., 0., -1.), (1., np.inf, np.inf)])
def test_check_never_passes_nonfinite_values_or_invalid_bounds(pred, saved, bound):
    c = diag.Check('x')
    c.add(np.array([pred], dtype=np.float32), np.array([saved], dtype=np.float32), np.array([bound]))
    rep = c.report()
    assert rep['violations'] == 1 and not rep['PASS']
    ok = diag.Check('y')
    ok.add(np.array([1.], dtype=np.float32), np.array([1.], dtype=np.float32), np.array([0.]))
    assert ok.report()['PASS'] and ok.report()['nonfinite_elements'] == 0


def test_negative_saved_adam_second_moment_is_rejected_before_sqrt(tmp_path):
    theta, m, g = rand(20, .02), rand(21, .003), rand(22, .004)
    v = {k: np.abs(x) + np.float32(1e-6) for k, x in rand(23, 1e-5).items()}
    (p, n), (gf,) = path_of(tmp_path, 'N', theta, m, v, [g])
    bad_v = {k: x.copy() for k, x in v.items()}
    bad_v[NAMES[0]][0, 0] = np.float32(-1e-6)
    bad = diag.Checkpoint(write_ckpt(tmp_path / 'negv', theta, m, bad_v, 52), label='negv')
    with pytest.raises(ValueError, match='negative saved Adam second moment'):
        diag.adam_step_report(bad, n, gf)


def test_shared_inherited_momentum_makes_opposite_gradient_arms_move_together(tmp_path):
    theta = rand(10, .02)
    m = rand(11, .05)
    v = {k: np.full(SHAPES[k], 2.5e-3, dtype=np.float32) for k in NAMES}
    gA0, gA1, gB1 = rand(12, .005), rand(13, .005), rand(14, .005)
    gB0 = {k: -x for k, x in gA0.items()}
    (pA, a53, a54), (fa0, fa1) = path_of(tmp_path, 'A', theta, m, v, [gA0, gA1])
    (_, b53, b54), (fb0, fb1) = path_of(tmp_path, 'B', theta, m, v, [gB0, gB1])
    cross = diag.cross_arm_report(pA, a53, a54, b53, b54, fa0, fa1, fb0, fb1)
    assert cross['cosine']['gA0|gB0'] < -.999
    assert cross['cosine']['dAtot|dBtot'] > .99
    two = diag.two_step_report(pA, a53, a54, fa0, fa1)
    shares = two['shares_along_total_delta']
    assert abs(sum(shares.values()) - 1) < 1e-2 and shares['parent_history_momentum_m52'] > .9 and two['residual_l2_over_total_l2'] < 1e-2
    lora = diag.lora_effective_report({'P': pA, 'A53': a53, 'A54': a54, 'B53': b53, 'B54': b54}, 2.)
    assert lora['modules'] == 2 and lora['total']['cos_effective_dA_tot_vs_dB_tot'] > .9


def test_lowrank_inner_equals_direct_frobenius():
    r = np.random.default_rng(0)
    a1, b1, a2, b2 = r.standard_normal((7, 3)), r.standard_normal((3, 5)), r.standard_normal((7, 3)), r.standard_normal((3, 5))
    x = [(1., a1, b1), (-1., a2, b2)]
    assert math.isclose(diag.lowrank_inner(x, x), float(np.sum((a1 @ b1 - a2 @ b2) ** 2)), rel_tol=1e-12)
    assert math.isclose(diag.lowrank_inner([(2., a1, b1)], [(1., a2, b2)]), float(np.sum(2 * (a1 @ b1) * (a2 @ b2))), rel_tol=1e-12)


# ================================================================ task gradients: refuse to invent
def test_task_cosine_refuses_without_separately_retained_task_gradients():
    assert diag.task_cosine(None)['status'] == 'NOT_VERIFIED'
    assert diag.task_cosine({'MRI': None, 'DALLAS': np.ones(3)})['cosine'] is None
    measured = diag.task_cosine({'MRI': np.array([1., 0.]), 'DALLAS': np.array([1., 1.])})
    assert measured['status'] == 'MEASURED' and math.isclose(measured['cosine'], 1 / math.sqrt(2))


def test_joint_norm_constraint_identifies_a_cosine_only_for_exactly_two_units():
    r = np.random.default_rng(3)
    g = [r.standard_normal(9) for _ in range(4)]
    two = [np.zeros(9), g[1], np.zeros(9), g[3]]
    joint = .25 * sum(two)
    res = diag.joint_norm_constraint(float(np.linalg.norm(joint)), [float(np.linalg.norm(x)) for x in two], ['MRI', 'DALLAS', 'MRI', 'DALLAS'])
    true = float(g[1] @ g[3] / np.linalg.norm(g[1]) / np.linalg.norm(g[3]))
    assert res['identifies_pairwise_cosine'] and math.isclose(res['derived_cosine'], true, rel_tol=1e-9) and not res['MRI_vs_Dallas']
    four = diag.joint_norm_constraint(float(np.linalg.norm(.25 * sum(g))), [float(np.linalg.norm(x)) for x in g], ['MRI', 'DALLAS', 'MRI', 'DALLAS'])
    assert not four['identifies_pairwise_cosine'] and 'derived_cosine' not in four


def test_inventory_labels_task_pure_segments_and_reports_no_per_task_files(tmp_path):
    d = tmp_path / 'stageA_PRIVATE' / 'arm_PRIVATE' / 'segment_1'
    d.mkdir(parents=True)
    (d / 'joint_gradient_float32_PRIVATE.safetensors').write_bytes(b'')
    units = lambda mri, dal: [{'task': 'MRI', 'effective_groups': mri}, {'task': 'DALLAS', 'effective_groups': dal}]
    inv = diag.task_gradient_inventory(tmp_path, {('A', 1): {'units': units(0, 2), 'saved_unit_l2': [0., 1.]},
                                                  ('B', 1): {'units': units(2, 1), 'saved_unit_l2': [1., 1.]}})
    assert inv['segments']['A1']['joint_task_composition'] == 'DALLAS_ONLY_EXACT_BY_GATE'
    assert inv['segments']['B1']['joint_task_composition'] == 'MIXED_NOT_SEPARABLE_FROM_JOINT'
    assert inv['per_task_or_per_unit_gradient_files'] == [] and inv['MRI_vs_Dallas_task_cosine']['status'] == 'NOT_VERIFIED'


# ================================================================ DEV greedy-path evidence
VOCAB = {'x': 1, '\\boxed{': 2, 'FL': 3, 'AIR': 4, '}': 5, 'con': 6, 'trast': 7}


def test_decision_token_and_divergence_are_located_without_a_model():
    dec = diag.TokenDecoder(VOCAB, [{'id': 99, 'content': '<think>'}])
    assert dec.piece(99) == b'<think>' and diag.decision_token_index([99, 1, 2, 3, 4, 5], dec) == 3
    ref = {'token_ids': [1, 2, 3, 4, 5], 'raw_selected_logps': [-.1, -.01, -.6, -.01, -.01]}
    flip = {'token_ids': [1, 2, 6, 7, 5], 'raw_selected_logps': [-.2, -.01, -.65, -.01, -.01]}
    same = {'token_ids': [1, 2, 3, 4, 5], 'raw_selected_logps': [-.1, -.01, -.9, -.01, -.01]}
    c = diag.selected_logp_comparison(ref, flip, dec)
    assert c['first_divergence_position'] == 2 and c['divergence_relative_to_decision'] == 'AT_DECISION'
    s = diag.selected_logp_comparison(ref, same, dec)
    assert s['divergence_relative_to_decision'] == 'IDENTICAL' and math.isclose(s['decision_token_logp_delta_child_minus_ref'], -.3)
    assert diag.selected_logp_comparison(ref, same)['ref_decision_token_index'] is None      # no vocabulary -> not verified


def test_dev_cases_use_pseudonymous_pairs_and_track_partner_and_category_changes():
    text = lambda label: '<think>r</think><answer>\\boxed{%s}</answer>' % label
    recs = [{'id': f'r{i}', 'row_PRIVATE': {'extra_info': {'patient_key': 'p%d' % (i // 2), 'level': 45}, 'reward_model': {'ground_truth': lab}}}
            for i, lab in enumerate(['FLAIR', 'contrast-enhanced T1'] * 2)]
    resp = {'token_ids': [1], 'raw_selected_logps': [-.1]}
    row = lambda label: {'text': text(label), 'actual_PRIVATE': {'response': resp}}
    parent = [row('FLAIR'), row('contrast-enhanced T1'), row('FLAIR'), row('contrast-enhanced T1')]
    child = [row('FLAIR'), row('FLAIR'), row('FLAIR'), row('contrast-enhanced T1')]
    cases = diag.dev_case_analysis(recs, {'Parent': parent, 'A': child, 'B': child})
    lost = [c for c in cases if c['transition_A'] == 'CORRECT_TO_WRONG']
    assert len(lost) == 1 and lost[0]['modality'] == 'T1' and lost[0]['partner_correct']['A']
    assert lost[0]['pair_category'] == {'Parent': 'DISCRIMINATES_CORRECTLY', 'A': 'CONSTANT_FLAIR', 'B': 'CONSTANT_FLAIR'}
    summary = diag.dev_summary(cases)
    assert summary['A']['transitions']['CORRECT_TO_WRONG'] == 1 and summary['A_B_identical_predictions']
    assert summary['A']['sign_test_two_sided_p_DESCRIPTIVE'] == 1.


def _dev_cases():
    text = lambda label: '<think>r</think><answer>\\boxed{%s}</answer>' % label
    recs = [{'id': f'r{i}', 'row_PRIVATE': {'extra_info': {'patient_key': 'p%d' % (i // 2), 'level': 45}, 'reward_model': {'ground_truth': lab}}}
            for i, lab in enumerate(['FLAIR', 'contrast-enhanced T1'] * 2)]
    resp = {'token_ids': [1], 'raw_selected_logps': [-.1]}
    row = lambda label: {'text': text(label), 'actual_PRIVATE': {'response': resp}}
    parent = [row('FLAIR'), row('contrast-enhanced T1'), row('FLAIR'), row('contrast-enhanced T1')]
    child = [row('FLAIR'), row('FLAIR'), row('FLAIR'), row('contrast-enhanced T1')]
    return diag.dev_case_analysis(recs, {'Parent': parent, 'A': child, 'B': child})


def test_public_aggregate_is_structurally_aggregate_only(synthetic_run):
    tmp, units = synthetic_run
    training = {arm: _rederive(tmp, units, arm) for arm in ('A', 'B')}
    cases = _dev_cases()
    report = {'status': 'X', 'boundaries': {}, 'measurements_by_hypothesis': {},
              'authentication': {**{k: 'v' for k in ('scope_file_sha256', 'spec_sha256', 'parent_adapter_sha256', 'reward_calibration_sha256',
                                                     'prompt_condition', 'DEV_rows_verified', 'segment_receipts')}, 'source_pins': {}},
              'adam_formula_pins': {'constructor': {'present': True}, 'installed_mlx_adam': {'status': 'MATCH'}}, 'training': training,
              'segment0_A_vs_B': diag.shared_segment0_comparison({a: training[a][0]['units'] for a in ('A', 'B')}),
              'task_gradient_inventory': {'segments': {}}, 'tensor_analysis': {}, 'evaluation_inventory': {},
              'joint_norm_constraints': {'A0': diag.joint_norm_constraint(1., [1., 0., 1., 0.], ['MRI', 'DALLAS', 'MRI', 'DALLAS'])},
              'dev': {'cases': cases, 'summary': diag.dev_summary(cases)}}
    pub = diag.public_aggregate(report)
    assert diag.public_schema_check(pub)['status'] == 'PASS'
    assert pub['dev']['A']['transitions']['CORRECT_TO_WRONG'] == 1 and pub['dev']['A']['transitions_by_modality']['T1']['CORRECT_TO_WRONG'] == 1
    assert pub['dev']['Parent']['pair_category_counts'] == {'DISCRIMINATES_CORRECTLY': 2}
    for arm in ('A', 'B'):
        assert 'raw_reward_counterfactual' in pub['training'][arm]['0']['summary']
    json.dumps(pub, allow_nan=False)
    with pytest.raises(ValueError, match='per-case'):                          # the previous public layout is rejected structurally
        diag.public_schema_check({'dev': {'summary': report['dev']['summary'], 'cases': [{'case': 0}]}})
    with pytest.raises(ValueError, match='per-case'):
        diag.public_schema_check({'training': {'A': {'0': {'groups': [{'members': ['x_0'], 'rewards': [.6]}]}}}})


# ================================================================ privacy and isolation
def test_privacy_check_rejects_identifiers_texts_and_paths():
    assert diag.privacy_check({'a': 1}, {'abcdef123456'})['status'] == 'PASS'
    with pytest.raises(ValueError, match='forbidden'):
        diag.privacy_check({'a': 'see abcdef123456'}, {'abcdef123456'})
    with pytest.raises(ValueError, match='absolute'):
        diag.privacy_check({'a': '/Users/someone/x'}, set())


def test_write_exclusive_never_overwrites(tmp_path):
    p = tmp_path / 'out.json'
    diag.write_exclusive(p, {'a': 1})
    with pytest.raises(FileExistsError):
        diag.write_exclusive(p, {'a': 2})


def test_module_is_cpu_isolated_and_guard_blocks_heavy_frameworks():
    code = textwrap.dedent('''
        from mri_rl import v2_regression_diagnosis_v1 as d
        assert not d.forbidden_modules_loaded(), d.forbidden_modules_loaded()
        d.install_import_guard()
        for name in ("mlx.core", "torch", "jax"):
            try:
                __import__(name)
            except ImportError as error:
                assert "blocked" in str(error), error
            else:
                raise SystemExit("not blocked: " + name)
        print("OK")
    ''')
    out = subprocess.run([sys.executable, '-c', code], env={**os.environ, 'PYTHONPATH': SRC}, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0 and out.stdout.strip() == 'OK', out.stderr
