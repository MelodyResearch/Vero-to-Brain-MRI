"""CPU/NumPy tests for the final balanced patient-paired sequence SFT candidate. No MLX, model, MRI data or native evidence."""
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import textwrap
import types
import numpy as np
import pytest
if str(Path(__file__).resolve().parents[1] / 'src') not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from mri_rl import final_balanced_sft_v1 as sft
from mri_rl.optimizer_checkpoint import load_optimizer_checkpoint
from mri_rl.accuracy_pilot_v2_core import CONCISE_INSTRUCTION

SRC = str(Path(__file__).resolve().parents[1] / 'src')
QUESTION = 'Which acquisition sequence is shown?'
SITES8 = [f'site{i}' for i in range(8)]
V, D = 128, 6


# ================================================================ synthetic rows / tokenizer
def people():
    return [f'sub-{i:03d}' for i in range(28)]


def sites():
    return {p: SITES8[i % 8] for i, p in enumerate(people())}


def make_rows(*, split='train', bad=None):
    rows, n = [], 0
    for person in people():
        for modality, label in (('flair', 'FLAIR'), ('t1ce', 'contrast-enhanced T1')):
            source = f'{person}_z045_{modality}_sequence'
            rows.append({'id': f'opaque_{n:03d}', 'ability': 'mri_sequence', 'data_source': 'OpenNeuro_ds007045', 'images': [f'/x/{source}.png'],
                         'reward_model': {'ground_truth': label},
                         'extra_info': {'patient_key': person, 'internal_split': split, 'patient_lineage': 'BIDS_SUBJECT_VERIFIED', 'question': QUESTION, 'index': n},
                         'paired_z045_data_v1': {'version': 1, 'level': 45, 'source_record_id_PRIVATE': source, 'modality_PRIVATE': modality}})
            n += 1
    if bad:
        bad(rows)
    return rows


def no_exclusions():
    return {'dev': set(), 'holdout16': set(), 'final100': set()}


def encode(text):
    return [ord(c) for c in text]          # char tokenizer: prefix/label/suffix boundaries are exact


EOS = 3


def plan_targets(seed=0):
    plan = sft.build_plan(make_rows(), sites(), seed=seed, excluded_patient_keys=no_exclusions())
    targets = sft.build_targets(encode, EOS)
    binding = sft.run_binding(plan=plan, targets=targets, parent_receipt_sha256='a' * 64, scope_sha256='b' * 64)
    return plan, targets, binding


# ================================================================ NumPy fixture backend (toy LM, real Adam semantics, real checkpoint files)
def log_softmax(z):
    z = np.asarray(z, dtype=np.float64)
    top = z.max(axis=-1, keepdims=True)
    return z - top - np.log(np.exp(z - top).sum(axis=-1, keepdims=True))


EMB = np.random.default_rng(5).normal(size=(V, D))


def features(row):
    seed = int(hashlib.sha256(row['id'].encode()).hexdigest()[:8], 16)
    return np.random.default_rng(seed).normal(size=D)


def toy_terms(W, x, ids):
    prev = [1] + list(ids[:-1])
    H = np.stack([x + EMB[p] for p in prev])            # (T, D): image feature + previous-token embedding
    return H, H @ W.T                                   # logits (T, V)


def toy_ce_and_grad(W, x, ids):
    H, logits = toy_terms(W, x, ids)
    lp = log_softmax(logits)
    logps = lp[np.arange(len(ids)), ids]
    probs = np.exp(lp)
    probs[np.arange(len(ids)), ids] -= 1.
    grad = probs.T @ H / len(ids)                       # d mean CE / dW
    return -logps.mean(), grad.astype(np.float32), logps


def save_arrays(path, arrays):
    with open(path, 'wb') as handle:
        np.savez(handle, **{k: np.asarray(v) for k, v in arrays.items()})


def load_arrays(path):
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


class Fake:
    evidence_kind = sft.SYNTHETIC_KIND

    def __init__(self, *, fail=None, mutate_in_backward=False, consume_rng=False, zero=False, charge_extra=False):
        self.W = np.random.default_rng(11).normal(scale=.1, size=(V, D)).astype(np.float32)
        self.m = np.random.default_rng(12).normal(scale=1e-3, size=(V, D)).astype(np.float32)   # inherited GSPO-era moments
        self.v = np.abs(np.random.default_rng(13).normal(scale=1e-4, size=(V, D))).astype(np.float32)
        self.step, self.rng = 52, random.Random(13)
        self.fail, self.mutate, self.consume, self.zero, self.charge_extra = fail, mutate_in_backward, consume_rng, zero, charge_extra
        self.pair = None
        self.calls = []

    # identities
    def policy_sha(self): return hashlib.sha256(self.W.tobytes()).hexdigest()
    def adam_digest(self): return hashlib.sha256(self.m.tobytes() + self.v.tobytes() + str(self.step).encode()).hexdigest()
    def rng_digest(self): return hashlib.sha256(repr(self.rng.getstate()).encode()).hexdigest()
    def optimizer_step(self): return self.step
    def begin_run(self): self.calls.append('begin')
    def bind_pair(self, pair): self.pair = pair

    def image_gradient(self, key, ids):
        if self.fail == ('image', self.pair['index'], key):
            raise RuntimeError('injected image failure')
        P = sft.CONCISE_SEQUENCE_PROMPT_LENGTH
        self.counters.charge_many({'gradient_forwards': 1, 'scorer_vision_embedding_calls': 1, 'scorer_language_model_calls': 1, 'teacherforced_input_positions': P + len(ids) - 1})
        self.counters.charge('backward_calls')
        if self.charge_extra:
            self.counters.charge('backward_calls')
        loss, grad, logps = toy_ce_and_grad(self.W, features(self.pair['rows'][key]), ids)
        if self.mutate:
            self.W[0, 0] += 1.
        if self.consume:
            self.rng.random()
        self.calls.append(('grad', self.pair['index'], key))
        return {'W': np.zeros_like(grad) if self.zero else grad}, {'loss': float(loss), 'response_logps_PRIVATE': [float(x) for x in logps]}

    def scale_gradient(self, tree, w): return {k: (v.astype(np.float32) * np.float32(w)) for k, v in tree.items()}
    def add_scaled(self, total, tree, w): return {k: total[k] + tree[k].astype(np.float32) * np.float32(w) for k in total}
    def gradient_l2(self, tree): return float(math.sqrt(sum(float((v.astype(np.float64) ** 2).sum()) for v in tree.values())))
    def gradient_dtypes(self, tree): return {str(v.dtype) for v in tree.values()}
    def release_graphs(self): self.calls.append('release')

    def apply_update(self, gradient):
        g = gradient['W']
        self.step += 1
        self.m = (.9 * self.m + .1 * g).astype(np.float32)
        self.v = (.999 * self.v + .001 * g * g).astype(np.float32)
        self.W = (self.W - 1e-5 * self.m / (np.sqrt(self.v) + 1e-8)).astype(np.float32)
        self.calls.append(('update', self.step))

    def optimizer_state(self):
        return {'step': np.array(self.step, dtype=np.int64), 'learning_rate': np.array(1e-5, dtype=np.float32), 'W': {'m': self.m, 'v': self.v}}

    def signature(self): return {'W': {'shape': [V, D], 'dtype': 'float32'}}

    def save_sft_checkpoint(self, directory, *, run, completed_pairs):
        if self.fail == ('save', completed_pairs - 1):
            raise OSError('injected save failure')
        return sft.save_checkpoint_files(directory, save_adapter=lambda p: save_arrays(p, {'W': self.W}), optimizer_state=self.optimizer_state(),
                                         rng_state={'python': self.rng.getstate()}, trainable_signature=self.signature(), run=run,
                                         completed_pairs=completed_pairs, save_arrays=save_arrays)

    def load_sft_checkpoint(self, directory, *, run, completed_pairs):
        directory = Path(directory)
        expected = {'run': run, 'adapter_sha256': sft.shared.file_sha256(directory / 'adapter.safetensors'), 'trainable_signature': self.signature()}
        state, rng, _ = load_optimizer_checkpoint(directory / 'optimizer', expected_bindings=json.loads(json.dumps(expected)), load_arrays=load_arrays)
        self.W = load_arrays(directory / 'adapter.safetensors')['W']
        self.step, self.m, self.v = int(state['step'].item()), state['W']['m'], state['W']['v']
        self.rng = random.Random()
        self.rng.setstate(tuple(tuple(x) if isinstance(x, list) else x for x in rng['python']))
        assert self.step == 52 + completed_pairs


def parent(backend): return {'policy': backend.policy_sha(), 'adam': backend.adam_digest(), 'rng': backend.rng_digest()}


def run(backend, out, *, resume=False, before=None, **kw):
    plan, targets, binding = plan_targets()
    return sft.run_sft(backend=backend, plan=plan, targets=targets, binding=binding, output=out, before=before or (lambda name: None),
                       expected_parent=None if resume else parent(backend), resume=resume, **kw)


# ================================================================ import hygiene
def test_importing_the_module_imports_no_mlx_and_builds_no_native_class():
    code = textwrap.dedent('''
        import sys
        sys.path.insert(0, sys.argv[1])
        import mri_rl.final_balanced_sft_v1 as m
        bad = sorted(x for x in sys.modules if x == 'mlx' or x.startswith('mlx.') or x.startswith('mlx_vlm'))
        assert not bad, bad
        assert m._NATIVE_CLASS is None
        assert 'mri_rl.safe_micro_native_v1' not in sys.modules and 'mri_rl.accuracy_pilot_v2_native' not in sys.modules
        ''')
    result = subprocess.run([sys.executable, '-I', '-c', code, SRC], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_lazy_native_class_uses_existing_backend_api_without_importing_mlx():
    code = textwrap.dedent('''
        import sys, ast, inspect, textwrap
        sys.path.insert(0, sys.argv[1])
        import mri_rl.final_balanced_sft_v1 as m
        from mri_rl.safe_micro_native_v1 import MicroNativeBackend
        from mri_rl.accuracy_stage_native_v1 import StageNativeBackend
        from mri_rl.shared_rollout_gspo_backend_v1 import SharedRolloutBackend
        cls = m.native_backend_class()
        assert issubclass(cls, MicroNativeBackend) and m.native_backend_class() is cls
        for name in m.REQUIRED_BACKEND[1:] + ('bind_unit', 'activate', '_context_logps', 'snapshot', 'save_adapter', 'save_arrays',
                                               'load_checkpoint', 'optimizer_state', 'rng_state', 'trainable_signature', 'configure_stage'):
            assert hasattr(cls, name), name
        assert cls.apply_update is StageNativeBackend.apply_update and cls.gradient_l2 is MicroNativeBackend.gradient_l2
        assert cls.__init__ is MicroNativeBackend.__init__                        # capped 12 GiB constructor unchanged
        b = cls.__new__(cls)
        b.configure_sft(prompt_lengths={'mri_sequence': 233}, run={'r': 1})
        expected = ast.parse(textwrap.dedent(inspect.getsource(SharedRolloutBackend._prepare_context))).body[0]
        import hashlib
        assert b.sft_context_ast_sha256 == hashlib.sha256(ast.dump(expected).encode()).hexdigest()
        assert b._sft_prepare.__func__.__globals__['require_prompt_cost']('mri_sequence', 233) == 233
        try:
            b._sft_prepare.__func__.__globals__['require_prompt_cost']('mri_sequence', 213); raise SystemExit('213 accepted')
        except ValueError:
            pass
        bad = sorted(x for x in sys.modules if x == 'mlx' or x.startswith('mlx.') or x.startswith('mlx_vlm'))
        assert not bad, bad
        ''')
    result = subprocess.run([sys.executable, '-I', '-c', code, SRC], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


# ================================================================ plan: deterministic order, leakage, split
def test_plan_is_28_complete_pairs_concise_and_deterministic():
    plan = sft.build_plan(make_rows(), sites(), seed=0, excluded_patient_keys=no_exclusions())
    again = sft.build_plan(make_rows(), sites(), seed=0, excluded_patient_keys=no_exclusions())
    assert plan['plan_sha256'] == again['plan_sha256'] and [p['patient'] for p in plan['pairs']] == [p['patient'] for p in again['pairs']]
    assert len(plan['pairs']) == 28 and sorted(p['patient'] for p in plan['pairs']) == people()
    for pair in plan['pairs']:
        f, t = pair['rows']['F'], pair['rows']['T']
        assert (f['reward_model']['ground_truth'], t['reward_model']['ground_truth']) == ('FLAIR', 'contrast-enhanced T1')
        assert f['extra_info']['patient_key'] == t['extra_info']['patient_key'] == pair['patient']
        assert f['extra_info']['question'] == QUESTION + '\n\n' + CONCISE_INSTRUCTION == t['extra_info']['question']
    assert len({sites()[p['patient']] for p in plan['pairs'][:8]}) == 8          # site-interleaved
    summary = sft.plan_summary_safe(plan)
    assert summary['class_images'] == {'FLAIR': 28, 'contrast-enhanced T1': 28} and 'sub-' not in json.dumps(summary)


def test_seed_changes_only_the_order_and_unknown_seed_is_refused():
    orders = {s: [p['patient'] for p in sft.build_plan(make_rows(), sites(), seed=s, excluded_patient_keys=no_exclusions())['pairs']] for s in sft.ALLOWED_SEEDS}
    assert len({tuple(o) for o in orders.values()}) == 3 and all(sorted(o) == people() for o in orders.values())
    with pytest.raises(ValueError, match='seed'):
        sft.build_plan(make_rows(), sites(), seed=3, excluded_patient_keys=no_exclusions())


@pytest.mark.parametrize('kind', sft.EXCLUSION_KINDS)
def test_excluded_dev_holdout_final100_patient_is_refused(kind):
    excluded = no_exclusions()
    excluded[kind] = {'sub-007'}
    with pytest.raises(ValueError, match='Excluded'):
        sft.build_plan(make_rows(), sites(), seed=0, excluded_patient_keys=excluded)


def test_missing_exclusion_sets_are_refused():
    with pytest.raises(ValueError, match='exclusion'):
        sft.build_plan(make_rows(), sites(), seed=0, excluded_patient_keys={'dev': set()})


def _set(index, path, value):
    def bad(rows):
        node = rows[index]
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
    return bad


@pytest.mark.parametrize('bad,match', [
    (_set(4, ('extra_info', 'internal_split'), 'dev'), 'TRAIN'),
    (_set(4, ('paired_z045_data_v1', 'level'), 91), 'z045'),
    (_set(4, ('paired_z045_data_v1', 'source_record_id_PRIVATE'), 'sub-002_z091_flair_sequence'), 'identity'),
    (_set(5, ('reward_model', 'ground_truth'), 'FLAIR'), 'Unique TRAIN modality'),
    (_set(4, ('extra_info', 'question'), 'other'), 'unchanged original question'),
    (_set(4, ('data_source',), 'other'), 'OpenNeuro'),
])
def test_row_leakage_and_identity_errors_are_refused(bad, match):
    with pytest.raises(ValueError, match=match):
        sft.build_plan(make_rows(bad=bad), sites(), seed=0, excluded_patient_keys=no_exclusions())


def test_missing_patient_pair_is_refused():
    rows = make_rows()[:-1]
    with pytest.raises(ValueError):
        sft.build_plan(rows, sites(), seed=0, excluded_patient_keys=no_exclusions())


# ================================================================ targets, masking and stable CE
def test_targets_pass_the_strict_parser_end_with_eos_and_discriminate_first_label_token():
    targets = sft.build_targets(encode, EOS)
    for label, item in targets['labels'].items():
        assert item['text'] == '<think>Visible MRI evidence.</think><answer>\\boxed{' + label + '}</answer>'
        assert item['target_ids'][-1] == EOS and EOS not in item['target_ids'][:-1] and item['response_tokens'] == len(item['target_ids'])
        assert ''.join(chr(item['target_ids'][p]) for p in item['label_positions']) == label
    with pytest.raises(ValueError, match='EOS'):
        sft.build_targets(encode, ord('<'))


def test_tokenizer_boundary_crossing_is_refused():
    merge = lambda text: encode(text.replace('{F', '\x01'))      # merges prefix end with label start
    with pytest.raises(ValueError, match='boundary'):
        sft.build_targets(merge, EOS)


def test_masked_full_sequence_ce_equals_response_slice_ce_and_is_stable():
    rng = np.random.default_rng(0)
    P, response = 7, [5, 9, 2, 3]
    ids = list(rng.integers(0, 12, size=P)) + response
    logits = rng.normal(size=(len(ids) - 1, 12)) * 1e4                 # extreme scale: stable log-softmax required
    reference = sft.reference_masked_ce(logits.tolist(), ids, P)
    sliced = log_softmax(logits[P - 1:P - 1 + len(response)])[np.arange(len(response)), response]
    assert math.isfinite(reference) and reference == pytest.approx(sft.mean_response_ce(sliced.tolist(), len(response)), rel=1e-9)
    changed = list(ids)
    changed[2] = 11                                                    # change a prompt target: masked, loss unchanged
    logits2 = logits.copy()
    assert sft.reference_masked_ce(logits2.tolist(), changed, P) == pytest.approx(reference)


def test_mean_response_ce_refuses_positive_or_incomplete_logps():
    with pytest.raises(ValueError):
        sft.mean_response_ce([-1., .5], 2)
    with pytest.raises(ValueError):
        sft.mean_response_ce([-1.], 2)
    assert sft.mean_response_ce([-1., -3.], 2) == 2.


def test_budget_is_exact_one_pass():
    targets = sft.build_targets(encode, EOS)
    limits = sft.run_budget(targets)
    lengths = [v['response_tokens'] for v in targets['labels'].values()]
    assert limits['backward_calls'] == limits['gradient_forwards'] == 56 and limits['optimizer_updates'] == 28
    assert limits['generated_response_tokens'] == limits['sampler_draws'] == limits['detached_old_forwards'] == 0
    assert limits['teacherforced_input_positions'] == 28 * sum(233 + n - 1 for n in lengths)


# ================================================================ gradients / normalization
def test_toy_analytic_gradient_matches_finite_differences():
    W = np.random.default_rng(1).normal(scale=.1, size=(V, D))
    x, ids = np.random.default_rng(2).normal(size=D), [60, 61, 62, EOS]
    _, grad, _ = toy_ce_and_grad(W, x, ids)
    for (i, j) in [(60, 0), (61, 3), (EOS, 5), (7, 2)]:
        h = 1e-6
        Wp, Wm = W.copy(), W.copy()
        Wp[i, j] += h
        Wm[i, j] -= h
        numeric = (toy_ce_and_grad(Wp, x, ids)[0] - toy_ce_and_grad(Wm, x, ids)[0]) / (2 * h)
        assert numeric == pytest.approx(grad[i, j], rel=1e-4, abs=1e-7)


def test_one_pair_update_is_adam_on_the_equal_mean_of_two_image_gradients(tmp_path):
    backend = Fake()
    plan, targets, binding = plan_targets()
    W0, m0, v0 = backend.W.copy(), backend.m.copy(), backend.v.copy()
    pair = plan['pairs'][0]
    gF = toy_ce_and_grad(W0, features(pair['rows']['F']), targets['labels']['FLAIR']['target_ids'])[1]
    gT = toy_ce_and_grad(W0, features(pair['rows']['T']), targets['labels']['contrast-enhanced T1']['target_ids'])[1]
    sft.run_sft(backend=backend, plan=plan, targets=targets, binding=binding, output=tmp_path / 'o', before=lambda n: None,
                expected_parent=parent(Fake()), max_pairs=1)
    g = (np.float32(.5) * gF + np.float32(.5) * gT).astype(np.float32)
    m = (.9 * m0 + .1 * g).astype(np.float32)
    v = (.999 * v0 + .001 * g * g).astype(np.float32)
    assert backend.step == 53 and np.array_equal(backend.m, m) and np.array_equal(backend.v, v)
    np.testing.assert_allclose(backend.W, W0 - 1e-5 * m / (np.sqrt(v) + 1e-8), rtol=0, atol=1e-9)
    assert [c for c in backend.calls if isinstance(c, tuple)] == [('grad', 0, 'F'), ('grad', 0, 'T'), ('update', 53)]
    receipt = json.loads((tmp_path / 'o/pairs/pair_0000/receipt_PRIVATE.json').read_text())
    assert receipt['pair_loss'] == pytest.approx(.5 * sum(i['loss'] for i in receipt['images']))


# ================================================================ full synthetic pass, cursor and atomic commits
def test_full_pass_commits_28_chained_checkpoints_and_exact_counters(tmp_path):
    backend = Fake()
    summary = run(backend, tmp_path / 'o')
    assert summary['completed_pairs'] == 28 and summary['adam_step_final'] == 80 and summary['science_claim'] is False
    assert summary['counters']['backward_calls'] == 56 and summary['counters']['optimizer_updates'] == 28
    assert summary['status'] == 'SYNTHETIC_FIXTURE_COMPLETE_NOT_NATIVE_EVIDENCE'
    plan, _, binding = plan_targets()
    cursor = sft.committed_cursor(tmp_path / 'o', binding_sha256=sft.shared.identity(binding))
    assert cursor['completed_pairs'] == 28 and cursor['expected_adam_step'] == 80
    for i in range(28):
        directory = tmp_path / 'o/pairs' / ('pair_%04d' % i) / 'checkpoint_PRIVATE'
        c = json.loads((directory / 'cursor.json').read_text())
        assert c == {'seed': 0, 'plan_sha256': plan['plan_sha256'], 'completed_pairs': i + 1, 'next_pair_index': i + 1, 'adam_step': 53 + i}
        manifest = json.loads((directory / 'optimizer/manifest.json').read_text())
        assert manifest['optimizer_updates'] == 53 + i and manifest['completed_groups'] == 115 + i
        assert set(manifest['bindings']) == {'run', 'adapter_sha256', 'trainable_signature'}      # StageNativeBackend.load_checkpoint schema
    assert not [p for p in (tmp_path / 'o/pairs').iterdir() if p.name.startswith('.')]


def test_resume_after_boundary_failure_reproduces_the_uninterrupted_pass(tmp_path):
    reference = Fake()
    run(reference, tmp_path / 'ref')

    def halt(name):
        if name == 'sft_pair_0005_start':
            raise RuntimeError('injected resource stop at a safe boundary')
    first = Fake()
    with pytest.raises(RuntimeError, match='resource stop'):
        run(first, tmp_path / 'o', before=halt)
    assert list((tmp_path / 'o').glob('STOP_*_PRIVATE.json'))
    plan, _, binding = plan_targets()
    cursor = sft.committed_cursor(tmp_path / 'o', binding_sha256=sft.shared.identity(binding))
    assert cursor['completed_pairs'] == 5 and cursor['next_pair_index'] == 5
    fresh = Fake()                                                      # new process: load last committed checkpoint
    fresh.load_sft_checkpoint(cursor['checkpoint'], run=binding, completed_pairs=5)
    summary = run(fresh, tmp_path / 'o', resume=True)
    assert summary['completed_pairs'] == 28 and summary['counters']['backward_calls'] == 56
    assert np.array_equal(fresh.W, reference.W) and np.array_equal(fresh.m, reference.m) and fresh.step == reference.step == 80


def test_mid_pair_failure_spends_budget_and_resume_is_refused_up_front(tmp_path):
    first = Fake(fail=('image', 5, 'T'))
    with pytest.raises(RuntimeError, match='injected image'):
        run(first, tmp_path / 'o')
    _, _, binding = plan_targets()
    cursor = sft.committed_cursor(tmp_path / 'o', binding_sha256=sft.shared.identity(binding))
    assert cursor['completed_pairs'] == 5
    used = json.loads((tmp_path / 'o/counters_PRIVATE.json').read_text())['used']
    assert used['backward_calls'] == 11 and used['optimizer_updates'] == 5            # pair 5 FLAIR backward is spent, not refunded
    fresh = Fake()
    fresh.load_sft_checkpoint(cursor['checkpoint'], run=binding, completed_pairs=5)
    with pytest.raises(ValueError, match='cannot complete the pass'):
        run(fresh, tmp_path / 'o', resume=True)


def test_save_failure_leaves_no_partial_commit(tmp_path):
    backend = Fake(fail=('save', 3))
    with pytest.raises(OSError):
        run(backend, tmp_path / 'o')
    names = sorted(p.name for p in (tmp_path / 'o/pairs').iterdir())
    assert names == ['pair_0000', 'pair_0001', 'pair_0002']


def test_indeterminate_or_tampered_commits_are_refused(tmp_path):
    run(Fake(), tmp_path / 'o', max_pairs=3)
    _, _, binding = plan_targets()
    sha = sft.shared.identity(binding)
    (tmp_path / 'o/pairs/.pending-x').mkdir()
    with pytest.raises(ValueError, match='INDETERMINATE'):
        sft.committed_cursor(tmp_path / 'o', binding_sha256=sha)
    (tmp_path / 'o/pairs/.pending-x').rmdir()
    (tmp_path / 'o/pairs/pair_0002/receipt_PRIVATE.json').rename(tmp_path / 'moved.json')
    with pytest.raises(ValueError, match='without receipt'):
        sft.committed_cursor(tmp_path / 'o', binding_sha256=sha)
    (tmp_path / 'moved.json').rename(tmp_path / 'o/pairs/pair_0002/receipt_PRIVATE.json')
    with (tmp_path / 'o/pairs/pair_0001/checkpoint_PRIVATE/adapter.safetensors').open('ab') as handle:
        handle.write(b'x')
    with pytest.raises(ValueError, match='payload changed'):
        sft.committed_cursor(tmp_path / 'o', binding_sha256=sha)
    with pytest.raises(ValueError, match='chain/binding'):
        sft.committed_cursor(tmp_path / 'o', binding_sha256='c' * 64)


# ================================================================ refusals
def test_refuses_non_parent_state_existing_output_and_native_without_authority(tmp_path):
    backend = Fake()
    expected = parent(backend)
    backend.W[0, 0] += 1
    plan, targets, binding = plan_targets()
    with pytest.raises(ValueError, match='trusted Adam52 parent'):
        sft.run_sft(backend=backend, plan=plan, targets=targets, binding=binding, output=tmp_path / 'a', before=lambda n: None, expected_parent=expected)
    (tmp_path / 'b').mkdir()
    with pytest.raises(FileExistsError):
        run(Fake(), tmp_path / 'b')
    native = Fake()
    native.evidence_kind = sft.NATIVE_KIND
    with pytest.raises(ValueError, match='checked root authorization'):
        run(native, tmp_path / 'c')
    with pytest.raises(ValueError, match='Pair cap'):
        run(Fake(), tmp_path / 'd', max_pairs=29)


@pytest.mark.parametrize('kwargs,match', [({'mutate_in_backward': True}, 'mutated'), ({'consume_rng': True}, 'mutated'),
                                          ({'zero': True}, 'nonzero'), ({'charge_extra': True}, 'budget exhausted')])
def test_backward_mutation_rng_consumption_zero_gradient_and_budget_overrun_stop(tmp_path, kwargs, match):
    with pytest.raises(ValueError, match=match):
        run(Fake(**kwargs), tmp_path / 'o')
    assert list((tmp_path / 'o').glob('STOP_*_PRIVATE.json'))


def test_binding_must_match_plan_targets_and_prompt(tmp_path):
    plan, targets, binding = plan_targets()
    other = sft.build_plan(make_rows(), sites(), seed=1, excluded_patient_keys=no_exclusions())
    with pytest.raises(ValueError, match='Run binding'):
        sft.run_sft(backend=Fake(), plan=other, targets=targets, binding=binding, output=tmp_path / 'o', before=lambda n: None, expected_parent=parent(Fake()))
    with pytest.raises(ValueError, match='Run binding'):
        sft.run_sft(backend=Fake(), plan=plan, targets=targets, binding=binding, output=tmp_path / 'o', before=lambda n: None,
                    expected_parent=parent(Fake()), prompt_length=213)


# ================================================================ native bridge (source only; fail closed; numpy shim)
def test_native_authority_fails_closed_without_root_entry(tmp_path, monkeypatch):
    built = []
    monkeypatch.setattr(sft, 'native_backend_class', lambda: built.append(1))
    gate = lambda name: None
    args = dict(resource_gate=gate, adapter=tmp_path / 'a', config={}, model_path=tmp_path, contract=None,
                cache={'dir': str(tmp_path)}, prompt_lengths={'mri_sequence': 233}, run={})
    with monkeypatch.context() as patch:
        patch.setattr(sft, 'ENTRY_MODULE', 'mri_rl.final_balanced_sft_entry_absent_for_test')
        with pytest.raises(sft.PendingAuthorization, match='absent'):
            sft.make_native_backend(authorization=object(), **args)
    with pytest.raises((sft.PendingAuthorization, ValueError)):          # whether or not a root entry exists: a forged object is refused
        sft.make_native_backend(authorization=object(), **args)
    assert built == []


def test_native_authority_accepts_only_exact_issued_scope(monkeypatch):
    class Authorization:
        def __init__(self, scope): self.scope, self.checked = scope, []
        def check(self, name): self.checked.append(name)
    entry = types.ModuleType(sft.ENTRY_MODULE)
    entry.Authorization = Authorization
    monkeypatch.setitem(sys.modules, sft.ENTRY_MODULE, entry)
    monkeypatch.setattr(sft, '_entry_authorization_class', lambda: Authorization)
    good = Authorization({'protocol': sft.PROTOCOL, 'execution_stage': 'SFT_TRAIN'})
    class Gate:
        def __init__(self, authorization): self.authorization = authorization
        def __call__(self, name): return None
    gate = Gate(good)
    with pytest.raises(ValueError, match='Resource gate'):
        sft.check_sft_authority(good, Gate(Authorization(good.scope)))
    assert isinstance(sft.check_sft_authority(good, gate), sft._CheckedAuthority) and good.checked == ['final_balanced_sft_native_authority']
    with pytest.raises(ValueError, match='not this SFT phase'):
        sft.check_sft_authority(Authorization({'protocol': 'GSPO', 'execution_stage': 'SFT_TRAIN'}), gate)
    with pytest.raises(ValueError, match='issued by the root'):
        sft.check_sft_authority(object(), gate)


class _Arr:
    def __init__(self, value): self.value = np.asarray(value, dtype=np.float64)
    def tolist(self): return self.value.tolist()
    def item(self): return float(self.value)


def test_native_image_gradient_uses_one_own_context_scorer_and_mean_response_ce():
    W0 = np.random.default_rng(3).normal(scale=.1, size=(V, D))
    x, ids = np.random.default_rng(4).normal(size=D), [70, 71, 72, EOS]
    events = []
    model = types.SimpleNamespace(W=W0.copy(), eval=lambda: events.append('eval'))

    def context_logps(context, tokens, counter):
        events.append(('score', context, counter))
        _, logits = toy_terms(model.W, x, tokens)
        return _Arr(log_softmax(logits)[np.arange(len(tokens)), tokens])

    def value_and_grad(m, fn):
        def call(mod, tokens):
            loss = fn(mod, tokens)
            grad = toy_ce_and_grad(m.W, x, tokens)[1]                     # analytic gradient of the same mean CE
            return _Arr(loss.value), {'W': grad}
        return call
    mx = types.SimpleNamespace(stop_gradient=lambda a: a, mean=lambda a: _Arr(a.value.mean()), eval=lambda *a: None)
    mx_neg = lambda a: _Arr(-a.value)
    _Arr.__neg__ = lambda self: mx_neg(self)
    counters = sft.shared.Counters({**{k: 0 for k in sft.shared.CHARGEABLE}, 'backward_calls': 1})
    backend = types.SimpleNamespace(mx=mx, nn=types.SimpleNamespace(value_and_grad=value_and_grad), model=model, counters=counters,
                                    activate=lambda key: events.append(('activate', key)) or 'ctx-' + key, _context_logps=context_logps,
                                    before_unit=lambda name: events.append(('before', name)), _checked_gradient=lambda g: float(np.linalg.norm(g['W'])))
    gradient, info = sft.native_image_gradient(backend, 'T', ids)
    assert events[:4] == [('activate', 'T'), 'eval', ('before', 'final_sft_image_backward'), ('score', 'ctx-T', 'gradient_forwards')]
    assert sum(1 for e in events if isinstance(e, tuple) and e[0] == 'score') == 1 and counters.used['backward_calls'] == 1
    assert info['loss'] == pytest.approx(sft.mean_response_ce(info['response_logps_PRIVATE'], len(ids)))
    assert info['loss'] == pytest.approx(toy_ce_and_grad(W0, x, ids)[0]) and len(info['response_logps_PRIVATE']) == len(ids)
    with pytest.raises(ValueError, match='budget'):
        sft.native_image_gradient(backend, 'T', ids)                     # second backward exceeds the counter before running


def test_checkpoint_files_round_trip_through_existing_optimizer_checkpoint(tmp_path):
    backend = Fake()
    backend.apply_update({'W': np.ones((V, D), dtype=np.float32)})
    _, _, binding = plan_targets()
    (tmp_path / 'c').mkdir()
    (tmp_path / 'd').mkdir()
    names = backend.save_sft_checkpoint(tmp_path / 'c', run=binding, completed_pairs=1)
    assert names == ['adapter.safetensors', 'optimizer/manifest.json', 'optimizer/state.safetensors']
    other = Fake()
    other.load_sft_checkpoint(tmp_path / 'c', run=binding, completed_pairs=1)
    assert parent(other) == parent(backend)
    with pytest.raises(ValueError, match='52 \\+ completed'):
        backend.save_sft_checkpoint(tmp_path / 'd', run=binding, completed_pairs=2)
    with pytest.raises(ValueError, match='binding'):
        Fake().load_sft_checkpoint(tmp_path / 'c', run={**binding, 'seed': 1}, completed_pairs=1)
