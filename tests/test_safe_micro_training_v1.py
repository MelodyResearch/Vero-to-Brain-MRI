"""Synthetic CPU tests for the safe-micro joint transaction and lazy native bridge. No MLX, model, MRI data or native evidence."""
import ast
import inspect
import json
import math
from pathlib import Path
import subprocess
import sys
import textwrap
import types
import numpy as np
import pytest
from test_multistep_gspo_v1 import FakeBackend, make_plan, LENGTHS, REWARD, sha
from mri_rl import multistep_gspo_core_v1 as ms
from mri_rl import safe_micro_training_v1 as stm

SRC = str(Path(__file__).resolve().parents[1] / 'src')
IDS = [f'{k}{i}' for k in 'FT' for i in range(4)] + [f'{k}_{i}' for k in ('D1', 'D2') for i in range(4)]
MIXED = {**dict(zip(IDS[:4], [1, 0, 1, 0])), **dict(zip(IDS[4:8], [1, 1, 0, 0])), **dict(zip(IDS[8:12], [1, 0, 0, 0])), **dict(zip(IDS[12:], [0, 1, 1, 0]))}
DALLAS_INACTIVE = {**MIXED, **{i: 1 for i in IDS[8:12]}, **{i: 0 for i in IDS[12:]}}


class MicroFake(FakeBackend):
    evidence_kind = stm.SYNTHETIC_KIND

    def __init__(self, script=None, **kwargs):
        super().__init__(**kwargs)
        self.step, self.m, self.v = 52, np.full(4, .01), np.full(4, .02)
        self.script, self.apply_calls, self.save_fail, self.apply_fail = script, 0, False, False

    def generate(self, unit, prompt_lengths):
        raw = super().generate(unit, prompt_lengths)
        for r in raw['responses'] if self.script else ():
            y = bool(self.script[r['response_id']])
            p = self.p_correct(r['context_key'], unit)
            r['grade'] = {**r['grade'], 'accuracy': float(y), 'raw_accuracy': float(y), 'score': float(y), 'total': float(y)}
            r['raw_selected_logps'] = [math.log(p if y else 1 - p), -.01]
        return raw

    def gradient(self, planned, own_old, unit):
        tree, info = super().gradient(planned, own_old, unit)
        return (None if tree is None else tree.astype(np.float32)), info

    def scale_gradient(self, tree, weight):
        return (tree * weight).astype(np.float32)

    def add_scaled(self, total, tree, weight):
        return (total + tree * weight).astype(np.float32)

    def gradient_l2(self, tree):
        return float(np.linalg.norm(tree))

    def gradient_dtypes(self, tree):
        return {str(tree.dtype)}

    def save_gradient(self, path, tree):
        with open(path, 'wb') as handle:
            np.save(handle, tree)

    def release_graphs(self):
        pass

    def optimizer_step(self):
        return self.step

    def apply_update(self, gradient, lr=0.05):
        if self.apply_fail:
            raise RuntimeError('simulated optimizer failure')
        self.apply_calls += 1
        return super().apply_update(gradient, lr)

    def save_checkpoint(self, directory):
        if self.save_fail:
            raise RuntimeError('simulated checkpoint failure')
        return super().save_checkpoint(directory)


def units():
    return ms.make_units(make_plan(1))[:2]


def parent_of(b):
    return {'policy': b.policy_sha(), 'adam': b.adam_digest(), 'rng': b.rng_digest()}


def run(tmp_path, backend=None, name='out', unit_list=None, before=None, **extra):
    backend = backend or MicroFake(MIXED)
    expected = parent_of(backend)
    summary = stm.execute_micro(backend=backend, units=unit_list or units(), prompt_lengths=LENGTHS, reward_policy_sha256=REWARD, output=tmp_path / name,
                                before=before or (lambda name: None), expected_parent=expected, **extra)
    return backend, expected, summary, tmp_path / name


def read(path):
    return json.loads(Path(path).read_text())


# ---------------------------------------------------------------- budget
def test_fixed_caps_and_tf_accounting():
    b = stm.micro_budget(units(), LENGTHS)
    t = b['total']
    assert (t['generated_response_tokens'], t['detached_old_forwards'], t['gradient_forwards'], t['backward_calls'], t['optimizer_updates']) == (4096, 16, 32, 32, 2)
    assert (b['common']['teacherforced_input_positions'], 2 * b['arm']['teacherforced_input_positions'], t['teacherforced_input_positions']) == (9024, 18048, 27072)
    assert b['arm']['optimizer_updates'] == 1 and b['arm']['generated_response_tokens'] == 0 and b['common']['optimizer_updates'] == 0
    assert b['automatic_retry'] is False and b['unused_budget_transfer'] is False


@pytest.mark.parametrize('bad', [lambda u: u[:1], lambda u: u[::-1], lambda u: ms.make_units(make_plan(2))])
def test_only_the_first_whole_round_is_accepted(bad):
    with pytest.raises(ValueError):
        stm.micro_budget(bad(units()), LENGTHS)


def test_second_update_in_one_arm_is_refused_by_the_ledger(tmp_path):
    ledger = stm.Ledger(stm.micro_budget(units(), LENGTHS), tmp_path / 'c.json')
    ledger.phase('A').charge('optimizer_updates')
    with pytest.raises(ValueError):
        ledger.phase('A').charge('optimizer_updates')
    ledger.phase('B').charge('optimizer_updates')
    assert read(tmp_path / 'c.json')['used']['A']['optimizer_updates'] == 1


# ---------------------------------------------------------------- same parents / common once
def test_common_16_generated_once_both_arms_share_parent_and_bundles(tmp_path):
    backend, expected, summary, out = run(tmp_path)
    assert backend.generate_calls == 2                                  # F4/T4 and D1/D2 once each; arms never generate
    assert parent_of(backend) == expected and backend.step == 52         # parent preserved after both arms
    receipts = {a: read(out / f'arm_{a}/receipt_PRIVATE.json') for a in 'AB'}
    assert receipts['A']['bundle_sha256'] == receipts['B']['bundle_sha256'] and len(receipts['A']['bundle_sha256']) == 2
    assert receipts['A']['parent_policy_sha256'] == receipts['B']['parent_policy_sha256'] == expected['policy']
    assert receipts['A']['common_receipt_sha256'] == receipts['B']['common_receipt_sha256'] == summary['common_receipt_sha256']
    assert read(out / 'common_PRIVATE/common_record_PRIVATE.json')['responses'] == 16
    a_in, b_in = (read(out / f'arm_{a}/inputs_PRIVATE.json') for a in 'AB')
    assert a_in['own_old'] == b_in['own_old'] and a_in['rewards'] == b_in['rewards'] and len(a_in['rewards']) == 16


def test_arm_groupings_sequence_differs_dallas_unchanged(tmp_path):
    _, _, _, out = run(tmp_path)
    blocks = {a: read(out / f'arm_{a}/gradient_evidence_PRIVATE.json')['blocks'] for a in 'AB'}
    assert blocks['A'][0]['plan']['groups'] == [['F0', 'F1', 'F2', 'F3'], ['T0', 'T1', 'T2', 'T3']]
    assert sorted(map(sorted, blocks['B'][0]['plan']['groups'])) == sorted(map(sorted, [['F0', 'F1', 'T0', 'T1'], ['F2', 'F3', 'T2', 'T3']]))
    dallas = [['D1_0', 'D1_1', 'D1_2', 'D1_3'], ['D2_0', 'D2_1', 'D2_2', 'D2_3']]
    assert blocks['A'][1]['plan']['groups'] == blocks['B'][1]['plan']['groups'] == dallas
    assert all(i['weight'] in (0., 1 / 8) for a in 'AB' for b in blocks[a] for i in b['plan']['items'])


# ---------------------------------------------------------------- one joint step
def expected_joint(out, arm, units_, parent_backend):
    evidence = read(out / f'arm_{arm}/gradient_evidence_PRIVATE.json')
    rewards = read(out / f'arm_{arm}/inputs_PRIVATE.json')['rewards']
    total = np.zeros(4)
    for block, unit in zip(evidence['blocks'], units_):
        for item in block['plan']['items']:
            if not item['admitted']:
                continue
            rid = item['response_id']
            key = rid[0] if rid[0] in 'FT' else rid.split('_')[0]
            p = parent_backend.p_correct(key, unit)
            y = rewards[rid] == 1.0
            assert item['weight'] == 1 / 8
            total += (1 / 16) * (-item['advantage']) * (1 - p if y else -p) * parent_backend.phi(key, unit['kind'])
    return total


def test_exactly_one_joint_step_equals_per_response_one_sixteenth(tmp_path):
    us = units()
    backend, expected, summary, out = run(tmp_path, unit_list=us)
    assert backend.apply_calls == 2 and summary['optimizer_updates'] == 2
    for arm in 'AB':
        receipt = read(out / f'arm_{arm}/receipt_PRIVATE.json')
        assert receipt['updated'] is True and receipt['adam_step_after'] == 53 and receipt['alias_of_parent'] is False
        want = expected_joint(out, arm, us, MicroFake())
        got = np.load(out / f'arm_{arm}/joint_gradient_float32_PRIVATE.safetensors')
        assert got.dtype == np.float32 and np.allclose(got, want, atol=1e-6) and np.linalg.norm(want) > 0
        ref = MicroFake(); ref.apply_update(want.astype(np.float32))
        state = np.load(out / f'arm_{arm}/checkpoint_PRIVATE/state.npz')
        assert int(state['step']) == 53 and np.allclose(state['theta'], ref.theta, atol=1e-5)


def test_inactive_block_is_not_reweighted(tmp_path):
    us = units()
    _, _, _, out = run(tmp_path, MicroFake(DALLAS_INACTIVE), unit_list=us)
    for arm in 'AB':
        evidence = read(out / f'arm_{arm}/gradient_evidence_PRIVATE.json')
        assert evidence['blocks'][1]['plan']['effective_groups'] == 0 and evidence['blocks'][0]['plan']['effective_groups'] >= 1
        got = np.load(out / f'arm_{arm}/joint_gradient_float32_PRIVATE.safetensors')
        want = expected_joint(out, arm, us, MicroFake())                      # still 1/16 per admitted response, not 1/8
        assert np.allclose(got, want, atol=1e-6)
        seq_only_full_weight = want * 2
        assert not np.allclose(got, seq_only_full_weight, atol=1e-6)


def test_no_gspo_signal_means_zero_updates_no_momentum_step_no_resample(tmp_path):
    backend, expected, summary, out = run(tmp_path, MicroFake(always=1))   # every group zero-variance
    assert summary['optimizer_updates'] == 0 and backend.apply_calls == 0 and backend.generate_calls == 2
    assert parent_of(backend) == expected and backend.step == 52
    for arm in 'AB':
        receipt = read(out / f'arm_{arm}/receipt_PRIVATE.json')
        assert receipt['updated'] is False and receipt['alias_of_parent'] and receipt['checkpoint'] is None and receipt['update_decision']['reason'] == 'NO_ADMITTED_GROUP'
        assert receipt['adam_step_after'] == 52 and receipt['rng_terminal'].startswith('R0')
        assert not (out / f'arm_{arm}/checkpoint_PRIVATE').exists() and not (out / f'arm_{arm}/joint_gradient_float32_PRIVATE.safetensors').exists()
    assert read(out / 'counters_PRIVATE.json')['used']['A']['optimizer_updates'] == 0


def test_updated_arms_end_at_the_persisted_post_all16_generation_rng(tmp_path):
    _, expected, summary, out = run(tmp_path)
    common = read(out / 'common_PRIVATE/common_record_PRIVATE.json')
    assert common['rng_after_all16_digest'] == summary['rng_after_all16_digest'] != expected['rng']
    for arm in 'AB':
        receipt = read(out / f'arm_{arm}/receipt_PRIVATE.json')
        assert receipt['rng_digest_after'] == common['rng_after_all16_digest'] and receipt['future_onpolicy_sharing'] is False


# ---------------------------------------------------------------- rollback / spent counters / atomic evidence
def test_failure_rolls_back_keeps_spent_counters_and_cannot_be_retried(tmp_path):
    backend = MicroFake(MIXED); backend.fault = 'grad_params'
    expected = parent_of(MicroFake(MIXED))
    with pytest.raises(ValueError, match='Backward mutated'):
        stm.execute_micro(backend=backend, units=units(), prompt_lengths=LENGTHS, reward_policy_sha256=REWARD, output=tmp_path / 'o',
                          before=lambda n: None, expected_parent=expected)
    assert parent_of(backend) == expected and backend.step == 52
    out = tmp_path / 'o'
    assert not (out / 'arm_B').exists() and not (out / 'arm_A/receipt_PRIVATE.json').exists() and not (out / 'micro_complete_SAFE.json').exists()
    stop = read(next(out.glob('STOP_*')))
    assert stop['rolled_back_to_parent'] is True and stop['automatic_retry'] is False
    assert read(out / 'counters_PRIVATE.json')['used']['A']['backward_calls'] > 0 and stop['counters_spent_not_refunded']['A']['backward_calls'] > 0
    backend.fault = None
    with pytest.raises(FileExistsError):
        stm.execute_micro(backend=backend, units=units(), prompt_lengths=LENGTHS, reward_policy_sha256=REWARD, output=out,
                          before=lambda n: None, expected_parent=expected)


def test_optimizer_failure_after_charge_is_rolled_back_and_stays_spent(tmp_path):
    backend = MicroFake(MIXED); backend.apply_fail = True
    expected = parent_of(MicroFake(MIXED))
    with pytest.raises(RuntimeError, match='optimizer'):
        stm.execute_micro(backend=backend, units=units(), prompt_lengths=LENGTHS, reward_policy_sha256=REWARD, output=tmp_path / 'o',
                          before=lambda n: None, expected_parent=expected)
    assert parent_of(backend) == expected and backend.step == 52
    assert read(tmp_path / 'o/counters_PRIVATE.json')['used']['A']['optimizer_updates'] == 1


def test_checkpoint_failure_commits_nothing_and_receipt_follows_checkpoint(tmp_path, monkeypatch):
    backend = MicroFake(MIXED); backend.save_fail = True
    expected = parent_of(MicroFake(MIXED))
    with pytest.raises(RuntimeError, match='checkpoint'):
        stm.execute_micro(backend=backend, units=units(), prompt_lengths=LENGTHS, reward_policy_sha256=REWARD, output=tmp_path / 'o',
                          before=lambda n: None, expected_parent=expected)
    a = tmp_path / 'o/arm_A'
    assert not (a / 'checkpoint_PRIVATE').exists() and not (a / 'receipt_PRIVATE.json').exists() and parent_of(backend) == expected
    # Receipt publication failing after a committed checkpoint leaves an UNRECEIPTED checkpoint (not evidence) and a rolled-back backend.
    real = stm.shared.publish_once
    def flaky(path, value):
        if Path(path).name == 'receipt_PRIVATE.json':
            assert (Path(path).parent / 'checkpoint_PRIVATE/state.npz').exists()       # checkpoint committed BEFORE the receipt
            raise OSError('receipt write failed')
        return real(path, value)
    monkeypatch.setattr(stm.shared, 'publish_once', flaky)
    backend2 = MicroFake(MIXED)
    with pytest.raises(OSError):
        stm.execute_micro(backend=backend2, units=units(), prompt_lengths=LENGTHS, reward_policy_sha256=REWARD, output=tmp_path / 'p',
                          before=lambda n: None, expected_parent=parent_of(MicroFake(MIXED)))
    assert (tmp_path / 'p/arm_A/checkpoint_PRIVATE').exists() and not (tmp_path / 'p/arm_A/receipt_PRIVATE.json').exists()
    assert parent_of(backend2) == parent_of(MicroFake(MIXED)) and backend2.step == 52


def test_receipt_hashes_cover_the_committed_checkpoint(tmp_path):
    _, _, _, out = run(tmp_path)
    for arm in 'AB':
        receipt = read(out / f'arm_{arm}/receipt_PRIVATE.json')
        assert set(receipt['files_sha256']) == {'state.npz', 'rng.json'}
        for name, digest in receipt['files_sha256'].items():
            assert stm.shared.file_sha256(out / f'arm_{arm}/checkpoint_PRIVATE' / name) == digest
        body = {k: v for k, v in receipt.items() if k != 'receipt_sha256'}
        assert receipt['receipt_sha256'] == stm.identity(body)


# ---------------------------------------------------------------- fail-closed inputs / no progression
def test_guard_is_called_at_boundaries_and_can_stop(tmp_path):
    seen = []
    run(tmp_path, before=seen.append)
    assert 'micro_common_generate_unit_0' in seen and 'micro_arm_A_before_update' in seen and 'micro_arm_B_checkpoint' in seen
    expected = parent_of(MicroFake(MIXED))
    backend = MicroFake(MIXED)
    def stop(name):
        if name == 'micro_arm_B_restore_parent':
            raise RuntimeError('HOLD')
    with pytest.raises(RuntimeError, match='HOLD'):
        stm.execute_micro(backend=backend, units=units(), prompt_lengths=LENGTHS, reward_policy_sha256=REWARD, output=tmp_path / 'g', before=stop, expected_parent=expected)
    assert parent_of(backend) == expected and (tmp_path / 'g/arm_A/receipt_PRIVATE.json').exists() and not (tmp_path / 'g/arm_B/receipt_PRIVATE.json').exists()


def test_refuses_wrong_parent_missing_protocol_and_non_float32(tmp_path):
    b = MicroFake(MIXED)
    with pytest.raises(ValueError, match='Adam52 common parent'):
        stm.execute_micro(backend=b, units=units(), prompt_lengths=LENGTHS, reward_policy_sha256=REWARD, output=tmp_path / 'w', before=lambda n: None,
                          expected_parent={**parent_of(b), 'adam': sha('other')})
    assert b.generate_calls == 0
    class Lacking(MicroFake):
        def __getattribute__(self, name):
            if name == 'scale_gradient':
                raise AttributeError(name)
            return super().__getattribute__(name)
    lacking = Lacking(MIXED)
    with pytest.raises(ValueError, match='scale_gradient'):
        stm.execute_micro(backend=lacking, units=units(), prompt_lengths=LENGTHS, reward_policy_sha256=REWARD, output=tmp_path / 'x', before=lambda n: None, expected_parent=parent_of(b))
    class Wide(MicroFake):
        def gradient_dtypes(self, tree):
            return {'float64'}
    with pytest.raises(ValueError, match='float32'):
        stm.execute_micro(backend=Wide(MIXED), units=units(), prompt_lengths=LENGTHS, reward_policy_sha256=REWARD, output=tmp_path / 'y', before=lambda n: None, expected_parent=parent_of(b))
    with pytest.raises(ValueError, match='guard'):
        stm.execute_micro(backend=b, units=units(), prompt_lengths=LENGTHS, reward_policy_sha256=REWARD, output=tmp_path / 'z', before=None, expected_parent=parent_of(b))


def test_synthetic_run_is_never_labeled_native_and_never_advances_stage(tmp_path):
    _, _, summary, out = run(tmp_path)
    assert summary['status'] == 'SYNTHETIC_FIXTURE_COMPLETE_NOT_NATIVE_EVIDENCE' and summary['evidence_kind'] == stm.SYNTHETIC_KIND
    assert summary['next_stage_authorized'] is False and summary['small_pilot_authorized'] is False and summary['dev24_authorized'] is False
    assert summary['automatic_stage_progression'] is False and summary['science_claim'] is False and summary['parent_preserved'] is True
    assert not any(hasattr(stm, n) for n in ('run_stage', 'run_small_pilot', 'run_dev24'))
    assert sorted(p.name for p in out.iterdir()) == ['arm_A', 'arm_B', 'common_PRIVATE', 'counters_PRIVATE.json', 'micro_complete_SAFE.json', 'start_PRIVATE.json']
    spent = summary['spent_total']
    assert spent['optimizer_updates'] == 2 and spent['generated_response_tokens'] <= 4096 and spent['backward_calls'] <= 32 and spent['gradient_forwards'] <= 32


# ================================================================ native bridge (source only; no MLX)
def native():
    import mri_rl.safe_micro_native_v1 as bridge
    return bridge


def test_importing_the_bridge_imports_no_mlx():
    code = textwrap.dedent('''
        import sys
        sys.path.insert(0, sys.argv[1])
        import mri_rl.safe_micro_native_v1, mri_rl.safe_micro_training_v1
        bad = sorted(m for m in sys.modules if m == 'mlx' or m.startswith('mlx.') or m.startswith('mlx_vlm'))
        assert not bad, bad
        ''')
    result = subprocess.run([sys.executable, '-I', '-c', code, SRC], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_capped_init_is_the_unchanged_ast_in_a_distinct_copied_namespace():
    bridge = native()
    import mri_rl.mlx_continuous_gspo as original
    binding = bridge.capped_init_binding()
    expected = ast.parse(textwrap.dedent(inspect.getsource(original.NativeGSPOBackend.__init__))).body[0]
    assert ast.dump(binding['node']) == ast.dump(expected)
    import hashlib
    assert binding['ast_sha256'] == hashlib.sha256(ast.dump(expected).encode()).hexdigest()
    assert bridge.shared.file_sha256(original.__file__) == bridge.SOURCE_FILE_SHA256 == binding['source_sha256']
    glob = binding['init'].__globals__
    assert glob is not vars(original) and glob['PACED_POLICY'] is not original.PACED_POLICY
    assert (glob['PACED_POLICY']['memory_guideline_gib'], glob['PACED_POLICY']['cache_limit_mib']) == (12, 256)
    assert original.PACED_POLICY['memory_guideline_gib'] == 18                                    # historical global never mutated
    assert {k: v for k, v in glob['PACED_POLICY'].items() if k not in ('memory_guideline_gib', 'cache_limit_mib')} == \
           {k: v for k, v in original.PACED_POLICY.items() if k not in ('memory_guideline_gib', 'cache_limit_mib')}
    assert all(glob[k] is v for k, v in vars(original).items() if k != 'PACED_POLICY')


def test_stage_backend_init_is_still_only_parent_init_plus_generate_alias_pop():
    from mri_rl.accuracy_stage_native_v1 import StageNativeBackend
    body = ast.parse(textwrap.dedent(inspect.getsource(StageNativeBackend.__init__))).body[0].body
    assert [ast.unparse(n) for n in body] == ['super().__init__(*args, **kwargs)', "self._legacy_generation_api = self.__dict__.pop('generate', None)"]


def test_source_drift_is_refused(monkeypatch):
    bridge = native()
    monkeypatch.setattr(bridge, 'SOURCE_FILE_SHA256', '0' * 64)
    with pytest.raises(ValueError, match='pinned reviewed source'):
        bridge.capped_init_binding()


class _Halt(BaseException):
    pass


def fake_runtime(monkeypatch, events):
    def mod(name, **attrs):
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
        return module
    mx = mod('mlx.core', set_memory_limit=lambda v: events.append(('memory', v)), set_cache_limit=lambda v: events.append(('cache', v)),
             clear_cache=lambda: None, random=types.SimpleNamespace(seed=lambda s: None))
    def load(path, lazy=True):
        events.append(('load',))
        raise _Halt()
    mod('mlx', core=mx)
    mod('mlx.nn'); mod('mlx.optimizers'); mod('mlx.utils', tree_flatten=lambda t: [], tree_map=lambda f, t: t)
    mod('mlx_vlm', generate=lambda *a, **k: None, load=load); mod('mlx_vlm.prompt_utils', apply_chat_template=None)
    mod('mlx_vlm.trainer'); mod('mlx_vlm.trainer.common'); mod('mlx_vlm.trainer.common.model', prepare_model_for_training=None)
    mod('mlx_vlm.trainer.common.utils', save_trainable_weights=None, enable_gradient_checkpointing=None)
    mod('mlx_vlm.utils', prepare_inputs=None, should_add_special_tokens=None)


def test_constructor_applies_12gib_and_256mib_before_any_load_or_materialization(tmp_path, monkeypatch):
    bridge = native()
    import mri_rl.paced_resources as paced
    events = []
    fake_runtime(monkeypatch, events)
    adapter = tmp_path / 'adapter.safetensors'; adapter.write_bytes(b'x')
    before = []
    with pytest.raises(_Halt):
        bridge.MicroNativeBackend(adapter, {}, model_path=tmp_path, contract=None, seed=13, before_unit=before.append, cache={'dir': str(tmp_path)})
    assert before[0] == 'native_model_load'
    assert events == [('memory', 12 * 1024 ** 3), ('cache', 256 * 1024 ** 2), ('load',)]
    assert paced.PACED_POLICY['memory_guideline_gib'] == 18 and paced.PACED_POLICY['cache_limit_mib'] == 256


# ---------------------------------------------------------------- authorization gate (fail closed)
def factory_args(tmp_path):
    return dict(adapter=tmp_path / 'a', config={}, model_path=tmp_path, contract=None, seed=13, cache={'dir': str(tmp_path)})


def test_factory_fails_closed_without_the_root_entry_authorization(tmp_path, monkeypatch):
    bridge = native()
    monkeypatch.setattr(bridge.importlib.util, 'find_spec', lambda name: None)
    with pytest.raises(bridge.PendingAuthorization, match='PENDING_GENUINE_AUTHORIZATION'):
        bridge.make_native_backend(authorization=lambda: None, resource_gate=lambda n: None, **factory_args(tmp_path))
    with pytest.raises(bridge.PendingAuthorization):
        bridge.execute_native_micro(authorization=None, resource_gate=None, backend=None, units=[], prompt_lengths={}, reward_policy_sha256=REWARD, output=tmp_path / 'o', expected_parent={})


def test_factory_requires_the_issued_class_the_micro_phase_and_the_matching_gate(tmp_path, monkeypatch):
    bridge = native()
    from mri_rl import safe_micro_resources_v1 as resources
    class Authorization:
        def __init__(self, scope):
            self.scope, self.checked = scope, []
        def check(self, name):
            self.checked.append(name)
    entry = types.ModuleType(bridge.ENTRY_MODULE); entry.Authorization = Authorization
    monkeypatch.setitem(sys.modules, bridge.ENTRY_MODULE, entry)
    monkeypatch.setattr(bridge.importlib.util, 'find_spec', lambda name: object())
    scope = {'resource_stage': 'B_MICRO', 'execution_stage': 'MICRO_TRAIN'}
    auth = Authorization(scope)
    def gate_for(authorization, stage='B_MICRO'):
        gate = object.__new__(resources.ResourceGate)
        gate.stage, gate.scope, gate.authorization = stage, authorization.scope, authorization
        return gate
    built = []
    class Sentinel:
        def __init__(self, *args, **kwargs):
            built.append(kwargs)
    monkeypatch.setattr(bridge, 'MicroNativeBackend', Sentinel)
    kwargs = factory_args(tmp_path)
    with pytest.raises(ValueError, match='issued by the root entry'):
        bridge.make_native_backend(authorization=types.SimpleNamespace(scope=scope, check=lambda n: None), resource_gate=gate_for(auth), **kwargs)
    with pytest.raises(ValueError, match='MICRO_TRAIN'):
        other = Authorization({**scope, 'execution_stage': 'SINGLE_LOAD_FORWARD'})
        bridge.make_native_backend(authorization=other, resource_gate=gate_for(other), **kwargs)
    with pytest.raises(ValueError, match='matching|built from this exact'):
        bridge.make_native_backend(authorization=auth, resource_gate=gate_for(Authorization(dict(scope))), **kwargs)
    with pytest.raises(ValueError, match='built from this exact'):
        bridge.make_native_backend(authorization=auth, resource_gate=lambda n: None, **kwargs)
    assert built == []
    gate = gate_for(auth)
    bridge.make_native_backend(authorization=auth, resource_gate=gate, **kwargs)
    assert len(built) == 1 and built[0]['before_unit'] is gate and auth.checked == ['micro_native_authority']
