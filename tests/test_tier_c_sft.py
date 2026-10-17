"""CPU tests for the same-C-data SFT control (NumPy fixture backend; no model)."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
import pytest
from mri_rl import tier_c_sft as sft, two_task_training as tt
from tests.test_two_task_training import ROWS
from tests.test_continuous_rl_driver import NumericBackend

IN = lambda r: 213 if r['ability'] == 'mri_sequence' else 405
TGT = {r['id']: 24 for r in ROWS}
PLAN = {'group_size': 4, 'max_tokens': 256, 'budget': {'input_tokens_min': 213, 'input_tokens_max': 405,
        'text_visual_token_cap': 169216, 'maximum_attempted_groups': 198}}


class SFTBackend(NumericBackend):
    """NumPy Adam fixture with an sft_step like the native one (deterministic non-collinear gradients)."""
    def sft_step(self, examples, work):
        g = np.array([1., -2.]) if int(self.state['step']) % 2 == 0 else np.array([3., 1.])
        moments = self.state.setdefault('weight', {'m': np.zeros(2), 'v': np.zeros(2)})
        moments['m'] = .9 * moments['m'] + .1 * g; moments['v'] = .999 * moments['v'] + .001 * g * g
        self.state['step'] += 1
        self.weight -= self.state['learning_rate'] * moments['m'] / (np.sqrt(moments['v']) + 1e-8)
        return {'loss_terms': [.5] * 4, 'gradient_l2': float(np.linalg.norm(g)),
                'prompt_tokens': [IN(e) for e in examples], 'target_tokens': [TGT[e['id']] for e in examples]}


def test_schedule_is_natural_proportional_deterministic_and_whole_step_bounded():
    steps, used = sft.build_steps(ROWS, 7, 169216, IN, TGT)
    again, _ = sft.build_steps(ROWS, 7, 169216, IN, TGT)
    assert steps == again and sft.build_steps(ROWS, 11, 169216, IN, TGT)[0] != steps
    assert used <= 169216 and len(steps) <= 256 and all(len(s) == 4 for s in steps)
    flat = [i for s in steps for i in s]
    first_epochs = Counter(flat[:76 * (len(flat) // 76)])
    assert set(first_epochs.values()) <= {len(flat) // 76} and len(first_epochs) == 76      # full epochs give equal exposure per row
    nxt = sum(IN(ROWS[i]) + TGT[ROWS[i]['id']] for i in range(4))
    assert used + 4 * 213 > 169216 - 4 * 429 or len(steps) == 256                         # stopped because the next step did not fit


def run(tmp_path, steps, cap=10**6, resume=False, name='o', factory=SFTBackend):
    plan = {**PLAN, 'budget': {**PLAN['budget'], 'text_visual_token_cap': cap}}
    led = tt.TokenLedger(plan)
    r = sft.run_sft(output=tmp_path / name, steps=sft.step_rows(ROWS, steps), bindings={'arm': 'SFT_C', 'seed': 7},
                    backend_factory=factory, authorize=lambda c: None, ledger=led, input_tokens=IN,
                    target_tokens=TGT, resume=resume)
    return r, led


def test_run_publishes_atomic_boundaries_one_update_per_step_and_resumes(tmp_path):
    steps, _ = sft.build_steps(ROWS, 7, 10**6, IN, TGT, max_steps=5)
    NumericBackend.instances = []
    r, led = run(tmp_path, steps)
    assert r['status'] == 'SCHEDULE_COMPLETE' and r['completed_steps'] == 5 and r['optimizer_updates'] == 5
    assert len(NumericBackend.instances) == 1                                    # one resident backend
    expected_cost = sum(IN(e) + TGT[e['id']] for s in steps for e in [ROWS[i] for i in s])
    assert led.used == expected_cost and led.charged_groups == 5
    again, _ = run(tmp_path, steps, resume=True)
    assert again['status'] == 'SCHEDULE_COMPLETE' and again['completed_steps'] == 5


def test_crash_then_resume_matches_the_continuous_run(tmp_path):
    steps, _ = sft.build_steps(ROWS, 7, 10**6, IN, TGT, max_steps=6)
    NumericBackend.instances = []
    run(tmp_path, steps, name='full'); full = NumericBackend.instances[-1]
    class Flaky(SFTBackend):
        calls = 0
        def sft_step(self, examples, work):
            Flaky.calls += 1
            if Flaky.calls == 3:
                raise RuntimeError('simulated native failure mid-run')
            return super().sft_step(examples, work)
    with pytest.raises(RuntimeError, match='simulated'):
        run(tmp_path, steps, name='crash', factory=Flaky)
    assert len(list((tmp_path / 'crash').glob('boundary_*'))) == 3                    # initial + 2 completed steps preserved
    r, _ = run(tmp_path, steps, name='crash', resume=True)
    resumed = NumericBackend.instances[-1]
    assert r['optimizer_updates'] == 6
    np.testing.assert_array_equal(full.weight, resumed.weight)
    for k in ('m', 'v'):
        np.testing.assert_array_equal(full.state['weight'][k], resumed.state['weight'][k])


def test_budget_stop_and_native_token_mismatch_and_changed_cap(tmp_path):
    steps, _ = sft.build_steps(ROWS, 7, 10**6, IN, TGT, max_steps=6)
    one = sum(IN(ROWS[i]) + TGT[ROWS[i]['id']] for i in steps[0])
    r, led = run(tmp_path, steps, cap=one * 2 + 1, name='b')
    assert r['status'] == 'TOKEN_BUDGET_EXHAUSTED' and r['completed_steps'] == 2 and led.used <= one * 2 + 1 + 1000
    class Wrong(SFTBackend):
        def sft_step(self, examples, work):
            out = super().sft_step(examples, work); out['prompt_tokens'][0] += 1; return out
    with pytest.raises(ValueError): run(tmp_path, steps, name='w', factory=Wrong)
    run(tmp_path, steps[:1], name='c')
    with pytest.raises(ValueError): run(tmp_path, steps[:1], cap=5 * 10**5, resume=True, name='c')


def test_native_backend_class_defers_mlx_and_targets_match_the_rl_answer_format():
    import sys
    sft.native_backend_class()                                                   # building the class must not import MLX
    assert 'mlx.core' not in sys.modules
    t = sft.sft_target(ROWS[0])
    assert t.startswith('<think>') and t.endswith('</answer>') and ROWS[0]['reward_model']['ground_truth'] in t


# ---- independent SFT counterexample regressions (results/TIER_C_SFT_INDEPENDENT_FAILURE_REVIEW.json) ----
def test_completed_resume_restores_and_reports_the_committed_ledger(tmp_path):
    steps, _ = sft.build_steps(ROWS, 7, 10**6, IN, TGT, max_steps=2)
    first, led = run(tmp_path, steps, name='done')
    again, led2 = run(tmp_path, steps, name='done', resume=True)
    assert again['status'] == 'SCHEDULE_COMPLETE' and again['ledger']['used_tokens'] == first['ledger']['used_tokens'] > 0
    assert led2.used == led.used and again['optimizer_updates'] == 2


def test_crashed_step_is_charged_once_as_imputed_and_cannot_exceed_the_cap(tmp_path):
    steps, _ = sft.build_steps(ROWS, 7, 10**6, IN, TGT, max_steps=2)
    costs = [sum(IN(ROWS[i]) + TGT[ROWS[i]['id']] for i in s) for s in steps]; cap = sum(costs)
    class Crash(SFTBackend):
        calls = 0
        def sft_step(self, examples, work):
            Crash.calls += 1
            if Crash.calls == 2:
                raise RuntimeError('crash after the first teacher-forced example')
            return super().sft_step(examples, work)
    with pytest.raises(RuntimeError, match='crash'):
        run(tmp_path, steps, cap=cap, name='p', factory=Crash)
    r, led = run(tmp_path, steps, cap=cap, name='p', resume=True)
    assert led.imputed == costs[1] and led.used == costs[0] + costs[1]                      # reserved step charged conservatively, separate
    assert r['status'] == 'TOKEN_BUDGET_EXHAUSTED' and r['completed_steps'] == 1 and led.used <= cap
    r2, led2 = run(tmp_path, steps, cap=cap, name='p', resume=True)
    assert led2.imputed == costs[1] and led2.used == led.used                                # not charged a second time


def test_image_changed_inside_the_native_callback_is_never_committed(tmp_path):
    steps, _ = sft.build_steps(ROWS, 7, 10**6, IN, TGT, max_steps=1)
    image = Path(ROWS[steps[0][1]]['images'][0]); old = image.read_bytes()
    class Mutate(SFTBackend):
        def sft_step(self, examples, work):
            Path(examples[1]['images'][0]).write_bytes(b'MUTATED AFTER THE RUNNER PRECHECK')
            return super().sft_step(examples, work)
    try:
        with pytest.raises(ValueError, match='image'):
            run(tmp_path, steps, name='m', factory=Mutate)
        assert len(list((tmp_path / 'm').glob('boundary_*'))) == 1                         # only the initial boundary; no updated state
    finally:
        image.write_bytes(old)


def test_native_step_rejects_nonfinite_loss_or_gradient_before_any_optimizer_update(tmp_path):
    import types, math, hashlib
    cls = sft.native_backend_class(); b = cls.__new__(cls); called = []
    b.mx = types.SimpleNamespace(array=np.array, float32=np.float32, eval=lambda *x: None, clear_cache=lambda: None, sum=np.sum)
    b.nn = types.SimpleNamespace(value_and_grad=lambda model, fn: lambda *a: (np.array(float('nan')), {'g': np.array([float('nan')], dtype=np.float32)}))
    b.model = types.SimpleNamespace(eval=lambda: None, config=types.SimpleNamespace(model_type='fixture'), trainable_parameters=lambda: {})
    b.contract = types.SimpleNamespace(system_instruction='s', enable_thinking=False, assistant_prefix='END')
    b.processor = types.SimpleNamespace(tokenizer=types.SimpleNamespace(encode=lambda *a, **k: [3]))
    b.template = lambda *a, **k: '<|image_pad|>END'; b.prepare_inputs = lambda *a, **k: {'input_ids': np.array([[1, 2]], dtype=np.int64)}
    b.special_tokens = lambda *a: False; b.before_unit = lambda name: None
    b.tree_map = lambda fn, *t: {k: fn(*(x[k] for x in t)) for k in t[0]}; b.tree_flatten = lambda d: list(d.items())
    b.optimizer = types.SimpleNamespace(update=lambda m, g: called.append(g), state={})
    ex = ROWS[:4]
    b.expected_images = {r['id']: hashlib.sha256(Path(r['images'][0]).read_bytes()).hexdigest() for r in ex}
    b.expected_target_ids = {r['id']: [3] for r in ex}
    with pytest.raises(ValueError, match='Nonfinite'):
        b.sft_step(ex, tmp_path)
    assert called == []
    with pytest.raises(ValueError, match='Approved image hashes'):
        del b.expected_images; b.sft_step(ex, tmp_path)


def _native_standin(ex, tokenizer_ids, grad_calls, update_calls):
    import types, hashlib
    cls = sft.native_backend_class(); b = cls.__new__(cls)
    b.mx = types.SimpleNamespace(array=lambda v, dtype=None: np.array(v), float32=np.float32, eval=lambda *x: None, clear_cache=lambda: None, sum=np.sum)
    def vg(model, fn):
        def call(model_, prompt_ids, target_ids, pixels, kwargs):
            grad_calls.append(target_ids.tolist()); return np.array(1.0), {'g': np.array([1.0], dtype=np.float32)}
        return call
    b.nn = types.SimpleNamespace(value_and_grad=vg)
    b.model = types.SimpleNamespace(eval=lambda: None, config=types.SimpleNamespace(model_type='fixture'), trainable_parameters=lambda: {})
    b.contract = types.SimpleNamespace(system_instruction='s', enable_thinking=False, assistant_prefix='END')
    b.processor = types.SimpleNamespace(tokenizer=types.SimpleNamespace(encode=lambda *a, **k: list(tokenizer_ids)))
    b.template = lambda *a, **k: '<|image_pad|>END'; b.prepare_inputs = lambda *a, **k: {'input_ids': np.array([[1, 2]], dtype=np.int64)}
    b.special_tokens = lambda *a: False; b.before_unit = lambda n: None
    b.tree_map = lambda fn, *t: {k: fn(*(x[k] for x in t)) for k in t[0]}; b.tree_flatten = lambda d: list(d.items())
    b.optimizer = types.SimpleNamespace(update=lambda m, g: update_calls.append(g), state={})
    b.expected_images = {r['id']: hashlib.sha256(Path(r['images'][0]).read_bytes()).hexdigest() for r in ex}
    return b


def test_same_length_wrong_target_ids_are_rejected_before_any_gradient_or_update(tmp_path):
    ex = ROWS[:4]; grads, updates = [], []
    b = _native_standin(ex, [999], grads, updates); b.expected_target_ids = {r['id']: [7] for r in ex}   # same count, different id
    with pytest.raises(ValueError, match='target token ids differ'):
        b.sft_step(ex, tmp_path)
    assert grads == [] and updates == []
    ok = _native_standin(ex, [7], grads, updates); ok.expected_target_ids = {r['id']: [7] for r in ex}
    out = ok.sft_step(ex, tmp_path)
    assert grads == [[7]] * 4 and len(updates) == 1 and out['target_tokens'] == [1] * 4                 # exact ids pass; math path unchanged


def test_native_step_fails_closed_without_frozen_target_ids(tmp_path):
    ex = ROWS[:4]; grads, updates = [], []
    b = _native_standin(ex, [7], grads, updates)
    with pytest.raises(ValueError, match='Frozen target token ids are required'):
        b.sft_step(ex, tmp_path)
    assert grads == [] and updates == []


def test_run_sft_hands_the_plan_target_ids_to_the_backend(tmp_path):
    steps, _ = sft.build_steps(ROWS, 7, 10**6, IN, TGT, max_steps=1)
    seen = {}
    class Capture(SFTBackend):
        def sft_step(self, examples, work):
            seen['ids'] = getattr(self, 'expected_target_ids', None); return super().sft_step(examples, work)
    plan = {**PLAN, 'budget': {**PLAN['budget'], 'text_visual_token_cap': 10**6}}
    expected = {r['id']: [1, 2] for r in ROWS}
    sft.run_sft(output=tmp_path / 'cap', steps=sft.step_rows(ROWS, steps), bindings={'arm': 'SFT_C', 'seed': 7},
                backend_factory=Capture, authorize=lambda c: None, ledger=tt.TokenLedger(plan), input_tokens=IN,
                target_tokens=TGT, expected_target_ids=expected)
    assert seen['ids'] == expected
