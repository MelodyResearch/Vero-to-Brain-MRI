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
