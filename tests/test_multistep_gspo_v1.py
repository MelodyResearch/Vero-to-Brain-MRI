"""CPU fake-backend tests for the limited multistep on-policy A/B. Never native, never MRI evidence."""
import copy
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pytest
from mri_rl import exposure_rounds_v1 as ex
from mri_rl import multistep_gspo_core_v1 as ms
from mri_rl import paired_prompt_groups_v1 as ppg
from mri_rl import shared_rollout_gspo_core_v1 as shared

EOS = 99
LENGTHS = {'mri_sequence': 213, 'mri_visit_correspondence': 405}
REWARD = hashlib.sha256(b'reward calibration').hexdigest()


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def pool():
    rows, sites = [], {}
    for i in range(28):
        person = f'sub-{i:02d}'
        sites[person] = ['BA', 'BO', 'FO', 'JS', 'KA', 'ME', 'NSK', 'TO'][i % 8]
        for label, tag in (('FLAIR', 'f'), ('contrast-enhanced T1', 't')):
            rows.append({'id': f'{person}_{tag}', 'ability': 'mri_sequence', 'images': [f'/x/{person}{tag}.png'],
                         'extra_info': {'patient_key': person, 'internal_split': 'train', 'question': 'q', 'level': 45},
                         'reward_model': {'ground_truth': label}})
    dallas = [{'id': f'd{i:02d}', 'ability': 'mri_visit_correspondence', 'images': [f'/x/d{i}.png'],
               'extra_info': {'internal_split': 'train', 'question': 'dq'}, 'reward_model': {'ground_truth': 'AB'[i % 2]}} for i in range(48)]
    return rows, dallas, sites


def make_plan(rounds):
    rows, dallas, sites = pool()
    return ex.build_round_plan(rows, dallas, sites, rounds=rounds)


class FakeBackend:
    """Deterministic numpy stand-in: LoRA=theta, persistent Adam, RNG. Counts like a native backend would."""
    def __init__(self, theta=None, counters=None, bias=0.0, always=None):
        self.theta = np.zeros(4) if theta is None else np.array(theta, dtype=float)
        self.m, self.v, self.step = np.zeros(4), np.zeros(4), 0
        self.rng = np.random.default_rng(7)
        self.counters = counters
        self.bias, self.always, self.fail_in, self.extra_charge = bias, always, None, 0
        self.generate_calls = 0
        self.fault = None

    # ----- identity
    def policy_sha(self):
        return sha('theta:' + self.theta.tobytes().hex() + f':{self.step}')

    def adam_digest(self):
        return sha('adam:' + self.m.tobytes().hex() + self.v.tobytes().hex() + f':{self.step}')

    def rng_digest(self):
        return sha('rng:' + json.dumps(self.rng.bit_generator.state, sort_keys=True, default=str))

    def save_rng(self, directory):
        (Path(directory) / 'rng.json').write_text(json.dumps(self.rng.bit_generator.state, default=str))

    def load_rng(self, directory):
        self.rng.bit_generator.state = json.loads((Path(directory) / 'rng.json').read_text())

    def snapshot(self):
        return (self.theta.copy(), self.m.copy(), self.v.copy(), self.step, copy.deepcopy(self.rng.bit_generator.state))

    def restore(self, snap):
        self.theta, self.m, self.v, self.step = snap[0].copy(), snap[1].copy(), snap[2].copy(), snap[3]
        self.rng.bit_generator.state = copy.deepcopy(snap[4])

    # ----- features / policy
    def phi(self, key, unit_kind):
        base = {'F': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'D1': [0, 0, 1, 0], 'D2': [0, 0, 0, 1]}[key]
        return np.array(base, dtype=float)

    def p_correct(self, key, unit):
        return sigmoid(float(self.theta @ self.phi(key, unit['kind'])) + self.bias)

    # ----- native-like operations
    def generate(self, unit, prompt_lengths):
        self.generate_calls += 1
        if self.fault == 'generate_params':
            self.theta = self.theta + 1e-9
        n = len(unit['contexts']) * 4
        if self.counters:
            self.counters.charge_many({'generated_response_tokens': n * 3, 'sampler_draws': n * 3, 'generation_language_model_calls': n * 3, 'generation_vision_embedding_calls': n})
        responses, contexts = [], {}
        for ctx in unit['contexts']:
            key = ctx['key']
            p = self.p_correct(key, unit)
            contexts[key] = {'prompt_text_sha256': sha('generic question'), 'image_sha256': sha('img' + ctx['row']['id']),
                             'rendered_prompt_token_ids_sha256': ppg.identity([1, key]), 'input_fingerprints_sha256': sha('fp' + key),
                             'prompt_length': prompt_lengths[ctx['row']['ability']]}
            for i in range(4):
                draw = bool(self.rng.random() < p)
                y = draw if self.always is None else bool(self.always)
                lp = math.log(p if y else 1 - p)
                grade = {'accuracy': float(y), 'raw_accuracy': float(y), 'format': 1.0, 'overlong': 0.0, 'score': float(y), 'total': float(y)}
                rid = f'{key}{i}' if key in ('F', 'T') else f'{key}_{i}'
                responses.append({'response_id': rid, 'context_key': key, 'text': f'<think>{rid}</think>', 'token_ids': [10 + i, EOS],
                                  'raw_selected_logps': [lp, -.01], 'finish_reason': 'stop', 'EOS_coverage': 'OBSERVED_AND_SCORED', 'generation_tokens': 2, 'grade': grade,
                                  '_y': y})
        return {'policy_sha256': self.policy_sha(), 'contexts': contexts, 'responses': responses, 'effective_eos_ids': {c['key']: [EOS] for c in unit['contexts']}}

    def _logp(self, unit, key, token_ids, y_from_tokens):
        p = self.p_correct(key, unit)
        return p if y_from_tokens else 1 - p

    def score_old(self, unit, responses):
        if self.counters:
            self.counters.charge_many({'detached_old_forwards': len(responses), 'scorer_vision_embedding_calls': len(responses), 'scorer_language_model_calls': len(responses),
                                       'teacherforced_input_positions': len(responses) * 10})
        if self.fault == 'old_params':
            self.theta = self.theta + 1e-9
        if self.fault == 'old_rng':
            self.rng.random()
        if self.fault == 'old_adam':
            self.m = self.m + 1e-9
        old = {}
        for r in responses:
            key = r.context.modality if r.context.modality in ('F', 'T') else r.response_id.split('_')[0]
            p = self.p_correct(key, unit)
            y = r.grade.accuracy == 1.0
            old[r.response_id] = [math.log(p if y else 1 - p), -.01]
        return old

    def gradient(self, planned, own_old, unit):
        if self.fail_in == 'gradient':
            raise RuntimeError('simulated backward failure')
        if self.fault == 'grad_params':
            self.theta = self.theta + 1e-9
        if self.fault == 'grad_rng':
            self.rng.random()
        used = [i for i in planned.plan.items if i.weight > 0]
        if self.counters:
            self.counters.charge_many({'backward_calls': len(used), 'gradient_forwards': len(used), 'scorer_vision_embedding_calls': len(used),
                                       'scorer_language_model_calls': len(used), 'teacherforced_input_positions': len(used) * 10 + self.extra_charge,
                                       'detached_old_forwards': 0})

        def one(item):
            key = item.response.context.modality if item.response.context.modality in ('F', 'T') else item.response.response_id.split('_')[0]
            p = self.p_correct(key, unit)
            y = item.response.grade.accuracy == 1.0
            dlogp = (1 - p if y else -p) * self.phi(key, unit['kind'])
            return {'loss': 0.0, 'gradient': -item.advantage * dlogp, 'old_current_max_abs': 0.0}
        total, evidence = shared.weighted_gradient_sum(planned.plan, one, lambda tree, w: tree * w, lambda a, b: a + b)
        norm = float(np.linalg.norm(total)) if total is not None else 0.0
        return total, {**evidence, 'gradient_l2': norm}

    def apply_update(self, gradient, lr=0.05):
        if self.fault == 'update_rng':
            self.rng.random()
        self.step += 1
        self.m = .9 * self.m + .1 * gradient
        self.v = .999 * self.v + .001 * gradient ** 2
        self.theta = self.theta - lr * self.m / (np.sqrt(self.v) + 1e-8)
        return {'optimizer_calls': 1}

    def save_checkpoint(self, directory):
        directory = Path(directory)
        np.savez(directory / 'state.npz', theta=self.theta, m=self.m, v=self.v, step=self.step)
        (directory / 'rng.json').write_text(json.dumps(self.rng.bit_generator.state, default=str))
        return ['state.npz', 'rng.json']

    def load_checkpoint(self, directory):
        z = np.load(Path(directory) / 'state.npz')
        self.theta, self.m, self.v, self.step = z['theta'], z['m'], z['v'], int(z['step'])
        self.rng.bit_generator.state = json.loads((Path(directory) / 'rng.json').read_text())


def build(tmp, rounds=3, common=False):
    plan = make_plan(rounds)
    units = ms.make_units(plan)
    return plan, units, ms.derive_arm_budget(units, LENGTHS, common_first_unit=common)


def runner(tmp_path, arm, backend, units, budget, common=None, guards=None, resume=False):
    return ms.ArmRunner(arm=arm, units=units, backend=backend, output=tmp_path / f'arm_{arm}', budget=budget, prompt_lengths=LENGTHS,
                        reward_policy_sha256=REWARD, common=common, guards=guards, resume=resume)


def parent_checkpoint(tmp_path, theta=(0, 0, 0, 0)):
    b = FakeBackend(theta)
    d = tmp_path / 'parent'
    d.mkdir()
    b.save_checkpoint(d)
    return d, b


# ------------------------------------------------------------------ structure
def test_units_are_pair_blocks_then_dallas_blocks_two_per_round():
    units = ms.make_units(make_plan(3))
    assert [u['kind'] for u in units] == ['SEQ_PAIR', 'DALLAS_BLOCK'] * 3 and [u['index'] for u in units] == list(range(6))
    assert [c['key'] for c in units[0]['contexts']] == ['F', 'T'] and [c['key'] for c in units[1]['contexts']] == ['D1', 'D2']
    assert {c['row']['reward_model']['ground_truth'] for c in units[0]['contexts']} == {'FLAIR', 'contrast-enhanced T1'}


def test_budget_is_exact_finite_and_identical_for_both_arms():
    units = ms.make_units(make_plan(8))
    b = ms.derive_arm_budget(units, LENGTHS)
    assert b['optimizer_updates'] == 16 and b['units'] == 16 and b['rounds'] == 8
    assert b['generated_response_tokens'] == 16 * 8 * 256 - 8 * 256          # unit 0 rollout is the COMMON one, charged once elsewhere
    assert b['common_rollout']['generated_response_tokens'] == 8 * 256
    assert b == ms.derive_arm_budget(ms.make_units(make_plan(8)), LENGTHS)
    assert b['limits']['backward_calls'] == 16 * 8 and b['claimed_cap_splits_pair'] is False


# ------------------------------------------------------------------ ownership / common rollout
def run_both(tmp_path, rounds=3, theta=(0, 0, 0, 0)):
    plan, units, budget = build(tmp_path, rounds, common=True)
    parent, pb = parent_checkpoint(tmp_path, theta)
    # common rollout from the exact parent
    common = ms.make_common_rollout(units[0], FakeBackend(theta), budget, prompt_lengths=LENGTHS, reward_policy_sha256=REWARD, output=tmp_path / 'common')
    results = {}
    for arm in 'AB':
        backend = FakeBackend(theta)
        r = runner(tmp_path, arm, backend, units, budget, common=common)
        results[arm] = (backend, r, r.run())
    return units, common, results


def test_common_rollout_once_then_each_arm_generates_its_own_responses(tmp_path):
    units, common, results = run_both(tmp_path, rounds=3)
    receipts = {a: results[a][1].receipts() for a in 'AB'}
    assert receipts['A'][0]['rollout_origin'] == receipts['B'][0]['rollout_origin'] == 'COMMON_PARENT'
    assert receipts['A'][0]['bundle_sha256'] == receipts['B'][0]['bundle_sha256'] == common.bundle.sha256
    for arm in 'AB':
        for k, rec in enumerate(receipts[arm][1:], start=1):
            assert rec['rollout_origin'] == 'OWN_POLICY' and rec['rollout_policy_sha256'] == rec['policy_sha_before']
    # after unit 0 the arms hold different policies (different grouping -> different update) and so different later rollouts
    assert receipts['A'][1]['policy_sha_before'] != receipts['B'][1]['policy_sha_before']
    assert receipts['A'][1]['bundle_sha256'] != receipts['B'][1]['bundle_sha256']
    assert results['A'][0].generate_calls == results['B'][0].generate_calls == 5      # 6 units - 1 common


def test_cross_arm_rollout_is_rejected_as_off_policy(tmp_path):
    plan, units, budget = build(tmp_path, 2)
    a, b = FakeBackend(), FakeBackend()
    ra, rb = runner(tmp_path, 'A', a, units, budget), runner(tmp_path, 'B', b, units, budget)
    ra.run_units(1), rb.run_units(1)          # no common: each generates its own unit 0 from the shared parent
    b.theta = b.theta + 0.5                    # B now holds a different policy
    foreign = b.generate(units[1], LENGTHS)
    with pytest.raises(ValueError, match='policy'):
        ra.inject_rollout_for_test(units[1], foreign)


def test_common_rollout_refused_when_arm_is_not_at_the_common_parent(tmp_path):
    plan, units, budget = build(tmp_path, 2, common=True)
    common = ms.make_common_rollout(units[0], FakeBackend(), budget, prompt_lengths=LENGTHS, reward_policy_sha256=REWARD, output=tmp_path / 'common')
    moved = FakeBackend()
    moved.theta = moved.theta + 1.0
    with pytest.raises(ValueError, match='common parent'):
        runner(tmp_path, 'A', moved, units, budget, common=common).run()


def test_common_rollout_is_not_consumable_beyond_unit_zero(tmp_path):
    plan, units, budget = build(tmp_path, 2, common=True)
    common = ms.make_common_rollout(units[0], FakeBackend(), budget, prompt_lengths=LENGTHS, reward_policy_sha256=REWARD, output=tmp_path / 'common')
    r = runner(tmp_path, 'A', FakeBackend(), units, budget, common=common)
    with pytest.raises(ValueError, match='unit 0'):
        r.inject_common_for_test(units[2])


# ------------------------------------------------------------------ math: loss batching
def test_pair_block_gradient_equals_existing_weighted_math_in_both_arms(tmp_path):
    plan, units, budget = build(tmp_path, 1)
    backend = FakeBackend((0.3, -0.2, 0.1, 0.4))
    raw = backend.generate(units[0], LENGTHS)
    for arm in 'AB':
        block = ms.build_block(units[0], raw, policy_sha=raw['policy_sha256'], reward_policy_sha256=REWARD)
        planned = ms.arm_plan_for(block, arm)
        old = backend.score_old(units[0], block.responses)
        gradient, evidence = backend.gradient(planned, old, units[0])
        jac = {}
        for r in block.responses:
            key = r.context.modality
            p = backend.p_correct(key, units[0])
            y = r.grade.accuracy == 1.0
            jac[r.response_id] = list(((1 - p) if y else -p) * backend.phi(key, 'SEQ_PAIR'))
        expected = ppg.local_descent(planned.plan, jac)          # existing CPU math: -grad = sum w*A*dlogp
        assert np.allclose(-np.array(expected), gradient if gradient is not None else 0)
        assert evidence['backward_calls'] == len(planned.admitted_response_ids)
        assert all(abs(i.weight - 1 / 8) < 1e-15 for i in planned.plan.items if i.admitted)


def test_dallas_block_uses_original_same_prompt_groups_with_one_eighth_weight(tmp_path):
    plan, units, budget = build(tmp_path, 1)
    raw = FakeBackend().generate(units[1], LENGTHS)
    block = ms.build_block(units[1], raw, policy_sha=raw['policy_sha256'], reward_policy_sha256=REWARD)
    for arm in 'AB':       # Dallas grouping is identical in both arms
        planned = ms.arm_plan_for(block, arm)
        groups = {}
        for i in planned.plan.items:
            groups.setdefault(i.group_index, set()).add(i.response.context.modality + i.response.response_id.split('_')[0])
        assert len(groups) == 2 and all(len(v) == 1 for v in groups.values())
        assert all(i.weight in (0.0, 1 / 8) for i in planned.plan.items)
    assert [i.response.response_id for i in ms.arm_plan_for(block, 'A').plan.items] == [i.response.response_id for i in ms.arm_plan_for(block, 'B').plan.items]


# ------------------------------------------------------------------ recovery / cursor / cost
def test_resume_from_last_committed_cursor_restores_moments_and_matches_uninterrupted(tmp_path):
    plan, units, budget = build(tmp_path, 2)
    ref = FakeBackend()
    runner(tmp_path / 'ref', 'A', ref, units, budget).run()
    first = FakeBackend()
    r1 = runner(tmp_path, 'A', first, units, budget)
    r1.run_units(2)                                        # crash after two committed units
    fresh = FakeBackend()                                  # new process: nothing in memory
    r2 = runner(tmp_path, 'A', fresh, units, budget, resume=True)
    assert r2.cursor()['next_unit'] == 2 and fresh.adam_digest() == first.adam_digest() and fresh.rng_digest() == first.rng_digest()
    r2.run()
    assert fresh.policy_sha() == ref.policy_sha() and fresh.adam_digest() == ref.adam_digest()


def test_exception_rolls_back_model_adam_and_rng_and_blocks_automatic_retry(tmp_path):
    plan, units, budget = build(tmp_path, 2)
    b = FakeBackend()
    r = runner(tmp_path, 'A', b, units, budget)
    r.run_units(1)
    before = (b.policy_sha(), b.adam_digest(), b.rng_digest())
    b.fail_in = 'gradient'
    with pytest.raises(RuntimeError):
        r.run_units(1)
    assert (b.policy_sha(), b.adam_digest(), b.rng_digest()) == before and r.cursor()['next_unit'] == 1
    b.fail_in = None
    with pytest.raises(RuntimeError, match='INCOMPLETE_UNIT'):
        runner(tmp_path, 'A', FakeBackend(), units, budget, resume=True)


def test_abandoned_unit_is_charged_in_full_and_never_free_to_retry(tmp_path):
    plan, units, budget = build(tmp_path, 2)
    b = FakeBackend()
    r = runner(tmp_path, 'A', b, units, budget)
    r.run_units(1)
    b.fail_in = 'gradient'
    with pytest.raises(RuntimeError):
        r.run_units(1)
    used_before = r.counters.snapshot()
    ms.abandon_incomplete_unit(tmp_path / 'arm_A', reason='test')
    r2 = runner(tmp_path, 'A', FakeBackend(), units, budget, resume=True)
    after = r2.counters.snapshot()
    unit_worst = ms.unit_worst_case(units[1], LENGTHS)
    assert all(after[k] >= used_before[k] for k in after) and after['generated_response_tokens'] >= used_before['generated_response_tokens']
    assert after['generated_response_tokens'] - used_before['generated_response_tokens'] <= unit_worst['generated_response_tokens']


def test_cost_overrun_stops_before_commit_and_rolls_back(tmp_path):
    plan, units, budget = build(tmp_path, 2)
    b = FakeBackend()
    r = runner(tmp_path, 'A', b, units, budget)
    r.run_units(1)
    snap = (b.policy_sha(), b.adam_digest())
    b.extra_charge = 10 ** 9
    with pytest.raises(ValueError, match='budget'):
        r.run_units(1)
    assert (b.policy_sha(), b.adam_digest()) == snap and r.cursor()['next_unit'] == 1


def test_zero_variance_groups_are_kept_without_resample_or_update(tmp_path):
    plan, units, budget = build(tmp_path, 1)
    b = FakeBackend(always=True)             # every response correct -> no informative group
    r = runner(tmp_path, 'A', b, units, budget)
    r.run()
    recs = r.receipts()
    assert [x['updated'] for x in recs] == [False, False] and all(x['effective_groups'] == 0 for x in recs)
    assert b.generate_calls == 2 and b.step == 0 and recs[0]['update_decision']['reason'] == 'NO_ADMITTED_GROUP'


def test_guard_hold_stops_at_a_unit_boundary_and_cursor_is_resumable(tmp_path):
    plan, units, budget = build(tmp_path, 2)
    calls = []

    def guards(phase, info):
        calls.append(phase)
        if phase == 'before_unit' and info['unit'] == 2:
            raise ms.GuardStop('AC power lost')
    b = FakeBackend()
    r = runner(tmp_path, 'A', b, units, budget, guards=guards)
    with pytest.raises(ms.GuardStop):
        r.run()
    assert r.cursor()['next_unit'] == 2 and r.cursor()['complete'] is False
    runner(tmp_path, 'A', FakeBackend(), units, budget, resume=True).run()


def test_arm_b_starts_from_parent_not_from_a_state(tmp_path):
    plan, units, budget = build(tmp_path, 1)
    a = FakeBackend()
    runner(tmp_path, 'A', a, units, budget).run()
    assert a.step > 0
    dirty = FakeBackend()
    dirty.load_checkpoint(tmp_path / 'arm_A' / 'unit_0001')       # A's final state
    with pytest.raises(ValueError, match='parent'):
        ms.ArmRunner(arm='B', units=units, backend=dirty, output=tmp_path / 'arm_B', budget=budget, prompt_lengths=LENGTHS, reward_policy_sha256=REWARD,
                     parent_policy_sha256=FakeBackend().policy_sha())


def test_update_cap_cannot_split_a_pair_in_the_budget(tmp_path):
    units = ms.make_units(make_plan(3))
    with pytest.raises(ValueError):
        ms.derive_arm_budget(units[:1], LENGTHS)               # a lone SEQ_PAIR (round split from its Dallas block) is not budgetable
    assert ms.derive_arm_budget(units[:2], LENGTHS)['rounds'] == 1


# ------------------------------------------------------------------ review regressions
def test_common_requires_full_parent_state_not_just_weights(tmp_path):
    plan, units, budget = build(tmp_path, 2, common=True)
    common = ms.make_common_rollout(units[0], FakeBackend(), budget, prompt_lengths=LENGTHS, reward_policy_sha256=REWARD, output=tmp_path / 'common')
    other_adam = FakeBackend()
    other_adam.m = other_adam.m + 1e-3                      # same weights, different Adam moments
    with pytest.raises(ValueError, match='trusted common parent'):
        runner(tmp_path, 'A', other_adam, units, budget, common=common)
    other_rng = FakeBackend()
    other_rng.rng.random()                                  # same weights/Adam, different R0
    with pytest.raises(ValueError, match='trusted common parent'):
        runner(tmp_path, 'B', other_rng, units, budget, common=common)
    ok = runner(tmp_path, 'A2', FakeBackend(), units, budget, common=common) if False else None
    assert common.parent_state['policy'] == FakeBackend().policy_sha() and common.rng_after_digest != common.parent_state['rng']


def _faulty(kind, tmp_path):
    plan, units, budget = build(tmp_path, 2)
    b = FakeBackend()
    r = runner(tmp_path, 'A', b, units, budget)
    r.run_units(1)
    before = (b.policy_sha(), b.adam_digest(), b.rng_digest())
    b.fault = kind
    with pytest.raises(ValueError):
        r.run_units(1)
    assert (b.policy_sha(), b.adam_digest(), b.rng_digest()) == before and r.cursor()['next_unit'] == 1
    assert list((tmp_path / 'arm_A').glob('STOP_unit_0001_*'))


@pytest.mark.parametrize('kind', ['old_params', 'old_rng', 'old_adam', 'grad_params', 'grad_rng', 'update_rng', 'generate_params'])
def test_state_contamination_anywhere_is_detected_and_rolled_back(kind, tmp_path):
    _faulty(kind, tmp_path)


def test_truncation_recomputes_sites_and_reports_potential_vs_actual_updates():
    p = make_plan(8)
    cut = ex.truncate_to_update_cap(p, max_updates=4)
    kept = {r['patient'] for r in cut['rounds']}
    assert set(cut['patient_sites']) == kept and set(cut['sites_covered']) == {p['patient_sites'][k] for k in kept}
    rep = ex.exposure_report(cut)
    assert rep['sites_covered'] == sorted(set(cut['patient_sites'].values())) and rep['updates_max_per_arm'] == 4 and 'POTENTIAL_MAX' in rep['updates_basis']
