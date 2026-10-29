"""SAFE MICRO joint one-step Vero GSPO A/B transaction: backend-neutral core. SOURCE_IMPLEMENTED_CPU_SYNTHETIC_NOT_NATIVE_PASS.

No MLX, model load, generation or optimizer here; a backend object supplies them (CPU fake in tests, the lazy native bridge
`safe_micro_native_v1` only after a separate exact authorization). Importing this module is inert.

Semantics (first frozen whole patient round = SEQ_PAIR F4/T4 + DALLAS_BLOCK D1 K4/D2 K4, 16 real responses):
  * the 16 responses are generated ONCE from the common Adam52 parent (before any update) and their own-old logps are scored there;
  * each arm restores the exact common parent (weights + full Adam + R0) separately, computes the sequence and Dallas block gradients
    serially (existing per-response weight 1/8), forms JOINT = .5*seq + .5*Dallas (= 1/16 per admitted response; an inactive block is
    NOT reweighted), and applies AT MOST ONE Adam step (52 -> 53);
  * zero admitted groups or a zero joint gradient is negative evidence: 0 updates, no resample, no momentum-only step;
  * arm A groups the sequence block F4/T4, arm B F2T2/F2T2; Dallas is the unchanged two same-prompt K4 groups in both arms;
  * state (checkpoint+optimizer+RNG) is committed atomically BEFORE the receipt; an updated arm ends at the persisted post-all-16
    generation RNG R1 (explicit protocol step); there is no on-policy sharing after divergence, no retry and no stage progression.
Counters are a spent ledger: a rollback restores model state but never refunds spent work.
"""
from __future__ import annotations
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
from mri_rl import multistep_gspo_core_v1 as ms
from mri_rl import shared_rollout_gspo_core_v1 as shared
from mri_rl.paired_prompt_groups_v1 import identity, canonical

STATUS = 'SOURCE_IMPLEMENTED_CPU_SYNTHETIC_NOT_NATIVE_PASS'
PROTOCOL = 'SAFE_MICRO_JOINT_ONE_STEP_VERO_GSPO_AB_V1'
CHARGEABLE = shared.CHARGEABLE
require = shared.require
ARMS = ('A', 'B')
BLOCK_WEIGHT = .5                      # joint = .5*sequence + .5*Dallas
PARENT_ADAM_STEP, UPDATED_ADAM_STEP = 52, 53
NATIVE_KIND, SYNTHETIC_KIND = 'NATIVE_MLX', 'CPU_SYNTHETIC_FIXTURE'
CAPS = {'generated_response_tokens': 4096, 'detached_old_forwards': 16, 'gradient_forwards': 32, 'backward_calls': 32,
        'optimizer_updates': 2, 'optimizer_updates_per_arm': 1}
REQUIRED_BACKEND = ('policy_sha', 'adam_digest', 'rng_digest', 'optimizer_step', 'snapshot', 'restore', 'generate', 'score_old', 'gradient',
                    'scale_gradient', 'add_scaled', 'gradient_l2', 'gradient_dtypes', 'save_gradient', 'release_graphs', 'apply_update',
                    'save_checkpoint', 'save_rng', 'load_rng', 'evidence_kind')


# ---------------------------------------------------------------- units / budget
def check_units(units):
    require(isinstance(units, list) and len(units) == 2, 'Exactly the first frozen whole patient round (two units) required')
    seq, dallas = units
    require([u['kind'] for u in units] == [ms.SEQ_KIND, ms.DALLAS_KIND] and [u['index'] for u in units] == [0, 1] and seq['round'] == dallas['round'],
            'Unit order must be SEQ_PAIR(0) then DALLAS_BLOCK(1) of one round')
    require([c['key'] for c in seq['contexts']] == ['F', 'T'] and [c['key'] for c in dallas['contexts']] == ['D1', 'D2'], 'Unexpected context inventory')
    return True


def micro_budget(units, prompt_lengths):
    """Exact worst-case endpoints: ONE common 16-response generation/own-old, then two identical arm phases (<=1 update each)."""
    check_units(units)
    require(set(prompt_lengths) == {ms.SEQ_TASK, ms.DALLAS_TASK} and all(type(v) is int and 0 < v <= 2048 for v in prompt_lengths.values()),
            'Measured expanded prompt lengths required')
    common, arm = ({k: 0 for k in CHARGEABLE} for _ in range(2))
    for unit in units:
        for key, value in ms.common_rollout_budget(unit, prompt_lengths).items():
            common[key] += value
        for key, value in ms.unit_worst_case(unit, prompt_lengths, common=True).items():
            arm[key] += value
    arm['optimizer_updates'] = 1                      # ONE joint step per arm, not one per block
    total = {k: common[k] + 2 * arm[k] for k in CHARGEABLE}
    require(total['generated_response_tokens'] == CAPS['generated_response_tokens'] and total['detached_old_forwards'] == CAPS['detached_old_forwards'] and
            total['gradient_forwards'] <= CAPS['gradient_forwards'] and total['backward_calls'] <= CAPS['backward_calls'] and
            total['optimizer_updates'] <= CAPS['optimizer_updates'] and arm['optimizer_updates'] <= CAPS['optimizer_updates_per_arm'],
            'Derived budget exceeds the fixed micro caps')
    require(total['teacherforced_input_positions'] == common['teacherforced_input_positions'] + 2 * arm['teacherforced_input_positions'], 'TF accounting differs')
    return {'protocol': PROTOCOL, 'common': common, 'arm': arm, 'total': total, 'prompt_lengths': dict(prompt_lengths), 'caps': dict(CAPS),
            'weight_per_admitted_response_in_joint': 1 / 16, 'block_gradient_weight': BLOCK_WEIGHT,
            'automatic_retry': False, 'unused_budget_transfer': False, 'extension': False, 'simultaneous_models': 1}


class Ledger:
    """Spent-work ledger: one finite integer Counters per phase (common, A, B), persisted atomically; never refunded on rollback."""
    def __init__(self, budget, path):
        self.path, self.limits = Path(path), {'common': budget['common'], 'A': budget['arm'], 'B': budget['arm']}
        self.used = {name: {k: 0 for k in CHARGEABLE} for name in self.limits}
        self.phases = {name: shared.Counters(limits, None, lambda used, name=name: self._write(name, used)) for name, limits in self.limits.items()}
        self._write('common', self.used['common'])

    def _write(self, name, used):
        self.used[name] = dict(used)
        shared.replace_atomic(self.path, {'protocol': PROTOCOL, 'limits': self.limits, 'used': self.used})

    def phase(self, name):
        return self.phases[name]

    def snapshot(self):
        return {name: dict(c.used) for name, c in self.phases.items()}


# ---------------------------------------------------------------- helpers
def _state(backend):
    return backend.policy_sha(), backend.adam_digest(), backend.rng_digest()


def _fsync_file(path):
    with Path(path).open('rb') as handle:
        os.fsync(handle.fileno())


def _check_backend(backend):
    missing = [name for name in REQUIRED_BACKEND if not hasattr(backend, name)]
    require(not missing, 'Backend lacks the micro-transaction protocol: ' + ','.join(missing))
    require(backend.evidence_kind in (NATIVE_KIND, SYNTHETIC_KIND), 'Backend must declare its evidence kind')


def _plan_record(planned):
    groups = {}
    for item in planned.plan.items:
        groups.setdefault(item.group_index, []).append(item.response.response_id)
    return {'grouping': planned.plan.grouping, 'effective_groups': planned.admitted_groups, 'admitted_response_ids': list(planned.admitted_response_ids),
            'groups': [groups[k] for k in sorted(groups)],
            'items': [{'response_id': i.response.response_id, 'group': i.group_index, 'advantage': i.advantage, 'weight': i.weight,
                       'admitted': i.admitted, 'reward': i.response.grade.total} for i in planned.plan.items]}


# ---------------------------------------------------------------- common 16-response rollout
def _common(*, backend, units, lengths, reward, ledger, parent, output, before, receipt):
    directory = Path(output) / 'common_PRIVATE'
    directory.mkdir()
    backend.counters = ledger.phase('common')
    policy, adam, rng0 = parent
    blocks = []
    for unit in units:
        before(f"micro_common_generate_unit_{unit['index']}")
        raw = backend.generate(unit, lengths)
        ms.ownership(raw['policy_sha256'], policy)
        require(backend.policy_sha() == policy and backend.adam_digest() == adam, 'Common generation changed parameters or Adam')
        blocks.append(ms.build_block(unit, raw, policy_sha=policy, reward_policy_sha256=reward, checkpoint_receipt_sha256=receipt))
    rng1 = backend.rng_digest()
    require(rng1 != rng0, 'Common generation did not consume RNG')
    rng_dir = directory / 'rng_after_all16'
    rng_dir.mkdir()
    backend.save_rng(rng_dir)
    olds = []
    for unit, block in zip(units, blocks):
        before(f"micro_common_score_old_unit_{unit['index']}")
        old = backend.score_old(unit, block.responses)
        ms.check_old(block, old)
        require(_state(backend) == (policy, adam, rng1), 'Own-old scoring changed weights, Adam or the post-generation RNG R1')
        olds.append(old)
    record = {'protocol': PROTOCOL, 'parent_state': {'policy': policy, 'adam': adam, 'rng': rng0}, 'rng_after_all16_digest': rng1,
              'bundle_sha256': [b.sha256 for b in blocks], 'responses': sum(len(b.responses) for b in blocks), 'counters': ledger.phase('common').snapshot()}
    for unit, block, old in zip(units, blocks, olds):
        shared.publish_once(directory / f"bundle_unit{unit['index']}_PRIVATE.json", block.raw)
        shared.publish_once(directory / f"own_old_unit{unit['index']}_PRIVATE.json", {'bundle_sha256': block.sha256, 'old': old})
    record['receipt_sha256'] = identity(record)
    shared.publish_once(directory / 'common_record_PRIVATE.json', record)
    return SimpleNamespace(blocks=blocks, olds=olds, rng_dir=rng_dir, rng1=rng1, record=record)


# ---------------------------------------------------------------- one arm
def _arm(*, arm, backend, units, common, ledger, snapshot, parent, output, before, kind):
    directory = Path(output) / f'arm_{arm}'
    directory.mkdir()
    counters = ledger.phase(arm)
    before(f'micro_arm_{arm}_restore_parent')
    backend.restore(snapshot)
    require(_state(backend) == parent and backend.optimizer_step() == PARENT_ADAM_STEP, 'Arm did not start from the exact common parent (weights, full Adam, R0)')
    backend.counters = counters
    plans = [ms.arm_plan_for(block, arm) for block in common.blocks]
    joint, evidence = None, []
    try:
        for unit, block, planned, old, name in zip(units, common.blocks, plans, common.olds, ('sequence', 'dallas')):
            before(f'micro_arm_{arm}_{name}_gradient')
            gradient, info = backend.gradient(planned, old, unit)
            require(_state(backend) == parent, 'Backward mutated parameters, Adam or RNG')
            require(info.get('backward_calls', 0) == planned.backward_calls_needed and (gradient is None) == (planned.backward_calls_needed == 0),
                    'Block gradient inventory differs from admitted responses')
            evidence.append({'block': name, 'plan': _plan_record(planned), 'gradient_l2': float(info.get('gradient_l2', 0.)),
                             'weighted_loss': float(info.get('weighted_loss', 0.)), 'responses': info.get('responses', [])})
            if gradient is not None:
                require(backend.gradient_dtypes(gradient) == {'float32'}, 'Block gradient must be float32')
                joint = backend.scale_gradient(gradient, BLOCK_WEIGHT) if joint is None else backend.add_scaled(joint, gradient, BLOCK_WEIGHT)
            del gradient
            backend.release_graphs()          # first task graph is gone before the second starts
        admitted = sum(p.admitted_groups for p in plans)
        require((joint is None) == (admitted == 0), 'Joint gradient existence differs from admitted groups')
        norm = 0. if joint is None else float(backend.gradient_l2(joint))
        decision = shared.update_decision(SimpleNamespace(admitted_groups=admitted), norm)
        if joint is not None:
            require(backend.gradient_dtypes(joint) == {'float32'}, 'Joint gradient must be stored float32 only')
        before(f'micro_arm_{arm}_archive_gradient')
        shared.publish_once(directory / 'inputs_PRIVATE.json', {
            'bundle_sha256': [b.sha256 for b in common.blocks], 'bundles': [b.raw for b in common.blocks], 'own_old': common.olds,
            'rewards': {r.response_id: r.grade.total for b in common.blocks for r in b.responses}})
        gradient_file = None
        if joint is not None:
            gradient_file = directory / 'joint_gradient_float32_PRIVATE.safetensors'
            backend.save_gradient(gradient_file, joint)
            _fsync_file(gradient_file)
        shared.publish_once(directory / 'gradient_evidence_PRIVATE.json', {
            'arm': arm, 'blocks': evidence, 'joint_gradient_l2': norm, 'decision': decision, 'joint_gradient_sha256': None if gradient_file is None else shared.file_sha256(gradient_file),
            'joint_definition': '.5*sequence_block_gradient + .5*dallas_block_gradient; each block uses per-response weight 1/8; no reweighting of inactive block'})
        updated, files, checkpoint = False, {}, None
        if decision['update']:
            before(f'micro_arm_{arm}_before_update')
            require(_state(backend) == parent, 'State changed before the optimizer step')
            counters.charge('optimizer_updates')
            backend.apply_update(joint)
            updated = True
            after = _state(backend)
            require(after[2] == parent[2] and after[0] != parent[0] and after[1] != parent[1] and backend.optimizer_step() == UPDATED_ADAM_STEP,
                    'Joint step must move weights and Adam 52->53 and leave R0 unchanged')
            backend.load_rng(common.rng_dir)            # explicit protocol step: terminal RNG is the persisted post-all-16 R1
            require(backend.rng_digest() == common.rng1, 'Restored R1 differs from the persisted post-all-16 generation RNG')
            del joint
            backend.release_graphs()
            before(f'micro_arm_{arm}_checkpoint')
            terminal = _state(backend)
            pending = Path(tempfile.mkdtemp(prefix='.pending-checkpoint-', dir=directory))
            names = list(backend.save_checkpoint(pending))
            require(_state(backend) == terminal, 'Checkpoint save changed state')
            for name in names:
                _fsync_file(pending / name)
            os.rename(pending, directory / 'checkpoint_PRIVATE')       # atomic: checkpoint+optimizer+RNG commit together, THEN the receipt
            shared.sync_dir(directory)
            checkpoint = 'checkpoint_PRIVATE'
            files = {name: shared.file_sha256(directory / checkpoint / name) for name in names}
        else:
            require(_state(backend) == parent and backend.optimizer_step() == PARENT_ADAM_STEP, 'No-update arm changed the parent state')
        final = _state(backend)
        receipt = {'protocol': PROTOCOL, 'arm': arm, 'evidence_kind': kind, 'updated': updated, 'update_decision': decision,
                   'adam_step_after': backend.optimizer_step(), 'alias_of_parent': not updated, 'checkpoint': checkpoint, 'files_sha256': files,
                   'common_receipt_sha256': common.record['receipt_sha256'], 'bundle_sha256': [b.sha256 for b in common.blocks],
                   'parent_policy_sha256': parent[0], 'policy_sha_after': final[0], 'adam_digest_after': final[1], 'rng_digest_after': final[2],
                   'rng_terminal': 'R1_POST_ALL16_GENERATION' if updated else 'R0_PARENT_UNCHANGED_NO_UPDATE', 'future_onpolicy_sharing': False,
                   'effective_groups': sum(p.admitted_groups for p in plans), 'admitted_responses': sum(len(p.admitted_response_ids) for p in plans),
                   'counters': counters.snapshot()}
        receipt['receipt_sha256'] = identity(receipt)
        shared.publish_once(directory / 'receipt_PRIVATE.json', receipt)
        summary = {k: receipt[k] for k in ('arm', 'updated', 'adam_step_after', 'alias_of_parent', 'effective_groups', 'admitted_responses', 'receipt_sha256')}
        summary.update(update_reason=decision['reason'], counters=counters.snapshot(), science_claim=False)
        shared.replace_atomic(directory / 'complete_SAFE.json', summary)
        return summary
    finally:
        joint = None
        backend.restore(snapshot)
        require(_state(backend) == parent and backend.optimizer_step() == PARENT_ADAM_STEP, 'Rollback to the exact common parent failed')


# ---------------------------------------------------------------- transaction
def execute_micro(*, backend, units, prompt_lengths, reward_policy_sha256, output, before, expected_parent, checkpoint_receipt_sha256=None,
                  verify_parent_disk=None):
    """Run the whole micro transaction once. Fails closed; a failed/partial output is evidence, never resumed or retried here.

    `before(name)` is the caller's authorization+resource guard, invoked at every safe boundary. `expected_parent` is the trusted
    {'policy','adam','rng'} identity of the restored Adam52 parent; the backend must already be at it.
    """
    _check_backend(backend)
    require(callable(before), 'Explicit authorization/resource guard required')
    require(isinstance(expected_parent, dict) and set(expected_parent) == {'policy', 'adam', 'rng'}, 'Trusted parent identity required')
    budget = micro_budget(units, prompt_lengths)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)            # fresh output only: no resume of an unfinished reservation
    parent = _state(backend)
    kind = backend.evidence_kind
    ledger = Ledger(budget, output / 'counters_PRIVATE.json')
    try:
        require(parent == (expected_parent['policy'], expected_parent['adam'], expected_parent['rng']) and backend.optimizer_step() == PARENT_ADAM_STEP,
                'Backend is not at the trusted Adam52 common parent')
        shared.publish_once(output / 'start_PRIVATE.json', {'protocol': PROTOCOL, 'budget': budget, 'evidence_kind': kind, 'parent_state': dict(expected_parent),
                                                            'adam_step': PARENT_ADAM_STEP, 'automatic_retry': False})
        snapshot = backend.snapshot()
        common = _common(backend=backend, units=units, lengths=prompt_lengths, reward=reward_policy_sha256, ledger=ledger, parent=parent,
                         output=output, before=before, receipt=checkpoint_receipt_sha256)
        arms = {}
        for arm in ARMS:
            arms[arm] = _arm(arm=arm, backend=backend, units=units, common=common, ledger=ledger, snapshot=snapshot, parent=parent,
                             output=output, before=before, kind=kind)
        require(_state(backend) == parent, 'Parent not preserved after both arms')
        if verify_parent_disk is not None:
            require(verify_parent_disk() is True, 'Parent files on disk changed')
        spent = ledger.snapshot()
        total = {k: sum(spent[p][k] for p in spent) for k in CHARGEABLE}
        require(all(total[k] <= budget['total'][k] for k in CHARGEABLE), 'Total spent work exceeds the fixed micro caps')
        summary = {'status': 'ACTUAL_MICRO_TRANSACTION_COMPLETE_PENDING_INDEPENDENT_REVIEW' if kind == NATIVE_KIND else 'SYNTHETIC_FIXTURE_COMPLETE_NOT_NATIVE_EVIDENCE',
                   'protocol': PROTOCOL, 'evidence_kind': kind, 'arms': arms, 'common_receipt_sha256': common.record['receipt_sha256'],
                   'rng_after_all16_digest': common.rng1, 'parent_preserved': True, 'spent_total': total, 'optimizer_updates': sum(a['updated'] for a in arms.values()),
                   'next_stage_authorized': False, 'small_pilot_authorized': False, 'dev24_authorized': False, 'automatic_stage_progression': False,
                   'science_claim': False}
        shared.publish_once(output / 'micro_complete_SAFE.json', summary)
        return summary
    except BaseException as error:
        restored = False
        try:
            if 'snapshot' in locals():
                backend.restore(snapshot)
                restored = _state(backend) == parent
        except BaseException:
            restored = False
        shared.replace_atomic(output / f'STOP_{time.time_ns()}_PRIVATE.json', {
            'error_type': type(error).__name__, 'error': str(error), 'rolled_back_to_parent': restored, 'counters_spent_not_refunded': ledger.snapshot(),
            'automatic_retry': False})
        raise
