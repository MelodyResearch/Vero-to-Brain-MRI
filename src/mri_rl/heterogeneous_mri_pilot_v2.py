"""Heterogeneous MRI pilot V2 (sequence FLAIR/T1ce + VQA-RAD midline shift): backend-neutral CPU core.
SOURCE_IMPLEMENTED_CPU_SYNTHETIC_NOT_NATIVE_RUN.

V1 (heterogeneous_mri_pilot_v1, executed once -> NO_SIGNAL/INVALID_CONTROL, all 6 greedy answers unboxed) is preserved unchanged
and reused: rows/units/grader/census/border controls/evaluation metrics/segment limits/STOP record. V2 changes ONLY:
  * ONE new frozen midline suffix (explicit symmetric boxed yes/no + full think/answer grammar); the original clinical question,
    the global system instruction, the sequence concise prompt, parser, reward, LR, LoRA, Adam and GSPO are unchanged;
  * Phase 1 hard gate = strict FORMAT on the two real TRUE TRAIN inputs only (FORMAT_BLOCKED otherwise). SWAP/BORDER answers and the
    v1 image-dependence classification are recorded as diagnostics and never veto (greedy decisions are not RL learnability);
  * Phase 2 signal gate before ANY backward: in BOTH tasks >=1 originally gate-admitted same-prompt K4 group must contain a
    strict-valid CORRECT and a strict-valid WRONG response (class contrast, not format-only reward variance). All 16 responses stay in
    the bundle with their original grades/advantages/weights; nothing is re-drawn, repaired or filtered. A zero/returned-nonfinite norm
    -> NO_GSPO_SIGNAL, 0 updates. Native nonfinite-leaf validation exceptions -> STOP (never swallowed).
    JOINT = .5*g_sequence + .5*g_midline, at most ONE Adam step 52->53;
  * every generated response persists text, token IDs, finish reason/EOS metadata and the F32/input trace (the v1 evidence gap);
  * Phase 3 only after a committed update (fresh processes): Parent midline EVAL2 (new prompt), child DEV24 + midline EVAL2 + the
    same 6 controls (diagnostic). Outcome REAL_PILOT_COMPLETED -> FEASIBLE / NEGATIVE_TRANSFER / NO_ACCURACY_GAIN.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from mri_rl import heterogeneous_mri_pilot_v1 as v1
from mri_rl import multistep_gspo_core_v1 as ms
from mri_rl import shared_rollout_gspo_core_v1 as shared
from mri_rl import safe_micro_training_v1 as stm
from mri_rl import vqarad_midline_task_v1 as vq
from mri_rl.paired_prompt_groups_v1 import identity

STATUS = 'SOURCE_IMPLEMENTED_CPU_SYNTHETIC_NOT_NATIVE_RUN'
PROTOCOL = 'HETEROGENEOUS_MRI_PILOT_V2_FORMAT_GATE_CLASS_CONTRAST_ONE_JOINT_STEP'
SEQ_TASK, MIDLINE_ABILITY, TASKS, TASK_WEIGHT = v1.SEQ_TASK, v1.MIDLINE_ABILITY, v1.TASKS, v1.TASK_WEIGHT
GATE_NAMES, MAX_TOKENS = v1.GATE_NAMES, v1.MAX_TOKENS
PARENT_ADAM_STEP, UPDATED_ADAM_STEP = v1.PARENT_ADAM_STEP, v1.UPDATED_ADAM_STEP
require = shared.require
# Frozen ONCE. '\\boxed' is a literal backslash (never the Python '\b' backspace). Symmetric: both labels shown in the full grammar.
MIDLINE_SUFFIX = ('Look at the image and answer yes or no. Keep the reasoning to one short sentence. Reply in exactly one of these two forms: '
                  '<think>one short sentence about the image</think><answer>\\boxed{yes}</answer> or '
                  '<think>one short sentence about the image</think><answer>\\boxed{no}</answer>')
MIDLINE_SUFFIX_SHA256 = hashlib.sha256(MIDLINE_SUFFIX.encode()).hexdigest()
FORMAT_GATE_CONTROLS = ('TRUE_0', 'TRUE_1')
SEQ_LABELS = ('FLAIR', 'contrast-enhanced T1')


# ================================================================ rows (v1 rows, only the frozen suffix differs)
def midline_row(record):
    row = v1.midline_row(record)
    original = row['extra_info']['original_question']
    require(row['extra_info']['question'] == original + '\n\n' + v1.MIDLINE_SUFFIX and MIDLINE_SUFFIX not in original, 'Unexpected v1 row form')
    row['extra_info'] = {**row['extra_info'], 'question': original + '\n\n' + MIDLINE_SUFFIX}
    return row


def strict_outcome(task, text, truth):
    """CORRECT / WRONG / INVALID under the UNCHANGED strict parsers (format must be 1; no repair, no auto-boxing)."""
    if task == SEQ_TASK:
        from mri_rl import accuracy_pilot_v2_core as v2core
        p = v2core.strict_label(text)
        return 'INVALID' if p < 0 else ('CORRECT' if p == SEQ_LABELS.index(truth) else 'WRONG')
    require(task == MIDLINE_ABILITY, 'Unknown task')
    g = vq.score_response(text, truth)
    return 'INVALID' if g['invalid'] else ('CORRECT' if g['strict_accuracy'] == 1 else 'WRONG')


def _jsonable(value):
    return json.loads(json.dumps(value, default=lambda o: sorted(o) if isinstance(o, (set, frozenset)) else repr(o)))


# ================================================================ Phase 1: hard FORMAT gate on TRUE inputs; SWAP/BORDER diagnostic only
def greedy_item(backend, *, task, name, row, image_sha256, parent, before, extra=None):
    """One greedy call; the record keeps text, token IDs, finish/EOS metadata and the whole F32/input trace."""
    before('hetero_v2_greedy_' + name)
    out = backend.greedy(row, image_sha256)
    require(backend.policy_sha() == parent[0] and backend.adam_digest() == parent[1], 'Greedy call changed parameters or Adam')
    require(isinstance(out.get('text'), str) and isinstance(out.get('token_ids'), list) and 0 < len(out['token_ids']) <= MAX_TOKENS and
            out.get('finish_reason') in ('stop', 'length'), 'Bounded greedy response with token IDs and finish reason required')
    truth = row['reward_model']['ground_truth']
    return {'name': name, 'task': task, 'id': row['id'], 'image_sha256': image_sha256, 'truth': truth, 'text': out['text'],
            'token_ids': list(out['token_ids']), 'tokens': len(out['token_ids']), 'finish_reason': out['finish_reason'],
            'EOS_coverage': out.get('EOS_coverage'), 'strict': strict_outcome(task, out['text'], truth),
            'actual_PRIVATE': _jsonable(out.get('actual_PRIVATE')), **(extra or {})}


def format_gate(records):
    """Hard gate: both TRUE TRAIN controls strictly valid. Image-following/question-following/border behaviour is DIAGNOSTIC ONLY."""
    by = {r['name']: r for r in records}
    require(set(by) == set(GATE_NAMES), 'Exactly the six frozen controls required')
    answers = {n: v1.parsed_yes_no(by[n]['text']) for n in GATE_NAMES}
    invalid = [n for n in FORMAT_GATE_CONTROLS if answers[n] is None]
    diagnostic = v1.image_dependence_gate(answers)
    return {'status': 'FORMAT_BLOCKED' if invalid else 'FORMAT_QUALIFIED', 'hard_gate': 'strict format on TRUE_0 and TRUE_1 (real TRAIN inputs)',
            'invalid_true_controls': invalid, 'answers': answers,
            'diagnostic_v1_image_dependence': {**diagnostic, 'use': 'DIAGNOSTIC_ONLY_NEVER_A_VETO'},
            'border_meaning': 'central occlusion, border retained: NOT a no-image or anatomy-grounding control',
            'greedy_note': 'identical greedy decisions do not imply identical image-conditional probabilities; no margins measured'}


# ================================================================ Phase 2: two-task class-contrast signal gate
def class_contrast(plan_record, outcomes):
    """plan_record: safe_micro_training_v1._plan_record of the ORIGINAL same-prompt plan. outcomes: response_id -> strict outcome.
    A group counts iff it was admitted by the inherited gate AND contains >=1 strict-valid CORRECT and >=1 strict-valid WRONG."""
    admitted = sorted({i['group'] for i in plan_record['items'] if i['admitted']})
    groups = {}
    for g in admitted:
        ids = plan_record['groups'][g]
        kinds = [outcomes[i] for i in ids]
        groups[g] = {'response_ids': ids, 'correct': kinds.count('CORRECT'), 'wrong': kinds.count('WRONG'), 'invalid': kinds.count('INVALID'),
                     'class_contrast': 'CORRECT' in kinds and 'WRONG' in kinds}
    return {'admitted_groups': len(admitted), 'groups_with_class_contrast': sum(v['class_contrast'] for v in groups.values()), 'groups': groups,
            'all_groups': len(plan_record['groups'])}


def _outcomes(unit, raw):
    truth = {c['key']: c['row']['reward_model']['ground_truth'] for c in unit['contexts']}
    task = SEQ_TASK if unit['kind'] == ms.SEQ_KIND else MIDLINE_ABILITY
    return {r['response_id']: strict_outcome(task, r['text'], truth[r['context_key']]) for r in raw['responses']}


def train_segment(*, backend, seq_unit, mid_unit, lengths, reward_policy_sha256, output, before, expected_parent, parent_receipt_sha256, row_lengths=None):
    """Generate the 16 real responses ONCE from Parent, own-old, original plans, two-task class-contrast gate, <=1 joint step,
    atomic checkpoint (adapter+full Adam+RNG) BEFORE the receipt. Any failure: STOP record, no in-process rollback (v1 _stop)."""
    stm._check_backend(backend)
    require(callable(before), 'Explicit authorization/resource guard required')
    require(seq_unit['kind'] == ms.SEQ_KIND and [c['key'] for c in seq_unit['contexts']] == ['F', 'T'] and
            mid_unit['kind'] == ms.DALLAS_KIND and [c['key'] for c in mid_unit['contexts']] == ['D1', 'D2'] and
            all(c['row']['ability'] == MIDLINE_ABILITY and c['row']['extra_info']['question'].endswith(MIDLINE_SUFFIX) for c in mid_unit['contexts']),
            'Exactly the sequence F/T unit and the V2 midline D1/D2 unit')
    require(set(lengths) == set(TASKS) and all(type(v) is int and 0 < v <= 2048 for v in lengths.values()), 'Frozen measured prompt reservations required')
    output = v1._fresh(output)
    before('hetero_segment_state_preflight')
    v1._require_parent(backend, expected_parent)
    parent = v1._state(backend)
    units = [(SEQ_TASK, seq_unit), (MIDLINE_ABILITY, mid_unit)]
    limits = v1.segment_limits([u for _, u in units], lengths, row_lengths)
    counters = shared.Counters(limits, None, lambda used: shared.replace_atomic(output / 'counters_PRIVATE.json', {'limits': limits, 'used': used}))
    backend.counters = counters
    before('hetero_segment_snapshot')
    backend.snapshot()          # native transaction_validate reference only; never used for an in-process rollback
    try:
        shared.publish_once(output / 'start_PRIVATE.json', {'protocol': PROTOCOL, 'parent_state': dict(expected_parent), 'limits': limits,
                                                            'task_weight': TASK_WEIGHT, 'automatic_retry': False, 'evidence_kind': backend.evidence_kind})
        blocks, raws = [], []
        for task, unit in units:
            before(f'hetero_v2_generate_{task}')
            raw = backend.generate(unit, lengths)
            ms.ownership(raw['policy_sha256'], parent[0])
            require(backend.policy_sha() == parent[0] and backend.adam_digest() == parent[1], 'Generation changed parameters or Adam')
            raws.append(raw)
            shared.publish_once(output / f'rollout_raw_{task}_PRIVATE.json', _jsonable(raw))      # all 8 responses, text+IDs+finish+EOS
            blocks.append(ms.build_block(unit, raw, policy_sha=parent[0], reward_policy_sha256=reward_policy_sha256,
                                         checkpoint_receipt_sha256=parent_receipt_sha256))
        rng1 = backend.rng_digest()
        olds = []
        for (task, unit), block in zip(units, blocks):
            before(f'hetero_v2_own_old_{task}')
            old = backend.score_old(unit, block.responses)
            ms.check_old(block, old)
            require(v1._state(backend) == (parent[0], parent[1], rng1), 'Own-old scoring changed weights, Adam or RNG')
            olds.append(old)
        plans = [ms.arm_plan_for(block, 'A') for block in blocks]              # unchanged same-prompt K4 grouping, 1/8 weights
        records = {task: stm._plan_record(p) for (task, _), p in zip(units, plans)}
        outcomes = {task: _outcomes(unit, raw) for (task, unit), raw in zip(units, raws)}
        contrast = {task: class_contrast(records[task], outcomes[task]) for task in TASKS}
        shared.publish_once(output / 'inputs_PRIVATE.json', {'bundle_sha256': [b.sha256 for b in blocks], 'bundles': [b.raw for b in blocks],
                                                             'own_old': olds, 'plans': records, 'strict_outcomes': outcomes, 'class_contrast': contrast})
        missing = [task for task in TASKS if contrast[task]['groups_with_class_contrast'] == 0]
        joint, evidence, backward = None, {}, 0
        if missing:
            decision = {'update': False, 'reason': 'NO_GSPO_SIGNAL', 'tasks_without_class_contrast': missing}
        else:
            nonfinite = []
            for (task, unit), planned, old in zip(units, plans, olds):
                before(f'hetero_v2_gradient_{task}')
                gradient, info = backend.gradient(planned, old, unit)
                require(v1._state(backend) == (parent[0], parent[1], rng1), 'Backward mutated parameters, Adam or RNG')
                require(gradient is not None and info.get('backward_calls', 0) == planned.backward_calls_needed, 'Task gradient inventory differs from admitted responses')
                require(backend.gradient_dtypes(gradient) == {'float32'}, 'Task gradient must be float32')
                backward += info['backward_calls']
                norm = float(backend.gradient_l2(gradient))
                evidence[task] = {'gradient_l2': norm if math.isfinite(norm) else None, 'backward_calls': info['backward_calls'], 'finite': math.isfinite(norm),
                                  'separate_from': 'format/correctness evidence (class_contrast) is recorded separately'}
                if not math.isfinite(norm):
                    nonfinite.append(task)
                joint = backend.scale_gradient(gradient, TASK_WEIGHT[task]) if joint is None else backend.add_scaled(joint, gradient, TASK_WEIGHT[task])
                del gradient
                backend.release_graphs()
            norm = float(backend.gradient_l2(joint))
            evidence['joint'] = {'gradient_l2': norm if math.isfinite(norm) else None, 'finite': math.isfinite(norm)}
            if nonfinite or not math.isfinite(norm) or norm == 0:
                decision = {'update': False, 'reason': 'NO_GSPO_SIGNAL', 'gradient_problem': 'NONFINITE' if (nonfinite or not math.isfinite(norm)) else 'ZERO',
                            'tasks_nonfinite': nonfinite}
                joint = None
            else:
                decision = {'update': True, 'reason': 'BOTH_TASKS_CLASS_CONTRAST_FINITE_NONZERO_JOINT_GRADIENT'}
        shared.publish_once(output / 'gradient_evidence_PRIVATE.json', {'decision': decision, 'evidence': evidence,
                                                                        'joint_definition': '.5*g_sequence + .5*g_midline; 1/8 per admitted response within task'})
        updated, files, checkpoint = False, {}, None
        if decision['update']:
            before('hetero_v2_before_update')
            require(v1._state(backend) == (parent[0], parent[1], rng1), 'State changed before the optimizer step')
            counters.charge('optimizer_updates')
            backend.apply_update(joint)
            updated = True
            after = v1._state(backend)
            require(after[0] != parent[0] and after[1] != parent[1] and after[2] == rng1 and backend.optimizer_step() == UPDATED_ADAM_STEP,
                    'Joint step must move weights and Adam 52->53 and leave the post-rollout RNG unchanged')
            joint = None
            backend.release_graphs()
            before('hetero_v2_checkpoint')
            pending = Path(tempfile.mkdtemp(prefix='.pending-checkpoint-', dir=output))
            names = list(backend.save_checkpoint(pending))
            require(v1._state(backend) == after, 'Checkpoint save changed state')
            for name in names:
                stm._fsync_file(pending / name)
            os.rename(pending, output / 'checkpoint_PRIVATE')
            shared.sync_dir(output)
            checkpoint = 'checkpoint_PRIVATE'
            files = {name: shared.file_sha256(output / checkpoint / name) for name in names}
        else:
            require(backend.policy_sha() == parent[0] and backend.adam_digest() == parent[1] and backend.optimizer_step() == PARENT_ADAM_STEP,
                    'No-update segment changed the Parent')
        final = v1._state(backend)
        receipt = {'protocol': PROTOCOL, 'evidence_kind': backend.evidence_kind, 'updated': updated, 'decision': decision, 'task_weight': TASK_WEIGHT,
                   'per_response_weight_within_task': ms.WEIGHT, 'admitted_groups': {t: c['admitted_groups'] for t, c in contrast.items()},
                   'class_contrast_groups': {t: c['groups_with_class_contrast'] for t, c in contrast.items()},
                   'strict_outcome_counts': {t: {k: list(o.values()).count(k) for k in ('CORRECT', 'WRONG', 'INVALID')} for t, o in outcomes.items()},
                   'responses': {t: len(o) for t, o in outcomes.items()}, 'backward_calls': backward, 'adam_step_after': backend.optimizer_step(),
                   'alias_of_parent': not updated, 'checkpoint': checkpoint, 'files_sha256': files, 'bundle_sha256': [b.sha256 for b in blocks],
                   'parent_policy_sha256': parent[0], 'parent_adam_digest': parent[1], 'parent_rng_digest': parent[2],
                   'policy_sha_after': final[0], 'adam_digest_after': final[1], 'rng_digest_after': final[2], 'counters': counters.snapshot(),
                   'block_transport': {MIDLINE_ABILITY: v1.TRANSPORT}, 'dallas_data': False}
        receipt['receipt_sha256'] = identity(receipt)
        shared.publish_once(output / 'receipt_PRIVATE.json', receipt)
        return receipt
    except BaseException as error:
        v1._stop(output, error, counters)
        raise


def run_train_phase(*, backend, seq_unit, midline_train, borders, lengths, reward_policy_sha256, output, before, expected_parent,
                    parent_receipt_sha256, row_lengths=None):
    """Parent process: 6 frozen controls (greedy) -> hard FORMAT gate on TRUE -> (qualified) 16 on-policy responses -> <=1 step."""
    output = v1._fresh(output)
    try:
        v1._require_parent(backend, expected_parent)
        parent = v1._state(backend)
        rng_dir = output / 'parent_rng_PRIVATE'
        rng_dir.mkdir()
        backend.save_rng(rng_dir)
        controls = v1.gate_controls(midline_train, borders)
        records = [greedy_item(backend, task=MIDLINE_ABILITY, name=c['name'], row=c['row'], image_sha256=c['image_sha256'], parent=parent, before=before,
                               extra={'kind': c['kind'], 'image_label': c['image_label']}) for c in controls]
        gate = format_gate(records)
        shared.publish_once(output / 'controls_pre_PRIVATE.json', {'gate': gate, 'controls': records})
        result = {'protocol': PROTOCOL, 'format_gate': gate, 'optimizer_updates': 0, 'greedy_calls': len(records)}
        if gate['status'] != 'FORMAT_QUALIFIED':
            result.update(status='FORMAT_BLOCKED', on_policy_responses=0, group_variance='NOT_APPLICABLE_NO_SAMPLED_GROUP')
            shared.publish_once(output / 'train_phase_complete_PRIVATE.json', result)
            return result
        before('hetero_before_rollout_rng_reload')
        backend.load_rng(rng_dir)                                                   # rollouts start from the exact saved Parent RNG
        require(v1._state(backend) == parent, 'Parent RNG could not be restored after greedy controls')
    except BaseException as error:
        v1._stop(output, error, None)
        raise
    segment = train_segment(backend=backend, seq_unit=seq_unit, mid_unit=v1.midline_unit(midline_train), lengths=lengths,
                            reward_policy_sha256=reward_policy_sha256, output=output / 'segment', before=before,
                            expected_parent=expected_parent, parent_receipt_sha256=parent_receipt_sha256, row_lengths=row_lengths)
    result.update(status='UPDATED' if segment['updated'] else 'NO_GSPO_SIGNAL', on_policy_responses=sum(segment['responses'].values()),
                  optimizer_updates=int(segment['updated']), segment_receipt_sha256=segment['receipt_sha256'], decision=segment['decision'],
                  class_contrast_groups=segment['class_contrast_groups'], admitted_groups=segment['admitted_groups'])
    shared.publish_once(output / 'train_phase_complete_PRIVATE.json', result)
    return result


# ================================================================ Phase 3 jobs, control comparison, outcome
def child_jobs(dev_rows, midline_eval, controls):
    """Child evaluation: the same frozen DEV24 rows, the same 2 midline EVAL rows and the SAME 6 control inputs as Phase 1."""
    require(len(dev_rows) == 24 and len(midline_eval) == 2 and [c['name'] for c in controls] == list(GATE_NAMES), 'Exact child inventory required')
    jobs = [(r['id'], SEQ_TASK, r, r['image_sha256'], {}) for r in dev_rows]
    jobs += [(r['id'], MIDLINE_ABILITY, r, r['image_sha256'], {}) for r in midline_eval]
    jobs += [(c['name'], MIDLINE_ABILITY, c['row'], c['image_sha256'], {'kind': c['kind'], 'image_label': c['image_label'], 'control': True}) for c in controls]
    return jobs


def control_comparison(pre, post):
    """Descriptive only: same prompts/images before (Parent) and after (child). Never an outcome input."""
    a, b = {r['name']: r for r in pre}, {r['name']: r for r in post}
    require(set(a) == set(b) == set(GATE_NAMES) and all(a[n]['id'] == b[n]['id'] and a[n]['image_sha256'] == b[n]['image_sha256'] for n in GATE_NAMES),
            'Post controls must use the identical prompts/images')
    return {n: {'pre': v1.parsed_yes_no(a[n]['text']), 'post': v1.parsed_yes_no(b[n]['text'])} for n in GATE_NAMES} | {'use': 'DIAGNOSTIC_ONLY'}


def outcome(parent, terminal, *, status, data_scale='DATA_TOO_SMALL_FOR_LEARNING_CLAIM'):
    """status: train-phase status. Only UPDATED (a committed step) can be REAL_PILOT_COMPLETED; then
    NEGATIVE_TRANSFER if any class recall (FLAIR, T1, yes, no) fell or any task's invalid count rose (hidden loss);
    FEASIBLE if equal-task strict balanced accuracy strictly increased with no hidden loss; otherwise NO_ACCURACY_GAIN."""
    base = {'accuracy_claim': 'NONE_TINY_FEASIBILITY_ONLY', 'efficacy_or_generalization_claim': False, 'data_scale': data_scale}
    if status != 'UPDATED':
        require(status in ('FORMAT_BLOCKED', 'NO_GSPO_SIGNAL'), 'Unknown train status')
        return {**base, 'outcome': status, 'completed_learning': False, 'evaluation': 'NOT_RUN_NO_COMMITTED_UPDATE'}
    p, t = parent['tasks'], terminal['tasks']
    recalls = {'FLAIR': (SEQ_TASK, 'FLAIR_recall'), 'T1': (SEQ_TASK, 'T1_recall'), 'yes': (MIDLINE_ABILITY, 'yes_recall'), 'no': (MIDLINE_ABILITY, 'no_recall')}
    losses = {k: t[task][key] < p[task][key] for k, (task, key) in recalls.items()}
    invalid_up = {task: t[task]['invalid'] > p[task]['invalid'] for task in TASKS}
    gain = terminal['equal_task'] > parent['equal_task']
    hidden = any(losses.values()) or any(invalid_up.values())
    label = 'NEGATIVE_TRANSFER' if hidden else ('FEASIBLE' if gain else 'NO_ACCURACY_GAIN')
    return {**base, 'outcome': 'REAL_PILOT_COMPLETED', 'result': label, 'completed_learning': True, 'class_recall_lost': losses, 'invalid_increased': invalid_up,
            'equal_task_gain': terminal['equal_task'] - parent['equal_task'], 'worst_task_delta': terminal['worst_task'] - parent['worst_task'],
            'per_task_strict_accuracy': {task: {'parent': p[task]['accuracy'] if task == SEQ_TASK else p[task]['strict_accuracy'],
                                                'child': t[task]['accuracy'] if task == SEQ_TASK else t[task]['strict_accuracy']} for task in TASKS}}


# ================================================================ exact prospective budget (3 serial loads)
PHASES = ('train', 'parent_eval', 'child_eval')
GREEDY = {'train': len(GATE_NAMES), 'parent_eval': 2, 'child_eval': 24 + 2 + len(GATE_NAMES)}


def budget(lengths, dev_rows=24):
    """lengths: {'sequence': int, 'by_row': {midline row id: int}, 'midline_train': {...}, 'midline_gate': {...}, 'midline_eval': {...}}."""
    flat = [lengths['sequence'], *lengths['midline_train'].values(), *lengths['midline_gate'].values(), *lengths['midline_eval'].values()]
    require(all(type(v) is int and 0 < v <= 2048 for v in flat), 'PENDING_PROCESSOR_MEASUREMENT: every V2 midline prompt length must be measured')
    require(len(lengths['midline_train']) == 2 and sorted(lengths['midline_gate']) == sorted(GATE_NAMES) and len(lengths['midline_eval']) == 2 and dev_rows == 24,
            'Exact midline inventory (2 TRAIN, 6 controls, 2 EVAL) and DEV24')
    S, T = lengths['sequence'], MAX_TOKENS
    tf_one = 8 * (S + T - 1) + sum(4 * (p + T - 1) for p in lengths['midline_train'].values())
    greedy = sum(GREEDY.values())
    return {'model_loads_max': 3, 'resident_models_max': 1, 'optimizer_updates_max': 1, 'sampled_rollout_responses': 16,
            'greedy_calls_max': greedy, 'greedy_by_phase': dict(GREEDY), 'parent_DEV24': 'AUTHENTICATED_V2_EP_CONCISE_ROWS_0_CALLS',
            'generations_max': 16 + greedy, 'generated_tokens_max': (16 + greedy) * T, 'detached_old_forwards': 16, 'gradient_forwards_max': 16,
            'backward_calls_max': 16, 'scorer_forwards_max': 32, 'own_old_TF_positions_max': tf_one, 'gradient_TF_positions_max': tf_one,
            'teacherforced_positions_total_max': 2 * tf_one,
            'greedy_prompt_prefill_positions_max': 2 * sum(lengths['midline_gate'].values()) + 2 * sum(lengths['midline_eval'].values()) + dev_rows * S,
            'conditional': {'parent_eval': 'only after a committed update', 'child_eval': 'only after a committed update and parent_eval PASS'},
            'automatic_retry': False, 'resampling': False, 'prompt_lengths': lengths}
