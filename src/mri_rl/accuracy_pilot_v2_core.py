"""Accuracy Pilot V2 backend-neutral core. IMPLEMENTED_CPU_NOT_NATIVE_RUN.

No MLX, model, generation or optimizer here; a backend supplies them (CPU fake in
tests, `accuracy_pilot_v2_native` only after a fresh exact human grant).

Frozen design (author: Claude Code, 2026-10-21):
  * Experiment 1 (prompt effect, NOT Vero RL): one predeclared concise instruction,
    reused verbatim from generation_recovery_v1 (already tested once on Adam60);
    appended to the mri_sequence question only. Verifier, parser, system prompt,
    assistant prefix and the 256-token cap are unchanged. Parent Adam52 greedy on
    the 8 frozen TRAIN images (the 4 training patients) under both prompts, then
    the frozen `prompt_decision` rule selects the common generation condition for
    ALL later rollouts and evaluations. No DEV selection.
  * Experiment 2 (Vero GSPO): the already-frozen 4-round TRAIN schedule (4 patient
    z045 FLAIR/T1 pairs + 8 Dallas engineering rows). Two joint updates per arm,
    each consuming TWO whole rounds (2 SEQ_PAIR + 2 DALLAS_BLOCK units, 32
    responses), so all 4 patients are exposed under the <=2 update cap. JOINT =
    sum_u 0.25*g_u where g_u is the unchanged unit gradient (1/8 per admitted
    response, inherited gate): task weight stays .5/.5 and each round counts 1/2.
    An inactive unit contributes zero and is NOT reweighted. V1 used one round per
    update (1/16 per admitted response); V2 is 1/32 per admitted response over a
    twice larger batch: this granularity difference is reported, never hidden.
  * Segment 0 consumes the ONE common parent rollout (units 0-3, generated once,
    immutable). After segment 0 every arm continues from the persisted post-common
    RNG R1 (explicit protocol step, as V1) and segment 1 (units 4-7) is generated,
    own-old scored and differentiated by that arm's OWN current policy. A rollout
    owned by another policy is rejected. No resampling, no forced update.
  * Experiment 3: DEV24 same-patient same-level FLAIR/T1 pairs. If the saved
    prompt/input_ids/mask/grid fingerprints are identical within a pair, the two
    real evaluations ARE an exact image-substitution comparison (no extra calls).
    It is labeled PAIRED_IMAGE_SUBSTITUTION_FROM_MAIN_EVALUATION, not a separate
    intervention.
"""
from __future__ import annotations
import copy
import hashlib
import math
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from mri_rl import multistep_gspo_core_v1 as ms
from mri_rl import shared_rollout_gspo_core_v1 as shared
from mri_rl.generation_recovery_v1 import SHORT_INSTRUCTION
from mri_rl.paired_prompt_groups_v1 import identity, canonical

STATUS = 'IMPLEMENTED_CPU_NOT_NATIVE_RUN'
PROTOCOL = 'ACCURACY_PILOT_V2_FOUR_PATIENT_TWO_UPDATE_VERO_GSPO_AB'
require = shared.require
CHARGEABLE = shared.CHARGEABLE
CONCISE_INSTRUCTION = SHORT_INSTRUCTION          # reused verbatim; never edited or searched
CONDITIONS = ('original', 'concise')
ORIGINAL_PROMPT_LENGTHS = {ms.SEQ_TASK: 213, ms.DALLAS_TASK: 405}
MAX_TOKENS = 256
PARENT_ADAM_STEP = 52
SEGMENTS = ((0, 1, 2, 3), (4, 5, 6, 7))
UNIT_WEIGHT = .25                     # .5 task weight x 1/2 rounds per block kind
LABELS = ('FLAIR', 'contrast-enhanced T1')
NATIVE_KIND, SYNTHETIC_KIND = 'NATIVE_MLX', 'CPU_SYNTHETIC_FIXTURE'
STAGES = ('P', 'C', 'A', 'B', 'EP', 'EA', 'EB')
EXECUTION = {'P': 'PROMPT_DIAGNOSTIC_TRAIN8', 'C': 'COMMON_PARENT_ROLLOUT', 'A': 'ARM_TRAIN', 'B': 'ARM_TRAIN',
             'EP': 'DEV24_EVAL', 'EA': 'DEV24_EVAL', 'EB': 'DEV24_EVAL'}
REQUIRED_BACKEND = ('policy_sha', 'adam_digest', 'rng_digest', 'optimizer_step', 'snapshot', 'generate', 'score_old', 'gradient',
                    'scale_gradient', 'add_scaled', 'gradient_l2', 'gradient_dtypes', 'save_gradient', 'release_graphs', 'apply_update',
                    'save_checkpoint', 'save_rng', 'load_rng', 'evidence_kind')
CONCISE_INSTRUCTION_TEXT = 'Keep the reasoning brief (at most two short sentences) and finish with the required boxed answer.'
require(CONCISE_INSTRUCTION == CONCISE_INSTRUCTION_TEXT, 'Predeclared concise instruction drifted from the frozen text')
CONCISE_INSTRUCTION_SHA256 = hashlib.sha256(CONCISE_INSTRUCTION.encode()).hexdigest()


# ================================================================ Experiment 1: one predeclared prompt
def condition_row(row, condition):
    """Return a copy of `row` under `condition`. Only the mri_sequence question text changes.

    The native template reads `extra_info['question']` (shared_rollout_gspo_backend_v1._prepare_context and
    paired_z045_matched_f32_eval_v1.prepared_inputs); id, images, labels, ability and Dallas rows are untouched.
    """
    require(condition in CONDITIONS, 'Unknown prompt condition')
    result = copy.deepcopy(row)
    if condition == 'original' or row['ability'] != ms.SEQ_TASK:
        return result
    question = result['extra_info']['question']
    require(isinstance(question, str) and question and CONCISE_INSTRUCTION not in question, 'Original sequence question required')
    result['extra_info']['question'] = question + '\n\n' + CONCISE_INSTRUCTION
    return result


def condition_units(units, condition):
    out = []
    for unit in units:
        u = copy.deepcopy(unit)
        u['contexts'] = [{**c, 'row': condition_row(c['row'], condition)} for c in unit['contexts']]
        out.append(u)
    return out


def prompt_lengths_for(condition, concise_sequence_length):
    require(condition in CONDITIONS, 'Unknown prompt condition')
    if condition == 'original':
        return dict(ORIGINAL_PROMPT_LENGTHS)
    require(type(concise_sequence_length) is int and ORIGINAL_PROMPT_LENGTHS[ms.SEQ_TASK] < concise_sequence_length <= 320,
            'Measured concise sequence prompt length required')
    return {ms.SEQ_TASK: concise_sequence_length, ms.DALLAS_TASK: ORIGINAL_PROMPT_LENGTHS[ms.DALLAS_TASK]}


def strict_label(text):
    from mri_rl.paired_z045_evaluation_v1 import strict_prediction
    return strict_prediction(text, ms.SEQ_TASK)


def greedy_summary(items):
    """items: [{'truth','text','tokens','wall_seconds'}]. Strict unchanged parser; invalid counted in denominators."""
    require(items, 'Rows required')
    counts, hits, predicted, invalid, tokens, wall, caps = [0, 0], [0, 0], {'FLAIR': 0, 'contrast-enhanced T1': 0, 'INVALID': 0}, 0, 0, 0., 0
    for item in items:
        y = LABELS.index(item['truth'])
        p = strict_label(item['text'])
        counts[y] += 1
        hits[y] += int(p == y)
        invalid += int(p < 0)
        predicted['INVALID' if p < 0 else LABELS[p]] += 1
        require(type(item['tokens']) is int and 0 <= item['tokens'] <= MAX_TOKENS, 'Bounded returned tokens required')
        tokens += item['tokens']
        caps += int(item['tokens'] >= MAX_TOKENS)
        wall += float(item.get('wall_seconds', 0.))
    recalls = [hits[i] / counts[i] if counts[i] else float('nan') for i in (0, 1)]
    return {'cases': len(items), 'correct': sum(hits), 'accuracy': sum(hits) / len(items), 'invalid': invalid, 'invalid_rate': invalid / len(items),
            'FLAIR_recall': recalls[0], 'T1_recall': recalls[1], 'balanced_accuracy': sum(recalls) / 2, 'class_counts': counts, 'class_correct': hits,
            'predicted_counts': predicted, 'returned_tokens': tokens, 'cap_hits': caps, 'generation_wall_seconds': wall}


PROMPT_DECISION_RULE = ('ADOPT_CONCISE iff, on the SAME 8 frozen TRAIN images and Parent Adam52 greedy: concise invalid <= original invalid, '
                        'no class loses a correct answer, and (concise invalid < original invalid OR concise correct > original correct). '
                        'Otherwise KEEP_ORIGINAL. Frozen before any call; DEV24 is never used for this decision.')


def prompt_decision(original, concise):
    require(original['cases'] == concise['cases'] == 8 and original['class_counts'] == concise['class_counts'] == [4, 4],
            'Exactly the same balanced 8 TRAIN images under both prompts required')
    checks = {'invalid_not_higher': concise['invalid'] <= original['invalid'],
              'no_class_loses_correct': all(concise['class_correct'][i] >= original['class_correct'][i] for i in (0, 1)),
              'strict_benefit': concise['invalid'] < original['invalid'] or concise['correct'] > original['correct']}
    decision = 'ADOPT_CONCISE' if all(checks.values()) else 'KEEP_ORIGINAL'
    return {'decision': decision, 'condition': 'concise' if decision == 'ADOPT_CONCISE' else 'original', 'checks': checks, 'rule': PROMPT_DECISION_RULE,
            'original': original, 'concise': concise, 'label': 'GENERATION_PROMPT_EFFECT_NOT_VERO_RL', 'unbounded_prompt_search': False,
            'token_cap_changed': False, 'verifier_changed': False}


# ================================================================ Experiment 2: schedule, budgets, batching
def validate_schedule(units, dev_patient_keys):
    """The already-frozen 4-round TRAIN schedule: 4 complete distinct patient pairs, 8 unique Dallas rows, TRAIN only, DEV-disjoint."""
    require(isinstance(units, list) and len(units) == 8, 'Exactly four whole rounds (eight units) required')
    require([u['kind'] for u in units] == [ms.SEQ_KIND, ms.DALLAS_KIND] * 4 and [u['index'] for u in units] == list(range(8)), 'SEQ_PAIR/DALLAS_BLOCK round order required')
    require([u['round'] for u in units] == [1, 1, 2, 2, 3, 3, 4, 4], 'Rounds 1..4 in order required')
    patients, levels, dallas = [], set(), []
    for unit in units:
        rows = [c['row'] for c in unit['contexts']]
        require(all(r['extra_info']['internal_split'] == 'train' for r in rows), 'TRAIN rows only')
        if unit['kind'] == ms.SEQ_KIND:
            require([c['key'] for c in unit['contexts']] == ['F', 'T'], 'F/T contexts required')
            require([r['reward_model']['ground_truth'] for r in rows] == list(LABELS), 'One FLAIR and one contrast-enhanced T1 row')
            require(len({r['extra_info']['patient_key'] for r in rows}) == 1, 'A pair is one patient')
            patients.append(rows[0]['extra_info']['patient_key'])
            levels.update(r['extra_info'].get('level') for r in rows)
        else:
            require([c['key'] for c in unit['contexts']] == ['D1', 'D2'] and all(r['ability'] == ms.DALLAS_TASK for r in rows), 'Dallas block required')
            dallas.extend(r['id'] for r in rows)
    require(len(set(patients)) == 4, 'Four distinct TRAIN patients required')
    require(len(set(dallas)) == 8, 'Eight unique Dallas engineering rows required')
    require(not set(patients) & set(dev_patient_keys), 'TRAIN/DEV patient membership must be disjoint')
    # Frozen TRAIN rows carry no model-visible level field by design. A missing field proves nothing about level: the
    # slice level is authenticated separately from the bound DATA56 source provenance (z045_provenance), never from rows.
    row_level = 'ABSENT_IN_MODEL_ROWS_BY_DESIGN' if levels == {None} else 'PRESENT_IN_ROWS_NOT_USED_AS_PROOF'
    return {'patients': 4, 'sequence_images': 8, 'class_counts': [4, 4], 'dallas_rows': 8, 'row_level_field': row_level,
            'DEV_disjoint': True, 'segments': [list(s) for s in SEGMENTS], 'patients_per_update': 2}


def z045_provenance(sequence_rows, pool, image_bindings):
    """Authenticate pinned z045 for every scheduled sequence row from the bound DATA56 capsule entries (actual schema:
    level, patient_key_PRIVATE, original_source_record_id_PRIVATE, modality_PRIVATE, expected_png_sha256, record_PRIVATE).
    Each scheduled row must equal its source record_PRIVATE exactly, with integer level 45, the source patient key, source id
    `<patient>_z045_<modality>_sequence`, modality matching the label, and expected_png_sha256 equal to the planned image
    SHA-256; 8 unique planned SHAs. pool: DATA56 entries; image_bindings: {row_id: sha256}. No legacy schema fallback."""
    require(len(sequence_rows) == 8 and len({r['id'] for r in sequence_rows}) == 8, 'Eight unique scheduled sequence rows required')
    by_id = {}
    for entry in pool:
        rec = entry['record_PRIVATE']; require(rec['id'] not in by_id, 'Duplicate DATA56 source row'); by_id[rec['id']] = entry
    pairs = {}
    for row in sequence_rows:
        require(row['id'] in by_id, 'Scheduled row absent from the bound DATA56 source')
        entry = by_id[row['id']]; rec = entry['record_PRIVATE']
        require(rec == row, 'Scheduled row differs from its bound DATA56 source record')
        require(type(entry.get('level')) is int and entry['level'] == 45, 'Bound source provenance must pin integer level 45')
        patient = entry.get('patient_key_PRIVATE')
        require(isinstance(patient, str) and patient == rec['extra_info']['patient_key'], 'Source patient key must equal the record patient')
        modality = entry.get('modality_PRIVATE')
        require(modality in ('flair', 't1ce') and {'flair': LABELS[0], 't1ce': LABELS[1]}[modality] == row['reward_model']['ground_truth'],
                'Source modality must match the target label')
        require(entry.get('original_source_record_id_PRIVATE') == '%s_z045_%s_sequence' % (patient, modality),
                'Source record identity must be this patient z045 with the target modality')
        require(row['id'] in image_bindings and entry.get('expected_png_sha256') == image_bindings[row['id']],
                'Source expected PNG SHA-256 must equal the planned image SHA-256')
        require(modality not in pairs.setdefault(patient, set()), 'Duplicate source modality for a patient'); pairs[patient].add(modality)
    require(len(pairs) == 4 and all(v == {'flair', 't1ce'} for v in pairs.values()), 'Four complete z045 FLAIR/T1 source pairs required')
    shas = [image_bindings[r['id']] for r in sequence_rows]
    require(all(shared.sha(s) for s in shas) and len(set(shas)) == 8, 'Eight actual unique planned MRI image SHA-256s required')
    return {'level': 45, 'basis': 'BOUND_DATA56_SOURCE_PROVENANCE: level + patient_key_PRIVATE + original_source_record_id_PRIVATE + '
                                  'modality_PRIVATE + expected_png_sha256 + exact record_PRIVATE per row',
            'rows_authenticated': 8, 'complete_patient_pairs': 4, 'unique_image_sha256': 8, 'row_level_field_used_as_proof': False}


def _sum(dicts):
    total = {k: 0 for k in CHARGEABLE}
    for d in dicts:
        for k, v in d.items():
            total[k] += v
    return total


def common_budget(units, lengths):
    require([u['index'] for u in units] == list(SEGMENTS[0]), 'Common rollout is exactly units 0-3')
    return _sum(ms.common_rollout_budget(u, lengths) for u in units)


def segment_budget(units, lengths, *, common):
    total = _sum(ms.unit_worst_case(u, lengths, common=common) for u in units)
    total['optimizer_updates'] = 1                      # one joint update per segment, never one per unit
    return total


def arm_budget(units, lengths):
    first = segment_budget([units[i] for i in SEGMENTS[0]], lengths, common=True)
    second = segment_budget([units[i] for i in SEGMENTS[1]], lengths, common=False)
    return {'segment0': first, 'segment1': second, 'limits': _sum((first, second)), 'optimizer_updates_max': 2,
            'weight_per_admitted_response_in_joint': UNIT_WEIGHT / ms.RESPONSES, 'v1_weight_per_admitted_response_in_joint': 1 / 16,
            'granularity_difference': 'V1: one round per update. V2: two rounds per update; per-response weight halves, batch doubles.',
            'automatic_retry': False, 'resampling': False, 'unused_budget_transfer': False}


def state(backend):
    return backend.policy_sha(), backend.adam_digest(), backend.rng_digest()


def check_backend(backend):
    missing = [n for n in REQUIRED_BACKEND if not hasattr(backend, n)]
    require(not missing, 'Backend lacks the V2 transaction protocol: ' + ','.join(missing))
    require(backend.evidence_kind in (NATIVE_KIND, SYNTHETIC_KIND), 'Backend must declare its evidence kind')


def _fsync(path):
    with Path(path).open('rb') as handle:
        os.fsync(handle.fileno())


def _publish_tree_once(pending, final):
    """pending dir (all files fsynced) -> atomic rename to final name."""
    for p in sorted(Path(pending).rglob('*')):
        if p.is_file():
            _fsync(p)
    for p in sorted((x for x in Path(pending).rglob('*') if x.is_dir()), reverse=True):
        shared.sync_dir(p)
    require(not Path(final).exists(), 'Committed segment cannot be replaced')
    os.rename(pending, final)
    shared.sync_dir(Path(final).parent)


# ---------------------------------------------------------------- the ONE shared parent rollout (units 0-3)
def make_common(*, backend, units, lengths, reward_policy_sha256, output, before, parent_receipt_sha256, counters):
    check_backend(backend)
    require([u['index'] for u in units] == list(SEGMENTS[0]), 'Common rollout covers exactly units 0-3')
    directory = Path(output)
    require(not directory.exists(), 'Common rollout is generated once only')
    pending = Path(tempfile.mkdtemp(prefix='.pending-common-', dir=directory.parent))
    backend.counters = counters
    policy, adam, rng0 = state(backend)
    require(backend.optimizer_step() == PARENT_ADAM_STEP, 'Common rollout only from the Adam52 parent')
    holder = backend.snapshot()
    raws, blocks = [], []
    for unit in units:
        before(f"v2_common_generate_unit_{unit['index']}")
        raw = backend.generate(unit, lengths)
        ms.ownership(raw['policy_sha256'], policy)
        require(backend.policy_sha() == policy and backend.adam_digest() == adam, 'Common generation changed parameters or Adam')
        raws.append(raw)
        blocks.append(ms.build_block(unit, raw, policy_sha=policy, reward_policy_sha256=reward_policy_sha256, checkpoint_receipt_sha256=parent_receipt_sha256))
    rng1 = backend.rng_digest()
    require(rng1 != rng0, 'Common generation did not consume RNG')
    (pending / 'rng_after_common').mkdir()
    backend.save_rng(pending / 'rng_after_common')
    olds = []
    for unit, block in zip(units, blocks):
        before(f"v2_common_score_old_unit_{unit['index']}")
        old = backend.score_old(unit, block.responses)
        ms.check_old(block, old)
        require(state(backend) == (policy, adam, rng1), 'Own-old scoring changed weights, Adam or R1')
        olds.append(old)
    holder = None
    record = {'protocol': PROTOCOL, 'parent_state': {'policy': policy, 'adam': adam, 'rng': rng0}, 'rng_after_common_digest': rng1,
              'bundle_sha256': [b.sha256 for b in blocks], 'responses': sum(len(b.responses) for b in blocks), 'counters': counters.snapshot(),
              'units': [u['index'] for u in units], 'shared_by_arms': ['A', 'B'], 'regenerated': False}
    record['receipt_sha256'] = identity(record)
    (pending / 'raw_rollout_and_own_old_PRIVATE.json').write_text(canonical({'raws': raws, 'olds': olds}) + '\n')
    (pending / 'common_record_PRIVATE.json').write_text(canonical(record) + '\n')
    _publish_tree_once(pending, directory)
    return record


def load_common(directory, units, *, reward_policy_sha256, parent_receipt_sha256, backend=None):
    """Rebuild the immutable common bundle; every arm verifies it against its own fresh-disk parent."""
    import json
    directory = Path(directory)
    record = json.loads((directory / 'common_record_PRIVATE.json').read_text())
    require(record['receipt_sha256'] == identity({k: v for k, v in record.items() if k != 'receipt_sha256'}), 'Common record changed')
    value = json.loads((directory / 'raw_rollout_and_own_old_PRIVATE.json').read_text())
    policy = record['parent_state']['policy']
    blocks = [ms.build_block(u, r, policy_sha=policy, reward_policy_sha256=reward_policy_sha256, checkpoint_receipt_sha256=parent_receipt_sha256)
              for u, r in zip(units, value['raws'])]
    require(len(blocks) == len(value['olds']) == 4 and record['bundle_sha256'] == [b.sha256 for b in blocks], 'Common bundle differs')
    for unit, raw, block, old in zip(units, value['raws'], blocks, value['olds']):
        ms.check_old(block, old)
        if backend is not None and hasattr(backend, '_rollout_contexts'):
            backend._rollout_contexts[(unit['index'], policy)] = raw['contexts']
    return SimpleNamespace(blocks=blocks, olds=value['olds'], rng_dir=directory / 'rng_after_common', rng1=record['rng_after_common_digest'],
                           record=record, parent_state=record['parent_state'])


# ---------------------------------------------------------------- one segment = one possible joint update
def _plan_record(planned, unit):
    groups = {}
    for item in planned.plan.items:
        groups.setdefault(item.group_index, []).append(item.response.response_id)
    rewards = {}
    for item in planned.plan.items:
        rewards.setdefault(item.group_index, []).append(item.response.grade.total)
    return {'unit': unit['index'], 'kind': unit['kind'], 'grouping': planned.plan.grouping, 'effective_groups': planned.admitted_groups,
            'admitted_response_ids': list(planned.admitted_response_ids), 'groups': [groups[k] for k in sorted(groups)],
            'zero_variance_groups': sum(1 for v in rewards.values() if max(v) - min(v) <= 1e-12),
            'items': [{'response_id': i.response.response_id, 'group': i.group_index, 'advantage': i.advantage, 'weight': i.weight,
                       'admitted': i.admitted, 'reward': i.response.grade.total, 'format': i.response.grade.format,
                       'raw_accuracy': i.response.grade.raw_accuracy, 'tokens': len(i.response.token_ids)} for i in planned.plan.items]}


def run_segment(*, arm, index, backend, units, lengths, reward_policy_sha256, counters, output, before, common=None, previous_receipt=None):
    """index 0: common parent rollout of units 0-3. index 1: OWN-policy rollout of units 4-7. <=1 joint update."""
    check_backend(backend)
    require(arm in shared.ARMS and index in (0, 1), 'Arm A/B and segment 0/1 required')
    require([u['index'] for u in units] == list(SEGMENTS[index]), 'Segment unit inventory differs')
    final = Path(output) / f'segment_{index}'
    require(not final.exists(), 'Committed segment is never replayed')
    pending = Path(tempfile.mkdtemp(prefix=f'.pending-segment{index}-', dir=Path(output)))
    backend.counters = counters
    before(f'v2_arm_{arm}_segment{index}_start')
    s0 = state(backend)
    step0 = backend.optimizer_step()
    holder = backend.snapshot()
    if index == 0:
        require(common is not None and previous_receipt is None, 'Segment 0 consumes only the common rollout')
        ps = common.parent_state
        require(s0 == (ps['policy'], ps['adam'], ps['rng']) and step0 == PARENT_ADAM_STEP,
                'Arm start (parameters+Adam+R0) differs from the trusted common parent: shared rollout not on-policy')
        blocks, olds, origin, post_gen = common.blocks, common.olds, 'COMMON_PARENT', s0
    else:
        require(common is None and shared.sha(previous_receipt), 'Segment 1 is own-policy and chained to segment 0')
        blocks = []
        for unit in units:
            before(f"v2_arm_{arm}_generate_unit_{unit['index']}")
            raw = backend.generate(unit, lengths)
            ms.ownership(raw['policy_sha256'], s0[0])
            require(backend.policy_sha() == s0[0] and backend.adam_digest() == s0[1], 'Generation changed parameters or Adam')
            blocks.append(ms.build_block(unit, raw, policy_sha=s0[0], reward_policy_sha256=reward_policy_sha256, checkpoint_receipt_sha256=previous_receipt))
            (pending / f"bundle_unit{unit['index']}_PRIVATE.json").write_text(canonical(blocks[-1].raw) + '\n')
        post_gen = state(backend)
        require(post_gen[2] != s0[2], 'Own generation did not consume RNG')
        olds = []
        for unit, block in zip(units, blocks):
            before(f"v2_arm_{arm}_score_old_unit_{unit['index']}")
            old = backend.score_old(unit, block.responses)
            ms.check_old(block, old)
            require(state(backend) == post_gen, 'Own-old scoring mutated parameters, Adam or RNG')
            olds.append(old)
        (pending / 'own_old_PRIVATE.json').write_text(canonical(olds) + '\n')
        origin = 'OWN_POLICY'
    joint, evidence, admitted = None, [], 0
    try:
        for unit, block, old in zip(units, blocks, olds):
            require(block.policy_sha256 == s0[0], 'Block policy differs from this arm policy before the update')
            before(f"v2_arm_{arm}_gradient_unit_{unit['index']}")
            planned = ms.arm_plan_for(block, arm)
            gradient, info = backend.gradient(planned, old, unit)
            require(state(backend) == post_gen, 'Backward mutated parameters, Adam or RNG')
            require(info.get('backward_calls', 0) == planned.backward_calls_needed and (gradient is None) == (planned.backward_calls_needed == 0),
                    'Unit gradient inventory differs from admitted responses')
            admitted += planned.admitted_groups
            evidence.append({**_plan_record(planned, unit), 'gradient_l2': float(info.get('gradient_l2', 0.)), 'weighted_loss': float(info.get('weighted_loss', 0.))})
            if gradient is not None:
                require(backend.gradient_dtypes(gradient) == {'float32'}, 'Unit gradient must be float32')
                joint = backend.scale_gradient(gradient, UNIT_WEIGHT) if joint is None else backend.add_scaled(joint, gradient, UNIT_WEIGHT)
            del gradient
            backend.release_graphs()
        require((joint is None) == (admitted == 0), 'Joint gradient existence differs from admitted groups')
        norm = 0. if joint is None else float(backend.gradient_l2(joint))
        decision = shared.update_decision(SimpleNamespace(admitted_groups=admitted), norm)
        if joint is not None:
            require(backend.gradient_dtypes(joint) == {'float32'}, 'Joint gradient must be float32')
            before(f'v2_arm_{arm}_segment{index}_archive_gradient')
            backend.save_gradient(pending / 'joint_gradient_float32_PRIVATE.safetensors', joint)
        before(f'v2_arm_{arm}_segment{index}_before_update')
        require(state(backend) == post_gen, 'State changed before the optimizer step')
        updated = False
        if decision['update']:
            counters.charge('optimizer_updates')
            backend.apply_update(joint)
            updated = True
            after = state(backend)
            require(after[2] == post_gen[2] and after[0] != post_gen[0] and after[1] != post_gen[1] and backend.optimizer_step() == step0 + 1,
                    'Joint step must move weights and Adam by one step and leave RNG unchanged')
        else:
            require(state(backend) == post_gen and backend.optimizer_step() == step0, 'No-update segment changed state')
    finally:
        joint = None
        backend.release_graphs()
    if index == 0:            # explicit protocol step (as V1): every arm continues from the persisted post-common R1
        backend.load_rng(common.rng_dir)
        require(backend.rng_digest() == common.rng1, 'Restored R1 differs from the persisted common-rollout RNG')
    holder = None
    before(f'v2_arm_{arm}_segment{index}_checkpoint')
    terminal = state(backend)
    (pending / 'checkpoint_PRIVATE').mkdir()
    names = list(backend.save_checkpoint(pending / 'checkpoint_PRIVATE'))
    require(state(backend) == terminal, 'Checkpoint save changed state')
    files = {name: shared.file_sha256(pending / 'checkpoint_PRIVATE' / name) for name in names}
    grades = [r.grade for b in blocks for r in b.responses]
    receipt = {'protocol': PROTOCOL, 'arm': arm, 'grouping': shared.ARMS[arm], 'segment': index, 'units': [u['index'] for u in units],
               'patients': [u['contexts'][0]['row']['extra_info']['patient_key'] for u in units if u['kind'] == ms.SEQ_KIND],
               'rollout_origin': origin, 'evidence_kind': backend.evidence_kind, 'bundle_sha256': [b.sha256 for b in blocks],
               'policy_sha_before': s0[0], 'adam_digest_before': s0[1], 'rng_digest_before': s0[2], 'adam_step_before': step0,
               'policy_sha_after': terminal[0], 'adam_digest_after': terminal[1], 'rng_digest_after': terminal[2], 'adam_step_after': backend.optimizer_step(),
               'updated': updated, 'update_decision': decision, 'joint_gradient_l2': norm, 'effective_groups': admitted,
               'admitted_responses': sum(len(e['admitted_response_ids']) for e in evidence), 'responses': len(grades),
               'invalid_format_responses': sum(g.format != 1 for g in grades), 'correct_responses': sum(g.raw_accuracy == 1 for g in grades),
               'zero_variance_groups': sum(e['zero_variance_groups'] for e in evidence),
               'rng_terminal': 'R1_POST_COMMON_GENERATION' if index == 0 else 'POST_OWN_GENERATION',
               'joint_definition': 'sum_u 0.25*g_u; g_u per-response 1/8 inherited gate; inactive unit contributes zero, not reweighted',
               'checkpoint': 'checkpoint_PRIVATE', 'files_sha256': files, 'prev_receipt_sha256': previous_receipt,
               'common_receipt_sha256': common.record['receipt_sha256'] if common is not None else None,
               'counters_after': counters.snapshot(), 'future_cross_arm_sharing': False, 'science_claim': False}
    receipt['receipt_sha256'] = identity(receipt)
    (pending / 'gradient_evidence_PRIVATE.json').write_text(canonical(evidence) + '\n')
    (pending / 'receipt_PRIVATE.json').write_text(canonical(receipt) + '\n')
    _publish_tree_once(pending, final)
    return receipt


def verify_segment_receipt(directory):
    import json
    directory = Path(directory)
    receipt = json.loads((directory / 'receipt_PRIVATE.json').read_text())
    require(receipt['receipt_sha256'] == identity({k: v for k, v in receipt.items() if k != 'receipt_sha256'}), 'Segment receipt changed')
    for name, digest in receipt['files_sha256'].items():
        require(shared.file_sha256(directory / 'checkpoint_PRIVATE' / name) == digest, 'Segment checkpoint payload changed')
    return receipt


def run_arm(*, arm, backend, units, lengths, reward_policy_sha256, common, counters, output, before, load_checkpoint=None):
    """Both segments of one arm in one process; a committed segment 0 may be resumed from its full checkpoint."""
    output = Path(output)
    # A committed final segment is terminal: reject BEFORE any checkpoint load or backend mutation.
    require(not (output / 'segment_1').exists(), 'Committed final segment is never replayed')
    output.mkdir(parents=True, exist_ok=True)
    receipts = []
    seg0 = output / 'segment_0'
    if seg0.exists():
        r0 = verify_segment_receipt(seg0)
        require(load_checkpoint is not None, 'Resume requires an explicit full checkpoint loader')
        load_checkpoint(seg0 / 'checkpoint_PRIVATE')
        require(state(backend) == (r0['policy_sha_after'], r0['adam_digest_after'], r0['rng_digest_after']) and backend.optimizer_step() == r0['adam_step_after'],
                'Resumed state differs from the committed segment-0 receipt')
    else:
        r0 = run_segment(arm=arm, index=0, backend=backend, units=[units[i] for i in SEGMENTS[0]], lengths=lengths,
                         reward_policy_sha256=reward_policy_sha256, counters=counters, output=output, before=before, common=common)
    receipts.append(r0)
    require(not (output / 'segment_1').exists(), 'Committed final segment is never replayed')
    r1 = run_segment(arm=arm, index=1, backend=backend, units=[units[i] for i in SEGMENTS[1]], lengths=lengths,
                     reward_policy_sha256=reward_policy_sha256, counters=counters, output=output, before=before, previous_receipt=r0['receipt_sha256'])
    receipts.append(r1)
    return arm_summary(arm, receipts, counters)


def arm_summary(arm, receipts, counters):
    return {'arm': arm, 'segments': len(receipts), 'optimizer_updates': sum(r['updated'] for r in receipts),
            'adam_step_after': receipts[-1]['adam_step_after'], 'policy_sha_after': receipts[-1]['policy_sha_after'],
            'patients_exposed': len({p for r in receipts for p in r['patients']}),
            'responses': sum(r['responses'] for r in receipts), 'admitted_responses': sum(r['admitted_responses'] for r in receipts),
            'effective_groups': sum(r['effective_groups'] for r in receipts), 'zero_variance_groups': sum(r['zero_variance_groups'] for r in receipts),
            'invalid_format_responses': sum(r['invalid_format_responses'] for r in receipts), 'correct_responses': sum(r['correct_responses'] for r in receipts),
            'update_reasons': [r['update_decision']['reason'] for r in receipts], 'receipt_sha256': [r['receipt_sha256'] for r in receipts],
            'common_rollout_segments': sum(r['rollout_origin'] == 'COMMON_PARENT' for r in receipts),
            'own_policy_segments': sum(r['rollout_origin'] == 'OWN_POLICY' for r in receipts), 'counters': counters.snapshot(), 'science_claim': False}


# ================================================================ Experiment 3: DEV24 paired image substitution
PAIR_FIELD = 'level'
SUBSTITUTION_LABEL = 'PAIRED_IMAGE_SUBSTITUTION_FROM_MAIN_EVALUATION'


def dev_pairs(records, field=PAIR_FIELD):
    """Group frozen DEV records into same-patient same-level FLAIR/T1 pairs. Returns [(F_index, T_index)]."""
    by = {}
    for i, rec in enumerate(records):
        extra = rec['row_PRIVATE']['extra_info']
        require('patient_key' in extra and field in extra, 'Pair key field absent: ' + field)
        by.setdefault((extra['patient_key'], str(extra[field])), {}).setdefault(rec['row_PRIVATE']['reward_model']['ground_truth'], []).append(i)
    pairs = []
    for key in sorted(by):
        group = by[key]
        require(set(group) == set(LABELS) and all(len(v) == 1 for v in group.values()), 'Every patient/level needs exactly one FLAIR and one T1')
        pairs.append((group[LABELS[0]][0], group[LABELS[1]][0]))
    return pairs


def substitution_equivalence(pairs, rows):
    """Exact check from saved generation fingerprints: identical prompt/input_ids/mask/grid/metadata, different pixels."""
    report = []
    for f, t in pairs:
        a, b = rows[f]['actual_PRIVATE'], rows[t]['actual_PRIVATE']
        fa, fb = a['input_fingerprints_PRIVATE'], b['input_fingerprints_PRIVATE']
        nonpixel = sorted(set(fa) | set(fb))
        same = {k: fa.get(k) == fb.get(k) for k in nonpixel if k != 'pixel_values'}
        report.append({'pair': len(report), 'prompt_sha256_equal': a['prompt_sha256'] == b['prompt_sha256'], 'prompt_length_equal': a['prompt_length'] == b['prompt_length'],
                       'nonpixel_fingerprints_equal': all(same.values()), 'nonpixel_fields': sorted(same), 'differing_nonpixel_fields': sorted(k for k, v in same.items() if not v),
                       'pixel_values_differ': fa.get('pixel_values', {}).get('sha256') != fb.get('pixel_values', {}).get('sha256'),
                       'image_sha256_differ': a['image_sha256'] != b['image_sha256']})
    exact = [r['prompt_sha256_equal'] and r['prompt_length_equal'] and r['nonpixel_fingerprints_equal'] and r['pixel_values_differ'] and r['image_sha256_differ'] for r in report]
    return {'pairs': len(report), 'exact_substitution_pairs': sum(exact), 'all_exact': all(exact), 'per_pair': report,
            'label': SUBSTITUTION_LABEL if all(exact) else 'NOT_EQUIVALENT_PREDECLARE_SEPARATE_SWAP',
            'independent_intervention': False}


def pair_category(pf, pt):
    if pf < 0 or pt < 0:
        return 'INVALID_INVOLVED'
    return {(0, 1): 'DISCRIMINATES_CORRECTLY', (0, 0): 'CONSTANT_FLAIR', (1, 1): 'CONSTANT_T1', (1, 0): 'REVERSED'}[(pf, pt)]


def pair_profile(pairs, rows):
    preds = [strict_label(r['text']) for r in rows]
    cats = [pair_category(preds[f], preds[t]) for f, t in pairs]
    valid = [c for c in cats if c != 'INVALID_INVOLVED']
    counts = {k: cats.count(k) for k in ('DISCRIMINATES_CORRECTLY', 'CONSTANT_FLAIR', 'CONSTANT_T1', 'REVERSED', 'INVALID_INVOLVED')}
    return {'categories': cats, 'counts': counts, 'valid_pairs': len(valid),
            'prediction_changes_with_image': sum(c in ('DISCRIMINATES_CORRECTLY', 'REVERSED') for c in valid),
            'predicted_T1': sum(p == 1 for p in preds), 'predicted_FLAIR': sum(p == 0 for p in preds), 'invalid': sum(p < 0 for p in preds)}


def binomial_two_sided(b, c):
    """Exact two-sided sign test on discordant counts. Descriptive only: cases are clustered in 4 patients."""
    n = b + c
    if n == 0:
        return 1.
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1., 2 * tail)


def compare_to_reference(pairs, reference_rows, child_rows):
    ref, child = pair_profile(pairs, reference_rows), pair_profile(pairs, child_rows)
    moves = list(zip(ref['categories'], child['categories']))
    gains = sum(r != 'DISCRIMINATES_CORRECTLY' and c == 'DISCRIMINATES_CORRECTLY' for r, c in moves)
    losses = sum(r == 'DISCRIMINATES_CORRECTLY' and c != 'DISCRIMINATES_CORRECTLY' for r, c in moves)
    to_t1 = sum(r == 'CONSTANT_FLAIR' and c == 'CONSTANT_T1' for r, c in moves)
    to_flair = sum(r == 'CONSTANT_T1' and c == 'CONSTANT_FLAIR' for r, c in moves)
    rp = [strict_label(r['text']) for r in reference_rows]
    cp = [strict_label(r['text']) for r in child_rows]
    truth = [LABELS.index(r['truth']) for r in reference_rows]
    require([LABELS.index(r['truth']) for r in child_rows] == truth and [r['id'] for r in child_rows] == [r['id'] for r in reference_rows], 'Same frozen rows required')
    w2c = sum(a != y and b == y for a, b, y in zip(rp, cp, truth))
    c2w = sum(a == y and b != y for a, b, y in zip(rp, cp, truth))
    if gains - losses >= 2 and gains > to_t1 + to_flair:
        reading = 'IMAGE_CONDITIONED_DISCRIMINATION_GAIN'
    elif (to_t1 + to_flair) > 0 and gains <= losses:
        reading = 'CLASS_PRIOR_SHIFT_PLAUSIBLE'
    else:
        reading = 'INCONCLUSIVE'
    return {'reference': ref, 'child': child, 'pair_transitions': [{'from': r, 'to': c} for r, c in moves],
            'discrimination_gains': gains, 'discrimination_losses': losses, 'constant_answer_shift_to_T1': to_t1, 'constant_answer_shift_to_FLAIR': to_flair,
            'case_wrong_to_correct': w2c, 'case_correct_to_wrong': c2w, 'sign_test_two_sided_p_DESCRIPTIVE': binomial_two_sided(w2c, c2w),
            'predicted_T1_change': child['predicted_T1'] - ref['predicted_T1'], 'reading': reading,
            'reading_rule': 'IMAGE gain iff pair gains-losses>=2 and gains>constant-answer shifts; PRIOR iff only constant-answer shifts with gains<=losses; else INCONCLUSIVE',
            'anatomical_grounding_claim': False}


# ================================================================ frozen scale decision (written before any V2 output)
SCALE_RULE = ('GO_PROPOSE_ROUND2 iff ALL: B balanced accuracy > A and > Parent (same frozen prompt); B correct exceeds both by >=2 cases; '
              'B T1 recall > Parent; B FLAIR recall >= Parent - 1/12; B invalid <= Parent invalid OR B beats Parent on cases valid for both; '
              'B-vs-Parent pair reading == IMAGE_CONDITIONED_DISCRIMINATION_GAIN; AND (method-specific, frozen 2026-10-21 before any V2 output) '
              'B-vs-A over ALL 12 DEV pairs: correct-discriminating pair gains - losses >= 2 with reading IMAGE_CONDITIONED_DISCRIMINATION_GAIN. '
              'If only B-vs-Parent passes, the result is INCONCLUSIVE (no Paired-Prompt-specific image evidence). '
              'NO_GO iff B balanced accuracy <= A and <= Parent, or B FLAIR recall < Parent - 1/12. '
              'Otherwise INCONCLUSIVE. Identical B/A pair category profiles are never reported as Paired-Prompt image grounding. '
              'Round 2 (<=8 patients/<=4 updates) is never automatic: it needs independent review and a new human grant.')
B_VS_A_MIN_NET_PAIR_GAIN = 2


def valid_both_correct(reference_rows, child_rows):
    rp = [strict_label(r['text']) for r in reference_rows]
    cp = [strict_label(r['text']) for r in child_rows]
    truth = [LABELS.index(r['truth']) for r in reference_rows]
    keep = [i for i in range(len(truth)) if rp[i] >= 0 and cp[i] >= 0]
    return {'cases_valid_for_both': len(keep), 'reference_correct': sum(rp[i] == truth[i] for i in keep), 'child_correct': sum(cp[i] == truth[i] for i in keep)}


def method_specific_image_evidence(pair_b_vs_a):
    """B-vs-A paired reading over all pairs (A is the reference). Descriptive; never an anatomical-grounding claim."""
    net = pair_b_vs_a['discrimination_gains'] - pair_b_vs_a['discrimination_losses']
    identical = pair_b_vs_a['reference']['categories'] == pair_b_vs_a['child']['categories']
    supported = not identical and net >= B_VS_A_MIN_NET_PAIR_GAIN and pair_b_vs_a['reading'] == 'IMAGE_CONDITIONED_DISCRIMINATION_GAIN'
    return {'pairs_denominator': len(pair_b_vs_a['reference']['categories']), 'B_minus_A_net_pair_gain': net,
            'identical_pair_category_profiles': identical, 'reading': pair_b_vs_a['reading'],
            'status': 'B_VS_A_PAIRED_GAIN_DESCRIPTIVE' if supported else ('NOT_DEMONSTRATED_IDENTICAL_PAIR_PROFILES' if identical else 'INCONCLUSIVE'),
            'paired_prompt_image_grounding_demonstrated': False}


def scale_decision(metrics, pair_vs_parent, valid_vs_parent, pair_b_vs_a):
    p, a, b = metrics['Parent'], metrics['A'], metrics['B']
    method = method_specific_image_evidence(pair_b_vs_a)
    checks = {'B_balanced_gt_A': b['balanced_accuracy'] > a['balanced_accuracy'], 'B_balanced_gt_Parent': b['balanced_accuracy'] > p['balanced_accuracy'],
              'B_beats_both_by_2_cases': b['correct'] - a['correct'] >= 2 and b['correct'] - p['correct'] >= 2,
              'B_T1_recall_gt_Parent': b['T1_recall'] > p['T1_recall'], 'B_FLAIR_not_below_Parent_minus_1_of_12': b['FLAIR_recall'] >= p['FLAIR_recall'] - 1 / 12 - 1e-12,
              'invalid_or_valid_only_gain': b['invalid'] <= p['invalid'] or valid_vs_parent['child_correct'] > valid_vs_parent['reference_correct'],
              'image_conditioned_pair_gain': pair_vs_parent['reading'] == 'IMAGE_CONDITIONED_DISCRIMINATION_GAIN',
              'B_vs_A_paired_image_gain': method['status'] == 'B_VS_A_PAIRED_GAIN_DESCRIPTIVE'}
    if all(checks.values()):
        decision = 'GO_PROPOSE_ROUND2'
    elif (b['balanced_accuracy'] <= a['balanced_accuracy'] and b['balanced_accuracy'] <= p['balanced_accuracy']) or not checks['B_FLAIR_not_below_Parent_minus_1_of_12']:
        decision = 'NO_GO'
    else:
        decision = 'INCONCLUSIVE'
    return {'decision': decision, 'checks': checks, 'rule': SCALE_RULE, 'method_specific_image_evidence': method, 'automatic_round2': False, 'generalization_claim': False,
            'cohort': 'FROZEN_INTERNAL_DEV24_PREVIOUSLY_EXPLORED'}


def effect_decomposition(parent_original, parent_frozen, a, b):
    """Prompt effect (Parent original->frozen prompt) is reported separately from RL effects at the frozen prompt."""
    keys = ('accuracy', 'balanced_accuracy', 'FLAIR_recall', 'T1_recall', 'invalid_rate')
    diff = lambda x, y: {k: x[k] - y[k] for k in keys}
    return {'prompt_effect_on_Parent': diff(parent_frozen, parent_original), 'RL_effect_A_vs_Parent_same_prompt': diff(a, parent_frozen),
            'RL_effect_B_vs_Parent_same_prompt': diff(b, parent_frozen), 'method_effect_B_vs_A': diff(b, a),
            'attribution_rule': 'Prompt gains are never attributed to Vero RL; RL effects compare checkpoints under the identical frozen prompt.'}
