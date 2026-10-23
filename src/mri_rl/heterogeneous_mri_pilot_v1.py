"""First real heterogeneous MRI pilot (sequence FLAIR/T1ce + VQA-RAD midline shift): backend-neutral CPU core.
SOURCE_IMPLEMENTED_CPU_SYNTHETIC_NOT_NATIVE_RUN.

No MLX, model, generation or optimizer here; a backend supplies them (CPU fake in tests, `heterogeneous_mri_pilot_native_v1`
only after exact review + fresh human permission). Reused, not copied: multistep_gspo_core_v1 blocks/plans (same-prompt K4,
inherited gate, 1/8 per admitted response), shared_rollout_gspo_core_v1 counters/decision/atomic publish, the
safe_micro_training_v1 joint-step/checkpoint pattern and vqarad_midline_task_v1 parser/case components.

Experiment (ONE arm, direct same-prompt Vero GSPO from Parent Adam52; no third arm, no new loss, no coefficient search):
  1. pre-update image-dependence gate on the 2 midline TRAIN items only (2 TRUE + 2 opposite-label image SWAPs + 2 frozen
     central-occlusion/border-retained controls, greedy); image dependence, not correctness;
  2. Parent midline EVAL (2 greedy) - Parent DEV24 sequence rows are the journal-authenticated V2 EP rows (same concise contract);
  3. 16 real responses generated ONCE: sequence F4/T4 (existing frozen TRAIN pair) + midline D1 K4/D2 K4 (2 TRAIN items);
  4. own-old -> plans -> NO_SIGNAL unless BOTH tasks have >=1 admitted group -> at most ONE joint Adam step 52->53 with
     JOINT = .5*g_sequence + .5*g_midline (explicit equal task weight; response counts never weight tasks);
     any failure -> STOP record, no in-process rollback (process release), nothing replayed;
  5. terminal greedy DEV24 + 2 midline EVAL in a fresh process (0-update terminal is the Parent alias, not evaluated).
"""
from __future__ import annotations
from collections import Counter, defaultdict
import hashlib
import os
from pathlib import Path
import re
import tempfile
import time
from types import SimpleNamespace
from mri_rl import multistep_gspo_core_v1 as ms
from mri_rl import shared_rollout_gspo_core_v1 as shared
from mri_rl import safe_micro_training_v1 as stm
from mri_rl import vqarad_midline_task_v1 as vq
from mri_rl.paired_prompt_groups_v1 import identity
from mri_rl.vero_mri_reward import _answer_and_format
from mri_rl.generation_recovery_v1 import SHORT_INSTRUCTION

STATUS = 'SOURCE_IMPLEMENTED_CPU_SYNTHETIC_NOT_NATIVE_RUN'
PROTOCOL = 'HETEROGENEOUS_MRI_PILOT_V1_DIRECT_SAME_PROMPT_ONE_JOINT_STEP'
SEQ_TASK = ms.SEQ_TASK
MIDLINE_ABILITY = 'vqarad_midline_shift'
MIDLINE_SOURCE = 'vqarad_midline_v1'
TASKS = (SEQ_TASK, MIDLINE_ABILITY)
TASK_WEIGHT = {SEQ_TASK: .5, MIDLINE_ABILITY: .5}                   # frozen explicit equal task weight
PARENT_ADAM_STEP, UPDATED_ADAM_STEP = stm.PARENT_ADAM_STEP, stm.UPDATED_ADAM_STEP
MAX_TOKENS = shared.MAX_TOKENS
require = shared.require
CHARGEABLE = shared.CHARGEABLE
MIDLINE_SUFFIX = 'Answer yes or no. ' + SHORT_INSTRUCTION           # frozen ONCE (task class + existing concise instruction); no search
MIDLINE_SUFFIX_SHA256 = hashlib.sha256(MIDLINE_SUFFIX.encode()).hexdigest()
BORDER_KIND = 'CENTRAL_OCCLUSION_BORDER_RETAINED__NOT_A_BRAIN_MASK'
GATE_NAMES = ('TRUE_0', 'TRUE_1', 'SWAP_0', 'SWAP_1', 'BORDER_0', 'BORDER_1')
NO_IMAGE_CONTROL = {'status': 'NOT_RUN_EXISTING_PATH_REQUIRES_EXACTLY_ONE_IMAGE',
                    'reason': 'native _prepare_context requires one <|image_pad|> and num_images=1; a no-image path would be new unqualified code'}

# ================================================================ once-only source census (CPU, metadata only)
MIDLINE = re.compile(r'\bmid(?:line|light)\b.*\bshift(?:ed|ing)?\b|\bshift(?:ed|ing)?\b.*\bmid(?:line|light)\b', re.I)
SPELLING = re.compile(r'\b(mid(?:line|light)|shift(?:ed|ing)?)\b', re.I)
MASS_SHIFT = re.compile(r'\bmass\b.*\bshift(?:ed|ing)?\b|\bshift(?:ed|ing)?\b.*\bmass\b', re.I)          # no explicit midline: AMBIGUOUS
RELATED = re.compile(r'\bshift(?:ed|ing)?\b.*\bmiddle\b|\bcross(?:es|ed|ing)?\b.*\bmidline\b|\bmidline\b.*\bdeviat', re.I)
MRI_WORDS = re.compile(r'\b(mri|mr|t1|t2|flair|dwi|adc|weighted|diffusion|gadolinium)\b', re.I)
CT_WORDS = re.compile(r'\b(ct|computed\s+tomography|cat\s+scan)\b', re.I)
THRESHOLD = {'train': {'yes': 2, 'no': 2}, 'eval': {'yes': 2, 'no': 2}}


def _yes_no(row):
    answer = str(row.get('answer', '')).strip().lower()
    return answer if answer in ('yes', 'no') else None


def _kinds(text):
    return ({'MRI'} if MRI_WORDS.search(text) else set()) | ({'CT'} if CT_WORDS.search(text) else set())


def _image_modality(rows):
    """Per image, original source text only: a MODALITY answer naming MRI/sequence or CT, or a 'yes' to a same-image question
    naming them (affirmed). A 'no' affirms nothing. Both kinds -> UNCERTAIN. Not an acquisition-metadata or visual check."""
    found = defaultdict(set)
    for row in rows:
        question, answer = str(row.get('question', '')), str(row.get('answer', '')).strip().lower()
        if 'MODALITY' in [t.strip() for t in str(row.get('question_type', '')).upper().split(',')]:
            found[row['image_name']] |= _kinds(answer)
        if answer == 'yes':
            found[row['image_name']] |= _kinds(question)
    return {image: (next(iter(kinds)) if len(kinds) == 1 else 'UNCERTAIN') for image, kinds in found.items() if kinds}


def _question_modality(row, modality):
    own = _kinds(str(row.get('question', '')))
    kinds = own | ({modality[row['image_name']]} if row['image_name'] in modality else set())
    return next(iter(kinds)) if len(kinds) == 1 else 'UNCERTAIN'


def census(rows):
    """Every HEAD yes/no midline-shift question (incl. 'midlight'/'shifted'); mass-shift wording counted separately as AMBIGUOUS.

    Eligible = MRI evidence + evaluated + CLOSED + PRES; deduplicated by conservative source-case component (image<->MedPix case
    union graph); a component with both answers is a CONFLICT and excluded. Counts only; no patient is invented or certified."""
    components = vq._case_components(rows)
    modality = _image_modality(rows)
    midline, ambiguous, related, spellings = [], 0, 0, set()
    for row in rows:
        if row.get('image_organ') != 'HEAD' or _yes_no(row) is None:
            continue
        question = str(row.get('question', ''))
        if MIDLINE.search(question):
            midline.append(row)
            spellings.update(m.lower() for m in SPELLING.findall(question))
        elif MASS_SHIFT.search(question):
            ambiguous += 1
        elif RELATED.search(question):
            related += 1
    kind = {id(r): _question_modality(r, modality) for r in midline}
    by_modality = Counter(kind[id(r)] for r in midline)
    evaluation = Counter(str(r.get('evaluation')) for r in midline)
    def components_of(admit):
        answers, missing = defaultdict(set), 0
        for r in midline:
            if kind[id(r)] == 'MRI' and admit(r):
                component = components.get(r['image_name'])
                if component is None:
                    missing += 1
                else:
                    answers[component].add(_yes_no(r))
        return {'yes': sum(a == {'yes'} for a in answers.values()), 'no': sum(a == {'no'} for a in answers.values()),
                'conflict': sum(len(a) > 1 for a in answers.values())}, missing
    eligible, no_component = components_of(lambda r: r.get('evaluation') == 'evaluated' and str(r.get('answer_type', '')).strip() == 'CLOSED'
                                           and 'PRES' in [t.strip() for t in str(r.get('question_type', '')).split(',')])
    all_mri, _ = components_of(lambda r: True)
    need = {k: THRESHOLD['train'][k] + THRESHOLD['eval'][k] for k in ('yes', 'no')}
    sufficient = eligible['yes'] >= need['yes'] and eligible['no'] >= need['no']
    return {'schema': 'heterogeneous_mri_midline_census_v1', 'source_rows': len(rows), 'midline_yesno_head_questions': len(midline),
            'mass_shift_ambiguous_questions': ambiguous, 'related_paraphrase_questions': related, 'matched_spellings': sorted(spellings),
            'by_modality_evidence': {k: by_modality[k] for k in ('MRI', 'CT', 'UNCERTAIN') if by_modality[k]},
            'evaluated': evaluation['evaluated'], 'not_evaluated': evaluation['not evaluated'], 'evaluation_status': dict(evaluation),
            'eligible_case_components': eligible, 'eligible_without_case_component': no_component, 'threshold': THRESHOLD,
            'MRI_components_including_not_evaluated_NOT_ADMITTED': all_mri,
            'admission_rule': 'evaluated + CLOSED + PRES + MRI source-text evidence; not-evaluated QA is reported, never promoted',
            'decision': 'SUFFICIENT_FOR_4_TRAIN_4_EVAL' if sufficient else 'DATA_TOO_SMALL_FOR_LEARNING_CLAIM',
            'modality_rule': 'original source text only: same-image MODALITY answer or affirmed (yes) MRI/sequence/CT question, or MRI wording '
                             'in the question itself; both kinds -> UNCERTAIN; no image reading, no acquisition metadata',
            'mass_shift_rule': 'mass + shift without explicit midline: AMBIGUOUS, never merged; "mass effect" alone is not a shift endpoint',
            'split_unit': 'conservative image<->MedPix case component; patient registry NOT_VERIFIED'}


# ================================================================ rows, grader, units
def midline_row(record):
    """Model row from one hash-bound vqarad_midline_source_candidate_v1 record. The original clinical question is kept verbatim
    (`original_question`); the prompt appends ONLY the frozen task-format suffix, joined exactly as V2 condition_row ('\n\n')."""
    require(record['split'] in ('train', 'eval') and record['ground_truth'] in ('yes', 'no'), 'TRAIN/EVAL yes/no record required (reserve never used)')
    require(record['model_input']['question'] == record['question'] and MIDLINE_SUFFIX not in record['question'], 'Original question must be unchanged')
    question = record['question'] + '\n\n' + MIDLINE_SUFFIX
    return {'id': 'vqarad_' + str(record['source_qid']), 'ability': MIDLINE_ABILITY, 'data_source': MIDLINE_SOURCE,
            'images': [record['model_input']['image']], 'image_sha256': record['source_image_sha256'],
            'reward_model': {'ground_truth': record['ground_truth']},
            'extra_info': {'question': question, 'original_question': record['question'], 'source_qid': str(record['source_qid']),
                           'split': record['split'], 'case_proxy_sha256': record.get('case_proxy_sha256')}}


def routing_grader(sequence_grader):
    """Task-routed REAL reward: midline -> unchanged vqarad yes/no verifier (raw 0.8*acc+0.2*format); others -> sequence grader."""
    def grader(row, text, config):
        if row['ability'] != MIDLINE_ABILITY:
            return sequence_grader(row, text, config)
        g = vq.score_response(text, row['reward_model']['ground_truth'])
        return {'score': g['score'], 'accuracy': g['accuracy'], 'format': g['format'], 'raw_accuracy': g['accuracy']}
    return grader


TRANSPORT = 'multistep DALLAS_BLOCK same-prompt two-context schema reused as TRANSPORT ONLY; actual task vqarad_midline_shift; no Dallas data'


def midline_unit(train_rows, index=1, round_=1):
    """The unchanged two-context same-prompt K4 block (multistep DALLAS_KIND path), contexts D1/D2 ordered by source qid."""
    labels = sorted(r['reward_model']['ground_truth'] for r in train_rows)
    require(len(train_rows) == 2 and labels == ['no', 'yes'] and len({r['id'] for r in train_rows}) == 2 and
            all(r['ability'] == MIDLINE_ABILITY and r['extra_info']['split'] == 'train' for r in train_rows),
            'Exactly two distinct label-balanced midline TRAIN rows required')
    rows = sorted(train_rows, key=lambda r: int(r['extra_info']['source_qid']))
    return {'index': index, 'round': round_, 'kind': ms.DALLAS_KIND, 'task': MIDLINE_ABILITY, 'transport_schema': TRANSPORT,
            'contexts': [{'key': 'D1', 'row': rows[0]}, {'key': 'D2', 'row': rows[1]}]}


# ================================================================ pre-update image-dependence gate (TRAIN only, greedy, 6 controls)
def border_controls(record, train_rows):
    """The 2 predeclared CPU-made central-occlusion/border-retained TRAIN images (frozen before any model output), one per TRAIN item,
    bound by qid, source image SHA and control SHA. LIMITED diagnostic: the rectangle is not a brain mask, may retain peripheral MRI
    pixels and certifies neither watermark exclusion nor anatomy grounding."""
    require(record.get('schema') == 'heterogeneous-central-occlusion-controls-v1' and record.get('model_outputs_used') is False and
            len(record.get('controls', [])) == 2, 'Exact frozen border-control record required')
    by = {r['extra_info']['source_qid']: r for r in train_rows}
    out = {}
    for c in record['controls']:
        row = by.get(str(c['source_qid']))
        require(row is not None and c['source_sha256'] == row['image_sha256'] and c['control_kind'] == BORDER_KIND and
                c['question_original_unchanged'] == row['extra_info']['original_question'] and shared.sha(c['control_sha256']) and
                c['control_sha256'] != row['image_sha256'], 'Border control does not bind its TRAIN item, source image and unchanged question')
        out[row['id']] = {'path': c['control_path'], 'sha256': c['control_sha256']}
    require(set(out) == set(by[q]['id'] for q in by), 'One border control per TRAIN item required')
    return out


def gate_controls(train_rows, borders):
    """TRUE_i (question_i, image_i), SWAP_i (question_i, opposite-label image_j), BORDER_i (question_i, central-occluded image_i).
    The question text is identical across a row's three controls."""
    rows = sorted(train_rows, key=lambda r: int(r['extra_info']['source_qid']))
    require(len(rows) == 2 and {r['reward_model']['ground_truth'] for r in rows} == {'yes', 'no'} and
            all(r['extra_info']['split'] == 'train' for r in rows), 'Gate uses the two label-balanced TRAIN items only')
    require(set(borders) == {r['id'] for r in rows}, 'Border controls must be exactly the two TRAIN items')
    controls = [{'name': f'TRUE_{i}', 'kind': 'TRUE', 'row': r, 'image_sha256': r['image_sha256'],
                 'image_label': r['reward_model']['ground_truth']} for i, r in enumerate(rows)]
    for i, r in enumerate(rows):
        other = rows[1 - i]
        swapped = {**r, 'id': r['id'] + '__image_of_' + other['id'], 'images': list(other['images']), 'image_sha256': other['image_sha256'],
                   'reward_model': {'ground_truth': other['reward_model']['ground_truth']}}
        controls.append({'name': f'SWAP_{i}', 'kind': 'SWAP', 'row': swapped, 'image_sha256': other['image_sha256'],
                         'image_label': other['reward_model']['ground_truth']})
    for i, r in enumerate(rows):
        b = borders[r['id']]
        occluded = {**r, 'id': r['id'] + '__central_occluded', 'images': [b['path']], 'image_sha256': b['sha256'],
                    'reward_model': {'ground_truth': r['reward_model']['ground_truth']}}
        controls.append({'name': f'BORDER_{i}', 'kind': 'BORDER', 'row': occluded, 'image_sha256': b['sha256'], 'image_label': None})
    require([c['name'] for c in controls] == list(GATE_NAMES), 'Control inventory/order differs')
    return controls


def parsed_yes_no(text):
    answer, fmt = _answer_and_format(text)
    answer = answer.strip().lower()
    return answer if fmt == 1. and answer in ('yes', 'no') else None


def image_dependence_gate(answers):
    """Predeclared conservative rule (image DEPENDENCE, not correctness). PASS iff all six strictly valid, TRUE_0 != TRUE_1, each SWAP
    answer equals the TRUE answer of the image it was shown (so the positive AND the inverted pattern pass), and the border-retained
    pair does NOT reproduce the TRUE pair (else the answer vector is explainable by retained borders/watermarks -> NO_SIGNAL).
    Not a grounding, anatomy or watermark-exclusion claim."""
    require(set(answers) == set(GATE_NAMES), 'Exactly six gate controls required')
    a = answers
    if any(v not in ('yes', 'no') for v in a.values()):
        reason = 'INVALID_CONTROL'
    elif len({a['TRUE_0'], a['TRUE_1'], a['SWAP_0'], a['SWAP_1']}) == 1:
        reason = 'CONSTANT_ANSWER'
    elif a['TRUE_0'] != a['TRUE_1'] and a['SWAP_0'] == a['TRUE_1'] and a['SWAP_1'] == a['TRUE_0']:
        reason = 'BORDER_REPRODUCES_TRUE' if (a['BORDER_0'], a['BORDER_1']) == (a['TRUE_0'], a['TRUE_1']) else 'IMAGE_FOLLOWING'
    elif a['TRUE_0'] != a['TRUE_1'] and a['SWAP_0'] == a['TRUE_0'] and a['SWAP_1'] == a['TRUE_1']:
        reason = 'QUESTION_FOLLOWING'
    else:
        reason = 'MIXED'
    return {'verdict': 'PASS' if reason == 'IMAGE_FOLLOWING' else 'NO_SIGNAL', 'reason': reason, 'answers': dict(a),
            'controls': len(GATE_NAMES), 'decoding': 'greedy', 'no_image_control': NO_IMAGE_CONTROL, 'uses_eval_or_dev': False,
            'grounding_claim': 'NOT_CLAIMED', 'border_control': 'LIMITED_DIAGNOSTIC_NOT_A_BRAIN_MASK',
            'rule': 'all 6 valid AND TRUE_0!=TRUE_1 AND SWAP_i==TRUE_(1-i) AND (BORDER_0,BORDER_1)!=(TRUE_0,TRUE_1)'}


# ================================================================ helpers
def _state(backend):
    return backend.policy_sha(), backend.adam_digest(), backend.rng_digest()


def _fresh(output):
    output = Path(output)
    require(not output.exists(), 'Output already exists: a pilot operation is never replayed or resumed')
    output.mkdir(parents=True)
    return output


def _require_parent(backend, expected_parent):
    require(isinstance(expected_parent, dict) and set(expected_parent) == {'policy', 'adam', 'rng'}, 'Trusted parent identity required')
    require(_state(backend) == (expected_parent['policy'], expected_parent['adam'], expected_parent['rng']) and
            backend.optimizer_step() == PARENT_ADAM_STEP, 'Backend is not at the trusted Parent Adam52 state')


def segment_limits(units, lengths, row_lengths=None):
    """Exact worst-case endpoints of the 16-response segment: own generation + own-old + <=16 backward, ONE update.
    With `row_lengths` (frozen measured expanded prompt length per row id) the teacher-forced cap is exact per context
    (own-old + gradient: 2 * sum_context 4*(P_row + 255)) instead of the unit's max-length reservation."""
    limits = {k: 0 for k in CHARGEABLE}
    for unit in units:
        for k, v in ms.unit_worst_case(unit, lengths).items():
            limits[k] += v
    if row_lengths is not None:
        ids = [c['row']['id'] for u in units for c in u['contexts']]
        require(all(type(row_lengths.get(i)) is int and 0 < row_lengths[i] <= lengths[c['row']['ability']]
                    for u in units for c in u['contexts'] for i in [c['row']['id']]) and len(ids) == len(set(ids)),
                'Exact frozen per-row lengths (within the task reservation) required for every rollout context')
        limits['teacherforced_input_positions'] = 2 * sum(ms.GROUP_SIZE * (row_lengths[i] + MAX_TOKENS - 1) for i in ids)
    limits['optimizer_updates'] = 1
    return limits


# ================================================================ the one joint transaction (or NO_SIGNAL)
def train_segment(*, backend, seq_unit, mid_unit, lengths, reward_policy_sha256, output, before, expected_parent, parent_receipt_sha256, row_lengths=None):
    """Generate 16 real responses once from Parent, score own-old, gate per task, <=1 joint step, atomic checkpoint, receipt."""
    stm._check_backend(backend)
    require(callable(before), 'Explicit authorization/resource guard required')
    before('hetero_segment_state_preflight')
    require(seq_unit['kind'] == ms.SEQ_KIND and [c['key'] for c in seq_unit['contexts']] == ['F', 'T'] and
            mid_unit['kind'] == ms.DALLAS_KIND and [c['key'] for c in mid_unit['contexts']] == ['D1', 'D2'] and
            all(c['row']['ability'] == MIDLINE_ABILITY for c in mid_unit['contexts']), 'Exactly the sequence F/T unit and the midline D1/D2 unit')
    require(set(lengths) == set(TASKS) and all(type(v) is int and 0 < v <= 2048 for v in lengths.values()), 'Frozen measured prompt reservations required')
    output = _fresh(output)
    _require_parent(backend, expected_parent)
    parent = _state(backend)
    units = [(SEQ_TASK, seq_unit), (MIDLINE_ABILITY, mid_unit)]
    limits = segment_limits([u for _, u in units], lengths, row_lengths)
    counters = shared.Counters(limits, None, lambda used: shared.replace_atomic(output / 'counters_PRIVATE.json', {'limits': limits, 'used': used}))
    backend.counters = counters
    before('hetero_segment_snapshot')
    backend.snapshot()          # native transaction_validate reference only; never used for an in-process rollback
    try:
        shared.publish_once(output / 'start_PRIVATE.json', {'protocol': PROTOCOL, 'parent_state': dict(expected_parent), 'limits': limits,
                                                            'task_weight': TASK_WEIGHT, 'automatic_retry': False, 'evidence_kind': backend.evidence_kind})
        blocks = []
        for task, unit in units:
            before(f'hetero_generate_{task}')
            raw = backend.generate(unit, lengths)
            ms.ownership(raw['policy_sha256'], parent[0])
            require(backend.policy_sha() == parent[0] and backend.adam_digest() == parent[1], 'Generation changed parameters or Adam')
            blocks.append(ms.build_block(unit, raw, policy_sha=parent[0], reward_policy_sha256=reward_policy_sha256,
                                         checkpoint_receipt_sha256=parent_receipt_sha256))
        rng1 = backend.rng_digest()
        olds = []
        for (task, unit), block in zip(units, blocks):
            before(f'hetero_own_old_{task}')
            old = backend.score_old(unit, block.responses)
            ms.check_old(block, old)
            require(_state(backend) == (parent[0], parent[1], rng1), 'Own-old scoring changed weights, Adam or RNG')
            olds.append(old)
        plans = [ms.arm_plan_for(block, 'A') for block in blocks]          # A == unchanged same-prompt K4 grouping for both tasks
        admitted = {task: p.admitted_groups for (task, _), p in zip(units, plans)}
        missing = [task for task in TASKS if admitted[task] == 0]
        shared.publish_once(output / 'inputs_PRIVATE.json', {
            'bundle_sha256': [b.sha256 for b in blocks], 'bundles': [b.raw for b in blocks], 'own_old': olds,
            'plans': {task: stm._plan_record(p) for (task, _), p in zip(units, plans)}})
        joint, evidence, backward = None, [], 0
        if missing:
            decision = {'update': False, 'reason': 'NO_SIGNAL', 'tasks_without_valid_contrast': missing}
        else:
            for (task, unit), planned, old in zip(units, plans, olds):
                before(f'hetero_gradient_{task}')
                gradient, info = backend.gradient(planned, old, unit)
                require(_state(backend) == (parent[0], parent[1], rng1), 'Backward mutated parameters, Adam or RNG')
                require(gradient is not None and info.get('backward_calls', 0) == planned.backward_calls_needed, 'Task gradient inventory differs from admitted responses')
                require(backend.gradient_dtypes(gradient) == {'float32'}, 'Task gradient must be float32')
                backward += info['backward_calls']
                evidence.append({'task': task, 'gradient_l2': float(info.get('gradient_l2', 0.)), 'backward_calls': info['backward_calls']})
                w = TASK_WEIGHT[task]
                joint = backend.scale_gradient(gradient, w) if joint is None else backend.add_scaled(joint, gradient, w)
                del gradient
                backend.release_graphs()
            norm = float(backend.gradient_l2(joint))
            decision = shared.update_decision(SimpleNamespace(admitted_groups=sum(admitted.values())), norm)
            evidence.append({'joint_gradient_l2': norm})
        shared.publish_once(output / 'gradient_evidence_PRIVATE.json', {'decision': decision, 'evidence': evidence, 'admitted_groups': admitted,
                                                                        'joint_definition': '.5*g_sequence + .5*g_midline; 1/8 per admitted response within task'})
        updated, files, checkpoint = False, {}, None
        if decision['update']:
            before('hetero_before_update')
            require(_state(backend) == (parent[0], parent[1], rng1), 'State changed before the optimizer step')
            counters.charge('optimizer_updates')
            backend.apply_update(joint)
            updated = True
            after = _state(backend)
            require(after[0] != parent[0] and after[1] != parent[1] and after[2] == rng1 and backend.optimizer_step() == UPDATED_ADAM_STEP,
                    'Joint step must move weights and Adam 52->53 and leave the post-rollout RNG unchanged')
            joint = None
            backend.release_graphs()
            before('hetero_checkpoint')
            pending = Path(tempfile.mkdtemp(prefix='.pending-checkpoint-', dir=output))
            names = list(backend.save_checkpoint(pending))
            require(_state(backend) == after, 'Checkpoint save changed state')
            for name in names:
                stm._fsync_file(pending / name)
            os.rename(pending, output / 'checkpoint_PRIVATE')                # adapter + full Adam + RNG commit together, then the receipt
            shared.sync_dir(output)
            checkpoint = 'checkpoint_PRIVATE'
            files = {name: shared.file_sha256(output / checkpoint / name) for name in names}
        else:
            require(backend.policy_sha() == parent[0] and backend.adam_digest() == parent[1] and backend.optimizer_step() == PARENT_ADAM_STEP,
                    'No-update segment changed the Parent')
        final = _state(backend)
        receipt = {'protocol': PROTOCOL, 'evidence_kind': backend.evidence_kind, 'updated': updated, 'decision': decision,
                   'task_weight': TASK_WEIGHT, 'per_response_weight_within_task': ms.WEIGHT, 'admitted_groups': admitted,
                   'admitted_responses': {task: len(p.admitted_response_ids) for (task, _), p in zip(units, plans)},
                   'backward_calls': backward, 'adam_step_after': backend.optimizer_step(), 'alias_of_parent': not updated,
                   'checkpoint': checkpoint, 'files_sha256': files, 'bundle_sha256': [b.sha256 for b in blocks],
                   'parent_policy_sha256': parent[0], 'parent_adam_digest': parent[1], 'parent_rng_digest': parent[2],
                   'policy_sha_after': final[0], 'adam_digest_after': final[1], 'rng_digest_after': final[2], 'counters': counters.snapshot(),
                   'tasks': {SEQ_TASK: 'existing calibrated sequence grader', MIDLINE_ABILITY: 'raw vqarad yes/no verifier (no calibration)'},
                   'block_transport': {MIDLINE_ABILITY: TRANSPORT}, 'dallas_data': False}
        receipt['receipt_sha256'] = identity(receipt)
        shared.publish_once(output / 'receipt_PRIVATE.json', receipt)
        return receipt
    except BaseException as error:
        _stop(output, error, counters)
        raise


def _stop(output, error, counters):
    """NO in-process rollback: a restore copies GPU state, which under RED/pause/fatal pressure is new work against the release
    policy. Record STOP (spent work stays charged), drop the counters reference, and let the child process be released; the Parent
    on disk is unchanged and an unfinished operation is never replayed."""
    shared.replace_atomic(Path(output) / f'STOP_{time.time_ns()}_PRIVATE.json', {
        'error_type': type(error).__name__, 'error': str(error), 'in_process_rollback': False, 'resident_state': 'TAINTED_RELEASE_PROCESS',
        'counters_spent_not_refunded': None if counters is None else counters.snapshot(), 'automatic_retry': False})


# ================================================================ Parent phase: 6 gate controls -> Parent midline EVAL -> segment
def _greedy(backend, row, image_sha256, parent, before, name):
    before('hetero_greedy_' + name)
    out = backend.greedy(row, image_sha256)
    require(backend.policy_sha() == parent[0] and backend.adam_digest() == parent[1], 'Greedy call changed parameters or Adam')
    require(isinstance(out.get('text'), str) and isinstance(out.get('token_ids'), list) and len(out['token_ids']) <= MAX_TOKENS, 'Bounded greedy response required')
    return out


def run_parent_phase(*, backend, seq_unit, midline_train, midline_eval, borders, lengths, reward_policy_sha256, output, before, expected_parent,
                     parent_receipt_sha256, row_lengths=None):
    output = _fresh(output)
    try:
        _require_parent(backend, expected_parent)
        parent = _state(backend)
        rng_dir = output / 'parent_rng_PRIVATE'
        rng_dir.mkdir()
        backend.save_rng(rng_dir)
        controls = gate_controls(midline_train, borders)
        records, answers = [], {}
        for c in controls:
            out = _greedy(backend, c['row'], c['image_sha256'], parent, before, c['name'])
            answers[c['name']] = parsed_yes_no(out['text'])
            records.append({'name': c['name'], 'kind': c['kind'], 'row_id': c['row']['id'], 'image_sha256': c['image_sha256'], 'text': out['text'],
                            'tokens': len(out['token_ids']), 'image_label': c['image_label'], 'actual_PRIVATE': out.get('actual_PRIVATE')})
        gate = image_dependence_gate(answers)
        gate['source_label_agreement_descriptive'] = {c['name']: answers[c['name']] == c['image_label'] for c in controls if c['image_label']}
        shared.publish_once(output / 'gate_PRIVATE.json', {'gate': gate, 'controls': records})
        result = {'protocol': PROTOCOL, 'gate': gate, 'optimizer_updates': 0, 'parent_midline_eval_count': 0, 'greedy_calls': len(controls)}
        if gate['verdict'] != 'PASS':
            result.update(status='NO_SIGNAL_IMAGE_GATE', gradient=False)
            shared.publish_once(output / 'parent_phase_complete_PRIVATE.json', result)
            return result
        require(len(midline_eval) == 2 and sorted(r['reward_model']['ground_truth'] for r in midline_eval) == ['no', 'yes'] and
                all(r['extra_info']['split'] == 'eval' for r in midline_eval), 'Exactly the two label-balanced midline EVAL rows')
        items = []
        for r in midline_eval:
            out = _greedy(backend, r, r['image_sha256'], parent, before, 'eval_' + r['extra_info']['source_qid'])
            items.append({'task': MIDLINE_ABILITY, 'id': r['id'], 'truth': r['reward_model']['ground_truth'], 'text': out['text'], 'tokens': len(out['token_ids']),
                          'actual_PRIVATE': out.get('actual_PRIVATE')})
        shared.publish_once(output / 'parent_midline_eval_PRIVATE.json', {'items': items})
        before('hetero_before_rollout_rng_reload')
        backend.load_rng(rng_dir)                                                   # rollouts start from the exact saved Parent RNG
        require(_state(backend) == parent, 'Parent RNG could not be restored after greedy controls')
    except BaseException as error:
        _stop(output, error, None)
        raise
    segment = train_segment(backend=backend, seq_unit=seq_unit, mid_unit=midline_unit(midline_train), lengths=lengths,
                            reward_policy_sha256=reward_policy_sha256, output=output / 'segment', before=before,
                            expected_parent=expected_parent, parent_receipt_sha256=parent_receipt_sha256, row_lengths=row_lengths)
    result.update(status='UPDATED' if segment['updated'] else 'NO_SIGNAL', gradient=segment['backward_calls'] > 0, greedy_calls=len(controls) + len(items),
                  optimizer_updates=int(segment['updated']), parent_midline_eval_count=len(items), segment_receipt_sha256=segment['receipt_sha256'],
                  decision=segment['decision'], admitted_groups=segment['admitted_groups'])
    shared.publish_once(output / 'parent_phase_complete_PRIVATE.json', result)
    return result


# ================================================================ evaluation + predeclared outcome
EVAL_INVENTORY = {SEQ_TASK: {'FLAIR': 12, 'contrast-enhanced T1': 12}, MIDLINE_ABILITY: {'yes': 1, 'no': 1}}


def evaluation_summary(rows, inventory=EVAL_INVENTORY):
    """Strict greedy metrics with the UNCHANGED parsers; invalid stays in every denominator. Sequence via V2 greedy_summary."""
    from mri_rl import accuracy_pilot_v2_core as v2core
    by = {task: [r for r in rows if r['task'] == task] for task in TASKS}
    require(len(rows) == sum(len(v) for v in by.values()) and len({r['id'] for r in rows}) == len(rows) and
            all(dict(Counter(r['truth'] for r in by[t])) == inventory[t] for t in TASKS), 'Evaluation inventory differs from the frozen DEV24 + 2 midline EVAL')
    seq = v2core.greedy_summary([{**r, 'tokens': r.get('tokens', 0)} for r in by[SEQ_TASK]])
    correct = {r['id']: int(v2core.strict_label(r['text']) == v2core.LABELS.index(r['truth'])) for r in by[SEQ_TASK]}
    hits, invalid = Counter(), 0
    for r in by[MIDLINE_ABILITY]:
        g = vq.score_response(r['text'], r['truth'])
        invalid += int(g['invalid'])
        hits[r['truth']] += int(g['strict_accuracy'] == 1)
        correct[r['id']] = int(g['strict_accuracy'] == 1)
    counts = Counter(r['truth'] for r in by[MIDLINE_ABILITY])
    recalls = {k: hits[k] / counts[k] for k in ('yes', 'no')}
    mid = {'items': len(by[MIDLINE_ABILITY]), 'strict_accuracy': sum(hits.values()) / len(by[MIDLINE_ABILITY]), 'invalid': invalid,
           'yes_recall': recalls['yes'], 'no_recall': recalls['no'], 'balanced_accuracy': (recalls['yes'] + recalls['no']) / 2}
    tasks = {SEQ_TASK: seq, MIDLINE_ABILITY: mid}
    scores = [tasks[t]['balanced_accuracy'] for t in TASKS]
    return {'tasks': tasks, 'equal_task': sum(scores) / 2, 'worst_task': min(scores), 'task_score': 'per-task balanced strict accuracy',
            'item_correct': correct, 'item_task': {r['id']: r['task'] for r in rows}}


def outcome(parent, terminal, *, updated, data_scale='DATA_TOO_SMALL_FOR_LEARNING_CLAIM'):
    """Predeclared, in this order:
      0 updates                                   -> NO_SIGNAL (terminal is the Parent alias);
      any task's strict invalid count increased   -> FAIL_FORMAT_COLLAPSE (not a transfer finding);
      FLAIR or T1 recall, or ANY task's balanced accuracy below Parent -> NEGATIVE_TRANSFER;
      otherwise                                   -> FEASIBLE.
    Per-item correct->wrong / wrong->correct flips are reported per task. Never an accuracy or generalization claim at this scale."""
    base = {'accuracy_claim': 'NONE_FEASIBILITY_ONLY', 'data_scale': data_scale, 'generalization_claim': False}
    if not updated:
        return {**base, 'outcome': 'NO_SIGNAL', 'terminal_is_parent_alias': True}
    p, t = parent['tasks'], terminal['tasks']
    require(set(parent['item_correct']) == set(terminal['item_correct']), 'Parent and terminal must be evaluated on the same items')
    flips = {task: {'correct_to_wrong': sum(1 for i, k in parent['item_task'].items() if k == task and parent['item_correct'][i] and not terminal['item_correct'][i]),
                    'wrong_to_correct': sum(1 for i, k in parent['item_task'].items() if k == task and not parent['item_correct'][i] and terminal['item_correct'][i])}
             for task in TASKS}
    format_checks = {task + '_invalid_not_higher': t[task]['invalid'] <= p[task]['invalid'] for task in TASKS}
    transfer_checks = {'FLAIR_recall_not_lower': t[SEQ_TASK]['FLAIR_recall'] >= p[SEQ_TASK]['FLAIR_recall'],
                       'T1_recall_not_lower': t[SEQ_TASK]['T1_recall'] >= p[SEQ_TASK]['T1_recall'],
                       **{task + '_balanced_not_lower': t[task]['balanced_accuracy'] >= p[task]['balanced_accuracy'] for task in TASKS}}
    deltas = {'equal_task': terminal['equal_task'] - parent['equal_task'], 'worst_task': terminal['worst_task'] - parent['worst_task'],
              **{task: t[task]['balanced_accuracy'] - p[task]['balanced_accuracy'] for task in TASKS}}
    label = ('FAIL_FORMAT_COLLAPSE' if not all(format_checks.values()) else 'NEGATIVE_TRANSFER' if not all(transfer_checks.values()) else 'FEASIBLE')
    return {**base, 'outcome': label, 'format_checks': format_checks, 'transfer_checks': transfer_checks, 'item_flips': flips,
            'deltas_descriptive': deltas, 'terminal_is_parent_alias': False}


# ================================================================ exact prospective budget
def budget(lengths, dev_rows=24):
    """lengths: {'sequence': int, 'midline_train': {row: int}, 'midline_gate': {control name: int}, 'midline_eval': {row: int}} (expanded tokens).
    Parent DEV24 is the authenticated V2 EP concise rows (0 calls)."""
    flat = [lengths['sequence'], *lengths['midline_train'].values(), *lengths['midline_gate'].values(), *lengths['midline_eval'].values()]
    require(all(type(v) is int and 0 < v <= 2048 for v in flat),
            'PENDING_PROCESSOR_MEASUREMENT: every midline prompt length must be measured with the pinned processor before scope freeze')
    require(len(lengths['midline_train']) == 2 and sorted(lengths['midline_gate']) == sorted(GATE_NAMES) and len(lengths['midline_eval']) == 2,
            'Exact midline inventory (2 TRAIN, 6 gate controls, 2 EVAL)')
    S, T = lengths['sequence'], MAX_TOKENS
    tf = 8 * (S + T - 1) + sum(4 * (p + T - 1) for p in lengths['midline_train'].values())
    gate, parent_eval, terminal = len(GATE_NAMES), 2, dev_rows + 2
    greedy = gate + parent_eval + terminal
    prefill = sum(lengths['midline_gate'].values()) + 2 * sum(lengths['midline_eval'].values()) + dev_rows * S
    return {'model_loads_max': 2, 'resident_models_max': 1, 'optimizer_updates_max': 1, 'sampled_rollout_responses': 16,
            'greedy_calls_max': greedy, 'greedy_breakdown': {'gate_TRAIN_controls': gate, 'parent_midline_EVAL': parent_eval, 'terminal_DEV24': dev_rows,
                                                              'terminal_midline_EVAL': 2, 'parent_DEV24': 'AUTHENTICATED_V2_EP_ROWS_0_CALLS'},
            'generations_max': 16 + greedy, 'generated_tokens_max': (16 + greedy) * T, 'own_old_TF_positions_max': tf, 'detached_old_forwards': 16,
            'backward_calls_max': 16, 'gradient_forwards_max': 16, 'scorer_forwards_max': 32, 'gradient_TF_positions_max': tf,
            'greedy_prompt_prefill_positions_max': prefill, 'segment_limits_formula': 'multistep_gspo_core_v1.unit_worst_case per unit, 1 update',
            'automatic_retry': False, 'resampling': False, 'prompt_lengths': lengths}


def authenticated_input(path, expected_identity):
    """A publish_once JSON file is canonical+newline, so its FILE sha differs from identity(dict); authenticate the canonical identity."""
    import json
    value = json.loads(Path(path).read_text())
    require(identity(value) == expected_identity, 'Pilot input canonical identity differs from the scope')
    return value
