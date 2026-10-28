"""Final accuracy sprint: ONE class-balanced patient-paired sequence SFT candidate. SOURCE_IMPLEMENTED_CPU_ONLY_NOT_NATIVE_RUN.

Supervised comparison (track B), NOT Vero GSPO and NOT the old auxiliary v1/v2/v3 objectives. Importing this module imports no
MLX and loads nothing; the native backend class is built lazily and its factory fails closed without a genuine root-issued
authorization (no entry module exists yet; this file never self-authorizes and is not a launcher).

Frozen candidate (author: Claude Code, 2026-10-27):
  * Parent: preserved P_calibrated13 / Adam52 / boundary114 adapter + full Adam + RNG via the existing `restore_parent`
    (`accuracy_stage_native_v1`) on the existing 12 GiB-capped `MicroNativeBackend`. Same 4-bit base, same LoRA (504 trainable).
  * Data: only the existing 28 TRAIN people at z045, BOTH acquisition modalities (FLAIR + contrast-enhanced T1) per update.
    No new patients, no DEV/holdout16/final100 people, nothing selected by model outputs. Deterministic site-interleaved salted
    hash order per seed (seed changes ONLY the patient order; the loss itself is deterministic).
  * Prompt: the UNCHANGED concise evaluation prompt (Pilot V2 `condition_row(row, 'concise')`, 233 expanded tokens).
  * Target: `<think>Visible MRI evidence.</think><answer>\\boxed{<ground truth>}</answer>` + EOS. The think text is the existing
    frozen generic candidate prefix (`answer_likelihood.PREFIX`): non-empty, no invented radiological observation. The target is
    checked by the strict Vero parser (format 1, single box, correct class).
  * Objective: per image, mean token cross-entropy over the RESPONSE positions only (prompt/image positions are never targets),
    computed from the existing F32 full-teacher-forced scorer (`f32_response_logps_observed` via `_context_logps`) under
    `nn.value_and_grad`. Pair loss = 0.5*CE_FLAIR + 0.5*CE_T1: the two image gradients are computed SERIALLY (one image graph
    resident), scaled 0.5 in float32, summed, then ONE `apply_update` (existing Adam path). One pass: <= 28 updates, <= 56
    image backwards per run; no retry, no extra epoch, no resampling.
  * Optimizer: the restored Adam52 state is CONTINUED (lr 1e-5, betas .9/.999, eps 1e-8, no bias correction - the inherited
    historical optimizer). Inherited GSPO first moment contributes .9**k of m at update k (0.9 at k=1, ~0.05 at k=28); this is a
    disclosed causal confound, not hidden (see research/FINAL_BALANCED_SFT_DESIGN_V1.md).
  * Durability: after EVERY completed pair update the adapter + full Adam + RNG + seed/order cursor are committed atomically
    (pending dir -> rename), THEN a hash-chained receipt is published. A failure is never rolled back in memory (the documented
    MLX optimizer container-alias hazard makes a post-update in-memory snapshot an unsafe rollback target); the last committed
    checkpoint on disk is the rollback/resume point. Counters are a spent ledger and are never refunded.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
from mri_rl import shared_rollout_gspo_core_v1 as shared
from mri_rl.answer_likelihood import PREFIX, SUFFIX, LABELS as LIKELIHOOD_LABELS
from mri_rl.paired_z045_data_v1 import LABELS, SEQ, canonical_sha

STATUS = 'SOURCE_IMPLEMENTED_CPU_ONLY_NOT_NATIVE_RUN'
PROTOCOL = 'FINAL_BALANCED_PATIENT_PAIRED_SEQUENCE_SFT_V1'
require = shared.require
NATIVE_KIND, SYNTHETIC_KIND = 'NATIVE_MLX', 'CPU_SYNTHETIC_FIXTURE'
PEOPLE, PAIR_KEYS = 28, ('F', 'T')
KEY_LABEL = {'F': LABELS[0], 'T': LABELS[1]}
MODALITY_KEY = {'flair': 'F', 't1ce': 'T'}
PAIR_WEIGHT = .5                                   # equal per-image weight: mean of the two per-image mean-CE losses
MAX_UPDATES, MAX_IMAGE_BACKWARDS = 28, 56          # one pass over 28 patient pairs
ALLOWED_SEEDS = (0, 1, 2)                          # order seeds; at most 3 seeds and only for one winner (root-enforced)
PARENT_ADAM_STEP, PARENT_COMPLETED_GROUPS = 52, 114
CONCISE_SEQUENCE_PROMPT_LENGTH = 233               # 213 original + 20 tokens of the frozen concise instruction (V2 report)
ORDER_SALT = 'final-balanced-sft-v1'
OPTIMIZER = {'name': 'Adam', 'source': 'CONTINUED_RESTORED_ADAM52_FULL_STATE', 'learning_rate': 1e-5, 'betas': [.9, .999], 'eps': 1e-8,
             'bias_correction': False, 'reset_moments': False, 'scheduler': 'NONE_CONSTANT'}
EXCLUSION_KINDS = ('dev', 'holdout16', 'final100')
REQUIRED_BACKEND = ('evidence_kind', 'policy_sha', 'adam_digest', 'rng_digest', 'optimizer_step', 'begin_run', 'bind_pair', 'image_gradient',
                    'scale_gradient', 'add_scaled', 'gradient_l2', 'gradient_dtypes', 'release_graphs', 'apply_update', 'save_sft_checkpoint')
ENTRY_MODULE = 'mri_rl.final_balanced_sft_entry_v1'


class PendingAuthorization(RuntimeError):
    """Fail-closed: no genuine root-issued authorization exists for native SFT. Never self-authorize."""


# ================================================================ data: leakage/split checks and deterministic order
def _source_identity(row):
    """(patient, modality) from the existing paired_z045_data_v1 source block; refuses anything not an exact TRAIN z045 row."""
    info, block = row['extra_info'], row.get('paired_z045_data_v1')
    require(row.get('ability') == SEQ and row.get('data_source') == 'OpenNeuro_ds007045', 'Typed OpenNeuro ds007045 sequence row required')
    require(info.get('internal_split') == 'train' and info.get('patient_lineage') == 'BIDS_SUBJECT_VERIFIED', 'Only verified-lineage TRAIN rows are allowed')
    require(isinstance(block, dict) and block.get('level') == 45, 'Existing z045 source-qualified pool row required')
    source = block.get('source_record_id_PRIVATE')
    match = re.fullmatch(re.escape(info['patient_key']) + r'_z045_(flair|t1ce)_sequence', source or '')
    require(match is not None and block.get('modality_PRIVATE') == match[1], 'Source patient/z045/modality identity differs')
    require(row['reward_model']['ground_truth'] == KEY_LABEL[MODALITY_KEY[match[1]]], 'Acquisition-modality label mismatch')
    require(isinstance(row.get('images'), list) and len(row['images']) == 1, 'Exactly one image per row required')
    return info['patient_key'], MODALITY_KEY[match[1]]


def _exclusions(excluded):
    require(isinstance(excluded, dict) and set(excluded) == set(EXCLUSION_KINDS), 'Explicit dev/holdout16/final100 exclusion sets required')
    require(all(isinstance(v, (set, frozenset)) and all(isinstance(k, str) and k for k in v) for v in excluded.values()), 'Exclusion sets of patient keys required')
    return {k: frozenset(v) for k, v in excluded.items()}


def patient_order(people, sites, seed):
    """Stratum-interleaved hash order. A single NON_SITE stratum uses no site information.

    The legacy argument name is retained for CPU/API compatibility. Callers must
    not present a non-site stratum as an authenticated acquisition-site join.
    """
    require(type(seed) is int and seed in ALLOWED_SEEDS, 'Order seed outside the frozen seed set')
    require(set(people) == set(sites), 'Authenticated TRAIN site join required')
    rank = lambda person: hashlib.sha256(f'{ORDER_SALT}:seed{seed}:{person}'.encode()).hexdigest()
    by_site = {}
    for person in people:
        by_site.setdefault(sites[person], []).append(person)
    for site in by_site:
        by_site[site].sort(key=rank)
    ordered = []
    for index in range(max(map(len, by_site.values()))):
        for site in sorted(by_site):
            if index < len(by_site[site]):
                ordered.append(by_site[site][index])
    return ordered


def build_plan(sequence_rows, sites, *, seed, excluded_patient_keys):
    """Exactly 28 TRAIN people x {FLAIR, T1} at z045 under the concise prompt. Pure CPU; no file or model access."""
    from mri_rl.paired_z045_exposure_v2 import patient_order as existing_pair_check
    from mri_rl.accuracy_pilot_v2_core import condition_row, CONCISE_INSTRUCTION_SHA256
    excluded = _exclusions(excluded_patient_keys)
    existing_pair_check(sequence_rows, sites)                     # existing check: 56 unique TRAIN rows, 28 complete pairs, site join
    groups = {}
    for row in sequence_rows:
        person, key = _source_identity(row)
        require(key not in groups.setdefault(person, {}), 'Duplicate pair member')
        groups[person][key] = row
    require(len(groups) == PEOPLE and all(set(g) == set(PAIR_KEYS) for g in groups.values()), 'Exactly 28 complete z045 FLAIR/T1 pairs required')
    leaked = {kind: sorted(set(groups) & keys) for kind, keys in excluded.items()}
    require(not any(leaked.values()), 'Excluded DEV/holdout16/final100 patient present in TRAIN plan')
    require(len({r['id'] for r in sequence_rows}) == 2 * PEOPLE, 'Duplicate row identity')
    require(len({r['extra_info']['question'] for r in sequence_rows}) == 1, 'All 56 rows must share one unchanged original question')
    order = patient_order(sorted(groups), sites, seed)
    pairs = [{'index': i, 'patient': p, 'rows': {k: condition_row(groups[p][k], 'concise') for k in PAIR_KEYS}} for i, p in enumerate(order)]
    identity = {'protocol': PROTOCOL, 'seed': seed, 'order_salt': ORDER_SALT, 'concise_instruction_sha256': CONCISE_INSTRUCTION_SHA256,
                'pairs': [{'index': x['index'], 'patient': x['patient'], **{k: x['rows'][k]['id'] for k in PAIR_KEYS}} for x in pairs]}
    return {'schema': 'final-balanced-sft-plan-v1', 'seed': seed, 'pairs': pairs, 'plan_sha256': canonical_sha(identity),
            'sites_covered': len(set(sites.values())), 'exclusion_set_sizes': {k: len(v) for k, v in excluded.items()},
            'selection': 'Existing 28 TRAIN people only; deterministic stratum/hash order; single NON_SITE stratum in this sprint; no outputs used'}


def plan_summary_safe(plan):
    """Aggregate-only description that carries no patient keys, row IDs or labels per person."""
    return {'pairs': len(plan['pairs']), 'images': 2 * len(plan['pairs']), 'class_images': {LABELS[0]: len(plan['pairs']), LABELS[1]: len(plan['pairs'])},
            'seed': plan['seed'], 'sites_covered': plan['sites_covered'], 'exclusion_set_sizes': plan['exclusion_set_sizes']}


# ================================================================ targets / objective
def target_text(label):
    require(label in LABELS, 'Canonical label required')
    return PREFIX + label + SUFFIX


def build_targets(encode, eos_id):
    """Token targets for both classes. `encode(text) -> list[int]` without special tokens; `eos_id` is the chat end token."""
    from mri_rl.paired_z045_evaluation_v1 import strict_prediction
    from mri_rl.vero_mri_reward import _answer_and_format
    require(tuple(LIKELIHOOD_LABELS) == tuple(LABELS), 'Frozen label order differs')
    require(type(eos_id) is int and eos_id >= 0, 'Concrete EOS token required')
    result = {}
    for index, label in enumerate(LABELS):
        text = target_text(label)
        require(_answer_and_format(text) == (label, 1.) and strict_prediction(text, SEQ) == index, 'Target fails the strict Vero parser')
        prefix, answer, suffix, full = (list(encode(s)) for s in (PREFIX, label, SUFFIX, text))
        require(answer and full == prefix + answer + suffix, 'Tokenizer boundary crosses fixed prefix/label/suffix')
        require(eos_id not in full, 'EOS inside target body')
        result[label] = {'text': text, 'target_ids': full + [eos_id], 'label_positions': list(range(len(prefix), len(prefix) + len(answer))),
                         'response_tokens': len(full) + 1}
    require(result[LABELS[0]]['target_ids'][len(prefix)] != result[LABELS[1]]['target_ids'][len(prefix)], 'First label token must discriminate the classes')
    return {'labels': result, 'targets_sha256': canonical_sha({k: v['target_ids'] for k, v in result.items()}), 'eos_id': eos_id,
            'masking': 'response positions only (think/answer tags, label, closing tags, EOS); prompt and image positions are never targets'}


def mean_response_ce(logps, n_targets):
    """Per-image objective: mean negative log-likelihood of the response tokens. Refuses positive/nonfinite log-probabilities."""
    values = [float(x) for x in logps]
    require(len(values) == n_targets > 0 and all(math.isfinite(x) and x <= 1e-6 for x in values), 'Complete finite response logps required')
    return -sum(values) / n_targets


def reference_masked_ce(full_logits, ids, prompt_length):
    """CPU reference over a full sequence: stable log-softmax, mask = positions predicting response tokens only.

    `full_logits[t]` predicts `ids[t+1]`. Equivalent to the scorer's slice logits[P-1 : P-1+len(response)].
    """
    require(0 < prompt_length < len(ids) and len(full_logits) == len(ids) - 1, 'Full causal logits/ids shape required')
    total, count = 0., 0
    for t, row in enumerate(full_logits):
        if t + 1 < prompt_length:
            continue                                   # masked: prompt/image positions
        top = max(row)
        log_z = top + math.log(sum(math.exp(v - top) for v in row))
        total += log_z - row[ids[t + 1]]
        count += 1
    return total / count


def run_budget(targets, prompt_length=CONCISE_SEQUENCE_PROMPT_LENGTH):
    """Exact finite per-run ledger: 28 updates, 56 own-context scorer forwards+backwards, no generation."""
    require(type(prompt_length) is int and 0 < prompt_length <= 2048, 'Measured expanded prompt length required')
    positions = PEOPLE * sum(prompt_length + v['response_tokens'] - 1 for v in targets['labels'].values())
    limits = {k: 0 for k in shared.CHARGEABLE}
    limits.update(gradient_forwards=MAX_IMAGE_BACKWARDS, backward_calls=MAX_IMAGE_BACKWARDS, scorer_vision_embedding_calls=MAX_IMAGE_BACKWARDS,
                  scorer_language_model_calls=MAX_IMAGE_BACKWARDS, teacherforced_input_positions=positions, optimizer_updates=MAX_UPDATES)
    return limits


def run_binding(*, plan, targets, parent_receipt_sha256, scope_sha256, prompt_length=CONCISE_SEQUENCE_PROMPT_LENGTH):
    require(shared.sha(parent_receipt_sha256) and shared.sha(scope_sha256), 'Exact parent receipt and scope digests required')
    return {'protocol': PROTOCOL, 'seed': plan['seed'], 'plan_sha256': plan['plan_sha256'], 'targets_sha256': targets['targets_sha256'],
            'parent_receipt_sha256': parent_receipt_sha256, 'scope_sha256': scope_sha256, 'prompt_condition': 'concise',
            'prompt_lengths': {SEQ: prompt_length}, 'optimizer': OPTIMIZER, 'pair_weight': PAIR_WEIGHT, 'max_updates': MAX_UPDATES}


# ================================================================ checkpoint files (shared by native and fixture backends)
def save_checkpoint_files(directory, *, save_adapter, optimizer_state, rng_state, trainable_signature, run, completed_pairs, save_arrays):
    """Same file layout / binding schema as StageNativeBackend.save_checkpoint, so its load_checkpoint restores it.

    completed_groups = 114 + completed pairs (one pair = one completed supervised group); optimizer_updates = Adam step = 52 + pairs.
    """
    from mri_rl.optimizer_checkpoint import save_optimizer_checkpoint
    directory = Path(directory)
    save_adapter(directory / 'adapter.safetensors')
    step = optimizer_state['step']
    step = int(step.item() if hasattr(step, 'item') else step)
    require(step == PARENT_ADAM_STEP + completed_pairs, 'Adam step differs from 52 + completed pair updates')
    binding = {'run': run, 'adapter_sha256': shared.file_sha256(directory / 'adapter.safetensors'), 'trainable_signature': trainable_signature}
    save_optimizer_checkpoint(directory / 'optimizer', optimizer_state=optimizer_state, rng_state=rng_state, bindings=binding,
                              completed_groups=PARENT_COMPLETED_GROUPS + completed_pairs, optimizer_updates=step, save_arrays=save_arrays)
    return ['adapter.safetensors', 'optimizer/manifest.json', 'optimizer/state.safetensors']


def _fsync_file(path):
    with Path(path).open('rb') as handle:
        os.fsync(handle.fileno())


def _state(backend):
    return {'policy': backend.policy_sha(), 'adam': backend.adam_digest(), 'rng': backend.rng_digest()}


def _check_backend(backend):
    missing = [name for name in REQUIRED_BACKEND if not hasattr(backend, name)]
    require(not missing, 'Backend lacks the SFT protocol: ' + ','.join(missing))
    require(backend.evidence_kind in (NATIVE_KIND, SYNTHETIC_KIND), 'Backend must declare its evidence kind')


def _pair_dirs(pairs_root):
    entries = sorted(p.name for p in Path(pairs_root).iterdir()) if Path(pairs_root).exists() else []
    stray = [n for n in entries if not re.fullmatch(r'pair_\d{4}', n)]
    require(not stray, 'INDETERMINATE: uncommitted/pending entries present; human review required, never auto-resumed')
    return entries


def committed_cursor(output, *, binding_sha256):
    """Validate the committed hash chain; return the seed/order cursor for resume. Refuses any indeterminate state."""
    pairs_root = Path(output) / 'pairs'
    previous, receipt = None, None
    entries = _pair_dirs(pairs_root)
    for index, name in enumerate(entries):
        require(name == 'pair_%04d' % index, 'Committed pairs must be contiguous from 0')
        directory = pairs_root / name
        require((directory / 'receipt_PRIVATE.json').is_file(), 'INDETERMINATE: committed checkpoint without receipt; human review required')
        receipt = json.loads((directory / 'receipt_PRIVATE.json').read_text())
        require(receipt['receipt_sha256'] == shared.identity({k: v for k, v in receipt.items() if k != 'receipt_sha256'}), 'Receipt changed')
        require(receipt['previous_receipt_sha256'] == previous and receipt['binding_sha256'] == binding_sha256, 'Receipt chain/binding differs')
        require(receipt['pair_index'] == index and receipt['cursor']['completed_pairs'] == index + 1, 'Cursor differs from committed pairs')
        for file, digest in receipt['files_sha256'].items():
            require(shared.file_sha256(directory / 'checkpoint_PRIVATE' / file) == digest, 'Committed checkpoint payload changed')
        previous = receipt['receipt_sha256']
    completed = len(entries)
    return {'completed_pairs': completed, 'next_pair_index': completed, 'last_receipt': receipt,
            'checkpoint': None if not completed else str(pairs_root / entries[-1] / 'checkpoint_PRIVATE'),
            'expected_adam_step': PARENT_ADAM_STEP + completed}


# ================================================================ backend-neutral one-pass transaction
def _pair(*, backend, pair, targets, counters, before, index, previous_receipt, binding, binding_sha, pairs_root, kind):
    before(f'sft_pair_{index:04d}_start')
    start = _state(backend)
    step0 = backend.optimizer_step()
    require(step0 == PARENT_ADAM_STEP + index, 'Adam step differs from the one-pass cursor')
    backend.bind_pair(pair)
    total, images = None, []
    for key in PAIR_KEYS:
        label = KEY_LABEL[key]
        require(pair['rows'][key]['reward_model']['ground_truth'] == label, 'Pair member label differs from its modality key')
        before(f'sft_pair_{index:04d}_image_{key}_backward')
        gradient, info = backend.image_gradient(key, targets['labels'][label]['target_ids'])
        require(_state(backend) == start, 'Image backward mutated parameters, Adam or RNG')
        loss = mean_response_ce(info['response_logps_PRIVATE'], targets['labels'][label]['response_tokens'])
        require(abs(loss - float(info['loss'])) <= 1e-4 * max(1., abs(loss)), 'Autograd loss differs from mean response CE of its own scorer')
        require(backend.gradient_dtypes(gradient) == {'float32'}, 'Image gradient must be float32')
        positions = targets['labels'][label]['label_positions']
        images.append({'key': key, 'loss': loss, 'gradient_l2': float(backend.gradient_l2(gradient)),
                       'label_logp_mean': sum(info['response_logps_PRIVATE'][p] for p in positions) / len(positions),
                       'first_label_logp': info['response_logps_PRIVATE'][positions[0]], 'response_tokens': len(info['response_logps_PRIVATE'])})
        total = backend.scale_gradient(gradient, PAIR_WEIGHT) if total is None else backend.add_scaled(total, gradient, PAIR_WEIGHT)
        del gradient
        backend.release_graphs()                       # first image graph is gone before the second starts
    norm = float(backend.gradient_l2(total))
    require(math.isfinite(norm) and norm > 0, 'Finite nonzero pair gradient required; no skipped or momentum-only update')
    before(f'sft_pair_{index:04d}_update')
    require(_state(backend) == start, 'State changed before the optimizer step')
    counters.charge('optimizer_updates')
    backend.apply_update(total)
    total = None
    after = _state(backend)
    require(backend.optimizer_step() == step0 + 1 and after['policy'] != start['policy'] and after['adam'] != start['adam'] and after['rng'] == start['rng'],
            'Exactly one Adam step moving weights and moments with RNG unconsumed required')
    backend.release_graphs()
    before(f'sft_pair_{index:04d}_checkpoint')
    cursor = {'seed': binding['seed'], 'plan_sha256': binding['plan_sha256'], 'completed_pairs': index + 1, 'next_pair_index': index + 1,
              'adam_step': step0 + 1}
    pending = Path(tempfile.mkdtemp(prefix='.pending-', dir=pairs_root))
    try:
        (pending / 'checkpoint_PRIVATE').mkdir()
        names = list(backend.save_sft_checkpoint(pending / 'checkpoint_PRIVATE', run=binding, completed_pairs=index + 1))
        shared.publish_once(pending / 'checkpoint_PRIVATE' / 'cursor.json', cursor)
        names.append('cursor.json')
        require(_state(backend) == after, 'Checkpoint save changed state')
        for name in names:
            _fsync_file(pending / 'checkpoint_PRIVATE' / name)
        shared.sync_dir(pending / 'checkpoint_PRIVATE')
        target = pairs_root / ('pair_%04d' % index)
        require(not target.exists(), 'Pair already committed; never overwritten')
        os.rename(pending, target)                     # atomic: adapter + full Adam + RNG + cursor appear together
        shared.sync_dir(pairs_root)
    finally:
        if pending.exists():
            shutil.rmtree(pending)
    files = {name: shared.file_sha256(target / 'checkpoint_PRIVATE' / name) for name in names}
    receipt = {'protocol': PROTOCOL, 'evidence_kind': kind, 'pair_index': index, 'patient_PRIVATE': pair['patient'],
               'row_ids_PRIVATE': {k: pair['rows'][k]['id'] for k in PAIR_KEYS}, 'images': images, 'pair_loss': PAIR_WEIGHT * sum(i['loss'] for i in images),
               'joint_gradient_l2': norm, 'state_before': start, 'state_after': after, 'adam_step_after': step0 + 1, 'cursor': cursor,
               'files_sha256': files, 'previous_receipt_sha256': previous_receipt, 'binding_sha256': binding_sha, 'counters': counters.snapshot()}
    receipt['receipt_sha256'] = shared.identity(receipt)
    shared.publish_once(target / 'receipt_PRIVATE.json', receipt)
    before(f'sft_pair_{index:04d}_committed')
    return receipt


def run_sft(*, backend, plan, targets, binding, output, before, expected_parent, prompt_length=CONCISE_SEQUENCE_PROMPT_LENGTH,
            resume=False, authority=None, max_pairs=MAX_UPDATES):
    """One pass over the frozen 28-pair order. `before(name)` is the caller's authorization + dynamic resource guard.

    Fresh: backend must be exactly at the trusted Adam52 parent and `output` must not exist. Resume: the launcher must have
    loaded the last committed checkpoint (`committed_cursor`) into a fresh process; state must equal that receipt's after-state.
    A native backend additionally requires the authority object returned by `check_sft_authority`.
    """
    _check_backend(backend)
    require(callable(before), 'Explicit authorization/resource guard required')
    if backend.evidence_kind == NATIVE_KIND:
        require(isinstance(authority, _CheckedAuthority), 'Native SFT requires the checked root authorization')
    require(type(max_pairs) is int and 1 <= max_pairs <= MAX_UPDATES, 'Pair cap must be within the one-pass 28-update ceiling')
    require(len(plan['pairs']) == PEOPLE and [p['index'] for p in plan['pairs']] == list(range(PEOPLE)), 'Frozen 28-pair plan required')
    require(binding['plan_sha256'] == plan['plan_sha256'] and binding['targets_sha256'] == targets['targets_sha256'] and
            binding['seed'] == plan['seed'] and binding['prompt_lengths'] == {SEQ: prompt_length}, 'Run binding differs from plan/targets/prompt')
    binding_sha = shared.identity(binding)
    output = Path(output)
    limits = run_budget(targets, prompt_length)
    if not resume:
        require(isinstance(expected_parent, dict) and set(expected_parent) == {'policy', 'adam', 'rng'}, 'Trusted parent identity required')
        require(_state(backend) == expected_parent and backend.optimizer_step() == PARENT_ADAM_STEP, 'Backend is not at the trusted Adam52 parent')
        output.mkdir(parents=True, exist_ok=False)     # fresh output only
        (output / 'pairs').mkdir()
        shared.publish_once(output / 'start_PRIVATE.json', {'protocol': PROTOCOL, 'binding': binding, 'binding_sha256': binding_sha, 'limits': limits,
                                                            'parent_state': dict(expected_parent), 'evidence_kind': backend.evidence_kind,
                                                            'automatic_retry': False})
        cursor, used = committed_cursor(output, binding_sha256=binding_sha), None
    else:
        start = json.loads((output / 'start_PRIVATE.json').read_text())
        require(start['binding_sha256'] == binding_sha and start['limits'] == limits, 'Resume binding/limits differ from the started run')
        cursor = committed_cursor(output, binding_sha256=binding_sha)
        require(cursor['completed_pairs'] >= 1, 'Nothing committed: a fresh run, not a resume, is required')
        last = cursor['last_receipt']
        require(_state(backend) == last['state_after'] and backend.optimizer_step() == cursor['expected_adam_step'],
                'Loaded state differs from the last committed checkpoint')
        used = json.loads((output / 'counters_PRIVATE.json').read_text())['used']    # spent ledger, never refunded
        remaining = max_pairs - cursor['completed_pairs']
        require(used['backward_calls'] + 2 * remaining <= limits['backward_calls'] and used['optimizer_updates'] + remaining <= limits['optimizer_updates'],
                'Remaining frozen one-pass budget cannot complete the pass (work spent by an uncommitted pair is not refunded); root decision required')
        shared.publish_once(output / f'resume_{cursor["completed_pairs"]:04d}_{time.time_ns()}_PRIVATE.json',
                            {'cursor': {k: v for k, v in cursor.items() if k != 'last_receipt'}, 'counters_used': used})
    counters = shared.Counters(limits, used, lambda value: shared.replace_atomic(output / 'counters_PRIVATE.json', {'limits': limits, 'used': value}))
    if used is None:
        shared.replace_atomic(output / 'counters_PRIVATE.json', {'limits': limits, 'used': counters.snapshot()})
    backend.counters = counters
    previous = None if cursor['last_receipt'] is None else cursor['last_receipt']['receipt_sha256']
    receipts = []
    try:
        before('sft_run_start')
        backend.begin_run()
        for index in range(cursor['next_pair_index'], max_pairs):
            receipt = _pair(backend=backend, pair=plan['pairs'][index], targets=targets, counters=counters, before=before, index=index,
                            previous_receipt=previous, binding=binding, binding_sha=binding_sha, pairs_root=output / 'pairs', kind=backend.evidence_kind)
            previous = receipt['receipt_sha256']
            receipts.append(receipt)
        final = committed_cursor(output, binding_sha256=binding_sha)
        require(final['completed_pairs'] == max_pairs, 'Committed pair count differs from the pass')
        summary = {'status': 'ACTUAL_SFT_PASS_COMPLETE_PENDING_INDEPENDENT_REVIEW' if backend.evidence_kind == NATIVE_KIND else 'SYNTHETIC_FIXTURE_COMPLETE_NOT_NATIVE_EVIDENCE',
                   'protocol': PROTOCOL, 'evidence_kind': backend.evidence_kind, 'completed_pairs': final['completed_pairs'],
                   'adam_step_final': backend.optimizer_step(), 'terminal_receipt_sha256': previous, 'binding_sha256': binding_sha,
                   'counters': counters.snapshot(), 'automatic_retry': False, 'evaluation_authorized': False, 'science_claim': False}
        shared.publish_once(output / 'complete_SAFE.json', summary)
        return summary
    except BaseException as error:
        # No in-memory rollback: the last committed checkpoint is the only trusted rollback/resume point.
        shared.replace_atomic(output / f'STOP_{time.time_ns()}_PRIVATE.json', {
            'error_type': type(error).__name__, 'error': str(error), 'committed_pairs_this_process': len(receipts),
            'counters_spent_not_refunded': counters.snapshot(), 'in_memory_rollback': False, 'automatic_retry': False})
        raise


# ================================================================ lazy native bridge (no MLX on import; never executed here)
class _CheckedAuthority:
    def __init__(self, authorization):
        self.authorization = authorization


def _entry_authorization_class():
    import importlib
    import importlib.util
    if importlib.util.find_spec(ENTRY_MODULE) is None:
        raise PendingAuthorization('PENDING_GENUINE_AUTHORIZATION: the root SFT entry module ' + ENTRY_MODULE + ' is absent')
    cls = getattr(importlib.import_module(ENTRY_MODULE), 'Authorization', None)
    if not isinstance(cls, type):
        raise PendingAuthorization('PENDING_GENUINE_AUTHORIZATION: the root SFT entry exposes no Authorization class')
    return cls


def check_sft_authority(authorization, resource_gate):
    """Exact root-issued authorization for this SFT phase plus a callable dynamic resource gate built from it."""
    cls = _entry_authorization_class()
    require(isinstance(authorization, cls), 'Authorization must be an instance issued by the root SFT entry')
    scope = getattr(authorization, 'scope', None)
    require(isinstance(scope, dict) and scope.get('protocol') == PROTOCOL and scope.get('execution_stage') == 'SFT_TRAIN', 'Authorization is not this SFT phase')
    require(callable(resource_gate) and getattr(resource_gate, 'authorization', None) is authorization, 'Resource gate must be built from this exact authorization')
    authorization.check('final_balanced_sft_native_authority')
    return _CheckedAuthority(authorization)


def native_image_gradient(backend, key, target_ids):
    """One own-context F32 teacher-forced forward + backward of the per-image mean response CE. No optimizer here."""
    mx = backend.mx
    context = backend.activate(key)
    backend.model.eval()
    fresh = []

    def loss_fn(model, tokens):
        logps = backend._context_logps(context, tokens, 'gradient_forwards')
        fresh.append(mx.stop_gradient(logps))
        return -mx.mean(logps)

    backend.counters.charge('backward_calls')
    backend.before_unit('final_sft_image_backward')
    loss, gradient = backend.nn.value_and_grad(backend.model, loss_fn)(backend.model, list(target_ids))
    mx.eval(loss, gradient, fresh)
    require(len(fresh) == 1, 'Exactly one own-context scorer per image backward')
    logps = [float(v) for v in fresh[0].tolist()]
    require(len(logps) == len(target_ids), 'Scorer response alignment differs')
    norm = backend._checked_gradient(gradient)
    del fresh
    return gradient, {'loss': float(loss.item()), 'response_logps_PRIVATE': logps, 'gradient_l2': norm}


_NATIVE_CLASS = None


def native_backend_class():
    """Build (once) the MicroNativeBackend subclass: unchanged context AST with only the concise prompt-length ledger swapped."""
    global _NATIVE_CLASS
    if _NATIVE_CLASS is not None:
        return _NATIVE_CLASS
    import ast
    import types
    from mri_rl.safe_micro_native_v1 import MicroNativeBackend
    from mri_rl.accuracy_pilot_v2_native import _context_node, exact_prompt_cost

    class BalancedSFTNativeBackend(MicroNativeBackend):
        """Existing capped loader/Adam/restore/F32 scorer; adds only the SFT protocol methods."""
        def configure_sft(self, *, prompt_lengths, run):
            module, node = _context_node()
            namespace = dict(vars(module))
            namespace['require_prompt_cost'] = exact_prompt_cost(prompt_lengths)
            exec(compile(ast.Module(body=[node], type_ignores=[]), '<unchanged-_prepare_context-concise-length>', 'exec'), namespace)
            self._sft_prepare = types.MethodType(namespace['_prepare_context'], self)
            self.stage_binding = run
            self.sft_context_ast_sha256 = hashlib.sha256(ast.dump(node).encode()).hexdigest()

        def _prepare_context(self, modality):
            require(getattr(self, '_sft_prepare', None) is not None, 'configure_sft must bind the concise prompt ledger first')
            return self._sft_prepare(modality)

        def begin_run(self):
            # Sets _unit_snapshot, consumed only by the inherited apply_update frozen-base validation. Never a restore target.
            self.snapshot()

        def bind_pair(self, pair):
            self.bind_unit({'index': pair['index'], 'contexts': [{'key': k, 'row': pair['rows'][k]} for k in PAIR_KEYS]})

        def image_gradient(self, key, target_ids):
            return native_image_gradient(self, key, target_ids)

        def save_sft_checkpoint(self, directory, *, run, completed_pairs):
            require(run == self.stage_binding, 'Checkpoint run binding differs from the configured SFT run')
            return save_checkpoint_files(directory, save_adapter=self.save_adapter, optimizer_state=self.optimizer_state(), rng_state=self.rng_state(),
                                         trainable_signature=self.trainable_signature(), run=run, completed_pairs=completed_pairs, save_arrays=self.save_arrays)

        def load_sft_checkpoint(self, directory, *, run, completed_pairs):
            self.stage_binding = run
            self.load_checkpoint(directory)            # inherited: binding + adapter hash + full Adam/RNG restore
            require(self.optimizer_step() == PARENT_ADAM_STEP + completed_pairs, 'Loaded Adam step differs from the cursor')

        def close(self):
            self._sft_prepare = None
            super().close()

    _NATIVE_CLASS = BalancedSFTNativeBackend
    return _NATIVE_CLASS


def make_native_backend(*, authorization, resource_gate, adapter, config, model_path, contract, cache, prompt_lengths, run):
    """Construct the single resident native backend. Refuses before any import/load without the genuine root authorization."""
    authority = check_sft_authority(authorization, resource_gate)
    require(isinstance(cache, dict) and set(cache) == {'dir'}, 'Exact certified cache directory required')
    require(prompt_lengths == {SEQ: CONCISE_SEQUENCE_PROMPT_LENGTH}, 'Frozen concise sequence prompt length required')
    backend = native_backend_class()(adapter, config, model_path=model_path, contract=contract, seed=13, before_unit=resource_gate, cache=cache)
    backend.configure_sft(prompt_lengths=prompt_lengths, run=run)
    return backend, authority


def prepare_native_parent(backend, *, expected_images, prompt_lengths, work, run, parent_dir, anchor):
    """Existing stage configuration + Adam52/504 restore_parent; returns the trusted parent identity and the native EOS id."""
    from mri_rl.accuracy_stage_native_v1 import restore_parent
    from mri_rl.paired_z045_f32_backend_v1 import reset_effective_eos
    backend.configure_stage(grader=None, expected_images=expected_images, prompt_lengths=prompt_lengths, work=work, binding=run)
    restore_parent(backend, parent_dir, anchor)
    tokenizer = backend.processor.tokenizer
    eos = tokenizer.encode('<|im_end|>', add_special_tokens=False)
    require(len(eos) == 1 and eos[0] in reset_effective_eos(tokenizer, backend.model.config.eos_token_id), 'Chat end token must be one effective EOS id')
    return {'policy': backend.policy_sha(), 'adam': backend.adam_digest(), 'rng': backend.rng_digest()}, eos[0]
