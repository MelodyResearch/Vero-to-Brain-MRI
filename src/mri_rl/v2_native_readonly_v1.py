"""V2_NATIVE_READONLY_V1: exact READ-ONLY native diagnosis of the Accuracy Pilot V2 regression at the saved A53/B53 endpoints.

Author: Claude Code (claude-opus-5-5), 2026-10-21. STATUS: IMPLEMENTED_CPU_TESTED_PENDING_INDEPENDENT_REVIEW_NOT_NATIVE_RUN.

This module is the CPU core: preparation (authenticated hash binding of the immutable V2 run, the frozen DEV selection
and the exact teacher-forced job list), finite counters, answer-span arithmetic on SAVED response token ids,
the descriptive fixed-plan GSPO surrogate, and the CPU analysis of committed native outputs. Importing it imports no
MLX/torch/JAX and loads no model; `prepare` runs under a temporary import blocker.

Experiment A (per checkpoint, A53 then B53, one fresh process each): the 32 Common TRAIN responses of V2 segment 0 are
teacher-forced under the loaded checkpoint with the SAME F32 scorer/image/prompt/token contract (32+32 = 64 forwards).
Their full selected-token log-probabilities are compared with the saved 52 own-old scores (the GSPO denominators).
New scores are EXPECTED to differ: the training old==current parity gates (1e-5 / .002) are NOT applied here.
Experiment B: 4 strict greedy DEV generations per checkpoint (8 total, <=2048 returned tokens), on the 2 previously
regressed cases plus their same-patient same-level opposite-modality partners, selected by a frozen rule from the
EXISTING Parent / A54 / B54 outputs only. Diagnostic only; not Adam53 DEV24 accuracy, not generalization.
0 backward, 0 optimizer updates, 0 new TRAIN generation, no final100 / new16.
"""
from __future__ import annotations
import contextlib
import json
import math
import sys
from pathlib import Path
from mri_rl import accuracy_pilot_v2_core as core
from mri_rl import shared_rollout_gspo_core_v1 as shared
from mri_rl import safe_micro_dynamic_resources_v1 as resources
from mri_rl import v2_regression_diagnosis_v1 as diag
from mri_rl.paired_prompt_groups_v1 import identity, CLIP_LOW, CLIP_HIGH, LOG_RATIO_MAX
from mri_rl.paired_z045_evaluation_v1 import LABELS as TASK_LABELS, strict_prediction
from mri_rl.safe_micro_watchdog_v2 import POLICY as GUARDIAN_POLICY
from mri_rl.vero_mri_reward import _answer_and_format, _boxed_contents, _FORMAT, _TEXT_WRAPPER

require = shared.require
digest = identity
STATUS = 'IMPLEMENTED_CPU_TESTED_PENDING_INDEPENDENT_REVIEW_NOT_NATIVE_RUN'
SCHEMA = 'v2-native-readonly-scope-v1'
KIND = 'FRESH_EXACT_V2_NATIVE_READONLY_A53_B53_SCOPE'
ACTION = 'v2_native_readonly_A53_B53_common32_TF_and_DEV4_greedy_once'
IMPLEMENTED_BY = 'Claude Code (claude-opus-5-5)'
CHECKPOINTS = ('A53', 'B53')
ARM_OF = {'A53': 'A', 'B53': 'B'}
PLANS = ('A', 'B')
CONDITION = 'concise'
TF_PER_CHECKPOINT, DEV_PER_CHECKPOINT = 32, 4
SEQ_TASK, DALLAS_TASK = 'mri_sequence', 'mri_visit_correspondence'
HEAVY_MODULES = diag.FORBIDDEN_MODULES
EXPECTED = {'v2_results_sha256': '386c90a30bda025f27f95b1aee65a78ca9aefbbce58f2d0186fedd7bd0760af0',
            'diagnosis_module_sha256': '11b0ce6292ce57e023f8315152ea89c820fb29f0641529407d3939f11b5d6a8f',
            'diagnosis_aggregate_sha256': 'fcd3028fb0226a21db7cb54ef20bec50a00047b4dc416781a68e5e58edef8bfa',
            'v2_scope_file_sha256': 'afb0d7ef4af2bc581ffce761faba8975a00f32d6ff2ba9c2f4975c838c115482'}
# Same per-greedy-call ceilings as the executed V2 DEV24 counter (accuracy_pilot_v2_native.PER_CALL).
GREEDY_LIMITS = {'generations': DEV_PER_CHECKPOINT, 'generated_tokens': DEV_PER_CHECKPOINT * core.MAX_TOKENS, 'LM': DEV_PER_CHECKPOINT * 260,
                 'vision': DEV_PER_CHECKPOINT, 'sampler': DEV_PER_CHECKPOINT * 257}
TOTALS = {'teacher_forced_scorer_forwards': 2 * TF_PER_CHECKPOINT, 'greedy_generations': 2 * DEV_PER_CHECKPOINT, 'returned_tokens_max': 2048, 'LM_max': 2080,
          'vision_max': 8, 'sampler_max': 2056, 'backward_calls': 0, 'optimizer_updates': 0, 'new_train_generation': 0, 'model_loads_max': 2, 'recoveries': 0,
          'final100_or_new16_outputs': 0}
# PLANNING ESTIMATE from the historical V1/V2 rates recorded in accuracy_pilot_v2_entry.COST_MODEL (load+release 90 s, 5 s per F32
# full TF forward, .25 s per generated token, +3 s per greedy call). It is NOT calibrated for this run and NOT a worst-case or
# wall-time guarantee. The HARD limits are RUN_LIMITS' operation-start windows (waits included, never reset).
COST_MODEL = {'load_release_seconds': 90., 'seconds_per_TF_forward': 5., 'seconds_per_generated_token': .25, 'seconds_per_call_overhead': 3.,
              'kind': 'PLANNING_ESTIMATE_FROM_HISTORICAL_RATES_NOT_CALIBRATED_NOT_A_WALLTIME_GUARANTEE',
              'source': 'accuracy_pilot_v2_entry.COST_MODEL (observed V1/V2 costs)'}
PLANNING_ESTIMATE_SECONDS_PER_CHECKPOINT = (COST_MODEL['load_release_seconds'] + TF_PER_CHECKPOINT * COST_MODEL['seconds_per_TF_forward'] +
                                            DEV_PER_CHECKPOINT * (core.MAX_TOKENS * COST_MODEL['seconds_per_generated_token'] + COST_MODEL['seconds_per_call_overhead']))
RUN_LIMITS = {'per_checkpoint_window_seconds': 1200, 'total_window_seconds': 2400, 'wait_counts_against_deadlines': True, 'windows_reset': False,
              'model_loads_max': 2, 'one_fresh_process_per_checkpoint': True, 'serial_order': list(CHECKPOINTS), 'automatic_retry': False,
              'automatic_recoveries': 0, 'red_restart': False, 'incomplete_operation': 'INDETERMINATE_NO_REPLAY', 'resume': False,
              'B53_requires': 'A53 child exit 0, reaped, lock released, terminal PASS and verified allocator release',
              'window_semantics': ('Windows gate the START of every operation boundary (resource gate / journal claim). An in-flight F32 forward '
                                   'or greedy generation is an atomic GPU operation that cannot be interrupted safely; the external guardian '
                                   'can still SIGUSR1/TERM/KILL the owned child.'),
              'nonresident_wait_poll_seconds': 5., 'wait_samples_before_launch': 3}
RESOURCE_POLICY = dict(resources.POLICY)
DEV_SELECTION_RULE = ('From the frozen DEV24 and the EXISTING journal-verified V2 greedy rows only: REGRESSED = sorted indices where Parent '
                      '(strict parser) is correct and A54 or B54 is not; require exactly 2. SELECTED = sorted(REGRESSED + their '
                      'same-patient same-level opposite-modality partners from core.dev_pairs); require exactly 4. No other case is added; '
                      'no new output is consulted. Frozen PRIVATE before any native outcome.')


# ================================================================ isolation
def heavy_modules_loaded():
    return sorted(m for m in sys.modules if m.split('.')[0] in HEAVY_MODULES)


class _Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in HEAVY_MODULES:
            raise ImportError('blocked by v2_native_readonly_v1 CPU-only preparation: ' + name)
        return None


@contextlib.contextmanager
def cpu_only_imports():
    """Temporary meta-path blocker: preparation/analysis cannot import MLX/torch/JAX. Removed afterwards (the child imports MLX later)."""
    require(not heavy_modules_loaded(), 'Heavy runtime already imported before CPU-only work')
    blocker = _Blocker()
    sys.meta_path.insert(0, blocker)
    try:
        yield
    finally:
        sys.meta_path.remove(blocker)
    require(not heavy_modules_loaded(), 'Heavy runtime imported during CPU-only work')


def load_json(path):
    return json.loads(Path(path).read_text())


# ================================================================ finite counters
def tf_counter_limits(forwards, positions):
    """Exact shared.Counters inventory for the read-only scorer (the reused _context_logps charges these four keys)."""
    require(type(forwards) is int and type(positions) is int and forwards >= 0 and positions >= 0, 'Exact integer TF budget required')
    limits = {k: 0 for k in shared.CHARGEABLE}
    limits.update(detached_old_forwards=forwards, scorer_vision_embedding_calls=forwards, scorer_language_model_calls=forwards,
                  teacherforced_input_positions=positions)
    return limits


class FiniteCounter:
    """Greedy call counter with the generate_one_v2 interface. Fresh only: never resumed, never refunded."""
    def __init__(self, limits, path):
        require(isinstance(limits, dict) and limits and all(type(v) is int and v >= 0 for v in limits.values()), 'Finite integer limits required')
        self.limits, self.path = dict(limits), Path(path)
        require(not self.path.exists(), 'Counter ledger already exists: no resume/replay')
        self.used = {k: 0 for k in limits}
        shared.publish_once(self.path, {'limits': self.limits, 'used': self.used})

    def charge(self, key, n=1):
        require(key in self.used and type(n) is int and n >= 0 and self.used[key] + n <= self.limits[key], 'Frozen greedy call/token budget exceeded')
        used = {**self.used, key: self.used[key] + n}
        shared.replace_atomic(self.path, {'limits': self.limits, 'used': used})
        self.used = used


def tf_job_positions(jobs, lengths):
    """Exact charged TF positions: P + len(response) - 1 per forward, P = the RECORDED rollout prompt length == frozen ledger."""
    total = 0
    for j in jobs:
        require(j['task'] in lengths and type(j['prompt_length']) is int and j['prompt_length'] == lengths[j['task']],
                'Recorded prompt length differs from the frozen V2 prompt-condition ledger')
        require(type(j['tokens']) is int and 0 < j['tokens'] <= core.MAX_TOKENS, 'Saved response length outside 1..256')
        total += j['prompt_length'] + j['tokens'] - 1
    return total


def budget_for(tf_jobs, lengths):
    require(len(tf_jobs) == TF_PER_CHECKPOINT and [j['index'] for j in tf_jobs] == list(range(TF_PER_CHECKPOINT)), 'Exactly the 32 Common TRAIN jobs required')
    positions = tf_job_positions(tf_jobs, lengths)
    return {'tf_jobs_per_checkpoint': TF_PER_CHECKPOINT, 'dev_jobs_per_checkpoint': DEV_PER_CHECKPOINT, 'tf_positions_per_checkpoint': positions,
            'tf_positions_total': 2 * positions, 'tf_limits': tf_counter_limits(TF_PER_CHECKPOINT, positions), 'greedy_limits': dict(GREEDY_LIMITS),
            'planning_estimate_seconds_per_checkpoint': PLANNING_ESTIMATE_SECONDS_PER_CHECKPOINT, 'cost_model': dict(COST_MODEL),
            'note': 'Completed API endpoints, not kernels; one unreturned AR lookahead per greedy response is included in the per-call LM/sampler ceilings.'}


# ================================================================ answer positions on SAVED token ids (never retokenized)
# Measurement only: the historical reward/strict parser (vero_mri_reward._answer_and_format, strict_prediction) is CALLED, never changed.
ANSWER_STATUSES = ('STRICT_CANONICAL', 'STRICT_NONCANONICAL', 'MULTIPLE_ANSWER_BOXES', 'EMPTY_ANSWER_BOX', 'MALFORMED_BOX', 'NO_ANSWER_BOX',
                   'FORMAT_INVALID', 'DECODE_MISMATCH')
MEASUREMENT_KINDS = ('LABEL_ONLY', 'WHOLE_OVERLAPPING_TOKEN_DIAGNOSTIC', 'NONE')
INELIGIBILITY_REASONS = ('EXTRA_BOX_MARKUP_IN_ANSWER', 'PARTIAL_BPE_BOUNDARY')


def _box_spans(text, base=0):
    """Char spans of the stripped (and \\text{}-unwrapped) payload of every balanced \\boxed{...}, scanned exactly like
    vero_mri_reward._boxed_contents (an unclosed box stops the scan). Returns ([(start, end)], stopped_by_unclosed_box)."""
    out, cursor = [], 0
    while (start := text.find('\\boxed', cursor)) >= 0:
        cursor = start + len('\\boxed')
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor >= len(text) or text[cursor] != '{':
            continue
        depth = 1
        begin = cursor = cursor + 1
        while cursor < len(text) and depth:
            depth += (text[cursor] == '{') - (text[cursor] == '}')
            cursor += 1
        if depth:
            return out, True
        s, e = begin, cursor - 1
        for _ in range(2):                                   # strip; then once more inside a \text{} wrapper
            while s < e and text[s].isspace():
                s += 1
            while e > s and text[e - 1].isspace():
                e -= 1
            wrapped = _TEXT_WRAPPER.fullmatch(text, s, e)
            if not wrapped:
                break
            s, e = wrapped.start(1), wrapped.end(1)
        out.append((base + s, base + e))
    require([text[s - base:e - base] for s, e in out] == _boxed_contents(text), 'Box span scan differs from the pinned reward helper')
    return out, False


def answer_span(token_ids, decoder, text, task, *, eos_last):
    """Answer-token MEASUREMENT ELIGIBILITY on the decoded SAVED token ids (no retokenization; no token probability is byte-split).

    The decoded ids (minus a final special EOS token) must equal the saved text up to outer whitespace and give the SAME historical
    parse (`_answer_and_format`), else DECODE_MISMATCH. Then, from the region the parser actually grades:
      STRICT_CANONICAL       strict format, exactly one balanced box INSIDE <answer>, canonical label (boxes in <think> are ignored,
                             exactly as the parser ignores them);
      STRICT_NONCANONICAL    strict format, one answer box, non-label content;
      MULTIPLE_ANSWER_BOXES  >1 balanced box inside <answer> (parser format .5; ambiguous);
      EMPTY_ANSWER_BOX / MALFORMED_BOX (unclosed) / NO_ANSWER_BOX;
      FORMAT_INVALID         tag/think gate fails (parser format 0); the last balanced box anywhere is kept only as a diagnostic.
    label_metric_eligible (measurement_kind LABEL_ONLY) iff STRICT_CANONICAL, the answer holds no other \\boxed markup, and the
    selected tokens start and end EXACTLY on the label bytes (no brace/wrapper/tag/space byte shares a label token). Every other span
    with overlapping tokens is WHOLE_OVERLAPPING_TOKEN_DIAGNOSTIC (whole tokens may carry non-label bytes); otherwise NONE.
    No raw answer text is retained."""
    require(task in TASK_LABELS, 'Known task required')
    pieces = [decoder.piece(t) for t in token_ids]
    spans, cursor = [], 0
    for p in pieces:
        spans.append((cursor, cursor + len(p)))
        cursor += len(p)
    full = b''.join(pieces)
    body_end = spans[-1][0] if eos_last and token_ids and token_ids[-1] in decoder.added else len(full)
    labels = TASK_LABELS[task]
    lower = {x.lower(): x for x in labels}
    saved_answer, saved_format = parsed = _answer_and_format(text)
    out = {'tokens': len(token_ids), 'eos_last': bool(eos_last), 'status': 'DECODE_MISMATCH', 'label_metric_eligible': False,
           'measurement_kind': 'NONE', 'ineligibility_reason': None, 'overlap_positions': [], 'label_positions': [], 'label': None,
           'decision_index': None, 'boundary_tokens_include_non_answer_bytes': False, 'content_canonical': None,
           'parser_format': saved_format, 'parser_answer_canonical': saved_answer.lower() in lower, 'decoded_text_exact': False,
           'answer_boxes': None}
    try:
        decoded = full[:body_end].decode('utf-8')
    except UnicodeDecodeError:
        return out
    if decoded.strip() != text.strip() or _answer_and_format(decoded) != parsed:
        return out
    out['decoded_text_exact'] = decoded == text
    if saved_format == 0.:
        boxes, _ = _box_spans(decoded)
        out['status'] = 'FORMAT_INVALID'
        selected = boxes[-1] if boxes else None
    else:
        match = _FORMAT.fullmatch(decoded)
        require(match is not None, 'Parser format >0 requires the strict tag pattern')
        raw = match.group('answer')
        region = raw.strip()
        base = match.start('answer') + len(raw) - len(raw.lstrip())
        boxes, unclosed = _box_spans(region, base)
        out['answer_boxes'] = len(boxes)
        selected = boxes[-1] if boxes else None
        if not boxes:
            out['status'] = 'MALFORMED_BOX' if '\\boxed' in region else 'NO_ANSWER_BOX'
        elif len(boxes) > 1:
            out['status'] = 'MULTIPLE_ANSWER_BOXES'
        elif selected[0] == selected[1]:
            out['status'], selected = 'EMPTY_ANSWER_BOX', None
        else:
            require(saved_format == 1., 'One nonempty answer box must be strict format under the pinned parser')
            content = decoded[selected[0]:selected[1]]
            if content.lower() not in lower:
                out['status'] = 'STRICT_NONCANONICAL'
            else:
                predicted = strict_prediction(text, task)
                if predicted < 0 or labels[predicted].lower() != content.lower():
                    return {**out, 'status': 'DECODE_MISMATCH'}
                out.update(status='STRICT_CANONICAL', label=labels[predicted])
                if region.count('\\boxed') != 1:
                    out['ineligibility_reason'] = 'EXTRA_BOX_MARKUP_IN_ANSWER'
    if selected is None:
        return out
    s, e = len(decoded[:selected[0]].encode()), len(decoded[:selected[1]].encode())
    positions = [i for i, (a, b) in enumerate(spans) if a < e and b > s]
    require(positions and spans[positions[-1]][1] <= body_end, 'Selected content must lie in the decoded response body')
    partial = spans[positions[0]][0] != s or spans[positions[-1]][1] != e
    out.update(overlap_positions=positions, boundary_tokens_include_non_answer_bytes=partial,
               content_canonical=decoded[selected[0]:selected[1]].lower() in lower, measurement_kind='WHOLE_OVERLAPPING_TOKEN_DIAGNOSTIC')
    if out['status'] == 'STRICT_CANONICAL' and out['ineligibility_reason'] is None:
        if partial:
            out['ineligibility_reason'] = 'PARTIAL_BPE_BOUNDARY'
        else:
            out.update(label_metric_eligible=True, measurement_kind='LABEL_ONLY', label_positions=list(positions), decision_index=positions[0])
    return out


# ================================================================ per-response arithmetic (post-update scores are NOT parity-gated)
def _finite_logps(values):
    return isinstance(values, (list, tuple)) and len(values) > 0 and all(type(v) in (int, float) and math.isfinite(v) and v <= 0 for v in values)


def _span_family(prefix, d, old, new, pos):
    if not pos:
        return {f'delta_{prefix}_sum': None, f'delta_{prefix}_mean': None, f'delta_{prefix}_first': None,
                f'{prefix}_logp_old_sum': None, f'{prefix}_logp_new_sum': None}
    da = [d[p] for p in pos]
    return {f'delta_{prefix}_sum': math.fsum(da), f'delta_{prefix}_mean': math.fsum(da) / len(da), f'delta_{prefix}_first': da[0],
            f'{prefix}_logp_old_sum': math.fsum(old[p] for p in pos), f'{prefix}_logp_new_sum': math.fsum(new[p] for p in pos)}


def response_measures(new, old, *, eos_last, overlap_positions, label_positions=None):
    """Fixed-response log-probability changes under IDENTICAL saved prefixes: full mean/sum/first, EOS, and two SEPARATE span families:
    *_overlap_*  WHOLE_OVERLAPPING_TOKEN diagnostic over every token overlapping the selected content (may carry non-label bytes);
    *_label_*    pure label-only, present ONLY when label_positions is given (the span is label_metric_eligible), else None.
    Full-response logp is NOT a canonical class margin. seq_log_ratio is the GSPO sequence log-ratio vs the 52 own-old."""
    require(_finite_logps(new) and _finite_logps(old) and len(new) == len(old), 'Finite nonpositive equal-length new/own-old vectors required')
    n = len(new)
    pos = list(overlap_positions)
    require(all(type(p) is int and 0 <= p < n for p in pos) and pos == sorted(set(pos)), 'Overlap positions must index the saved response')
    require(label_positions is None or (pos and list(label_positions) == pos), 'Label-only positions must be exactly the nonempty eligible span')
    d = [float(a) - float(b) for a, b in zip(new, old)]
    mean = math.fsum(d) / n
    out = {'tokens': n, 'delta_sum': math.fsum(d), 'delta_mean': mean, 'delta_first': d[0], 'mean_logp_old': math.fsum(old) / n,
           'mean_logp_new': math.fsum(new) / n, 'delta_eos': d[-1] if eos_last else None,
           'delta_mean_excluding_eos': (math.fsum(d[:-1]) / (n - 1) if n > 1 else None) if eos_last else mean,
           'seq_log_ratio': mean, 'raw_ratio': math.exp(min(mean, LOG_RATIO_MAX)), 'log_ratio_capped': mean > LOG_RATIO_MAX,
           'overlap_tokens': len(pos), 'label_metric_eligible': label_positions is not None,
           'label_tokens': len(pos) if label_positions is not None else 0}
    out.update(_span_family('overlap', d, old, new, pos))
    out.update(_span_family('label', d, old, new, pos if label_positions is not None else []))
    return out


def gspo_surrogate(ratio, advantage):
    """Forward value of the frozen native GSPO token loss (identical per token): max(-A r, -A clip(r, 1-.0003, 1+.0004))."""
    require(type(ratio) in (int, float) and math.isfinite(ratio) and ratio > 0 and type(advantage) in (int, float) and math.isfinite(advantage),
            'Finite ratio/advantage required')
    clipped = min(max(ratio, 1 - CLIP_LOW), 1 + CLIP_HIGH)
    return max(-advantage * ratio, -advantage * clipped)


def _plan_term(job, measure, plan):
    """One response's fixed-plan objective terms (old, new, unclipped, outside_clip), or None when not admitted (weight 0)."""
    p = job['plans'][plan]
    if not p['weight']:
        return None
    w, a, r = core.UNIT_WEIGHT * p['weight'], p['advantage'], measure['raw_ratio']
    return w * -a, w * gspo_surrogate(r, a), w * -a * r, not (1 - CLIP_LOW <= r <= 1 + CLIP_HIGH)


def plan_objective(jobs, measures, plan):
    """Joint fixed-plan objective sum_u .25 * sum_i w_i * loss_i on the FIXED saved responses (r=1 at the 52 own-old)."""
    old, new, unclipped, by_task, admitted, outside = [], [], [], {}, 0, 0
    for j in jobs:
        term = _plan_term(j, measures[j['index']], plan)
        if term is None:
            continue
        o, nw, un, out_of_clip = term
        old.append(o); new.append(nw); unclipped.append(un)
        admitted += 1
        outside += out_of_clip
        t = by_task.setdefault(j['task'], {'admitted': 0, 'delta': []})
        t['admitted'] += 1; t['delta'].append(nw - o)
    return {'objective_old': math.fsum(old), 'objective_new': math.fsum(new), 'delta': math.fsum(b - a for a, b in zip(old, new)),
            'delta_unclipped': math.fsum(b - a for a, b in zip(old, unclipped)), 'admitted': admitted, 'outside_clip': outside,
            'by_task': {k: {'admitted': v['admitted'], 'delta': math.fsum(v['delta'])} for k, v in sorted(by_task.items())},
            'label': ('DESCRIPTIVE fixed-plan clipped GSPO surrogate on fixed saved responses (float64 recomputation; own-old = saved 52 scores); '
                      'NOT a counterfactual training result, NOT accuracy evidence')}


def _stats(values):
    v = [x for x in values if x is not None]
    return {'n': len(v), 'mean': math.fsum(v) / len(v) if v else None, 'min': min(v) if v else None, 'max': max(v) if v else None}


def _sign(plan):
    if not plan['weight']:
        return 'NOT_ADMITTED'
    return 'POSITIVE' if plan['advantage'] > 1e-12 else 'NEGATIVE' if plan['advantage'] < -1e-12 else 'ZERO'


def _task(j):
    return 'MRI' if j['task'] == SEQ_TASK else 'DALLAS'


def format_class(j):
    """Recorded historical grade format: 1 strict, .5 tags ok but not exactly one box in <answer>, 0 tag/think gate failed."""
    return {1: 'STRICT_VALID', .5: 'PARTIAL_FORMAT', 0: 'FORMAT_INVALID'}.get(j['format'], 'FORMAT_OTHER')


def raw_outcome(j):
    """Why raw accuracy is 0: never call a format-gated zero a wrong class. Canonicality is that of the historical parser's answer."""
    if j['raw_accuracy'] == 1:
        return 'CORRECT'
    if j['format'] == 0:
        return 'RAW0_FORMAT_GATE'
    return 'RAW0_CANONICAL_WRONG_CLASS' if j['answer']['parser_answer_canonical'] else 'RAW0_NONCANONICAL_OR_MISSING_ANSWER'


STRATA = {'task': _task, 'image': lambda j: j['image_class'], 'format': format_class,
          'raw': lambda j: 'CORRECT' if j['raw_accuracy'] == 1 else 'RAW_ZERO', 'raw_outcome': raw_outcome,
          'advantage_sign_plan_A': lambda j: _sign(j['plans']['A']), 'advantage_sign_plan_B': lambda j: _sign(j['plans']['B']),
          'answer_status': lambda j: j['answer']['status'], 'measurement_kind': lambda j: j['answer']['measurement_kind']}
# Pooled strata carry full-response measures and the GATED label-only family (None for ineligible spans, so never pooled).
# Whole-overlapping-token measures are NEVER pooled here: they appear only per status in answer_measurement.
LABEL_KEYS = ('delta_label_mean', 'delta_label_sum', 'delta_label_first')
OVERLAP_KEYS = ('delta_overlap_mean', 'delta_overlap_sum', 'delta_overlap_first')
MEASURE_KEYS = ('delta_mean', 'delta_sum', 'delta_first', 'delta_eos', 'delta_mean_excluding_eos', *LABEL_KEYS, 'raw_ratio')
LABEL_ONLY_NOTE = ('LABEL_ONLY: strict-canonical single answer box whose selected tokens hold exactly the label bytes; '
                   'ineligible responses excluded (never pooled)')
OVERLAP_NOTE = ('CONTENT_OVERLAP_DIAGNOSTIC (WHOLE_OVERLAPPING_TOKEN): label-ineligible spans only, whole tokens may include '
                'brace/wrapper/tag bytes or non-label/ambiguous/format-invalid content; NOT visual accuracy, NOT a label probability')


def _cell(jobs, measures):
    ms = [measures[j['index']] for j in jobs]
    return {'n': len(ms), 'label_metric_eligible': sum(j['answer']['label_metric_eligible'] for j in jobs),
            **{k: _stats([m[k] for m in ms]) for k in MEASURE_KEYS}}


def stratify(jobs, measures):
    out = {}
    for name, key in STRATA.items():
        groups = {}
        for j in jobs:
            groups.setdefault(key(j), []).append(j)
        out[name] = {g: _cell(js, measures) for g, js in sorted(groups.items())}
    return out


def joint_strata(jobs, measures):
    """task x recorded format class x raw outcome, with the SAME checkpoint's fixed-plan admission/advantage and surrogate terms.
    Cells partition the 32 responses; per plan the surrogate deltas sum to plan_objective(...)['delta']."""
    groups = {}
    for j in jobs:
        groups.setdefault(f'{_task(j)}|{format_class(j)}|{raw_outcome(j)}', []).append(j)
    out = {}
    for g, js in sorted(groups.items()):
        cell = _cell(js, measures)
        cell['measurement_kind_counts'] = _counts(j['answer']['measurement_kind'] for j in js)
        cell['plans'] = {}
        for plan in PLANS:
            terms = [(j, _plan_term(j, measures[j['index']], plan)) for j in js]
            admitted = [(j, t) for j, t in terms if t is not None]
            signs = [_sign(j['plans'][plan]) for j, _ in admitted]
            cell['plans'][plan] = {'admitted': len(admitted), 'advantage_positive': signs.count('POSITIVE'),
                                   'advantage_negative': signs.count('NEGATIVE'), 'advantage_zero': signs.count('ZERO'),
                                   'surrogate_delta': math.fsum(t[1] - t[0] for _, t in admitted)}
        out[g] = cell
    return out


def answer_measurement(jobs, measures):
    """Eligibility inventory plus the two separately named span families (label-only pooled over eligible spans only;
    whole-overlapping-token diagnostics only for ineligible spans and only per status)."""
    eligible = [j for j in jobs if j['answer']['label_metric_eligible']]
    by_status, by_task, overlap = {}, {}, {}
    for j in jobs:
        a = j['answer']
        s = by_status.setdefault(a['status'], {'n': 0, 'label_metric_eligible': 0, 'ineligibility_reasons': {}})
        s['n'] += 1; s['label_metric_eligible'] += a['label_metric_eligible']
        if a['ineligibility_reason']:
            s['ineligibility_reasons'][a['ineligibility_reason']] = s['ineligibility_reasons'].get(a['ineligibility_reason'], 0) + 1
        t = by_task.setdefault(_task(j), {'n': 0, 'label_metric_eligible': 0})
        t['n'] += 1; t['label_metric_eligible'] += a['label_metric_eligible']
        if a['measurement_kind'] == 'WHOLE_OVERLAPPING_TOKEN_DIAGNOSTIC':
            overlap.setdefault(a['status'], []).append(measures[j['index']])

    def label_family(js):
        return {'n': len(js), **{k: _stats([measures[j['index']][k] for j in js]) for k in LABEL_KEYS}, 'label': LABEL_ONLY_NOTE}
    return {'label_metric_eligible': len(eligible), 'label_metric_ineligible': len(jobs) - len(eligible),
            'by_status': dict(sorted(by_status.items())), 'by_task': dict(sorted(by_task.items())),
            'measurement_kind_counts': _counts(j['answer']['measurement_kind'] for j in jobs),
            'decoded_text_exact': sum(j['answer']['decoded_text_exact'] for j in jobs),
            'parser_format_differs_from_recorded_grade': sum(j['answer']['parser_format'] != j['format'] for j in jobs),
            'label_only': label_family(eligible),
            'label_only_by_task': {t: label_family([j for j in eligible if _task(j) == t]) for t in sorted({_task(j) for j in eligible})},
            'whole_overlapping_token_diagnostic_by_status': {s: {'n': len(ms), **{k: _stats([m[k] for m in ms]) for k in OVERLAP_KEYS},
                                                                 'label': OVERLAP_NOTE} for s, ms in sorted(overlap.items())}}


def analyze_train(jobs, new_by_checkpoint):
    """Experiment A: per checkpoint, per response measures; descriptive 2x2 (checkpoint x fixed plan); strata; cross-checkpoint."""
    require(set(new_by_checkpoint) == set(CHECKPOINTS), 'Both A53 and B53 scores required')
    indices = [j['index'] for j in jobs]
    measures = {}
    for ck in CHECKPOINTS:
        require(sorted(new_by_checkpoint[ck]) == sorted(indices) and len(indices) == TF_PER_CHECKPOINT, 'All 32 fixed responses required per checkpoint')
        measures[ck] = {}
        for j in jobs:
            a = j['answer']
            require(a['label_metric_eligible'] == (a['measurement_kind'] == 'LABEL_ONLY') and
                    (a['label_positions'] == a['overlap_positions'] != [] if a['label_metric_eligible'] else a['label_positions'] == []),
                    'Inconsistent answer measurement eligibility')
            measures[ck][j['index']] = response_measures(new_by_checkpoint[ck][j['index']], j['old_logps_PRIVATE'], eos_last=j['eos_last'],
                                                         overlap_positions=a['overlap_positions'],
                                                         label_positions=a['label_positions'] if a['label_metric_eligible'] else None)
    diffs = [measures['A53'][i]['delta_mean'] - measures['B53'][i]['delta_mean'] for i in indices]
    same = sum((measures['A53'][i]['delta_mean'] > 0) == (measures['B53'][i]['delta_mean'] > 0) for i in indices)
    return {'surrogate_2x2': {ck: {plan: plan_objective(jobs, measures[ck], plan) for plan in PLANS} for ck in CHECKPOINTS},
            'strata': {ck: stratify(jobs, measures[ck]) for ck in CHECKPOINTS},
            'joint_strata': {ck: joint_strata(jobs, measures[ck]) for ck in CHECKPOINTS},
            'answer_measurement': {ck: answer_measurement(jobs, measures[ck]) for ck in CHECKPOINTS},
            'cross_checkpoint': {'n': len(indices), 'mean_delta_mean_A53_minus_B53': math.fsum(diffs) / len(diffs),
                                 'mean_abs_delta_mean_difference': math.fsum(abs(x) for x in diffs) / len(diffs), 'same_sign_delta_mean': same},
            'measures_PRIVATE': {ck: {str(i): m for i, m in v.items()} for ck, v in measures.items()},
            'interpretation_limits': ['Fixed responses sampled from the 52 parent; post-update scores are expected to differ (no parity gate).',
                                      'Full-response mean logp and answer-span changes are not canonical class margins.',
                                      'Label-only metrics pool ONLY label_metric_eligible spans; whole-overlapping-token metrics are '
                                      'per-status content-overlap diagnostics, not visual accuracy.',
                                      'Joint strata are descriptive cells of a fixed 32-response sample; small cells (n=1) are single responses.',
                                      'The 2x2 crosses checkpoint with fixed plan descriptively; it is not a counterfactual training result.']}


# ================================================================ Experiment B: frozen DEV selection and analysis
LABELS = core.LABELS


def select_dev_cases(records, rows):
    """Deterministic predeclared rule over EXISTING Parent/A54/B54 rows only (see DEV_SELECTION_RULE)."""
    require(set(rows) == {'Parent', 'A', 'B'} and all(len(rows[m]) == len(records) for m in rows), 'Parent/A54/B54 rows for every DEV record required')
    truth = [LABELS.index(r['row_PRIVATE']['reward_model']['ground_truth']) for r in records]
    preds = {m: [core.strict_label(x['text']) for x in rows[m]] for m in rows}
    regressed = sorted(i for i in range(len(records)) if preds['Parent'][i] == truth[i] and (preds['A'][i] != truth[i] or preds['B'][i] != truth[i]))
    require(len(regressed) == 2, 'Predeclared design requires exactly the 2 observed regressions')
    pairs = core.dev_pairs(records)
    partner = {}
    for f, t in pairs:
        partner[f], partner[t] = t, f
    selected = sorted(set(regressed) | {partner[i] for i in regressed})
    require(len(selected) == 4, 'The two regressions must lie in two distinct pairs (4 images); no substitute case is added')
    chosen = [list(p) for p in pairs if p[0] in selected]
    return {'rule': DEV_SELECTION_RULE, 'regressed_PRIVATE': regressed, 'selected_PRIVATE': selected, 'pairs_PRIVATE': chosen,
            'roles_PRIVATE': {str(i): 'REGRESSED' if i in regressed else 'PARTNER' for i in selected}, 'uses_new_outputs': False,
            'reference_outputs': ['Parent (V2 EP, concise)', 'A54 (V2 EA)', 'B54 (V2 EB)']}


def analyze_dev(dev_jobs, new_rows, decoder=None):
    require(set(new_rows) == set(CHECKPOINTS) and [d['index'] for d in dev_jobs] == list(range(DEV_PER_CHECKPOINT)), 'Exact 4-image DEV inventory required')
    truth = [LABELS.index(d['truth']) for d in dev_jobs]
    refs = {lab: [d['reference_PRIVATE'][lab] for d in dev_jobs] for lab in ('Parent', 'A54', 'B54')}
    ref_pred = {lab: [core.strict_label(r['text']) for r in v] for lab, v in refs.items()}
    pairs = {}
    for i, d in enumerate(dev_jobs):
        pairs.setdefault(d['pair'], {})[d['truth']] = i
    require(all(set(p) == set(LABELS) for p in pairs.values()) and len(pairs) == 2, 'Two complete FLAIR/T1 pairs required')
    out, preds = {}, {}
    for ck in CHECKPOINTS:
        rows = new_rows[ck]
        require(len(rows) == DEV_PER_CHECKPOINT, 'Four new greedy rows per checkpoint required')
        pred = preds[ck] = [core.strict_label(r['text']) for r in rows]
        ref54 = 'A54' if ck == 'A53' else 'B54'
        timing = {'REGRESSION_PRESENT_AT_53': 0, 'REGRESSION_APPEARS_AT_54': 0, 'NOT_A_REGRESSION_IN_THIS_ARM': 0}
        partner, per = {}, []
        for i, d in enumerate(dev_jobs):
            c53, c54, cp = pred[i] == truth[i], ref_pred[ref54][i] == truth[i], ref_pred['Parent'][i] == truth[i]
            if d['role'] == 'REGRESSED':
                timing['NOT_A_REGRESSION_IN_THIS_ARM' if (not cp or c54) else ('REGRESSION_PRESENT_AT_53' if not c53 else 'REGRESSION_APPEARS_AT_54')] += 1
            else:
                key = 'PARENT_%s_53_%s_54_%s' % tuple('CORRECT' if x else 'WRONG' for x in (cp, c53, c54))
                partner[key] = partner.get(key, 0) + 1
            vs_parent = diag.selected_logp_comparison(refs['Parent'][i], rows[i], decoder)
            vs_54 = diag.selected_logp_comparison(refs[ref54][i], rows[i], decoder)
            per.append({'role': d['role'], 'truth': d['truth'], 'pred53': pred[i], 'correct53': c53, 'vs_parent': vs_parent, 'vs_54': vs_54})
        cats = {lab: {} for lab in ('53', 'Parent', '54')}
        for p in pairs.values():
            f, t = p[LABELS[0]], p[LABELS[1]]
            for lab, pr in (('53', pred), ('Parent', ref_pred['Parent']), ('54', ref_pred[ref54])):
                c = core.pair_category(pr[f], pr[t])
                cats[lab][c] = cats[lab].get(c, 0) + 1
        hist = {}
        for x in per:
            k = x['vs_parent'].get('divergence_relative_to_decision', 'NOT_VERIFIED')
            hist[k] = hist.get(k, 0) + 1
        out[ck] = {'images': DEV_PER_CHECKPOINT, 'correct': sum(x['correct53'] for x in per), 'invalid': sum(p < 0 for p in pred), 'regression_timing': timing,
                   'partner_outcomes': partner, 'pair_categories': cats['53'], 'pair_categories_parent': cats['Parent'], 'pair_categories_54': cats['54'],
                   'identical_tokens_to_parent': sum(x['vs_parent']['identical_token_sequence'] for x in per),
                   'identical_tokens_to_54': sum(x['vs_54']['identical_token_sequence'] for x in per),
                   'divergence_relative_to_decision_vs_parent': hist, 'returned_tokens': sum(len(r['token_ids']) for r in rows), 'per_image_PRIVATE': per}
    a, b = new_rows['A53'], new_rows['B53']
    out['A53_vs_B53'] = {'identical_predictions': sum(x == y for x, y in zip(preds['A53'], preds['B53'])),
                         'identical_token_sequences': sum(list(x['token_ids']) == list(y['token_ids']) for x, y in zip(a, b))}
    out['scope_label'] = ('DIAGNOSTIC_ONLY: 4 preselected, previously explored DEV images per checkpoint (2 regressions + partners); '
                          'NOT Adam53 DEV24 accuracy, NOT independent generalization')
    return out


# ================================================================ public aggregate
def public_aggregate(analysis):
    t, d = analysis['train'], analysis['dev']
    keep = ('images', 'correct', 'invalid', 'regression_timing', 'partner_outcomes', 'pair_categories', 'pair_categories_parent', 'pair_categories_54',
            'identical_tokens_to_parent', 'identical_tokens_to_54', 'divergence_relative_to_decision_vs_parent', 'returned_tokens')
    pub = {'schema': SCHEMA + '-aggregate', 'boundaries': analysis['boundaries'],
           'train': {'surrogate_2x2': t['surrogate_2x2'], 'strata': t['strata'], 'joint_strata': t['joint_strata'],
                     'answer_measurement': t['answer_measurement'], 'cross_checkpoint': t['cross_checkpoint'],
                     'interpretation_limits': t['interpretation_limits']},
           'dev': {**{ck: {k: d[ck][k] for k in keep} for ck in CHECKPOINTS}, 'A53_vs_B53': d['A53_vs_B53'], 'scope_label': d['scope_label']}}
    text = json.dumps(pub)
    require('_PRIVATE' not in text and '/Users/' not in text and '/home/' not in text, 'Public aggregate carries private fields or paths')
    pub['public_schema'] = diag.public_schema_check(pub)
    return pub


# ================================================================ scope assembly / checkpoint binding
def check_checkpoint_entry(entry):
    """The ORIGINAL segment-0 run binding (from its own manifest) and every payload hash; never the new scope binding."""
    require(set(entry) >= {'checkpoint_dir', 'receipt_PRIVATE', 'run_binding', 'payload_sha256'}, 'Incomplete checkpoint entry')
    r = entry['receipt_PRIVATE']
    require(r.get('adam_step_after') == 53 and all(isinstance(r.get(k), str) and r[k] for k in ('policy_sha_after', 'adam_digest_after', 'rng_digest_after')),
            'Exact Adam53 receipt identity required')
    ck = Path(entry['checkpoint_dir'])
    require(ck.is_dir() and not ck.is_symlink(), 'Checkpoint directory required')
    files = {str(p): p for p in sorted(ck.rglob('*'))}
    require(not any(p.is_symlink() for p in files.values()), 'Checkpoint must not contain symlinks')
    present = {k for k, p in files.items() if p.is_file()}
    require(present == set(entry['payload_sha256']), 'Checkpoint file inventory differs from the bound payload')
    for name, h in entry['payload_sha256'].items():
        require(shared.file_sha256(name) == h, 'Checkpoint payload changed: ' + Path(name).name)
    manifest = load_json(ck / 'optimizer' / 'manifest.json')
    require(isinstance(entry['run_binding'], dict) and entry['run_binding'] and manifest['bindings']['run'] == entry['run_binding'],
            'Checkpoint must be restored under its ORIGINAL manifest run binding')
    require(manifest['bindings'].get('adapter_sha256') == shared.file_sha256(ck / 'adapter.safetensors'), 'Manifest adapter binding differs')
    return True


def assemble_scope(*, sources, files, run_output, checkpoints, units, recorded_contexts, tf_jobs, dev_jobs, selection, lengths, runtime, pins,
                   expected_images=None, tokenizer=None, preparation=None):
    require(set(checkpoints) == set(CHECKPOINTS), 'A53 and B53 entries required')
    for entry in checkpoints.values():
        check_checkpoint_entry(entry)
    require(len(dev_jobs) == DEV_PER_CHECKPOINT and [d['index'] for d in dev_jobs] == list(range(DEV_PER_CHECKPOINT)), 'Exactly 4 frozen DEV jobs required')
    payload = {}
    for entry in checkpoints.values():
        payload.update(entry['payload_sha256'])
    budget = budget_for(tf_jobs, lengths)
    inputs = {'files': files, 'payload': payload, 'tf_jobs': identity(tf_jobs), 'dev_jobs': identity(dev_jobs), 'selection': identity(selection),
              'units': identity(units), 'recorded_contexts': identity(recorded_contexts), 'checkpoints': identity(checkpoints)}
    return {'schema': SCHEMA, 'scope_kind': KIND, 'action': ACTION, 'implemented_by': IMPLEMENTED_BY, 'status_at_preparation': 'PREPARED_PENDING_EXACT_REVIEW_AND_FRESH_HUMAN',
            'source_sha256': dict(sources), 'file_sha256': dict(files), 'payload_sha256': payload, 'input_sha256': identity(inputs), 'run_output': str(run_output),
            'checkpoints_PRIVATE': checkpoints, 'units_PRIVATE': units, 'recorded_contexts_PRIVATE': recorded_contexts, 'tf_jobs_PRIVATE': tf_jobs,
            'dev_jobs_PRIVATE': dev_jobs, 'selection_PRIVATE': selection, 'lengths': dict(lengths), 'condition': CONDITION, 'budget': budget, 'totals': dict(TOTALS),
            'run_limits': dict(RUN_LIMITS), 'resource_policy': dict(RESOURCE_POLICY), 'external_guardian_policy': GUARDIAN_POLICY, 'runtime_PRIVATE': dict(runtime),
            'pins': dict(pins), 'expected_images_PRIVATE': dict(expected_images or {}), 'tokenizer_PRIVATE': dict(tokenizer or {}),
            'preparation_PRIVATE': dict(preparation or {}), 'automatic_retry': False, 'training': False,
            'forbidden_inputs': ['final100', 'sealed new16 outputs'], 'old_current_parity_gate_applied_to_post_update_scores': False}


def commit_complete(journal, out, result):
    """Final journal operation; a pause before its claim leaves no unbound terminal file (same semantics as V2)."""
    complete = Path(out) / 'native_complete_PRIVATE.json'
    def commit(_):
        require(not complete.exists(), 'Uncommitted completion file cannot be adopted')
        shared.publish_once(complete, result)
        return {'value': result, 'files': {str(complete.resolve()): shared.file_sha256(complete)}}
    return journal.run('native_complete', commit)


# ================================================================ CPU preparation over the immutable V2 run (metadata + streamed hashes)
def _context_key(unit, response_id):
    return response_id[0] if unit['kind'] == 'SEQ_PAIR' else response_id.split('_')[0]


def derive_tf_jobs(units4, raw, blocks, plans, lengths, decoder):
    jobs = []
    for unit, rollout, block, old in zip(units4, raw['raws'], blocks, raw['olds']):
        by_raw = {r['response_id']: r for r in rollout['responses']}
        rows = {c['key']: c['row'] for c in unit['contexts']}
        for response in block.responses:
            rid = response.response_id
            key = _context_key(unit, rid)
            item, row, recorded = by_raw[rid], rows[key], rollout['contexts'][key]
            tokens = list(response.token_ids)
            require(list(item['token_ids']) == tokens and old[rid] and len(old[rid]) == len(tokens), 'Saved token/own-old inventory differs')
            eos_last = item.get('finish_reason') == 'stop'
            pa, pb = plans['A'][(unit['index'], rid)], plans['B'][(unit['index'], rid)]
            jobs.append({'index': len(jobs), 'unit': unit['index'], 'task': row['ability'], 'context_key': key, 'response_id': rid,
                         'row_id_sha256': identity({'row_id': row['id']}), 'prompt_length': recorded['prompt_length'], 'tokens': len(tokens),
                         'tf_positions': recorded['prompt_length'] + len(tokens) - 1, 'token_ids': tokens, 'token_sha256': identity(tuple(tokens)),
                         'old_logps_PRIVATE': [float(v) for v in old[rid]], 'old_logps_sha256': identity(old[rid]),
                         'reward_receipt_sha256': response.reward_receipt_sha256, 'block_sha256': block.sha256,
                         'recorded_context_sha256': identity(recorded), 'raw_accuracy': response.grade.raw_accuracy, 'format': response.grade.format,
                         'reward_total': response.grade.total, 'image_class': {'F': 'FLAIR_IMAGE', 'T': 'T1_IMAGE'}.get(key, 'DALLAS'),
                         'eos_last': eos_last, 'finish_reason': item.get('finish_reason'),
                         'answer': answer_span(tokens, decoder, item['text'], row['ability'], eos_last=eos_last),
                         'plans': {arm: {k: p[k] for k in ('advantage', 'weight', 'admitted', 'group')} for arm, p in (('A', pa), ('B', pb))}})
    tf_job_positions(jobs, lengths)
    return jobs


def _reference(row):
    r = row['actual_PRIVATE']['response']
    return {'text': row['text'], 'token_ids': list(r['token_ids']), 'raw_selected_logps': list(r['raw_selected_logps'])}


def checkpoint_entry(run_dir, receipt, label):
    ck = Path(run_dir) / f'stage{ARM_OF[label]}_PRIVATE' / 'arm_PRIVATE' / 'segment_0' / 'checkpoint_PRIVATE'
    require(receipt['updated'] is True and receipt['adam_step_after'] == 53 and receipt['segment'] == 0, 'Segment-0 receipt must be one actual Adam53 update')
    payload = {str(p): shared.file_sha256(p) for p in sorted(ck.rglob('*')) if p.is_file()}
    for name, h in receipt['files_sha256'].items():
        require(payload.get(str(ck / name)) == h, 'Checkpoint payload differs from its journal-bound receipt: ' + name)
    run_binding = load_json(ck / 'optimizer' / 'manifest.json')['bindings']['run']
    entry = {'checkpoint_dir': str(ck), 'run_binding': run_binding, 'payload_sha256': payload,
             'receipt_PRIVATE': {k: receipt[k] for k in ('policy_sha_after', 'adam_digest_after', 'rng_digest_after', 'adam_step_after', 'receipt_sha256')}}
    check_checkpoint_entry(entry)
    return entry


def prepare(*, root, worktree, run_dir=None, results_path=None, out_dir=None, expected_v2_scope_sha256=EXPECTED['v2_scope_file_sha256']):
    """CPU-only exact preparation. Reads metadata/JSON, stream-hashes checkpoint payloads, loads no model and imports no MLX."""
    from mri_rl import multistep_gspo_core_v1 as ms
    from mri_rl.safe_micro_entry_v1 import _closure
    from mri_rl.safe_micro_recovery_entry_v1 import bind_runtime_sources
    root, worktree = Path(root).resolve(), Path(worktree).resolve()
    run_dir = Path(run_dir or root / 'local_data/accuracy_pilot_v2/pilot_v2_attempt1_PRIVATE').resolve()
    results_path = Path(results_path or worktree / 'results/ACCURACY_PILOT_V2_ACTUAL_RESULTS_V1.json').resolve()
    out_dir = Path(out_dir or root / 'local_data/v2_native_readonly_v1').resolve()
    require(out_dir.is_relative_to(root / 'local_data') and not out_dir.is_relative_to(run_dir) and not (root / 'local_data').is_symlink(),
            'Private preparation output must be a new directory under local_data, outside the immutable V2 run')
    with cpu_only_imports():
        aggregate = worktree / 'results/V2_REGRESSION_DIAGNOSIS_ACTUAL_CPU_V1.json'
        pins = {'v2_results_sha256': diag.require_file_sha(results_path, EXPECTED['v2_results_sha256'], 'committed V2 results'),
                'diagnosis_module_sha256': diag.require_file_sha(worktree / 'src/mri_rl/v2_regression_diagnosis_v1.py', EXPECTED['diagnosis_module_sha256'],
                                                                 'reused diagnosis module'),
                'diagnosis_aggregate_sha256': diag.require_file_sha(aggregate, EXPECTED['diagnosis_aggregate_sha256'], 'diagnosis aggregate')}
        ctx = diag.authenticate(root=root, worktree=worktree, run_dir=run_dir, results_path=results_path, expected_scope_sha256=expected_v2_scope_sha256)
        require(ctx['condition'] == CONDITION, 'V2 frozen prompt condition must be concise')
        spec, runtime, v2scope = ctx['spec'], ctx['runtime'], ctx['scope']
        lengths = core.prompt_lengths_for(CONDITION, spec['concise_sequence_prompt_length'])
        units4 = ctx['units'][:4]
        require([u['index'] for u in units4] == list(core.SEGMENTS[0]), 'Common units 0-3 required')
        raw_path = ctx['common_dir'] / 'raw_rollout_and_own_old_PRIVATE.json'
        raw = load_json(raw_path)
        parent_state = ctx['common']['parent_state']
        blocks = diag.rebuild_common_blocks(units4, raw['raws'], parent_policy=parent_state['policy'], reward_sha=ctx['reward_sha'], parent_receipt=ctx['parent_receipt'])
        require([b.sha256 for b in blocks] == ctx['common']['bundle_sha256'], 'Rebuilt Common bundles differ from the common record')
        for block, old in zip(blocks, raw['olds']):
            ms.check_old(block, old)                    # the SAVED 52 own-old vs rollout parity (historical); not applied to new scores
        plans, evidence_files = {}, {}
        for arm in ('A', 'B'):
            seg = run_dir / f'stage{arm}_PRIVATE' / 'arm_PRIVATE' / 'segment_0'
            evidence = load_json(seg / 'gradient_evidence_PRIVATE.json')
            records = [diag.plan_records(b, u, arm) for b, u in zip(blocks, units4)]
            require(all(diag.compare_with_saved_evidence(r, e)['match'] for r, e in zip(records, evidence)), f'{arm}: re-derived plan differs from saved evidence')
            plans[arm] = {(r['unit'], x['response_id']): x for r in records for x in r['responses']}
            evidence_files[str(seg / 'gradient_evidence_PRIVATE.json')] = shared.file_sha256(seg / 'gradient_evidence_PRIVATE.json')
            evidence_files[str(seg / 'receipt_PRIVATE.json')] = shared.file_sha256(seg / 'receipt_PRIVATE.json')
        model = Path(runtime['model'])
        tokenizer = {'vocab': str(model / 'vocab.json'), 'tokenizer_json': str(model / 'tokenizer.json')}
        decoder = diag.TokenDecoder.from_files(tokenizer['vocab'], tokenizer['tokenizer_json'])
        tf_jobs = derive_tf_jobs(units4, raw, blocks, plans, lengths, decoder)
        checkpoints = {label: checkpoint_entry(run_dir, ctx['receipts'][(ARM_OF[label], 0)], label) for label in CHECKPOINTS}
        records = ctx['dev']['records_PRIVATE']
        selection = select_dev_cases(records, ctx['dev_rows'])
        dev_jobs, row_files = [], {}
        pair_of = {i: p for p, pair in enumerate(selection['pairs_PRIVATE']) for i in pair}
        for k, i in enumerate(selection['selected_PRIVATE']):
            rec = records[i]
            row = core.condition_row(rec['row_PRIVATE'], CONDITION)
            refs = {}
            for label, stage, name in (('Parent', 'EP', 'Parent'), ('A54', 'EA', 'A'), ('B54', 'EB', 'B')):
                path = run_dir / f'stage{stage}_PRIVATE' / 'rows_PRIVATE' / f'{name}_row_{i:02d}_PRIVATE.json'
                row_files[str(path)] = shared.file_sha256(path)
                refs[label] = _reference(ctx['dev_rows'][name][i])
            dev_jobs.append({'index': k, 'dev_case_index_PRIVATE': i, 'pair': pair_of[i], 'role': selection['roles_PRIVATE'][str(i)],
                             'truth': rec['row_PRIVATE']['reward_model']['ground_truth'], 'row_PRIVATE': row, 'row_sha256': identity(row),
                             'image_sha256': rec['image_sha256'], 'reference_PRIVATE': refs})
        stage_input = load_json(spec['stage_input'])
        train_images = {x['record_id_PRIVATE']: x['image_binding_PRIVATE']['sha256'] for x in stage_input['planned_image_bindings_PRIVATE']}
        expected_images = {**{c['row']['id']: train_images[c['row']['id']] for u in units4 for c in u['contexts']},
                           **{d['row_PRIVATE']['id']: d['image_sha256'] for d in dev_jobs}}
        spec_path = root / 'local_data/accuracy_pilot_v2/PILOT_V2_SPEC_PRIVATE.json'
        files = {str(Path(p)): h for p, h in v2scope['file_sha256'].items()}
        for p, h in files.items():
            require(shared.file_sha256(p) == h, 'V2-bound metadata changed: ' + Path(p).name)
        for p in (spec_path, results_path, aggregate, raw_path, ctx['common_dir'] / 'common_record_PRIVATE.json', Path(tokenizer['vocab']),
                  Path(tokenizer['tokenizer_json']), Path(spec['runtime_spec']), Path(spec['stage_input']), Path(spec['dev24']), Path(runtime['calibration'])):
            files[str(Path(p))] = shared.file_sha256(p)
        files.update(evidence_files)
        files.update(row_files)
        own = ('src/mri_rl/v2_native_readonly_v1.py', 'src/mri_rl/v2_native_readonly_entry_v1.py', 'src/mri_rl/v2_native_readonly_backend_v1.py',
               'scripts/run_v2_native_readonly_v1.py', 'src/mri_rl/v2_regression_diagnosis_v1.py', 'src/mri_rl/accuracy_pilot_v2_native.py',
               'src/mri_rl/accuracy_pilot_v2_entry.py', 'src/mri_rl/safe_micro_watchdog_v2.py')
        sources = _closure({str(worktree / p): shared.file_sha256(worktree / p) for p in own})
        for p, h in v2scope['source_sha256'].items():            # source-state pins: every source of the executed V2 run is unchanged
            require(shared.file_sha256(p) == h and sources.get(p, h) == h, 'Executed V2 source changed: ' + Path(p).name)
            sources[p] = h
        sources = bind_runtime_sources(sources, runtime)
        recorded = {str(u['index']): rollout['contexts'] for u, rollout in zip(units4, raw['raws'])}
        scope = assemble_scope(sources=sources, files=files, run_output=out_dir / 'run_PRIVATE', checkpoints=checkpoints, units=units4,
                               recorded_contexts=recorded, tf_jobs=tf_jobs, dev_jobs=dev_jobs, selection=selection, lengths=lengths,
                               runtime={'runtime_spec': spec['runtime_spec'], 'global_lock': runtime['global_lock'], 'recovery_spec': spec['recovery_spec'],
                                        'v2_spec': str(spec_path)},
                               pins={**pins, 'v2_scope_file_sha256': expected_v2_scope_sha256, 'v2_journal_commits_verified': len(ctx['record']['journal_commits'])},
                               expected_images=expected_images, tokenizer=tokenizer,
                               preparation={'root': str(root), 'worktree': str(worktree), 'run_dir': str(run_dir), 'results_path': str(results_path),
                                            'out_dir': str(out_dir), 'expected_v2_scope_sha256': expected_v2_scope_sha256})
    summary = {'tf_forwards': TOTALS['teacher_forced_scorer_forwards'], 'tf_positions_total': scope['budget']['tf_positions_total'],
               'tf_positions_per_checkpoint': scope['budget']['tf_positions_per_checkpoint'], 'greedy_generations': TOTALS['greedy_generations'],
               'returned_tokens_max': TOTALS['returned_tokens_max'], 'planning_estimate_seconds_per_checkpoint': PLANNING_ESTIMATE_SECONDS_PER_CHECKPOINT,
               'windows': {k: RUN_LIMITS[k] for k in ('per_checkpoint_window_seconds', 'total_window_seconds')}, 'model_loads_max': 2,
               'answer_status_counts': _counts(j['answer']['status'] for j in tf_jobs),
               'measurement_kind_counts': _counts(j['answer']['measurement_kind'] for j in tf_jobs),
               'label_metric_eligible_by_task': _counts(_task(j) for j in tf_jobs if j['answer']['label_metric_eligible']),
               'ineligibility_reasons': _counts(j['answer']['ineligibility_reason'] for j in tf_jobs if j['answer']['ineligibility_reason']),
               'decoded_text_exact': sum(j['answer']['decoded_text_exact'] for j in tf_jobs),
               'parser_format_differs_from_recorded_grade': sum(j['answer']['parser_format'] != j['format'] for j in tf_jobs),
               'joint_task_format_outcome_counts': _counts(f'{_task(j)}|{format_class(j)}|{raw_outcome(j)}' for j in tf_jobs),
               'tasks': _counts(j['task'] for j in tf_jobs),
               'heavy_modules_loaded': heavy_modules_loaded()}
    return {'status': 'PREPARED_PENDING_EXACT_REVIEW_AND_FRESH_HUMAN', 'executable_scope': scope, 'scope_sha256': digest(scope), 'native_calls': 0,
            'public_summary': summary}


def _counts(values):
    out = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items()))


# ================================================================ CPU analysis of committed native outputs (no model)
def run_analysis(scope, *, decoder=None):
    from mri_rl.safe_micro_journal_v1 import Journal
    base = Path(scope['run_output'])
    tf_jobs, dev_jobs = scope['tf_jobs_PRIVATE'], scope['dev_jobs_PRIVATE']
    new, dev_new, completes = {}, {}, {}
    for label in CHECKPOINTS:
        out = base / f'{label}_PRIVATE'
        with Journal(out / 'journal_PRIVATE', {'scope': digest(scope), 'checkpoint': label, 'input': scope['input_sha256']}) as journal:
            completes[label] = journal.get('native_complete')
            require(completes[label] is not None and completes[label]['status'] == 'PASS' and completes[label]['release_verified'] is True,
                    'Committed PASS with verified release required: ' + label)
            new[label] = {}
            for j in tf_jobs:
                v = journal.get('tf_%02d' % j['index'])
                rec = load_json(v['path'])
                require(identity(rec['logps']) == v['logps_sha256'] and rec['token_sha256'] == j['token_sha256'] and rec['index'] == j['index'],
                        'Committed TF row differs from its job')
                new[label][j['index']] = rec['logps']
            dev_new[label] = []
            for d in dev_jobs:
                item = load_json(journal.get('dev_%d' % d['index'])['path'])
                require(item['index'] == d['index'] and item['truth'] == d['truth'], 'Committed DEV row differs from its job')
                r = item['actual_PRIVATE']['response']
                dev_new[label].append({'text': item['text'], 'token_ids': list(r['token_ids']), 'raw_selected_logps': list(r['raw_selected_logps'])})
    if decoder is None:
        tok = scope['tokenizer_PRIVATE']
        for p in tok.values():
            require(shared.file_sha256(p) == scope['file_sha256'][str(Path(p))], 'Tokenizer file changed')
        decoder = diag.TokenDecoder.from_files(tok['vocab'], tok['tokenizer_json'])
    analysis = {'schema': SCHEMA + '-analysis', 'boundaries': dict(TOTALS), 'train': analyze_train(tf_jobs, new), 'dev': analyze_dev(dev_jobs, dev_new, decoder),
                'native_completes_PRIVATE': completes,
                'questions': {'Q1_when_regression': 'dev.*.regression_timing (53 vs saved 54)', 'Q2_A_B_divergence': 'dev.A53_vs_B53; train.cross_checkpoint',
                              'Q3_paired_collapse': 'dev.*.pair_categories vs _parent / _54', 'Q4_fixed_TRAIN_surrogate_vs_DEV': 'train.surrogate_2x2 / strata vs dev'}}
    public = public_aggregate(analysis)
    shared.publish_once(base / 'analysis_PRIVATE.json', analysis)
    shared.publish_once(base / 'AGGREGATE_PUBLIC_CANDIDATE.json', public)
    return {'public': public, 'model_calls_in_analysis': 0}
