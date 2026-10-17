"""Shared Tier-C training surface for the two-task comparison (Claude-authored; reuses, does not replace, the helpers).

* schedule: pinned-Vero-style PerBatchDomainSampler semantics (256-prompt logical block, floor/largest-remainder
  category counts, shuffled category pools with exhaustion reshuffle, within-block shuffle) -> repeated OCCURRENCE ids
  that keep the original source id. Parity with the pinned Vero implementation is SEMANTIC only (NOT_VERIFIED line by line).
* grading: explicit source routing (ds007045 sequence via vero_mri_reward; ds004856 correspondence via the data agent's
  scorer) and the MedGRPO Eq.3 accuracy transform from FROZEN common-initializer statistics, zero IQR blocks.
* ledger: whole-group token reservation/charging via full_arm_protocol, persisted so a resume never gets free compute.
Standard library + numpy only; no model imports.
"""
from __future__ import annotations
import hashlib
import json
from collections import Counter

from mri_rl.full_arm_protocol import can_start_group, charge_group
from mri_rl.reward_replay import _logistic
from mri_rl.vero_mri_reward import compute_score_from_data_source
from mri_rl.visit_correspondence import SOURCE as DALLAS_SOURCE, score_visit_correspondence

SEQ_SOURCE = 'OpenNeuro_ds007045'
ROUTES = {'mri_sequence': SEQ_SOURCE, 'mri_visit_correspondence': DALLAS_SOURCE}


class CalibrationBlocked(ValueError):
    """Zero/invalid IQR: the calibrated configuration is blocked; no floor is invented."""


def validate_training_row(row):
    """Typed two-source boundary: source/route agree, training split, patient lineage, one image."""
    e = row.get('extra_info', {})
    route = row.get('ability')
    if ROUTES.get(route) != row.get('data_source') or e.get('reward_type') != route:
        raise ValueError('Source/route mismatch')
    if e.get('internal_split') != 'train' or e.get('patient_lineage') != 'BIDS_SUBJECT_VERIFIED' or not e.get('patient_key'):
        raise ValueError('Training-only verified lineage required')
    if len(row.get('images', [])) != 1:
        raise ValueError('One image per row required')
    if route == 'mri_visit_correspondence':
        from mri_rl.visit_correspondence import validate_correspondence_row
        validate_correspondence_row(row, expected_split='train')
    return route


def category_weights(arm, rows):
    """Weights derived from the admitted manifest: specialist / proportional / uniform."""
    counts = Counter(r['ability'] for r in rows)
    if arm['sampling'] == 'specialist':
        return {t: (1.0 if t == arm['task'] else 0.0) for t in ROUTES}
    if arm['sampling'] == 'uniform':
        return {t: 1 / len(ROUTES) for t in ROUTES}
    if arm['sampling'] == 'proportional':
        return {t: counts[t] / len(rows) for t in ROUTES}
    raise ValueError('Unknown sampling')


def build_schedule(rows, arm, seed, n_attempts):
    """Thin wrapper over Codex's pinned-class-AST-parity sampler (vero_category_sampler). Returns the frozen occurrence
    schedule: opaque unique top-level id, top-level source_record_id (original frozen row id), all else unchanged."""
    from mri_rl.vero_category_sampler import build_category_schedule
    mode = arm['sampling']
    out = build_category_schedule(rows, mode=mode, seed=seed, max_groups=n_attempts,
                                  specialist_category=arm.get('task') if mode == 'specialist' else None)
    return out['schedule'], out['binding']


def realized_report(schedule):
    c = Counter(r['ability'] for r in schedule)
    unique = {t: len({r['source_record_id'] for r in schedule if r['ability'] == t}) for t in ROUTES}
    patients = {t: len({r['extra_info']['patient_key'] for r in schedule if r['ability'] == t}) for t in ROUTES}
    return {'attempts': len(schedule), 'category_counts': dict(c), 'unique_source_rows': unique, 'distinct_patients': patients,
            'schedule_sha256': hashlib.sha256(json.dumps([r['id'] for r in schedule]).encode()).hexdigest()}


def input_tokens_by_source(measurement_rows, source_rows):
    """Occurrence row -> measured native input tokens (incl. visual). Measurements are keyed by canonical_sha(source row)
    exactly as scripts/measure_native_training_tokens.py wrote them; every frozen source row must be covered."""
    from mri_rl.two_task_pool import canonical_sha
    by_sha = {m['record_sha256']: int(m['input_tokens']) for m in measurement_rows}
    table = {}
    for row in source_rows:
        key = canonical_sha(row)
        if key not in by_sha:
            raise ValueError('Frozen source row has no native token measurement')
        table[row['id']] = by_sha[key]
    return lambda row: table[row.get('source_record_id', row['id'])]


def make_grader(calibration=None):
    """Return grader(row, response_text, config) -> grade dict. calibration: {task: {'p50':..,'iqr':..}} or None (raw)."""
    def grader(row, text, config):
        e = row['extra_info']
        if row['data_source'] == DALLAS_SOURCE:
            g = score_visit_correspondence(row['data_source'], text, row['reward_model']['ground_truth'], e)
        else:
            g = compute_score_from_data_source(row['data_source'], text, row['reward_model']['ground_truth'], e)
        g = dict(g)
        g['raw_accuracy'] = g['accuracy']
        if calibration is not None:
            stats = calibration[row['ability']]
            if not stats['iqr'] > 0:
                raise CalibrationBlocked('Zero or invalid IQR for ' + row['ability'])
            g['accuracy'] = _logistic(g['accuracy'], stats['p50'], stats['iqr'])
            g['score'] = 0.8 * g['accuracy'] + 0.2 * g['format']          # callback form: 0.8*acc + 0.2*format
        return g
    return grader


class TokenLedger:
    """Persisted whole-group token accounting around full_arm_protocol.can_start_group/charge_group.

    `used` = measured actual cost + `imputed` conservative charges for unobserved crashed in-flight work (kept separate;
    imputed amounts are NOT claimed as actual consumption)."""
    def __init__(self, plan, used=0, charged_groups=0, imputed=0):
        self.plan, self.used, self.charged_groups, self.imputed = plan, used, charged_groups, imputed

    def can_start(self, input_tokens):
        return can_start_group(self.used, input_tokens, self.plan)

    def charge(self, input_tokens, generated_lengths):
        self.used += charge_group(input_tokens, generated_lengths, self.plan)
        self.charged_groups += 1
        return self.used

    def charge_imputed(self, tokens):
        self.used += tokens
        self.imputed += tokens

    def state(self):
        return {'used_tokens': self.used, 'charged_groups': self.charged_groups, 'imputed_unknown_tokens': self.imputed,
                'cap': self.plan['budget']['text_visual_token_cap']}
