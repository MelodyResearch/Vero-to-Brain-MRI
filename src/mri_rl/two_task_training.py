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
import numpy as np

from mri_rl.full_arm_protocol import can_start_group, charge_group
from mri_rl.reward_replay import _logistic
from mri_rl.vero_mri_reward import compute_score_from_data_source
from mri_rl.vero_official_launch import expected_batch_counts
from mri_rl.visit_correspondence import SOURCE as DALLAS_SOURCE, score_visit_correspondence

SEQ_SOURCE = 'OpenNeuro_ds007045'
ROUTES = {'mri_sequence': SEQ_SOURCE, 'mri_visit_correspondence': DALLAS_SOURCE}
BLOCK = 256


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


def build_schedule(rows, weights, seed, n_attempts):
    """Deterministic occurrence schedule of length n_attempts (<= a few blocks). Returns rows with occurrence ids."""
    by_cat = {t: [r for r in rows if r['ability'] == t] for t in ROUTES}
    for r in rows:
        validate_training_row(r)
    active = {t: w for t, w in weights.items() if w > 0}
    if abs(sum(active.values()) - 1) > 1e-9 or any(not by_cat[t] for t in active):
        raise ValueError('Weights must sum to 1 over non-empty categories')
    rng = np.random.default_rng(seed)
    pools = {t: [] for t in active}
    seen = Counter()
    out = []
    while len(out) < n_attempts:
        counts = expected_batch_counts(active, BLOCK)
        block = []
        for t, c in counts.items():
            for _ in range(c):
                if not pools[t]:                       # exhaustion -> reshuffle the whole category
                    pools[t] = list(rng.permutation(len(by_cat[t])))
                block.append(by_cat[t][pools[t].pop()])
        order = rng.permutation(len(block))            # within-block shuffle
        for i in order:
            src = block[int(i)]
            occ = seen[src['id']]; seen[src['id']] += 1
            out.append({**src, 'id': f"{src['id']}#occ{occ}", 'extra_info': {**src['extra_info'], 'source_id': src['id'], 'occurrence': occ}})
    out = out[:n_attempts]                             # token-limited prefix: realized weights are reported, not the nominal ones
    return out


def realized_report(schedule):
    c = Counter(r['ability'] for r in schedule)
    unique = {t: len({r['extra_info']['source_id'] for r in schedule if r['ability'] == t}) for t in ROUTES}
    patients = {t: len({r['extra_info']['patient_key'] for r in schedule if r['ability'] == t}) for t in ROUTES}
    return {'attempts': len(schedule), 'category_counts': dict(c), 'unique_source_rows': unique, 'distinct_patients': patients,
            'schedule_sha256': hashlib.sha256(json.dumps([r['id'] for r in schedule]).encode()).hexdigest()}


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
    """Persisted whole-group token accounting around full_arm_protocol.can_start_group/charge_group."""
    def __init__(self, plan, used=0, charged_groups=0):
        self.plan, self.used, self.charged_groups = plan, used, charged_groups

    def can_start(self, input_tokens):
        return can_start_group(self.used, input_tokens, self.plan)

    def charge(self, input_tokens, generated_lengths):
        self.used += charge_group(input_tokens, generated_lengths, self.plan)
        self.charged_groups += 1
        return self.used

    def state(self):
        return {'used_tokens': self.used, 'charged_groups': self.charged_groups, 'cap': self.plan['budget']['text_visual_token_cap']}
