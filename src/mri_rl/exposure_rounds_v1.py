"""Finite PAIRED-ROUND exposure plan for the matched Vero-style A/B on the paired z045 pool (CPU only, no model).

Replaces the old finite prefix, where only 4 of 12 exposed patients ever received both modalities. A ROUND is the atomic
exposure unit:  SEQ_PAIR block (one patient's FLAIR + contrast-enhanced-T1 rows = two K4 groups = 8 responses = one update)
followed by DALLAS_BLOCK (two unchanged Dallas rows = two K4 groups = 8 responses = one update). Task share is therefore
explicitly 50/50 by K4 groups, as in the original uniform sampler, and class share is exactly 50/50 within sequence.
The plan is fixed BEFORE any model output (deterministic hashes only) and is identical in both arms: only the sequence
grouping (A same-image K4 vs B F2T2) differs. A cap can only truncate WHOLE rounds, so a pair is never split.
A prefix of rounds is reported as a prefix; ALL56 is claimed only when all 28 patients are in the plan.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import copy
import hashlib
from mri_rl.paired_z045_data_v1 import SEQ, DALLAS, LABELS, canonical_sha, require
from mri_rl.paired_z045_exposure_v2 import patient_order

SALT_DALLAS = 'paired-round-dallas-order-v1'
UPDATES_PER_ROUND = 2
K4_GROUPS_PER_BLOCK = 2
RESPONSES_PER_BLOCK = 8
MAX_TOKENS = 256
GROUPINGS = {'A': 'same_prompt', 'B': 'paired_prompt'}


def _dallas_order(dallas):
    """Deterministic hash order, round-robin across ground-truth strata so classes stay balanced in every prefix."""
    require(len({r['id'] for r in dallas}) == len(dallas) and all(r['ability'] == DALLAS for r in dallas), 'Unique typed Dallas rows required')
    strata = defaultdict(list)
    for row in dallas:
        strata[str(row['reward_model']['ground_truth'])].append(row)
    rank = lambda row: hashlib.sha256((SALT_DALLAS + ':' + row['id']).encode()).hexdigest()
    for key in strata:
        strata[key].sort(key=rank)
    ordered = []
    for index in range(max(map(len, strata.values()))):
        for key in sorted(strata):
            if index < len(strata[key]):
                ordered.append(strata[key][index])
    return ordered


def build_round_plan(sequence_pool, dallas_pool, sites, *, rounds):
    require(type(rounds) is int and 1 <= rounds <= 28, 'Between 1 and 28 patient rounds required')
    order, groups = patient_order(sequence_pool, sites)          # existing deterministic site-interleaved patient order; asserts 28 complete pairs
    dallas = _dallas_order(dallas_pool)
    require(len(dallas) >= 2, 'At least two Dallas rows required')       # >24 rounds cycle the fixed order (Dallas is engineering-only)
    plan_rounds = []
    for r in range(rounds):
        person = order[r]
        labels = LABELS if r % 2 == 0 else tuple(reversed(LABELS))      # alternate orientation independently of results
        seq = [copy.deepcopy(groups[person][label]) for label in labels]
        blocks = [{'kind': 'SEQ_PAIR', 'task': SEQ, 'rows': seq}, {'kind': 'DALLAS_BLOCK', 'task': DALLAS, 'rows': [copy.deepcopy(dallas[(2 * r) % len(dallas)]), copy.deepcopy(dallas[(2 * r + 1) % len(dallas)])]}]
        plan_rounds.append({'round': r + 1, 'patient': person, 'blocks': blocks})
    plan = {'schema': 'paired-round-exposure-plan-v1', 'rounds': plan_rounds, 'patient_sites': {r['patient']: sites[r['patient']] for r in plan_rounds}, 'sites_covered': sorted({sites[r['patient']] for r in plan_rounds}), 'policy': {
        'unit': 'ROUND = SEQ_PAIR(2 K4 groups, 8 responses, 1 update) + DALLAS_BLOCK(2 K4 groups, 8 responses, 1 update)',
        'task_balance': 'explicit 50/50 by K4 groups (original uniform sampler logical weights 0.5/0.5)',
        'class_balance': 'exactly one FLAIR and one contrast-enhanced T1 row per selected patient; Dallas strata round-robin',
        'selection': 'deterministic site-interleaved salted hash order, fixed before any model output',
        'cap_rule': 'whole rounds only; a patient pair is never split by an update/wall cap',
        'dallas': 'unchanged rows/question/grader; ENGINEERING_ONLY, not a second scientific task block',
        'arms': 'identical plan in both arms; only the sequence-block grouping differs'}}
    plan['plan_sha256'] = canonical_sha({'rounds': [{'round': r['round'], 'patient': r['patient'], 'blocks': [{'kind': b['kind'], 'rows': [x['id'] for x in b['rows']]} for b in r['blocks']]} for r in plan_rounds],
                                         'policy': plan['policy']})
    return plan


def arm_view(plan, arm):
    require(arm in GROUPINGS, 'Arm A or B required')
    return {**copy.deepcopy(plan), 'arm': arm, 'grouping': GROUPINGS[arm]}


def truncate_to_update_cap(plan, *, max_updates):
    require(type(max_updates) is int and max_updates >= UPDATES_PER_ROUND, 'Cap smaller than one whole round would split a patient pair')
    keep = max_updates // UPDATES_PER_ROUND
    cut = copy.deepcopy(plan)
    cut['rounds'] = cut['rounds'][:keep]
    cut['patient_sites'] = {r['patient']: plan['patient_sites'][r['patient']] for r in cut['rounds']}      # recomputed from the KEPT patients
    cut['sites_covered'] = sorted(set(cut['patient_sites'].values()))
    cut['plan_sha256'] = canonical_sha({'truncated_of': plan['plan_sha256'], 'rounds': keep})
    return cut


def exposure_report(plan):
    seq_rows = [row for r in plan['rounds'] for b in r['blocks'] if b['kind'] == 'SEQ_PAIR' for row in b['rows']]
    dallas_rows = [row for r in plan['rounds'] for b in r['blocks'] if b['kind'] == 'DALLAS_BLOCK' for row in b['rows']]
    by_person = defaultdict(set)
    for row in seq_rows:
        by_person[row['extra_info']['patient_key']].add(row['reward_model']['ground_truth'])
    patients = len(by_person)
    k4 = {SEQ: len(seq_rows), DALLAS: len(dallas_rows)}
    total = sum(k4.values())
    return {'rounds': len(plan['rounds']), 'patients_exposed': patients, 'patients_with_both_modalities': sum(len(v) == 2 for v in by_person.values()),
            'patients_with_one_modality': sum(len(v) == 1 for v in by_person.values()), 'sequence_rows': len(seq_rows),
            'sequence_rows_repeated': len(seq_rows) - len({r['id'] for r in seq_rows}), 'class_rows': dict(Counter(r['reward_model']['ground_truth'] for r in seq_rows)),
            'k4_groups': k4, 'task_share': {k: v / total for k, v in k4.items()} if total else {},
            'dallas_rows_unique': len({r['id'] for r in dallas_rows}), 'dallas_class_rows': dict(Counter(str(r['reward_model']['ground_truth']) for r in dallas_rows)),
            'sites_covered': list(plan.get('sites_covered', [])),
            'updates_max_per_arm': UPDATES_PER_ROUND * len(plan['rounds']),
            'updates_basis': 'POTENTIAL_MAX 2 per round; actual updates = blocks with an admitted group (a no-admitted block makes 0)',
            'all_56_rows_exposed': patients == 28 and len(seq_rows) == 56,
            'coverage_claim': 'ALL_28_PATIENTS_56_ROWS' if patients == 28 and len(seq_rows) == 56 else f'PREFIX_{patients}_OF_28_PATIENTS',
            'plan_sha256': plan['plan_sha256']}


def original_pool_exposure(schedule_rows):
    """Exposure of an ORIGINAL (one-modality-per-person) finite schedule, same report keys where defined."""
    by_person = defaultdict(set)
    for row in schedule_rows:
        by_person[row['extra_info']['patient_key']].add(row['reward_model']['ground_truth'])
    return {'patients_exposed': len(by_person), 'patients_with_both_modalities': sum(len(v) == 2 for v in by_person.values()),
            'patients_with_one_modality': sum(len(v) == 1 for v in by_person.values()), 'sequence_rows': len(schedule_rows)}


def compare_with_original(new_report, original_report):
    keys = ('patients_exposed', 'patients_with_both_modalities', 'patients_with_one_modality', 'sequence_rows')
    return {'original': {k: original_report[k] for k in keys}, 'new': {k: new_report[k] for k in keys},
            'both_modality_fraction_original': original_report['patients_with_both_modalities'] / max(1, original_report['patients_exposed']),
            'both_modality_fraction_new': new_report['patients_with_both_modalities'] / max(1, new_report['patients_exposed'])}


def describe_costs(plan, *, prompt_lengths):
    """Worst-case generation charge per K4 group = 4 x (measured expanded prompt + 256 response tokens)."""
    require(set(prompt_lengths) == {SEQ, DALLAS} and all(type(v) is int and 0 < v <= 2048 for v in prompt_lengths.values()), 'Measured expanded prompt lengths for both tasks required')
    groups = exposure_report(plan)['k4_groups']
    return {'k4_groups': groups, 'generation_charged_tokens': sum(groups[t] * 4 * (prompt_lengths[t] + MAX_TOKENS) for t in groups),
            'max_returned_response_tokens': sum(groups.values()) * 4 * MAX_TOKENS, 'updates_max_per_arm': UPDATES_PER_ROUND * len(plan['rounds']),
            'qualification': 'Conditional on the measured expanded prompt lengths; unreturned AR lookahead and checkpoint recomputation are additional work.'}
