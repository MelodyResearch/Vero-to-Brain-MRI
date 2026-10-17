"""Common-B28 training-only calibration rollouts and frozen MedGRPO Eq.3 statistics (Claude-authored; new file).

Statistics unit: the per-prompt mean raw accuracy over K rollouts of the COMMON B28 start on the admitted TRAINING rows of
each task (no updates, no development/test rows). p50 = median, IQR = q75 - q25 over prompts. Zero (or non-finite) IQR
blocks the calibrated configuration for that task; no floor is invented. Rollouts are durable (fsynced jsonl, resumable,
never repeated) and their token cost is charged to a shared calibration ledger, reported separately from the arms.
"""
from __future__ import annotations
import json
import math
import os
from pathlib import Path
import numpy as np

from mri_rl.continuous_rl import digest, require
from mri_rl.two_task_training import ROUTES, make_grader, validate_training_row


def stats_from_accuracies(by_task):
    """{task: [per-prompt mean accuracy]} -> {task: {p50, iqr, n, blocked, levels}}."""
    out = {}
    for task in ROUTES:
        v = by_task.get(task, [])
        require(len(v) >= 2 and all(isinstance(x, (int, float)) and math.isfinite(x) and 0 <= x <= 1 for x in v),
                'Calibration needs finite per-prompt accuracies for every task')
        q25, p50, q75 = (float(x) for x in np.percentile(np.array(v, dtype=float), [25, 50, 75]))
        iqr = q75 - q25
        out[task] = {'p50': p50, 'iqr': iqr, 'n': len(v), 'blocked': not iqr > 0, 'distinct_levels': len(set(v))}
    return out


def calibration_for_arms(stats):
    """Frozen statistics usable by make_grader; raises for any blocked task (zero IQR)."""
    blocked = [t for t, s in stats.items() if s['blocked']]
    if blocked:
        raise ValueError('Calibrated configuration blocked by zero IQR: ' + ','.join(blocked))
    return {t: {'p50': s['p50'], 'iqr': s['iqr']} for t, s in stats.items()}


def run_calibration(*, rows, backend_factory, authorize, output, group_size, input_tokens, resume=False):
    """Roll out the common start on every training row once (K=group_size samples), raw-grade, persist per row."""
    for r in rows:
        validate_training_row(r)
    output = Path(output)
    if output.exists() and not resume:
        raise FileExistsError('Existing calibration requires explicit resume')
    output.mkdir(parents=True, exist_ok=True)
    log = output / 'calibration_rows_PRIVATE.jsonl'
    done = {}
    if log.exists():
        for line in log.read_text().splitlines():
            rec = json.loads(line)
            done[rec['id']] = rec
    grader = make_grader(None)
    pending = [r for r in rows if r['id'] not in done]
    authorize({'phase': 'load', 'pending': len(pending)})
    backend = backend_factory() if pending else None
    try:
        for index, row in enumerate(pending):
            authorize({'phase': 'row', 'id': row['id']})
            work = output / f'.work-{row["id"]}'
            work.mkdir(exist_ok=True)
            raw = backend.rollout(row, index, work)
            require(raw.get('record_id') == row['id'] and raw.get('image_sha256') == digest(row['images'][0]) and
                    len(raw['responses']) == group_size, 'Calibration rollout differs from its frozen input')
            grades = [grader(row, r['text'], {}) for r in raw['responses']]
            gen = [len(r['token_ids']) for r in raw['responses']]
            rec = {'id': row['id'], 'task': row['ability'], 'accuracy': [g['accuracy'] for g in grades],
                   'mean_accuracy': sum(g['accuracy'] for g in grades) / len(grades), 'format': [g['format'] for g in grades],
                   'input_tokens': input_tokens(row), 'generated_tokens': gen,
                   'cost_tokens': group_size * input_tokens(row) + sum(gen)}
            with log.open('a') as f:
                f.write(json.dumps(rec, sort_keys=True) + '\n'); f.flush(); os.fsync(f.fileno())
            done[row['id']] = rec
    finally:
        if backend is not None:
            backend.close()
    by_task = {t: [d['mean_accuracy'] for d in done.values() if d['task'] == t] for t in ROUTES}
    stats = stats_from_accuracies(by_task)
    return {'rows': len(done), 'stats': stats, 'calibrated_arms_unblocked': not any(s['blocked'] for s in stats.values()),
            'cost_tokens': sum(d['cost_tokens'] for d in done.values()),
            'accuracy_support_levels': {t: sorted({round(x, 6) for x in by_task[t]}) for t in ROUTES}}
