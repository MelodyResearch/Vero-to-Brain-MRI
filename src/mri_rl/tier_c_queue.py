"""Finite, sequential, resumable Tier C stage queue (Claude-authored; new file; no new resource framework).

Order: raw RL arms -> common-B28 calibration -> calibrated RL arms + same-data SFT -> both-task dev/test evaluations.
One heavy stage at a time. A stage runs ONLY with an independently supplied approval entry (path + sha256 that matches the
file); the queue never approves, never retries a failed stage, never reorders, and records every status durably. Calibrated
stages are BLOCKED (not failed) when calibration reports a zero IQR. Runners are injected (native work lives elsewhere).
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path

SEEDS = (7, 11, 13)
RAW_ARMS = ('S_sequence', 'S_correspondence', 'P_raw', 'U_raw')
CAL_ARMS = ('P_calibrated', 'U_calibrated')


def plan_stages(seeds=SEEDS):
    st = [{'id': f'rl:{a}:{s}', 'kind': 'rl', 'arm': a, 'seed': s} for s in seeds for a in RAW_ARMS]
    st.append({'id': 'calibration', 'kind': 'calibration'})
    st += [{'id': f'rl:{a}:{s}', 'kind': 'rl', 'arm': a, 'seed': s, 'needs_calibration': True} for s in seeds for a in CAL_ARMS]
    st += [{'id': f'sft:{s}', 'kind': 'sft', 'seed': s} for s in seeds]
    st.append({'id': 'eval:B28', 'kind': 'eval', 'checkpoint': 'B28'})
    st += [{'id': f'eval:{x["id"]}', 'kind': 'eval', 'checkpoint': x['id']} for x in list(st) if x['kind'] in ('rl', 'sft')]
    return st


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _save(path, status):
    tmp = Path(str(path) + '.tmp')
    with tmp.open('w') as f:
        json.dump(status, f, indent=1, sort_keys=True); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def run_queue(stages, status_path, approvals, runners, *, calibration_unblocked=None):
    """Execute pending stages in order; returns the status dict. Stops at the first failure or missing approval."""
    status_path = Path(status_path)
    status = json.loads(status_path.read_text()) if status_path.exists() else {'stages': {}}
    for st in stages:
        rec = status['stages'].get(st['id'], {})
        if rec.get('state') in ('DONE', 'BLOCKED'):
            continue
        if rec.get('state') == 'FAILED':
            status['halted'] = f"{st['id']} FAILED earlier; preserved, no automatic retry"; break
        if st.get('needs_calibration'):
            cal = status['stages'].get('calibration', {})
            unblocked = cal.get('summary', {}).get('calibrated_arms_unblocked') if calibration_unblocked is None else calibration_unblocked
            if cal.get('state') != 'DONE' or unblocked is False:
                status['stages'][st['id']] = {'state': 'BLOCKED', 'reason': 'calibration missing or zero-IQR'}
                _save(status_path, status); continue
        if st['kind'] == 'eval' and status['stages'].get(st.get('checkpoint'), {}).get('state') == 'BLOCKED':
            status['stages'][st['id']] = {'state': 'BLOCKED', 'reason': 'trained checkpoint stage is blocked'}
            _save(status_path, status); continue
        appr = approvals.get(st['id'])
        if not appr or not Path(appr['path']).is_file() or _sha(appr['path']) != appr['sha256']:
            status['stages'][st['id']] = {'state': 'NOT_APPROVED'}
            status['halted'] = f"{st['id']} has no matching independent approval"; _save(status_path, status); break
        status['stages'][st['id']] = {'state': 'RUNNING', 'approval_sha256': appr['sha256']}; _save(status_path, status)
        try:
            summary = runners[st['kind']](st, appr)
            status['stages'][st['id']] = {'state': 'DONE', 'approval_sha256': appr['sha256'], 'summary': summary}
        except Exception as exc:                        # preserve, never retry blindly
            status['stages'][st['id']] = {'state': 'FAILED', 'approval_sha256': appr['sha256'], 'error': type(exc).__name__ + ': ' + str(exc)[:300]}
            status['halted'] = f"{st['id']} FAILED"; _save(status_path, status); break
        _save(status_path, status)
    else:
        status.pop('halted', None)
    _save(status_path, status)
    return status
