import json
from pathlib import Path
import hashlib
from mri_rl import tier_c_queue as q


def approvals(tmp_path, stages, skip=()):
    out = {}
    for s in stages:
        if s['id'] in skip: continue
        p = tmp_path / (s['id'].replace(':', '_') + '.json'); p.write_text(json.dumps({'for': s['id']}))
        out[s['id']] = {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
    return out


def test_plan_has_exact_order_counts_and_dependencies():
    st = q.plan_stages(); ids = [s['id'] for s in st]
    assert sum(s['kind'] == 'rl' for s in st) == 18 and sum(s['kind'] == 'sft' for s in st) == 3
    assert ids.index('calibration') > max(i for i, s in enumerate(st) if s['kind'] == 'rl' and not s.get('needs_calibration'))
    assert all(s.get('needs_calibration') for s in st if s.get('arm') in q.CAL_ARMS)
    assert ids[-1].startswith('eval:') and ids.index('eval:B28') > ids.index('sft:13')
    assert len([s for s in st if s['kind'] == 'eval']) == 1 + 18 + 3


def test_runs_in_order_with_durable_status_and_resumes_without_rerunning(tmp_path):
    st = q.plan_stages((7,)); ran = []
    runners = {k: (lambda s, a, k=k: ran.append(s['id']) or {'calibrated_arms_unblocked': True} if k == 'calibration' else ran.append(s['id']) or {}) for k in ('rl', 'calibration', 'sft', 'eval')}
    status = q.run_queue(st, tmp_path / 's.json', approvals(tmp_path, st), runners)
    assert ran == [s['id'] for s in st] and all(v['state'] == 'DONE' for v in status['stages'].values()) and 'halted' not in status
    ran.clear(); q.run_queue(st, tmp_path / 's.json', approvals(tmp_path, st), runners)
    assert ran == []


def test_missing_or_mismatched_approval_halts_and_failure_is_never_retried(tmp_path):
    st = q.plan_stages((7,)); ran = []
    runners = {k: (lambda s, a: ran.append(s['id']) or {}) for k in ('rl', 'calibration', 'sft', 'eval')}
    status = q.run_queue(st, tmp_path / 'a.json', approvals(tmp_path, st, skip={'rl:P_raw:7'}), runners)
    assert ran == ['rl:S_sequence:7', 'rl:S_correspondence:7'] and status['stages']['rl:P_raw:7']['state'] == 'NOT_APPROVED'
    ap = approvals(tmp_path, st); Path(ap['rl:P_raw:7']['path']).write_text('tampered')           # hash mismatch
    assert q.run_queue(st, tmp_path / 'a.json', ap, runners)['stages']['rl:P_raw:7']['state'] == 'NOT_APPROVED'
    def boom(s, a): raise RuntimeError('native failure')
    ap = approvals(tmp_path, st); r2 = {**runners, 'rl': boom}
    status = q.run_queue(st, tmp_path / 'f.json', ap, r2)
    assert status['stages']['rl:S_sequence:7']['state'] == 'FAILED' and 'FAILED' in status['halted']
    calls = []
    status = q.run_queue(st, tmp_path / 'f.json', ap, {**runners, 'rl': lambda s, a: calls.append(s['id']) or {}})
    assert calls == [] and 'no automatic retry' in status['halted']


def test_zero_iqr_calibration_blocks_calibrated_stages_but_not_the_rest(tmp_path):
    st = q.plan_stages((7,)); ran = []
    runners = {'rl': lambda s, a: ran.append(s['id']) or {}, 'sft': lambda s, a: ran.append(s['id']) or {},
               'eval': lambda s, a: ran.append(s['id']) or {}, 'calibration': lambda s, a: {'calibrated_arms_unblocked': False}}
    status = q.run_queue(st, tmp_path / 'b.json', approvals(tmp_path, st), runners)
    assert status['stages']['rl:P_calibrated:7']['state'] == 'BLOCKED' and status['stages']['rl:U_calibrated:7']['state'] == 'BLOCKED'
    assert 'rl:P_calibrated:7' not in ran and 'sft:7' in ran and any(i.startswith('eval:') for i in ran)
    assert status['stages']['eval:rl:P_calibrated:7']['state'] == 'BLOCKED' and 'eval:rl:P_calibrated:7' not in ran
