"""Lightweight Git-anchored plan freeze / append-only log check; no approval override."""
import difflib
import hashlib
import json

PLAN='research/CURRENT_RESEARCH_PLAN.md'
LOCK='research/CURRENT_RESEARCH_PLAN.lock.json'
LOG='research/EXPERIMENT_LOG.md'
PATHS=(PLAN,LOCK,LOG)


def check_trees(base,candidate):
    errors=[];old=base.get(PLAN);new=candidate.get(PLAN)
    if not new:errors.append('PLAN_MISSING')
    if (old is None and any(k in base for k in (LOCK,LOG))) or (old is not None and any(k not in base for k in PATHS)):errors.append('INVALID_BASELINE')
    if old is not None:
        if new!=old:errors.append('PLAN_CHANGED')
        if candidate.get(LOCK)!=base.get(LOCK):errors.append('LOCK_CHANGED')
        if not candidate.get(LOG,b'').startswith(base.get(LOG,b'')) or LOG not in candidate:errors.append('LOG_NOT_APPEND_ONLY')
    if not candidate.get(LOG):errors.append('LOG_MISSING')
    try:
        lock=json.loads(candidate[LOCK])
        valid=(lock.get('schema')=='research-plan-lock-v1' and isinstance(lock.get('initial_researcher_instruction_sha256'),str)
               and len(lock['initial_researcher_instruction_sha256'])==64)
        if not valid:errors.append('INVALID_LOCK')
        if new is None or lock.get('plan_sha256')!=hashlib.sha256(new).hexdigest():errors.append('LOCK_DIGEST_MISMATCH')
    except (KeyError,ValueError,TypeError):errors.append('LOCK_MISSING_OR_INVALID')
    return {'status':'BLOCKED' if errors else ('INITIAL_FREEZE' if old is None else 'UNCHANGED_PLAN_APPEND_ONLY_LOG'),
        'errors':errors,'human_approval':'NOT_MACHINE_VERIFIED',
        'plan_sha256':hashlib.sha256(new).hexdigest() if new else None,
        'plan_diff':''.join(difflib.unified_diff((old or b'').decode('utf-8',errors='replace').splitlines(True),
            (new or b'').decode('utf-8',errors='replace').splitlines(True),fromfile='Git baseline plan',tofile='candidate plan')) if old!=new else '',
        'policy':'No approved=true, CLI flag or local JSON authorizes a plan change. Future change requires actual experiment evidence, independent review, proposed diff and explicit researcher approval through a separately reviewed change.',
        'initial_creation_note':'Initial freeze is requested by the current researcher instruction; its SHA is provenance, not machine proof of human identity.'}
