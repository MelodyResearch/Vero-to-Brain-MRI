"""Byte-preserving adoption of independently reviewed pilot evidence.

No identity rewriting: only an exact immutable prefix may retain its historical
execution identity. Scoring/backend/model/input bindings must remain identical.
The new exact review scope binds this bridge and its digest.
"""
import ast
import hashlib
import json
from pathlib import Path
from mri_rl.frozen_eval import sha256
from mri_rl.experiment_contract import canonical_hash

EXECUTION_ONLY={'paced_answer_likelihood','likelihood_battery','likelihood_resident','likelihood_adoption'}
PROFILE=('contract_sha256','checkpoint','adapter','backend_sources','temperature','max_generation_tokens',
         'lora_rank','lora_alpha','quantization_bits','group_size','weight_updates')


def numerical_loop_sha256():
    source=Path(__file__).with_name('paced_answer_likelihood.py').read_text()
    loops=[n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='record']
    if len(loops)!=1:raise ValueError('Ambiguous numerical record loop')
    return hashlib.sha256(ast.dump(loops[0],include_attributes=False).encode()).hexdigest()


def verify_adoption(path,current,*,private_root=None):
    root=Path(private_root or Path(__file__).resolve().parents[2]/'local_data').resolve()
    path=Path(path).resolve()
    if not path.is_relative_to(root):raise ValueError('Adoption must remain private')
    digest=sha256(path);d=json.loads(path.read_text())
    if d.get('kind')!='likelihood_immutable_prefix_v1':raise ValueError('Unknown adoption contract')
    prefix=Path(d['prefix_path']).resolve();approval=Path(d['approval_path']).resolve();audit=Path(d['audit_path']).resolve()
    if not all(p.is_relative_to(root) for p in (prefix,approval,audit)):raise ValueError('Adoption source must remain private')
    if sha256(prefix)!=d['prefix_sha256'] or sha256(approval)!=d['approval_sha256']:raise ValueError('Adoption source digest mismatch')
    if sha256(audit)!=d['audit_sha256'] or d['numeric_loop_sha256']!=numerical_loop_sha256():raise ValueError('Adopted numerical loop/audit changed')
    reviewed=json.loads(audit.read_text())
    if reviewed.get('verdict') not in ('PASS','PASS_WITH_LIMITATIONS') or reviewed.get('output_sha256')!=d['prefix_sha256'] or reviewed.get('approval_sha256')!=d['approval_sha256']:
        raise ValueError('Pilot actual result lacks exact independent verification')
    review=json.loads(approval.read_text());old=review['scope']
    if review.get('kind')!='answer_likelihood_v1' or review.get('verdict') not in ('PASS','PASS_WITH_LIMITATIONS') or not review.get('reviewer'):
        raise ValueError('Original prefix lacks independent approval')
    if any(old.get(k)!=current.get(k) for k in PROFILE):raise ValueError('Adopted numerical/model/input profile changed')
    oldnumeric={k:v for k,v in old['source_hashes'].items() if k not in EXECUTION_ONLY}
    newnumeric={k:v for k,v in current['source_hashes'].items() if k not in EXECUTION_ONLY}
    if oldnumeric!=newnumeric:raise ValueError('Adopted numeric source changed')
    identity=canonical_hash({'contract':old['contract_sha256'],'checkpoint':old['checkpoint'],
        'sources':old['source_hashes'],'backend':old['backend_sources'],'battery_diagnostic':old.get('battery_exception') is not None})
    if identity!=old['run_identity']:raise ValueError('Original identity cannot be reconstructed')
    raw=prefix.read_bytes()
    if not raw or not raw.endswith(b'\n'):raise ValueError('Torn/empty adopted prefix')
    rows=[json.loads(line) for line in raw.splitlines()]
    if type(d['records']) is not int or not 1<=d['records']<=57 or len(rows)!=d['records'] or any(r.get('run_identity')!=identity for r in rows):
        raise ValueError('Invalid adopted prefix identities/count')
    if sha256(audit)!=d['audit_sha256'] or sha256(path)!=digest or sha256(prefix)!=d['prefix_sha256'] or sha256(approval)!=d['approval_sha256']:
        raise ValueError('Adoption evidence changed during validation')
    return {'bytes':raw,'identities':[identity]*len(rows),'binding':{'path':str(path),'sha256':digest,'records':len(rows),
        'prefix_sha256':d['prefix_sha256'],'approval_sha256':d['approval_sha256']}}
