"""Non-native parent process: durable child exit/log/resource witness, never a grant.

Does not initialize MLX, consume permission, acquire the GPU lock, kill a process,
or advance phases. The child's exact entry authenticates its own authorization.
"""
from __future__ import annotations
import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess
import time
from mri_rl.shared_rollout_gspo_core_v1 import publish_once, replace_atomic


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def witness_child(argv, output, *, scope_sha256, sampler, lock_inspector):
    """Exactly one child; `argv` is fixed by the production CLI, synthetic in tests.

    os.wait4 supplies THIS child's peak RSS, not cumulative RUSAGE_CHILDREN.
    Deadlines/resources are enforced by the child at safe boundaries. There is
    no forced wall-time kill during an atomic native call and no automatic retry.
    """
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    started = time.time(); t0 = time.monotonic()
    publish_once(output/'launch_PRIVATE.json', {'scope_file_sha256':scope_sha256,
        'argv_sha256':hashlib.sha256(json.dumps(argv).encode()).hexdigest(), 'started_unix':started,
        'automatic_retry':False, 'automatic_next_stage':False})
    with (output/'stdout_PRIVATE.log').open('xb') as stdout, (output/'stderr_PRIVATE.log').open('xb') as stderr:
        child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr)
        publish_once(output/'child_pid_PRIVATE.json', {'pid':child.pid,'parent_pid':os.getpid()})
        pid, status, usage = os.wait4(child.pid, 0)
        child.returncode = os.waitstatus_to_exitcode(status)
        stdout.flush(); stderr.flush(); os.fsync(stdout.fileno()); os.fsync(stderr.fileno())
    unit = 1024**3 if platform.system()=='Darwin' else 1024**2
    result = {'schema':'safe-micro-external-exit-v1', 'scope_file_sha256':scope_sha256,
        'native_exit_code':child.returncode, 'child_pid':pid, 'child_reaped':True,
        'started_unix':started, 'finished_unix':time.time(), 'wall_seconds':time.monotonic()-t0,
        'stdout_sha256':digest(output/'stdout_PRIVATE.log'), 'stderr_sha256':digest(output/'stderr_PRIVATE.log'),
        'child_peak_RSS_GiB':usage.ru_maxrss/unit, 'peak_RSS_provenance':'os.wait4 exact child; platform-native units',
        'post_child_resource_sample':{'status':'PENDING'}, 'post_child_global_lock':{'status':'PENDING'},
        'post_child_inspection_complete':False, 'post_child_inspection_errors':{},
        'allocator_release_proof':'SEPARATE_CHILD_RECEIPT_OR_NOT_MEASURED',
        'parent_Adam_RNG_integrity':'NOT_ATTESTED_BY_EXIT_WITNESS',
        'automatic_retry':False, 'automatic_next_stage':False}
    publish_once(output/'actual_exit_PRIVATE.json', result)
    # The reaped exit and closed logs are durable BEFORE optional OS sensors.
    # Missing telemetry is unknown, never an absent lock or a release proof.
    for key, observe in (('post_child_resource_sample',sampler), ('post_child_global_lock',lock_inspector)):
        try:
            result[key]=observe()
        except Exception as error:
            result[key]={'status':'UNAVAILABLE','error_type':type(error).__name__}
            result['post_child_inspection_errors'][key]={'type':type(error).__name__,'message':str(error)}
    result['post_child_inspection_complete']=True
    replace_atomic(output/'actual_exit_PRIVATE.json', result)
    return result


def inspect_lock(path):
    """Read-only inspection; no flock/open-for-write/create/unlink."""
    result = subprocess.run(['/usr/sbin/lsof','-nP','-F','pfn',str(path)],capture_output=True,text=True,timeout=10)
    return {'exists':Path(path).is_file(), 'no_open_holders':result.returncode==1 and not result.stdout.strip() and not result.stderr.strip(),
            'inspection_error':result.stderr.strip() or None, 'lock_acquired_or_modified':False}


def main(argv=None):
    import argparse
    from dataclasses import asdict
    from mri_rl import accuracy_recovery_v1 as authority
    from mri_rl.resource_supervisor import sample_macos
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('spec','scope','human','review','cache-execution'):parser.add_argument('--'+name,type=Path,required=True)
    for name in ('scope','human','review','cache-execution'):parser.add_argument('--expected-'+name+'-sha256',required=True)
    args=parser.parse_args(argv)
    if digest(args.scope)!=args.expected_scope_sha256:raise ValueError('Exact scope file changed')
    scope=json.loads(args.scope.read_text())
    if scope.get('stage')!=1 or scope.get('phase')!='A' or scope.get('automatic_next_stage') is not False:
        raise ValueError('This launcher only runs standalone CACHE_ONLY stage1')
    root=authority.canonical_project_root();source=Path(__file__).resolve().parents[2]
    entry=source/'scripts/run_safe_micro_v1.py';launcher=source/'scripts/launch_safe_micro_v1.py'
    for p in (Path(__file__).resolve(),entry,launcher):
        if scope['source_sha256'].get(str(p))!=digest(p):raise ValueError('Unbound launcher/entry source')
    output=Path(scope['output']);allowed=root/'local_data/safe_micro_v1'
    if not output.resolve().is_relative_to(allowed.resolve()):raise ValueError('Private canonical output required')
    nested=json.loads(Path(args.spec).read_text())['cache_contract']['path']
    lock=Path(json.loads(Path(nested).read_text())['roots']['global_lock'])
    python=root/'work/accuracy-recovery-venv-v1/bin/python'
    child=[str(python),'-I',str(entry),'--execute']
    for name in ('spec','scope','human','review','cache_execution'):
        child.extend(['--'+name.replace('_','-'),str(getattr(args,name))])
    for name in ('scope','human','review','cache_execution'):
        child.extend(['--expected-'+name.replace('_','-')+'-sha256',getattr(args,'expected_'+name+'_sha256')])
    result=witness_child(child,output.with_name(output.name+'_external_witness_PRIVATE'),scope_sha256=args.expected_scope_sha256,
        sampler=lambda:asdict(sample_macos()),lock_inspector=lambda:inspect_lock(lock))
    print(json.dumps({'native_exit_code':result['native_exit_code'],'child_reaped':True,'automatic_next_stage':False},sort_keys=True))
    return result['native_exit_code']
