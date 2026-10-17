"""Finite, source-bound deferred launcher: wait (no model, no LLM turns) until the EXISTING load-headroom
gate says CONTINUE, then run ONE already-approved launch script exactly once.

It never retries a failed launch, never touches other apps, and does not replace the launched script's own
gates (stage swap cap, pressure, thermal, concurrency). It only decides *when* to start the single attempt.
Reuses paced_resources.decisions and resource_supervisor.sample_macos. Aggregate-only stdout/status.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from mri_rl.paced_resources import decisions  # noqa: E402
from mri_rl.resource_supervisor import sample_macos  # noqa: E402


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wait_and_launch(script, expected_sha256, status_path, *, deadline_seconds, poll_seconds,
                    sampler=sample_macos, sleep=time.sleep, clock=time.monotonic, runner=subprocess.run):
    script, status_path = Path(script), Path(status_path)
    def write(state, **extra):
        tmp = status_path.with_suffix('.tmp')
        tmp.write_text(json.dumps({'state': state, 'script_sha256': expected_sha256, **extra}, sort_keys=True))
        os.replace(tmp, status_path)
    if status_path.exists():
        raise FileExistsError('Status exists: a launch was already attempted; inspect it, never auto-retry')
    if sha256(script) != expected_sha256:
        write('REFUSED_SCRIPT_HASH_MISMATCH'); return 'REFUSED_SCRIPT_HASH_MISMATCH'
    end = clock() + deadline_seconds; checks = 0; last = None
    while True:
        s = sampler(); checks += 1
        action, reasons = decisions(s, loading=True)
        last = {'available_gib': s.available_gib, 'pressure_free_percent': s.pressure_free_percent,
                'power_source': s.power_source, 'reasons': reasons}
        if action == 'STOP':
            write('STOPPED_SEVERE_PRESSURE', checks=checks, last=last); return 'STOPPED_SEVERE_PRESSURE'
        if action == 'CONTINUE':
            break
        if clock() >= end:
            write('DEADLINE_NOT_READY', checks=checks, last=last); return 'DEADLINE_NOT_READY'
        sleep(poll_seconds)
    if sha256(script) != expected_sha256:  # re-check immediately before the single launch
        write('REFUSED_SCRIPT_CHANGED', checks=checks, last=last); return 'REFUSED_SCRIPT_CHANGED'
    write('LAUNCHING', checks=checks, last=last)
    result = runner(['sh', str(script)], capture_output=True, text=True)
    state = 'LAUNCH_EXITED_0' if result.returncode == 0 else 'LAUNCH_EXITED_NONZERO_NO_RETRY'
    write(state, checks=checks, last=last, returncode=result.returncode)
    return state


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--script', type=Path, required=True)
    p.add_argument('--expected-sha256', required=True)
    p.add_argument('--status', type=Path, required=True)
    p.add_argument('--deadline-seconds', type=int, default=3600)
    p.add_argument('--poll-seconds', type=int, default=60)
    a = p.parse_args()
    print(wait_and_launch(a.script, a.expected_sha256, a.status, deadline_seconds=a.deadline_seconds,
                          poll_seconds=a.poll_seconds))


if __name__ == '__main__':
    main()
