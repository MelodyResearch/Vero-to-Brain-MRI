import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest

spec = importlib.util.spec_from_file_location('lwr', Path(__file__).parents[1] / 'scripts/launch_when_ready.py')
lwr = importlib.util.module_from_spec(spec); spec.loader.exec_module(lwr)


def sample(avail, pressure=50, power='AC', heavy=0):
    return SimpleNamespace(available_gib=avail, pressure_free_percent=pressure, power_source=power,
                           other_heavy_processes=heavy, thermal_state='NORMAL', mlx_active_gib=None)


def setup(tmp_path):
    script = tmp_path / 'go.sh'; script.write_text('exit 0\n')
    return script, hashlib.sha256(script.read_bytes()).hexdigest(), tmp_path / 'status.json'


class Runner:
    def __init__(self, code=0): self.calls, self.code = [], code
    def __call__(self, cmd, **k): self.calls.append(cmd); return SimpleNamespace(returncode=self.code)


def clock_factory():
    t = [0.0]
    def clock(): return t[0]
    def sleep(s): t[0] += s
    return clock, sleep


def test_waits_then_launches_once_when_headroom_recovers(tmp_path):
    script, sha, status = setup(tmp_path); clock, sleep = clock_factory(); r = Runner()
    seq = iter([sample(5.0), sample(6.0), sample(9.0)])
    out = lwr.wait_and_launch(script, sha, status, deadline_seconds=600, poll_seconds=60,
                              sampler=lambda: next(seq), sleep=sleep, clock=clock, runner=r)
    assert out == 'LAUNCH_EXITED_0' and len(r.calls) == 1


def test_deadline_without_headroom_never_launches(tmp_path):
    script, sha, status = setup(tmp_path); clock, sleep = clock_factory(); r = Runner()
    out = lwr.wait_and_launch(script, sha, status, deadline_seconds=180, poll_seconds=60,
                              sampler=lambda: sample(5.9), sleep=sleep, clock=clock, runner=r)
    assert out == 'DEADLINE_NOT_READY' and not r.calls


def test_hash_mismatch_severe_pressure_and_failed_launch_do_not_retry(tmp_path):
    script, sha, status = setup(tmp_path); clock, sleep = clock_factory(); r = Runner()
    assert lwr.wait_and_launch(script, 'bad', status, deadline_seconds=1, poll_seconds=1,
        sampler=lambda: sample(20), sleep=sleep, clock=clock, runner=r) == 'REFUSED_SCRIPT_HASH_MISMATCH'
    status.unlink()
    assert lwr.wait_and_launch(script, sha, status, deadline_seconds=100, poll_seconds=1,
        sampler=lambda: sample(1.0), sleep=sleep, clock=clock, runner=r) == 'STOPPED_SEVERE_PRESSURE'
    status.unlink(); r2 = Runner(code=1)
    assert lwr.wait_and_launch(script, sha, status, deadline_seconds=100, poll_seconds=1,
        sampler=lambda: sample(20), sleep=sleep, clock=clock, runner=r2) == 'LAUNCH_EXITED_NONZERO_NO_RETRY'
    with pytest.raises(FileExistsError):
        lwr.wait_and_launch(script, sha, status, deadline_seconds=100, poll_seconds=1,
            sampler=lambda: sample(20), sleep=sleep, clock=clock, runner=r2)
    assert len(r2.calls) == 1 and not r.calls


def test_script_changed_after_wait_is_refused(tmp_path):
    script, sha, status = setup(tmp_path); clock, sleep = clock_factory(); r = Runner()
    calls = []
    def sampler():
        calls.append(1)
        if len(calls) == 2: script.write_text('rm -rf /\n')
        return sample(5.0) if len(calls) == 1 else sample(20)
    out = lwr.wait_and_launch(script, sha, status, deadline_seconds=600, poll_seconds=1,
                              sampler=sampler, sleep=sleep, clock=clock, runner=r)
    assert out == 'REFUSED_SCRIPT_CHANGED' and not r.calls
