import json

from mri_rl.resource_supervisor import (
    ResourceGate, ResourcePolicy, ResourceSample, _memory_pressure, _power,
    _swap_used_gib, _thermal, _vm_available_gib, sample_macos,
)
from mri_rl.mlx_sft_warmstart import _save_step_checkpoint


def sample(t, free=70, available=12, swap=30, thermal="SENSOR_UNAVAILABLE",
           power="AC", battery=50):
    return ResourceSample(t, available, free, swap, 16, thermal, power, battery)


def test_macos_parsers_and_unavailable_sensors():
    assert _memory_pressure("System-wide free percentage: 79%") == 79
    assert _memory_pressure("System-wide memory free percentage: 59%") == 59
    assert _vm_available_gib("Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
                             "Pages free: 100000.\nPages inactive: 200000.\n"
                             "Pages speculative: 100000.") == 400000 * 16384 / 1024**3
    assert _swap_used_gib("vm.swapusage: total = 32.00G  used = 30569.00M  free = 1.00G") == 30569 / 1024
    assert _power("Now drawing from 'AC Power'\n -InternalBattery-0 98%; discharging") == ("AC", 98)
    assert _thermal("CPU_Speed_Limit = 75") == "THROTTLED"
    assert _thermal("No thermal warning level has been recorded") == "SENSOR_UNAVAILABLE"


def test_large_but_falling_swap_is_normal():
    policy = ResourcePolicy()
    for t, swap in enumerate((31, 30.9, 30.8, 30.7)):
        level, reasons = policy.observe(sample(t * 60, swap=swap, free=79))
        assert level == "NORMAL"
        assert not reasons


def test_warning_critical_and_hysteretic_recovery():
    policy = ResourcePolicy()
    for t in range(3):
        level, _ = policy.observe(sample(t * 60, free=12, available=5))
    assert level == "WARNING"
    policy.observe(sample(180, free=2, available=1))
    assert policy.observe(sample(240, free=2, available=1))[0] == "CRITICAL"
    assert policy.observe(sample(300))[0] == "CRITICAL"
    assert policy.observe(sample(360))[0] == "CRITICAL"
    assert policy.observe(sample(420))[0] == "NORMAL"


def test_sustained_swap_growth_without_absolute_swap_trigger():
    policy = ResourcePolicy()
    for t, swap in enumerate((31, 31.3, 31.6, 31.9)):
        level, _ = policy.observe(sample(t * 60, swap=swap, free=25, available=7))
    assert level == "WARNING"


def test_swap_churn_with_healthy_pressure_does_not_pause_inference():
    policy = ResourcePolicy()
    for t, swap in enumerate((28, 32, 32.5, 33)):
        level, reasons = policy.observe(sample(t * 60, swap=swap, free=60, available=18))
        assert level == "NORMAL"
        assert "critical_resource_signal" not in reasons


def test_imminent_oom_blocks_even_first_baseline_sample():
    policy = ResourcePolicy()
    assert policy.observe(sample(0, free=0.5, available=0.5))[0] == "CRITICAL"


def test_sampling_survives_missing_optional_tools():
    def command(argv):
        return None
    got = sample_macos(pid=123, now=lambda: 7, command=command)
    assert got.thermal_state == "SENSOR_UNAVAILABLE"
    assert got.power_source == "SENSOR_UNAVAILABLE"
    assert got.available_gib is None
    assert got.other_heavy_processes is None


def test_gate_pauses_before_new_unit_and_logs_recovery(tmp_path):
    sequence = iter([sample(0, free=2, available=1),
                     sample(60, free=2, available=1),
                     sample(120), sample(180), sample(240)])
    slept = []
    gate = ResourceGate(tmp_path / "events.jsonl", sampler=lambda: next(sequence),
                        sleep=slept.append, interval_seconds=1)
    assert gate.before_unit("record-0") == "NORMAL"
    assert gate.before_unit("record-1") == "NORMAL"
    assert slept == [1, 1, 1]
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [event["level"] for event in events] == ["NORMAL", "CRITICAL", "CRITICAL", "CRITICAL", "NORMAL"]


def test_step_checkpoint_is_durable_and_explicitly_not_exact_resume(tmp_path):
    def fake_save(_model, path):
        path.write_bytes(b"adapter-step-1")
    receipt = _save_step_checkpoint(fake_save, object(), tmp_path, 1,
                                    {"model_config_sha256": "pinned"})
    assert receipt["completed_optimizer_steps"] == 1
    assert receipt["exact_resume_supported"] is False
    assert receipt["optimizer_state_saved"] is False
    assert (tmp_path / "checkpoints" / "step_001_adapters.safetensors").read_bytes() == b"adapter-step-1"
    assert not list((tmp_path / "checkpoints").glob("*.partial*"))
