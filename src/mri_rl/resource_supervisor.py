"""Lightweight, cooperative macOS resource gate for sequential MLX jobs.

The gate never sends signals to workers or changes a scientific configuration.
Call it only between completed inference examples or training groups. Optional
signals are explicitly unavailable when macOS does not expose them.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class ResourceSample:
    monotonic_seconds: float
    available_gib: float | None = None
    pressure_free_percent: float | None = None
    swap_used_gib: float | None = None
    process_rss_gib: float | None = None
    thermal_state: str = "SENSOR_UNAVAILABLE"
    power_source: str = "SENSOR_UNAVAILABLE"
    battery_percent: int | None = None
    other_heavy_processes: int | None = None
    mlx_active_gib: float | None = None
    mlx_peak_gib: float | None = None


def _run(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=2,
                                check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout if result.returncode == 0 else None


def _memory_pressure(text: str | None) -> float | None:
    match = re.search(r"System-wide (?:memory )?free percentage:\s*(\d+(?:\.\d+)?)%", text or "")
    return float(match.group(1)) if match else None


def _vm_available_gib(text: str | None) -> float | None:
    if not text:
        return None
    page = re.search(r"page size of (\d+) bytes", text)
    if not page:
        return None
    counts = {}
    for name in ("Pages free", "Pages inactive", "Pages speculative"):
        match = re.search(rf"^{name}:\s*([\d.]+)\.?", text, re.MULTILINE)
        if match:
            counts[name] = int(float(match.group(1)))
    if not counts:
        return None
    return sum(counts.values()) * int(page.group(1)) / 1024**3


def _swap_used_gib(text: str | None) -> float | None:
    match = re.search(r"used\s*=\s*([\d.]+)([KMG])", text or "")
    if not match:
        return None
    return float(match.group(1)) * {"K": 1 / 1024**2, "M": 1 / 1024, "G": 1}[match.group(2)]


def _power(text: str | None) -> tuple[str, int | None]:
    if not text:
        return "SENSOR_UNAVAILABLE", None
    if "discharging" in text.lower():
        source = "BATTERY"
    elif "Now drawing from 'AC Power'" in text:
        source = "AC"
    elif "Now drawing from 'Battery Power'" in text:
        source = "BATTERY"
    else:
        source = "SENSOR_UNAVAILABLE"
    match = re.search(r"(\d+)%", text)
    return source, int(match.group(1)) if match else None


def _thermal(text: str | None) -> str:
    if not text:
        return "SENSOR_UNAVAILABLE"
    # pmset exposes limits on some Macs; a temperature is deliberately not inferred.
    speeds = re.findall(r"(?:CPU_Speed_Limit|GPU_Speed_Limit)\s*=\s*(\d+)", text)
    if speeds:
        return "THROTTLED" if min(map(int, speeds)) < 80 else "NORMAL"
    return "SENSOR_UNAVAILABLE"


def sample_macos(pid: int | None = None, *, now: Callable[[], float] = time.monotonic,
                 command: Callable[[list[str]], str | None] = _run) -> ResourceSample:
    """Read documented command output; missing permissions/sensors stay unknown."""
    pid = pid or os.getpid()
    pressure = _memory_pressure(command(["memory_pressure", "-Q"]))
    available = _vm_available_gib(command(["vm_stat"]))
    swap = _swap_used_gib(command(["sysctl", "vm.swapusage"]))
    power, battery = _power(command(["pmset", "-g", "batt"]))
    thermal = _thermal(command(["pmset", "-g", "therm"]))
    process_text = command(["ps", "-o", "rss=", "-p", str(pid)])
    try:
        rss = int((process_text or "").strip()) / 1024**2
    except ValueError:
        rss = None
    # Process inventory is advisory; never terminate another job automatically.
    inventory = command(["ps", "-axo", "pid=,rss=,command="])
    heavy = None
    if inventory is not None:
        heavy = 0
        for line in inventory.splitlines():
            fields = line.strip().split(maxsplit=2)
            if len(fields) != 3:
                continue
            try:
                other_pid, other_rss = int(fields[0]), int(fields[1])
            except ValueError:
                continue
            if other_pid != pid and other_rss >= 2 * 1024**2 and any(
                marker in fields[2].lower() for marker in ("mlx", "frozen_eval", "gspo", "qwen")
            ):
                heavy += 1
    mlx_active = mlx_peak = None
    # Read an existing MLX runtime only. Monitoring must not initialize Metal.
    mx = sys.modules.get("mlx.core")
    metal = getattr(mx, "metal", None) if mx is not None else None
    if metal is not None:
        try:
            mlx_active = metal.get_active_memory() / 1024**3
            mlx_peak = metal.get_peak_memory() / 1024**3
        except (AttributeError, RuntimeError, OSError):
            pass
    return ResourceSample(now(), available, pressure, swap, rss, thermal,
                          power, battery, heavy, mlx_active, mlx_peak)


class ResourcePolicy:
    """Sustained warning/critical detection with recovery hysteresis."""

    def __init__(self, warning_samples: int = 3, critical_samples: int = 2,
                 recovery_samples: int = 3):
        self.level = "NORMAL"
        self.warning_samples = warning_samples
        self.critical_samples = critical_samples
        self.recovery_samples = recovery_samples
        self._warning = self._critical = self._recovery = 0
        self._last: ResourceSample | None = None

    def observe(self, sample: ResourceSample) -> tuple[str, list[str]]:
        swap_rate = None
        if (self._last and sample.swap_used_gib is not None and
                self._last.swap_used_gib is not None):
            minutes = (sample.monotonic_seconds - self._last.monotonic_seconds) / 60
            if minutes > 0:
                swap_rate = (sample.swap_used_gib - self._last.swap_used_gib) / minutes
        self._last = sample
        reasons = []
        if sample.pressure_free_percent is not None and sample.pressure_free_percent <= 15:
            reasons.append("low_memory_pressure_free_percent")
        if (sample.available_gib is not None and sample.available_gib < 8 and
                sample.pressure_free_percent is not None and sample.pressure_free_percent < 30):
            reasons.append("low_available_memory_with_pressure")
        # macOS can churn compressed/swap pages during a healthy model load.
        # Require memory pressure corroboration before throttling on a rate.
        swap_warning = (swap_rate is not None and swap_rate > 0.20 and
                        ((sample.pressure_free_percent is not None and
                          sample.pressure_free_percent < 40) or
                         (sample.available_gib is not None and sample.available_gib < 10)))
        swap_critical = (swap_rate is not None and swap_rate > 0.75 and
                         ((sample.pressure_free_percent is not None and
                           sample.pressure_free_percent < 15) or
                          (sample.available_gib is not None and sample.available_gib < 4)))
        if swap_warning:
            reasons.append("sustained_swap_growth")
        if sample.thermal_state == "THROTTLED":
            reasons.append("thermal_throttling")
        if sample.power_source == "BATTERY" and sample.battery_percent is not None and sample.battery_percent <= 10:
            reasons.append("battery_low")
        critical = (
            sample.pressure_free_percent is not None and sample.pressure_free_percent <= 3
        ) or (
            sample.available_gib is not None and sample.available_gib < 2 and
            sample.pressure_free_percent is not None and sample.pressure_free_percent <= 8
        ) or swap_critical or (
            sample.power_source == "BATTERY" and sample.battery_percent is not None and sample.battery_percent <= 3
        )
        emergency = (sample.pressure_free_percent is not None and
                     sample.pressure_free_percent <= 1 and
                     sample.available_gib is not None and sample.available_gib < 1)
        if critical:
            reasons.append("critical_resource_signal")
        self._critical = self._critical + 1 if critical else 0
        self._warning = self._warning + 1 if reasons else 0
        self._recovery = self._recovery + 1 if not reasons else 0
        if emergency or self._critical >= self.critical_samples:
            self.level = "CRITICAL"
        elif self._warning >= self.warning_samples and self.level == "NORMAL":
            self.level = "WARNING"
        elif self._recovery >= self.recovery_samples:
            self.level = "NORMAL"
        return self.level, reasons


class ResourceGate:
    """Pause only before a new safe unit; log every sample and state transition."""

    def __init__(self, log_path: Path, *, sampler: Callable[[], ResourceSample] = sample_macos,
                 policy: ResourcePolicy | None = None, sleep: Callable[[float], None] = time.sleep,
                 interval_seconds: float = 15, warning_cool_seconds: float = 5):
        self.log_path = log_path
        self.sampler = sampler
        self.policy = policy or ResourcePolicy()
        self.sleep = sleep
        self.interval_seconds = interval_seconds
        self.warning_cool_seconds = warning_cool_seconds
        log_path.parent.mkdir(parents=True, exist_ok=True)

    def before_unit(self, unit_id: str) -> str:
        """Return NORMAL/WARNING; wait for recovery from CRITICAL."""
        while True:
            sample = self.sampler()
            previous = self.policy.level
            level, reasons = self.policy.observe(sample)
            event = {"unit_id": unit_id, "level": level, "previous_level": previous,
                     "reasons": reasons, "sample": asdict(sample)}
            with self.log_path.open("a") as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            if level != "CRITICAL":
                if level == "WARNING" and reasons and self.warning_cool_seconds > 0:
                    self.sleep(self.warning_cool_seconds)
                return level
            self.sleep(self.interval_seconds)
