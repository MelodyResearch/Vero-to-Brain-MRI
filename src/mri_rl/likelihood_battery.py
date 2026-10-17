"""Researcher-authorized, read-only battery exception; never a training gate."""
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import subprocess
import time
from mri_rl.paced_resources import decisions
from mri_rl.resource_supervisor import ResourcePolicy, sample_macos

BATTERY_POLICY = {'version':'read-only-likelihood-battery-v4-low-reserve-only', 'runner':'answer_likelihood_v1',
    'stop_percent':20, 'initial_sample_gap_seconds':31,
    'stage_max_swap_growth_gib':1,
    'rate_warning_min_seconds':30, 'rate_warning_min_observations':2,
    'warning_hold_rechecks':3,'warning_hold_seconds':15,'warning_recovery_samples':2,
    'power_watts':'UNAVAILABLE; battery-percent trend is a coarse proxy', 'weight_updates':0}

def cache_required_gib(profile,name,mlx_active_gib):
    """Phase-aware available-memory requirement from the measured cache footprint.

    initial load: packed cache + adapter + critical reserve; materialization: bytes still to be made resident
    (active-aware) + reserve; resident work: observed per-record inference increment + reserve."""
    reserve=profile['reserve_gib']
    if name.startswith(('before_runtime_import','model_load')):return profile['packed_gib']+reserve
    if name.startswith(('materialized:','quantized_model_ready','cache_parameters_verified')):
        return max(0.0,profile['packed_gib']-(mlx_active_gib or 0.0))+reserve
    return profile['resident_increment_gib']+reserve


def os_pressure_level():
    """kern.memorystatus_vm_pressure_level: 1 normal, 2 warn (logged only), 4 critical (stop); None if unavailable."""
    try:
        out=subprocess.run(['sysctl','-n','kern.memorystatus_vm_pressure_level'],capture_output=True,text=True,timeout=2,check=False).stdout.strip()
        return int(out) if out in ('1','2','4') else None
    except (OSError,subprocess.TimeoutExpired,ValueError):return None


class BatteryDiagnosticGate:
    def __init__(self,path,*,runner='answer_likelihood_v1',sampler=sample_macos,sleep=time.sleep,cache_profile=None):
        if runner != BATTERY_POLICY['runner']:raise ValueError('Read-only likelihood exception only')
        self.path=Path(path);self.sampler=sampler;self.sleep=sleep;self.cache_profile=cache_profile
        self.policy=ResourcePolicy();self.initial=None;self.history=[]
        self._rate_start=None;self._rate_count=0;self._rate_phase=None
        self._rate_only_episode=False
        self.resident_start=None   # captured exactly once, only after an ADMITTED quantized_model_ready (cache path)

    def _check(self,name,loading,starting,*,recovering=False,clean_samples=0):
        s=self.sampler();level,reasons=self.policy.observe(s)
        sole_rate=reasons==['sustained_swap_growth']
        # The generic supervisor counts observations, not elapsed time. Only its
        # provably rate-only WARNING episode may use this duration-based rule.
        if level=='NORMAL':self._rate_only_episode=sole_rate
        elif level=='CRITICAL' or any(r!='sustained_swap_growth' for r in reasons):
            self._rate_only_episode=False
        rate_only_level=level=='WARNING' and self._rate_only_episode
        # Materialization uses the unit memory gate intentionally; phase labeling
        # must not silently replace that with the stricter loading headroom gate.
        phase='loading' if loading or name.startswith('materialized:') or name.startswith('quantized_model_ready') else 'resident'
        if sole_rate:
            if self._rate_start is None or phase!=self._rate_phase:
                self._rate_start=s.monotonic_seconds;self._rate_count=0
            self._rate_count+=1
        else:self._rate_start=None;self._rate_count=0
        self._rate_phase=phase
        duration=0 if self._rate_start is None else s.monotonic_seconds-self._rate_start
        sustained=(self._rate_count>=BATTERY_POLICY['rate_warning_min_observations'] and
                   duration>=BATTERY_POLICY['rate_warning_min_seconds'])
        # The exception substitutes only the verified battery source for the AC check.
        known=s.power_source in ('AC','BATTERY')
        effective=replace(s,power_source='AC') if known else s
        action,reasons=decisions(effective,loading=loading,supervisor_reasons=reasons)
        # Cache path only (measured profile, not the generic 8/6 GiB heuristics): replace ONLY the memory-headroom reason.
        if self.cache_profile is not None:
            need=cache_required_gib(self.cache_profile,name,s.mlx_active_gib)
            reasons=[r for r in reasons if r!='insufficient_memory_headroom']
            if s.available_gib is None or s.available_gib<need:
                reasons.append('insufficient_memory_headroom')
            if action!='STOP':action='HOLD' if reasons else 'CONTINUE'
        os_level=os_pressure_level()
        if os_level==4:reasons.append('os_memory_pressure_critical')
        if s.battery_percent is None:reasons.append('battery_sensor_unknown')
        elif s.battery_percent<=BATTERY_POLICY['stop_percent']:reasons.append('battery_low_reserve_stop')
        if s.swap_used_gib is None:reasons.append('swap_sensor_unknown')
        cache_phase=None;total_growth=net_growth=None;limits=None
        if self.initial is not None and s.swap_used_gib is not None:
            total_growth=s.swap_used_gib-self.initial.swap_used_gib
        if self.cache_profile is None:
            # Generic (non-cache) path: the legacy 1 GiB stage cap, exactly as before.
            if total_growth is not None and total_growth>=BATTERY_POLICY['stage_max_swap_growth_gib']:
                reasons.append('stage_swap_growth')
        else:
            # Cache path: ONE phase machine; the original initial/history/ResourcePolicy are never reset.
            packed=self.cache_profile['packed_gib'];cap=BATTERY_POLICY['stage_max_swap_growth_gib']
            cache_phase='resident' if self.resident_start is not None else 'load'
            limits={'load_total_growth_lt':packed,'resident_net_growth_lt':cap,'resident_total_growth_lt':packed+cap}
            if total_growth is not None:
                if cache_phase=='load':
                    if total_growth>=packed:reasons.append('load_swap_growth_budget')
                else:
                    net_growth=s.swap_used_gib-self.resident_start.swap_used_gib
                    if net_growth>=cap:reasons.append('stage_swap_growth')
                    if total_growth>=packed+cap:reasons.append('total_swap_growth_budget')
        if level!='NORMAL' and not rate_only_level:reasons.append('supervisor_not_normal')
        if reasons:
            if reasons==['sustained_swap_growth']:
                # Cache load phase: a rate-only warning under otherwise healthy gates is logged and continues (no dwell/HOLD).
                if self.cache_profile is not None and cache_phase=='load' and not recovering:action='WARNING'
                else:action='HOLD' if recovering or sustained else 'WARNING'
            else:action='STOP'
        elif recovering and clean_samples<1:
            action='HOLD_RECOVERY'
        event={'unit_id':name,'action':action,'reasons':reasons,'supervisor_level':level,'os_pressure_level':os_level,'cache_profile':self.cache_profile,'cache_phase':cache_phase,
            'swap_growth_total_gib':total_growth,'swap_growth_resident_net_gib':net_growth,'swap_limits_gib':limits,
            'resident_start_swap_gib':None if self.resident_start is None else self.resident_start.swap_used_gib,
            'sample':asdict(s),'battery_exception':BATTERY_POLICY,
            'phase':phase,'rate_warning_duration_seconds':duration,
            'rate_warning_observations':self._rate_count,
            'rate_only_supervisor_warning':rate_only_level,
            'admitted':action in ('CONTINUE','WARNING')}
        with self.path.open('a') as f:
            f.write(json.dumps(event,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
        if self.initial is None and action!='STOP':self.initial=s
        if (self.cache_profile is not None and self.resident_start is None and name=='quantized_model_ready'
                and action in ('CONTINUE','WARNING')):self.resident_start=s   # load budget was checked above; never reset
        self.history.append(s)
        return s,action,reasons

    def _admit(self,name,loading,starting):
        sample,action,reasons=self._check(name,loading,starting)
        if action in ('CONTINUE','WARNING'):return sample
        if action=='STOP':raise RuntimeError('Battery diagnostic stopped at boundary: '+','.join(reasons))
        clean=0
        for attempt in range(BATTERY_POLICY['warning_hold_rechecks']):
            self.sleep(BATTERY_POLICY['warning_hold_seconds'])
            sample,action,reasons=self._check(name+':warning_hold:'+str(attempt+1),loading,starting,recovering=True,clean_samples=clean)
            if action=='STOP':raise RuntimeError('Battery diagnostic stopped during HOLD: '+','.join(reasons))
            clean=clean+1 if not reasons else 0
            if action=='CONTINUE':return sample
        with self.path.open('a') as f:
            f.write(json.dumps({'unit_id':name,'action':'STOP','reasons':['warning_hold_not_recovered'],
                'sample':asdict(sample),'battery_exception':BATTERY_POLICY},sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
        raise RuntimeError('Battery diagnostic stopped: warning_hold_not_recovered')

    def before_unit(self,name,loading=False):
        starting=self.initial is None
        sample=self._admit(name,loading,starting)
        if starting:
            self.sleep(BATTERY_POLICY['initial_sample_gap_seconds'])
            sample=self._admit(name+':second_preflight',loading,True)
            if sample.monotonic_seconds-self.initial.monotonic_seconds<30:
                raise RuntimeError('Insufficient preflight sampling interval')
        return sample
