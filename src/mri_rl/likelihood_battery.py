"""Researcher-authorized, read-only battery exception; never a training gate."""
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import time
from mri_rl.paced_resources import decisions
from mri_rl.resource_supervisor import ResourcePolicy, sample_macos

BATTERY_POLICY = {'version':'read-only-likelihood-battery-v1', 'runner':'answer_likelihood_v1',
    'start_min_percent':75, 'stop_percent':65, 'initial_sample_gap_seconds':31,
    'rapid_drain_min_seconds':60, 'rapid_drain_min_points':2, 'rapid_drain_points_per_minute':1,
    'stage_max_drain_points':5, 'stage_max_swap_growth_gib':1,
    'power_watts':'UNAVAILABLE; battery-percent trend is a coarse proxy', 'weight_updates':0}

class BatteryDiagnosticGate:
    def __init__(self,path,*,runner='answer_likelihood_v1',sampler=sample_macos,sleep=time.sleep):
        if runner != BATTERY_POLICY['runner']:raise ValueError('Read-only likelihood exception only')
        self.path=Path(path);self.sampler=sampler;self.sleep=sleep
        self.policy=ResourcePolicy();self.initial=None;self.history=[]

    def _check(self,name,loading,starting):
        s=self.sampler();level,reasons=self.policy.observe(s)
        # The exception substitutes only the verified battery source for the AC check.
        known=s.power_source in ('AC','BATTERY')
        effective=replace(s,power_source='AC') if known else s
        action,reasons=decisions(effective,loading=loading,supervisor_reasons=reasons)
        if s.battery_percent is None:reasons.append('battery_sensor_unknown')
        elif starting and s.battery_percent<BATTERY_POLICY['start_min_percent']:reasons.append('battery_stage_start_below_75')
        elif s.battery_percent<=BATTERY_POLICY['stop_percent']:reasons.append('battery_stop_at_65')
        if s.swap_used_gib is None:reasons.append('swap_sensor_unknown')
        if self.initial is not None:
            if s.swap_used_gib is not None and s.swap_used_gib-self.initial.swap_used_gib>=BATTERY_POLICY['stage_max_swap_growth_gib']:
                reasons.append('stage_swap_growth')
            if s.battery_percent is not None and self.initial.battery_percent-s.battery_percent>=BATTERY_POLICY['stage_max_drain_points']:
                reasons.append('stage_battery_budget')
            for prior in self.history:
                dt=s.monotonic_seconds-prior.monotonic_seconds
                if s.battery_percent is not None and prior.battery_percent is not None:
                    drop=prior.battery_percent-s.battery_percent
                    if dt>=60 and drop>=2 and drop/(dt/60)>=1:
                        reasons.append('rapid_battery_drain');break
        if reasons:action='STOP'
        event={'unit_id':name,'action':action,'reasons':reasons,'supervisor_level':level,
            'sample':asdict(s),'battery_exception':BATTERY_POLICY}
        with self.path.open('a') as f:
            f.write(json.dumps(event,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
        if action!='CONTINUE':raise RuntimeError('Battery diagnostic stopped at boundary: '+','.join(reasons))
        if self.initial is None:self.initial=s
        self.history.append(s)
        return s

    def before_unit(self,name,loading=False):
        starting=self.initial is None
        sample=self._check(name,loading,starting)
        if starting:
            self.sleep(BATTERY_POLICY['initial_sample_gap_seconds'])
            sample=self._check(name+':second_preflight',loading,True)
            if sample.monotonic_seconds-self.initial.monotonic_seconds<30:
                raise RuntimeError('Insufficient preflight sampling interval')
        return sample
