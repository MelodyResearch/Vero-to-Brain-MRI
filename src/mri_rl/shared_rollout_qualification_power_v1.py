"""Human-authorized battery exception for ONE readonly qualification, never training.

Actual battery telemetry is preserved. Only the AC requirement is excepted; the
existing low-battery warning threshold (10%) becomes an immediate safe-unit stop.
Global PacedGate and ResourcePolicy remain unchanged.
"""
from dataclasses import asdict
import json
import os
from mri_rl.paced_resources import PacedGate,PACED_POLICY,decisions as original


def decisions(sample,*,loading,supervisor_reasons=()):
    action,reasons=original(sample,loading=loading,supervisor_reasons=supervisor_reasons)
    if sample.power_source=='BATTERY':
        if sample.battery_percent is None or sample.battery_percent<=10:
            return 'STOP',reasons+['qualification_battery_low_or_unknown']
        reasons=[r for r in reasons if r!='power_not_verified_AC']
        if action!='STOP':action='HOLD' if reasons else 'CONTINUE'
    return action,reasons


class QualificationBatteryGate(PacedGate):
    def before_unit(self,name,loading=False):
        for _ in range(PACED_POLICY['wait_rechecks']):
            sample=self.sampler();level,reasons=self.policy.observe(sample)
            action,reasons=decisions(sample,loading=loading,supervisor_reasons=reasons)
            event={'unit_id':name,'action':action,'supervisor_level':level,'reasons':reasons,
                   'sample':asdict(sample),'power_exception':'ONE_READONLY_SHARED_ROLLOUT_QUALIFICATION'}
            with self.path.open('a') as f:
                f.write(json.dumps(event,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
            if action=='CONTINUE':return sample
            if action=='STOP':raise RuntimeError('Qualification stopped at safe unit: '+','.join(reasons))
            self.sleep(PACED_POLICY['wait_seconds'])
        raise RuntimeError('Resource hold persists; no automatic restart')
