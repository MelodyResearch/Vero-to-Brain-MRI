"""Conservative boundaries for the explicitly reauthorized single-process SFT.

No promise against OS failure. Holds occur only after evaluated computations.
"""
from dataclasses import asdict
import json
import os
from pathlib import Path
import time
from mri_rl.resource_supervisor import sample_macos, ResourcePolicy

PACED_POLICY = {'version':'paced-local-sft-v1','stage_end_steps':[1,2,*range(4,29,2)],
    'example_pause_seconds':2,'initial_stage_pause_seconds':30,'stage_pause_seconds':15,
    'memory_guideline_gib':18,'cache_limit_mib':256,'load_available_gib':8,
    'unit_available_gib':6,'load_pressure_free_percent':35,'unit_pressure_free_percent':30,
    'wait_rechecks':20,'wait_seconds':15,'gradient_checkpointing':'language_layers',
    'loading':'lazy_base_then_materialize_restored_quantization'}


def decisions(sample, *, loading, supervisor_reasons=()):
    reasons=list(supervisor_reasons)
    if (sample.pressure_free_percent is not None and sample.pressure_free_percent<=3) or (
        sample.available_gib is not None and sample.available_gib<2):
        return 'STOP',reasons+['severe_resource_pressure']
    if sample.power_source!='AC':reasons.append('power_not_verified_AC')
    if sample.other_heavy_processes is None or sample.other_heavy_processes:
        reasons.append('concurrent_process_inventory_unknown_or_occupied')
    minimum=PACED_POLICY['load_available_gib' if loading else 'unit_available_gib']
    pressure=PACED_POLICY['load_pressure_free_percent' if loading else 'unit_pressure_free_percent']
    if sample.available_gib is None or sample.available_gib<minimum:reasons.append('insufficient_memory_headroom')
    if sample.pressure_free_percent is None or sample.pressure_free_percent<pressure:reasons.append('memory_pressure')
    if sample.thermal_state=='THROTTLED':reasons.append('thermal_throttling')
    if sample.mlx_active_gib is not None and sample.mlx_active_gib>18:reasons.append('active_mlx_memory_above_guideline')
    return ('HOLD' if reasons else 'CONTINUE'),reasons


class PacedGate:
    def __init__(self,path,*,sampler=sample_macos,sleep=time.sleep):
        self.path=Path(path);self.sampler=sampler;self.sleep=sleep;self.policy=ResourcePolicy()
    def before_unit(self,name,loading=False):
        for attempt in range(PACED_POLICY['wait_rechecks']):
            sample=self.sampler();level,reasons=self.policy.observe(sample)
            action,reasons=decisions(sample,loading=loading,supervisor_reasons=reasons)
            event={'unit_id':name,'action':action,'supervisor_level':level,'reasons':reasons,'sample':asdict(sample)}
            with self.path.open('a') as f:
                f.write(json.dumps(event,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
            if action=='CONTINUE':return sample
            if action=='STOP':raise RuntimeError('Paced workload stopped at safe boundary: '+','.join(reasons))
            print('Resource hold at '+name+': '+','.join(reasons),flush=True)
            self.sleep(PACED_POLICY['wait_seconds'])
        raise RuntimeError('Resource hold persists; completed checkpoints preserved, no automatic restart')
    def after_step(self,number):
        if number in PACED_POLICY['stage_end_steps']:
            self.sleep(PACED_POLICY['initial_stage_pause_seconds'] if number<=2 else PACED_POLICY['stage_pause_seconds'])
        self.before_unit(f'completed_step_{number}')
