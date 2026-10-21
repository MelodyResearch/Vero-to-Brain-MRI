"""THIS micro pilot only:9GiB available AND50%free load admission.

V1 and all global/historical gates remain byte-unchanged. Resident limits, original
swap debt and hard stops are inherited exactly. No host or MLX work on import.
"""
from pathlib import Path
import json
from mri_rl import safe_micro_resources_v1 as v1
from mri_rl import accuracy_recovery_gpu_cache_v3 as v3, resource_supervisor

BASELINE_SWAP_GIB=v1.BASELINE_SWAP_GIB
WINDOWS=dict(v1.WINDOWS)
MODEL_POLICY={**v1.MODEL_POLICY,'load_available_gib_min':9.,'load_pressure_free_percent_min':50.}
MODEL_MEMORY_LIMIT_BYTES=v1.MODEL_MEMORY_LIMIT_BYTES
MODEL_CACHE_LIMIT_BYTES=v1.MODEL_CACHE_LIMIT_BYTES
require=v1.require


class ModelController(v1.ModelController):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  self.policy.update(load_available_gib_min=9.,load_pressure_free_percent_min=50.)
 def boundary(self,name,**kwargs):
  loading='native_model_load' in str(name)
  self.policy['continue_available_gib_min']=9. if loading else 6.
  self.policy['continue_pressure_free_percent_min']=50. if loading else 35.
  def deadline():
   if self.clock()-self.t0>=self.policy['operation_start_window_seconds']:raise v1.v2.WindowExceeded('Authorized phase operation-start deadline exhausted; holds/loading included')
  deadline();result=v3.ResourceController.boundary(self,name,**kwargs);deadline();return result


class ResourceGate(v1.ResourceGate):
 def __init__(self,stage,spec,scope,authorization):
  require(stage=='B_MICRO' and callable(getattr(authorization,'check',None)) and getattr(authorization,'scope',None)==scope,'Exact local model authorization required')
  require(scope.get('resource_stage')==stage and scope.get('resource_policy')==MODEL_POLICY,'Exact local9GiB/50 resource policy required')
  phase=scope.get('execution_stage');require(phase in ('SINGLE_LOAD_FORWARD','MICRO_TRAIN','DEV24_EVAL') and scope.get('action'),'Exact finite pilot phase required')
  self.stage,self.scope,self.authorization=stage,scope,authorization
  self.output=Path(scope['output']);require(self.output.is_absolute(),'Absolute output required')
  runtime=json.loads(Path(spec['runtime_spec']).read_text());self.lock=Path(runtime['global_lock'])
  self.need=scope.get('disk_required_bytes',0);require(type(self.need) is int and self.need>=0,'Finite disk forecast required')
  self.controller=ModelController(sampler=lambda:resource_supervisor.sample_macos(),os_level=lambda:v3.os_pressure_level(),baseline_swap_gib=BASELINE_SWAP_GIB,
   log_path=self.output.with_name(self.output.name+'_resources_PRIVATE.jsonl'),window_seconds=WINDOWS[phase])


def make_resource_gate(stage,spec,scope,authorization):return ResourceGate(stage,spec,scope,authorization)
