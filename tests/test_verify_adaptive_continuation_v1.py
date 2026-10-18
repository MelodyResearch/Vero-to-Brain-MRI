import importlib.util
from pathlib import Path
import numpy as np
import pytest

def module():
 p=Path(__file__).resolve().parents[1]/'scripts/verify_adaptive_continuation_v1.py';s=importlib.util.spec_from_file_location('adaptive_cpu_proof',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_parent_tree_requires_exact_adam_rng_continuity():
 m=module();a={'optimizer':{'step':np.array(52,dtype=np.uint32),'m':np.array([.1])},'rng':{'mlx':[np.array([1,2],dtype=np.uint32)]}}
 import copy
 b=copy.deepcopy(a);assert m.same_tree(a,b);b['optimizer']['step']=np.array(0,dtype=np.uint32);assert not m.same_tree(a,b);b=copy.deepcopy(a);b['rng']['mlx'][0][0]=3;assert not m.same_tree(a,b)

def test_last_terminal_absolute_counter_and_no_best_dev_selection(tmp_path):
 m=module();r={'completed_groups':7,'optimizer_updates':4,'absolute_completed_groups':121,'absolute_optimizer_updates':56};t={**r,'status':'UPDATE_BUDGET_REACHED','boundary':str(tmp_path/'boundary_0007')};assert m.terminal_selection(t,r,tmp_path/'boundary_0007')==(7,4)
 with pytest.raises(ValueError):m.terminal_selection(t,r,tmp_path/'boundary_0006')
 with pytest.raises(ValueError):m.terminal_selection({**t,'absolute_optimizer_updates':4},r,tmp_path/'boundary_0007')
 with pytest.raises(ValueError):m.terminal_selection({**t,'status':'BOUNDED_STAGE_COMPLETE'},r,tmp_path/'boundary_0007')

def test_inactive_first_child_boundary_does_not_claim_update(tmp_path):
 m=module();r={'completed_groups':16,'optimizer_updates':0,'absolute_completed_groups':130,'absolute_optimizer_updates':52};t={**r,'status':'SCHEDULE_COMPLETE','boundary':str(tmp_path/'boundary_0016')};assert m.terminal_selection(t,r,tmp_path/'boundary_0016')==(16,0)
