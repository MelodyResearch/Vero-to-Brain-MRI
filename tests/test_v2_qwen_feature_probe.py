import numpy as np
import pytest
from mri_rl.v2_qwen_feature_probe import select_sequence_records,pool_merged,dual_ridge,sequence_metrics

def rows(split,people):
 return [{'id':f'{p}_z{z:03d}_{label}_sequence','ability':'mri_sequence','extra_info':{'patient_key':p,'level':z,'internal_split':split},'reward_model':{'ground_truth':label},'images':['unread']} for p in people for z in (45,91,136) for label in ('FLAIR','contrast-enhanced T1')]
def test_selection_first_four_people_balanced_same_level_and_dev_disjoint():
 a=rows('train',['f','a','e','b','d','c']);b=rows('internal_dev',['g','h','i','j']);r=select_sequence_records(a,b)
 assert len(r)==48 and {x['extra_info']['patient_key'] for x in r[:24]}=={'a','b','c','d'}
 assert {x['extra_info']['patient_key'] for x in r[24:]}=={'g','h','i','j'}
 with pytest.raises(ValueError):select_sequence_records(a,b[:-1])
 with pytest.raises(ValueError):select_sequence_records(a,rows('internal_dev',['a','h','i','j']))
def test_mean_pool_validates_actual_merged_grid_shape_and_finiteness():
 x=np.arange(24,dtype=float).reshape(6,4);assert np.array_equal(pool_merged(x,[[1,4,6]],2,4),x.mean(axis=0))
 for values,grid in ((x[:-1],[[1,4,6]]),(x,[[2,2,6]]),(x,[[1,3,8]]),(x,[[1,2,4],[1,2,2]]),(x*np.nan,[[1,4,6]])):
  with pytest.raises(ValueError):pool_merged(values,grid,2,4)
def test_dual_ridge_bounded_gram_and_train_only_scaling():
 x=np.tile(np.array([[-2.],[-1.],[1.],[2.]]),(1,4096));y=np.array([0,0,1,1]);dev=x*10
 p,proof=dual_ridge(x,y,dev);assert list(p)==list(y) and proof['gram_shape']==[4,4] and proof['primal_D_by_D_formed'] is False
 _,other=dual_ridge(x,y,dev*100);assert proof==other
 with pytest.raises(ValueError):dual_ridge(x,np.zeros(4),dev)
def test_sequence_metrics_use_people_and_explicit_invalids():
 r=rows('internal_dev',['a','b','c','d']);y=np.array([int(x['reward_model']['ground_truth']=='contrast-enhanced T1') for x in r]);p=y.copy();p[:6]=-1;m=sequence_metrics(r,p)
 assert m['people']==4 and m['patient_macro_accuracy']==.75 and m['invalid']==6
 assert m['recall']=={'FLAIR':.75,'contrast-enhanced T1':.75}

def test_nonintegral_class_prediction_is_rejected():
 r=rows('internal_dev',['a','b','c','d'])
 with pytest.raises(ValueError):sequence_metrics(r,np.full(24,.5))

def test_runner_binds_actual_worktree_imported_sources(tmp_path,monkeypatch):
 import importlib.util,hashlib
 from pathlib import Path
 path=Path(__file__).resolve().parents[1]/'scripts/run_v2_qwen_mean48.py';spec=importlib.util.spec_from_file_location('mean48runner',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 primary=tmp_path/'primary';worktree=tmp_path/'worktree'
 for root in (primary,worktree):
  p=root/'src/mri_rl/backend.py';p.parent.mkdir(parents=True);p.write_text('bound')
 monkeypatch.setattr(module,'W',worktree);sources={'backend.py':hashlib.sha256(b'bound').hexdigest()};module.verify_source_closure(primary,sources)
 (worktree/'src/mri_rl/backend.py').write_text('changed')
 with pytest.raises(AssertionError,match='Worktree'):module.verify_source_closure(primary,sources)

def test_release_checks_native_allocation_and_model_pointer():
 import importlib.util
 from pathlib import Path
 from types import SimpleNamespace
 path=Path(__file__).resolve().parents[1]/'scripts/run_v2_qwen_mean48.py';spec=importlib.util.spec_from_file_location('release_runner',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 def backend(remaining=0,drop=True):
  b=SimpleNamespace(model=object(),mx=SimpleNamespace(clear_cache=lambda:None,get_active_memory=lambda:remaining))
  def close():
   if drop:b.model=None
  b.close=close;return b
 ok=m.release_backend(backend());assert ok['model_is_none'] and ok['active_MLX_bytes']==0 and ok['native_release_verified']
 with pytest.raises(AssertionError):m.release_backend(backend(128*1024**2))
 with pytest.raises(AssertionError):m.release_backend(backend(drop=False))

def test_all48_without_completion_allows_integrity_finalization_no_new_forwards():
 import importlib.util
 from pathlib import Path
 path=Path(__file__).resolve().parents[1]/'scripts/run_v2_qwen_mean48.py';spec=importlib.util.spec_from_file_location('budget_runner',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 assert m.work_budget(48,False)=={'new_vision_forwards':0,'max_model_loads':1,'mode':'INTEGRITY_FINALIZATION_ONLY'}
 assert m.work_budget(48,True)=={'new_vision_forwards':0,'max_model_loads':0,'mode':'ALREADY_RELEASED_CPU_ANALYSIS_ONLY'}
 assert m.work_budget(47,False)=={'new_vision_forwards':1,'max_model_loads':1,'mode':'EXTRACT_REMAINING_AND_FINALIZE'}
 with pytest.raises(AssertionError):m.work_budget(47,True)

def test_completion_requires_postclose_release_and_exact_integrity_binding():
 import importlib.util,copy
 from pathlib import Path
 path=Path(__file__).resolve().parents[1]/'scripts/run_v2_qwen_mean48.py';spec=importlib.util.spec_from_file_location('completion_runner',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 hashes=[str(i) for i in range(48)];r={'run_identity':'run','plan_sha256':'plan','record_sha256':hashes,'adapter_tensors_compared_after':504,'optimizer_step':0,'weight_updates':0,'release':{'native_release_verified':True,'model_is_none':True,'active_MLX_bytes':0,'active_limit_bytes':128*1024**2}}
 m.validate_completion(r,'run','plan',hashes)
 missing=copy.deepcopy(r);missing.pop('release')
 with pytest.raises(KeyError):m.validate_completion(missing,'run','plan',hashes)
 for key,value in [('optimizer_step',1),('adapter_tensors_compared_after',503),('record_sha256',['changed']*48)]:
  bad=copy.deepcopy(r);bad[key]=value
  with pytest.raises(AssertionError):m.validate_completion(bad,'run','plan',hashes)
 bad=copy.deepcopy(r);bad['release']['model_is_none']=False
 with pytest.raises(AssertionError):m.validate_completion(bad,'run','plan',hashes)

def test_direct_analysis_dependencies_are_bound_and_worktree_tamper_rejected(tmp_path,monkeypatch):
 import importlib.util
 from pathlib import Path
 path=Path(__file__).resolve().parents[1]/'scripts/run_v2_qwen_mean48.py';spec=importlib.util.spec_from_file_location('dependency_runner',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 primary=tmp_path/'primary';worktree=tmp_path/'worktree';names=('experiments.py','two_task_pool.py','vero_mri_reward.py')
 for root in (primary,worktree):
  (root/'src/mri_rl').mkdir(parents=True)
  for name in names:(root/'src/mri_rl'/name).write_text('bound_'+name)
 monkeypatch.setattr(m,'W',worktree);bound=m.bind_direct_dependencies(primary,{})
 assert set(bound)==set(names)
 for name in names:
  p=worktree/'src/mri_rl'/name;p.write_text('changed')
  with pytest.raises(AssertionError,match='Worktree'):m.bind_direct_dependencies(primary,{})
  p.write_text('bound_'+name)
