"""Independent CPU counterexamples; no MLX imports or model/runtime calls."""
import importlib.util,json,struct,sys
from pathlib import Path
import numpy as np
S=Path(__file__).resolve().parents[1]
p=S/'scripts/verify_safe_micro_A_salvage_v1.py'
s=importlib.util.spec_from_file_location('independent_salvage_review',p);v=importlib.util.module_from_spec(s);s.loader.exec_module(v)

def test_weight_matches_but_moment_is_from_wrong_gradient():
    w=np.array([.01,-.02],np.float32);g=np.array([.25,-.5],np.float32);m=np.array([.01,.015],np.float32);vv=np.array([.02,.03],np.float32)
    good=v.numpy_adam(w,g,m,vv,1e-5);bad=v.numpy_adam(w,-g,m,vv,1e-5)
    r=v.compare_step(w,g,m,vv,good['w'],bad['m'],good['v'],1e-5)
    assert not r['pass'] and r['m']['failed_elements']==2

def test_wrong_old_moments_rejected_despite_same_step_and_gradient():
    w=np.array([.01],np.float32);g=np.array([.25],np.float32);m=np.array([.01],np.float32);vv=np.array([.02],np.float32)
    bad=v.numpy_adam(w,g,m*2,vv,1e-5)
    assert not v.compare_step(w,g,m,vv,bad['w'],bad['m'],bad['v'],1e-5)['pass']

def test_reader_rejects_duplicate_json_keys(tmp_path):
    h=b'{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'
    p=tmp_path/'duplicate.safetensors';p.write_bytes(struct.pack('<Q',len(h))+h+b'\0'*4)
    try:v.TensorFile(p)
    except ValueError:pass
    else:raise AssertionError('duplicate tensor entry accepted')

def test_no_native_imported():
    assert not any(x.split('.')[0] in ('mlx','mlx_vlm','torch') for x in sys.modules)
