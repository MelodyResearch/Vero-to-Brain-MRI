"""Small CPU fixtures pin forensic reader and non-bias-corrected recurrence."""
import importlib.util
import json
import struct
from pathlib import Path
import numpy as np
import pytest

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/verify_safe_micro_A_salvage_v1.py'

def module():
    assert SCRIPT.is_file(), 'bounded forensic verifier is not implemented'
    spec=importlib.util.spec_from_file_location('salvage',SCRIPT);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def safefile(path,arrays):
    header={};data=b''
    for key,array in arrays.items():
        a=np.asarray(array);raw=a.tobytes();header[key]={'dtype':{'float32':'F32','uint64':'U64','uint32':'U32'}[a.dtype.name], 'shape':list(a.shape),'data_offsets':[len(data),len(data)+len(raw)]};data+=raw
    raw=json.dumps(header).encode();path.write_bytes(struct.pack('<Q',len(raw))+raw+data);return path

def test_reader_returns_only_requested_array_and_rejects_truncated_payload(tmp_path):
    m=module();p=safefile(tmp_path/'one.safetensors',{'x':np.array([1.,2.],np.float32),'n':np.array(52,np.uint64)})
    with m.TensorFile(p) as f:
        assert set(f.keys)=={'x','n'} and np.array_equal(f.get('x'),[1.,2.]) and int(f.get('n'))==52
    p.write_bytes(p.read_bytes()[:-1])
    with pytest.raises(ValueError):m.TensorFile(p)

def test_reader_rejects_overlapping_tensor_ranges(tmp_path):
    m=module();h={'x':{'dtype':'F32','shape':[1],'data_offsets':[0,4]},'y':{'dtype':'F32','shape':[1],'data_offsets':[0,4]}};b=json.dumps(h).encode();p=tmp_path/'bad.safetensors';p.write_bytes(struct.pack('<Q',len(b))+b+b'\0'*4)
    with pytest.raises(ValueError):m.TensorFile(p)

def test_adam_one_step_uses_saved_moments_and_no_bias_correction():
    m=module();w=np.array([1.],np.float32);g=np.array([2.],np.float32);old=np.array([0.],np.float32)
    result=m.numpy_adam(w,g,old,old,.1)
    assert np.allclose(result['m'],[.2],rtol=0,atol=1e-8)
    assert np.allclose(result['v'],[.004],rtol=0,atol=1e-9)
    assert np.allclose(result['w'],[.6837723],rtol=0,atol=1e-7)
    # A common wrong bias-corrected answer would be .9.
    assert abs(float(result['w'][0])-.9)>.1

def test_rounding_bounds_reject_wrong_update_and_nonfinite_or_negative_second_moment():
    m=module();w=np.array([.5],np.float32);g=np.array([.3],np.float32);mom=np.array([.01],np.float32);var=np.array([.03],np.float32)
    expected=m.numpy_adam(w,g,mom,var,1e-5)
    ok=m.compare_step(w,g,mom,var,expected['w'],expected['m'],expected['v'],1e-5)
    assert ok['pass'] and ok['w']['max_abs_error']==0
    wrong=expected['w']+np.float32(1e-3)
    assert not m.compare_step(w,g,mom,var,wrong,expected['m'],expected['v'],1e-5)['pass']
    for bad in (np.array([np.nan],np.float32),np.array([-1.],np.float32)):
        with pytest.raises(ValueError):m.compare_step(w,g,mom,bad,expected['w'],expected['m'],expected['v'],1e-5)

def test_layout_refuses_duplicate_tensor_keys_and_unreferenced_header_arrays(tmp_path):
    m=module();p=safefile(tmp_path/'a.safetensors',{'x':np.array(1,np.uint64),'extra':np.array(2,np.uint64)})
    node={'kind':'tensor','key':'x','shape':[],'dtype':'mlx.core.uint64'}
    with m.TensorFile(p) as f:
        with pytest.raises(ValueError):m.layout_index({'kind':'dict','items':[['a',node],['b',node]]},f)
        with pytest.raises(ValueError):m.layout_index(node,f)
