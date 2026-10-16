"""Native adapter safety boundaries without importing a model/backend runtime."""
import importlib.util
from pathlib import Path
import sys
import pytest


def module():
    path=Path(__file__).parents[1]/'src/mri_rl/mlx_continuous_gspo.py'
    assert path.exists(), 'Integratable native GSPO backend missing'
    spec=importlib.util.spec_from_file_location('native_continuous_test',path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def test_native_module_import_and_denied_constructor_never_load_runtime(tmp_path,monkeypatch):
    class RejectNative:
        def find_spec(self, fullname, *args):
            if fullname.split('.')[0] in {'mlx','mlx_vlm','torch','transformers'}:
                raise AssertionError('Forbidden native import before authorization')
    blocker=RejectNative();sys.meta_path.insert(0,blocker)
    try:
        m=module()
        adapter=tmp_path/'adapter';adapter.write_bytes(b'fixture')
        def deny(unit): raise ValueError('independent gate denied')
        with pytest.raises(ValueError,match='independent gate denied'):
            m.NativeGSPOBackend(adapter,m.PROFILE,model_path=tmp_path,contract=object(),seed=7,before_unit=deny)
    finally:sys.meta_path.remove(blocker)


def test_native_backend_requires_explicit_adapter_and_gate(tmp_path):
    m=module()
    with pytest.raises(ValueError,match='resource authorizer'):
        m.NativeGSPOBackend(None,m.PROFILE,model_path=tmp_path,contract=object(),seed=7,before_unit=None)
    with pytest.raises(ValueError,match='starting adapter'):
        m.NativeGSPOBackend(None,m.PROFILE,model_path=tmp_path,contract=object(),seed=7,before_unit=lambda _:None)

@pytest.mark.parametrize('bad', [None,'weight','moment','variance'])
def test_native_result_validation_with_finite_and_invalid_arrays(bad):
    import numpy as np
    from types import SimpleNamespace
    m=module();backend=m.NativeGSPOBackend.__new__(m.NativeGSPOBackend)
    weight=np.array([1.,2.]);moment=np.array([.1,.2]);variance=np.array([.3,.4])
    if bad=='weight':weight[0]=np.nan
    if bad=='moment':moment[0]=np.nan
    if bad=='variance':variance[0]=-1.
    backend.mx=SimpleNamespace(all=np.all,isfinite=np.isfinite)
    backend.model=SimpleNamespace(trainable_parameters=lambda:{'weight':weight})
    backend.optimizer_state=lambda:{'weight.m':moment,'weight.v':variance}
    backend.tree_flatten=lambda value:list(value.items())
    result=backend.validate_state()
    assert result['parameters_finite'] is (bad!='weight')
    assert result['optimizer_finite'] is (bad!='moment')
    assert result['second_moments_nonnegative'] is (bad!='variance')
