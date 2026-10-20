import numpy as np
import pytest
from mri_rl import mixture_adam_cpu_v3 as m


def test_mixture_preconditions_adam_instead_of_interpolating_deltas():
    one=np.array([2.,-1.],dtype=np.float32);two=np.array([-3.,4.],dtype=np.float32)
    first=np.array([.2,-.3],dtype=np.float32);second=np.array([.01,.04],dtype=np.float32)
    w=.4;g=m.mix_arrays(one,two,w)
    actual=m.adam_delta(g,first,second,1e-5)
    expected=-np.float32(1e-5)*(np.float32(.9)*first+np.float32(.1)*g)/(np.sqrt(np.float32(.999)*second+np.float32(.001)*g*g)+np.float32(1e-8))
    np.testing.assert_array_equal(actual,expected)
    linear=(1-w)*m.adam_delta(one,first,second,1e-5)+w*m.adam_delta(two,first,second,1e-5)
    assert np.max(np.abs(actual-linear))>1e-7


def test_mix_endpoints_and_readonly_state():
    a=np.array([1.,2.],np.float32);b=-a;first=a.copy();second=a*a;old=[x.copy() for x in (a,b,first,second)]
    np.testing.assert_array_equal(m.mix_arrays(a,b,0),a)
    np.testing.assert_array_equal(m.mix_arrays(a,b,1),b)
    m.adam_delta(m.mix_arrays(a,b,.3),first,second,1e-5)
    for x,y in zip((a,b,first,second),old):np.testing.assert_array_equal(x,y)


def test_unknown_nonfinite_negative_or_foreign_inputs_fail():
    a=np.ones(2,np.float32)
    for w in (-.01,1.01,float('nan')):
        with pytest.raises(ValueError):m.mix_arrays(a,a,w)
    for bad in (np.array([np.nan,1],np.float32),np.array([np.inf,1],np.float32)):
        with pytest.raises(ValueError):m.adam_delta(bad,a,a,1e-5)
    with pytest.raises(ValueError):m.adam_delta(a,a,-a,1e-5)
    with pytest.raises(ValueError):m.adam_delta(a,a,np.ones(3,np.float32),1e-5)


def test_total_and_increment_protection_are_distinct():
    assert m.class_protected({'mT':.01,'mF':.01,'mDallas':.01},guard=1e-6)
    assert not m.class_protected({'mT':.01,'mF':-.001,'mDallas':.01},guard=1e-6)
    assert not m.class_protected({'mT':0,'mF':.01,'mDallas':.01},guard=0)
    assert not m.class_protected({'mT':float('nan'),'mF':.01,'mDallas':.01},guard=0)
