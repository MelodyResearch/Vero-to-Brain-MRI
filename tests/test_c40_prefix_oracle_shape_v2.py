import importlib.util
from pathlib import Path
import pytest

def mod():
 p=Path(__file__).parents[1]/'scripts/run_c40_prefix_oracle_shape_v2.py';s=importlib.util.spec_from_file_location('shape',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_exact_causal_position_and_only_future_zero_padding():
 m=mod();prompt=list(range(115));prefix=list(range(200,211));ids,position=m.padded_oracle_layout(prompt,prefix,15)
 assert len(ids)==129 and position==125
 assert ids[:126]==prompt+prefix and ids[126:]==[0,0,0]
 assert position==len(prompt)+len(prefix)-1 and position!=len(ids)-1

def test_no_padding_when_shape_already_equal_and_invalid_short_shape_rejected():
 m=mod();assert m.padded_oracle_layout([4,5],[6],2)==([4,5,6],2)
 with pytest.raises(ValueError):m.padded_oracle_layout([4,5],[6,7],2)
