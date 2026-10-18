"""Independent finite generation evidence guard; no native calls."""
import importlib.util
from pathlib import Path
import pytest

def helper():
 p=Path(__file__).parents[1]/'src/mri_rl/generation_recovery_v1.py';s=importlib.util.spec_from_file_location('independent_generation_guard',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_length_termination_before_requested_cap_is_rejected():
 row={'response':'incomplete','token_ids':[1],'generation_tokens':1,'finish_reason':'length','hit_max_tokens':False}
 with pytest.raises(ValueError):helper().validate_greedy(row,256)

def test_eos_stop_at_exact_cap_is_not_misclassified_as_length():
 row={'response':'completed','token_ids':[1,2],'generation_tokens':2,'finish_reason':'stop','hit_max_tokens':True}
 helper().validate_greedy(row,2)
