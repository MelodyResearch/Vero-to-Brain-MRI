import importlib.util
from pathlib import Path
import pytest
W=Path(__file__).parents[1]
def runner():
 s=importlib.util.spec_from_file_location("generation_recovery_timing",W/"scripts/run_generation_recovery_v1.py");m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_measures_actual_call_preserving_all_scientific_fields(monkeypatch):
 m=runner();times=iter([100.0,102.25,200.0,200.5]);monkeypatch.setattr(m.time,"monotonic",lambda:next(times))
 original={"token_ids":[1,2],"response":"frozen","scores":[-0.4,-2.0]};calls=[]
 for field,expected in [("generation_wall_seconds",2.25),("candidate_wall_seconds",.5)]:
  result=m.timed_row(lambda:(calls.append(1) or original.copy()),field)
  assert result==dict(original,**{field:expected}) and field not in original
 assert len(calls)==2

def test_rejects_invalid_monotonic_duration(monkeypatch):
 m=runner();times=iter([2.,1.]);monkeypatch.setattr(m.time,"monotonic",lambda:next(times))
 with pytest.raises(ValueError):m.timed_row(lambda:{},"generation_wall_seconds")
