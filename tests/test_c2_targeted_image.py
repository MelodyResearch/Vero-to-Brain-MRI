import importlib.util
from pathlib import Path
import pytest
s=importlib.util.spec_from_file_location('c2img',Path(__file__).resolve().parents[1]/'scripts/c2_targeted_image.py');m=importlib.util.module_from_spec(s)
if Path(s.origin).exists():s.loader.exec_module(m)


def test_selection_uses_only_threshold_eligible_new_training_subjects():
    assert hasattr(m,'select_one')
    masks=[{'subject_PRIVATE':'old','levels':{'136':{'tumor':2000}}},{'subject_PRIVATE':'new','levels':{'136':{'tumor':1000}}},{'subject_PRIVATE':'small','levels':{'136':{'tumor':999}}}]
    assert m.select_one(masks,{'old'},set())['subject_PRIVATE']=='new'
    with pytest.raises(ValueError):m.select_one(masks,{'old'},{'new'})


def test_selection_is_order_invariant():
    assert hasattr(m,'select_one')
    masks=[{'subject_PRIVATE':p,'levels':{'136':{'tumor':1000}}} for p in ('a','b')]
    assert m.select_one(masks,set(),set())==m.select_one(masks[::-1],set(),set())
