import importlib.util
from pathlib import Path
import pytest
s=importlib.util.spec_from_file_location('c2cohort',Path(__file__).resolve().parents[1]/'scripts/c2_frozen_cohort.py');m=importlib.util.module_from_spec(s)
if Path(s.origin).exists():s.loader.exec_module(m)


def test_frozen_union_uses_only_selected_new_geometry_rows():
    assert hasattr(m,'selected_new_union')
    d={'categories':{'tumor':{'selected_PRIVATE':[{'patient':'old'},{'patient':'a','MRI_image_status':'NOT_DOWNLOADED_OR_RENDERED'}]},
                     'enh':{'selected_PRIVATE':[{'patient':'a','MRI_image_status':'NOT_DOWNLOADED_OR_RENDERED'},{'patient':'b','MRI_image_status':'NOT_DOWNLOADED_OR_RENDERED'}]}}}
    assert m.selected_new_union(d)=={'a','b'}


def test_scope_rejects_reserved_subject_and_excess_download_bytes():
    assert hasattr(m,'check_scope')
    rows=[{'subject_PRIVATE':'new','payloads':[{'bytes':5},{'bytes':6}]}]
    m.check_scope(rows,set(),{'reserved'},20)
    with pytest.raises(ValueError):m.check_scope(rows,{'new'},set(),20)
    with pytest.raises(ValueError):m.check_scope(rows,set(),{'new'},20)
    with pytest.raises(ValueError):m.check_scope(rows,set(),set(),10)
