import importlib.util
from pathlib import Path
import numpy as np
s=importlib.util.spec_from_file_location('imagebaseline',Path(__file__).resolve().parents[1]/'scripts/c1_image_shortcut_baseline.py');m=importlib.util.module_from_spec(s)
if Path(s.origin).exists():s.loader.exec_module(m)


def test_lopo_excludes_all_same_patient_rows_even_when_nearest():
    assert hasattr(m,'lopo_predictions')
    rows=[{'patient':'a','record_id':'a1','label':'yes'},{'patient':'a','record_id':'a2','label':'yes'},{'patient':'b','record_id':'b','label':'no'}]
    assert m.lopo_predictions(rows,np.array([[0.],[0.],[10.]]))==['no','no','yes']


def test_equal_distances_have_record_id_tie_break_not_label_tie_break():
    assert hasattr(m,'lopo_predictions')
    rows=[{'patient':'a','record_id':'0','label':'yes'},{'patient':'b','record_id':'2','label':'yes'},{'patient':'c','record_id':'1','label':'no'}]
    assert m.lopo_predictions(rows,np.ones((3,2)))[0]=='no'
