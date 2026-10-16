import importlib.util
from pathlib import Path
s=importlib.util.spec_from_file_location('candidate',Path(__file__).resolve().parents[1]/'scripts/build_slice_aware_candidate.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)


def test_matching_never_reuses_patients_and_fails_infeasible():
    a={'patient':'a','record_id':'a'};b={'patient':'b','record_id':'b'}
    assert len({r['patient'] for r in m.match_slots({'x':[a,b],'y':[a]},['x','y'])})==2
    assert m.match_slots({'x':[a],'y':[a]},['x','y']) is None


def test_small_positive_masks_are_not_relabeled_to_negative():
    rows=[{'patient':str(i),'record_id':str(i),'category':'mri_tumor_presence','level':45,'label':'yes','tumor':5} for i in range(28)]
    selected,quotas=m.binary_schedule(rows,'mri_tumor_presence')
    assert selected==[] and quotas is None
    assert all(r['label']=='yes' for r in rows)
