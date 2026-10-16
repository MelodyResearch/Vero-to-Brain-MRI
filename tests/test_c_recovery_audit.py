import importlib.util
from pathlib import Path
import numpy as np
import pytest
p=Path(__file__).resolve().parents[1]/'scripts/c_recovery_audit.py'
s=importlib.util.spec_from_file_location('crecovery',p);m=importlib.util.module_from_spec(s)
if p.exists():s.loader.exec_module(m)


def test_centroid_outside_disconnected_regions_and_grid_disagreement():
    assert hasattr(m,'spatial_summary')
    mask=np.zeros((9,9),dtype=bool);mask[4,0:2]=True;mask[4,7:9]=True
    result=m.spatial_summary(mask)
    assert result['components']==2
    assert result['centroid_on_mask'] is False
    assert result['global_grid']=='center-middle'
    assert result['largest_component_grid']=='left-middle'
    assert result['majority_cell_fraction']==0.5


def test_display_mapping_uses_rotated_plane_and_panel_origin():
    assert hasattr(m,'display_xy')
    assert m.display_xy(0,0,(182,218),0)==(4.5,462.5)
    assert m.display_xy(181,217,(182,218),1)==(746.5,28.5)


def test_site_holdout_excludes_same_site_and_same_patient():
    assert hasattr(m,'heldout_neighbors')
    rows=[{'patient':'a','site':'x','record_id':'0'}, {'patient':'b','site':'x','record_id':'1'}, {'patient':'c','site':'y','record_id':'2'}]
    assert m.heldout_neighbors(rows,np.array([[0.],[0.],[100.]]))==[2,2,0]
    rows[2]['patient']='a'
    with pytest.raises(ValueError,match='No eligible'):
        m.heldout_neighbors(rows,np.array([[0.],[0.],[100.]]))
