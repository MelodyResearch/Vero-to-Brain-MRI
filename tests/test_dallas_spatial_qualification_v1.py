import numpy as np
import pytest
from PIL import Image
from mri_rl.dallas_spatial_qualification_v1 import require_train,spatial_geometry,cranial_roi_occlusion,grid

def test_nontrain_rejected():
    with pytest.raises(ValueError):require_train([{'extra_info':{'internal_split':'internal_dev'}}])

def test_rot90_frame_and_multifocal_centroid_ambiguity():
    mask=np.zeros((18,24),bool);mask[2:4,10:12]=1;mask[14:16,10:12]=1
    g=spatial_geometry(mask)
    assert g['components']==2 and g['global_grid']=='center-middle'
    assert 'CENTROID_OFF_MASK' in g['geometric_flags']
    assert g['largest_component_grid']!='center-middle'
    assert g['centroid_xy_display_pixelcenter']==[17.5,25.5]

def test_zero_mask_remains_unadjudicated():
    assert spatial_geometry(np.zeros((18,24)))['geometric_flags']==['MASK_NEGATIVE_NOT_VISUAL_NEGATIVE']

def test_occlusion_is_fixed_and_retains_no_text_or_gold():
    im=Image.new('L',(384,384),255);out=np.asarray(cranial_roi_occlusion(im))
    assert out[45,64]==255 and out[120,64]==0 and out[0,0]==0

def test_screen_grid_boundary():
    assert grid(2.999,3,9,9)=='left-middle'
    assert grid(3,3,9,9)=='center-middle'
