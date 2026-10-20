"""Exact endpoint regressions: geometric mass centers are not visible-focus gold."""
import numpy as np
import pytest
from mri_rl.spatial_ontology_resolution_v1 import endpoints, source_labels, amended_row


def test_pixel_mass_center_uses_display_edge_coordinates():
    mask=np.zeros((3,3),bool);mask[0,0]=True
    e=endpoints(mask)
    assert e['union_centroid_display_edge_xy']==[1.,1.]
    assert e['grid']=='left-top'
    assert e['legacy_centroid_display_index_xy']==[0.,0.]


def test_half_open_thirds_boundary_uses_right_bottom_cell():
    mask=np.zeros((3,3),bool);mask[0,0:2]=True
    e=endpoints(mask)
    assert e['union_centroid_display_edge_xy']==[2.,1.]
    assert e['grid']=='center-top'
    assert e['legacy_grid']=='left-top'
    assert e['boundary_distance_display_pixels']==0


def test_disconnected_regions_union_center_can_be_off_mask():
    mask=np.zeros((9,9),bool);mask[4,0]=True;mask[4,8]=True
    e=endpoints(mask)
    assert e['components_8']==2
    assert e['union_centroid_on_mask_nearest_pixel'] is False
    assert e['grid']=='center-middle'
    assert {x['grid'] for x in e['components']}=={'left-middle','right-middle'}
    assert e['visible_focus_center']=='NOT_ADJUDICATED'


def test_single_connected_ring_still_has_off_mask_centroid():
    mask=np.ones((5,5),bool);mask[1:4,1:4]=False
    e=endpoints(mask)
    assert e['components_8']==1
    assert not e['union_centroid_on_mask_nearest_pixel']


def test_empty_mask_never_acquires_center_or_negative_clinical_gold():
    e=endpoints(np.zeros((5,5),bool))
    assert e['grid'] is None and e['union_centroid_display_edge_xy'] is None
    assert e['visible_focus_center']=='NOT_ADJUDICATED'


@pytest.mark.parametrize('a',[np.array([[4]]),np.array([[1.5]]),np.array([[np.nan]])])
def test_unknown_or_noninteger_source_labels_fail_closed(a):
    with pytest.raises(ValueError):source_labels(a)


def test_observed_numeric_3_does_not_itself_prove_semantics():
    assert source_labels(np.array([[0,1],[2,3]]))=={'0':1,'1':1,'2':1,'3':1}


def test_amendment_preserves_original_target_and_cannot_authorize_training():
    r={'id':'a','ability':'mri_tumor_grid','extra_info':{'internal_split':'train'},'reward_model':{'ground_truth':'left-top'},'prompt':[{'role':'user'}]}
    e=endpoints(np.array([[True,True,False],[False,False,False],[False,False,False]]))
    out=amended_row(r,e,{'label3':'VERIFIED_SOURCE_RELEASE_DICTIONARY'})
    assert out['reward_model']==r['reward_model'] and out['prompt']==r['prompt']
    assert out['spatial_ontology_resolution_v1']['proposed_arithmetic_target_PRIVATE']=='center-top'
    assert out['spatial_ontology_resolution_v1']['target_status']=='OBJECTIVE_CONVENTION_CHANGE_REQUIRES_VERSIONED_ENDPOINT'
    assert out['training_authorized'] is False


def test_nontrain_amendment_rejected():
    with pytest.raises(ValueError):amended_row({'extra_info':{'internal_split':'test'}},endpoints(np.zeros((2,2),bool)),{})


def test_pointer_lineage_requires_exact_local_release_blob():
    from mri_rl.spatial_ontology_resolution_v1 import pointer_lineage
    import hashlib
    target='../../.git/annex/objects/ab/cd/SHA256E-s3--'+'a'*64+'.nii.gz'
    b=target.encode();blob=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
    t={'x':{'sha':blob,'mode':'120000'}}
    assert pointer_lineage('x',target,t,t)['historical_same_annotation_pointer'] is True
    assert pointer_lineage('x',target,{},t)['historical_same_annotation_pointer'] is False
    with pytest.raises(ValueError):pointer_lineage('x',target,t,{'x':{'sha':'bad','mode':'120000'}})


def test_matching_blob_in_regular_file_mode_cannot_verify_annotation_semantics():
    from mri_rl.spatial_ontology_resolution_v1 import pointer_lineage
    import hashlib
    target='../../.git/annex/SHA256E-s3--'+'a'*64+'.nii.gz';b=target.encode()
    blob=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
    result=pointer_lineage('x',target,{'x':{'sha':blob,'mode':'100644'}},{'x':{'sha':blob,'mode':'120000'}})
    assert result['historical_same_annotation_pointer'] is False
    assert result['label3_semantics']=='NOT_VERIFIED_NEW_RELEASE_ANNOTATION'
