"""Fixed source-only selection and unchanged renderer; no model or patient fixtures."""
import hashlib
from pathlib import Path
import numpy as np
import pytest
from mri_rl.paired_signal_holdout16_v1 import select_subjects,parse_annex,verify_payload,render_z045,check_pair_geometry


def rows():
    return [{'subject_PRIVATE':f'sub-{site}{i}','site':site,'processed_pair_available':True} for site in ['BO','KA'] for i in [1,2,3]]


def test_site_quota_sha_order_is_frozen_and_input_order_independent():
    x=rows();q={'BO':1,'KA':2};a=select_subjects(x,set(),q);b=select_subjects(list(reversed(x)),set(),q)
    assert a==b
    assert [r['subject_PRIVATE'] for r in a if r['site']=='BO']==sorted([r['subject_PRIVATE'] for r in x if r['site']=='BO'],key=lambda s:hashlib.sha256(('paired-signal-independent16-v1:'+s).encode()).hexdigest())[:1]


def test_exclusions_and_metadata_unavailability_are_applied_before_ranking():
    x=rows();x[0]['processed_pair_available']=False
    a=select_subjects(x,{'sub-BO2'},{'BO':1,'KA':1})
    assert next(r['subject_PRIVATE'] for r in a if r['site']=='BO')=='sub-BO3'
    with pytest.raises(ValueError):select_subjects(x,{'sub-BO2','sub-BO3'},{'BO':1})


def test_pixel_or_score_fields_cannot_change_fixed_source_selection():
    x=rows();a=select_subjects(x,set(),{'BO':2})
    for i,r in enumerate(x):r['model_score']=i*100;r['pixel_eligibility']=False
    assert [r['subject_PRIVATE'] for r in select_subjects(x,set(),{'BO':2})]==[r['subject_PRIVATE'] for r in a]


def test_duplicate_subject_cannot_silently_replace_case():
    x=rows();x.append(x[0].copy())
    with pytest.raises(ValueError):select_subjects(x,set(),{'BO':1})


def test_expected_annex_sha_size_are_not_inferred_from_pointer_bytes():
    a=parse_annex('../../annex/SHA256E-s123--'+'a'*64+'.nii.gz/SHA256E-s123--'+'a'*64+'.nii.gz')
    assert a=={'bytes':123,'sha256':'a'*64}
    with pytest.raises(ValueError):parse_annex('URL-without-source-byte-hash')


def test_verified_payload_resume_requires_both_hash_and_size(tmp_path):
    p=tmp_path/'source.nii.gz';p.write_bytes(b'payload');h=hashlib.sha256(b'payload').hexdigest()
    assert verify_payload(p,7,h)
    assert not verify_payload(p,7,'b'*64)
    assert not verify_payload(p,8,h)
    assert not verify_payload(tmp_path/'missing',7,h)


def test_render_matches_existing_frozen_numpy_pillow_contract():
    from mri_rl.sequence_qualification_v1 import source_sequence_render
    x=np.arange(8*9*50,dtype=np.float32).reshape(8,9,50);x[0]=0
    image,window,stats=render_z045(x)
    old,oldwindow=source_sequence_render(x,45)
    assert window==oldwindow and image.tobytes()==old.tobytes()
    assert image.size==(364,436) and image.mode=='RGB'
    assert stats['clinical_visibility']=='NOT_ADJUDICATED'


def test_empty_or_nonfinite_volume_cannot_produce_qualified_image():
    with pytest.raises(ValueError):render_z045(np.zeros((3,4,50)))
    x=np.ones((3,4,50));x[1,1,1]=np.nan
    with pytest.raises(ValueError):render_z045(x)


def test_pair_requires_same_shape_affine_and_original_LAS_storage():
    affine=np.diag([-1.,1.,1.,1.])
    result=check_pair_geometry((182,218,182),affine,(182,218,182),affine)
    assert result['storage_axcodes']==['L','A','S'] and result['world_coordinate_convention']=='NIfTI RAS+ world;not RAS storage'
    with pytest.raises(ValueError):check_pair_geometry((182,218,182),affine,(182,218,182),np.eye(4))
    with pytest.raises(ValueError):check_pair_geometry((182,218,182),np.eye(4),(182,218,182),np.eye(4))
