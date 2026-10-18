"""CPU rules exercise actual ambiguity gates without MRI/model imports."""
import copy
import pytest
from mri_rl.second_task_eligibility_v1 import plane_triage, metadata_ceiling, matched_label_controls


def geometry(area=1400,components=1,flags=(),distance=20):
    return {'area':area,'components':components,'geometric_flags':list(flags),
        'boundary_distance_display_pixels':distance,'global_grid':'left-middle'}


def notes(apparent=True,text='Asymmetric focus; etiology not adjudicated.'):
    return {'MRI_only_nonexpert_observation':'APPARENT_FOCAL_OR_ASYMMETRIC_ABNORMALITY_REVIEW_REQUIRED' if apparent else 'NO_CONSPICUOUS_FOCUS_AT_CONTACT_SHEET_RESOLUTION_NOT_A_NEGATIVE_GOLD',
        'notes':text,'clinical_presence_truth':'NOT_ADJUDICATED','enhancement_truth':'NOT_ADJUDICATED','expert_eligibility':'NOT_VERIFIED'}


def test_geometry_clean_apparent_plane_is_only_an_expert_review_candidate():
    r=plane_triage({'tumor_geometry':geometry()},notes())
    assert r['primary_queue']=='SPATIAL_REVIEW_CANDIDATE_EXPERT_PENDING'
    assert r['spatial_review_candidate'] is True
    assert r['training_authorized'] is False


@pytest.mark.parametrize('change,reason',[
    ({'area':999},'SMALL_SOURCE_MASK_REQUIRES_REVIEW'),
    ({'components':2},'MULTICOMPONENT_MASK_NOT_MULTIFOCAL_DIAGNOSIS'),
    ({'distance':9.99},'CENTROID_NEAR_GRID_BOUNDARY'),
    ({'flags':['CENTROID_OFF_MASK']},'CENTROID_OFF_MASK'),
    ({'flags':['GLOBAL_VS_LARGEST_COMPONENT_GRID_DISAGREEMENT']},'GLOBAL_LARGEST_REGION_GRID_DISAGREEMENT'),
    ({'flags':['PIXEL_CENTER_CHANGES_GRID']},'PIXEL_CENTER_CONVENTION_CHANGES_TARGET'),
])
def test_each_existing_geometry_risk_blocks_clean_candidate(change,reason):
    r=plane_triage({'tumor_geometry':geometry(**change)},notes())
    assert reason in r['reasons'] and not r['spatial_review_candidate']


def test_mask_negative_is_not_visible_negative_and_cavity_is_not_postoperative_gold():
    negative=plane_triage({'tumor_geometry':geometry(area=0,components=0)},notes(False))
    assert negative['primary_queue']=='MASK_NEGATIVE_NEEDS_EXPERT_NEGATIVE_ADJUDICATION'
    conflict=plane_triage({'tumor_geometry':geometry(area=0,components=0)},notes())
    assert conflict['primary_queue']=='MASK_NEGATIVE_MRI_APPARENT_DISCORDANCE'
    cavity=plane_triage({'tumor_geometry':geometry()},notes(text='Cavity/rim-like appearance, etiology unknown.'))
    assert cavity['primary_queue']=='CAVITY_RIM_ETIOLOGY_ADJUDICATION'
    assert cavity['postoperative_status']=='NOT_VERIFIED_SOURCE_COHORT_PREOPERATIVE'


def test_sparse_and_mask_positive_without_apparent_focus_have_separate_queues():
    n=notes(False);n['MRI_only_nonexpert_observation']='SPARSE_SUPERIOR_FRAME_UNANSWERABLE'
    assert plane_triage({'tumor_geometry':geometry()},n)['primary_queue']=='COVERAGE_UNANSWERABLE'
    assert plane_triage({'tumor_geometry':geometry()},notes(False))['primary_queue']=='MASK_POSITIVE_MRI_NOT_CONSPICUOUS'


def test_expert_status_cannot_be_created_by_changing_a_note():
    n=notes();n['expert_eligibility']='PASS'
    with pytest.raises(ValueError):plane_triage({'tumor_geometry':geometry()},n)


def test_metadata_contingency_and_matched_controls_do_not_fit_a_model():
    rows=[{'id':'a','patient':'p1','site':'S','level':91,'label':'left-middle'},
          {'id':'b','patient':'p2','site':'S','level':91,'label':'right-middle'},
          {'id':'c','patient':'p3','site':'S','level':45,'label':'left-middle'}]
    assert metadata_ceiling(rows,('site','level'))['majority_correct']==2
    controls=matched_label_controls(rows)
    assert controls['pairs']==1 and controls['pairs_PRIVATE'][0]['record_ids']==['a','b']
    assert controls['metadata_only_pair_ceiling']==.5
