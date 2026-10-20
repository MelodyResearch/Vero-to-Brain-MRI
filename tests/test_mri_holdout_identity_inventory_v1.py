import hashlib
import pytest
from mri_rl import mri_holdout_identity_inventory_v1 as p


def test_actual_anonymous_crosswalk_and_metadata_paths_join_without_labels():
    people=['sub-BO1','sub-KA2'];index=p.identity_index(people)
    token=hashlib.sha256(people[0].encode()).hexdigest()[:12]
    obj={'patient_key':token,'source_path':'/cache/sub-KA2/anat/scan.nii.gz',
         'ground_truth_PRIVATE':'sub-BO999','response':'sub-BO999'}
    assert p.metadata_identities(obj,index)==set(people)


def test_outcome_fields_cannot_influence_exclusion_or_selection():
    index=p.identity_index(['sub-BO1'])
    assert p.metadata_identities({'model_outputs':[{'patient_key':'sub-BO1'}],
        'prediction':'sub-BO1','ground_truth_PRIVATE':'sub-BO1'},index)==set()


def test_final_salted_patient_ids_join_via_source_identity_not_legacy_hash():
    index=p.identity_index(['sub-BO1','sub-KA2'])
    registry={'people':2,'cases_PRIVATE':[{'patient_key':'different-salted-key','ground_truth_PRIVATE':'unused'}],
              'files_PRIVATE':[{'subject_PRIVATE':'sub-BO1'},{'subject_PRIVATE':'sub-KA2'}]}
    assert p.final_identity_projection(registry,index)=={'sub-BO1','sub-KA2'}
    registry['files_PRIVATE'].pop()
    with pytest.raises(ValueError):p.final_identity_projection(registry,index)


def test_annex_pointer_is_not_an_available_mri_payload(tmp_path):
    subject='sub-BO1';d=tmp_path/subject/'anat';d.mkdir(parents=True)
    for suffix in ('FLAIR','run-02_T1w'):
        (d/(subject+'_'+suffix+'.json')).write_text('{}')
        (d/(subject+'_'+suffix+'.nii.gz')).write_text('/annex/objects/not-image')
    x=p.source_availability(tmp_path,subject)
    assert x['both_sidecars'] and not x['both_gzip_payloads']
    assert all(v['status']=='NON_GZIP_POINTER_OR_OTHER' for v in x['modalities'].values())


def test_gzip_header_only_is_not_claimed_as_image_qualification(tmp_path):
    subject='sub-KA2';d=tmp_path/subject/'anat';d.mkdir(parents=True)
    for suffix in ('FLAIR','run-02_T1w'):
        (d/(subject+'_'+suffix+'.json')).write_text('{}')
        (d/(subject+'_'+suffix+'.nii.gz')).write_bytes(b'\x1f\x8bnot-qualified')
    x=p.source_availability(tmp_path,subject)
    assert x['both_gzip_payloads'] is True
    assert x['image_geometry_or_label_qualified'] is False
