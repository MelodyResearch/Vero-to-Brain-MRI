"""Finite Dallas task safeguards: provenance, folds, hidden targets, rendering and scoring."""
import copy
import hashlib
import importlib.util
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from PIL import Image


def module():
    spec = importlib.util.find_spec('mri_rl.visit_correspondence')
    assert spec is not None, 'Missing typed Dallas correspondence helper'
    from mri_rl import visit_correspondence
    return visit_correspondence


def plan_fixture():
    people = {}
    splits = {'train': ['secret-a', 'secret-b'], 'internal_dev': ['secret-c', 'secret-d'], 'test': ['secret-e', 'secret-f']}
    cycles = []
    for split, subjects in splits.items():
        cycles.append({'split': split, 'subjects': subjects, 'sex': 'f', 'age_gap': 2})
        for subject in subjects:
            images = [{'relative_path': f'{subject}/ses-wave{v}/anat/{subject}_T1w.nii.gz',
                       'sha256': hashlib.sha256(f'{subject}{v}'.encode()).hexdigest(),
                       'bytes': 1234, 'tr': .006, 'te': .002, 'manufacturer': 'Philips'} for v in (1, 2)]
            people[subject] = {'images': images, 'sex': 'f', 'age_wave1': 40, 'age_wave2': 44, 'interval_years': 4}
    return {'source': 'OpenNeuro ds004856 v1.3.0', 'splits_PRIVATE': splits,
            'matched_cycles_PRIVATE': cycles, 'selected_people_PRIVATE': people,
            'selected_people': 6, 'selected_expected_volumes': 12}


def test_source_equality_targets_balanced_no_same_image_and_cycle_cluster():
    m = module(); cases = m.build_cases(plan_fixture())
    assert {k: len(v) for k, v in cases.items()} == {'train': 4, 'internal_dev': 4, 'test': 4}
    for split, rows in cases.items():
        assert sorted(r['target'] for r in rows) == ['different', 'different', 'same', 'same']
        assert len({r['cluster_key'] for r in rows}) == 1
        for r in rows:
            assert r['target'] == ('same' if r['query_subject'] == r['candidate_subject'] else 'different')
            assert r['query']['sha256'] != r['candidate']['sha256']
            assert r['query_subject'] in plan_fixture()['splits_PRIVATE'][split]
            assert r['candidate_subject'] in plan_fixture()['splits_PRIVATE'][split]


@pytest.mark.parametrize('damage', ['cross_fold', 'duplicate_source', 'cycle_missing', 'wrong_source'])
def test_rejects_source_fold_or_cycle_corruption(damage):
    m = module(); p = plan_fixture()
    if damage == 'cross_fold': p['splits_PRIVATE']['test'][0] = 'secret-a'
    if damage == 'duplicate_source': p['selected_people_PRIVATE']['secret-a']['images'][1]['sha256'] = p['selected_people_PRIVATE']['secret-a']['images'][0]['sha256']
    if damage == 'cycle_missing': p['matched_cycles_PRIVATE'].pop()
    if damage == 'wrong_source': p['source'] = 'OpenNeuro ds007045 v2.0.1'
    with pytest.raises(ValueError): m.build_cases(p)


def test_typed_row_hides_source_ids_dates_and_gold_from_model_inputs(tmp_path):
    m = module(); case = m.build_cases(plan_fixture())['train'][0]
    image = tmp_path / (case['case_id'] + '.png'); Image.new('L', (24, 12)).save(image)
    row = m.make_record(case, image, 'a'*64, 'b'*64)
    visible = str(row['prompt']) + str(row['images']) + row['extra_info']['question']
    assert not any(x in visible for x in ('secret-', 'wave1', 'wave2', 'Philips', 'ses-'))
    assert row['data_source'] == 'OpenNeuro_ds004856'
    assert row['ability'] == row['extra_info']['reward_type'] == 'mri_visit_correspondence'
    assert row['reward_model']['ground_truth'] == case['target']
    assert 'answer' not in row['extra_info']
    m.validate_correspondence_row(row, expected_split='train')
    bad = copy.deepcopy(row); bad['reward_model']['ground_truth'] = 'different' if case['target']=='same' else 'same'
    with pytest.raises(ValueError, match='equality'): m.validate_correspondence_row(bad)
    with pytest.raises(ValueError, match='split'): m.validate_correspondence_row(row, expected_split='internal_dev')


def test_renderer_canonical_orientation_and_label_independent_geometry(tmp_path):
    m = module(); a = np.arange(12*14*16, dtype=np.float32).reshape(12, 14, 16) + 1
    p = tmp_path/'a.nii.gz'; q = tmp_path/'q.nii.gz'
    nib.save(nib.Nifti1Image(a, np.eye(4)), p)
    # Same world image stored L-A-S instead of R-A-S must render identically.
    affine = np.eye(4); affine[0,0] = -1; affine[0,3] = 11
    nib.save(nib.Nifti1Image(a[::-1], affine), q)
    im1, qc1 = m.render_volume(p); im2, qc2 = m.render_volume(q)
    assert im1.size == im2.size == (384,384)
    assert np.array_equal(np.array(im1), np.array(im2))
    assert qc1['canonical_orientation'] == qc2['canonical_orientation'] == ['R','A','S']
    assert len(qc1['tile_nonzero_fraction']) == 9
    assert qc1['finite_voxels'] is True
    paired = m.compose_pair(im1, im2)
    assert paired.size == (768,384)
    assert np.array_equal(np.array(paired)[:,:384], np.array(im1))
    assert np.array_equal(np.array(paired)[:,384:], np.array(im2))


def test_test_split_rejected_before_any_source_volume_access(tmp_path):
    m = module()
    with pytest.raises(ValueError, match='sealed'):
        m.render_split(plan_fixture(), 'test', tmp_path/'missing', tmp_path/'out', 'a'*64)
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('target,answer,want', [('same','same',1),('same','different',0),('different','different',1)])
def test_strict_reward_source_bound_and_format_preserved(target,answer,want):
    m=module(); response='<think>Compare anatomy.</think><answer>\\boxed{'+answer+'}</answer>'
    score=m.score_visit_correspondence('OpenNeuro_ds004856', response, target, {'reward_type':'mri_visit_correspondence'})
    assert score['accuracy']==want and score['format']==1 and score['score']==pytest.approx(.8*want+.2)
    assert m.score_visit_correspondence('OpenNeuro_ds004856', '\\boxed{'+answer+'}', target, {'reward_type':'mri_visit_correspondence'})['score']==0
    with pytest.raises(ValueError): m.score_visit_correspondence('OpenNeuro_ds007045',response,target,{'reward_type':'mri_visit_correspondence'})
    with pytest.raises(ValueError): m.score_visit_correspondence('OpenNeuro_ds004856',response,'yes',{'reward_type':'mri_visit_correspondence'})


def test_real_render_split_persists_exact_rows_and_is_idempotent(tmp_path):
    m=module(); p=plan_fixture(); source=tmp_path/'source'; out=tmp_path/'out'
    for i,person in enumerate(p['selected_people_PRIVATE'].values()):
        for v,item in enumerate(person['images']):
            a=np.arange(12*14*16,dtype=np.float32).reshape(12,14,16)+1+10*i+v
            path=source/item['relative_path'];path.parent.mkdir(parents=True,exist_ok=True)
            nib.save(nib.Nifti1Image(a,np.eye(4)),path)
            item['bytes']=path.stat().st_size;item['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    summary=m.render_split(p,'train',source,out,'a'*64)
    assert (summary['rows'],summary['source_volumes'],summary['people'],summary['cycles'])==(4,4,2,1)
    assert summary['blank_or_low_contrast_tile_warnings']==0
    assert not (out/'test.verl.jsonl').exists()
    first=(out/'train.verl.jsonl').read_bytes()
    assert m.render_split(p,'train',source,out,'a'*64)['manifest_sha256']==summary['manifest_sha256']
    assert first==(out/'train.verl.jsonl').read_bytes()
    # A changed known source cannot silently become a new render at the same frozen path.
    item=p['selected_people_PRIVATE']['secret-a']['images'][0]
    with (source/item['relative_path']).open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='payload changed'): m.render_split(p,'train',source,out,'a'*64)


def test_train_only_threshold_and_dev_fixed_scoring():
    m=module()
    result=m.fit_train_threshold([-2.,-1.,1.,2.],['different','different','same','same'])
    assert result['threshold']==0 and result['train_accuracy']==1
    assert m.fixed_threshold_metrics([-1.,1.],['same','different'],result['threshold'])['accuracy']==0
    assert m.fixed_threshold_metrics([1.,-1.],['same','different'],result['threshold'])['accuracy']==1
