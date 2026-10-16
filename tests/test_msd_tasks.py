import json

import nibabel as nib
import numpy as np

from mri_rl.msd_tasks import build_tasks, central_slice_index, derive_targets, render_image_only_slice, validate_pair


def test_fixed_slice_does_not_search_for_mask_foreground():
    mask = np.zeros((10, 12, 9), dtype=np.uint8)
    mask[2:5, 3:6, 8] = 1
    assert central_slice_index(mask.shape) == 4
    assert derive_targets(mask, np.eye(4)) is None


def test_centroid_and_dominance_targets():
    mask = np.zeros((10, 12, 9), dtype=np.uint8)
    mask[2:5, 3:6, 4] = 1
    mask[5, 5, 4] = 2
    affine = np.diag([2.0, 3.0, 4.0, 1.0])
    result = derive_targets(mask, affine)
    assert result["slice_index_k"] == 4
    assert result["central_slice_counts"] == {"anterior": 9, "posterior": 1}
    assert result["dominant_region"] == "anterior"
    assert result["centroid_voxel_ij"] == [3, 4]
    assert np.allclose(result["centroid_world_mm"], [6.4, 12.3, 16.0])


def test_pair_geometry_and_mask_labels_are_enforced(tmp_path):
    image = np.ones((10, 12, 9), dtype=np.float32)
    mask = np.zeros_like(image)
    mask[2:5, 3:6, 4] = 1
    image_path, mask_path = tmp_path / "image.nii.gz", tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(image, np.eye(4)), image_path)
    nib.save(nib.Nifti1Image(mask, np.eye(4)), mask_path)
    loaded_image, loaded_mask, _ = validate_pair(image_path, mask_path)
    assert loaded_image.shape == loaded_mask.shape
    mask[1, 1, 1] = 3
    nib.save(nib.Nifti1Image(mask, np.eye(4)), mask_path)
    try:
        validate_pair(image_path, mask_path)
    except ValueError as exc:
        assert "unexpected segmentation label" in str(exc)
    else:
        raise AssertionError("invalid label accepted")


def test_render_depends_only_on_image_and_fixed_slice(tmp_path):
    image = np.arange(10 * 12 * 9, dtype=np.float32).reshape((10, 12, 9))
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    render_image_only_slice(image, 4, a)
    render_image_only_slice(image.copy(), 4, b)
    assert a.read_bytes() == b.read_bytes()


def test_build_writes_private_records_and_aggregate_report(tmp_path):
    root = tmp_path / "dataset"
    (root / "imagesTr").mkdir(parents=True)
    (root / "labelsTr").mkdir()
    image = np.arange(10 * 12 * 9, dtype=np.float32).reshape((10, 12, 9))
    mask = np.zeros_like(image)
    mask[2:5, 3:6, 4] = 1
    mask[5, 5, 4] = 2
    nib.save(nib.Nifti1Image(image, np.eye(4)), root / "imagesTr" / "x.nii.gz")
    nib.save(nib.Nifti1Image(mask, np.eye(4)), root / "labelsTr" / "x.nii.gz")
    (root / "dataset.json").write_text(json.dumps({
        "name": "Hippocampus", "numTraining": 1, "test": [],
        "training": [{"image": "./imagesTr/x.nii.gz", "label": "./labelsTr/x.nii.gz"}],
    }))
    private = tmp_path / "private"
    audit_path = tmp_path / "audit.md"
    audit = build_tasks(root, private, audit_path)
    assert audit["geometry_valid_pairs"] == 1
    assert audit["task_record_count"] == 2
    records = [json.loads(line) for line in (private / "tasks.jsonl").read_text().splitlines()]
    assert {record["task_id"] for record in records} == {
        "central_slice_hippocampal_centroid", "central_slice_anterior_posterior_dominance"
    }
    assert all("UNVERIFIED_PATIENT_IDENTITY" in record["split_status"] for record in records)
    assert "No patient-disjoint held-out split" in audit_path.read_text()
