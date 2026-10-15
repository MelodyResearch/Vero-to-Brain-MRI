import nibabel as nib
import numpy as np
import pytest

from mri_rl.volume import load_canonical_volume, voxel_to_world, world_to_voxel


def test_nifti_orientation_and_affine_roundtrip(tmp_path):
    volume = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    affine = np.diag([-2.0, 2.0, 3.0, 1.0])
    p = tmp_path / "scan.nii.gz"
    nib.save(nib.Nifti1Image(volume, affine), p)
    result = load_canonical_volume(p)
    assert result["orientation"] == ("R", "A", "S")
    assert result["spacing_mm"] == pytest.approx((2.0, 2.0, 3.0))
    point = (1, 2, 3)
    assert world_to_voxel(voxel_to_world(point, result["affine"]), result["affine"]) == pytest.approx(point)


def test_rejects_non_3d_or_nonfinite_affine(tmp_path):
    p = tmp_path / "bad.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((2, 2, 2, 2)), np.eye(4)), p)
    with pytest.raises(ValueError, match="3D"):
        load_canonical_volume(p)
