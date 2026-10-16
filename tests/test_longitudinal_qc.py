"""Meaningful geometry and known-displacement checks for longitudinal QC."""

import nibabel as nib
import numpy as np
import pytest

from mri_rl.longitudinal_qc import check_physical_geometry, register_pair


def test_physical_geometry_rejects_singular_affine():
    image = nib.Nifti1Image(np.ones((8, 8, 8), dtype=np.float32), np.eye(4))
    image._affine = np.diag([1, 1, 0, 1])  # malformed in-memory input
    with pytest.raises(ValueError, match="Invalid physical affine"):
        check_physical_geometry(image)


def test_rigid_registration_recovers_known_translation(tmp_path):
    grid = np.indices((48, 48, 48), dtype=float)
    brain = (((grid[0] - 24) / 14) ** 2 + ((grid[1] - 23) / 12) ** 2 + ((grid[2] - 25) / 11) ** 2) < 1
    texture = 1 + 0.2 * np.sin(grid[0] / 3) + 0.15 * np.cos(grid[1] / 4)
    image = (brain * texture).astype(np.float32)
    fixed_affine = np.eye(4)
    moving_affine = np.eye(4)
    moving_affine[:3, 3] = [6, -5, 4]
    fixed = tmp_path / "fixed.nii.gz"
    moving = tmp_path / "moving.nii.gz"
    nib.save(nib.Nifti1Image(image, fixed_affine), fixed)
    nib.save(nib.Nifti1Image(image, moving_affine), moving)
    result = register_pair(fixed, moving, registration_mm=2, evaluation_mm=1.5)
    assert result["after"]["foreground_dice"] > result["before"]["foreground_dice"] + 0.08
    assert result["after"]["foreground_pearson"] > result["before"]["foreground_pearson"] + 0.1
    assert result["after"]["foreground_dice"] > 0.90
