"""Four reviewer-demonstrated integrity bugs for mri_data_prep_v1 (synthetic fixtures only; no patient data).

Adapted from the reviewer RED fixture: same four cases, imports the package module, and the mask test uses the
current API key names (label_set_exact_preserved) with stronger assertions.
"""
import json

import nibabel as nib
import numpy as np
import pytest

from mri_rl import mri_data_prep_v1 as dp


# 1. QC flag -> abstain propagation ---------------------------------------------------------
def test_low_contrast_candidate_must_abstain(tmp_path):
    v = np.broadcast_to(np.arange(1, 11, dtype=np.float32)[None, None, :], (12, 12, 10)).copy()
    aff = np.diag([-1.0, 1.0, 1.0, 1.0])
    img = nib.Nifti1Image(v, aff)
    img.header.set_xyzt_units("mm")
    img.set_qform(aff, 1)
    img.set_sform(aff, 1)
    path = tmp_path / "synthetic.nii.gz"
    nib.save(img, path)
    m = {"schema": dp.MANIFEST_SCHEMA, "legacy_levels": [2, 4, 6],
         "rows": [{"opaque_id": "r1", "modality": "flair", "nifti_path": str(path), "pair_key": "k1", "split": "train"}]}
    dp.run_pilot(m, tmp_path / "private", tmp_path / "public")
    rows = json.loads((tmp_path / "public/pilot_public_aggregate.json").read_text())["per_row_opaque"]["r1"]["candidate_primary"]
    assert len(rows) == 3
    assert all("LOW_CONTRAST" in r["qc_flags"] for r in rows)
    assert all(r["abstain"] is True for r in rows)
    # reason is retained, and the pre-pixel (selection-stage) abstain state is not silently overwritten
    assert all("LOW_CONTRAST" in r["abstain_reasons"] for r in rows)
    assert all(r["abstain_before_pixel_qc"] is False for r in rows)


def test_clean_candidate_does_not_abstain_and_has_no_reasons(tmp_path):
    rng = np.random.default_rng(1)
    v = np.zeros((30, 40, 24), np.float32)
    x, y, z = np.indices(v.shape)
    inside = ((x - 15) / 14) ** 2 + ((y - 20) / 19) ** 2 + ((z - 12) / 11) ** 2 < 1
    v[inside] = 100 + 400 * rng.random(inside.sum())
    aff = np.diag([-1.0, 1.0, 1.0, 1.0])
    img = nib.Nifti1Image(v, aff)
    img.header.set_xyzt_units("mm")
    img.set_qform(aff, 1)
    img.set_sform(aff, 1)
    p = tmp_path / "ok.nii.gz"
    nib.save(img, p)
    m = {"schema": dp.MANIFEST_SCHEMA, "legacy_levels": [8, 12, 16],
         "rows": [{"opaque_id": "r1", "modality": "flair", "nifti_path": str(p), "pair_key": "k1", "split": "train"}]}
    dp.run_pilot(m, tmp_path / "private", tmp_path / "public")
    rows = json.loads((tmp_path / "public/pilot_public_aggregate.json").read_text())["per_row_opaque"]["r1"]["candidate_primary"]
    for r in rows:
        assert r["qc_flags"] == [] and r["abstain"] is False and r["abstain_reasons"] == []


# 2. outside-support lateral / presence must abstain ------------------------------------------
def _support():
    s = np.zeros((20, 20), bool)
    s[:, 6:14] = True
    return s


def test_outside_support_lateral_mask_must_abstain():
    mask = np.zeros((20, 20), bool)
    mask[5:15, :4] = True
    r = dp.verify_lateral_side(mask, _support(), "L")
    assert r["answer"] is None and r["reason"] == "MASK_OUTSIDE_SUPPORT"


def test_mostly_outside_support_lateral_abstains_but_inside_mask_answers():
    s = _support()
    mostly_out = np.zeros_like(s)
    mostly_out[5:15, 2:4] = True
    mostly_out[5:7, 12:14] = True  # ~10% inside
    assert dp.verify_lateral_side(mostly_out, s, "L")["answer"] is None
    inside = np.zeros_like(s)
    inside[5:15, 11:14] = True
    assert dp.verify_lateral_side(inside, s, "L")["answer"] in ("left", "right")


def test_outside_support_lesion_presence_must_abstain():
    s = _support()
    out = np.zeros_like(s)
    out[5:15, :4] = True
    r = dp.verify_lesion_presence(out, 1.0, min_area_mm2=5.0, support2d=s)
    assert r["answer"] is None and r["reason"] == "MASK_OUTSIDE_SUPPORT" and r["support_checked"] is True
    inside = np.zeros_like(s)
    inside[5:15, 8:12] = True
    assert dp.verify_lesion_presence(inside, 1.0, min_area_mm2=5.0, support2d=s)["answer"] == "present"
    assert dp.verify_lesion_presence(np.zeros_like(s), 1.0, 5.0, support2d=s)["answer"] == "absent"
    # nonzero mask with an empty support cannot be checked -> abstain
    assert dp.verify_lesion_presence(inside, 1.0, 5.0, support2d=np.zeros_like(s))["answer"] is None


def test_lesion_presence_without_support_is_flagged_unchecked():
    m = np.zeros((20, 20), bool)
    m[5:15, 8:12] = True
    r = dp.verify_lesion_presence(m, 1.0, min_area_mm2=5.0)
    assert r["support_checked"] is False


# 3. hidden metadata must not overwrite explicit truth ------------------------------------------
@pytest.mark.parametrize("key", ["truth", "TRUTH", "opaque_id", "prompt_visible", "hidden_metadata"])
def test_hidden_metadata_reserved_keys_rejected(key):
    with pytest.raises(ValueError, match="reserved"):
        dp.bind_source_label("r1", "Choose a class", ["A", "B"], "A", {key: "B"})


def test_bound_truth_preserved_with_ordinary_hidden_metadata():
    r = dp.bind_source_label("r1", "Choose a class", ["A", "B"], "A", {"source_key": "zzq-77"})
    assert r["hidden_metadata"]["truth"] == "A" and r["hidden_metadata"]["source_key"] == "zzq-77"
    assert r["opaque_id"] == "r1" and r["prompt_visible"] == {"prompt": "Choose a class", "options": ["A", "B"]}


# 4. mask entirely cropped by the image grid is ineligible -------------------------------------
def test_mask_outside_entire_fov_not_preserved_and_ineligible():
    vol = np.ones((8, 8, 8), np.float32)
    mask = np.zeros_like(vol, dtype=np.uint8)
    mask[2:4, 2:4, 2:4] = 1
    ma = np.eye(4)
    ma[0, 3] = 100
    r = dp.check_image_mask_geometry(vol, np.eye(4), mask, ma, {0, 1})
    assert r["mask_nonzero_voxels"] == 0
    assert r["label_set_before"] == [0, 1] and r["label_set_after"] == [0]
    assert r["label_set_exact_preserved"] is False
    assert r["missing_labels"] == [1]
    assert "MASK_OUTSIDE_IMAGE_SUPPORT" in r["flags"] and "LABEL_MISSING_AFTER_RESAMPLE" in r["flags"]
    assert r["mask_eligible"] is False and "ENTIRE_NONEMPTY_MASK_LOST" in r["ineligible_reasons"]


def test_exact_mask_is_eligible():
    vol = np.ones((8, 8, 8), np.float32)
    mask = np.zeros_like(vol, dtype=np.uint8)
    mask[2:4, 2:4, 2:4] = 1
    r = dp.check_image_mask_geometry(vol, np.eye(4), mask, np.eye(4), {0, 1})
    assert r["mask_eligible"] is True and r["ineligible_reasons"] == [] and r["label_set_exact_preserved"] is True


def test_tiny_label_disappearance_is_ineligible():
    vol = np.ones((8, 8, 8), np.float32)
    mask = np.zeros_like(vol, dtype=np.uint8)
    mask[1:5, 1:5, 1:5] = 1
    mask[7, 7, 7] = 2  # one-voxel label that the shifted grid drops
    ma = np.eye(4)
    ma[0, 3] = 1.0
    r = dp.check_image_mask_geometry(vol, np.eye(4), mask, ma, {0, 1, 2})
    assert 2 in r["missing_labels"] and r["mask_eligible"] is False
