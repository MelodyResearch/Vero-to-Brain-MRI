"""Root review findings 1-7 for mri_data_prep_v1 (synthetic fixtures only; no private data)."""
import json

import nibabel as nib
import numpy as np
import pytest
from PIL import Image

from mri_rl import mri_data_prep_v1 as dp


def las_affine(shape=(30, 40, 24), z0=-10.0):
    a = np.diag([-1.0, 1.0, 1.0, 1.0])
    a[:3, 3] = [shape[0] / 2, -shape[1] / 2, z0]
    return a


def brain_volume(shape=(30, 40, 24), seed=0):
    rng = np.random.default_rng(seed)
    v = np.zeros(shape, dtype=np.float32)
    x, y, z = np.indices(shape)
    r = ((x - 15) / 11) ** 2 + ((y - 20) / 16) ** 2 + ((z - 12) / 9) ** 2
    v[r < 1] = 100 + 50 * rng.random((r < 1).sum())
    return v


def write(tmp_path, name, data, affine, units="mm"):
    img = nib.Nifti1Image(data, affine)
    img.set_qform(affine, code=1)
    img.set_sform(affine, code=1)
    if units:
        img.header.set_xyzt_units(units)
    p = tmp_path / name
    nib.save(img, str(p))
    return p


def rot_affine(deg=20, axis="z"):
    th = np.deg2rad(deg)
    c, s = np.cos(th), np.sin(th)
    r = {"z": [[c, -s, 0], [s, c, 0], [0, 0, 1]], "y": [[c, 0, s], [0, 1, 0], [-s, 0, c]]}[axis]
    a = las_affine()
    a[:3, :3] = np.array(r) @ a[:3, :3]
    return a


def pair_manifest(tmp_path, a, b, levels=(8, 12), aff_b=None, extra_a=None):
    f = write(tmp_path, "a.nii.gz", a, las_affine())
    t = write(tmp_path, "b.nii.gz", b, aff_b if aff_b is not None else las_affine())
    rows = [{"opaque_id": "r001", "modality": "flair", "nifti_path": str(f), "pair_key": "k1", "split": "train", **(extra_a or {})},
            {"opaque_id": "r002", "modality": "t1ce", "nifti_path": str(t), "pair_key": "k1", "split": "train"}]
    return {"schema": dp.MANIFEST_SCHEMA, "legacy_levels": list(levels), "rows": rows}


def public_rows(tmp_path, sub="pub"):
    return json.loads((tmp_path / sub / "pilot_public_aggregate.json").read_text())["per_row_opaque"]


# ---- 2: geometry fail-closed ---------------------------------------------
def test_axis_aligned_precondition_enforced_on_oblique_and_offdiagonal():
    v = brain_volume()
    assert dp.is_axis_aligned(las_affine()) is True
    for a in (rot_affine(20, "z"), rot_affine(10, "y")):
        assert dp.is_axis_aligned(a) is False
        with pytest.raises(ValueError):
            dp.select_slices_coverage(v, a)
        with pytest.raises(ValueError):
            dp.axial_display_plane(v, a, 5)
        with pytest.raises(ValueError):
            dp.offsets_2p5d(v, a, 5, (0,))
    shear = las_affine()
    shear[2, 0] = 0.5  # world z depends on i as well
    assert dp.is_axis_aligned(shear) is False


def test_geometry_eligibility_blocks_each_unsupported_class(tmp_path):
    good = dp.inspect_nifti(write(tmp_path, "g.nii.gz", brain_volume(), las_affine()))
    assert dp.geometry_eligibility(good) == {"eligible": True, "reasons": []}
    unk = dp.inspect_nifti(write(tmp_path, "u.nii.gz", brain_volume(), las_affine(), units=None))
    assert "UNITS_UNKNOWN" in dp.geometry_eligibility(unk)["reasons"]
    obl = dp.inspect_nifti(write(tmp_path, "o.nii.gz", brain_volume(), rot_affine()))
    e = dp.geometry_eligibility(obl)
    assert e["eligible"] is False and "OBLIQUE_OR_SINGULAR" in e["reasons"]
    ras = dp.inspect_nifti(write(tmp_path, "r.nii.gz", brain_volume(), np.diag([1.0, 1, 1, 1])))
    assert "NOT_LAS_STORAGE" in dp.geometry_eligibility(ras)["reasons"]
    img = nib.Nifti1Image(brain_volume(), las_affine())
    img.set_sform(las_affine(), code=1)
    q = las_affine()
    q[0, 3] += 5
    img.set_qform(q, code=1)
    img.header.set_xyzt_units("mm")
    nib.save(img, str(tmp_path / "c.nii.gz"))
    assert "QFORM_SFORM_CONFLICT" in dp.geometry_eligibility(dp.inspect_nifti(tmp_path / "c.nii.gz"))["reasons"]


def test_pilot_fails_closed_no_physical_candidate_for_oblique(tmp_path):
    man = pair_manifest(tmp_path, brain_volume(), brain_volume(seed=3), aff_b=rot_affine())
    res = dp.run_pilot(man, tmp_path / "priv", tmp_path / "pub")
    rows = public_rows(tmp_path)
    bad = rows["r002"]
    assert bad["technical_ineligible"] is True and "OBLIQUE_OR_SINGULAR" in bad["ineligible_reasons"]
    for key in ("legacy", "candidate_primary", "candidate_independent_exploratory", "selection", "canonicalization"):
        assert key not in bad
    assert not list((tmp_path / "priv" / "images").glob("r002*"))
    assert rows["r001"]["technical_ineligible"] is False and res["ineligible_rows"] == 1
    assert res["pair_geometry"]["k1"]["paired_selection"] is None  # a pair needs two eligible rows


# ---- 1: legacy PNG binding ----------------------------------------------------
def test_legacy_self_determinism_is_not_png_preservation(tmp_path):
    man = pair_manifest(tmp_path, brain_volume(), brain_volume(seed=2) * 1.7, levels=(8, 12, 16))
    res = dp.run_pilot(man, tmp_path / "p1", tmp_path / "u1")
    assert "legacy_control_pixels_unchanged" not in res
    assert res["legacy_render_self_deterministic"] is True
    assert res["existing_legacy_png_verification"]["status"] == "NOT_SUPPLIED"
    img, _ = dp.legacy_render(brain_volume(), 12)
    good = tmp_path / "good.png"
    img.save(good)
    arr = np.array(img)
    arr[5, 5, :] = (arr[5, 5, :].astype(int) + 1) % 256
    badp = tmp_path / "bad.png"
    Image.fromarray(arr).save(badp)
    man["rows"][0]["legacy_png"] = {"12": str(good)}
    res = dp.run_pilot(man, tmp_path / "p2", tmp_path / "u2")
    assert res["existing_legacy_png_verification"] == {"status": "ALL_EXACT", "compared": 1, "exact": 1, "mismatched": 0}
    man["rows"][0]["legacy_png"] = {"12": str(badp)}
    res = dp.run_pilot(man, tmp_path / "p3", tmp_path / "u3")
    v = res["existing_legacy_png_verification"]
    assert v["status"] == "MISMATCH" and v["mismatched"] == 1
    assert str(tmp_path) not in (tmp_path / "u3" / "pilot_public_aggregate.json").read_text()
    man["rows"][0]["legacy_png"] = {"99": str(good)}  # level not among legacy_levels
    with pytest.raises(ValueError):
        dp.run_pilot(man, tmp_path / "p4", tmp_path / "u4")


# ---- 3: mask labels -----------------------------------------------------------
def test_mask_fractional_and_nonfinite_labels_rejected_not_cast():
    v, aff = brain_volume(), las_affine()
    m = np.zeros(v.shape, np.float32)
    m[10:14, 15:20, 8:12] = 1.5
    with pytest.raises(ValueError):
        dp.check_image_mask_geometry(v, aff, m, aff, allowed_labels={0, 1})
    m2 = np.zeros(v.shape, np.float32)
    m2[10, 10, 10] = np.nan
    with pytest.raises(ValueError):
        dp.check_image_mask_geometry(v, aff, m2, aff, allowed_labels={0, 1})
    m3 = np.zeros(v.shape, np.float32)
    m3[10:14, 15:20, 8:12] = 2.0  # integer-valued float is accepted
    r = dp.check_image_mask_geometry(v, aff, m3, aff, allowed_labels={0, 2})
    assert r["label_set_after"] == [0, 2] and r["label_set_exact_preserved"] is True


def test_resample_label_report_distinguishes_novel_missing_and_background():
    v, aff = brain_volume(), las_affine()
    m = np.ones((10, 10, 10), np.uint8)
    maff = aff.copy()
    maff[:3, 3] += [2, 2, 2]
    r = dp.check_image_mask_geometry(v, aff, m, maff, allowed_labels={0, 1})
    assert r["label_set_before"] == [1] and r["label_set_after"] == [0, 1]
    assert r["background_zero_introduced"] is True and r["novel_nonzero_labels"] == []
    assert r["label_set_exact_preserved"] is False and r["missing_labels"] == []
    fine = np.zeros((60, 80, 48), np.uint8)
    fine[1, 1, 1] = 4
    faff = aff.copy()
    faff[:3, :3] = aff[:3, :3] * 0.5
    r2 = dp.check_image_mask_geometry(v, aff, fine, faff, allowed_labels={0, 4})
    assert r2["missing_labels"] == [4] and r2["label_set_exact_preserved"] is False
    assert "LABEL_MISSING_AFTER_RESAMPLE" in r2["flags"]


def test_pilot_mask_requires_pinned_ontology(tmp_path):
    mk = write(tmp_path, "m.nii.gz", np.zeros((30, 40, 24), np.uint8), las_affine())
    man = pair_manifest(tmp_path, brain_volume(), brain_volume(seed=2), extra_a={"mask_path": str(mk)})
    with pytest.raises(ValueError):
        dp.run_pilot(man, tmp_path / "p", tmp_path / "u")


# ---- 5: RAS mapping in the pilot ------------------------------------------------
def test_pilot_reports_ras_canonicalization_and_mapped_legacy_slice(tmp_path):
    man = pair_manifest(tmp_path, brain_volume(), brain_volume(seed=2) * 1.7)
    dp.run_pilot(man, tmp_path / "priv", tmp_path / "pub")
    row = public_rows(tmp_path)["r001"]
    c = row["canonicalization"]
    assert c["new_axcodes"] == "RAS" and c["max_world_error_mm"] < 1e-6 and c["verified_samples"] >= 8
    assert "original_affine_preserved" not in c
    for leg in row["legacy"]:
        assert isinstance(leg["ras_canonical_slice_index"], int)
        assert leg["display_plane_equal_after_canonicalization"] is True
    priv = json.loads((tmp_path / "priv" / "full_geometry_PRIVATE.json").read_text())
    assert priv["rows"]["r001"]["affine"] and priv["rows"]["r001"]["ras_affine"]


# ---- 6: paired selection --------------------------------------------------------
def test_paired_selection_uses_one_reference_and_maps_world_z():
    v = brain_volume()
    ref = dp.select_slices_coverage(v, las_affine(z0=-10))
    p = dp.select_paired_slices(v, las_affine(z0=-10), v, las_affine(z0=-12))  # partner starts 2 mm lower
    for r, q in zip(ref["slices"], p["partner"]):
        assert q["index"] == r["index"] + 2 and q["world_z_residual_mm"] == pytest.approx(0)
    assert p["reference_world_z_mm"] == [s["world_z_mm"] for s in ref["slices"]]
    assert p["correspondence"] == "SAME_WORLD_Z_FROM_REFERENCE_MODALITY"


def test_independent_selection_difference_reported_as_non_corresponding():
    a, b = brain_volume(), brain_volume()
    b[:, :, :6] = 0
    aff = las_affine()
    d = dp.independent_selection_difference(dp.select_slices_coverage(a, aff), dp.select_slices_coverage(b, aff))
    assert max(abs(x) for x in d["world_z_diff_mm"]) > 0 and d["corresponding"] is False


def test_partner_out_of_volume_abstains():
    short = brain_volume()[:, :, :10]
    p = dp.select_paired_slices(brain_volume(), las_affine(z0=-10), short, las_affine(z0=-10))
    assert any(q["out_of_volume"] and q["abstain"] for q in p["partner"])


def test_pilot_primary_candidates_are_paired_from_reference_modality(tmp_path):
    b = brain_volume(seed=2)
    b[:, :, :6] = 0
    man = pair_manifest(tmp_path, brain_volume(), b)
    res = dp.run_pilot(man, tmp_path / "priv", tmp_path / "pub")
    rows = public_rows(tmp_path)
    pr = [x["world_z_mm"] for x in rows["r001"]["candidate_primary"]]
    pt = [x["world_z_mm"] for x in rows["r002"]["candidate_primary"]]
    assert pr == pt
    ex = rows["r002"]["candidate_independent_exploratory"]
    assert [x["world_z_mm"] for x in ex] != pt and ex[0]["correspondence"] == "NOT_CORRESPONDING_ACROSS_MODALITIES"
    ps = res["pair_geometry"]["k1"]["paired_selection"]
    assert ps["reference_modality"] == "flair" and ps["independent_selection_difference"]["corresponding"] is False


# ---- 7: token metadata ----------------------------------------------------------
def test_token_estimate_metadata_vs_processor_observation():
    t = dp.visual_token_estimate((364, 436), n_images=3)
    assert t["per_image_tokens"] == 154 and t["total_tokens"] == 462
    assert t["kind"] == "METADATA_DERIVED_APPROXIMATE" and t["native_processor_observed"] is False
    assert t["resized_wh_px"] == [352, 448]
    assert t["min_pixels"] == 65536 and t["max_pixels"] == 16777216
    assert t["metadata_source"] == "ROOT_REPORTED_NOT_READ_BY_THIS_AGENT"
    one = dp.visual_token_estimate((364, 436), n_images=1)
    assert one["total_tokens"] == 154 and one["n_images"] == 1
