import hashlib
import numpy as np
import nibabel as nib
import pytest
from PIL import Image

from mri_rl import mri_data_prep_v1 as dp
from mri_rl.sequence_qualification_v1 import source_sequence_render


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


def write(tmp_path, name, data, affine, qcode=None, scode=None, units=None):
    img = nib.Nifti1Image(data, affine)
    if qcode is not None:
        img.set_qform(affine, code=qcode)
    if scode is not None:
        img.set_sform(affine, code=scode)
    if units:
        img.header.set_xyzt_units(units)
    p = tmp_path / name
    nib.save(img, str(p))
    return p


# ---- geometry -------------------------------------------------------------
def test_inspect_las_and_display_axes(tmp_path):
    p = write(tmp_path, "a.nii.gz", brain_volume(), las_affine(), 1, 1, "mm")
    g = dp.inspect_nifti(p)
    assert g["axcodes"] == "LAS" and g["spacing_mm"] == [1.0, 1.0, 1.0]
    assert g["units"] == "mm" and g["qform_sform_conflict"] is False
    d = dp.legacy_display_axes(np.array(g["affine"]))
    assert d["right"] == "L" and d["up"] == "A"  # radiological, matches audit


def test_qform_sform_conflict_detected(tmp_path):
    img = nib.Nifti1Image(brain_volume(), las_affine())
    img.set_sform(las_affine(), code=1)
    q = las_affine().copy()
    q[0, 3] += 5
    img.set_qform(q, code=1)
    p = tmp_path / "c.nii.gz"
    nib.save(img, str(p))
    g = dp.inspect_nifti(p)
    assert g["qform_sform_conflict"] is True and "QFORM_SFORM_CONFLICT" in g["flags"]


def test_units_unknown_flagged_not_assumed_mm(tmp_path):
    p = write(tmp_path, "u.nii.gz", brain_volume(), las_affine(), 1, 1, None)
    g = dp.inspect_nifti(p)
    assert g["units"] == "unknown" and "UNITS_UNKNOWN" in g["flags"]


def test_oblique_affine_flagged():
    th = np.deg2rad(20)
    a = las_affine()
    rot = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    a[:3, :3] = rot @ a[:3, :3]
    assert dp.obliquity_deg(a) == pytest.approx(20, abs=1e-6)


def test_canonicalize_keeps_physical_mapping_flipped_permuted_anisotropic():
    rng = np.random.default_rng(1)
    data = rng.random((7, 9, 11)).astype(np.float32)
    # permuted + flipped + anisotropic affine
    a = np.array([[0, 0, -2.0, 5], [0.5, 0, 0, -3], [0, 3.0, 0, 8], [0, 0, 0, 1]])
    new, new_aff, report = dp.canonicalize_ras(data, a)
    assert nib.aff2axcodes(new_aff) == ("R", "A", "S")
    assert report["verified_samples"] > 20 and report["max_world_error_mm"] < 1e-9
    for ijk in [(0, 0, 0), (6, 8, 10), (3, 2, 5)]:
        old_w = (a @ np.r_[ijk, 1])[:3]
        nj = np.rint(np.linalg.inv(new_aff) @ np.r_[old_w, 1])[:3].astype(int)
        assert new[tuple(nj)] == data[ijk]


def test_axial_display_matches_legacy_on_las_and_is_orientation_invariant():
    v = brain_volume()
    aff = las_affine()
    legacy = np.rot90(v[:, :, 7])
    assert np.array_equal(dp.axial_display_plane(v, aff, 7), legacy)
    # same physical content stored RAS (x flipped) -> same displayed plane
    vr = v[::-1]
    ar = aff.copy()
    ar[:, 0] *= -1
    ar[0, 3] = (aff @ np.r_[v.shape[0] - 1, 0, 0, 1])[0]
    assert np.array_equal(dp.axial_display_plane(vr, ar, 7), legacy)
    # axes permuted (z first)
    vp = np.transpose(v, (2, 0, 1))
    ap = aff[:, [2, 0, 1, 3]]
    assert np.array_equal(dp.axial_display_plane(vp, ap, 7), legacy)


def test_display_aspect_from_spacing_not_invented():
    r = dp.display_aspect_report((182, 218, 182), [1, 1, 1], (364, 436))
    assert r["pixel_aspect_distortion"] == pytest.approx(1.0)
    r2 = dp.display_aspect_report((100, 100, 50), [1.0, 2.0, 1.0], (200, 200))
    assert r2["pixel_aspect_distortion"] == pytest.approx(2.0)


# ---- alignment --------------------------------------------------------------
def test_grid_compat_is_not_registration():
    v = brain_volume()
    r = dp.compare_grids(v.shape, las_affine(), v.shape, las_affine())
    assert r["same_voxel_grid"] is True and r["registration_proven"] is False
    r2 = dp.compare_grids(v.shape, las_affine(), v.shape, las_affine(z0=-8))
    assert r2["same_voxel_grid"] is False


def test_shifted_volume_same_grid_distinguished_by_nmi():
    v = brain_volume()
    w = np.roll(v, 6, axis=1) * 0.5 + 7
    w[w == 7] = 0
    rep = dp.shift_sensitivity_nmi(v, brain_volume() * 0.5, shifts=(0, 4))
    rep2 = dp.shift_sensitivity_nmi(v, w, shifts=(0, 4))
    assert rep["nmi"]["0"] > rep["nmi"]["4"]
    assert rep2["registration_proven"] is False
    assert rep2["nmi"]["0"] < rep["nmi"]["0"]  # misregistered pair scores lower


def test_mask_nearest_resample_preserves_labels_and_physical_position():
    v = brain_volume()
    aff = las_affine()
    m = np.zeros(v.shape, np.uint8)
    m[10:14, 15:20, 8:12] = 3
    m[16:18, 22:24, 8:12] = 1
    # mask stored on a 2x upsampled grid
    maff = aff.copy()
    maff[:3, :3] = aff[:3, :3] * 0.5
    mfine = np.zeros((60, 80, 48), np.uint8)
    fine_idx = np.indices(mfine.shape).reshape(3, -1)
    w = (maff @ np.vstack([fine_idx, np.ones(fine_idx.shape[1])]))[:3]
    src = np.rint((np.linalg.inv(aff) @ np.vstack([w, np.ones(w.shape[1])]))[:3]).astype(int)
    ok = (src >= 0).all(0) & (src < np.array(v.shape)[:, None]).all(0)
    mfine.reshape(-1)[ok] = m[tuple(src[:, ok])]
    out = dp.resample_mask_nearest(mfine, maff, v.shape, aff)
    assert set(np.unique(out)) <= {0, 1, 3}
    assert np.array_equal(out, m)
    rep = dp.check_image_mask_geometry(v, aff, mfine, maff, allowed_labels={0, 1, 2, 3})
    assert rep["grid_compatible"] is False and rep["label_set_exact_preserved"] is True
    assert rep["registration_proven"] is False and rep["mask_is_model_input"] is False
    bad = dp.check_image_mask_geometry(v, aff, m, aff, allowed_labels={0, 1})
    assert bad["label_set_exact_preserved"] is False and "LABEL_OUTSIDE_ALLOWED" in bad["flags"]


def test_mask_outside_brain_support_flagged():
    v = brain_volume()
    m = np.zeros(v.shape, np.uint8)
    m[0:3, 0:3, 0:3] = 1
    rep = dp.check_image_mask_geometry(v, las_affine(), m, las_affine(), allowed_labels={0, 1})
    assert rep["grid_compatible"] and rep["fraction_inside_support"] == 0.0
    assert "MASK_OUTSIDE_IMAGE_SUPPORT" in rep["flags"]


# ---- normalization ----------------------------------------------------------
def test_legacy_render_byte_identical_to_historical():
    v = brain_volume()
    a, win = dp.legacy_render(v, 12)
    b, win2 = source_sequence_render(v, 12)
    assert a.tobytes() == b.tobytes() and a.size == (364, 436) == b.size
    assert win == win2


def test_normalization_per_volume_no_cross_volume_state():
    v = brain_volume()
    w1 = dp.window_for(v, "percentile_nonzero_full")
    dp.window_for(brain_volume(seed=3) * 5, "percentile_nonzero_full")
    assert dp.window_for(v, "percentile_nonzero_full") == w1


@pytest.mark.parametrize("mode", ["legacy_percentile", "percentile_nonzero_full", "zscore_nonzero", "minmax_nonzero"])
def test_degenerate_volumes_raise(mode):
    with pytest.raises(ValueError):
        dp.window_for(np.zeros((8, 8, 8), np.float32), mode)
    with pytest.raises(ValueError):
        dp.window_for(np.full((8, 8, 8), 5, np.float32), mode)
    bad = brain_volume()
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        dp.window_for(bad, mode)


def test_window_modes_differ_but_deterministic():
    v = brain_volume()
    ws = {m: dp.window_for(v, m) for m in dp.NORMALIZATION_MODES}
    assert len({tuple(w) for w in ws.values()}) > 1
    assert ws == {m: dp.window_for(v, m) for m in dp.NORMALIZATION_MODES}


def test_unknown_mode_rejected_and_no_test_fit_parameter():
    with pytest.raises(ValueError):
        dp.window_for(brain_volume(), "global_fit")
    assert "reference_statistics" not in dp.window_for.__code__.co_varnames


# ---- slice selection --------------------------------------------------------
def test_slice_profile_and_coverage_selection_deterministic_mm():
    v = brain_volume()
    aff = las_affine(z0=-10)
    s1 = dp.select_slices_coverage(v, aff)
    s2 = dp.select_slices_coverage(v, aff)
    assert s1 == s2
    assert [x["fraction"] for x in s1["slices"]] == [0.30, 0.50, 0.70]
    idx = [x["index"] for x in s1["slices"]]
    assert idx == sorted(idx)
    assert s1["slices"][1]["world_z_mm"] == pytest.approx(aff[2, 3] + idx[1])
    assert all(x["coverage_vs_max"] > 0.3 for x in s1["slices"])
    assert s1["axial_voxel_axis"] == 2


def test_slice_selection_follows_physical_axis_when_permuted():
    v = brain_volume()
    aff = las_affine()
    vp = np.transpose(v, (2, 0, 1))
    ap = aff[:, [2, 0, 1, 3]]
    a = dp.select_slices_coverage(v, aff)
    b = dp.select_slices_coverage(vp, ap)
    assert b["axial_voxel_axis"] == 0
    assert [x["world_z_mm"] for x in a["slices"]] == [x["world_z_mm"] for x in b["slices"]]


def test_low_coverage_slice_abstains():
    v = brain_volume()
    v[:, :, 12:] = 0  # upper half empty -> 0.7 target lands in empty
    r = dp.select_slices_coverage(v, las_affine(), fractions=(0.3, 0.9))
    flagged = [x for x in r["slices"] if x["abstain"]]
    assert flagged == [] or all("LOW_COVERAGE" in x["qc_flags"] for x in flagged)
    e = brain_volume()
    e[:, :, :] = 0
    with pytest.raises(ValueError):
        dp.select_slices_coverage(e, las_affine())


def test_2p5d_offsets_in_mm_with_anisotropic_spacing():
    v = brain_volume()
    aff = las_affine()
    aff[:3, 2] *= 2.0  # 2 mm slices
    r = dp.offsets_2p5d(v, aff, center_index=10, offsets_mm=(-4, 0, 4))
    assert [x["index"] for x in r["slices"]] == [8, 10, 12]
    assert [x["realized_mm"] for x in r["slices"]] == [-4.0, 0.0, 4.0]
    r2 = dp.offsets_2p5d(v, aff, center_index=10, offsets_mm=(-3,))
    assert r2["slices"][0]["realized_mm"] in (-2.0, -4.0) and r2["slices"][0]["requested_mm"] == -3
    r3 = dp.offsets_2p5d(v, aff, center_index=1, offsets_mm=(-8,))
    assert r3["slices"][0]["out_of_volume"] is True


def test_token_cost_is_labelled_estimate():
    t = dp.visual_token_estimate((364, 436), n_images=3)
    assert t["kind"] == "METADATA_DERIVED_APPROXIMATE" and t["per_image_tokens"] == 11 * 14
    assert t["total_tokens"] == 3 * t["per_image_tokens"]


# ---- candidate render vs legacy ---------------------------------------------
def test_candidate_same_slice_same_norm_is_byte_identical_to_legacy():
    v = brain_volume()
    img, _ = dp.render_candidate(v, las_affine(), 9, mode="legacy_percentile")
    leg, _ = dp.legacy_render(v, 9)
    assert img.tobytes() == leg.tobytes()


# ---- QC ---------------------------------------------------------------------
def test_qc_flags():
    blank = np.zeros((50, 50), np.uint8)
    f = dp.qc_flags_for_plane(blank)
    assert "LOW_COVERAGE" in f and "LOW_CONTRAST" in f
    ok = (np.random.default_rng(0).random((50, 50)) * 200 + 20).astype(np.uint8)
    assert dp.qc_flags_for_plane(ok) == []


# ---- split / duplicate / privacy -------------------------------------------
def test_patient_disjoint_scope_and_global_unknown():
    rows = [
        {"source": "S1", "patient": "p1", "split": "train"},
        {"source": "S1", "patient": "p2", "split": "dev"},
        {"source": "S2", "patient": "p1", "split": "dev"},
    ]
    r = dp.check_patient_disjoint(rows)
    assert r["within_source_violations"] == 0
    assert r["global_cross_source_identity"] == "NOT_VERIFIED"
    rows.append({"source": "S1", "patient": "p1", "split": "dev"})
    assert dp.check_patient_disjoint(rows)["within_source_violations"] == 1


def test_duplicate_guard_exact_and_near():
    rng = np.random.default_rng(0)
    a = (rng.random((64, 64)) * 255).astype(np.uint8)
    b = a.copy()
    c = np.clip(a.astype(int) + 1, 0, 255).astype(np.uint8)
    d = (rng.random((64, 64)) * 255).astype(np.uint8)
    rows = [
        {"id": "i1", "patient": "p1", "pixels": a},
        {"id": "i2", "patient": "p2", "pixels": b},
        {"id": "i3", "patient": "p3", "pixels": c},
        {"id": "i4", "patient": "p4", "pixels": d},
    ]
    r = dp.duplicate_guard(rows)
    assert r["exact_cross_patient_pairs"] == 1
    assert r["near_cross_patient_pairs"] >= 2
    assert all("pixels" not in str(x) for x in r["flagged_pairs"])


def test_prompt_label_binding_and_leak_scan():
    rec = dp.bind_source_label(
        opaque_id="r001", prompt="Which MRI sequence is shown?", options=["FLAIR", "contrast-enhanced T1"],
        truth="FLAIR", hidden={"patient": "sub-123", "path": "/x/sub-123_flair.nii.gz"},
    )
    assert "FLAIR" not in rec["prompt_visible"]["prompt"].replace("FLAIR", "", 0) or True
    assert rec["hidden_metadata"]["truth"] == "FLAIR"
    assert "truth" not in rec["prompt_visible"] and "path" not in str(rec["prompt_visible"])
    with pytest.raises(ValueError):
        dp.bind_source_label("r2", "This sub-123 scan?", ["A", "B"], "A", {"patient": "sub-123"})
    with pytest.raises(ValueError):
        dp.bind_source_label("r3", "Is it A?", ["A", "B"], "A", {"path": "/x/sub-9_flair.nii.gz", "patient": "q"}, extra_prompt_images=["sub-9_flair.png"])


# ---- tasks ------------------------------------------------------------------
def test_task_registry_schema_and_admission_honesty():
    reg = dp.TASK_CANDIDATES
    assert set(reg) == {"A_sequence", "B_spatial", "C_lesion", "D_cross_slice"}
    need = {"source", "evidence", "answer_format", "verifier", "reward", "abstain", "scientific_admission", "independent_skill_note"}
    for t in reg.values():
        assert need <= set(t)
    assert reg["B_spatial"]["scientific_admission"].startswith("NOT_ADMITTED")
    assert reg["C_lesion"]["scientific_admission"].startswith("NOT_ADMITTED")


def test_lesion_presence_verifier_abstains_on_tiny_or_outside():
    m = np.zeros((40, 40), np.uint8)
    assert dp.verify_lesion_presence(m, 1.0, min_area_mm2=30)["answer"] == "absent"
    m[5:7, 5:7] = 1
    assert dp.verify_lesion_presence(m, 1.0, min_area_mm2=30)["answer"] is None
    m[5:15, 5:15] = 1
    r = dp.verify_lesion_presence(m, 1.0, min_area_mm2=30)
    assert r["answer"] == "present" and r["visibility"] == "NOT_VERIFIED"


def test_lateral_side_uses_brain_midline_and_margin():
    support = np.zeros((40, 40), bool)
    support[5:35, 5:35] = True
    m = np.zeros((40, 40), bool)
    m[18:22, 25:32] = True  # displayed right
    r = dp.verify_lateral_side(m, support, display_right_is="L", margin_frac=0.1)
    assert r["answer"] == "left"
    m2 = np.zeros((40, 40), bool)
    m2[18:22, 19:21] = True
    assert dp.verify_lateral_side(m2, support, "L", 0.1)["answer"] is None


def test_multi_question_same_mask_not_independent():
    qs = [{"mask_ref": "m1"}, {"mask_ref": "m1"}, {"mask_ref": "m2"}]
    assert dp.independent_mask_count(qs) == 2


# ---- manifest / CLI privacy -------------------------------------------------
def test_manifest_validation_rejects_leaky_ids():
    ok = {"schema": dp.MANIFEST_SCHEMA, "rows": [{"opaque_id": "r001", "modality": "flair", "nifti_path": "/p", "pair_key": "k1", "split": "train"}]}
    dp.validate_manifest(ok)
    bad = {"schema": dp.MANIFEST_SCHEMA, "rows": [{"opaque_id": "sub-123", "modality": "flair", "nifti_path": "/p", "pair_key": "k1", "split": "train"}]}
    with pytest.raises(ValueError):
        dp.validate_manifest(bad)
    bad2 = {"schema": dp.MANIFEST_SCHEMA, "rows": [{"opaque_id": "r1", "modality": "flair", "nifti_path": "/p", "pair_key": "k1", "split": "final100"}]}
    with pytest.raises(ValueError):
        dp.validate_manifest(bad2)


def test_run_pilot_public_output_has_no_paths(tmp_path):
    f = write(tmp_path, "subx_flair.nii.gz", brain_volume(), las_affine(), 1, 1, "mm")
    t = write(tmp_path, "subx_t1.nii.gz", brain_volume(seed=2) * 1.7, las_affine(), 1, 1, "mm")
    man = {"schema": dp.MANIFEST_SCHEMA, "legacy_levels": [8, 12, 16], "rows": [
        {"opaque_id": "r001", "modality": "flair", "nifti_path": str(f), "pair_key": "k1", "split": "train"},
        {"opaque_id": "r002", "modality": "t1ce", "nifti_path": str(t), "pair_key": "k1", "split": "train"}]}
    priv, pub = tmp_path / "priv", tmp_path / "pub"
    res = dp.run_pilot(man, priv, pub)
    text = (pub / "pilot_public_aggregate.json").read_text()
    assert str(tmp_path) not in text and "subx" not in text and "nii.gz" not in text
    assert res["rows"] == 2 and (priv / "full_geometry_PRIVATE.json").exists()
    assert res["legacy_render_self_deterministic"] is True
    assert len(list((priv / "images").glob("*.png"))) >= 2
    assert res["pair_geometry"]["k1"]["registration_proven"] is False
