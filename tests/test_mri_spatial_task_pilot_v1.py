import json
import numpy as np
import nibabel as nib
import pytest
from PIL import Image
from mri_rl.dallas_spatial_qualification_v1 import render_slice

from mri_rl import mri_spatial_task_pilot_v1 as sp
from mri_rl import mri_data_prep_v1 as dp

SHAPE = (182, 218, 60)
LEVEL = 45


def las_affine(shape=SHAPE):
    a = np.diag([-1.0, 1.0, 1.0, 1.0])
    a[:3, 3] = [90.5, -108.5, -72.0]
    return a


def brain(seed=0, box=(slice(2, 180), slice(2, 216), slice(2, 58))):
    rng = np.random.default_rng(seed)
    v = np.zeros(SHAPE, np.float32)
    v[box] = 100 + 50 * rng.random(v[box].shape)
    return v


def save(path, data, aff=None, dtype=None):
    img = nib.Nifti1Image(data, las_affine() if aff is None else aff)
    img.set_qform(img.affine, code=1)
    img.set_sform(img.affine, code=1)
    img.header.set_xyzt_units("mm")
    nib.save(img, str(path))
    return path


def blob(i0, j0, rad, label=2, level=LEVEL, base=None):
    m = np.zeros(SHAPE, np.int16) if base is None else base
    x, y = np.indices(SHAPE[:2])
    m[((x - i0) ** 2 + (y - j0) ** 2 <= rad ** 2), level] = label
    return m


def paired_canvas(vol, other=None):
    """752x470 paired frame: FLAIR panel at (4,28), T1ce panel at (384,28); title strip left blank (no header text needed)."""
    c = Image.new("RGB", sp.PAIRED_CANVAS_SIZE)
    c.paste(render_slice(brain(seed=99) if other is None else other, LEVEL).convert("RGB"), (4, 28))
    c.paste(render_slice(vol, LEVEL).convert("RGB"), (384, 28))
    return c


def make_case(tmp_path, mask, name="c01", target=None, vol=None, png=True, prompt=None, aff=None, mask_aff=None, modality="t1ce"):
    vol = brain() if vol is None else vol
    f = save(tmp_path / f"{name}_img.nii.gz", vol, aff)
    mk = save(tmp_path / f"{name}_mask.nii.gz", mask, mask_aff)
    case = {"opaque_id": name, "pair_key": "p" + name, "source_order": 0, "split": "train", "legacy_slice_index": LEVEL,
            "image_nifti_path": str(f), "mask_nifti_path": str(mk), "original_ground_truth": target or "right-top",
            "image_modality": modality, "annotation_release_lineage": "NOT_REATTESTED"}
    if png:
        p = tmp_path / f"{name}.png"
        paired_canvas(vol).save(p)
        case["legacy_png_path"] = str(p)
        case["legacy_png_sha256"] = sp.dp.sha256_file(p)
    if prompt:
        case["visible_prompt_text"] = prompt
    return case


# ---- selection (before any voxel is read) ---------------------------------
def row(i, patient, ability="mri_tumor_grid", split="train"):
    return {"id": f"r{i}", "ability": ability, "extra_info": {"patient_key": patient, "internal_split": split}}


def test_selection_first_three_distinct_people_in_source_order():
    rows = [row(0, "a", "mri_sequence"), row(1, "a"), row(2, "a"), row(3, "b", split="internal_dev"),
            row(4, "b"), row(5, "c"), row(6, "d")]
    assert [r["id"] for r in sp.select_first_distinct_train_grid(rows, 3)] == ["r1", "r4", "r5"]
    with pytest.raises(ValueError):
        sp.select_first_distinct_train_grid(rows[:5], 3)


# ---- endpoint / abstain semantics ------------------------------------------
def test_clean_blob_is_arithmetic_clean_but_never_scientifically_admitted():
    m = sp.plane_report(np.rot90(blob(150, 190, 20)[:, :, LEVEL]), original_target="right-top")
    assert m["grid"] == "right-top" and m["abstain_reasons"] == []
    assert m["technical_status"] == "TECHNICAL_ARITHMETIC_CLEAN_EXPERT_PENDING"
    assert m["scientific_admitted"] == 0 and m["expert_status"] == "EXPERT_ADJUDICATION_REQUIRED"
    assert m["mask_derived_not_visible_truth"] is True


@pytest.mark.parametrize("mk,reason", [
    (lambda: blob(150, 190, 8), "SMALL_SOURCE_MASK"),
    (lambda: np.zeros(SHAPE, np.int16), "EMPTY_MASK_NOT_NEGATIVE_GOLD"),
    (lambda: blob(150, 190, 20, base=blob(30, 40, 20)), "MULTICOMPONENT_2D"),
])
def test_abstain_reasons_retained(mk, reason):
    r = sp.plane_report(np.rot90(mk()[:, :, LEVEL]), original_target="right-top")
    assert reason in r["abstain_reasons"] and r["technical_status"] == "ABSTAIN"
    assert r["scientific_admitted"] == 0


def test_boundary_and_off_mask_centroid_flagged():
    near = sp.plane_report(np.rot90(blob(120, 190, 20)[:, :, LEVEL]), original_target="right-top")
    assert "CENTROID_NEAR_GRID_BOUNDARY" in near["abstain_reasons"]
    m = np.zeros(SHAPE, np.int16)
    x, y = np.indices(SHAPE[:2])
    ring = ((x - 91) ** 2 + (y - 109) ** 2 <= 45 ** 2) & ((x - 91) ** 2 + (y - 109) ** 2 >= 35 ** 2)
    m[ring, LEVEL] = 2
    r = sp.plane_report(np.rot90(m[:, :, LEVEL]), original_target="center-middle")
    assert "CENTROID_OFF_MASK" in r["abstain_reasons"]


def test_display_frame_is_explicit_and_radiological():
    m = np.rot90(blob(160, 200, 20)[:, :, LEVEL])
    assert sp.plane_report(m, "right-top")["grid"] == "right-top"
    assert sp.plane_report(np.rot90(blob(20, 20, 20)[:, :, LEVEL]), "left-bottom")["grid"] == "left-bottom"
    f = sp.display_frame(las_affine())
    assert f["display_right_is_anatomical"] == "Left" and f["display_left_is_anatomical"] == "Right"
    assert f["display_up_is_anatomical"] == "Anterior"


def test_exact_third_boundary_belongs_to_right_bottom_cell():
    # 3x3 panel of native pixels; centroid edge coordinate exactly on the first vertical third.
    m = np.zeros((18, 18), bool)
    m[8:10, 5:7] = True  # centroid_x idx 5.5 -> edge 2*5.5+1=12 = 2*18/3
    r = sp.plane_report(m, "center-middle", require_display_scale=False, min_area=1)
    assert r["union"]["union_centroid_display_edge_xy"][0] == 12 and r["grid"].startswith("center")


def test_union_and_numeric3_component_are_separate_endpoints():
    mask = blob(150, 190, 20, label=2)
    mask = blob(100, 60, 6, label=3, base=mask)
    r = sp.plane_report(np.rot90(mask[:, :, LEVEL]), "right-top", label3_semantics="NOT_VERIFIED")
    assert r["union"]["area_native_pixels"] > r["numeric3_component"]["area_native_pixels"] > 0
    assert r["numeric3_component"]["admitted"] is False
    assert r["numeric3_component"]["label3_semantics"] == "NOT_VERIFIED"
    assert "MULTICOMPONENT_2D" in r["abstain_reasons"]
    assert r["numeric3_component"]["grid"] is not None and r["union"]["components_8"] == 2
    assert r["visible_focus_center"] == "NOT_ADJUDICATED"


def test_original_target_disagreement_retained_not_rewritten():
    r = sp.plane_report(np.rot90(blob(150, 190, 20)[:, :, LEVEL]), original_target="left-bottom")
    assert "ORIGINAL_TARGET_DISAGREES_WITH_ARITHMETIC" in r["abstain_reasons"]
    assert r["original_target"] == "left-bottom" and r["grid"] == "right-top"


def test_convention_sensitive_grid_flagged():
    # mean idx 5.5 of 18: legacy 2*mean=11 -> left; edge 2*mean+1=12 sits exactly on the third -> center (right/bottom rule)
    m = np.zeros((18, 18), bool)
    m[8:10, 5:7] = True
    r = sp.plane_report(m, "center-middle", require_display_scale=False, min_area=1)
    assert r["union"]["legacy_grid"].startswith("left") and r["grid"].startswith("center")
    assert "CONVENTION_SENSITIVE_GRID" in r["abstain_reasons"]


# ---- case-level processing ---------------------------------------------------
def test_process_case_clean_end_to_end(tmp_path):
    case = make_case(tmp_path, blob(150, 190, 20))
    pr, pub = sp.process_case(case)
    assert pr["geometry"]["image_mask_grid_identical"] is True and pr["resampled"] is False
    assert pr["legacy_png"] == "EXACT_PANEL" and pr["technical_status"] == "TECHNICAL_ARITHMETIC_CLEAN_EXPERT_PENDING"
    assert pr["mask_is_model_input"] is False and pr["world"]["plane_world_z_mm"] == -27.0
    assert pr["verifier"]["synthetic_response_fixtures_only"] is True and pr["verifier"]["all_ok"] is True
    assert "image_nifti_path" not in json.dumps(pub) and str(tmp_path) not in json.dumps(pub)


def test_unknown_label_holds(tmp_path):
    m = blob(150, 190, 20, label=4)
    pr, _ = sp.process_case(make_case(tmp_path, m))
    assert "UNKNOWN_ONTOLOGY_LABEL" in pr["abstain_reasons"] and pr["technical_status"] == "ABSTAIN"


def test_fractional_mask_holds_not_cast(tmp_path):
    m = blob(150, 190, 20).astype(np.float32)
    m[0, 0, 0] = 0.5
    pr, _ = sp.process_case(make_case(tmp_path, m))
    assert "MASK_NONINTEGER_LABELS" in pr["abstain_reasons"] and "grid" not in pr


def test_grid_mismatch_holds_and_never_resamples(tmp_path):
    a = las_affine()
    a[0, 3] += 3.0
    pr, _ = sp.process_case(make_case(tmp_path, blob(150, 190, 20), mask_aff=a))
    assert "IMAGE_MASK_GRID_MISMATCH_NO_RESAMPLE" in pr["abstain_reasons"]
    assert pr["resampled"] is False and pr["technical_status"] == "ABSTAIN"


def test_non_las_geometry_holds_no_pixels(tmp_path):
    ras = np.diag([1.0, 1.0, 1.0, 1.0])
    ras[:3, 3] = [-90, -108, -72]
    pr, _ = sp.process_case(make_case(tmp_path, blob(150, 190, 20), aff=ras, mask_aff=ras))
    assert any(x.startswith("GEOMETRY_") for x in pr["abstain_reasons"]) and "grid" not in pr


def test_legacy_png_mismatch_abstains(tmp_path):
    case = make_case(tmp_path, blob(150, 190, 20))
    im = np.asarray(Image.open(case["legacy_png_path"]).convert("RGB")).copy()
    im[28 + 10, 384 + 10] = (im[28 + 10, 384 + 10].astype(int) + 9) % 256
    Image.fromarray(im).save(case["legacy_png_path"])
    case.pop("legacy_png_sha256")
    pr, _ = sp.process_case(case)
    assert pr["legacy_png"] == "MISMATCH" and "LEGACY_PNG_MISMATCH" in pr["abstain_reasons"]


def test_png_sha_mismatch_abstains(tmp_path):
    case = make_case(tmp_path, blob(150, 190, 20))
    case["legacy_png_sha256"] = "0" * 64
    assert "LEGACY_PNG_SHA_MISMATCH" in sp.process_case(case)[0]["abstain_reasons"]


def test_only_the_right_t1ce_panel_is_compared_not_the_whole_frame(tmp_path):
    case = make_case(tmp_path, blob(150, 190, 20))
    pr, _ = sp.process_case(case)
    assert pr["legacy_png_panel"]["matches"] == {"flair": False, "t1ce": True} and pr["legacy_png_panel"]["panel"] == "t1ce"
    assert pr["legacy_png_panel"]["canvas_size"] == [752, 470]


def test_declared_modality_must_match_its_panel(tmp_path):
    case = make_case(tmp_path, blob(150, 190, 20), modality="flair")  # volume is T1ce-panel content
    pr, _ = sp.process_case(case)
    assert "LEGACY_PNG_MISMATCH" in pr["abstain_reasons"]


def test_undeclared_modality_needs_exactly_one_matching_panel(tmp_path):
    ok = make_case(tmp_path, blob(150, 190, 20), modality=None)
    ok.pop("image_modality")
    pr, _ = sp.process_case(ok)
    assert pr["image_modality"] == "not_declared" and pr["legacy_png_panel"]["panel"] == "t1ce" and "LEGACY_PNG_MISMATCH" not in pr["abstain_reasons"]


def test_single_364x436_png_is_accepted_as_the_panel(tmp_path):
    case = make_case(tmp_path, blob(150, 190, 20))
    render_slice(brain(), LEVEL).convert("RGB").save(case["legacy_png_path"])
    case.pop("legacy_png_sha256")
    pr, _ = sp.process_case(case)
    assert pr["legacy_png"] == "EXACT_PANEL" and pr["legacy_png_panel"]["panel"] == "single"


def test_canvas_offset_never_shifts_the_target_cell(tmp_path):
    assert sp.canvas_to_panel_xy(384 + 10, 28 + 20, "t1ce") == (10, 20)
    assert sp.canvas_to_panel_xy(4 + 10, 28 + 20, "flair") == (10, 20)
    with pytest.raises(ValueError):
        sp.canvas_to_panel_xy(10, 20, "t1ce")  # a FLAIR-panel pixel is not a T1ce-panel pixel
    # same mask, PNG shifted differently in a bigger/other canvas arrangement: the arithmetic target is mask-only
    pr, _ = sp.process_case(make_case(tmp_path, blob(150, 190, 20)))
    assert pr["grid"] == "right-top" and "PANEL_RELATIVE" in pr["target_coordinates_frame"]


def test_clean_case_saves_mask_free_panel_crop_and_single_image_row(tmp_path):
    case = make_case(tmp_path, blob(150, 190, 20))
    d = tmp_path / "mi"
    d.mkdir()
    pr, pub = sp.process_case(case, d)
    row = pr["candidate_row"]
    saved = np.asarray(Image.open(row["images"][0]).convert("L"))
    assert np.array_equal(saved, np.asarray(render_slice(brain(), LEVEL)))  # unmasked render, no header strip, no overlay
    assert Image.open(row["images"][0]).size == (364, 436)
    typed = [c["image"] for c in row["prompt"][0]["content"] if c["type"] == "image"]
    assert typed == row["images"] and row["reward_model"]["ground_truth"] == "right-top"
    assert row["extra_info"]["prompt_state"] == "NEW_SINGLE_IMAGE_CANDIDATE_NOT_MODEL_TESTED" and pub["model_input_candidate_saved"] is True
    text = [c["text"] for c in row["prompt"][0]["content"] if c["type"] == "text"][0]
    assert text == sp.SINGLE_IMAGE_PROMPT and all(g in text for g in sp.GRID_LABELS)
    assert "FLAIR" not in text and "T1" not in text and "c01" not in text


def test_abstained_case_creates_no_candidate_row(tmp_path):
    d = tmp_path / "mi"
    d.mkdir()
    pr, pub = sp.process_case(make_case(tmp_path, blob(150, 190, 8)), d)
    assert pr["candidate_row"] == "NOT_CREATED_ABSTAIN" and pub["model_input_candidate_saved"] is False and not list(d.iterdir())


def test_manifest_aliases_and_key_diagnosis_expose_no_values(tmp_path):
    raw = {"opaque_id": "S01", "split": "train", "flair_path": "/secret/a.nii.gz", "mask_path": "/secret/m.nii.gz", "level": 45}
    c = sp.normalize_case(raw)
    assert c["image_nifti_path"] == "/secret/a.nii.gz" and c["mask_nifti_path"] == "/secret/m.nii.gz" and c["legacy_slice_index"] == 45
    d = sp.describe_keys({"schema": "x", "cases": [raw]})
    assert "/secret" not in json.dumps(d) and d["first_row_keys"] == sorted(raw) and d["n_rows"] == 1
    assert len(sp.manifest_cases([raw])) == 1 and len(sp.manifest_cases({"rows": [raw]})) == 1


def test_visible_text_leakage_abstains(tmp_path):
    case = make_case(tmp_path, blob(150, 190, 20), prompt="Where is the lesion? file c01_img")
    pr, _ = sp.process_case(case)
    assert "VISIBLE_TEXT_LEAKAGE" in pr["abstain_reasons"]
    ok = make_case(tmp_path, blob(150, 190, 20), name="c02", prompt="In which region of the image is the tumor centered?")
    assert "VISIBLE_TEXT_LEAKAGE" not in sp.process_case(ok)[0]["abstain_reasons"]


def test_source_files_unmodified(tmp_path):
    case = make_case(tmp_path, blob(150, 190, 20))
    before = {k: sp.dp.sha256_file(case[k]) for k in ("image_nifti_path", "mask_nifti_path", "legacy_png_path")}
    sp.process_case(case)
    assert before == {k: sp.dp.sha256_file(case[k]) for k in before}


def test_expected_sha_mismatch_fails_closed(tmp_path):
    case = make_case(tmp_path, blob(150, 190, 20))
    case["image_expected_sha256"] = "0" * 64
    pr, _ = sp.process_case(case)
    assert "SOURCE_HASH_MISMATCH" in pr["abstain_reasons"] and "grid" not in pr


# ---- verifier -----------------------------------------------------------------
def test_verifier_fixtures_executable_for_all_nine_cells():
    for g in sorted(sp.GRID_LABELS):
        v = sp.verify_reward_fixtures(g)
        assert v["all_ok"] and v["synthetic_response_fixtures_only"] is True
        assert v["correct"]["score"] == 1.0 and v["wrong"]["accuracy"] == 0.0
        assert v["no_tags"]["format"] == 0.0 and v["no_tags"]["accuracy"] == 0.0
        assert v["unboxed_in_answer"]["format"] == 0.5 and v["unboxed_in_answer"]["accuracy"] == 1.0
    with pytest.raises(ValueError):
        sp.verify_reward_fixtures("upper-left")


# ---- runner ----------------------------------------------------------------------
def inp(cases):
    return {"schema": "mri-spatial-task-pilot-input-v1", "cases": cases}


def test_runner_end_to_end_privacy_and_outputs(tmp_path):
    cases = [make_case(tmp_path, blob(150, 190, 20), name="c01"), make_case(tmp_path, blob(30, 40, 8), name="c02")]
    out = tmp_path / "out"
    pub = tmp_path / "pub.json"
    res = sp.run_pilot(inp(cases), out, pub)
    assert res["cases_attempted"] == 2 and res["status_counts"] == {"TECHNICAL_ARITHMETIC_CLEAN_EXPERT_PENDING": 1, "ABSTAIN": 1}
    text = pub.read_text()
    assert str(tmp_path) not in text and "_img" not in text and ".nii" not in text
    j = json.loads(text)
    assert j["scientific_admitted"] == 0 and j["models_loaded"] == 0 and j["optimizer_calls"] == 0
    packet = json.loads((out / "expert_packet_PRIVATE.json").read_text())
    assert all(v is None for item in packet["items"] for v in item["expert_fields"].values())
    assert packet["ai_expert_assertions"] == 0
    assert (out / "case_c01_PRIVATE.json").exists()


def test_runner_rejects_nonempty_output_and_more_than_three(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "x").write_text("1")
    c = make_case(tmp_path, blob(150, 190, 20))
    with pytest.raises(ValueError):
        sp.run_pilot(inp([c]), out, tmp_path / "p.json")
    cs = [make_case(tmp_path, blob(150, 190, 20), name=f"c0{i}") for i in range(1, 5)]
    with pytest.raises(ValueError):
        sp.run_pilot(inp(cs), tmp_path / "o2", tmp_path / "p2.json")


def test_runner_keeps_failure_and_continues_without_replacement(tmp_path):
    bad = make_case(tmp_path, blob(150, 190, 20), name="c01")
    bad["image_nifti_path"] = str(tmp_path / "missing.nii.gz")
    good = make_case(tmp_path, blob(150, 190, 20), name="c02")
    res = sp.run_pilot(inp([bad, good]), tmp_path / "o", tmp_path / "p.json")
    assert res["status_counts"].get("FAILED") == 1 and res["cases_attempted"] == 2
    assert (tmp_path / "o" / "case_c01_FAILED_PRIVATE.json").exists()
    assert str(tmp_path) not in (tmp_path / "p.json").read_text()


def test_soft_time_budget_stops_remaining_after_safe_save(tmp_path):
    ticks = iter([0, 0, 500, 500, 500, 500, 500, 500, 500, 500])
    cs = [make_case(tmp_path, blob(150, 190, 20), name=f"c0{i}") for i in range(1, 4)]
    res = sp.run_pilot(inp(cs), tmp_path / "o", tmp_path / "p.json", clock=lambda: next(ticks))
    assert res["cases_attempted"] == 1 and res["stopped_remaining"] == 2
    assert (tmp_path / "o" / "case_c01_PRIVATE.json").exists()


def test_rss_budget_stops_remaining(tmp_path):
    cs = [make_case(tmp_path, blob(150, 190, 20), name=f"c0{i}") for i in range(1, 3)]
    res = sp.run_pilot(inp(cs), tmp_path / "o", tmp_path / "p.json", rss=lambda: 600 * 2 ** 20)
    assert res["cases_attempted"] == 0 and res["stopped_remaining"] == 2 and res["stop_reason"] == "RSS_BUDGET"


def test_non_train_split_rejected(tmp_path):
    c = make_case(tmp_path, blob(150, 190, 20))
    c["split"] = "test"
    with pytest.raises(ValueError):
        sp.run_pilot(inp([c]), tmp_path / "o", tmp_path / "p.json")


def test_mask_outside_image_support_abstains(tmp_path):
    vol = brain(box=(slice(100, 180), slice(100, 216), slice(2, 58)))
    pr, _ = sp.process_case(make_case(tmp_path, blob(30, 40, 20, base=None), vol=vol, target="left-bottom"))
    assert "MASK_OUTSIDE_IMAGE_SUPPORT" in pr["abstain_reasons"] and pr["mask_fraction_inside_nonzero_image_support"] < 0.8
