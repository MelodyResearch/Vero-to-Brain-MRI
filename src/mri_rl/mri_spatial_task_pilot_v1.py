"""Minimal real-TRAIN spatial multiclass (mri_tumor_grid) CPU pilot, v1.

Scope: arithmetic technical gold for the 3x3 display-grid task, computed from the source segmentation with the pinned
formal ontology (0 background, 1 necrotic core, 2 non-enhancing/edematous, 3 contrast-enhancing; publication v1.0.1 != current
v2.0.1). Nothing here is clinical or visible truth: every case stays EXPERT_ADJUDICATION_REQUIRED and scientific_admitted is
always 0. No model, torch, MLX, training, resampling, mask overlay. Endpoint arithmetic is reused from
spatial_ontology_resolution_v1; geometry helpers from mri_data_prep_v1; the historical renderer from dallas_spatial_qualification_v1.

Image modality is generic (the fixed pilot images are T1ce). The historical spatial PNGs are PAIRED canvases (752x470: FLAIR panel
box (4,28,368,464), T1ce panel box (384,28,748,464), each 364x436); only the panel crop is compared and all endpoint coordinates
are PANEL-relative, never canvas-relative. A single-modality panel crop is a NEW model-input candidate with a NEW single-image
prompt state; the old paired-frame prompt/target are retained only as private provenance.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import traceback
import numpy as np
import nibabel as nib
from PIL import Image

from mri_rl import mri_data_prep_v1 as dp
from mri_rl.dallas_spatial_qualification_v1 import render_slice
from mri_rl.second_task_eligibility_v1 import RULES as _ELIG
from mri_rl.spatial_ontology_resolution_v1 import endpoints
from mri_rl.vero_mri_reward import compute_score_from_data_source, _GRID_LABELS as GRID_LABELS

VERSION = "mri-spatial-task-pilot-v1"
FORMAL_ONTOLOGY = {0: "background/non-tumor tissue", 1: "necrotic core", 2: "non-enhancing/edematous tumor",
                   3: "contrast-enhancing tumor"}
ALLOWED_LABELS = sorted(FORMAL_ONTOLOGY)
RELEASES = {"publication": "ds007045 v1.0.1 (337 people)", "current": "ds007045 v2.0.1 (363 people)"}
LABEL3_LINEAGE = ("PAPER_RELEASE_SAME_ANNOTATION", "NEW_RELEASE_LABEL3_NOT_VERIFIED", "NOT_REATTESTED", "NOT_SUPPLIED")
MODALITIES = ("flair", "t1ce")
PAIRED_CANVAS_SIZE = (752, 470)
PANEL_BOX = {"flair": (4, 28, 368, 464), "t1ce": (384, 28, 748, 464)}  # verified in build_dallas_spatial_qualification_v1
PANEL_SIZE = dp.LEGACY_SIZE  # (364, 436)
MAX_CASES = 3
SOFT_SECONDS_PER_CASE = 120.0
RSS_LIMIT_BYTES = 512 * 2 ** 20
MIN_AREA = _ELIG["area_screen_native_pixels"]          # inherited engineering screens, not validated visibility thresholds
MIN_BOUNDARY = _ELIG["boundary_screen_display_pixels"]
EXPERT_FIELDS = ("lesion_visible_in_frame", "target_cell_matches_visible_focus", "multifocal_clinical", "negative_frame_adjudication",
                 "question_annotation_endpoint_aligned", "annotator_identity", "annotation_date")
# NEW single-image prompt state (PROPOSED; never model-tested). The old paired-frame question is private provenance only.
SINGLE_IMAGE_PROMPT = (
    "This is one axial brain MRI slice. Divide the image, as displayed, into a 3x3 grid with two equal vertical and two equal "
    "horizontal lines. In which grid cell is the center of the whole abnormal lesion region? Choose exactly one of: "
    + ", ".join(f"{x}-{y}" for y in ("top", "middle", "bottom") for x in ("left", "center", "right"))
    + ". Give the final answer in \\boxed{...}.")
KEY_ALIASES = {"image_nifti_path": ("image_nifti_path", "flair_path", "t1ce_path", "nifti_path"),
               "mask_nifti_path": ("mask_nifti_path", "mask_path"),
               "legacy_slice_index": ("legacy_slice_index", "level"),
               "image_expected_sha256": ("image_expected_sha256",), "mask_expected_sha256": ("mask_expected_sha256",),
               "legacy_png_sha256": ("legacy_png_sha256",), "legacy_png_path": ("legacy_png_path",),
               "image_modality": ("image_modality", "modality"), "original_ground_truth": ("original_ground_truth", "ground_truth"),
               "annotation_release_lineage": ("annotation_release_lineage",), "visible_prompt_text": ("visible_prompt_text",),
               "pair_key": ("pair_key",), "opaque_id": ("opaque_id",), "split": ("split",), "source_order": ("source_order",)}


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


# ----------------------------------------------------------------- selection
def select_first_distinct_train_grid(rows, n=3):
    """First n original TRAIN mri_tumor_grid records in source order with distinct people (no voxel/outcome input)."""
    picks, seen = [], set()
    for r in rows:
        ei = r.get("extra_info", {})
        if r.get("ability") != "mri_tumor_grid" or ei.get("internal_split") != "train" or ei.get("patient_key") in seen:
            continue
        seen.add(ei["patient_key"])
        picks.append(r)
        if len(picks) == n:
            return picks
    raise ValueError("Fewer than %d distinct-person TRAIN mri_tumor_grid records" % n)


def normalize_case(raw):
    """Map a private manifest row onto canonical keys (aliases only; no value is printed or guessed)."""
    out = {}
    for key, names in KEY_ALIASES.items():
        for name in names:
            if raw.get(name) is not None:
                out[key] = raw[name]
                break
    out.setdefault("pair_key", out.get("opaque_id"))
    return out


def manifest_cases(inp):
    """Accept {'cases':[...]} / {'rows':[...]} / a bare list. Returns normalized cases."""
    rows = inp if isinstance(inp, list) else inp.get("cases") or inp.get("rows")
    require(isinstance(rows, list) and rows, "No case list in input")
    return [normalize_case(r) for r in rows]


def describe_keys(inp):
    """Key NAMES only (never values) of the first case, for schema diagnosis without exposing private content."""
    rows = inp if isinstance(inp, list) else inp.get("cases") or inp.get("rows") or []
    return {"top_level_keys": sorted(inp) if isinstance(inp, dict) else "LIST", "n_rows": len(rows),
            "first_row_keys": sorted(rows[0]) if rows else []}


# ------------------------------------------------------------------- frames
_ANAT = {"L": "Left", "R": "Right", "A": "Anterior", "P": "Posterior", "S": "Superior", "I": "Inferior"}
_OPP = {"Left": "Right", "Right": "Left", "Anterior": "Posterior", "Posterior": "Anterior"}


def display_frame(affine):
    """Explicit LAS radiological display (np.rot90(volume[:,:,z])) and what the grid words mean anatomically."""
    d = dp.legacy_display_axes(affine)
    right, up = _ANAT[d["right"]], _ANAT[d["up"]]
    return {"display_right_is_anatomical": right, "display_left_is_anatomical": _OPP[right], "display_up_is_anatomical": up,
            "display_down_is_anatomical": _OPP[up], "storage_axcodes": d["storage_axcodes"],
            "grid_words": "left/center/right and top/middle/bottom name DISPLAY columns/rows, not anatomical sides",
            "coordinates": "PANEL-relative image edges (never canvas coordinates); first pixel [0,1) center .5; thirds half-open; boundary -> right/bottom cell"}


def canvas_to_panel_xy(x, y, modality):
    """Convert a paired-canvas pixel to panel-relative display coordinates (canvas offset must never move a target cell)."""
    x0, y0, x1, y1 = PANEL_BOX[modality]
    require(x0 <= x < x1 and y0 <= y < y1, "Canvas point outside the modality panel")
    return x - x0, y - y0


# --------------------------------------------------------------------- plane
def plane_report(display_mask, original_target, *, label3_semantics="NOT_SUPPLIED", min_area=MIN_AREA,
                 require_display_scale=True, extra_reasons=()):
    """Endpoint arithmetic + frozen abstain rules for one display-oriented integer label plane.

    Primary endpoint: whole-lesion union (label>0) centroid cell. The numeric-3 component and per-component cells are reported
    separately and never substituted; visible focus is NOT_ADJUDICATED.
    """
    m = np.asarray(display_mask)
    h, w = m.shape
    reasons = list(extra_reasons)
    if require_display_scale and (w * 2, h * 2) != PANEL_SIZE:
        reasons.append("DISPLAY_SCALE_NOT_2X")
    union = endpoints(m > 0)
    num3 = endpoints(m == 3)
    if union["area_native_pixels"] == 0:
        reasons.append("EMPTY_MASK_NOT_NEGATIVE_GOLD")
    else:
        if union["area_native_pixels"] < min_area:
            reasons.append("SMALL_SOURCE_MASK")
        if union["components_8"] > 1:
            reasons.append("MULTICOMPONENT_2D")
        if union["boundary_distance_display_pixels"] < MIN_BOUNDARY:
            reasons.append("CENTROID_NEAR_GRID_BOUNDARY")
        if not union["union_centroid_on_mask_nearest_pixel"]:
            reasons.append("CENTROID_OFF_MASK")
        if union["component_grids_disagree_with_union"]:
            reasons.append("COMPONENT_GRID_DISAGREES_WITH_UNION")
        if union["grid"] != union["legacy_grid"]:
            reasons.append("CONVENTION_SENSITIVE_GRID")
        if original_target is not None and original_target not in (union["grid"], union["legacy_grid"]):
            reasons.append("ORIGINAL_TARGET_DISAGREES_WITH_ARITHMETIC")
    status = "ABSTAIN" if reasons else "TECHNICAL_ARITHMETIC_CLEAN_EXPERT_PENDING"
    n3 = {k: num3[k] for k in ("area_native_pixels", "components_8", "grid", "union_centroid_display_edge_xy") if k in num3}
    n3.update(admitted=False, route_enabled=False, label3_semantics=label3_semantics,
              note="separate numeric-3 component endpoint; its clinical meaning (enhancing) is not asserted unless the release lineage is "
                   "attested; never substituted for the whole-lesion union target")
    return {"union": union, "grid": union["grid"], "legacy_grid": union["legacy_grid"], "original_target": original_target,
            "numeric3_component": n3, "abstain_reasons": list(dict.fromkeys(reasons)), "technical_status": status,
            "scientific_admitted": 0, "expert_status": "EXPERT_ADJUDICATION_REQUIRED",
            "mask_derived_not_visible_truth": True, "visible_focus_center": "NOT_ADJUDICATED",
            "clinical_multifocal_status": "NOT_ADJUDICATED", "endpoint": "whole_lesion_union(label>0)_centroid_cell"}


# ------------------------------------------------------------------ verifier
def verify_reward_fixtures(target):
    """Run the real Vero-format MRI reward on SYNTHETIC response strings (never model outputs)."""
    require(target in GRID_LABELS, "Unknown grid target")
    other = sorted(GRID_LABELS - {target})[0]

    def run(resp):
        return compute_score_from_data_source("OpenNeuro_ds007045", resp, target, {"reward_type": "mri_tumor_grid"})
    out = {"correct": run(f"<think>fixture</think><answer>\\boxed{{{target}}}</answer>"),
           "wrong": run(f"<think>fixture</think><answer>\\boxed{{{other}}}</answer>"),
           "no_tags": run(f"\\boxed{{{target}}}"),
           "unboxed_in_answer": run(f"<think>fixture</think><answer>{target}</answer>")}
    out["all_ok"] = (out["correct"]["score"] == 1.0 and out["wrong"]["accuracy"] == 0.0 and out["wrong"]["format"] == 1.0
                     and out["no_tags"]["accuracy"] == 0.0 and out["no_tags"]["format"] == 0.0
                     and out["unboxed_in_answer"]["accuracy"] == 1.0 and out["unboxed_in_answer"]["format"] == 0.5)
    out["synthetic_response_fixtures_only"] = True
    return out


def single_image_row(opaque_id, image_path, arithmetic_target):
    """NEW single-image candidate row for a technically clean case. Image bound in BOTH typed prompt and top-level images."""
    require(arithmetic_target in GRID_LABELS, "Unknown grid target")
    path = str(image_path)
    return {"id": f"{opaque_id}_single", "data_source": "OpenNeuro_ds007045", "ability": "mri_tumor_grid",
            "prompt": [{"role": "user", "content": [{"type": "image", "image": path}, {"type": "text", "text": SINGLE_IMAGE_PROMPT}]}],
            "images": [path], "reward_model": {"style": "rule", "ground_truth": arithmetic_target},
            "extra_info": {"reward_type": "mri_tumor_grid", "task_type": "mri_tumor_grid", "internal_split": "train",
                           "prompt_state": "NEW_SINGLE_IMAGE_CANDIDATE_NOT_MODEL_TESTED",
                           "target_provenance": "whole-lesion union centroid cell from source mask; technical arithmetic only, expert pending"}}


# --------------------------------------------------------------------- case
def _sha(path):
    return dp.sha256_file(path)


def _stat(path):
    s = Path(path).stat()
    return (s.st_size, s.st_mtime_ns)


def _hidden_strings(case):
    hidden = {k: case[k] for k in ("image_nifti_path", "mask_nifti_path", "legacy_png_path", "pair_key") if case.get(k)}
    hidden["case_id"] = case["opaque_id"]  # 'opaque_id' itself is a reserved binding key
    return hidden


def compare_png_panel(png_path, rendered_gray, declared_modality):
    """Exact comparison of the historical render against the panel crop(s) of a paired canvas (or a single 364x436 image)."""
    with Image.open(png_path) as im:
        canvas = im.convert("RGB")
    size = canvas.size
    if size == PAIRED_CANVAS_SIZE:
        panels = {m: canvas.crop(b) for m, b in PANEL_BOX.items()}
    elif size == PANEL_SIZE:
        panels = {"single": canvas}
    else:
        return {"status": "UNEXPECTED_CANVAS_SIZE", "canvas_size": list(size), "matches": {}, "panel": None, "crop": None}
    gray = np.asarray(rendered_gray, np.uint8)
    require(gray.shape == (PANEL_SIZE[1], PANEL_SIZE[0]), "Rendered panel has an unexpected shape")
    ref_rgb = np.repeat(gray[:, :, None], 3, axis=2)  # exact RGB comparison: a colored/overlaid pixel can never equal gray
    matches = {m: bool(np.array_equal(np.asarray(p), ref_rgb)) for m, p in panels.items()}
    if size == PANEL_SIZE:
        panel = "single" if matches["single"] else None
    elif declared_modality in PANEL_BOX:
        panel = declared_modality if matches[declared_modality] else None
    else:
        hit = [m for m, ok in matches.items() if ok]
        panel = hit[0] if len(hit) == 1 else None
    return {"status": "EXACT_PANEL" if panel else "MISMATCH", "canvas_size": list(size), "matches": matches, "panel": panel,
            "crop": panels[panel] if panel else None}


def process_case(case, model_input_dir=None):
    """Return (private_record, public_record). Reads source read-only; no resampling, no mask rendering.

    A technically clean case may additionally save the mask-free single-modality panel crop as a NEW model-input candidate."""
    case = normalize_case(case)
    t0 = time.perf_counter()
    oid = case["opaque_id"]
    pr = {"opaque_id": oid, "abstain_reasons": [], "resampled": False, "mask_is_model_input": False}
    pub = {"opaque_id": oid}
    paths = [k for k in ("image_nifti_path", "mask_nifti_path", "legacy_png_path") if case.get(k)]
    stat0 = {k: _stat(case[k]) for k in paths}
    reasons = pr["abstain_reasons"]

    def finish():
        pr["abstain_reasons"] = list(dict.fromkeys(reasons))
        pr["technical_status"] = "ABSTAIN" if pr["abstain_reasons"] else "TECHNICAL_ARITHMETIC_CLEAN_EXPERT_PENDING"
        pr["scientific_admitted"] = 0
        pr["expert_status"] = "EXPERT_ADJUDICATION_REQUIRED"
        pr["elapsed_s"] = round(time.perf_counter() - t0, 3)
        pr["source_unmodified"] = all(_stat(case[k]) == stat0[k] for k in stat0)
        pub.update(technical_status=pr["technical_status"], abstain_reasons=pr["abstain_reasons"], scientific_admitted=0,
                   expert_status=pr["expert_status"], elapsed_s=pr["elapsed_s"], source_unmodified=pr["source_unmodified"])
        return pr, pub

    modality = (case.get("image_modality") or "not_declared").lower()
    require(modality in MODALITIES or modality == "not_declared", "Unknown image modality")
    pr["image_modality"] = pub["image_modality"] = modality
    img_geo, mask_geo = dp.inspect_nifti(case["image_nifti_path"]), dp.inspect_nifti(case["mask_nifti_path"])
    sha_i, sha_m = _sha(case["image_nifti_path"]), _sha(case["mask_nifti_path"])
    if (case.get("image_expected_sha256") and case["image_expected_sha256"] != sha_i) or \
            (case.get("mask_expected_sha256") and case["mask_expected_sha256"] != sha_m):
        reasons.append("SOURCE_HASH_MISMATCH")
        pr["geometry"] = {}
        return finish()
    pr["source_sha256_PRIVATE"] = {"image": sha_i, "mask": sha_m}
    ge, gm = dp.geometry_eligibility(img_geo), dp.geometry_eligibility(mask_geo)
    for who, g in (("IMAGE", ge), ("MASK", gm)):
        for r_ in g["reasons"]:
            reasons.append(f"GEOMETRY_{who}_{r_}")
    grids = dp.compare_grids(img_geo["shape"], img_geo["affine"], mask_geo["shape"], mask_geo["affine"])
    keys = ("shape", "axcodes", "units", "spacing_mm", "qform_code", "sform_code", "flags")
    pr["geometry"] = {"image": {k: img_geo[k] for k in keys}, "mask": {k: mask_geo[k] for k in keys},
                      "image_mask_grid_identical": grids["same_voxel_grid"], "grid_compare": grids, "registration_proven": False}
    if not grids["same_voxel_grid"]:
        reasons.append("IMAGE_MASK_GRID_MISMATCH_NO_RESAMPLE")
    if reasons:
        return finish()
    level = int(case["legacy_slice_index"])
    vol = nib.load(case["image_nifti_path"]).get_fdata(dtype=np.float32)  # same read as the historical renderer
    raw_mask = np.asanyarray(nib.load(case["mask_nifti_path"]).dataobj)
    if not 0 <= level < vol.shape[2]:
        reasons.append("LEVEL_OUT_OF_RANGE")
        return finish()
    try:
        mask = dp.integer_labels(raw_mask)
    except ValueError:
        reasons.append("MASK_NONINTEGER_LABELS")
        return finish()
    labels = {int(v): int(n) for v, n in zip(*np.unique(mask, return_counts=True))}
    pr["volume_label_voxel_counts"] = labels
    if set(labels) - set(ALLOWED_LABELS):
        reasons.append("UNKNOWN_ONTOLOGY_LABEL")
        pr["labels_outside_ontology"] = sorted(set(labels) - set(ALLOWED_LABELS))
        return finish()
    aff = np.asarray(img_geo["affine"], float)
    pr["display_frame"] = display_frame(aff)
    img_plane = dp.axial_display_plane(vol, aff, level)
    mask_plane = dp.axial_display_plane(mask, aff, level)
    require(np.array_equal(mask_plane, np.rot90(mask[:, :, level])), "Display plane differs from the historical rot90 frame")
    lineage = case.get("annotation_release_lineage", "NOT_SUPPLIED")
    require(lineage in LABEL3_LINEAGE, "Unknown annotation lineage")
    l3 = "VERIFIED_PAPER_RELEASE_SAME_ANNOTATION" if lineage == "PAPER_RELEASE_SAME_ANNOTATION" else "NOT_VERIFIED"
    rep = plane_report(mask_plane, case.get("original_ground_truth"), label3_semantics=l3)
    pr.update(rep)
    reasons.extend(rep["abstain_reasons"])
    pr["plane_label_pixel_counts"] = {int(v): int(n) for v, n in zip(*np.unique(mask_plane, return_counts=True))}
    plane_z = float((aff @ np.array([0, 0, level, 1.0]))[2])
    world = {"plane_world_z_mm": plane_z, "image_mask_center_offset_mm": grids["center_offset_mm"]}
    if rep["union"]["area_native_pixels"]:
        cx, cy = rep["union"]["legacy_centroid_display_index_xy"][0] / 2, rep["union"]["legacy_centroid_display_index_xy"][1] / 2
        Y = mask.shape[1]  # rot90 display: voxel i = column, voxel j = Y-1-row
        world["union_centroid_world_xyz_mm_PRIVATE"] = [round(float(x), 3) for x in (aff @ np.array([cx, Y - 1 - cy, level, 1.0]))[:3]]
    pr["world"] = world
    support, pos = img_plane != 0, mask_plane > 0
    if pos.any():
        inside = float((pos & support).sum() / pos.sum())
        pr["mask_fraction_inside_nonzero_image_support"] = inside
        if inside < dp.MIN_MASK_INSIDE_SUPPORT:
            reasons.append("MASK_OUTSIDE_IMAGE_SUPPORT")
        other = support & ~pos
        if other.any():
            a, b = img_plane[pos].astype(float), img_plane[other].astype(float)
            sd = float(np.sqrt((a.var() + b.var()) / 2)) or 1.0
            pr["image_contrast_effect_size_descriptive"] = round(float((a.mean() - b.mean()) / sd), 3)
            pr["image_contrast_note"] = ("image intensity inside vs outside the union mask in THIS modality; descriptive, NOT visibility or "
                                         "answerability evidence and not assumed equal to another modality's contrast")
    # historical render vs existing PNG (panel crop only; the mask never enters any image)
    rendered = render_slice(vol, level)
    rgray = np.asarray(rendered)
    for f in dp.qc_flags_for_plane(rgray):
        reasons.append(f"QC_{f}")
    pr["legacy_render_hash16"] = dp._h16(rgray)
    pr["legacy_png"] = "NOT_SUPPLIED"
    crop = None
    if case.get("legacy_png_path"):
        if case.get("legacy_png_sha256") and case["legacy_png_sha256"] != _sha(case["legacy_png_path"]):
            reasons.append("LEGACY_PNG_SHA_MISMATCH")
            pr["legacy_png"] = "SHA_MISMATCH"
        else:
            cmp_ = compare_png_panel(case["legacy_png_path"], rgray, None if modality == "not_declared" else modality)
            crop = cmp_.pop("crop")
            pr["legacy_png"] = cmp_["status"]
            pr["legacy_png_panel"] = {k: cmp_[k] for k in ("canvas_size", "matches", "panel")}
            pr["target_coordinates_frame"] = "PANEL_RELATIVE; canvas offset (384,28)/(4,28) never applied to the endpoint"
            if cmp_["status"] != "EXACT_PANEL":
                reasons.append("LEGACY_PNG_MISMATCH")
                crop = None
    if case.get("visible_prompt_text"):
        try:
            dp.bind_source_label(oid, case["visible_prompt_text"], sorted(GRID_LABELS),
                                 case.get("original_ground_truth") or sorted(GRID_LABELS)[0], _hidden_strings(case))
            pr["visible_text_check"] = "NO_HIDDEN_IDENTIFIER_FOUND"
        except ValueError:
            reasons.append("VISIBLE_TEXT_LEAKAGE")
            pr["visible_text_check"] = "LEAKAGE"
    else:
        pr["visible_text_check"] = "NOT_SUPPLIED"
    tgt = rep["grid"] if rep["grid"] else case.get("original_ground_truth")
    pr["verifier"] = verify_reward_fixtures(tgt if tgt in GRID_LABELS else sorted(GRID_LABELS)[0])
    pr["independent_mask_derived_sources"] = 1
    pr["independent_skill_note"] = "presence, enhancement and grid targets from this one mask count as ONE mask-derived source, not independent skills"
    clean = not list(dict.fromkeys(reasons))
    pr["candidate_row"] = "NOT_CREATED_ABSTAIN" if not clean else "NOT_CREATED_NO_MODEL_INPUT_DIR"
    if clean and crop is not None and model_input_dir is not None:
        target = Path(model_input_dir) / f"model_input_{oid}_PRIVATE.png"
        crop.convert("RGB").save(target)  # mask-free, header-free panel crop (no metadata chunks)
        pr["model_input_candidate_sha256_PRIVATE"] = _sha(target)
        pr["candidate_row"] = single_image_row(oid, target, rep["grid"])
        pr["candidate_row_prompt_state"] = "NEW_SINGLE_IMAGE_CANDIDATE_NOT_MODEL_TESTED"
    pub.update(grid=rep["grid"], legacy_grid=rep["legacy_grid"], components_8=rep["union"]["components_8"],
               union_area_native_pixels=rep["union"]["area_native_pixels"],
               boundary_distance_display_pixels=rep["union"].get("boundary_distance_display_pixels"),
               label_set=sorted(labels), legacy_png=pr["legacy_png"], legacy_render_hash16=pr["legacy_render_hash16"],
               image_mask_grid_identical=True, resampled=False, plane_world_z_mm=plane_z,
               numeric3_component_area=rep["numeric3_component"]["area_native_pixels"],
               label3_semantics=l3, verifier_all_ok=pr["verifier"]["all_ok"],
               model_input_candidate_saved=isinstance(pr["candidate_row"], dict))
    return finish()


# -------------------------------------------------------------------- runner
def _rss_bytes():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r if sys.platform == "darwin" else r * 1024


def _write_atomic(path, obj):
    tmp = Path(str(path) + ".tmp")
    with tmp.open("w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    os.replace(tmp, path)


def run_pilot(inp, out_dir, public_json, *, clock=time.perf_counter, rss=_rss_bytes):
    cases = manifest_cases(inp)
    require(len(cases) <= MAX_CASES, "At most %d cases" % MAX_CASES)
    ids = [c.get("opaque_id") for c in cases]
    require(all(ids) and len(set(ids)) == len(ids) and all(dp._ID.match(i) and not i.lower().startswith(("sub", "ses")) for i in ids),
            "Opaque ids must be unique and non-identifying")
    require(all(c.get("split") == "train" for c in cases), "Pilot is TRAIN-only")
    require(all(c.get("image_nifti_path") and c.get("mask_nifti_path") and c.get("legacy_slice_index") is not None for c in cases),
            "Each case needs image_nifti_path, mask_nifti_path and legacy_slice_index")
    out, pub_path = Path(out_dir), Path(public_json)
    require(not out.exists() or not any(out.iterdir()), "Output directory must be new/empty")
    require(not pub_path.exists(), "Public output must not pre-exist")
    out.mkdir(parents=True, exist_ok=True)
    model_dir = out / "model_input_candidates_PRIVATE"
    model_dir.mkdir()
    t_start = clock()
    private_rows, public_rows, by_id = [], [], {}
    stop_reason, attempted, counts = None, 0, {}
    for case in cases:
        if rss() > RSS_LIMIT_BYTES:
            stop_reason = "RSS_BUDGET"
            break
        t0 = clock()
        try:
            pr, pb = process_case(case, model_dir)
            _write_atomic(out / f"case_{case['opaque_id']}_PRIVATE.json", pr)
            private_rows.append(pr)
            by_id[pr["opaque_id"]] = case
        except Exception as e:  # real paths/ids stay in the private failure log only
            pb = {"opaque_id": case["opaque_id"], "technical_status": "FAILED", "error_class": type(e).__name__,
                  "scientific_admitted": 0, "expert_status": "EXPERT_ADJUDICATION_REQUIRED"}
            _write_atomic(out / f"case_{case['opaque_id']}_FAILED_PRIVATE.json",
                          {"opaque_id": case["opaque_id"], "error": repr(e), "traceback": traceback.format_exc()})
        attempted += 1
        pb["peak_rss_mib_so_far"] = round(rss() / 2 ** 20, 1)
        public_rows.append(pb)
        counts[pb["technical_status"]] = counts.get(pb["technical_status"], 0) + 1
        if clock() - t0 > SOFT_SECONDS_PER_CASE:
            stop_reason = "SOFT_TIME_BUDGET"
            break
    packet = {"schema": "spatial-expert-packet-v1", "ai_expert_assertions": 0, "instructions":
              "Blank. Independent qualified expert only; AI must not fill. Source masks are not visible/clinical truth. "
              "Shown frame = the mask-free single-modality panel crop, never the paired canvas.",
              "items": [{"opaque_id": r["opaque_id"], "level": int(by_id[r["opaque_id"]]["legacy_slice_index"]),
                         "legacy_png_PRIVATE": by_id[r["opaque_id"]].get("legacy_png_path"),
                         "original_target_PRIVATE": by_id[r["opaque_id"]].get("original_ground_truth"),
                         "arithmetic_target_PRIVATE": r.get("grid"), "abstain_reasons": r["abstain_reasons"],
                         "expert_fields": {k: None for k in EXPERT_FIELDS}} for r in private_rows]}
    _write_atomic(out / "expert_packet_PRIVATE.json", packet)
    cand = [r["candidate_row"] for r in private_rows if isinstance(r.get("candidate_row"), dict)]
    with (out / "candidate_rows_PRIVATE.jsonl").open("w") as f:
        for row in cand:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    code_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result = {"schema": "mri-spatial-task-pilot-public-v1", "version": VERSION, "cases_attempted": attempted,
              "cases_configured": len(cases), "stopped_remaining": len(cases) - attempted, "stop_reason": stop_reason,
              "status_counts": counts, "rows": public_rows, "candidate_rows_created": len(cand), "scientific_admitted": 0,
              "training_authorized": False, "models_loaded": 0, "optimizer_calls": 0, "mri_volumes_read": attempted,
              "source_masks_read": attempted, "single_image_prompt_state": "NEW_SINGLE_IMAGE_CANDIDATE_NOT_MODEL_TESTED",
              "ontology": {"labels": {str(k): v for k, v in FORMAL_ONTOLOGY.items()}, "releases": RELEASES,
                           "note": "publication v1.0.1 and current v2.0.1 are not conflated; numeric-3 semantics per case lineage"},
              "elapsed_s": round(clock() - t_start, 3), "peak_rss_mib": round(rss() / 2 ** 20, 1), "code_sha256_16": code_sha[:16],
              "evidence_tier": "OBSERVED arithmetic on source data; NOT visible/clinical truth; NOT model evidence"}
    _write_atomic(out / "run_receipt_PRIVATE.json", {**result, "input_sha256": hashlib.sha256(
        json.dumps(cases, sort_keys=True).encode()).hexdigest()})
    _write_atomic(pub_path, result)
    return result
