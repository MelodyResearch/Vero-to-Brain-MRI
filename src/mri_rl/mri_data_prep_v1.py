"""MRI data preparation v1: source-grounded geometry, normalization, slice selection, QC.

CPU-only (numpy/nibabel/Pillow/scipy). No model, torch, MLX, training or fitting.
Independent of, and never overwriting, the historical renderer: ``legacy_render``
reproduces ``np.rot90(volume[:,:,z])`` + nonzero[::16] 0.5/99.5 percentile window +
uint8 + 364x436 bilinear exactly and is kept as the Legacy Control.

Alignment checks here are GEOMETRIC (grid/affine compatibility, intensity-based
proxies). None of them proves anatomical registration; reports say so explicitly.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import time
import numpy as np
import nibabel as nib
from nibabel import orientations as ori
from PIL import Image
from scipy import ndimage

VERSION = "mri-data-prep-v1"
MANIFEST_SCHEMA = "mri-data-prep-manifest-v1"
LEGACY_SIZE = (364, 436)
NORMALIZATION_MODES = ("legacy_percentile", "percentile_nonzero_full", "zscore_nonzero", "minmax_nonzero")
MASKED_MODES = ("percentile_given_mask",)
PRIMARY_FRACTIONS = (0.30, 0.50, 0.70)  # frozen before any pilot output exists


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha256_file(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


# ---------------------------------------------------------------- geometry
def obliquity_deg(affine):
    a = np.asarray(affine, float)[:3, :3]
    cols = a / np.linalg.norm(a, axis=0)
    return float(np.degrees(np.arccos(np.clip(np.abs(cols).max(axis=0), -1, 1))).max())


def inspect_nifti(path):
    """Header/affine contract for one source volume (no pixel data kept)."""
    img = nib.load(str(path))
    h = img.header
    qcode, scode = int(h["qform_code"]), int(h["sform_code"])
    aff = np.asarray(img.affine, float)
    flags = []
    unit = h.get_xyzt_units()[0]
    if unit != "mm":
        flags.append("UNITS_UNKNOWN" if unit == "unknown" else "UNITS_NOT_MM")
    conflict = False
    if qcode > 0 and scode > 0:
        conflict = bool(np.abs(np.asarray(img.get_qform()) - np.asarray(img.get_sform())).max() > 1e-4)
    if conflict:
        flags.append("QFORM_SFORM_CONFLICT")
    if qcode == 0 and scode == 0:
        flags.append("NO_VALID_XFORM")
    finite = bool(np.isfinite(aff).all())
    if not finite:
        flags.append("NONFINITE_AFFINE")
    spacing = np.linalg.norm(aff[:3, :3], axis=0) if finite else np.full(3, np.nan)
    zooms = np.asarray(h.get_zooms()[:3], float)
    if finite and not np.allclose(spacing, zooms, rtol=0, atol=1e-3):
        flags.append("SPACING_HEADER_AFFINE_MISMATCH")
    obl = obliquity_deg(aff) if finite and abs(np.linalg.det(aff[:3, :3])) > 1e-12 else float("nan")
    if not obl <= 1.0:
        flags.append("OBLIQUE_OR_SINGULAR")
    return {"shape": list(img.shape), "dtype": str(h.get_data_dtype()), "axcodes": "".join(nib.aff2axcodes(aff)),
            "units": unit, "qform_code": qcode, "sform_code": scode, "qform_sform_conflict": conflict,
            "spacing_mm": [round(float(x), 6) for x in spacing], "header_zooms": [round(float(x), 6) for x in zooms],
            "determinant_sign": int(np.sign(np.linalg.det(aff[:3, :3]))), "obliquity_deg": obl,
            "affine": aff.tolist(), "flags": flags}


def legacy_display_axes(affine):
    """What the historical np.rot90(volume[:,:,z]) shows, from the storage axis codes."""
    codes = nib.aff2axcodes(np.asarray(affine, float))
    return {"right": codes[0], "up": codes[1], "slice_axis_code": codes[2], "storage_axcodes": "".join(codes),
            "note": "right/up follow increasing axis0 / increasing axis1; axcode letters name the direction each axis points TOWARD"}


def display_aspect_report(shape, spacing, size):
    col_mm = spacing[0] * shape[0] / size[0]
    row_mm = spacing[1] * shape[1] / size[1]
    return {"mm_per_display_pixel_col": col_mm, "mm_per_display_pixel_row": row_mm,
            "pixel_aspect_distortion": max(col_mm, row_mm) / min(col_mm, row_mm), "source": "header/affine spacing, not assumed"}


def canonicalize_ras(data, affine, max_samples=64):
    """Reorient to RAS keeping the original affine mapping; verify same physical position per sample."""
    aff = np.asarray(affine, float)
    data = np.asarray(data)
    xfm = ori.ornt_transform(ori.io_orientation(aff), ori.axcodes2ornt(("R", "A", "S")))
    new = ori.apply_orientation(data, xfm)
    new_aff = aff @ ori.inv_ornt_aff(xfm, data.shape)
    corners = [(i, j, k) for i in (0, data.shape[0] - 1) for j in (0, data.shape[1] - 1) for k in (0, data.shape[2] - 1)]
    rng = np.random.default_rng(0)
    extra = [tuple(int(rng.integers(0, s)) for s in data.shape) for _ in range(max_samples - len(corners))]
    inv = np.linalg.inv(new_aff)
    worst = 0.0
    for ijk in corners + extra:
        w = aff @ np.r_[ijk, 1]
        nj = inv @ w
        nij = np.rint(nj[:3]).astype(int)
        require(np.abs(nj[:3] - nij).max() < 1e-6 and (nij >= 0).all() and (nij < new.shape).all(), "Canonical voxel not integral/in-bounds")
        worst = max(worst, float(np.abs((new_aff @ np.r_[nij, 1]) - w).max()))
        require(new[tuple(nij)] == data[ijk] or (np.isnan(new[tuple(nij)]) and np.isnan(data[ijk])), "Canonical intensity mismatch")
    require(worst < 1e-6, "Canonical world mapping error")
    return new, new_aff, {"verified_samples": len(corners) + len(extra), "max_world_error_mm": worst,
                          "old_axcodes": "".join(nib.aff2axcodes(aff)), "new_axcodes": "RAS", "original_affine_preserved": aff.tolist()}


def axial_axis(affine):
    return int(np.argmax(np.abs(np.asarray(affine, float)[2, :3])))


def axial_display_plane(volume, affine, index):
    """Axial plane in the fixed display frame: right=anatomical Left, up=Anterior.

    For LAS storage with axial axis last this equals np.rot90(volume[:,:,index]) exactly.
    """
    aff = np.asarray(affine, float)
    k = axial_axis(aff)
    plane = np.take(np.asarray(volume), index, axis=k)
    others = [a for a in range(3) if a != k]
    ax_x = max(others, key=lambda a: abs(aff[0, a]))
    ax_y = [a for a in others if a != ax_x][0]
    if others.index(ax_y) > others.index(ax_x):  # arrange as [y, x]
        plane = plane.T
    if aff[1, ax_y] > 0:
        plane = plane[::-1]
    if aff[0, ax_x] > 0:
        plane = plane[:, ::-1]
    return plane


def compare_grids(shape_a, aff_a, shape_b, aff_b, atol=1e-4):
    a, b = np.asarray(aff_a, float), np.asarray(aff_b, float)
    diff = float(np.abs(a - b).max())
    ca = a @ np.r_[(np.asarray(shape_a) - 1) / 2, 1]
    cb = b @ np.r_[(np.asarray(shape_b) - 1) / 2, 1]
    same = tuple(shape_a) == tuple(shape_b) and diff <= atol
    return {"same_shape": tuple(shape_a) == tuple(shape_b), "affine_max_abs_diff": diff, "same_voxel_grid": bool(same),
            "same_axcodes": nib.aff2axcodes(a) == nib.aff2axcodes(b), "center_offset_mm": float(np.linalg.norm(ca[:3] - cb[:3])),
            "registration_proven": False, "scope": "grid/affine compatibility only; not anatomical registration"}


def _nmi(x, y, bins=32):
    h, _, _ = np.histogram2d(x, y, bins=bins)
    p = h / h.sum()
    px, py = p.sum(1), p.sum(0)
    ent = lambda q: -(q[q > 0] * np.log(q[q > 0])).sum()
    return float((ent(px) + ent(py)) / ent(p.ravel()))


def shift_sensitivity_nmi(a, b, shifts=(0, 2, 4, 8), stride=2):
    """Normalized mutual information of foreground voxels under axis-1 shifts of b.

    A peak at shift 0 is consistent with co-location but is NOT a registration proof.
    """
    require(a.shape == b.shape, "Same-grid volumes required")
    out = {}
    for s in shifts:
        aa, bb = (a, b) if s == 0 else (a[:, s:], b[:, :-s])
        aa, bb = aa[::stride, ::stride, ::stride], bb[::stride, ::stride, ::stride]
        m = (aa != 0) | (bb != 0)
        out[str(s)] = _nmi(aa[m].astype(float), bb[m].astype(float)) if m.sum() > 100 else float("nan")
    return {"nmi": out, "registration_proven": False,
            "interpretation": "intensity-association proxy for gross co-location; contrast-dependent and not a registration verdict"}


def resample_mask_nearest(mask, mask_aff, target_shape, target_aff):
    m = np.linalg.inv(np.asarray(mask_aff, float)) @ np.asarray(target_aff, float)
    return ndimage.affine_transform(np.asarray(mask), m[:3, :3], offset=m[:3, 3], output_shape=tuple(target_shape), order=0, mode="constant", cval=0)


def check_image_mask_geometry(volume, img_aff, mask, mask_aff, allowed_labels):
    g = compare_grids(volume.shape, img_aff, mask.shape, mask_aff)
    flags = []
    before = {int(x) for x in np.unique(mask)}
    if g["same_voxel_grid"]:
        on_grid = np.asarray(mask)
    else:
        on_grid = resample_mask_nearest(mask, mask_aff, volume.shape, img_aff)
        flags.append("RESAMPLED_NEAREST")
    after = {int(x) for x in np.unique(on_grid)}
    outside = sorted(before - set(allowed_labels))
    if outside:
        flags.append("LABEL_OUTSIDE_ALLOWED")
    preserved = after <= before and not outside
    nz = on_grid > 0
    frac = float((np.asarray(volume)[nz] != 0).mean()) if nz.any() else None
    if frac is not None and frac < 0.5:
        flags.append("MASK_OUTSIDE_IMAGE_SUPPORT")
    return {"grid_compatible": bool(g["same_voxel_grid"]), "grid": g, "label_set_before": sorted(before), "label_set_after": sorted(after),
            "label_set_preserved": bool(preserved), "mask_nonzero_voxels": int(nz.sum()), "fraction_inside_support": frac,
            "registration_proven": False, "mask_is_model_input": False,
            "scope": "geometric grid check + image-support proxy; a grid-compatible mask can still be misregistered", "flags": flags}


# ------------------------------------------------------------ normalization
def window_for(volume, mode, support_mask=None, percentiles=(0.5, 99.5)):
    """Per-volume intensity window [lo, hi]. Uses only this volume (no cross-volume or test statistics)."""
    v = np.asarray(volume)
    require(np.isfinite(v).all(), "Nonfinite volume")
    if mode == "percentile_given_mask":
        require(support_mask is not None and np.shape(support_mask) == v.shape, "Mask required for masked normalization")
        sel = v[np.asarray(support_mask, bool)]
        require(sel.size > 1, "Empty mask foreground")
        lo, hi = np.percentile(sel, percentiles)
    else:
        require(mode in NORMALIZATION_MODES, "Unknown normalization mode")
        nz = v[v != 0]
        require(nz.size > 1, "Insufficient foreground")
        if mode == "legacy_percentile":
            lo, hi = np.percentile(nz[::16], [0.5, 99.5])
            require(nz[::16].size > 1, "Insufficient foreground")
        elif mode == "percentile_nonzero_full":
            lo, hi = np.percentile(nz, percentiles)
        elif mode == "zscore_nonzero":
            mu, sd = float(nz.mean()), float(nz.std())
            lo, hi = mu - 2 * sd, mu + 4 * sd  # z in [-2, 4] -> [0, 255]
        else:
            lo, hi = nz.min(), nz.max()
    require(hi > lo, "Degenerate normalization")
    return [float(lo), float(hi)]


def apply_window(plane, lo, hi):
    return np.uint8(np.round(np.clip((plane - lo) / (hi - lo), 0, 1) * 255))


def _to_image(pixels, size=LEGACY_SIZE):
    return Image.fromarray(pixels).resize(size, Image.Resampling.BILINEAR).convert("RGB")


def legacy_render(volume, level, size=LEGACY_SIZE):
    """Legacy Control: byte-identical to the historical renderer."""
    require(volume.ndim == 3 and 0 <= level < volume.shape[2] and np.isfinite(volume).all(), "Invalid source volume/level")
    nonzero = volume[volume != 0][::16]
    require(nonzero.size > 1, "Insufficient source foreground")
    lo, hi = np.percentile(nonzero, [0.5, 99.5])
    require(hi > lo, "Degenerate normalization")
    plane = np.rot90(volume[:, :, level])
    return _to_image(apply_window(plane, lo, hi), size), [float(lo), float(hi)]


def render_candidate(volume, affine, index, mode="legacy_percentile", size=LEGACY_SIZE, support_mask=None):
    lo, hi = window_for(volume, mode, support_mask)
    plane = axial_display_plane(volume, affine, index)
    return _to_image(apply_window(plane, lo, hi), size), [lo, hi]


# ---------------------------------------------------------- slice selection
def select_slices_coverage(volume, affine, fractions=PRIMARY_FRACTIONS, min_support_frac=0.10, min_coverage=0.50):
    """Deterministic physical-height slice choice from foreground coverage.

    Support extent = axial slices whose foreground area >= min_support_frac * max area.
    Each fraction f picks the slice nearest world z = zmin + f*(zmax-zmin) (inferior=0).
    A slice with area < min_coverage * max area is flagged and abstains (never substituted).
    """
    aff = np.asarray(affine, float)
    v = np.asarray(volume)
    require(np.isfinite(v).all(), "Nonfinite volume")
    k = axial_axis(aff)
    area = np.array([np.count_nonzero(np.take(v, i, axis=k)) for i in range(v.shape[k])], float)
    require(area.max() > 0, "No foreground")
    support = np.flatnonzero(area >= min_support_frac * area.max())
    step = aff[2, k]
    z_of = lambda t: float(aff[2, 3] + t * step)  # other axes at index 0; obliquity is flagged by inspect_nifti
    zs = np.array([z_of(t) for t in support])
    zmin, zmax = float(zs.min()), float(zs.max())
    out = []
    for f in fractions:
        target = zmin + f * (zmax - zmin)
        idx = int(support[np.argmin(np.abs(zs - target))])
        cov = float(area[idx] / area.max())
        flags = ["LOW_COVERAGE"] if cov < min_coverage else []
        out.append({"fraction": f, "index": idx, "world_z_mm": z_of(idx), "coverage_vs_max": cov, "abstain": bool(flags), "qc_flags": flags})
    return {"axial_voxel_axis": k, "slice_spacing_mm": abs(float(step)), "support_index_range": [int(support.min()), int(support.max())],
            "support_world_z_mm": [zmin, zmax], "slices": out, "rule": "coverage_fraction_v1", "area_profile_max_voxels": float(area.max())}


def coverage_at(volume, affine, index):
    k = axial_axis(affine)
    v = np.asarray(volume)
    area = np.array([np.count_nonzero(np.take(v, i, axis=k)) for i in range(v.shape[k])], float)
    return float(area[index] / area.max()), area


def offsets_2p5d(volume, affine, center_index, offsets_mm):
    aff = np.asarray(affine, float)
    k = axial_axis(aff)
    step = float(aff[2, k])
    out = []
    for mm in offsets_mm:
        d = int(np.floor(mm / step + 0.5))
        idx = center_index + d
        out.append({"requested_mm": float(mm), "index": idx, "realized_mm": float(d * step),
                    "out_of_volume": bool(idx < 0 or idx >= np.shape(volume)[k]),
                    "world_z_mm": float(aff[2, 3] + idx * step)})
    return {"slices": out, "rounding": "floor(mm/step+0.5), signed by affine axis direction"}


def visual_token_estimate(size, n_images, patch=16, merge=2):
    f = patch * merge
    per = max(1, round(size[0] / f)) * max(1, round(size[1] / f))
    return {"kind": "ESTIMATE_NOT_MEASURED", "per_image_tokens": per, "n_images": n_images, "total_tokens": per * n_images,
            "assumption": f"patch {patch}, merge {merge}, round-to-{f}px, no min/max-pixel clamp; real processor settings NOT_VERIFIED"}


# ------------------------------------------------------------------- QC
def qc_flags_for_plane(pixels):
    p = np.asarray(pixels)
    flags = []
    if np.count_nonzero(p) / p.size < 0.20:
        flags.append("LOW_COVERAGE")
    if p.std() < 8:
        flags.append("LOW_CONTRAST")
    if (p == 255).mean() > 0.20:
        flags.append("SATURATED")
    return flags


# ------------------------------------------------- split / dedup / privacy
def check_patient_disjoint(rows):
    splits = defaultdict(set)
    for r in rows:
        splits[(r["source"], r["patient"])].add(r["split"])
    by_patient = defaultdict(set)
    for s, p in splits:
        by_patient[p].add(s)
    return {"within_source_violations": sum(len(v) > 1 for v in splits.values()), "patients_within_source": len(splits),
            "same_string_across_sources": sum(len(v) > 1 for v in by_patient.values()),
            "global_cross_source_identity": "NOT_VERIFIED",
            "scope": "disjointness holds only inside each source namespace; cross-source identity needs lineage evidence"}


def _ahash(pixels):
    small = np.asarray(Image.fromarray(np.asarray(pixels, np.uint8)).resize((16, 16), Image.Resampling.BOX), float)
    return (small > small.mean()).ravel()


def duplicate_guard(rows, hamming=6):
    keys = [(r["id"], r["patient"], hashlib.sha256(np.asarray(r["pixels"], np.uint8).tobytes()).hexdigest(), _ahash(r["pixels"])) for r in rows]
    exact = near = same_patient = 0
    flagged = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            ex = a[2] == b[2]
            ham = int((a[3] != b[3]).sum())
            if a[1] == b[1]:
                same_patient += ex or ham <= hamming
                continue
            if ex:
                exact += 1
            if ham <= hamming:
                near += 1
                flagged.append({"a": a[0], "b": b[0], "kind": "exact" if ex else "near", "hamming": ham})
    return {"exact_cross_patient_pairs": exact, "near_cross_patient_pairs": near, "same_patient_similar_pairs": same_patient,
            "flagged_pairs": flagged, "scope": "within provided rows; global external duplication NOT_VERIFIED"}


def bind_source_label(opaque_id, prompt, options, truth, hidden, extra_prompt_images=()):
    """Truth and source identifiers live in hidden metadata; scan the visible prompt/image names for leakage."""
    require(truth in options, "Truth must be one of the options")
    secrets = set()
    for v in hidden.values():
        s = str(v)
        if len(s) >= 3:
            secrets.add(s.lower())
            base = Path(s).name
            secrets.add(base.lower())
            secrets.add(base.split(".")[0].lower())
    secrets = {s for s in secrets if len(s) >= 3}
    visible = " ".join([prompt, *map(str, options), *map(str, extra_prompt_images)]).lower()
    for s in secrets:
        require(s not in visible, "Hidden source identifier appears in model-visible text")
    return {"opaque_id": opaque_id, "prompt_visible": {"prompt": prompt, "options": list(options)},
            "hidden_metadata": {"truth": truth, **hidden}}


# --------------------------------------------------------------- tasks
def _task(source, evidence, answer, verifier, reward, abstain, admission, note):
    return {"source": source, "evidence": evidence, "answer_format": answer, "verifier": verifier, "reward": reward,
            "abstain": abstain, "scientific_admission": admission, "independent_skill_note": note}


TASK_CANDIDATES = {
    "A_sequence": _task("Source acquisition modality from pinned dataset metadata (FLAIR vs post-contrast T1w)",
                        "Whole displayed slice; contrast appearance. Dallas/whole-head and sparse-high-slice cases excluded.",
                        "exactly one of {FLAIR, contrast-enhanced T1} in boxed answer", "exact string match to hidden source label",
                        "existing binary exact-match route (unchanged)", "QC LOW_COVERAGE/LOW_CONTRAST slices abstain",
                        "ENGINEERING_BASELINE: label from metadata, limited visual check; not clinical reasoning",
                        "one skill (sequence recognition); per-slice repeats of one patient are not independent"),
    "B_spatial": _task("Lesion mask centroid in physical/display frame, brain-support midline from the same image",
                       "Lesion visible on the shown slice AND inside brain support; display frame stated in prompt",
                       "left|right (anatomical) via display convention right=L,up=A", "verify_lateral_side on hidden mask",
                       "binary exact-match", "mask straddles midline, offset < margin, tiny area, or low-contrast proxy",
                       "NOT_ADMITTED: visibility and endpoint qualification unverified; engineering only",
                       "shares mask with C/D; count by independent mask not by question"),
    "C_lesion": _task("Segmentation mask with qualified source ontology (pinned label semantics)",
                      "Lesion pixels visible at displayed resolution; mask nonzero is NOT proof of visibility",
                      "present|absent", "verify_lesion_presence on hidden slice mask",
                      "binary exact-match", "area between 0 and min_area_mm2 abstains; mask-outside-support abstains",
                      "NOT_ADMITTED: independent visual answerability and endpoint qualification missing",
                      "shares mask with B/D"),
    "D_cross_slice": _task("Mask extent across 2-3 physically spaced slices from one volume",
                           "Each asked slice displayed with stated physical height; pattern must be checkable from images",
                           "pattern over ordered slices, e.g. which slices contain lesion", "per-slice verify_lesion_presence",
                           "exact match of full pattern (no partial credit invented)", "any ambiguous slice abstains the item",
                           "NOT_ADMITTED: needs visible qualification; geometry/mask-derived", "shares mask with B/C"),
}


def verify_lesion_presence(mask2d, pixel_area_mm2, min_area_mm2=30.0):
    area = float(np.count_nonzero(mask2d) * pixel_area_mm2)
    ans = "absent" if area == 0 else ("present" if area >= min_area_mm2 else None)
    return {"answer": ans, "area_mm2": area, "visibility": "NOT_VERIFIED",
            "note": "mask-derived truth; mask extent does not prove the lesion is visibly discernible"}


def verify_lateral_side(mask2d, support2d, display_right_is, margin_frac=0.1):
    m, s = np.asarray(mask2d, bool), np.asarray(support2d, bool)
    if not m.any() or not s.any():
        return {"answer": None, "reason": "EMPTY"}
    cols = np.flatnonzero(s.any(axis=0))
    c0, c1 = cols.min(), cols.max()
    mid, width = (c0 + c1) / 2, c1 - c0 + 1
    mc = np.nonzero(m)[1]
    off = (mc.mean() - mid) / width
    far = (mc < mid).mean() if off > 0 else (mc > mid).mean()
    if abs(off) < margin_frac:
        return {"answer": None, "reason": "NEAR_MIDLINE", "offset": float(off)}
    if far > 0.2:
        return {"answer": None, "reason": "STRADDLES_MIDLINE", "offset": float(off)}
    other = "R" if display_right_is == "L" else "L"
    side = display_right_is if off > 0 else other
    return {"answer": {"L": "left", "R": "right"}[side], "offset": float(off)}


def independent_mask_count(questions):
    return len({q["mask_ref"] for q in questions})


# ------------------------------------------------------- manifest / pilot
_ID = re.compile(r"^[A-Za-z][A-Za-z0-9]{0,23}$")


def validate_manifest(m):
    require(m.get("schema") == MANIFEST_SCHEMA and m.get("rows"), "Bad manifest schema")
    seen = set()
    for r in m["rows"]:
        oid = r["opaque_id"]
        require(_ID.match(oid) and not oid.lower().startswith(("sub", "ses")) and oid not in seen, "Opaque ID must be non-identifying and unique")
        seen.add(oid)
        require(r["modality"] in ("flair", "t1ce"), "Unsupported modality")
        require(r["split"] == "train", "Pilot is TRAIN-only")
        require(_ID.match(r["pair_key"]) and not r["pair_key"].lower().startswith(("sub", "ses")), "Pair key must be opaque")
        require(isinstance(r["nifti_path"], str) and r["nifti_path"], "nifti_path required")


def _h16(pixels):
    return hashlib.sha256(np.ascontiguousarray(pixels).tobytes()).hexdigest()[:16]


def run_pilot(manifest, private_dir, public_dir):
    validate_manifest(manifest)
    priv, pub = Path(private_dir), Path(public_dir)
    (priv / "images").mkdir(parents=True, exist_ok=True)
    pub.mkdir(parents=True, exist_ok=True)
    levels = manifest.get("legacy_levels", [45, 91, 136])
    rows_pub, rows_priv, loaded = {}, {}, {}
    unchanged = True
    dup_rows = []
    for r in manifest["rows"]:
        t0 = time.perf_counter()
        oid = r["opaque_id"]
        geo = inspect_nifti(r["nifti_path"])
        sha = sha256_file(r["nifti_path"])
        if r.get("expected_sha256"):
            require(sha == r["expected_sha256"], "Source hash mismatch")
        vol = np.asanyarray(nib.load(r["nifti_path"], mmap=True).dataobj)
        require(np.isfinite(vol).all(), "Nonfinite source")
        aff = np.array(geo["affine"])
        k = axial_axis(aff)
        sel = select_slices_coverage(vol, aff)
        e0, e1 = sel["support_world_z_mm"]
        legacy = []
        for z in levels:
            if not 0 <= z < vol.shape[2]:
                legacy.append({"level": z, "out_of_volume": True})
                continue
            a, win = legacy_render(vol, z)
            b, _ = legacy_render(vol, z)
            unchanged &= a.tobytes() == b.tobytes()
            a.save(priv / "images" / f"{oid}_legacy_z{z:03d}.png")
            wz = float(aff[2, 3] + z * aff[2, 2]) if k == 2 else None
            cov = coverage_at(vol, aff, z)[0] if k == 2 else None
            px = np.asarray(a)[:, :, 0]
            legacy.append({"level": z, "window": win, "world_z_mm": wz, "support_fraction": (wz - e0) / (e1 - e0) if wz is not None and e1 > e0 else None,
                           "coverage_vs_max": cov, "pixel_hash16": _h16(px), "qc_flags": qc_flags_for_plane(px)})
            dup_rows.append({"id": f"{oid}L{z}", "patient": r["pair_key"], "pixels": px})
        cand = []
        for i, s in enumerate(sel["slices"]):
            img, win = render_candidate(vol, aff, s["index"], "legacy_percentile")
            px = np.asarray(img)[:, :, 0]
            img.save(priv / "images" / f"{oid}_cand_f{int(round(s['fraction'] * 100)):02d}.png")
            diff = None
            if i < len(legacy) and "pixel_hash16" in legacy[i]:
                leg = np.asarray(Image.open(priv / "images" / f"{oid}_legacy_z{levels[i]:03d}.png"))[:, :, 0].astype(float)
                diff = float(np.abs(leg - px).mean())
            cand.append({**s, "window": win, "pixel_hash16": _h16(px), "qc_flags": s["qc_flags"] + [f for f in qc_flags_for_plane(px) if f not in s["qc_flags"]],
                         "mean_abs_pixel_diff_vs_legacy_level": diff,
                         "difference_reason": ("same axial index" if diff == 0 else "different axial index / physical height")})
            dup_rows.append({"id": f"{oid}C{i}", "patient": r["pair_key"], "pixels": px})
        center = sel["slices"][len(sel["slices"]) // 2]["index"]
        ctrl = {}
        base_win = window_for(vol, "legacy_percentile")
        for mode in NORMALIZATION_MODES:
            lo, hi = window_for(vol, mode)
            plane = axial_display_plane(vol, aff, center)
            ctrl[mode] = {"window": [lo, hi], "pixel_hash16": _h16(apply_window(plane, lo, hi)),
                          "mean_abs_diff_vs_legacy_norm": float(np.abs(apply_window(plane, lo, hi).astype(float) - apply_window(plane, *base_win).astype(float)).mean())}
        nz = vol[vol != 0]
        stats = {"nonzero_fraction": float(nz.size / vol.size), "p0.5_p99.5": [float(x) for x in np.percentile(nz[::16], [0.5, 99.5])],
                 "mean_nonzero": float(nz.mean()), "std_nonzero": float(nz.std())}
        ess = offsets_2p5d(vol, aff, center, (-8, 0, 8))
        info = {"modality": r["modality"], "pair_key": r["pair_key"], "shape": geo["shape"], "axcodes": geo["axcodes"], "spacing_mm": geo["spacing_mm"],
                "units": geo["units"], "qform_code": geo["qform_code"], "sform_code": geo["sform_code"],
                "qform_sform_conflict": geo["qform_sform_conflict"], "obliquity_deg": geo["obliquity_deg"], "geometry_flags": geo["flags"],
                "display": {**legacy_display_axes(aff), **display_aspect_report(geo["shape"], geo["spacing_mm"], LEGACY_SIZE)},
                "intensity": stats, "selection": {k2: v for k2, v in sel.items() if k2 != "slices"}, "legacy": legacy, "candidate_primary": cand,
                "normalization_controls_at_center": ctrl, "offsets_2p5d_mm": ess["slices"],
                "token_estimate_primary_3_images": visual_token_estimate(LEGACY_SIZE, 3), "token_estimate_2p5d_3_images": visual_token_estimate(LEGACY_SIZE, 3),
                "elapsed_s": round(time.perf_counter() - t0, 3)}
        if r.get("mask_path"):
            mimg = nib.load(r["mask_path"])
            info["mask"] = check_image_mask_geometry(vol, aff, np.asanyarray(mimg.dataobj), np.asarray(mimg.affine), set(manifest.get("mask_allowed_labels", [0, 1, 2, 3, 4])))
        rows_pub[oid] = info
        rows_priv[oid] = {"nifti_path": r["nifti_path"], "sha256": sha, "affine": geo["affine"], "mask_path": r.get("mask_path")}
        loaded[oid] = (r, vol, aff)
    pairs = defaultdict(list)
    for oid, (r, vol, aff) in loaded.items():
        pairs[r["pair_key"]].append(oid)
    pair_geo = {}
    for key, ids in pairs.items():
        if len(ids) == 2:
            (ra, va, aa), (rb, vb, ab) = loaded[ids[0]], loaded[ids[1]]
            g = compare_grids(va.shape, aa, vb.shape, ab)
            if g["same_voxel_grid"]:
                g["shift_nmi"] = shift_sensitivity_nmi(va, vb)
            pair_geo[key] = g
    dup = duplicate_guard(dup_rows)
    flags = Counter(f for v in rows_pub.values() for f in v["geometry_flags"])
    result = {"version": VERSION, "rows": len(rows_pub), "legacy_control_pixels_unchanged": bool(unchanged), "pair_geometry": pair_geo,
              "duplicate_guard": {k: v for k, v in dup.items() if k != "flagged_pairs"}, "geometry_flag_counts": dict(flags),
              "modalities": dict(Counter(v["modality"] for v in rows_pub.values())), "threads": os.environ.get("OMP_NUM_THREADS"),
              "model_loaded": False, "training": False, "clinical_adjudication": False}
    (pub / "pilot_public_aggregate.json").write_text(json.dumps({**result, "per_row_opaque": rows_pub}, indent=1, sort_keys=True))
    (priv / "full_geometry_PRIVATE.json").write_text(json.dumps({"rows": rows_priv, "duplicate_pairs": dup["flagged_pairs"]}, indent=1))
    return result
