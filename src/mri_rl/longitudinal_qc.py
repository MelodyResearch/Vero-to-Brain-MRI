"""Bounded physical-space, within-person T1 registration feasibility audit.

This is image-alignment QC, not a longitudinal biological change estimator.
No subject-level result or NIfTI derivative should be committed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
from dipy.align.imaffine import (
    AffineRegistration,
    MutualInformationMetric,
    transform_centers_of_mass,
)
from dipy.align.transforms import RigidTransform3D
from nibabel.processing import resample_from_to, resample_to_output


def check_physical_geometry(image: nib.Nifti1Image) -> dict:
    """Reject unusable 3D geometry and describe its physical voxel grid."""
    if len(image.shape) != 3 or any(n < 2 for n in image.shape):
        raise ValueError("Expected nondegenerate 3D volume")
    affine = np.asarray(image.affine, dtype=float)
    if not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-6:
        raise ValueError("Invalid physical affine")
    spacing = np.asarray(nib.affines.voxel_sizes(affine), dtype=float)
    if not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("Invalid voxel spacing")
    return {
        "shape": list(image.shape),
        "spacing_mm": spacing.tolist(),
        "orientation": list(nib.aff2axcodes(affine)),
        "voxel_volume_mm3": float(abs(np.linalg.det(affine[:3, :3]))),
    }


def _scaled_intensity(data: np.ndarray) -> np.ndarray:
    """Monotone robust scaling for MI without depending on scanner gain."""
    data = np.nan_to_num(np.asarray(data, dtype=np.float32), nan=0, posinf=0, neginf=0)
    positive = data[data > 0]
    if positive.size < 100:
        raise ValueError("Too few nonzero voxels")
    upper = float(np.percentile(positive, 99.5))
    if upper <= 0:
        raise ValueError("Nonpositive intensity scale")
    return np.clip(data / upper, 0, 1)


def _similarity(static: np.ndarray, moving: np.ndarray) -> dict:
    """Symmetric foreground overlap and Pearson similarity in common foreground."""
    static_mask = static > 0.05
    moving_mask = moving > 0.05
    inter = static_mask & moving_mask
    denom = int(static_mask.sum() + moving_mask.sum())
    dice = float(2 * inter.sum() / denom) if denom else float("nan")
    if inter.sum() < 100:
        corr = float("nan")
    else:
        a, b = static[inter].astype(float), moving[inter].astype(float)
        corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
    return {"foreground_dice": dice, "foreground_pearson": corr, "intersection_voxels": int(inter.sum())}


def register_pair(
    wave1_path: str | Path,
    wave2_path: str | Path,
    *,
    registration_mm: float = 4.0,
    evaluation_mm: float = 3.0,
) -> dict:
    """Fit 6-DOF MI rigid alignment on 4-mm grids; assess on 3-mm grids.

    World-space NIfTI affines are honored throughout. The moving wave2 scan is
    transformed into the fixed wave1 space. The evaluation grid is separate
    from the grid used to estimate the transform.
    """
    fixed = nib.load(str(wave1_path))
    moving = nib.load(str(wave2_path))
    fg, mg = check_physical_geometry(fixed), check_physical_geometry(moving)
    fixed_coarse = resample_to_output(fixed, voxel_sizes=(registration_mm,) * 3, order=1)
    moving_coarse = resample_to_output(moving, voxel_sizes=(registration_mm,) * 3, order=1)
    f = _scaled_intensity(fixed_coarse.get_fdata(dtype=np.float32))
    m = _scaled_intensity(moving_coarse.get_fdata(dtype=np.float32))
    start = transform_centers_of_mass(f, fixed_coarse.affine, m, moving_coarse.affine)
    reg = AffineRegistration(
        metric=MutualInformationMetric(nbins=24, sampling_proportion=0.2),
        level_iters=[30, 15, 5], sigmas=[2, 1, 0], factors=[4, 2, 1], verbosity=0,
    )
    mapping = reg.optimize(
        f, m, RigidTransform3D(), None,
        static_grid2world=fixed_coarse.affine,
        moving_grid2world=moving_coarse.affine,
        starting_affine=start.affine,
    )
    fixed_eval = resample_to_output(fixed, voxel_sizes=(evaluation_mm,) * 3, order=1)
    fe = _scaled_intensity(fixed_eval.get_fdata(dtype=np.float32))
    # Physical-grid identity is the no-registration control.
    raw = resample_from_to(moving, fixed_eval, order=1)
    before = _similarity(fe, _scaled_intensity(raw.get_fdata(dtype=np.float32)))
    # AffineMap stores the static-world -> moving-world coordinate map.
    moving_eval = resample_to_output(moving, voxel_sizes=(evaluation_mm,) * 3, order=1)
    aligned = mapping.transform(
        _scaled_intensity(moving_eval.get_fdata(dtype=np.float32)),
        interpolation="linear",
        image_grid2world=moving_eval.affine,
        sampling_grid_shape=fixed_eval.shape,
        sampling_grid2world=fixed_eval.affine,
    )
    after = _similarity(fe, aligned)
    xform = np.asarray(mapping.affine, dtype=float)
    rotation = xform[:3, :3]
    angle_deg = float(np.degrees(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1))))
    return {
        "fixed_geometry": fg,
        "moving_geometry": mg,
        "registration_mm": registration_mm,
        "evaluation_mm": evaluation_mm,
        "before": before,
        "after": after,
        "rigid_angle_deg": angle_deg,
        "rigid_translation_mm": float(np.linalg.norm(xform[:3, 3])),
        "transform_world": xform.tolist(),
        "qc_pass": bool(
            np.isfinite(after["foreground_dice"])
            and np.isfinite(after["foreground_pearson"])
            and after["foreground_dice"] >= before["foreground_dice"] - 0.01
            and after["foreground_pearson"] >= before["foreground_pearson"] - 0.01
            and after["foreground_dice"] >= 0.75
            and after["foreground_pearson"] >= 0.50
        ),
    }


def run_local_audit(dataset_dir: Path, receipt_paths: list[Path], output_path: Path) -> dict:
    """Execute all verified wave1/wave2 T1 pairs; private output only."""
    started = time.perf_counter()
    records = []
    for receipt_path in receipt_paths:
        receipt = json.loads(receipt_path.read_text())
        rows = receipt["records"]
        if "images" not in rows[0]:
            by_subject = {}
            for image in rows:
                if image["relative_path"].endswith("T1w.nii.gz"):
                    by_subject.setdefault(image["subject"], []).append(image)
            rows = [{"subject": s, "images": xs} for s, xs in sorted(by_subject.items())]
        for row in rows:
            images = [x for x in row["images"] if x["relative_path"].endswith("T1w.nii.gz")]
            if len(images) != 2:
                continue
            images.sort(key=lambda x: x["relative_path"])
            if "ses-wave1" not in images[0]["relative_path"] or "ses-wave2" not in images[1]["relative_path"]:
                raise ValueError("Unexpected visit ordering")
            paths = [dataset_dir / x["relative_path"] for x in images]
            pair_started = time.perf_counter()
            result = register_pair(*paths)
            records.append({"subject": row["subject"], "paths": [str(x) for x in paths],
                            "runtime_seconds": time.perf_counter() - pair_started,
                            "visual_qc_status": "NOT_CERTIFIED",
                            "image_evaluation_eligible": False,
                            **result})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {"runtime_seconds": time.perf_counter() - started, "records": records}
    output_path.write_text(json.dumps(output, indent=2))
    return output
