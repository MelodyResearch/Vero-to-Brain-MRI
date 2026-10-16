"""Build image-only visual reasoning tasks from MSD Task04 masks.

This builder never uses a segmentation mask to choose a slice or alter model input.
The mask is read only for private target derivation and source validation.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw


def central_slice_index(shape: tuple[int, ...]) -> int:
    """Return the prespecified middle slice in native voxel axis 2."""
    return (shape[2] - 1) // 2


def validate_pair(image_path: Path, mask_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image_nifti = nib.load(str(image_path))
    mask_nifti = nib.load(str(mask_path))
    image = np.asarray(image_nifti.dataobj, dtype=np.float32)
    mask = np.asarray(mask_nifti.dataobj)
    if image.ndim != 3 or image.shape != mask.shape:
        raise ValueError("image/mask must be matching 3D volumes")
    if not np.allclose(image_nifti.affine, mask_nifti.affine, atol=1e-4):
        raise ValueError("image/mask affines do not match")
    if not np.isfinite(image_nifti.affine).all() or abs(np.linalg.det(image_nifti.affine[:3, :3])) < 1e-8:
        raise ValueError("invalid image affine")
    if not np.isfinite(image).all() or not np.isfinite(mask).all():
        raise ValueError("nonfinite image or mask values")
    if not np.all(np.isin(mask, [0, 1, 2])):
        raise ValueError("unexpected segmentation label")
    if not np.array_equal(mask, mask.astype(np.uint8)):
        raise ValueError("noninteger segmentation label")
    return image, mask.astype(np.uint8), image_nifti.affine


def derive_targets(mask: np.ndarray, affine: np.ndarray) -> dict | None:
    """Use only a geometry-fixed central slice; never search labels for a better slice."""
    k = central_slice_index(mask.shape)
    plane = mask[:, :, k]
    foreground = np.argwhere(plane > 0)
    if not len(foreground):
        return None
    center_ij = foreground.mean(axis=0)
    center_rounded = np.rint(center_ij).astype(int)
    world = affine @ np.array([center_ij[0], center_ij[1], k, 1.0])
    counts = np.bincount(plane.ravel(), minlength=3)
    if counts[1] == counts[2]:
        dominant = "tie"
    else:
        dominant = "anterior" if counts[1] > counts[2] else "posterior"
    return {
        "slice_index_k": k,
        "centroid_voxel_ij": center_rounded.tolist(),
        "centroid_world_mm": [round(float(x), 3) for x in world[:3]],
        "central_slice_counts": {"anterior": int(counts[1]), "posterior": int(counts[2])},
        "dominant_region": dominant,
    }


def render_image_only_slice(image: np.ndarray, k: int, destination: Path) -> None:
    """Render with intensity statistics from the image, plus voxel axes only."""
    plane = image[:, :, k]
    lo, hi = np.percentile(plane, [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    pixels = np.uint8(np.rint(255 * np.clip((plane - lo) / (hi - lo), 0, 1)))
    # Horizontal display coordinate is native voxel i, vertical coordinate is j.
    base = Image.fromarray(pixels.T).resize((384, 384), Image.Resampling.NEAREST).convert("RGB")
    canvas = Image.new("RGB", (452, 438), "white")
    canvas.paste(base, (50, 12))
    draw = ImageDraw.Draw(canvas)
    draw.text((50, 402), f"i: 0 to {plane.shape[0] - 1}", fill="black")
    draw.text((3, 12), "j=0", fill="black")
    draw.text((3, 382), f"j={plane.shape[1] - 1}", fill="black")
    draw.text((280, 402), f"k={k}", fill="black")
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def build_tasks(dataset_root: Path, private_output: Path, aggregate_audit: Path) -> dict:
    manifest = json.loads((dataset_root / "dataset.json").read_text())
    pairs = manifest["training"]
    if len(pairs) != manifest["numTraining"]:
        raise ValueError("manifest training count mismatch")
    private_output.mkdir(parents=True, exist_ok=True)
    image_dir = private_output / "images"
    counts = Counter()
    shapes = Counter()
    affines = Counter()
    errors: list[dict] = []
    records: list[dict] = []
    for ordinal, entry in enumerate(pairs):
        image_path = dataset_root / entry["image"]
        mask_path = dataset_root / entry["label"]
        try:
            image, mask, affine = validate_pair(image_path, mask_path)
            targets = derive_targets(mask, affine)
            counts["geometry_valid_pairs"] += 1
            shapes[str(tuple(image.shape))] += 1
            affines[str(tuple(np.round(affine.ravel(), 4)))] += 1
            if targets is None:
                counts["central_slice_no_foreground"] += 1
                continue
            if targets["dominant_region"] == "tie":
                counts["central_slice_ties"] += 1
            image_name = f"scan_{ordinal:03d}_central.png"
            render_image_only_slice(image, targets["slice_index_k"], image_dir / image_name)
            common = {
                "source_image": str(image_path.resolve()),
                "source_mask": str(mask_path.resolve()),
                "input_image": str((image_dir / image_name).resolve()),
                "image_shape_ijk": list(image.shape),
                "affine_voxel_to_world_mm": affine.tolist(),
                "slice_index_k": targets["slice_index_k"],
                "split_status": "UNVERIFIED_PATIENT_IDENTITY_NO_HELD_OUT_CLAIM",
                "target_provenance": "MSD Task04 segmentation mask; not independent between task categories",
            }
            records.append({
                **common,
                "task_id": "central_slice_hippocampal_centroid",
                "prompt": "In this fixed central MRI slice, give the center of the hippocampal region as native voxel coordinates (i,j). Return two integers.",
                "answer": targets["centroid_voxel_ij"],
                "answer_world_mm_reference": targets["centroid_world_mm"],
                "scoring": "Euclidean distance in native (i,j) voxels <= 3; report continuous distance too",
            })
            if targets["dominant_region"] != "tie":
                records.append({
                    **common,
                    "task_id": "central_slice_anterior_posterior_dominance",
                    "prompt": "In this fixed central MRI slice, which hippocampal subregion occupies more area: anterior or posterior? Answer one word.",
                    "answer": targets["dominant_region"],
                    "scoring": "Exact category match against mask-derived central-slice pixel counts",
                    "private_label_counts": targets["central_slice_counts"],
                })
                counts[f"dominant_{targets['dominant_region']}"] += 1
            counts["rendered_pairs"] += 1
        except Exception as exc:
            errors.append({"manifest_ordinal": ordinal, "error": f"{type(exc).__name__}: {exc}"})
    (private_output / "tasks.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
    audit = {
        "dataset_name": manifest.get("name"),
        "manifest_training_count": len(pairs),
        "manifest_test_count_unlabelled": len(manifest.get("test", [])),
        "geometry_valid_pairs": counts["geometry_valid_pairs"],
        "rendered_pairs": counts["rendered_pairs"],
        "central_slice_no_foreground": counts["central_slice_no_foreground"],
        "central_slice_ties": counts["central_slice_ties"],
        "task_record_count": len(records),
        "task_counts": dict(Counter(x["task_id"] for x in records)),
        "dominance_counts": {"anterior": counts["dominant_anterior"], "posterior": counts["dominant_posterior"]},
        "shape_distribution": dict(shapes),
        "affine_distribution_count": len(affines),
        "validation_errors": errors,
    }
    aggregate_audit.parent.mkdir(parents=True, exist_ok=True)
    aggregate_audit.write_text(_format_report(audit))
    return audit


def _format_report(audit: dict) -> str:
    return (
        "# MSD Task04 first image-only task build\n\n"
        "Status: `TESTED_CPU` for data validation and task construction; model inference and RL training: `NOT_RUN`.\n\n"
        f"- Training image/mask entries: {audit['manifest_training_count']}; unlabelled official test scans: {audit['manifest_test_count_unlabelled']}.\n"
        f"- Pairs passing full-image/mask load, dimensional, shape, affine, finite-value, and label checks: {audit['geometry_valid_pairs']}.\n"
        f"- Fixed central slices rendered: {audit['rendered_pairs']}; no foreground: {audit['central_slice_no_foreground']}; exact anterior/posterior area ties: {audit['central_slice_ties']}.\n"
        f"- Private task records: {audit['task_record_count']}; categories: {json.dumps(audit['task_counts'], sort_keys=True)}.\n"
        f"- Dominance answers: {json.dumps(audit['dominance_counts'], sort_keys=True)}.\n"
        f"- Distinct image/mask shapes: {len(audit['shape_distribution'])}; distinct affines: {audit['affine_distribution_count']}; validation errors: {len(audit['validation_errors'])}.\n\n"
        "## Task construction and limits\n\n"
        "Both prompts use the same image-only native-axis central slice, selected as `(shape[2]-1)//2` before reading mask contents. The image is displayed with voxel axes and no mask pixels or mask-derived crop. The private mask supplies (1) a 2D centroid coordinate target and (2) an anterior-versus-posterior area category. The answers differ in format, but they share one segmentation source and cannot establish independent cross-task generalization. A slice is not a native 3D encoder. The source volumes are cropped hippocampus MR images, not whole-brain or longitudinal scans.\n\n"
        "The anterior/posterior category has a severe majority-class baseline: "
        f"{max(audit['dominance_counts'].values())}/{sum(audit['dominance_counts'].values())} "
        f"= {max(audit['dominance_counts'].values()) / max(1, sum(audit['dominance_counts'].values())):.1%}. "
        "It should not be counted as a convincing learned reasoning result without balanced evaluation and shortcut controls. "
        "The manifest does not establish patient identity or repeated-scan linkage. No patient-disjoint held-out split or clinical validity is claimed. The official test scans have no downloadable masks in this package and are excluded from supervised evaluation. Before any claimed model comparison: verify case lineage/patient keys, lock the split, test image-only and metadata-only shortcuts, and retain a separate independent dataset. These tasks are a first engineering substrate for the broader Vero multi-category transfer, not a sufficient final taxonomy.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--aggregate-audit", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_tasks(args.dataset_root, args.private_output, args.aggregate_audit), indent=2))


if __name__ == "__main__":
    main()
