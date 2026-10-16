"""Small, deterministic 3D T1 retrieval diagnostic, not a clinical model.

Only the private runner reads BIDS paths/identifiers to assemble ground truth.
The spatial scoring functions receive image arrays alone. Affines are used to
orient the arrays consistently, never as spatial-score features.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom


def spatial_signature(data: np.ndarray, size: int = 32) -> np.ndarray:
    """Robustly normalize a foreground-cropped 3D volume to a fixed grid."""
    a = np.asarray(data, dtype=np.float32)
    if a.ndim != 3 or not np.isfinite(a).all():
        raise ValueError("Expected a finite 3D image")
    positive = a[a > 0]
    if positive.size < 100:
        raise ValueError("Too few foreground voxels")
    where = np.where(a > 0)
    if any(axis.size == 0 for axis in where):
        raise ValueError("Empty foreground")
    lo = [max(0, int(axis.min()) - 2) for axis in where]
    hi = [min(a.shape[i], int(axis.max()) + 3) for i, axis in enumerate(where)]
    crop = a[tuple(slice(lo[i], hi[i]) for i in range(3))]
    lower, upper = np.percentile(positive, [2, 98])
    if upper <= lower:
        raise ValueError("Degenerate intensity range")
    crop = np.clip((crop - lower) / (upper - lower), 0, 1)
    resized = zoom(crop, tuple(size / n for n in crop.shape), order=1, prefilter=False)
    # scipy.ndimage.zoom may differ by one voxel from target due to rounding.
    resized = resized[tuple(slice(0, size) for _ in range(3))]
    if resized.shape != (size, size, size):
        raise ValueError("Unexpected resampled shape")
    return resized.ravel().astype(np.float32)


def cosine_score(a: np.ndarray, b: np.ndarray) -> float:
    """Centered cosine (Pearson) score of spatially corresponding voxels."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("Signatures must be equal-length vectors")
    a, b = a - a.mean(), b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom > 0 else float("-inf")


def rank_candidates(scores: np.ndarray) -> np.ndarray:
    """Return deterministic zero-based candidate order, highest score first."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("Expected finite one-dimensional scores")
    return np.lexsort((np.arange(scores.size), -scores))


def _header_vector(image: nib.Nifti1Image) -> np.ndarray:
    """Geometry-only shortcut: shape, spacing, orientation and center in mm."""
    affine = np.asarray(image.affine, dtype=float)
    spacing = nib.affines.voxel_sizes(affine)
    direction = affine[:3, :3] / spacing
    center = nib.affines.apply_affine(affine, (np.asarray(image.shape) - 1) / 2)
    return np.r_[np.asarray(image.shape) / 256, spacing, direction.ravel(), center / 100]


def _summary_vector(data: np.ndarray) -> np.ndarray:
    """Nonspatial intensity/foreground shortcut; no voxel coordinates."""
    positive = data[data > 0]
    if positive.size < 100:
        raise ValueError("Too few positive voxels")
    quantiles = np.percentile(positive, [10, 25, 50, 75, 90, 99])
    scale = max(float(quantiles[2]), 1.0)
    return np.r_[positive.size / data.size, quantiles / scale]


def _scaled_distance_scores(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Untrained Euclidean control with only predeclared feature scaling."""
    # Per-feature scale comes from the 12 gallery scans, without query labels.
    scale = np.std(candidates, axis=0)
    scale = np.where(scale > 1e-6, scale, 1.0)
    return -np.linalg.norm((candidates - query) / scale, axis=1)


def retrieval_metrics(score_matrix: np.ndarray) -> dict:
    """Score independent queries; diagonal is the private ground-truth match."""
    scores = np.asarray(score_matrix, dtype=float)
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("Expected square query-by-candidate score matrix")
    ranks = []
    for i, row in enumerate(scores):
        order = rank_candidates(row)
        ranks.append(int(np.flatnonzero(order == i)[0] + 1))
    n = len(ranks)
    correct = sum(r == 1 for r in ranks)
    return {"queries": n, "top1_correct": correct, "top1_accuracy": correct / n,
            "mean_reciprocal_rank": float(np.mean(1 / np.asarray(ranks))),
            "median_rank": float(np.median(ranks)), "ranks_private": ranks}


def wilson_interval(correct: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if not 0 <= correct <= n or n == 0:
        raise ValueError("Invalid count")
    p = correct / n
    d = 1 + z*z/n
    center = (p + z*z/(2*n)) / d
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return center-half, center+half


def _verified_pairs(repo: Path) -> list[tuple[str, Path, Path]]:
    receipts = [repo / "local_data/ds004856_evidence/private_receipt.json",
                repo / "local_data/dallas_expansion_private_receipt.json",
                repo / "local_data/dallas_second_batch_private_receipt.json"]
    pairs = {}
    for receipt_path in receipts:
        receipt = json.loads(receipt_path.read_text())
        rows = receipt["records"]
        if "images" in rows[0]:
            for row in rows:
                pairs[row["subject"]] = [x for x in row["images"] if x["relative_path"].endswith("T1w.nii.gz")]
        else:
            for row in rows:
                if row["relative_path"].endswith("T1w.nii.gz"):
                    pairs.setdefault(row["subject"], []).append(row)
    result = []
    for subject, images in sorted(pairs.items()):
        if len(images) != 2:
            raise ValueError("Expected exactly two T1 scans per subject")
        images.sort(key=lambda x: x["relative_path"])
        if "ses-wave1" not in images[0]["relative_path"] or "ses-wave2" not in images[1]["relative_path"]:
            raise ValueError("Unexpected visit ordering")
        paths = []
        for row in images:
            path = repo / "local_data/ds004856" / row["relative_path"]
            if not path.exists() or path.stat().st_size != row["bytes"]:
                raise ValueError("Missing or wrong-sized verified volume")
            with path.open("rb") as handle:
                h = hashlib.file_digest(handle, "sha256").hexdigest()
            if h != row["sha256"]:
                raise ValueError("SHA-256 mismatch")
            paths.append(path)
        result.append((subject, *paths))
    return result


def run_private_experiment(repo: Path, private_output: Path) -> dict:
    """Run all controls; write identifiers and row-wise ranks only locally."""
    pairs = _verified_pairs(repo)
    n = len(pairs)
    if n != 20:
        raise ValueError(f"Expected 20 verified pairs, found {n}")
    spatial, header, summary = [[], []], [[], []], [[], []]
    for _, wave1, wave2 in pairs:
        for visit, path in enumerate((wave1, wave2)):
            image = nib.load(str(path))
            data = nib.as_closest_canonical(image).get_fdata(dtype=np.float32)
            spatial[visit].append(spatial_signature(data))
            header[visit].append(_header_vector(image))
            summary[visit].append(_summary_vector(data))
    spatial_scores = np.array([[cosine_score(q, c) for c in spatial[0]] for q in spatial[1]])
    controls = {}
    for name, vectors in (("header_geometry", header), ("nonspatial_intensity", summary)):
        gallery = np.asarray(vectors[0])
        controls[name] = np.asarray([_scaled_distance_scores(q, gallery) for q in vectors[1]])
    results = {"spatial_image": retrieval_metrics(spatial_scores)}
    results.update({name: retrieval_metrics(scores) for name, scores in controls.items()})
    results["metadata_identifier"] = {"queries": n, "top1_correct": n, "top1_accuracy": 1.0,
                                      "note": "Trivial BIDS subject-key lookup, never a model feature"}
    results["uniform_chance"] = {"queries": n, "expected_correct": 1.0,
                                  "expected_top1_accuracy": 1/n, "expected_mrr": sum(1/k for k in range(1,n+1))/n}
    output = {"dataset": "OpenNeuro ds004856 v1.3.0", "pair_count": n,
              "subjects_private": [p[0] for p in pairs], "results": results,
              "spatial_score_matrix_private": spatial_scores.tolist()}
    private_output.parent.mkdir(parents=True, exist_ok=True)
    private_output.write_text(json.dumps(output, indent=2))
    return output
