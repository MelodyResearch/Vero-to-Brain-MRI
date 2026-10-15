"""Deterministic native-grid 3D MRI evidence with invertible pixel mapping.

Images use the neurological RAS+ convention: R points right, S points up.
No annotation or answer is used for slice selection.
"""

from pathlib import Path

import numpy as np
from PIL import Image


def _axis(plane):
    try:
        return {"sagittal": 0, "coronal": 1, "axial": 2}[plane]
    except KeyError as exc:
        raise ValueError(f"Unknown plane {plane!r}") from exc


def extract_plane(volume: np.ndarray, plane: str, voxel: tuple[int, int, int]):
    """Return a 2D array in displayed orientation plus fixed voxel index."""
    if volume.ndim != 3 or len(voxel) != 3 or any(not 0 <= v < n for v, n in zip(voxel, volume.shape)):
        raise ValueError("Expected in-bounds 3D voxel")
    fixed = voxel[_axis(plane)]
    if plane == "axial":
        slice_data = volume[:, :, fixed]
    elif plane == "coronal":
        slice_data = volume[:, fixed, :]
    else:
        slice_data = volume[fixed, :, :]
    return slice_data.T[::-1, :], fixed


def voxel_to_pixel(voxel, plane, shape):
    i, j, k = voxel
    if any(not 0 <= v < n for v, n in zip(voxel, shape)):
        raise ValueError("Voxel outside volume")
    if plane == "axial":
        return i, shape[1] - 1 - j
    if plane == "coronal":
        return i, shape[2] - 1 - k
    if plane == "sagittal":
        return j, shape[2] - 1 - k
    raise ValueError(f"Unknown plane {plane!r}")


def pixel_to_voxel(pixel, plane, slice_index, shape):
    x, y = pixel
    if plane == "axial":
        voxel = (x, shape[1] - 1 - y, slice_index)
    elif plane == "coronal":
        voxel = (x, slice_index, shape[2] - 1 - y)
    elif plane == "sagittal":
        voxel = (slice_index, x, shape[2] - 1 - y)
    else:
        raise ValueError(f"Unknown plane {plane!r}")
    if any(not 0 <= v < n for v, n in zip(voxel, shape)):
        raise ValueError("Pixel outside plane")
    return voxel


def display_to_voxel(pixel, plane, slice_index, shape, scale):
    """Invert isotropic-display pixel center to nearest native voxel."""
    native = tuple(int(round((p + 0.5) / s - 0.5)) for p, s in zip(pixel, scale))
    return pixel_to_voxel(native, plane, slice_index, shape)


def render_volume_evidence(data: np.ndarray, affine: np.ndarray, output_dir: Path) -> dict:
    """Render 3 axial slices and one coronal/sagittal on the native grid."""
    if data.ndim != 3 or not np.isfinite(data).all() or not (data > 0).any():
        raise ValueError("Need finite nonempty 3D MRI data")
    if np.asarray(affine).shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError("Need finite 4x4 affine")
    nonzero = data > 0
    bounds = []
    for axis in range(3):
        used = np.any(nonzero, axis=tuple(i for i in range(3) if i != axis))
        indexes = np.flatnonzero(used)
        bounds.append((int(indexes[0]), int(indexes[-1])))
    center = tuple((lo + hi) // 2 for lo, hi in bounds)
    lo, hi = np.percentile(data[nonzero], [1, 99])
    if hi <= lo:
        raise ValueError("No usable MRI intensity contrast")
    axial = sorted(set(int(round(bounds[2][0] + q * (bounds[2][1] - bounds[2][0]))) for q in (0.25, 0.5, 0.75)))
    if len(axial) != 3:
        raise ValueError("Insufficient occupied axial slices")
    selections = [("axial", (center[0], center[1], k)) for k in axial]
    selections += [("coronal", center), ("sagittal", center)]
    spacing = np.linalg.norm(np.asarray(affine)[:3, :3], axis=0)
    if np.any(spacing <= 0):
        raise ValueError("Invalid voxel spacing")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    views = []
    for plane, voxel in selections:
        pixels, fixed = extract_plane(data, plane, voxel)
        mapped = np.uint8(np.rint(255 * np.clip((pixels - lo) / (hi - lo), 0, 1)))
        if plane == "axial":
            horizontal_mm, vertical_mm = spacing[0], spacing[1]
        elif plane == "coronal":
            horizontal_mm, vertical_mm = spacing[0], spacing[2]
        else:
            horizontal_mm, vertical_mm = spacing[1], spacing[2]
        base_mm = min(horizontal_mm, vertical_mm)
        width = max(1, int(round(mapped.shape[1] * horizontal_mm / base_mm)))
        height = max(1, int(round(mapped.shape[0] * vertical_mm / base_mm)))
        scale = (width / mapped.shape[1], height / mapped.shape[0])
        path = output_dir / f"{plane}_{fixed:04d}.png"
        Image.fromarray(mapped).resize((width, height), Image.Resampling.BILINEAR).save(path)
        anchor_native = voxel_to_pixel(voxel, plane, data.shape)
        anchor_display = tuple(int(round((p + 0.5) * s - 0.5)) for p, s in zip(anchor_native, scale))
        views.append({"plane": plane, "slice_index": fixed, "path": str(path),
                      "shape_pixels_native": list(mapped.shape[::-1]), "shape_pixels_display": [width, height],
                      "display_scale": list(scale), "voxel_anchor": list(voxel),
                      "anchor_pixel_native": list(anchor_native), "anchor_pixel_display": list(anchor_display)})
    return {"shape_voxels": list(data.shape), "affine": np.asarray(affine).tolist(),
            "intensity_window": [float(lo), float(hi)], "selection": "occupied_bbox_25_50_75_axial_plus_center_coronal_sagittal",
            "views": views}
