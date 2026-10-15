"""NIfTI spatial geometry utilities; no diagnosis or annotation inference."""

from pathlib import Path

import nibabel as nib
import numpy as np


def load_canonical_volume(path: str | Path) -> dict:
    image = nib.load(str(path))
    if len(image.shape) != 3:
        raise ValueError("Expected a 3D NIfTI volume")
    if not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3, :3])) < 1e-8:
        raise ValueError("Invalid affine")
    canonical = nib.as_closest_canonical(image)
    data = canonical.get_fdata(dtype=np.float32)
    if not np.isfinite(data).all():
        raise ValueError("Nonfinite MRI intensity")
    return {"data": data, "affine": canonical.affine,
            "orientation": nib.aff2axcodes(canonical.affine),
            "spacing_mm": tuple(float(v) for v in nib.affines.voxel_sizes(canonical.affine)),
            "source": str(path)}


def voxel_to_world(voxel, affine):
    return tuple(float(v) for v in nib.affines.apply_affine(affine, voxel))


def world_to_voxel(world, affine):
    return tuple(float(v) for v in nib.affines.apply_affine(np.linalg.inv(affine), world))
