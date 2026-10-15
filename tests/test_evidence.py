from pathlib import Path

import numpy as np
from PIL import Image

from mri_rl.evidence import display_to_voxel, extract_plane, pixel_to_voxel, render_volume_evidence, voxel_to_pixel


def test_plane_coordinates_roundtrip():
    volume = np.arange(5 * 6 * 7).reshape(5, 6, 7)
    for plane, voxel in [('axial', (2, 3, 4)), ('coronal', (2, 3, 4)), ('sagittal', (2, 3, 4))]:
        image, slice_index = extract_plane(volume, plane, voxel)
        pixel = voxel_to_pixel(voxel, plane, volume.shape)
        assert pixel_to_voxel(pixel, plane, slice_index, volume.shape) == voxel
        assert image[pixel[1], pixel[0]] == volume[voxel]


def test_render_evidence_produces_real_array_slices(tmp_path):
    data = np.zeros((12, 14, 16), dtype=np.float32)
    data[2:10, 3:12, 4:14] = np.arange(8 * 9 * 10).reshape(8, 9, 10) + 1
    receipt = render_volume_evidence(data, np.eye(4), tmp_path)
    assert len(receipt['views']) == 5
    assert len({item['slice_index'] for item in receipt['views'] if item['plane'] == 'axial'}) == 3
    assert all(Path(item['path']).exists() for item in receipt['views'])
    assert all(Image.open(item['path']).getextrema()[1] > 0 for item in receipt['views'])


def test_anisotropic_view_respects_physical_aspect_and_mapping(tmp_path):
    data = np.ones((10, 10, 8), dtype=np.float32)
    data[2:8, 2:8, 2:6] = 10
    affine = np.diag([1.0, 1.0, 5.0, 1.0])
    receipt = render_volume_evidence(data, affine, tmp_path)
    coronal = next(v for v in receipt['views'] if v['plane'] == 'coronal')
    assert Image.open(coronal['path']).size == (10, 40)
    assert display_to_voxel(coronal['anchor_pixel_display'], coronal['plane'], coronal['slice_index'],
                            data.shape, coronal['display_scale']) == tuple(coronal['voxel_anchor'])
