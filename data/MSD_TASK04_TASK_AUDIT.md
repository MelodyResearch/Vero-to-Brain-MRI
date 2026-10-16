# MSD Task04 first image-only task build

Status: `TESTED_CPU` for data validation and task construction; model inference and RL training: `NOT_RUN`.

- Training image/mask entries: 260; unlabelled official test scans: 130.
- Pairs passing full-image/mask load, dimensional, shape, affine, finite-value, and label checks: 260.
- Fixed central slices rendered: 260; no foreground: 0; exact anterior/posterior area ties: 0.
- Private task records: 520; categories: {"central_slice_anterior_posterior_dominance": 260, "central_slice_hippocampal_centroid": 260}.
- Dominance answers: {"anterior": 14, "posterior": 246}.
- Distinct image/mask shapes: 236; distinct affines: 2; validation errors: 0.

## Task construction and limits

Both prompts use the same image-only native-axis central slice, selected as `(shape[2]-1)//2` before reading mask contents. The image is displayed with voxel axes and no mask pixels or mask-derived crop. The private mask supplies (1) a 2D centroid coordinate target and (2) an anterior-versus-posterior area category. The answers differ in format, but they share one segmentation source and cannot establish independent cross-task generalization. A slice is not a native 3D encoder. The source volumes are cropped hippocampus MR images, not whole-brain or longitudinal scans.

The anterior/posterior category has a severe majority-class baseline: 246/260 = 94.6%. It should not be counted as a convincing learned reasoning result without balanced evaluation and shortcut controls. The manifest does not establish patient identity or repeated-scan linkage. No patient-disjoint held-out split or clinical validity is claimed. The official test scans have no downloadable masks in this package and are excluded from supervised evaluation. Before any claimed model comparison: verify case lineage/patient keys, lock the split, test image-only and metadata-only shortcuts, and retain a separate independent dataset. These tasks are a first engineering substrate for the broader Vero multi-category transfer, not a sufficient final taxonomy.
