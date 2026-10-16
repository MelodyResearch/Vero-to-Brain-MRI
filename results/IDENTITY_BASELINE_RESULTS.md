# Dallas T1 same-person retrieval: bounded real-image diagnostic

**Status:** `TESTED_CPU` for a deterministic image-similarity rule; `NOT_RUN` for VLM inference, learned embedding models, fine-tuning, and RL. This is an engineering diagnostic, not biological change measurement or a clinically validated identity system.

## Predeclared experiment

- Source: OpenNeuro Dallas ds004856 v1.3.0. Two prior pilot subjects and ten first-expansion subjects provided the initial **12 people × two visits = 24 T1 MPRAGE NIfTI volumes**. A second, separately selected batch added eight, bringing the frozen rerun to **20 people × two visits = 40 T1 volumes**. The private source receipts specify exact relative files, byte lengths, and SHA-256 hashes; the runner rechecked all 40 files before the rerun. Selection was intentionally age-spread and Philips-MPRAGE constrained, so neither stage is a population sample.
- For each wave-2 query, rank all wave-1 candidates in the stage's gallery (12 in the initial stage; 20 in the frozen rerun). The only true answer is the same person's wave-1 scan. **Top-1 accuracy per query** is the primary metric. Candidate order breaks exact ties deterministically. Mean reciprocal rank is descriptive. The score cells are not treated as independent trials.
- Spatial rule, fixed before inspecting labels: reorient each 3D image to canonical axes using its NIfTI affine; crop its positive foreground bounding box with a two-voxel margin; clip intensities to the positive-voxel 2nd–98th percentiles; trilinearly resize to 32³; center each vector; rank by Pearson/cosine similarity. The affine is used only for physical orientation, not included in the spatial score. No training or hyperparameter selection on either the 12-person initial stage or 20-person rerun.
- Controls: uniform independent random choice (expected top-1 1/n for a gallery of n people); BIDS subject-key lookup (trivial metadata upper bound, never a model input); untrained standardized Euclidean distance on NIfTI header geometry; and distance on **nonspatial intensity/foreground summaries**. The latter removes voxel locations but does still derive from image intensities, so it is an acquisition/summary shortcut control rather than literally image-free.

## Observed query-level results: frozen 20-person rerun

| Rule | Correct / 20 | Top-1 | Mean reciprocal rank | Descriptive Wilson 95% interval for top-1 |
|---|---:|---:|---:|---:|
| Spatial 3D image similarity | 8 | 40.0% | 0.580 | 21.9%–61.3% |
| Header geometry shortcut | 3 | 15.0% | 0.300 | 5.2%–36.0% |
| Nonspatial intensity/foreground summary | 10 | 50.0% | 0.656 | 29.9%–70.1% |
| BIDS identifier lookup, trivial | 20 | 100% | 1.000 | Not meaningful |
| Uniform prior-free chance | 1 expected | 5.0% expected | 0.180 expected | Not applicable |

On paired queries in the 20-person run, the spatial rule alone was right for four people, the summary rule alone for six, both for four, and neither for six. The spatial rule exceeds the random-choice expectation, but uncertainty is wide and the stronger nonspatial control prevents an anatomical-identity claim. The cohort's deliberate age spread, scanner/protocol matching, and finite gallery can all affect retrieval. We did not estimate a generalization rate, an independent external cohort effect, or longitudinal biological change.

## Initial 12-person stage, preserved

The first execution, before the second batch was available, used the same frozen spatial rule and controls: spatial 4/12 (33.3%, MRR 0.495), header geometry 3/12 (25.0%, MRR 0.436), nonspatial summary 8/12 (66.7%, MRR 0.790), metadata key 12/12, random expected 1/12 (8.3%). Descriptive Wilson intervals were 13.8%–60.9%, 8.9%–53.2%, and 39.1%–86.2%, respectively. Its ignored private receipt is preserved separately. The 20-person result is a **new gallery of larger size**, not eight independent additional Bernoulli trials or a direct paired comparison of accuracies.

## Interpretation and next gate

This no-metadata spatial rule is **not a clean verifier** for same-person image reasoning. An intensity distribution without spatial anatomy performed better at both stages. A stronger identity claim would require a frozen method on more people and scanners, a well-matched negative gallery, an acquisition-normalized/defaced control, and a spatial-versus-nonspatial ablation evaluated by independent people. The current result is suitable for discovering shortcut risk and for requiring a no-image/nonspatial control in subsequent MRI reward experiments; it is not suitable as a Vero reward or BrainTRACE evaluation result.

## Reproduction and privacy

Code: `src/mri_rl/identity_baseline.py`; tests: `tests/test_identity_baseline.py`. Run from the repository root with:

```sh
PYTHONPATH=src python3 - <<'PY'
from pathlib import Path
from mri_rl.identity_baseline import run_private_experiment
repo = Path.cwd()
run_private_experiment(repo, repo / "local_data/identity_baseline_20_subjects_private.json")
PY
```

The ignored local JSON contains subject IDs, per-query ranks, and the score matrix for audit. The initial 12-person record is `local_data/identity_baseline_12_subjects_private.json`; the frozen rerun is `local_data/identity_baseline_20_subjects_private.json`. No subject IDs, image data, or private score rows are included here. `PYTHONPATH=src python3 -m pytest tests/test_identity_baseline.py -q` passed (3 tests).
