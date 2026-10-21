#!/usr/bin/env python3
"""CPU-only MRI data-preparation pilot over a PRIVATE input manifest.

Usage: run_mri_data_prep_v1.py --manifest PRIVATE.json --private-out DIR --public-out DIR
Private DIR gets images + full geometry (paths, hashes). Public DIR gets only an
aggregate JSON with opaque IDs (no paths, subject IDs, source hashes or images).
No model, torch, MLX, training or fitting. Threads fixed to 1.
"""
from __future__ import annotations
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = "1"
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mri_rl.mri_data_prep_v1 import run_pilot  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--private-out", required=True)
    ap.add_argument("--public-out", required=True)
    a = ap.parse_args()
    res = run_pilot(json.loads(Path(a.manifest).read_text()), a.private_out, a.public_out)
    print(json.dumps({k: res[k] for k in ("rows", "legacy_control_pixels_unchanged", "geometry_flag_counts", "duplicate_guard")}, sort_keys=True))


if __name__ == "__main__":
    main()
