#!/usr/bin/env python3
"""CPU-only prepare entry for the frozen slice-input comparison (NOT_RUN: no model, no GPU).

Usage: prepare_mri_slice_comparison_v1.py --binding PRIVATE.json --private-out NEW_DIR --public-out NEW_FILE.json
"""
from __future__ import annotations
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = "1"
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mri_rl.mri_slice_comparison_v1 import main  # noqa: E402

if __name__ == "__main__":
    try:
        sys.exit(main(["prepare", *sys.argv[1:]]))
    except Exception as e:  # no paths/ids on stdout
        print(json.dumps({"status": "ERROR", "error_class": type(e).__name__}))
        sys.exit(2)
