#!/usr/bin/env python3
"""CPU analyzer for saved slice-comparison model rows (needs a declared provenance; real outputs need a receipt sha256).

Usage: analyze_mri_slice_comparison_v1.py --inputs comparison_inputs_PRIVATE.json --results RESULTS.json
       --provenance ACTUAL_MODEL_OUTPUT|SYNTHETIC_FIXTURE [--receipt-sha256 H] --public-out NEW_FILE.json
No model is loaded; there are currently no model rows (NOT_RUN).
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
        sys.exit(main(["analyze", *sys.argv[1:]]))
    except Exception as e:  # no paths/ids on stdout
        print(json.dumps({"status": "ERROR", "error_class": type(e).__name__}))
        sys.exit(2)
