#!/usr/bin/env python3
"""CPU-only spatial (mri_tumor_grid) task pilot over a PRIVATE input (schema mri-spatial-task-pilot-input-v1).

Usage: run_mri_spatial_task_pilot_v1.py --input PRIVATE.json --private-out NEW_DIR --public-out NEW_FILE.json
       run_mri_spatial_task_pilot_v1.py --input PRIVATE.json --describe-keys   (key names only)
Max 3 cases, threads 1, ~512 MiB RSS and 120 s/case soft budgets. No model/torch/MLX/training.
Stdout carries only aggregate counts, opaque ids and an error class; details go to the private log.
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
from mri_rl.mri_spatial_task_pilot_v1 import describe_keys, run_pilot  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--private-out")
    ap.add_argument("--public-out")
    ap.add_argument("--describe-keys", action="store_true", help="print only key NAMES of the first case, never values")
    a = ap.parse_args()
    try:
        if a.describe_keys:
            print(json.dumps(describe_keys(json.loads(Path(a.input).read_text())), sort_keys=True))
            return 0
        if not (a.private_out and a.public_out):
            raise ValueError("--private-out and --public-out required")
        res = run_pilot(json.loads(Path(a.input).read_text()), a.private_out, a.public_out)
    except Exception as e:  # no paths/ids on stdout
        print(json.dumps({"status": "RUN_ERROR", "error_class": type(e).__name__}))
        return 2
    print(json.dumps({k: res[k] for k in ("cases_attempted", "stopped_remaining", "stop_reason", "status_counts", "elapsed_s", "peak_rss_mib")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
