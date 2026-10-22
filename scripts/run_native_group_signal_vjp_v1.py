#!/usr/bin/env python3
"""Bounded native read-only A/B/C VJP experiment (Parent Adam52 + Common32).

  --prepare   CPU only (root runs locally): authenticate V2, derive jobs/budget, bind inputs/sources/payloads; writes a NEW attempt dir.
  --launch    controller: exact scope/review/fresh-human admission, non-resident admission, ONE guarded child, then CPU analysis.
  --child     the single native child (spawned by --launch only).
  --analyze   CPU analysis of a committed run (after the child exited).
No GPU/native work happens without --launch plus an independently reviewed scope and a fresh human permission file.
"""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_k, '1')
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from mri_rl.native_group_signal_vjp_entry_v1 import main    # noqa: E402

if __name__ == '__main__':
    raise SystemExit(main())
