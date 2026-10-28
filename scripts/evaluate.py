#!/usr/bin/env python3
"""Read-only final evaluator; defaults to non-native validation."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from mri_rl.final_evaluation import main
if __name__=='__main__':raise SystemExit(main())
