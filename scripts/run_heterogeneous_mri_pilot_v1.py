"""Heterogeneous MRI pilot v1 exact authority (census / prepare / launch / child). Census and preparation make no model call."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from mri_rl.heterogeneous_mri_pilot_entry_v1 import main
if __name__=='__main__':raise SystemExit(main())
