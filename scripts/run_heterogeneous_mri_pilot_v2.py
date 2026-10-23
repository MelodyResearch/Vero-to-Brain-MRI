"""Heterogeneous MRI pilot V2 exact authority (prepare / launch / child). Preparation makes no model call."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from mri_rl.heterogeneous_mri_pilot_entry_v2 import main
if __name__=='__main__':raise SystemExit(main())
