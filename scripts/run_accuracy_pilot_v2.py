"""Accuracy Pilot V2 exact authority (prepare / launch / resume / child). Preparation makes no model call."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from mri_rl.accuracy_pilot_v2_entry import main
if __name__=='__main__':raise SystemExit(main())
