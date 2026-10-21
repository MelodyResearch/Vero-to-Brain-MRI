"""Exact dynamic resource/recovery pilot entry. No native call without new grant."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from mri_rl.safe_micro_pilot_entry_v3 import main
if __name__=='__main__':raise SystemExit(main())
