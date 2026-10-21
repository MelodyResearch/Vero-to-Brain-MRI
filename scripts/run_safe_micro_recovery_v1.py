"""Fresh exact Q→B→DEV24 authority; no old scope retry or model work during preparation."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from mri_rl.safe_micro_recovery_entry_v1 import main
if __name__=='__main__':raise SystemExit(main())
