"""External once-only CACHE_ONLY child witness; no GPU/model imports."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from mri_rl.safe_micro_witness_v1 import main
if __name__=='__main__':raise SystemExit(main())
