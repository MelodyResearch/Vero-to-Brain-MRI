"""V2_NATIVE_READONLY_V1 exact authority (CPU prepare / --launch / --child). Preparation makes no model call and imports no MLX."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from mri_rl.v2_native_readonly_entry_v1 import main
if __name__ == '__main__':
    raise SystemExit(main())
