"""Gated GPU re-quantization of the original base cache: `plan` / `adopt-retained` never import MLX; `execute` needs a fresh
authorization artifact bound to this script's and the module's sha256. No approval flag, no config switch."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from mri_rl.accuracy_recovery_gpu_cache_v2 import main
if __name__=='__main__':main()
