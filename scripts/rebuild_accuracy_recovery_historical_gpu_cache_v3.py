"""Cache v3 continuation: `plan` / `prepare-scope` / `adopt-continuation` never import MLX; `execute` needs a prepared scope + human authorization +
independent review (exact sha256 pins each). No approval flag, no config switch."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from mri_rl.accuracy_recovery_gpu_cache_v3 import main
if __name__=='__main__':main()
