"""CPU-only exact quant-cache plan/rebuild (no model load; `plan` never imports MLX). No approval flag."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from mri_rl.accuracy_recovery_cpu_cache_v1 import main
if __name__=='__main__':main()
