"""CPU scope preparation or separately authorized one-shot historical GPU cache continuation."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from mri_rl.accuracy_recovery_gpu_cache_v4 import main
if __name__=='__main__':main()
