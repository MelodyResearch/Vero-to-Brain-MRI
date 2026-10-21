"""CPU preparation/analysis, or strictly gated future native Parent52/A/B evaluation for VERO_TO_MRI_ACCURACY_RECOVERY_V1. No approval flag."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from mri_rl.accuracy_recovery_eval_v1 import main
if __name__=='__main__':main()
