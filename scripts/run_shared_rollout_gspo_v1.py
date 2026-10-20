"""CPU preparation, or strictly gated future native shared-rollout one-step GSPO control. No approval flag."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from mri_rl.shared_rollout_gspo_entry_v1 import main
if __name__=='__main__':main()
