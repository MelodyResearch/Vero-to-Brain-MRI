"""Prepare exact CPU scope, or admit a separately authorized future compound run."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from mri_rl.accuracy_stage_entry_v1 import main
if __name__=='__main__':main()
