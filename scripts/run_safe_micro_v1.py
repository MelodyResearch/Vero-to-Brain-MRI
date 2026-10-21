"""CPU stage1 cache-only scope preparation; execution requires distinct exact permission."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from mri_rl.safe_micro_entry_v1 import main
if __name__=='__main__':main()
