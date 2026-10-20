"""Inspect CPU preparation or invoke strictly gated future native continuation."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from mri_rl.class_margin_aux_v2_entry import main
if __name__=='__main__':main()
