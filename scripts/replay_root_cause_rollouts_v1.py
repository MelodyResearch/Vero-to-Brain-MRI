#!/usr/bin/env python3
"""One CPU process; fixed historical train-only evidence inventory."""
import argparse
import importlib.abc
import json
import os
import sys
from pathlib import Path
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'mlx','mlx_vlm','torch','transformers','tensorflow','jax'}:raise RuntimeError('Native/model import forbidden: '+fullname)
sys.meta_path.insert(0,NoModels())
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from mri_rl.root_cause_reward_replay_v1 import run_replay
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--evidence-root',type=Path,required=True)
p.add_argument('--private-output',type=Path,required=True)
p.add_argument('--public-output',type=Path,required=True)
a=p.parse_args()
r=run_replay(a.evidence_root,a.private_output,a.public_output)
print(json.dumps({'status':r['status'],'groups':r['groups'],'accepted_updates':r['accepted_updates'],'totals':r['totals']},sort_keys=True))
