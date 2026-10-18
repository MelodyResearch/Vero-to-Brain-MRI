#!/usr/bin/env python3
"""Run bounded CPU parity diagnosis with all native model frameworks blocked."""
import argparse
import importlib.abc
import json
import os
from pathlib import Path
import sys
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
class BlockNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'mlx','mlx_vlm','torch','transformers','safetensors'}:
            raise RuntimeError('CPU-only parity audit blocks model frameworks')
sys.meta_path.insert(0,BlockNative())
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mri_rl.root_cause_gspo_parity_v1 import build_report, sha

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--official-root',type=Path,required=True)
    args=ap.parse_args()
    result=build_report(ROOT,args.official_root)
    out=ROOT/'results/ROOT_CAUSE_GSPO_PARITY_V1.json'
    out.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    private=ROOT/'local_data/root_cause_cpu_v1/parity'
    private.mkdir(parents=True,exist_ok=True)
    (private/'receipt.json').write_text(json.dumps({'public_report_sha256':sha(out),
        'script_sha256':sha(__file__),'module_sha256':sha(ROOT/'src/mri_rl/root_cause_gspo_parity_v1.py'),
        'official_root':str(args.official_root.resolve()),'model_framework_imports_blocked':True,
        'native_execution':'NOT_RUN','counterexample_count':len(result['counterexamples'])},indent=2)+'\n')
    print(json.dumps({'status':result['status'],'counterexamples':len(result['counterexamples']),
        'report_sha256':sha(out),'native_e2e':'NOT_VERIFIED'}))
if __name__=='__main__':main()
