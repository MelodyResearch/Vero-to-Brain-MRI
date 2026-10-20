"""Run deterministic CPU math/counterexamples only; no native launch interface."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mri_rl.paired_prompt_groups_v1 import mechanism_report,canonical


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/PAIRED_PROMPT_CPU_MECHANISM_V1.json')
    args=parser.parse_args();report=mechanism_report()
    files=('src/mri_rl/paired_prompt_groups_v1.py','tests/test_paired_prompt_groups_v1.py','scripts/check_paired_prompt_mechanism_v1.py')
    report['implementation_sha256']={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files}
    text=canonical(report)+'\n';args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists():
        if args.output.read_text()!=text:raise ValueError('Preserve immutable previous result; choose a new output path')
    else:
        with args.output.open('x') as stream:stream.write(text)
    assert not any(name=='mlx' or name.startswith('mlx.') for name in sys.modules)
    print(json.dumps({'status':report['status'],'native_calls':0,'optimizer_updates':0,'result_sha256':hashlib.sha256(text.encode()).hexdigest()}))

if __name__=='__main__':main()
