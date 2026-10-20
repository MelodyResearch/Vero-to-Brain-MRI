#!/usr/bin/env python3
"""Build a source-bound finite A/B proposal; no execute path or model imports."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mri_rl.paired_prompt_ab_preparation_v1 import proposal,refuse_native_execution

INPUT_SHA='f4257ce14dd2918e5efd1bb01d27d86a1fcbbe890c881c527f07f2bbe3e63c5c'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--execute',action='store_true',help='Always refuses; this is a preparation entry only')
    args=p.parse_args()
    if args.execute:refuse_native_execution()
    if args.inputs is None or args.output is None:p.error('--inputs and --output are required for CPU preparation')
    assert sha(args.inputs)==INPUT_SHA,'Frozen input metadata changed'
    inputs=json.loads(args.inputs.read_text())
    assert inputs['same_inputs_both_arms'] and inputs['future_training_each_K4_one_unchanged_policy_required']
    assert [r['role'] for r in inputs['rows_PRIVATE']]==['SEQUENCE_F','SEQUENCE_T','DALLAS_SAME','DALLAS_DIFFERENT']
    assert all(r['record_PRIVATE']['extra_info']['internal_split']=='train' for r in inputs['rows_PRIVATE'])
    assert inputs['budget']['charged_cap_per_arm']==9040 and inputs['parent']['Adam_step']==60
    d=proposal();d['input_plan_sha256']=INPUT_SHA
    d['parent_receipt_sha256']=inputs['parent']['receipt_sha256']
    d['parent_adapter_sha256_from_receipt']=inputs['parent']['adapter_sha256_from_receipt']
    d['parent_payloads_verified_this_preparation']=False
    d['source_sha256']={name:sha(ROOT/name) for name in ('research/CURRENT_RESEARCH_PLAN.md','research/CURRENT_RESEARCH_PLAN.lock.json',
        'results/PAIRED_PROMPT_CPU_MECHANISM_V1.json','results/PAIRED_PROMPT_SAVED_MATCHED_AVAILABILITY_V2.json',
        'src/mri_rl/paired_prompt_groups_v1.py','src/mri_rl/paired_prompt_ab_preparation_v1.py')}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(d,f,indent=2,sort_keys=True);f.write('\n')
    assert not any(k.split('.')[0] in ('mlx','mlx_vlm','torch','transformers') for k in sys.modules)
    print(json.dumps({'status':d['status'],'output_sha256':sha(args.output),'model_loads':0,'optimizer_updates':0}))


if __name__=='__main__':main()
