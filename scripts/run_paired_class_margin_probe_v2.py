#!/usr/bin/env python3
"""Prepare only by default;fresh exact human+independent evidence for execution."""
import argparse,json,sys
from pathlib import Path
W=Path(__file__).resolve().parents[1];sys.path.insert(0,str(W/'src'))
from mri_rl import paired_class_margin_probe_v2 as p


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--spec',type=Path,required=True);a.add_argument('--scope-output',type=Path,required=True)
    a.add_argument('--execute',action='store_true');a.add_argument('--human-authorization',type=Path);a.add_argument('--expected-human-sha256');a.add_argument('--review',type=Path);a.add_argument('--expected-review-sha256');args=a.parse_args()
    scope=p.prepare(json.loads(args.spec.read_text()));p.require(args.scope_output.resolve().is_relative_to(W/'local_data/paired_class_margin_probe_v2'),'Private new scope required')
    args.scope_output.parent.mkdir(parents=True,exist_ok=True)
    if args.scope_output.exists():p.require(json.loads(args.scope_output.read_text())==scope,'Frozen scope differs;write new version')
    else:args.scope_output.write_text(json.dumps(scope,indent=2,sort_keys=True)+'\n')
    if args.execute:
        p.require(args.human_authorization and args.expected_human_sha256 and args.review and args.expected_review_sha256,'Fresh authorization/review required;old consumed permission cannot execute')
        p.execute(json.loads(args.spec.read_text()),scope,human_path=args.human_authorization,human_sha=args.expected_human_sha256,review_path=args.review,review_sha=args.expected_review_sha256)
    else:print(json.dumps({'status':scope['status'],'scope_sha256':p.canonical(scope),'scope_file_sha256':p.hashlib.sha256(args.scope_output.read_bytes()).hexdigest(),'budget':scope['budget'],'model_loads':0,'training_authorized':False}))


if __name__=='__main__':main()
