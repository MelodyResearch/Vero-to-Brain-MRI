"""CPU inspect/preflight by default; fresh human+independent review for execute."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mri_rl import paired_signal_candidate_probe_v1 as p


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--spec',type=Path,required=True);parser.add_argument('--scope-output',type=Path,required=True);parser.add_argument('--execute',action='store_true')
    parser.add_argument('--human-authorization',type=Path);parser.add_argument('--expected-human-sha256');parser.add_argument('--review',type=Path);parser.add_argument('--expected-review-sha256');args=parser.parse_args()
    spec=json.loads(args.spec.read_text());scope=p.prepare(spec);p.require(args.scope_output.resolve().is_relative_to(ROOT/'local_data/paired_signal_candidate_probe_v1'),'Private scope output required');args.scope_output.parent.mkdir(parents=True,exist_ok=True)
    if args.scope_output.exists():p.require(json.loads(args.scope_output.read_text())==scope,'Immutable scope differs; use fresh version')
    else:p.entry.driver.write_json(args.scope_output,scope)
    if args.execute:
        p.require(args.human_authorization and args.expected_human_sha256 and args.review and args.expected_review_sha256,'No human permission yet: new explicit authorization and independent exact review required')
        p.execute(spec,scope,human_path=args.human_authorization,human_sha=args.expected_human_sha256,review_path=args.review,review_sha=args.expected_review_sha256)
    else:print(json.dumps({'status':scope['status'],'scope_sha256':p.canonical(scope),'scope_file_sha256':p.entry.digest(args.scope_output),'model_loads':0,'training_authorized':False,'budget':p.budget()}))

if __name__=='__main__':main()
