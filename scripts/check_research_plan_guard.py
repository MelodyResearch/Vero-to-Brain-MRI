#!/usr/bin/env python3
"""Check staged or committed trees without trusting a mutable local approval flag."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from mri_rl.research_plan_guard import PATHS,check_trees


def git(*args,check=True):
    p=subprocess.run(['git',*args],capture_output=True)
    if check and p.returncode:raise ValueError(p.stderr.decode().strip())
    return p


def snapshot(revision):
    result={}
    for path in PATHS:
        obj=f':{path}' if revision==':' else f'{revision}:{path}'
        probe=git('cat-file','-e',obj,check=False)
        if probe.returncode==0:result[path]=git('show',obj).stdout
    return result


def history_report(head,*,require_plan=True):
    # Only commits touching the three governance files need tree checks. First
    # parent follows the committed branch state, including merge results.
    commits=git('log','--reverse','--first-parent','--format=%H',head,'--',*PATHS).stdout.decode().splitlines()
    previous={};checks=[]
    scientific_source=snapshot('a71bb39a5b75d4cb65b7a958fa79548ff0805254')
    for commit in commits:
        current=snapshot(commit)
        if not previous and not current:continue
        # A publication branch may have a short result log before importing the
        # already-frozen scientific plan. This is not the raw scientific log.
        # Only exact source-plan/lock bytes qualify; no approval flag or arbitrary
        # new plan is admitted, and later mutations still use ordinary guards.
        if PATHS[0] not in previous and PATHS[0] not in current and PATHS[1] not in current:
            previous=current
            continue
        if PATHS[0] not in previous and PATHS[2] in previous and PATHS[0] in current:
            if any(current.get(k)!=scientific_source.get(k) or k not in scientific_source for k in PATHS[:2]):
                return {'status':'BLOCKED','errors':['IMPORTED_PLAN_NOT_EXACT_SCIENTIFIC_SOURCE'],'history_checks':checks}
            report=check_trees({},current)
            report['imported_from_scientific_source']='a71bb39a5b75d4cb65b7a958fa79548ff0805254'
        else:
            report=check_trees(previous,current)
        checks.append({'commit':commit,'status':report['status'],'errors':report['errors']})
        if report['errors']:
            return {**report,'history_checks':checks,'history_anchor':'first first-parent governance addition'}
        previous=current
    if not previous and not snapshot(head) and not require_plan:
        return {'status':'PRE_FREEZE_BASELINE','errors':[],'history_checks':checks}
    final=check_trees(previous,snapshot(head))
    return {**final,'history_checks':checks,'history_anchor':'first first-parent governance addition'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',default='HEAD')
    mode=p.add_mutually_exclusive_group();mode.add_argument('--staged',action='store_true');mode.add_argument('--head')
    p.add_argument('--history',action='store_true',help='Check every first-parent governance transition; no HEAD^ new-branch shortcut')
    args=p.parse_args()
    try:
        base=git('rev-parse','--verify',args.base+'^{commit}').stdout.decode().strip()
        candidate=':' if not args.head else git('rev-parse','--verify',args.head+'^{commit}').stdout.decode().strip()
        if candidate==':' and base!=git('rev-parse','HEAD').stdout.decode().strip():
            raise ValueError('Staged baseline must be actual HEAD; an older --base cannot reauthorize a changed plan')
        committed=base if candidate==':' else candidate
        history=history_report(committed,require_plan=candidate!=':')
        # Even a staged-only check rejects unauthorized changes hidden in earlier
        # commits. --history is a stricter committed-tree reporting mode.
        if history['errors']:report=history
        elif args.history and candidate!=':':report=history
        else:
            report=check_trees(snapshot(base),snapshot(candidate))
            report['history_checks']=history['history_checks']
        report.update(base_commit=base,candidate=candidate)
        if not report['errors']:report['plan_diff']=''
        print(json.dumps(report,ensure_ascii=False,indent=2));return 2 if report['errors'] else 0
    except ValueError as e:print(str(e),file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
