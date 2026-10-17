"""Finite controller replay of authenticated, already generated B28 training groups.

Repeated cases reuse their saved four outcomes. This is fixed-policy logged
replay, not new inference, off-policy learning, or actual optimizer updates.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mri_rl.v2_effective_balance import QuotaControl,TASKS,PILOT,FULL,digest,require
from mri_rl.mlx_gspo_smoke import task_group_is_informative


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--primary-root',type=Path,required=True)
    ap.add_argument('--expected-proof-sha256',required=True);ap.add_argument('--expected-train-sha256',required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--private-output',type=Path,required=True);a=ap.parse_args()
    base=a.primary_root/'local_data/c_two_task_pool_v1';proofp=a.primary_root/'results/COMMON_B28_CALIBRATION76_INDEPENDENT_QUANTILES.json'
    require(sha(proofp)==a.expected_proof_sha256,'External calibration proof digest differs');proof=json.loads(proofp.read_text())
    require(proof['verdict']=='PASS_WITH_LIMITATIONS' and proof['weight_updates']==0 and proof['rows']==76 and proof['ordered_raw_76_verified'] is True,'Actual common-B28 fixed-policy76 proof required')
    train=base/'train.verl.jsonl';require(sha(train)==a.expected_train_sha256,'Frozen training data differs')
    rows=[json.loads(x) for x in train.read_text().splitlines() if x.strip()];ids={r['id']:r for r in rows}
    run=base/'calibration_v1/run';chain=proof['evidence_sha256']['journal_receipt_chain'];require(len(chain)==77,'Complete76 receipt chain required')
    require(sha(run/'initial_state/receipt.json')==chain[0],'Initial fixed-policy receipt changed')
    cases={};tokens={};support={t:Counter() for t in TASKS}
    for i in range(76):
        p=run/f'row_{i:04d}/receipt.json';require(sha(p)==chain[i+1],'Accepted calibration receipt changed');r=json.loads(p.read_text());v=r['record'];rid=v['id']
        require(rid in ids and ids[rid]['ability']==v['task'] and r['optimizer_updates']==0,'Training case/route/zero-update mismatch')
        gs=[{'accuracy':acc,'format':fmt,'overlong':0.,'total':.8*acc+.2*fmt} for acc,fmt in zip(v['accuracy'],v['format'],strict=True)]
        cases[rid]=(gs,v['generated_tokens']);tokens[rid]=v['input_tokens'];t=v['task']
        support[t]['rows']+=1;support[t]['accepted_fixed_groups']+=int(task_group_is_informative(gs))
        support[t]['zero_accuracy_variance']+=int(len(set(v['accuracy']))==1)
        support[t]['not_all_full_format']+=int(any(x!=1 for x in v['format']))
    require(set(cases)==set(ids),'Exact76 training cases required')
    public=[];private=[]
    for mode,cfg in [('pilot',PILOT),('full_predeclared',FULL)]:
        for seed in (7,11,13):
            c=QuotaControl(rows,tokens,seed=seed,**cfg)
            while c.phase=='READY':
                d=c.offer()
                if d is None:break
                gs,lengths=cases[d['row']['id']];c.commit(c.assess(d,gs,lengths))
            public.append({'configuration':mode,'seed':seed,'terminal':c.phase,'target_per_task':c.target,'token_cap':c.cap,'replayed_token_cost':c.used,
                'replayed_complete_groups':len(c.history),'accepted_logged_groups':c.accepted,'attempted_logged_groups':dict(Counter(h['task'] for h in c.history)),
                'actual_new_optimizer_updates':0,'new_model_generations':0,'controller_state_sha256':digest(c.snapshot())})
            private.append({'configuration':mode,'seed':seed,'controller_state_PRIVATE':c.snapshot()})
    require(a.private_output.resolve().is_relative_to(a.primary_root.resolve()/'local_data'),'Individual replay state must stay private')
    a.private_output.parent.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(private,indent=2,sort_keys=True)+'\n'
    if a.private_output.exists():require(a.private_output.read_text()==payload,'Preserve previous private replay')
    else:a.private_output.write_text(payload)
    result={'schema':'V2-fixed-common-B28-training-group-replay-v1','evidence_tier':'CPU_REPLAY_OF_AUTHENTICATED_TRAINING_ONLY_LOGGED_GROUPS',
        'author':'Codex /root','calibration_proof_sha256':a.expected_proof_sha256,'training_manifest_sha256':a.expected_train_sha256,
        'controller_source_sha256':sha(ROOT/'src/mri_rl/v2_effective_balance.py'),'script_sha256':sha(__file__),'private_trace_sha256':sha(a.private_output),
        'fixed_policy_case_support':{t:dict(v) for t,v in support.items()},'replays':public,'actual_new_optimizer_updates':0,'new_model_generations':0,
        'limitations':['Repeated cases reuse their original fixed-B28 four responses; accepted logged groups are not actual updates or independent draws.',
            'No learned policy is simulated; no causal MRI benefit, true future V2 feasibility or patient generalization is established.',
            'The callback native/controller checkpoint integration still requires separate review and real restore evidence before any V2 training.'],
        'sealed_test_used':False,'frozen_original_training_changed':False}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'output_sha256':sha(a.output),'replays':public},sort_keys=True))


if __name__=='__main__':main()
