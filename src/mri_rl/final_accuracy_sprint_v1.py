"""Bounded supervised accuracy sprint contracts. Importing performs no MLX work.

DEV24 is repeatedly explored development data, never an independent test.
The sealed16 cohort cannot be supplied here, and this module cannot unseal it.
"""
from __future__ import annotations
import hashlib
import itertools
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from mri_rl.shared_rollout_gspo_core_v1 import publish_once, replace_atomic

SEEDS=(0,1,2)
LABELS=('FLAIR','contrast-enhanced T1')
BUDGET={'candidate_methods':1,'training_runs':3,'updates_per_run':28,
        'backward_per_run':56,'TF_positions_per_run':14028,
        'greedy_per_run':24,'returned_tokens_per_run':6144,
        'model_loads_per_run':1,'run_window_seconds':3600,
        'total_window_seconds':10800,'cache_quantizations':0}


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def verify_files(pins):
    for path,h in pins.items():require(Path(path).is_file() and sha(path)==h,'Frozen input changed: '+str(path))


def claim_run(path,seed,scope):
    """Claim before native import; consumed/failed runs cannot be renamed/retried."""
    path=Path(path)
    value=json.loads(path.read_text()) if path.exists() else {'scope':scope,'seeds':[],'started_unix':time.time()}
    require(value['scope']==scope and seed in SEEDS and seed not in value['seeds'],'Consumed run or different scope')
    require(len(value['seeds'])<3,'At most three runs for one candidate')
    require(time.time()-value['started_unix']<BUDGET['total_window_seconds'],'Original total window exhausted')
    if seed!=0:require(0 in value['seeds'],'Initial seed0 must run first')
    value['seeds'].append(seed);replace_atomic(path,value)
    return value


def replication_allowed(m):
    return m['correct']>=19 and m['invalid']==0 and m['class_correct'][0]>=10 and m['class_correct'][1]>=8 and m['both_correct_pairs']>=7


def validate_dev(records):
    require(len(records)==24 and len({r['id'] for r in records})==24,'Exact24 unique DEV rows required')
    people=defaultdict(set)
    for item in records:
        r=item['row_PRIVATE'];e=r['extra_info']
        require(r['ability']=='mri_sequence' and e['internal_split']=='internal_dev','Only frozen sequence DEV records')
        require(e.get('level') in (45,91,136),'Frozen DEV level required')
        y=r['reward_model']['ground_truth'];require(y in LABELS,'Two source sequence labels only')
        key=(e['level'],y);require(key not in people[e['patient_key']],'Duplicate patient/level/modality')
        people[e['patient_key']].add(key)
    expected={(z,y) for z in (45,91,136) for y in LABELS}
    require(len(people)==4 and all(v==expected for v in people.values()),'Exact four people with six paired images each required')
    return set(people)


def metrics(rows):
    counts=[0,0];hits=[0,0];confusion=[[0,0,0],[0,0,0]];people=defaultdict(list)
    for r in rows:
        y=LABELS.index(r['truth']);p=r['prediction'];require(p in (-1,0,1),'Unknown prediction')
        counts[y]+=1;hits[y]+=int(p==y);confusion[y][2 if p<0 else p]+=1;people[r['patient']].append(int(p==y))
    require(all(counts),'Both sequence labels required')
    values={p:sum(v)/len(v) for p,v in people.items()}
    # Enumerate cluster bootstrap for small DEV; 4^4=256 draws, not24 independent draws.
    draws=[sum(v)/len(v) for v in itertools.product(values.values(),repeat=len(values))] if len(values)<=6 else []
    draws.sort()
    ci=[draws[math.floor(.025*(len(draws)-1))],draws[math.ceil(.975*(len(draws)-1))]] if draws else None
    return {'cases':len(rows),'patients':len(people),'correct':sum(hits),'accuracy':sum(hits)/len(rows),
            'class_counts':counts,'class_correct':hits,'FLAIR_recall':hits[0]/counts[0],'T1_recall':hits[1]/counts[1],
            'balanced_accuracy':sum(hits[i]/counts[i] for i in (0,1))/2,
            'invalid':sum(c[2] for c in confusion),'confusion':confusion,'confusion_columns':['FLAIR','contrast-enhanced T1','INVALID'],
            'patient_accuracy':values,'patient_cluster_bootstrap95':ci,
            'uncertainty_limit':'Four explored DEV patients; descriptive cluster bootstrap, not independent generalization.'}
