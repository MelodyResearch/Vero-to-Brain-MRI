#!/usr/bin/env python3
"""CPU-only frozen TRAIN pair routing and intervention/cost comparison."""
import argparse
import hashlib
import json
from pathlib import Path
from mri_rl.paired_interaction_signal_v1 import attach_pairs, operation_budget, pair_loss, score_derivatives
from mri_rl.paired_z045_exposure_v2 import describe

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(x,f,indent=2,sort_keys=True);f.write('\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--old-worktree',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];old=Path(a.old_worktree).resolve();data=old/'local_data/paired_z045_data_v1/candidate_v2'
    paths={'pool':data/'paired_z045_sequence56_PRIVATE.verl.jsonl','membership':data/'source_and_pair_membership_PRIVATE.json',
        'schedule':data/'reference_common_stage_next32_PRIVATE.json',
        'forced':old/'local_data/paired_z045_exposure_v2/forced_pair_same32_PRIVATE.json'}
    if sha(paths['pool'])!='c257ef5481c0e97a907fdc75bfcba6345744bd3769b3e6391f0563147a17a770' or sha(paths['membership'])!='bf92f7e560ce3ed3b3629427f590d8c6f6cf4c4c3bae6ab04dc14af42ad4ce99' or sha(paths['schedule'])!='44a874e50d45ee5f6007b9ee3800d8d7e05dcf092fa4fa5adade4fedfd8ba17d':raise ValueError('Previously qualified TRAIN source changed')
    rows=[json.loads(s) for s in paths['pool'].read_text().splitlines()];membership=json.loads(paths['membership'].read_text());members={m['record_id_PRIVATE']:m for m in membership}
    normalized=[];by_key={}
    for row in rows:
        m=members[row['id']];n={'id':row['id'],'patient':row['extra_info']['patient_key'],'level':m['level'],
            'label':row['reward_model']['ground_truth'],'image_sha256':m['PNG_sha256'],'question':row['extra_info']['question']}
        if sha(row['images'][0])!=n['image_sha256'] or row['extra_info']['internal_split']!='train':raise ValueError('TRAIN image binding changed')
        normalized.append(n);by_key[n['patient'],n['label']]=n
    main=json.loads(paths['schedule'].read_text())['schedule'];slots=[]
    for row in main:
        seq=row['ability']=='mri_sequence'
        slots.append({'task':'sequence' if seq else 'Dallas','record_id':by_key[row['extra_info']['patient_key'],row['reward_model']['ground_truth']]['id'] if seq else row['id']})
    routing=attach_pairs(slots,normalized)
    forced=json.loads(paths['forced'].read_text());forced=forced['schedule'] if isinstance(forced,dict) else forced
    compare=[]
    for cap in (8,32):
        compare.append({'group_cap':32,'accepted_update_ceiling':cap,'token_reservation':70784,
            'full_schedule':describe(forced,70784),'earliest_possible_update_stop_prefix':describe(forced[:cap],70784),
            'interpretation':'32 is a noncensoring ceiling, not a target. Safety/time/token gates may still stop early; accepted-gradient coverage unknown.' if cap==32 else '8 accepted updates can stop at slot8, censoring planned nine-pair exposure; rejected groups can delay that endpoint.'})
    report={'schema':'paired-interaction-signal-preparation-v1','status':'TESTED_CPU_NOT_NATIVE_INTEGRATED',
        'intervention':'Original28+48 main GSPO pool and original next uniform seed13 slots8:40 unchanged; same-person opposite z045 image is auxiliary-only.',
        'objective':'L_total=L_GSPO+0.1*softplus(-((s_F(I_F)-s_T(I_F))-(s_F(I_T)-s_T(I_T)))/2), s=mean canonical label-only logp; tau=1.',
        'not_original_vero':'Explicit supervised paired auxiliary. Never fabricated rollout/reward/advantage.',
        'source_sha256':{k:sha(v) for k,v in paths.items()},
        'full32_auxiliary':{'units':18,'unique_pairs':len({tuple(x['pair_ids']) for x in routing if x['pair_ids']}),'unique_images':len({i for x in routing if x['pair_ids'] for i in x['pair_ids']}),**operation_budget(18,14)},
        'preferred_future_micro':{'slot_count':4,'sequence':2,'Dallas':2,'accepted_joint_update_cap':2,'minimum_updates_required':0,
            'wall_operation_start_seconds':1200,'parent':'Preserved Adam60 for literal chain continuity, not best-final selection',
            'optimizer':'Same trainable structure permits full Adam60/RNG import in an explicitly new objective phase; native atomicity not yet integrated or verified.',
            'cost':operation_budget(2,2),'estimated_training_minutes':[10,20],
            'timing_basis':'Prior F32 two-group read-only qualification actual307.59s includes load/216 returnedtokens/0backward; auxiliary and training backward unmeasured. Planning envelope, not throughput forecast.',
            'selection':'First four original28-pool uniform-stage slots, no correct-rollout filtering, no resampling.'},
        'forced_pair32_sampling_only_alternatives':compare,
        'fixtures':{'all_wrong_equal_rollout_rewards_GSPO_gradient':0,'pair_scores':[[-.1,-3.],[-.2,-2.]],
            'pair_loss':pair_loss([[-.1,-3.],[-.2,-2.]]),'score_derivatives':score_derivatives([[-.1,-3.],[-.2,-2.]])},
        'failure_boundaries':['Image-insensitive parameter Jacobians cancel despite nonzero score-space derivatives.',
            'Common additive class priors cancel, but image-specific shortcuts do not.',
            'Larger separation need not improve absolute T1 margin/recall; explicit CPU counterexample passes.',
            'Shared parameters can harm Dallas. Extra paired exposure and auxiliary math are inseparable without exposure-matched control.',
            'Prior two-update independent CE from B28 was negative; this interaction objective is not validated by that experiment.'],
        'matched_evaluation':{'same_F32_parent_child':True,'primary':'Strict greedy sequence balanced accuracy and both class recalls',
            'secondary':['mean/sum/first candidate balanced accuracy and correct-minus-wrong margins','same-person image-swap direction','Dallas same/different recall','macro/worst task','invalids','charged tokens and wall cost'],
            'decision':'No scale-up unless T1 and candidate improve without FLAIR or Dallas degradation; one local comparison is not a causal proof.',
            'data':'New independent16-person source plan/payload qualification separate; old40 engineering only; no opened final100 output access.',
            'evaluation_budget':'Existing matched F32 parent+child80greedy+240candidate API calls, <=20480 generated and102000 TF positions; <=2400s operation-start; actual throughput unmeasured.'},
        'training_readiness':'NO','blockers':['Common-prefix TRAIN class scores/image dependence not measured; new read-only scope requires explicit authorization.',
            'F32 on-policy active-group backward remains PARTIAL.','Auxiliary native serial-gradient integration/state audit not verified.',
            'New holdout model-blind qualification and prospective acceptance lock required.'],
        'training_runs':0,'optimizer_updates':0,'training_authorized':False}
    out=Path(a.output);write(out,{'routing_PRIVATE':routing,'main_original_schedule_PRIVATE':main,'normalized_pair_records_PRIVATE':normalized,'public_report':report})
    write(root/'results/PAIRED_INTERACTION_SIGNAL_CPU_PREPARATION_V1.json',report)
    print(json.dumps({'pairs':report['full32_auxiliary']['unique_pairs'],'micro_tokens':13560,'updates':0}))
if __name__=='__main__':main()
