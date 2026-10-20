"""CPU-only comparison of actual finite source exposure; never launches training."""
from pathlib import Path
import argparse,copy,hashlib,json
from mri_rl.paired_z045_data_v1 import SEQ,DALLAS,canonical_sha,require
from mri_rl.paired_z045_exposure_v2 import pair_schedule,describe

ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    out=a.output.resolve();require(out.is_relative_to(ROOT/'local_data') and not out.exists(),'Fresh private destination required')
    base=ROOT/'local_data/paired_z045_data_v1/candidate_v2'
    data=json.loads((ROOT/'results/PAIRED_Z045_DATA_V1.json').read_text())
    names=['paired_z045_sequence56_PRIVATE.verl.jsonl','source_and_pair_membership_PRIVATE.json',
           'candidate_common_stage_next32_PRIVATE.json','reference_common_stage_next32_PRIVATE.json']
    for name in names:require(sha(base/name)==data['artifact_sha256'][name],'Frozen input changed: '+name)
    pool=[json.loads(x) for x in (base/names[0]).read_text().splitlines()]
    members=json.loads((base/names[1]).read_text());sites={x['patient_PRIVATE']:x['site_PRIVATE'] for x in members}
    unforced=json.loads((base/names[2]).read_text())['schedule']
    control=json.loads((base/names[3]).read_text())['schedule']
    paired=pair_schedule(unforced,pool,sites)
    # Cost-only alternatives preserve the same14 Dallas occurrence objects;
    # additional sequence slots change task allocation and are NOT recommended.
    alternatives={}
    for count in (32,56):
        mask=copy.deepcopy(unforced)
        extra=count-18
        mask.extend([{'id':'placeholder-'+str(i),'ability':SEQ,'extra_info':{'internal_split':'train'}} for i in range(extra)])
        alternatives['forced_'+str(len(mask))]=pair_schedule(mask,pool,sites)
    plans={'historical28_pool_same32':control,'unforced56_pool_same32':unforced,
           'forced_pair_same32':paired,**alternatives}
    out.mkdir(parents=True)
    summary={}
    timing=json.loads((ROOT/'results/PAIRED_Z045_HISTORICAL_BUDGET_INDEPENDENT_V1.json').read_text())['historical_timing_seconds']
    for name,rows in plans.items():
        save(out/(name+'_PRIVATE.json'),{'schedule':rows,'sampling_intervention':name.startswith('forced'),
            'training_authorized':False,'same_Vero_row_sampling_claimed':not name.startswith('forced')})
        x=describe(rows,70784)
        x['same_task_Dallas_slots_as_original32']=len(rows)==32
        x['Dallas_occurrence_content_unchanged']=[r for r in rows if r['ability']==DALLAS]==[r for r in unforced if r['ability']==DALLAS]
        x['historical_speed_planning_seconds']={
            'mean_rate_plus_historical_load':timing['load_authorization_to_first_group']+len(rows)*timing['complete_intergroup_statistics']['mean'],
            'all_groups_at_historical_p90_plus_load':timing['load_authorization_to_first_group']+len(rows)*timing['complete_intergroup_statistics']['p90_linear'],
            'not_native_F32_training_measurement':True,'holds_excluded':True}
        summary[name]=x
    require(summary['forced_pair_same32']['exposure']['people_both_modalities_observed']==9,'Nine pair exposure required')
    prefixes={str(n):describe(paired[:n],70784)['exposure'] for n in (8,12,16,20,24,28,32)}
    result={'schema':'paired-z045-finite-exposure-comparison-v2','status':'CPU_PREPARED_NOT_TRAINED',
        'evidence_tier':'OBSERVED_CPU_SCHEDULE_ARITHMETIC_NOT_LEARNING','plans':summary,'forced32_prefix_exposure':prefixes,
        'recommendation':{'schedule':'forced_pair_same32','32_attempts_if_all_completed':True,
            'sampling_change':'Explicit site-round-robin deterministic patient ordering, alternating within-pair orientation, adjacent sequence occurrences paired; exact historical task/Dallas mask preserved.',
            'reward_LR_LoRA_loss_changed':False,'not_original_Vero_source_sampling':True,
            'existing8_update_early_stop_cannot_guarantee9_pairs':True,
            'coverage_complete_candidate_update_ceiling':32,
            'ceiling_rationale':'At most one update/group;32 ceiling prevents update-count censoring before32 planned inputs. It is a proposed ceiling, not an optimization target or permission to train.',
            'current32_group8_update_entry_not_modified_or_authorized_for_this_candidate':True,
            'token_cap':70784,'operation_start_seconds':3000,'atomic_finish_can_overrun':True,
            'incomplete_schedule_result':'INCONCLUSIVE_FOR_PLANNED9_PAIR_EXPOSURE; report actual pairs, never silently extend.',
            'training_authorized':False},
        'causal_limit':'Without matched old-pool continuation at identical actual budget, parent/child only measures local change; pairing causality not identified.',
        'qualification':'Real F32 backward qualification and fresh exact training permission still required; no new model execution in this command.',
        'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'src/mri_rl/paired_z045_exposure_v2.py',ROOT/'results/PAIRED_Z045_DATA_V1.json']},
        'private_plan_sha256':{p.name:sha(p) for p in out.glob('*.json')},
        'optimizer_updates_this_command':0,'new_model_loads':0,'final100_outcomes_read':False}
    save(ROOT/'results/PAIRED_Z045_FINITE_EXPOSURE_COMPARISON_V2.json',result)
    print(json.dumps({k:{'groups':v['attempted_group_slots'],'pairs':v['exposure']['people_both_modalities_observed'],
        'patients':v['exposure']['sequence_people_observed'],'images':v['unique_sequence_image_paths'],'reserved_tokens':v['charged_token_reservation']} for k,v in summary.items()}))


if __name__=='__main__':main()
