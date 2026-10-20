#!/usr/bin/env python3
"""Prepare fixed broader TRAIN checks and dormant <=2-step micro; no execute mode."""
import argparse
from collections import defaultdict,Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mri_rl import paired_classification_mix_v3 as mix


def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous',type=Path,required=True)
    parser.add_argument('--paired-source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--public-output',type=Path,required=True)
    args=parser.parse_args();previous=args.previous;source=args.paired_source
    feasibility=read(ROOT/'results/MIXTURE_ADAM60_FEASIBILITY_V3_CPU_V1.json')
    assert sha(ROOT/'results/MIXTURE_ADAM60_FEASIBILITY_V3_CPU_V1.json')==mix.FEASIBILITY_SHA256
    assert feasibility['candidate']['w']==mix.WEIGHT
    scope_path=previous/'local_data/paired_class_margin_probe_v2/scope_v4_PRIVATE.json'
    assert sha(scope_path)=='0dc854bcd5834ecb8f27fb38a867304084f7c2772f2be00a5c1ce08e55670d1f'
    scope=read(scope_path)
    prior_path=previous/'local_data/class_margin_aux_v2/preparation_v3_PRIVATE.json'
    assert sha(prior_path)=='51cd57b04444b4c70d9f6533f4823a72cc19ebd61e478bd81b5b5f9e33aea139'
    prior=read(prior_path)
    membership=read(source/'source_and_pair_membership_PRIVATE.json')
    rows=[json.loads(line) for line in (source/'paired_z045_sequence56_PRIVATE.verl.jsonl').read_text().splitlines() if line.strip()]
    by_id={r['id']:r for r in rows};members=defaultdict(list)
    for m in membership:members[m['patient_PRIVATE']].append(m)
    excluded={c['row_PRIVATE']['extra_info']['patient_key'] for c in scope['contexts_PRIVATE'] if c['row_PRIVATE']['ability']=='mri_sequence'}
    anchor_sites={m['site_PRIVATE'] for p in excluded for m in members[p]}
    by_site=defaultdict(list)
    for p,group in members.items():
        if p in excluded:continue
        assert len(group)==2 and {m['modality_PRIVATE'] for m in group}=={'flair','t1ce'} and {m['level'] for m in group}=={45}
        sites={m['site_PRIVATE'] for m in group};assert len(sites)==1
        site=next(iter(sites))
        if site not in anchor_sites:by_site[site].append(p)
    sites=sorted(by_site)[:3];assert len(sites)==3
    selected=[]
    for site in sites:
        patient=sorted(by_site[site])[0]
        group=sorted(members[patient],key=lambda m:('flair','t1ce').index(m['modality_PRIVATE']))
        pair=[]
        for member in group:
            row=by_id[member['record_id_PRIVATE']]
            assert row['extra_info']['internal_split']=='train' and member['pixel_check_PRIVATE']['pixel_exact']
            assert sha(row['images'][0])==member['PNG_sha256']==row['rc_data_v1']['image_sha256']
            pair.append(row)
        selected.append({'site_PRIVATE':site,'patient_PRIVATE':patient,'pair_PRIVATE':pair,'original_QC_flags_retained_PRIVATE':[r['sequence_qualification_v1'] for r in pair]})
    # Old original task schedule only; deduplicate by source identity, not outputs.
    original_plan_path=previous.parent/'vero-mri-paired-signal/local_data/paired_interaction_signal_v1/preparation_PRIVATE.json'
    all_rows=read(original_plan_path)['main_original_schedule_PRIVATE']
    old_dallas={c['row_PRIVATE']['extra_info']['patient_key'] for c in scope['contexts_PRIVATE'] if c['row_PRIVATE']['ability']=='mri_visit_correspondence'}
    dallas=[];seen=set()
    for label in ('same','different'):
        eligible=[r for r in all_rows if r['ability']=='mri_visit_correspondence' and r['reward_model']['ground_truth']==label
                  and r['extra_info']['patient_key'] not in old_dallas and r['extra_info']['patient_key'] not in seen]
        row=min(eligible,key=lambda r:r['source_record_id']);assert row['extra_info']['internal_split']=='train'
        dallas.append(row);seen.add(row['extra_info']['patient_key'])
    assert len(selected)==3 and len(dallas)==2
    binding={str(prior_path):sha(prior_path),str(scope_path):sha(scope_path),str(original_plan_path):sha(original_plan_path),str(source/'source_and_pair_membership_PRIVATE.json'):sha(source/'source_and_pair_membership_PRIVATE.json'),
             str(source/'paired_z045_sequence56_PRIVATE.verl.jsonl'):sha(source/'paired_z045_sequence56_PRIVATE.verl.jsonl')}
    preparation={'schema':'mixture-v3-broader-TRAIN-and-micro-preparation','status':'CPU_PREPARED_NOT_NATIVE_AUTHORIZED',
        'weight':mix.WEIGHT,'source_parent_commit':'a6ebe8d12e0dc7a8e1633acb64e5730cc5c6897b',
        'main_schedule_PRIVATE':prior['schedule_PRIVATE'],'companions_PRIVATE':prior['companions_PRIVATE'],
        'original_configuration':prior['configuration'],'parent':prior['parent'],
        'guard_reference_contexts_PRIVATE':{c['name']:c for c in scope['contexts_PRIVATE'][:3]},'guard_token_plans_PRIVATE':scope['token_plans_PRIVATE'],
        'new_CHECKS_TRAIN_ONLY_PRIVATE':selected,'Dallas_ENGINEERING_ONLY_checks_PRIVATE':dallas,
        'broader_check_rule':'Excluding previous gradient patient/site, lexicographic first3sites and firstpatient/site; retain QC flags/no outcome-dependent exclusions. Dallas firstsource eachclass with distinct storedpatientkeys; cross-case person independence NOT_VERIFIED.',
        'next_readonly_call_bound_PROPOSED':{'sequence_pairs':3,'sequence_images':6,'Dallas_images':2,'full_TF_forwards':16,'margin_backward_calls':8,
            'expected_TF_positions':4382,'hard_TF_positions':6800,'generation_calls':0,'optimizer_updates':0,'maximum_operation_start_seconds':1200,
            'scope_status':'CALL_BOUND_AND_INPUTS_ONLY_NOT_EXACT_NATIVE_LAUNCH_SCOPE','fresh_human_permission_required':True},
        'future_micro_hard_caps':prior['budget'],
        'micro_cost_note':'Additional current-model guard VJPs debit SAME40TF/20back/19860positions counters. Do not enlarge cap to complete checks; stop/rollback if exhausted. Both initial slots active upper bound28TF/14back/12500TFpositions. 2TOTALsteps can stop before4slots.',
        'loss':'unchanged_GSPO + .1*((1-.457130)*L_v1 + .457130*L_v2), explicit supervised auxiliary',
        'pre_step_guard':'Sequence slots protect actual current auxiliary F/T pair; Dallas slots protect fixed reference F/T pair; Dallas target is fixedcase always. Persist exact per-target image/question/row identities. Recompute current-model VJPs and current-Adam JOINT direction, never stale Adam60 targets. All3TOTALdirections>=1e-6; NEWdelta/auxgradient endpointenvelope; reject beforestep without altering gradients/advantages. This is case-scoped protection, not classwide/oldanchor protection on sequence slots.',
        'missing_before_training':['Fresh independent exact source/input/scope review and human permission for broader TRAIN read-only VJPs.',
            'Broader TRAIN check real results and exact independent review; no retuningw after per-patient failures.',
            'Actual current combinedGSPO+aux Adam direction checked dynamically before each proposed step; native joint path still not measured.',
            'Exact future micro launch scope, explicit human training authorization accepting native limitations, sealed evaluation protocol before model use.'],
        'bindings_PRIVATE':binding,'model_loads':0,'new_native_forwards':0,'new_native_backwards':0,'optimizer_updates':0,'holdout_or_final_reads':0,'training_authorized':False}
    public={k:preparation[k] for k in ('schema','status','weight','broader_check_rule','next_readonly_call_bound_PROPOSED','future_micro_hard_caps','micro_cost_note','loss','pre_step_guard','missing_before_training','model_loads','new_native_forwards','new_native_backwards','optimizer_updates','holdout_or_final_reads','training_authorized')}
    public['selected_TRAIN_counts']={'sequence_patients':3,'sequence_sites':3,'sequence_images':6,'sequence_pairs':3,'Dallas_cases':2,'Dallas_classes':['same','different']}
    public['historical_pool_site_patient_counts']={s:len({m['patient_PRIVATE'] for m in membership if m['site_PRIVATE']==s}) for s in sorted({m['site_PRIVATE'] for m in membership})}
    public['old_main_pool_and_next4schedule_unchanged']=True
    for path,value in [(args.output,preparation),(args.public_output,public)]:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')
    assert not any(k.split('.')[0] in ('mlx','mlx_vlm','torch','transformers') for k in sys.modules)
    print(json.dumps({'status':public['status'],'counts':public['selected_TRAIN_counts'],'private_sha256':sha(args.output),'public_sha256':sha(args.public_output),'model_loads':0}))


if __name__=='__main__':main()
