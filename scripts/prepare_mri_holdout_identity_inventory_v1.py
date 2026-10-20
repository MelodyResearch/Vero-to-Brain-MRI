"""Collect an explicit historical identity exclusion catalog, without predictions.

No final inference outputs, final native evaluation manifest, model arrays or MRI
pixels are opened. Final recipe registries contribute identity metadata ONLY.
This inventory does not establish an independent qualified holdout.
"""
import argparse,csv,hashlib,json,re
from collections import Counter,defaultdict
from pathlib import Path
from mri_rl.mri_holdout_identity_inventory_v1 import identity_index,metadata_identities,source_availability,final_identity_projection

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--primary-root',type=Path,required=True);a.add_argument('--output',type=Path,required=True);args=a.parse_args()
    R=args.primary_root.resolve();out=args.output.resolve();meta=R.parent.parent/'work/ds007045_metadata'
    if not out.is_relative_to(ROOT/'local_data') or out.exists():raise ValueError('Fresh private destination required')
    participants=list(csv.DictReader((meta/'participants.tsv').open(),delimiter='\t'))
    subjects={x['participant_id'] for x in participants};index=identity_index(subjects)
    legacy=R/'local_data/ds007045_bounded/visual_prompts/private_manifest.json'
    old=json.loads(legacy.read_text())
    if not all(index[c['anonymous_token']]==c['private_subject'] for c in old['private_cases']):raise ValueError('Historical anonymous crosswalk mismatch')
    catalog=[('original40_source_partition',legacy),
        ('original_all_task_TRAIN',R/'local_data/ds007045_bounded/vero_training_v1/train.verl.jsonl'),
        ('original_internal_dev',R/'local_data/ds007045_bounded/vero_training_v1/internal_dev.verl.jsonl'),
        ('original_two_task_TRAIN',R/'local_data/c_two_task_pool_v1/train.verl.jsonl'),
        ('final_sequence_IDENTITY_REGISTRY_ONLY',R/'local_data/prospective_sequence_test14_v1/recipe_registry_PRIVATE.json'),
        ('final_Dallas_IDENTITY_REGISTRY_ONLY',R/'local_data/dallas_correspondence_prospective40_v1/task_v1/sealed_test_registry_PRIVATE.json'),
        ('reviewed_z045_pair_membership',ROOT/'local_data/paired_z045_data_v1/candidate_v2/source_and_pair_membership_PRIVATE.json')]
    # Prior representation/diagnostic input plans, not output or score files.
    v2=R/'.worktrees/vero-mri-v2-controls/local_data'
    for pattern in ['qwen_mean48_v*/plan_PRIVATE.json','v2_cpu_representation_v1/frozen_inputs_PRIVATE.json',
                    'pilot8_cpu_controls_v1/frozen_inputs_PRIVATE.json','c40_early_visual_v1/plan*.json',
                    'paired_discrimination_pilot_v2/plan*.json']:
        catalog.extend(('prior_input_plan_'+str(p.relative_to(v2)),p) for p in sorted(v2.glob(pattern)))
    q=R/'.worktrees/vero-mri-native-path-qualification/local_data'
    catalog.extend([('reviewed168_TRAIN_manifest',q/'sequence_qualification_v1/audit_v1/future_loader_manifest_reviewed_PRIVATE.jsonl'),
        ('Dallas_spatial_TRAIN_candidates',q/'dallas_spatial_qualification_v1/candidate_v2/candidate_manifest_PRIVATE.jsonl')])
    provenance=defaultdict(list);details=[];missing=[]
    for name,path in catalog:
        if not path.is_file():missing.append(name);continue
        if path.stat().st_size>8*1024**2:raise ValueError('Unexpected large metadata input')
        x=[json.loads(t) for t in path.read_text().splitlines() if t] if path.suffix=='.jsonl' else json.loads(path.read_text())
        final_people=None
        if name=='final_sequence_IDENTITY_REGISTRY_ONLY':
            final_people=final_identity_projection(x,index);x=None
        elif name=='final_Dallas_IDENTITY_REGISTRY_ONLY':
            x=[{'patient_key':v['query_patient_key'],'candidate_patient_key':v['candidate_patient_key']} for v in x]
        used=final_people if final_people is not None else metadata_identities(x,index)
        for subject in used:provenance[subject].append(name)
        details.append({'logical_name':name,'path_PRIVATE':str(path),'sha256':sha(path),'sequence_source_identities_found':len(used)})
    # Conservatively exclude all historical raw-payload subject directories too,
    # including reserved/not-scored subjects; no payload contents are inspected.
    raw_dirs=[p.name for p in (R/'local_data/ds007045_bounded').glob('sub-*') if p.is_dir() and p.name in subjects]
    for subject in raw_dirs:provenance[subject].append('historical_local_payload_directory_conservative_exclusion')
    excluded=set(provenance);remaining=subjects-excluded
    available=[];sitecounts=defaultdict(Counter)
    for subject in sorted(subjects):
        match=re.fullmatch(r'sub-([A-Za-z]+)\d+',subject)
        if match is None:raise ValueError('Unknown source site format')
        site=match[1];sitecounts[site]['all_metadata_people']+=1
        if subject in excluded:sitecounts[site]['excluded_people']+=1;continue
        item=source_availability(meta,subject);item.update({'subject_PRIVATE':subject,'anonymous_token_PRIVATE':hashlib.sha256(subject.encode()).hexdigest()[:12],'site':site})
        available.append(item);sitecounts[site]['remaining_metadata_people']+=1
        sitecounts[site]['remaining_both_sidecars']+=int(item['both_sidecars'])
        sitecounts[site]['remaining_both_local_gzip_headers']+=int(item['both_gzip_payloads'])
    out.mkdir(parents=True)
    save(out/'historical_identity_exclusion_PRIVATE.json',{'source_namespace':'OpenNeuro_ds007045','identity_rule':'sha256(BIDSsubject)[:12], verified against historical40 crosswalk',
        'excluded':[{ 'subject_PRIVATE':s,'anonymous_token_PRIVATE':hashlib.sha256(s.encode()).hexdigest()[:12],'reasons':provenance[s]} for s in sorted(excluded)],
        'metadata_sources':details,'missing_catalog_inputs':missing,'cross_source_global_identity':'NOT_VERIFIED'})
    save(out/'remaining_source_availability_PRIVATE.json',available)
    desired=all(sitecounts[s]['remaining_both_sidecars']>=2 for s in sitecounts)
    proof={'schema':'MRI-new-holdout-identity-and-source-inventory-v1','status':'METADATA_INVENTORY_COMPLETE_HOLDOUT_NOT_ESTABLISHED',
        'source_namespace':'OpenNeuro_ds007045','metadata_people':len(subjects),'excluded_unique_people':len(excluded),'remaining_metadata_people':len(remaining),
        'source_files_examined':len(details),'missing_catalog_inputs':missing,'sites':{k:dict(v) for k,v in sorted(sitecounts.items())},
        'prospective16_two_per_each8site_metadata_feasible':len(sitecounts)==8 and desired,
        'new_patients_selected':0,'MRI_pixels_decoded':0,'source_payloads_downloaded':0,
        'old_final100_outputs_or_accuracy_accessed':False,'final_registry_use':'Patient identity projection for exclusion only; no outcomes/task-label selection.',
        'historical_exclusion_scope':'Explicit original partition/TRAIN/dev/used final identity registries/feature-probe input catalogs/review candidates plus conservative local raw directories; prospective cross-source lineage and unseen external exposures not certified.',
        'new_holdout_qualified':False,'global_cross_source_identity':'NOT_VERIFIED',
        'remaining_requirements':['Resolve missing catalogs if any and certify closed historical exposure union, including external/manual exposures.',
            'Freeze feasible site availability rule before selecting any cohort or inspecting outputs.',
            'Retrieve real paired payloads and qualify acquisition metadata/geometry/rendering/labels; pointers or gzip headers alone are not qualification.',
            'Deduplicate source lineage and predeclare independent evaluation before model selection.'],
        'participants_sha256':sha(meta/'participants.tsv'),'source_readme_sha256':sha(meta/'README.txt'),
        'helper_sha256':sha(ROOT/'src/mri_rl/mri_holdout_identity_inventory_v1.py'),'builder_sha256':sha(__file__),
        'private_artifact_sha256':{p.name:sha(p) for p in out.glob('*.json')},'new_model_loads':0,'optimizer_updates':0}
    save(ROOT/'results/MRI_NEW_HOLDOUT_IDENTITY_AND_SOURCE_INVENTORY_V1.json',proof)
    print(json.dumps({k:proof[k] for k in ['metadata_people','excluded_unique_people','remaining_metadata_people','source_files_examined','missing_catalog_inputs','sites','prospective16_two_per_each8site_metadata_feasible','new_holdout_qualified']}))


if __name__=='__main__':main()
