#!/usr/bin/env python3
"""Build a source-bound second MRI task review packet from saved TRAIN evidence.

No image decoding/rendering, model imports, fitting, inference or updates.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import sys

W=Path(__file__).resolve().parents[1];sys.path.insert(0,str(W/'src'))
from mri_rl.second_task_eligibility_v1 import RULES,require,plane_triage,metadata_ceiling,matched_label_controls

PINS={
 'MRI_TRAIN':'0efb588b328e3d9af2a339d9237836357a5baff5d9db8b376dc716de3d1f10e3',
 'Dallas_TRAIN':'6adf06674050e395b704e4b76103f067166b905016920195603e271690de4f14',
 'qualification':'219cd2bbe4fc336085378da5d0a22ff7a9092e57774bd497ac2ee8ddc5934944',
 'plane_geometry':'889620f2b147523273dbb3ccbc84b3e7a9b676664ff16c33ac1ff9b3e3512dc1',
 'MRI_only_notes':'6edb737d44b64f50e7a7286ae7c71cf8a27478a2b7288c564e3ad83d28f5fb37',
 'crosscheck':'6bb02193ae5a15ad3d13b6b5b68d326f3517cbed939ecddfe3a3cf8eeebb0c39',
 'TRAIN_source_metadata':'eca155b538b5f83fdcc6b6242f113aebdbfac9210511b68f4a6c01bc53571390',
 'source_README':'66efa8e9ffd194b0aca9e52e9a62201db1b447f9b1f38a11ec8daec265293d96',
 'source_description':'748fb52a1d943c6f50bb6e4795082e973c4a8694e01a287383a9bf1f20fff763',
 'original_renderer':'4d7a353c4b6da53bd9781a0073544fb610cc8012512e31fdf68024f9cc31588e',
}
TASKS={'mri_tumor_presence':84,'mri_enhancing_presence':84,'mri_tumor_grid':46}


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read_bound(path,pin):
    raw=path.read_bytes();require(hashlib.sha256(raw).hexdigest()==pin,'Frozen TRAIN evidence SHA mismatch: '+path.name)
    return raw.decode('utf-8')


def jsonlines(text):return [json.loads(line) for line in text.splitlines() if line.strip()]


def save(path,obj):
    with path.open('x') as f:json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')


def save_rows(path,rows):
    with path.open('x') as f:
        for row in rows:f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')


def unique(rows,field):
    result={r[field]:r for r in rows};require(len(result)==len(rows),'Duplicate evidence key: '+field);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--primary-root',type=Path,required=True)
    p.add_argument('--prior-worktree',type=Path,required=True);p.add_argument('--destination',type=Path,default=W/'local_data/second_task_eligibility_v1/candidate_v1');a=p.parse_args()
    require(all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS')),'One-thread CPU caps required')
    R=a.primary_root.resolve();prior=a.prior_worktree.resolve();out=a.destination.resolve();public=W/'results/SECOND_MRI_TASK_REVIEW_QUEUE_V1.json'
    require(out.is_relative_to(W/'local_data/second_task_eligibility_v1') and not out.exists() and not public.exists(),'New scoped output required')
    B=prior/'local_data/dallas_spatial_qualification_v1';work=R.parent.parent/'work'
    paths={'MRI_TRAIN':R/'local_data/ds007045_bounded/vero_training_v1/train.verl.jsonl',
        'Dallas_TRAIN':R/'local_data/dallas_correspondence_prospective40_v1/task_v1/train.verl.jsonl',
        'qualification':B/'candidate_v2/candidate_manifest_PRIVATE.jsonl','plane_geometry':B/'candidate_v1/tumor_truth_separate_PRIVATE.json',
        'MRI_only_notes':B/'candidate_v1/MRI_only_nonexpert_observations_BEFORE_OVERLAY_PRIVATE.json',
        'crosscheck':B/'candidate_v2/post_overlay_crosscheck_PRIVATE.json','TRAIN_source_metadata':prior/'local_data/sequence_qualification_v1/audit_v1/audit_PRIVATE.json',
        'source_README':work/'ds007045_metadata/README.txt','source_description':work/'ds007045_metadata/dataset_description.json',
        'original_renderer':work/'ds007045_prompts.py'}
    text={k:read_bound(path,PINS[k]) for k,path in paths.items()}
    # Freeze executable rules before deriving this build's subset counts.
    out.mkdir(parents=True);rules_path=out/'PREDECLARED_RULES_SAFE.json';save(rules_path,RULES)
    mri=jsonlines(text['MRI_TRAIN']);dallas=jsonlines(text['Dallas_TRAIN']);qualification=jsonlines(text['qualification'])
    require(all(r['extra_info']['internal_split']=='train' for r in mri+dallas),'Only TRAIN registry rows accepted')
    source_rows=[r for r in mri if r['ability'] in TASKS]
    require(Counter(r['ability'] for r in source_rows)==TASKS and len(dallas)==48,'Frozen task coverage differs')
    originals=unique(source_rows,'id');qrows=unique(qualification,'record_id_PRIVATE')
    require(set(qrows)==set(originals)|{r['id'] for r in dallas},'Qualification exact TRAIN-ID coverage differs')
    for r in source_rows+dallas:
        q=qrows[r['id']];require(q['task']==r['ability'] and q['split']=='train' and q['training_authorized'] is False and q['expert_eligibility']=='NOT_VERIFIED','Qualification task/split/admission changed')
    keys=json.loads(text['plane_geometry']);notes=json.loads(text['MRI_only_notes'])['tumor'];cross=json.loads(text['crosscheck'])['nonexpert_mri_mask_crosscheck']
    keymap=unique(keys,'review_id');notemap=unique(notes,'review_id');crossmap=unique(cross,'review_id')
    require(len(keys)==84 and set(keymap)==set(notemap)==set(crossmap),'Exact84 TRAIN plane notes required')
    require(set(rid for k in keys for rid in k['records_PRIVATE'])==set(originals),'Plane-to-record coverage incomplete')
    require(sum(len(k['records_PRIVATE']) for k in keys)==214,'Plane-to-record duplicate')
    raw_metadata=json.loads(text['TRAIN_source_metadata'])['row_metadata_PRIVATE'];metadata=defaultdict(set)
    for value in raw_metadata.values():metadata[value['patient'],value['level']].add((value['site'],value['scanner']))
    require(len(metadata)==84 and all(len(s)==1 for s in metadata.values()),'Source site/scanner/level metadata ambiguous')
    candidate_rows=[];expert_queue=[];source_keys=[];spatial=[];contingencies=defaultdict(list);planes=[]
    sheet_hashes={}
    for review_id,key in sorted(keymap.items()):
        note=notemap[review_id];triage=plane_triage(key,note);plane_rows=[originals[rid] for rid in key['records_PRIVATE']]
        require(crossmap[review_id]['mask_geometry_positive']==triage['mask_positive'] and crossmap[review_id]['MRI_only_apparent_abnormality']==triage['MRI_only_nonexpert_apparent'],'Saved MRI-mask crosscheck differs')
        position=int(review_id[1:])-1;page=position//12+1;slot=position%12
        sheet=B/'candidate_v1/sheets_PRIVATE'/f'tumor_MRI_ONLY_{page:02d}.png'
        if str(sheet) not in sheet_hashes:sheet_hashes[str(sheet)]=sha(sheet)
        form={'review_id':review_id,'MRI_only_sheet_PRIVATE':str(sheet),'sheet_sha256':sheet_hashes[str(sheet)],
            'position_zero_based':{'row':slot//4,'column':slot%4},
            'first_pass_instruction':'Inspect MRI-only sheet before source/overlay key. Do not infer postoperative status or tumor absence from appearance or a zero mask.',
            'expert_fields_to_complete':{'annotator_identity':None,'credentials':None,'sufficient_frame_coverage':None,
                'visible_abnormal_region_present_absent_indeterminate':None,'visible_region_annotation':None,
                'enhancement_visible_absent_indeterminate':None,'single_center_meaningful':None,'ambiguity_and_abstention_reasons':[],
                'etiology_cavity_necrosis_artifact_or_postoperative_if_supportable':None},
            'training_authorized':False,'expert_adjudication':'NOT_COMPLETED'}
        expert_queue.append(form);planes.append({'review_id':review_id,**triage})
        source_keys.append({'review_id':review_id,'saved_source_key_PRIVATE':key,'saved_nonexpert_note_PRIVATE':note,'triage':triage,
            'second_pass_instruction':'After independent MRI-only judgment, inspect prior source mask overlay and resolve question/source-endpoint alignment. Original references must not be relabeled automatically.'})
        for original in plane_rows:
            row=copy.deepcopy(original);task=row['ability'];label=row['reward_model']['ground_truth'];patient=row['extra_info']['patient_key']
            m=re.fullmatch(re.escape(patient)+r'_z(045|091|136)_(tumor_presence|enhancing_presence|tumor_grid)',row['id']);require(m is not None,'Source patient/level/task linkage differs')
            level=int(m[1]);site,scanner=next(iter(metadata[patient,level]));g=key['enhancing_geometry' if task=='mri_enhancing_presence' else 'tumor_geometry']
            expected=key['tumor_geometry']['global_grid'] if task=='mri_tumor_grid' else ('yes' if g['area']>0 else 'no')
            require(label==key['truth_PRIVATE'][task]==expected,'Existing hidden mask target changed')
            reasons=list(triage['reasons'])
            if task=='mri_enhancing_presence':reasons.append('SOURCE_NUMERIC_LABEL3_ENHANCING_MAPPING_NOT_EXPLICIT_IN_LOCAL_SOURCE_DOCS')
            if task!='mri_tumor_grid' and label=='no':reasons.append('MASK_NEGATIVE_IS_NOT_EXPERT_VISIBLE_NEGATIVE')
            row['training_authorized']=False
            row['second_task_eligibility_v1']={'version':1,'review_id_PRIVATE':review_id,'training_authorized':False,'scientific_admission':'HOLD',
                'primary_review_queue':triage['primary_queue'],'expert_adjudication':'NOT_COMPLETED','blocking_reasons':reasons,
                'spatial_review_candidate':task=='mri_tumor_grid' and triage['spatial_review_candidate'],
                'original_source_target_unchanged':True,'source_registry_sha256':PINS['MRI_TRAIN'],'rules_sha256':sha(rules_path)}
            candidate_rows.append(row)
            entry={'id':row['id'],'patient':patient,'site':site,'scanner':scanner,'level':level,'label':label}
            contingencies[task].append(entry)
            if task=='mri_tumor_grid' and triage['spatial_review_candidate']:spatial.append(entry)
    require(len(candidate_rows)==214 and len({r['id'] for r in candidate_rows})==214,'Candidate coverage differs')
    require(all(r['reward_model']==originals[r['id']]['reward_model'] and r['prompt']==originals[r['id']]['prompt'] for r in candidate_rows),'Source target/prompt mutated')
    plane_counts=dict(Counter(r['primary_queue'] for r in planes));row_counts=dict(Counter(r['second_task_eligibility_v1']['primary_review_queue'] for r in candidate_rows))
    for row in dallas:
        row['training_authorized']=False;row['second_task_eligibility_v1']={'version':1,'training_authorized':False,'scientific_admission':'HOLD',
            'route_status':'DALLAS_WHOLEHEAD_IDENTITY_ENGINEERING_ONLY','blocking_reasons':['REAL_BRAIN_ONLY_INPUT_AND_IDENTITY_ANSWERABILITY_GOLD_MISSING'],
            'original_source_target_unchanged':True,'source_registry_sha256':PINS['Dallas_TRAIN']}
    candidate_path=out/'second_task_candidates_PRIVATE.verl.jsonl';save_rows(candidate_path,candidate_rows)
    spatial_ids={r['id'] for r in spatial};spatial_path=out/'spatial_priority_review_candidates_PRIVATE.verl.jsonl'
    save_rows(spatial_path,[r for r in candidate_rows if r['id'] in spatial_ids])
    save_rows(out/'Dallas_engineering_hold_PRIVATE.verl.jsonl',dallas)
    save_rows(out/'expert_MRI_only_adjudication_queue_PRIVATE.jsonl',expert_queue)
    save(out/'expert_second_pass_source_key_PRIVATE.json',source_keys)
    controls={'all_original_grid_rows':matched_label_controls(contingencies['mri_tumor_grid']),
        'strict_spatial_review_candidates':matched_label_controls(spatial)}
    save(out/'site_level_matched_label_controls_PRIVATE.json',controls)
    per_task={}
    for task,rows in contingencies.items():
        per_task[task]={'rows':len(rows),'labels':dict(Counter(r['label'] for r in rows)),
            'primary_queue_rows':dict(Counter(r['second_task_eligibility_v1']['primary_review_queue'] for r in candidate_rows if r['ability']==task)),
            'blocking_reason_counts':dict(Counter(reason for r in candidate_rows if r['ability']==task for reason in r['second_task_eligibility_v1']['blocking_reasons'])),
            'metadata_contingencies':{'+'.join(fields):metadata_ceiling(rows,fields) for fields in [('site',),('level',),('site','level'),('scanner',),('patient',)]}}
    source_semantics={'dataset_release':'OpenNeuro ds007045 v2.0.1','source_metadata_commit':'283007b96977b5b3285bd22cd67b7d97114b1eed',
        'supported_by_pinned_README':['Adults with histologically confirmed primary glioblastoma; preoperative MRI before surgery.',
            'SegResNet-initialized tumor masks following BraTS convention, subsequently reviewed/corrected by neuroradiologists.',
            'Segmentation masks correspond spatially to preprocessed MRI; N4/MNI152/skull stripping are existing derivatives.'],
        'not_supported_by_these_source_docs':['Per-plane visible tumor/absence/enhancement gold at the exact frozen 2D rendering.',
            'An explicit numeric0/1/2/3 dictionary confirming local mask==3 enhancement semantics.',
            'Per-subject manual correction/QC completeness, independent wholly hand-traced masks, or postoperative diagnoses from cavity-like MRI.'],
        'local_renderer_endpoints':{'mri_tumor_presence':'any(mask>0)','mri_enhancing_presence':'any(mask==3), numeric semantic mapping NOT_VERIFIED locally',
            'mri_tumor_grid':'global centroid of np.rot90(mask>0),legacy doubled display coordinate,3x3 FLAIR-panel grid'},
        'scientific_consequence':'Programmatic source-mask arithmetic is reproducible. Image-only clinical visibility and precise endpoint interpretation still require independent expert adjudication.'}
    save(out/'source_annotation_semantics_SAFE.json',source_semantics)
    save(out/'binding_PRIVATE.json',{'input_paths_PRIVATE':{k:str(v) for k,v in paths.items()},'input_sha256':PINS,
        'prior_sheet_sha256_PRIVATE':sheet_hashes,'candidate_manifest_sha256':sha(candidate_path),'rules_sha256':sha(rules_path),
        'plane_triage_PRIVATE':planes,'spatial_control_rows_PRIVATE':spatial})
    report={'schema':'second-mri-task-review-queue-v1','status':'ACTUAL_CPU_VERSIONED_REVIEW_CANDIDATES_BUILT_SCIENTIFIC_ADMISSION_HOLD',
        'evidence_tier':'OBSERVED_SAVED_TRAIN_SOURCE_AND_NONEXPERT_EVIDENCE','research_tag':'C_TASK_ELIGIBILITY',
        'mission_alignment':{'link':'Resolve the second heterogeneous image-grounded task requirement for Vero-to-MRI transfer.',
            'claim_falsifier':'A deterministic existing-source geometry/nonexpert screen identifies a smaller review queue; exact source labels and counts must reproduce or the build fails.',
            'isolation':'Only new qualification metadata and expert packet; existing images, tasks, hidden labels, weights and reviews remain frozen.',
            'provenance':'Pinned TRAIN notes, geometry, registries, metadata, renderer and README hashes.',
            'shortcut':'Compute training site/level/scanner/patient contingencies and build actual same-site/same-level/different-person/different-label control pairs.',
            'resources_decision':'Single-thread text/JSON/hash work; zero image decoding/rendering/model work. No scientific admission or training authorization.'},
        'source_annotation_semantics':source_semantics,'rules_sha256':sha(rules_path),'input_sha256':PINS,
        'plane_partition':{'planes':84,'people':28,'primary_queue_counts':plane_counts},
        'candidate_rows':214,'candidate_task_counts':TASKS,'primary_queue_row_counts':row_counts,'per_task':per_task,
        'spatial_priority_review_subset':{'rows':len(spatial),'people':len({r['patient'] for r in spatial}),
            'labels':dict(Counter(r['label'] for r in spatial)),'sites':len({r['site'] for r in spatial}),
            'levels':dict(Counter(str(r['level']) for r in spatial)),
            'definition':'Single-component robust whole-mask-centroid geometry plus prior MRI-only apparent abnormality and no cavity/rim lexical concern; expert pending, not visual or clinical gold.',
            'training_authorized':False,'manifest_sha256':sha(spatial_path)},
        'actual_matched_shortcut_control_index':{name:{k:v for k,v in value.items() if not k.endswith('_PRIVATE')} for name,value in controls.items()},
        'expert_packet':{'MRI_only_first_pass_cases':84,'blank_expert_fields':True,'second_pass_reference_key_separate':True,'prior_sheets_reused':7,
            'independent_expert_records_created':0,'adjudication_completed':False},
        'Dallas':{'rows':48,'status':'ENGINEERING_ONLY','missing':'Validated brain-only input plus independently verified identity answerability; prior fixed ROI is not skull stripping.',
            'new_crops_or_models':0,'training_authorized':False},
        'manifest_sha256':sha(candidate_path),'scientifically_admitted_second_task_rows':0,'training_authorized':False,
        'read_scope':{'existing_TRAIN_JSON_and_source_docs_only':True,'image_files_decoded':0,'new_renders':0,'repeated_blind_judgments':0,
            'mask_or_MRI_NIfTI_files_opened':0,'prior_sheet_files_hashed_only':7,'dev_or_final100_files_read':0,
            'models_loaded':0,'optimizer_updates':0,'classifier_fits':0},
        'new_concrete_fix':'Replace undifferentiated214-row lesion eligibility with source-preserving deterministic plane/row review queues, a strict spatial priority subset and a ready-to-fill84-case expert adjudication packet.',
        'remaining_decisive_blocker':'Independent expert frame-level visibility and question/annotation endpoint adjudication; enhancement route additionally needs source numeric-label semantics confirmation.',
        'limitations':['A clean geometry screen is not evidence of a visible lesion or an independently valid second task.',
            'Mask components are not clinical multifocal tumors; cavity/rim lexical flags are review prompts, not postoperative diagnoses.',
            'Thresholds1000 native pixels/10 display pixels inherit earlier engineering screens and are not clinically calibrated.',
            'Metadata majority values are TRAIN resubstitution contingencies, not held-out predictive accuracy; control pairs are indices, not completed model experiments.'],
        'code_sha256':{'helper':sha(W/'src/mri_rl/second_task_eligibility_v1.py'),'builder':sha(__file__),
            'tests':sha(W/'tests/test_second_task_eligibility_v1.py')},'cross_model_review':'NOT_VERIFIED'}
    save(public,report)
    print(json.dumps({'planes':84,'candidate_rows':214,'plane_queues':plane_counts,'spatial_priority_rows':len(spatial),
        'matched_control_pairs':{k:v['pairs'] for k,v in controls.items()},'candidate_manifest_sha256':sha(candidate_path),'training_authorized':False},sort_keys=True))


if __name__=='__main__':main()
