#!/usr/bin/env python3
"""One-thread, existing TRAIN PNG audit; never launch a model or fit a classifier."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

W=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(W/'src'))
from PIL import Image,ImageChops,ImageDraw
from mri_rl.root_cause_data_v1 import require,sha,validate_sequence_rows,validate_dallas_rows,patient_label_confounding,image_qc
from mri_rl.visit_correspondence import token,RENDER

PINS={
 'sequence_train':('local_data/ds007045_bounded/vero_training_v1/train.verl.jsonl','0efb588b328e3d9af2a339d9237836357a5baff5d9db8b376dc716de3d1f10e3'),
 'original_train':('local_data/c_two_task_pool_v1/train.verl.jsonl','455efc650444f2820f26832bc29b042ce2ec3babc0c9fe745491705a3fdf7e2c'),
 'sequence_inventory':('local_data/ds007045_bounded/c_recovery_v1/sequence_inventory_PRIVATE.json','260082c4b2ea27357bbb708caf1afdf50679a9142cf2bbfa1de3f7c46277d400'),
 'train_geometry':('local_data/ds007045_bounded/safe_resume_v1/historical112_sampling_v2_PRIVATE.json','8abe468dfe50de53c4da23d31ea00f613ae38df7996677eaff14cdf2f49226b6'),
 'dallas_train':('local_data/dallas_correspondence_prospective40_v1/task_v1/train.verl.jsonl','6adf06674050e395b704e4b76103f067166b905016920195603e271690de4f14'),
}


def save_new(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path,rows):
    with path.open('x') as stream:
        for row in rows:stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')


def verify_dallas_panels(rows,root):
    """Check only train panel/QC artifacts, and exact pair panel concatenation."""
    base=root/'local_data/dallas_correspondence_prospective40_v1/task_v1'
    panels={};qcs={};bytes_read=0
    expected_render=hashlib.sha256(json.dumps(RENDER,sort_keys=True).encode()).hexdigest()
    for row in rows:
        e=row['extra_info']; require(e['render_contract_sha256']==expected_render,'Dallas render contract changed')
        for key in ('query_source_sha256','candidate_source_sha256'):
            h=e[key]
            if h in panels:continue
            qc_path=base/'qc_PRIVATE'/(token('qc',h)+'.json')
            qc=json.loads(qc_path.read_text());require(qc['source_sha256']==h,'Source/panel QC linkage changed')
            panel=base/'assets'/(token('volume',h)+'.png')
            require(qc['panel_file']==panel.name and sha(panel)==qc['panel_sha256'],'Frozen source panel changed')
            require(qc['canonical_orientation']==['R','A','S'] and qc['finite_voxels'] is True,'Canonical orientation/finite QC missing')
            require(len(qc['tile_std'])==len(qc['tile_nonzero_fraction'])==9,'Nine-view QC missing')
            with Image.open(panel) as im:
                require(im.mode=='L' and im.size==(384,384),'Source panel geometry changed');im.load()
            panels[h]=panel;qcs[h]=qc;bytes_read+=panel.stat().st_size
        with Image.open(row['images'][0]) as im,Image.open(panels[e['query_source_sha256']]) as left,Image.open(panels[e['candidate_source_sha256']]) as right:
            require(ImageChops.difference(im.crop((0,0,384,384)),left).getbbox() is None,'Left panel source mismatch')
            require(ImageChops.difference(im.crop((384,0,768,384)),right).getbbox() is None,'Right panel source mismatch')
    require(len(panels)==48,'Dallas train source-volume count changed')
    return {'train_source_panels':len(panels),'pair_compositions_exact':len(rows),
        'render_contract':RENDER,'render_contract_sha256':expected_render,
        'technical_tile_warning_volumes':sum(any(v<1 for v in q['tile_std']) or any(v<.01 for v in q['tile_nonzero_fraction']) for q in qcs.values()),
        'qc_files_read':len(qcs),'panel_bytes_read':bytes_read,
        'raw_NIfTI_rerender':'NOT_RUN; source hash to saved panel QC verified, upstream NIfTI bytes not reopened',
        'scanner_model_and_site':'UNVERIFIED; manufacturer/protocol matching in original builder is not verified site control'}


def make_montages(rows,out):
    """Private, unlabeled contact sheets for technical visual inspection only."""
    files=[]
    for task,width,columns,chunk in [('mri_sequence',110,6,42),('mri_visit_correspondence',320,4,24)]:
        group=sorted((r for r in rows if r['ability']==task),key=lambda r:r['id'])
        for number,start in enumerate(range(0,len(group),chunk),1):
            selection=group[start:start+chunk];tile_height=150 if task=='mri_sequence' else 180
            canvas=Image.new('RGB',(columns*width,((len(selection)+columns-1)//columns)*tile_height),'#222222')
            draw=ImageDraw.Draw(canvas)
            for i,row in enumerate(selection):
                with Image.open(row['images'][0]) as im:
                    im=im.convert('RGB');im.thumbnail((width-4,tile_height-20));x=(i%columns)*width;y=(i//columns)*tile_height
                    canvas.paste(im,(x,y));draw.text((x+2,y+tile_height-18),str(start+i+1),fill='white')
            path=out/f'{task}_technical_contact_{number:02d}_PRIVATE.png';canvas.save(path);files.append(path.name)
    return files


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--primary-root',type=Path,required=True)
    p.add_argument('--destination',type=Path,required=True);p.add_argument('--public-output',type=Path,required=True);a=p.parse_args()
    root=a.primary_root.resolve();out=a.destination.resolve();public=a.public_output.resolve()
    require(all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS')),'Explicit one-thread caps required')
    require(out.is_relative_to(W/'local_data/root_cause_cpu_v1/data') and not out.exists(),'New scoped private output directory required')
    require(public==W/'results/ROOT_CAUSE_DATA_TRAIN_ONLY_V1.json' and not public.exists(),'New named public aggregate required')
    paths={}
    for key,(rel,pin) in PINS.items():
        path=root/rel;require(sha(path)==pin,'Frozen train-only source changed: '+key);paths[key]=path
    source=jsonl(paths['sequence_train']);require(all(r['extra_info']['internal_split']=='train' for r in source),'Source registry contains nontrain rows')
    seq=[r for r in source if r['ability']=='mri_sequence'];original=jsonl(paths['original_train'])
    require(all(r['extra_info']['internal_split']=='train' for r in original),'Original pool contains nontrain rows')
    old_seq=[r for r in original if r['ability']=='mri_sequence'];dallas=[r for r in original if r['ability']=='mri_visit_correspondence']
    require(len(old_seq)==28 and len(dallas)==48 and len(original)==76,'Original fixed train pool changed')
    require({r['id'] for r in old_seq}<={r['id'] for r in seq},'Original sequence rows absent from candidate')
    require(dallas==jsonl(paths['dallas_train']),'Mixed pool Dallas rows differ from source train registry')
    require({r['extra_info']['patient_key'] for r in old_seq}=={r['extra_info']['patient_key'] for r in seq},'Expansion changes patient cohort')
    refs=json.loads(paths['sequence_inventory'].read_text())['rows_PRIVATE']
    require(all(Path(r['image_PRIVATE']).resolve().is_relative_to(root/'local_data/ds007045_bounded/visual_prompts') for r in refs),'Unexpected source-image path')
    require(all(Path(r['images'][0]).resolve().is_relative_to(root/'local_data/dallas_correspondence_prospective40_v1/task_v1/assets') for r in dallas),'Unexpected Dallas image path')
    seq_report=validate_sequence_rows(seq,refs);dallas_report=validate_dallas_rows(dallas)
    panel_report=verify_dallas_panels(dallas,root)
    history=json.loads(paths['train_geometry'].read_text());geometry={r['record_id']:r for r in history['training_references_PRIVATE']}
    require(len(geometry)==len(source) and set(geometry)=={r['id'] for r in source},'Train geometry registry differs')
    require(all(geometry[r['id']]['image_sha256']==sha(r['images'][0]) for r in source),'Historical train image hash drift')
    excluded=[r for r in source if r['ability']!='mri_sequence']; lesion={}
    for task in sorted({r['ability'] for r in excluded}):
        group=[r for r in excluded if r['ability']==task];g=[geometry[r['id']] for r in group]
        lesion[task]={'rows':len(group),'labels':dict(Counter(r['reward_model']['ground_truth'] for r in group)),
            'level_counts':dict(Counter(str(r['level']) for r in g)),
            'geometry_positive_planes':sum(r['enhancing' if task=='mri_enhancing_presence' else 'tumor']>0 for r in g),
            'status':'ENGINEERING_ONLY_NOT_ADMITTED','visible_lesion_evidence':'NOT_VERIFIED',
            'independent_mask_reread':'NOT_RUN; saved train-only geometry receipt verified',
            'missing_evidence':['Independent image-linked visual adjudication and ambiguity/abstention labels',
                'Source/site/level shortcut controls and clinically justified negatives',
                'Frame-aware localization target and centroid/multifocal/boundary ambiguity adjudication' if task=='mri_tumor_grid' else 'Visible lesion evidence distinct from mask area']}
    out.mkdir(parents=True)
    candidates=copy.deepcopy(seq+dallas)
    for row in candidates:
        is_seq=row['ability']=='mri_sequence'
        row['rc_data_v1']={'technical_candidate':True,'image_sha256':sha(row['images'][0]),
            'visual_qualification':'NOT_VERIFIED','shortcut_admission':'NOT_VERIFIED','training_ready':False,
            'candidate_route_status':'PAIRED_SEQUENCE_VISUAL_QUALIFICATION_PENDING' if is_seq else 'DALLAS_IDENTITY_ROUTE_UNCERTAIN',
            'source_train_registry_sha256':PINS['sequence_train' if is_seq else 'dallas_train'][1]}
        if is_seq:
            qc=image_qc(row['images'][0],(364,436))
            row['rc_data_v1']['nonzero_fraction']=qc['nonzero_fraction']
            row['rc_data_v1']['sparse_foreground_engineering_warning']=qc['nonzero_fraction']<.05
    candidate_path=out/'candidate_manifest_PRIVATE.jsonl';write_jsonl(candidate_path,candidates)
    # This deliberately empty manifest is the only artifact called qualified training.
    qualified_path=out/'qualified_training_manifest_PRIVATE.jsonl';write_jsonl(qualified_path,[])
    excluded_path=out/'excluded_engineering_routes_PRIVATE.jsonl';write_jsonl(excluded_path,excluded)
    montage_files=make_montages(candidates,out)
    private={'candidate_manifest_sha256':sha(candidate_path),'candidate_rows':len(candidates),
        'qualified_training_manifest_sha256':sha(qualified_path),'qualified_training_rows':0,
        'original_sequence_ids_PRIVATE':[r['id'] for r in old_seq],
        'contact_sheets_PRIVATE':montage_files,'source_train_only_paths_PRIVATE':{k:str(v) for k,v in paths.items()},
        'sources_sha256':{k:pin for k,(rel,pin) in PINS.items()}}
    save_new(out/'audit_PRIVATE.json',private)
    report={'schema':'root-cause-data-train-only-v1','status':'TESTED_CPU_TECHNICAL_CANDIDATE_VISUAL_ADMISSION_PENDING',
        'evidence_tier':'OBSERVED_EXISTING_TRAIN_PNG_AND_REGISTRY_CHECKS','research_tag':'C_TASK_ELIGIBILITY',
        'mission_alignment':{'main_question_link':'Prepare admissible image-linked task candidates for the locked Vero-to-MRI transfer experiment.',
            'claim_falsifier':'168 source-linked sequence PNGs cover the same28 train patients at three levels and both modalities; any missing pair, changed target, split, pixel hash or prompt rejects the build.',
            'change_isolation':'Candidate expands original28 sequence exposures to168 at the same28 patients; Dallas48 are unchanged. This is a distinct C candidate, not matched-budget B or an accuracy result.',
            'evidence_provenance':'Pinned preexisting train registries and image hashes; private candidate and public aggregate hashes.',
            'shortcut_baseline':'Site/level and query identity class-majority ceilings computed from matched train counts; no classifier fitted; Dallas whole-head/intensity/protocol shortcuts remain unresolved.',
            'resources_decision':'One thread, bounded PNG/QC reads only; zero models/optimizer updates. HOLD scientific training admission pending independent visual and shortcut qualification.'},
        'original_pool':{'rows':76,'sequence_rows':28,'Dallas_rows':48,
            'sequence_labels':dict(Counter(r['reward_model']['ground_truth'] for r in old_seq)),
            'sequence_levels':dict(Counter(r['id'].split('_z')[1][:3] for r in old_seq))},
        'candidate_pool':{'rows':216,'sequence_rows':168,'Dallas_uncertain_rows':48,'scientifically_qualified_rows':0,
            'new_people_added':0,'new_images_downloaded':0,'training_ready':False,
            'candidate_manifest_sha256':sha(candidate_path),'qualified_manifest_sha256':sha(qualified_path)},
        'sequence':seq_report,'Dallas':{**dallas_report,'render_and_source_panel_checks':panel_report},
        'excluded_engineering_routes':lesion,'source_train_registry_sha256':private['sources_sha256'],
        'source_code_sha256':{'helper':sha(W/'src/mri_rl/root_cause_data_v1.py'),'builder':sha(__file__),
            'tests':sha(W/'tests/test_root_cause_data_v1.py')},
        'controls':{'new_classifier_fits':0,'new_VLM_generations':0,'new_model_or_optimizer_steps':0,
            'patient_target_confounding':{'original28':patient_label_confounding(old_seq),'paired168':patient_label_confounding(seq),
                'confirmed_design_defect':'Global14/14 balance leaves each original training person paired with only one modality truth; the same-person three-level two-modality candidate removes this specific contingency confound.',
                'causal_limit':'This does not establish that a VLM used identity, improved accuracy, or removed site/protocol/intensity shortcuts.'},
            'source_site_level_majority_ceiling_sequence':.5,'query_identity_majority_ceiling_Dallas':.5,
            'blank_or_no_image_model_control':'NOT_RUN in this CPU-only build',
            'same_patient_modality_swap_model_control':'NOT_RUN; both modalities are paired within all84 patient-level units',
            'Dallas_brain_only_vs_whole_head_control':'NOT_VERIFIED'},
        'read_scope':{'runtime_sources':'Only five pinned TRAIN resources plus Dallas TRAIN source panels/QC and existing train images',
            'test_content_or_outcomes_read_by_builder':False,'internal_dev_content_or_outcomes_read_by_builder':False,
            'upstream_NIfTI_arrays_read_by_builder':0,'cross_source_global_patient_identity':'UNVERIFIED',
            'exploratory_inspection_outside_builder':{
                'fully_train_only':False,
                'mixed_metadata_files_parsed':['ds007045 visual_prompts/private_manifest.json','ds007045 private_receipt_40.json',
                    'Dallas metadata_plan_PRIVATE.json','Dallas task_receipt_SAFE.json'],
                'disclosure':'During source discovery the complete bytes of mixed source-provenance JSON were parsed. The ds007045 source receipt includes unused source-heldout mask-QC summaries; the Dallas safe receipt includes unused internal-development controls. Only train records and train control fields were selected for inspection. Exploration must not be described as zero mixed-metadata bytes read.',
                'final_test_released_projection_final100_or_test_prediction_files_opened':False,
                'nontrain_pixels_or_predictions_used_for_design':False,
                'downstream_builder':'Excludes all four mixed metadata files; reproducible candidate depends solely on pinned TRAIN resources.'}},
        'limitations':['Candidate construction does not prove visual answerability, anatomical grounding, improved accuracy or scientific training readiness.',
            'Sequence source file name/modality identity and saved train image hashes agree; upstream source MRI rerender and source byte-to-PNG reconstruction were not repeated.',
            'Dallas equality is source BIDS identity across visits, not biological change or a clinical label; whole-head face/skull/neck cues may answer it.',
            'Dallas manufacturer/TR/TE matching is an existing construction rule; scanner model/site and brain-only answerability are unverified here.',
            'Lesion/localization labels remain mask-derived engineering evidence. A nonzero mask is not independently adjudicated visible lesion evidence.',
            'Cross-source global patient linkage is unverified. The candidate stays within immutable preexisting TRAIN registries; no heldout independence claim is newly established.'],
        'cross_model_review':'NOT_VERIFIED'}
    save_new(public,report)
    print(json.dumps({'status':report['status'],'candidate_rows':216,'qualified_rows':0,
        'candidate_manifest_sha256':sha(candidate_path),'public_receipt_sha256':sha(public)},sort_keys=True))


if __name__=='__main__':main()
