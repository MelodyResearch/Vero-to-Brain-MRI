#!/usr/bin/env python3
"""Freeze, serially collect, and technically qualify a fixed independent16 source cohort.

Each phase is explicit. Selection is immutable before payload fetch. Source-data
work only: no MLX, model, inference, fitting, training or optimizer imports.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import urllib.error
import urllib.request

W=Path(__file__).resolve().parents[1];sys.path.insert(0,str(W/'src'))
from mri_rl.paired_signal_holdout16_v1 import select_subjects,parse_annex,verify_payload,render_z045,check_pair_geometry,require,SELECTION_PREFIX
COMMIT='283007b96977b5b3285bd22cd67b7d97114b1eed'
QUOTAS={'BO':4,'KA':4,'NSK':2,'TO':2,'JS':2,'FO':1,'ME':1}
LIMIT=400_000_000
SOURCE_BASE='derivatives/processing_MNI152_skull-stripped_N4/processing_MNI152_skull-stripped_N4'
PINS={'exclusion':'6946aff5c4f40faceebc18ea8269e0599c1465e7e2f5d13c3208dfd36698492b',
 'availability':'181cf17c09263352b098466e10f0f2ca95f148abeba9fe418799fa3cf35f472e',
 'inventory_safe':'c7d92f44d3aeaf45850c425a3c643b9d81984d71e12a8220f66fce695e962d7d',
 'README':'66efa8e9ffd194b0aca9e52e9a62201db1b447f9b1f38a11ec8daec265293d96',
 'description':'748fb52a1d943c6f50bb6e4795082e973c4a8694e01a287383a9bf1f20fff763'}
QUESTION='Which MRI sequence is shown? Answer exactly FLAIR or contrast-enhanced T1. Give the final answer in \\boxed{...}.'


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def now():return datetime.now(timezone.utc).isoformat()


def save(p,obj):
    with Path(p).open('x') as f:json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())


def append(p,obj):
    with Path(p).open('a') as f:f.write(json.dumps(obj,sort_keys=True,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())


def code_hashes():return {k:sha(W/p) for k,p in {'helper':'src/mri_rl/paired_signal_holdout16_v1.py','collector':'scripts/collect_paired_signal_holdout16_v1.py','tests':'tests/test_paired_signal_holdout16_v1.py'}.items()}


def checked(path,pin):
    require(sha(path)==pin,'Pinned identity/source metadata changed: '+path.name)
    return json.loads(path.read_text())


def freeze(a):
    R=a.primary_root.resolve();old=a.identity_worktree.resolve();meta=a.metadata_root.resolve();out=a.destination.resolve()
    require(not out.exists(),'New holdout output required;never overwrite a frozen cohort')
    require(subprocess.check_output(['git','-C',str(meta),'rev-parse','HEAD'],text=True).strip()==COMMIT,'Source metadata commit changed')
    B=old/'local_data/mri_holdout_identity_inventory_v2'
    paths={'exclusion':B/'historical_identity_exclusion_PRIVATE.json','availability':B/'remaining_source_availability_PRIVATE.json',
        'inventory_safe':old/'results/MRI_NEW_HOLDOUT_IDENTITY_AND_SOURCE_INVENTORY_V1.json','README':meta/'README.txt','description':meta/'dataset_description.json'}
    exclude=checked(paths['exclusion'],PINS['exclusion']);available=checked(paths['availability'],PINS['availability']);proof=checked(paths['inventory_safe'],PINS['inventory_safe'])
    require(sha(paths['README'])==PINS['README'] and sha(paths['description'])==PINS['description'],'Official acquisition docs changed')
    require(json.loads(paths['description'].read_text())['License']=='CC0','Source license differs')
    excluded={r['subject_PRIVATE'] for r in exclude['excluded']};require(len(excluded)==54 and len(available)==309 and proof['excluded_unique_people']==54,'Historical exclusion coverage changed')
    require(not exclude['missing_catalog_inputs'],'Missing exposure catalogs remain')
    records=[]
    for r in available:
        subject=r['subject_PRIVATE'];require(subject not in excluded,'Exclusion/remaining inventory overlap')
        ptrs=[meta/SOURCE_BASE/subject/f'{subject}_{m}_MNI152_brain_n4.nii.gz' for m in ['flair','t1ce']]
        records.append({'subject_PRIVATE':subject,'site':r['site'],'processed_pair_available':all(p.is_symlink() for p in ptrs)})
    selected=select_subjects(records,excluded,QUOTAS);require(len(selected)==16,'Exact16 source people required')
    historical_source_sha=set()
    for subject in excluded:
        for modality in ['flair','t1ce']:
            p=meta/SOURCE_BASE/subject/f'{subject}_{modality}_MNI152_brain_n4.nii.gz'
            if p.is_symlink():historical_source_sha.add(parse_annex(os.readlink(p))['sha256'])
    files=[];sidecars=[]
    for row in selected:
        subject=row['subject_PRIVATE']
        for modality,label in [('flair','FLAIR'),('t1ce','contrast-enhanced T1')]:
            rel=f'{SOURCE_BASE}/{subject}/{subject}_{modality}_MNI152_brain_n4.nii.gz';ptr=meta/rel;expected=parse_annex(os.readlink(ptr))
            require(expected['sha256'] not in historical_source_sha,'Selected source payload duplicates a historical excluded annotation/image')
            side=meta/subject/'anat'/f"{subject}_{'FLAIR' if modality=='flair' else 'run-02_T1w'}.json"
            require(side.is_file() and not side.is_symlink(),'Acquisition sidecar bytes absent')
            sd=json.loads(side.read_text());require(isinstance(sd,dict),'Acquisition sidecar malformed')
            sid=hashlib.sha256(('paired-signal-holdout16-opaque-v1:'+subject+':'+modality).encode()).hexdigest()[:24]
            files.append({'subject_PRIVATE':subject,'site_PRIVATE':row['site'],'modality_PRIVATE':modality,'label_PRIVATE':label,
                'source_relative_path_PRIVATE':rel,'URL_PRIVATE':'https://s3.amazonaws.com/openneuro.org/ds007045/'+rel,
                'official_pointer_target_PRIVATE':os.readlink(ptr),'source_bytes':expected['bytes'],'source_sha256':expected['sha256'],
                'source_sidecar_path_PRIVATE':str(side),'source_sidecar_sha256':sha(side),'source_sidecar_metadata_PRIVATE':sd,
                'opaque_record_id':sid,'local_volume_relative_PRIVATE':f'volumes_PRIVATE/{sid}.nii.gz','local_PNG_relative_PRIVATE':f'images_PRIVATE/{sid}.png'})
    require(len(files)==32 and len({f['source_sha256'] for f in files})==32,'Source payload duplicates within selected cohort')
    total=sum(f['source_bytes'] for f in files);require(total<=LIMIT,'Frozen source payload exceeds total byte cap')
    out.mkdir(parents=True)
    plan={'schema':'paired-signal-independent16-source-plan-v1','frozen_at_UTC':now(),'source_metadata_commit':COMMIT,'source_license':'CC0',
        'site_quotas':QUOTAS,'BA_quota':0,'selection_prefix':SELECTION_PREFIX,'selection_rule':'Within-site ascending SHA256(prefix+BIDSsubject);metadata complete processed pair only;exclude all54 identities;no outcome/pixel substitution.',
        'selected_subjects_PRIVATE':selected,'files_PRIVATE':files,'source_expected_bytes':total,'network_received_byte_cap':LIMIT,
        'network_policy':'One connection at a time;one retry only for connection/read failures;skip hash+size-verified payloads;resume partial via Range;never replace a fixed subject.',
        'historical_excluded_subject_count':54,'historical_processed_payload_SHA_count':len(historical_source_sha),'historical_source_SHA_overlap':0,
        'input_paths_PRIVATE':{k:str(v) for k,v in paths.items()},'input_sha256':PINS,'metadata_root_PRIVATE':str(meta),
        'code_sha256':code_hashes(),'pixels_or_models_inspected_at_freeze':False,'training_authorized':False,
        'technical_acceptance':{'shape':[182,218,182],'storage_axes':['L','A','S'],'spacing_mm':[1,1,1],
            'finite_voxels_and_affine':True,'paired_affine_atol':1e-6,'z':45,'render':'np.rot90;nonzero[::16]percentile.5/99.5;round/clampuint8;2xbilinearRGB364x436',
            'require_nondegenerate_foreground_window':True,'source_or_geometry_failure':'NOT_ESTABLISHED;retain fixed subject/no substitution',
            'blackness_stats':'Descriptive only;no pixel-based reselection or clinical judgment'},
        'independence_scope':'Disjoint54 historical within-source BIDS identities and processed source SHA;global cross-source/external exposures NOT_VERIFIED',
        'new_holdout_model_outputs':'NOT_RUN'}
    planpath=out/'FROZEN_SOURCE_PLAN_PRIVATE.json';save(planpath,plan)
    public={'schema':'paired-signal-independent16-source-plan-safe-v1','status':'FROZEN_BEFORE_PAYLOAD_FETCH','source_metadata_commit':COMMIT,
        'site_quotas':QUOTAS,'BA_quota':0,'people':16,'paired_volumes':32,'level':45,'source_expected_bytes':total,'network_byte_cap':LIMIT,
        'historical_excluded_people':54,'historical_source_SHA_overlap':0,'selection_rule':plan['selection_rule'],
        'plan_sha256':sha(planpath),'input_sha256':PINS,'code_sha256':code_hashes(),'source_license':'CC0','payloads_fetched':0,
        'models_loaded':0,'optimizer_updates':0,'training_authorized':False,'holdout_technical_qualification':'PENDING_ACTUAL_PAYLOADS'}
    save(W/'results/PAIRED_SIGNAL_HOLDOUT16_PLAN_V1.json',public)
    print(json.dumps({'status':public['status'],'people':16,'volumes':32,'expected_bytes':total,'plan_sha256':sha(planpath)},sort_keys=True),flush=True)


def load_plan(a):
    p=a.destination/'FROZEN_SOURCE_PLAN_PRIVATE.json';require(a.expected_plan_sha256 and sha(p)==a.expected_plan_sha256,'Externally bound frozen plan SHA required')
    plan=json.loads(p.read_text());require(plan['code_sha256']==code_hashes(),'Frozen source code changed after selection')
    require(plan['site_quotas']==QUOTAS and plan['source_expected_bytes']<=LIMIT,'Frozen source scope changed')
    return plan


def received(journal):
    return sum(json.loads(s).get('received_bytes',0) for s in journal.read_text().splitlines()) if journal.exists() else 0


def fetch(a):
    plan=load_plan(a);out=a.destination;journal=out/'fetch_journal_PRIVATE.jsonl';completed=0
    for index,item in enumerate(plan['files_PRIVATE']):
        p=out/item['local_volume_relative_PRIVATE'];p.parent.mkdir(exist_ok=True);n=item['source_bytes'];expected=item['source_sha256']
        if verify_payload(p,n,expected):completed+=1;continue
        require(not p.exists(),'Existing unverified final payload must be inspected;no overwrite')
        part=p.with_suffix(p.suffix+'.partial')
        for attempt in range(2):
            offset=part.stat().st_size if part.exists() else 0;require(offset<=n,'Partial payload exceeds expected source bytes')
            if offset==n:
                require(verify_payload(part,n,expected),'Completed partial SHA differs from official source')
                os.replace(part,p);break
            budget=received(journal);require(budget+(n-offset)<=LIMIT,'Remaining scoped network budget insufficient')
            req=urllib.request.Request(item['URL_PRIVATE'],headers={'Range':f'bytes={offset}-'} if offset else {})
            try:
                with urllib.request.urlopen(req,timeout=45) as response:
                    require(response.geturl()==item['URL_PRIVATE'],'Unexpected source redirect;do not bypass access requirements')
                    require(response.status==(206 if offset else 200),'Source did not honor expected full/Range response')
                    length=response.headers.get('Content-Length')
                    require(length is None or int(length)==n-offset,'Source Content-Length differs from frozen annex bytes')
                    if offset:require(response.headers.get('Content-Range','').startswith(f'bytes {offset}-'),'Range response start mismatch')
                    with part.open('ab' if offset else 'xb') as f:
                        written=offset
                        while True:
                            chunk=response.read(min(1024*1024,n-written+1))
                            if not chunk:break
                            require(budget+len(chunk)<=LIMIT,'Actual network byte cap exceeded')
                            f.write(chunk);f.flush();os.fsync(f.fileno());written+=len(chunk);budget+=len(chunk)
                            append(journal,{'kind':'RECEIVED','file_index':index,'attempt':attempt+1,'received_bytes':len(chunk),'UTC':now()})
                            require(written<=n,'Source exceeds frozen annex size')
                require(verify_payload(part,n,expected),'Downloaded source bytes/SHA mismatch')
                os.replace(part,p);append(journal,{'kind':'VERIFIED','file_index':index,'bytes':n,'sha256':expected,'UTC':now()});break
            except (urllib.error.URLError,TimeoutError,socket.timeout,ConnectionResetError) as error:
                append(journal,{'kind':'CONNECTION_FAILURE','file_index':index,'attempt':attempt+1,'error_PRIVATE':str(error),'UTC':now()})
                if attempt:raise
                continue
        require(verify_payload(p,n,expected),'Source file remains unverified')
        completed+=1
        print(json.dumps({'verified_volumes':completed,'of':32,'network_received_bytes':received(journal)}),flush=True)
    save(out/'FETCH_COMPLETE_SAFE.json',{'schema':'paired-signal-holdout16-fetch-v1','status':'ALL32_SOURCE_PAYLOADS_HASH_SIZE_VERIFIED',
        'plan_sha256':a.expected_plan_sha256,'verified_files':32,'verified_source_bytes':plan['source_expected_bytes'],
        'network_received_bytes':received(journal),'network_byte_cap':LIMIT,'model_loads':0,'optimizer_updates':0})


def qualify(a):
    import numpy as np
    import nibabel as nib
    from PIL import Image
    plan=load_plan(a);out=a.destination;done=out/'TECHNICAL_COMPLETE_SAFE.json'
    require(not done.exists(),'Completed technical qualification must not be repeated')
    require((out/'FETCH_COMPLETE_SAFE.json').is_file(),'Complete verified source fetch required')
    forms=[];records=[];qual=[];image_dir=out/'images_PRIVATE';image_dir.mkdir(exist_ok=False)
    subjects=[r['subject_PRIVATE'] for r in plan['selected_subjects_PRIVATE']]
    for subject in subjects:
        pair=[r for r in plan['files_PRIVATE'] if r['subject_PRIVATE']==subject];require(len(pair)==2,'Frozen pair missing')
        paths=[out/r['local_volume_relative_PRIVATE'] for r in pair]
        require(all(verify_payload(p,r['source_bytes'],r['source_sha256']) for p,r in zip(paths,pair)),'Source mutated before qualification')
        imgs=[nib.load(p) for p in paths];g=check_pair_geometry(imgs[0].shape,imgs[0].affine,imgs[1].shape,imgs[1].affine)
        per_pair=[]
        for img,item in zip(imgs,pair):
            data=np.asanyarray(img.dataobj);image,window,stats=render_z045(data)
            p=out/item['local_PNG_relative_PRIVATE'];image.save(p,format='PNG');pngsha=sha(p)
            with Image.open(p) as saved:require(saved.size==(364,436) and saved.mode=='RGB' and not saved.info,'Opaque PNG schema changed')
            patient=hashlib.sha256(subject.encode()).hexdigest()[:12];e={'index':len(records),'internal_split':'independent_validation',
                'patient_key':patient,'patient_lineage':'BIDS_SUBJECT_VERIFIED','level':45,'question':QUESTION,
                'answer':item['label_PRIVATE'],'reward_type':'mri_sequence','task_type':'mri_sequence',
                'target_provenance':'Official ds007045 acquisition modality identity;FLAIR collection/run02 postGdT1 mapping'}
            row={'id':item['opaque_record_id'],'ability':'mri_sequence','data_source':'OpenNeuro_ds007045','extra_info':e,'images':[str(p.resolve())],
                'prompt':[{'role':'user','content':[{'type':'image','image':str(p.resolve())},{'type':'text','text':QUESTION}]}],
                'reward_model':{'ground_truth':item['label_PRIVATE'],'style':'rule'},'training_authorized':False,
                'paired_signal_holdout16_v1':{'plan_sha256':a.expected_plan_sha256,'source_sha256':item['source_sha256'],'render_sha256':pngsha,
                    'source_acquisition_modality_technical_eligible':True,'clinical_expert_gate_required':False,'new_holdout_model_outputs':'NOT_RUN',
                    'normalization_window_PRIVATE':window,'geometry_PRIVATE':g,'technical_FOV_stats_PRIVATE':stats}}
            records.append(row);per_pair.append({'record_id_PRIVATE':row['id'],'modality_PRIVATE':item['modality_PRIVATE'],'PNG_sha256':pngsha,
                'source_sha256':item['source_sha256'],'dtype':str(data.dtype),'normalization_window_PRIVATE':window,'stats':stats})
            del data,image
        qual.append({'subject_PRIVATE':subject,'site_PRIVATE':pair[0]['site_PRIVATE'],'geometry':g,'affine_PRIVATE':imgs[0].affine.tolist(),'rows_PRIVATE':per_pair})
        del imgs
    require(len(records)==32 and len({r['id'] for r in records})==32,'Exact candidate coverage differs')
    target=out/'holdout32_PRIVATE.verl.jsonl'
    with target.open('x') as f:
        for r in records:f.write(json.dumps(r,sort_keys=True,allow_nan=False)+'\n')
        f.flush();os.fsync(f.fileno())
    save(out/'source_geometry_render_qualification_PRIVATE.json',qual)
    # Future model outputs are separate. Sealing records no inference or outcome.
    sealed={'schema':'paired-signal-independent16-sealed-inputs-v1','sealed_at_UTC':now(),'plan_sha256':a.expected_plan_sha256,
        'manifest_sha256':sha(target),'qualification_sha256':sha(out/'source_geometry_render_qualification_PRIVATE.json'),
        'render_sha256_by_record_PRIVATE':{r['id']:r['paired_signal_holdout16_v1']['render_sha256'] for r in records},
        'source_payload_sha256_PRIVATE':[r['source_sha256'] for r in plan['files_PRIVATE']],
        'training_forbidden':True,'model_outputs':'NOT_RUN','model_output_rows':0}
    save(out/'SEALED_INPUTS_PRIVATE.json',sealed)
    stats=[r['paired_signal_holdout16_v1']['technical_FOV_stats_PRIVATE'] for r in records];fetchmeta=json.loads((out/'FETCH_COMPLETE_SAFE.json').read_text())
    result={'schema':'paired-signal-independent16-technical-v1','status':'ACTUAL32_VOLUMES_AND_FIXED_Z045_RENDERS_TECHNICALLY_QUALIFIED_SEALED_NOT_EVALUATED',
        'evidence_tier':'OBSERVED','research_tag':'C_TASK_ELIGIBILITY','mission_alignment':{'link':'Provide actual new-patient acquisition-modality validation inputs for the locked Vero-to-MRI comparison.',
            'falsifier':'Fixed source IDs/payloadSHA/bytes/geometry/affine/metadata/render contract failure blocks establishment without replacement.',
            'isolation':'New evaluation sources only;no training samples,model,loss,normalization or task prompt intervention.',
            'provenance':'Pinned publicsource metadata and54-exclusion proof;fixed SHA-ranked16 selected beforepayload;actual32 SHAverified.',
            'shortcut':'Samez045 pairedmodalities pernewpatient;7sites;BA unavailable;site/protocol/intensity shortcuts not excluded.',
            'resources_decision':'Single-thread boundedserial<=400MB sourcefetch;32decoded volumes;nomodel/fitting/updates;sealedfutureoutputsNOT_RUN.'},
        'people':16,'sequence_records':32,'site_people':QUOTAS,'BA_people':0,'level':45,'modality_counts':{'FLAIR':16,'contrast-enhanced T1':16},
        'source_metadata_commit':COMMIT,'source_license':'CC0','plan_sha256':a.expected_plan_sha256,'input_sha256':PINS,'code_sha256':code_hashes(),
        'source_payload_files_verified':32,'verified_source_bytes':plan['source_expected_bytes'],'actual_network_received_bytes':fetchmeta['network_received_bytes'],
        'network_byte_cap':LIMIT,'source_payload_expectedSHA_size_matches':32,'source_payload_SHA_unique':32,
        'pair_geometry_qualified':16,'source_shape':[182,218,182],'source_storage_axcodes':{'LAS':32},'world_coordinates':'NIfTI affine RAS+;LAS array storage preserved',
        'spacing_mm':[1,1,1],'paired_affine_equal_atol_1e_6':16,'all_voxels_finite':32,'new_source_volumes_decoded':32,'segmentation_masks_read':0,
        'acquisition_modality_endpoint':'OfficialFLAIR collection andrun02T1 postGd mapping;clinicalexpert adjudication not required for this metadata identity endpoint.',
        'render_contract':plan['technical_acceptance']['render'],'rendered_PNGs':32,'PNG_shape':[364,436],'PNG_mode':'RGB','opaque_filenames':True,'PNG_text_metadata_keys':0,
        'technical_FOV_ranges':{k:[min(s[k] for s in stats),max(s[k] for s in stats)] for k in ['z045_nonzero_fraction','display_black_fraction','display_saturated_fraction','display_std']},
        'technical_stats_used_for_reselection':False,'clinical_visual_adjudication':'NOT_RUN_NOT_INVENTED','new_model_outputs':'NOT_RUN','model_output_rows':0,
        'identity_independence':{'excluded_historical_people':54,'selected_subject_overlap':0,'historical_processed_source_SHA_overlap':0,
            'scope':plan['independence_scope'],'external_exposure_or_cross_source_global_identity':'NOT_VERIFIED'},
        'manifest_sha256':sha(target),'sealed_inputs_sha256':sha(out/'SEALED_INPUTS_PRIVATE.json'),
        'private_artifact_sha256':{p.name:sha(p) for p in out.iterdir() if p.is_file()},
        'read_scope':{'old_final_registry_or_outputs_opened':0,'old_final_outcomes_used':False,'historical_identity_catalog_only':True,
            'models_loaded':0,'MLX_imports':0,'inference_calls':0,'classifier_fits':0,'optimizer_updates':0,'clinical_diagnoses':0},
        'holdout_input_technical_eligibility':True,'training_authorized':False,'generalization_result_claimed':False,'cross_model_review':'NOT_VERIFIED',
        'remaining_limits':['Evaluation and modeloutputs are NOT_RUN;technical source validity is not an accuracy result.',
            'Within-source54-catalog exclusion does not certify unknown external/manual or cross-source historical exposure.',
            'Seven source sites represented;BA has no eligible remainder;not equal-site or site-held-out validation.',
            'MNI/skullstrip/N4 andper-volume percentile normalization retain possible acquisition/site/intensity shortcuts;no clinical visibility label created.']}
    save(done,result);save(W/'results/PAIRED_SIGNAL_HOLDOUT16_TECHNICAL_V1.json',result)
    print(json.dumps({'status':result['status'],'people':16,'records':32,'verified_bytes':plan['source_expected_bytes'],'manifest_sha256':sha(target),'new_model_outputs':'NOT_RUN'}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--freeze',action='store_true');mode.add_argument('--fetch',action='store_true');mode.add_argument('--qualify',action='store_true')
    p.add_argument('--primary-root',type=Path,required=True);p.add_argument('--identity-worktree',type=Path,required=True);p.add_argument('--metadata-root',type=Path,required=True)
    p.add_argument('--destination',type=Path,default=W/'local_data/paired_signal_holdout16_v1/candidate_v1');p.add_argument('--expected-plan-sha256');a=p.parse_args()
    require(all(os.environ.get(k)=='1' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']),'One-thread CPU required')
    a.destination=a.destination.resolve();require(a.destination.is_relative_to(W/'local_data/paired_signal_holdout16_v1'),'Scoped private output required')
    try:
        if a.freeze:freeze(a)
        elif a.fetch:fetch(a)
        else:qualify(a)
    except Exception as error:
        if a.destination.exists():
            append(a.destination/'FAILURES_PRIVATE.jsonl',{'phase':'freeze' if a.freeze else 'fetch' if a.fetch else 'qualify','error_type':type(error).__name__,'error_PRIVATE':str(error),'UTC':now()})
        print(json.dumps({'status':'NOT_ESTABLISHED','phase':'freeze' if a.freeze else 'fetch' if a.fetch else 'qualify','error_type':type(error).__name__}),flush=True)
        raise


if __name__=='__main__':main()
