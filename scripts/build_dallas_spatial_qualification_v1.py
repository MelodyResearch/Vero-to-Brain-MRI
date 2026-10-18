#!/usr/bin/env python3
"""One-thread TRAIN source reread and label-separated visual review material."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib,json,os,sys
from pathlib import Path
W=Path(__file__).resolve().parents[1];sys.path.insert(0,str(W/'src'))
import nibabel as nib
import numpy as np
from PIL import Image,ImageChops
from mri_rl.dallas_spatial_qualification_v1 import (sha,require,require_train,render_slice,spatial_geometry,cranial_roi_occlusion,sheet)
from mri_rl.visit_correspondence import token,render_volume,compose_pair

def save(path,obj):
    with path.open('x') as f:json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')

def read_rows(path,pin):
    require(sha(path)==pin,'Frozen TRAIN registry drift')
    rows=[json.loads(l) for l in path.read_text().splitlines() if l.strip()];require_train(rows);return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--primary-root',type=Path,required=True);p.add_argument('--destination',type=Path,required=True);a=p.parse_args()
    require(all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS')),'One-thread caps required')
    R=a.primary_root.resolve();out=a.destination.resolve()
    require(out.is_relative_to(W/'local_data/dallas_spatial_qualification_v1') and not out.exists(),'New private versioned destination required')
    dtrain=R/'local_data/dallas_correspondence_prospective40_v1/task_v1/train.verl.jsonl'
    ttrain=R/'local_data/ds007045_bounded/vero_training_v1/train.verl.jsonl'
    dallas=read_rows(dtrain,'6adf06674050e395b704e4b76103f067166b905016920195603e271690de4f14')
    train=read_rows(ttrain,'0efb588b328e3d9af2a339d9237836357a5baff5d9db8b376dc716de3d1f10e3')
    out.mkdir(parents=True);(out/'assets_PRIVATE').mkdir();(out/'sheets_PRIVATE').mkdir()
    records=[];source_qc=[];dpanels={};dpaths={};file_reads=[]
    # Directory names alone are enumerated. Only TRAIN-token-matching paths are opened.
    people={r['extra_info'][k] for r in dallas for k in ('patient_key','candidate_patient_key')}
    dmap={token('patient',x.name):x for x in (R/'local_data/ds004856').glob('sub-*') if token('patient',x.name) in people}
    require(len(dmap)==len(people)==24,'Dallas TRAIN source directory match incomplete')
    expected={r['extra_info'][k] for r in dallas for k in ('query_source_sha256','candidate_source_sha256')}
    for key,directory in sorted(dmap.items()):
        for source in sorted(directory.glob('ses-wave*/anat/*T1w.nii.gz')):
            source_sha=sha(source);require(source_sha in expected,'Unexpected non-TRAIN Dallas payload')
            panel,qc=render_volume(source);frozen=R/'local_data/dallas_correspondence_prospective40_v1/task_v1/assets'/(token('volume',source_sha)+'.png')
            with Image.open(frozen) as old:require(ImageChops.difference(panel,old).getbbox() is None,'Dallas source rerender differs')
            ni=nib.load(str(source));sidecar=source.with_name(source.name.replace('.nii.gz','.json'));meta={}
            if sidecar.exists():meta=json.loads(sidecar.read_text());file_reads.append(str(sidecar))
            source_qc.append({'kind':'Dallas','source_PRIVATE':str(source),'source_sha256':source_sha,'patient_key_PRIVATE':key,
                'original_orientation':list(nib.aff2axcodes(ni.affine)),'qform_code':int(ni.header['qform_code']),'sform_code':int(ni.header['sform_code']),
                'affine_PRIVATE':ni.affine.tolist(),'source_to_frozen_panel_exact':True,**qc,
                'scanner_model':meta.get('ManufacturersModelName','NOT_VERIFIED_NO_TRAIN_SIDECAR'),
                'site':meta.get('InstitutionName','NOT_VERIFIED_NO_TRAIN_SIDECAR')})
            dpanels[source_sha]=panel;dpaths[source_sha]=str(source);file_reads.extend([str(source),str(frozen)])
    require(set(dpanels)==expected and len(dpanels)==48,'Dallas volume inventory differs')
    dorder=sorted(dallas,key=lambda r:hashlib.sha256(('dallas-blind-v1|'+r['id']).encode()).hexdigest())
    dkeys=[];whole=[];occluded=[]
    for i,r in enumerate(dorder,1):
        e=r['extra_info'];q=dpanels[e['query_source_sha256']];c=dpanels[e['candidate_source_sha256']];pair=compose_pair(q,c)
        with Image.open(r['images'][0]) as old:require(ImageChops.difference(pair,old).getbbox() is None,'Dallas frozen pair mismatch')
        identifier=f'D{i:03d}';whole.append(pair);occluded.append(compose_pair(cranial_roi_occlusion(q),cranial_roi_occlusion(c)))
        dkeys.append({'review_id':identifier,'record_id_PRIVATE':r['id'],'source_equality_truth_PRIVATE':r['reward_model']['ground_truth'],
                     'query_source_PRIVATE':dpaths[e['query_source_sha256']],'candidate_source_PRIVATE':dpaths[e['candidate_source_sha256']]})
        records.append({'review_id':identifier,'record_id_PRIVATE':r['id'],'task':r['ability'],'split':'train','source_pixel_reconstruction':'PASS_EXACT',
                        'technical_eligibility':True,'visual_eligibility':'NOT_VERIFIED','expert_eligibility':'NOT_VERIFIED','training_authorized':False,
                        'route_status':'WHOLE_HEAD_IDENTITY_ENGINEERING_ONLY_BRAIN_GROUNDING_UNPROVEN'})
    for start in range(0,48,6):
        ids=[r['review_id'] for r in dkeys[start:start+6]]
        for name,images in [('wholehead',whole),('fixed_cranial_roi_occlusion_NOT_SKULLSTRIP',occluded)]:
            sheet(images[start:start+6],ids,out/'sheets_PRIVATE'/f'dallas_{name}_{start//6+1:02d}.png',2,768,408)
    save(out/'dallas_truth_separate_PRIVATE.json',dkeys)
    print(json.dumps({'stage':'Dallas48_NIFTI_reconstructed','volumes':48,'pairs':48}),flush=True)
    # All existing TRAIN tumor-presence/enhancement/grid rows; source arrays are read one modality/person at a time.
    lesion=[r for r in train if r['ability']!='mri_sequence'];tpeople={r['extra_info']['patient_key'] for r in lesion}
    tbase=R/'local_data/ds007045_bounded';tmap={hashlib.sha256(x.name.encode()).hexdigest()[:12]:x for x in tbase.glob('sub-*') if hashlib.sha256(x.name.encode()).hexdigest()[:12] in tpeople}
    require(len(tmap)==len(tpeople)==28,'Tumor TRAIN source directory match incomplete')
    planes={};technical=[]
    for key,directory in sorted(tmap.items()):
        selected=[r for r in lesion if r['extra_info']['patient_key']==key];levels=sorted({int(r['id'].split('_z')[1][:3]) for r in selected})
        maskpath=directory/(directory.name+'_MNI152_tumor-mask.nii.gz');maskni=nib.load(str(maskpath));mask_sha=sha(maskpath);file_reads.append(str(maskpath))
        require(maskni.shape==(182,218,182) and np.isfinite(maskni.affine).all(),'Invalid mask grid')
        mp={z:np.asanyarray(maskni.dataobj[:,:,z]) for z in levels}
        require(all(set(np.unique(v))<={0,1,2,3} for v in mp.values()),'Unexpected mask labels')
        renders={}
        for modality in ('flair','t1ce'):
            source=directory/(directory.name+'_'+modality+'_MNI152_brain_n4.nii.gz');ni=nib.load(str(source));file_reads.append(str(source))
            require(ni.shape==maskni.shape and np.allclose(ni.affine,maskni.affine,atol=1e-5,rtol=0),'Image/mask grid mismatch; no silent resampling')
            data=ni.get_fdata(dtype=np.float32);source_hash=sha(source)
            source_qc.append({'kind':'Tumor','source_PRIVATE':str(source),'source_sha256':source_hash,'mask_sha256':mask_sha,
                'orientation':list(nib.aff2axcodes(ni.affine)),'shape':list(ni.shape),'affine_PRIVATE':ni.affine.tolist(),
                'image_mask_affine_equal_atol':1e-5,'resampling':'NONE_SAME_GRID_VERIFIED','finite_voxels':bool(np.isfinite(data).all()),
                'qform_code':int(ni.header['qform_code']),'sform_code':int(ni.header['sform_code'])})
            for z in levels:
                im=render_slice(data,z);frozen=tbase/'visual_prompts'/f'{key}_z{z:03d}_{modality}.png'
                with Image.open(frozen) as old:require(ImageChops.difference(im,old.convert('L')).getbbox() is None,'Source MRI historical render mismatch')
                renders[modality,z]=im;file_reads.append(str(frozen))
            del data,ni
        for z in levels:
            relevant=[r for r in selected if int(r['id'].split('_z')[1][:3])==z];imagepath=Path(relevant[0]['images'][0]);pair=Image.open(imagepath).convert('RGB');file_reads.append(str(imagepath))
            require(pair.size==(752,470),'Paired MRI frame changed')
            for modality,x in [('flair',4),('t1ce',384)]:
                require(ImageChops.difference(pair.crop((x,28,x+364,464)).convert('L'),renders[modality,z]).getbbox() is None,'Paired display frame mismatch')
            geom=spatial_geometry(mp[z]>0);enh=spatial_geometry(mp[z]==3)
            cropped=Image.new('RGB',(728,436));cropped.paste(renders['flair',z],(0,0));cropped.paste(renders['t1ce',z],(364,0))
            overlay=cropped.copy();arr=np.asarray(overlay).copy();tm=np.asarray(Image.fromarray(np.rot90((mp[z]>0).astype('uint8')*255)).resize((364,436),Image.Resampling.NEAREST))>0
            em=np.asarray(Image.fromarray(np.rot90((mp[z]==3).astype('uint8')*255)).resize((364,436),Image.Resampling.NEAREST))>0
            for panel in (0,1):
                aview=arr[:,panel*364:(panel+1)*364,:];aview[tm]=(aview[tm]*.55+np.array([255,0,0])*.45).astype('uint8');aview[em]=(aview[em]*.45+np.array([0,255,255])*.55).astype('uint8')
            planes[key,z]={'image':cropped,'overlay':Image.fromarray(arr),'geometry':geom,'enhancing_geometry':enh,'source_mask_sha256':mask_sha,'rows':relevant}
            for r in relevant:
                truth=r['reward_model']['ground_truth'];g=enh if r['ability']=='mri_enhancing_presence' else geom
                expected_truth=geom['global_grid'] if r['ability']=='mri_tumor_grid' else ('yes' if g['area'] else 'no')
                require(truth==expected_truth,'Mask-derived label does not match pixels/frame')
                technical.append({'record_id_PRIVATE':r['id'],'task':r['ability'],'split':'train','source_pixel_reconstruction':'PASS_EXACT',
                    'technical_eligibility':True,'geometric_flags':g['geometric_flags'],'visual_eligibility':'NOT_VERIFIED',
                    'expert_eligibility':'NOT_VERIFIED','training_authorized':False,'label_provenance':'MASK_DERIVED_NOT_VISIBLE_LESION_GOLD'})
        del maskni,mp,renders
    ordered=sorted(planes,key=lambda x:hashlib.sha256(('tumor-blind-v1|'+x[0]+'|'+str(x[1])).encode()).hexdigest())
    tkeys=[]
    for i,key in enumerate(ordered,1):
        v=planes[key];identifier=f'T{i:03d}';v['review_id']=identifier
        tkeys.append({'review_id':identifier,'records_PRIVATE':[r['id'] for r in v['rows']],
            'truth_PRIVATE':{r['ability']:r['reward_model']['ground_truth'] for r in v['rows']},
            'tumor_geometry':v['geometry'],'enhancing_geometry':v['enhancing_geometry'],'mask_sha256':v['source_mask_sha256']})
        for r in technical:
            if r['record_id_PRIVATE'] in tkeys[-1]['records_PRIVATE']:r['review_id']=identifier
    for start in range(0,len(ordered),12):
        group=[planes[k] for k in ordered[start:start+12]];ids=[v['review_id'] for v in group]
        for name,field in [('MRI_ONLY','image'),('MASK_OVERLAY_NOT_BLIND','overlay')]:
            sheet([v[field] for v in group],ids,out/'sheets_PRIVATE'/f'tumor_{name}_{start//12+1:02d}.png',4,368,242)
    records.extend(technical);save(out/'tumor_truth_separate_PRIVATE.json',tkeys);save(out/'source_qc_PRIVATE.json',source_qc)
    with (out/'candidate_manifest_PRIVATE.jsonl').open('x') as f:
        for r in records:f.write(json.dumps(r,sort_keys=True)+'\n')
    (out/'qualified_training_manifest_PRIVATE.jsonl').touch(exist_ok=False)
    save(out/'read_inventory_PRIVATE.json',{'TRAIN_registries':[str(dtrain),str(ttrain)],'source_and_image_files_opened':sorted(set(file_reads)),
        'directory_names_enumerated_only':'Local sub-* names were hashed to existing TRAIN tokens before selecting any file payload',
        'mixed_metadata_files_opened':[],'nontrain_pixels_or_predictions_opened':False,'model_loads':0,'optimizer_updates':0})
    save(out/'aggregate_SAFE.json',{'schema':'dallas-spatial-qualification-v1','status':'TESTED_CPU_REAL_TRAIN_SOURCES_VISUAL_AND_EXPERT_ADMISSION_PENDING',
        'evidence_tier':'OBSERVED_SOURCE_NIFTI_AND_EXACT_PIXEL_RECONSTRUCTION','research_tag':'C_TASK_ELIGIBILITY',
        'mission_alignment':{'link':'Qualify actual MRI task evidence for locked Vero-to-MRI task-balanced transfer.','falsifier':'Any source hash, image/mask affine, label, or frozen display-pixel mismatch rejects the candidate.',
            'isolation':'Frozen inputs and all historical models unchanged; new TRAIN-only technical qualification and review assets.',
            'controls':'Wholehead versus fixed geometric cranial ROI occlusion; unvalidated occlusion cannot establish brain-only answerability.',
            'resources':'One thread, source data only, zero models and zero updates.','decision':'HOLD training; technical qualification is not clinical visibility validation.'},
        'Dallas':{'pairs':48,'people':24,'NIfTI_volumes_reopened':48,'exact_source_panel_and_pair_reconstructions':48,
            'scanner_model_or_site_verified':sum(x['scanner_model']!='NOT_VERIFIED_NO_TRAIN_SIDECAR' for x in source_qc if x['kind']=='Dallas'),
            'wholehead_shortcut_status':'UNRESOLVED','fixed_occlusion':'NOT_VALIDATED_SKULLSTRIP_NOT_BRAIN_ONLY'},
        'tumor':{'rows':len(technical),'people':28,'planes':len(planes),'MRI_volumes_reopened':56,'mask_volumes_reopened':28,
            'same_grid_affine_and_exact_source_pixel_reconstruction':True,'resampling':'NONE','frame':'np.rot90 native LAS; FLAIR origin(4,28),T1ce origin(384,28);2x bilinear; full728x436 review strip removes titles only',
            'task_rows':dict(Counter(r['task'] for r in technical)),
            'geometric_flag_row_counts':dict(Counter(flag for r in technical for flag in r['geometric_flags'])),
            'independent_task_limit':'Three routes share one mask source; no independent clinical gold.',
            'visible_lesion_evidence':'NOT_VERIFIED_MASK_NONZERO_IS_NOT_VISIBILITY'},
        'source_registry_sha256':{'Dallas_TRAIN':sha(dtrain),'tumor_TRAIN':sha(ttrain)},
        'candidate_manifest_sha256':sha(out/'candidate_manifest_PRIVATE.jsonl'),'qualified_rows':0,'training_authorized':False,
        'model_loads':0,'new_inference':0,'optimizer_updates':0,'new_downloads':0,'cross_model_review':'NOT_VERIFIED',
        'read_scope':'Only TRAIN registries, token-filtered TRAIN source NIfTIs and TRAIN frozen images; no mixed source metadata, final100, dev pixels or predictions.'})
    print(json.dumps({'stage':'complete','tumor_rows':len(technical),'tumor_planes':len(planes),'qualified_training_rows':0}),flush=True)

if __name__=='__main__':main()
