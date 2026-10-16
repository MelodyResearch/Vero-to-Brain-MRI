"""Retrieve only the new-patient union of the frozen C2 geometry match.

Plan/fetch contain no array imports. Render requires a separate explicit call in
a GPU-idle window. Geometry eligibility never grants scientific task admission.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import urllib.request


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write_json(path,obj):
    if path.exists():raise FileExistsError(path)
    with path.open('x') as f:json.dump(obj,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())


def selected_new_union(combined):
    return {r['patient'] for c in combined['categories'].values() for r in c['selected_PRIVATE'] if r.get('MRI_image_status')=='NOT_DOWNLOADED_OR_RENDERED'}


def check_scope(rows,existing,reserved,max_bytes=150_000_000):
    subjects=[r['subject_PRIVATE'] for r in rows]
    if len(subjects)!=len(set(subjects)) or set(subjects)&(set(existing)|set(reserved)):raise ValueError('Patient exclusion/inventory failed')
    if sum(p['bytes'] for r in rows for p in r['payloads'][:2])>max_bytes:raise ValueError('Bounded image byte budget exceeded')


def plan(root,out):
    if out.exists():raise FileExistsError(out)
    combined_path=root/'overnight_c2_masks_v2/combined_geometry_PRIVATE.json';combined=json.loads(combined_path.read_text())
    selection_path=root/'overnight_c1_v2/matched_subsets_v2_PRIVATE.json';selection=json.loads(selection_path.read_text())['C2']
    mask_path=root/'overnight_c2_masks_v2/receipt_PRIVATE.json';masks=json.loads(mask_path.read_text())
    old=json.loads((root/'visual_prompts/private_manifest.json').read_text())['private_cases']
    reserve=json.loads((root/'balanced_sft_independent_audit_v1/prospective_metadata_selection_PRIVATE.json').read_text())['selected_PRIVATE']
    mapping={hashlib.sha256(('C2-training-v2:'+c['subject_PRIVATE']).encode()).hexdigest()[:16]:c for c in selection['selected_PRIVATE']}
    union=selected_new_union(combined)
    if not union<=set(mapping) or len(union)!=10:raise ValueError('Frozen union changed')
    chosen=[dict(mapping[k],patient=k) for k in sorted(union)];check_scope(chosen,{c['private_subject'] for c in old},{c['subject_PRIVATE'] for c in reserve})
    old_pair=root/'overnight_c2_upper_image_v1';prior=json.loads((old_pair/'download_PRIVATE.json').read_text())
    files=[]
    for c in chosen:
        for p in c['payloads'][:2]:
            reused=c['subject_PRIVATE']==prior['subject_PRIVATE']
            local=old_pair/Path(p['path']).name if reused else out/Path(p['path']).name
            if reused and (not local.is_file() or local.stat().st_size!=p['bytes'] or sha(local)!=p['sha256']):raise ValueError('Existing pair invalid')
            files.append({**p,'subject_PRIVATE':c['subject_PRIVATE'],'patient':c['patient'],'reused':reused,'local_path_PRIVATE':str(local.resolve()),'source_url':'https://s3.amazonaws.com/openneuro.org/ds007045/'+p['path']})
    scope={'tag':'C_TASK_ELIGIBILITY','status':'RETRIEVAL_SCOPE_FROZEN','source_commit':selection['source_commit'],
        'source_code_sha256':sha(__file__),'root_PRIVATE':str(root.resolve()),'combined_geometry_sha256':sha(combined_path),
        'selection_sha256':sha(selection_path),'mask_receipt_sha256':sha(mask_path),'new_patients':len(chosen),'image_files':len(files),
        'image_bytes':sum(x['bytes'] for x in files),'missing_files':sum(not x['reused'] for x in files),
        'missing_bytes':sum(x['bytes'] for x in files if not x['reused']),'max_total_bytes':150_000_000,
        'chosen_PRIVATE':chosen,'files_PRIVATE':files,'existing_and_reserved_excluded':True,
        'mission_alignment':{'main_question':'Enable real-image C task eligibility for later Vero-to-MRI mixture comparisons.',
          'claim_and_falsifier':'Frozen matched mask cohorts can be linked to source MRI; checksum/grid/label failure invalidates affected records.',
          'change_isolation':'Only the10 new patients already required by the frozen32/12 matches; no changes toA/B, thresholds or reserved evaluation.',
          'evidence_provenance':'Pinned official annex hashes, frozen selection/match/mask receipt hashes; no synthetic targets.',
          'shortcut_baseline':'Preserve within-level balance; site and visible-answerability confounds remain; mask eligibility is not clinical evidence.',
          'resources_decision':'≤150MB total; reuse oneverifiedpair; serial streamed fetch; decode onevolume atatime only in explicitGPUidlewindow; noCtraining.'}}
    out.mkdir();write_json(out/'scope_PRIVATE.json',scope)
    print(json.dumps({k:scope[k] for k in ('new_patients','image_files','image_bytes','missing_files','missing_bytes')}))


def verify_scope(out):
    scope=json.loads((out/'scope_PRIVATE.json').read_text());root=Path(scope['root_PRIVATE'])
    if sha(__file__)!=scope['source_code_sha256']:raise ValueError('Scope source changed')
    for name,path in [('combined_geometry_sha256',root/'overnight_c2_masks_v2/combined_geometry_PRIVATE.json'),('selection_sha256',root/'overnight_c1_v2/matched_subsets_v2_PRIVATE.json'),('mask_receipt_sha256',root/'overnight_c2_masks_v2/receipt_PRIVATE.json')]:
        if sha(path)!=scope[name]:raise ValueError('Frozen source changed')
    return scope,root


def fetch(out):
    scope,_=verify_scope(out)
    if (out/'download_PRIVATE.json').exists():raise FileExistsError('Download receipt already exists')
    fetched=0
    for item in scope['files_PRIVATE']:
        path=Path(item['local_path_PRIVATE'])
        if path.exists():
            if path.stat().st_size!=item['bytes'] or sha(path)!=item['sha256']:raise ValueError('Existing payload differs; preserve for investigation')
            continue
        if item['reused']:raise ValueError('Reused file disappeared')
        digest=hashlib.sha256();total=0
        with urllib.request.urlopen(item['source_url'],timeout=45) as response,path.open('xb') as f:
            while True:
                block=response.read(65536)
                if not block:break
                total+=len(block)
                if total>item['bytes']:raise ValueError('Per-file byte cap exceeded')
                f.write(block);digest.update(block)
            f.flush();os.fsync(f.fileno())
        if total!=item['bytes'] or digest.hexdigest()!=item['sha256']:raise ValueError('Downloaded SHA256/size mismatch')
        fetched+=total
        with (out/'download_progress_PRIVATE.jsonl').open('a') as f:f.write(json.dumps({'sha256':item['sha256'],'bytes':total})+'\n');f.flush();os.fsync(f.fileno())
    receipt={'status':'IMAGE_PAYLOADS_VERIFIED','scope_sha256':sha(out/'scope_PRIVATE.json'),'new_patients':scope['new_patients'],
        'verified_image_files':len(scope['files_PRIVATE']),'total_verified_bytes':scope['image_bytes'],'new_download_bytes_this_call':fetched,
        'rendering':'NOT_RUN','task_admission':'ENGINEERING_ONLY'}
    write_json(out/'download_PRIVATE.json',receipt);print(json.dumps(receipt))


def render(out):
    import numpy as np
    import nibabel as nib
    from PIL import Image,ImageDraw
    scope,root=verify_scope(out)
    if (out/'render_PRIVATE.json').exists():raise FileExistsError('Render exists')
    download=json.loads((out/'download_PRIVATE.json').read_text())
    if download['scope_sha256']!=sha(out/'scope_PRIVATE.json'):raise ValueError('Download scope mismatch')
    combined=json.loads((root/'overnight_c2_masks_v2/combined_geometry_PRIVATE.json').read_text())
    masks=json.loads((root/'overnight_c2_masks_v2/receipt_PRIVATE.json').read_text())['checks_PRIVATE']
    mask_by_subject={c['subject_PRIVATE']:c for c in masks};panels_by_case={};geometry=[]
    for case in scope['chosen_PRIVATE']:
        patient=case['patient'];subject=case['subject_PRIVATE']
        rows=[r for v in combined['categories'].values() for r in v['selected_PRIVATE'] if r['patient']==patient];levels=sorted({r['level'] for r in rows})
        mask_info=mask_by_subject[subject];maskpath=root/'overnight_c2_masks_v2'/Path(mask_info['source_url']).name
        if sha(maskpath)!=mask_info['sha256']:raise ValueError('Source mask changed')
        mask=nib.load(str(maskpath));panels={z:[] for z in levels}
        for item in [x for x in scope['files_PRIVATE'] if x['patient']==patient]:
            path=Path(item['local_path_PRIVATE'])
            if path.stat().st_size!=item['bytes'] or sha(path)!=item['sha256']:raise ValueError('Image payload changed')
            image=nib.load(str(path))
            if image.shape!=mask.shape or not np.allclose(image.affine,mask.affine,rtol=0,atol=1e-6):raise ValueError('Image-mask grid mismatch')
            if np.prod(image.shape)*max(8,image.get_data_dtype().itemsize)>64*1024*1024:raise ValueError('Volume allocation cap exceeded')
            volume=np.asanyarray(image.dataobj)
            if not np.isfinite(volume).all():raise ValueError('Nonfinite MRI')
            nonzero=volume[volume!=0][::16];lo,hi=np.percentile(nonzero,[0.5,99.5])
            if not hi>lo:raise ValueError('Degenerate intensities')
            kind='flair' if '_flair_' in path.name else 't1ce'
            for z in levels:
                plane=np.rot90(volume[:,:,z]);pixels=np.uint8(np.round(np.clip((plane-lo)/(hi-lo),0,1)*255))
                pic=Image.fromarray(pixels).resize((364,436),Image.Resampling.BILINEAR).convert('RGB');panels[z].append((kind,pic))
                del plane,pixels
            geometry.append({'patient':patient,'modality':kind,'shape':list(image.shape),'orientation':''.join(nib.aff2axcodes(image.affine)),
                'spacing_mm':list(map(float,image.header.get_zooms()[:3])),'same_mask_grid':True,'finite':True,
                'image_sha256':item['sha256'],'normalization_percentiles':[float(lo),float(hi)],'array_bytes':volume.nbytes})
            del volume,nonzero,image;gc.collect()
        for z,parts in panels.items():
            if [k for k,_ in parts]!=['flair','t1ce']:raise ValueError('Panel ordering changed')
            canvas=Image.new('RGB',(752,470),'black');canvas.paste(parts[0][1],(4,28));canvas.paste(parts[1][1],(384,28));draw=ImageDraw.Draw(canvas)
            draw.text((8,7),'FLAIR',fill='white');draw.text((388,7),'contrast-enhanced T1',fill='white')
            path=out/f'{patient}_z{z:03d}_paired.png';canvas.save(path);panels_by_case[patient,z]=path
        del panels,mask;gc.collect()
    training={r['id']:r for r in map(json.loads,(root/'vero_training_v1/train.verl.jsonl').read_text().splitlines())}
    cohort=[]
    for task,item in combined['categories'].items():
        for row in item['selected_PRIVATE']:
            is_new=row.get('MRI_image_status')=='NOT_DOWNLOADED_OR_RENDERED'
            image=panels_by_case[row['patient'],row['level']] if is_new else Path(training[row['record_id']]['images'][0])
            digest=sha(image)
            if not is_new and digest!=row['image_sha256']:raise ValueError('Existing MRI changed')
            cohort.append({**row,'image_PRIVATE':str(image.resolve()),'image_sha256':digest,'MRI_image_status':'IMAGE_LINKED_GEOMETRY_VERIFIED',
                'task_admission':'ENGINEERING_ONLY','visual_answerability':'NOT_CLINICALLY_ADJUDICATED','new_source_patient':is_new})
    receipt={'status':'FROZEN_COHORT_IMAGE_LINKED','tag':'C_TASK_ELIGIBILITY','scope_sha256':sha(out/'scope_PRIVATE.json'),
        'download_receipt_sha256':sha(out/'download_PRIVATE.json'),'new_source_patients':10,'verified_volumes':len(geometry),
        'new_paired_frames':len(panels_by_case),'task_rows':len(cohort),'category_rows':{k:v['n'] for k,v in combined['categories'].items()},
        'geometry_PRIVATE':geometry,'rows_PRIVATE':cohort,'task_admission':'ENGINEERING_ONLY','clinical_visibility':'NOT_VERIFIED','C_training_run':False}
    write_json(out/'render_PRIVATE.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items() if not k.endswith('_PRIVATE')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['plan','fetch','render']);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    private=Path(__file__).resolve().parents[1]/'local_data'
    if not a.output.resolve().is_relative_to(private):raise ValueError('Private output required')
    if a.stage=='plan':plan(a.root.resolve(),a.output.resolve())
    elif a.stage=='fetch':fetch(a.output.resolve())
    else:render(a.output.resolve())
