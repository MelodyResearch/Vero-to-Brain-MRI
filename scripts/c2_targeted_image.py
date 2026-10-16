"""One C2 upper-level real MRI pair: separate fetch and GPU-idle render stages."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def select_one(checks,existing,reserved):
    eligible=[c for c in checks if c['levels']['136']['tumor']>=1000 and c['subject_PRIVATE'] not in set(existing)|set(reserved)]
    if not eligible:raise ValueError('No new eligible upper-level training subject')
    return min(eligible,key=lambda c:hashlib.sha256(('C2-upperlevel-pair-v1:'+c['subject_PRIVATE']).encode()).hexdigest())


def fetch(root,output):
    if output.exists():raise ValueError('New output directory required')
    mask_path=root/'overnight_c2_masks_v2/receipt_PRIVATE.json';selection_path=root/'overnight_c1_v2/matched_subsets_v2_PRIVATE.json'
    masks=json.loads(mask_path.read_text());selection=json.loads(selection_path.read_text())['C2']
    existing={c['private_subject'] for c in json.loads((root/'visual_prompts/private_manifest.json').read_text())['private_cases']}
    reserved={c['subject_PRIVATE'] for c in json.loads((root/'balanced_sft_independent_audit_v1/prospective_metadata_selection_PRIVATE.json').read_text())['selected_PRIVATE']}
    chosen=select_one(masks['checks_PRIVATE'],existing,reserved)
    candidate=next(c for c in selection['selected_PRIVATE'] if c['subject_PRIVATE']==chosen['subject_PRIVATE']);items=candidate['payloads'][:2]
    if len(items)!=2 or sum(i['bytes'] for i in items)>25_000_000:raise ValueError('Two-image25MBscope exceeded')
    output.mkdir();files=[]
    for item in items:
        url='https://s3.amazonaws.com/openneuro.org/ds007045/'+item['path'];path=output/Path(item['path']).name
        with urllib.request.urlopen(url,timeout=45) as response, path.open('xb') as f:
            total=0;digest=hashlib.sha256()
            while True:
                b=response.read(65536)
                if not b:break
                total+=len(b)
                if total>item['bytes']:raise ValueError('Source byte budget exceeded')
                f.write(b);digest.update(b)
        if total!=item['bytes'] or digest.hexdigest()!=item['sha256']:raise ValueError('Source checksum mismatch')
        files.append({**item,'local_name':path.name,'source_url':url,'actual_bytes':total})
    receipt={'tag':'C_TASK_ELIGIBILITY','status':'IMAGE_BYTES_VERIFIED_RENDER_PENDING','source_commit':selection['source_commit'],
        'source_code_sha256':sha(__file__),'mask_receipt_sha256':sha(mask_path),'selection_sha256':sha(selection_path),
        'selection_rule':'SHA256 C2-upperlevel-pair-v1 tie-break among new12patients withz136tumor≥1000; exclude40+14',
        'subject_PRIVATE':chosen['subject_PRIVATE'],'site':chosen['site'],'z':136,'mask_geometry':chosen,
        'mask_path_PRIVATE':str((root/'overnight_c2_masks_v2'/Path(chosen['source_url']).name).resolve()),
        'files':files,'bytes_verified':sum(i['actual_bytes'] for i in files),'clinical_visibility':'NOT_VERIFIED',
        'mission_alignment':{'main_question':'Unblock image-grounded upper-level task eligibility for later Vero MRI recipe transfer.',
        'claim_and_falsifier':'New geometry-eligible mask can be linked to real MRI; reject mismatched hashes/affines or ambiguous visible evidence.',
        'change_isolation':'One new training-source patient; frozen A/B and reserved evaluation cohorts unchanged.',
        'evidence_provenance':'Pinned official annex SHA256 and source identities; no synthetic images.',
        'shortcut_baseline':'Upper-level support only; source/site/lesion-size shortcuts and blinded visibility audit remain.',
        'resources_decision':'Twofiles≤25MB; serial download; render onevolume atatime only duringGPUidle; no taskadmission from geometry.'}}
    (output/'download_PRIVATE.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'status':receipt['status'],'images':2,'bytes_verified':receipt['bytes_verified']}))


def render(output):
    import gc
    import numpy as np
    import nibabel as nib
    from PIL import Image,ImageDraw
    if (output/'render_PRIVATE.json').exists():raise ValueError('Render receipt exists')
    receipt=json.loads((output/'download_PRIVATE.json').read_text());mask_path=Path(receipt['mask_path_PRIVATE'])
    if sha(mask_path)!=receipt['mask_geometry']['sha256']:raise ValueError('Mask changed')
    mask=nib.load(str(mask_path));panels=[];geometry=[]
    for item in receipt['files']:
        path=output/item['local_name']
        if path.stat().st_size!=item['bytes'] or sha(path)!=item['sha256']:raise ValueError('Image payload changed')
        image=nib.load(str(path))
        if image.shape!=mask.shape or not np.allclose(image.affine,mask.affine,rtol=0,atol=1e-6):raise ValueError('Image-mask grid mismatch')
        if np.prod(image.shape)*max(8,image.get_data_dtype().itemsize)>64*1024*1024:raise ValueError('Volume allocation budget exceeded')
        volume=np.asanyarray(image.dataobj)
        if not np.isfinite(volume).all():raise ValueError('Nonfinite MRI')
        nonzero=volume[volume!=0][::16];lo,hi=np.percentile(nonzero,[0.5,99.5])
        if not hi>lo:raise ValueError('Degenerate intensity range')
        plane=np.rot90(volume[:,:,136]);pixels=np.uint8(np.round(np.clip((plane-lo)/(hi-lo),0,1)*255))
        panel=Image.fromarray(pixels).resize((pixels.shape[1]*2,pixels.shape[0]*2),Image.Resampling.BILINEAR).convert('RGB')
        name='flair' if '_flair_' in item['local_name'] else 't1ce';panel.save(output/f'upper_z136_{name}.png');panels.append(panel)
        geometry.append({'modality':name,'shape':list(image.shape),'orientation':''.join(nib.aff2axcodes(image.affine)),
            'spacing_mm':list(map(float,image.header.get_zooms()[:3])),'affine':image.affine.tolist(),'same_mask_grid':True,'finite':True,
            'normalization_percentiles':[float(lo),float(hi)],'volume_array_bytes':volume.nbytes})
        del nonzero,volume,plane,pixels,image;gc.collect()
    if [g['modality'] for g in geometry]!=['flair','t1ce']:raise ValueError('Channel order changed')
    canvas=Image.new('RGB',(752,470),'black');canvas.paste(panels[0],(4,28));canvas.paste(panels[1],(384,28));draw=ImageDraw.Draw(canvas)
    draw.text((8,7),'FLAIR',fill='white');draw.text((388,7),'contrast-enhanced T1',fill='white');path=output/'upper_z136_paired.png';canvas.save(path)
    result={'status':'IMAGE_LINKAGE_AND_RENDER_VERIFIED','tag':'C_TASK_ELIGIBILITY','download_receipt_sha256':sha(output/'download_PRIVATE.json'),
        'source_code_sha256':sha(__file__),'images':2,'patients':1,'geometry':geometry,'paired_png_sha256':sha(path),
        'render_policy':'Historical np.rot90,2xbilinear,nonzero-volume[::16]0.5/99.5percentile; modalitytitlesonly; no maskoverlay.',
        'clinical_visibility':'NOT_VERIFIED','task_admission':'ENGINEERING_ONLY','new_training_run':False}
    (output/'render_PRIVATE.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['fetch','render']);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    private=Path(__file__).resolve().parents[1]/'local_data'
    if not a.output.resolve().is_relative_to(private):raise ValueError('Private output required')
    if a.stage=='fetch':fetch(a.root,a.output)
    else:render(a.output)
