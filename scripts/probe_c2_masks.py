"""Bounded official-public C2 mask-only probe; all source subjects stay private."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import urllib.request


def verify_payload(blob,item):
    if len(blob)!=item['bytes'] or hashlib.sha256(blob).hexdigest()!=item['sha256']:
        raise ValueError('Source payload size or SHA256 mismatch')


def verify_candidates(selected,existing,reserved,count=12,bytes_total=318016):
    subjects=[c['subject_PRIVATE'] for c in selected]
    if len(subjects)!=len(set(subjects)) or len(subjects)!=count:raise ValueError('Candidate inventory changed')
    if set(subjects)&(set(existing)|set(reserved)):raise ValueError('Reserved/existing patient contamination')
    if sum(c['payloads'][2]['bytes'] for c in selected)!=bytes_total:raise ValueError('Mask-only byte budget changed')


def main():
    import nibabel as nib
    import numpy as np
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--selection',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    private=Path(__file__).resolve().parents[1]/'local_data'
    if not a.output_dir.resolve().is_relative_to(private) or a.output_dir.exists():raise ValueError('New private directory required')
    source=json.loads(a.selection.read_text());selected=source['C2']['selected_PRIVATE']
    cases=json.loads((a.root/'visual_prompts/private_manifest.json').read_text())['private_cases']
    reserved=json.loads((a.root/'balanced_sft_independent_audit_v1/prospective_metadata_selection_PRIVATE.json').read_text())['selected_PRIVATE']
    verify_candidates(selected,{c['private_subject'] for c in cases},{c['subject_PRIVATE'] for c in reserved})
    anchor=cases[0]['private_subject'];anchor=nib.load(str(a.root/anchor/f'{anchor}_MNI152_tumor-mask.nii.gz'))
    expected_shape=anchor.shape;expected_affine=anchor.affine.copy()
    a.output_dir.mkdir();checks=[]
    for c in selected:
        item=c['payloads'][2];url='https://s3.amazonaws.com/openneuro.org/ds007045/'+item['path']
        if not item['path'].startswith('derivatives/segmentation/') or not item['path'].endswith('_MNI152_tumor-mask.nii.gz'):raise ValueError('Mask-only scope')
        with urllib.request.urlopen(url,timeout=30) as response:blob=response.read(item['bytes']+1)
        verify_payload(blob,item);path=a.output_dir/Path(item['path']).name;path.write_bytes(blob)
        mask=nib.load(str(path))
        if mask.shape!=expected_shape or not np.allclose(mask.affine,expected_affine,rtol=0,atol=1e-6):raise ValueError('Unexpected source grid before array allocation')
        if np.prod(mask.shape)*max(8,mask.get_data_dtype().itemsize)>64*1024*1024:raise ValueError('Mask array budget exceeded')
        array=np.asanyarray(mask.dataobj)
        geometry={'shape':list(mask.shape),'orientation':''.join(nib.aff2axcodes(mask.affine)),
                  'spacing_mm':list(map(float,mask.header.get_zooms()[:3])),
                  'affine':mask.affine.tolist(),'same_grid_as_existing_source':bool(mask.shape==expected_shape and np.allclose(mask.affine,expected_affine,rtol=0,atol=1e-6)),
                  'finite':bool(np.isfinite(array).all()),'labels':np.unique(array).tolist(),'array_bytes':array.nbytes}
        if not geometry['same_grid_as_existing_source'] or not geometry['finite'] or not set(geometry['labels'])<={0,1,2,3}:raise ValueError('Mask geometry/label integrity failure')
        levels={str(z):{'tumor':int(np.count_nonzero(array[:,:,z]>0)),'enhancing':int(np.count_nonzero(array[:,:,z]==3))} for z in (45,91,136)}
        checks.append({'subject_PRIVATE':c['subject_PRIVATE'],'site':c['site'],'bytes':len(blob),'sha256':item['sha256'],
                       'source_url':url,'geometry':geometry,'levels':levels})
        del array,mask,blob
        with (a.output_dir/'progress_PRIVATE.jsonl').open('a') as f:f.write(json.dumps(checks[-1])+'\n')
    counts={category:{z:{'eligible_yes':sum(c['levels'][z][category]>=1000 for c in checks),
                         'zero_no':sum(c['levels'][z][category]==0 for c in checks),
                         'small_positive_excluded':sum(0<c['levels'][z][category]<1000 for c in checks)} for z in ('45','91','136')} for category in ('tumor','enhancing')}
    result={'status':'MASK_PAYLOAD_VERIFIED','source':'officialpublicOpenNeuroS3','source_commit':source['C2']['source_commit'],
            'selection_sha256':hashlib.sha256(a.selection.read_bytes()).hexdigest(),'source_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'subjects':len(checks),'bytes_verified':sum(c['bytes'] for c in checks),'positive_area_threshold':1000,
            'existing_and_reserved_exclusion_verified':True,'MRI_payload_downloaded':False,'MRI_visual_admission':False,
            'site_counts':dict(Counter(c['site'] for c in checks)),'counts':counts,'checks_PRIVATE':checks}
    (a.output_dir/'receipt_PRIVATE.json').write_text(json.dumps(result,indent=2)+'\n')
    result.pop('checks_PRIVATE');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
