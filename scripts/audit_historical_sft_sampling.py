"""Audit all historical112 examples with one source mask proxy at a time.

No model imports, no full cohort arrays. Original record identity is reconstructed
from frozen scheduler and verified against historical step/category/patient tuples.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from statistics import median
from mri_rl.mlx_sft_warmstart import _schedule_train_rows
from mri_rl.sft_evidence import verify_history


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def summarize(rows):
    result={}
    for category in sorted({r['category'] for r in rows}):
        group=[r for r in rows if r['category']==category]
        areas=sorted(r['mask_area'] for r in group);positive=[v for v in areas if v>0]
        counts=Counter(r['label'] for r in group)
        result[category]={'n':len(group),'labels':dict(counts),'levels':dict(Counter(r['level'] for r in group)),
            'distinct_patients':len({r['patient'] for r in group}),
            'positive_fraction':counts.get('yes',0)/len(group) if set(counts)<= {'yes','no'} else None,
            'distinct_images':len({r['image_sha256'] for r in group}),
            'image_reuse_excess':len(group)-len({r['image_sha256'] for r in group}),
            'distinct_targets':len({r['target'] for r in group}),
            'target_reuse_excess':len(group)-len({r['target'] for r in group}),
            'mask_area':{'min':min(areas),'median':median(areas),'max':max(areas),
                         'zero_count':areas.count(0),'positive_min':min(positive) if positive else None,
                         'positive_median':median(positive) if positive else None,
                         'positive_le_100':sum(0<v<=100 for v in areas)}}
    return result


def audit(root, output):
    import nibabel as nib
    import numpy as np
    private=Path(__file__).resolve().parents[1]/'local_data'
    if not output.resolve().is_relative_to(private):raise ValueError('Private audit output required')
    if output.exists():raise FileExistsError(output)
    train=root/'vero_training_v1/train.verl.jsonl'
    schedule=_schedule_train_rows(train,28,7,'legacy_lexical_v1')
    receipt=json.loads((root/'sft_warmstart_28step_v1/receipt.json').read_text())
    verify_history(receipt,schedule)
    cases=json.loads((root/'visual_prompts/private_manifest.json').read_text())['private_cases']
    mapping={c['anonymous_token']:c['private_subject'] for c in cases}
    rows=[]; references=[]; selected=defaultdict(list)
    training=[json.loads(line) for line in train.read_text().splitlines() if line.strip()]
    for step,batch in enumerate(schedule):
        for row in batch:selected[row['extra_info']['patient_key']].append((step,row))
    for patient, batch in selected.items():
        subject=mapping[patient]
        path=root/subject/f'{subject}_MNI152_tumor-mask.nii.gz'
        image=nib.load(str(path),mmap=True)
        if len(image.shape)!=3:raise ValueError('Mask must be3D')
        slices={}
        for row in [r for r in training if r['extra_info']['patient_key']==patient]:
            z=int(re.search(r'_z(\d+)_',row['id']).group(1))
            if z not in slices:
                plane=np.asanyarray(image.dataobj[:,:,z])
                coords=np.argwhere(np.rot90(plane)>0)
                boundary=None
                if coords.size:
                    cy,cx=coords.mean(axis=0);height,width=np.rot90(plane).shape
                    boundary=float(min(abs(cx-width/3),abs(cx-2*width/3),abs(cy-height/3),abs(cy-2*height/3))*2)
                slices[z]={'tumor':int(np.count_nonzero(plane>0)), 'enhancing':int(np.count_nonzero(plane==3)), 'grid_boundary_distance_display_pixels':boundary}
                del plane
        orientation=''.join(nib.aff2axcodes(image.affine))
        for row in [r for r in training if r['extra_info']['patient_key']==patient]:
            z=int(re.search(r'_z(\d+)_',row['id']).group(1))
            references.append({'record_id':row['id'],'patient':patient,'category':row['ability'],
                'label':row['reward_model']['ground_truth'],'level':z,'orientation':orientation,**slices[z],
                'image_sha256':sha(row['images'][0]),'question_sha256':hashlib.sha256(row['extra_info']['question'].encode()).hexdigest(),
                'visual_answerability':'SOURCE_LABEL_VERIFIED_VISIBILITY_NOT_ADJUDICATED'})
        for step,row in batch:
            z=int(re.search(r'_z(\d+)_',row['id']).group(1))
            area=slices[z]['enhancing' if row['ability']=='mri_enhancing_presence' else 'tumor']
            label=row['reward_model']['ground_truth']
            if row['ability'] in ('mri_tumor_presence','mri_enhancing_presence') and label!=('yes' if area else 'no'):
                raise ValueError('Mask-derived label differs')
            rows.append({'step':step,'record_id':row['id'],'category':row['ability'],'label':label,
                'level':z,'patient':patient,'image_sha256':sha(row['images'][0]),'mask_sha256':sha(path),
                'target':f'<think>Visible MRI evidence.</think><answer>\\boxed{{{label}}}</answer>',
                'mask_area':area,'tumor_area':slices[z]['tumor'],'enhancing_area':slices[z]['enhancing'],
                'orientation':orientation,'question_sha256':hashlib.sha256(row['extra_info']['question'].encode()).hexdigest(),
                'visual_answerability':'SOURCE_LABEL_VERIFIED_VISIBILITY_NOT_ADJUDICATED','area_reference':'enhancing_pixels' if row['ability']=='mri_enhancing_presence' else 'tumor_pixels'})
        del image,slices
    rows.sort(key=lambda r:(r['step'],r['category']))
    report={'historical_receipt_sha256':sha(root/'sft_warmstart_28step_v1/receipt.json'),
            'train_sha256':sha(train),'record_identity':'reconstructed_not_originally_logged',
            'examples':len(rows),'patients':len(selected),'by_category':summarize(rows),
            'cross_category_unique_images':len({r['image_sha256'] for r in rows}),
            'cross_category_image_reuse_excess':len(rows)-len({r['image_sha256'] for r in rows}),
            'repeated_image_question_excess':len(rows)-len({(r['image_sha256'],r['question_sha256']) for r in rows}),
            'orientations':dict(Counter(r['orientation'] for r in rows)),
            'rows_PRIVATE':rows,'training_references_PRIVATE':references}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if not k.endswith('_PRIVATE')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();audit(a.root,a.output)
