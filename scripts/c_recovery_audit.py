"""Training-only C recovery: bounded mask geometry and site-held-out PNG probes."""
import argparse
from collections import Counter, defaultdict, deque
import gc
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'local_data/ds007045_bounded'


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write_new(path, data):
    with path.open('x') as f:json.dump(data,f,indent=2);f.write('\n')


def display_xy(i,j,shape,panel=0):
    """Source voxel center -> rot90, 2x pixel-center transform, paired canvas."""
    return (4 if panel==0 else 384)+2*i+0.5,28+2*(shape[1]-1-j)+0.5


def grid(x,y,width,height):
    return ['left','center','right'][min(2,int(x//(width/3)))]+ '-' + ['top','middle','bottom'][min(2,int(y//(height/3)))]


def spatial_summary(rotated):
    positive=np.asarray(rotated,dtype=bool);height,width=positive.shape
    coords=np.argwhere(positive)
    if not len(coords):return {'area':0,'components':0}
    cy,cx=coords.mean(axis=0);seen=np.zeros_like(positive);components=[]
    for y,x in coords:
        if seen[y,x]:continue
        q=deque([(int(y),int(x))]);seen[y,x]=True;part=[]
        while q:
            yy,xx=q.popleft();part.append((yy,xx))
            for dy in (-1,0,1):
                for dx in (-1,0,1):
                    ny,nx=yy+dy,xx+dx
                    if 0<=ny<height and 0<=nx<width and positive[ny,nx] and not seen[ny,nx]:
                        seen[ny,nx]=True;q.append((ny,nx))
        components.append(part)
    largest=max(components,key=len)  # row-major deterministic tie
    ly,lx=np.asarray(largest).mean(axis=0)
    cells=Counter(grid(float(x),float(y),width,height) for y,x in coords)
    legacy=grid(cx,cy,width,height);center=grid(cx+0.25,cy+0.25,width,height)
    return {'area':len(coords),'components':len(components),'largest_component_fraction':len(largest)/len(coords),
            'centroid_xy_panel_legacy':[float(cx*2),float(cy*2)],
            'centroid_xy_panel_pixelcenter':[float(cx*2+.5),float(cy*2+.5)],
            'centroid_on_mask':bool(positive[int(np.rint(cy)),int(np.rint(cx))]),
            'global_grid':legacy,'pixelcenter_grid':center,'largest_component_grid':grid(lx,ly,width,height),
            'boundary_distance_display_pixels':float(min(abs(cx-width/3),abs(cx-2*width/3),abs(cy-height/3),abs(cy-2*height/3))*2),
            'occupied_grid_cells':len(cells),'cell_counts':dict(cells),'majority_cell_fraction':max(cells.values())/len(coords),
            'largest_component_centroid_on_mask':bool(positive[int(np.rint(ly)),int(np.rint(lx))])}


def heldout_neighbors(rows,features):
    result=[]
    for i,r in enumerate(rows):
        candidates=[j for j,s in enumerate(rows) if s['site']!=r['site'] and s['patient']!=r['patient']]
        if not candidates:raise ValueError('No eligible cross-site patient neighbor')
        result.append(min(candidates,key=lambda j:(float(np.sum((features[i]-features[j])**2)),rows[j]['record_id'])))
    return result


def load_inputs():
    train=[json.loads(l) for l in (BASE/'vero_training_v1/train.verl.jsonl').read_text().splitlines()]
    patients={r['extra_info']['patient_key'] for r in train};assert len(patients)==28
    cases=json.loads((BASE/'visual_prompts/private_manifest.json').read_text())['private_cases']
    old={c['anonymous_token']:{'subject':c['private_subject'],'site':c['private_site']} for c in cases if c['anonymous_token'] in patients}
    c2scope=json.loads((BASE/'overnight_c2_frozen_cohort_v1/scope_PRIVATE.json').read_text())
    new={c['patient']:{'subject':c['subject_PRIVATE'],'site':c['site']} for c in c2scope['chosen_PRIVATE']}
    reserved={c['subject_PRIVATE'] for c in json.loads((BASE/'balanced_sft_independent_audit_v1/prospective_metadata_selection_PRIVATE.json').read_text())['selected_PRIVATE']}
    assert len(reserved)==14 and not reserved & {v['subject'] for v in [*old.values(),*new.values()]}
    assert not {v['subject'] for v in new.values()} & {c['private_subject'] for c in cases}
    cohort=json.loads((BASE/'overnight_c2_frozen_cohort_v1/render_PRIVATE.json').read_text())['rows_PRIVATE']
    for r in cohort:assert r['patient'] in old or r['patient'] in new
    return train,old,new,cohort


def mask_audit(out):
    # Only explicitly GPU-idle execution; one small mask plane at a time, no MRI intensities.
    import nibabel as nib
    train,old,new,cohort=load_inputs();mapping={**old,**new}
    masks=json.loads((BASE/'overnight_c2_masks_v2/receipt_PRIVATE.json').read_text())['checks_PRIVATE']
    newmasks={c['subject_PRIVATE']:c for c in masks}
    oldhash={r['patient']:r['mask_sha256'] for r in json.loads((BASE/'safe_resume_v1/historical112_sampling_v2_PRIVATE.json').read_text())['rows_PRIVATE']}
    targets={(r['patient'],r['level']):r for r in cohort}
    # Add existing train-only spatial candidates; same mask files, no new patients.
    candidate=json.loads((BASE/'safe_resume_v1/slice_label_candidate_v1_PRIVATE.json').read_text())['rows_PRIVATE']
    selected_grid={r['record_id']:r for r in candidate if r['category']=='mri_tumor_grid'}
    refs=json.loads((BASE/'safe_resume_v1/historical112_sampling_v2_PRIVATE.json').read_text())['training_references_PRIVATE']
    spatial_refs=[r for r in refs if r['category']=='mri_tumor_grid']
    for r in spatial_refs:targets.setdefault((r['patient'],r['level']),r)
    bypatient=defaultdict(list)
    for patient,z in targets:bypatient[patient].append(z)
    output=[]
    for patient,levels in sorted(bypatient.items()):
        subject=mapping[patient]['subject']
        if patient in old:path=BASE/subject/f'{subject}_MNI152_tumor-mask.nii.gz';expected=oldhash[patient]
        else:
            check=newmasks[subject];path=BASE/'overnight_c2_masks_v2'/Path(check['source_url']).name;expected=check['sha256']
        if sha(path)!=expected:raise ValueError('Mask bytes drift')
        mask=nib.load(str(path),mmap=True)
        if mask.shape!=(182,218,182) or ''.join(nib.aff2axcodes(mask.affine))!='LAS':raise ValueError('Mask grid drift')
        for z in sorted(set(levels)):
            plane=np.asanyarray(mask.dataobj[:,:,z]);rot=np.rot90(plane)
            r={'patient':patient,'site':mapping[patient]['site'],'level':z,'mask_sha256':expected,
               'tumor':spatial_summary(rot>0),'enhancing':spatial_summary(rot==3),
               'in_C2':any(x['patient']==patient and x['level']==z for x in cohort),
               'in_grid_candidate':any(x['patient']==patient and x['level']==z for x in selected_grid.values())}
            assert np.count_nonzero((rot==3)&~(rot>0))==0
            ref=next((x for x in spatial_refs if x['patient']==patient and x['level']==z),None)
            if ref and r['tumor']['global_grid']!=ref['label']:raise ValueError('Existing spatial label mismatch')
            original=targets[patient,z]
            if 'tumor' in original and int(original['tumor'])!=r['tumor']['area']:raise ValueError('Area mismatch')
            if 'enhancing' in original and int(original['enhancing'])!=r['enhancing']['area']:raise ValueError('Enhancing area mismatch')
            output.append(r);del plane,rot
        del mask;gc.collect()
    groups={}
    for name,rows in [('all_planes',output),('C2_planes',[r for r in output if r['in_C2']]),('grid_candidates',[r for r in output if r['in_grid_candidate']])]:
        positive=[r['tumor'] for r in rows if r['tumor']['area']>=1000]
        groups[name]={'planes':len(rows),'patients':len({r['patient'] for r in rows}),'tumor_positive_ge1000':len(positive),
          'centroid_off_mask':sum(not r['centroid_on_mask'] for r in positive),
          'global_vs_largest_grid_disagree':sum(r['global_grid']!=r['largest_component_grid'] for r in positive),
          'less_than_half_mask_in_centroid_cell':sum(r['cell_counts'][r['global_grid']]<r['area']/2 if r['global_grid'] in r['cell_counts'] else True for r in positive),
          'boundary_under10':sum(r['boundary_distance_display_pixels']<10 for r in positive),
          'pixelcenter_vs_legacy_grid_disagree':sum(r['global_grid']!=r['pixelcenter_grid'] for r in positive)}
    result={'status':'MASK_GEOMETRY_AUDITED','source_sha256':sha(__file__),'clinical_visibility':'NOT_VERIFIED','threshold':1000,'groups':groups,'rows_PRIVATE':output}
    write_new(out/'masks_PRIVATE.json',result);print(json.dumps(groups,indent=2))



def pixel_audit(out):
    import importlib.util
    from PIL import Image
    spec=importlib.util.spec_from_file_location('c2features',ROOT/'scripts/c2_image_shortcut_baseline.py')
    features=importlib.util.module_from_spec(spec);spec.loader.exec_module(features)
    train,old,new,cohort=load_inputs();mapping={**old,**new}
    refs=json.loads((BASE/'safe_resume_v1/historical112_sampling_v2_PRIVATE.json').read_text())
    old_ids={r['record_id'] for r in refs['rows_PRIVATE']}
    bref=json.loads((BASE/'balanced_sft_resume_step028_v1/receipt.json').read_text())['examples']
    new_ids={r['record_id'] for r in bref}
    reference_hash={r['record_id']:r['image_sha256'] for r in refs['training_references_PRIVATE']}
    sequence=[]
    for r in train:
        if r['ability']!='mri_sequence':continue
        patient=r['extra_info']['patient_key'];level=int(r['id'].split('_z')[1][:3]);path=Path(r['images'][0])
        expected=reference_hash[r['id']]
        if sha(path)!=expected:raise ValueError('Sequence image drift')
        with Image.open(path) as im:
            if im.size!=(364,436):raise ValueError('Single-frame dimensions drift')
        sequence.append({'record_id':r['id'],'patient':patient,'site':old[patient]['site'],'level':level,
            'label':r['reward_model']['ground_truth'],'image_PRIVATE':str(path),'image_sha256':expected,
            'A28_directly_supervised_reconstructed':r['id'] in old_ids,'B28_directly_supervised':r['id'] in new_ids})
    assert len(sequence)==168
    pairs=defaultdict(list)
    for r in sequence:pairs[r['patient'],r['level']].append(r)
    assert len(pairs)==84 and all(Counter(r['label'] for r in pair)=={'FLAIR':1,'contrast-enhanced T1':1} for pair in pairs.values())
    write_new(out/'sequence_inventory_PRIVATE.json',{'status':'PNG_READY','rows_PRIVATE':sequence,'source_train_sha256':sha(BASE/'vero_training_v1/train.verl.jsonl'),
      'source_A28_reconstruction_sha256':sha(BASE/'safe_resume_v1/historical112_sampling_v2_PRIVATE.json'),
      'source_B28_receipt_sha256':sha(BASE/'balanced_sft_resume_step028_v1/receipt.json'),
      'counts':{str(z):{'rows':sum(r['level']==z for r in sequence),'patients':len({r['patient'] for r in sequence if r['level']==z}),
        'A_supervised':sum(r['level']==z and r['A28_directly_supervised_reconstructed'] for r in sequence),
        'B_supervised':sum(r['level']==z and r['B28_directly_supervised'] for r in sequence)} for z in (45,91,136)}})
    groups={category:[{**r,'site':mapping[r['patient']]['site']} for r in cohort if r['category']==category] for category in sorted({r['category'] for r in cohort})}
    groups.update({f'sequence_z{z}':[r for r in sequence if r['level']==z] for z in (45,91,136)})
    result={'status':'TESTED_CPU','method':'prespecified leave-one-site-out1NN and training-fold majority; no tuning',
            'source_sha256':sha(__file__),'original_feature_source_sha256':sha(ROOT/'scripts/c2_image_shortcut_baseline.py'),
            'cohort_sha256':sha(BASE/'overnight_c2_frozen_cohort_v1/render_PRIVATE.json'),
            'sequence_inventory_sha256':sha(out/'sequence_inventory_PRIVATE.json'),'groups':{}}
    for name,rows in groups.items():
        for patient in {r['patient'] for r in rows}:assert len({r['site'] for r in rows if r['patient']==patient})==1
        pix=[features.extract_features(Path(r['image_PRIVATE']),r['image_sha256']) for r in rows]
        methods={}
        for mode in ('thumbnail16','histogram16','training_fold_majority','training_fold_level_majority'):
            predictions=[]
            if mode in ('thumbnail16','histogram16'):
                neighbors=heldout_neighbors(rows,np.array([p[0 if mode=='thumbnail16' else 1] for p in pix]))
            for i,r in enumerate(rows):
                fitting=[x for x in rows if x['site']!=r['site'] and x['patient']!=r['patient']]
                if mode=='training_fold_level_majority':
                    level=[x for x in fitting if x['level']==r['level']]
                    fitting=level or fitting
                if mode.startswith('training_'):
                    counts=Counter(x['label'] for x in fitting);guess=min(counts,key=lambda x:(-counts[x],x));neighbor=None
                else:neighbor=rows[neighbors[i]];guess=neighbor['label']
                predictions.append({'record_id':r['record_id'],'patient':r['patient'],'site':r['site'],'gold':r['label'],'prediction':guess,
                    'neighbor_record_id':neighbor['record_id'] if neighbor else None,'neighbor_patient':neighbor['patient'] if neighbor else None,
                    'neighbor_site':neighbor['site'] if neighbor else None})
            bysite={site:{'n':sum(r['site']==site for r in predictions),'correct':sum(r['site']==site and r['gold']==r['prediction'] for r in predictions)} for site in sorted({r['site'] for r in rows})}
            bypatient={patient:{'n':sum(r['patient']==patient for r in predictions),'correct':sum(r['patient']==patient and r['gold']==r['prediction'] for r in predictions)} for patient in sorted({r['patient'] for r in rows})}
            methods[mode]={'n':len(rows),'correct':sum(r['gold']==r['prediction'] for r in predictions),'site_macro':sum(v['correct']/v['n'] for v in bysite.values())/len(bysite),
                'patient_macro':sum(v['correct']/v['n'] for v in bypatient.values())/len(bypatient),'by_site':bysite,'predictions_PRIVATE':predictions}
        result['groups'][name]={'patients':len({r['patient'] for r in rows}),'sites':len({r['site'] for r in rows}),'methods':methods}
    write_new(out/'site_baselines_PRIVATE.json',result)
    print(json.dumps({g:{m:{k:v for k,v in d.items() if k not in ('predictions_PRIVATE','by_site')} for m,d in data['methods'].items()} for g,data in result['groups'].items()},indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['masks','pixels']);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if not a.output.resolve().is_relative_to(ROOT/'local_data'):raise ValueError('Private output required')
    a.output.mkdir(exist_ok=True)
    if a.stage=='masks':mask_audit(a.output)
    else:pixel_audit(a.output)


if __name__=='__main__':main()
