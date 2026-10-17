"""Source-typed Dallas visit correspondence; images contain no metadata or gold.

This engineering candidate is NOT an admitted clinical or generalization task.
The frozen source equality target requires no change/registration biomarker.
All canonicalization and slice selection are independent per volume, never
conditioned on whether two source people are equal. Test pixels stay sealed.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

from mri_rl.vero_mri_reward import _answer_and_format

SOURCE = 'OpenNeuro_ds004856'
SOURCE_VERSION = 'OpenNeuro ds004856 v1.3.0'
ROUTE = 'mri_visit_correspondence'
RENDER = {'version': 'canonical-independent-multiview-v1', 'orientation': 'RAS',
          'axes': ['sagittal', 'coronal', 'axial'], 'fractions': [.35, .50, .65],
          'tile_pixels': 128, 'window_positive_percentiles': [1, 99],
          'crop': 'none', 'registration': 'none', 'pair_order': 'query-left_candidate-right'}
QUESTION = ('The left panel A and right panel B each show nine standardized T1 MRI views '
            '(rows: sagittal, coronal, axial; columns: 35%, 50%, 65% of the field of view). '
            'Do these MRI visits belong to the same person or different people? '
            'Compare anatomy across views. The images are from separate visits. '
            'Answer only same or different in the final \\boxed{...}.')


def sha_file(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def token(*parts):
    return hashlib.sha256(('ds004856-v1|' + '|'.join(parts)).encode()).hexdigest()[:24]


def _require(value, message):
    if not value:
        raise ValueError(message)


def build_cases(plan):
    """Derive equality labels from frozen source keys, keeping full cycles in-fold."""
    _require(plan['source'] == SOURCE_VERSION, 'Source/version mismatch')
    folds = plan['splits_PRIVATE']; people = plan['selected_people_PRIVATE']
    _require(set(folds) == {'train', 'internal_dev', 'test'}, 'Required split inventory')
    flat = [s for group in folds.values() for s in group]
    _require(len(flat) == len(set(flat)) == len(people) == plan['selected_people'], 'Patient split duplication or omission')
    _require(set(flat) == set(people), 'Patient inventory mismatch')
    hashes = []
    for s, person in people.items():
        images = person['images']
        _require(len(images) == 2, 'Exactly two visits required')
        for number, image in enumerate(images, 1):
            rel = Path(image['relative_path'])
            _require(not rel.is_absolute() and '..' not in rel.parts and rel.parts[0] == s
                     and rel.parts[1] == f'ses-wave{number}', 'Source visit lineage mismatch')
            _require(re.fullmatch('[0-9a-f]{64}', image['sha256']) is not None, 'Source hash malformed')
            hashes.append(image['sha256'])
    _require(len(hashes) == len(set(hashes)), 'Duplicate source image; same-image shortcut')
    _require(len(hashes) == plan['selected_expected_volumes'], 'Volume inventory mismatch')
    rows = {k: [] for k in folds}; covered = set()
    for cycle in plan['matched_cycles_PRIVATE']:
        split = cycle['split']; subjects = cycle['subjects']
        _require(split in folds and len(subjects) == 2 and len(set(subjects)) == 2
                 and set(subjects) <= set(folds[split]), 'Matched cycle crosses split')
        _require(not set(subjects) & covered, 'Duplicate cycle patient'); covered.update(subjects)
        a, b = (people[s] for s in subjects)
        _require(a['sex'] == b['sex'] and abs(a['age_wave1']-b['age_wave1']) <= 5, 'Negative demographic match failed')
        for first, second in ((a['images'][1], b['images'][0]), (b['images'][1], a['images'][0])):
            _require(first['manufacturer'] == second['manufacturer'] == 'Philips'
                     and all(0 < min(first[k], second[k]) and max(first[k], second[k])/min(first[k], second[k]) <= 1.03
                             for k in ('tr', 'te')), 'Negative protocol match failed')
        cluster = token('cycle', *sorted(subjects))
        for query_subject in subjects:
            for candidate_subject in subjects:
                query = people[query_subject]['images'][1]; candidate = people[candidate_subject]['images'][0]
                rows[split].append({'case_id': token('case', query['sha256'], candidate['sha256']),
                    'split': split, 'cluster_key': cluster,
                    'query_subject': query_subject, 'candidate_subject': candidate_subject,
                    'query': query, 'candidate': candidate,
                    'target': 'same' if query_subject == candidate_subject else 'different'})
    _require(covered == set(people), 'Matched cycle missing a patient')
    return {split: sorted(items, key=lambda x: x['case_id']) for split, items in rows.items()}


def render_volume(path):
    """One finite 3D array at a time; no pair, label, date or patient input."""
    image = nib.load(str(path))
    _require(len(image.shape) == 3, 'Expected a 3D MRI')
    _require(np.isfinite(image.affine).all() and abs(np.linalg.det(image.affine[:3,:3])) > 1e-8, 'Invalid affine')
    image = nib.as_closest_canonical(image)
    data = image.get_fdata(dtype=np.float32)
    _require(np.isfinite(data).all(), 'Nonfinite MRI')
    positive = data[data > 0]
    _require(positive.size >= 100, 'Insufficient positive MRI foreground')
    low, high = np.percentile(positive, [1, 99])
    _require(high > low, 'Degenerate MRI intensity window')
    quantiles = np.percentile(positive, [10,25,50,75,90,99])
    qc = {'canonical_orientation': list(nib.aff2axcodes(image.affine)), 'shape': list(data.shape),
          'spacing_mm': [float(v) for v in nib.affines.voxel_sizes(image.affine)],
          'finite_voxels': True, 'foreground_fraction': float(positive.size/data.size),
          'nonspatial_summary': [float(positive.size/data.size)] + [float(v/max(quantiles[2],1)) for v in quantiles],
          'window': [float(low),float(high)], 'tile_nonzero_fraction': [], 'tile_std': []}
    canvas = Image.new('L', (384,384))
    spacing = nib.affines.voxel_sizes(image.affine)
    for row, axis in enumerate(range(3)):
        plane_axes = [i for i in range(3) if i != axis]
        for col, fraction in enumerate(RENDER['fractions']):
            index = round((data.shape[axis]-1)*fraction)
            plane = np.take(data, index, axis=axis)
            normalized = np.clip((plane-low)/(high-low),0,1)
            pixels = (255*normalized.T[::-1]).astype(np.uint8)
            tile = Image.fromarray(pixels)
            physical = [plane.shape[i]*spacing[plane_axes[i]] for i in range(2)]
            ratio = min(128/physical[0],128/physical[1])
            size = (max(1,round(physical[0]*ratio)), max(1,round(physical[1]*ratio)))
            tile = tile.resize(size, Image.Resampling.BILINEAR)
            padded = Image.new('L',(128,128)); padded.paste(tile,((128-size[0])//2,(128-size[1])//2))
            canvas.paste(padded,(col*128,row*128))
            qc['tile_nonzero_fraction'].append(float(np.mean(pixels>0)))
            qc['tile_std'].append(float(np.std(pixels)))
    return canvas, qc


def compose_pair(query, candidate):
    _require(query.size == candidate.size == (384,384), 'Multiview panel shape mismatch')
    result=Image.new('L',(768,384)); result.paste(query,(0,0)); result.paste(candidate,(384,0))
    return result


def make_record(case, image, plan_sha256, render_sha256):
    question = QUESTION
    return {'id': case['case_id'], 'data_source': SOURCE, 'ability': ROUTE,
        'images': [str(Path(image).resolve())],
        'prompt': [{'role':'user', 'content':[{'type':'image','image':str(Path(image).resolve())}, {'type':'text','text':question}]}],
        'reward_model': {'style':'rule','ground_truth':case['target']},
        'extra_info': {'question':question,'reward_type':ROUTE,'task_type':ROUTE,
            'internal_split':case['split'], 'patient_lineage':'BIDS_SUBJECT_VERIFIED',
            'patient_key':token('patient',case['query_subject']),
            'candidate_patient_key':token('patient',case['candidate_subject']),
            'statistical_cluster_key':case['cluster_key'],
            'target_provenance':'Hidden OpenNeuro ds004856 BIDS subject equality across distinct source visits',
            'source_version':SOURCE_VERSION,'query_source_sha256':case['query']['sha256'],
            'candidate_source_sha256':case['candidate']['sha256'],
            'metadata_plan_sha256':plan_sha256,'render_contract_sha256':render_sha256,
            'visual_task_admission':'NOT_VERIFIED'}}


def validate_correspondence_row(row, expected_split=None):
    e=row.get('extra_info',{}); target=row.get('reward_model',{}).get('ground_truth')
    _require(row.get('data_source')==SOURCE and row.get('ability')==ROUTE and e.get('reward_type')==ROUTE, 'Source/route mismatch')
    _require(e.get('source_version')==SOURCE_VERSION and e.get('patient_lineage')=='BIDS_SUBJECT_VERIFIED', 'Source lineage mismatch')
    _require(e.get('internal_split') in {'train','internal_dev','test'} and
             (expected_split is None or e['internal_split']==expected_split), 'Internal split mismatch')
    _require(all(re.fullmatch('[0-9a-f]{24}',e.get(k,'')) for k in ('patient_key','candidate_patient_key','statistical_cluster_key')), 'Missing patient or cycle keys')
    _require(target==('same' if e['patient_key']==e['candidate_patient_key'] else 'different'), 'Ground truth differs from source equality')
    _require(all(re.fullmatch('[0-9a-f]{64}',e.get(k,'')) for k in ('query_source_sha256','candidate_source_sha256','metadata_plan_sha256','render_contract_sha256')), 'Missing source binding')
    _require(e['query_source_sha256'] != e['candidate_source_sha256'], 'Same image cannot be a cross-visit positive')
    _require(len(row.get('images',[]))==1 and e.get('question')==QUESTION, 'Model input contract mismatch')
    expected_prompt=[{'role':'user','content':[{'type':'image','image':row['images'][0]},{'type':'text','text':QUESTION}]}]
    _require(row.get('prompt')==expected_prompt,'Hidden fields or altered model prompt')


def score_visit_correspondence(data_source, solution_str, ground_truth, extra_info, format_score=.2):
    _require(data_source==SOURCE and isinstance(extra_info,dict) and extra_info.get('reward_type')==ROUTE, 'Source/route mismatch')
    _require(ground_truth in {'same','different'} and 0<=format_score<=1,'Invalid correspondence target/format weight')
    answer, formatting=_answer_and_format(solution_str)
    accuracy=float(answer.lower()==ground_truth) if formatting>0 else 0.
    return {'score':(1-format_score)*accuracy+format_score*formatting,'accuracy':accuracy,'format':formatting}


def _durable(path, content):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        _require(path.read_bytes()==content,'Refuse to overwrite changed task artifact'); return
    with path.open('xb') as f: f.write(content);f.flush();os.fsync(f.fileno())


def _png(path, image):
    import io
    buffer=io.BytesIO();image.save(buffer,format='PNG')
    _durable(path,buffer.getvalue())


def render_split(plan, split, source_root, destination, plan_sha256):
    _require(split in {'train','internal_dev'},'Test pixels are sealed; train/internal_dev rendering only')
    cases=build_cases(plan)[split]; destination=Path(destination); assets=destination/'assets'
    contract_sha=hashlib.sha256(json.dumps(RENDER,sort_keys=True).encode()).hexdigest()
    inventory={x['sha256']:x for c in cases for x in (c['query'],c['candidate'])}
    qc_rows=[]; panels={}
    for source_sha,item in sorted(inventory.items()):
        path=Path(source_root)/item['relative_path']
        _require(path.stat().st_size==item['bytes'] and sha_file(path)==source_sha,'Frozen MRI payload changed')
        panel,qc=render_volume(path); asset=assets/(token('volume',source_sha)+'.png')
        _png(asset,panel);panels[source_sha]=asset
        qc_rows.append({'source_sha256':source_sha,'panel_sha256':sha_file(asset),'panel_file':asset.name,**qc})
        _durable(destination/'qc_PRIVATE'/(token('qc',source_sha)+'.json'),(json.dumps(qc_rows[-1],sort_keys=True)+'\n').encode())
    rows=[]
    for case in cases:
        with Image.open(panels[case['query']['sha256']]) as query, Image.open(panels[case['candidate']['sha256']]) as candidate:
            pair=compose_pair(query,candidate)
        image=assets/(case['case_id']+'.png');_png(image,pair)
        row=make_record(case,image,plan_sha256,contract_sha);validate_correspondence_row(row,split);rows.append(row)
    _durable(destination/(split+'.verl.jsonl'),(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows)).encode())
    return {'split':split,'rows':len(rows),'people':len(plan['splits_PRIVATE'][split]),'source_volumes':len(inventory),
            'cycles':len({c['cluster_key'] for c in cases}),'render_contract':RENDER,'render_contract_sha256':contract_sha,
            'blank_or_low_contrast_tile_warnings':sum(any(v<.01 for v in q['tile_nonzero_fraction']) or any(v<1 for v in q['tile_std']) for q in qc_rows),
            'qc_PRIVATE':qc_rows,'manifest_sha256':sha_file(destination/(split+'.verl.jsonl')),'visual_task_admission':'NOT_VERIFIED'}


def fixed_threshold_metrics(scores, targets, threshold):
    scores=np.asarray(scores,dtype=float)
    _require(scores.ndim==1 and len(scores)==len(targets) and len(scores)>0 and np.isfinite(scores).all(), 'Invalid shortcut scores')
    _require(all(t in {'same','different'} for t in targets),'Invalid shortcut targets')
    truth=np.array([t=='same' for t in targets]); predicted=scores>=threshold
    return {'cases':len(targets),'correct':int(np.sum(predicted==truth)),
            'accuracy':float(np.mean(predicted==truth)),
            'same_recall':float(np.mean(predicted[truth])) if truth.any() else None,
            'different_recall':float(np.mean(~predicted[~truth])) if (~truth).any() else None}


def fit_train_threshold(scores, targets):
    values=np.unique(np.asarray(scores,dtype=float))
    _require(len(values)>0 and np.isfinite(values).all(),'Invalid training shortcut scores')
    margin=max(float(np.ptp(values)),1.)*1e-6
    choices=[float(values[0]-margin)]+[float((a+b)/2) for a,b in zip(values[:-1],values[1:])]+[float(values[-1]+margin)]
    # Fixed direction: larger similarity means same. First threshold breaks ties.
    threshold=max(choices,key=lambda t:fixed_threshold_metrics(scores,targets,t)['accuracy'])
    return {'threshold':threshold,'train_accuracy':fixed_threshold_metrics(scores,targets,threshold)['accuracy'],
            'fit_split':'train_only','score_direction':'higher_means_same'}
