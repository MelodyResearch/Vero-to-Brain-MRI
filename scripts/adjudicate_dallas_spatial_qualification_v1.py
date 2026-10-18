#!/usr/bin/env python3
"""Bind actual nonexpert MRI-only observations to separate engineering keys.

This is a review receipt, never an expert label generator or training permission.
"""
import argparse,json,sys
from collections import Counter
from pathlib import Path
W=Path(__file__).resolve().parents[1];sys.path.insert(0,str(W/'src'))
import nibabel as nib
import numpy as np
from mri_rl.dallas_spatial_qualification_v1 import require,sha

def write(path,obj):
    with path.open('x') as f:json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--primary-root',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--destination',type=Path,required=True);a=p.parse_args()
    source=a.source.resolve();dest=a.destination.resolve();R=a.primary_root.resolve()
    require(source==W/'local_data/dallas_spatial_qualification_v1/candidate_v1' and dest==W/'local_data/dallas_spatial_qualification_v1/candidate_v2' and not dest.exists(),'New v2 destination required')
    observations_path=source/'MRI_only_nonexpert_observations_BEFORE_OVERLAY_PRIVATE.json'
    require(sha(observations_path)=='6edb737d44b64f50e7a7286ae7c71cf8a27478a2b7288c564e3ad83d28f5fb37','Pre-overlay observations drift')
    observations=json.loads(observations_path.read_text());keys=json.loads((source/'tumor_truth_separate_PRIVATE.json').read_text());dkeys=json.loads((source/'dallas_truth_separate_PRIVATE.json').read_text())
    historical=R/'local_data/ds007045_bounded/safe_resume_v1/historical112_sampling_v2_PRIVATE.json'
    require(sha(historical)=='8abe468dfe50de53c4da23d31ea00f613ae38df7996677eaff14cdf2f49226b6','Pinned TRAIN reference drift')
    hist=json.loads(historical.read_text());maskpins={r['patient']:r['mask_sha256'] for r in hist['rows_PRIVATE']};geometry={r['record_id']:r for r in hist['training_references_PRIVATE']}
    qc=json.loads((source/'source_qc_PRIVATE.json').read_text());header_checks=[]
    for row in qc:
        path=Path(row['source_PRIVATE']);ni=nib.load(str(path));q,qcode=ni.get_qform(coded=True);s,scode=ni.get_sform(coded=True)
        require(np.isfinite(ni.affine).all() and abs(np.linalg.det(ni.affine[:3,:3]))>1e-8,'Invalid affine')
        header_checks.append({'kind':row['kind'],'source_sha256':row['source_sha256'],'qform_sform_max_difference_mm':float(np.max(abs(q-s))) if qcode and scode else None,
            'qform_code':int(qcode),'sform_code':int(scode),'chosen_affine':'nibabel NIfTI best affine; same-grid mask agreement verified by builder'})
        if row['kind']=='Tumor':
            patient=__import__('hashlib').sha256(path.parent.name.encode()).hexdigest()[:12]
            require(row['mask_sha256']==maskpins[patient],'Original TRAIN mask source bytes changed')
    for k in keys:
        for record in k['records_PRIVATE']:
            ref=geometry[record];require(ref['tumor']==k['tumor_geometry']['area'] and ref['enhancing']==k['enhancing_geometry']['area'],'Saved TRAIN geometry differs from independent reread')
    for k in dkeys:
        q,c=Path(k['query_source_PRIVATE']),Path(k['candidate_source_PRIVATE'])
        require(q.parts[-3]=='ses-wave2' and c.parts[-3]=='ses-wave1','Dallas visit direction changed')
        same=q.parts[-4]==c.parts[-4]
        require(k['source_equality_truth_PRIVATE']==('same' if same else 'different'),'Source visits do not support identity label')
    byid={k['review_id']:k for k in keys};obs={r['review_id']:r for r in observations['tumor']};cross=[]
    for rid,k in byid.items():
        apparent=obs[rid]['MRI_only_nonexpert_observation'].startswith('APPARENT')
        positive=k['tumor_geometry']['area']>0
        cross.append({'review_id':rid,'mask_geometry_positive':positive,'MRI_only_apparent_abnormality':apparent,
            'review_status':'NONEXPERT_APPARENT_REGION_MASK_POSITIVE_REQUIRES_EXPERT' if apparent and positive else ('MASK_POSITIVE_WITHOUT_CONSPICUOUS_NONEXPERT_MRI_FOCUS' if positive else ('MASK_NEGATIVE_WITH_APPARENT_MRI_ABNORMALITY' if apparent else 'MASK_NEGATIVE_NOT_CLINICALLY_CONFIRMED_NEGATIVE')),
            'clinical_gold':'NOT_VERIFIED'})
    dest.mkdir(parents=True);rows=[json.loads(l) for l in (source/'candidate_manifest_PRIVATE.jsonl').read_text().splitlines()]
    crossmap={r['review_id']:r for r in cross}
    for row in rows:
        row['visual_review_evidence']='Actual Codex nonexpert view_image inspection; not radiologist/expert adjudication'
        row['visual_eligibility']='NOT_ADMITTED_NONEXPERT_INSPECTION_ONLY'
        row['training_authorized']=False
        if row['review_id'].startswith('T'):row['nonexpert_triage']=crossmap[row['review_id']]['review_status']
        else:row['nonexpert_triage']='WHOLEHEAD_ANATOMY_VIEWED_BRAIN_ONLY_UNPROVEN'
    with (dest/'candidate_manifest_PRIVATE.jsonl').open('x') as f:
        for row in rows:f.write(json.dumps(row,sort_keys=True)+'\n')
    (dest/'qualified_training_manifest_PRIVATE.jsonl').touch(exist_ok=False)
    write(dest/'post_overlay_crosscheck_PRIVATE.json',{'pre_overlay_observations_sha256':sha(observations_path),'nonexpert_mri_mask_crosscheck':cross,
        'all84_overlay_planes_viewed':True,'overlay_visual_findings':'Mask overlays generally follow the asymmetric MRI regions. Cavity/rim-like abnormalities also occur without positive mask; subtle positive FLAIR regions were not confidently noticed in MRI-only contact review. This is qualitative nonexpert evidence, not expert agreement or visibility gold.',
        'independent_expert_review':'NOT_VERIFIED','clinical_visibility_labels_created':0})
    write(dest/'header_source_checks_PRIVATE.json',header_checks)
    agg=json.loads((source/'aggregate_SAFE.json').read_text());agg['schema']='dallas-spatial-qualification-v1-review-v2'
    agg['status']='TESTED_CPU_AND_NONEXPERT_VISUAL_REVIEW_ENGINEERING_ONLY'
    agg['Dallas']['source_identity_and_cross_visit_labels_independently_checked']=48
    agg['Dallas']['nonexpert_wholehead_pairs_actually_viewed']=48
    agg['Dallas']['visual_admission']='HOLD_BRAIN_ONLY_IDENTITY_ANSWERABILITY_NOT_ESTABLISHED'
    agg['Dallas']['occlusion_visual_failure']='Actual inspected fixed geometric ROI cuts brain and retains extracranial signal; cannot isolate intracranial cues.'
    agg['tumor']['mask_source_hashes_match_pinned_original_TRAIN_reference']=28
    agg['tumor']['geometry_matches_pinned_all_TRAIN_reference_rows']=214
    agg['tumor']['MRI_only_planes_actually_viewed_before_overlay']=84
    agg['tumor']['overlay_planes_actually_viewed_after_notes_frozen']=84
    agg['tumor']['nonexpert_crosscheck_plane_counts']=dict(Counter(r['review_status'] for r in cross))
    agg['tumor']['visual_admission']='HOLD_INDEPENDENT_EXPERT_VISIBILITY_AND_NEGATIVE_ADJUDICATION_MISSING'
    agg['tumor']['scientific_limit']='Nonexpert apparent abnormality may be cavity, edema, artifact or postoperative change; no independent clinical truth or expert assessment has been invented.'
    agg['header_checks']={'volumes':len(header_checks),'both_qform_and_sform_present':sum(r['qform_sform_max_difference_mm'] is not None for r in header_checks),
        'qform_sform_disagreement_gt_1e-3_mm':sum(r['qform_sform_max_difference_mm'] is not None and r['qform_sform_max_difference_mm']>1e-3 for r in header_checks)}
    agg['candidate_rows']=len(rows);agg['qualified_rows']=0;agg['training_authorized']=False
    agg['candidate_manifest_sha256']=sha(dest/'candidate_manifest_PRIVATE.jsonl')
    agg['source_code_sha256']={str(path.relative_to(W)):sha(path) for path in [W/'src/mri_rl/dallas_spatial_qualification_v1.py',W/'scripts/build_dallas_spatial_qualification_v1.py',Path(__file__),W/'tests/test_dallas_spatial_qualification_v1.py']}
    agg['pre_overlay_observations_sha256']=sha(observations_path)
    agg['review_assets']={'wholehead_sheets':8,'fixed_occlusion_sheets':8,'MRI_only_sheets':7,'mask_overlay_sheets':7,'truth_keys':'PRIVATE_SEPARATE_FROM_IMAGE_SHEETS',
        'image_privacy_limit':'Sheets suppress filenames, patient IDs and labels. Wholehead pixels retain reconstructable face/skull anatomy and are private research data, not guaranteed deidentified MRI.'}
    write(dest/'aggregate_SAFE.json',agg)
    public=W/'results/DALLAS_SPATIAL_QUALIFICATION_V1.json';write(public,agg)
    write(dest/'artifact_hashes_PRIVATE.json',{str(path.relative_to(source)):sha(path) for path in sorted(source.rglob('*')) if path.is_file()})
    print(json.dumps({'rows':len(rows),'qualified':0,'crosscheck_counts':agg['tumor']['nonexpert_crosscheck_plane_counts'],'header_checks':agg['header_checks'],'aggregate_sha256':sha(public)},sort_keys=True))

if __name__=='__main__':main()
