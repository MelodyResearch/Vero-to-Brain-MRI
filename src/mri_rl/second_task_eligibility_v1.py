"""Deterministic TRAIN review queues; source masks are not slice-visibility gold.

This module imports no imaging, model or optimizer packages. It consumes only
already saved TRAIN source geometry and immutable nonexpert observations.
"""
from __future__ import annotations
from collections import Counter,defaultdict
import re

RULES={
    'version':'second-task-eligibility-v1',
    'task_under_review':'Existing mri_tumor_grid: 2D source whole-mask union centroid grid in the FLAIR display frame',
    'whole_mask':'mask>0 in original source renderer; no relabeling to visible/clinical tumor',
    'area_screen_native_pixels':1000,'boundary_screen_display_pixels':10,
    'threshold_scope':'Inherited engineering screens, not validated visibility thresholds; never turn small positives into negatives',
    'clean_spatial_review_candidate_requires':['mask_positive','saved MRI-only apparent abnormality',
        'coverage not sparse','one mask component','area>=1000','centroid boundary distance>=10 display pixels',
        'centroid on mask','global/largest-component grid agreement','pixel-center convention invariant',
        'no cavity/rim/resection lexical flag in saved notes'],
    'cavity_note_regex':r'cavit|rim|resect|post.?operat',
    'scientific_requirements':['Independent expert frame visibility/negative adjudication',
        'Explicit question versus annotation-endpoint alignment','Source label3 semantic confirmation before enhancing route admission'],
    'source_population':'README states preoperative histologically confirmed primary glioblastoma; cavity-like appearance does not establish postoperative status',
    'training_authorized':False,
}


def require(value,message):
    if not value:raise ValueError(message)


def plane_triage(key,note):
    require(note.get('clinical_presence_truth')=='NOT_ADJUDICATED' and note.get('enhancement_truth')=='NOT_ADJUDICATED'
        and note.get('expert_eligibility')=='NOT_VERIFIED','Saved note is not unadjudicated nonexpert evidence')
    g=key['tumor_geometry'];area=g['area'];components=g['components'];flags=set(g['geometric_flags'])
    require(isinstance(area,int) and area>=0 and isinstance(components,int) and components>=0,'Invalid saved geometry')
    require((area==0)==(components==0),'Area/component inconsistency')
    observation=note['MRI_only_nonexpert_observation'];apparent=observation.startswith('APPARENT_')
    sparse=observation=='SPARSE_SUPERIOR_FRAME_UNANSWERABLE'
    require(apparent or sparse or observation=='NO_CONSPICUOUS_FOCUS_AT_CONTACT_SHEET_RESOLUTION_NOT_A_NEGATIVE_GOLD','Unknown saved observation')
    cavity=bool(re.search(RULES['cavity_note_regex'],note.get('notes',''),flags=re.IGNORECASE))
    reasons=['EXPERT_FRAME_VISIBILITY_NOT_ADJUDICATED','QUESTION_ANNOTATION_ENDPOINT_ALIGNMENT_NOT_ADJUDICATED']
    if sparse:reasons.append('INSUFFICIENT_ANATOMICAL_COVERAGE')
    if area==0:reasons.append('MASK_ZERO_NOT_VISIBLE_NEGATIVE_GOLD')
    else:
        if area<RULES['area_screen_native_pixels']:reasons.append('SMALL_SOURCE_MASK_REQUIRES_REVIEW')
        if components>1:reasons.append('MULTICOMPONENT_MASK_NOT_MULTIFOCAL_DIAGNOSIS')
        if g['boundary_distance_display_pixels']<RULES['boundary_screen_display_pixels']:reasons.append('CENTROID_NEAR_GRID_BOUNDARY')
    for source_flag,reason in [('CENTROID_OFF_MASK','CENTROID_OFF_MASK'),
        ('GLOBAL_VS_LARGEST_COMPONENT_GRID_DISAGREEMENT','GLOBAL_LARGEST_REGION_GRID_DISAGREEMENT'),
        ('PIXEL_CENTER_CHANGES_GRID','PIXEL_CENTER_CONVENTION_CHANGES_TARGET')]:
        if source_flag in flags:reasons.append(reason)
    if cavity:reasons.append('CAVITY_RIM_ETIOLOGY_UNADJUDICATED')
    geometry_risks=set(reasons)-{'EXPERT_FRAME_VISIBILITY_NOT_ADJUDICATED','QUESTION_ANNOTATION_ENDPOINT_ALIGNMENT_NOT_ADJUDICATED'}
    if sparse:queue='COVERAGE_UNANSWERABLE'
    elif area==0 and apparent:queue='MASK_NEGATIVE_MRI_APPARENT_DISCORDANCE'
    elif area>0 and not apparent:queue='MASK_POSITIVE_MRI_NOT_CONSPICUOUS'
    elif area==0:queue='MASK_NEGATIVE_NEEDS_EXPERT_NEGATIVE_ADJUDICATION'
    elif cavity:queue='CAVITY_RIM_ETIOLOGY_ADJUDICATION'
    elif geometry_risks:queue='POSITIVE_GEOMETRY_COMPLEXITY_ADJUDICATION'
    else:queue='SPATIAL_REVIEW_CANDIDATE_EXPERT_PENDING'
    return {'primary_queue':queue,'spatial_review_candidate':queue=='SPATIAL_REVIEW_CANDIDATE_EXPERT_PENDING',
        'reasons':reasons,'mask_positive':area>0,'MRI_only_nonexpert_apparent':apparent,
        'postoperative_status':'NOT_VERIFIED_SOURCE_COHORT_PREOPERATIVE','training_authorized':False}


def metadata_ceiling(rows,fields):
    """Empirical majority contingency, never a fitted prediction/generalization."""
    require(bool(rows),'Nonempty TRAIN contingency required');groups=defaultdict(Counter)
    for r in rows:groups[tuple(str(r[f]) for f in fields)][r['label']]+=1
    n=len(rows);correct=sum(max(c.values()) for c in groups.values())
    return {'rows':n,'groups':len(groups),'majority_correct':correct,'resubstitution_ceiling':correct/n,
        'interpretation':'Training contingency only; not a fitted classifier, unseen-patient score or evidence that a VLM used metadata'}


def matched_label_controls(rows):
    """Actual disjoint case pairs: same site+level, different people and targets.

    A fixed metadata-only answer cannot solve both cases in a pair. This builds
    the review/evaluation index only; no new images or model outputs are made.
    """
    groups=defaultdict(list)
    for r in rows:groups[r['site'],r['level']].append(r)
    pairs=[];used=set()
    for key,group in sorted(groups.items()):
        group=sorted(group,key=lambda r:r['id'])
        for r in group:
            if r['id'] in used:continue
            choices=[s for s in group if s['id'] not in used and s['id']!=r['id'] and s['patient']!=r['patient'] and s['label']!=r['label']]
            if not choices:continue
            other=choices[0];used.update((r['id'],other['id']))
            pairs.append({'site':key[0],'level':key[1],'record_ids':[r['id'],other['id']],
                'targets_PRIVATE':[r['label'],other['label']],'different_patients_verified':True})
    return {'pairs':len(pairs),'covered_rows':2*len(pairs),'unmatched_rows':len(rows)-2*len(pairs),
        'metadata_only_pair_ceiling':.5 if pairs else None,'pairs_PRIVATE':pairs,
        'model_or_expert_control_execution':'NOT_RUN; deterministic existing-TRAIN index only'}
