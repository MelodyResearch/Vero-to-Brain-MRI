"""Explicit TRAIN source-mask arithmetic. No mask becomes clinical/visible gold.

Coordinates are panel-relative image-edge coordinates: the first displayed
pixel occupies [0,1); its center is .5. A native mask pixel occupies a 2x2
block, whose center is 2*(native_index+.5). Thirds are half-open; a boundary
belongs to the cell on its right/bottom. All components use 8-connectivity.
"""
from __future__ import annotations
import copy
import numpy as np
from scipy import ndimage


def source_labels(mask):
    a=np.asarray(mask)
    if not np.isfinite(a).all() or not np.isin(a,[0,1,2,3]).all():
        raise ValueError('Source mask contains noninteger or unknown numeric labels')
    vals,counts=np.unique(a,return_counts=True)
    return {str(int(v)):int(n) for v,n in zip(vals,counts)}


def _center(mask):
    y,x=np.nonzero(mask);n=len(x);h,w=mask.shape
    sx=int(x.sum());sy=int(y.sum())
    # Integer arithmetic makes exact-third tie handling independent of float roundoff.
    col=min(2,(3*(2*sx+n))//(2*n*w));row=min(2,(3*(2*sy+n))//(2*n*h))
    cx=sx/n;cy=sy/n;edge=[2*cx+1,2*cy+1]
    legacy_col=min(2,int(3*cx/w));legacy_row=min(2,int(3*cy/h))
    names_x=['left','center','right'];names_y=['top','middle','bottom']
    return {'area_native_pixels':n,'union_centroid_display_edge_xy':edge,
        'legacy_centroid_display_index_xy':[2*cx,2*cy],
        'prior_pixelcenter_display_index_xy':[2*cx+.5,2*cy+.5],
        'grid':names_x[col]+'-'+names_y[row],
        'legacy_grid':names_x[legacy_col]+'-'+names_y[legacy_row],
        'boundary_distance_display_pixels':min(
            abs(edge[0]-2*w/3),abs(edge[0]-4*w/3),abs(edge[1]-2*h/3),abs(edge[1]-4*h/3)),
        'union_centroid_on_mask_nearest_pixel':bool(mask[min(h-1,int(np.floor(cy+.5))),min(w-1,int(np.floor(cx+.5)))])}


def endpoints(mask):
    a=np.asarray(mask)
    if a.ndim!=2 or not a.size or a.dtype!=np.dtype(bool):
        raise ValueError('Nonempty 2D boolean mask required')
    base={'coordinate_frame':'panel-relative image edges;2x scale;half-open thirds;boundary right/bottom',
        'visible_focus_center':'NOT_ADJUDICATED','clinical_multifocal_status':'NOT_ADJUDICATED',
        'union_centroid_display_edge_xy':None,'grid':None,'legacy_grid':None,
        'area_native_pixels':int(a.sum()),'components_8':0,'components':[]}
    if not a.any():return base
    labels,n=ndimage.label(a,structure=np.ones((3,3),dtype=np.uint8));parts=[]
    for k in range(1,n+1):
        part=_center(labels==k);part['component_index']=k;parts.append(part)
    parts.sort(key=lambda p:(-p['area_native_pixels'],p['component_index']))
    union=_center(a);base.update(union);base.update(components_8=int(n),components=parts,
        largest_component_grid=parts[0]['grid'],
        component_grids_disagree_with_union=any(p['grid']!=union['grid'] for p in parts))
    return base


def amended_row(row,endpoint,ontology):
    if row.get('extra_info',{}).get('internal_split')!='train':raise ValueError('Only TRAIN accepted')
    if row.get('ability')!='mri_tumor_grid':raise ValueError('Only existing spatial task accepted')
    out=copy.deepcopy(row);same=row['reward_model']['ground_truth']==endpoint['grid']
    out['training_authorized']=False
    out['spatial_ontology_resolution_v1']={'version':1,'training_authorized':False,
        'scientific_admission':'HOLD','expert_adjudication':'NOT_COMPLETED',
        'original_prompt_and_target_preserved':True,'endpoint_PRIVATE':endpoint,'ontology':ontology,
        'proposed_arithmetic_target_PRIVATE':endpoint['grid'],
        'target_status':'ARITHMETIC_TARGET_UNCHANGED_EXPERT_PENDING' if same else 'OBJECTIVE_CONVENTION_CHANGE_REQUIRES_VERSIONED_ENDPOINT',
        'question_annotation_alignment':'NOT_ADJUDICATED; source union centroid is not automatically a visible-focus center',
        'visible_focus_or_dominant_lesion_target':'NOT_CREATED',
        'blocking_reasons':['EXPERT_FRAME_VISIBILITY_NOT_ADJUDICATED','QUESTION_ANNOTATION_ENDPOINT_ALIGNMENT_NOT_ADJUDICATED']}
    return out


def pointer_lineage(relative_path,local_target,historical_tree,current_tree):
    """Offline exact public Git-annex pointer comparison; no network requests."""
    import hashlib
    import re
    b=local_target.encode();blob=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
    current=current_tree.get(relative_path,{})
    if current.get('mode')!='120000' or current.get('sha')!=blob:
        raise ValueError('Local pointer does not match exact current public release tree')
    m=re.search(r'SHA256E-s(\d+)--([0-9a-f]{64})\.nii\.gz',local_target)
    if not m:raise ValueError('Expected SHA256E source annotation pointer')
    old=historical_tree.get(relative_path,{})
    same=old.get('mode')=='120000' and old.get('sha')==blob
    return {'local_pointer_gitblob_sha1':blob,'current_release_pointer_matches':True,
        'historical_pointer_present':bool(old),'historical_same_annotation_pointer':same,
        'source_payload_sha256':m[2],'source_payload_bytes':int(m[1]),
        'label3_semantics':'VERIFIED_PAPER_RELEASE_SAME_ANNOTATION' if same else 'NOT_VERIFIED_NEW_RELEASE_ANNOTATION'}
