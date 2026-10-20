"""Bounded independent source collection. No model or training dependency."""
from __future__ import annotations
from collections import Counter
import hashlib
from pathlib import Path
import re
import numpy as np
from PIL import Image

SELECTION_PREFIX='paired-signal-independent16-v1:'


def require(ok,message):
    if not ok:raise ValueError(message)


def select_subjects(records,excluded,quotas):
    require(len({r['subject_PRIVATE'] for r in records})==len(records),'Duplicate source subject')
    selected=[]
    for site,count in sorted(quotas.items()):
        require(type(count) is int and count>=0,'Invalid fixed quota')
        candidates=[r for r in records if r['site']==site and r['subject_PRIVATE'] not in excluded and r['processed_pair_available'] is True]
        candidates.sort(key=lambda r:hashlib.sha256((SELECTION_PREFIX+r['subject_PRIVATE']).encode()).hexdigest())
        require(len(candidates)>=count,'Insufficient source metadata availability for frozen site quota')
        selected.extend(candidates[:count])
    require(Counter(r['site'] for r in selected)=={k:v for k,v in quotas.items() if v},'Quota mismatch')
    return selected


def parse_annex(target):
    found=re.findall(r'SHA256E-s(\d+)--([0-9a-f]{64})\.nii\.gz',target)
    require(found and len(set(found))==1,'Expected one unambiguous SHA256E NIfTI source identity')
    n,h=found[0];require(int(n)>0,'Empty source object')
    return {'bytes':int(n),'sha256':h}


def verify_payload(path,size,sha):
    path=Path(path)
    if not path.is_file() or path.is_symlink() or path.stat().st_size!=size:return False
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()==sha


def render_z045(volume):
    a=np.asarray(volume)
    require(a.ndim==3 and a.shape[2]>45 and np.isfinite(a).all(),'Finite 3D source with z045 required')
    nonzero=a[a!=0][::16];require(nonzero.size>1,'Insufficient source foreground')
    lo,hi=np.percentile(nonzero,[.5,99.5]);require(hi>lo,'Degenerate normalization window')
    plane=np.rot90(a[:,:,45]);pixels=np.uint8(np.round(np.clip((plane-lo)/(hi-lo),0,1)*255))
    image=Image.fromarray(pixels).resize((364,436),Image.Resampling.BILINEAR).convert('RGB')
    display=np.asarray(image)[:,:,0]
    stats={'source_nonzero_voxels':int(np.count_nonzero(a)),'z045_nonzero_fraction':float(np.count_nonzero(plane)/plane.size),
        'display_nonzero_fraction':float(np.count_nonzero(display)/display.size),'display_black_fraction':float((display==0).mean()),
        'display_saturated_fraction':float((display==255).mean()),'display_min':int(display.min()),'display_max':int(display.max()),
        'display_std':float(display.std()),'clinical_visibility':'NOT_ADJUDICATED','technical_stats_used_for_case_substitution':False}
    return image,[float(lo),float(hi)],stats


def check_pair_geometry(shape_a,affine_a,shape_b,affine_b):
    from nibabel import aff2axcodes
    a=np.asarray(affine_a,dtype=float);b=np.asarray(affine_b,dtype=float)
    require(tuple(shape_a)==tuple(shape_b)==(182,218,182),'Expected same existing MNI182 geometry')
    require(a.shape==b.shape==(4,4) and np.isfinite(a).all() and np.isfinite(b).all(),'Finite NIfTI affines required')
    require(np.allclose(a,b,rtol=0,atol=1e-6) and abs(np.linalg.det(a[:3,:3]))>1e-8,'Paired source world geometry mismatch')
    axes=list(aff2axcodes(a));require(axes==list(aff2axcodes(b))==['L','A','S'],'Original LAS storage required;no automatic canonicalization')
    spacing=np.linalg.norm(a[:3,:3],axis=0);require(np.allclose(spacing,[1,1,1],rtol=0,atol=1e-5),'Expected MNI1mm spacing')
    return {'shape':[182,218,182],'storage_axcodes':axes,'world_coordinate_convention':'NIfTI RAS+ world;not RAS storage',
        'spacing_mm':spacing.tolist(),'paired_affine_equal_atol_1e_6':True,'affine_finite':True,'canonicalization_applied':False}
