"""Metadata-only historical exclusion and source availability, no outcome scoring."""
import hashlib
from pathlib import Path
import re

SKIP={'response','responses','prediction','predictions','model_outputs','outputs',
      'answer','label','labels','ground_truth','ground_truth_PRIVATE',
      'source_equality_target','modality_target_PRIVATE','correct','accuracy','scores'}


def identity_index(subjects):
    result={}
    for subject in subjects:
        token=hashlib.sha256(subject.encode()).hexdigest()[:12]
        if token in result:raise ValueError('Anonymous patient identity collision')
        result[token]=subject;result[subject]=subject
    return result


def metadata_identities(obj,index):
    result=set()
    def walk(value):
        if isinstance(value,dict):
            for key,child in value.items():
                if key not in SKIP:walk(child)
        elif isinstance(value,list):
            for child in value:walk(child)
        elif isinstance(value,str):
            if value in index:result.add(index[value])
            for token in re.findall(r'(?<![A-Za-z0-9])(?:sub-[A-Za-z]+\d+|[0-9a-f]{12})(?![A-Za-z0-9])',value):
                if token in index:result.add(index[token])
    walk(obj)
    return result


def final_identity_projection(registry,index):
    # Later formal recipes use a different opaque patient-key namespace. Their
    # approved SOURCE identity crosswalk is authoritative; no test labels read.
    projected=[{'patient_key':v['patient_key']} for v in registry['cases_PRIVATE']]
    projected.extend({'subject_PRIVATE':v['subject_PRIVATE']} for v in registry['files_PRIVATE'])
    people=metadata_identities(projected,index)
    if len(people)!=registry['people']:raise ValueError('Final source identity count does not reconcile')
    return people


def source_availability(metadata,subject):
    directory=Path(metadata)/subject/'anat';modalities={}
    for modality,suffix in [('FLAIR','FLAIR'),('T1ce','run-02_T1w')]:
        image=directory/(subject+'_'+suffix+'.nii.gz');sidecar=directory/(subject+'_'+suffix+'.json')
        if image.is_file():
            with image.open('rb') as f:magic=f.read(2)
            status='GZIP_HEADER_ONLY_NOT_QUALIFIED' if magic==b'\x1f\x8b' else 'NON_GZIP_POINTER_OR_OTHER'
        elif image.is_symlink():status='DANGLING_ANNEX_SYMLINK_NO_PAYLOAD'
        else:status='NO_LOCAL_PAYLOAD'
        modalities[modality]={'status':status,'sidecar_present':sidecar.is_file(),
            'sidecar_sha256':hashlib.sha256(sidecar.read_bytes()).hexdigest() if sidecar.is_file() else None,
            'image_path_PRIVATE':str(image),'symlink_target_PRIVATE':str(image.readlink()) if image.is_symlink() else None}
    return {'modalities':modalities,'both_sidecars':all(x['sidecar_present'] for x in modalities.values()),
            'both_gzip_payloads':all(x['status']=='GZIP_HEADER_ONLY_NOT_QUALIFIED' for x in modalities.values()),
            'image_geometry_or_label_qualified':False,'pixel_decode_performed':False}
