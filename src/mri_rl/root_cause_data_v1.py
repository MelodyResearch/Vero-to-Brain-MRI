"""CPU-only source-bound candidates. Technical checks never confer visual admission.

No model, optimizer, fitting or evaluation module is imported. The only pixels
accepted here are existing train PNGs. A caller must supply frozen train-only
references, rather than infer patient/site/labels from appearance.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageStat

from mri_rl.visit_correspondence import validate_correspondence_row

SEQUENCE_QUESTION = ('Which MRI sequence is shown? Answer exactly FLAIR or contrast-enhanced T1. '
                     'Give the final answer in \\boxed{...}.')
LEVELS = (45,91,136)
MODALITIES = {'flair':'FLAIR','t1ce':'contrast-enhanced T1'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def image_qc(path, size, expected_sha=None):
    """Decode bounded PNG, record contrast; positive statistics are not visibility."""
    if expected_sha is not None:
        require(sha(path)==expected_sha,'Frozen image hash changed')
    with Image.open(path) as image:
        require(image.format=='PNG' and image.size==size,'PNG geometry changed')
        require(image.mode in {'L','RGB'},'Unexpected image mode')
        image.load(); gray=image.convert('L'); histogram=gray.histogram(); count=sum(histogram)
        require(count>0 and len([n for n in histogram if n])>1,'Blank or constant image')
        return {'size':list(size),'mode':image.mode,'gray_std':float(ImageStat.Stat(gray).stddev[0]),
                'nonzero_fraction':1-histogram[0]/count,'image_sha256':sha(path)}


def validate_sequence_rows(rows, references, expected_people=28):
    refs={r['record_id']:r for r in references}
    require(len(refs)==len(references),'Duplicate train references')
    require(len(rows)==expected_people*6,'Incomplete patient/modality/level inventory')
    require(len({r['id'] for r in rows})==len(rows),'Duplicate row IDs')
    pairs=defaultdict(list); controls=defaultdict(Counter); qcs=[]; hashes=set(); patients=set()
    for row in rows:
        e=row.get('extra_info',{}); key=e.get('patient_key','')
        require(row.get('ability')=='mri_sequence' and row.get('data_source')=='OpenNeuro_ds007045','Source/route mismatch')
        require(e.get('internal_split')=='train' and e.get('patient_lineage')=='BIDS_SUBJECT_VERIFIED','Only verified train patients accepted')
        require(e.get('reward_type')==e.get('task_type')=='mri_sequence','Typed task mismatch')
        match=re.fullmatch(re.escape(key)+r'_z(045|091|136)_(flair|t1ce)_sequence',row['id'])
        require(match is not None,'Patient/level/modality ID linkage mismatch')
        level=int(match[1]); label=MODALITIES[match[2]]
        require(row.get('reward_model',{}).get('ground_truth')==label and e.get('answer')==label,'Source modality label mismatch')
        require(len(row.get('images',[]))==1 and e.get('question')==SEQUENCE_QUESTION,'Image/question mismatch')
        image=row['images'][0]
        require(row.get('prompt')==[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':SEQUENCE_QUESTION}]}],'Unexpected model-visible prompt field')
        ref=refs.get(row['id']); require(ref is not None,'Not in frozen train image inventory')
        require((ref['patient'],ref['level'],ref['label'],ref['image_PRIVATE'])==(key,level,label,image),'Frozen train lineage changed')
        require(isinstance(ref.get('site'),str) and ref['site'],'Source site missing; never infer site')
        qc=image_qc(image,(364,436),ref['image_sha256']); qcs.append(qc)
        require(qc['image_sha256'] not in hashes,'Duplicate exact image across task examples');hashes.add(qc['image_sha256'])
        patients.add(key);pairs[key,level].append(label);controls[ref['site'],level][label]+=1
    require(len(patients)==expected_people and len(pairs)==expected_people*3,'Unexpected patient/level coverage')
    require(all(Counter(labels)==Counter(MODALITIES.values()) for labels in pairs.values()),'Within-patient same-level modality pairing missing')
    return {'rows':len(rows),'people':len(patients),'patient_level_pairs':len(pairs),
        'levels':{str(z):sum(len(v) for (p,level),v in pairs.items() if level==z) for z in LEVELS},
        'labels':dict(Counter(r['reward_model']['ground_truth'] for r in rows)),
        'source_sites':len({site for site,z in controls}),
        'source_site_level_majority_ceiling':sum(max(c.values()) for c in controls.values())/len(rows),
        'gray_std_min':min(q['gray_std'] for q in qcs),'blank_constant_images':0,
        'nonzero_fraction_min':min(q['nonzero_fraction'] for q in qcs),
        'nonzero_under_5percent_warning_rows':sum(q['nonzero_fraction']<.05 for q in qcs),
        'sparse_foreground_flag_scope':'Descriptive engineering warning, not a clinical eligibility threshold or source-label change',
        'scientifically_qualified_rows':0,'visual_qualification':'NOT_VERIFIED',
        'source_patient_independence':'WITHIN_SOURCE_TRAIN_REGISTRY_ONLY; cross_source_global_identity_UNVERIFIED'}


def validate_dallas_rows(rows, expected_people=24, expected_cycles=12):
    require(len(rows)==expected_cycles*4,'Incomplete matched-cycle inventory')
    require(len({r['id'] for r in rows})==len(rows),'Duplicate Dallas row IDs')
    cycles=defaultdict(list);people=set();queries=defaultdict(Counter)
    for row in rows:
        validate_correspondence_row(row,expected_split='train')
        e=row['extra_info']; people.update((e['patient_key'],e['candidate_patient_key']))
        cycles[e['statistical_cluster_key']].append(row)
        queries[e['patient_key']][row['reward_model']['ground_truth']]+=1
        # Dallas images need not have different bytes per case: pair linkage and
        # source panels below are authoritative. A fully constant PNG is rejected.
        image_qc(row['images'][0],(768,384))
    require(len(people)==expected_people and len(cycles)==expected_cycles,'Unexpected Dallas independent-unit inventory')
    require(all(c=={'same':1,'different':1} for c in queries.values()),'Each query must have a same/different matched pair')
    seen=set()
    for cluster,group in cycles.items():
        members={r['extra_info']['patient_key'] for r in group}
        require(len(group)==4 and len(members)==2 and not members&seen,'Broken or duplicated two-person cycle')
        require({(r['extra_info']['patient_key'],r['extra_info']['candidate_patient_key']) for r in group}=={(a,b) for a in members for b in members},'Cycle is not full ordered cross-product')
        # Each person's visit is fixed across positive and negative pairings.
        for member in members:
            require(len({r['extra_info']['query_source_sha256'] for r in group if r['extra_info']['patient_key']==member})==1,'Query volume changed within cycle')
            require(len({r['extra_info']['candidate_source_sha256'] for r in group if r['extra_info']['candidate_patient_key']==member})==1,'Candidate volume changed within cycle')
        seen.update(members)
    return {'rows':len(rows),'people':len(people),'cycles':len(cycles),'statistical_unit':'two_person_matched_cycle',
        'labels':dict(Counter(r['reward_model']['ground_truth'] for r in rows)),
        'task_scope':'cross_visit_BIDS_identity_not_clinical_change',
        'query_identity_majority_ceiling':.5,'scientifically_qualified_rows':0,
        'visual_qualification':'NOT_VERIFIED','scanner_site':'UNVERIFIED; manufacturer is not site'}


def training_rows(candidates):
    """No valid source-bound visual review exists in v1; no self-attested bypass.

    A future independently reviewed evidence schema is an explicit code change.
    Merely replacing status strings cannot turn this inventory into training.
    """
    require(not candidates,'Independent source-bound visual qualification required before training; v1 has no admitted review receipt')
    return []


def patient_label_confounding(rows):
    """Exact TRAIN contingency arithmetic; no estimator fitting or predictions.

    The majority statistic is a resubstitution lookup ceiling. It cannot measure
    generalization to unseen people and does not demonstrate that a VLM used it.
    """
    require(bool(rows),'Nonempty train rows required')
    groups=defaultdict(Counter)
    for row in rows:
        require(row.get('ability')=='mri_sequence' and row['extra_info'].get('internal_split')=='train','Sequence TRAIN rows only')
        label=row['reward_model']['ground_truth'];require(label in MODALITIES.values(),'Unknown modality truth')
        groups[row['extra_info']['patient_key']][label]+=1
    n=len(rows);entropy_terms=[];correct=0
    for counts in groups.values():
        size=sum(counts.values());correct+=max(counts.values())
        entropy_terms.append(size/n*(-sum((c/size)*math.log2(c/size) for c in counts.values())))
    return {'rows':n,'people':len(groups),'label_counts':dict(Counter(r['reward_model']['ground_truth'] for r in rows)),
        'conditional_label_entropy_bits':math.fsum(entropy_terms),'patient_majority_resubstitution_correct':correct,
        'patient_majority_resubstitution_accuracy':correct/n,
        'single_label_patients':sum(len(c)==1 for c in groups.values()),
        'interpretation':'Exact training contingency ceiling; not fitted model accuracy, unseen-patient generalization or proof of a VLM shortcut.'}
