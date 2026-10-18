#!/usr/bin/env python3
"""Join reviewed TRAIN qualification into standard Vero candidate rows.

Data assembly only: no image/model imports, fitting, inference, optimizer or
training entry point. Every row retains training_authorized=false.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path

W=Path(__file__).resolve().parents[1]
PINS={
    'sequence_reviewed':'761c1618a455031098eda68c3080808d9d610beb63293bb50af597eea3caaf9b',
    'Dallas_tumor_qualification':'219cd2bbe4fc336085378da5d0a22ff7a9092e57774bd497ac2ee8ddc5934944',
    'original_MRI_TRAIN':'0efb588b328e3d9af2a339d9237836357a5baff5d9db8b376dc716de3d1f10e3',
    'original_Dallas_TRAIN':'6adf06674050e395b704e4b76103f067166b905016920195603e271690de4f14',
}
COUNTS={'mri_sequence':168,'mri_visit_correspondence':48,'mri_tumor_presence':84,'mri_enhancing_presence':84,'mri_tumor_grid':46}
MRI_TASKS={'mri_sequence','mri_tumor_presence','mri_enhancing_presence','mri_tumor_grid'}


def require(condition,message):
    if not condition:raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_bound_rows(path,pin):
    raw=Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest()==pin,'Frozen input SHA mismatch')
    return [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]


def check_schema(row):
    require(isinstance(row.get('id'),str) and row['id'],'Missing row ID')
    task=row.get('ability');require(task in COUNTS,'Unknown task')
    expected_source='OpenNeuro_ds004856' if task=='mri_visit_correspondence' else 'OpenNeuro_ds007045'
    require(row.get('data_source')==expected_source,'Task/source disagreement')
    e=row.get('extra_info',{})
    require(e.get('internal_split')=='train' and e.get('patient_lineage')=='BIDS_SUBJECT_VERIFIED' and e.get('patient_key'),'NonTRAIN or unverified source patient')
    require(e.get('reward_type')==e.get('task_type')==task,'Typed reward/task mismatch')
    require(len(row.get('images',[]))==1 and isinstance(row['images'][0],str) and Path(row['images'][0]).is_absolute(),'Exactly one absolute image path required')
    question=e.get('question');require(isinstance(question,str) and question,'Question missing')
    expected=[{'role':'user','content':[{'type':'image','image':row['images'][0]},{'type':'text','text':question}]}]
    require(row.get('prompt')==expected,'Standard Vero image/question prompt changed')
    reward=row.get('reward_model',{})
    require(reward.get('style')=='rule' and isinstance(reward.get('ground_truth'),str) and reward['ground_truth'],'Hidden source target missing')


def unique(rows,key,label):
    out={}
    for row in rows:
        value=key(row);require(value not in out,'Duplicate '+label);out[value]=row
    return out


def assemble(sequence,qualification,mri_train,dallas_train,*,source_hashes,expected_counts=None):
    counts=COUNTS if expected_counts is None else expected_counts
    require(set(source_hashes)==set(PINS),'All four input SHA bindings required')
    for rows in (sequence,mri_train,dallas_train):
        for row in rows:check_schema(row)
        unique(rows,lambda r:r['id'],'source/candidate ID')
    require(all(r['ability'] in MRI_TASKS for r in mri_train),'Unexpected source in MRI registry')
    require(all(r['ability']=='mri_visit_correspondence' for r in dallas_train),'Unexpected source in Dallas registry')
    require(Counter(r['ability'] for r in mri_train)=={k:v for k,v in counts.items() if k in MRI_TASKS},'MRI TRAIN registry task coverage differs')
    require(len(dallas_train)==counts.get('mri_visit_correspondence',0),'Dallas TRAIN registry coverage differs')
    # The original source index is immutable within ds007045. Reviewed sequence
    # IDs and filenames were deliberately replaced with opaque blind IDs.
    source_sequences=[r for r in mri_train if r['ability']=='mri_sequence']
    by_index=unique(source_sequences,lambda r:r['extra_info'].get('index'),'original sequence index')
    require(None not in by_index,'Original sequence index missing')
    require(len(sequence)==counts['mri_sequence'],'Reviewed sequence coverage differs')
    require(len({r['extra_info'].get('index') for r in sequence})==len(sequence),'Duplicate reviewed sequence source index')
    require({r['extra_info'].get('index') for r in sequence}==set(by_index),'Reviewed sequence source IDs incomplete')
    out=[]
    for row in sequence:
        source=by_index[row['extra_info']['index']]
        require(row['ability']=='mri_sequence' and row['data_source']==source['data_source'],'Sequence task/source changed')
        require(row['extra_info']==source['extra_info'] and row['reward_model']==source['reward_model'],'Reviewed sequence hidden target or source metadata changed')
        q=row.get('sequence_qualification_v1',{})
        require(q.get('training_authorized') is False and q.get('expert_review')=='PENDING' and
            q.get('independent_visual_review')=='VISUAL_NONEXPERT_REVIEW_COMPLETE_EXPERT_PENDING','Sequence qualification promoted or missing')
        adequate=q.get('nonexpert_technical_coverage_adequate');require(isinstance(adequate,bool),'Sequence coverage state missing')
        require(q.get('blind_review_abstained') is (not adequate),'Sequence coverage/abstention conflict')
        status='SEQUENCE_NONEXPERT_COVERAGE_CANDIDATE_EXPERT_PENDING' if adequate else 'SEQUENCE_VISUAL_ABSTENTION_HOLD'
        item=copy.deepcopy(row);item['training_authorized']=False
        item['train_qualification_bundle_v1']={'version':1,'training_authorized':False,'scientific_admission':'HOLD',
            'route_status':status,'source_record_id_PRIVATE':source['id'],
            'source_registry_sha256':source_hashes['original_MRI_TRAIN'],
            'qualification_source_sha256':source_hashes['sequence_reviewed'],'qualification':copy.deepcopy(q)}
        out.append(item)
    other=dallas_train+[r for r in mri_train if r['ability']!='mri_sequence']
    original=unique(other,lambda r:r['id'],'nonsequence source ID')
    indexed=unique(qualification,lambda r:r.get('record_id_PRIVATE'),'qualification source ID')
    require(None not in indexed and set(indexed)==set(original),'Qualification/source exact ID coverage differs')
    for source in other:
        q=indexed[source['id']]
        require(q.get('task')==source['ability'] and q.get('split')=='train','Qualification task/split changed')
        require(q.get('training_authorized') is False and q.get('expert_eligibility')=='NOT_VERIFIED','Qualification promoted or missing HOLD')
        require(q.get('visual_eligibility')=='NOT_ADMITTED_NONEXPERT_INSPECTION_ONLY' or
            q.get('route_status')=='WHOLE_HEAD_IDENTITY_ENGINEERING_ONLY_BRAIN_GROUNDING_UNPROVEN','Visual eligibility unexpectedly admitted')
        is_dallas=source['ability']=='mri_visit_correspondence';item=copy.deepcopy(source);item['training_authorized']=False
        item['train_qualification_bundle_v1']={'version':1,'training_authorized':False,'scientific_admission':'HOLD',
            'route_status':'DALLAS_DOWNGRADED_WHOLEHEAD_IDENTITY_ONLY' if is_dallas else 'TUMOR_MASK_DERIVED_EXPERT_VISIBILITY_HOLD',
            'source_record_id_PRIVATE':source['id'],
            'source_registry_sha256':source_hashes['original_Dallas_TRAIN' if is_dallas else 'original_MRI_TRAIN'],
            'qualification_source_sha256':source_hashes['Dallas_tumor_qualification'],'qualification':copy.deepcopy(q)}
        out.append(item)
    require(Counter(r['ability'] for r in out)==counts,'Unified task counts differ')
    unique(out,lambda r:r['id'],'unified candidate ID')
    require(len({r['train_qualification_bundle_v1']['source_record_id_PRIVATE'] for r in out})==len(out),'Original source ID appears more than once')
    require(all(r['training_authorized'] is False and r['train_qualification_bundle_v1']['training_authorized'] is False for r in out),'Training authorization must remain false')
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--primary-root',type=Path,required=True)
    p.add_argument('--destination',type=Path,default=W/'local_data/train_qualification_bundle_v1/candidate_v1');a=p.parse_args()
    root=a.primary_root.resolve();out=a.destination.resolve();public=W/'results/TRAIN_QUALIFICATION_BUNDLE_V1.json'
    require(out.is_relative_to(W/'local_data/train_qualification_bundle_v1') and not out.exists(),'New scoped private destination required')
    require(not public.exists(),'Public bundle registry already exists')
    paths={'sequence_reviewed':W/'local_data/sequence_qualification_v1/audit_v1/future_loader_manifest_reviewed_PRIVATE.jsonl',
        'Dallas_tumor_qualification':W/'local_data/dallas_spatial_qualification_v1/candidate_v2/candidate_manifest_PRIVATE.jsonl',
        'original_MRI_TRAIN':root/'local_data/ds007045_bounded/vero_training_v1/train.verl.jsonl',
        'original_Dallas_TRAIN':root/'local_data/dallas_correspondence_prospective40_v1/task_v1/train.verl.jsonl'}
    bound={key:read_bound_rows(path,PINS[key]) for key,path in paths.items()}
    rows=assemble(*(bound[k] for k in PINS),source_hashes=PINS)
    require(Counter(r['train_qualification_bundle_v1']['route_status'] for r in rows)=={
        'SEQUENCE_NONEXPERT_COVERAGE_CANDIDATE_EXPERT_PENDING':166,'SEQUENCE_VISUAL_ABSTENTION_HOLD':2,
        'DALLAS_DOWNGRADED_WHOLEHEAD_IDENTITY_ONLY':48,'TUMOR_MASK_DERIVED_EXPERT_VISIBILITY_HOLD':214},'Frozen qualification count partition differs')
    out.mkdir(parents=True);manifest=out/'train_candidates_PRIVATE.verl.jsonl'
    with manifest.open('x') as f:
        for row in rows:f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
    private={'schema':'train-qualification-bundle-v1','input_paths_PRIVATE':{k:str(v) for k,v in paths.items()},
        'input_sha256':PINS,'manifest_sha256':sha(manifest),'rows':len(rows),'training_authorized':False}
    with (out/'binding_PRIVATE.json').open('x') as f:json.dump(private,f,indent=2,sort_keys=True);f.write('\n')
    registry={'schema':'train-qualification-bundle-v1','status':'TESTED_CPU_SCHEMA_ASSEMBLY_CANDIDATE_ONLY','evidence_tier':'OBSERVED_CPU',
        'research_tag':'ENGINEERING_UNBLOCK','mission_alignment':'Expose existing TRAIN task qualifications as one parseable Vero candidate input without changing targets or authorizing training.',
        'rows':430,'task_counts':dict(Counter(r['ability'] for r in rows)),
        'route_status_counts':dict(Counter(r['train_qualification_bundle_v1']['route_status'] for r in rows)),
        'manifest_sha256':sha(manifest),'input_sha256':PINS,
        'standard_vero_fields':['id','data_source','ability','images','prompt','reward_model','extra_info'],
        'qualification_namespace':'train_qualification_bundle_v1','training_authorized':False,'scientific_admission':'HOLD',
        'checks':{'source_SHA_bindings':4,'exact_source_ID_task_split_coverage':True,'duplicate_ids':0,'nonTRAIN_rows':0,
            'hidden_targets_preserved':430,'independent_expert_review_promotions':0,'training_launchers_created':0},
        'guard':'Candidate data only. Every row has training_authorized=false; external qualification review and new researcher authorization remain necessary before any future training executor may use it.',
        'limitations':['Sequence166 coverage rows remain nonexpert candidates;2 rows retain visual abstention.','Dallas48 remain downgraded whole-head identity candidates.','Tumor214 retain mask-derived targets and expert visibility/negative-adjudication HOLD.'],
        'code_sha256':{'assembler':sha(__file__),'tests':sha(W/'tests/test_assemble_train_qualification_bundle_v1.py')}}
    with public.open('x') as f:json.dump(registry,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({'rows':len(rows),'manifest_sha256':sha(manifest),'public_registry_sha256':sha(public),'training_authorized':False},sort_keys=True))


if __name__=='__main__':main()
