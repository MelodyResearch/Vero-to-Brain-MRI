"""Exact-row assembly of the new C pool; standard library only, no model imports.

No sampling, relabeling, prompt edits, image processing or test content is done.
Sequence rows are recovered from the completed balanced B28 schedule evidence.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

SOURCES={'mri_sequence':'OpenNeuro_ds007045','mri_visit_correspondence':'OpenNeuro_ds004856'}


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def read_rows(path):
    result=[];seen=set()
    for raw in Path(path).read_bytes().splitlines(keepends=True):
        if not raw.strip():continue
        row=json.loads(raw);require(row['id'] not in seen,'Duplicate source record ID');seen.add(row['id'])
        require(raw.endswith(b'\n'),'Source JSONL line lacks terminal newline')
        result.append({'row':row,'raw':raw})
    return result


def validate_model_boundary(row,split):
    e=row['extra_info'];route=row['ability'];images=row['images']
    require(row['data_source']==SOURCES.get(route) and e['reward_type']==route,'Typed source/route mismatch')
    require(e['internal_split']==split and e.get('patient_lineage')=='BIDS_SUBJECT_VERIFIED' and e.get('patient_key'),'Patient/split lineage mismatch')
    require(len(images)==1 and Path(images[0]).is_file(),'One existing image required')
    expected=[{'role':'user','content':[{'type':'image','image':images[0]},{'type':'text','text':e['question']}]}]
    require(row['prompt']==expected,'Model prompt and hidden-metadata boundary mismatch')
    target=row['reward_model']['ground_truth']
    if route=='mri_sequence':require(target in {'FLAIR','contrast-enhanced T1'},'Sequence target mismatch')
    else:
        require(target==('same' if e['patient_key']==e.get('candidate_patient_key') else 'different'),'Dallas source equality target mismatch')
        require(e.get('statistical_cluster_key') and e.get('query_source_sha256')!=e.get('candidate_source_sha256'),'Dallas cluster/source mismatch')


def extract_frozen_sequence(train,inventory):
    source={x['row']['id']:x for x in read_rows(train)};selected=[]
    for entry in inventory:
        if entry['category']!='mri_sequence':continue
        require(entry['record_id'] in source,'Frozen sequence record missing')
        item=source[entry['record_id']];row=item['row'];validate_model_boundary(row,'train')
        require(row['ability']=='mri_sequence' and row['extra_info']['patient_key']==entry['patient_key'],'Schedule patient/route mismatch')
        require(row['reward_model']['ground_truth']==entry['target_label'],'Schedule target mismatch')
        require(hashlib.sha256(row['extra_info']['question'].encode()).hexdigest()==entry['question_sha256'],'Schedule question mismatch')
        require(sha(row['images'][0])==entry['image_sha256'],'Schedule image byte mismatch')
        selected.append(item)
    require(len({x['row']['id'] for x in selected})==len(selected),'Duplicated frozen sequence ID')
    return selected


def check_split_keys(train,dev):
    def keys(rows):
        patients=set();clusters=set()
        for row in rows:
            source=row['data_source'];e=row['extra_info']
            patients.add((source,e['patient_key']))
            if e.get('candidate_patient_key'):patients.add((source,e['candidate_patient_key']))
            if e.get('statistical_cluster_key'):clusters.add((source,e['statistical_cluster_key']))
        return patients,clusters
    a,b=keys(train),keys(dev)
    require(not a[0]&b[0] and not a[1]&b[1],'Source patient/candidate/cycle split overlap')


def write_exact(path,content):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():require(path.read_bytes()==content,'Frozen output already differs');return
    temporary=path.with_suffix(path.suffix+'.partial')
    with temporary.open('wb') as f:f.write(content);f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)


def assemble(sequence_train,sequence_dev,schedule,b28_receipt,sequence_contract,dallas_train,dallas_dev,destination):
    contract=json.loads(Path(sequence_contract).read_text());completed=json.loads(Path(b28_receipt).read_text())
    require(sha(sequence_train)==contract['train_manifest_sha256'] and sha(sequence_dev)==contract['dev_manifest_sha256'],'Historical sequence contract source hashes differ')
    schedule_obj=json.loads(Path(schedule).read_text());inventory=schedule_obj['rows'];schedule_sha=canonical_sha(inventory)
    require(schedule_obj['sha256']==completed['schedule_sha256']==schedule_sha,'Completed B28 schedule hash mismatch')
    require(completed['schedule_policy']=='balanced_sequence_v2' and completed['completed_control_28steps'] is True
            and completed['stage_end_step']==28,'Exact completed balanced B28 evidence required')
    seq_train=extract_frozen_sequence(sequence_train,inventory)
    require(len(seq_train)==28 and Counter(x['row']['reward_model']['ground_truth'] for x in seq_train)=={'FLAIR':14,'contrast-enhanced T1':14},'Expected exact balanced 14/14 sequence rows')
    seq_dev=[x for x in read_rows(sequence_dev) if x['row']['ability']=='mri_sequence']
    require(len(seq_dev)==24 and len({x['row']['extra_info']['patient_key'] for x in seq_dev})==4,'Fixed 24-case four-patient sequence dev required')
    dal_train=read_rows(dallas_train);dal_dev=read_rows(dallas_dev)
    require(len(dal_train)==48 and len(dal_dev)==16,'Frozen Dallas counts differ')
    merged={'train':seq_train+dal_train,'internal_dev':seq_dev+dal_dev}
    image_inventory={};source_counts={}
    for split,items in merged.items():
        ids=set()
        for item in items:
            row=item['row'];validate_model_boundary(row,split)
            require(row['id'] not in ids,'Merged source record ID collision');ids.add(row['id'])
            image_inventory[row['id']]=sha(row['images'][0])
        source_counts[split]=dict(Counter(x['row']['ability'] for x in items))
    check_split_keys([x['row'] for x in merged['train']],[x['row'] for x in merged['internal_dev']])
    destination=Path(destination)
    for split,items in merged.items():write_exact(destination/(split+'.verl.jsonl'),b''.join(x['raw'] for x in items))
    paths={'sequence_train':sequence_train,'sequence_dev':sequence_dev,'completed_B28_schedule':schedule,'completed_B28_receipt':b28_receipt,
           'sequence_contract':sequence_contract,'dallas_train':dallas_train,'dallas_dev':dallas_dev}
    safe={'status':'C_TWO_TASK_POOL_FROZEN_NO_NEW_MODEL_OUTPUTS','evidence_tier':'OBSERVED_CPU_EXACT_ROW_ASSEMBLY',
          'mission_alignment':'Freeze sequence acquisition recognition plus whole-head cross-visit correspondence inputs for the controlled Vero task-balanced comparison.',
          'manifest_sha256':{split:sha(destination/(split+'.verl.jsonl')) for split in merged},
          'source_file_sha256':{k:sha(v) for k,v in paths.items()},'source_schedule_sha256':schedule_sha,
          'pool_rows':{k:len(v) for k,v in merged.items()},'task_counts':source_counts,
          'sequence_train_labels':{'FLAIR':14,'contrast-enhanced T1':14},
          'source_statistical_units':{'sequence_train_patients':28,'sequence_dev_patients':4,'dallas_train_people':24,'dallas_train_cycles':12,'dallas_dev_people':8,'dallas_dev_cycles':4},
          'proportional_train_weights':{'mri_sequence':28/76,'mri_visit_correspondence':48/76},
          'original_selected_row_bytes_preserved':True,'images_reprocessed':False,'image_bytes_verified':True,
          'native_system_instruction':contract['system_instruction'],'native_assistant_prefix':contract['assistant_prefix'],
          'enable_thinking':contract['enable_thinking'],'source_split_keys_disjoint':True,
          'cross_source_global_person_independence':'NOT_VERIFIED; namespace keys establish within-source separation only; cohort/task effects remain confounded.',
          'internal_dev_status':'Previously used sequence dev; no independent sealed-test or generalization claim.',
          'sealed_test_content_used':False,'new_model_outputs_generated':False,'token_measurements':'PENDING_NATIVE_PROCESSOR_ONLY_SAFE_WINDOW',
          'limitations':['Dallas whole-head correspondence retains face/skull/neck and nonspatial shortcut risks.',
                         'Sequence rows retain exact B28 slice-level limitations; this is a new mixed pool, not correction of anatomical slice sampling.',
                         '76 questions are not 76 independent patients; Dallas two-person cycles are conservative clusters.']}
    private={'source_row_ids':{k:[x['row']['id'] for x in v] for k,v in merged.items()},'image_sha256_by_record':image_inventory}
    write_exact(destination/'row_inventory_PRIVATE.json',(json.dumps(private,indent=2,sort_keys=True)+'\n').encode())
    safe['private_image_inventory_sha256']=sha(destination/'row_inventory_PRIVATE.json')
    write_exact(destination/'pool_receipt_SAFE.json',(json.dumps(safe,indent=2,sort_keys=True)+'\n').encode())
    return safe


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('sequence-train','sequence-dev','schedule','b28-receipt','sequence-contract','dallas-train','dallas-dev','destination'):p.add_argument('--'+flag,type=Path,required=True)
    a=p.parse_args();require(a.destination.resolve().is_relative_to(Path(__file__).resolve().parents[2]/'local_data'),'Task pool must remain private under local_data')
    result=assemble(**vars(a));print(json.dumps({'status':result['status'],'pool_rows':result['pool_rows'],'task_counts':result['task_counts'],'manifest_sha256':result['manifest_sha256']}))


if __name__=='__main__':main()
