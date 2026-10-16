"""C1 v2: exact patient-distinct, within-level balanced training-only subsets.

A geometry eligibility probe, not clinical/visual adjudication or a training gate.
No model, MLX, MRI-array or held-out outcome imports.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
from itertools import product
import json
from pathlib import Path
import re
import subprocess

LEVELS=(45,91,136)
SOURCE_COMMIT='283007b96977b5b3285bd22cd67b7d97114b1eed'


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def matching(options,slots):
    patient_slot={}; selected={}
    def assign(i,seen):
        for row in sorted(options[slots[i]],key=lambda r:hashlib.sha256(('C1-v2:'+r['record_id']).encode()).hexdigest()):
            p=row['patient']
            if p in seen:continue
            seen.add(p);old=patient_slot.get(p)
            if old is None or assign(old,seen):
                patient_slot[p]=i;selected[i]=row;return True
        return False
    for i in sorted(range(len(slots)),key=lambda i:len(options[slots[i]])):
        if not assign(i,set()):return None
    return [selected[i] for i in range(len(slots))]


def maximum_subset(rows,category):
    field='enhancing' if category=='mri_enhancing_presence' else 'tumor'
    options=defaultdict(list);inconsistent=0
    for r in rows:
        if r['category']!=category:continue
        if r['label'] not in ('yes','no'):raise ValueError('Nonbinary label')
        area=r[field]
        if (r['label']=='yes') != (area>0):inconsistent+=1;continue
        if r['level'] not in LEVELS:raise ValueError('Unexpected slice level')
        if r['label']=='no' or area>=1000:options[r['level'],r['label']].append(r)
    supported=[z for z in LEVELS if options[z,'yes'] and options[z,'no']]
    capacity={z:min(len({r['patient'] for r in options[z,label]}) for label in ('yes','no')) for z in supported}
    candidates=[]
    if len(supported)>=2:
        cap=len({r['patient'] for r in rows if r['category']==category})//2
        candidates=[q for q in product(*(range(1,capacity[z]+1) for z in supported)) if sum(q)<=cap]
        candidates.sort(key=lambda q:(-sum(q),sum((x-sum(q)/len(q))**2 for x in q),q))
    attempted=0;selected=[];chosen=None
    for q in candidates:
        attempted+=1
        slots=[(z,label) for z,n in zip(supported,q) for label in ('no','yes') for _ in range(n)]
        found=matching(options,slots)
        if found is not None:selected=found;chosen=q;break
    return {'category':category,'supported_levels':supported,'excluded_levels':[z for z in LEVELS if z not in supported],
            'eligibility_counts':{str(z):{label:len(options[z,label]) for label in ('no','yes')} for z in LEVELS},
            'excluded_inconsistent_label':inconsistent,'pair_quotas':chosen,'quotas_tested':attempted,
            'maximum_proven':True,'maximum_scope':'All currently supported levels required; exact within-level balance; distinct patients; area threshold1000',
            'selected':selected}


def audit_prompt(row):
    texts=[]
    for message in row.get('prompt',[]):
        content=message.get('content','')
        if isinstance(content,str):texts.append(content)
        else:texts.extend(item.get('text','') for item in content if item.get('type')=='text')
    prompt='\n'.join(texts)
    reasons=[]
    for value in (row.get('id'),row.get('extra_info',{}).get('patient_key')):
        if value and value in prompt:reasons.append('private_identifier_in_prompt')
    if re.search(r'ground_truth|reward_model|tumor_pixels|enhancing_pixels|mask_area',prompt,re.I):reasons.append('gold_metadata_field_in_prompt')
    return {'pass':not reasons,'reasons':reasons,
            'scope':'model-visible text components only; image transport locators excluded, runner boundary independently checked elsewhere'}


def summarize(rows):
    def majority(group):return sorted(Counter(r['label'] for r in group).items(),key=lambda x:(-x[1],x[0]))[0][0]
    def oracle(fields):
        buckets=defaultdict(list)
        for r in rows:buckets[tuple(r[k] for k in fields)].append(r)
        return sum(max(Counter(r['label'] for r in g).values()) for g in buckets.values())
    def lopo(fields):
        correct=0
        for r in rows:
            training=[s for s in rows if s['patient']!=r['patient']]
            same=[s for s in training if all(s[k]==r[k] for k in fields)]
            pred=majority(same or training);correct+=pred==r['label']
        return correct
    n=len(rows)
    return {'n':n,'patients':len({r['patient'] for r in rows}),'labels':dict(Counter(r['label'] for r in rows)),
            'levels':dict(Counter(r['level'] for r in rows)),'sites':dict(Counter(r['site'] for r in rows)),
            'site_label_level':dict(Counter(f"{r['site']}|{r['label']}|{r['level']}" for r in rows)),
            'baselines':{'fixed_no_correct':sum(r['label']=='no' for r in rows),
                         'level_resubstitution_correct':oracle(['level']),
                         'site_resubstitution_correct':oracle(['site']),
                         'site_level_resubstitution_correct':oracle(['site','level']),
                         'site_lopo_correct':lopo(['site']), 'site_level_lopo_correct':lopo(['site','level']),
                         'denominator':n,'model_no_image':'NOT_RUN',
                         'interpretation':'Resubstitution measures metadata association, not generalization. LOPO descriptive and unstable with tiny strata.'}} if rows else {'n':0}


def broader_metadata(metadata,existing,reserved):
    def git(*args):return subprocess.check_output(['git','-C',str(metadata),*args],text=True)
    paths=set(git('ls-tree','-r','--name-only',SOURCE_COMMIT).splitlines())
    subjects=[line.split('\t')[0] for line in git('show',f'{SOURCE_COMMIT}:participants.tsv').splitlines()[1:] if line.strip()]
    if len(subjects)!=len(set(subjects)) or set(existing)&set(reserved):raise ValueError('Invalid exclusion inventory')
    excluded=set(existing)|set(reserved); candidates=[]
    for subject in subjects:
        if subject in excluded:continue
        site=re.match(r'sub-([A-Z]+)',subject).group(1)
        required=[f'derivatives/processing_MNI152_skull-stripped_N4/processing_MNI152_skull-stripped_N4/{subject}/{subject}_{modality}_MNI152_brain_n4.nii.gz' for modality in ('flair','t1ce')]
        required += [f'derivatives/segmentation/{subject}/{subject}_MNI152_tumor-mask.nii.gz']
        if not all(p in paths for p in required+[f'{subject}/anat/{subject}_FLAIR.json',f'{subject}/anat/{subject}_run-02_T1w.json']):continue
        payloads=[]
        for path in required:
            pointer=git('show',f'{SOURCE_COMMIT}:{path}')
            hit=re.search(r'SHA256E-s(\d+)--([a-f0-9]{64})',pointer)
            if not hit:raise ValueError('Expected checksum-bound annex pointer')
            payloads.append({'path':path,'bytes':int(hit[1]),'sha256':hit[2]})
        candidates.append({'subject_PRIVATE':subject,'site':site,'payloads':payloads,
                           'selection_hash':hashlib.sha256(f'C2-train-v2:{SOURCE_COMMIT}:{subject}'.encode()).hexdigest()})
    selected=[]
    for site in sorted({c['site'] for c in candidates}):
        selected += sorted([c for c in candidates if c['site']==site],key=lambda c:c['selection_hash'])[:2]
    return {'version':'C2_training_metadata_v2','source_commit':SOURCE_COMMIT,'existing_excluded':len(existing),
            'reserved_excluded':len(reserved),'available':len(candidates),'site_counts':dict(Counter(c['site'] for c in candidates)),
            'selected_count':len(selected),'selected_sites':dict(Counter(c['site'] for c in selected)),
            'selected_payload_bytes':sum(p['bytes'] for c in selected for p in c['payloads']),
            'selected_mask_bytes':sum(c['payloads'][2]['bytes'] for c in selected),
            'selected_PRIVATE':selected,'status':'METADATA_ONLY_NOT_DOWNLOADED',
            'selection_rule':'up to2 per site, SHA256 deterministic; excludes all40existing and14prospective',
            'limitation':'Metadata does not establish positive mask area or visibility; inspect masks before corresponding image downloads.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--metadata-repo',type=Path)
    a=p.parse_args();private=Path(__file__).resolve().parents[1]/'local_data'
    if not a.output.resolve().is_relative_to(private) or a.output.exists():raise ValueError('New privateoutput required')
    refs_path=a.root/'safe_resume_v1/historical112_sampling_v2_PRIVATE.json'
    refs=json.loads(refs_path.read_text())['training_references_PRIVATE']
    cases=json.loads((a.root/'visual_prompts/private_manifest.json').read_text())['private_cases']
    sites={c['anonymous_token']:c['private_site'] for c in cases}
    records=[json.loads(line) for line in (a.root/'vero_training_v1/train.verl.jsonl').read_text().splitlines() if line.strip()]
    train_patients={r['extra_info']['patient_key'] for r in records}
    if len(train_patients)!=28 or {r['patient'] for r in refs}!=train_patients:raise ValueError('Training patient inventory differs')
    for r in refs:r['site']=sites[r['patient']]
    result={'version':'C1_matched_subset_v2','references_sha256':sha(refs_path),'train_sha256':sha(a.root/'vero_training_v1/train.verl.jsonl'),
            'source_code_sha256':sha(__file__),'status':'TESTED_CPU','task_admission':'ENGINEERING_ONLY',
            'visual_adjudication_complete':False,'positive_area_threshold_native_pixels':1000,
            'heldout_outcomes_used':False,'categories':{},'prompt_audit':{'rows':len(records),'failures':sum(not audit_prompt(r)['pass'] for r in records)}}
    for category in ('mri_tumor_presence','mri_enhancing_presence'):
        solution=maximum_subset(refs,category);selected=solution.pop('selected')
        solution['summary']=summarize(selected);solution['selected_PRIVATE']=selected;result['categories'][category]=solution
    if a.metadata_repo:
        reserved=json.loads((a.root/'balanced_sft_independent_audit_v1/prospective_metadata_selection_PRIVATE.json').read_text())['selected_PRIVATE']
        result['C2']=broader_metadata(a.metadata_repo,[c['private_subject'] for c in cases],[c['subject_PRIVATE'] for c in reserved])
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    public=json.loads(json.dumps(result))
    for c in public['categories'].values():c.pop('selected_PRIVATE')
    if 'C2' in public:public['C2'].pop('selected_PRIVATE')
    print(json.dumps(public,indent=2))


if __name__=='__main__':main()
