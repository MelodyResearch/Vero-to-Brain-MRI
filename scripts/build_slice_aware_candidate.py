"""Training-only candidate C; mask geometry eligibility is not visual adjudication."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path

LEVELS=(45,91,136)


def match_slots(options, slots):
    """Deterministic bipartite match: one patient per slot, no replacement."""
    patient_slot={};selected={}
    def assign(i,seen):
        for row in sorted(options[slots[i]],key=lambda r:hashlib.sha256(('C-v1:'+r['record_id']).encode()).hexdigest()):
            p=row['patient']
            if p in seen:continue
            seen.add(p)
            old=patient_slot.get(p)
            if old is None or assign(old,seen):
                patient_slot[p]=i;selected[i]=row;return True
        return False
    for i in sorted(range(len(slots)),key=lambda i:len(options[slots[i]])):
        if not assign(i,set()):return None
    return [selected[i] for i in range(len(slots))]


def binary_schedule(rows,category):
    area='enhancing' if category=='mri_enhancing_presence' else 'tumor'
    options=defaultdict(list)
    for r in rows:
        if r['category']==category and ((r['label']=='no' and r[area]==0) or (r['label']=='yes' and r[area]>=1000)):
            options[(r['level'],r['label'])].append(r)
    # Prespecified selection objective: closest to equal level counts, exact14/14
    # labels and within-level class equality. Metadata only, never model scores.
    quotas=[(a,b,14-a-b) for a in range(1,13) for b in range(1,14-a) if 14-a-b>=1]
    quotas.sort(key=lambda q:(sum((x-14/3)**2 for x in q),q))
    for q in quotas:
        slots=[(z,label) for z,n in zip(LEVELS,q) for label in ('no','yes') for _ in range(n)]
        matched=match_slots(options,slots)
        if matched is not None:return matched,q
    return [],None


def build(references):
    seq=[r for r in references if r['category']=='mri_sequence']
    patients=sorted({r['patient'] for r in seq},key=lambda p:hashlib.sha256(('7:mri_sequence:'+p).encode()).hexdigest())
    if len(patients)!=28:raise ValueError('Expected exact28training patients')
    offset=int(hashlib.sha256(b'7:sequence-label-offset').hexdigest(),16)%2
    out=[];status={}
    for i,p in enumerate(patients):
        label=('FLAIR','contrast-enhanced T1')[(i+offset)%2];z=LEVELS[i%3]
        choices=[r for r in seq if r['patient']==p and r['level']==z and r['label']==label]
        if len(choices)!=1:raise ValueError('Missing unique sequence candidate')
        out+=choices
    for category in ('mri_tumor_presence','mri_enhancing_presence'):
        matched,q=binary_schedule(references,category);out+=matched
        status[category]={'matched':len(matched),'within_level_balanced_pair_quotas':q,
            'status':'GEOMETRY_CANDIDATE_VISIBILITY_PENDING' if matched else 'NO_FEASIBLE_BALANCED_ASSIGNMENT'}
    grid=[];level_counts=Counter();label_counts=Counter()
    for p in patients:
        choices=[r for r in references if r['patient']==p and r['category']=='mri_tumor_grid' and r['tumor']>=1000
                 and r['grid_boundary_distance_display_pixels'] is not None and r['grid_boundary_distance_display_pixels']>=10]
        if choices:
            r=min(choices,key=lambda r:(level_counts[r['level']],label_counts[r['label']],hashlib.sha256(r['record_id'].encode()).hexdigest()))
            grid.append(r);level_counts[r['level']]+=1;label_counts[r['label']]+=1
    out+=grid;status['mri_tumor_grid']={'matched':len(grid),'status':'ENGINEERING_ONLY_EXACT_MASK_CENTROID','class_balance_guaranteed':False}
    summaries={c:{'n':len(g),'patients':len({r['patient'] for r in g}),
                  'labels':dict(Counter(r['label'] for r in g)),'levels':dict(Counter(r['level'] for r in g))}
               for c in sorted({r['category'] for r in out}) for g in [[r for r in out if r['category']==c]]}
    return {'version':'slice_label_candidate_v1','source':'train-only geometry references',
            'positive_geometry_min_native_pixels':1000,'grid_min_display_boundary_distance':10,
            'not_a_new_clinical_gold_rule':True,'visual_adjudication_complete':False,'gpu_admitted':False,
            'status':status,'by_category':summaries,'rows_PRIVATE':out,
            'intervention':'Changes sequence images acrosslevels AND other taskimage exposure/eligibility; notlabel-only B'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--references',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    private=Path(__file__).resolve().parents[1]/'local_data'
    if not a.output.resolve().is_relative_to(private) or a.output.exists():raise ValueError('New privateoutput required')
    result=build(json.loads(a.references.read_text())['training_references_PRIVATE'])
    a.output.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='rows_PRIVATE'},indent=2))
