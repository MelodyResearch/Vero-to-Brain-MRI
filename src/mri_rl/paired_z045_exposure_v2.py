"""CPU-only finite exposure controls. Forced pairing is an explicit intervention.

No model, loss, sampler-RNG, image decoding or training dependencies. These are
prospective schedules, not evidence any image received a model update.
"""
from collections import defaultdict
import copy
import hashlib
from mri_rl.paired_z045_data_v1 import (
    SEQ, DALLAS, LABELS, canonical_sha, exposure_summary, require,
)

SALT='paired-z045-patient-order-v2'


def patient_order(pool,sites):
    groups=defaultdict(dict)
    for row in pool:
        person=row['extra_info']['patient_key'];label=row['reward_model']['ground_truth']
        require(row['ability']==SEQ and row['extra_info']['internal_split']=='train' and
                label in LABELS and label not in groups[person], 'Unique TRAIN modality rows required')
        groups[person][label]=row
    require(len(pool)==56 and len(groups)==28 and all(set(g)==set(LABELS) for g in groups.values()),
            'Exactly28 complete existing pairs required')
    require(set(groups)==set(sites), 'Authenticated TRAIN site join required')
    bysite=defaultdict(list)
    for person in groups:bysite[sites[person]].append(person)
    rank=lambda s:hashlib.sha256((SALT+':'+s).encode()).hexdigest()
    for site in bysite:bysite[site].sort(key=rank)
    ordered=[]
    for index in range(max(map(len,bysite.values()))):
        for site in sorted(bysite):
            if index<len(bysite[site]):ordered.append(bysite[site][index])
    return ordered,groups


def pair_schedule(reference,pool,sites):
    """Preserve every task/Dallas slot; pair adjacent SEQUENCE occurrences.

    Dallas can occur between paired sequence slots. This is not unchanged Vero
    source sampling or a claim of old row-RNG continuity.
    """
    order,groups=patient_order(pool,sites)
    n=sum(r['ability']==SEQ for r in reference)
    require(0<n<=56 and n%2==0, 'Even finite sequence exposure within existing56 required')
    stream=[]
    for index,person in enumerate(order[:n//2]):
        # Alternate within-pair orientation independently of labels/results.
        labels=LABELS if index%2==0 else tuple(reversed(LABELS))
        for label in labels:stream.append(groups[person][label])
    result=[];index=0
    for position,old in enumerate(reference):
        require(old['ability'] in (SEQ,DALLAS), 'Only unchanged two tasks allowed')
        if old['ability']==DALLAS:result.append(copy.deepcopy(old));continue
        row=copy.deepcopy(stream[index]);row['source_record_id']=row['id']
        row['id']=canonical_sha({'version':'paired-z045-forced-pair-v2','slot':position,
                                'source':row['source_record_id']})[:24]
        result.append(row);index+=1
    return result


def describe(schedule,token_cap):
    require(type(token_cap) is int and token_cap>0,'Explicit declared token envelope required')
    prompt=sum(4*({SEQ:213,DALLAS:405}[r['ability']]) for r in schedule)
    response=len(schedule)*4*256
    return {'attempted_group_slots':len(schedule),'exposure':exposure_summary(schedule),
            'unique_sequence_image_paths':len({p for r in schedule if r['ability']==SEQ for p in r['images']}),
            'paired_patient_fraction_of_existing28':exposure_summary(schedule)['people_both_modalities_observed']/28,
            'prompt_visual_reservation':prompt,'returned_response_reservation':response,
            'charged_token_reservation':prompt+response,'declared_token_cap':token_cap,
            'fits_declared_cap':prompt+response<=token_cap,
            'reservation_qualification':'Conditional on actual expanded213/405 input counts; existing4-response256 policy. Installed unreturned AR lookahead recorded separately.',
            'training_executed':False,'actual_optimizer_updates':0}
