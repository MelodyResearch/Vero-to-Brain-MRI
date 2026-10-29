"""TRAIN paired-z045 preparation. No model/native/image decoding dependencies.

Acquisition-modality identity is a source metadata endpoint, not clinical lesion
adjudication. Preparation retains training_authorized=False. The existing full-block
sampler selects source support within frozen historical task/Dallas slots only.
"""
from __future__ import annotations
from collections import Counter,defaultdict
import copy
import hashlib
import json
import re

SEQ='mri_sequence';DALLAS='mri_visit_correspondence'
LABELS=('FLAIR','contrast-enhanced T1')


def require(ok,message):
    if not ok:raise ValueError(message)


def canonical_sha(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def select_z045(reviewed,full,expected_people=28):
    source=[r for r in full if r['ability']==SEQ]
    require(len(source)==len(reviewed)==expected_people*6,'Exact complete three-level two-modality inventory required')
    by_index={r['extra_info']['index']:r for r in source};seen=set();people=defaultdict(set);result=[]
    require(len(by_index)==len(source),'Duplicate source index')
    require(len({r['id'] for r in reviewed})==len(reviewed),'Duplicate opaque row ID')
    for row in reviewed:
        e=row['extra_info'];idx=e['index'];require(idx in by_index and idx not in seen,'Duplicate/missing reviewed source index');seen.add(idx)
        original=by_index[idx];oe=original['extra_info'];q=row.get('sequence_qualification_v1',{})
        require(row['ability']==SEQ and row['data_source']=='OpenNeuro_ds007045','Typed sequence source required')
        require(e==oe and row['reward_model']==original['reward_model'],'Source label/metadata changed')
        require(e['internal_split']=='train' and e['patient_lineage']=='BIDS_SUBJECT_VERIFIED','TRAIN verified lineage required')
        m=re.fullmatch(re.escape(e['patient_key'])+r'_z(045|091|136)_(flair|t1ce)_sequence',original['id'])
        require(m is not None,'Source patient/level/modality identity differs')
        level=int(m[1]);modality=m[2];label={'flair':LABELS[0],'t1ce':LABELS[1]}[modality]
        require(row['reward_model']['ground_truth']==label,'Source modality target mismatch')
        require(q.get('source_pixel_equivalence') is True and q.get('training_authorized') is False,'Frozen source-pixel qualification missing')
        people[e['patient_key']].add((level,modality))
        if level!=45:continue
        require(q.get('nonexpert_technical_coverage_adequate') is True and q.get('blind_review_abstained') is False,'z045 coverage abstention requires exclusion')
        item=copy.deepcopy(row);item['training_authorized']=False
        item['paired_z045_data_v1']={'version':1,'level':45,'source_record_id_PRIVATE':original['id'],
            'modality_PRIVATE':modality,'acquisition_modality_source_eligible':True,'clinical_expert_gate_required':False,
            'qualification_basis':'Pinned acquisition identity plus existing source/pixel qualification;not clinical lesion gold',
            'legacy_expert_pending_field':'Preserved historical nonexpert-review metadata;not a clinical-expert prerequisite for acquisition identity',
            'training_authorized':False}
        result.append(item)
    expected={(z,m) for z in [45,91,136] for m in ['flair','t1ce']}
    require(len(people)==expected_people and all(v==expected for v in people.values()),'Incomplete patient-level source pair inventory')
    result.sort(key=lambda r:r['extra_info']['index'])
    require(len(result)==expected_people*2,'Exact z045 pair coverage differs')
    return result


def replace_sequence_slots(reference,sequence_pool,*,seed=13,sequence_draw_offset=0,sampler_sequence_stream=None,sampler_binding=None):
    """Select only from an actual existing full-block sampler's sequence stream.

    No RNG, permutation, quota or pair-adjacency logic is implemented here. The
    caller computes the existing sampler over the complete two-task source pool.
    Fixed historical category/Dallas slots condition which sequence draws occur.
    """
    require(type(seed) is int and seed>=0 and type(sequence_draw_offset) is int and sequence_draw_offset>=0,'Invalid sampling stage')
    require(reference and sequence_pool,'Nonempty reference and sequence pool required')
    require(sampler_sequence_stream is not None and sampler_binding is not None,'Existing full sampler stream and binding required')
    require(sampler_binding.get('mode')=='uniform' and sampler_binding.get('seed')==seed and sampler_binding.get('maximum_occurrences')==198,'Existing uniform198 seed stage required')
    require(len({r['id'] for r in reference})==len(reference),'Duplicate occurrence ID')
    by_id={r['id']:r for r in sequence_pool};require(len(by_id)==len(sequence_pool),'Duplicate source ID')
    for r in sequence_pool:
        require(r['ability']==SEQ and r['extra_info']['internal_split']=='train' and 'source_record_id' not in r,'Unexpanded TRAIN sequence source rows required')
    for r in reference:
        require(r['ability'] in [SEQ,DALLAS] and r['extra_info']['internal_split']=='train','Typed TRAIN reference slots required')
    for r in sampler_sequence_stream:
        rid=r.get('source_record_id');require(r['ability']==SEQ and rid in by_id,'Sampler references foreign sequence source')
        restored=copy.deepcopy(r);restored['id']=restored.pop('source_record_id')
        require(restored==by_id[rid],'Sampler occurrence source content changed')
    count=sum(r['ability']==SEQ for r in reference)
    selected=sampler_sequence_stream[sequence_draw_offset:sequence_draw_offset+count]
    require(len(selected)==count,'Existing full sampler lacks enough sequence draws')
    it=iter(selected);schedule=[];ordinal=sequence_draw_offset
    for old in reference:
        if old['ability']==DALLAS:schedule.append(copy.deepcopy(old));continue
        sampled=next(it);source=by_id[sampled['source_record_id']];row=copy.deepcopy(source);row['source_record_id']=source['id']
        row['id']=canonical_sha({'stage':'paired-z045-existing-full-sampler-stream-v1','seed':seed,'sequence_ordinal':ordinal,'source_id':source['id']})[:24]
        ordinal+=1;schedule.append(row)
    require(len({r['id'] for r in schedule})==len(schedule),'Occurrence collision')
    binding={'schema':'paired-z045-slot-replacement-v2','source_sampling_stage':{'name':'existing-full-block-category-sampler-v1','seed':seed,'sequence_draw_offset':sequence_draw_offset,
        'law':'Existing build_category_schedule198:128 draws/category,exhaustion reshuffle,whole256-block shuffle;filter sequence then skip prior2;fixed historical task/Dallas mask'},
        'reference_schedule_sha256':canonical_sha(reference),'sequence_pool_sha256':canonical_sha(sequence_pool),
        'full_sampler_binding':copy.deepcopy(sampler_binding),'full_sequence_stream_sha256':canonical_sha(sampler_sequence_stream),
        'schedule_sha256':canonical_sha(schedule),'task_order_unchanged':True,'Dallas_occurrences_content_unchanged':True,'forced_pair_adjacency':False,
        'pool_pairing_does_not_guarantee_finite_pair_exposure':True,'model_or_optimizer_RNG_touched':False,
        'historical_row_RNG_continuity_claimed':False,'new_standalone_sequence_RNG_implemented':False,'training_authorized':False}
    return {'schedule':schedule,'binding':binding}


def exposure_summary(schedule):
    seq=[r for r in schedule if r['ability']==SEQ];by_person=defaultdict(set);by_row=Counter()
    for r in seq:
        by_person[r['extra_info']['patient_key']].add(r['reward_model']['ground_truth']);by_row[r.get('source_record_id',r['id'])]+=1
    return {'task_occurrences':dict(Counter(r['ability'] for r in schedule)),'sequence_occurrences':len(seq),
        'sequence_label_occurrences':dict(Counter(r['reward_model']['ground_truth'] for r in seq)),
        'unique_sequence_rows':len(by_row),'repeated_sequence_occurrences':len(seq)-len(by_row),
        'sequence_people_observed':len(by_person),'people_both_modalities_observed':sum(len(v)==2 for v in by_person.values()),
        'people_one_modality_observed':sum(len(v)==1 for v in by_person.values()),
        'maximum_repeats_per_sequence_row':max(by_row.values(),default=0)}
