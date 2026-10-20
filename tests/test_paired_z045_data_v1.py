"""Pool intervention and task-slot invariants; no model or image fixtures."""
import copy
from collections import Counter
import numpy as np
import pytest
from mri_rl.paired_z045_data_v1 import select_z045,replace_sequence_slots,exposure_summary


def inventories(n=2):
    full=[];reviewed=[]
    for p in range(n):
        for z in [45,91,136]:
            for m,label in [('flair','FLAIR'),('t1ce','contrast-enhanced T1')]:
                idx=len(full);r={'id':f'p{p}_z{z:03d}_{m}_sequence','ability':'mri_sequence','data_source':'OpenNeuro_ds007045',
                    'extra_info':{'index':idx,'patient_key':f'p{p}','internal_split':'train','patient_lineage':'BIDS_SUBJECT_VERIFIED','question':'which?','reward_type':'mri_sequence'},
                    'reward_model':{'ground_truth':label},'images':[f'/source/{idx}.png'],'prompt':['same question']}
                full.append(r);r=copy.deepcopy(r);r['id']=f'opaque{idx}';r['images']=[f'/opaque/{idx}.png'];r['sequence_qualification_v1']={'source_pixel_equivalence':True,'nonexpert_technical_coverage_adequate':True,'blind_review_abstained':False,'training_authorized':False};reviewed.append(r)
    return reviewed,full


def test_only_exact_z045_complete_opposite_class_pairs_selected_by_source_index():
    reviewed,full=inventories();new=select_z045(reviewed,full,expected_people=2)
    assert len(new)==4
    assert {r['paired_z045_data_v1']['level'] for r in new}=={45}
    assert {r['extra_info']['index'] for r in new}=={0,1,6,7}
    assert all(r['training_authorized'] is False for r in new)
    assert all(r['paired_z045_data_v1']['acquisition_modality_source_eligible'] for r in new)
    assert all(r['paired_z045_data_v1']['clinical_expert_gate_required'] is False for r in new)


@pytest.mark.parametrize('defect',['missing','duplicate','nontrain','wronglabel','unverified_pixels'])
def test_incomplete_or_corrupt_source_inventory_fails_closed(defect):
    reviewed,full=inventories()
    if defect=='missing':reviewed.pop()
    if defect=='duplicate':reviewed[-1]=copy.deepcopy(reviewed[0])
    if defect=='nontrain':reviewed[0]['extra_info']['internal_split']='dev'
    if defect=='wronglabel':reviewed[0]['reward_model']['ground_truth']='contrast-enhanced T1'
    if defect=='unverified_pixels':reviewed[0]['sequence_qualification_v1']['source_pixel_equivalence']=False
    with pytest.raises(ValueError):select_z045(reviewed,full,expected_people=2)


def reference():
    return [{'id':f'occ{i}','source_record_id':f'old{i}','ability':'mri_sequence' if i%2==0 else 'mri_visit_correspondence','extra_info':{'internal_split':'train'}} for i in range(12)]


def sampler_stream(pool,indices=None):
    indices=indices if indices is not None else [0,0,2,1,3,2,0,1]
    result=[]
    for i in indices:
        row=copy.deepcopy(pool[i%len(pool)]);row['source_record_id']=row['id'];row['id']='sample'+str(len(result));result.append(row)
    return result


def apply(ref,pool,offset=0,stream=None):
    return replace_sequence_slots(ref,pool,seed=13,sequence_draw_offset=offset,
        sampler_sequence_stream=sampler_stream(pool) if stream is None else stream,
        sampler_binding={'mode':'uniform','seed':13,'maximum_occurrences':198})


def test_replacement_preserves_task_slots_and_dallas_content_exactly():
    pool=select_z045(*inventories(),expected_people=2);ref=reference();out=apply(ref,pool)
    assert [r['ability'] for r in out['schedule']]==[r['ability'] for r in ref]
    assert all(a==b for a,b in zip(out['schedule'],ref) if b['ability']=='mri_visit_correspondence')
    assert Counter(r['ability'] for r in out['schedule'])==Counter(r['ability'] for r in ref)
    assert len({r['id'] for r in out['schedule']})==len(ref)
    assert out==apply(ref,pool)


def test_existing_full_block_sequence_stream_preserves_early_duplicates():
    pool=select_z045(*inventories(),expected_people=2);stream=sampler_stream(pool)
    out=apply(reference(),pool,stream=stream)
    assert [r['source_record_id'] for r in out['schedule'] if r['ability']=='mri_sequence']==[r['source_record_id'] for r in stream[:6]]
    assert exposure_summary(out['schedule'])['repeated_sequence_occurrences']==2
    assert out['binding']['new_standalone_sequence_RNG_implemented'] is False


def test_sequence_substream_skips_prior_two_actual_occurrences():
    pool=select_z045(*inventories(),expected_people=2);stream=sampler_stream(pool)
    out=apply(reference(),pool,offset=2,stream=stream)
    assert [r['source_record_id'] for r in out['schedule'] if r['ability']=='mri_sequence']==[r['source_record_id'] for r in stream[2:8]]
    assert out['binding']['source_sampling_stage']['sequence_draw_offset']==2


def test_nontrain_schedule_and_already_expanded_pool_rejected():
    pool=select_z045(*inventories(),expected_people=2);ref=reference();ref[0]['extra_info']['internal_split']='test'
    with pytest.raises(ValueError):apply(ref,pool)
    pool[0]['source_record_id']='bad'
    with pytest.raises(ValueError):apply(reference(),pool)


def test_missing_full_sampler_stream_cannot_silently_create_alternate_rng():
    pool=select_z045(*inventories(),expected_people=2)
    with pytest.raises(ValueError):replace_sequence_slots(reference(),pool,seed=13)


def test_sampler_stream_foreign_or_changed_source_content_rejected():
    pool=select_z045(*inventories(),expected_people=2);stream=sampler_stream(pool)
    stream[0]['reward_model']['ground_truth']='forged'
    with pytest.raises(ValueError):apply(reference(),pool,stream=stream)
