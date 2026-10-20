import copy
import pytest
from mri_rl import paired_z045_exposure_v2 as p


def source(person, label):
    return {'id': person+'-'+label, 'ability': 'mri_sequence',
            'extra_info': {'patient_key': person, 'internal_split': 'train'},
            'reward_model': {'ground_truth': label}, 'images': [person+'-'+label+'.png']}


def fixture():
    pool=[source('p'+str(i),label) for i in range(28) for label in p.LABELS]
    mask=[{'id':'d'+str(i),'ability':p.DALLAS,'extra_info':{'internal_split':'train'}}
          if i%3==0 else {'id':'s'+str(i),'ability':p.SEQ,'extra_info':{'internal_split':'train'}} for i in range(27)]
    #18sequence and9Dallas; non32 length tests function independently.
    sites={'p'+str(i):'site'+str(i%8) for i in range(28)}
    return mask,pool,sites


def test_same_task_slots_dallas_bytes_and_nine_actual_pairs():
    old,pool,sites=fixture();before=copy.deepcopy((old,pool))
    new=p.pair_schedule(old,pool,sites)
    assert [r['ability'] for r in new]==[r['ability'] for r in old]
    assert [r for r in new if r['ability']==p.DALLAS]==[r for r in old if r['ability']==p.DALLAS]
    seq=[r for r in new if r['ability']==p.SEQ]
    assert len({r['source_record_id'] for r in seq})==18
    assert all(seq[i]['extra_info']['patient_key']==seq[i+1]['extra_info']['patient_key'] for i in range(0,18,2))
    assert len({sites[r['extra_info']['patient_key']] for r in seq[:16]})==8
    assert (old,pool)==before


def test_prefix_coverage_does_not_claim_full_pairs_after_early_update_stop():
    mask,pool,sites=fixture();schedule=p.pair_schedule(mask,pool,sites)
    early=p.describe(schedule[:8],70784)
    full=p.describe(schedule,70784)
    assert early['exposure']['people_both_modalities_observed']<9
    assert full['exposure']['people_both_modalities_observed']==9
    assert full['charged_token_reservation']<70784
    assert full['training_executed'] is False


def test_budget_reserves_allfour_fullresponses_and_visualinputs():
    rows=[source('a',p.LABELS[0]),{'id':'d','ability':p.DALLAS}]
    x=p.describe(rows,10000)
    assert x['charged_token_reservation']==4*(213+405)+2*4*256


def test_incomplete_pair_rejected_and_no_synthetic_modality_added():
    mask,pool,sites=fixture()
    with pytest.raises(ValueError):p.pair_schedule(mask,pool[:-1],sites)
