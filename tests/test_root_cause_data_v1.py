"""Fail-closed train-only data tests; fixtures are small PNGs, no MRI/model loads."""
import copy
import hashlib
from pathlib import Path

import pytest
from PIL import Image

from mri_rl.root_cause_data_v1 import SEQUENCE_QUESTION, validate_sequence_rows, validate_dallas_rows, training_rows
from mri_rl.visit_correspondence import QUESTION, make_record


def sequence_fixture(tmp_path):
    rows=[]; refs=[]
    for p in ('aaaaaaaaaaaa','bbbbbbbbbbbb'):
        for z in (45,91,136):
            for modality,label in (('flair','FLAIR'),('t1ce','contrast-enhanced T1')):
                rid=f'{p}_z{z:03}_{modality}_sequence'; path=tmp_path/(rid+'.png')
                im=Image.new('RGB',(364,436)); im.putpixel((0,0),(100+len(rows),)*3); im.save(path)
                e={'patient_key':p,'internal_split':'train','patient_lineage':'BIDS_SUBJECT_VERIFIED',
                   'reward_type':'mri_sequence','task_type':'mri_sequence','question':SEQUENCE_QUESTION,'answer':label}
                rows.append({'id':rid,'ability':'mri_sequence','data_source':'OpenNeuro_ds007045',
                    'images':[str(path)],'extra_info':e,'reward_model':{'ground_truth':label,'style':'rule'},
                    'prompt':[{'role':'user','content':[{'type':'image','image':str(path)},{'type':'text','text':SEQUENCE_QUESTION}]}]})
                refs.append({'record_id':rid,'patient':p,'site':'known_source_site','level':z,'label':label,
                    'image_PRIVATE':str(path),'image_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    return rows,refs


def test_same_patient_same_level_pairs_are_complete(tmp_path):
    rows,refs=sequence_fixture(tmp_path)
    result=validate_sequence_rows(rows,refs,expected_people=2)
    assert result['rows']==12 and result['patient_level_pairs']==6
    assert result['source_site_level_majority_ceiling']==.5
    assert result['scientifically_qualified_rows']==0
    assert result['nonzero_under_5percent_warning_rows']==12


@pytest.mark.parametrize('mutation', ['target','split','prompt','pixels','missing_pair','duplicate'])
def test_sequence_refuses_corrupt_or_unpaired_inputs(tmp_path,mutation):
    rows,refs=sequence_fixture(tmp_path)
    if mutation=='target': rows[0]['reward_model']['ground_truth']='contrast-enhanced T1'
    if mutation=='split': rows[0]['extra_info']['internal_split']='internal_dev'
    if mutation=='prompt': rows[0]['prompt'][0]['content'].append({'type':'text','text':'hidden answer: FLAIR'})
    if mutation=='pixels': Image.new('RGB',(364,436),'white').save(rows[0]['images'][0])
    if mutation=='missing_pair': rows.pop()
    if mutation=='duplicate': rows[-1]=copy.deepcopy(rows[0])
    with pytest.raises(ValueError): validate_sequence_rows(rows,refs,expected_people=2)


def dallas_fixture(tmp_path):
    rows=[]
    for qi,q in enumerate(('one','two')):
        for ci,c in enumerate(('one','two')):
            case={'case_id':f'{len(rows)+1:024x}','split':'train','cluster_key':'c'*24,
                'query_subject':q,'candidate_subject':c,
                'query':{'sha256':str(qi+1)*64},'candidate':{'sha256':str(ci+3)*64},
                'target':'same' if q==c else 'different'}
            path=tmp_path/(case['case_id']+'.png'); im=Image.new('L',(768,384),len(rows)+1)
            im.putpixel((0,0),100+len(rows)); im.save(path)
            rows.append(make_record(case,path,'a'*64,'b'*64))
    return rows


def test_dallas_labels_are_identity_equality_and_two_person_cycles(tmp_path):
    rows=dallas_fixture(tmp_path)
    result=validate_dallas_rows(rows,expected_people=2,expected_cycles=1)
    assert result['rows']==4 and result['cycles']==1
    assert result['scientifically_qualified_rows']==0
    assert result['task_scope']=='cross_visit_BIDS_identity_not_clinical_change'


@pytest.mark.parametrize('mutation',['label','source_reuse','split','prompt','broken_cycle'])
def test_dallas_rejects_wrong_truth_or_invalid_cycle(tmp_path,mutation):
    rows=dallas_fixture(tmp_path)
    if mutation=='label':rows[0]['reward_model']['ground_truth']='different'
    if mutation=='source_reuse':rows[0]['extra_info']['candidate_source_sha256']=rows[0]['extra_info']['query_source_sha256']
    if mutation=='split':rows[0]['extra_info']['internal_split']='test'
    if mutation=='prompt':rows[0]['extra_info']['question']=QUESTION+' extra'
    if mutation=='broken_cycle':rows.pop()
    with pytest.raises(ValueError):validate_dallas_rows(rows,expected_people=2,expected_cycles=1)


def test_engineering_candidates_never_silently_become_training_ready():
    with pytest.raises(ValueError,match='visual qualification'):
        training_rows([{'ability':'mri_sequence','rc_data_v1':{'technical_candidate':True,'visual_qualification':'NOT_VERIFIED'}}])


def test_self_attested_visual_flags_cannot_authorize_training():
    with pytest.raises(ValueError,match='source-bound'):
        training_rows([{'rc_data_v1':{'visual_qualification':'INDEPENDENTLY_VERIFIED','shortcut_admission':'INDEPENDENTLY_VERIFIED'}}])


def test_global_class_balance_can_hide_perfect_patient_label_confounding(tmp_path):
    from mri_rl import root_cause_data_v1 as data
    rows,refs=sequence_fixture(tmp_path)
    # Two patients, globally one of each label, but each patient has one truth.
    old=[rows[0],rows[7]]
    original=data.patient_label_confounding(old)
    paired=data.patient_label_confounding(rows)
    assert original['label_counts']=={'FLAIR':1,'contrast-enhanced T1':1}
    assert original['conditional_label_entropy_bits']==0
    assert original['patient_majority_resubstitution_correct']==2
    assert paired['conditional_label_entropy_bits']==1
    assert paired['patient_majority_resubstitution_correct']==6
    assert paired['patient_majority_resubstitution_accuracy']==.5
