"""The unified bundle is a candidate data schema, never training permission."""
import copy
import hashlib
import importlib.util
from pathlib import Path
import pytest

P=Path(__file__).resolve().parents[1]/'scripts/assemble_train_qualification_bundle_v1.py'
spec=importlib.util.spec_from_file_location('bundle_assembler',P)
assembler=importlib.util.module_from_spec(spec)
spec.loader.exec_module(assembler)


def record(rid,task,index):
    image=f'/private/fixture/{rid}.png';question='Fixed MRI question'
    return {'id':rid,'ability':task,'data_source':'OpenNeuro_ds004856' if task=='mri_visit_correspondence' else 'OpenNeuro_ds007045',
        'images':[image],'prompt':[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':question}]}],
        'reward_model':{'style':'rule','ground_truth':'same' if task=='mri_visit_correspondence' else 'FLAIR' if task=='mri_sequence' else 'no'},
        'extra_info':{'index':index,'internal_split':'train','patient_key':'verified_patient','patient_lineage':'BIDS_SUBJECT_VERIFIED',
            'question':question,'reward_type':task,'task_type':task}}


def fixtures():
    original=[record('seq0','mri_sequence',0),record('seq1','mri_sequence',1),record('tumor','mri_tumor_presence',2)]
    sequence=copy.deepcopy(original[:2])
    for i,r in enumerate(sequence):
        r['id']=f'opaque{i}';r['sequence_qualification_v1']={'training_authorized':False,'expert_review':'PENDING',
            'independent_visual_review':'VISUAL_NONEXPERT_REVIEW_COMPLETE_EXPERT_PENDING',
            'nonexpert_technical_coverage_adequate':i==0,'blind_review_abstained':i!=0}
    dallas=[record('dallas','mri_visit_correspondence',3)]
    index=[{'record_id_PRIVATE':'dallas','task':'mri_visit_correspondence','split':'train','training_authorized':False,
            'expert_eligibility':'NOT_VERIFIED','route_status':'WHOLE_HEAD_IDENTITY_ENGINEERING_ONLY_BRAIN_GROUNDING_UNPROVEN'},
           {'record_id_PRIVATE':'tumor','task':'mri_tumor_presence','split':'train','training_authorized':False,
            'expert_eligibility':'NOT_VERIFIED','visual_eligibility':'NOT_ADMITTED_NONEXPERT_INSPECTION_ONLY'}]
    counts={'mri_sequence':2,'mri_visit_correspondence':1,'mri_tumor_presence':1}
    return sequence,index,original,dallas,counts


def build(f):
    return assembler.assemble(*f[:4],source_hashes={k:'a'*64 for k in assembler.PINS},expected_counts=f[4])


def test_standard_vero_targets_and_qualifications_are_joined_without_authorization():
    f=fixtures();before=copy.deepcopy(f);rows=build(f)
    assert f==before and len(rows)==4
    assert {r['id'] for r in rows}=={'opaque0','opaque1','dallas','tumor'}
    assert all(r['training_authorized'] is False for r in rows)
    assert [r['reward_model'] for r in rows]==[f[0][0]['reward_model'],f[0][1]['reward_model'],f[3][0]['reward_model'],f[2][2]['reward_model']]
    assert [r['train_qualification_bundle_v1']['route_status'] for r in rows]==[
        'SEQUENCE_NONEXPERT_COVERAGE_CANDIDATE_EXPERT_PENDING','SEQUENCE_VISUAL_ABSTENTION_HOLD',
        'DALLAS_DOWNGRADED_WHOLEHEAD_IDENTITY_ONLY','TUMOR_MASK_DERIVED_EXPERT_VISIBILITY_HOLD']


def test_input_byte_hash_is_checked_before_json_rows_are_accepted(tmp_path):
    path=tmp_path/'TRAIN.jsonl';path.write_text('{"id":"bound"}\n')
    pin=hashlib.sha256(path.read_bytes()).hexdigest()
    assert assembler.read_bound_rows(path,pin)==[{'id':'bound'}]
    path.write_text('{"id":"changed"}\n')
    with pytest.raises(ValueError,match='SHA'):assembler.read_bound_rows(path,pin)


@pytest.mark.parametrize('defect',['missing','duplicate','nontrain','wrongtask','wrongtarget','seq_index_duplicate','seq_target','seq_promoted','idx_promoted','unknownsource'])
def test_missing_duplicate_changed_or_nontrain_rows_fail_closed(defect):
    f=fixtures()
    if defect=='missing':f[1].pop()
    if defect=='duplicate':f[1].append(copy.deepcopy(f[1][0]))
    if defect=='nontrain':f[2][2]['extra_info']['internal_split']='internal_dev'
    if defect=='wrongtask':f[1][0]['task']='mri_tumor_grid'
    if defect=='wrongtarget':f[0][0]['reward_model']['ground_truth']='changed'
    if defect=='seq_index_duplicate':f[2][1]['extra_info']['index']=0
    if defect=='seq_target':f[0][0]['extra_info']['question']='target leaked'
    if defect=='seq_promoted':f[0][0]['sequence_qualification_v1']['expert_review']='PASS'
    if defect=='idx_promoted':f[1][0]['training_authorized']=True
    if defect=='unknownsource':f[3][0]['data_source']='invented'
    with pytest.raises(ValueError):build(f)
