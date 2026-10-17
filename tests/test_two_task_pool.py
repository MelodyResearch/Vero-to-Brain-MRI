"""Regression tests for exact historical-row extraction, not a new sampler."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def mod():
    assert importlib.util.find_spec('mri_rl.two_task_pool') is not None,'Missing exact two-task pool assembler'
    from mri_rl import two_task_pool
    return two_task_pool


def test_extract_frozen_rows_preserves_original_line_bytes_and_validates_evidence(tmp_path):
    m=mod(); image=tmp_path/'input.png';image.write_bytes(b'known image bytes')
    rows=[]
    for number,label in enumerate(('FLAIR','contrast-enhanced T1')):
        rows.append({'id':f'q{number}','data_source':'OpenNeuro_ds007045','ability':'mri_sequence','images':[str(image)],
          'prompt':[{'role':'user','content':[{'type':'image','image':str(image)},{'type':'text','text':'Question'}]}],
          'reward_model':{'ground_truth':label},'extra_info':{'patient_key':f'p{number}','patient_lineage':'BIDS_SUBJECT_VERIFIED','internal_split':'train','question':'Question','reward_type':'mri_sequence'}})
    source=tmp_path/'train.jsonl'; lines=[json.dumps(r,indent=None)+'\n' for r in rows];source.write_text(''.join(lines))
    entries=[{'record_id':r['id'],'category':'mri_sequence','patient_key':r['extra_info']['patient_key'],
              'image_sha256':hashlib.sha256(image.read_bytes()).hexdigest(),'target_label':r['reward_model']['ground_truth'],
              'question_sha256':hashlib.sha256(b'Question').hexdigest()} for r in rows[::-1]]
    selected=m.extract_frozen_sequence(source,entries)
    assert [x['row']['id'] for x in selected]==['q1','q0']
    assert [x['raw'] for x in selected]==[lines[1].encode(),lines[0].encode()]
    entries[0]['target_label']='FLAIR'
    with pytest.raises(ValueError,match='target'): m.extract_frozen_sequence(source,entries)


def test_model_question_and_prompt_mismatch_fails_closed(tmp_path):
    m=mod(); image=tmp_path/'x.png';image.write_bytes(b'x')
    row={'id':'q','data_source':'OpenNeuro_ds007045','ability':'mri_sequence','images':[str(image)],
         'prompt':[{'role':'user','content':[{'type':'image','image':str(image)},{'type':'text','text':'Hidden target leaked here'}]}],
         'reward_model':{'ground_truth':'FLAIR'},'extra_info':{'patient_key':'p','patient_lineage':'BIDS_SUBJECT_VERIFIED','internal_split':'train','question':'Question','reward_type':'mri_sequence'}}
    with pytest.raises(ValueError,match='prompt'): m.validate_model_boundary(row,'train')


def test_namespaced_source_split_overlap_detects_candidate_reuse():
    m=mod()
    def row(split):
        return {'data_source':'OpenNeuro_ds004856','extra_info':{'internal_split':split,'patient_key':split,
                'candidate_patient_key':'same-hidden-candidate','statistical_cluster_key':split}}
    with pytest.raises(ValueError,match='overlap'):m.check_split_keys([row('train')],[row('internal_dev')])
