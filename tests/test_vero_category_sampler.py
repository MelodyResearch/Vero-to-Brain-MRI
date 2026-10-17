"""Exact pinned Vero sampler source parity; no Torch, MLX or model imports."""
import ast
import copy
import hashlib
from pathlib import Path
import subprocess

import numpy as np
import pytest

from mri_rl.vero_category_sampler import (build_category_schedule, OFFICIAL_CLASS_SHA256,
                                          OFFICIAL_VERO_COMMIT)


def reference():
    root = Path(__file__).resolve().parents[3] / 'work/vero-official'
    assert subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()==OFFICIAL_VERO_COMMIT
    path=root/'vero-rl/verl/utils/dataset/rl_dataset.py'; source=path.read_text()
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='PerBatchDomainSampler')
    assert hashlib.sha256(ast.get_source_segment(source,node).encode()).hexdigest()==OFFICIAL_CLASS_SHA256
    namespace={'np':np,'Sampler':type('Sampler',(),{'__class_getitem__':classmethod(lambda cls,item:cls)}),'Stateful':type('Stateful',(),{})}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    return namespace['PerBatchDomainSampler']


def records(tmp_path):
    image=tmp_path/'opaque.png';image.write_bytes(b'fixture image, never decoded')
    rows=[]
    for route,source,count in [('mri_sequence','OpenNeuro_ds007045',2),('mri_visit_correspondence','OpenNeuro_ds004856',3)]:
        for n in range(count):
            question='Inspect the supplied image.'
            patient=f'{route}-patient-{n}'
            e={'question':question,'reward_type':route,'internal_split':'train',
               'patient_lineage':'BIDS_SUBJECT_VERIFIED','patient_key':patient}
            if route=='mri_visit_correspondence':
                e.update(candidate_patient_key=patient,statistical_cluster_key=f'cycle-{n}',
                         query_source_sha256=str(n)*64,candidate_source_sha256=str(n+1)*64)
            rows.append({'id':f'source-{route}-{n}','data_source':source,'ability':route,'images':[str(image)],
                         'prompt':[{'role':'user','content':[{'type':'image','image':str(image)},{'type':'text','text':question}]}],
                         'reward_model':{'ground_truth':'FLAIR' if route=='mri_sequence' else 'same'},'extra_info':e})
    return rows


@pytest.mark.parametrize('mode,task', [('uniform',None),('proportional',None),
                                      ('specialist','mri_sequence'),('specialist','mri_visit_correspondence')])
@pytest.mark.parametrize('seed',[7,11,13])
def test_exact_source_order_exhaustion_and_rng_parity(tmp_path,mode,task,seed):
    rows=records(tmp_path); plan=build_category_schedule(rows,mode=mode,seed=seed,max_groups=198,specialist_category=task)
    b=plan['binding']; names=list(b['sampling_weights'])
    pools={name:[i for i,r in enumerate(rows) if r['ability']==name] for name in names}
    official=reference()(pools,b['sampling_weights'],256,256,seed=seed)
    full=list(official)
    assert plan['source_indices']==full[:198]
    assert b['logical_batch_category_counts']==dict(zip(names,official._batch_counts))
    assert b['rng_state_after_logical_blocks_sha256']==hashlib.sha256(__import__('json').dumps(
        official.rng.bit_generator.state,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert len({r['id'] for r in plan['schedule']})==198
    for i,row in enumerate(plan['schedule']):
        original=rows[full[i]]
        assert row['source_record_id']==original['id']
        assert row['prompt']==original['prompt'] and row['reward_model']==original['reward_model']
        assert {k:v for k,v in row.items() if k not in {'id','source_record_id'}}=={k:v for k,v in original.items() if k!='id'}
    assert b['source_pool_patient_counts']=={'mri_sequence':2,'mri_visit_correspondence':3}
    assert len(set(plan['source_indices']))<=5


def test_frozen_prefix_resume_keeps_occurrences_and_inputs_immutable(tmp_path):
    rows=records(tmp_path); original=copy.deepcopy(rows)
    long=build_category_schedule(rows,mode='uniform',seed=7,max_groups=198)
    short=build_category_schedule(rows,mode='uniform',seed=7,max_groups=43)
    assert rows==original and short['schedule']==long['schedule'][:43]
    completed=43
    assert short['schedule']+long['schedule'][completed:]==long['schedule']
    assert long==build_category_schedule(rows,mode='uniform',seed=7,max_groups=198)
    assert long['binding']['logical_batch_category_counts']=={'mri_sequence':128,'mri_visit_correspondence':128}


@pytest.mark.parametrize('damage',['mode','source','target','split','duplicate','rescheduled','budget','specialist'])
def test_malformed_or_out_of_scope_schedule_fails_closed(tmp_path,damage):
    rows=records(tmp_path); kwargs={'mode':'uniform','seed':7,'max_groups':198}
    if damage=='mode':kwargs['mode']='iid'
    if damage=='source':rows[-1]['data_source']='OpenNeuro_ds007045'
    if damage=='target':rows[-1]['reward_model']['ground_truth']='different'
    if damage=='split':rows[0]['extra_info']['internal_split']='test'
    if damage=='duplicate':rows[1]['id']=rows[0]['id']
    if damage=='rescheduled':rows[0]['source_record_id']='already-occurrence'
    if damage=='budget':kwargs['max_groups']=199
    if damage=='specialist':kwargs['specialist_category']='mri_sequence'
    with pytest.raises(ValueError):build_category_schedule(rows,**kwargs)
