"""CPU-only calibration journal counterexamples. No native approvals or images."""
import hashlib
import json
from pathlib import Path
import random
import numpy as np
import pytest
from mri_rl import tier_c_calibration as cal
from mri_rl.continuous_rl import digest
from tests.test_two_task_training import ROWS
from tests.test_continuous_rl_driver import NumericBackend

SUB=ROWS[:3]+ROWS[28:31]
PROMPT='CPU FIXTURE <|image_pad|>\n'
SIGNATURES={r['id']:{'rendered_prompt_sha256':hashlib.sha256(PROMPT.encode()).hexdigest(),
                    'input_token_ids':[[0]*213]} for r in SUB}
BIND={'kind':'CPU_FIXTURE_NOT_NATIVE','starting_adapter_sha256':'a'*64,'seed':117}


class JournalBackend(NumericBackend):
    made=[]
    def __init__(self):super().__init__(None,{});JournalBackend.made.append(self)
    def parameter_fingerprint(self):return hashlib.sha256(self.weight.tobytes()).hexdigest()
    def rollout(self,row,index,directory):
        responses=[]
        for j in range(4):
            draw=self.rng.random();target=row['reward_model']['ground_truth']
            other=({'FLAIR':'contrast-enhanced T1','contrast-enhanced T1':'FLAIR','same':'different','different':'same'})[target]
            label=target if draw>.5 else other
            responses.append({'text':'<think>Fixture '+str(draw)+'.</think><answer>\\boxed{'+label+'}</answer>',
                              'token_ids':[j+1],'finish_reason':'stop'})
            if getattr(self,'crash_at',None)==index and j==1:
                raw={'record_id':row['id'],'group_index':index,'image_sha256':digest(row['images'][0]),
                     'rendered_prompt':PROMPT,'prompt_sha256':hashlib.sha256(PROMPT.encode()).hexdigest(),
                     'prompt_token_ids':[[0]*213],'responses':responses}
                (Path(directory)/'rollout_progress_PRIVATE.json').write_text(json.dumps(raw))
                raise RuntimeError('partial generation crash')
        return {'record_id':row['id'],'group_index':index,'image_sha256':digest(row['images'][0]),
                'rendered_prompt':PROMPT,'prompt_sha256':hashlib.sha256(PROMPT.encode()).hexdigest(),
                'prompt_token_ids':[[0]*213],'responses':responses}


def run(path,*,resume=False,factory=JournalBackend,rows=SUB,bindings=BIND,prompts=SIGNATURES):
    return cal.run_calibration(rows=rows,backend_factory=factory,authorize=lambda c:None,output=path,
         group_size=4,input_tokens=lambda r:213,resume=resume,bindings=bindings,prompt_signatures=prompts,max_tokens=256)


def test_complete_resume_preserves_exact_rows_without_constructing_backend(tmp_path):
    JournalBackend.made=[];first=run(tmp_path/'c');assert first['rows']==6 and first['complete']
    again=run(tmp_path/'c',resume=True)
    assert len(JournalBackend.made)==1 and again['cost_tokens']==first['cost_tokens']
    assert len(list((tmp_path/'c').glob('row_*')))==6


def test_unknown_log_id_and_changed_binding_are_rejected_before_backend(tmp_path):
    run(tmp_path/'c');before=len(JournalBackend.made)
    with pytest.raises(ValueError):run(tmp_path/'c',resume=True,bindings={**BIND,'seed':99})
    with (tmp_path/'c/calibration_rows_PRIVATE.jsonl').open('a') as f:f.write('{"id":"ALIEN"}\n')
    with pytest.raises(ValueError):run(tmp_path/'c',resume=True)
    assert len(JournalBackend.made)==before


def test_partial_retry_preserves_sampling_rng_and_charges_known_plus_unknown_cost(tmp_path):
    full=run(tmp_path/'full');full_rng=JournalBackend.made[-1].rng.getstate()
    class Crash(JournalBackend):crash_at=3
    with pytest.raises(RuntimeError,match='partial'):run(tmp_path/'split',factory=Crash)
    prefix=[p.read_bytes() for p in sorted((tmp_path/'split').glob('row_*/raw_PRIVATE.json'))]
    resumed=run(tmp_path/'split',resume=True)
    assert resumed['rows']==6 and resumed['complete']
    assert [p.read_bytes() for p in sorted((tmp_path/'split').glob('row_*/raw_PRIVATE.json'))][:3]==prefix
    assert [p.read_bytes() for p in sorted((tmp_path/'split').glob('row_*/raw_PRIVATE.json'))]==[p.read_bytes() for p in sorted((tmp_path/'full').glob('row_*/raw_PRIVATE.json'))]
    assert JournalBackend.made[-1].rng.getstate()==full_rng
    assert resumed['measured_tokens']==full['cost_tokens']+2*(213+1)
    assert resumed['imputed_unknown_tokens']==2*(213+256)
    assert resumed['cost_tokens']==resumed['measured_tokens']+resumed['imputed_unknown_tokens']
    again=run(tmp_path/'split',resume=True);assert again['cost_tokens']==resumed['cost_tokens']


@pytest.mark.parametrize('corruption',['unknown_row','wrong_prompt','too_many_tokens','empty_tokens','updated_weights'])
def test_bad_native_group_is_not_committed(tmp_path,corruption):
    class Bad(JournalBackend):
        def rollout(self,row,index,directory):
            raw=super().rollout(row,index,directory)
            if corruption=='unknown_row':raw['record_id']='ALIEN'
            elif corruption=='wrong_prompt':raw['prompt_token_ids']=[[1]*213]
            elif corruption=='too_many_tokens':raw['responses'][0]['token_ids']=[1]*257
            elif corruption=='empty_tokens':raw['responses'][0]['token_ids']=[]
            else:self.weight[0]+=1
            return raw
    with pytest.raises(ValueError):run(tmp_path/'bad',factory=Bad)
    assert not list((tmp_path/'bad').glob('row_*'))


def test_changed_image_duplicate_rows_and_partial_task_statistics_are_rejected(tmp_path):
    run(tmp_path/'c');p=Path(SUB[0]['images'][0]);old=p.read_bytes()
    try:
        p.write_bytes(b'CHANGED')
        with pytest.raises(ValueError):run(tmp_path/'c',resume=True)
    finally:p.write_bytes(old)
    with pytest.raises(ValueError):run(tmp_path/'dup',rows=SUB+[SUB[0]])
    with pytest.raises(ValueError):cal.calibration_for_arms({'mri_sequence':{'p50':.5,'iqr':.5,'blocked':False}})


def test_journal_payload_tampering_is_rejected_before_new_backend(tmp_path):
    run(tmp_path/'c');before=len(JournalBackend.made)
    p=tmp_path/'c/row_0002/raw_PRIVATE.json';p.write_text('{}')
    with pytest.raises(ValueError):run(tmp_path/'c',resume=True)
    assert len(JournalBackend.made)==before
