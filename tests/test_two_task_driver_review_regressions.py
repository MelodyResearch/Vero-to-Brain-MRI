"""Independent review regressions for the new Tier-C hooks; synthetic CPU backend only.

These expose concrete ledger defects in 1aa6361. They do not authorize native
training, alter a legacy pilot, or claim that NumPy fixture gradients are Qwen.
"""
import importlib.util
import json
from pathlib import Path

import pytest

from mri_rl import continuous_rl as driver
from mri_rl.two_task_training import TokenLedger

_PATH=Path(__file__).with_name('test_continuous_rl_driver.py')
_SPEC=importlib.util.spec_from_file_location('review_fixture_backend',_PATH)
_FIXTURE=importlib.util.module_from_spec(_SPEC);_SPEC.loader.exec_module(_FIXTURE)


class FourResponseBackend(_FIXTURE.NumericBackend):
    def rollout(self,row,index,directory):
        raw=super().rollout(row,index,directory)
        raw['group_index']=index
        raw['responses']=raw['responses']*2
        return raw
    def update(self,row,raw,advantages):
        result=super().update(row,raw,advantages)
        result['loss_terms']*=2;result['old_current_logp_max_abs_errors']*=2
        return result


def inputs(tmp_path,n=3):
    image=tmp_path/'opaque.png';image.write_bytes(b'CPU fixture only')
    rows=[{'id':f'occurrence-{i}','source_record_id':f'source-{i}','data_source':'OpenNeuro_ds007045',
           'ability':'mri_sequence','images':[str(image)],'reward_model':{'ground_truth':'FLAIR'},
           'extra_info':{'reward_type':'mri_sequence','internal_split':'train','patient_lineage':'BIDS_SUBJECT_VERIFIED',
                         'patient_key':f'p{i}','question':'Fixture only'}} for i in range(n)]
    plan={'group_size':4,'max_tokens':256,'budget':{'input_tokens_min':213,'input_tokens_max':405,
            'text_visual_token_cap':169216,'maximum_attempted_groups':198,'maximum_optimizer_updates':198}}
    return rows,plan


def run(tmp_path,rows,plan,*,resume=False,factory=FourResponseBackend,config=None,count=1):
    return driver.run(output=tmp_path/'run',schedule=rows,bindings={'review_fixture':'one unchanged scientific plan'},
        backend_factory=factory,authorize=lambda c:None,max_new_groups=count,resume=resume,
        config=config or {**driver.PROFILE,'group_size':4},ledger=TokenLedger(plan),
        input_tokens=lambda r:213,max_groups_cap=198)


@pytest.mark.parametrize('completed_responses',[0,1])
def test_unknown_inflight_generation_cost_is_not_free_on_resume(tmp_path,completed_responses):
    rows,plan=inputs(tmp_path)
    class Crash(FourResponseBackend):
        def rollout(self,row,index,directory):
            raw=super().rollout(row,index,directory)
            if completed_responses:
                raw['responses']=raw['responses'][:completed_responses]
                (Path(directory)/'rollout_progress_PRIVATE.json').write_text(json.dumps(raw))
            # Simulate dying inside the next response before native progress fsync.
            raise RuntimeError('generation in flight; unknown generated length')
    with pytest.raises(RuntimeError,match='in flight'):
        run(tmp_path,rows,plan,factory=Crash)
    result=run(tmp_path,rows,plan,resume=True)
    receipt=json.loads((Path(result['boundary'])/'receipt.json').read_text())
    # Actual resumed group costs4*(213+1); each completed old response costs214;
    # unknown old in-flight response needs a conservative reservation213+256.
    assert receipt['ledger']['used_tokens'] >= 4*214 + completed_responses*214 + 213+256


@pytest.mark.parametrize('damage',['group_size','max_tokens','group_count'])
def test_ledger_contract_mismatch_is_rejected_before_backend_load(tmp_path,damage):
    rows,plan=inputs(tmp_path,n=199 if damage=='group_count' else 3)
    config={**driver.PROFILE,'group_size':4}
    if damage=='group_size':config['group_size']=2
    if damage=='max_tokens':config['max_tokens']=512
    def forbidden_load(*args):raise RuntimeError('backend constructed before ledger preflight')
    with pytest.raises(ValueError):
        run(tmp_path,rows,plan,factory=forbidden_load,config=config)


def test_resume_cannot_change_the_compiled_token_cap_under_same_run_binding(tmp_path):
    rows,plan=inputs(tmp_path);run(tmp_path,rows,plan)
    changed=json.loads(json.dumps(plan));changed['budget']['text_visual_token_cap']*=2
    def forbidden_load(*args):raise RuntimeError('changed ledger accepted before backend construction')
    with pytest.raises(ValueError):
        run(tmp_path,rows,changed,resume=True,factory=forbidden_load)


def test_pending_wrong_image_binding_is_rejected_instead_of_charged_as_valid_work(tmp_path):
    rows,plan=inputs(tmp_path);pending=tmp_path/'.pending-tampered';pending.mkdir()
    raw={'record_id':rows[0]['id'],'group_index':0,'image_sha256':'0'*64,
         'responses':[{'text':'fixture','token_ids':[1]}]}
    (pending/'rollout_progress_PRIVATE.json').write_text(json.dumps(raw))
    # Supply a VALID new recovery context/reservation so this regression reaches
    # the wrong-image check rather than passing merely because context is missing.
    context={'binding_sha256':'b'*64,'boundary_groups':{'c'*64:0},'schedule':rows,
             'image_sha256':{r['id']:driver.digest(r['images'][0]) for r in rows},
             'group_size':4,'max_tokens':256}
    (pending/'attempt_reservation.json').write_text(json.dumps({
        'schema':'attempt-reservation-v1','run_binding_sha256':'b'*64,'parent_receipt_sha256':'c'*64,
        'index':0,'record_id':rows[0]['id'],'image_sha256':context['image_sha256'][rows[0]['id']],
        'input_tokens':213,'group_size':4,'max_tokens':256}))
    with pytest.raises(ValueError,match='image|frozen'):
        driver._partial_cost(pending,{r['id']:r for r in rows},lambda row:213,context=context)
