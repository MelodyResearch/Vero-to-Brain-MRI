"""Focused second-pass crash reservation regressions; no model or native imports."""
import hashlib
import json
from pathlib import Path
import shutil

import pytest
from mri_rl import continuous_rl as driver
from mri_rl.two_task_training import TokenLedger
from tests.test_two_task_driver_review_regressions import FourResponseBackend,inputs,run


def crash_prefix(tmp_path,rows,plan,n=1):
    class Crash(FourResponseBackend):
        def rollout(self,row,index,directory):
            raw=super().rollout(row,index,directory);raw['responses']=raw['responses'][:n]
            (Path(directory)/'rollout_progress_PRIVATE.json').write_text(json.dumps(raw))
            raise RuntimeError('crash during next generation')
    with pytest.raises(RuntimeError,match='crash'):
        run(tmp_path,rows,plan,factory=Crash)
    return next((tmp_path/'run').glob('.pending-*'))


@pytest.mark.parametrize('damage',['index','group_size','max_tokens','raw_index','too_many','overlong','noninteger'])
def test_malformed_or_mismatched_pending_work_is_rejected(tmp_path,damage):
    rows,plan=inputs(tmp_path);pending=crash_prefix(tmp_path,rows,plan)
    reservation=pending/'attempt_reservation.json';progress=pending/'rollout_progress_PRIVATE.json'
    reserved=json.loads(reservation.read_text());raw=json.loads(progress.read_text())
    if damage=='index':reserved['index']=99
    if damage=='group_size':reserved['group_size']=1
    if damage=='max_tokens':reserved['max_tokens']=0
    if damage=='raw_index':raw['group_index']=99
    if damage=='too_many':raw['responses']*=5
    if damage=='overlong':raw['responses'][0]['token_ids']=[1]*257
    if damage=='noninteger':raw['responses'][0]['token_ids']=['bad-token']
    reservation.write_text(json.dumps(reserved));progress.write_text(json.dumps(raw))
    with pytest.raises(ValueError):run(tmp_path,rows,plan,resume=True)


def test_pending_cannot_be_imported_from_a_different_run_policy_binding(tmp_path):
    rows,plan=inputs(tmp_path);pending=crash_prefix(tmp_path,rows,plan)
    other=tmp_path/'other'
    def gate(ctx):
        if ctx['phase']=='group':raise RuntimeError('stop before any different-run attempt')
    kwargs=dict(output=other,schedule=rows,bindings={'review_fixture':'DIFFERENT arm or policy'},
                backend_factory=FourResponseBackend,max_new_groups=1,config={**driver.PROFILE,'group_size':4},
                ledger=TokenLedger(plan),input_tokens=lambda r:213,max_groups_cap=198)
    with pytest.raises(RuntimeError,match='before any'):
        driver.run(authorize=gate,**kwargs)
    shutil.copytree(pending,other/pending.name)
    with pytest.raises(ValueError):driver.run(authorize=lambda ctx:None,resume=True,**kwargs)


def test_unknown_charge_is_separate_and_not_recounted_after_a_completed_boundary(tmp_path):
    rows,plan=inputs(tmp_path);crash_prefix(tmp_path,rows,plan,n=3)
    first=run(tmp_path,rows,plan,resume=True)
    r1=json.loads((Path(first['boundary'])/'receipt.json').read_text())
    assert r1['ledger']['used_tokens']==3*214+469+856
    assert r1['ledger']['imputed_unknown_tokens']==469
    second=run(tmp_path,rows,plan,resume=True)
    r2=json.loads((Path(second['boundary'])/'receipt.json').read_text())
    assert r2['ledger']['used_tokens']==r1['ledger']['used_tokens']+856
    assert r2['ledger']['imputed_unknown_tokens']==469
    assert r2['ledger']['partial_dirs_included']==r1['ledger']['partial_dirs_included']


@pytest.mark.parametrize('when',['post_rollout','save'])
def test_complete_native_progress_is_measured_without_imputing_extra_work(tmp_path,when):
    rows,plan=inputs(tmp_path)
    class Crash(FourResponseBackend):
        def rollout(self,row,index,directory):
            raw=super().rollout(row,index,directory)
            (Path(directory)/'rollout_progress_PRIVATE.json').write_text(json.dumps(raw))
            return raw
        def after_rollout(self):
            if when=='post_rollout':raise RuntimeError('completed rollout crash')
        def save_adapter(self,path):
            if when=='save' and int(self.state['step']):raise RuntimeError('completed rollout save crash')
            super().save_adapter(path)
    with pytest.raises(RuntimeError,match='crash'):run(tmp_path,rows,plan,factory=Crash)
    result=run(tmp_path,rows,plan,resume=True)
    receipt=json.loads((Path(result['boundary'])/'receipt.json').read_text())
    assert receipt['ledger']['used_tokens']==856*2
    assert receipt['ledger']['imputed_unknown_tokens']==0


def test_authoritative_complete_rollout_file_is_not_ignored_if_progress_is_missing(tmp_path):
    rows,plan=inputs(tmp_path)
    class Crash(FourResponseBackend):
        def after_rollout(self):raise RuntimeError('driver has durably written complete rollout')
    with pytest.raises(RuntimeError,match='durably'):
        run(tmp_path,rows,plan,factory=Crash)
    result=run(tmp_path,rows,plan,resume=True)
    receipt=json.loads((Path(result['boundary'])/'receipt.json').read_text())
    assert receipt['ledger']['used_tokens']==856*2
    assert receipt['ledger']['imputed_unknown_tokens']==0
