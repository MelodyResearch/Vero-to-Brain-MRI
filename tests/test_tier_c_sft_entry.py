"""Codex-authored SFT entry checks. CPU fixtures are never native approvals."""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pytest
from mri_rl import tier_c_sft as sft
from mri_rl.two_task_pool import canonical_sha
from tests.test_two_task_training import ROWS
from tests.test_tier_c_sft import run as old_run, SFTBackend, IN, TGT
from tests.test_continuous_rl_driver import NumericBackend


def entry():
    path=Path(__file__).parents[1]/'src/mri_rl/tier_c_sft_entry.py'
    assert path.is_file(), 'Executable SFT entry is missing'
    spec=importlib.util.spec_from_file_location('sft_entry_test',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


def test_one_step_stage_binds_full_schedule_and_continues_exactly(tmp_path):
    steps,_=sft.build_steps(ROWS,7,10**6,IN,TGT,max_steps=6)
    plan={'group_size':4,'max_tokens':256,'budget':{'maximum_attempted_groups':198,'text_visual_token_cap':10**6}}
    from mri_rl.two_task_training import TokenLedger
    def go(name,resume=False,maximum=None):
        return sft.run_sft(output=tmp_path/name,steps=sft.step_rows(ROWS,steps),bindings={'fixture':'full6'},
            backend_factory=SFTBackend,authorize=lambda ctx:None,ledger=TokenLedger(plan),input_tokens=IN,
            target_tokens=TGT,resume=resume,max_new_steps=maximum)
    full=go('full'); full_backend=NumericBackend.instances[-1]
    first=go('staged',maximum=1)
    assert first['completed_steps']==1 and first['status']=='BOUNDED_STAGE_COMPLETE'
    b0=json.loads((tmp_path/'staged/boundary_0000/receipt.json').read_text())
    assert len(b0['binding']['schedule_record_ids'])==6
    later=go('staged',resume=True); resumed=NumericBackend.instances[-1]
    assert later['completed_steps']==full['completed_steps']==6
    assert later['ledger']==full['ledger']
    np.testing.assert_array_equal(resumed.weight,full_backend.weight)
    for k in ('m','v'):np.testing.assert_array_equal(resumed.state['weight'][k],full_backend.state['weight'][k])
    assert resumed.rng.random()==full_backend.rng.random()


@pytest.mark.parametrize('maximum',[0,-1,True,1.5])
def test_bad_stage_limit_fails_before_backend(tmp_path,maximum):
    from mri_rl.two_task_training import TokenLedger
    steps=sft.step_rows(ROWS,[[0,1,2,3]])
    loaded=[]
    with pytest.raises(ValueError):
        sft.run_sft(output=tmp_path/'bad',steps=steps,bindings={'f':1},backend_factory=lambda *a:loaded.append(a),
            authorize=lambda c:None,ledger=TokenLedger({'budget':{'maximum_attempted_groups':198,'text_visual_token_cap':10000}}),
            input_tokens=IN,target_tokens=TGT,max_new_steps=maximum)
    assert not loaded


@pytest.fixture
def compiled(tmp_path):
    e=entry()
    from tokenizers import Tokenizer,models
    targets=sorted({sft.sft_target(r) for r in ROWS})
    tok=Tokenizer(models.WordLevel({'[UNK]':0,**{x:i+1 for i,x in enumerate(targets)}},unk_token='[UNK]'))
    tokenizer=tmp_path/'tokenizer.json';tok.save(str(tokenizer))
    train=tmp_path/'train.jsonl';train.write_text(''.join(json.dumps(r)+'\n' for r in ROWS))
    adapter=tmp_path/'start.safetensors';adapter.write_bytes(b'CPU_START')
    (tmp_path/'adapter_config.json').write_text('{"fine_tune_type":"lora"}')
    measurement={'schema':'frozen-training-token-measurements-v1','split':'train','manifest_sha256':e.digest(train),
        'processor_sha256':'a'*64,'prompt_contract_sha256':'b'*64,
        'rows':[{'record_sha256':canonical_sha(r),'route':r['ability'],'source':r['data_source'],
                 'input_tokens':IN(r),'visual_tokens':154 if r['ability']=='mri_sequence' else 288} for r in ROWS]}
    plan=e.compile_sft_plan(train,measurement,tokenizer,adapter,'c'*64)
    return e,plan,train,measurement,tokenizer,adapter,tmp_path


def test_compiler_freezes_three_full_schedules_targets_images_and_cap(compiled):
    e,p,train,m,tok,a,tmp=compiled
    assert p['seeds']==[7,11,13] and p['budget']['text_visual_token_cap']==169216
    assert set(p['sft']['schedules'])=={'7','11','13'}
    assert len(p['sft']['target_ids'])==len(p['sft']['image_sha256'])==76
    assert p['sft']['schedules']['7']!=p['sft']['schedules']['11']
    assert e.compile_sft_plan(train,m,tok,a,'c'*64)==p
    assert 'tier_c_sft_entry.py' in e.source_hashes() and 'tier_c_entry.py' in e.source_hashes()


@pytest.mark.parametrize('changed',['measurement','target','image','tokenizer','budget'])
def test_prepare_rejects_changed_frozen_inputs_before_any_backend(compiled,changed):
    e,p,train,m,tok,a,tmp=compiled
    old=None;image=None
    if changed=='measurement':m={**m,'rows':m['rows'][:-1]}
    elif changed=='target':
        rows=[json.loads(l) for l in train.read_text().splitlines()];rows[0]['reward_model']['ground_truth']='FLAIR';train.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    elif changed=='image':image=Path(ROWS[0]['images'][0]);old=image.read_bytes();image.write_bytes(b'CHANGED')
    elif changed=='tokenizer':tok.write_text(tok.read_text()+'\n')
    else:p={**p,'budget':{**p['budget'],'text_visual_token_cap':99999999}}
    try:
        with pytest.raises(ValueError):e.prepare_sft(p,train,m,tok,7)
    finally:
        if image is not None:image.write_bytes(old)


def test_stage_scope_changes_without_mutating_stable_run_binding(compiled):
    e,p,train,m,tok,a,tmp=compiled;prep=e.prepare_sft(p,train,m,tok,7)
    original=json.dumps(prep['bindings'],sort_keys=True)
    first=e.review_scope(prep,output=tmp/'out',runtime={'bound':'fixture'},resume=False,max_new_steps=1,tier='B')
    assert first['max_new_steps']==1 and first['weight_updates_at_most']==1
    assert first['full_schedule_steps']==len(prep['steps'])>1
    assert first['full_schedule_sha256']==prep['bindings']['schedule_sha256']
    assert first['resume'] is False and first['starting_output_state'] is None
    with pytest.raises(ValueError):e.review_scope(prep,output=tmp/'out',runtime={},resume=False,max_new_steps=None,tier='B')
    with pytest.raises(ValueError):e.review_scope(prep,output=tmp/'out',runtime={},resume=True,max_new_steps=1,tier='B')
    assert json.dumps(prep['bindings'],sort_keys=True)==original


def test_readonly_import_does_not_import_model_runtime():
    entry();assert not any(x in sys.modules for x in ('mlx.core','mlx_vlm','torch'))


def test_continuation_uses_committed_SFT_update_without_extra_generation(compiled):
    e,p,train,m,tok,a,tmp=compiled;prep=e.prepare_sft(p,train,m,tok,7)
    class Control(NumericBackend):
        def sft_step(self,examples,work):
            update=super().update(examples[0],{},[1.])
            return {'loss_terms':[.5]*4,'gradient_l2':update['gradient_l2'],
                    'prompt_tokens':[prep['input_tokens'](x) for x in examples],
                    'target_tokens':[prep['target_tokens'][x['id']] for x in examples]}
    r=e.train_prepared(prep,output=tmp/'pilot',adapter=None,backend_factory=Control,
                       authorize=lambda c:None,max_new_steps=1)
    assert r['optimizer_updates']==1
    proof=e.sft_update_proof(tmp/'pilot/boundary_0001/receipt.json',p)
    assert proof['optimizer_updates']==1 and proof['native_SFT_reload']=='NOT_SEPARATELY_RUN'
    assert proof['continuation_requires_shared_native_restore'] is True
    bad={**p,'manifest_sha256':'0'*64}
    with pytest.raises(ValueError):e.sft_update_proof(tmp/'pilot/boundary_0001/receipt.json',bad)
