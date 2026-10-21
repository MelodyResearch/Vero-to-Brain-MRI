"""Entry fixtures: verified original P_calibrated parent metadata, canonical private output root, launch scope. No model, MLX or patient data."""
import importlib.util
import json
from pathlib import Path
import pytest
from mri_rl import accuracy_recovery_v1 as ar
from mri_rl import accuracy_recovery_entry_v1 as entry
from mri_rl import paired_z045_continuation_entry_v1 as inherited
from mri_rl import paired_z045_continuation_driver_v1 as old


def load(name):
    spec=importlib.util.spec_from_file_location(name[:-3],Path(__file__).with_name(name));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


d=load('test_accuracy_recovery_driver_v1.py')
f,c,c2=d.f,d.c,d.c2
reset=d.reset
CALIBRATION={'mri_sequence':{'p50':.5,'iqr':.2}}


def build(tmp_path,monkeypatch,*,sampling='proportional',rule=ar.PARENT_ACTIVE_GROUP_RULE,seed=13):
    R,W=c2.make_repo(tmp_path);(R/'local_data/accuracy_recovery_v1').mkdir(parents=True)
    cal=tmp_path/'calibration.json';cal.write_text(json.dumps(CALIBRATION));digest=inherited.scope_digest(CALIBRATION)
    binding={'experiment':{'arm':{**ar.PARENT_ARM,'sampling':sampling},'seed':seed,'calibration_sha256':digest},'configuration':{**old.PROFILE,'group_size':4,'max_tokens':256,'active_group_rule':rule}}
    parent,receipt=c.original_parent(tmp_path,f.Fake,name='adam52',binding=binding);frozen=c.frozen_for(parent,receipt)
    _,_,rows=f.parent_fixture(tmp_path)
    (tmp_path/'schedule.json').write_text(json.dumps(rows))
    (tmp_path/'cache').mkdir();cert=tmp_path/'cert.json';cert.write_text('{}')
    (tmp_path/'runtime.json').write_text(json.dumps({'parent_checkpoint':str(parent),'parent_receipt_sha256':receipt,'calibration':str(cal),'cache_dir':str(tmp_path/'cache'),'cache_certificate':str(cert),
        'cache_certificate_sha256':inherited.digest(cert),'model':str(tmp_path),'contract':'c','global_lock':str(tmp_path/'lock'),'runtime_source_sha256':{'/x.py':'0'*64},'file_sha256':{'calibration':inherited.digest(cal)}}))
    orig=ar.adam52_parent_anchor
    monkeypatch.setattr(ar,'adam52_parent_anchor',lambda path,sha:orig(path,sha,frozen=frozen))
    monkeypatch.setattr(ar,'PARENT_CALIBRATION_SHA256',digest)
    monkeypatch.setattr(inherited,'source_closure',lambda runtime:{'/installed/runtime.py':'1'*64})
    monkeypatch.setattr(inherited,'cache_metadata',lambda runtime:{'expected_binding':{'cache':'fixture'}})
    spec={'runtime_spec':str(tmp_path/'runtime.json'),'schedule':str(tmp_path/'schedule.json'),'output':str(R/'local_data/accuracy_recovery_v1/run1'),
          'expected_image_sha256':{r['id']:inherited.digest(r['images'][0]) for r in rows}}
    return spec,W,R


def test_prepare_binds_verified_original_parent_arm_and_emits_an_unlaunchable_until_authorized_scope(tmp_path,monkeypatch):
    spec,W,R=build(tmp_path,monkeypatch);prepared=entry.prepare(spec,start=W)
    assert prepared['blockers']==[] and prepared['parent']['groups']==114 and prepared['parent']['updates']==52 and prepared['output']==str(R/'local_data/accuracy_recovery_v1/run1')
    assert prepared['stage_change_vs_parent']['parent_rule']==ar.PARENT_ACTIVE_GROUP_RULE and prepared['parent_original_active_group_rule']==ar.PARENT_ACTIVE_GROUP_RULE
    scope=entry.launch_scope(prepared)
    assert scope['scope_kind']==ar.SCOPE_KIND and scope['implemented_by']=='claude_code' and not any(k in scope for k in ('mode','approved','authorized'))
    human,review=c.signed({**scope,'parent':scope['parent']})
    with pytest.raises(ValueError,match='frozen'):                              # the real entry scope carries the synthetic parent: the default frozen identity refuses it
        ar.authorize_launch(scope,human,review,consumed=ar.consume_once(tmp_path/'ledger'))


@pytest.mark.parametrize('edit',['sampling','rule','seed','arm_spec','calibration'])
def test_prepare_refuses_any_deviation_from_the_verified_original_metadata(tmp_path,monkeypatch,edit):
    kwargs={'sampling':{'sampling':'uniform'},'rule':{'rule':'valid-accuracy-varies_mixed-format_equal-overlong_nonzero-total-v1'},'seed':{'seed':7}}.get(edit,{})
    spec,W,R=build(tmp_path,monkeypatch,**kwargs)
    if edit=='arm_spec':spec['expected_parent_arm']={'name':'U_calibrated','reward':'medgrpo_eq3','sampling':'uniform'}
    if edit=='calibration':monkeypatch.setattr(ar,'PARENT_CALIBRATION_SHA256','0'*64)
    with pytest.raises(ValueError):entry.prepare(spec,start=W)


@pytest.mark.parametrize('where',['worktree','elsewhere','traversal'])
def test_prepare_refuses_outputs_outside_the_canonical_private_root(tmp_path,monkeypatch,where):
    spec,W,R=build(tmp_path,monkeypatch)
    spec['output']={'worktree':str(W/'local_data/accuracy_recovery_v1/run1'),'elsewhere':str(tmp_path/'run1'),'traversal':str(R/'local_data/accuracy_recovery_v1/../run1')}[where]
    with pytest.raises(ValueError):entry.prepare(spec,start=W)
