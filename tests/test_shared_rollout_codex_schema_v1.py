"""Codex regression: a recomputed content digest does not replace schema validation."""
import copy
import pytest
from mri_rl import shared_rollout_gspo_core_v1 as core
from test_shared_rollout_gspo_v1 import frozen


@pytest.mark.parametrize('field,value',[
    ('receipt_sha256','not-a-SHA'),('receipt_sha256',None),
    ('adapter_sha256','not-a-SHA'),('adapter_sha256',True),
])
def test_recomputed_outer_hash_still_rejects_malformed_parent(field,value):
    raw=copy.deepcopy(frozen());raw['parent'][field]=value
    raw['bundle_sha256']=core.identity({k:v for k,v in raw.items() if k!='bundle_sha256'})
    with pytest.raises(ValueError,match='parent receipt/adapter'):core.load_bundle(raw)


@pytest.mark.parametrize('field,value',[
    ('execution_subtype','SEQUENTIAL_ON_POLICY_GSPO'),('numerical_policy','BF16'),
])
def test_bundle_cannot_relabel_execution_contract(field,value):
    raw=copy.deepcopy(frozen());raw[field]=value
    raw['bundle_sha256']=core.identity({k:v for k,v in raw.items() if k!='bundle_sha256'})
    with pytest.raises(ValueError,match='subtype/numerical policy'):core.load_bundle(raw)


@pytest.mark.parametrize('adopt',[None,'/reviewed/qualification'])
def test_prepare_cli_emits_only_applicable_scopes(tmp_path,monkeypatch,adopt):
    import json
    import sys
    from mri_rl import shared_rollout_gspo_entry_v1 as entry
    spec=tmp_path/'spec.json';spec.write_text('{}')
    out=tmp_path/'local_data/prepared.json'
    monkeypatch.setattr(entry,'ROOT',tmp_path)
    prepared={'implementation_status':core.STATUS,'budgets':{m:core.derive_budget({'F':213,'T':213},m) for m in core.MODES},
              'source_sha256':{'/fixture.py':'a'*64},'input_sha256':'b'*64,'output':'/private/future',
              'invocation':{'output':'/private/future','adopt_qualification_run':adopt,'resume':False}}
    monkeypatch.setattr(entry,'prepare',lambda spec,**kw:prepared)
    monkeypatch.setattr(sys,'argv',['shared','prepare','--spec',str(spec),'--output',str(out)])
    entry.main()
    assert json.loads(out.read_text())==prepared
    assert out.with_name('prepared_train_scope.json').is_file()
    assert out.with_name('prepared_qualify_scope.json').exists()==(adopt is None)
