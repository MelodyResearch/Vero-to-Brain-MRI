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
