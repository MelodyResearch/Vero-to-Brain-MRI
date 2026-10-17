"""Independent failure-accounting regressions; no native runtime or MRI payloads."""
import importlib.util
from pathlib import Path
import pytest


def module():
    path=Path(__file__).parents[1]/'src/mri_rl/v2_effective_balance.py'
    spec=importlib.util.spec_from_file_location('independent_v2',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def fixture():
    m=module();rows=[{'id':t,'ability':t,'extra_info':{'internal_split':'train'}} for t in m.TASKS]
    c=m.QuotaControl(rows,{r['id']:10 for r in rows},seed=7,target=2,cap=10000)
    return m,c


def test_failure_rejects_unauthenticated_negative_cost():
    _,c=fixture();c.offer()
    with pytest.raises(ValueError):c.fail({'cost':-1})
    assert c.used==0


def test_failure_debt_is_charged_once():
    _,c=fixture();c.offer();c.fail();first=c.used
    with pytest.raises(ValueError):c.fail()
    assert c.used==first


def test_failure_rejects_cost_from_other_pending_group():
    _,c=fixture();c.offer()
    with pytest.raises(ValueError):c.fail({'cost':2})
    assert c.used==0
