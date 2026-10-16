import importlib.util
from pathlib import Path
from collections import Counter
import pytest

spec=importlib.util.spec_from_file_location('c1',Path(__file__).resolve().parents[1]/'scripts/build_c1_matched_subsets.py')
m=importlib.util.module_from_spec(spec)
if spec.loader and Path(spec.origin).exists(): spec.loader.exec_module(m)


def row(p,z,label,area=None):
    return {'patient':p,'record_id':f'{p}-{z}-{label}','category':'mri_tumor_presence','level':z,
            'label':label,'tumor':(1000 if label=='yes' else 0) if area is None else area,'site':'a'}


def test_maximum_balanced_subset_exhaustively_respects_patient_competition():
    assert hasattr(m,'maximum_subset'), 'C1 exact subset solver missing'
    rows=[row('a',45,'yes'),row('a',91,'yes'),row('b',45,'no'),row('c',91,'yes'),row('d',91,'no'),row('e',45,'yes')]
    result=m.maximum_subset(rows,'mri_tumor_presence')
    assert len(result['selected'])==4
    assert len({r['patient'] for r in result['selected']})==4
    for z in (45,91): assert Counter(r['label'] for r in result['selected'] if r['level']==z)=={'yes':1,'no':1}
    assert result['maximum_proven'] is True


def test_small_positive_and_nonzero_negative_are_ineligible_and_unsupported_level_removed():
    assert hasattr(m,'maximum_subset'), 'C1 solver missing'
    rows=[row('a',45,'yes'),row('b',45,'no'),row('c',91,'yes'),row('d',91,'no'),row('e',136,'yes',10),row('f',136,'no'),row('g',45,'no',10)]
    result=m.maximum_subset(rows,'mri_tumor_presence')
    assert result['supported_levels']==[45,91]
    assert {r['patient'] for r in result['selected']}=={'a','b','c','d'}
    assert result['excluded_inconsistent_label']==1


def test_single_level_is_not_admitted_as_multilevel():
    assert hasattr(m,'maximum_subset'), 'C1 solver missing'
    assert m.maximum_subset([row('a',45,'yes'),row('b',45,'no')],'mri_tumor_presence')['selected']==[]


def test_prompt_audit_rejects_identifier_and_visible_gold_field():
    assert hasattr(m,'audit_prompt'), 'prompt audit missing'
    good={'prompt':[{'role':'user','content':'<image> Is a tumor visible?'}], 'images':['private/patient-label.png'], 'extra_info':{'question':'Is a tumor visible?','patient_key':'patient-secret'},'reward_model':{'ground_truth':'yes'},'id':'record-secret'}
    assert m.audit_prompt(good)['pass'] is True
    bad=dict(good,prompt=[{'role':'user','content':'patient-secret; ground_truth: yes'}])
    assert m.audit_prompt(bad)['pass'] is False


def test_image_transport_path_not_mistaken_for_model_visible_text():
    r={'prompt':[{'role':'user','content':[{'type':'image','image':'patient-secret_flair.png'},{'type':'text','text':'Which sequence?'}]}],'extra_info':{'patient_key':'patient-secret'}}
    assert m.audit_prompt(r)['pass'] is True
