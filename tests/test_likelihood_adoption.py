import json
from pathlib import Path
import pytest
from mri_rl.frozen_eval import sha256
from mri_rl.experiment_contract import canonical_hash
from mri_rl.likelihood_adoption import verify_adoption,numerical_loop_sha256


def fixture(tmp_path):
    old={'contract_sha256':'contract','checkpoint':'A28','adapter':{'sha256':'adapter'},'backend_sources':{'b':'sha'},'source_hashes':{'answer_likelihood':'a','paced_answer_likelihood':'old'},'battery_exception':{'version':'old'},'run_identity':'unused','temperature':0,'max_generation_tokens':256,'lora_rank':4,'lora_alpha':8,'quantization_bits':4,'group_size':64,'weight_updates':0}
    old['run_identity']=canonical_hash({'contract':'contract','checkpoint':'A28','sources':old['source_hashes'],'backend':old['backend_sources'],'battery_diagnostic':True})
    approval=tmp_path/'approval.json';approval.write_text(json.dumps({'kind':'answer_likelihood_v1','verdict':'PASS_WITH_LIMITATIONS','reviewer':'independent','scope':old}))
    prefix=tmp_path/'prefix.jsonl';prefix.write_text(json.dumps({'record_id':'r','run_identity':old['run_identity']})+'\n')
    audit=tmp_path/'audit.json';audit.write_text(json.dumps({'verdict':'PASS_WITH_LIMITATIONS','output_sha256':sha256(prefix),'approval_sha256':sha256(approval)}))
    bridge=tmp_path/'bridge.json';bridge.write_text(json.dumps({'kind':'likelihood_immutable_prefix_v1','prefix_path':str(prefix),'prefix_sha256':sha256(prefix),'approval_path':str(approval),'approval_sha256':sha256(approval),'records':1,'audit_path':str(audit),'audit_sha256':sha256(audit),'numeric_loop_sha256':numerical_loop_sha256()}))
    current={**old,'source_hashes':{**old['source_hashes'],'paced_answer_likelihood':'new'}}
    return bridge,prefix,approval,current


def test_preserves_original_bytes_and_identity(tmp_path):
    b,p,a,c=fixture(tmp_path)
    result=verify_adoption(b,c,private_root=tmp_path)
    assert result['bytes']==p.read_bytes()
    assert result['identities']==[json.loads(p.read_text())['run_identity']]

@pytest.mark.parametrize('mutation',['prefix','approval','backend','numeric','precision','checkpoint','ast','audit','path'])
def test_adoption_rejects_mutation(tmp_path,mutation):
    b,p,a,c=fixture(tmp_path)
    if mutation=='prefix':p.write_text(p.read_text()+'\n')
    elif mutation=='approval':a.write_text(a.read_text()+' ')
    elif mutation=='backend':c['backend_sources']={'b':'bad'}
    elif mutation=='numeric':c['source_hashes']['answer_likelihood']='bad'
    elif mutation=='precision':c['quantization_bits']=8
    elif mutation=='checkpoint':c['checkpoint']='B28'
    elif mutation=='ast':
        d=json.loads(b.read_text());d['numeric_loop_sha256']='bad';b.write_text(json.dumps(d))
    elif mutation=='audit':(tmp_path/'audit.json').write_text('{}')
    else:
        d=json.loads(b.read_text());d['prefix_path']='/etc/hosts';b.write_text(json.dumps(d))
    with pytest.raises(ValueError):verify_adoption(b,c,private_root=tmp_path)


def test_mixed_prefix_only_accepts_exact_historical_region(tmp_path):
    from mri_rl.paced_answer_likelihood import completed_prefix
    from test_paced_answer_likelihood import numeric_fixture
    records=[{'record_id':'first','image_sha256':'x'},{'record_id':'second','image_sha256':'y'}]
    old={**records[0],**numeric_fixture(),'run_identity':'old','response':'ok'}
    new={**records[1],**numeric_fixture(),'run_identity':'new','response':'ok'}
    raw=(json.dumps(old)+'\n').encode();adopt={'bytes':raw,'identities':['old']}
    p=tmp_path/'out.jsonl';p.write_bytes(raw+(json.dumps(new)+'\n').encode())
    assert completed_prefix(p,records,'new',adoption=adopt)==2
    new['run_identity']='old';p.write_bytes(raw+(json.dumps(new)+'\n').encode())
    with pytest.raises(ValueError):completed_prefix(p,records,'new',adoption=adopt)
    p.write_bytes(raw.replace(b'"ok"',b'"different"'))
    with pytest.raises(ValueError):completed_prefix(p,records,'new',adoption=adopt)
