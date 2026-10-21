"""Scientific-criterion regression: equal admitted counts do NOT mean equal advantage coefficients; counts are diagnostics only.
Also: kernel-shaped row payloads and the prospective evaluation proposal/finalize path. CPU only; existing core with a typed synthetic bundle."""
import importlib.util
import json
from pathlib import Path
import re
import numpy as np
import pytest
from mri_rl import accuracy_recovery_v1 as ar
from mri_rl import accuracy_recovery_eval_v1 as ae
from mri_rl import paired_z045_continuation_entry_v1 as inherited
from mri_rl import shared_rollout_gspo_eval_v1 as shared_eval


def load(name):
    spec=importlib.util.spec_from_file_location(name[:-3],Path(__file__).with_name(name));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


e=load('test_accuracy_recovery_eval_v1.py')
d,f,c,c2,m=e.d,e.f,e.c,e.c2,e.m
reset=d.reset
ADVANTAGE={'F':[1,0,1,0],'T':[1,1,0,0]}        # all format-valid, binary rewards (Codex counterexample)


def coefficients(arm):
    planned=ar.arm_plan(f.bundle(accuracy=ADVANTAGE),arm)
    return planned,{i.response.response_id:i.advantage for i in planned.plan.items}


def test_equal_admitted_counts_do_not_imply_equal_advantage_coefficients_or_gradient_directions():
    pa,adv_a=coefficients('A');pb,adv_b=coefficients('B')
    assert pa.admitted_groups==pb.admitted_groups==2 and pa.admitted_response_ids==pb.admitted_response_ids and len(pa.admitted_response_ids)==8      # the old count predicate sees "identical"
    np.testing.assert_allclose([adv_a[k] for k in ('F0','F1','F2','F3')],[.86603,-.86603,.86603,-.86603],atol=1e-3)
    np.testing.assert_allclose([adv_b[k] for k in ('F0','F1','T0','T1')],[.5,-1.5,.5,.5],atol=1e-3)
    np.testing.assert_allclose([adv_b[k] for k in ('F2','F3','T2','T3')],[1.5,-.5,-.5,-.5],atol=1e-3)
    vector_a=np.array([adv_a[k] for k in ar.RESPONSE_IDS]);vector_b=np.array([adv_b[k] for k in ar.RESPONSE_IDS])
    assert not np.allclose(vector_a,vector_b)
    basis=np.eye(8)                                    # per-response gradients treated as independent directions: g_arm = sum_i (1/8) adv_i e_i
    ga,gb=basis.T@(vector_a/8),basis.T@(vector_b/8);cosine=float(ga@gb/np.linalg.norm(ga)/np.linalg.norm(gb))
    assert cosine<.999                                    # same counts, same coefficient norm (sample-std groups have sum adv^2 = n-1 each), DIFFERENT direction


ENDPOINT=lambda t1:ae.endpoint_vector(e.rows_for(12,t1,6))


@pytest.mark.parametrize('diagnostics,expected',[
    ({'admitted_groups':{'A':2,'B':2},'admitted_responses':{'A':8,'B':8},'cosine':1.0},'CONTINUE_CANDIDATE_REQUIRES_NEW_AUTHORIZATION'),     # equal counts + beneficial endpoints -> still GO
    ({'admitted_groups':{'A':2,'B':1},'admitted_responses':{'A':8,'B':4},'cosine':.2},'CONTINUE_CANDIDATE_REQUIRES_NEW_AUTHORIZATION'),     # fewer B counts cannot veto a beneficial endpoint
    (None,'CONTINUE_CANDIDATE_REQUIRES_NEW_AUTHORIZATION')])
def test_training_signal_counts_can_never_override_a_beneficial_endpoint_decision(diagnostics,expected):
    parent=ENDPOINT(3);out=ae.verdict(parent,parent,ENDPOINT(6),None,diagnostics)
    assert out['decision']==expected and out['training_signal_diagnostics']['diagnostic_only_never_a_criterion'] is True


@pytest.mark.parametrize('diagnostics',[{'admitted_groups':{'A':2,'B':2},'cosine':.3},{'admitted_groups':{'A':1,'B':2},'cosine':.0}])
def test_favourable_training_diagnostics_cannot_rescue_a_no_benefit_endpoint_and_the_comparison_is_always_complete(diagnostics):
    parent=ENDPOINT(3);out=ae.verdict(parent,parent,parent,None,diagnostics)
    assert out['decision']=='STOP_CANDIDATE' and out['benefit'] is False and set(out['B_minus_A'])==set(ae.ev.KEYS)           # full B-A comparison computed even for tied counts


def test_protocol_no_longer_claims_added_variance_is_the_only_signal_or_stops_on_equal_counts():
    text=(Path(ar.__file__).resolve().parents[2]/'research/VERO_TO_MRI_ACCURACY_RECOVERY_V1.md').read_text()
    assert 'the only way B can add signal' not in text and 'B has no more admitted groups/responses than A' not in text
    assert 'diagnostics only' in text.lower() and 'equal admitted counts' in text.lower() and 'advantage coefficient' in text.lower() and 'never a go/no-go criterion' in text.lower()


# ------------------------------------------------------------------ kernel-shaped rows (paired_z045_matched_f32_eval_v1.run_row return keys)
def kernel_row(task,truth,answer,means,row_id):
    labels=ae.ev.LABELS[task]
    return {'id':row_id,'task':task,'truth':truth,'image_sha256':'0'*64,'prompt_sha256':'1'*64,
        'greedy':{'text':'<think>x</think><answer>\\boxed{'+answer+'}</answer>','token_ids_PRIVATE':[1,2],'raw_selected_logps_PRIVATE':[-.1,-.2],'generation_tokens':2,'finish_reason':'stop',
                  'EOS_coverage':'OBSERVED_AND_SCORED','generation_dtype_cache_evidence_PRIVATE':[]},
        'candidates':{'labels':list(labels),'per_label':[{'label':l,'mean':v,'sum':v*2,'first':v,'token_count':2,'label_logps_PRIVATE':[v,v]} for l,v in zip(labels,means)],
                      'prefix_oracle':{'max_abs_error':0.,'tolerance':.002,'first_logps_PRIVATE':list(means)}},
        'costs':{'greedy_calls':1,'candidate_score_API_calls':3,'generated_tokens':2,'teacherforced_input_positions':10}}


def test_loader_and_endpoints_integrate_the_actual_run_row_payload_and_ignore_integrity_files(tmp_path):
    directory=tmp_path/'parent_PRIVATE/rows_PRIVATE';directory.mkdir(parents=True)
    for i,row in enumerate(e.rows_for(12,3,6)):
        payload=kernel_row(row['task'],row['truth'],row['greedy']['text'].split('boxed{')[1].split('}')[0],[x['mean'] for x in row['candidates']['per_label']],f'r{i}')
        with (directory/f'r{i}_PRIVATE.json').open('x') as handle:json.dump(payload,handle,indent=2,sort_keys=True,allow_nan=False)           # exactly how utility.durable writes it
    (directory/'before_restore_integrity_PRIVATE.json').write_text('{"unchanged":true}');(directory/'restored_state_PRIVATE.json').write_text('{"unchanged":true}')
    rows=ae.load_rows(tmp_path/'parent_PRIVATE');assert len(rows)==40
    assert ae.endpoint_vector(rows)==ae.endpoint_vector(e.rows_for(12,3,6))


# ------------------------------------------------------------------ prospective evaluation proposal -> finalize
def pre_world(tmp_path,monkeypatch):
    parent,receipt,_=d.world(tmp_path);frozen=c.frozen_for(parent,receipt);R,W=c2.make_repo(tmp_path/'repo');(R/'local_data/accuracy_recovery_v1').mkdir(parents=True)
    (tmp_path/'cache').mkdir();cert=tmp_path/'cert.json';cert.write_text('{}')
    (tmp_path/'runtime.json').write_text(json.dumps({'parent_checkpoint':str(parent),'parent_receipt_sha256':receipt,'cache_dir':str(tmp_path/'cache'),'cache_certificate':str(cert),
        'cache_certificate_sha256':inherited.digest(cert),'model':str(tmp_path),'contract':'c','global_lock':str(tmp_path/'lock'),'runtime_source_sha256':{'/x.py':'0'*64}}))
    (tmp_path/'evaluation.json').write_text(json.dumps(m.frozen()))
    orig=ar.adam52_parent_anchor
    monkeypatch.setattr(ar,'adam52_parent_anchor',lambda path,sha,frozen=frozen:orig(path,sha,frozen=frozen))
    monkeypatch.setattr(inherited,'source_closure',lambda runtime:{'/installed/runtime.py':'1'*64})
    monkeypatch.setattr(inherited,'cache_metadata',lambda runtime:{'expected_binding':{'cache':'fixture'}})
    return {'runtime_spec':str(tmp_path/'runtime.json'),'evaluation':str(tmp_path/'evaluation.json')},W,R


def test_proposal_exists_before_any_terminal_contains_no_child_hashes_and_is_not_executable(tmp_path,monkeypatch):
    spec,W,R=pre_world(tmp_path,monkeypatch);proposal=ae.propose(spec,encode=e.encode)
    assert proposal['status']=='PROPOSAL_NOT_EXECUTABLE_BEFORE_TERMINALS' and proposal['blockers']==[] and proposal['model_PASS'] is False and proposal['model_loads_this_preparation']==0
    assert proposal['frozen']['budget_ceiling']==shared_eval.eval_budget(3) and proposal['frozen']['max_checkpoints']==3 and proposal['frozen']['stop_rule']==ae.STOP_RULE
    derivation=json.dumps(proposal['derivation'])
    assert 'PENDING_DERIVED_ONLY_FROM_AUTHENTICATED_TERMINAL_RECEIPT' in derivation and not re.search(r'\b[0-9a-f]{64}\b',derivation)         # no phantom child hashes
    assert 'arms' not in proposal and 'evaluated_checkpoints' not in proposal and not hasattr(ae,'launch_scope') or 'scope_kind' not in proposal
    assert ae.propose(spec,encode=e.encode)==proposal                                   # deterministic


def test_finalize_derives_the_actual_scope_only_from_authenticated_terminals_and_rejects_any_frozen_drift(tmp_path,monkeypatch):
    spec,W,R=pre_world(tmp_path,monkeypatch);proposal=ae.propose(spec,encode=e.encode)
    full={**spec,'training_run':str(tmp_path/'run'),'output':str(R/'local_data/accuracy_recovery_v1/eval1')}
    with pytest.raises(ValueError,match='pending'):ae.finalize(proposal,full,encode=e.encode,start=W)                  # no terminals yet: nothing to execute
    d.launch(tmp_path,tmp_path/'run')
    prepared=ae.finalize(proposal,full,encode=e.encode,start=W)
    assert prepared['frozen']==proposal['frozen'] and prepared['evaluated_checkpoints']==['parent','A','B'] and all(prepared['arms'][a]['receipt_sha256'] for a in 'AB')
    assert all(prepared['budget'][k]<=proposal['frozen']['budget_ceiling'][k] for k in prepared['budget'] if isinstance(prepared['budget'][k],(int,float)))
    tampered={**proposal,'frozen':{**proposal['frozen'],'max_checkpoints':5}}
    with pytest.raises(ValueError,match='Intact'):ae.finalize(tampered,full,encode=e.encode,start=W)
    drift=dict(proposal);drift['frozen']={**proposal['frozen'],'evaluation_sha256':'f'*64};drift['proposal_sha256']=ar.identity({k:v for k,v in drift.items() if k!='proposal_sha256'})
    with pytest.raises(ValueError,match='changed since the proposal'):ae.finalize(drift,full,encode=e.encode,start=W)
    data=json.loads((tmp_path/'evaluation.json').read_text());data['records_PRIVATE'][0]['row_PRIVATE']['extra_info']['question']='edited'
    (tmp_path/'evaluation.json').write_text(json.dumps(data))
    with pytest.raises(ValueError):ae.finalize(proposal,full,encode=e.encode,start=W)


def test_cli_propose_writes_only_a_proposal_under_the_canonical_root_and_never_a_scope(tmp_path,monkeypatch,capsys):
    spec,W,R=pre_world(tmp_path,monkeypatch);(tmp_path/'spec.json').write_text(json.dumps(spec))
    monkeypatch.setattr(ae,'_encode',lambda runtime,encode:(e.encode,[]));out=R/'local_data/accuracy_recovery_v1/proposal.json'
    ae.main(['propose','--spec',str(tmp_path/'spec.json'),'--output',str(out)],start=W)
    printed=json.loads(capsys.readouterr().out);assert printed['status']=='PROPOSAL_NOT_EXECUTABLE_BEFORE_TERMINALS' and printed['model_PASS'] is False
    assert out.exists() and not list(out.parent.glob('*_scope.json'))
