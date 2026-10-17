"""Synthetic release fixtures; no actual test case, pixel or model is opened."""
import copy
import json
from pathlib import Path
import pytest
from mri_rl.two_task_pool import canonical_sha, sha


def fixture(tmp_path):
    from mri_rl.final_test_release import BINDINGS, required_checkpoints
    plan={'schema':'two-task-finite-arm-plan-v1','seeds':[7,11,13],
          'rl_arms':[{'name':a} for a in ('S_sequence','S_correspondence','P_raw','U_raw','P_calibrated','U_calibrated')],
          'common_bindings':{'starting_adapter_sha256':None}}
    data=tmp_path/'fixture.bin';data.write_bytes(b'fixture adapter, not model weights')
    config=tmp_path/'config.json';config.write_text('{}')
    evidence=tmp_path/'receipt.json';evidence.write_text('{"fixture":"not native execution"}')
    plan['common_bindings']['starting_adapter_sha256']=sha(data)
    inventory=[]
    for key in required_checkpoints(plan):
        inventory.append({'checkpoint_id':key,'adapter':{'path':str(data),'sha256':sha(data)},
                          'adapter_config':{'path':str(config),'sha256':sha(config)},
                          'completion_evidence':[] if key=='B28' else [{'path':str(evidence),'sha256':sha(evidence)}]})
    bindings={k:('a'*64) for k in BINDINGS}
    from mri_rl import final_test_release as release_module
    for key,name in {'scorer_source_sha256':'vero_mri_reward.py','raw_adapter_source_sha256':'two_task_native_results.py',
                     'statistics_source_sha256':'two_task_analysis.py','projection_source_sha256':'two_task_evaluation.py','release_source_sha256':'final_test_release.py',
                     'dallas_renderer_source_sha256':'visit_correspondence.py'}.items():
        bindings[key]=sha(Path(release_module.__file__).with_name(name))
    bindings['plan_sha256']=canonical_sha(plan)
    scope={'kind':'final-two-task-test-release-v1','bindings':bindings,'checkpoint_inventory_sha256':canonical_sha(inventory),
           'training_completion_verified':True,'checkpoint_selection':'final_budget_boundary','test_outcomes_used':False,
           'all_final_receipt_chains_and_budgets_verified':True}
    approval={'reviewer':'independent fixture reviewer','verdict':'PASS_WITH_LIMITATIONS','scope':scope}
    return plan,inventory,bindings,approval


def test_complete_reviewed_inventory_has_22_checkpoints_and_48_matrix_slots(tmp_path):
    from mri_rl.final_test_release import validate_release, required_matrix_keys
    p,i,b,a=fixture(tmp_path)
    proof=validate_release(p,i,b,a,expected_approval_sha256=canonical_sha(a))
    assert len(i)==22 and len(required_matrix_keys(p))==24
    assert proof['matrix_cells']==48 and proof['actual_test_execution'] is False


@pytest.mark.parametrize('defect',['missing','extra','partial','wrong_checkpoint','wrong_start','changed_binding','unapproved','selection','test_used'])
def test_invalid_release_fails_before_sealed_loader_is_called(tmp_path,defect):
    from mri_rl.final_test_release import release_projection
    p,i,b,a=fixture(tmp_path)
    if defect=='missing':i.pop()
    elif defect=='extra':i.append(copy.deepcopy(i[0]))
    elif defect=='partial':i[1]['completion_evidence']=[]
    elif defect=='wrong_checkpoint':i[1]['adapter']['sha256']='0'*64
    elif defect=='wrong_start':p['common_bindings']['starting_adapter_sha256']='0'*64
    elif defect=='changed_binding':b['scorer_source_sha256']='0'*64
    elif defect=='selection':a['scope']['checkpoint_selection']='best_dev'
    elif defect=='test_used':a['scope']['test_outcomes_used']=True
    expected=canonical_sha(a) if defect!='unapproved' else '0'*64
    called=[]
    with pytest.raises(ValueError):
        release_projection(p,i,b,a,expected_approval_sha256=expected,load_released_rows=lambda:called.append(True),
                           contract_path=tmp_path/'not-to-be-opened.json',destination=tmp_path/'out')
    assert called==[]


def test_final_matrix_requires_both_tasks_each_seed_and_exact_authenticated_predictions(tmp_path):
    from mri_rl.final_test_release import required_matrix_keys, adapt_final_matrix
    p,i,b,a=fixture(tmp_path)
    with pytest.raises(ValueError,match='coverage'):
        adapt_final_matrix(p,{}, {},n_bootstrap=10)
    # Baseline reuse maps one immutable B28 output to each seed; it creates no new patients.
    assert ('B28',7) in required_matrix_keys(p) and ('same_C_SFT',13) in required_matrix_keys(p)


def test_synthetic_complete_projection_hides_targets_and_patient_cycle_metadata(tmp_path):
    from mri_rl.final_test_release import release_projection
    p,i,b,a=fixture(tmp_path)
    contract={'system_instruction':'Frozen system','assistant_prefix':'<|im_start|>assistant\n','enable_thinking':False}
    cp=tmp_path/'prompt.json';cp.write_text(json.dumps(contract))
    b['prompt_contract_sha256']=sha(cp);b['released_row_loader_source_sha256']=sha(__file__)
    a['scope']['bindings']=b
    image=tmp_path/'neutral.png';image.write_bytes(b'synthetic, not MRI')
    rows=[]
    def row(key,task,patient,target,**extra):
        return {'id':key,'ability':task,'data_source':'OpenNeuro_ds007045' if task=='mri_sequence' else 'OpenNeuro_ds004856',
                'images':[str(image)],'reward_model':{'ground_truth':target},
                'extra_info':{'reward_type':task,'internal_split':'test','patient_lineage':'BIDS_SUBJECT_VERIFIED',
                              'patient_key':patient,'question':'Frozen question for '+task,**extra}}
    for person in range(14):
        for target in ('FLAIR','contrast-enhanced T1'):
            for level in range(3):rows.append(row(f'seq-{person}-{target}-{level}','mri_sequence',f'sealed-person-{person}',target))
    for cycle in range(4):
        people=[f'sealed-dallas-{cycle}-{q}' for q in range(2)]
        for query in people:
            for candidate in people:
                rows.append(row(f'{query}-{candidate}','mri_visit_correspondence',query,'same' if query==candidate else 'different',
                                candidate_patient_key=candidate,statistical_cluster_key=f'sealed-cycle-{cycle}'))
    called=[]
    def reviewed_loader():called.append(True);return rows
    source,manifest,proof=release_projection(p,i,b,a,expected_approval_sha256=canonical_sha(a),
        load_released_rows=reviewed_loader,contract_path=cp,destination=tmp_path/'released')
    text=manifest.read_text()
    assert called==[True] and proof['cases']==100 and proof['actual_model_execution'] is False
    assert 'ground_truth' not in text and 'sealed-person' not in text and 'sealed-cycle' not in text
    records=[json.loads(line) for line in text.splitlines()]
    assert all(r['prompt'][0]=={'role':'system','content':'Frozen system'} for r in records)
    assert 'ground_truth' in source.read_text()  # Private scorer reference remains separate.


def test_changed_dallas_scorer_source_rejected_without_touching_training_source(tmp_path,monkeypatch):
    from mri_rl import final_test_release as release
    p,i,b,a=fixture(tmp_path);original_sha=release.sha
    def simulated_source_change(path):
        return '0'*64 if Path(path).name=='visit_correspondence.py' else original_sha(path)
    monkeypatch.setattr(release,'sha',simulated_source_change)
    with pytest.raises(ValueError,match='source changed'):
        release.validate_release(p,i,b,a,expected_approval_sha256=canonical_sha(a))
