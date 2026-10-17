"""Small public-proof aggregator checks; no native or private research artifacts."""
import copy
import importlib.util
import json
from pathlib import Path
import pytest


def module():
    p=Path(__file__).parents[1]/'scripts/compare_completed_arm_training_signals.py'
    assert p.is_file(),'Completed-arm sanitized-proof aggregator missing'
    s=importlib.util.spec_from_file_location('completed_signal_test',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def test_unknown_answer_keys_are_grouped_and_not_exported():
    m=module();raw={'responses':8,'answer:flair':3,'answer:contrast-enhanced t1':2,'answer:':1,
        'answer:PRIVATE_FREE_TEXT_MUST_NOT_ESCAPE':1,'withheld_non_categorical_answer_count':1}
    counts=m.safe_answer_counts(raw,'mri_sequence')
    assert counts=={'flair':3,'contrast-enhanced t1':2,'WITHHELD_NONCATEGORICAL_OR_UNPARSED':3}
    assert 'PRIVATE_FREE_TEXT' not in json.dumps(counts)
    with pytest.raises(ValueError):m.safe_answer_counts({**raw,'responses':9},'mri_sequence')


def test_attempt_and_update_shares_are_not_conflated():
    m=module();p={'completed_groups':125,'optimizer_updates':39,'attempted_groups_by_task':{'mri_sequence':61,'mri_visit_correspondence':64},'accepted_update_groups_by_task':{'mri_sequence':16,'mri_visit_correspondence':23}}
    x=m.allocations(p)
    assert x['mri_sequence']['attempt_share']==61/125
    assert x['mri_sequence']['accepted_update_share']==16/39
    assert x['mri_sequence']['accepted_update_rate']==16/61
    p['optimizer_updates']=0;p['accepted_update_groups_by_task']={}
    assert m.allocations(p)['mri_sequence']['accepted_update_share'] is None


def test_seed_variation_preserves_each_seed_and_has_no_learning_claim():
    m=module();rows=[{'arm':'U_raw','seed':s,'completed_groups':125,'optimizer_updates':u,'used_tokens':167000,'tasks':{'mri_sequence':{'accepted_update_rate':r}}} for s,u,r in ((7,39,16/61),(11,5,4/57))]
    x=m.seed_variation(rows)['U_raw']
    assert x['completed_seeds']==[7,11] and x['optimizer_updates']['minimum']==5 and x['optimizer_updates']['maximum']==39
    assert x['seed_rows']==rows and x['accuracy_or_causal_improvement_claimed'] is False


def test_changed_input_digest_fails_before_payload_interpretation(tmp_path):
    m=module();p=tmp_path/'proof.json';p.write_text('{}')
    with pytest.raises(ValueError,match='digest'):m.load_verified_proof(p,{'sha256':'0'*64},{})


def test_valid_proof_and_changed_schema_source_or_budget_rejected(tmp_path):
    m=module();h='a'*64;shared={'initial_adapter':'b'*64,'training_manifest':'c'*64,'plan_canonical':'d'*64,'tokenizer':'e'*64,'frozen_native_inputs':'f'*64}
    p={'schema':'completed-RL-independent-numerical-verification-v1','verdict':'PASS_WITH_LIMITATIONS','reviewer_kind':'isolated_codex','reviewer':'CPU_FIXTURE_NOT_NATIVE_PROOF',
        'evidence_tier':'REPRODUCED_CPU_FROM_ACTUAL_NATIVE_CHECKPOINTS','full_arm_completed':True,'stage_id':'rl:S_sequence:7',
        'arm':{'name':'S_sequence','reward':'raw'},'seed':7,'completed_groups':1,'optimizer_updates':1,'actual_saved_optimizer_RNG_states_verified':2,
        'native_terminal_status':'TOKEN_BUDGET_EXHAUSTED','attempted_groups_by_task':{'mri_sequence':1},'accepted_update_groups_by_task':{'mri_sequence':1},
        'cap':169216,'used_tokens_recomputed':856,'unspent_tokens':169216-856,'completed_rollout_tokens':856,'measured_crashed_work_tokens':0,'imputed_unknown_tokens':0,
        'terminal_boundary_receipt_sha256':h,'checks':{**{k:True for k in m.REQUIRED_CHECKS},'critical_source_hashes':19,'all_saved_file_hashes_verified':1,'independent_reward_response_recomputations':4},
        'evidence_sha256':{**shared,'verifier_source':h,'exit_witness':h,'native_stdout':h,'release_approval':h,'receipt_chain':[h,h]},
        'raw_response_counts_and_correctness_by_task':{'mri_sequence':{'responses':4,'answer:flair':2,'answer:contrast-enhanced t1':2,'native_correct':2,'full_format_correct':2,'full_format':4,'invalid':0}},
        'token_decomposition':{'mri_sequence:input':852,'mri_sequence:generated':4},'variance_diagnostics':{'mri_sequence:zero_total_variance':0,'mri_sequence:task_varies':1}}
    path=tmp_path/'proof.json';path.write_text(json.dumps(p));spec={'schema':p['schema'],'arm':'S_sequence','seed':7,'verifier_source_sha256':h,'sha256':m.file_sha(path)}
    assert m.load_verified_proof(path,spec,shared)==p
    for kind in ('schema','source','budget','variance'):
        bad=copy.deepcopy(p)
        if kind=='schema':bad['schema']='UNVERIFIED'
        elif kind=='source':bad['evidence_sha256']['verifier_source']='0'*64
        elif kind=='budget':bad['used_tokens_recomputed']+=1
        else:bad['variance_diagnostics']['mri_sequence:zero_total_variance']=1
        path.write_text(json.dumps(bad));changed={**spec,'sha256':m.file_sha(path)}
        with pytest.raises(ValueError):m.load_verified_proof(path,changed,shared)
