"""Synthetic forward-evidence and strict all-case analysis only."""
import pytest
from mri_rl import safe_micro_pilot_native_v1 as n


def test_forward_gate_requires_actual_parity_and_nonzero_same_token_image_effect():
 assert n.forward_verdict([.001,.0019],different_pixels=True,same_target_delta=.01)=='PASS'
 assert n.forward_verdict([.001,.0019],different_pixels=True,same_target_delta=0.)=='PARTIAL'
 with pytest.raises(ValueError):n.forward_verdict([.003,.001],different_pixels=True,same_target_delta=.01)
 with pytest.raises(ValueError):n.forward_verdict([float('nan'),0.],different_pixels=True,same_target_delta=.01)
 with pytest.raises(ValueError):n.forward_verdict([0.,0.],different_pixels=False,same_target_delta=.1)


def test_greedy_metrics_count_invalid_wrong_and_both_recalls():
 rows=[{'truth':'FLAIR','text':'<think>sequence</think><answer>\\boxed{FLAIR}</answer>'},
       {'truth':'contrast-enhanced T1','text':'<think>sequence</think><answer>\\boxed{contrast-enhanced T1}</answer>'},
       {'truth':'FLAIR','text':'FLAIR'},
       {'truth':'contrast-enhanced T1','text':'<think>sequence</think><answer>\\boxed{FLAIR}</answer>'}]
 x=n.greedy_metrics(rows)
 assert x['cases']==4 and x['accuracy']==.5 and x['FLAIR_recall']==.5 and x['T1_recall']==.5 and x['invalid']==1


def test_zero_update_alias_requires_exact_parent_policy_and_no_checkpoint(tmp_path):
 parent='a'*64
 receipt={'updated':False,'alias_of_parent':True,'checkpoint':None,'files_sha256':{},'parent_policy_sha256':parent,'policy_sha_after':parent,'adam_step_after':52}
 assert n.terminal_choice(receipt,parent,tmp_path)['alias_of_parent']
 with pytest.raises(ValueError):n.terminal_choice({**receipt,'policy_sha_after':'b'*64},parent,tmp_path)
 with pytest.raises(ValueError):n.terminal_choice({**receipt,'adam_step_after':53},parent,tmp_path)


def test_finite_local_counter_charges_before_native_calls(tmp_path):
 c=n.CallCounter({'generated_tokens':256,'LM':260},tmp_path/'c.json')
 c.charge('generated_tokens',256)
 with pytest.raises(ValueError):c.charge('generated_tokens')
 assert c.used['generated_tokens']==256
 with pytest.raises(ValueError):c.charge('unapproved')


def test_terminal_selection_is_bound_to_completed_training_receipt(tmp_path):
 from mri_rl.shared_rollout_gspo_core_v1 import identity
 r={'updated':False,'alias_of_parent':True,'checkpoint':None,'files_sha256':{},'parent_policy_sha256':'a'*64,'policy_sha_after':'a'*64,'adam_step_after':52};h=identity(r);r['receipt_sha256']=h
 n.verify_terminal_receipt(r,h)
 with pytest.raises(ValueError):n.verify_terminal_receipt(r,'b'*64)
 r['adam_step_after']=53
 with pytest.raises(ValueError):n.verify_terminal_receipt(r,h)


def test_positive_signal_requires_B_beats_both_baselines_without_FLAIR_loss():
 parent={'balanced_accuracy':.5,'T1_recall':.25,'FLAIR_recall':.75};A=dict(parent);B={'balanced_accuracy':.625,'T1_recall':.5,'FLAIR_recall':.75}
 assert n.pilot_decision({'Parent':parent,'A':A,'B':B})['decision']=='POSITIVE_ENGINEERING_SIGNAL_NO_AUTOMATIC_SCALE'
 assert n.pilot_decision({'Parent':parent,'A':B,'B':B})['decision']=='NO_POSITIVE_ADAPTATION_SIGNAL'
 assert n.pilot_decision({'Parent':parent,'A':A,'B':{**B,'FLAIR_recall':.5}})['decision']=='NO_POSITIVE_ADAPTATION_SIGNAL'
