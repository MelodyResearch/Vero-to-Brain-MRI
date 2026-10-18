import importlib.util
from pathlib import Path
import numpy as np
import pytest

def mod():
 p=Path(__file__).parents[1]/'src/mri_rl/paired_discrimination_pilot_v2.py';s=importlib.util.spec_from_file_location('paired_v2',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_mean_label_loss_excludes_format_tokens_and_handles_unequal_label_lengths():
 m=mod();parts=[{'prefix_ids':[1],'label_ids':[2],'suffix_ids':[3]},{'prefix_ids':[1],'label_ids':[4,5],'suffix_ids':[3]}]
 chosen=[np.array([-9.,-2.,-8.]),np.array([-7.,-1.,-3.,-6.])]
 assert m.discrimination_loss(chosen,parts,0,np)==pytest.approx(np.log(2))
 chosen[0][[0,2]]=-99.;chosen[1][[0,3]]=-88.
 assert m.discrimination_loss(chosen,parts,0,np)==pytest.approx(np.log(2))

def test_finite_difference_gradients_only_label_positions_and_opposite_supervision():
 m=mod();parts=[{'prefix_ids':[1],'label_ids':[2],'suffix_ids':[3]}]*2
 values=[np.array([-.4,-1.,-.7]),np.array([-.8,-2.,-.6])]
 def grad(truth,candidate,position):
  epsilon=1e-6;a=[x.copy() for x in values];a[candidate][position]+=epsilon;b=[x.copy() for x in values];b[candidate][position]-=epsilon
  return (m.discrimination_loss(a,parts,truth,np)-m.discrimination_loss(b,parts,truth,np))/(2*epsilon)
 assert grad(0,0,1)<0 and grad(0,1,1)>0 and grad(1,0,1)>0 and grad(1,1,1)<0
 for candidate in range(2):
  for position in (0,2):assert grad(0,candidate,position)==0.

def test_pair_selection_is_training_only_balanced_and_deterministic():
 m=mod()
 def row(person,level,label):return {'id':f'{person}-{level}-{label}','ability':'mri_sequence','images':['image'],'reward_model':{'ground_truth':label},'extra_info':{'patient_key':person,'level':level,'internal_split':'train','question':'same'}}
 rows=[row(p,z,label) for p in ('b','a') for z in (91,45) for label in m.LABELS]
 selected=m.select_four_pairs(rows)
 assert len(selected)==8 and [r['reward_model']['ground_truth'] for r in selected]==list(m.LABELS)*4 and selected[0]['extra_info']['patient_key']=='a'
 rows[0]['extra_info']['internal_split']='internal_dev'
 with pytest.raises(ValueError):m.select_four_pairs(rows)


def test_raw_boundary_and_expanded_native_image_prefix_are_distinct_but_both_verified():
 m=mod()
 class Tokenizer:
  def encode(self,text,add_special_tokens=False):return {'PROMPT':[4,9,5],'TARGET':[6,7],'PROMPTTARGET':[4,9,5,6,7]}[text]
 part={'target_ids':[6,7],'prefix_ids':[6],'label_ids':[7],'label_logit_positions':[5]}
 assert Tokenizer().encode('PROMPTTARGET')!=[4,9,9,9,5]+part['target_ids'] # reproduced invalid old assertion
 m.validate_candidate_alignment(Tokenizer(),'PROMPT',[4,9,9,9,5],'TARGET',part,9,3)
 with pytest.raises(ValueError,match='expanded'):m.validate_candidate_alignment(Tokenizer(),'PROMPT',[4,9,9,9,8],'TARGET',part,9,3)
 bad={**part,'label_logit_positions':[4]}
 with pytest.raises(ValueError,match='causal'):m.validate_candidate_alignment(Tokenizer(),'PROMPT',[4,9,9,9,5],'TARGET',bad,9,3)


def test_actual_shared_sft_keyword_and_contract_identity_apis_cpu_only():
 import inspect,sys
 primary=Path(__file__).parents[1].parents[1];sys.path.insert(0,str(primary/'src'))
 from mri_rl.tier_c_sft import run_sft
 from mri_rl.experiment_contract import ExperimentContract
 # Same actual shared API invoked by the V2 wrapper, no backend instantiated.
 inspect.signature(run_sft).bind(output='x',steps=[],bindings={},backend_factory=object(),authorize=object(),ledger=object(),input_tokens=object(),target_tokens={},resume=False,starting_adapter='x',max_new_steps=2,expected_target_ids={})
 assert callable(ExperimentContract.identity) and list(inspect.signature(ExperimentContract.identity).parameters)==['self']
