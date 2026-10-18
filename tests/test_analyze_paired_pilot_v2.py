import importlib.util,json,hashlib
from pathlib import Path
import pytest

def mod():
 p=Path(__file__).parents[1]/'scripts/analyze_paired_discrimination_pilot_v2.py';s=importlib.util.spec_from_file_location('paired_analyze',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def fixture():
 rows=[];candidate={}
 for person in range(4):
  for level in (45,91,136):
   for truth in (0,1):
    key=f's{person}_{level}_{truth}';rows.append({'id':key,'ability':'mri_sequence','data_source':'OpenNeuro_ds007045','reward_model':{'ground_truth':('FLAIR','contrast-enhanced T1')[truth]},'extra_info':{'internal_split':'internal_dev','patient_lineage':'BIDS_SUBJECT_VERIFIED','reward_type':'mri_sequence','patient_key':f's{person}','level':level}});candidate[key]={'record_id':key,'condition':'real_image','category':'mri_sequence','comparisons':{'mean_t1_minus_flair':-1.,'raw_t1_minus_flair':-1.},'prefix_oracle_first_logps':[-4.,-5.]}
 for cycle in range(4):
  for query in range(2):
   for same in (0,1):
    p=f'd{cycle}_{query}';c=p if same else f'd{cycle}_{1-query}';rows.append({'id':f'{p}_{same}','ability':'mri_visit_correspondence','data_source':'OpenNeuro_ds004856','reward_model':{'ground_truth':'same' if same else 'different'},'extra_info':{'internal_split':'internal_dev','patient_lineage':'BIDS_SUBJECT_VERIFIED','reward_type':'mri_visit_correspondence','patient_key':p,'candidate_patient_key':c,'statistical_cluster_key':str(cycle),'query_source_sha256':p+'_visit1','candidate_source_sha256':c+'_visit2'}})
 candidate['NULL']={'record_id':'NULL','category':'mri_sequence','condition':'no_image','comparisons':{'mean_t1_minus_flair':-1.,'raw_t1_minus_flair':-1.},'prefix_oracle_first_logps':[-4.,-5.]}
 return rows,candidate

def test_common_prior_shift_is_not_paired_visual_improvement_or_24_null_samples():
 import copy
 m=mod();rows,base=fixture();post=copy.deepcopy(base)
 for r in post.values():r['comparisons']={'mean_t1_minus_flair':1.,'raw_t1_minus_flair':1.};r['prefix_oracle_first_logps']=[-5.,-4.]
 x=m.sequence_controls(rows,base,post)
 assert x['U2_minus_B28']['mean']['delta_null_margin']==2. and x['U2_minus_B28']['mean']['delta_patient_mean_paired_separation']==0.
 assert x['checkpoints']['B28']['mean']['null_unique_queries']==1 and x['checkpoints']['B28']['mean']['implied_constant_class_accuracy_on24_truths']==.5

def test_image_dependent_discrimination_changes_separation_while_null_stays_fixed():
 import copy
 m=mod();rows,base=fixture();post=copy.deepcopy(base)
 for row in rows:
  if row['ability']=='mri_sequence':
   margin=1. if row['reward_model']['ground_truth']=='contrast-enhanced T1' else -1.;r=post[row['id']];r['comparisons']={'mean_t1_minus_flair':margin,'raw_t1_minus_flair':margin};r['prefix_oracle_first_logps']=[-4.,-4.+margin]
 x=m.sequence_controls(rows,base,post)
 assert x['U2_minus_B28']['mean']['delta_patient_mean_paired_separation']==2. and x['U2_minus_B28']['mean']['delta_null_margin']==0.
 assert m.candidate_predictions(rows,base,'mean')==[0]*24
 assert m.candidate_predictions(rows,post,'mean')==[i%2 for i in range(24)]

def test_exact_rows_reject_missing_duplicate_and_tampered_record():
 m=mod();records=[{'record_id':'a','image_sha256':'x','prompt':['frozen']}];row={**records[0],'run_identity':'R','weight_updates':0,'response':'ok'}
 assert m.exact_rows([row],records,'R',lambda x:None)=={'a':row}
 for data in ([],[row,row],[{**row,'image_sha256':'changed'}]):
  with pytest.raises(ValueError):m.exact_rows(data,records,'R',lambda x:None)

def test_actual_exit_and_closed_log_are_externally_bound(tmp_path):
 m=mod();log=tmp_path/'log';log.write_text('closed');scope=tmp_path/'scope';scope.write_text('{}');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();receipt=tmp_path/'exit';receipt.write_text(json.dumps({'exit_code':0,'raw_log_sha256':sha(log),'scope_file_sha256':sha(scope)}))
 assert m.authenticate_exit(receipt,sha(receipt),log,sha(log),sha(scope))['exit_code']==0
 receipt.write_text(json.dumps({'exit_code':1,'raw_log_sha256':sha(log),'scope_file_sha256':sha(scope)}))
 with pytest.raises(ValueError):m.authenticate_exit(receipt,sha(receipt),log,sha(log),sha(scope))


def test_primary_is_matched_candidate_readout_not_old_greedy_baseline():
 import sys
 m=mod();rows,_=fixture();seq=[r for r in rows if r['ability']=='mri_sequence'];sys.path.insert(0,str(Path(__file__).parents[1]/'src'))
 from mri_rl.v2_qwen_feature_probe import sequence_metrics
 from mri_rl.v2_cpu_representation import paired_units
 perfect=[i%2 for i in range(24)]
 # Even if separately measured greedy were only15/24, candidate24/24 ->24/24 is ZERO gain.
 x=m.matched_candidate_summary(seq,{'B28':perfect,'post2':perfect},sequence_metrics,paired_units)
 assert x['post2_minus_B28_balanced_accuracy']==0. and x['checkpoints']['B28']['balanced_accuracy']==1.
 assert x['paired_people_uncertainty_descriptive']['subjects']==4


def test_metric_dependency_tampering_is_rejected_before_import(tmp_path):
 m=mod();folder=tmp_path/'src/mri_rl';folder.mkdir(parents=True)
 for name in ('v2_cpu_representation.py','v2_qwen_feature_probe.py'):(folder/name).write_text("raise RuntimeError('unreviewed helper must not execute')\n")
 with pytest.raises(ValueError,match='dependency changed'):m.load_statistics(tmp_path)
