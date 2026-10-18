import importlib.util
from pathlib import Path
import pytest
W=Path(__file__).parents[1]
def mod():
 p=W/'scripts/analyze_generation_recovery_v1.py';assert p.exists(),'Saved generation-recovery analyzer missing';s=importlib.util.spec_from_file_location('gen_recovery_analysis_test',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_strict_generation_and_candidate_readouts_stay_distinct():
 m=mod();v=W.parent/'vero-mri-v2-controls';s=importlib.util.spec_from_file_location('generation_shared_fixture',v/'tests/test_analyze_paired_pilot_v2.py');f=importlib.util.module_from_spec(s);s.loader.exec_module(f);rows,candidates=f.fixture();greedy={r['id']:{'response':'<think>x</think><answer>\\boxed{'+r['reward_model']['ground_truth']+'}</answer>'} for r in rows};greedy[rows[0]['id']]['response']='<think>x</think><answer>FLAIR</answer>'
 x,pred=m.cohort_summary(v,rows,greedy,candidates)
 assert x['strict_greedy']['tasks']['mri_sequence']['question_accuracy']==23/24
 assert x['candidate']['mean']['balanced_accuracy']==.5
 assert x['greedy_invalid_by_task']['mri_sequence']==1
 assert x['generation_metadata']['records_with_actual_token_ids']==0 and x['generation_metadata']['saved_generated_tokens_sum'] is None

def test_generation_metadata_uses_saved_tokens_not_retokenized_text():
 m=mod();r=m.generation_metadata([{'response':'a long visible text','token_ids':[1,2],'generation_tokens':2,'finish_reason':'stop','hit_max_tokens':False},{'response':'no original emitted tokens'}]);assert r['records_with_actual_token_ids']==1 and r['saved_generated_tokens_sum']==2 and r['records_without_emitted_token_metadata']==1

def test_external_saved_file_hash_mismatch_rejected(tmp_path):
 m=mod();p=tmp_path/'saved.json';p.write_text('{}');entry={'path':str(p),'sha256':m.sha(p)};assert m.auth(entry)=={}
 p.write_text('{"tampered":true}')
 with pytest.raises(ValueError):m.auth(entry)


def test_saved_wall_times_are_not_invented_for_legacy_records():
 m=mod();x=m.saved_wall_times([{"generation_wall_seconds":2.5},{"generation_wall_seconds":1.25},{}],"generation_wall_seconds")
 assert x=={"records":3,"records_with_actual_elapsed":2,"records_without_actual_elapsed":1,"actual_elapsed_seconds_sum":3.75,"actual_elapsed_seconds_mean":1.875}
 assert m.saved_wall_times([{}],"candidate_wall_seconds")["actual_elapsed_seconds_sum"] is None
 for bad in [float("nan"),-1]:
  with pytest.raises(ValueError):m.saved_wall_times([{"generation_wall_seconds":bad}],"generation_wall_seconds")


def test_longer_cap_token_prefix_is_observed_not_assumed():
 m=mod();base=[{"token_ids":[1,2],"response":"old","hit_max_tokens":True,"finish_reason":"length"},{"token_ids":[3,4],"response":"same","hit_max_tokens":False},{"token_ids":[5,6],"response":"old","hit_max_tokens":True,"finish_reason":"length"}]
 control=[{"token_ids":[1,2,7],"response":"new"},{"token_ids":[3,4],"response":"same"},{"token_ids":[5,8,9],"response":"new"}]
 x=m.cap_prefix_agreement(base,control,[0,1,-1],[1,1,0],baseline_cap=2)
 assert x["records"]==3 and x["shared_prefix_identical"]==2 and x["baseline_capped_records"]==2 and x["capped_baseline_shared_prefix_identical"]==1
 assert x["first_divergence_zero_based_position_counts"]=={"1":1}
 assert x["completed_baseline_records"]==1 and x["completed_baseline_text_unchanged"]==1 and x["completed_baseline_strict_label_unchanged"]==1
 assert x["baseline_tokens_are_complete_prefix_of_control"]==2
