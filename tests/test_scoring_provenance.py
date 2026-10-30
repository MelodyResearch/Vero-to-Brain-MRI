import importlib.util
from pathlib import Path
import pytest
W=Path(__file__).parents[1]
def module():
 from types import SimpleNamespace
 from mri_rl.final_results import recount_saved_responses
 return SimpleNamespace(recount=recount_saved_responses)
def parser():
 p=W/"src/mri_rl/vero_mri_reward.py";s=importlib.util.spec_from_file_location("final22_exact_parser",p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m._answer_and_format
def fixture():
 ref=[{"id":"a","ability":"mri_sequence","reward_model":{"ground_truth":"FLAIR"}},{"id":"b","ability":"mri_sequence","reward_model":{"ground_truth":"contrast-enhanced T1"}},{"id":"c","ability":"mri_visit_correspondence","reward_model":{"ground_truth":"same"}},{"id":"d","ability":"mri_visit_correspondence","reward_model":{"ground_truth":"different"}}]
 texts=["<think>x</think><answer>FLAIR</answer>","<think>x</think><answer>\\boxed{FLAIR}</answer>","malformed","<think>x</think><answer>\\boxed{different}</answer>"]
 rows=[{"record_id":r["id"],"category":r["ability"],"response":t,"token_ids":[1,2,3,4],"generation_tokens":4,"max_tokens":4,"hit_max_tokens":True,"finish_reason":"stop","elapsed_seconds":.5} for r,t in zip(ref,texts)]
 return ref,rows
def test_native_half_format_and_strict_counts_remain_distinct():
 m=module();ref,rows=fixture();x=m.recount(ref,rows,parser());s=x["mri_sequence"];assert s["native_correct"]==1 and s["full_format_correct"]==0 and s["format_score_counts"]["0.5"]==1;assert s["class_counts"]["FLAIR"]["native_recall"]==1 and s["class_counts"]["FLAIR"]["full_format_recall"]==0;assert x["mri_visit_correspondence"]["invalid_answers"]==1
def test_eos_at_cap_is_not_truncation_and_cost_is_saved_only():
 m=module();ref,rows=fixture();rows[0]["finish_reason"]="length";rows[1].pop("elapsed_seconds");x=m.recount(ref,rows,parser())["mri_sequence"];assert x["length_truncated_records"]==1 and x["literal_hit_max_tokens_records"]==2;assert x["tokens"]["sum"]==8 and x["call_time"]["actual_elapsed_seconds_sum"]==.5 and x["call_time"]["records_without_elapsed"]==1
def test_rejects_duplicate_unknown_and_nonfinite_actual_metadata():
 m=module();ref,rows=fixture()
 with pytest.raises(ValueError):m.recount(ref,rows[:-1]+[rows[0]],parser())
 rows[0]["elapsed_seconds"]=float("nan")
 with pytest.raises(ValueError):m.recount(ref,rows,parser())


def test_original22_nine_source_pins_remain_recoverable_from_immutable_history():
 import hashlib,json,subprocess
 matrix=json.loads((W/'results/ORIGINAL22_FINAL100_ACTUAL_MATRIX_V1.json').read_text())
 assert len(matrix['source_sha256'])==9
 for name,expected in matrix['source_sha256'].items():
  data=subprocess.check_output(['git','-C',str(W),'show','1e0d7be1bb099b42e717eb4eb081facc12607ed3:src/mri_rl/'+name])
  assert hashlib.sha256(data).hexdigest()==expected


def test_plan_guard_rejects_log_truncation_and_lock_replacement():
 from mri_rl.research_plan_guard import check_trees,PATHS
 before={p:(W/p).read_bytes() for p in PATHS}
 truncated=dict(before);truncated[PATHS[2]]=b'not the original prefix'
 assert 'LOG_NOT_APPEND_ONLY' in check_trees(before,truncated)['errors']
 changed=dict(before);changed[PATHS[1]]=b'{}'
 assert 'LOCK_CHANGED' in check_trees(before,changed)['errors']
