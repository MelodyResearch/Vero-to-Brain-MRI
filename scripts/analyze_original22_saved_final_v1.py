"""Prospective saved-only bridge for frozen original22 final100. No inference.
All22 externally bound exit0 witnesses precede ANY prediction/reference read.
Future completion hashes must be supplied by the independent coordinator; this
script never fills missing hashes or grants itself permission to score a subset.
"""
from __future__ import annotations
import argparse,hashlib,json,re,sys
from pathlib import Path
W=Path(__file__).resolve().parents[1]
BOOTSTRAPS=2000

def require(ok,message):
 if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return json.loads(Path(path).read_text())
def auth(d):
 require(isinstance(d,dict) and set(d)=={'path','sha256'} and isinstance(d['sha256'],str) and re.fullmatch('[0-9a-f]{64}',d['sha256']),'Externally supplied exact file hash required')
 require(sha(d['path'])==d['sha256'],'Externally bound saved file changed');return load(d['path'])
def checkpoint_key(arm,seed):return 'B28' if arm=='B28' else f'sft:{seed}' if arm=='same_C_SFT' else f'rl:{arm}:{seed}'

def verify_all_completed(runs,completed,reviews):
 require(len(runs)==22 and len({r['checkpoint_id'] for r in runs})==22,'Exact22 prepared runs required')
 require(len(completed)==22 and len({r['checkpoint_id'] for r in completed})==22 and {r['checkpoint_id'] for r in completed}=={r['checkpoint_id'] for r in runs},'All22 external completed-run bindings required')
 by={r['checkpoint_id']:r for r in completed};witnesses={}
 # Phase1 reads ONLY externally authenticated exit metadata. No prediction,
 # closed-log contents, projected manifest, image or reference labels are read.
 for run in runs:
  key=run['checkpoint_id'];d=by[key];require(set(d)=={'checkpoint_id','exit','log','output'},'Exact completion descriptor fields required')
  require(d['exit']['path']==run['exit'],'Completed exit path differs');e=auth(d['exit'])
  require(e.get('exit_code')==0,'All22 actual exit0 witnesses required before scoring')
  require(e.get('checkpoint_id')==key and e.get('review_sha256')==reviews[key]['sha256'] and e.get('scope_sha256')==run['scope']['sha256'],'Actual checkpoint/review/scope witness differs')
  require(e.get('records')==100 and d['output']['path']==run['output'] and d['log']['path']==run['log'],'Full100 output/log identities required')
  require(isinstance(d['output']['sha256'],str) and re.fullmatch('[0-9a-f]{64}',d['output']['sha256']) and d['output']['sha256']==e.get('output_sha256'),'External completed output digest differs')
  require(isinstance(d['log']['sha256'],str) and re.fullmatch('[0-9a-f]{64}',d['log']['sha256']) and d['log']['sha256']==e.get('raw_log_sha256'),'External closed-log digest differs');witnesses[key]=e
 # Phase2 starts only after every one of the22 metadata gates has passed.
 for d in completed:
  for field in ('log','output'):require(sha(d[field]['path'])==d[field]['sha256'],'Completed output/closed log changed')
 return by

def analyze(spec):
 require(spec.get('schema')=='original22-final-saved-analysis-input-v1' and spec.get('status')=='ALL22_COMPLETED_HASHES_EXTERNALLY_SUPPLIED','NOT_RUN: all22 future completion hashes must be independently supplied')
 require(spec.get('analyzer_sha256')==sha(__file__) and spec.get('bootstrap_replicates')==BOOTSTRAPS,'Frozen analyzer/canonical2000 bootstrap differs')
 primary=Path(spec['primary_root']).resolve()
 for name,expected in spec['source_sha256'].items():require(sha(primary/'src/mri_rl'/name)==expected,'Bound source dependency changed')
 required_sources={'original22_final_eval.py','final_test_release.py','two_task_native_results.py','two_task_analysis.py','two_task_evaluation.py','two_task_pool.py','vero_mri_reward.py','experiments.py','experiment_contract.py'}
 require(set(spec['source_sha256'])==required_sources,'Exact saved-analysis source closure required');sys.path.insert(0,str(primary/'src'))
 from mri_rl import original22_final_eval as execution
 from mri_rl.final_test_release import required_checkpoints,required_matrix_keys,adapt_final_matrix
 from mri_rl.two_task_pool import canonical_sha
 require(Path(execution.__file__).resolve()==primary/'src/mri_rl/original22_final_eval.py','Wrong primary package imported')
 b=auth(spec['bundle']);ss=auth(spec['scope_set']);auth(spec['native_review_set']);auth(spec['release_review'])
 require([r['checkpoint_id'] for r in ss['runs']]==list(required_checkpoints(b['plan'])),'Frozen22 checkpoint order differs')
 reviews=execution.validate_review_set(spec['native_review_set']['path'],spec['native_review_set']['sha256'],ss,spec['scope_set']['sha256'])
 completed=verify_all_completed(ss['runs'],spec['completed_runs'],reviews)
 # Only now permit existing release validation, reference reads and scoring.
 b=execution.bundle_at(spec['bundle']['path'],spec['bundle']['sha256']);release=execution.release_bundle(b,spec['release_review']['path'],spec['expected_release_approval_sha256'])
 require(ss['executor_source_sha256']==b['executor_source_sha256'] and ss['inventory_sha256']==canonical_sha(b['inventory']) and ss['release_approval_sha256']==spec['expected_release_approval_sha256'],'Scope set/frozen inventory/release differ')
 p=Path(b['projection_directory']);require(sha(p/'projection_receipt_SAFE.json')==ss['projection_receipt_sha256'],'Exact released projection differs')
 runs={r['checkpoint_id']:r for r in ss['runs']};matrix={}
 for arm,seed in sorted(required_matrix_keys(b['plan'])):
  key=checkpoint_key(arm,seed);r=runs[key];scope=auth(r['scope'])
  matrix[(arm,seed)]={'source_manifest':str(p/'final_test.verl.jsonl'),'eval_manifest':str(p/'final_test_native_system_matched.jsonl'),'projection_receipt':str(p/'projection_receipt_SAFE.json'),'contract_path':b['contract_path'],'output_path':r['output'],'expected_config':r['expected_config'],'approved_scope':scope,'expected_scope_sha256':canonical_sha(scope),'expected_projection_sha256':ss['projection_receipt_sha256'],'expected_output_sha256':completed[key]['output']['sha256'],'expected_cache_binding':scope['derived_cache']}
 result=adapt_final_matrix(b['plan'],matrix,release,n_bootstrap=BOOTSTRAPS)
 require(len(result['matrix']['cells'])==48 and len(result['matrix']['arm_seed_summary'])==24,'Complete48 task cells/24 arm-seed rows required')
 result.update(schema='original22-final-saved-result-v1',status='ALL22_ACTUAL_OUTPUTS_AUTHENTICATED_AND_SCORED',research_tag='PRIMARY_VERO_TRANSFER',evidence_tier='OBSERVED',analyzer_sha256=sha(__file__),bootstrap_replicates=BOOTSTRAPS,distinct_model_inference_runs=22,common_B28_inference_runs=1,common_B28_seed_rows=3,common_B28_training_seed_replicates=0,native_models_loaded_by_analysis=0,sealed_outcomes_used_to_select_methods=False,adaptive_or_short_prompt_arms_included=False,source_sha256=spec['source_sha256'],input_evidence_sha256={k:spec[k]['sha256'] for k in ('bundle','scope_set','native_review_set','release_review')},mission_alignment='Frozen original Vero-to-MRI specialist/proportional/uniform/raw/calibrated/SFT cross-task comparison; no method or endpoint selected after final-test outcomes.',baseline_reuse_note='The canonical B28 seed-indexed rows repeat one immutable common initializer output, not three independent inference or training runs. Observed training-seed ranges do not add patients.')
 return result,{'schema':'original22-final-saved-binding-provenance-v1','completed_runs':completed,'input_spec':spec}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True);p.add_argument('--expected-spec-sha256',required=True);p.add_argument('--private-output',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(sha(a.spec)==a.expected_spec_sha256,'Externally supplied complete analysis-spec digest differs')
 require(a.private_output.resolve().is_relative_to(W/'local_data') and a.output.resolve().is_relative_to(W/'results'),'Private provenance/public aggregate destinations required')
 result,private=analyze(load(a.spec))
 for path,value in ((a.private_output,private),(a.output,result)):
  with path.open('x') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({'status':result['status'],'aggregate_sha256':sha(a.output),'task_cells':48,'distinct_model_runs':22,'native_models_loaded_by_analysis':0}))
if __name__=='__main__':main()
