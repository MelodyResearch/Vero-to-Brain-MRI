"""Saved actual parent/adaptive outcomes only; no model imports, fitting or new forward calls."""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,re,sys
from pathlib import Path
from statistics import mean
W=Path(__file__).resolve().parents[1]
HELPER_SHA='7c6defde436210ac00d502533c3200ffb02e8843eb3abe2a5c9f357489610b81'
PARENT='P_calibrated13_parent';CHILD='adaptive'
PARENT_ADAPTER='c70b426b58191446545ba6333f7595e96f4fdb4cc689ddbaf5206e276b329db6'
PARENT_RECEIPT='e0696a8186d413b2bb35034b5c5eb2568ddc0b05e1b719778de42fa491cc882a'
def require(ok,message):
 if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return json.loads(Path(path).read_text())
def authenticated(item):
 require(sha(item['path'])==item['sha256'],'External immutable evidence digest differs');return load(item['path'])
def load_helpers(v2):
 path=Path(v2)/'scripts/analyze_paired_discrimination_pilot_v2.py';require(sha(path)==HELPER_SHA,'Reviewed pure analysis dependency changed');s=importlib.util.spec_from_file_location('adaptive_fixed_analysis_helpers',path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def labels(value):
 """The old pure helper requires internal aliases; never export those checkpoint labels."""
 mapping={'B28':PARENT,'post2':CHILD,'U2_minus_B28':'adaptive_minus_parent','post2_minus_B28_balanced_accuracy':'adaptive_minus_parent_balanced_accuracy'}
 if isinstance(value,dict):return {mapping.get(k,k):labels(v) for k,v in value.items()}
 if isinstance(value,list):return [labels(v) for v in value]
 return value

def visual_controls(helper,rows,parent,child):return labels(helper.sequence_controls(rows,parent,child))
def validate_training_proof(proof,receipt,receipt_sha):
 n=receipt['completed_groups'];k=receipt['optimizer_updates']
 require(type(n) is int and 0<=n<=16 and type(k) is int and 0<=k<=min(n,4),'Invalid actual adaptive budget/counters')
 require(proof.get('schema')=='adaptive-continuation-focused-CPU-proof-v1' and proof.get('verdict')=='PASS_WITH_LIMITATIONS' and proof.get('preserved_initial_parent_state') is True and proof.get('parent_native_step')==52,'Independent inherited-state proof required')
 require(proof['completed_groups']==n and proof['new_optimizer_updates']==k and proof['absolute_optimizer_updates']==52+k and proof['absolute_completed_groups']==114+n and receipt['absolute_optimizer_updates']==52+k and receipt['absolute_completed_groups']==114+n,'Relative/absolute optimizer continuity differs')
 require(proof['terminal_boundary_receipt_sha256']==receipt_sha and proof['final_adapter_sha256']==receipt['files_sha256']['adapter.safetensors'],'Proof does not authenticate selected terminal checkpoint')
 return n,k

def strict_prediction(response,labels,parser):
 answer,fmt=parser(response);lookup={v.lower():i for i,v in enumerate(labels)}
 return lookup.get(answer.lower(),-1) if fmt==1. else -1

def resource_summary(events):
 require(bool(events),'Actual resource samples required');times=[e['sample']['monotonic_seconds'] for e in events]
 require(all(isinstance(t,(int,float)) for t in times) and all(a<=b for a,b in zip(times,times[1:])),'Resource monotonic chronology differs')
 admitted={}
 for e in events:
  unit=e.get('unit_id','')
  if re.fullmatch(r'rollout_group_\d+',unit) and e.get('action')=='CONTINUE':admitted.setdefault(unit,e['sample']['monotonic_seconds'])
 starts=list(admitted.values())
 result={'samples':len(events),'observed_monotonic_span_seconds':times[-1]-times[0],
  'complete_intergroup_intervals_seconds':[b-a for a,b in zip(starts,starts[1:])],
  'final_group_observed_lower_bound_seconds':times[-1]-starts[-1] if starts else None,
  'timing_limit':'Each group begins at its first admitted CONTINUE observation; repeated HOLD/recovery samples do not create groups. Observed monotonic sampling span; intergroup intervals include preceding group work/checkpoint/guards. Last-group span is only a lower bound, not full process runtime.',
  'actions':dict(__import__('collections').Counter(e.get('action','UNKNOWN') for e in events)),
  'supervisor_levels':dict(__import__('collections').Counter(e.get('supervisor_level','UNKNOWN') for e in events))}
 for key in ('available_gib','pressure_free_percent','swap_used_gib','mlx_peak_gib'):
  values=[e['sample'].get(key) for e in events if e['sample'].get(key) is not None]
  if values:result[key]={'min':min(values),'max':max(values)}
 return result

def analyze(spec):
 R=Path(spec['primary_root']);V=Path(spec['v2_root']);sys.path.insert(0,str(R/'src'));h=load_helpers(V);stats,seqstats=h.load_statistics(V)
 entry=h.module(V/'scripts/run_paired_discrimination_pilot_v2.py','adaptive_actual_evaluator',h.ENTRY_SHA)
 early=h.module(V/'scripts/run_c40_early_visual_probe_v1.py','adaptive_actual_loader','8cafacbf3a9086160c8e257673c6f933070df208bffc5c32dec38bc0e1e82e06');api=early.primary_api(R)
 evaluation=authenticated(spec['evaluation_contract']);require(spec['evaluation_contract']['sha256']==h.CONTRACT_SHA,'Frozen full40/25 evaluation required')
 frozen=V/'local_data/v2_cpu_representation_v1/frozen_inputs_PRIVATE.json';require(sha(frozen)==h.FROZEN_SHA,'Frozen dev inputs changed');dev=load(frozen)['dev_PRIVATE'];stats._validate_rows(dev,'internal_dev');seq=[r for r in dev if r['ability']=='mri_sequence'];require(len(dev)==40 and len(seq)==24,'Full40/24 coverage required')
 train_scope=authenticated(spec['training_scope']);training=authenticated(spec['native_complete']);proof=authenticated(spec['training_proof']);child_receipt=authenticated(spec['child_receipt'])
 n,k=validate_training_proof(proof,child_receipt,spec['child_receipt']['sha256'])
 h.authenticate_exit(spec['training_exit']['path'],spec['training_exit']['sha256'],spec['training_closed_log']['path'],spec['training_closed_log']['sha256'],spec['training_scope']['sha256'])
 require(train_scope['kind']=='adaptive_accuracy_gate_v1' and child_receipt['binding']==train_scope['run_binding'] and training['scope_sha256']==h.canon(train_scope),'Adaptive native checkpoint/scope binding differs')
 require(training['completed_groups']==n and training['optimizer_updates']==k and training['absolute_optimizer_updates']==52+k and training['absolute_completed_groups']==114+n and training['status'] in {'UPDATE_BUDGET_REACHED','SCHEDULE_COMPLETE','TOKEN_BUDGET_EXHAUSTED'},'Actual finite terminal differs')
 require(Path(training['boundary'])/'receipt.json'==Path(spec['child_receipt']['path']),'Selected child is not exact terminal')
 original=api.validate_plan(R/'local_data/c_two_task_pool_v1/u7_visual_controls_v1/probe_plan_PRIVATE.json')
 from tokenizers import Tokenizer
 from mri_rl.vero_mri_reward import _answer_and_format
 raw=Tokenizer.from_file(str(Path(original['model_path'])/'tokenizer.json'))
 class Raw:
  def encode(self,text,add_special_tokens=False):return raw.encode(text,add_special_tokens=add_special_tokens).ids
 scopes={};greedy={};candidates={};evidence={}
 for cp,arg,kind,checkpoint in ((PARENT,'parent','adaptive_parent_read_only_evaluation_v1','P_calibrated13'),(CHILD,'child','adaptive_post_read_only_evaluation_v1','P_calibrated13_adaptive')):
  bundle=spec[arg];scope=authenticated(bundle['scope']);receipt=authenticated(bundle['eval_receipt'])
  h.authenticate_exit(bundle['exit_receipt']['path'],bundle['exit_receipt']['sha256'],bundle['closed_log']['path'],bundle['closed_log']['sha256'],bundle['scope']['sha256'])
  require(scope['kind']==kind and scope['checkpoint']==checkpoint and scope['entry_sha256']==h.ENTRY_SHA and scope['ready'] is True and scope['evaluation_contract_sha256']==h.CONTRACT_SHA and scope['run_identity']==entry.scope_identity(scope),'Exact parent/adaptive evaluated scope differs')
  require(scope['native_sources']==api.source_hashes() and scope['derived_cache']==original['derived_cache'] and scope['greedy_max_tokens']==256 and scope['temperature']==0,'Native numerical/runtime contract differs')
  if cp==PARENT:
   parent=scope['parent_checkpoint'];require(scope['adapter']['sha256']==PARENT_ADAPTER and parent['receipt_sha256']==PARENT_RECEIPT and parent['completed_groups']==114 and parent['optimizer_updates']==52,'Actual parent baseline differs')
  else:
   selected=scope['adaptive_child'];require(selected['receipt_sha256']==spec['child_receipt']['sha256'] and selected['proof_sha256']==spec['training_proof']['sha256'] and selected['relative_completed_groups']==n and selected['relative_optimizer_updates']==k and scope['adapter']['sha256']==child_receipt['files_sha256']['adapter.safetensors'],'Evaluated child differs from independently proved terminal')
  require(sha(scope['adapter']['path'])==scope['adapter']['sha256'],'Actual evaluated weights changed')
  feature=scope['feature_evidence'];entry.feature_binding(feature['plan_path'],feature['plan_sha256'],feature['receipt_path'],feature['receipt_sha256']);fp=load(feature['plan_path']);fc=load(feature['receipt_path']);inputs={}
  for i,row in enumerate(fp['rows_PRIVATE']):
   path=Path(feature['plan_path']).parent/f'record_{i:04d}.json';require(sha(path)==fc['record_sha256'][i],'Captured native input evidence differs');inputs[row['id']]=load(path)
  require(sha(evaluation['B28_prefix_path'])==evaluation['B28_prefix_sha256'],'Original exact null input evidence changed')
  null=next(r for r in entry.read_rows(evaluation['B28_prefix_path']) if r['condition']=='no_image' and r['category']=='mri_sequence')
  output=Path(scope['output_directory']);gp=output/'greedy_PRIVATE.jsonl';cpth=output/'candidates_PRIVATE.jsonl'
  require(Path(bundle['greedy_output']['path'])==gp and sha(gp)==bundle['greedy_output']['sha256'] and Path(bundle['candidate_output']['path'])==cpth and sha(cpth)==bundle['candidate_output']['sha256'],'Externally frozen output files differ')
  entry.validate_completion(receipt,scope,evaluation,gp,cpth)
  def vg(row):require(isinstance(row['response'],str) and isinstance(row.get('token_ids'),list) and len(row['token_ids'])<=256 and all(type(t) is int and t>=0 for t in row['token_ids']) and row['image_placeholder_count']==1,'Missing bounded native greedy evidence')
  def vc(row):
   api.validate_numeric_route(row,Raw());expected=null if row['condition']=='no_image' else inputs[row['source_record_id']]
   require(row['prompt_ids']==expected['prompt_ids'] and row['rendered_prompt_sha256']==expected.get('prompt_sha256',expected.get('rendered_prompt_sha256')),'Independent native prompt/prefix expansion differs')
   if row['condition']=='real_image':require(row['image_grid_thw']==expected['image_grid_thw'] and row['image_token_count']==expected['image_token_count'],'Visual token grid differs')
  greedy[cp]=h.exact_rows(entry.read_rows(gp),evaluation['greedy_records_PRIVATE'],scope['run_identity'],vg);candidates[cp]=h.exact_rows(entry.read_rows(cpth),evaluation['candidate_records_PRIVATE'],scope['run_identity'],vc)
  require(all(sha(r['image'])==r['image_sha256'] for r in evaluation['greedy_records_PRIVATE']),'Displayed images changed')
  scopes[cp]=scope;evidence[cp]={name:item['sha256'] for name,item in bundle.items()}
 require(scopes[PARENT]['native_sources']==scopes[CHILD]['native_sources'] and scopes[PARENT]['feature_evidence']==scopes[CHILD]['feature_evidence'],'Parent/child evaluation interfaces differ')
 def gp(rows):
  out=[]
  for row in dev:
   out.append(strict_prediction(rows[row['id']]['response'],stats.LABELS[row['ability']],_answer_and_format))
  return out
 predictions={cp:gp(rows) for cp,rows in greedy.items()};gm={cp:stats.metrics(dev,p) for cp,p in predictions.items()}
 for m in gm.values():
  for t in m['tasks'].values():t['balanced_accuracy']=mean(t['class_recall'].values())
 by_kind={};private={'greedy_predictions_PRIVATE':predictions,'candidate_predictions_PRIVATE':{}}
 for kind in h.KINDS:
  pred={cp:h.candidate_predictions(dev,rows,kind) for cp,rows in candidates.items()};by_kind[kind]=labels(h.matched_candidate_summary(seq,{'B28':pred[PARENT],'post2':pred[CHILD]},seqstats.sequence_metrics,stats.paired_units));private['candidate_predictions_PRIVATE'][kind]=pred
 resources=spec['training_resources'];require(sha(resources['path'])==resources['sha256'],'Closed training resource log changed');events=entry.read_rows(resources['path'])
 result={'schema':'adaptive-accuracy-parent-child-actual-outcomes-v1','status':'ACTUAL_SAVED_PARENT_AND_ADAPTIVE_OUTCOMES_AUTHENTICATED','primary_endpoint':{'readout':'FULL24 sequence MEAN-label candidate balanced accuracy; identical readout parent/adaptive','result':by_kind['mean']},'sequence_candidate_secondary':{a:b for a,b in by_kind.items() if a!='mean'},'full40_greedy_separate_readout':{'format_rule':'Exactly one nonempty boxed answer with full Vero tag format (format score1); partial-credit format is INVALID','checkpoints':gm,'paired_units_uncertainty_descriptive':stats.paired_units(dev,predictions[PARENT],predictions[CHILD])},'visual_controls':visual_controls(h,dev,candidates[PARENT],candidates[CHILD]),'actual_continuation':{'attempted_groups':n,'new_optimizer_updates':k,'parent_optimizer_step':52,'terminal_optimizer_step':52+k,'parent_schedule_cursor':114,'terminal_schedule_cursor':114+n,'training_proof_sha256':spec['training_proof']['sha256']},'training_resources':resource_summary(events),'input_evidence':evidence,'all_external_evidence_sha256':{key:value['sha256'] for key,value in spec.items() if isinstance(value,dict) and 'path' in value and 'sha256' in value},'analyzer_sha256':sha(__file__),'pure_analysis_dependency_sha256':HELPER_SHA,'metric_source_sha256':h.METRIC_SOURCE_SHA256,'null_unique_queries_per_checkpoint':1,'blank_image_controls':'NOT_RUN','sealed_final100_used':False,'limitations':['Previously examined internal dev:4sequence people and4Dallas cycles; clustered intervals are descriptive, not independent generalization.','Single-factor adaptive continuation has no matched strict-gate continuation; before/after cannot isolate the admission-rule causal effect from additional optimization.','Candidate scoring is compared with the same candidate readout; greedy is separate. A shared null query is not24 independent observations.','Acquisition/intensity shortcuts remain; paired separation change is not proof of anatomical reasoning.']}
 return result,private

def main():
 p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--expected-spec-sha256',required=True);p.add_argument('--private-output',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(sha(a.spec)==a.expected_spec_sha256,'Externally authenticated analysis input required');result,private=analyze(load(a.spec));require(a.private_output.absolute().is_relative_to(W/'local_data') and a.output.absolute().is_relative_to(W/'results'),'Private/sanitized destinations required')
 for path,value in ((a.private_output,private),(a.output,result)):
  with path.open('x') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({'status':result['status'],'aggregate_sha256':sha(a.output),'native_models_loaded':0}))
if __name__=='__main__':main()
