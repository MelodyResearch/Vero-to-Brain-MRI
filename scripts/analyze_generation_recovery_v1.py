"""Saved-only fixed40/25 comparison: four existing weights, then two single-factor Adam60 generation controls."""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,math,statistics,sys
from collections import Counter
from pathlib import Path
W=Path(__file__).resolve().parents[1]
ADAPTER_ANALYSIS_SHA='8e05cbe91fb046ca1bd53b991ff15d543edeafa7af9784ae04cfeec0e3342904'
ENTRY_SHA='d0202667c4a580c4ea34cdd22a31a2238fd2de8ad81f26277600f351b5dd152c'
EVAL_SHA='8c52a40f572393f4a01c6fbd2eb84d044bc14ecd151ae00ef56d9e7600a2539b'
REWARD_SHA='411ab099edcfb45305630c675f7ca4870d6501e69dd084c9e1218ad2cc7a0ed8'
NAMES=('B28','Adam52','Adam56','Adam60')
BASELINE_IDENTITIES={
 'B28':('B28','b316db3d5eadf2e1fc0cbe80ccc93c7f7405858769f7d32b39a137dea3d6c572','f0b74a9474713b7588f5170053ecc5133497a691916d4a6a3c89c205e24bad11'),
 'Adam52':('P_calibrated13','c70b426b58191446545ba6333f7595e96f4fdb4cc689ddbaf5206e276b329db6','cc57a97ec4e5825d5cd86a691ec35b13ffac18cf520e187b39a2c8765d1c1d31'),
 'Adam56':('P_calibrated13_adaptive','23b6ce522f770bf39acbadd9b66ccf00529de7822a5bb4173e7c4b96cd8618a7','022c97281506fce016d8dc74dcec983c0aa52797f34c92baf02c4189717a84f8'),
 'Adam60':('uniform_category_continuation','1741ff2feb2f569cfe062b92cb79b62f61f83463e6683a928d2c458a5f547310','860fde061f6dfe106878b1c5bfb50850ccae541460afbf4bb17c2e8fec060a6f')}
def require(ok,message):
 if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return json.loads(Path(path).read_text())
def check(item):require(sha(item['path'])==item['sha256'],'Externally frozen saved evidence differs')
def auth(item):check(item);return load(item['path'])
def module(path,name,expected):
 require(sha(path)==expected,'Frozen analysis dependency changed');s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def libraries(v2):
 adapter=module(W/'scripts/analyze_adaptive_accuracy_outcomes_v1.py','generation_saved_adapter',ADAPTER_ANALYSIS_SHA);h=adapter.load_helpers(v2);stats,seqstats=h.load_statistics(v2);reward=module(Path(v2)/'src/mri_rl/vero_mri_reward.py','generation_frozen_strict_reward',REWARD_SHA);return adapter,h,stats,seqstats,reward

def saved_wall_times(rows,field):
 values=[r[field] for r in rows if field in r]
 require(all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) and v>=0 for v in values),"Invalid saved actual wall duration")
 return {"records":len(rows),"records_with_actual_elapsed":len(values),"records_without_actual_elapsed":len(rows)-len(values),"actual_elapsed_seconds_sum":sum(values) if values else None,"actual_elapsed_seconds_mean":statistics.mean(values) if values else None}

def generation_metadata(rows):
 present=[r for r in rows if isinstance(r.get('token_ids'),list)];lengths=[len(r['token_ids']) for r in present]
 return {'records':len(rows),'records_with_actual_token_ids':len(present),'records_without_emitted_token_metadata':len(rows)-len(present),'saved_generated_tokens_sum':sum(lengths) if lengths else None,'saved_generated_token_counts':{'min':min(lengths),'median':statistics.median(lengths),'max':max(lengths)} if lengths else None,'wall_time':saved_wall_times(rows,'generation_wall_seconds'),'finish_reason_counts':dict(Counter(r.get('finish_reason','NOT_RECORDED_IN_THIS_SAVED_BRIDGE') for r in rows)),'hit_max_tokens_records':sum(r.get('hit_max_tokens') is True for r in rows),'interpretation':'Only emitted token_ids from the actual saved generation are counted; visible text is never re-encoded as actual compute.'}

def cap_prefix_agreement(baseline,control,baseline_predictions,control_predictions,baseline_cap=256):
 require(len(baseline)==len(control)==len(baseline_predictions)==len(control_predictions),"Paired token-prefix coverage differs")
 count=Counter();divergence=Counter()
 for old,new,p,q in zip(baseline,control,baseline_predictions,control_predictions):
  a=old.get('token_ids');b=new.get('token_ids');require(isinstance(a,list) and isinstance(b,list),"Actual saved token IDs required for cap-prefix comparison")
  different=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),None);same=different is None;capped=old.get('finish_reason')=='length' and len(a)==baseline_cap
  count['shared_prefix_identical']+=same;count['baseline_tokens_are_complete_prefix_of_control']+=same and len(b)>=len(a)
  if different is not None:divergence[str(different)]+=1
  if capped:count['baseline_capped_records']+=1;count['capped_baseline_shared_prefix_identical']+=same
  else:
   count['completed_baseline_records']+=1;count['completed_baseline_text_unchanged']+=old['response']==new['response'];count['completed_baseline_strict_label_unchanged']+=p==q
 return {'records':len(baseline),**{k:count[k] for k in ('shared_prefix_identical','baseline_tokens_are_complete_prefix_of_control','baseline_capped_records','capped_baseline_shared_prefix_identical','completed_baseline_records','completed_baseline_text_unchanged','completed_baseline_strict_label_unchanged')},'first_divergence_zero_based_position_counts':dict(divergence),'interpretation':'Shared-prefix equality is measured, never required. A cap-dependent first-token divergence prevents interpreting that trajectory as simply extending the original truncated output.'}

def cohort_summary(v2,dev,greedy,candidates):
 adapter,h,stats,seqstats,reward=libraries(v2);pred=[];formats=Counter();invalid=Counter()
 for r in dev:
  text=greedy[r['id']]['response'];formats[str(reward._answer_and_format(text)[1])]+=1;p=adapter.strict_prediction(text,stats.LABELS[r['ability']],reward._answer_and_format);pred.append(p)
  if p<0:invalid[r['ability']]+=1
 metrics=stats.metrics(dev,pred)
 for task in metrics['tasks'].values():task['balanced_accuracy']=statistics.mean(task['class_recall'].values())
 seq=[r for r in dev if r['ability']=='mri_sequence'];candidate={};cpred={}
 for kind in h.KINDS:
  cpred[kind]=h.candidate_predictions(dev,candidates,kind);candidate[kind]=seqstats.sequence_metrics(seq,cpred[kind]);candidate[kind]['balanced_accuracy']=statistics.mean(candidate[kind]['recall'].values())
 rows=[greedy[r['id']] for r in dev];summary={'strict_greedy':metrics,'strict_format_score_counts':dict(formats),'greedy_invalid_by_task':{t:invalid[t] for t in stats.LABELS},'generation_metadata':generation_metadata(rows),'generation_metadata_by_task':{t:generation_metadata([greedy[r['id']] for r in dev if r['ability']==t]) for t in stats.LABELS},'candidate':candidate,'candidate_wall_time':saved_wall_times(list(candidates.values()),'candidate_wall_seconds'),'null_greedy':'NOT_MEASURED_IN_STANDARD40_EVALUATION','visual_summary':h.sequence_controls(dev,candidates,candidates)['checkpoints']['B28']}
 return summary,{'greedy':pred,'candidate':cpred}

def renamed(x,parent,child):
 keys={'B28':parent,'post2':child,'U2_minus_B28':'difference','post2_minus_B28_balanced_accuracy':'difference_balanced_accuracy'}
 if isinstance(x,dict):return {keys.get(k,k):renamed(v,parent,child) for k,v in x.items()}
 if isinstance(x,list):return [renamed(v,parent,child) for v in x]
 return x

def analyze(spec,baseline_only):
 R=Path(spec['primary_root']);V=Path(spec['v2_root']);sys.path.insert(0,str(R/'src'));adapter,h,stats,seqstats,reward=libraries(V);entry=module(V/'scripts/run_paired_discrimination_pilot_v2.py','generation_evaluator_validation',ENTRY_SHA);early=module(V/'scripts/run_c40_early_visual_probe_v1.py','generation_loader_validation','8cafacbf3a9086160c8e257673c6f933070df208bffc5c32dec38bc0e1e82e06');api=early.primary_api(R)
 evaluation=auth(spec['evaluation_contract']);require(spec['evaluation_contract']['sha256']==EVAL_SHA,'Exact full40/25 evaluation required');frozen=V/'local_data/v2_cpu_representation_v1/frozen_inputs_PRIVATE.json';require(sha(frozen)==h.FROZEN_SHA,'Frozen dev pack changed');dev=load(frozen)['dev_PRIVATE'];stats._validate_rows(dev,'internal_dev');require(len(dev)==40,'Full40 only')
 from tokenizers import Tokenizer
 tokenizer=Tokenizer.from_file(str(R/'local_data/models/qwen3vl8b-base/tokenizer.json'))
 class Raw:
  def encode(self,text,add_special_tokens=False):return tokenizer.encode(text,add_special_tokens=add_special_tokens).ids
 tables={};predictions={};greedy={};candidates={};evidence={};scopes={}
 require(set(spec['baselines'])==set(NAMES),'All four predetermined checkpoints required')
 for name in NAMES:
  b=spec['baselines'][name];scope=auth(b['scope']);receipt=auth(b['eval_receipt']);h.authenticate_exit(b['exit_receipt']['path'],b['exit_receipt']['sha256'],b['closed_log']['path'],b['closed_log']['sha256'],b['scope']['sha256']);require(scope['ready'] and scope['entry_sha256']==ENTRY_SHA and scope['evaluation_contract_sha256']==EVAL_SHA and scope['run_identity']==entry.scope_identity(scope) and scope['temperature']==0 and scope['greedy_max_tokens']==256,'Frozen standard generation contract differs')
  require((scope['checkpoint'],scope['adapter']['sha256'],b['scope']['sha256'])==BASELINE_IDENTITIES[name],'Named checkpoint, adapter and exact scope identity disagree')
  check(b['greedy_output']);check(b['candidate_output']);entry.validate_completion(receipt,scope,evaluation,Path(b['greedy_output']['path']),Path(b['candidate_output']['path']))
  greedy[name]=h.exact_rows(entry.read_rows(Path(b['greedy_output']['path'])),evaluation['greedy_records_PRIVATE'],scope['run_identity'],lambda r:require(isinstance(r['response'],str),'Missing saved response'))
  candidates[name]=h.exact_rows(entry.read_rows(Path(b['candidate_output']['path'])),evaluation['candidate_records_PRIVATE'],scope['run_identity'],lambda r:api.validate_numeric_route(r,Raw()))
  tables[name],predictions[name]=cohort_summary(V,dev,greedy[name],candidates[name]);tables[name]['adapter_sha256']=scope['adapter']['sha256'];tables[name]['candidate_provenance']='AUTHENTICATED_EXISTING_ORIGINAL_PROMPT_RECORDS';tables[name]['resource_time']=None
  if 'resource_log' in b:
   check(b['resource_log']);tables[name]['resource_time']=adapter.resource_summary(entry.read_rows(Path(b['resource_log']['path'])))
  tables[name]['time_interpretation']='Observed evaluator resource sampling span, including candidate work; not generation-only runtime or training time. B28 reuses40 saved greedy outputs and5candidate rows, unlike the later full fresh evaluations.';evidence[name]={k:v['sha256'] for k,v in b.items()};scopes[name]=scope
 if not baseline_only:
  g=spec['controls'];scope=auth(g['scope']);receipt=auth(g['completion']);h.authenticate_exit(g['exit_receipt']['path'],g['exit_receipt']['sha256'],g['closed_log']['path'],g['closed_log']['sha256'],g['scope']['sha256']);require(scope['kind']=='generation_recovery_inference_v1' and scope['ready'] and scope['adapter']==scopes['Adam60']['adapter'] and scope['evaluation_contract_sha256']==EVAL_SHA and scope['optimizer_updates']==scope['backward_calls']==0,'Exact fixed-weight generation controls required')
  runner=module(W/'scripts/run_generation_recovery_v1.py','generation_control_contract',scope['native_sources']['runner']);ctl=module(W/'src/mri_rl/generation_recovery_v1.py','generation_control_records',scope['native_sources']['helper']);require(scope['baseline_evidence']=={k:spec['baselines']['Adam60'][k] for k in scope['baseline_evidence']},'Control baseline differs from actual Adam60 records');plans=ctl.control_records(evaluation);paths=ctl.output_paths(scope['output_directory']);ctl.validate_completion(receipt,scope,paths);require(receipt['scope_sha256']==ctl.canon(scope),'Control completion is not bound to the actually executed scope')
  for name,key in (('Adam60_original_512','original'),('Adam60_short_reason_256','short')):
   condition='original_512' if key=='original' else 'short_reason_256';plan=plans[condition];cap=plan['max_tokens'];grows=ctl.durable_prefix(paths[key+'_greedy'],plan['greedy'],scope['run_identity'],lambda r:ctl.validate_greedy(r,cap));require(len(grows)==40,'Complete40 control generations required');null=next(r for r in plan['candidates'] if r['condition']=='no_image');nullrows=ctl.durable_prefix(paths[key+'_null'],[null],scope['run_identity'],lambda r:ctl.validate_greedy(r,cap));require(len(nullrows)==1,'One actual control null generation required');greedy[name]={r['record_id']:r for r in grows}
   if key=='original':candidates[name]=candidates['Adam60']
   else:
    rows=ctl.durable_prefix(paths['short_candidates'],plan['candidates'],scope['run_identity'],lambda r:api.validate_numeric_route(r,Raw()));require(len(rows)==25,'Complete25 short-prompt candidates required');candidates[name]={r['record_id']:r for r in rows}
   tables[name],predictions[name]=cohort_summary(V,dev,greedy[name],candidates[name]);tables[name].update(null_greedy={'unique_queries':1,'strict_label_index':adapter.strict_prediction(nullrows[0]['response'],stats.LABELS['mri_sequence'],reward._answer_and_format),'format_score':reward._answer_and_format(nullrows[0]['response'])[1],'generation_metadata':generation_metadata(nullrows),'interpretation':'One shared sequence null query, not24 independent null samples.'},adapter_sha256=scope['adapter']['sha256'],candidate_provenance='BYTE_EXACT_ADAM60_ORIGINAL_PROMPT_BASELINE_REFERENCE' if key=='original' else 'ACTUAL_NEW_SHORT_PROMPT_SCORES_SAME_WEIGHTS',maximum_generation_tokens=cap,weights_changed=False)
  evidence['controls']={k:v['sha256'] for k,v in g.items()};check(g['resource_log']);tables['shared_control_resource_time']=adapter.resource_summary(entry.read_rows(Path(g['resource_log']['path'])))
 comparisons={}
 if not baseline_only:
  tables['Adam60_original_512']['actual_token_prefix_vs_256']=cap_prefix_agreement([greedy['Adam60'][r['id']] for r in dev],[greedy['Adam60_original_512'][r['id']] for r in dev],predictions['Adam60']['greedy'],predictions['Adam60_original_512']['greedy'])
 for parent,child,kind in [('B28','Adam52','WEIGHT_CHECKPOINT_OBSERVED'),('Adam52','Adam56','WEIGHT_CHECKPOINT_OBSERVED'),('Adam56','Adam60','WEIGHT_CHECKPOINT_OBSERVED')]+([] if baseline_only else [('Adam60','Adam60_original_512','CAP_ONLY_FIXED_WEIGHTS'),('Adam60','Adam60_short_reason_256','PROMPT_ONLY_FIXED_WEIGHTS')]):
  comparisons[parent+'__to__'+child]={'factor':kind,'strict_greedy_paired_units':stats.paired_units(dev,predictions[parent]['greedy'],predictions[child]['greedy']),'candidate':{k:renamed(h.matched_candidate_summary([r for r in dev if r['ability']=='mri_sequence'],{'B28':predictions[parent]['candidate'][k],'post2':predictions[child]['candidate'][k]},seqstats.sequence_metrics,stats.paired_units),parent,child) for k in h.KINDS},'visual_controls':renamed(h.sequence_controls(dev,candidates[parent],candidates[child]),parent,child)}
 result={'schema':'generation-recovery-saved-fixed-cohort-comparison-v1','status':'ACTUAL_FOUR_EXISTING_CHECKPOINTS_AUTHENTICATED' if baseline_only else 'ACTUAL_TWO_GENERATION_CONTROLS_AND_FOUR_CHECKPOINTS_AUTHENTICATED','endpoint':'Strict full-Vero-format greedy accuracy is the generation-control endpoint; candidate readouts remain separate diagnostics and do not alter historical experiment endpoints.','cohorts':tables,'paired_comparisons':comparisons,'evidence_sha256':evidence,'analysis_adapter_sha256':ADAPTER_ANALYSIS_SHA,'metric_source_sha256':h.METRIC_SOURCE_SHA256,'script_sha256':sha(__file__),'native_models_loaded':0,'sealed_final100_used':False,'blank_image_controls':'NOT_RUN','limitations':['All40 records are repeatedly examined internal development:4sequence people and4Dallas cycles; clustered intervals are descriptive.','A512 and short256 are each compared only against the same-weight original256 Adam60 baseline; no combined condition was run.','Weight-checkpoint differences lack matched strict/proportional continuation counterfactuals and do not isolate an algorithmic benefit.','Forced candidate scoring is not generated-answer accuracy; cap-only candidate scores are referenced, not recomputed.','No missing emitted token metadata is replaced by re-encoded visible text; resource spans are not pure generation runtimes.']}
 return result,{'predictions_PRIVATE':predictions}

def main():
 p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--expected-spec-sha256',required=True);p.add_argument('--baseline-only',action='store_true');p.add_argument('--private-output',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(sha(a.spec)==a.expected_spec_sha256,'Externally bound complete analysis input required');result,private=analyze(load(a.spec),a.baseline_only);require(a.private_output.absolute().is_relative_to(W/'local_data') and a.output.absolute().is_relative_to(W/'results'),'Private/public destination required')
 for path,value in ((a.private_output,private),(a.output,result)):
  with path.open('x') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({'status':result['status'],'aggregate_sha256':sha(a.output),'native_models_loaded':0}))
if __name__=='__main__':main()
