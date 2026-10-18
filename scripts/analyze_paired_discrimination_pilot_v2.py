"""Outcome processing ONLY after actual exit0 of fixed B28/post2 evaluations.
Full24 mean-candidate balanced accuracy is primary; full40 greedy is separate.
No model, training, inference, hyperparameter selection or sealed-test access.
"""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,math,sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
W=Path(__file__).resolve().parents[1]
ENTRY_SHA='d0202667c4a580c4ea34cdd22a31a2238fd2de8ad81f26277600f351b5dd152c'
CONTRACT_SHA='8c52a40f572393f4a01c6fbd2eb84d044bc14ecd151ae00ef56d9e7600a2539b'
FROZEN_SHA='446808a0203ba1e6dd9f45e7f052a7667420b016ffff5a5b88f295d728fc79ab'
KINDS=('mean','raw','first_label');LABELS=('FLAIR','contrast-enhanced T1')
METRIC_SOURCE_SHA256={'v2_cpu_representation.py':'da2097c1dbcbf465fae1568fc31004418dd7acc65b498d1b267a11ecb91029e1','v2_qwen_feature_probe.py':'01127773780070ef4824cf2ebe1b9c4308b9e48bc8b9d272e770216d13ea55fc'}
def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def canon(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def module(path,name,expected=None):
 if expected:require(sha(path)==expected,'Frozen analysis dependency changed')
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def load_statistics(root):
 return tuple(module(Path(root)/'src/mri_rl'/name,'paired_bound_'+name.removesuffix('.py'),expected) for name,expected in METRIC_SOURCE_SHA256.items())

def authenticated(path,expected):require(sha(path)==expected,'Externally supplied evidence digest differs');return load(path)
def authenticate_exit(path,expected,log,log_sha,scope_sha):
 r=authenticated(path,expected);require(sha(log)==log_sha==r['raw_log_sha256'] and r['exit_code']==0 and r['scope_file_sha256']==scope_sha,'Actual exit0/closed-log/scope witness mismatch');return r

def exact_rows(rows,records,identity,validator):
 require(len(rows)==len(records) and len({r['record_id'] for r in rows})==len(rows),'Exact unique full output coverage required');result={}
 for row,expected in zip(rows,records):
  json.dumps(row,allow_nan=False);require(row['run_identity']==identity and row['weight_updates']==0 and all(row.get(k)==v for k,v in expected.items()),'Frozen prompt/displayed image/target/identity differs')
  validator(row);result[row['record_id']]=row
 return result

def margin(row,kind):
 if kind=='first_label':return row['prefix_oracle_first_logps'][1]-row['prefix_oracle_first_logps'][0]
 return row['comparisons'][kind+'_t1_minus_flair']
def sign_prediction(value):return 1 if value>0 else 0 if value<0 else -1
def candidate_predictions(rows,candidates,kind):return [sign_prediction(margin(candidates[r['id']],kind)) for r in rows if r['ability']=='mri_sequence']
def matched_candidate_summary(sequence_rows,predictions,metric,paired_units):
 report={cp:metric(sequence_rows,pred) for cp,pred in predictions.items()}
 for value in report.values():value['balanced_accuracy']=mean(value['recall'].values())
 return {'checkpoints':report,'post2_minus_B28_balanced_accuracy':report['post2']['balanced_accuracy']-report['B28']['balanced_accuracy'],'paired_people_uncertainty_descriptive':paired_units(sequence_rows,predictions['B28'],predictions['post2'])['mri_sequence']}

def sequence_controls(rows,base,post):
 sequence=[r for r in rows if r['ability']=='mri_sequence'];require(len(sequence)==24,'Full24 sequence images required');pairs=defaultdict(dict)
 for row in sequence:
  e=row['extra_info'];key=(e['patient_key'],e['level']);truth=row['reward_model']['ground_truth'];require(truth in LABELS and truth not in pairs[key],'Invalid displayed-truth sequence pair');pairs[key][truth]=row['id']
 require(len(pairs)==12 and len({k[0] for k in pairs})==4 and all(set(v)==set(LABELS) for v in pairs.values()),'Full12pairs clustered4people required')
 result={'checkpoints':{},'U2_minus_B28':{}};private={}
 for checkpoint,candidates in [('B28',base),('post2',post)]:
  null=[r for r in candidates.values() if r['condition']=='no_image'];require(len(null)==1,'Exactly ONE null query, not24 null samples');summary={};private[checkpoint]={}
  for kind in KINDS:
   null_margin=margin(null[0],kind);pred=sign_prediction(null_margin);units=defaultdict(list)
   for (person,level),pair in pairs.items():units[person].append(margin(candidates[pair[LABELS[1]]],kind)-margin(candidates[pair[LABELS[0]]],kind))
   values=[margin(candidates[r['id']],kind) for r in sequence];unit_values={p:mean(v) for p,v in units.items()}
   summary[kind]={'null_unique_queries':1,'null_margin':null_margin,'null_preferred_class':LABELS[pred] if pred>=0 else 'TIE',
    'implied_constant_class_accuracy_on24_truths':sum(pred==LABELS.index(r['reward_model']['ground_truth']) for r in sequence)/24,
    'null_accuracy_interpretation':'One unique no-image query; its single preference is applied to the24 frozen balanced reference labels, not24 independent null measurements.',
    'real_mean_margin':mean(values),'mean_real_minus_null_margin':mean(v-null_margin for v in values),
    'patient_mean_paired_separation':mean(unit_values.values()),'patient_pair_separation_range':[min(unit_values.values()),max(unit_values.values())],
    'sequence_pairs':12,'independent_people':4,'levels_per_person':3}
   private[checkpoint][kind]=unit_values
  result['checkpoints'][checkpoint]=summary
 for kind in KINDS:
  a=result['checkpoints']['B28'][kind];b=result['checkpoints']['post2'][kind]
  result['U2_minus_B28'][kind]={'delta_null_margin':b['null_margin']-a['null_margin'],'delta_patient_mean_paired_separation':b['patient_mean_paired_separation']-a['patient_mean_paired_separation'],'delta_mean_real_minus_null_margin':b['mean_real_minus_null_margin']-a['mean_real_minus_null_margin'],'separation_interpretation':'Common additive answer-prior shifts cancel in opposite-image paired separation; changed separation is a descriptive interaction, not proof of brain-specific grounding.'}
 return result

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--primary-root',type=Path,required=True)
 for name in ('evaluation-contract','training-receipt'):
  p.add_argument('--'+name,type=Path,required=True);p.add_argument('--'+name+'-sha256',required=True)
 for cp in ('b28','post2'):
  for name in ('scope','eval-receipt','exit-receipt','closed-log'):
   p.add_argument('--'+cp+'-'+name,type=Path,required=True);p.add_argument('--'+cp+'-'+name+'-sha256',required=True)
 p.add_argument('--private-output',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 entry=module(W/'scripts/run_paired_discrimination_pilot_v2.py','paired_analyzer_entry',ENTRY_SHA);require(a.evaluation_contract_sha256==CONTRACT_SHA,'Prespecified evaluation contract required');evaluation=authenticated(a.evaluation_contract,a.evaluation_contract_sha256)
 R=a.primary_root.resolve();sys.path.insert(0,str(R/'src'));early=module(W/'scripts/run_c40_early_visual_probe_v1.py','paired_analyzer_loader','8cafacbf3a9086160c8e257673c6f933070df208bffc5c32dec38bc0e1e82e06');api=early.primary_api(R)
 stats,seqstats=load_statistics(W)
 frozen_path=W/'local_data/v2_cpu_representation_v1/frozen_inputs_PRIVATE.json';require(sha(frozen_path)==FROZEN_SHA,'Frozen dev pack changed');dev=load(frozen_path)['dev_PRIVATE'];stats._validate_rows(dev,'internal_dev');seqdev=[r for r in dev if r['ability']=='mri_sequence']
 require(len(dev)==40 and len(seqdev)==24,'Fixed full40/24 dev coverage required')
 training=authenticated(a.training_receipt,a.training_receipt_sha256);require(training['status']=='ACTUAL_TWO_UPDATE_PAIRED_PILOT_COMPLETE_PENDING_ACCURACY' and training['checkpoint']['cursor']==training['checkpoint']['optimizer_updates']==2 and training['evaluation_contract_sha256']==CONTRACT_SHA,'Exact two-update completed pilot required')
 from mri_rl.experiment_contract import load_contract
 from mri_rl.vero_mri_reward import _answer_and_format
 from tokenizers import Tokenizer
 original=api.validate_plan(R/'local_data/c_two_task_pool_v1/u7_visual_controls_v1/probe_plan_PRIVATE.json');rawtok=Tokenizer.from_file(str(Path(original['model_path'])/'tokenizer.json'))
 class Raw:
  def encode(self,text,add_special_tokens=False):return rawtok.encode(text,add_special_tokens=add_special_tokens).ids
 saved,saved_sha=api.saved_predictions('B28');require(saved_sha==evaluation['B28_saved_greedy_sha256'],'Saved B28 full40 baseline changed')
 scopes={};greedy={};candidates={};evidence={}
 for cp,arg in [('B28','b28'),('post2','post2')]:
  scope=authenticated(getattr(a,arg+'_scope'),getattr(a,arg+'_scope_sha256'));receipt=authenticated(getattr(a,arg+'_eval_receipt'),getattr(a,arg+'_eval_receipt_sha256'))
  authenticate_exit(getattr(a,arg+'_exit_receipt'),getattr(a,arg+'_exit_receipt_sha256'),getattr(a,arg+'_closed_log'),getattr(a,arg+'_closed_log_sha256'),getattr(a,arg+'_scope_sha256'))
  require(scope['kind']=='paired_discrimination_pilot_eval_v2' and scope['checkpoint']==cp and scope['entry_sha256']==ENTRY_SHA and scope['ready'] is True and scope['evaluation_contract_sha256']==CONTRACT_SHA and scope['run_identity']==entry.scope_identity(scope) and scope['native_sources']==api.source_hashes(),'Exact reviewed native evaluation scope differs')
  require(scope['greedy_max_tokens']==256 and scope['temperature']==0 and scope['derived_cache']==original['derived_cache'],'Frozen precision/cache/generation contract differs')
  entry.feature_binding(scope['feature_evidence']['plan_path'],scope['feature_evidence']['plan_sha256'],scope['feature_evidence']['receipt_path'],scope['feature_evidence']['receipt_sha256'])
  feature_plan=load(scope['feature_evidence']['plan_path']);feature_complete=load(scope['feature_evidence']['receipt_path']);feature_records={}
  for index,expected in enumerate(feature_plan['rows_PRIVATE']):
   path=Path(scope['feature_evidence']['plan_path']).parent/f'record_{index:04d}.json';require(sha(path)==feature_complete['record_sha256'][index],'Actual feature input evidence hash differs');value=load(path);require(value['record_id']==expected['id'] and value['image_sha256']==sha(expected['images'][0]),'Actual feature prompt/image binding differs');feature_records[value['record_id']]=value
  old_prefix=entry.read_rows(evaluation['B28_prefix_path']);require(sha(evaluation['B28_prefix_path'])==evaluation['B28_prefix_sha256'],'Original null prompt input evidence differs');old_null=next(r for r in old_prefix if r['condition']=='no_image' and r['category']=='mri_sequence')
  if cp=='B28':require(scope['adapter']==original['checkpoints']['B28'],'No-new-supervision B28 baseline differs')
  else:
   require(scope['starting_journal']==training['checkpoint'] and scope['adapter']['sha256']==training['checkpoint']['adapter_sha256'],'Evaluated model is not exact two-update journal checkpoint')
   paired=module(W/'src/mri_rl/paired_discrimination_pilot_v2.py','paired_analyzer_objective',scope['paired_source_sha256']);pilot=authenticated(scope['pilot_plan_path'],scope['pilot_plan_sha256']);state=entry.checkpoint_state(Path(scope['run_directory']),pilot,paired)
   require(state==training['checkpoint'],'Two-update adapter/optimizer/RNG journal hash chain differs')
  out=Path(scope['output_directory']);gp=out/'greedy_PRIVATE.jsonl';cpth=out/'candidates_PRIVATE.jsonl';entry.validate_completion(receipt,scope,evaluation,gp,cpth)
  def validate_greedy(row):
   require(isinstance(row['response'],str),'Missing model response')
   if cp=='B28':require(row['response']==saved[row['record_id']]['response'] and row['reused_saved_common_B28_greedy'] is True,'B28 matching-image response bridge changed')
   else:require(isinstance(row.get('token_ids'),list) and len(row['token_ids'])<=256 and all(type(t) is int and t>=0 for t in row['token_ids']) and row['image_placeholder_count']==1,'Actual bounded greedy token/image evidence missing')
  grows=exact_rows(entry.read_rows(gp),evaluation['greedy_records_PRIVATE'],scope['run_identity'],validate_greedy)
  def validate_candidate(row):
   api.validate_numeric_route(row,Raw())
   expected=old_null if row['condition']=='no_image' else feature_records[row['source_record_id']]
   require(row['prompt_ids']==(expected['prompt_ids']) and row['rendered_prompt_sha256']==expected.get('prompt_sha256',expected.get('rendered_prompt_sha256')),'Independent actual input-token/rendered-prompt evidence differs')
   if row['condition']=='real_image':require(row['image_grid_thw']==expected['image_grid_thw'] and row['image_token_count']==expected['image_token_count'],'Actual visual-grid expansion differs')
  crows=exact_rows(entry.read_rows(cpth),evaluation['candidate_records_PRIVATE'],scope['run_identity'],validate_candidate)
  require(all(sha(r['image'])==r['image_sha256'] for r in evaluation['greedy_records_PRIVATE']),'Displayed image bytes changed')
  greedy[cp]=grows;candidates[cp]=crows;scopes[cp]=scope;evidence[cp]={field:getattr(a,arg+'_'+field+'_sha256') for field in ('scope','eval_receipt','exit_receipt','closed_log')}
 require(scopes['B28']['pilot_plan_sha256']==scopes['post2']['pilot_plan_sha256'] and scopes['B28']['sources']==scopes['post2']['sources'] and scopes['B28']['feature_evidence']==scopes['post2']['feature_evidence'],'Before/after experimental bindings differ')
 def greedy_predictions(rows):
  result=[]
  for row in dev:
   answer,fmt=_answer_and_format(rows[row['id']]['response']);labels=stats.LABELS[row['ability']];lookup={v.lower():i for i,v in enumerate(labels)};result.append(lookup.get(answer.lower(),-1) if fmt>0 else -1)
  return result
 gpred={cp:greedy_predictions(rows) for cp,rows in greedy.items()};greedy_metrics={cp:stats.metrics(dev,pred) for cp,pred in gpred.items()}
 for value in greedy_metrics.values():
  for task in value['tasks'].values():task['balanced_accuracy']=mean(task['class_recall'].values())
 by_kind={};private={'greedy_predictions_PRIVATE':gpred,'candidate_predictions_PRIVATE':{}}
 for kind in KINDS:
  pred={cp:candidate_predictions(dev,rows,kind) for cp,rows in candidates.items()};by_kind[kind]=matched_candidate_summary(seqdev,pred,seqstats.sequence_metrics,stats.paired_units);private['candidate_predictions_PRIVATE'][kind]=pred
 controls=sequence_controls(dev,candidates['B28'],candidates['post2'])
 result={'schema':'paired-discrimination-two-update-actual-outcomes-v2','status':'ACTUAL_FIXED_TWO_UPDATE_MODEL_AND_B28_AUTHENTICATED','primary_endpoint':{'readout':'FULL24 sequence MEAN-label candidate balanced accuracy, SAME readout before/after','result':by_kind['mean']},'sequence_candidate_secondary':{k:v for k,v in by_kind.items() if k!='mean'},
  'full40_greedy_separate_readout':{'checkpoints':greedy_metrics,'paired_units_uncertainty_descriptive':stats.paired_units(dev,gpred['B28'],gpred['post2'])},'visual_controls':controls,'input_evidence':evidence,'training_receipt_sha256':a.training_receipt_sha256,'evaluation_contract_sha256':CONTRACT_SHA,'analyzer_sha256':sha(__file__),'bound_metric_source_sha256':METRIC_SOURCE_SHA256,
  'optimizer_updates':2,'checkpoint_selection':'Exact prespecified boundary0002 only; no best-checkpoint selection','null_unique_queries_per_checkpoint':1,'blank_image_controls':'NOT_RUN','sealed_final100_used':False,
  'limitations':['Previously examined internal development:4sequence people,4Dallas cycles; descriptive clustered bootstrap does not establish population, scanner/site or independent-seed generalization.',
   'Primary gain is same candidate readout before/after; saved old greedy baseline is never compared with candidate scoring as a weight-improvement estimate.',
   'One null query is applied to24 balanced truth labels only as constant-class reference, never counted as24 independent null observations.',
   'Changed paired image discrimination does not establish anatomical reasoning; acquisition/nonbrain shortcuts remain possible.']}
 require(a.private_output.resolve().is_relative_to(W/'local_data') and a.output.resolve().is_relative_to(W/'results'),'Private details and sanitized aggregate destinations required')
 entry.save(a.private_output,private);result['private_analysis_sha256']=sha(a.private_output);entry.save(a.output,result);print(json.dumps({'status':result['status'],'primary_delta':by_kind['mean']['post2_minus_B28_balanced_accuracy'],'aggregate_sha256':sha(a.output),'native_models_loaded':0}))
if __name__=='__main__':main()
