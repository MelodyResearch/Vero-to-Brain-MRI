"""Independent CPU arithmetic over completed parent/adaptive saved outputs; no native calls."""
import argparse,collections,hashlib,importlib.util,json,sys
from pathlib import Path
W=Path(__file__).resolve().parents[1];V=W.parent/'vero-mri-v2-controls';R=W.parents[1]
PURE_SHA='f07f08a810e77f0d30cfea41a98207303ab8acbc5ff6f70369cdf80e11b411a0'
ANALYZER_SHA='8e05cbe91fb046ca1bd53b991ff15d543edeafa7af9784ae04cfeec0e3342904'
SPEC_SHA='3080fdb62ceaba2a6ff224afe443ee31ce07e8ba1180d38e3156eb5310e34569'
RESULT_SHA='6d5313e8da010ec4acc5392851faf18726132676de4b4d592e1e57b0a2c6ab14'
CHECKPOINTS=('P_calibrated13_parent','adaptive');KINDS=('mean','raw','first_label')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def req(v,msg):
 if not v:raise ValueError(msg)
def module(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def verify(a):
 req(a.reviewer_id.strip(),'Truthful actual reviewer required');pure=V/'scripts/verify_paired_pilot_outcomes_cpu.py';req(sha(pure)==PURE_SHA,'Independent arithmetic helper changed');m=module(pure,'independent_saved_arithmetic')
 sp=W/'local_data/adaptive_accuracy_gate_v1/analysis_spec_ACTUAL_COMPLETE_PRIVATE.json';req(sha(sp)==SPEC_SHA,'Externally supplied actual spec changed');spec=load(sp)
 # Authenticate all external evidence before reading any numerical analyzer result.
 for value in spec.values():
  if isinstance(value,dict) and 'path' in value: req(sha(value['path'])==value['sha256'],'External input digest changed')
 contract=load(spec['evaluation_contract']['path']);req(spec['evaluation_contract']['sha256']=='8c52a40f572393f4a01c6fbd2eb84d044bc14ecd151ae00ef56d9e7600a2539b','Frozen readout differs')
 fp=V/'local_data/v2_cpu_representation_v1/frozen_inputs_PRIVATE.json';req(sha(fp)=='446808a0203ba1e6dd9f45e7f052a7667420b016ffff5a5b88f295d728fc79ab','Frozen dev changed');dev=load(fp)['dev_PRIVATE'];seq=[r for r in dev if r['ability']=='mri_sequence'];req(len(dev)==40 and len(seq)==24,'Wrong40/24coverage')
 parserpath=R/'src/mri_rl/vero_mri_reward.py';req(sha(parserpath)=='411ab099edcfb45305630c675f7ca4870d6501e69dd084c9e1218ad2cc7a0ed8','Strict parser source differs');parser=module(parserpath,'independent_frozen_strict_parser')._answer_and_format
 scopes={};candidates={};outcomes={};units={};evidence={};partitions={}
 for cp,key in zip(CHECKPOINTS,('parent','child')):
  bundle=spec[key]
  for item in bundle.values():req(sha(item['path'])==item['sha256'],'Actual checkpoint evidence changed')
  scope=load(bundle['scope']['path']);receipt=load(bundle['eval_receipt']['path']);ex=load(bundle['exit_receipt']['path']);reviewpath=Path(bundle['scope']['path']).parent/'INDEPENDENT_REVIEW_PRIVATE.json'
  req(ex['exit_code']==0 and ex['raw_log_sha256']==bundle['closed_log']['sha256'] and ex['scope_file_sha256']==bundle['scope']['sha256'] and ex['review_sha256']==sha(reviewpath),'Actual native exit/log/review binding differs');review=load(reviewpath);req(review['verdict'] in ('PASS','PASS_WITH_LIMITATIONS'),'Scope approval absent')
  selector=W/'scripts'/('run_adaptive_parent_eval_v1.py' if key=='parent' else 'run_adaptive_post_eval_v1.py');req(ex['source_sha256']==sha(selector) and scope['adapter_selector_sha256']==sha(W/'scripts/run_adaptive_parent_eval_v1.py'),'Actual selector source differs')
  req(scope['entry_sha256']==sha(V/'scripts/run_paired_discrimination_pilot_v2.py')=='d0202667c4a580c4ea34cdd22a31a2238fd2de8ad81f26277600f351b5dd152c','Same reviewed native evaluator differs');req(scope['evaluation_contract_sha256']==spec['evaluation_contract']['sha256'] and scope['greedy_max_tokens']==256 and scope['temperature']==0,'Same numerical readout differs')
  req(receipt['native_release_verified'] is True and receipt['weight_updates']==0 and receipt['run_identity']==scope['run_identity'] and receipt['adapter_sha256']==scope['adapter']['sha256'] and receipt['scope_sha256']==hashlib.sha256(json.dumps(scope,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'Native readonly completion identity differs')
  req(sha(scope['adapter']['path'])==scope['adapter']['sha256'] and receipt['candidate_sha256']==bundle['candidate_output']['sha256'] and receipt['greedy_sha256']==bundle['greedy_output']['sha256'],'Evaluated adapter/output differs')
  cs=m.lines(bundle['candidate_output']['path']);gs=m.lines(bundle['greedy_output']['path']);req(len(cs)==25 and len(gs)==40,'Incomplete saved coverage')
  for rows,expected in ((cs,contract['candidate_records_PRIVATE']),(gs,contract['greedy_records_PRIVATE'])):
   req(len({r['record_id'] for r in rows})==len(rows),'Duplicate output')
   for row,x in zip(rows,expected):req(row['run_identity']==scope['run_identity'] and row['weight_updates']==0 and all(row.get(k)==v for k,v in x.items()),'Actual target/prompt/image/order differs')
  feature=scope['feature_evidence'];req(sha(feature['plan_path'])==feature['plan_sha256'] and sha(feature['receipt_path'])==feature['receipt_sha256'],'Native prompt evidence changed');featureplan=load(feature['plan_path']);fc=load(feature['receipt_path']);inputs={}
  for i,x in enumerate(featureplan['rows_PRIVATE']):
   p=Path(feature['plan_path']).parent/f'record_{i:04d}.json';req(sha(p)==fc['record_sha256'][i],'Native captured record changed');inputs[x['id']]=load(p)
  null=next(x for x in m.lines(contract['B28_prefix_path']) if x['condition']=='no_image' and x['category']=='mri_sequence');req(sha(contract['B28_prefix_path'])==contract['B28_prefix_sha256'],'Null prompt evidence changed')
  for row in cs:
   expected=null if row['condition']=='no_image' else inputs[row['source_record_id']];req(row['prompt_ids']==expected['prompt_ids'] and row['rendered_prompt_sha256']==expected.get('prompt_sha256',expected.get('rendered_prompt_sha256')),'Captured native prompt differs')
   if row['condition']=='real_image':req(row['image_grid_thw']==expected['image_grid_thw'] and row['image_token_count']==expected['image_token_count'],'Native visual grid differs')
  candidates[cp]={x['record_id']:{'condition':x['condition'],'margins':m.raw_margins(x)} for x in cs};partitions[cp]={x['record_id']:{l:v['partition'] for l,v in x['candidate_scores'].items()} for x in cs};outcomes[cp]={};units[cp]={}
  for kind in KINDS:
   v,u=m.scores(seq,[m.prediction(candidates[cp][r['id']]['margins'][kind]) for r in seq]);outcomes[cp][kind]=v['mri_sequence'];units[cp][kind]=u['mri_sequence']
  greedy={r['record_id']:r for r in gs};pred=[]
  for row in dev:
   ans,fmt=parser(greedy[row['id']]['response']);labs=[x.lower() for x in m.LABELS[row['ability']]];pred.append(labs.index(ans.lower()) if fmt==1.0 and ans.lower() in labs else -1)
  outcomes[cp]['greedy'],units[cp]['greedy']=m.scores(dev,pred);scopes[cp]=scope;evidence[cp]={k:x['sha256'] for k,x in bundle.items()};evidence[cp]['review_sha256']=sha(reviewpath);evidence[cp]['selector_source_sha256']=sha(selector)
 req(scopes[CHECKPOINTS[0]]['native_sources']==scopes[CHECKPOINTS[1]]['native_sources'] and scopes[CHECKPOINTS[0]]['feature_evidence']==scopes[CHECKPOINTS[1]]['feature_evidence'] and partitions[CHECKPOINTS[0]]==partitions[CHECKPOINTS[1]],'Paired native interface or candidate tokenization differs')
 pairs=collections.defaultdict(dict)
 for row in seq:pairs[(row['extra_info']['patient_key'],row['extra_info']['level'])][row['reward_model']['ground_truth']]=row['id']
 req(len(pairs)==12 and len({k[0] for k in pairs})==4 and all(set(x)==set(m.LABELS['mri_sequence']) for x in pairs.values()),'Wrong12pairs/fourpeople')
 controls={}
 for cp in CHECKPOINTS:
  ns=[x for x in candidates[cp].values() if x['condition']=='no_image'];req(len(ns)==1,'Exactly one null required');controls[cp]={}
  for kind in KINDS:
   nm=ns[0]['margins'][kind];by=collections.defaultdict(list)
   for (person,level),pair in pairs.items():by[person].append(candidates[cp][pair['contrast-enhanced T1']]['margins'][kind]-candidates[cp][pair['FLAIR']]['margins'][kind])
   people=[m.avg(x) for x in by.values()];real=[candidates[cp][row['id']]['margins'][kind] for row in seq];controls[cp][kind]={'null_margin':nm,'real_mean_margin':m.avg(real),'mean_real_minus_null_margin':m.avg([x-nm for x in real]),'patient_mean_paired_separation':m.avg(people),'patient_pair_separation_range':[min(people),max(people)]}
 bootstrap={kind:m.bootstrap(units[CHECKPOINTS[0]][kind],units[CHECKPOINTS[1]][kind]) for kind in KINDS};greedyboot={t:m.bootstrap(units[CHECKPOINTS[0]]['greedy'][t],units[CHECKPOINTS[1]]['greedy'][t]) for t in m.LABELS}
 # Persist independently computed numbers BEFORE opening the analyst aggregate.
 preliminary={'outcomes':outcomes,'visual_controls':controls,'candidate_bootstrap':bootstrap,'greedy_bootstrap':greedyboot,'evidence':evidence}
 private=W/'local_data/adaptive_accuracy_gate_v1'/(a.output.stem+'_BEFORE_ANALYST_COMPARISON_PRIVATE.json');req(not private.exists(),'Preserve prior independent calculation');private.write_text(json.dumps(preliminary,sort_keys=True,indent=2,allow_nan=False)+'\n')
 print(json.dumps({'independent_primary':{cp:outcomes[cp]['mean'] for cp in CHECKPOINTS},'independent_strict_greedy':{cp:outcomes[cp]['greedy'] for cp in CHECKPOINTS},'independent_visual_mean':{cp:controls[cp]['mean'] for cp in CHECKPOINTS}},sort_keys=True),flush=True)
 resultpath=W/'results/ADAPTIVE_ACCURACY_GATE_PARENT_CHILD_ACTUAL_OUTCOMES_V1.json';req(sha(resultpath)==RESULT_SHA and sha(W/'scripts/analyze_adaptive_accuracy_outcomes_v1.py')==ANALYZER_SHA,'Supplied analyst/source changed');result=load(resultpath);req(result['analyzer_sha256']==ANALYZER_SHA,'Result source differs')
 for kind in KINDS:
  saved=result['primary_endpoint']['result'] if kind=='mean' else result['sequence_candidate_secondary'][kind]
  for cp in CHECKPOINTS:
   x=outcomes[cp][kind];y=saved['checkpoints'][cp]
   for field in ('question_accuracy','balanced_accuracy'):m.near(x[field],y[field])
   m.near(x['unit_macro_accuracy'],y['patient_macro_accuracy']);m.same(x['class_recall'],y['recall']);m.same(x['confusion'],y['confusion']);req(y['people']==x['independent_units']==4,'People inflated')
  m.near(outcomes[CHECKPOINTS[1]][kind]['balanced_accuracy']-outcomes[CHECKPOINTS[0]][kind]['balanced_accuracy'],saved['adaptive_minus_parent_balanced_accuracy']);m.same(bootstrap[kind],saved['paired_people_uncertainty_descriptive'])
 for cp in CHECKPOINTS:
  saved=result['full40_greedy_separate_readout']['checkpoints'][cp];vals=[]
  for task,x in outcomes[cp]['greedy'].items():
   y=saved['tasks'][task]
   for field in ('question_accuracy','unit_macro_accuracy','balanced_accuracy'):m.near(x[field],y[field])
   m.same(x['class_recall'],y['class_recall']);m.same(x['confusion'],y['confusion']);req(x['independent_units']==y['independent_units']==4,'Greedy clusters inflated');vals.append(x['unit_macro_accuracy'])
  m.near(m.avg(vals),saved['equal_task_unit_macro']);m.near(min(vals),saved['worst_task_accuracy'])
  for kind in KINDS:
   y=result['visual_controls']['checkpoints'][cp][kind]
   for key,value in controls[cp][kind].items():m.same(value,y[key])
   req(y['null_unique_queries']==1 and y['sequence_pairs']==12 and y['independent_people']==4,'Control units inflated');m.near(sum(m.prediction(controls[cp][kind]['null_margin'])==m.LABELS['mri_sequence'].index(r['reward_model']['ground_truth']) for r in seq)/24,y['implied_constant_class_accuracy_on24_truths'])
 m.same(greedyboot,result['full40_greedy_separate_readout']['paired_units_uncertainty_descriptive'])
 for kind in KINDS:
  y=result['visual_controls']['adaptive_minus_parent'][kind]
  for output,key in [('delta_null_margin','null_margin'),('delta_patient_mean_paired_separation','patient_mean_paired_separation'),('delta_mean_real_minus_null_margin','mean_real_minus_null_margin')]:m.near(controls[CHECKPOINTS[1]][kind][key]-controls[CHECKPOINTS[0]][kind][key],y[output])
 req(result['actual_continuation']=={'attempted_groups':7,'new_optimizer_updates':4,'parent_optimizer_step':52,'terminal_optimizer_step':56,'parent_schedule_cursor':114,'terminal_schedule_cursor':121,'training_proof_sha256':'39d62aa3d9949b3ec39a52fb9b64fe82eb1233fdafdd63158fb7e0d558a87967'},'Actual continuation differs');req(not any(k.split('.')[0] in {'mlx','mlx_vlm','torch','transformers'} for k in sys.modules),'Native model import occurred')
 report={'schema':'ADAPTIVE_PARENT_CHILD_ACTUAL_SAVED_NUMERICAL_INDEPENDENT_V1','verdict':'PASS_WITH_LIMITATIONS','reviewer_kind':'isolated_codex','reviewer_id':a.reviewer_id,'verifier_author':'/root/full_arm_execution_contract','verifier_sha256':sha(__file__),'pure_independent_arithmetic_source_sha256':PURE_SHA,'actual_spec_sha256':SPEC_SHA,'actual_analyst_result_sha256':RESULT_SHA,'actual_analyst_source_sha256':ANALYZER_SHA,'independent_precomparison_sha256':sha(private),'evidence':evidence,'candidate_outcomes':{cp:{k:outcomes[cp][k] for k in KINDS} for cp in CHECKPOINTS},'strict_greedy_outcomes_separate_readout':{cp:outcomes[cp]['greedy'] for cp in CHECKPOINTS},'visual_controls':controls,'candidate_paired_people_bootstrap_descriptive':bootstrap,'greedy_paired_clusters_bootstrap_descriptive':greedyboot,'actual_new_optimizer_updates':4,'actual_attempted_groups':7,'native_models_or_refits_by_reviewer':0,'unique_null_queries_per_checkpoint':1,'paired_images':12,'independent_sequence_people':4,'independent_Dallas_cycles':4,'all_compared_candidate_greedy_visual_outcome_metrics_match_independent_recalculation':True,'same_native_source_inventory_sha256':hashlib.sha256(json.dumps(scopes[CHECKPOINTS[0]]['native_sources'],sort_keys=True,separators=(',',':')).encode()).hexdigest(),'shared_native_evaluator_sha256':scopes[CHECKPOINTS[0]]['entry_sha256'],'overall_improvement_GO':False,'limitations':['Numerical PASS applies to saved evidence and arithmetic, not MRI understanding or method efficacy.','Candidate MEAN primary, RAW/FIRST secondary and strict full-format greedy remain distinct readouts.','Four repeatedly examined sequence people and four Dallas cycles; bootstrap intervals descriptive, no sealed final data or independent-patient claim.','One null query per checkpoint, twelve pairs clustered within four people; no inflation to24 null queries or12 independent people.','No matched strict-gate continuation; before/after cannot identify causal benefit of relaxed eligibility versus four additional updates.','Saved chosen-token scores were recalculated; native forward/backward and mean-feature extraction not independently rerun.','Dallas strict-greedy accuracy regresses; sequence gains alone do not satisfy joint-task improvement GO.','Intensity/acquisition/whole-head shortcuts remain possible; paired separation is not anatomical understanding.']}
 req(not a.output.exists(),'Preserve existing public proof');a.output.write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n');print(json.dumps({'verdict':report['verdict'],'proof_sha256':sha(a.output),'overall_improvement_GO':False},sort_keys=True))
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reviewer-id',required=True);p.add_argument('--output',type=Path,required=True);verify(p.parse_args())
if __name__=='__main__':main()
