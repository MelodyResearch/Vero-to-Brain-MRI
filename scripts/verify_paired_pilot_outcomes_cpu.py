"""Independent arithmetic from actual B28/post2 saved token scores and responses.
No model/forward/refit; primary candidate and separate greedy readouts never mix.
"""
import argparse,collections,hashlib,json,math,random,sys
from pathlib import Path
W=Path(__file__).resolve().parents[1]
LABELS={'mri_sequence':('FLAIR','contrast-enhanced T1'),'mri_visit_correspondence':('different','same')}
KINDS=('mean','raw','first_label')
def req(x,msg):
 if not x:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def near(a,b):req(math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-10),'Independent scalar differs')
def same(a,b):
 if isinstance(a,dict):
  req(isinstance(b,dict) and set(a)==set(b),'Independent dictionary scope differs')
  for k,v in a.items():same(v,b[k])
 elif isinstance(a,list):
  req(isinstance(b,list) and len(a)==len(b),'Independent vector length differs')
  for v,w in zip(a,b):same(v,w)
 elif type(a) is float:near(a,b)
 else:req(a==b,'Independent categorical/integer value differs')
def avg(xs):return math.fsum(xs)/len(xs)
def prediction(x):return 1 if x>0 else 0 if x<0 else -1

def raw_margins(row):
 values={}
 for label in LABELS['mri_sequence']:
  c=row['candidate_scores'][label];p=c['partition'];logps=c['target_token_logps'];n=len(p['prefix_ids']);a=len(p['label_ids']);req(a>0 and len(logps)==len(p['target_ids']) and all(math.isfinite(x) and x<=1e-6 for x in logps),'Invalid saved chosen-token evidence')
  req(p['target_ids']==p['prefix_ids']+p['label_ids']+p['suffix_ids'],'Saved token partition differs');raw=math.fsum(logps[n:n+a]);near(raw,c['label_logp_sum']);near(raw/a,c['label_logp_mean']);values[label]=(raw/a,raw)
 f=row['prefix_oracle_first_logps'];req(len(f)==2 and all(math.isfinite(x) for x in f),'Bad independent first-label vector')
 margins={'mean':values['contrast-enhanced T1'][0]-values['FLAIR'][0],'raw':values['contrast-enhanced T1'][1]-values['FLAIR'][1],'first_label':f[1]-f[0]}
 for kind in ('mean','raw'):near(margins[kind],row['comparisons'][kind+'_t1_minus_flair'])
 return margins

def scores(rows,pred):
 req(len(rows)==len(pred) and all(type(v) is int and v in (-1,0,1) for v in pred),'Wrong predicted coverage/classes');groups=collections.defaultdict(lambda:collections.defaultdict(list));conf={t:{y:{p:0 for p in (*labs,'INVALID')} for y in labs} for t,labs in LABELS.items()}
 for r,p in zip(rows,pred):
  t=r['ability'];e=r['extra_info'];truth=r['reward_model']['ground_truth'];got=LABELS[t][p] if p>=0 else 'INVALID';unit=e['patient_key'] if t=='mri_sequence' else e['statistical_cluster_key'];groups[t][unit].append(int(got==truth));conf[t][truth][got]+=1
 result={};units={}
 for t,gs in groups.items():
  unit={key:avg(v) for key,v in gs.items()};units[t]=unit;count=sum(sum(v.values()) for v in conf[t].values());recall={label:conf[t][label][label]/sum(conf[t][label].values()) for label in LABELS[t]};result[t]={'questions':count,'correct':sum(conf[t][l][l] for l in LABELS[t]),'question_accuracy':sum(conf[t][l][l] for l in LABELS[t])/count,'unit_macro_accuracy':avg(list(unit.values())),'class_recall':recall,'balanced_accuracy':avg(list(recall.values())),'confusion':conf[t],'independent_units':len(unit)}
 return result,units

def bootstrap(a,b):
 req(set(a)==set(b),'Different bootstrap people/cycles');ds=[b[k]-a[k] for k in sorted(a)];rng=random.Random(7);draws=sorted(avg(rng.choices(ds,k=len(ds))) for _ in range(2000));return {'subjects':len(ds),'mean_difference':avg(ds),'confidence_interval':[draws[49],draws[1949]]}

def verify(a):
 req(a.reviewer_id.strip(),'Actual reviewer required');req(sha(a.analysis)==a.expected_analysis_sha256,'Actual reviewed analyzer result differs');result=load(a.analysis);req(result['analyzer_sha256']==sha(W/'scripts/analyze_paired_discrimination_pilot_v2.py')=='7c6defde436210ac00d502533c3200ffb02e8843eb3abe2a5c9f357489610b81','Reviewed analyzer source differs');D=W/'local_data/paired_discrimination_pilot_v2/entry_v2';contract=load(D/'evaluation_contract_FROZEN_PRIVATE.json');req(sha(D/'evaluation_contract_FROZEN_PRIVATE.json')==result['evaluation_contract_sha256']=='8c52a40f572393f4a01c6fbd2eb84d044bc14ecd151ae00ef56d9e7600a2539b','Frozen evaluation contract differs')
 packpath=W/'local_data/v2_cpu_representation_v1/frozen_inputs_PRIVATE.json';req(sha(packpath)=='446808a0203ba1e6dd9f45e7f052a7667420b016ffff5a5b88f295d728fc79ab','Frozen dev rows differ');pack=load(packpath);dev=pack['dev_PRIVATE'];seq=[r for r in dev if r['ability']=='mri_sequence'];req(len(dev)==40 and len(seq)==24,'Full frozen readout coverage required')
 R=Path(pack['source_paths_PRIVATE']['mixed_train']).parents[2];req(sha(R/'src/mri_rl/vero_mri_reward.py')=='411ab099edcfb45305630c675f7ca4870d6501e69dd084c9e1218ad2cc7a0ed8','Existing strict parser source changed');sys.path.insert(0,str(R/'src'));from mri_rl.vero_mri_reward import _answer_and_format
 candidate={};greedy={};proofs={};outcome={};patient={}
 for cp,scope_name,expected_scope in [('B28','B28_baseline_READY_FOR_REVIEW_PRIVATE.json','f0b74a9474713b7588f5170053ecc5133497a691916d4a6a3c89c205e24bad11'),('post2','post2_evaluation_AFTER_TWO_UPDATES_READY_FOR_REVIEW_PRIVATE.json','ce15d73d1252b3c22eba0be914a22ffb384fe480337e79b957167a82106b8744')]:
  scopepath=D/scope_name;req(sha(scopepath)==expected_scope==result['input_evidence'][cp]['scope'],'Actual scope digest differs');scope=load(scopepath);out=Path(scope['output_directory']);receipt=load(out/'eval_complete_SAFE.json');exitpath=out/'actual_exit_PRIVATE.json';log=out/'native_stdout_PRIVATE.log';witness=load(exitpath);ev=result['input_evidence'][cp]
  req(sha(exitpath)==ev['exit_receipt'] and sha(log)==ev['closed_log']==witness['raw_log_sha256'] and witness['exit_code']==0 and witness['scope_file_sha256']==expected_scope,'Actual completed native exit/log not authenticated');req(sha(out/'eval_complete_SAFE.json')==ev['eval_receipt'] and receipt['native_release_verified'] is True and receipt['weight_updates']==0 and receipt['run_identity']==scope['run_identity'] and receipt['adapter_sha256']==scope['adapter']['sha256'],'Readonly completion identity/release differs')
  gp=out/'greedy_PRIVATE.jsonl';cpth=out/'candidates_PRIVATE.jsonl';req(sha(gp)==receipt['greedy_sha256'] and sha(cpth)==receipt['candidate_sha256'],'Saved output hashes differ');gs=lines(gp);cs=lines(cpth);req(len(gs)==40 and len(cs)==25,'Incomplete actual output coverage')
  for rows,expected in [(gs,contract['greedy_records_PRIVATE']),(cs,contract['candidate_records_PRIVATE'])]:
   req(len({r['record_id'] for r in rows})==len(rows),'Duplicate actual output')
   for r,x in zip(rows,expected):req(r['run_identity']==scope['run_identity'] and r['weight_updates']==0 and all(r.get(k)==v for k,v in x.items()),'Actual row/prompt/image/target/order differs')
  candidate[cp]={r['record_id']:{'condition':r['condition'],'margins':raw_margins(r)} for r in cs};greedy[cp]={r['record_id']:r for r in gs};outcome[cp]={};patient[cp]={}
  for kind in KINDS:
   pred=[prediction(candidate[cp][r['id']]['margins'][kind]) for r in seq];m,u=scores(seq,pred);outcome[cp][kind]=m['mri_sequence'];patient[cp][kind]=u['mri_sequence']
  p=[]
  for r in dev:
   answer,fmt=_answer_and_format(greedy[cp][r['id']]['response']);labels=[x.lower() for x in LABELS[r['ability']]];p.append(labels.index(answer.lower()) if fmt>0 and answer.lower() in labels else -1)
  gm,gu=scores(dev,p);outcome[cp]['greedy']=gm;patient[cp]['greedy']=gu;proofs[cp]={'actual_exit_sha256':sha(exitpath),'closed_log_sha256':sha(log),'candidate_sha256':sha(cpth),'greedy_sha256':sha(gp),'adapter_sha256':scope['adapter']['sha256']}
 for kind in KINDS:
  recorded=result['primary_endpoint']['result'] if kind=='mean' else result['sequence_candidate_secondary'][kind]
  for cp in ('B28','post2'):
   m=outcome[cp][kind];r=recorded['checkpoints'][cp];near(m['question_accuracy'],r['question_accuracy']);near(m['unit_macro_accuracy'],r['patient_macro_accuracy']);same(m['class_recall'],r['recall']);same(m['confusion'],r['confusion']);near(m['balanced_accuracy'],r['balanced_accuracy']);req(m['independent_units']==r['people']==4,'Wrong sequence unit count')
  near(outcome['post2'][kind]['balanced_accuracy']-outcome['B28'][kind]['balanced_accuracy'],recorded['post2_minus_B28_balanced_accuracy']);same(bootstrap(patient['B28'][kind],patient['post2'][kind]),recorded['paired_people_uncertainty_descriptive'])
 for cp in ('B28','post2'):
  saved=result['full40_greedy_separate_readout']['checkpoints'][cp];acc=[]
  for t,m in outcome[cp]['greedy'].items():
   r=saved['tasks'][t]
   for field in ('question_accuracy','unit_macro_accuracy','balanced_accuracy'):near(m[field],r[field])
   same(m['class_recall'],r['class_recall']);same(m['confusion'],r['confusion']);req(m['independent_units']==r['independent_units']==4,'Wrong greedy statistical units');acc.append(m['unit_macro_accuracy'])
  near(avg(acc),saved['equal_task_unit_macro']);near(min(acc),saved['worst_task_accuracy'])
 for t in LABELS:same(bootstrap(patient['B28']['greedy'][t],patient['post2']['greedy'][t]),result['full40_greedy_separate_readout']['paired_units_uncertainty_descriptive'][t])
 pairs=collections.defaultdict(dict)
 for r in seq:pairs[(r['extra_info']['patient_key'],r['extra_info']['level'])][r['reward_model']['ground_truth']]=r['id']
 req(len(pairs)==12 and len({k[0] for k in pairs})==4 and all(set(v)==set(LABELS['mri_sequence']) for v in pairs.values()),'Different12pair/4person control');controls={}
 for cp in ('B28','post2'):
  null=[v for v in candidate[cp].values() if v['condition']=='no_image'];req(len(null)==1,'Exactly ONE unique null query required');controls[cp]={}
  for kind in KINDS:
   nm=null[0]['margins'][kind];pred=prediction(nm);units=collections.defaultdict(list)
   for (person,level),p in pairs.items():units[person].append(candidate[cp][p['contrast-enhanced T1']]['margins'][kind]-candidate[cp][p['FLAIR']]['margins'][kind])
   perperson={p:avg(v) for p,v in units.items()};real=[candidate[cp][r['id']]['margins'][kind] for r in seq];v={'null_margin':nm,'real_mean_margin':avg(real),'mean_real_minus_null_margin':avg([x-nm for x in real]),'patient_mean_paired_separation':avg(list(perperson.values()))};controls[cp][kind]=v;r=result['visual_controls']['checkpoints'][cp][kind]
   for k,x in v.items():near(x,r[k])
   same([min(perperson.values()),max(perperson.values())],r['patient_pair_separation_range']);req(r['null_unique_queries']==1 and r['sequence_pairs']==12 and r['independent_people']==4,'Inflated null/pair denominator');near(sum(pred==LABELS['mri_sequence'].index(x['reward_model']['ground_truth']) for x in seq)/24,r['implied_constant_class_accuracy_on24_truths'])
 for kind in KINDS:
  before=controls['B28'][kind];after=controls['post2'][kind];r=result['visual_controls']['U2_minus_B28'][kind]
  for output,key in [('delta_null_margin','null_margin'),('delta_patient_mean_paired_separation','patient_mean_paired_separation'),('delta_mean_real_minus_null_margin','mean_real_minus_null_margin')]:near(after[key]-before[key],r[output])
 req(result['optimizer_updates']==2 and result['null_unique_queries_per_checkpoint']==1 and result['sealed_final100_used'] is False,'Wrong experiment outcome scope');req(not any(k.split('.')[0] in {'mlx','mlx_vlm','torch','transformers'} for k in sys.modules),'Native model imported')
 report={'schema':'PAIRED_PILOT_ACTUAL_SAVED_OUTCOME_INDEPENDENT_V1','verdict':'PASS_WITH_LIMITATIONS','reviewer_kind':'isolated_codex','reviewer_id':a.reviewer_id,'verifier_author':'/root/full_arm_execution_contract','verifier_sha256':sha(__file__),'actual_analyzer_result_sha256':a.expected_analysis_sha256,'evidence':proofs,'primary_readout':'SAME24sequence mean-label candidate balanced accuracy B28 versus post2','candidate_outcomes':{cp:{k:v for k,v in x.items() if k!='greedy'} for cp,x in outcome.items()},'primary_balanced_accuracy_delta':outcome['post2']['mean']['balanced_accuracy']-outcome['B28']['mean']['balanced_accuracy'],'greedy_outcomes_separate_readout':{cp:x['greedy'] for cp,x in outcome.items()},'visual_controls':controls,'patient_bootstrap_comparisons_verified':5,'paired_images':12,'independent_sequence_people':4,'unique_null_queries_per_checkpoint':1,'actual_model_updates':2,'native_models_or_refits_by_reviewer':0,'limitations':['Saved chosen-token scores and predictions verified; native logits/backward not independently re-executed.','Primary candidate/greedy/feature-head readouts remain separate; no mixing to claim a weight-improvement gain.','One unique null query and12opposite-image pairs clusteredwithin4people; no24independent null measurements or12independent patients.','Repeated internal-dev4people/4Dallascycles only; no final-patient, clinical, scanner/seed generalization or anatomical grounding claim.','Any prior shift or changed paired separation is descriptive; acquisition/intensity/nonbrain shortcuts remain possible.']}
 req(not a.output.exists(),'Preserve previous outcome proof');a.output.write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n');print(json.dumps({'verdict':report['verdict'],'proof_sha256':sha(a.output),'primary_delta':report['primary_balanced_accuracy_delta'],'candidate':report['candidate_outcomes'],'greedy':report['greedy_outcomes_separate_readout']},sort_keys=True))
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--analysis',type=Path,required=True);p.add_argument('--expected-analysis-sha256',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--reviewer-id',required=True);verify(p.parse_args())
if __name__=='__main__':main()
