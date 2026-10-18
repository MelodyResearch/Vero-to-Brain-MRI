"""Independent saved-head arithmetic and grouped metrics; no refit or VLM import."""
from pathlib import Path
import argparse,collections,hashlib,json,math,random,re,sys
import numpy as np
LABELS={'mri_sequence':('FLAIR','contrast-enhanced T1'),'mri_visit_correspondence':('different','same')}
def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def near(a,b):require(math.isclose(float(a),float(b),rel_tol=1e-11,abs_tol=1e-11),'Independent arithmetic mismatch')
def scalarmean(x):return math.fsum(x)/len(x)
def units(rows,pred):
 d=collections.defaultdict(lambda:collections.defaultdict(lambda:collections.defaultdict(list)))
 for r,p in zip(rows,pred):
  t=r['ability'];e=r['extra_info'];u=e['patient_key'] if t=='mri_sequence' else e['statistical_cluster_key'];d[t][u][e['patient_key']].append(int(p==LABELS[t].index(r['reward_model']['ground_truth'])))
 return {t:{u:scalarmean([scalarmean(v) for v in ps.values()]) for u,ps in us.items()} for t,us in d.items()}
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--directory',type=Path,required=True);ap.add_argument('--result',type=Path,required=True);ap.add_argument('--expected-result-sha256',required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--reviewer-id',required=True);a=ap.parse_args();require(a.reviewer_id.strip(),'Actual reviewer required');require(sha(a.result)==a.expected_result_sha256,'Actual result differs');result=load(a.result);e=result['evidence_sha256'];f=a.directory/'frozen_inputs_PRIVATE.json';p=a.directory/'fits_and_predictions_PRIVATE.json';z=a.directory/'features_PRIVATE.npz'
 for path,key in ((f,'frozen_inputs'),(p,'fits_and_predictions_private'),(z,'features_private')):require(sha(path)==e[key],'Frozen saved evidence differs')
 frozen=load(f);saved=load(p);train=frozen['train_PRIVATE'];dev=frozen['dev_PRIVATE'];require(saved['dev_rows_PRIVATE']==[r['id'] for r in dev],'Saved dev ordering differs')
 require(frozen['configuration']==result['configuration'],'Frozen fit configuration differs')
 for name,path in frozen['source_paths_PRIVATE'].items():require(sha(path)==frozen['source_sha256'][name],'Frozen source changed')
 people={};cycles={};paircounts={};image_hashes={}
 for split,rows in (('train',train),('internal_dev',dev)):
  ps=set();cs=set();groups=collections.defaultdict(list);queries=collections.defaultdict(dict)
  for r in rows:
   t=r['ability'];x=r['extra_info'];y=r['reward_model']['ground_truth'];require(x['internal_split']==split and y in LABELS[t],'Reference/split differs');ps.add((r['data_source'],x['patient_key']))
   if t=='mri_sequence':
    m=re.fullmatch(re.escape(x['patient_key'])+r'_z(\d{3})_(flair|t1ce)_sequence',r['id']);require(m and int(m[1])==x['level'] and y=={'flair':'FLAIR','t1ce':'contrast-enhanced T1'}[m[2]],'Same-person/level label mismatch');groups[(x['patient_key'],x['level'])].append(y)
   else:
    ps.add((r['data_source'],x['candidate_patient_key']));cs.add(x['statistical_cluster_key']);require(y==('same' if x['patient_key']==x['candidate_patient_key'] else 'different'),'Correspondence GT identity mismatch');queries[(x['statistical_cluster_key'],x['patient_key'])][y]=x['candidate_patient_key']
  require(all(len(v)==2 and set(v)==set(LABELS['mri_sequence']) for v in groups.values()),'Unmatched sequence pairs');require(all(set(v)=={'same','different'} and v['same']==q and v['different']!=q for (c,q),v in queries.items()),'Unmatched Dallas query')
  people[split]=ps;cycles[split]=cs;paircounts[split]=len(groups);image_hashes[split]={frozen['images_PRIVATE'][r['id']] for r in rows}
 require(not people['train']&people['internal_dev'] and not cycles['train']&cycles['internal_dev'] and not image_hashes['train']&image_hashes['internal_dev'],'Patient/cycle/exact-image leakage')
 predictions=saved['predictions_PRIVATE'];require(all(len(v)==len(dev) and all(type(p) is int and p in (-1,0,1) for p in v) for v in predictions.values()),'Actual predictions are not exact integer classes')
 feature_checks=0
 with np.load(z,allow_pickle=False) as features:
  for task in LABELS:
   tr=[r for r in train if r['ability']==task];dv=[r for r in dev if r['ability']==task];idx=[i for i,r in enumerate(dev) if r['ability']==task]
   for method,key in (('spatial16_L2_logistic','spatial'),('histogram16_L2_logistic','histogram'),('source_site_level_L2_logistic','metadata')):
    fit=saved['fits_PRIVATE'][task+'|'+method];A=features[task+'__'+key+'__train'];B=features[task+'__'+key+'__dev'];mu=np.asarray(fit['training_feature_mean']);scale=np.asarray(fit['training_feature_scale']);coef=np.asarray(fit['coefficient'])[0];inter=float(fit['intercept'][0]);require(A.shape[0]==len(tr) and B.shape[0]==len(dv) and np.isfinite(A).all() and np.isfinite(B).all() and np.isfinite(coef).all() and math.isfinite(inter),'Saved feature/head shapes differ')
    computed_mean=A.mean(axis=0);computed_scale=np.sqrt(((A-computed_mean)**2).mean(axis=0));computed_scale[computed_scale==0]=1.;require(np.allclose(mu,computed_mean,rtol=1e-11,atol=1e-11) and np.allclose(scale,computed_scale,rtol=1e-11,atol=1e-11),'Training-only scaler differs')
    guess=(((B-mu)/scale)@coef+inter>0).astype(int);require(guess.tolist()==[predictions[method][i] for i in idx],'Independent saved-head predictions differ');feature_checks+=1
    if key=='spatial':require((((-mu/scale)@coef+inter>0)*np.ones(len(dv),dtype=int)).tolist()==[predictions['spatial_head_blank_dev'][i] for i in idx],'Blank control differs')
   y=[LABELS[task].index(r['reward_model']['ground_truth']) for r in tr];majority=int(scalarmean(y)>.5);require(all(predictions['null_training_majority'][i]==majority for i in idx),'Train-only majority differs')
 # Saved native generation classification uses the existing separately reviewed strict parser; no generation occurs.
 R=Path(frozen['source_paths_PRIVATE']['mixed_train']).parents[2];sys.path.insert(0,str(R/'src'));from mri_rl.vero_mri_reward import _answer_and_format
 for method,path,key,digest in (('B28_saved_native',R/'local_data/c_two_task_pool_v1/evaluation_v1/C40_B28_baseline_independent_scoring_PRIVATE.json','normalized_response_components_PRIVATE','32ce2cd88a9d316e94f75177eb8c6c36e33bcd351fe4028ed9e88aa3c26dda8f'),('U7_saved_native',R/'local_data/c_two_task_pool_v1/tier_c_v1/U_raw_seed7_final_C40_release_v1/final_C40_scoring_PRIVATE.json','normalized_new_outputs_PRIVATE','de73178d59d5894d7ddfe4494f05dd785aa35211905b9457e9a1757e1f8b059e')):
  require(sha(path)==digest,'Saved native outputs differ');values=load(path)[key];require(set(values)==set(saved['dev_rows_PRIVATE']),'Native ID alignment differs')
  for i,r in enumerate(dev):
   answer,fmt=_answer_and_format(values[r['id']]['response']);allowed=[x.lower() for x in LABELS[r['ability']]];pred=allowed.index(answer.lower()) if fmt>0 and answer.lower() in allowed else -1;require(pred==predictions[method][i],'Native prediction mapping differs')
 summaries={};unitmaps={}
 for method,pred in predictions.items():
  u=units(dev,pred);unitmaps[method]=u;summaries[method]={};acc=[]
  for task,labels in LABELS.items():
   indices=[i for i,r in enumerate(dev) if r['ability']==task];conf={l:{k:0 for k in (*labels,'INVALID')} for l in labels};correct=0
   for i in indices:
    truth=dev[i]['reward_model']['ground_truth'];got=labels[pred[i]] if pred[i]>=0 else 'INVALID';conf[truth][got]+=1;correct+=int(got==truth)
   recorded=result['metrics'][method]['tasks'][task];require(conf==recorded['confusion'],'Confusion differs');near(correct/len(indices),recorded['question_accuracy']);macro=scalarmean(list(u[task].values()));near(macro,recorded['unit_macro_accuracy']);require(len(u[task])==recorded['independent_units'],'Unit count differs');acc.append(macro)
   for label in labels:near(conf[label][label]/sum(conf[label].values()),recorded['class_recall'][label])
   summaries[method][task]={'correct_questions':correct,'questions':len(indices),'unit_macro_accuracy':macro,'independent_units':len(u[task])}
  near(scalarmean(acc),result['metrics'][method]['equal_task_unit_macro']);near(min(acc),result['metrics'][method]['worst_task_accuracy'])
 bootstraps=0
 for control,bytask in result['spatial_paired_difference_from_control'].items():
  for task,recorded in bytask.items():
   dif=[unitmaps['spatial16_L2_logistic'][task][u]-unitmaps[control][task][u] for u in sorted(unitmaps[control][task])];rng=random.Random(7);draws=sorted(sum(rng.choices(dif,k=len(dif)))/len(dif) for _ in range(2000));near(sum(dif)/len(dif),recorded['mean_difference']);require(recorded['subjects']==len(dif),'Bootstrap units differ');near(draws[49],recorded['confidence_interval'][0]);near(draws[1949],recorded['confidence_interval'][1]);bootstraps+=1
 report={'schema':'V2-CPU-representation-saved-arithmetic-independent-v1','reviewer_kind':'isolated_codex','reviewer_id':a.reviewer_id,'verifier_author':'/root/full_arm_execution_contract','verdict':'PASS_WITH_LIMITATIONS','source_sha256':sha(__file__),'actual_result_sha256':a.expected_result_sha256,'frozen_input_sha256':e['frozen_inputs'],'saved_head_predictions_reproduced':feature_checks,'all_actual_predictions_exact_integer_classes':True,'train_sequence_pairs':paircounts['train'],'dev_sequence_pairs':paircounts['internal_dev'],'within_source_patient_and_cycle_disjoint':True,'exact_image_hash_disjoint':True,'native_saved_ID_prediction_alignment_verified':True,'independent_confusion_and_unit_metrics':summaries,'unit_bootstrap_comparisons_verified':bootstraps,'models_refitted':0,'native_models_imported':False,'limitations':['Saved feature arrays and head parameters were used; feature pixel extraction was not rerun by this reviewer.','Current source uses train-only scaling/fitting and fixed C1 with no dev-label search; this is repeated internal development, not untouched preregistered evidence.','Sequence supervision expands28 to168 rows and models/budgets differ from VLM; no causal VLM accuracy gain or Qwen feature-encoder diagnosis follows.','Histogram and whole-head/acquisition shortcuts remain; no brain-specific grounding, clinical generalization or seed robustness is established.','Only4 sequence people and4 Dallas cycles bootstrap uncertainty; no scanner/seed/sample-selection uncertainty.']}
 require(not any(k.split('.')[0] in {'mlx','mlx_vlm','torch','transformers'} for k in sys.modules),'Native model import occurred');require(not a.output.exists(),'Preserve prior proof');a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n');print(json.dumps({'verdict':report['verdict'],'output_sha256':sha(a.output),'heads':feature_checks,'bootstrap_comparisons':bootstraps}))
if __name__=='__main__':main()
