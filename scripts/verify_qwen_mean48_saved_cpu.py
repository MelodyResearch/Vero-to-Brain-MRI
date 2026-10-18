"""Independent saved-vector/head arithmetic for actual mean48; no native/refit.
Raw merged vision tokens are not saved, so native pooling/504-tensor comparisons
remain source/execution-receipt bound rather than independently re-executed.
"""
import argparse,collections,hashlib,json,math,random,sys
from pathlib import Path
import numpy as np
W=Path(__file__).resolve().parents[1]
LABELS=('FLAIR','contrast-enhanced T1')
def req(x,msg):
 if not x:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def canon(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def near(a,b):req(math.isclose(float(a),float(b),abs_tol=1e-11,rel_tol=1e-11),'Saved metric differs')
def metrics(rows,pred):
 req(len(rows)==len(pred)==24 and all(type(v) is int and v in (-1,0,1) for v in pred),'Exact integer predictions required')
 conf={label:{p:0 for p in (*LABELS,'INVALID')} for label in LABELS};units=collections.defaultdict(list)
 for r,p in zip(rows,pred):
  y=r['reward_model']['ground_truth'];guess=LABELS[p] if p>=0 else 'INVALID';conf[y][guess]+=1;units[r['extra_info']['patient_key']].append(int(y==guess))
 req(len(units)==4 and all(len(v)==6 for v in units.values()),'Expected4people/6images each')
 unit={k:math.fsum(v)/len(v) for k,v in units.items()}
 return {'rows':24,'people':4,'patient_macro_accuracy':math.fsum(unit.values())/4,'question_accuracy':sum(conf[l][l] for l in LABELS)/24,'recall':{l:conf[l][l]/sum(conf[l].values()) for l in LABELS},'confusion':conf,'invalid':sum(conf[l]['INVALID'] for l in LABELS),'exploratory_internal_dev_only':True},unit

def verify(a):
 req(a.reviewer_id.strip(),'Truthful actual reviewer required');D=a.directory.resolve();scope=load(D/'scope_READY_REQUIRES_REVIEW_PRIVATE.json');plan=load(D/'plan_PRIVATE.json');complete=load(D/'native_complete_SAFE.json');result=load(D/'analysis_SAFE.json');fits=load(D/'linear_fits_predictions_PRIVATE.json')
 req(sha(D/'scope_READY_REQUIRES_REVIEW_PRIVATE.json')=='96c3720b7402993b7d8ced5b56cf901e723b0ee5863c43a9e006e3e03dce0fe8','Actual scoped mean48 release changed')
 req(sha(D/'INDEPENDENT_REVIEW_PRIVATE.json')=='82148169b59665b58e66e9fd7b361e4b6a8462b736d336e112abc0cef9bc4eb7','Independent exact source review changed')
 req(sha(a.native_exit)==a.expected_native_exit_sha256 and sha(a.closed_log)==a.expected_closed_log_sha256,'Actual exit/log witness digest differs')
 witness=load(a.native_exit);req(witness['exit_code']==0 and witness['raw_log_sha256']==a.expected_closed_log_sha256 and witness['scope_file_sha256']==sha(D/'scope_READY_REQUIRES_REVIEW_PRIVATE.json'),'Actual native exit0/closed log/scoped launch differs')
 req(sha(D/'analysis_SAFE.json')==a.expected_analysis_sha256,'Externally supplied actual analysis digest differs')
 req(sha(W/'scripts/run_v2_qwen_mean48.py')==scope['source_sha256']['V2_feature_entry']=='2f6418829aac434f5c91c5559962c4a1bd574a9843eb35aac6e1459127ef8963','Actual native source differs')
 req(scope['plan_sha256']==sha(D/'plan_PRIVATE.json')=='c901ea1a691d87a1a3e3a39482dca14717b7a409724ccbae760e72589111c8fc' and scope['run_identity']==canon(plan),'Actual plan/run identity differs')
 rows=plan['rows_PRIVATE'];req(len(rows)==48 and plan['train_rows']==plan['dev_rows']==24 and plan['train_people']==plan['dev_people']==4,'Frozen48plan differs')
 packpath=W/'local_data/v2_cpu_representation_v1/frozen_inputs_PRIVATE.json';req(sha(packpath)==plan['input_sha256']=='446808a0203ba1e6dd9f45e7f052a7667420b016ffff5a5b88f295d728fc79ab','Frozen original pack differs');pack=load(packpath)
 fulltrain=[r for r in pack['train_PRIVATE'] if r['ability']=='mri_sequence'];fulldev=[r for r in pack['dev_PRIVATE'] if r['ability']=='mri_sequence'];chosen=set(sorted({r['extra_info']['patient_key'] for r in fulltrain})[:4]);expected=sorted([r for r in fulltrain if r['extra_info']['patient_key'] in chosen],key=lambda r:r['id'])+sorted(fulldev,key=lambda r:r['id']);req(rows==expected,'Actual row selection differs')
 for split,group in [('train',rows[:24]),('internal_dev',rows[24:])]:
  people=collections.defaultdict(list)
  for r in group:
   e=r['extra_info'];req(e['internal_split']==split and r['ability']=='mri_sequence','Wrong split/task');people[e['patient_key']].append((e['level'],r['reward_model']['ground_truth']))
  req(len(people)==4 and all(len(v)==6 and set(v)=={(z,l) for z in (45,91,136) for l in LABELS} for v in people.values()),'Paired person/level coverage differs')
 req(not {r['extra_info']['patient_key'] for r in rows[:24]}&{r['extra_info']['patient_key'] for r in rows[24:]},'Train/dev person overlap')
 features=[];hashes=[]
 for i,r in enumerate(rows):
  path=D/f'record_{i:04d}.json';v=load(path);hashes.append(sha(path));req(v['index']==i and v['record_id']==r['id'] and v['run_identity']==scope['run_identity'] and v['image_sha256']==pack['images_PRIVATE'][r['id']]==sha(r['images'][0]),'Native feature image/order/identity differs')
  x=np.asarray(v['mean_feature'],dtype=np.float64);grid=np.asarray(v['image_grid_thw']);req(x.shape==(4096,) and np.isfinite(x).all() and v['feature_sha256']==canon(v['mean_feature']),'Nonfinite or changed4096vector')
  req(v['forward_count']==1 and v['weight_updates']==0 and v['prompt_tokens']==len(v['prompt_ids'])<=2048 and all(type(t) is int and t>=0 for t in v['prompt_ids']),'Native record budget/input evidence differs')
  req(grid.shape==(1,3) and grid[0,0]==1 and np.all(grid>0) and np.all(grid==grid.astype(int)) and grid[0,1]%2==grid[0,2]%2==0,'Native grid differs');n=int(np.prod(grid)//4);req(v['merged_shape']==[n,4096] and v['image_token_count']==n==v['prompt_ids'].count(151655),'Actual merged grid/token-count differs');req(len(v['prompt_sha256'])==64,'Missing native prompt hash');features.append(x)
 req(len(list(D.glob('record_*.json')))==48 and not list(D.glob('*.partial')),'Incomplete/extra feature records')
 req(complete['status']=='ACTUAL48_VISION_FORWARDS_READONLY_COMPLETE_AND_RELEASED' and complete['run_identity']==scope['run_identity'] and complete['plan_sha256']==scope['plan_sha256'] and complete['record_sha256']==hashes and complete['adapter_sha256']==scope['adapter']['sha256'],'Native completion chain differs')
 req(complete['adapter_tensors_compared_after']==504 and complete['optimizer_step']==complete['weight_updates']==0,'Source-executed readonly checks absent');release=complete['release'];req(release['native_release_verified'] is True and release['model_is_none'] is True and 0<=release['active_MLX_bytes']<release['active_limit_bytes']==128*1024**2,'Actual post-close release absent')
 req(result['status']=='ACTUAL_FEATURE_EXTRACTION_AND_CPU_HEAD_FIT_COMPLETE' and result['vision_forwards']==48 and result['head_training_images']==result['dev_images']==24,'Actual final analysis status differs');e=result['evidence_sha256'];req(e['plan']==sha(D/'plan_PRIVATE.json') and e['native_completion']==sha(D/'native_complete_SAFE.json') and e['native_record_files']==hashes and e['fitted_private_parameters']==sha(D/'linear_fits_predictions_PRIVATE.json'),'Saved analysis hash chain differs')
 req(fits['dev_record_ids_PRIVATE']==[r['id'] for r in rows[24:]],'Private prediction row ordering differs')
 old=W/'local_data/v2_cpu_representation_v1';req(sha(old/'features_PRIVATE.npz')=='8581ea599bd7600227556a0dcec171f1eada4febeb89a2962ac02514325b84e3' and sha(old/'fits_and_predictions_PRIVATE.json')=='d8297ce9407a88252170ae5c727a58c88c9c38a3a6c5ebc5772e690c0b461b3b','Prior authenticated feature/control evidence changed')
 matrices={'Qwen_merged_mean':np.asarray(features)};lookup={r['id']:i for i,r in enumerate(fulltrain)};devlookup={r['id']:i for i,r in enumerate(fulldev)}
 with np.load(old/'features_PRIVATE.npz',allow_pickle=False) as z:
  for name,key in [('pixel_thumbnail16','spatial'),('intensity_histogram16','histogram')]:matrices[name]=np.vstack((z['mri_sequence__'+key+'__train'][[lookup[r['id']] for r in rows[:24]]],z['mri_sequence__'+key+'__dev'][[devlookup[r['id']] for r in rows[24:]]]))
 y=np.array([LABELS.index(r['reward_model']['ground_truth']) for r in rows[:24]]);head_checks={}
 for name,M in matrices.items():
  proof=fits['fits_PRIVATE'][name];mu=M[:24].mean(axis=0);sd=M[:24].std(axis=0);sd=np.where(sd>1e-12,sd,1.);req(np.allclose(mu,proof['training_feature_mean'],atol=1e-11,rtol=1e-11) and np.allclose(sd,proof['training_feature_scale'],atol=1e-11,rtol=1e-11),'Train-only saved scaler differs')
  A=(M[:24]-mu)/sd;B=(M[24:]-mu)/sd;coef=np.asarray(proof['dual_coefficients']);intercept=float(proof['intercept']);req(coef.shape==(24,) and np.isfinite(coef).all() and proof['alpha']==1 and proof['gram_shape']==[24,24] and proof['primal_D_by_D_formed'] is False and proof['fitted_rows']==24 and proof['feature_dimensions']==M.shape[1] and proof['dev_used_for_fit_or_hyperparameters'] is False,'Fixed bounded dual-head shape/config differs');near(intercept,float((2*y-1).mean()))
  residual=(A@A.T+np.eye(24))@coef-((2*y-1)-intercept);maximum=float(np.max(np.abs(residual)));req(maximum<1e-7,'Saved fixed-ridge stationarity residual too large');decision=(B@A.T)@coef+intercept;req(np.isfinite(decision).all() and (decision>0).astype(int).tolist()==fits['predictions_PRIVATE'][name],'Saved-head independent decisions differ');head_checks[name]={'saved_predictions_reproduced':24,'feature_dimensions':int(M.shape[1]),'linear_system_max_abs_residual':maximum,'refitted':False}
 predictions=fits['predictions_PRIVATE'];req(predictions['null_training_prior']==[int(y.mean()>.5)]*24,'Balanced null prior differs');previous=load(old/'fits_and_predictions_PRIVATE.json');index={r:i for i,r in enumerate(previous['dev_rows_PRIVATE'])}
 for name in ('B28_saved_native','U7_saved_native'):req(predictions[name]==[previous['predictions_PRIVATE'][name][index[r['id']]] for r in rows[24:]],'Prior native prediction ID mapping differs')
 summaries={};unitmaps={}
 for name,pred in predictions.items():
  m,u=metrics(rows[24:],pred);recorded=result['metrics'][name];near(m['patient_macro_accuracy'],recorded['patient_macro_accuracy']);req({k:v for k,v in m.items() if k!='patient_macro_accuracy'}=={k:v for k,v in recorded.items() if k!='patient_macro_accuracy'},'Independent class/patient/confusion metrics differ');summaries[name]=m;unitmaps[name]=u
 bootstrap_checks=0
 for name,v in result['Qwen_head_paired_delta'].items():
  dif=[unitmaps['Qwen_merged_mean'][p]-unitmaps[name][p] for p in sorted(unitmaps[name])];rng=random.Random(7);draws=sorted(sum(rng.choices(dif,k=4))/4 for _ in range(2000));near(sum(dif)/4,v['mean_difference']);req(v['subjects']==4,'Wrong bootstrap unit count');near(draws[49],v['confidence_interval'][0]);near(draws[1949],v['confidence_interval'][1]);bootstrap_checks+=1
 req(not any(k.split('.')[0] in {'mlx','mlx_vlm','torch','transformers'} for k in sys.modules),'Forbidden native import')
 report={'schema':'QWEN_MEAN48_ACTUAL_SAVED_NUMERICS_INDEPENDENT_V1','verdict':'PASS_WITH_LIMITATIONS','reviewer_kind':'isolated_codex','reviewer_id':a.reviewer_id,'verifier_author':'/root/full_arm_execution_contract','verifier_source_sha256':sha(__file__),'actual_native_exit_sha256':a.expected_native_exit_sha256,'actual_closed_log_sha256':a.expected_closed_log_sha256,'scope_sha256':sha(D/'scope_READY_REQUIRES_REVIEW_PRIVATE.json'),'native_completion_sha256':sha(D/'native_complete_SAFE.json'),'analysis_sha256':a.expected_analysis_sha256,'saved_fit_sha256':sha(D/'linear_fits_predictions_PRIVATE.json'),'plan_sha256':scope['plan_sha256'],'actual_records_verified':48,'native_source_recorded_vision_forwards':48,'head_stationarity_and_decisions':head_checks,'independent_metrics':summaries,'patient_bootstrap_comparisons_verified':bootstrap_checks,'train_people':4,'dev_people':4,'train_images':24,'dev_images':24,'model_updates':0,'refits_or_model_loads_by_reviewer':0,'release_receipt':release,'readonly_integrity_evidence':'Approved source executed504 adapter comparisons/step0 beforeclose; actual completed native exit and immutable receipt verified. No after-model tensor snapshot persisted for independent tensor recompare.','limitations':['Raw merged vision tokens are not saved; native feature extraction/pooling is source/execution-receipt bound and was not independently re-executed.','Saved vectors, coefficients and matrix residuals checked without solving/refitting a head or importing native models.','Repeatedly examined internal-dev4people only; not independent scanner/seed/final-patient generalization.','Mean pooling discards positions/deepstack; either success or failure does not establish anatomical reasoning or complete encoder adequacy.','Acquisition/intensity shortcuts remain; differing supervision/model/budget prevents a causal VLM-gain claim.']}
 req(not a.output.exists(),'Preserve earlier proof');a.output.write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n');print(json.dumps({'verdict':report['verdict'],'proof_sha256':sha(a.output),'head_count':3,'actual_records':48,'metrics':summaries},sort_keys=True))
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('directory','native-exit','closed-log','output'):p.add_argument('--'+name,type=Path,required=True)
 for name in ('expected-native-exit-sha256','expected-closed-log-sha256','expected-analysis-sha256','reviewer-id'):p.add_argument('--'+name,required=True)
 verify(p.parse_args())
if __name__=='__main__':main()
