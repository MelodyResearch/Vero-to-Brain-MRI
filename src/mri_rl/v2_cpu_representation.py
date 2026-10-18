"""Finite image-only CPU probes on explicitly paired train/internal-dev inputs.
Features preserve C1/C2 thumbnail16 and histogram16 definitions. No Qwen model.
"""
from collections import defaultdict
import numpy as np
from PIL import Image
from mri_rl.experiments import paired_patient_difference
LABELS={'mri_sequence':('FLAIR','contrast-enhanced T1'),'mri_visit_correspondence':('different','same')}
def require(ok,message):
 if not ok:raise ValueError(message)
def _validate_rows(rows,split):
 require(rows and len({r['id'] for r in rows})==len(rows),'Duplicate/empty rows')
 require({r['ability'] for r in rows}==set(LABELS),'Two task pack required')
 sequence=defaultdict(list);cycles=defaultdict(lambda:defaultdict(list));patients=set();patient_cycle={}
 for r in rows:
  t=r['ability'];e=r['extra_info'];y=r['reward_model']['ground_truth'];p=e['patient_key']
  require(e['internal_split']==split and e['patient_lineage']=='BIDS_SUBJECT_VERIFIED' and e['reward_type']==t and y in LABELS[t],'Split/lineage/target mismatch')
  require(r['data_source']=={'mri_sequence':'OpenNeuro_ds007045','mri_visit_correspondence':'OpenNeuro_ds004856'}[t],'Task source mismatch')
  patients.add((r['data_source'],p))
  if t=='mri_sequence':sequence[(p,e['level'])].append(y)
  else:
   c=e['candidate_patient_key'];cluster=e['statistical_cluster_key'];patients.add((r['data_source'],c))
   require(y==('same' if p==c else 'different') and e['query_source_sha256']!=e['candidate_source_sha256'],'Correspondence truth/source mismatch')
   for q in (p,c):require(q not in patient_cycle or patient_cycle[q]==cluster,'Patient in multiple cycles');patient_cycle[q]=cluster
   cycles[cluster][p].append((c,y))
 require(all(len(v)==2 and set(v)==set(LABELS['mri_sequence']) for v in sequence.values()),'Missing same-patient same-level modality pair')
 for queries in cycles.values():
  require(len(queries)==2,'Two people per cycle required')
  for refs in queries.values():require(len(refs)==2 and {c for c,y in refs}==set(queries) and {y for c,y in refs}=={'same','different'},'Missing matched same/different cycle')
 return patients,set(cycles)
def validate_pack(train,dev):
 a,ac=_validate_rows(train,'train');b,bc=_validate_rows(dev,'internal_dev');require(not a&b and not ac&bc,'Patient/cycle leakage')
def pixel_features(gray):
 gray=gray.convert('L');a=np.asarray(gray)
 require(a.ndim==2 and a.size<=1_000_000,'Bounded PNG frame required')
 thumbnail=np.asarray(gray.resize((16,16),Image.Resampling.BILINEAR),dtype=np.float64).ravel()/255
 h=np.histogram(a,bins=16,range=(0,256))[0];return thumbnail,h/h.sum()
def pair_features(left,right):
 a,h=pixel_features(left);b,j=pixel_features(right)
 # Symmetric to panel reversal: no source identities, target, order or labels.
 return np.abs(a-b),np.abs(h-j)
def fit_predict(train,y,dev):
 from sklearn.pipeline import make_pipeline
 from sklearn.preprocessing import StandardScaler
 from sklearn.linear_model import LogisticRegression
 train=np.asarray(train,dtype=np.float64);dev=np.asarray(dev,dtype=np.float64);y=np.asarray(y)
 require(train.ndim==dev.ndim==2 and train.shape[1]==dev.shape[1] and len(y)==len(train) and set(y)=={0,1} and np.isfinite(train).all() and np.isfinite(dev).all(),'Invalid finite two-class feature fit')
 model=make_pipeline(StandardScaler(),LogisticRegression(C=1.,solver='liblinear',max_iter=1000,random_state=7))
 model.fit(train,y);scaler=model.steps[0][1];head=model.steps[1][1]
 proof={'trained_rows':len(train),'feature_dimensions':train.shape[1],'fitted_parameters':int(head.coef_.size+head.intercept_.size),'training_feature_mean':scaler.mean_.tolist(),'training_feature_scale':scaler.scale_.tolist(),'coefficient':head.coef_.tolist(),'intercept':head.intercept_.tolist(),'iterations':head.n_iter_.tolist(),'fit_train_accuracy':float(model.score(train,y)),'C':1.,'penalty':'l2','solver':'liblinear','random_state':7,'hyperparameter_selection':'FIXED_BEFORE_FIRST_DEV_FIT; no dev tuning'}
 return model.predict(dev),proof
def _unit_scores(rows,prediction):
 by=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
 for r,p in zip(rows,prediction,strict=True):
  t=r['ability'];e=r['extra_info'];unit=e['patient_key'] if t=='mri_sequence' else e['statistical_cluster_key'];by[t][unit][e['patient_key']].append(int(p==LABELS[t].index(r['reward_model']['ground_truth'])))
 return {t:{u:float(np.mean([np.mean(v) for v in people.values()])) for u,people in units.items()} for t,units in by.items()}
def metrics(rows,prediction):
 _validate_rows(rows,'internal_dev');require(len(rows)==len(prediction) and all(p in (-1,0,1) for p in prediction),'Prediction coverage/domain mismatch');unit=_unit_scores(rows,prediction);tasks={}
 for t,labels in LABELS.items():
  cases=[(r,int(p)) for r,p in zip(rows,prediction) if r['ability']==t];truth=[labels.index(r['reward_model']['ground_truth']) for r,p in cases];pred=[p for r,p in cases]
  confusion={gold:{label:sum(y==i and p==j for y,p in zip(truth,pred)) for j,label in [*enumerate(labels),(-1,'INVALID')]} for i,gold in enumerate(labels)}
  tasks[t]={'questions':len(cases),'patients':len({r['extra_info']['patient_key'] for r,p in cases}),'independent_units':len(unit[t]),'unit':'patient' if t=='mri_sequence' else 'matched_two_person_cycle','question_accuracy':float(np.mean(np.array(truth)==pred)),'unit_macro_accuracy':float(np.mean(list(unit[t].values()))),'class_recall':{l:confusion[l][l]/sum(confusion[l].values()) for l in labels},'confusion':confusion}
 accuracies=[v['unit_macro_accuracy'] for v in tasks.values()];return {'tasks':tasks,'equal_task_unit_macro':float(np.mean(accuracies)),'worst_task_accuracy':min(accuracies),'descriptive_internal_dev_only':True}
def paired_units(rows,base,candidate):
 a=_unit_scores(rows,base);b=_unit_scores(rows,candidate);result={}
 for t,values in a.items():
  paired=[{'subject_id':u,'base':v,'candidate':b[t][u]} for u,v in values.items()]
  result[t]=paired_patient_difference(paired,n_bootstrap=2000,seed=7) if len(paired)>1 else {'subjects':len(paired),'mean_difference':float(np.mean([x['candidate']-x['base'] for x in paired])),'confidence_interval':None,'status':'TOO_FEW_UNITS'}
 return result

def apply_fitted(proof,features):
 x=np.asarray(features,dtype=np.float64);mean=np.array(proof['training_feature_mean']);scale=np.array(proof['training_feature_scale']);coef=np.array(proof['coefficient'])[0];intercept=proof['intercept'][0]
 require(x.ndim==2 and x.shape[1]==len(mean)==len(scale)==len(coef) and np.isfinite(x).all() and (scale>0).all(),'Saved linear-head feature contract mismatch')
 return (((x-mean)/scale)@coef+intercept>0).astype(int)
