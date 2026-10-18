"""Pure CPU contracts for a finite frozen-Qwen vision diagnostic. No native imports."""
from collections import defaultdict
import numpy as np
from mri_rl.experiments import paired_patient_difference
LABELS=('FLAIR','contrast-enhanced T1')
def require(ok,msg):
 if not ok:raise ValueError(msg)
def _validate(rows,split,people=None):
 require(rows and len({r['id'] for r in rows})==len(rows),'Duplicate/empty sequence cases');groups=defaultdict(list)
 for r in rows:
  e=r['extra_info'];require(r['ability']=='mri_sequence' and e['internal_split']==split and r['reward_model']['ground_truth'] in LABELS and e['level'] in (45,91,136),'Sequence split/level/label mismatch');groups[e['patient_key']].append((e['level'],r['reward_model']['ground_truth']))
 expected={(z,l) for z in (45,91,136) for l in LABELS}
 require(all(len(v)==6 and set(v)==expected for v in groups.values()),'Need both modalities at each fixed level per person')
 if people is not None:require(len(groups)==people,'Unexpected independent patient count')
 return set(groups)
def select_sequence_records(train,dev):
 a=[r for r in train if r['ability']=='mri_sequence'];b=[r for r in dev if r['ability']=='mri_sequence'];people=_validate(a,'train');other=_validate(b,'internal_dev',4);require(len(people)>=4 and not people&other,'Patient leakage or insufficient train people')
 chosen=set(sorted(people)[:4]);return sorted([r for r in a if r['extra_info']['patient_key'] in chosen],key=lambda r:r['id'])+sorted(b,key=lambda r:r['id'])
def pool_merged(values,grid,merge,dimension):
 g=np.asarray(grid);x=np.asarray(values,dtype=np.float32)
 require(g.shape==(1,3) and merge==2 and all(int(v)==v and v>0 for v in g.ravel()) and g[0,0]==1 and g[0,1]%merge==0 and g[0,2]%merge==0,'Single still-image merged grid mismatch')
 n=int(np.prod(g[0])//merge**2);require(x.shape==(n,dimension) and np.isfinite(x).all(),'Merged vision token shape or finiteness mismatch')
 return x.astype(np.float64).mean(axis=0)
def dual_ridge(train,y,dev):
 x=np.asarray(train,dtype=np.float64);z=np.asarray(dev,dtype=np.float64);y=np.asarray(y)
 require(x.ndim==z.ndim==2 and x.shape[1]==z.shape[1] and len(x)==len(y) and set(y)=={0,1} and np.isfinite(x).all() and np.isfinite(z).all(),'Finite two-class train/dev features required')
 mean=x.mean(axis=0);scale=x.std(axis=0);scale=np.where(scale>1e-12,scale,1.);a=(x-mean)/scale;b=(z-mean)/scale;target=2*y.astype(float)-1;intercept=float(target.mean());gram=a@a.T;coef=np.linalg.solve(gram+np.eye(len(a)),target-intercept);decision=(b@a.T)@coef+intercept
 require(np.isfinite(decision).all(),'Nonfinite ridge decisions')
 proof={'gram_shape':list(gram.shape),'primal_D_by_D_formed':False,'fitted_rows':len(x),'feature_dimensions':x.shape[1],'alpha':1.,'training_feature_mean':mean.tolist(),'training_feature_scale':scale.tolist(),'dual_coefficients':coef.tolist(),'intercept':intercept,'decision_values':decision.tolist(),'dev_used_for_fit_or_hyperparameters':False}
 # Decisions are per-output evidence, not a fitting parameter.
 parameters={k:v for k,v in proof.items() if k!='decision_values'}
 return (decision>0).astype(int),parameters
def sequence_metrics(rows,prediction):
 people=_validate(rows,'internal_dev',4);require(len(prediction)==len(rows) and all(x in (-1,0,1) for x in prediction),'Prediction coverage/domain mismatch');groups=defaultdict(list);conf={label:{p:0 for p in (*LABELS,'INVALID')} for label in LABELS}
 for r,p in zip(rows,prediction):
  truth=r['reward_model']['ground_truth'];guess=LABELS[int(p)] if int(p)>=0 else 'INVALID';conf[truth][guess]+=1;groups[r['extra_info']['patient_key']].append(int(truth==guess))
 return {'rows':len(rows),'people':len(people),'patient_macro_accuracy':float(np.mean([np.mean(v) for v in groups.values()])),'question_accuracy':sum(conf[l][l] for l in LABELS)/len(rows),'recall':{l:conf[l][l]/sum(conf[l].values()) for l in LABELS},'confusion':conf,'invalid':sum(conf[l]['INVALID'] for l in LABELS),'exploratory_internal_dev_only':True}
def paired_sequence(rows,base,candidate):
 grouped=defaultdict(lambda:[[],[]])
 for r,a,b in zip(rows,base,candidate,strict=True):
  y=LABELS.index(r['reward_model']['ground_truth']);v=grouped[r['extra_info']['patient_key']];v[0].append(int(a==y));v[1].append(int(b==y))
 return paired_patient_difference([{'subject_id':p,'base':float(np.mean(v[0])),'candidate':float(np.mean(v[1]))} for p,v in grouped.items()],n_bootstrap=2000,seed=7)
