"""Fit fixed image-only controls on the exact added supervision of B/C pilots.
CPU only; immutable cached features; no hyperparameter or subset search.
"""
import hashlib,json,os,resource,sys
from pathlib import Path
import numpy as np
W=Path(__file__).resolve().parents[1];sys.path.insert(0,str(W/'src'))
from mri_rl.v2_cpu_representation import LABELS,validate_pack,fit_predict,apply_fitted,metrics,paired_units
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def main():
 assert all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'))
 old=W/'local_data/v2_cpu_representation_v1';pack=json.loads((old/'frozen_inputs_PRIVATE.json').read_text());assert sha(old/'frozen_inputs_PRIVATE.json')=='446808a0203ba1e6dd9f45e7f052a7667420b016ffff5a5b88f295d728fc79ab'
 priorpath=W/'results/V2_MATCHED_PIXEL_LINEAR_PROBE.json';assert sha(priorpath)=='fe9402cc4e1f4bbb2b1eb396922abbabda168f56d624441cc8b2750f2a576cf7';prior=json.loads(priorpath.read_text())
 for name,key in [('features_PRIVATE.npz','features_private'),('fits_and_predictions_PRIVATE.json','fits_and_predictions_private')]:assert sha(old/name)==prior['evidence_sha256'][key]
 bp=W/'local_data/paired_discrimination_pilot_v2/plan_v2_MEASURED_PENDING_FEATURE_AND_REVIEW_PRIVATE.json';cp=W/'local_data/c_correspondence_discriminative_pilot_v1/config_PENDING_NATIVE_MEASUREMENT_AND_REVIEW_PRIVATE.json'
 assert sha(bp)=='94aac6d202b050e231c6765449adfe80c856f9ab43606c43c80a13cf0009d748' and sha(cp)=='2f7424e1221f3597ec7d91af559140961de8d3623b3ee05d068111975347f3d6'
 b=json.loads(bp.read_text());c=json.loads(cp.read_text());train=b['rows_PRIVATE']+[json.loads(x) for x in Path(c['train_manifest']['path']).read_text().splitlines()];dev=pack['dev_PRIVATE'];validate_pack(train,dev)
 assert len(train)==16 and len(dev)==40 and all(sum(r['ability']==t for r in train)==8 for t in LABELS)
 old_by={r['id']:r for r in pack['train_PRIVATE']};assert all(r==old_by[r['id']] for r in train)
 config={'hypothesis':'Fixed spatial/intensity/source heads with exactly eight added training rows per task; no outcome-dependent row or hyperparameter selection.','train_tasks':{t:8 for t in LABELS},'dev_tasks':{'mri_sequence':24,'mri_visit_correspondence':16},'selected_training_units':{'sequence_people':len({r['extra_info']['patient_key'] for r in train if r['ability']=='mri_sequence'}),'sequence_pairs':4,'Dallas_people':4,'Dallas_cycles':2},'head':{'C':1.,'solver':'liblinear','max_iter':1000,'seed':7,'standardization':'selected8 training rows only'},'selection':'Exactly frozen paired B four patient-level keys and C first two complete train cycles; no new choice.','comparability':'Matches added pilot supervision rows, not architecture, pretraining history, optimizer budget or total B28 prior supervision.','feature_reuse':'Authenticate original immutable feature NPZ; slice selected training rows and all unchanged dev rows. No image/feature re-extraction.','VLM_or_RL_training':False,'sealed_final100_used':False,'dev_hyperparameter_tuning':False}
 out=W/'local_data/pilot8_cpu_controls_v1';assert not out.exists();out.mkdir()
 frozen={**pack,'configuration':config,'train_PRIVATE':train,'helper_source_sha256':sha(W/'src/mri_rl/v2_cpu_representation.py'),'runner_source_sha256':sha(__file__),'parent_inputs_sha256':sha(old/'frozen_inputs_PRIVATE.json'),'parent_features_sha256':sha(old/'features_PRIVATE.npz'),'B_pilot_sha256':sha(bp),'C_pilot_sha256':sha(cp)}
 save(out/'frozen_inputs_PRIVATE.json',frozen) # before any new fit
 methods={k:np.full(40,-1,dtype=int) for k in ('spatial16_L2_logistic','histogram16_L2_logistic','source_site_level_L2_logistic','null_training_majority','spatial_head_blank_dev')};fits={};arrays={}
 with np.load(old/'features_PRIVATE.npz',allow_pickle=False) as original:
  for task in LABELS:
   full=[r for r in pack['train_PRIVATE'] if r['ability']==task];chosen=[r for r in train if r['ability']==task];lookup={r['id']:i for i,r in enumerate(full)};take=[lookup[r['id']] for r in chosen];idx=[i for i,r in enumerate(dev) if r['ability']==task];y=np.array([LABELS[task].index(r['reward_model']['ground_truth']) for r in chosen]);assert list(np.bincount(y))==[4,4]
   for method,key in [('spatial16_L2_logistic','spatial'),('histogram16_L2_logistic','histogram'),('source_site_level_L2_logistic','metadata')]:
    A=original[task+'__'+key+'__train'][take];B=original[task+'__'+key+'__dev'];guess,proof=fit_predict(A,y,B);assert np.array_equal(guess,apply_fitted(proof,B)) and max(proof['iterations'])<1000
    methods[method][idx]=guess;fits[task+'|'+method]=proof;arrays[task+'__'+key+'__train']=A;arrays[task+'__'+key+'__dev']=B
    if key=='spatial':methods['spatial_head_blank_dev'][idx]=apply_fitted(proof,np.zeros_like(B))
   methods['null_training_majority'][idx]=0
 previous=json.loads((old/'fits_and_predictions_PRIVATE.json').read_text())
 assert previous['dev_rows_PRIVATE']==[r['id'] for r in dev]
 for name in ('B28_saved_native','U7_saved_native'):methods[name]=np.array(previous['predictions_PRIVATE'][name])
 np.savez(out/'features_PRIVATE.npz',**arrays);save(out/'fits_and_predictions_PRIVATE.json',{'fits_PRIVATE':fits,'dev_rows_PRIVATE':[r['id'] for r in dev],'predictions_PRIVATE':{k:v.tolist() for k,v in methods.items()}})
 report={'schema':'pilot8-matched-added-supervision-CPU-controls-v1','status':'ACTUAL_SIX_CPU_HEAD_FITS_AND_FIXED_DEV_EVALUATION_COMPLETE','executor_id':'Codex /root','configuration':config,'metrics':{k:metrics(dev,v) for k,v in methods.items()},'spatial_paired_difference_from_control':{name:paired_units(dev,methods[name],methods['spatial16_L2_logistic']) for name in methods if name!='spatial16_L2_logistic'},'actual_linear_heads_fitted':6,'new_Qwen_or_RL_updates':0,'feature_extraction_repeated':False,'evidence_sha256':{'frozen_inputs':sha(out/'frozen_inputs_PRIVATE.json'),'features_private':sha(out/'features_PRIVATE.npz'),'fits_and_predictions_private':sha(out/'fits_and_predictions_PRIVATE.json'),'runner':sha(__file__),'helper':sha(W/'src/mri_rl/v2_cpu_representation.py')},'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'limitations':['Exploratory repeatedly examined four sequence people/four Dallas cycles; no independent clinical or seed generalization.','Feature metadata vocabulary originates from permitted training pool; selected-row-only scaler/fit means unselected one-hot columns have zero coefficients. No dev outcomes enter fitting.','Matches added supervision rows only; different model/pretraining/compute from B28/VLM. Not a causal VLM gain.','Intensity/acquisition and whole-head identity shortcuts remain unresolved; no MRI anatomical reasoning claim.','Native paired B/C pilot and frozen-Qwen feature extraction remain separate and NOT_RUN.']}
 assert not any(x.split('.')[0] in {'mlx','mlx_vlm','torch','transformers'} for x in sys.modules)
 target=W/'results/PILOT8_MATCHED_ADDED_SUPERVISION_CPU_CONTROLS.json';save(target,report);print(json.dumps({'result_sha256':sha(target),'metrics':report['metrics'],'models':6},sort_keys=True))
if __name__=='__main__':main()

