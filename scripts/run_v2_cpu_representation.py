"""Run matched MRI pixel linear probes; CPU only, all scientific inputs frozen before fit."""
import argparse,collections,copy,hashlib,importlib.util,json,os,re,resource,sys
from pathlib import Path
import numpy as np
from PIL import Image,__version__ as pillow_version
W=Path(__file__).resolve().parents[1];sys.path.insert(0,str(W/'src'))
from mri_rl.v2_cpu_representation import LABELS,validate_pack,pixel_features,pair_features,fit_predict,apply_fitted,metrics,paired_units
from mri_rl.vero_mri_reward import _answer_and_format
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(path,value):
 raw=json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n'
 with Path(path).open('x') as f:f.write(raw);f.flush();os.fsync(f.fileno())
def main():
 p=argparse.ArgumentParser();p.add_argument('--primary-root',type=Path,required=True);p.add_argument('--destination',type=Path,required=True);a=p.parse_args();R=a.primary_root.resolve();out=a.destination.resolve()
 assert out.is_relative_to(W/'local_data') and not out.exists(),'New ignored private output required'
 assert all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS')),'One-thread caps required'
 files={'full_sequence_train':R/'local_data/ds007045_bounded/vero_training_v1/train.verl.jsonl','mixed_train':R/'local_data/c_two_task_pool_v1/train.verl.jsonl','mixed_dev':R/'local_data/c_two_task_pool_v1/internal_dev.verl.jsonl','source_partition_metadata':R/'local_data/ds007045_bounded/visual_prompts/private_manifest.json','prospective_test_exclusion_metadata':R/'local_data/prospective_sequence_test14_v1/sealed_access_receipt_SAFE.json'}
 assert sha(files['mixed_train'])=='455efc650444f2820f26832bc29b042ce2ec3babc0c9fe745491705a3fdf7e2c';assert sha(files['mixed_dev'])=='95a489d0182c917657042f542868c629bcda827fe8d396bac562bc5318e5cd09'
 def rows(path):return [json.loads(x) for x in path.read_text().splitlines() if x]
 train=[r for r in rows(files['full_sequence_train']) if r['ability']=='mri_sequence'];mixed=rows(files['mixed_train']);dev=rows(files['mixed_dev']);original_sequence=[r for r in mixed if r['ability']=='mri_sequence']
 assert len(train)==168 and len({r['extra_info']['patient_key'] for r in train})==28;assert {r['id'] for r in original_sequence}<={r['id'] for r in train};assert len({r['extra_info']['patient_key'] for r in original_sequence})==28
 train += [r for r in mixed if r['ability']=='mri_visit_correspondence'];train=copy.deepcopy(train);dev=copy.deepcopy(dev)
 partition=json.loads(files['source_partition_metadata'].read_text());cases=partition['private_cases'];allowed={c['anonymous_token'] for c in cases if c['split']=='train'};forbidden={c['anonymous_token'] for c in cases if c['split']=='test'};site={c['anonymous_token']:c['private_site'] for c in cases if c['split']=='train'}
 assert len(allowed)==32 and len(forbidden)==8 and not allowed&forbidden
 for r in train+dev:
  if r['ability']=='mri_sequence':
   e=r['extra_info'];assert e['patient_key'] in allowed and e['patient_key'] not in forbidden;m=re.fullmatch(re.escape(e['patient_key'])+r'_z(\d{3})_(flair|t1ce)_sequence',r['id']);assert m and int(m[1]) in (45,91,136) and r['reward_model']['ground_truth']=={'flair':'FLAIR','t1ce':'contrast-enhanced T1'}[m[2]];e['level']=int(m[1]);e['source_site_control_PRIVATE']=site[e['patient_key']]
  else:r['extra_info']['level']=0;r['extra_info']['source_site_control_PRIVATE']='ds004856_Philips_matched_cycle'
 validate_pack(train,dev);assert len(train)==216 and len(dev)==40
 inventory={r['id']:sha(r['images'][0]) for r in train+dev};assert len(inventory)==256
 # Only prior safe receipt metadata is read for prospective test exclusion; no test registry cases, labels or pixels.
 excluded=json.loads(files['prospective_test_exclusion_metadata'].read_text())
 c2=W/'scripts/c2_image_shortcut_baseline.py';spec=importlib.util.spec_from_file_location('existing_c2',c2);baseline=importlib.util.module_from_spec(spec);spec.loader.exec_module(baseline)
 configuration={'hypothesis':'Do simple spatial MRI pixels discriminate held-out internal-dev classes better than histogram/metadata/null controls, while native B28/U7 answers may be collapsed?','train_rows':216,'dev_rows':40,'train_tasks':{'mri_sequence':168,'mri_visit_correspondence':48},'dev_tasks':{'mri_sequence':24,'mri_visit_correspondence':16},'source_units':{'train_sequence_people':28,'dev_sequence_people':4,'train_Dallas_people':24,'train_Dallas_cycles':12,'dev_Dallas_people':8,'dev_Dallas_cycles':4},'sequence_matching':'Each patient has both modalities at each exact level045/091/136; unchanged original PNGs.','methods':['spatial16_L2_logistic','histogram16_L2_logistic','source_site_level_L2_logistic','null_training_majority','spatial_head_blank_dev'],'head':{'C':1.,'solver':'liblinear','regularization':'L2','max_iter':1000,'seed':7,'standardization':'training only','hyperparameter_tuning':False},'feature_definition':{'sequence':'Existing C2 thumbnail16 grayscale/255 vs normalized16-bin histogram','Dallas':'Symmetric absolute difference between existing left/right384px panel thumbnail16 or histogram16; truth invariant under panel reversal','source_site_level':'Training-vocabulary one-hot source/site/level; hidden metadata shortcut control only','blank':'Apply same fitted spatial head to zero thumbnail (sequence) or zero panel-difference (Dallas), no refit'},'outcome_independent_control_selection':True,'Qwen_frozen_visual_features':'NOT_RUN; only actual rendered-pixel features used. No Qwen feature extraction or head training.','VLM_or_RL_training':False,'supervision_comparability':'V2 expands sequence supervision from original28 one-modality rows to168 same-patient matched rows. Not a matched-budget causal comparator to B28/U7.','sealed_test_pixels_or_gold_used':False,'prospective_sealed_test_exclusion_basis':'Original40 source people only; prior14-patient sealed-access receipt bound. No new subjects and no test payload read.','gate':'All inputs and methods saved before any feature-label fit; development targets used only for scoring.'}
 out.mkdir(parents=True);save(out/'frozen_inputs_PRIVATE.json',{'configuration':configuration,'source_sha256':{k:sha(v) for k,v in files.items()},'source_paths_PRIVATE':{k:str(v) for k,v in files.items()},'images_PRIVATE':inventory,'train_PRIVATE':train,'dev_PRIVATE':dev,'helper_source_sha256':sha(W/'src/mri_rl/v2_cpu_representation.py'),'runner_source_sha256':sha(__file__),'existing_feature_source_sha256':sha(c2),'sealed_access_receipt_bound':True})
 X={s:{'spatial':[],'histogram':[]} for s in ('train','dev')};features={}
 for split,group in [('train',train),('dev',dev)]:
  for r in group:
   image=Path(r['images'][0]);assert sha(image)==inventory[r['id']]
   if r['ability']=='mri_sequence':spatial,hist=baseline.extract_features(image,inventory[r['id']])
   else:
    with Image.open(image) as frame:
     assert frame.size==(768,384);g=frame.convert('L');left,right=g.crop((0,0,384,384)),g.crop((384,0,768,384));spatial,hist=pair_features(left,right);rs,rh=pair_features(right,left);assert np.array_equal(spatial,rs) and np.array_equal(hist,rh)
   features[r['id']]={'spatial':spatial,'histogram':hist}
 methods={m:np.full(len(dev),-1,dtype=int) for m in ('spatial16_L2_logistic','histogram16_L2_logistic','source_site_level_L2_logistic','null_training_majority','spatial_head_blank_dev')};fits={};npz={}
 for task in LABELS:
  tr=[r for r in train if r['ability']==task];dv=[r for r in dev if r['ability']==task];indices=[i for i,r in enumerate(dev) if r['ability']==task];y=np.array([LABELS[task].index(r['reward_model']['ground_truth']) for r in tr]);vocab=sorted({(r['data_source'],r['extra_info']['source_site_control_PRIVATE'],r['extra_info']['level']) for r in tr})
  for name,key in [('spatial16_L2_logistic','spatial'),('histogram16_L2_logistic','histogram'),('source_site_level_L2_logistic','metadata')]:
   def vector(r):return np.array([float((r['data_source'],r['extra_info']['source_site_control_PRIVATE'],r['extra_info']['level'])==v) for v in vocab]) if key=='metadata' else features[r['id']][key]
   A=np.array([vector(r) for r in tr]);B=np.array([vector(r) for r in dv]);guess,proof=fit_predict(A,y,B);assert np.array_equal(guess,apply_fitted(proof,B));assert max(proof['iterations'])<1000;methods[name][indices]=guess;fits[task+'|'+name]=proof;npz[task+'__'+key+'__train']=A;npz[task+'__'+key+'__dev']=B
   if key=='spatial':methods['spatial_head_blank_dev'][indices]=apply_fitted(proof,np.zeros_like(B))
  methods['null_training_majority'][indices]=int(y.mean()>.5)
 # Same40 previously authenticated native outputs; no generation or feature extraction is repeated.
 for name,path,key,expected in [('B28_saved_native',R/'local_data/c_two_task_pool_v1/evaluation_v1/C40_B28_baseline_independent_scoring_PRIVATE.json','normalized_response_components_PRIVATE','32ce2cd88a9d316e94f75177eb8c6c36e33bcd351fe4028ed9e88aa3c26dda8f'),('U7_saved_native',R/'local_data/c_two_task_pool_v1/tier_c_v1/U_raw_seed7_final_C40_release_v1/final_C40_scoring_PRIVATE.json','normalized_new_outputs_PRIVATE','de73178d59d5894d7ddfe4494f05dd785aa35211905b9457e9a1757e1f8b059e')]:
  assert sha(path)==expected;normalized=json.loads(path.read_text())[key];assert set(normalized)=={r['id'] for r in dev};prediction=[]
  for r in dev:
   answer,fmt=_answer_and_format(normalized[r['id']]['response']);allowed=[v.lower() for v in LABELS[r['ability']]];prediction.append(allowed.index(answer.lower()) if fmt>0 and answer.lower() in allowed else -1)
  methods[name]=np.array(prediction)
 np.savez(out/'features_PRIVATE.npz',**npz);save(out/'fits_and_predictions_PRIVATE.json',{'fits_PRIVATE':fits,'dev_rows_PRIVATE':[r['id'] for r in dev],'predictions_PRIVATE':{k:v.tolist() for k,v in methods.items()}})
 scores={k:metrics(dev,v) for k,v in methods.items()};paired={name:paired_units(dev,methods[name],methods['spatial16_L2_logistic']) for name in ('histogram16_L2_logistic','source_site_level_L2_logistic','null_training_majority','spatial_head_blank_dev','B28_saved_native','U7_saved_native')}
 assert not any(x in sys.modules for x in ('mlx','mlx.core','mlx_vlm','torch','transformers'))
 import sklearn
 report={'schema':'v2-matched-image-CPU-linear-probe-v1','status':'ACTUAL_CPU_FIT_AND_INTERNAL_DEV_EVALUATION_COMPLETE','reviewer':'Codex /root/two_task_admission','configuration':configuration,'metrics':scores,'spatial_paired_difference_from_control':paired,'bootstrap_scope':'Reused experiments.paired_patient_difference:2000 percentile resamples within each task,4 people or4 cycles. Exploratory repeated-dev uncertainty only, no seed/scanner/generalization coverage.','actual_linear_heads_fitted':6,'actual_Qwen_features_or_updates':0,'new_model_generations':0,'peak_process_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'versions':{'numpy':np.__version__,'pillow':pillow_version,'sklearn':sklearn.__version__},'evidence_sha256':{'frozen_inputs':sha(out/'frozen_inputs_PRIVATE.json'),'features_private':sha(out/'features_PRIVATE.npz'),'fits_and_predictions_private':sha(out/'fits_and_predictions_PRIVATE.json'),'helper':sha(W/'src/mri_rl/v2_cpu_representation.py'),'runner':sha(__file__),'tests':sha(W/'tests/test_v2_cpu_representation.py'),'existing_C2_features':sha(c2)},'limitations':['This is an actual small CPU classification-head fit on simple pixel features; it is not Qwen feature extraction, VLM SFT, GSPO or vision-adapter training.','Both classes are matched within every sequence person/level and every Dallas query/cycle, and train/dev patients/cycles are disjoint within source. Cross-source global identity is not verified.','Sequence168 paired rows expand original28-row supervision; different supervision/model families prevent causal attribution relative to B28/U7.','Repeatedly examined4 sequence dev people and4 Dallas cycles support exploratory inference only, not independent clinical generalization or seed stability.','Histogram/acquisition and Dallas whole-head face/skull/neck shortcuts remain possible. Spatial accuracy alone does not establish brain-grounded reasoning.','No sealed final100 image, label or prediction was used. No hyperparameter was selected on dev outcomes.']}
 result=W/'results/V2_MATCHED_PIXEL_LINEAR_PROBE.json';save(result,report);print(json.dumps({'public_result_sha256':sha(result),'frozen_inputs_sha256':report['evidence_sha256']['frozen_inputs'],'metrics':scores,'actual_linear_heads_fitted':6,'peak_RSS_bytes':report['peak_process_RSS_bytes']},sort_keys=True))
if __name__=='__main__':main()
