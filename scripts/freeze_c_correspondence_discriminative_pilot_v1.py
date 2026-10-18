"""Freeze a finite, outcome-independent Dallas C pilot. CPU/PIL only; no fit/model.
Never reads a sealed/test artifact. Outputs private exact rows plus sanitized counts.
"""
import ast,collections,hashlib,json,os,sys
from pathlib import Path
from PIL import Image
W=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
canon=lambda x:json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)
def save(p,raw):
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():assert p.read_bytes()==raw,'Refuse changed frozen artifact'
 else:
  with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
def js(p,x):save(p,(json.dumps(x,sort_keys=True,indent=2,allow_nan=False)+'\n').encode())
def freeze():
 packpath=W/'local_data/v2_cpu_representation_v1/frozen_inputs_PRIVATE.json';assert sha(packpath)=='446808a0203ba1e6dd9f45e7f052a7667420b016ffff5a5b88f295d728fc79ab'
 p=json.loads(packpath.read_text());R=Path(p['source_paths_PRIVATE']['mixed_train']).parents[2]
 source=W/'src/mri_rl/visit_correspondence.py';tree=ast.parse(source.read_text());question=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='QUESTION' for t in n.targets))
 train=[r for r in p['train_PRIVATE'] if r['ability']=='mri_visit_correspondence'];dev=[r for r in p['dev_PRIVATE'] if r['ability']=='mri_visit_correspondence']
 assert len(train)==48 and len(dev)==16
 groups={};people={};checks=[]
 for split,rows,ncycles in [('train',train,12),('internal_dev',dev,4)]:
  by=collections.defaultdict(list);users=set()
  for r in rows:
   e=r['extra_info'];y=r['reward_model']['ground_truth'];q=e['patient_key'];c=e['candidate_patient_key'];users.update((q,c));by[e['statistical_cluster_key']].append(r)
   assert e['internal_split']==split and r['data_source']=='OpenNeuro_ds004856' and e['source_version']=='OpenNeuro ds004856 v1.3.0' and e['patient_lineage']=='BIDS_SUBJECT_VERIFIED'
   assert y==('same' if q==c else 'different') and e['query_source_sha256']!=e['candidate_source_sha256']
   assert e['question']==question and r['prompt']==[{'role':'user','content':[{'type':'image','image':r['images'][0]},{'type':'text','text':question}]}]
   assert all(value not in question for value in (r['id'],q,c,e['statistical_cluster_key'],e['query_source_sha256'],e['candidate_source_sha256']))
   image=Path(r['images'][0]);assert sha(image)==p['images_PRIVATE'][r['id']]
   with Image.open(image) as frame:
    assert frame.mode=='L' and frame.size==(768,384)
    for index,name in enumerate(('query_source_sha256','candidate_source_sha256')):
     token=hashlib.sha256(('ds004856-v1|volume|'+e[name]).encode()).hexdigest()[:24]
     panel=image.parent/(token+'.png')
     with Image.open(panel) as canonical:
      assert canonical.size==(384,384) and frame.crop((384*index,0,384*(index+1),384)).tobytes()==canonical.tobytes()
   checks.append({'record_id':r['id'],'image_sha256':sha(image),'canonical_halves_verified':True,'source_equality_truth_verified':True,'prompt_hidden_metadata_free':True})
  assert len(by)==ncycles and len(users)==2*ncycles
  for cycle,rs in by.items():
   assert len(rs)==4 and collections.Counter(r['reward_model']['ground_truth'] for r in rs)=={'same':2,'different':2}
   ps={r['extra_info']['patient_key'] for r in rs};assert len(ps)==2
   assert {(r['extra_info']['patient_key'],r['extra_info']['candidate_patient_key']) for r in rs}=={(a,b) for a in ps for b in ps}
   assert all(r['extra_info']['candidate_patient_key'] in ps for r in rs)
  groups[split]=by;people[split]=users
 assert not people['train']&people['internal_dev'] and not set(groups['train'])&set(groups['internal_dev'])
 cycles=sorted(groups['train'])[:2];selected=[];schedule=[]
 for cycle in cycles:
  rs=sorted(groups['train'][cycle],key=lambda r:(r['extra_info']['patient_key'],r['reward_model']['ground_truth']))
  schedule.append(list(range(len(selected),len(selected)+4)));selected.extend(rs)
 assert len(selected)==8 and len({r['extra_info']['patient_key'] for r in selected})==4
 # All existing dev rows, order and original assets remain fixed.
 dest=W/'local_data/c_correspondence_discriminative_pilot_v1'
 for name,rows in [('train8.verl.jsonl',selected),('internal_dev16.verl.jsonl',dev)]:save(dest/name,(''.join(canon(r)+'\n' for r in rows)).encode())
 old=W/'results/DALLAS_CORRESPONDENCE_TASK_V1.json';new=W/'results/V2_MATCHED_PIXEL_LINEAR_PROBE.json';oldr=json.loads(old.read_text());newr=json.loads(new.read_text())
 oldstats=oldr['shortcut_diagnostics']['controls'];newstats={k:v['tasks']['mri_visit_correspondence'] for k,v in newr['metrics'].items()}
 assert oldstats['spatial_multiview_correlation']['internal_dev']['correct']==12
 assert newstats['spatial16_L2_logistic']['question_accuracy']==14/16
 reconciled={'historical_spatial':{'method':'96x96 per-panel centered normalized correlation; one scalar threshold selected on48train pairs; fixed direction higher=same','train_correct':34,'train_rows':48,'dev_correct':12,'dev_rows':16,'same_recall':.5,'different_recall':1.,'source_sha256':sha(old)},'new_spatial':{'method':'Absolute difference of16x16 grayscale panel thumbnails, train-standardized L2 logistic(C1/liblinear/seed7);48train pairs','dev_correct':14,'dev_rows':16,'same_recall':.75,'different_recall':1.,'source_sha256':sha(new)},'new_nonspatial_histogram':{'dev_correct':11,'dev_rows':16,'same_recall':.625,'different_recall':.75},'historical_nonspatial_intensity_foreground':{'dev_correct':11,'dev_rows':16,'same_recall':.375,'different_recall':1.},'constant_class_and_saved_B28_U7':{'dev_correct':8,'dev_rows':16,'B28_U7_constant':'different'},'interpretation':'Same frozen16dev cases, different feature/fitting methods.14/16 does not replicate12/16, establish a VLM gain or brain-specific reasoning. Strong48-train-pair CPU models are descriptive controls, not same8-row-supervision comparisons.'}
 objective=W/'src/mri_rl/paired_discrimination_pilot_v2.py';assert sha(objective)=='79963d311b4797e8010a5b6c8de81c21f2f0adb1f5088613f2e867afcdfd021e'
 config={'schema':'C_same_different_discriminative_train_only_pilot_v1','status':'FROZEN_CPU_DATA_ONLY_NATIVE_NOT_RUN','author':'Codex /root/full_arm_execution_contract','source_pack_sha256':sha(packpath),'source_manifest_sha256':{k:p['source_sha256'][k] for k in ('mixed_train','mixed_dev')},'train_manifest':{'path':str(dest/'train8.verl.jsonl'),'sha256':sha(dest/'train8.verl.jsonl'),'rows':8,'matched_cycles':2,'people':4,'same':4,'different':4},'dev_manifest':{'path':str(dest/'internal_dev16.verl.jsonl'),'sha256':sha(dest/'internal_dev16.verl.jsonl'),'rows':16,'matched_cycles':4,'people':8,'same':8,'different':8},'selection':'Lexically first2 complete statistical_cluster_key values among frozen48train rows. No dev result or per-example model score enters selection. Each query paired with same-person and cycle-mate candidate; each update covers exactly one whole cycle.','selected_cycle_keys_PRIVATE':cycles,'schedule':schedule,'candidate_order':['different','same'],'objective':'Reuse label-agnostic discrimination_loss: binary cross-entropy of mean label-token log probabilities; no prefix/suffix/format loss. Mean of4image losses per update.','objective_reference':{'path':str(objective),'sha256':sha(objective),'function':'discrimination_loss'},'integration':'Reuse existing NativeSFT atomic weight/Adam/Python+MLX RNG/checkpoint journal and validated B target alignment through a separate C label-parameter adapter. Do not mutate B LABELS or call B-specific hardcoded factory for C.','initialization':'Exact common B28, independently from any sequence pilot result; common certified4bit model/prompt/precision.','starting_adapter_sha256':'b316db3d5eadf2e1fc0cbe80ccc93c7f7405858769f7d32b39a137dea3d6c572','seed':7,'optimizer':{'name':'Adam','learning_rate':1e-5,'betas':[.9,.999],'eps':1e-8,'bias_correction':False},'trainable_parameters':'Existing504 language LoRA tensors only; vision frozen','maximum_updates':2,'image_presentations':8,'maximum_training_candidate_forwards':16,'generations_during_training':0,'token_contract':{'native_input_counts':None,'candidate_target_ids':None,'target_prefix_label_suffix_partitions':None,'native_processor_measurement':'PENDING; do not copy old405 counts across corrected assistant-prefix contracts','exact_training_token_budget':None,'formula':'sum_over8rows(2*actual_input_tokens + len(target_different_ids)+len(target_same_ids))'},'evaluation':{'base':'Same commonB28 frozen16dev cases','updated':'Final2-update C checkpoint only; no dev-driven checkpoint selection','conditions':['real_image','no_image'],'paired_control':'Within each dev query, compare same-person vs matched cycle-mate candidate; these are the existing two rows, no new target construction','maximum_scored_rows_per_checkpoint':32,'maximum_base_scoring_forwards_per_checkpoint':96,'maximum_optional_shape_guard_forwards_per_checkpoint':16,'generation_maximum_per_checkpoint':32,'generation_max_tokens':32,'metrics':['same_recall','different_recall','balanced_accuracy','exact_class_accuracy','cycle_macro_accuracy','cycle_paired_deltas','real_minus_null_candidate_margin','query_matched_same_minus_different_margin','answer_class_frequencies'],'uncertainty':'2000 bootstrap resamples of4matched cycles/seed7; descriptive internal-dev uncertainty only','scoring':'Keep candidate values and strict original numeric guard. Only separately reviewed versioned shape guard may handle known null-prefix case; never silently drop or relax failures.','saved_CPU_controls':reconciled},'go_kill':{'engineering_hold':'Any input/label/prompt/reward mismatch, nonfinite loss/gradient, absent genuine update, wrong optimizer/RNG restore or unsafe resources blocks only affected stage','after2updates':'Report every outcome; do not tune or expand using dev accuracy. Positive image-discrimination evidence requires both classes retained, real-image balanced accuracy above both B28 and no-image, and positive matched-query candidate separation. A constant-class shift fails this criterion.','expansion':'Not authorized by this frozen pilot. A separate finite expansion may be justified only by training-only loss/margin evidence and frozen before any further dev outcome is examined.'},'checks':{'full_train48_dev16_cycles_and_people_validated':True,'all64_existing_pair_PNG_hashes_and128_source_panel_halves_verified':True,'hidden_source_identity_truth_and_metadata_free_model_prompt_verified':True,'train_dev_patient_cycle_disjoint':True,'sealed_final_data_read':False},'limitations':['Whole-head source-person correspondence, not clinical longitudinal change. Face/neck/skull and acquisition/intensity identity cues remain.','Two training cycles are4people, not8patients;4devcycles contain8people with linkednegativepairs.','Repeatedly examined dev is exploratory; no independent-patient generalization, seed robustness or publication-level accuracy claim.','No numerical/methodological novelty claimed from applying existing discriminative CE and LoRA.','Native C adapter/target tokens/input counts/exact budget/independent scope and actual post-update evaluation remain NOT_RUN.'],'mission_alignment':{'tag':'C_TASK_ELIGIBILITY / EXPLORATORY_DIAGNOSTIC / PRIMARY_VERO_TRANSFER','link':'Determine whether task-specific supervised image discrimination repairs collapsed correspondence answers before further VeroRL transfer. Original18RL+3SFT controls unchanged.','hypothesis':'Matched train-only same/different supervision can improve real-image correspondence beyond answer-prior changes.','falsifier':'No real-null or query-paired separation gain, loss of either class recall, or onlyconstant-answer prior shift contradicts image-discrimination improvement.','evidence_tier':'PROPOSED_NATIVE_EXPERIMENT / REPRODUCED_EXISTING_CPU_BASELINES'}}
 js(dest/'alignment_checks_PRIVATE.json',checks);config['alignment_checks_sha256']=sha(dest/'alignment_checks_PRIVATE.json');js(dest/'config_PENDING_NATIVE_MEASUREMENT_AND_REVIEW_PRIVATE.json',config)
 safe={k:config[k] for k in ('schema','status','author','source_pack_sha256','maximum_updates','image_presentations','maximum_training_candidate_forwards','checks','limitations','mission_alignment')};safe.update({'frozen_config_sha256':sha(dest/'config_PENDING_NATIVE_MEASUREMENT_AND_REVIEW_PRIVATE.json'),'train':{k:v for k,v in config['train_manifest'].items() if k!='path'},'dev':{k:v for k,v in config['dev_manifest'].items() if k!='path'},'baseline_reconciliation':reconciled,'token_measurement_and_C_native_adapter':'PENDING_NOT_APPROVED_FOR_NATIVE_LAUNCH','builder_sha256':sha(__file__)})
 js(W/'results/C_CORRESPONDENCE_DISCRIMINATIVE_PILOT_FROZEN_PREPARATION.json',safe)
 assert not any(k in sys.modules for k in ('mlx','mlx.core','mlx_vlm','torch','transformers'))
 print(json.dumps({'config_sha256':safe['frozen_config_sha256'],'public_sha256':sha(W/'results/C_CORRESPONDENCE_DISCRIMINATIVE_PILOT_FROZEN_PREPARATION.json'),'train_sha256':safe['train']['sha256'],'dev_sha256':safe['dev']['sha256'],'models_loaded':0,'sealed_data_read':False},sort_keys=True))
if __name__=='__main__':freeze()
