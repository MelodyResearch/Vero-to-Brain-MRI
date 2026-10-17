
"""CPU-only C40 paired-output diagnosis and score-independent probe freeze.
No native runtime, model launch, training state edits or sealed-test access.
"""
from pathlib import Path
import argparse,hashlib,json,collections,sys
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from mri_rl.c40_visual_controls import build_pairs,paired_greedy_summary,select_probe_rows,LABELS,SALT,CENTRAL_LEVEL
from mri_rl.two_task_evaluation import project_records
from mri_rl.two_task_pool import sha,canonical_sha,write_exact,require
from mri_rl.two_task_analysis import score_predictions
from mri_rl import answer_likelihood as al

B=R/'local_data/c_two_task_pool_v1'
def load(path):return json.loads(Path(path).read_text())
def save(path,obj):write_exact(path,(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n').encode())

def main():
 p=argparse.ArgumentParser();p.add_argument('--destination',type=Path,required=True);a=p.parse_args()
 destination=a.destination.resolve();require(destination.is_relative_to(R/'local_data'),'Raw pairing/selection must stay private')
 source=B/'internal_dev.verl.jsonl';require(sha(source)=='95a489d0182c917657042f542868c629bcda827fe8d396bac562bc5318e5cd09','Frozen C40 source changed')
 base_public=R/'results/C40_B28_COMMON_START_BASELINE.json';up_public=R/'results/U_RAW_SEED7_FINAL_C40_PAIRED_EVALUATION.json'
 require(sha(base_public)=='9e838d469736e373b44626baba8acba388e8dbe44bcaa0bed97018bfad561c4a','B28 authenticated public receipt changed')
 require(sha(up_public)=='09d65834f4e61db5ecdc846af61bf5c0af16639720d0b51887998601e1bce5dd','U7 authenticated paired evaluation changed')
 bp=load(base_public);up=load(up_public)
 base_private=B/'evaluation_v1/C40_B28_baseline_independent_scoring_PRIVATE.json'
 updir=B/'tier_c_v1/U_raw_seed7_final_C40_release_v1';up_private=updir/'final_C40_scoring_PRIVATE.json'
 require(sha(base_private)==bp['independent_private_receipt_sha256'] and sha(up_private)==up['evidence_sha256']['private_numeric_analysis'],'Previously authenticated normalized response artifacts changed')
 ba=load(base_private);ua=load(up_private)
 predictions={'B28':ba['normalized_response_components_PRIVATE'],'U_raw_seed7':ua['normalized_new_outputs_PRIVATE']}
 bindings=load(updir/'analysis_bindings_PRIVATE.json')
 require(sha(updir/'analysis_bindings_PRIVATE.json')==load(R/'results/U_RAW_SEED7_FINAL_C40_SCOPED_RELEASE.json')['analysis_bindings_sha256'],'Source-bound final scope changed')
 require(sha(bindings['scope']['output'])==up['evidence_sha256']['output']==ua['strict_native_output_binding_proof']['output_sha256'],'Actual U7 output bytes changed')
 require(sha(B/'evaluation_v1/legacy_B28_sequence24_original_rows_PRIVATE.jsonl')==bp['components']['legacy_sequence']['preserved_original_rows_sha256'],'Preserved legacy24 changed')
 require(sha(B/'evaluation_v1/B28_common_start_outputs_PRIVATE.jsonl')==bp['components']['new_dallas']['output_sha256'],'Actual baseline Dallas16 changed')
 rows=[json.loads(x) for x in source.read_text().splitlines()]
 # Freeze selection before using outputs in this diagnostic. The selector takes no scores.
 selected=select_probe_rows(rows);contract=load(bindings['contract_path']);projected=project_records(rows,contract)
 by_id={r['record_id']:r for r in projected};pairs=build_pairs(rows)
 inventory={r['id']:{'image_sha256':sha(r['images'][0])} for r in rows}
 for key,item in inventory.items():require(item['image_sha256']==bindings['scope']['image_sha256'][key],'Source pixel bytes differ from authenticated actual native output')
 # Verify candidate replacement really preserves the query pixels. Source-specific
 # panel hashes are local only; no new images, rendering or registration is performed.
 from PIL import Image
 panels_by_source={};dallas_pixels={}
 for row in rows:
  if row['ability']!='mri_visit_correspondence':continue
  with Image.open(row['images'][0]) as original:
   require(original.size==(768,384),'Frozen pair geometry changed')
   image=original.convert('L');halves=[image.crop((0,0,384,384)),image.crop((384,0,768,384))]
  hashes=[hashlib.sha256(x.tobytes()).hexdigest() for x in halves]
  e=row['extra_info']
  for source_sha,pixel_sha in zip((e['query_source_sha256'],e['candidate_source_sha256']),hashes):
   require(source_sha not in panels_by_source or panels_by_source[source_sha]==pixel_sha,'A repeated source volume changed pixels between pairs')
   panels_by_source[source_sha]=pixel_sha
  dallas_pixels[row['id']]={'query_pixels_sha256':hashes[0],'candidate_pixels_sha256':hashes[1]}
 for pair in pairs['mri_visit_correspondence']:
  require(dallas_pixels[pair['negative']['id']]['query_pixels_sha256']==dallas_pixels[pair['positive']['id']]['query_pixels_sha256'],'Replacement altered query panel')
 substitutions=[];private_pairs=[]
 for task,group in pairs.items():
  for pair in group:
   left,right=pair['negative'],pair['positive']
   private_pairs.append({'task':task,'unit':pair['unit'],'level':pair['level'],'negative_id':left['id'],'positive_id':right['id'],
      'negative_image_sha256':inventory[left['id']]['image_sha256'],'positive_image_sha256':inventory[right['id']]['image_sha256'],
      'query_source_sha256':left['extra_info'].get('query_source_sha256'),'candidate_source_sha256':[left['extra_info'].get('candidate_source_sha256'),right['extra_info'].get('candidate_source_sha256')]})
   for original,replacement in ((left,right),(right,left)):
    require(by_id[original['id']]['prompt']==by_id[replacement['id']]['prompt'],'Input permutation changed native system/question')
    if original in selected:
     substitutions.append({'original_source_record_id':original['id'],'displayed_source_record_id':replacement['id'],
       'actual_displayed_target':replacement['reward_model']['ground_truth'],'reuses_existing_input_and_greedy_prediction':True,
       'condition':'same_patient_opposite_sequence' if task=='mri_sequence' else 'same_query_replaced_candidate'})
 records=[]
 for row in selected:
  record=by_id[row['id']]
  records.append({**record,'image_sha256':inventory[row['id']]['image_sha256'],'source_record_id':row['id'],
   'candidate_labels':list(LABELS[row['ability']]),'visible_target_PRIVATE':row['reward_model']['ground_truth'],
   'condition':'real_image','reuse_greedy_output':True})
 for task in LABELS:
  questions={r['extra_info']['question'] for r in selected if r['ability']==task};require(len(questions)==1,'No-image task question differs')
  records.append({'record_id':'no_image_'+task,'category':task,'image':None,'image_sha256':None,
    'prompt':[{'role':'system','content':contract['system_instruction']},{'role':'user','content':next(iter(questions))}],
    'candidate_labels':list(LABELS[task]),'visible_target_PRIVATE':None,'condition':'no_image','reuse_greedy_output':False})
 adapters={}
 for label,path,expected in (
  ('B28',B/'tier_c_v1/U_raw_seed7_run_v1/boundary_0000/adapter.safetensors',bp['common_adapter_sha256']),
  ('U_raw_seed7',B/'tier_c_v1/U_raw_seed7_run_v1/boundary_0125/adapter.safetensors',up['fresh_final_adapter_reload']['adapter_sha256'])):
  require(sha(path)==expected and sha(path.with_name('adapter_config.json'))==bp['adapter_config_sha256'],'Exact checkpoint/config differs')
  adapters[label]={'path':str(path),'sha256':expected,'adapter_config_sha256':bp['adapter_config_sha256']}
 plan={'schema':'C40-fixed-visual-controls-probe-v1','status':'FROZEN_NOT_RUN','research_tag':'EXPLORATORY_DIAGNOSTIC','weight_updates':0,
  'selection':{'rule':'First two sequence source people and first Dallas complete matched cycle by SHA256(salt|hidden unit key); fixed z091; no output or correctness argument to selector.',
   'salt':SALT,'sequence_level':CENTRAL_LEVEL,'sequence_people':2,'dallas_people':2,'dallas_cycles':1,'real_image_records':8,'unique_no_image_prompts':2},
  'source_manifest':str(source),'source_manifest_sha256':sha(source),'projection_sha256':sha(bindings['eval_manifest']),
  'prompt_contract_path':bindings['contract_path'],'prompt_contract_sha256':sha(bindings['contract_path']),
  'model_path':bindings['expected_config']['model_path'],'derived_cache':bindings['scope']['derived_cache'],
  'checkpoints':adapters,'records_PRIVATE':records,'derived_substitutions_PRIVATE':substitutions,
  'likelihood':{'prefix':al.PREFIX,'suffix':al.SUFFIX,'label_logp':'Existing summarize_tokens: primary mean label tokens; raw sum and first-label-token margins retained separately.',
    'candidate_order':{k:list(v) for k,v in LABELS.items()},'candidate2_minus_candidate1':{'mri_sequence':'T1 minus FLAIR','mri_visit_correspondence':'same minus different'},
    'candidate_forwards_per_record':2,'independent_prefix_forwards_per_record':1,'forwards_per_checkpoint':30,'common_prefix_tolerance':.002,
    'partition_function_code':'answer_likelihood.partition_target with task-local LABELS binding; original module unchanged',
    'comparison_function_code':'answer_likelihood.compare_candidates with task-local LABELS binding; legacy t1 field names must be explicitly interpreted as candidate2 minus candidate1'},
  'greedy':{'max_tokens':256,'temperature':0,'new_generations_per_checkpoint':2,'existing_real_image_generations_reused_per_checkpoint':8},
  'source_sha256':{n:sha(R/'src/mri_rl'/n) for n in ('c40_visual_controls.py','answer_likelihood.py','paced_answer_likelihood.py','two_task_analysis.py','two_task_evaluation.py','two_task_native_results.py','vero_mri_reward.py','visit_correspondence.py')},
  'launch_conditions':['Original21training stages terminal before any new8B load; independent exact read-only scope review; one model process; current memory/power/thermal safeguards; save/flush each completed record.'],
  'sealed_test_used':False,'no_image_accuracy_defined':False,'panel_order_swap_truth':'Equality is symmetric; panel reversal never flips same/different target. Panel-order model inference NOT_RUN.'}
 save(destination/'probe_plan_PRIVATE.json',plan)
 save(destination/'pairing_inventory_PRIVATE.json',{'pairs':private_pairs,'image_inventory':inventory,'dallas_panels_PRIVATE':dallas_pixels})
 noimage=destination/'no_image_native.jsonl';write_exact(noimage,(''.join(json.dumps({k:r[k] for k in ('record_id','category','image','prompt')},sort_keys=True)+'\n' for r in records if r['condition']=='no_image')).encode())
 summaries={}
 for label,normalized in predictions.items():
  scored=score_predictions(rows,normalized)
  require(scored['public']==(bp['primary_metrics'] if label=='B28' else up['updated_U_raw_seed7']),'Reused predictions do not reproduce authenticated task scores')
  summaries[label]=paired_greedy_summary(rows,normalized)
 report={'schema':'C40-saved-output-visual-controls-v1','status':'ACTUAL_SAVED_GREEDY_PAIRED_ANALYSIS_COMPLETED_PENDING_NEW_PROBES','evidence_tier':'REPRODUCED_FROM_AUTHENTICATED_NATIVE_OUTPUTS',
  'reviewer':'/root/two_task_admission','research_tag':'EXPLORATORY_DIAGNOSTIC','paired_saved_output_results':summaries,
  'new_model_generations':0,'new_likelihood_forwards':0,'pending_probe_status':'NOT_RUN','pending_plan_sha256':sha(destination/'probe_plan_PRIVATE.json'),
  'frozen_probe_counts':plan['selection'],'pending_new_greedy_generations_total':4,'pending_likelihood_forwards_total':60,
  'derived_substitution_controls':'Exact permutations of existing native image+system+question inputs; gold is the displayed source image/candidate reference. No extra inference or new patients.',
  'Dallas_query_pixel_invariance_verified':True,'Dallas_source_panels_verified':len(panels_by_source),
  'Dallas_panel_order_swap':'Truth same/different is invariant under panel reversal; model response to reversed panels NOT_RUN.',
  'evidence_sha256':{'C40_source':sha(source),'B28_public':sha(base_public),'B28_private':sha(base_private),'U7_public':sha(up_public),'U7_private':sha(up_private),
    'pairing_private':sha(destination/'pairing_inventory_PRIVATE.json'),'probe_plan_private':sha(destination/'probe_plan_PRIVATE.json'),'no_image_native_manifest':sha(noimage),'analysis_source':sha(__file__),
    'helper_source':sha(R/'src/mri_rl/c40_visual_controls.py')},
  'mission_alignment_check':{'main_question':'Diagnose whether real Vero-to-MRI updates changed source-relevant image discrimination versus shared answer preference.',
   'claim_and_falsifier':'Correctly directed same-person/level or same-query pair switches support sensitivity; constant answers or reverse switches fail this greedy diagnostic, without proving absence of all latent image information.',
   'change_isolation':'Original18RL/3SFT scientific contracts and native outputs untouched; new exploratory probes separately frozen.',
   'provenance':'Existing authenticated C40 source, B28 and finalU7 outputs; hidden source patient/cycle and image-byte hashes retained locally.',
   'shortcut_baseline':'Pair contrasts within-person for sequence, within-query for Dallas; actual no-image/likelihood still NOT_RUN.',
   'resources':'CPU-only existing internal-dev PNG metadata/pixels and saved text. No models, queue edits or sealed test.'},
  'limitations':['Four repeatedly examined sequence people and four Dallas cycles; descriptive internal-dev diagnosis, not independent generalization.',
   'Greedy answer-pair invariance does not prove image embeddings or candidate likelihoods are invariant.',
   'Dallas whole-head face/skull/neck/intensity shortcuts remain; source and task confounded.',
   'No-image and fixed-prefix likelihood need actual future approved native runs; no causal explanation of U7 decline identified by these saved controls alone.'],
  'sealed_test_used':False,'independent_patient_generalization_claimed':False,'whole_research_completed':False}
 save(R/'results/U7_C40_VISUAL_CONTROLS_SAVED_OUTPUTS.json',report)
 print(json.dumps({'safe_report_sha256':sha(R/'results/U7_C40_VISUAL_CONTROLS_SAVED_OUTPUTS.json'),'private_probe_plan_sha256':sha(destination/'probe_plan_PRIVATE.json'),'paired_results':summaries,'pending_likelihood_forwards':60,'new_model_generations':0},sort_keys=True))
if __name__=='__main__':main()
