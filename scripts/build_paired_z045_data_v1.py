#!/usr/bin/env python3
"""CPU-only preparation: paired z045 pool, unchanged Dallas, frozen task slots.

Use the existing tokenizer-capable Python environment. No MLX/model import,
training launch, source-volume decoding, rendering, fitting or network access.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys

W=Path(__file__).resolve().parents[1];sys.path.insert(0,str(W/'src'))
from mri_rl.paired_z045_data_v1 import select_z045,replace_sequence_slots,exposure_summary,canonical_sha,require,SEQ,DALLAS,LABELS
PINS={'reviewed168':'761c1618a455031098eda68c3080808d9d610beb63293bb50af597eea3caaf9b',
 'source_audit':'eca155b538b5f83fdcc6b6242f113aebdbfac9210511b68f4a6c01bc53571390',
 'blind_key':'b72735cb28b26c065d33bc5717b8ecfd1213f5b2e3f24330a0029baa840bc0b0',
 'full_MRI_train':'0efb588b328e3d9af2a339d9237836357a5baff5d9db8b376dc716de3d1f10e3',
 'original76':'455efc650444f2820f26832bc29b042ce2ec3babc0c9fe745491705a3fdf7e2c',
 'Dallas48':'6adf06674050e395b704e4b76103f067166b905016920195603e271690de4f14',
 'parent_uniform_receipt':'3c2567723e7bc90a65464c8c3645f49ad91c38a0ad886961c508bc1f8a0e3879',
 'native_token_measurements':'0839e1a9c10e3a7d2b7d45d8f7bceeda46cdcb94146adb3b2233abc82ebec5e9',
 'source_README':'66efa8e9ffd194b0aca9e52e9a62201db1b447f9b1f38a11ec8daec265293d96'}


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def bound(path,name,jsonl=False):
    require(sha(path)==PINS[name],'Frozen TRAIN input mismatch: '+name)
    return [json.loads(s) for s in path.read_text().splitlines() if s] if jsonl else json.loads(path.read_text())


def save(path,obj):
    with path.open('x') as f:json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')


def lines(rows):return ''.join(json.dumps(r,sort_keys=True,allow_nan=False)+'\n' for r in rows).encode()


def pool_stats(rows):
    labels=Counter(r['reward_model']['ground_truth'] for r in rows);people=defaultdict(Counter)
    for r in rows:people[r['extra_info']['patient_key']][r['reward_model']['ground_truth']]+=1
    majority=sum(max(v.values()) for v in people.values());n=len(rows)
    entropy=sum(-sum(c/n*math.log2(c/sum(v.values())) for c in v.values()) for v in people.values())
    return {'rows':n,'people':len(people),'modality_rows':dict(labels),'rows_per_person_counts':dict(Counter(str(sum(v.values())) for v in people.values())),
        'people_both_modalities':sum(set(v)==set(LABELS) for v in people.values()),'patient_only_lookup_resubstitution_correct':majority,
        'patient_only_lookup_resubstitution_fraction':majority/n,'H_modality_given_patient_bits':entropy,
        'interpretation':'TRAIN contingency only;not a fitted model or generalization score'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--primary-root',type=Path,required=True);p.add_argument('--qualification-worktree',type=Path,required=True)
    p.add_argument('--destination',type=Path,default=W/'local_data/paired_z045_data_v1/candidate_v2');a=p.parse_args()
    require(all(os.environ.get(k)=='1' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']),'One CPU thread required')
    from PIL import Image
    from tokenizers import Tokenizer
    from mri_rl.vero_category_sampler import build_category_schedule
    from mri_rl.two_task_pool import canonical_sha as old_row_sha
    from mri_rl.answer_likelihood import PREFIX,SUFFIX
    R=a.primary_root.resolve();Q=a.qualification_worktree.resolve();out=a.destination.resolve();public=W/'results/PAIRED_Z045_DATA_V1.json'
    require(out.is_relative_to(W/'local_data/paired_z045_data_v1') and not out.exists() and not public.exists(),'New scoped outputs required')
    A=Q/'local_data/sequence_qualification_v1/audit_v1';work=R.parent.parent/'work'
    paths={'reviewed168':A/'future_loader_manifest_reviewed_PRIVATE.jsonl','source_audit':A/'audit_PRIVATE.json','blind_key':A/'blind_key_PRIVATE.json',
        'full_MRI_train':R/'local_data/ds007045_bounded/vero_training_v1/train.verl.jsonl','original76':R/'local_data/c_two_task_pool_v1/train.verl.jsonl',
        'Dallas48':R/'local_data/dallas_correspondence_prospective40_v1/task_v1/train.verl.jsonl',
        'parent_uniform_receipt':R/'.worktrees/mri-adaptive-accuracy-gate/local_data/uniform_balance_continuation_v1/run/boundary_0008/receipt.json',
        'native_token_measurements':R/'local_data/c_two_task_pool_v1/native_tokens_v2/measurements_PRIVATE.json',
        'source_README':work/'ds007045_metadata/README.txt'}
    reviewed=bound(paths['reviewed168'],'reviewed168',True);audit=bound(paths['source_audit'],'source_audit');blind=bound(paths['blind_key'],'blind_key')
    full=bound(paths['full_MRI_train'],'full_MRI_train',True);old=bound(paths['original76'],'original76',True);dallas=bound(paths['Dallas48'],'Dallas48',True)
    parent=bound(paths['parent_uniform_receipt'],'parent_uniform_receipt');measurements=bound(paths['native_token_measurements'],'native_token_measurements')
    require(sha(paths['source_README'])==PINS['source_README'],'Official modality README changed')
    old28=[r for r in old if r['ability']==SEQ];require(len(old)==76 and len(old28)==28 and len(dallas)==48,'Original task coverage differs')
    require([r for r in old if r['ability']==DALLAS]==dallas,'Dallas objects changed')
    original_rows={r['extra_info']['index']:r for r in full if r['ability']==SEQ}
    for r in old28:require(r==original_rows[r['extra_info']['index']] and '_z045_' in r['id'],'Historical sequence source/level differs')
    require(len({r['extra_info']['patient_key'] for r in old28})==28,'Historical source people differ')
    new56=select_z045(reviewed,full);people_order=[r['extra_info']['patient_key'] for r in old28]
    require(set(people_order)=={r['extra_info']['patient_key'] for r in new56},'Candidate source people differ')
    # Fixed source order: historical patient order, then FLAIR/T1; no score-dependent choice.
    new56.sort(key=lambda r:(people_order.index(r['extra_info']['patient_key']),LABELS.index(r['reward_model']['ground_truth'])))
    source_records={(r['patient_PRIVATE'],r['modality']):r for r in audit['source_records_PRIVATE']}
    pixel={r['record_id_PRIVATE']:r for r in audit['pixel_checks_PRIVATE']};geometry={r['patient_PRIVATE']:r for r in audit['paired_geometry_PRIVATE']}
    blind_map={side['source_row_PRIVATE']['extra_info']['index']:side for pair in blind['pairs_PRIVATE'] for side in pair['sides_PRIVATE'].values()}
    require(len(blind_map)==168 and len(source_records)==56,'Existing source qualification inventory differs')
    source_bindings=[];site_counts=Counter();scanner_matched=0
    for person in people_order:
        g=geometry[person];require(g['shape_equal'] and g['affine_equal_atol_1e_6'] and g['orientation_equal'],'Saved paired geometry mismatch')
        scanner_matched+=len({audit['row_metadata_PRIVATE'][original_rows[r['extra_info']['index']]['id']]['scanner'] for r in new56 if r['extra_info']['patient_key']==person})==1
    for row in new56:
        e=row['extra_info'];q=row['paired_z045_data_v1'];s=source_records[e['patient_key'],q['modality_PRIVATE']];original=original_rows[e['index']];check=pixel[original['id']];b=blind_map[e['index']]
        require(b['source_row_PRIVATE']['extra_info']==original['extra_info'] and check['level']==45 and check['pixel_exact'] and check['max_absolute_pixel_difference']==0,'Saved original pixel/source linkage differs')
        require(check['source_sha256']==s['source_sha256'] and check['modality_PRIVATE']==q['modality_PRIVATE'],'Source volume modality SHA mismatch')
        # Annexpointer/header/sidecar metadata only; actual volume bytes are not reopened.
        pointer=os.readlink(s['official_annex_pointer_PRIVATE']);require(s['source_sha256'] in pointer,'Saved source SHA differs from current official annex identity')
        require(sha(Path(s['sidecar_path_PRIVATE']))==s['sidecar_sha256'],'Acquisition sidecar changed')
        require((s['modality']=='flair' and Path(s['sidecar_path_PRIVATE']).name.endswith('_FLAIR.json')) or
            (s['modality']=='t1ce' and Path(s['sidecar_path_PRIVATE']).name.endswith('_run-02_T1w.json')),'Source modality sidecar collection mismatch')
        require(sha(Path(original['images'][0]))==check['png_sha256']==b['source_png_sha256'],'Original rendered PNG changed')
        actual=sha(Path(row['images'][0]));require(actual==b['opaque_png_sha256']==b['source_png_sha256'],'Opaque/source PNG byte identity differs')
        with Image.open(row['images'][0]) as im:
            require(im.size==(364,436) and im.mode=='RGB' and not im.info,'Frozen RGB dimensions/embedded metadata changed')
        expected=[{'role':'user','content':[{'type':'image','image':row['images'][0]},{'type':'text','text':e['question']}]}]
        require(row['prompt']==expected and row['reward_model']==original['reward_model'],'Opaque prompt or hidden target changed')
        require(not any(t in Path(row['images'][0]).name.lower() for t in ['flair','t1ce',e['patient_key'].lower()]),'Semantic image filename leak')
        m=audit['row_metadata_PRIVATE'][original['id']];site_counts[m['site']]+=1
        source_bindings.append({'record_id_PRIVATE':row['id'],'original_record_id_PRIVATE':original['id'],'source_index_PRIVATE':e['index'],
            'patient_PRIVATE':e['patient_key'],'level':45,'modality_PRIVATE':q['modality_PRIVATE'],'PNG_sha256':actual,
            'source_volume_sha256':s['source_sha256'],'source_sidecar_sha256':s['sidecar_sha256'],'source_record_PRIVATE':s,
            'pixel_check_PRIVATE':check,'site_PRIVATE':m['site'],'scanner_PRIVATE':m['scanner'],'size':[364,436],'mode':'RGB'})
    require(parent['completed_groups']==8 and parent['optimizer_updates']==4 and parent['absolute_completed_groups']==129 and parent['absolute_optimizer_updates']==60,'Actual parent cursor differs')
    ref40=build_category_schedule(old,mode='uniform',seed=13,max_groups=198)
    require([r['id'] for r in ref40['schedule'][:8]]==parent['binding']['schedule_record_ids'][:8],'Recomputed first8 differs from saved uniform stage')
    require(Counter(r['ability'] for r in ref40['schedule'][:8])=={SEQ:2,DALLAS:6},'Prior sequence offset differs')
    reference=ref40['schedule'][8:40];require(Counter(r['ability'] for r in reference)=={SEQ:18,DALLAS:14},'Next32 task allocation differs')
    new_full=build_category_schedule(new56+dallas,mode='uniform',seed=13,max_groups=198)
    old_stream=[r for r in ref40['schedule'] if r['ability']==SEQ]
    new_stream=[r for r in new_full['schedule'] if r['ability']==SEQ]
    control=replace_sequence_slots(reference,old28,seed=13,sequence_draw_offset=2,
        sampler_sequence_stream=old_stream,sampler_binding=ref40['binding'])
    candidate=replace_sequence_slots(reference,new56,seed=13,sequence_draw_offset=2,
        sampler_sequence_stream=new_stream,sampler_binding=new_full['binding'])
    require(control['binding']['source_sampling_stage']==candidate['binding']['source_sampling_stage'],'Existing sampler algorithm stage differs')
    for arow,brow in zip(control['schedule'],reference):
        require({k:v for k,v in arow.items() if k!='id'}=={k:v for k,v in brow.items() if k!='id'},'Old-pool control fails to restore historical source occurrence')
    cost_source={r['record_sha256']:r for r in measurements['rows']};require(len(cost_source)==76,'Incomplete historical native measurement inventory')
    seq_costs={cost_source[old_row_sha(r)]['input_tokens'] for r in old28};visual={cost_source[old_row_sha(r)]['visual_tokens'] for r in old28}
    require(seq_costs=={213} and visual=={154},'Expected existing native sequence costs differ')
    by_id={r['id']:r for r in old};costs={}
    for name,schedule in [('reference_original_next32',reference),('reference_common_stage_next32',control['schedule']),('candidate_common_stage_next32',candidate['schedule'])]:
        table={r['id']:(213 if r['ability']==SEQ else cost_source[old_row_sha(by_id[r['source_record_id']])]['input_tokens']) for r in schedule}
        require(sum(4*(v+256) for v in table.values())==70784,'Matched finite token charge differs');costs[name]=table
    # CPU tokenizer only. These are text counts, not new native processor observations.
    model=R/'local_data/models/qwen3vl8b-base';tokenizer_path=model/'tokenizer.json'
    require(sha(tokenizer_path)==measurements['processor_artifacts']['tokenizer.json'],'Tokenizer differs from historical native measurement')
    tokenizer=Tokenizer.from_file(str(tokenizer_path));enc=lambda text:tokenizer.encode(text,add_special_tokens=False).ids
    require(len({r['extra_info']['question'] for r in new56+old28})==1,'Question changed')
    counts={'question_tokens':len(enc(new56[0]['extra_info']['question'])),'label_tokens':{label:len(enc(label)) for label in LABELS},
        'fixed_canonical_target_tokens':{label:len(enc(PREFIX+label+SUFFIX)) for label in LABELS},'tokenizer_sha256':sha(tokenizer_path),
        'scope':'Actual CPU tokenizer text counts;expanded input costs separately derived from pinned native baseline and identical dimensions/question.'}
    require(not any(k=='mlx' or k.startswith('mlx.') for k in sys.modules),'MLX import forbidden')
    out.mkdir(parents=True)
    raw_old=paths['original76'].read_bytes().splitlines(keepends=True)
    raw28=b''.join(raw for raw in raw_old if json.loads(raw)['ability']==SEQ);raw_dallas=b''.join(raw for raw in raw_old if json.loads(raw)['ability']==DALLAS)
    require(raw_dallas==paths['Dallas48'].read_bytes(),'Dallas raw JSONL bytes changed')
    (out/'original28_sequence_PRIVATE.verl.jsonl').write_bytes(raw28);(out/'Dallas48_UNCHANGED_PRIVATE.verl.jsonl').write_bytes(raw_dallas)
    shutil.copyfile(paths['original76'],out/'original76_UNCHANGED_PRIVATE.verl.jsonl')
    (out/'paired_z045_sequence56_PRIVATE.verl.jsonl').write_bytes(lines(new56));(out/'mixed104_PRIVATE.verl.jsonl').write_bytes(lines(new56)+raw_dallas)
    save(out/'reference_original_next32_PRIVATE.json',{'schedule':reference,'original_schedule_binding':ref40['binding'],'source_slot_range':[8,40]})
    save(out/'reference_common_stage_next32_PRIVATE.json',control);save(out/'candidate_common_stage_next32_PRIVATE.json',candidate)
    for name,table in costs.items():save(out/(name+'_input_costs_PRIVATE.json'),table)
    save(out/'source_and_pair_membership_PRIVATE.json',source_bindings)
    save(out/'token_counts_SAFE.json',counts)
    binding={'schema':'paired-z045-data-binding-v1','input_paths_PRIVATE':{k:str(v) for k,v in paths.items()},'input_sha256':PINS,
        'parent_absolute_groups':129,'parent_Adam_step':60,'original_slot_range':[8,40],'sequence_draw_offset':2,
        'reference_first8_match_actual_parent':True,'common_existing_sampler_stage':candidate['binding']['source_sampling_stage'],
        'input_cost_provenance':{'method':'Derived baseline-native-equivalent costs conditional on unchanged processor:identical364x436 RGB dimensions and question/system/template;Dallas exact source measurement.',
            'new_native_measurement':False,'sequence_input_tokens':213,'sequence_visual_tokens':154,'measurements_sha256':PINS['native_token_measurements'],
            'processor_sha256':measurements['processor_sha256'],'required_unchanged_processor_artifacts':measurements['processor_artifacts'],
            'token_counts':counts,'expanded_token_ID_equivalence_executed':False},
        'artifact_sha256':{p.name:sha(p) for p in out.iterdir() if p.is_file()},'training_authorized':False}
    save(out/'binding_PRIVATE.json',binding)
    receipt={'schema':'paired-z045-data-v1','status':'TESTED_CPU_PREPARED_NOT_TRAINED','evidence_tier':'OBSERVED','research_tag':'C_TASK_ELIGIBILITY',
        'mission_alignment':{'link':'Prepare a single source-pool intervention for the existing Vero-style sequence/Dallas training comparison.',
            'falsifier':'Source/index/modality/pixel SHA/pair linkage/task slots/Dallas bytes/cursor mismatch aborts.',
            'isolation':'Same28TRAINpeople,samez045,unchangedquestion/targets/Dallas/taskslots/reward;onlysequencepool28to56.',
            'provenance':'Existing pinned TRAIN acquisition/pixel/visual qualification receipts;no new clinical judgments.',
            'shortcut':'Both modalities per patient remove perfect patient-label lookup in this pool;site/protocol/normalization shortcuts remain unexcluded.',
            'resources_decision':'Single-thread CPU preparation;no native/model/update;separate review and future numerical gates required.'},
        'old_sequence_pool':pool_stats(old28),'new_sequence_pool':pool_stats(new56),'mixed_pool_rows':104,'Dallas_rows':48,
        'level':45,'other_levels_in_candidate':0,'source_acquisition_qualification':{'rows':56,'people':28,'paired_opposite_modality_groups':28,
            'source_volumes_previously_verified':56,'current_official_annex_identity_matches_saved_SHA':56,'current_acquisition_sidecar_SHA_matches':56,
            'original_and_opaque_PNG_SHA_matches':56,'RGB_header_dimensions':[364,436],'source_and_render_pixels_recomputed':False,
            'basis':'OfficialREADME post-Gd run02_T1w mapping;FLAIR collection/sidecars;sourceNIfTI/annex/render SHA and paired geometry from existing frozen audit.',
            'paired_shape_affine_orientation_qualified':28,'paired_scanner_model_field_equal':int(scanner_matched),'nonexpert_z045_coverage_adequate_pairs':28,
            'clinical_expert_gate_required_for_acquisition_identity':False,'clinical_diagnoses_created':0,'acquisition_endpoint_is_not_anatomical_grounding':True},
        'source_sites_aggregate_rows':dict(site_counts),'site_scanner_invariant_generalization':'NOT_ESTABLISHED',
        'prompt_leakage':{'question_variants':1,'opaque_semantic_filename_leaks':0,'PNG_metadata_keys':0,'hidden_targets_remain_PRIVATE':True,
            'native_filename_text_exclusion':'Existing native source audit only;no new native token-path execution claimed.'},
        'sampling':{'original_uniform_seed':13,'historical_completed_slots':8,'historical_completed_sequence_slots':2,'next_original_slot_range':[8,40],
            'reference_first8_matches_actual_Adam60':True,'category_slot_counts':{SEQ:18,DALLAS:14},'logical_category_weights':{SEQ:.5,DALLAS:.5},
            'new_pool_proportions_not_used_as_task_weights':True,'sequence_stage':candidate['binding']['source_sampling_stage'],
            'new_dataset_row_stage_not_exact_historical_row_RNG_resume':True,'forced_pair_adjacency':False,'control_stage_common_to_old_and_new':True,
            'old28_control_restores_historical_source_occurrences_exactly_except_top_id':True,
            'new_standalone_sequence_RNG_or_sampler_implementation':False,
            'full_existing_sampler_block_shuffle_reused':True,
            'historical_reference_exposure':exposure_summary(reference),'common_stage_old28_exposure':exposure_summary(control['schedule']),
            'candidate_new56_exposure':exposure_summary(candidate['schedule']),'max_attempted_groups':32,'group_size':4,'max_tokens':256,'matched_token_charge_cap':70784},
        'token_accounting':counts,'expanded_input_costs':{'sequence':213,'Dallas':405,'new_native_measurements':0,
            'status':'DERIVED_FROM_PINNED_NATIVE_BASELINE_SAME_DIMENSIONS_AND_TEXT_CONDITIONAL_ON_UNCHANGED_PROCESSOR','full_new_expanded_token_ID_equality':'NOT_RUN'},
        'Dallas_original_bytes_and_objects_unchanged':True,'original28_and_original76_preserved_exactly':True,
        'artifact_sha256':{p.name:sha(p) for p in out.iterdir() if p.is_file()},'source_sha256':PINS,
        'code_sha256':{'helper':sha(W/'src/mri_rl/paired_z045_data_v1.py'),'builder':sha(__file__),'tests':sha(W/'tests/test_paired_z045_data_v1.py'),
            'existing_category_sampler':sha(W/'src/mri_rl/vero_category_sampler.py')},
        'read_scope':{'TRAIN_source_volumes_or_masks_decoded':0,'PNG_headers_read':56,'PNG_pixels_decoded':0,'new_renders':0,'new_blind_judgments':0,
            'models_loaded':0,'MLX_imports':0,'optimizer_updates':0,'classifier_fits':0,'dev_final100_or_native_model_outputs_read':0,
            'prior_mixed_audit_bytes_read':True,'prior_retrieval_outcomes_not_used':True,
            'preserved_pre_review_builds':1,'session_total_PNG_header_reads_including_preview':112,'session_unique_PNGs':56},
        'scientific_scope':'Acquisition-modality recognition qualified by source metadata;no clinical task or patient-held-out/site generalization result.',
        'training_authorized':False,'native_execution':'NOT_RUN','new_independent_validation':'ROOT_SEPARATE_PROTOCOL_NOT_ESTABLISHED_BY_THIS_BUILDER',
        'cross_model_review':'NOT_VERIFIED'}
    save(public,receipt)
    print(json.dumps({'new56':receipt['new_sequence_pool'],'sampling_exposures':{k:v for k,v in receipt['sampling'].items() if k.endswith('_exposure')},
        'candidate_manifest_sha256':sha(out/'mixed104_PRIVATE.verl.jsonl'),'training_authorized':False},sort_keys=True))


if __name__=='__main__':main()
