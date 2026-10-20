#!/usr/bin/env python3
"""Offline source ontology binding and TRAIN mask endpoint resolution, no model.

Public primary documents are supplied as already downloaded bytes. This builder
has no network operations and never renders or decodes MRI images.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import xml.etree.ElementTree as ET

W=Path(__file__).resolve().parents[1];sys.path.insert(0,str(W/'src'))
from mri_rl.spatial_ontology_resolution_v1 import endpoints,source_labels,amended_row,pointer_lineage

PINS={'priority':'549751aee8bfb1f963f2e2475fc15194cd0fd88178304a48e8fd8a7d72e8f7a2',
 'keys':'0dcdb7ce1280105338d816596353397b6ac864e99f21d30a41a99d5639b56a6e',
 'expert':'d6d1afd84e6fee2d7dda3006ab015d59db2bd73e6c3dd171d9d061ce79c13cae',
 'train':'0efb588b328e3d9af2a339d9237836357a5baff5d9db8b376dc716de3d1f10e3',
 'paper':'52ec470531eb503dc92707de2ab5e276280d773ededa79df7b82229c4547b1a5',
 'tree_v101':'c631539db8d85ea225891610df849443d0a8ebaaf63229216ad43414792a5b6f',
 'tree_v201':'ce5be333d00fd55718e38f22d64c4c46498d897d40f82c7d5f3f7e9c215802cf'}
SOURCE_URLS={'paper':'https://www.ebi.ac.uk/europepmc/webservices/rest/PMC13498554/fullTextXML',
 'tree_v101':'https://api.github.com/repos/OpenNeuroDatasets/ds007045/git/trees/20e149769eaad41c117f51e9be5e95b12bb56b5a?recursive=1',
 'tree_v201':'https://api.github.com/repos/OpenNeuroDatasets/ds007045/git/trees/283007b96977b5b3285bd22cd67b7d97114b1eed?recursive=1'}


def require(ok,msg):
    if not ok:raise ValueError(msg)


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def load_bound(p,k,jsonl=False):
    require(sha(p)==PINS[k],'Source SHA mismatch: '+k)
    return [json.loads(s) for s in p.read_text().splitlines() if s] if jsonl else json.loads(p.read_text())


def save(p,obj):
    with p.open('x') as f:json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')


def rows_save(p,rows):
    with p.open('x') as f:
        for r in rows:f.write(json.dumps(r,sort_keys=True,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--primary-root',type=Path,required=True)
    p.add_argument('--prior-worktree',type=Path,required=True);p.add_argument('--primary-source-dir',type=Path,required=True)
    p.add_argument('--metadata-root',type=Path,required=True);p.add_argument('--destination',type=Path,default=W/'local_data/spatial_ontology_resolution_v1/candidate_v1');a=p.parse_args()
    require(all(os.environ.get(k)=='1' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']),'CPU threads must be1')
    import numpy as np
    import nibabel as nib
    from scipy import ndimage
    R=a.primary_root.resolve();prior=a.prior_worktree.resolve();S=a.primary_source_dir.resolve();out=a.destination.resolve()
    public=W/'results/SPATIAL_ONTOLOGY_RESOLUTION_V1.json'
    require(out.is_relative_to(W/'local_data/spatial_ontology_resolution_v1') and not out.exists() and not public.exists(),'New scoped outputs required')
    B=prior/'local_data/second_task_eligibility_v1/candidate_v1'
    paths={'priority':B/'spatial_priority_review_candidates_PRIVATE.verl.jsonl','keys':B/'expert_second_pass_source_key_PRIVATE.json',
        'expert':B/'expert_MRI_only_adjudication_queue_PRIVATE.jsonl','train':R/'local_data/ds007045_bounded/vero_training_v1/train.verl.jsonl',
        'paper':S/'paper.xml','tree_v101':S/'release_v101_tree.json','tree_v201':S/'release_v201_tree.json'}
    priority=load_bound(paths['priority'],'priority',True);keys=load_bound(paths['keys'],'keys');expert=load_bound(paths['expert'],'expert',True);train=load_bound(paths['train'],'train',True)
    require(len(priority)==15 and len(keys)==84 and len(expert)==84 and len(train)==382,'Frozen coverage differs')
    require(all(r['extra_info']['internal_split']=='train' for r in train+priority),'TRAIN only')
    originals={r['id']:r for r in train};require(len(originals)==len(train),'Duplicate TRAIN record')
    for r in priority:
        o=originals[r['id']];require(all(r[f]==o[f] for f in ['ability','prompt','reward_model','extra_info','images']),'Priority original identity mismatch')
        require(r['ability']=='mri_tumor_grid' and r['training_authorized'] is False,'Priority scope differs')
    require(len({r['id'] for r in priority})==15,'Duplicate priority record')
    require(all(x['expert_adjudication']=='NOT_COMPLETED' and x['training_authorized'] is False and x['expert_fields_to_complete']['annotator_identity'] is None for x in expert),'Expert evidence must remain blank')
    require(sha(paths['paper'])==PINS['paper'],'Primary paper SHA mismatch')
    article=ET.parse(paths['paper']).getroot();paragraphs=[''.join(n.itertext()) for n in article.iter('p')]
    ontology_paragraph=next(t for t in paragraphs if 'Each voxel in the mask file contains an integer label' in t)
    for phrase in ['0 – background/non-tumor tissue','1 – necrotic core','2 – non-enhancing/edematous tumor','3 – contrast-enhancing tumor']:
        require(phrase in ontology_paragraph,'Primary explicit ontology not located')
    require(any('10.18112/openneuro.ds007045.v1.0.1' in t for t in paragraphs),'Paper release differs')
    trees=[]
    for k in ['tree_v101','tree_v201']:
        d=load_bound(paths[k],k);require(d.get('truncated') is False,'Incomplete release inventory')
        trees.append({x['path']:x for x in d['tree']})
    wanted={x['saved_source_key_PRIVATE']['records_PRIVATE'][0].split('_z')[0] for x in keys}
    require(len(wanted)==28,'Expected28 TRAIN people')
    # Directory names only are read to reverse the existing deterministic source adapter key.
    matches={}
    for d in (R/'local_data/ds007045_bounded').glob('sub-*'):
        k=hashlib.sha256(d.name.encode()).hexdigest()[:12]
        if k in wanted:
            require(k not in matches,'Ambiguous source lineage');matches[k]=d
    require(set(matches)==wanted,'TRAIN source crosswalk incomplete')
    priority_ids={r['id'] for r in priority};priority_people={r['extra_info']['patient_key'] for r in priority}
    key_by_patient={k:[] for k in wanted}
    for k in keys:key_by_patient[k['saved_source_key_PRIVATE']['records_PRIVATE'][0].split('_z')[0]].append(k)
    payloads=[];plane_results=[];by_review={};volume_results={};label_counts=Counter()
    for patient,d in sorted(matches.items()):
        rel=f'derivatives/segmentation/{d.name}/{d.name}_MNI152_tumor-mask.nii.gz'
        target=os.readlink(a.metadata_root/rel);lineage=pointer_lineage(rel,target,*trees)
        maskpath=d/f'{d.name}_MNI152_tumor-mask.nii.gz';actual=sha(maskpath)
        require(actual==lineage['source_payload_sha256'] and maskpath.stat().st_size==lineage['source_payload_bytes'],'Actual payload differs from source annex SHA/size')
        group=key_by_patient[patient];require(all(k['saved_source_key_PRIVATE']['mask_sha256']==actual for k in group),'Prior TRAIN packet mask SHA mismatch')
        img=nib.load(maskpath);require(img.shape==(182,218,182) and nib.aff2axcodes(img.affine)==('L','A','S'),'Frozen mask spatial frame differs')
        data=np.asanyarray(img.dataobj);labels=source_labels(data);label_counts.update(labels)
        payloads.append({'patient_PRIVATE':patient,'subject_PRIVATE':d.name,'source_relative_path_PRIVATE':rel,'local_mask_path_PRIVATE':str(maskpath),
            'mask_sha256':actual,'lineage':lineage,'observed_source_numeric_labels':labels,'shape':list(img.shape),'axes':['L','A','S']})
        if patient in priority_people:
            vol,n=ndimage.label(data>0,structure=np.ones((3,3,3),dtype=np.uint8));areas=np.bincount(vol.ravel())[1:]
            volume_results[patient]={'components_26':int(n),'component_voxels_descending':sorted([int(v) for v in areas],reverse=True),
                'clinical_multifocal_status':'NOT_ADJUDICATED;3D mask components are not separate tumor diagnoses'}
            del vol,areas
        for key in group:
            saved=key['saved_source_key_PRIVATE'];rid=saved['records_PRIVATE'][0];level=int(re.search(r'_z(\d+)_',rid)[1])
            slab=np.rot90(data[:,:,level]);union=endpoints(slab>0);enh=endpoints(slab==3)
            require(union['area_native_pixels']==saved['tumor_geometry']['area'] and union['components_8']==saved['tumor_geometry']['components'],'Saved tumor geometry mismatch')
            require(enh['area_native_pixels']==saved['enhancing_geometry']['area'] and enh['components_8']==saved['enhancing_geometry']['components'],'Saved enhancement geometry mismatch')
            require(saved['truth_PRIVATE']['mri_tumor_presence']==('yes' if union['area_native_pixels'] else 'no'),'Presence target differs')
            require(saved['truth_PRIVATE']['mri_enhancing_presence']==('yes' if enh['area_native_pixels'] else 'no'),'Label3 target differs')
            if union['area_native_pixels']:
                require(saved['truth_PRIVATE']['mri_tumor_grid']==union['legacy_grid'],'Original arithmetic target mismatch')
                require(np.allclose(union['prior_pixelcenter_display_index_xy'],saved['tumor_geometry']['centroid_xy_display_pixelcenter'],rtol=0,atol=1e-10),'Prior coordinate arithmetic mismatch')
            result={'review_id':key['review_id'],'patient_PRIVATE':patient,'level_PRIVATE':level,'records_PRIVATE':saved['records_PRIVATE'],
                'original_targets_PRIVATE':saved['truth_PRIVATE'],'source_mask_sha256':actual,'ontology':lineage,
                'union_endpoint_PRIVATE':union,'label3_endpoint_PRIVATE':enh,
                'endpoint_action':'HOLD_CONVENTION_SENSITIVE_GRID' if union['grid']!=union['legacy_grid'] else 'EXPERT_VISIBILITY_AND_ENDPOINT_ALIGNMENT_PENDING',
                'training_authorized':False}
            plane_results.append(result);by_review[key['review_id']]=result
        del data
    require(len(plane_results)==84 and len(by_review)==84,'Duplicate/missing TRAIN planes')
    out.mkdir(parents=True);sources=out/'primary_sources_PRIVATE';sources.mkdir()
    for k in ['paper','tree_v101','tree_v201']:shutil.copyfile(paths[k],sources/paths[k].name)
    amended=[]
    for r in priority:
        e=by_review[r['second_task_eligibility_v1']['review_id_PRIVATE']]
        row=amended_row(r,e['union_endpoint_PRIVATE'],e['ontology'])
        row['spatial_ontology_resolution_v1']['source_volume_geometry_PRIVATE']=volume_results[r['extra_info']['patient_key']]
        amended.append(row)
    candidate=out/'spatial_priority_amended_PRIVATE.verl.jsonl';rows_save(candidate,amended)
    rows_save(out/'expert_MRI_only_adjudication_queue_PRIVATE.jsonl',expert)
    save(out/'expert_second_pass_endpoint_key_PRIVATE.json',plane_results)
    save(out/'mask_source_lineage_PRIVATE.json',payloads)
    save(out/'binding_PRIVATE.json',{'input_paths_PRIVATE':{k:str(v) for k,v in paths.items()},'input_sha256':PINS,
        'primary_source_urls':SOURCE_URLS,'candidate_sha256':sha(candidate),'source_payloads_PRIVATE':payloads})
    positive=[p for p in plane_results if p['union_endpoint_PRIVATE']['area_native_pixels']]
    pe=[x['spatial_ontology_resolution_v1'] for x in amended]
    def metrics(values):
        return {'rows':len(values),'coordinate_convention_changes_grid':sum(e['grid']!=e['legacy_grid'] for e in values),
            'centroid_off_mask':sum(not e['union_centroid_on_mask_nearest_pixel'] for e in values),
            'multicomponent_8':sum(e['components_8']>1 for e in values),
            'component_grid_disagrees_with_union':sum(e['component_grids_disagree_with_union'] for e in values),
            'boundary_distance_lt10_display_pixels':sum(e['boundary_distance_display_pixels']<10 for e in values)}
    report={'schema':'spatial-ontology-resolution-v1','status':'TESTED_CPU_SOURCE_ARITHMETIC_RESOLVED_EXPERT_HOLD','research_tag':'C_TASK_ELIGIBILITY','evidence_tier':'OBSERVED',
        'mission_alignment':{'link':'Unblock an explicitly defined source spatial endpoint for heterogeneous Vero-to-MRI tasks.',
            'falsifier':'Reject unknown labels, source hashes, TRAIN coverage or original arithmetic mismatches.',
            'isolation':'Only versioned source/endpoint qualification; original tasks, targets, prompts and images retained.',
            'provenance':'Pinned primary article, complete public release trees, local annex pointer and actual TRAIN mask hashes.',
            'shortcut':'No new model or fit; reuse existing site/level controls; mask is not visible-focus gold.',
            'resources':'One CPU thread;28 existing TRAIN masks;zero MRI decoding/renders/updates;expert HOLD.'},
        'primary_ontology':{'paper_doi':'10.1038/s41597-026-07953-2','PMCID':'PMC13498554','source_release':'ds007045 v1.0.1;337 people',
            'section':'Data Records / derivatives/segmentation','dictionary':{'0':'background/non-tumor tissue','1':'necrotic core','2':'non-enhancing/edematous tumor','3':'contrast-enhancing tumor'},
            'source_urls':SOURCE_URLS,'source_sha256':{k:PINS[k] for k in SOURCE_URLS},
            'current_release':'ds007045 v2.0.1;363 people','TRAIN_masks_current_pointer_bound':28,
            'TRAIN_masks_byte_identical_annotation_pointer_to_paper_release':sum(x['lineage']['historical_same_annotation_pointer'] for x in payloads),
            'TRAIN_masks_new_to_v201_numeric3_semantics_NOT_VERIFIED':sum(not x['lineage']['historical_same_annotation_pointer'] for x in payloads),
            'priority_rows_paper_release_exact_lineage':sum(e['ontology']['historical_same_annotation_pointer'] for e in pe),
            'priority_rows_new_release_label3_semantics_NOT_VERIFIED':sum(not e['ontology']['historical_same_annotation_pointer'] for e in pe),
            'interpretation':'Numeric3 ontology is directly documented and source-bound for unchanged annotations. A new-release annotation needs its own numeric-label version evidence; this does not itself invalidate whole-mask>0 spatial arithmetic.'},
        'actual_arithmetic':{'TRAIN_planes':84,'TRAIN_people':28,'positive_union_planes':len(positive),'negative_union_planes':84-len(positive),
            'label3_positive_planes':sum(p['label3_endpoint_PRIVATE']['area_native_pixels']>0 for p in plane_results),
            'original_hidden_target_presence_enhancement_grid_reproduced':214,
            'all_positive_union_endpoint_metrics':metrics([p['union_endpoint_PRIVATE'] for p in positive]),
            'priority_endpoint_metrics':metrics([e['endpoint_PRIVATE'] for e in pe]),
            'priority_rows':15,'priority_people':13,'priority_3D_source_masks_with_multiple_26_components':sum(v['components_26']>1 for v in volume_results.values()),
            '3D_component_interpretation':'Source-mask topology only, not clinical multifocal disease. Slice endpoint remains the 2D union, not volume centroid.',
            'original_targets_changed':0,'source_images_or_masks_modified':0,
            'priority_geometric_exclusions':0,'priority_expert_blocked_rows':15,
            'all_grid_rows_convention_sensitive_HOLD':sum(p['union_endpoint_PRIVATE']['grid']!=p['union_endpoint_PRIVATE']['legacy_grid'] for p in positive)},
        'concrete_fix':{'new_endpoint_contract':'Panel-relative image-edge mass centroid:2*(native mean index+.5),half-open thirds,exact boundary right/bottom;store each component and global union separately.',
            'old_conventions':'Legacy used2*mean index;prior qualification used2*mean index+.5 in display pixel-index coordinates. New edge coordinates distinguish physical center from index origin.',
            'candidate_corrections':'Original target/prompt preserved;versioned proposed arithmetic target stored separately;coordinate boundary disagreement cannot silently rewrite original gold.',
            'visible_focus_target_created':False,'components_are_not_visible_lesion_centers':True,
            'retrospective_label_bug_claim':False,
            'convention_sensitive_case_handling':'One nonpriority row is explicitly flagged HOLD_CONVENTION_SENSITIVE_GRID in private second-pass packet;original target retained for endpoint adjudication.'},
        'expert_packet':{'existing_MRI_only_forms_preserved':84,'existing_15_priority_rows_amended':15,'new_expert_annotations':0,'blank_credentials_and_identity':True,
            'second_pass_endpoint_key_separate':True,'new_image_adjudication':0},
        'candidate_sha256':sha(candidate),'private_artifact_sha256':{p.name:sha(p) for p in out.glob('*') if p.is_file()},
        'remaining_blockers':['Expert frame visibility and union-centroid/question alignment for all15 spatial candidates.',
            'Clinical dominant/visible focus centers, multifocal interpretation and postoperative/necrotic etiology remain unadjudicated.',
            'Numeric3 mapping for the two new-release TRAIN annotations remains NOT_VERIFIED; do not mechanically reject whole-mask grid because this enhancement subtype is unresolved.'],
        'Dallas':'ENGINEERING_ONLY;independent brain-only identity answerability evidence still absent;no Dallas payload opened.',
        'training_authorized':False,'scientifically_admitted_rows':0,
        'read_scope':{'TRAIN_mask_files_decoded':28,'priority_3D_topology_files':13,'MRI_images_decoded':0,'new_renders':0,'repeated_blind_reviews':0,
            'models_loaded':0,'optimizer_updates':0,'classifier_fits':0,'final100_or_dev_outputs_read':0,
            'public_release_tree_metadata_contains_all_source_paths':True,'nonTRAIN_MRI_payloads_read':0,
            'rejected_TRAIN_filtered_outbound_pointer_requests_executed':0,
            'approved_public_inventory_fetch':'Complete fixed release-level Git tree URLs;no local selection identifiers in requests;subsequent comparisons entirely offline.'},
        'code_sha256':{'helper':sha(W/'src/mri_rl/spatial_ontology_resolution_v1.py'),'builder':sha(__file__),'tests':sha(W/'tests/test_spatial_ontology_resolution_v1.py')},
        'cross_model_review':'NOT_VERIFIED'}
    save(public,report)
    print(json.dumps({'priority_metrics':report['actual_arithmetic']['priority_endpoint_metrics'],
        'all_positive_metrics':report['actual_arithmetic']['all_positive_union_endpoint_metrics'],
        'candidate_sha256':sha(candidate),'training_authorized':False},sort_keys=True))


if __name__=='__main__':main()
