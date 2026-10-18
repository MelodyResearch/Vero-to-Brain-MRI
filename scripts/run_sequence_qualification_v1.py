#!/usr/bin/env python3
"""Finite TRAIN-only sequence data qualification. No model or optimizer imports."""
import argparse
from collections import Counter,defaultdict
import copy
import hashlib
import json
import os
from pathlib import Path
import secrets
import sys
import re
import ast

W=Path(__file__).resolve().parents[1];sys.path.insert(0,str(W/'src'))
from PIL import Image,ImageDraw,ImageChops
import numpy as np
from mri_rl.sequence_qualification_v1 import blind_pairs,require,source_sequence_render,lopo_nearest_neighbor,unblind_judgments

CANDIDATE_SHA='e33259ef98ecdb4e8931edcf8648d2044cbc80cecbee15554551e65d5234e258'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def jsonl(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def save_rows(path,rows):
    raw=''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows).encode()
    if path.exists():require(path.read_bytes()==raw,'Immutable private manifest changed');return
    with path.open('xb') as f:f.write(raw)


def blind(candidate,out):
    require(sha(candidate)==CANDIDATE_SHA,'Frozen train candidate hash differs')
    require(not out.exists(),'New private destination required');out.mkdir(parents=True)
    rows=[r for r in jsonl(candidate) if r['ability']=='mri_sequence'];require(len(rows)==168,'Exact168 TRAIN sequence images required')
    seed=secrets.randbits(256);pairs=blind_pairs(rows,seed);require(len(pairs)==84,'Exact84 matched TRAIN pairs required')
    (out/'blind_pairs').mkdir();(out/'opaque_assets').mkdir();(out/'blind_contact_sheets').mkdir()
    key=[];future=[];pair_paths=[]
    for item in pairs:
        case=item['case_id'];entry={'case_id':case,'sides_PRIVATE':{}};canvas=Image.new('RGB',(752,466),'black');draw=ImageDraw.Draw(canvas)
        draw.text((260,5),case,fill='white');draw.text((8,5),'L',fill='white');draw.text((388,5),'R',fill='white')
        for side,x in [('left',4),('right',384)]:
            row=item[side];path=Path(row['images'][0]);require(sha(path)==row['rc_data_v1']['image_sha256'],'Frozen TRAIN PNG changed')
            with Image.open(path) as image:
                require(image.size==(364,436),'Original display dimensions changed');pixels=image.convert('RGB')
                opaque=out/'opaque_assets'/f'{case}_{side}.png';pixels.save(opaque)
                with Image.open(opaque) as saved:require(ImageChops.difference(pixels,saved).getbbox() is None,'Opaque copy pixel mismatch')
                canvas.paste(pixels,(x,26))
                entry['sides_PRIVATE'][side]={'source_row_PRIVATE':row,'source_png_sha256':sha(path),'opaque_png_sha256':sha(opaque),'source_png_metadata_keys':sorted(image.info)}
            f=copy.deepcopy(row);f['id']=f'{case}_{side}';f['images']=[str(opaque.resolve())]
            f['prompt'][0]['content'][0]['image']=f['images'][0]
            f['sequence_qualification_v1']={'training_authorized':False,'independent_visual_review':'PENDING','source_pixel_equivalence':True}
            future.append(f)
        pair_path=out/'blind_pairs'/f'{case}.png';canvas.save(pair_path);pair_paths.append(pair_path);key.append(entry)
    sheets=[]
    for page,start in enumerate(range(0,len(pair_paths),8),1):
        selection=pair_paths[start:start+8];canvas=Image.new('RGB',(1016,4*332),'#202020')
        for i,path in enumerate(selection):
            with Image.open(path) as image:
                image.thumbnail((500,316));x=(i%2)*508;y=(i//2)*332
                canvas.paste(image,(x,y));ImageDraw.Draw(canvas).text((x+2,y+316),path.stem,fill='white')
        path=out/'blind_contact_sheets'/f'sheet_{page:02}.png';canvas.save(path);sheets.append(str(path.resolve()))
    save(out/'blind_key_PRIVATE.json',{'seed_PRIVATE':str(seed),'candidate_sha256':CANDIDATE_SHA,'pairs_PRIVATE':key})
    with (out/'future_loader_manifest_PRIVATE.jsonl').open('x') as f:
        for r in future:f.write(json.dumps(r,sort_keys=True)+'\n')
    save(out/'blind_manifest_SAFE.json',{'pairs':84,'source_images':168,'sheets':11,'pairs_per_sheet_max':8,'displayed_source_panel_approx_pixels':[242,290],'sheet_sha256':{Path(p).name:sha(p) for p in sheets},
        'randomization':'Saved private256bit seed; shuffled patient-level pairs and independently shuffled left/right modalities',
        'case_id':'Random64bit nonsemantic token; no source hash, patient, modality, level or label in displayed text',
        'training_authorized':False,'independent_visual_review':'PENDING','blind_key_sha256':sha(out/'blind_key_PRIVATE.json'),
        'future_loader_manifest_sha256':sha(out/'future_loader_manifest_PRIVATE.jsonl')})
    print(json.dumps({'stage':'BLIND_READY','pairs':84,'sheet_paths_PRIVATE':sheets,'training_authorized':False},indent=2),flush=True)


def audit(candidate,root,out):
    """Decode only the source volumes whose deterministic token is in TRAIN168."""
    import nibabel as nib
    require(sha(candidate)==CANDIDATE_SHA,'Frozen TRAIN candidate differs')
    require((out/'blind_key_PRIVATE.json').exists() and not (out/'audit_PRIVATE.json').exists(),'Existing blind pack and new audit required')
    rows=sorted((r for r in jsonl(candidate) if r['ability']=='mri_sequence'),key=lambda r:r['id'])
    require(len(rows)==168 and all(r['extra_info']['internal_split']=='train' for r in rows),'Exact TRAIN168 required')
    patients={r['extra_info']['patient_key'] for r in rows};require(len(patients)==28,'Exact28 TRAIN people required')
    work=root.parent.parent/'work';metadata=work/'ds007045_metadata';renderer=work/'ds007045_prompts.py'
    renderer_text=renderer.read_text();require('token=hashlib.sha256(id.encode()).hexdigest()[:12]' in renderer_text,'Original deterministic patient-token mapping changed')
    # Directory names only: source images/sidecars are opened solely after the
    # full hash-derived source key is proven to be in the immutable TRAIN pool.
    mapping={}
    for directory in metadata.glob('sub-*'):
        key=hashlib.sha256(directory.name.encode()).hexdigest()[:12]
        if key in patients:
            require(key not in mapping,'Ambiguous source patient hash');mapping[key]=directory.name
    require(set(mapping)==patients,'TRAIN source identity mapping incomplete')
    inventory_path=root/'local_data/ds007045_bounded/c_recovery_v1/sequence_inventory_PRIVATE.json'
    inventory=json.loads(inventory_path.read_text())['rows_PRIVATE'];site={r['patient']:r['site'] for r in inventory}
    require(set(site)==patients,'TRAIN site provenance differs')
    source_records=[];row_metadata={};arrays={};pixel_checks=[];upstream_bytes=0
    reference_by_group=defaultdict(list)
    for r in rows:
        m=re.search(r'_z(045|091|136)_(flair|t1ce)_sequence$',r['id']);require(m is not None,'Source modality/level ID missing')
        reference_by_group[r['extra_info']['patient_key'],m[2]].append((int(m[1]),r))
    for (person,modality),group in sorted(reference_by_group.items()):
        subject=mapping[person];name=f'{subject}_{modality}_MNI152_brain_n4.nii.gz'
        pointer=metadata/'derivatives/processing_MNI152_skull-stripped_N4/processing_MNI152_skull-stripped_N4'/subject/name
        require(pointer.is_symlink(),'Source git-annex pointer missing')
        link=os.readlink(pointer);match=re.search(r'SHA256E-s(\d+)--([0-9a-f]{64})',link);require(match is not None,'Source SHA/size pointer malformed')
        raw=root/'local_data/ds007045_bounded'/subject/name
        require(raw.stat().st_size==int(match[1]) and sha(raw)==match[2],'TRAIN source raw bytes differ from official annex identity')
        side_name=subject+('_FLAIR.json' if modality=='flair' else '_run-02_T1w.json')
        sidepath=metadata/subject/'anat'/side_name;side=json.loads(sidepath.read_text())
        scanner={'manufacturer':side.get('Manufacturer'),'model':side.get('ManufacturersModelName'),'field_T':side.get('MagneticFieldStrength')}
        scanner_key=json.dumps(scanner,sort_keys=True)
        image=nib.load(str(raw),mmap=True)
        require(image.shape==(182,218,182),'Unexpected source volume grid')
        orientation=''.join(nib.aff2axcodes(image.affine));spacing=list(map(float,nib.affines.voxel_sizes(image.affine)))
        arrays[person,modality]={'affine':image.affine.tolist(),'shape':list(image.shape),'orientation':orientation}
        data=np.asanyarray(image.dataobj);require(data.nbytes<=128*1024*1024 and np.isfinite(data).all(),'Source array cap/nonfinite failure')
        window=None
        for level,row in sorted(group):
            rendered,window=source_sequence_render(data,level)
            with Image.open(row['images'][0]) as existing:
                require(existing.size==(364,436) and existing.mode=='RGB','PNG geometry/mode changed')
                delta=np.asarray(rendered,dtype=np.int16)-np.asarray(existing,dtype=np.int16)
                check={'record_id_PRIVATE':row['id'],'source_patient_PRIVATE':subject,'modality_PRIVATE':modality,'level':level,
                    'source_sha256':match[2],'png_sha256':sha(row['images'][0]),'pixel_exact':bool(np.all(delta==0)),
                    'max_absolute_pixel_difference':int(np.abs(delta).max()),'differing_channel_values':int(np.count_nonzero(delta))}
                pixel_checks.append(check)
                row_metadata[row['id']]={'patient':person,'site':site[person],'scanner':scanner_key,'level':level,'label':row['reward_model']['ground_truth'],
                    'manufacturer':side.get('Manufacturer'),'model':side.get('ManufacturersModelName'),'field_T':side.get('MagneticFieldStrength'),
                    'TR_s':side.get('RepetitionTime'),'TE_s':side.get('EchoTime'),'inversion_s':side.get('InversionTime'),
                    'protocol_PRIVATE':side.get('ProtocolName'),'series_description_PRIVATE':side.get('SeriesDescription')}
        source_records.append({'patient_PRIVATE':person,'subject_PRIVATE':subject,'modality':modality,'source_path_PRIVATE':str(raw),
            'official_annex_pointer_PRIVATE':str(pointer),'source_sha256':match[2],'source_bytes':int(match[1]),
            'sidecar_sha256':sha(sidepath),'sidecar_path_PRIVATE':str(sidepath),'orientation':orientation,'spacing_mm':spacing,
            'shape':list(image.shape),'dtype':str(data.dtype),'normalization_window':window,'source_metadata':scanner})
        upstream_bytes+=raw.stat().st_size;del data,image
    require(len(source_records)==56 and len(pixel_checks)==168,'Source/render coverage incomplete')
    paired_geometry=[]
    for person in sorted(patients):
        a=arrays[person,'flair'];b=arrays[person,'t1ce']
        paired_geometry.append({'patient_PRIVATE':person,'shape_equal':a['shape']==b['shape'],
            'affine_equal_atol_1e_6':bool(np.allclose(a['affine'],b['affine'],rtol=0,atol=1e-6)),
            'orientation_equal':a['orientation']==b['orientation']})
    # Predefined feature rules; no fitted means/scales, heads, optimization,
    # hyperparameter search or held-out dataset reads.
    hists=[];intensities=[];labels=[];persons=[]
    for row in rows:
        with Image.open(row['images'][0]) as image:
            a=np.asarray(image.convert('L'),dtype=np.float64);h=np.histogram(a,bins=16,range=(0,256))[0].astype(float)
            q=np.percentile(a,[10,50,90])/255
            intensities.append([float(a.mean()/255),float(a.std()/255),float(np.mean(a>0)),*q.tolist()]);hists.append((h/h.sum()).tolist())
        labels.append(int(row['reward_model']['ground_truth']=='contrast-enhanced T1'));persons.append(row['extra_info']['patient_key'])
    diagnostics={name:lopo_nearest_neighbor(x,labels,persons) for name,x in [('normalized16bin_histogram',hists),('fixed01_intensity_summary',intensities)]}
    for result in diagnostics.values():
        result['label_map']={'0':'FLAIR','1':'contrast-enhanced T1'}
        predictions=result['prediction_PRIVATE']
        result['level_counts']={str(level):{'rows':sum(row_metadata[r['id']]['level']==level for r in rows),
            'correct':sum(row_metadata[r['id']]['level']==level and predictions[i]==labels[i] for i,r in enumerate(rows))} for level in (45,91,136)}
        result['site_counts']={s:{'rows':sum(row_metadata[r['id']]['site']==s for r in rows),
            'correct':sum(row_metadata[r['id']]['site']==s and predictions[i]==labels[i] for i,r in enumerate(rows))} for s in sorted(set(site.values()))}
    cross={}
    for field in ('patient','site','scanner','level'):
        groups=defaultdict(Counter)
        for m in row_metadata.values():groups[str(m[field])][m['label']]+=1
        cross[field]={key:dict(value) for key,value in sorted(groups.items())}
    require(len(set(r['extra_info']['question'] for r in rows))==1,'Question varies with hidden source labels')
    prompt_question=rows[0]['extra_info']['question']
    canonical_prompt_errors=0
    for r in rows:
        expected=[{'role':'user','content':[{'type':'image','image':r['images'][0]},{'type':'text','text':prompt_question}]}]
        canonical_prompt_errors+=int(r['prompt']!=expected)
    # AST source inspection only: actual model/tokenizer loading is forbidden.
    native=W/'src/mri_rl/mlx_continuous_gspo.py';native_text=native.read_text();tree=ast.parse(native_text)
    rollout=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='rollout')
    messages=next(n.value for n in rollout.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='messages' for t in n.targets))
    msg_source=ast.unparse(messages)
    require("row['extra_info']['question']" in msg_source and "row['images']" not in msg_source and 'ground_truth' not in msg_source and "row['id']" not in msg_source,'Native message construction changed')
    utils=work/'mlx-venv/lib/python3.13/site-packages/mlx_vlm/utils.py';utils_text=utils.read_text()
    utils_ast=ast.parse(utils_text);load_image=next(n for n in utils_ast.body if isinstance(n,ast.FunctionDef) and n.name=='load_image')
    source_load_image=ast.unparse(load_image)
    require('Image.open(image_source)' in source_load_image and '.convert(\'RGB\')' in source_load_image,'Installed image loader source changed')
    private={'candidate_sha256':CANDIDATE_SHA,'source_records_PRIVATE':source_records,'pixel_checks_PRIVATE':pixel_checks,
        'paired_geometry_PRIVATE':paired_geometry,'row_metadata_PRIVATE':row_metadata,'cross_tables_PRIVATE':cross,
        'features_PRIVATE':{'row_ids':[r['id'] for r in rows],'hist16':hists,'intensity6':intensities},'TRAIN_internal_retrieval_PRIVATE':diagnostics,
        'read_ledger':{'candidate_rows':168,'TRAIN_NIfTI_sources':56,'TRAIN_sidecars':56,'TRAIN_source_bytes_hashed':upstream_bytes,
            'metadata_subject_directory_names_hashed_only':True,'mixed_source_manifest_files_read':False,'mask_files_read':0,
            'nontrain_image_sidecar_QC_or_output_files_read':0,'model_optimizer_or_checkpoint_files_read':0}}
    save(out/'audit_PRIVATE.json',private)
    publish_report(candidate,root,out)


def unblind(out):
    judgments_path=out.parent/'blind_root_judgments_BEFORE_UNBLIND_PRIVATE.json'
    expected='382a5379f42d4d4e5d8fc62cfdead6ea64c51c8300c621bc4b632087ad79be56'
    require(sha(judgments_path)==expected,'Frozen pre-unblind judgments changed')
    judgments=json.loads(judgments_path.read_text());require(judgments['blind_key_opened_before_commit'] is False,'Review was not blinded')
    keypath=out/'blind_key_PRIVATE.json';key=json.loads(keypath.read_text());safe=json.loads((out/'blind_manifest_SAFE.json').read_text())
    require(sha(keypath)==safe['blind_key_sha256'],'Saved blind key changed')
    result=unblind_judgments(key['pairs_PRIVATE'],judgments['judgments'])
    detail={r['case_id']:r for r in result['pair_details_PRIVATE']}
    manifest=jsonl(out/'future_loader_manifest_PRIVATE.jsonl')
    for row in manifest:
        case,side=row['id'].rsplit('_',1);d=detail[case]
        row['sequence_qualification_v1'].update({'training_authorized':False,
            'independent_visual_review':'VISUAL_NONEXPERT_REVIEW_COMPLETE_EXPERT_PENDING',
            'nonexpert_coverage':d['coverage'],'nonexpert_confidence':d['confidence'],
            'nonexpert_technical_coverage_adequate':d['eligible_visual_review'],
            'blind_review_abstained':not d['answered'],'expert_review':'PENDING',
            'frozen_blind_review_sha256':expected})
    reviewed=out/'future_loader_manifest_reviewed_PRIVATE.jsonl'
    save_rows(reviewed,manifest)
    partitions={True:out/'future_nonexpert_coverage_candidate_PRIVATE.jsonl',False:out/'future_visual_abstention_PRIVATE.jsonl'}
    for adequate,path in partitions.items():
        save_rows(path,[r for r in manifest if r['sequence_qualification_v1']['nonexpert_technical_coverage_adequate']==adequate])
    result.update({'frozen_blind_judgments_sha256':expected,'blind_key_sha256':sha(keypath),
        'reviewed_manifest_sha256':sha(reviewed),'source_png_metadata_key_counts':dict(Counter(k for p in key['pairs_PRIVATE'] for s in p['sides_PRIVATE'].values() for k in s['source_png_metadata_keys'])),
        'coverage_candidate_manifest_sha256':sha(partitions[True]),'visual_abstention_manifest_sha256':sha(partitions[False]),
        'reviewer_credentials':judgments['expert_credentials'],'training_authorized':False})
    save(out/'unblind_PRIVATE.json',result)
    print(json.dumps({k:v for k,v in result.items() if not k.endswith('_PRIVATE')},indent=2),flush=True)


def publish_report(candidate,root,out):
    """Publish already saved CPU audit receipts, without reopening NIfTI arrays."""
    require(sha(candidate)==CANDIDATE_SHA,'Frozen TRAIN candidate differs')
    private=json.loads((out/'audit_PRIVATE.json').read_text());require(private['candidate_sha256']==CANDIDATE_SHA,'Saved audit binding differs')
    rows=sorted((r for r in jsonl(candidate) if r['ability']=='mri_sequence'),key=lambda r:r['id'])
    source_records=private['source_records_PRIVATE'];pixel_checks=private['pixel_checks_PRIVATE'];paired_geometry=private['paired_geometry_PRIVATE']
    diagnostics=private['TRAIN_internal_retrieval_PRIVATE'];cross=private['cross_tables_PRIVATE']
    work=root.parent.parent/'work';metadata=work/'ds007045_metadata';renderer=work/'ds007045_prompts.py'
    native=W/'src/mri_rl/mlx_continuous_gspo.py';utils=work/'mlx-venv/lib/python3.13/site-packages/mlx_vlm/utils.py'
    # Additional cross-checks over saved TRAIN metadata; no arrays reopened.
    for m in private['row_metadata_PRIVATE'].values():
        subject=next(r['subject_PRIVATE'] for r in source_records if r['patient_PRIVATE']==m['patient'])
        require(re.fullmatch(r'sub-([A-Z]+)\d+',subject)[1]==m['site'],'Recorded source site conflicts with official BIDS site code')
    per_person_scanners={p:{m['scanner'] for m in private['row_metadata_PRIVATE'].values() if m['patient']==p} for p in {m['patient'] for m in private['row_metadata_PRIVATE'].values()}}
    prompt_question=rows[0]['extra_info']['question']
    canonical_prompt_errors=sum(r['prompt']!=[{'role':'user','content':[{'type':'image','image':r['images'][0]},{'type':'text','text':prompt_question}]}] for r in rows)
    clean_diag={name:{k:v for k,v in result.items() if not k.endswith('_PRIVATE')} for name,result in diagnostics.items()}
    for result in clean_diag.values():result['label_map']={'0':'FLAIR','1':'contrast-enhanced T1'}
    report={'schema':'sequence-qualification-v1','status':'ACTUAL_CPU_SOURCE_PIXEL_AND_TRAIN_RETRIEVAL_AUDIT_COMPLETE','evidence_tier':'OBSERVED_CPU',
        'research_tag':'C_TASK_ELIGIBILITY','mission_alignment':{'main_question':'Qualify existing genuine MRI sequence inputs before future Vero transfer comparisons.',
            'claim_falsifier':'Every claimed TRAIN source/patient/modality/level must hash-bind to official annex identity and reproduce PNG pixels exactly; any mismatch remains a blocker.',
            'change_isolation':'Existing168 pixels unchanged; blind copies remove file-name semantics. Fixed TRAIN-only1NN feature diagnostics add no neural or optimizer training.',
            'provenance':'Exact candidate/source/code/annex/sidecar/render hashes retained privately; public aggregate hashes only.',
            'shortcut':'Fixed histogram and intensity retrieval exclude the complete query person; site/scanner/level contingencies are recorded separately.',
            'decision':'One-thread finite CPU audit; external independent visual/native-path review and researcher approval remain required.'},
        'source_audit':{'people':28,'volumes':56,'images':168,'pairs':84,'TRAIN_source_sha256_matches_official_annex':56,
            'exact_pixel_matches':sum(r['pixel_exact'] for r in pixel_checks),'mismatched_images':sum(not r['pixel_exact'] for r in pixel_checks),
            'max_absolute_pixel_difference':max(r['max_absolute_pixel_difference'] for r in pixel_checks),
            'same_patient_paired_shape_and_affine':sum(r['shape_equal'] and r['affine_equal_atol_1e_6'] for r in paired_geometry),
            'source_orientations':dict(Counter(r['orientation'] for r in source_records)),
            'people_with_same_scanner_model_field_across_modalities':sum(len(v)==1 for v in per_person_scanners.values()),
            'render':'SourceLAS182×218×182 -> native z045/091/136 ->np.rot90 -> volume nonzero[::16]0.5/99.5 percentile window ->round/clamp uint8 ->2x bilinearRGB364×436; no new crop/canonicalization/mask overlay.',
            'display_axes':'For LAS source: displayed right follows source axis0 toward anatomical left; displayed down follows reversed axis1 toward posterior. Upstream MNI/skull stripping is an existing derivative, not this renderer.',
            'source_modality_basis':'Pinned official README maps run-02_T1w to post-GdT1; BIDS_FLAIR sidecars and derivative flair/t1ce paths agree with original renderer mapping.'},
        'blind_review':{'pairs':84,'sheets':11,'max_pairs_per_sheet':8,'case_ids_semantic':False,'source_filenames_or_targets_displayed':False,
            'key_separate_PRIVATE':True,'review':'PENDING_EXTERNAL_RECORD','blind_manifest_sha256':sha(out/'blind_manifest_SAFE.json')},
        'future_loader':{'rows':168,'opaque_image_paths':True,'training_authorized':False,'external_review_required':True,
            'manifest_sha256':sha(out/'future_loader_manifest_PRIVATE.jsonl')},
        'prompt_and_filename_audit':{'source_filenames_contain_modality_truth':168,'constant_question_variants':1,'canonical_prompt_mismatches':canonical_prompt_errors,
            'filename_target_lookup_if_exposed_as_text':{'rows':168,'correct':sum(('flair' in Path(r['images'][0]).name)==(r['reward_model']['ground_truth']=='FLAIR') for r in rows),
                'interpretation':'Potential direct label exposure if filenames were serialized as model text; inspected native message path does not do so.'},
            'PNG_embedded_text_metadata':'To be cross-checked against separate blind key; opaque copies were re-encoded without PNG metadata',
            'native_GSPO_message_AST_only_question_and_system':True,'native_model_input_filename_as_text':'No evidence in inspected source: filenames passed only as image loader paths; question is constant across targets.',
            'hidden_reward_targets_and_patient_keys':'Present in PRIVATE training record; excluded from inspected native message construction.',
            'installed_image_loader':'PILImage.open(path), EXIF transpose, convertRGB; image path text is not converted into image pixels.',
            'runtime_token_equivalence':'NOT_RUN here; static source audit is not executed native tokenizer proof',
            'legitimate_contrast':'FLAIR vs post-GdT1 intensity/CSF contrast is task-relevant signal; histogram success alone neither proves leaked metadata nor anatomical reasoning.'},
        'TRAIN_internal_leave_person_out_retrieval':clean_diag,'cross_tables_aggregate':{k:v for k,v in cross.items() if k!='patient'},
        'patient_cross_table_aggregate':{'patients':28,'rows_per_patient':6,'each_patient_modality_counts':{'FLAIR':3,'contrast-enhanced T1':3}},
        'categorical_metadata_resubstitution_majority_ceiling':{field:sum(max(counts.values()) for counts in groups.values())/168 for field,groups in cross.items()},
        'read_scope':private['read_ledger'],'new_neural_model_loads':0,'optimizer_updates':0,'fitted_classifier_parameters':0,
        'limitations':['TRAIN-internal retrieval excludes same-person examples but reuses the TRAIN cohort and source protocols; no untouched/dev/final or independent-generalization claim.',
            'Intensity/histogram association may reflect genuine sequence contrast, normalization, site/protocol or other shortcuts. Retrieval alone cannot distinguish them.',
            'Automatic source/pixel checks and blinded nonexpert review do not establish clinical validity or authorize training. Sparse high-slice images need explicit visual qualification.',
            'Full affine/header and actual pixels were checked only for the28 TRAIN people. Cross-source global identity and scanner-independent generalization remain unverified.'],
        'source_sha256':{'helper':sha(W/'src/mri_rl/sequence_qualification_v1.py'),'runner':sha(__file__),
            'tests':sha(W/'tests/test_sequence_qualification_v1.py'),'original_renderer':sha(renderer),
            'official_README':sha(metadata/'README.txt'),'native_GSPO':sha(native),'installed_image_utils':sha(utils),
            'private_audit':sha(out/'audit_PRIVATE.json')},'training_authorized':False,'cross_model_review':'NOT_VERIFIED'}
    if (out/'unblind_PRIVATE.json').exists():
        unblind_result=json.loads((out/'unblind_PRIVATE.json').read_text())
        report['blind_review'].update({k:v for k,v in unblind_result.items() if not k.endswith('_PRIVATE')})
        report['blind_review']['review']='VISUAL_NONEXPERT_REVIEW_COMPLETE_EXPERT_PENDING'
        report['future_loader'].update({'manifest_sha256':unblind_result['reviewed_manifest_sha256'],
            'nonexpert_coverage_adequate_rows':2*(84-unblind_result['abstained_pairs']),
            'nonexpert_coverage_abstained_rows':2*unblind_result['abstained_pairs'],'expert_review':'PENDING'})
        for name in ('coverage_candidate_manifest_sha256','visual_abstention_manifest_sha256'):
            if name in unblind_result:report['future_loader'][name]=unblind_result[name]
        report['prompt_and_filename_audit']['PNG_embedded_text_metadata']={'source_PNG_metadata_key_counts':unblind_result['source_png_metadata_key_counts'],
            'opaque_assets_reencoded_without_metadata':True}
    save(W/'results/SEQUENCE_QUALIFICATION_V1.json',report)
    print(json.dumps({'stage':'AUDIT_COMPLETE','source_volumes':56,'exact_pixel_matches':report['source_audit']['exact_pixel_matches'],
        'retrieval':{name:{'correct':r['correct'],'rows':r['rows']} for name,r in diagnostics.items()},'training_authorized':False},indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['blind','audit','unblind','report']);p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--primary-root',type=Path,required=True);p.add_argument('--destination',type=Path,required=True);a=p.parse_args()
    require(all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS')),'One-thread caps required')
    out=a.destination.resolve();require(out.is_relative_to(W/'local_data/sequence_qualification_v1'),'Private scoped destination required')
    if a.stage=='blind':blind(a.candidate.resolve(),out)
    elif a.stage=='audit':audit(a.candidate.resolve(),a.primary_root.resolve(),out)
    elif a.stage=='unblind':unblind(out)
    else:publish_report(a.candidate.resolve(),a.primary_root.resolve(),out)
if __name__=='__main__':main()
