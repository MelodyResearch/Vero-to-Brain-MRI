"""Frozen source-specific final100 loader; no data is opened at import/factory time.

Only the returned closure, after the full externally reviewed22-checkpoint gate,
may read sealed registries/targets and render their complete100fixed cases. This
adds no registration, crop, label choice or post-outcome case selection. Sequence
uses ONLY the two pure functions extracted from the hash-bound original renderer;
Dallas reuses the existing independent-canonical whole-head multiview helper.
"""
from __future__ import annotations
import ast
from collections import Counter,defaultdict
import hashlib
import io
import json
from pathlib import Path
import re

from mri_rl.final_test_release import validate_release, _test_counts
from mri_rl.two_task_pool import require,sha,write_exact

PRIVATE_ROOT=Path(__file__).resolve().parents[2]/'local_data'


def load_json(path):return json.loads(Path(path).read_text())


def sequence_functions(source):
    """Execute only normalize/image_at definitions; never original top-level loops."""
    import numpy as np
    from PIL import Image
    tree=ast.parse(Path(source).read_text())
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'normalize','image_at'}]
    require(len(functions)==2 and {n.name for n in functions}=={'normalize','image_at'},'Original pure sequence renderer definitions missing')
    namespace={'np':np,'Image':Image}
    exec(compile(ast.Module(body=functions,type_ignores=[]),'<frozen-sequence-renderer-pure-functions>','exec'),namespace)
    return namespace['normalize'],namespace['image_at']


def validate_sequence_registry(registry,question):
    files=registry['files_PRIVATE'];cases=registry['cases_PRIVATE']
    require(len(files)==28 and len(cases)==84,'Complete frozen28-volume/84case sequence registry required')
    by_sha={f['sha256']:f for f in files}
    require(len(by_sha)==28 and all(re.fullmatch('[0-9a-f]{64}',h) for h in by_sha),'Duplicate/malformed sequence source hashes')
    require(len({c['record_id'] for c in cases})==84,'Duplicate sequence case IDs')
    person_to_key={};key_to_person={};groups=defaultdict(list);used=set()
    for c in cases:
        require(c['source']=='OpenNeuro_ds007045' and c['route']=='mri_sequence' and c['split']=='sealed_test','Frozen source/route/split mismatch')
        require(c['source_sha256'] in by_sha and c['pixels_rendered'] is False and c['question']==question,'Changed source/render/question recipe')
        f=by_sha[c['source_sha256']];person=f['subject_PRIVATE'];key=c['patient_key']
        require(c['source_bytes']==f['bytes'] and c['ground_truth_PRIVATE']==f['modality_target_PRIVATE']
                and c['ground_truth_PRIVATE'] in {'FLAIR','contrast-enhanced T1'} and c['slice_z'] in (45,91,136),'Frozen acquisition/geometry target differs')
        require(re.fullmatch('[0-9a-f]{24}',key) and re.fullmatch('[0-9a-f]{24}',c['record_id']),'Opaque case/patient keys required')
        require(person not in person_to_key or person_to_key[person]==key,'Source person maps to multiple patient keys')
        require(key not in key_to_person or key_to_person[key]==person,'Different source people share a patient key')
        person_to_key[person]=key;key_to_person[key]=person;groups[key].append(c);used.add(c['source_sha256'])
    require(len(person_to_key)==14 and used==set(by_sha),'Frozen sequence person/source inventory differs')
    for group in groups.values():
        require(len(group)==6 and {(c['ground_truth_PRIVATE'],c['slice_z']) for c in group}=={(t,z) for t in ('FLAIR','contrast-enhanced T1') for z in (45,91,136)},'Patient modality/slice recipe incomplete')
        require(len({c['source_sha256'] for c in group})==2,'Exactly two source acquisitions per sequence person required')
    return by_sha


def sequence_record(case,image,registry_sha,renderer_sha):
    question=case['question'];image=str(Path(image).resolve())
    return {'id':case['record_id'],'data_source':'OpenNeuro_ds007045','ability':'mri_sequence','images':[image],
        'prompt':[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':question}]}],
        'reward_model':{'style':'rule','ground_truth':case['ground_truth_PRIVATE']},
        'extra_info':{'question':question,'reward_type':'mri_sequence','task_type':'mri_sequence','internal_split':'test',
            'patient_lineage':'BIDS_SUBJECT_VERIFIED','patient_key':case['patient_key'],'source_version':'OpenNeuro ds007045 v2.0.1',
            'source_volume_sha256':case['source_sha256'],'slice_z':case['slice_z'],'frozen_registry_sha256':registry_sha,
            'original_renderer_source_sha256':renderer_sha,'target_provenance':'Frozen source-acquisition modality identity; not tumor mask or clinical diagnosis'}}


def expected_dallas_registry(cases):
    from mri_rl.visit_correspondence import token
    return [{'case_id':c['case_id'],'split':'test','cluster_key':c['cluster_key'],
        'query_patient_key':token('patient',c['query_subject']),'candidate_patient_key':token('patient',c['candidate_subject']),
        'query_source_sha256':c['query']['sha256'],'candidate_source_sha256':c['candidate']['sha256'],
        'source_equality_target':c['target'],'pixels_rendered':False} for c in sorted(cases,key=lambda c:c['case_id'])]


def validate_dallas_registry(registry,cases):
    require(len(cases)==16 and len({c['query_subject'] for c in cases})==8 and len({c['cluster_key'] for c in cases})==4,'Frozen Dallas16/8people/4cycle inventory differs')
    require(len(registry)==16 and sorted(registry,key=lambda c:c['case_id'])==expected_dallas_registry(cases),'Frozen Dallas source/equality/cycle registry differs')


def _private(path):
    path=Path(path).resolve();require(path.is_relative_to(PRIVATE_ROOT.resolve()),'Rendered MRI/reference output must stay under ignored local_data');return path


def _png(path,image):
    buffer=io.BytesIO();image.save(buffer,format='PNG');write_exact(_private(path),buffer.getvalue())


def _render_sequence(registry,by_sha,source,registry_sha,renderer_sha):
    # Imports/arrays happen only after the outer full gate and registry validation.
    import nibabel as nib
    import numpy as np
    normalize,image_at=sequence_functions(source);cases=registry['cases_PRIVATE'];rows=[]
    by_volume=defaultdict(list)
    for case in cases:by_volume[case['source_sha256']].append(case)
    for digest,file in sorted(by_sha.items()):
        path=_private(file['local_path_PRIVATE'])
        require(path.stat().st_size==file['bytes'] and sha(path)==digest,'Frozen sequence payload bytes changed')
        image=nib.load(str(path));require(image.shape==(182,218,182),'Frozen sequence stored array shape differs')
        volume=np.asanyarray(image.dataobj)  # Same original stored convention/dtype; no canonicalization/resampling.
        require(np.isfinite(volume).all(),'Nonfinite sequence array')
        low,high=normalize(volume);require(np.isfinite([low,high]).all() and high>low,'Degenerate frozen sequence window')
        for case in sorted(by_volume[digest],key=lambda c:c['slice_z']):
            destination=_private(case['future_neutral_image_path_PRIVATE'])
            require(destination.parent.name=='sealed_renders' and re.fullmatch('[0-9a-f]{12,64}\\.png',destination.name),'Frozen neutral sequence asset path differs')
            rendered=image_at(volume,case['slice_z'],low,high)
            require(rendered.size==(364,436) and rendered.mode=='RGB','Original sequence image geometry differs')
            _png(destination,rendered);rows.append(sequence_record(case,destination,registry_sha,renderer_sha))
        del volume,image
    return sorted(rows,key=lambda r:r['id'])


def _render_dallas(cases,source_root,destination,plan_sha):
    from PIL import Image
    from mri_rl.visit_correspondence import render_volume,compose_pair,make_record,validate_correspondence_row,token,RENDER
    source_root=_private(source_root);destination=_private(destination);panels={};qc=[]
    contract_sha=hashlib.sha256(json.dumps(RENDER,sort_keys=True).encode()).hexdigest()
    inventory={i['sha256']:i for c in cases for i in (c['query'],c['candidate'])}
    require(len(inventory)==16,'Frozen Dallas test requires16source volumes')
    for digest,item in sorted(inventory.items()):
        path=_private(source_root/item['relative_path'])
        require(path.stat().st_size==item['bytes'] and sha(path)==digest,'Frozen Dallas payload bytes changed')
        panel,quality=render_volume(path);asset=destination/(token('volume',digest)+'.png');_png(asset,panel);panels[digest]=asset
        qc.append({'source_sha256':digest,'panel_sha256':sha(asset),**quality})
    rows=[]
    for case in sorted(cases,key=lambda c:c['case_id']):
        with Image.open(panels[case['query']['sha256']]) as q,Image.open(panels[case['candidate']['sha256']]) as c:
            rendered=compose_pair(q,c)
        asset=destination/(case['case_id']+'.png');_png(asset,rendered)
        row=make_record(case,asset,plan_sha,contract_sha);validate_correspondence_row(row,'test');rows.append(row)
    write_exact(destination/'technical_qc_PRIVATE.json',(json.dumps(qc,sort_keys=True)+'\n').encode())
    return rows


def make_released_loader(*,release_bundle,sequence_registry,dallas_registry,dallas_plan,dallas_source_root,destination,
                         sequence_renderer_source,training_manifest=None,development_manifest=None):
    """Return the source-identifiable callback expected by release_projection.

    Factory does no data I/O. The closure rechecks full approval itself, so calling
    it directly cannot bypass the22checkpoint/scoring gate. Registry/hash-bound
    recipes and private references are opened only after that gate succeeds.
    """
    def load_released_rows():
        p=release_bundle;bindings=p['bindings']
        validate_release(p['plan'],p['inventory'],bindings,p['approval'],expected_approval_sha256=p['expected_approval_sha256'])
        require(sha(__file__)==bindings['released_row_loader_source_sha256'],'Reviewed final loader source changed')
        for path,key in ((sequence_registry,'sequence_registry_sha256'),(dallas_registry,'dallas_registry_sha256'),
                         (dallas_plan,'dallas_metadata_plan_sha256'),(sequence_renderer_source,'sequence_renderer_source_sha256')):
            require(sha(path)==bindings[key],'Frozen registry/source recipe hash differs')
        from mri_rl import visit_correspondence as vc
        require(sha(vc.__file__)==bindings['dallas_renderer_source_sha256'],'Frozen Dallas renderer/scorer source differs')
        train=Path(training_manifest) if training_manifest else PRIVATE_ROOT/'c_two_task_pool_v1/train.verl.jsonl'
        require(sha(train)==p['plan']['manifest_sha256'],'Frozen training question reference differs')
        training=[json.loads(l) for l in train.read_text().splitlines() if l.strip()]
        questions={r['extra_info']['question'] for r in training if r['ability']=='mri_sequence'}
        require(len(questions)==1,'Sequence question reference is not unique')
        sequence=load_json(sequence_registry);by_sha=validate_sequence_registry(sequence,next(iter(questions)))
        plan=load_json(dallas_plan);cases=vc.build_cases(plan)['test'];validate_dallas_registry(load_json(dallas_registry),cases)
        render_destination=_private(destination)
        rows=_render_sequence(sequence,by_sha,sequence_renderer_source,bindings['sequence_registry_sha256'],bindings['sequence_renderer_source_sha256'])
        rows+=_render_dallas(cases,dallas_source_root,render_destination/'dallas_assets',bindings['dallas_metadata_plan_sha256'])
        _test_counts(rows)
        # Namespace-level overlap check supplements the already frozen raw-source
        # reserve audit; it is not a claim of cross-source global identity proof.
        exposed={(r['data_source'],r['extra_info']['patient_key']) for r in training}
        if development_manifest:
            exposed|={(r['data_source'],r['extra_info']['patient_key']) for r in [json.loads(l) for l in Path(development_manifest).read_text().splitlines() if l.strip()]}
        test={(r['data_source'],key) for r in rows for key in (r['extra_info']['patient_key'],r['extra_info'].get('candidate_patient_key')) if key}
        require(not exposed&test,'Final test source-namespace patient key overlaps training/dev')
        return rows
    return load_released_rows
