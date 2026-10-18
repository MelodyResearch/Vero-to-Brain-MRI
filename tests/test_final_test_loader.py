"""Synthetic-only loader fixtures. Actual sealed registries/pixels are never opened."""
import ast
import hashlib
from pathlib import Path
import numpy as np
import pytest
from PIL import Image


def module():
    from mri_rl import final_test_loader
    return final_test_loader


def recipe_fixture():
    files=[];cases=[]
    for person in range(14):
        for label in ('FLAIR','contrast-enhanced T1'):
            digest=hashlib.sha256(f'{person}:{label}'.encode()).hexdigest()
            files.append({'sha256':digest,'bytes':1,'subject_PRIVATE':f'source-{person}','modality_target_PRIVATE':label,'local_path_PRIVATE':'fixture-only'})
            for z in (45,91,136):
                key=hashlib.sha256(f'{digest}:{z}'.encode()).hexdigest()[:24]
                cases.append({'record_id':key,'source':'OpenNeuro_ds007045','route':'mri_sequence','split':'sealed_test',
                    'patient_key':hashlib.sha256(f'patient-{person}'.encode()).hexdigest()[:24],
                    'source_sha256':digest,'source_bytes':1,'slice_z':z,'ground_truth_PRIVATE':label,
                    'question':'Frozen sequence question','future_neutral_image_path_PRIVATE':f'/fixture/sealed_renders/{key}.png','pixels_rendered':False})
    return {'files_PRIVATE':files,'cases_PRIVATE':cases}


def test_invalid_release_does_not_open_any_registry_or_import_volume_reader(monkeypatch,tmp_path):
    m=module();opened=[]
    def fail(*a,**k):raise ValueError('incomplete final inventory')
    monkeypatch.setattr(m,'validate_release',fail)
    monkeypatch.setattr(m,'load_json',lambda p:opened.append(p))
    loader=m.make_released_loader(release_bundle={'plan':{},'inventory':[],'bindings':{},'approval':{},'expected_approval_sha256':'x'},
        sequence_registry=tmp_path/'seq.json',dallas_registry=tmp_path/'dallas.json',dallas_plan=tmp_path/'plan.json',
        dallas_source_root=tmp_path/'payloads',destination=tmp_path/'out',sequence_renderer_source=tmp_path/'original.py')
    assert opened==[]
    with pytest.raises(ValueError,match='incomplete'):loader()
    assert opened==[]


@pytest.mark.parametrize('defect',['duplicate_id','wrong_target','wrong_slice','wrong_source','wrong_patient','rendered','wrong_question','omitted'])
def test_registry_complete_frozen_identity_and_geometry_validation(defect):
    m=module();r=recipe_fixture()
    if defect=='duplicate_id':r['cases_PRIVATE'][1]['record_id']=r['cases_PRIVATE'][0]['record_id']
    elif defect=='wrong_target':r['cases_PRIVATE'][0]['ground_truth_PRIVATE']='contrast-enhanced T1'
    elif defect=='wrong_slice':r['cases_PRIVATE'][0]['slice_z']=92
    elif defect=='wrong_source':r['cases_PRIVATE'][0]['source']='other source'
    elif defect=='wrong_patient':r['cases_PRIVATE'][0]['patient_key']=r['cases_PRIVATE'][6]['patient_key']
    elif defect=='rendered':r['cases_PRIVATE'][0]['pixels_rendered']=True
    elif defect=='wrong_question':r['cases_PRIVATE'][0]['question']='Changed prompt'
    else:r['cases_PRIVATE'].pop()
    with pytest.raises(ValueError):m.validate_sequence_registry(r,'Frozen sequence question')


def test_frozen_sequence_transform_is_byte_exact_to_original_pure_functions():
    m=module();source=Path(__file__).parents[3]/'work/ds007045_prompts.py'
    # Read only the pre-existing renderer SOURCE, not any data or its top-level loops.
    normalize,image_at=m.sequence_functions(source)
    volume=np.arange(5*7*9,dtype=np.float32).reshape(5,7,9);volume[0]=0
    lo,hi=normalize(volume);expected_window=np.percentile(volume[volume!=0][::16],[.5,99.5])
    assert np.array_equal([lo,hi],expected_window)
    plane=np.rot90(volume[:,:,4]);pixels=np.uint8(np.round(np.clip((plane-lo)/(hi-lo),0,1)*255))
    expected=Image.fromarray(pixels).resize((10,14),Image.Resampling.BILINEAR).convert('RGB')
    assert image_at(volume,4,lo,hi).tobytes()==expected.tobytes()


def test_typed_sequence_row_keeps_gold_and_patient_private():
    m=module();r=recipe_fixture();m.validate_sequence_registry(r,'Frozen sequence question');case=r['cases_PRIVATE'][0]
    row=m.sequence_record(case,Path('/private/opaque.png'),'a'*64,'b'*64)
    assert row['data_source']=='OpenNeuro_ds007045' and row['ability']=='mri_sequence'
    assert row['extra_info']['internal_split']=='test' and row['extra_info']['patient_key']==case['patient_key']
    assert row['reward_model']['ground_truth']==case['ground_truth_PRIVATE']
    text=str(row['prompt'])
    assert 'ground_truth' not in text and case['patient_key'] not in text and 'source-0' not in text


def test_dallas_registry_exact_source_cycle_and_hidden_target_mapping():
    m=module();from mri_rl.visit_correspondence import token
    cases=[]
    for c in range(4):
        people=[f'person-{c}-{i}' for i in range(2)];cluster=token('cycle',*people)
        for q in people:
            for other in people:
                qsha=hashlib.sha256((q+'wave2').encode()).hexdigest();csha=hashlib.sha256((other+'wave1').encode()).hexdigest()
                cases.append({'case_id':token('case',qsha,csha),'split':'test','cluster_key':cluster,
                    'query_subject':q,'candidate_subject':other,'query':{'sha256':qsha},'candidate':{'sha256':csha},
                    'target':'same' if q==other else 'different'})
    registry=m.expected_dallas_registry(cases);m.validate_dallas_registry(registry,cases)
    registry[0]['source_equality_target']='different' if registry[0]['source_equality_target']=='same' else 'same'
    with pytest.raises(ValueError):m.validate_dallas_registry(registry,cases)


def test_sequence_io_uses_each_synthetic_volume_once_and_persists_all_frozen_views(monkeypatch,tmp_path):
    m=module();import nibabel as nib
    from types import SimpleNamespace
    monkeypatch.setattr(m,'PRIVATE_ROOT',tmp_path)
    complete=recipe_fixture();cases=[dict(c) for c in complete['cases_PRIVATE'][:6]];files=[]
    for number,old in enumerate(complete['files_PRIVATE'][:2]):
        file=dict(old);path=tmp_path/f'opaque-volume-{number}.nii.gz';path.write_bytes(f'fixture bytes{number}'.encode())
        digest=hashlib.sha256(path.read_bytes()).hexdigest();files.append({**file,'sha256':digest,'bytes':path.stat().st_size,'local_path_PRIVATE':str(path)})
        for case in cases:
            if case['source_sha256']==old['sha256']:
                case['source_sha256']=digest;case['source_bytes']=path.stat().st_size
                case['future_neutral_image_path_PRIVATE']=str(tmp_path/'sealed_renders'/(case['record_id']+'.png'))
    loaded=[];z_used=[]
    def read(path):loaded.append(path);return SimpleNamespace(shape=(182,218,182),dataobj=np.arange(8,dtype=np.float32).reshape(2,2,2))
    monkeypatch.setattr(nib,'load',read)
    def fake_render(volume,z,lo,hi):z_used.append(z);return Image.new('RGB',(364,436),(z,0,0))
    monkeypatch.setattr(m,'sequence_functions',lambda source:(lambda volume:(0.,7.),fake_render))
    rows=m._render_sequence({'cases_PRIVATE':cases},{f['sha256']:f for f in files},tmp_path/'not-read.py','a'*64,'b'*64)
    assert len(loaded)==2 and sorted(z_used)==[45,45,91,91,136,136] and len(rows)==6
    assert all(Path(r['images'][0]).is_file() for r in rows)
    assert all(r['extra_info']['internal_split']=='test' for r in rows)


def test_actual_factory_closure_reaches_render_after_synthetic_approved_checks(monkeypatch,tmp_path):
    """Exercise nested closure, not _render_sequence directly; no real registry I/O."""
    import json
    m=module();from mri_rl import visit_correspondence as vc
    bindings={k:'f'*64 for k in ('released_row_loader_source_sha256','sequence_registry_sha256',
        'dallas_registry_sha256','dallas_metadata_plan_sha256','sequence_renderer_source_sha256','dallas_renderer_source_sha256')}
    training=tmp_path/'synthetic_train.jsonl'
    training.write_text(json.dumps({'ability':'mri_sequence','extra_info':{'question':'Synthetic frozen question'}})+'\n')
    reached=[];private_checks=[]
    monkeypatch.setattr(m,'validate_release',lambda *a,**k:None)  # Explicit synthetic approval fixture only.
    monkeypatch.setattr(m,'sha',lambda path:'f'*64)
    monkeypatch.setattr(m,'load_json',lambda path:{'synthetic':True})
    monkeypatch.setattr(m,'validate_sequence_registry',lambda *a:{})
    monkeypatch.setattr(m,'validate_dallas_registry',lambda *a:None)
    monkeypatch.setattr(vc,'build_cases',lambda plan:{'test':[]})
    def private(path):private_checks.append(path);return Path(path)
    monkeypatch.setattr(m,'_private',private)
    def before_render(*args):reached.append(True);raise RuntimeError('synthetic render boundary reached')
    monkeypatch.setattr(m,'_render_sequence',before_render)
    destination=tmp_path/'synthetic_render_destination'
    loader=m.make_released_loader(release_bundle={'plan':{'manifest_sha256':'f'*64},'inventory':[],
        'bindings':bindings,'approval':{},'expected_approval_sha256':'fixture-only'},
        sequence_registry=tmp_path/'seq',dallas_registry=tmp_path/'dallas',dallas_plan=tmp_path/'plan',
        dallas_source_root=tmp_path/'source',destination=destination,sequence_renderer_source=tmp_path/'renderer',
        training_manifest=training)
    with pytest.raises(RuntimeError,match='synthetic render boundary reached'):loader()
    assert reached==[True] and private_checks==[destination]
