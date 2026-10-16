import importlib.util
import json
from pathlib import Path
import pytest
s=importlib.util.spec_from_file_location('controls',Path(__file__).resolve().parents[1]/'src/mri_rl/paced_sequence_controls.py');m=importlib.util.module_from_spec(s)
if Path(s.origin).exists():s.loader.exec_module(m)


def pair_rows():
    out=[]
    for kind in ('flair','t1ce'):
        image=f'/private/patient_z045_{kind}.png'
        out.append({'record_id':f'patient_z045_{kind}_sequence','category':'mri_sequence','image':image,
                    'prompt':[{'role':'system','content':'system'},{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':'same question'}]}]})
    return out


def test_no_image_control_removes_pixels_only_and_preserves_question():
    assert hasattr(m,'derived_controls')
    original=pair_rows();controls=m.derived_controls(original,'no_image')
    assert controls[0]['image'] is None and controls[0]['prompt'][1]['content']==[{'type':'text','text':'same question'}]
    assert original[0]['prompt'][1]['content'][0]['type']=='image'


def test_swap_is_same_patient_level_opposite_modality_not_arbitrary_change():
    assert hasattr(m,'derived_controls')
    original=pair_rows();controls=m.derived_controls(original,'swap')
    assert controls[0]['image']==original[1]['image'] and controls[1]['image']==original[0]['image']
    assert controls[0]['prompt'][1]['content'][-1]==original[0]['prompt'][1]['content'][-1]
    with pytest.raises(ValueError):m.derived_controls(original[:1],'swap')


def test_control_baseline_rejects_changed_pixels_and_prompt(tmp_path,monkeypatch):
    assert hasattr(m,'verify_control_baseline')
    image=tmp_path/'i.png';image.write_bytes(b'oldpixels')
    row={'record_id':'id','category':'mri_sequence','image':str(image),'prompt':'question'}
    expected={'model_path':'model','model_config_sha256':'config','model_template_sha256':'template','template_override_sha256':None,'max_tokens':256,'temperature':0,'manifest_sha256':'manifest','adapter_config_sha256':m.pe.BASELINE_ADAPTER_CONFIG_SHA256}
    baseline={**row,**expected,'image_sha256':m.pe.sha256(image),'adapter_checkpoint_sha256':m.HISTORICAL_ADAPTER,
              'adapter_verified_parameter_tensors':504,'image_placeholder_count':1,'response':'response'}
    path=tmp_path/'baseline.jsonl';path.write_text(json.dumps(baseline)+'\n')
    spec={'baseline_sha256':m.pe.sha256(path),'images':1}
    m.verify_control_baseline(path,[row],expected,spec,expected_count=1)
    image.write_bytes(b'changedpixels')
    with pytest.raises(ValueError):m.verify_control_baseline(path,[row],expected,spec,expected_count=1)


def test_incomplete_primary_evaluation_blocks_control_planning():
    assert hasattr(m,'require_completed_primary')
    with pytest.raises(ValueError):m.require_completed_primary([{'record_id':'a'}],[],{})


def test_control_review_cannot_be_reused_for_other_mode(tmp_path):
    assert hasattr(m,'verify_review')
    path=tmp_path/'review.json';scope={'mode':'no_image','max_new_records':4}
    path.write_text(json.dumps({'kind':'paced_sequence_controls_v1','verdict':'PASS_WITH_LIMITATIONS','reviewer':'independent','scope':scope}))
    m.verify_review(path,m.pe.sha256(path),scope)
    with pytest.raises(ValueError):m.verify_review(path,m.pe.sha256(path),dict(scope,mode='swap'))


def test_baseline_prompt_drift_and_digest_change_rejected(tmp_path):
    record={'record_id':'id','category':'mri_sequence','image':None,'prompt':'question'}
    expected={'model_path':'model','model_config_sha256':'config','model_template_sha256':'template','template_override_sha256':None,'max_tokens':256,'temperature':0,'manifest_sha256':'manifest','adapter_config_sha256':m.pe.BASELINE_ADAPTER_CONFIG_SHA256}
    baseline={**record,**expected,'image_sha256':None,'adapter_checkpoint_sha256':m.HISTORICAL_ADAPTER,'adapter_verified_parameter_tensors':504,'image_placeholder_count':0,'response':'response'}
    path=tmp_path/'baseline.jsonl';path.write_text(json.dumps(baseline)+'\n');spec={'baseline_sha256':m.pe.sha256(path),'images':0}
    m.verify_control_baseline(path,[record],expected,spec,expected_count=1)
    with pytest.raises(ValueError):m.verify_control_baseline(path,[dict(record,prompt='changed question')],expected,spec,expected_count=1)
    path.write_text(json.dumps(dict(baseline,max_tokens=64))+'\n')
    with pytest.raises(ValueError):m.verify_control_baseline(path,[record],expected,spec,expected_count=1)
    spec['baseline_sha256']=m.pe.sha256(path)
    with pytest.raises(ValueError):m.verify_control_baseline(path,[record],expected,spec,expected_count=1)


def test_full_primary_completion_validates_adapter_images_and_actual_metrics(tmp_path):
    image=tmp_path/'i.png';image.write_bytes(b'pixels');records=[];outputs=[]
    for i in range(54):
        record={'record_id':str(i),'category':'category','prompt':'question','image':str(image)};records.append(record)
        outputs.append({**record,'identity':'frozen','image_sha256':m.pe.sha256(image),'adapter_verified_parameter_tensors':504,'image_placeholder_count':1,
                        'response':'answer','generation_tokens':3,'prompt_tokens':10,'total_tokens':13,'finish_reason':'stop','generation_peak_memory_gb':2.5})
    m.require_completed_primary(records,outputs,{'identity':'frozen'})
    outputs[-1]['adapter_verified_parameter_tensors']=503
    with pytest.raises(ValueError):m.require_completed_primary(records,outputs,{'identity':'frozen'})
    outputs[-1]['adapter_verified_parameter_tensors']=504;outputs[-1].pop('generation_tokens')
    with pytest.raises(ValueError):m.require_completed_primary(records,outputs,{'identity':'frozen'})


def test_unknown_control_mode_rejected():
    with pytest.raises(ValueError):m.derived_controls(pair_rows(),'arbitrary-image')


def test_cpu_import_has_no_model_side_effects():
    import os,subprocess,sys
    env=dict(os.environ,PYTHONPATH=str(Path(__file__).resolve().parents[1]/'src'))
    result=subprocess.run([sys.executable,'-c',"import sys;import mri_rl.paced_sequence_controls;assert 'mlx.core' not in sys.modules;assert 'mlx_vlm' not in sys.modules"],env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
