import json
from pathlib import Path
import pytest
from mri_rl import sft_evidence as module


def test_historical_structure_rejects_wrong_patient_or_nonfinite():
    schedule=[[{'ability':'mri_sequence','extra_info':{'patient_key':'a'}}]]
    source={'optimizer_steps':1,'examples':[{'step':0,'category':'mri_sequence','patient_key':'a'}], 'loss_terms':[1.0], 'gradient_l2_by_step':[0.1]}
    module.verify_history(source,schedule)
    source['examples'][0]['patient_key']='b'
    with pytest.raises(ValueError,match='schedule'): module.verify_history(source,schedule)
    source['examples'][0]['patient_key']='a';source['loss_terms']=[float('nan')]
    with pytest.raises(ValueError,match='finite'): module.verify_history(source,schedule)


def test_binding_rejects_mutated_artifact_and_escape(tmp_path):
    file=tmp_path/'receipt.json';file.write_text('{}')
    binding={'path':str(file),'sha256':module.sha256_file(file)}
    assert module.bound_file(binding,tmp_path)==file
    file.write_text('mutated')
    with pytest.raises(ValueError,match='changed'):module.bound_file(binding,tmp_path)
    with pytest.raises(ValueError,match='private'):module.bound_file({'path':'/etc/passwd','sha256':'x'},tmp_path)


def test_complete_schedule_hash_detects_changed_image(tmp_path):
    from mri_rl.mlx_sft_warmstart import schedule_inventory
    p=tmp_path/'image.png';p.write_bytes(b'first')
    rows=[[{'id':'a','ability':'mri_sequence','images':[str(p)],'reward_model':{'ground_truth':'FLAIR'},'extra_info':{'patient_key':'p','question':'which?'}}]]
    before=schedule_inventory(rows)
    p.write_bytes(b'changed')
    assert schedule_inventory(rows)!=before
    assert before[0]['target_label']=='FLAIR'


def test_sft_rejects_any_nonempty_output_and_changed_selected_row(tmp_path):
    from mri_rl.mlx_sft_warmstart import verify_frozen_row, require_empty_output, schedule_inventory
    output=tmp_path/'run';output.mkdir();(output/'before_adapters.safetensors').write_bytes(b'partial')
    with pytest.raises(ValueError,match='nonempty'):require_empty_output(output)
    p=tmp_path/'image';p.write_bytes(b'original')
    row={'id':'a','ability':'mri_sequence','images':[str(p)],'reward_model':{'ground_truth':'FLAIR'},'extra_info':{'patient_key':'p','question':'which?'}}
    frozen=schedule_inventory([[row]])[0]
    verify_frozen_row(row,0,frozen)
    p.write_bytes(b'new')
    with pytest.raises(ValueError,match='changed'):verify_frozen_row(row,0,frozen)


def test_prefix_binding_rejects_unexecuted_or_wrong_prefix():
    inventory=[{'step':i//4,'id':i} for i in range(112)]
    config={'executed_prefix_sha256':module.canonical_hash(inventory[:4]),
            'executed_prefix_examples':4}
    module.verify_prefix_binding(inventory,config,1)
    with pytest.raises(ValueError,match='prefix'):
        module.verify_prefix_binding(inventory,config,2)
    config['executed_prefix_sha256']=module.canonical_hash(inventory[4:8])
    with pytest.raises(ValueError,match='prefix'):
        module.verify_prefix_binding(inventory,config,1)
