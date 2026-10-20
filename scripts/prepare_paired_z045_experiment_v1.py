"""Assemble CPU-only preparation artifacts. There is no model launch option."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from mri_rl.paired_z045_continuation_entry_v1 import prepare,require

ROOT=Path(__file__).resolve().parents[1]
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--primary-root',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();primary=a.primary_root.resolve();data=a.data.resolve();out=a.output.resolve()
    require(data.is_relative_to(ROOT/'local_data') and out.is_relative_to(ROOT/'local_data') and not out.exists(),'New private preparation destination required')
    proof=json.loads((ROOT/'results/PAIRED_Z045_DATA_V1.json').read_text())
    for name,expected in proof['artifact_sha256'].items():require(sha(data/name)==expected,'Prepared data artifact changed: '+name)
    parent=primary/'.worktrees/mri-adaptive-accuracy-gate/local_data/uniform_balance_continuation_v1/run/boundary_0008'
    previous=primary/'.worktrees/mri-adaptive-accuracy-gate/local_data/uniform_balance_continuation_v1/spec_NOT_APPROVED_PRIVATE.json'
    require(sha(previous)=='bee47ec33b51b4075103ff4cece96acb152d678f305d8555055c19f48b1bd535','Preserved historical runtime specification changed')
    old=json.loads(previous.read_text())
    for name,expected in old['original_source_sha256'].items():require(sha(ROOT/'src/mri_rl'/name)==expected,'Historical frozen GSPO source changed: '+name)
    previous_numeric=primary/'.worktrees/vero-mri-margin-ontology/local_data/class_margin_v1/scope_v2_PRIVATE.json'
    require(sha(previous_numeric)=='563e1d54812526c8f168ce5a86c7a68e20ccaca436bbc784e68d9bfe4bedb996','Preserved numerical diagnostic source scope changed')
    runtime={path:expected for path,expected in json.loads(previous_numeric.read_text())['source_sha256'].items() if '/work/mlx-venv/' in path}
    require(runtime and all(sha(path)==expected for path,expected in runtime.items()),'Installed numerical source changed; prepare a separate contract')
    files={'reference_schedule':data/'reference_original_next32_PRIVATE.json',
        'candidate_schedule':data/'candidate_common_stage_next32_PRIVATE.json',
        'input_costs':data/'candidate_common_stage_next32_input_costs_PRIVATE.json',
        'sequence56':data/'paired_z045_sequence56_PRIVATE.verl.jsonl',
        'mixed104':data/'mixed104_PRIVATE.verl.jsonl','data_binding':data/'binding_PRIVATE.json',
        'calibration':Path(old['calibration']),'contract':Path(old['contract']),'cache_certificate':Path(old['cache_certificate']),
        'measurements':Path(old['measurements']),
        'data_membership':data/'source_and_pair_membership_PRIVATE.json',
        'original_image_inventory':primary/'local_data/c_two_task_pool_v1/row_inventory_PRIVATE.json',
        'original_pool_receipt':primary/'local_data/c_two_task_pool_v1/pool_receipt_SAFE.json'}
    for key in ('calibration','contract','cache_certificate','measurements'):require(sha(files[key])==old['file_sha256'][key],'Historical runtime/calibration changed: '+key)
    require(sha(files['original_pool_receipt'])=='684f3bcf47c4502e472617921473e4bc83c2159bed24e1c3c125dea78ef1a89f' and
            sha(files['original_image_inventory'])=='fb78af69995bae9d6191fb5c6143c14b894710b54f0f878acf727fd599c4df7c',
            'Historical expected-image provenance changed')
    spec={'parent_checkpoint':str(parent),'parent_receipt_sha256':proof['source_sha256']['parent_uniform_receipt'],
        **{k:str(v) for k,v in files.items()},'file_sha256':{k:sha(v) for k,v in files.items()},
        **{k:old[k] for k in ('model','cache_dir','cache_certificate_sha256','global_lock')},
        'runtime_source_sha256':runtime,
        'original_source_sha256':old['original_source_sha256'],
        'text_visual_token_cap':70784,'maximum_seconds':3000,'minimum_group_window_seconds':60,
        'output':str(out/'run_NOT_LAUNCHED')}
    scope=prepare(spec);out.mkdir(parents=True)
    save(out/'spec_PREPARATION_ONLY_PRIVATE.json',spec);save(out/'scope_PREPARATION_ONLY_PRIVATE.json',scope)
    # Deliberately only executable CPU inspection; no generated train command or
    # fake user/reviewer authorization artifact can be copied into a launch.
    command={'argv':['python3','-m','mri_rl.paired_z045_continuation_entry_v1','inspect','--spec',str(out/'spec_PREPARATION_ONLY_PRIVATE.json')],
        'model_launch':False,'training_authorized':False}
    save(out/'command_CPU_INSPECTION_ONLY_PRIVATE.json',command)
    safe={'schema':'paired-z045-execution-preparation-v1','status':scope['status'],'training_authorized':False,
        'parent_optimizer_step':60,'parent_absolute_groups':129,'parent_adapter_sha256':scope['parent']['adapter_sha256'],
        'parent_receipt_sha256':spec['parent_receipt_sha256'],'parent_array_bytes_reverified_this_command':False,
        'budget':scope['budget'],'configuration':scope['configuration'],
        'spec_sha256':sha(out/'spec_PREPARATION_ONLY_PRIVATE.json'),'scope_sha256':sha(out/'scope_PREPARATION_ONLY_PRIVATE.json'),
        'data_proof_sha256':sha(ROOT/'results/PAIRED_Z045_DATA_V1.json'),
        'evaluation_proof_sha256':sha(ROOT/'results/PAIRED_Z045_EVALUATION_PREPARATION_V1.json'),
        'budget_proof_sha256':sha(ROOT/'results/PAIRED_Z045_HISTORICAL_BUDGET_INDEPENDENT_V1.json'),
        'numeric_prerequisites':scope['numeric_prerequisites'],'new_model_loads':0,'new_generation':0,'new_optimizer_updates':0,
        'historical_source_unchanged':True,'source_sha256':{(str(Path(k).relative_to(ROOT)) if Path(k).is_relative_to(ROOT) else 'runtime/'+k.split('/site-packages/')[-1]):v for k,v in scope['source_sha256'].items()},
        'compiler_sha256':sha(__file__),
        'comparison_limit':'One parent/terminal engineering pilot cannot causally separate pairing benefit from extra optimization without a matched old-pool continuation; no such extra run is authorized.',
        'next_execution':'Only a separately scoped, source-bound, zero-update F32 numerical qualification may be proposed next. Request finite training authorization only AFTER its actual PASS and complete source/scope review.',
        'readiness_for_training':'NO','final100_access':False,'independent_patient_holdout':'NOT_ESTABLISHED'}
    save(ROOT/'results/PAIRED_Z045_EXECUTION_PREPARATION_V1.json',safe)
    print(json.dumps({'status':scope['status'],'scope_sha256':safe['scope_sha256'],'training_authorized':False,'model_loads':0}))

if __name__=='__main__':main()
