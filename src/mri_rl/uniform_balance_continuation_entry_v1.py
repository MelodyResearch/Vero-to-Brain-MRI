"""Finite, independently approved continuation entry; imports native code only after approval and the global lock."""
from __future__ import annotations
import argparse, fcntl, hashlib, json, os
from pathlib import Path
from mri_rl import uniform_balance_continuation_driver_v1 as driver
from mri_rl.uniform_balance_continuation_gate_v1 import parent_anchor
from mri_rl.continuous_rl import canonical, digest, require
from mri_rl.tier_c_entry import prepare_arm, verify_runtime_artifacts, CRITICAL_SOURCES
from mri_rl.two_task_training import TokenLedger, validate_training_row
ROOT=Path(__file__).resolve().parents[2]
def validate_sampler_only_change(previous, current):
    changed={key for key in set(previous)|set(current) if previous.get(key)!=current.get(key)}
    require(changed=={'arm','schedule_binding_sha256'}, 'Only standard sampler/arm binding may change')
    a=previous['arm'];b=current['arm']
    require(a['name']=='P_calibrated' and a['sampling']=='proportional' and b['name']=='U_calibrated' and b['sampling']=='uniform' and
            {k:v for k,v in a.items() if k not in ('name','sampling')}=={k:v for k,v in b.items() if k not in ('name','sampling')},
            'Reward or another arm factor changed')

NEW_SOURCES=('uniform_balance_continuation_gate_v1.py','uniform_balance_continuation_driver_v1.py','uniform_balance_continuation_entry_v1.py')

def sha(value):return hashlib.sha256(canonical(value).encode()).hexdigest()

def prepare(spec):
    """Declare a NEW existing uniform seed13 sampler stage: first16; model RNG continues the parent."""
    for key,expected in spec['file_sha256'].items():
        require(digest(spec[key])==expected, 'Frozen adaptive input changed: '+key)
    anchor=parent_anchor(spec['parent_checkpoint'],spec['parent_receipt_sha256'])
    require(anchor['groups']==121 and anchor['updates']==56 and anchor['local_groups']==7 and anchor['local_updates']==4, 'This reviewed version is specific to nested parent local7/4, absolute121/Adam56')
    parent_proof=json.loads(Path(spec['parent_proof']).read_text())
    require(parent_proof['verdict']=='PASS_WITH_LIMITATIONS' and parent_proof['preserved_initial_parent_state'] is True and parent_proof['absolute_optimizer_updates']==56 and parent_proof['terminal_boundary_receipt_sha256']==spec['parent_receipt_sha256'] and parent_proof['final_adapter_sha256']==anchor['adapter_sha256'], 'Actual independent nested-parent proof differs')
    for name,expected in spec['original_source_sha256'].items():
        require(name in CRITICAL_SOURCES and digest(ROOT/'src/mri_rl'/name)==expected, 'Original GSPO dependency changed: '+name)
    require(set(spec['original_source_sha256'])==set(CRITICAL_SOURCES), 'Incomplete original source closure')
    plan=json.loads(Path(spec['plan']).read_text());measurements=json.loads(Path(spec['measurements']).read_text())
    rows=[json.loads(l) for l in Path(spec['train']).read_text().splitlines() if l.strip()]
    calibration=json.loads(Path(spec['calibration']).read_text())
    prepared=prepare_arm(plan,'U_calibrated',13,rows,measurements['rows'],calibration)
    validate_sampler_only_change(anchor['original_binding']['experiment']['parent_experiment'],prepared['bindings'])
    from mri_rl.experiment_contract import load_contract
    runtime=verify_runtime_artifacts(plan,measurements,spec['common_adapter'],Path(spec['contract']),load_contract)
    from mri_rl.continuous_rl_pilot import cache_binding_for
    from types import SimpleNamespace
    cache_binding=cache_binding_for(SimpleNamespace(cache_dir=Path(spec['cache_dir']),cache_certificate=Path(spec['cache_certificate']),cache_certificate_sha256=spec['cache_certificate_sha256'],model=Path(spec['model'])))
    require(cache_binding['certificate_sha256']==plan['common_bindings']['cache_certificate_sha256'], 'Parent cache differs')
    schedule=prepared['schedule'][:16]
    require(len(schedule)==16 and set(r['ability'] for r in schedule)=={'mri_sequence','mri_visit_correspondence'}, 'Exact next16 must remain multi-task')
    config={**driver.PROFILE,'group_size':plan['group_size'],'max_tokens':plan['max_tokens']}
    require(config['group_size']==4 and config['max_tokens']==256, 'Frozen native group/token profile differs')
    require(config==anchor['original_binding']['configuration'], 'Configuration other than standard category sampler changed')
    # New finite stage pays for all requested work; the exhausted parent cap is neither reset nor relabelled.
    budget={**plan['budget'],'maximum_attempted_groups':16,'text_visual_token_cap':sum(4*(prepared['input_tokens'](r)+256) for r in schedule)}
    ledger_plan={'group_size':4,'max_tokens':256,'budget':budget}
    bindings={'parent_experiment':prepared['bindings'],'stage':'uniform_balance_continuation_v1','seed':13,'sampler_stage':{'method':'existing_prepare_arm_U_calibrated','first_schedule_index':0,'new_frozen_stage':True},
              'parent_proof_sha256':spec['file_sha256']['parent_proof'],'max_optimizer_updates':4,
              'source_sha256':{n:digest(ROOT/'src/mri_rl'/n) for n in NEW_SOURCES}}
    fullbindings={**bindings,'adaptive_parent':anchor}
    binding=driver._binding(schedule,fullbindings,config,validate_training_row,ledger_plan)
    output=Path(spec['output']); require(output.is_relative_to(ROOT/'local_data'),'Ignored isolated output required')
    current=driver.inspect_boundaries(output,binding) if output.exists() else None
    require(not output.exists() or current is not None, 'Incomplete output requires forensic inspection')
    require(bool(spec.get('resume'))==(current is not None),'Explicit resume must match durable child state')
    scope={'kind':'uniform_balance_continuation_v1','execution_tier':'C','phase':'continuation','spec_sha256':sha(spec),
           'run_binding':binding,'native_runtime':runtime,'derived_cache':cache_binding,'output':str(output),
           'source_sha256':{n:digest(ROOT/'src/mri_rl'/n) for n in (*CRITICAL_SOURCES,*NEW_SOURCES)},
           'group_size':4,'max_attempted_groups':16,'max_optimizer_updates':4,'new_token_cap':budget['text_visual_token_cap'],
           'starting_child_receipt_sha256':digest(current['path']/'receipt.json') if current else None,
           'parent_native_optimizer_step':56,'parent_schedule_cursor':121,'parent_local_checkpoint_groups':7,'parent_local_checkpoint_updates':4,'prespecified_category_counts':{task:sum(row['ability']==task for row in schedule) for task in sorted({row['ability'] for row in schedule})},'sampler_stage_first_index':0,'parent_evidence_tier':'REPRODUCED',
           'claim':'Single standard proportional-to-uniform category-sampling change, inherited gate unchanged; finite new stage, not original U arm and no established MRI gain'}
    return scope,dict(output=output,schedule=schedule,bindings=bindings,config=config,
                      parent_checkpoint=spec['parent_checkpoint'],parent_receipt_sha256=spec['parent_receipt_sha256'],
                      max_new_groups=16,max_groups_cap=16,max_optimizer_updates=4,resume=bool(spec.get('resume')),
                      grader=prepared['grader'],row_validator=validate_training_row,ledger=TokenLedger(ledger_plan),input_tokens=prepared['input_tokens'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--prepare-only',action='store_true');p.add_argument('--scope-output',type=Path);p.add_argument('--execute',action='store_true');p.add_argument('--review',type=Path);p.add_argument('--expected-review-sha256');a=p.parse_args()
    require(a.prepare_only != a.execute,'Choose prepare or execute')
    spec=json.loads(a.spec.read_text());scope,kwargs=prepare(spec)
    if a.prepare_only:
        require(a.scope_output is not None,'Scope output required');driver.write_json(a.scope_output,scope)
        print(canonical({'status':'READY_FOR_INDEPENDENT_REVIEW_NOT_RUN','scope_sha256':sha(scope),'max_groups':16,'max_updates':4,'token_cap':scope['new_token_cap']}));return
    from mri_rl.experiment_contract import _check_review,load_contract
    _check_review(review_verdict_path=a.review,expected_review_sha256=a.expected_review_sha256,scope=scope)
    # Shared original-queue lock: never signal or retire any process here.
    with Path(spec['global_lock']).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        scope2,kwargs=prepare(spec);require(scope2==scope,'Adaptive inputs changed while waiting for exclusive lock')
        from mri_rl.mlx_continuous_gspo import backend_factory,is_loading_unit
        from mri_rl.paced_resources import PacedGate
        gate=PacedGate(Path(spec['output']).with_name(Path(spec['output']).name+'_resources_PRIVATE.jsonl'))
        cache={'dir':spec['cache_dir']}
        def before_unit(name):gate.before_unit(name,loading=is_loading_unit(name,cache))
        result=driver.run(**kwargs,backend_factory=backend_factory(model_path=Path(spec['model']),contract=load_contract(Path(spec['contract'])),seed=13,before_unit=before_unit,cache=cache),authorize=lambda ctx:before_unit('authorize_'+ctx['phase']))
        receipt={**result,'scope_sha256':sha(scope),'science_claim':False,'new_stage_budget':scope['new_token_cap']}
        receipt['boundary']=str(receipt['boundary']);driver.write_json(Path(spec['output']).parent/'native_complete_SAFE.json',receipt)
        print(canonical({k:v for k,v in receipt.items() if k!='boundary'}))
if __name__=='__main__':main()
