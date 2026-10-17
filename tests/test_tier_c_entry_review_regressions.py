"""Independent CLI-preflight counterexamples, with all native entry points stubbed.

Approval objects here are labelled CPU_FIXTURE_NOT_A_NATIVE_APPROVAL and are never
used with a native backend. No MLX, Torch, processor or model imports occur.
"""
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType,SimpleNamespace

import pytest
from mri_rl import tier_c_entry as entry
from mri_rl.full_arm_protocol import compile_plan
from mri_rl.two_task_pool import canonical_sha
from tests.test_two_task_training import ROWS


def h(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def cli_case(tmp_path,monkeypatch):
    # Keep actual reviewed code hashes while isolating every output under a fake repository root.
    sources=entry.source_hashes();monkeypatch.setattr(entry,'ROOT',tmp_path)
    monkeypatch.setattr(entry,'source_hashes',lambda:dict(sources))
    output=tmp_path/'local_data/out';output.parent.mkdir()
    adapter=tmp_path/'initial.safetensors';adapter.write_bytes(b'CPU frozen starting weights')
    alternative_adapter=tmp_path/'changed.safetensors';alternative_adapter.write_bytes(b'CPU different starting weights')
    contract=tmp_path/'contract.json';contract.write_text(json.dumps({'system_instruction':'Frozen system.',
        'assistant_prefix':'<|im_start|>assistant\n','enable_thinking':False,'generation':{'max_tokens':256,'temperature':1.}}))
    altered_contract=tmp_path/'wrong_contract.json';altered_contract.write_text(json.dumps({'system_instruction':'Altered system.',
        'assistant_prefix':'<|im_start|>assistant\n','enable_thinking':False,'generation':{'max_tokens':256,'temperature':1.}}))
    train=tmp_path/'train.jsonl';train.write_text(''.join(json.dumps(r)+'\n' for r in ROWS))
    metrics={'schema':'frozen-training-token-measurements-v1','split':'train','manifest_sha256':h(train),
             'processor_sha256':'a'*64,'prompt_contract_sha256':h(contract),
             'rows':[{'record_sha256':canonical_sha(r),'source':r['data_source'],'route':r['ability'],
                      'input_tokens':213 if r['ability']=='mri_sequence' else 405,
                      'visual_tokens':154 if r['ability']=='mri_sequence' else 288} for r in ROWS]}
    measurements=tmp_path/'measurements.json';measurements.write_text(json.dumps(metrics))
    plan=compile_plan(metrics,entry.common_bindings(metrics,adapter,'b'*64))
    plan_path=tmp_path/'plan.json';plan_path.write_text(json.dumps(plan))
    prepared=entry.prepare_arm(plan,'U_raw',7,ROWS,metrics['rows'])
    # [Claude adaptation] the approved scope now also binds the validated runtime artifacts (contract bridge, config, versions).
    runtime=entry.verify_runtime_artifacts(plan,metrics,adapter,contract,lambda p:SimpleNamespace(**json.loads(Path(p).read_text())))
    runtime['dynamic_evidence_sha256']={}
    scope=entry.review_scope(prepared,output=output,runtime=runtime)
    verdict={**scope,'verdict':'PASS','reviewer_kind':'isolated_codex',
             'reviewer_id':'CPU_FIXTURE_NOT_A_NATIVE_APPROVAL','reviewer_findings':['Pure CPU preflight test only'],
             'substantial_training_authorized':True,'acceptance_criteria':['CPU fixture only'],
             'matched_budget_checks':['CPU fixture only']}
    review=tmp_path/'review.json';review.write_text(json.dumps(verdict))
    # Cache certificate validation and backend factory behavior are outside these
    # contract/adapter/image/measurement omission tests. Stub to prevent native imports.
    pilot=ModuleType('mri_rl.continuous_rl_pilot')
    pilot.cache_binding_for=lambda args:{'certificate_sha256':'b'*64}
    native=ModuleType('mri_rl.mlx_continuous_gspo')
    native.backend_factory=lambda **kwargs: ('CPU_FACTORY',kwargs)
    native.is_loading_unit=lambda *args:False
    paced=ModuleType('mri_rl.paced_resources')
    paced.PacedGate=lambda *args:SimpleNamespace(before_unit=lambda *args,**kwargs:None)
    monkeypatch.setitem(sys.modules,'mri_rl.continuous_rl_pilot',pilot)
    monkeypatch.setitem(sys.modules,'mri_rl.mlx_continuous_gspo',native)
    monkeypatch.setitem(sys.modules,'mri_rl.paced_resources',paced)
    from mri_rl import experiment_contract
    monkeypatch.setattr(experiment_contract,'load_contract',lambda p:SimpleNamespace(**json.loads(Path(p).read_text())))
    calls=[]
    monkeypatch.setattr(entry,'train_arm',lambda *args,**kwargs:calls.append(kwargs) or {'status':'CPU_FIXTURE_ACCEPTED'})
    args=['tier_c_entry','train','--measurements',str(measurements),'--train',str(train),'--adapter',str(adapter),
          '--cache-certificate-sha256','b'*64,'--plan',str(plan_path),'--arm','U_raw','--seed','7','--output',str(output),
          '--model',str(tmp_path/'model'),'--contract',str(contract),'--cache-dir',str(tmp_path/'cache'),
          '--cache-certificate',str(tmp_path/'certificate.json'),'--review-verdict',str(review),'--expected-review-sha256',h(review)]
    old_image_bytes={r['images'][0]:Path(r['images'][0]).read_bytes() for r in ROWS}
    yield {'args':args,'adapter':alternative_adapter,'contract':altered_contract,'metrics':metrics,'measurements':measurements,
           'calls':calls,'image':Path(ROWS[0]['images'][0])}
    for image,blob in old_image_bytes.items():Path(image).write_bytes(blob)


@pytest.mark.parametrize('changed',['adapter','contract','image','measurements'])
def test_changed_runtime_artifact_cannot_reuse_same_approved_scope(cli_case,monkeypatch,changed):
    args=cli_case['args'][:]
    if changed in {'adapter','contract'}:
        args[args.index('--'+changed)+1]=str(cli_case[changed])
    elif changed=='image':
        cli_case['image'].write_bytes(b'CHANGED PIXELS under same frozen image path')
    else:
        metrics=cli_case['metrics']
        for row in metrics['rows']:
            if row['route']=='mri_visit_correspondence':row['input_tokens']=213
        cli_case['measurements'].write_text(json.dumps(metrics))
    monkeypatch.setattr(sys,'argv',args)
    with pytest.raises(ValueError):entry.main()
    assert not cli_case['calls']


@pytest.mark.parametrize('filename',['experiment_contract.py','mlx_gspo_smoke.py','vero_official_launch.py'])
def test_approval_scope_binds_critical_called_source(filename):
    assert filename in entry.CRITICAL_SOURCES


def test_unchanged_cpu_fixture_scope_is_accepted_without_native_import(cli_case,monkeypatch):
    monkeypatch.setattr(sys,'argv',cli_case['args'])
    entry.main()
    assert len(cli_case['calls'])==1
    assert 'mlx.core' not in sys.modules and 'torch' not in sys.modules
