"""Finite common-B28 read-only calibration (Codex entry; no optimizer updates).

Protocol is fixed before sampling: all 76 C training rows once in manifest order,
K4, seed7, temperature1, max256; at most179440 input+generated tokens. Costs
are separate from the169216-token RL arms. This is a MedGRPO-style binary
mechanistic control, not faithful replication of a medical-video experiment.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.metadata as md
import json
from pathlib import Path

from mri_rl import tier_c_calibration as cal, tier_c_entry as common
from mri_rl.continuous_rl import canonical,digest,require,write_json,sync_dir
from mri_rl.experiment_contract import _check_review,load_contract
from mri_rl.full_arm_protocol import compile_plan
from mri_rl.two_task_pool import read_rows,canonical_sha,validate_model_boundary
from mri_rl.two_task_training import input_tokens_by_source

ROOT=Path(__file__).resolve().parents[2]
SOURCES=tuple(dict.fromkeys((*common.CRITICAL_SOURCES,'tier_c_calibration.py','tier_c_calibration_entry.py')))


def sha(value):return hashlib.sha256(canonical(value).encode()).hexdigest()


def source_hashes():return {n:digest(ROOT/'src/mri_rl'/n) for n in SOURCES}


def data_contract(train,measurements,native_inputs):
    require(digest(train)==measurements['manifest_sha256'],'Training manifest differs from native receipt')
    rows=[x['row'] for x in read_rows(train)];counts=dict(Counter(r['ability'] for r in rows))
    require(len(rows)==76 and counts=={'mri_sequence':28,'mri_visit_correspondence':48},'Exact C76 training coverage required')
    for r in rows:validate_model_boundary(r,'train')
    measured={x['record_sha256']:x for x in measurements['rows']}
    require(len(measured)==len(measurements['rows'])==76 and set(measured)=={canonical_sha(r) for r in rows},'Exact native input measurement coverage required')
    refs={r['record_id']:r for r in native_inputs}
    require(len(refs)==len(native_inputs)==76 and set(refs)=={r['id'] for r in rows},'Exact unique native prompt coverage required')
    lookup=input_tokens_by_source(measurements['rows'],rows);prompts={}
    for row in rows:
        m=measured[canonical_sha(row)];ref=refs[row['id']];ids=ref['input_token_ids']
        require(m['route']==row['ability'] and m['source']==row['data_source'],'Native route/source differs')
        require(ref['image_sha256']==digest(row['images'][0]) and canonical_sha(ids)==ref['input_token_ids_sha256'] and
                len(ids)==1 and len(ids[0])==lookup(row) and all(type(t) is int and t>=0 for t in ids[0]),'Frozen native image/token IDs differ')
        require(isinstance(ref['rendered_prompt_sha256'],str) and len(ref['rendered_prompt_sha256'])==64 and ref['image_grid_thw'],'Native prompt/grid signature missing')
        prompts[row['id']]={'rendered_prompt_sha256':ref['rendered_prompt_sha256'],'input_token_ids':ids,'image_grid_thw':ref['image_grid_thw']}
    return rows,lookup,prompts


def compile_calibration_plan(train,measurements,native_inputs,adapter,certificate_sha256,prompt_semantics):
    rows,lookup,prompts=data_contract(Path(train),measurements,native_inputs)
    bindings=common.common_bindings(measurements,adapter,certificate_sha256);base=compile_plan(measurements,bindings)
    require(prompt_semantics['generation']=={'max_tokens':256,'temperature':1.} and
            prompt_semantics['enable_thinking'] is False and prompt_semantics['assistant_prefix']=='<|im_start|>assistant\n',
            'Calibration prompt/decoding semantics differ from frozen native C contract')
    total=sum(lookup(r) for r in rows);require(total==25404,'Frozen C76 input-token sum changed')
    cap=4*total+76*4*256;require(cap==179440,'Calibration worst-case budget changed')
    return {'schema':'common-B28-training-calibration-v1','status':'COMPILED_NOT_AUTHORIZED',
        'common_bindings':bindings,'adapter_config_sha256':digest(Path(adapter).parent/'adapter_config.json'),
        'manifest_sha256':digest(train),'measurements_sha256':common.measurements_sha256(measurements),
        'native_inputs_sha256':sha(native_inputs),'processor_sha256':measurements['processor_sha256'],
        'source_sha256':source_hashes(),'row_ids':[r['id'] for r in rows],
        'row_sha256':[canonical_sha(r) for r in rows],'image_sha256':{r['id']:digest(r['images'][0]) for r in rows},
        'prompt_signatures':prompts,'prompt_semantics':prompt_semantics,
        'task_counts':dict(Counter(r['ability'] for r in rows)),'seed':7,'group_size':4,'max_tokens':256,'temperature':1.,
        'input_token_sum':total,'token_cap':cap,'weight_updates':0,
        'statistics':{'unit':'per-prompt mean raw accuracy','quantiles':'numpy linear percentile25/50/75',
                      'zero_IQR':'BLOCK_NO_FLOOR','source':'common initializer TRAINING only'},
        'separate_RL_arm_token_cap':base['budget']['text_visual_token_cap'],
        'limitations':['MedGRPO-style binary mechanistic/null control, not faithful medical-video replication.',
                      'No development/test sampling; calibration costs reported separately from every RL arm.']}


def prepare_calibration(plan,train,measurements,native_inputs):
    require(plan.get('schema')=='common-B28-training-calibration-v1' and plan['source_sha256']==source_hashes(),'Calibration plan/source changed')
    require(plan['seed']==7 and plan['group_size']==4 and plan['max_tokens']==256 and plan['temperature']==1. and plan['weight_updates']==0,'Frozen calibration protocol changed')
    require(common.measurements_sha256(measurements)==plan['measurements_sha256'] and sha(native_inputs)==plan['native_inputs_sha256'],'Measurement/native input receipt changed')
    rows,lookup,prompts=data_contract(Path(train),measurements,native_inputs)
    require(digest(train)==plan['manifest_sha256'] and [r['id'] for r in rows]==plan['row_ids'] and
            [canonical_sha(r) for r in rows]==plan['row_sha256'] and prompts==plan['prompt_signatures'] and
            {r['id']:digest(r['images'][0]) for r in rows}==plan['image_sha256'],'Frozen calibration row order/targets/images/prompts changed')
    require(sum(lookup(r) for r in rows)==plan['input_token_sum']==25404 and plan['token_cap']==179440,'Frozen calibration token budget changed')
    return {'rows':rows,'input_tokens':lookup,'prompts':prompts,'plan':plan}


def runtime_bindings(a,plan,measurements):
    # Existing prompt/common-artifact verifier accepts the same shared keys; calibration does not claim a training step.
    runtime=common.verify_runtime_artifacts(plan,measurements,a.adapter,a.contract,load_contract,a.measurement_contract)
    require(common.prompt_semantics(load_contract(a.contract))==plan['prompt_semantics'],'Runtime native prompt semantics changed')
    require(all(runtime['installed_versions'].values()),'Native package metadata unavailable')
    current={}
    for name,value in measurements['processor_artifacts'].items():
        if name.startswith('runtime_'):
            current[name]=md.version({'runtime_mlx_vlm':'mlx-vlm','runtime_transformers':'transformers'}[name])
        elif name=='native_processor_patch_sha256':
            current[name]=digest(md.distribution('mlx-vlm').locate_file('mlx_vlm/models/qwen3_vl/processing_qwen3_vl.py'))
        else:
            require(Path(name).name==name,'Processor artifact filename must stay in model directory');current[name]=digest(a.model/name)
    require(current==measurements['processor_artifacts'] and canonical_sha(current)==measurements['processor_sha256'],'Native processor/tokenizer/runtime changed')
    from mri_rl.continuous_rl_pilot import cache_binding_for
    cache=cache_binding_for(a);require(cache and cache['certificate_sha256']==plan['common_bindings']['cache_certificate_sha256'],'Common B28 cache certificate differs')
    from mri_rl.paced_resources import PACED_POLICY
    runtime.update({'model_path':str(a.model),'adapter_path':str(a.adapter),'derived_cache':cache,
                    'processor_artifacts':current,'resource_policy':dict(PACED_POLICY)})
    return runtime


def review_scope(prepared,*,output,runtime,resume):
    output=Path(output)
    if resume:
        require(output.is_dir(),'Calibration resume output missing')
        state={str(p.relative_to(output)):digest(p) for p in sorted(output.rglob('*')) if p.is_file()}
    else:require(not output.exists(),'Fresh calibration output already exists');state=None
    p=prepared['plan']
    return {'kind':'common_B28_calibration_readonly_v1','execution_tier':'B','plan_sha256':sha(p),
        'source_sha256':source_hashes(),'runtime':runtime,'output':str(output.resolve()),'resume':resume,
        'starting_output_state':state,'training_rows':76,'ordered_rows_sha256':sha(p['row_ids']),
        'seed':7,'group_size':4,'temperature':1.,'max_tokens':256,'maximum_native_input_generated_tokens':179440,
        'weight_updates':0,'costs_separate_from_RL_arms':True}


def native_backend_factory(prepared,*,adapter,model,contract,cache,before_unit):
    """Wrapper only: unchanged native rollout and step0 restore, checked against frozen C inputs."""
    from mri_rl.mlx_continuous_gspo import NativeGSPOBackend
    class NativeCalibrationBackend(NativeGSPOBackend):
        def __init__(self):
            self.calibration_row=None
            super().__init__(adapter,{'group_size':4,'max_tokens':256},model_path=model,contract=contract,seed=7,
                             before_unit=before_unit,cache=cache)
            original_before=self.before_unit;original_prepare=self.prepare_inputs
            def guarded_before(name):
                result=original_before(name)
                if self.calibration_row is not None:self.check_image()
                return result
            def checked_prepare(*args,**kwargs):
                result=original_prepare(*args,**kwargs);self.check_image()
                expected=prepared['prompts'][self.calibration_row['id']]
                require(result['input_ids'].tolist()==expected['input_token_ids'] and
                        result['image_grid_thw'].tolist()==expected['image_grid_thw'],'Native calibration prepared input/grid differs')
                return result
            self.before_unit=guarded_before;self.prepare_inputs=checked_prepare
        def check_image(self):
            row=self.calibration_row
            require(row is not None and digest(row['images'][0])==prepared['plan']['image_sha256'][row['id']],
                    'Frozen calibration image changed after resource callback')
        def rollout(self,row,index,directory):
            self.calibration_row=row
            try:
                self.check_image();raw=super().rollout(row,index,directory);self.check_image();return raw
            finally:self.calibration_row=None
        def parameter_fingerprint(self):
            import numpy as np
            pairs=sorted(self.tree_flatten(self.model.trainable_parameters()));require(len(pairs)==504,'Common-B28 adapter inventory changed')
            h=hashlib.sha256()
            for name,value in pairs:
                self.mx.eval(value)
                h.update(canonical([name,str(value.dtype),list(value.shape)]).encode())
                a=np.asarray(value.view(self.mx.uint16) if value.dtype==self.mx.bfloat16 else value);h.update(a.tobytes())
            return h.hexdigest()
    return NativeCalibrationBackend


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('compile','scope','train-readonly'))
    for name in ('train','measurements','native-inputs','adapter','model','contract','plan'):
        p.add_argument('--'+name,type=Path,required=True)
    for name in ('measurement-contract','cache-dir','cache-certificate','output','scope-output','review-verdict'):
        p.add_argument('--'+name,type=Path)
    p.add_argument('--cache-certificate-sha256',required=True);p.add_argument('--expected-review-sha256');p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    for name,value in vars(a).items():
        if isinstance(value,Path):setattr(a,name,value.resolve())
    require(a.plan.is_relative_to(ROOT/'local_data'),'Calibration plan must remain private')
    measure=json.loads(a.measurements.read_text());native=json.loads(a.native_inputs.read_text())
    if a.command=='compile':
        contract=load_contract(a.contract)
        plan=compile_calibration_plan(a.train,measure,native,a.adapter,a.cache_certificate_sha256,common.prompt_semantics(contract))
        # Exact file or approved prompt-semantics bridge, before the plan is presented for review.
        common.verify_runtime_artifacts(plan,measure,a.adapter,a.contract,load_contract,a.measurement_contract)
        a.plan.parent.mkdir(parents=True,exist_ok=True);write_json(a.plan,plan);sync_dir(a.plan.parent)
        print(json.dumps({'status':'COMPILED_NOT_AUTHORIZED','plan_sha256':sha(plan),'training_rows':76,'group_size':4,'maximum_tokens':179440}));return
    require(a.output is not None and a.output.is_relative_to(ROOT/'local_data') and a.cache_dir is not None and a.cache_certificate is not None,'Exact private output/cache paths required')
    plan=json.loads(a.plan.read_text());prepared=prepare_calibration(plan,a.train,measure,native)
    runtime=runtime_bindings(a,plan,measure);scope=review_scope(prepared,output=a.output,runtime=runtime,resume=a.resume)
    if a.command=='scope':
        require(a.scope_output is not None and a.scope_output.is_relative_to(ROOT/'local_data'),'Private calibration scope output required')
        write_json(a.scope_output,scope)
        print(json.dumps({'status':'SCOPE_NOT_AUTHORIZED','scope_sha256':sha(scope),'training_rows':76,'maximum_tokens':179440,'weight_updates':0}));return
    _check_review(review_verdict_path=a.review_verdict,expected_review_sha256=a.expected_review_sha256,scope=scope)
    from mri_rl.paced_resources import PacedGate
    from mri_rl.mlx_continuous_gspo import is_loading_unit
    cache={'dir':str(a.cache_dir)};gate=PacedGate(a.output.with_name(a.output.name+'_resource_events.jsonl'))
    def before(name):return gate.before_unit(name,loading=is_loading_unit(name,cache))
    factory=native_backend_factory(prepared,adapter=a.adapter,model=a.model,contract=load_contract(a.contract),cache=cache,before_unit=before)
    result=cal.run_calibration(rows=prepared['rows'],backend_factory=factory,authorize=lambda c:before('authorize_'+c['phase']),
        output=a.output,group_size=4,input_tokens=prepared['input_tokens'],resume=a.resume,max_tokens=256,
        prompt_signatures=prepared['prompts'],bindings={'starting_adapter_sha256':plan['common_bindings']['starting_adapter_sha256'],
            'plan_sha256':sha(plan),'runtime_sha256':sha(runtime),'seed':7})
    result.update({'plan_sha256':sha(plan),'approved_scope_sha256':sha(scope),'approved_review_sha256':a.expected_review_sha256,
                   'journal_binding_sha256':digest(a.output/'binding_PRIVATE.json'),
                   'method':'MedGRPO-style binary mechanistic/null control; not faithful medical-video replication'})
    if result['complete']:
        if result['calibrated_arms_unblocked']:
            path=a.output/'calibration_for_arms_PRIVATE.json';value=cal.calibration_for_arms(result['stats'])
            if path.exists():require(json.loads(path.read_text())==value,'Frozen calibration statistics changed')
            else:write_json(path,value)
            result['calibration_file_sha256']=digest(path)
        result['final_row_receipt_sha256']=digest(a.output/'row_0075/receipt.json')
    receipt=a.output/('calibration_complete_SAFE.json' if result['complete'] else 'calibration_incomplete_SAFE.json')
    if not receipt.exists():write_json(receipt,result);sync_dir(a.output)
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
