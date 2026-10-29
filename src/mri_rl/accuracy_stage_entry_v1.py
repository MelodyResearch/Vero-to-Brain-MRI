"""CPU exact-scope preparation for four whole TRAIN rounds, then frozen DEV40.

Preparation reads source/configuration/receipt metadata only. Missing connectors
produce BLOCKED, never an executable scope. A conditional cache scope may be
included before cache completion; its own fixed source/limits/expected hashes are
bound. No preparation result, source-readiness string or battery waiver authorizes
native work. Execution rederives the scope, authenticates distinct review and
fresh human artifacts, then (and only then) imports the new native bridge.
"""
from __future__ import annotations
import argparse
import ast
import importlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from mri_rl import accuracy_recovery_v1 as authority
from mri_rl import exposure_rounds_v1 as exposure
from mri_rl import multistep_gspo_core_v1 as multi
from mri_rl import shared_rollout_gspo_eval_v1 as evaluation
from mri_rl import accuracy_stage_resources_v1 as resources

ROOT=Path(__file__).resolve().parents[2]
FROZEN_PARENT=authority.FROZEN_PARENT
IMPLEMENTED_BY='Codex/root/rc_data'
ACTION='accuracy_stage_v1_cache_then_four_round_AB_then_DEV40'
KIND='FUTURE_EXACT_ACCURACY_STAGE_V1_COMPOUND_SCOPE'
STATUS='PREPARED_EXACT_SCOPE_PENDING_REVIEW_AND_FRESH_HUMAN'
NATIVE_MODULE='mri_rl.accuracy_stage_native_v1'
INPUT_KEYS=('stage_input','runtime_spec','native_contract','cache_contract','evaluation_contract','evaluation')
RUNTIME_KEYS=('parent_checkpoint','parent_receipt_sha256','calibration','model','cache_dir','contract','global_lock','runtime_source_sha256')
OWN_SOURCES=('src/mri_rl/accuracy_stage_entry_v1.py','scripts/run_accuracy_stage_v1.py',
 'src/mri_rl/exposure_rounds_v1.py','src/mri_rl/paired_z045_exposure_v2.py',
 'src/mri_rl/paired_z045_data_v1.py','src/mri_rl/multistep_gspo_core_v1.py',
 'src/mri_rl/shared_rollout_gspo_eval_v1.py','src/mri_rl/accuracy_recovery_v1.py')
RESOURCE_POLICY={'this_round_only':True,'battery_percent_min':20,'unknown_power_or_battery':'HOLD',
 'global_wall_seconds':10800,'train_wall_seconds_per_arm':3600,'cache_wall_seconds':1800,
 'one_model':True,'one_image_graph':True,'no_global_policy_mutation':True,
 'model_policy':resources.MODEL_POLICY,'original_swap_baseline_gib':resources.BASELINE_SWAP_GIB}
require=authority.require
scope_digest=authority.scope_digest
sha=authority.sha
file_sha256=authority.file_sha256


def _load(path):
    require(not any(t in str(path).lower() for t in ('final100','holdout16','new16')),'Forbidden stage input path')
    return json.loads(Path(path).read_text())


def _metadata_hash(path):
    p=Path(path)
    require(p.suffix not in ('.safetensors','.npz','.npy','.bin','.png','.gz'),'Only source/metadata hashing permitted by preparation')
    return file_sha256(p)


def _flag_free(value):
    if isinstance(value,dict):
        return all(k not in authority.APPROVAL_KEYS and _flag_free(v) for k,v in value.items())
    if isinstance(value,list):return all(_flag_free(v) for v in value)
    return True


def _closure(closures):
    result={}
    for closure in closures:
        require(isinstance(closure,dict) and closure,'Nonempty source closure required')
        for path,digest in closure.items():
            require(Path(path).is_absolute() and sha(digest),'Absolute source path/hash required')
            require(path not in result or result[path]==digest,'Conflicting source bindings')
            require(Path(path).is_file() and _metadata_hash(path)==digest,'Source closure changed: '+Path(path).name)
            result[path]=digest
    return result


def _functions(closure,basename):
    paths=[Path(p) for p in closure if Path(p).name==basename]
    if len(paths)!=1:return set()
    return {n.name for n in ast.parse(paths[0].read_text()).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}


def _local_source_closure(initial):
    """Bind static local imports, including deferred imports, without importing them."""
    result=dict(initial);pending=[Path(p) for p in initial if Path(p).suffix=='.py'];seen=set()
    while pending:
        path=pending.pop()
        if path in seen:continue
        seen.add(path)
        for node in ast.walk(ast.parse(path.read_text())):
            names=[]
            if isinstance(node,ast.Import):names=[n.name for n in node.names]
            elif isinstance(node,ast.ImportFrom) and node.module:
                names=[node.module]+[node.module+'.'+n.name for n in node.names]
            for name in names:
                if not name.startswith('mri_rl.'):continue
                target=ROOT/'src'/Path(*name.split('.')).with_suffix('.py')
                if target.is_file() and str(target) not in result:
                    result[str(target)]=_metadata_hash(target);pending.append(target)
    return result


def _output(path,start):
    root=authority.canonical_project_root(start)
    base=root/'local_data/accuracy_stage_v1';p=Path(path)
    require(p.is_absolute() and str(p)==str(p.resolve()),'Canonical absolute output required')
    require(p.is_relative_to(base) and p!=base and not (root/'local_data').is_symlink() and not base.is_symlink(),'Output must be inside canonical accuracy_stage_v1')
    return str(p)


def _stage(stage,runtime):
    require(stage.get('schema')=='accuracy-stage-four-round-input-v1','Wrong stage capsule schema')
    parent=stage['parent52_metadata_PRIVATE']
    require(parent['Adam_step']==52 and parent['groups']==114 and parent['receipt_sha256']==FROZEN_PARENT['receipt_sha256'] and parent['adapter_sha256_from_existing_metadata']==FROZEN_PARENT['adapter_sha256'],'Parent is not frozen Adam52')
    require(parent['checkpoint_path']==runtime['parent_checkpoint'] and runtime['parent_receipt_sha256']==parent['receipt_sha256'],'Runtime parent differs from capsule')
    receipt=Path(runtime['parent_checkpoint'])/'receipt.json'
    require(receipt.is_file() and _metadata_hash(receipt)==FROZEN_PARENT['receipt_sha256'],'Parent receipt missing or changed')
    r=_load(receipt)
    require(r.get('completed_groups')==114 and r.get('optimizer_updates')==52,'Original parent counters differ')
    capsules={}
    required={'DATA56_INPUT_PRIVATE.json','DALLAS48_INPUT_v2_PRIVATE.json','DEV40_INPUT_PRIVATE.json'}
    require(set(stage.get('input_capsules_PRIVATE',{}))==required,'All three frozen source capsules required')
    for name,b in stage['input_capsules_PRIVATE'].items():
        require(Path(b['path']).is_file() and _metadata_hash(b['path'])==b['sha256'],'Stage source capsule changed')
        capsules[name]=_load(b['path'])
    pool=capsules['DATA56_INPUT_PRIVATE.json']['rows'];dallas=capsules['DALLAS48_INPUT_v2_PRIVATE.json']['rows']
    sites={r['patient_key_PRIVATE']:r['site_PRIVATE'] for r in pool}
    rebuilt=exposure.build_round_plan([r['record_PRIVATE'] for r in pool],[r['record_PRIVATE'] for r in dallas],sites,rounds=4)
    plan=stage['round_plan_PRIVATE'];require(plan==rebuilt,'Plan differs from exact source pool membership/order')
    units=multi.make_units(plan)
    require(len(plan['rounds'])==4 and len(units)==8 and units==stage['units_PRIVATE'],'Exactly four whole rounds and source-derived units required')
    report=exposure.exposure_report(plan)
    require(report['patients_with_both_modalities']==4 and report['patients_with_one_modality']==0 and report['sequence_rows_repeated']==0 and report['dallas_rows_unique']==8,'Fixed complete unique patient pairs/Dallas rows required')
    require(report['k4_groups']=={'mri_sequence':8,'mri_visit_correspondence':8},'Equal16 K4 task exposure required')
    require(stage['prompt_lengths']=={'mri_sequence':213,'mri_visit_correspondence':405},'Fixed conditional213/405 prompt bounds required')
    rows=[c['row'] for u in units for c in u['contexts']]
    require(all(x['extra_info']['internal_split']=='train' for x in rows),'NonTRAIN schedule rejected')
    require(len({x['id'] for x in rows})==16,'Schedule rows must be unique')
    bindings=stage['planned_image_bindings_PRIVATE']
    expected={b['record_id_PRIVATE']:b['image_binding_PRIVATE'] for b in bindings}
    require(len(bindings)==len(expected)==16 and set(expected)=={x['id'] for x in rows},'Exact planned image bindings required')
    for row in rows:
        b=expected[row['id']]
        require(row['images']==[b['path']] and sha(b['sha256']) and type(b['bytes']) is int and b['bytes']>0,'Frozen image metadata/path mismatch')
        require([part['image'] for m in row['prompt'] for part in m['content'] if part['type']=='image']==row['images'],'Typed/top image routing mismatch')
        sources=pool if row['ability']=='mri_sequence' else dallas
        original=next(r for r in sources if r['record_PRIVATE']['id']==row['id'])
        pin=({'path':original['image_path'],'sha256':original['expected_png_sha256'],'bytes':original['PNGbytes']}
             if row['ability']=='mri_sequence' else original['current_image_file_bindings_PRIVATE'][0])
        require(b==pin,'Planned image hash differs from authenticated source capsule')
    lengths=stage['prompt_lengths']
    return {'per_arm':multi.derive_arm_budget(units,lengths,maximum_seconds=3600,common_first_unit=True),
      'shared_first_rollout_once':multi.common_rollout_budget(units[0],lengths),'evaluation':evaluation.eval_budget(3),
      'generation_exposure_per_arm':exposure.describe_costs(plan,prompt_lengths=lengths)},report


def prepare_stage_spec(specpath,*,start=None):
    """Source/config metadata only. BLOCKED has no executable_scope; never imports native bridge."""
    specpath=Path(specpath);spec=_load(specpath)
    require(spec.get('schema')=='accuracy-stage-spec-v1','Stage spec schema required')
    require(_flag_free(spec),'Local approval flags cannot authorize execution')
    base={'schema':'accuracy-stage-preparation-v1','status':'BLOCKED','blockers':[], 'model_loads':0,'optimizer_updates':0}
    blockers=base['blockers'];files={};loaded={}
    pins=spec.get('file_sha256',{})
    for key in INPUT_KEYS:
        if not spec.get(key) or not Path(spec[key]).is_file():blockers.append(key.upper()+'_NOT_PRESENT');continue
        require(sha(pins.get(key)),'Exact metadata hash missing: '+key)
        require(_metadata_hash(spec[key])==pins[key],'Bound input changed: '+key)
        files[spec[key]]=pins[key];loaded[key]=_load(spec[key])
    if blockers:return base
    runtime=loaded['runtime_spec'];missing=[k for k in RUNTIME_KEYS if k not in runtime]
    if missing:blockers.append('RUNTIME_KEYS_NOT_BOUND:'+','.join(missing));return base
    for key in ('parent_checkpoint','model'):
        if not Path(runtime[key]).is_dir():blockers.append(key.upper()+'_NOT_PRESENT')
    for key in ('calibration','contract'):
        if not Path(runtime[key]).is_file():blockers.append(key.upper()+'_NOT_PRESENT');continue
        require(sha(runtime.get('file_sha256',{}).get(key)),'Frozen runtime metadata SHA required: '+key)
        require(_metadata_hash(runtime[key])==runtime['file_sha256'][key],'Runtime metadata changed: '+key)
        files[runtime[key]]=runtime['file_sha256'][key]
    if not runtime['runtime_source_sha256']:blockers.append('RUNTIME_SOURCE_CLOSURE_NOT_BOUND')
    native=loaded['native_contract'];ev=loaded['evaluation_contract'];cache=loaded['cache_contract']
    hooks={'cache','train','evaluate','resource_gate'}
    if native.get('status')!='SOURCE_READY_NOT_NATIVE_VALIDATED' or not hooks.issubset(native.get('wired_hooks',[])):blockers.append('NATIVE_CONNECTORS_NOT_WIRED')
    if native.get('module')!=NATIVE_MODULE or native.get('entrypoint')!='run_stage' or native.get('resource_gate_entrypoint')!='make_resource_gate':blockers.append('NATIVE_ENTRYPOINTS_NOT_BOUND')
    if ev.get('status')!='SOURCE_READY_NOT_NATIVE_VALIDATED' or ev.get('entrypoint')!='run_checkpoint' or ev.get('real_rows')!=40 or ev.get('swap_aliases')!=8:blockers.append('EVALUATION_NOT_WIRED')
    if cache.get('scope_kind')!='FRESH_EXACT_HISTORICAL_GPU_CACHE_V4_CONTINUATION_SCOPE' or not cache.get('expected') or not cache.get('limits'):blockers.append('CACHE_V4_SCOPE_NOT_BOUND')
    if blockers:return base
    own={str(ROOT/name):_metadata_hash(ROOT/name) for name in OWN_SOURCES}
    closure=_local_source_closure(_closure([own,runtime['runtime_source_sha256'],native.get('source_sha256',{}),ev.get('source_sha256',{}),cache.get('source_sha256',{})]))
    if not {'run_stage','make_resource_gate'}.issubset(_functions(native['source_sha256'],'accuracy_stage_native_v1.py')):blockers.append('NATIVE_SOURCE_HOOKS_NOT_PRESENT')
    if not {'run_checkpoint'}.issubset(_functions(ev['source_sha256'],'paired_z045_matched_f32_eval_v1.py')):blockers.append('EVALUATION_SOURCE_HOOK_NOT_PRESENT')
    if blockers:return base
    for binding in loaded['stage_input'].get('input_capsules_PRIVATE',{}).values():
        require(Path(binding['path']).is_file() and _metadata_hash(binding['path'])==binding['sha256'],'Stage source capsule changed');files[binding['path']]=binding['sha256']
    budget,report=_stage(loaded['stage_input'],runtime)
    from mri_rl import paired_z045_matched_f32_eval_v1 as matched
    from mri_rl.paired_z045_evaluation_v1 import LABELS
    evplan=matched.compile_records(loaded['evaluation'])
    tokens=loaded['evaluation'].get('token_plans_PRIVATE',{})
    require(set(tokens)==set(LABELS),'Exact evaluation token plans required')
    for task,t in tokens.items():
        require(t['label_order']==list(LABELS[task]) and len(t['labels'])==2,'Frozen evaluation class order differs')
        for label in t['labels']:matched.score_input_positions(405,len(t['prefix']),len(label))
    devref=_load(loaded['stage_input']['input_capsules_PRIVATE']['DEV40_INPUT_PRIVATE.json']['path'])
    devby={r['original_source_row_sha256']:r for r in devref['rows']}
    require(len(devby)==40,'Unique DEV40 source rows required')
    seen=set()
    for record in evplan['real']:
        key=record['original_source_row_sha256'];require(key in devby and key not in seen,'Evaluation source membership differs');seen.add(key)
        original=devby[key];row=json.loads(json.dumps(record['row_PRIVATE']))
        if row['ability']=='mri_sequence':row['extra_info'].pop('level',None)
        require(row==original['record_PRIVATE'] and record['image_sha256']==original['current_image_file_bindings_PRIVATE'][0]['sha256'],'Evaluation record/target/image differs from frozen DEV40')
    parent_receipt=str(Path(runtime['parent_checkpoint'])/'receipt.json');files[parent_receipt]=FROZEN_PARENT['receipt_sha256']
    require('output' in spec,'Private output required')
    output=_output(spec['output'],start);require(not Path(output).exists(),'Fresh execution output required; no automatic resume')
    files[str(specpath.resolve())]=_metadata_hash(specpath)
    inp=authority.identity({'file_sha256':files,'plan_sha256':loaded['stage_input']['round_plan_PRIVATE']['plan_sha256'],'parent':FROZEN_PARENT,'budget':budget})
    scope={'schema':'accuracy-stage-exact-scope-v1','scope_kind':KIND,'action':ACTION,'implemented_by':IMPLEMENTED_BY,
      'input_sha256':inp,'source_sha256':closure,'file_sha256':files,'budget':{**budget,'cache':cache['limits']},
      'parent':{k:FROZEN_PARENT[k] for k in ('receipt_sha256','adapter_sha256','groups','updates')},
      'output':output,'resource_policy':RESOURCE_POLICY,
      'cache_prerequisite':{'status':'CONDITIONAL_EXECUTE_FIRST','scope_file_sha256':pins['cache_contract'],'scope_sha256':scope_digest(cache),'expected':cache['expected'],'must_match_before_model_load':True},
      'runtime_spec_sha256':pins['runtime_spec'],'native_module':NATIVE_MODULE,
      'evaluation':{'checkpoints':['Parent52','A_terminal','B_terminal'],'real_rows':40,'swap_aliases':8,'input_sha256':pins['evaluation'],'interim_selection':False,'terminal_hashes':'PENDING_DERIVED_FROM_COMMITTED_RECEIPTS'},
      'exposure':{k:v for k,v in report.items() if k not in ('sites_covered','class_rows','dallas_class_rows')},
      'scope_limitation':'Source-ready is not native-PASS; no model/accuracy/generalization evidence. Component wall maxima are subordinate to global10800s.'}
    require(_flag_free(scope),'Approval flags forbidden in launch scope')
    return {**base,'status':STATUS,'executable_scope':scope,'scope_sha256':scope_digest(scope)}


_ISSUER=object()
class Authorization:
    """Issued by exact human+review admission, checked at every native boundary."""
    def __init__(self,issuer,scope,expires):
        require(issuer is _ISSUER,'Authorization cannot be constructed directly')
        self.scope=json.loads(json.dumps(scope));self.scope_sha256=scope_digest(scope)
        self._verified={};self._expires=expires;self._started=time.monotonic()
    def check(self,phase):
        require(scope_digest(self.scope)==self.scope_sha256,'Authorized scope mutated')
        require(datetime.now(timezone.utc)<self._expires,'Human authorization time expired')
        require(time.monotonic()-self._started<RESOURCE_POLICY['global_wall_seconds'],'Global operation-start deadline exhausted')
        # Each call verifies already-hashed file identity. Full content hashes are
        # repeated at explicit safety boundaries or whenever filesystem metadata
        # changes. A changed stamp is never accepted without checking the bytes.
        full=any(marker in phase.lower() for marker in ('admission','stage_start','cache_complete','native_import',
            'model_load','restore_parent','unit','update','matched_eval_row','complete'))
        for path,digest in {**self.scope['source_sha256'],**self.scope['file_sha256']}.items():
            p=Path(path);require(p.is_file(),'Bound source/input missing before '+phase)
            before=p.stat();stamp=(before.st_size,before.st_mtime_ns,before.st_ctime_ns,before.st_ino,before.st_dev)
            if full or self._verified.get(path)!=stamp:
                require(_metadata_hash(p)==digest,'Bound source/input changed before '+phase)
                after=p.stat();nowstamp=(after.st_size,after.st_mtime_ns,after.st_ctime_ns,after.st_ino,after.st_dev)
                require(stamp==nowstamp,'Source/input changed while hashing');self._verified[path]=stamp
        return True


def authorize_stage(scope,human,review,*,consumed):
    require(callable(consumed) and scope.get('scope_kind')==KIND and scope.get('action')==ACTION and _flag_free(scope),'Exact compound scope required')
    require(scope.get('implemented_by')==IMPLEMENTED_BY and scope.get('resource_policy')==RESOURCE_POLICY,'Stage implementer/resource scope differs')
    require(scope.get('parent')=={k:FROZEN_PARENT[k] for k in ('receipt_sha256','adapter_sha256','groups','updates')},'Frozen Adam52 scope parent required')
    binding=scope_digest(scope)
    require(human.get('authorized') is True and human.get('action')==ACTION and human.get('scope_sha256')==binding and sha(human.get('human_instruction_sha256')) and human.get('approved_by'),'Fresh exact human instruction required')
    require(set(human.get('included_actions',[]))=={'cache','training','evaluation'} and human.get('cache_scope_sha256')==scope['cache_prerequisite']['scope_sha256'],'Human instruction must explicitly include exact cache, training and evaluation subscopes')
    try:
        issued,expires=(datetime.fromisoformat(human[k].replace('Z','+00:00')) for k in ('issued_at_utc','expires_at_utc'))
        now=datetime.now(timezone.utc)
        current=issued.tzinfo is not None and expires.tzinfo is not None and issued<=now<expires and (expires-issued).total_seconds()<=86400 and (now-issued).total_seconds()<=1800
    except (KeyError,TypeError,ValueError):current=False
    require(current,'Human authorization time must be current, <=24h duration, and start within1800s')
    require(review.get('verdict') in ('PASS','PASS_WITH_LIMITATIONS') and review.get('scope_sha256')==binding and review.get('reviewer_kind') in ('isolated_codex','claude_code') and review.get('reviewer_id') and review['reviewer_id']!=IMPLEMENTED_BY and review.get('source_sha256')==scope['source_sha256'] and review.get('input_sha256')==scope['input_sha256'],'Independent exact source/input/scope review required')
    token=Authorization(_ISSUER,scope,expires);token.check('admission')
    consumed(human['human_instruction_sha256']);return token


def execute_stage(args,*,start=None):
    required=('spec','scope','expected_scope_sha256','human','expected_human_sha256','review','expected_review_sha256','cache_execution','expected_cache_execution_sha256')
    require(all(getattr(args,k,None) for k in required),'NOT_AUTHORIZED: exact scope/human/review paths and hashes required before native import')
    for key in ('scope','human','review','cache_execution'):
        require(_metadata_hash(getattr(args,key))==getattr(args,'expected_'+key+'_sha256'),'Exact admission artifact changed: '+key)
    prepared=prepare_stage_spec(args.spec,start=start)
    require(not prepared['blockers'],'Stage prerequisites remain blocked')
    scope=_load(args.scope);require(scope==prepared['executable_scope'],'Exact current scope differs from supplied artifact')
    spec=_load(args.spec);human=_load(args.human);cache_execution=_load(args.cache_execution)
    cachekeys={'scope_path','scope_sha256','human_path','human_sha256','review_path','review_sha256','repo_root'}
    require(set(cache_execution)==cachekeys,'Exact cache credential keys required; backend/test overrides forbidden')
    require(cache_execution['scope_path']==spec['cache_contract'] and cache_execution['scope_sha256']==spec['file_sha256']['cache_contract'],'Cache execution must use bound conditional sub-scope')
    for key in ('human','review'):
        require(_metadata_hash(cache_execution[key+'_path'])==cache_execution[key+'_sha256'],'Cache credential hash changed')
    cache_human=_load(cache_execution['human_path'])
    require(cache_human.get('human_instruction_sha256')==human.get('human_instruction_sha256') and cache_human.get('scope_sha256')==scope['cache_prerequisite']['scope_sha256'],'Cache and compound permissions must bind the same fresh human instruction and exact subscope')
    token=authorize_stage(scope,human,_load(args.review),consumed=authority.consume_once(Path(scope['output']).with_name(Path(scope['output']).name+'_authorizations_PRIVATE')))
    token.check('before_native_import')
    bridge=importlib.import_module(NATIVE_MODULE)
    # Credentials are issued after approval, supplied separately, and never mutate the frozen spec.
    native_spec={**spec,'cache_contract':{'path':spec['cache_contract'],'sha256':spec['file_sha256']['cache_contract']},'cache_execution':cache_execution}
    gate=bridge.make_resource_gate(native_spec,scope,token)
    require(callable(gate),'Resource gate factory did not return a callable gate')
    token.check('before_run_stage')
    # Cache and native bridge own their sequential flocks; a second entry flock would self-conflict.
    return bridge.run_stage(native_spec,scope,token,gate)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',type=Path,required=True);parser.add_argument('--scope-output',type=Path)
    parser.add_argument('--execute',action='store_true');parser.add_argument('--scope',type=Path)
    for name in ('human','review','cache-execution'):parser.add_argument('--'+name,type=Path)
    for name in ('scope','human','review','cache-execution'):parser.add_argument('--expected-'+name+'-sha256')
    args=parser.parse_args(argv)
    if args.execute:return execute_stage(args)
    prepared=prepare_stage_spec(args.spec)
    if not prepared['blockers'] and args.scope_output:
        _output(str(args.scope_output),None)
        authority.publish_once(args.scope_output,prepared['executable_scope'])
    safe={k:v for k,v in prepared.items() if k!='executable_scope'}
    print(json.dumps(safe,sort_keys=True));return safe
