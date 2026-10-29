"""Stage1 cache-only exact CPU scope. Stages2/3/4/5 have no execution entry here.

Cache v4 source and operation limits remain unchanged. Stage1 requires its own
fresh human permission AND separately bound nested-cache permission from that
same instruction. Cache completion always ends this invocation. Prospective
training/evaluation budget helpers are arithmetic, never permissions or scopes.
No model/MLX import occurs on prepare. No old full-stage entry is imported.
"""
from __future__ import annotations
import argparse,ast,json,time
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
from mri_rl import accuracy_recovery_v1 as authority

ROOT=Path(__file__).resolve().parents[2]
EXTERNAL_LAUNCHER=ROOT/'scripts/launch_safe_micro_v1.py'
IMPLEMENTED_BY='Codex/root/rc_data'
CACHE_SCOPE_FILE_SHA256='e832655fe226607d8c7028c1132345e43ca340039521c817b1f0cbe10b4a7b65'
ACTION='safe_micro_v1_stage1_cache_only_then_stop'
KIND='FRESH_EXACT_SAFE_MICRO_STAGE1_CACHE_ONLY_SCOPE'
require=authority.require
scope_digest=authority.scope_digest
H=authority.file_sha256


def micro_budget():
    """PROSPECTIVE stage3 only: one shared16-response draw, one joint step/arm."""
    positions=8*(213+256-1)+8*(405+256-1)
    return {'stage':3,'status':'PROSPECTIVE_NOT_AUTHORIZED','training':{
      'generated_response_tokens':16*256,'sampler_draws':16*257,'generation_language_model_calls':16*260,
      'generation_vision_embedding_calls':16,'detached_old_forwards':16,'gradient_forwards':32,'backward_calls':32,
      'scorer_language_model_calls':48,'scorer_vision_embedding_calls':48,'teacherforced_input_positions':3*positions,'optimizer_updates':2},
      'generation_charged_tokens':8*(213+256)+8*(405+256),
      'batching':{'K4_groups_per_arm':4,'responses_per_arm':16,'common_generation_once':True,'weight_per_response':1/16,
       'sequence_gradient_weight':.5,'Dallas_gradient_weight':.5,'maximum_joint_updates_per_arm':1,
       'interpretation':'One .5 sequence block gradient + .5 Dallas block gradient before at most one Adam step; same granularity both arms, no advantage forcing or active-response renormalization.'},
      'prompt_bounds':{'mri_sequence':213,'mri_visit_correspondence':405},'actual_length_guard_required':True}


def eval24_budget():
    """PROSPECTIVE separately authorized stage5: frozen24seq, Parent/A/B at most."""
    return {'stage':5,'status':'PROSPECTIVE_NOT_AUTHORIZED','checkpoints':3,'real_queries_per_checkpoint':24,
      'greedy_generations':72,'candidate_score_API_calls':216,'maximum_generated_tokens':18432,
      'teacherforced_input_positions':91800,'generation_language_model_calls':72*258,
      'vision_embedding_calls':72+216,'optimizer_updates':0,'sequence_class_counts_per_checkpoint':[12,12],
      'Dallas_used':False,'additional_swap_calls':0,'no_interim_best_dev_selection':True}


def _load(path):
    require(not any(t in str(path).lower() for t in ('final100','new16','holdout16')),'Forbidden input path')
    return json.loads(Path(path).read_text())


def _hash(path):
    require(Path(path).suffix not in ('.safetensors','.npy','.npz','.png','.gz','.bin'),'Preparation hashes metadata/source only')
    return H(path)


def _binding(binding,files):
    require(isinstance(binding,dict) and set(binding)=={'path','sha256'} and authority.sha(binding['sha256']),'Exact path/SHA binding required')
    p=Path(binding['path']);require(p.is_file() and _hash(p)==binding['sha256'],'Bound metadata changed')
    files[str(p)]=binding['sha256'];return _load(p)


def _closure(initial):
    result=dict(initial);pending=[Path(p) for p in result if Path(p).suffix=='.py'];seen=set()
    while pending:
        path=pending.pop()
        if path in seen:continue
        seen.add(path)
        for node in ast.walk(ast.parse(path.read_text())):
            names=[]
            if isinstance(node,ast.Import):names=[n.name for n in node.names]
            elif isinstance(node,ast.ImportFrom) and node.module:names=[node.module]+[node.module+'.'+n.name for n in node.names]
            for name in names:
                if name.startswith('mri_rl.'):
                    p=ROOT/'src'/Path(*name.split('.')).with_suffix('.py')
                    if p.is_file() and str(p) not in result:result[str(p)]=_hash(p);pending.append(p)
    require(all(Path(p).is_file() and _hash(p)==sha for p,sha in result.items()),'Source closure changed')
    return result


def _output(path,start=None):
    root=authority.canonical_project_root(start);base=root/'local_data/safe_micro_v1';p=Path(path)
    require(p.is_absolute() and str(p)==str(p.resolve()) and p.is_relative_to(base) and p!=base,'Canonical safe_micro_v1 private output required')
    require(not (root/'local_data').is_symlink() and not base.is_symlink(),'Private root symlink forbidden')
    return p


def prepare(specpath,*,start=None):
    spec=_load(specpath);require(spec.get('schema')=='safe-micro-spec-v1','Safe micro spec required')
    stage=spec.get('stage');base={'schema':'safe-micro-preparation-v1','status':'BLOCKED','stage':stage,'phase':spec.get('phase'),'blockers':[],'model_loads':0,'optimizer_updates':0}
    if stage!=1:return {**base,'blockers':['STAGE_'+str(stage)+'_NOT_IMPLEMENTED_NOT_AUTHORIZED']}
    require(spec.get('phase')=='A','Stage1 is cache-only A; old B action invalid')
    if not spec.get('cache_contract') or not spec.get('revoked_full_scope') or not spec.get('output'):
        return {**base,'blockers':['STAGE1_CACHE_REVOCATION_OUTPUT_BINDINGS_REQUIRED']}
    if not EXTERNAL_LAUNCHER.is_file():return {**base,'blockers':['EXTERNAL_EXIT_WITNESS_SOURCE_NOT_PRESENT']}
    files={str(Path(specpath).resolve()):_hash(specpath)}
    cache=_binding(spec['cache_contract'],files);_binding(spec['revoked_full_scope'],files)
    require(spec['cache_contract']['sha256']==CACHE_SCOPE_FILE_SHA256,'Cache operations must retain exact existing v4 scope')
    require(cache.get('scope_kind')=='FRESH_EXACT_HISTORICAL_GPU_CACHE_V4_CONTINUATION_SCOPE','Existing cachev4 scope required')
    require(cache.get('limits') and cache.get('expected') and cache.get('source_sha256'),'Cache limits/expected/source pins required')
    out=_output(spec['output'],start);require(not out.exists(),'Fresh standalone stage1 output; no retry/resume')
    own={str(p):_hash(p) for p in [Path(__file__),ROOT/'scripts/run_safe_micro_v1.py',EXTERNAL_LAUNCHER]}
    sources=_closure({**cache['source_sha256'],**own})
    scope={'schema':'safe-micro-stage1-scope-v1','scope_kind':KIND,'stage':1,'phase':'A','action':ACTION,'implemented_by':IMPLEMENTED_BY,
      'cache_scope_sha256':scope_digest(cache),'cache_scope_file_sha256':spec['cache_contract']['sha256'],
      'input_sha256':authority.identity(files),'source_sha256':sources,'file_sha256':files,'output':str(out),
      'budget':{'cache':cache['limits'],'model_loads':0,'training_updates':0,'generation':0,'forward':0,'backward':0},
      'model_loads':0,'automatic_phase_B':False,'automatic_next_stage':False,
      'external_witness':{'required_before_stage2':True,'path':str(out.with_name(out.name+'_external_witness_PRIVATE')/'actual_exit_PRIVATE.json'),
        'must_bind':'Actual child exit, closed stdout/stderr hashes, peak RSS, post-exit resource sample and lock inspection; no inferred release bytes.'},
      'terminal':'Return only after cache execute has returned through release; save resource sample and STOP. Stage2 requires a separate human scope.',
      'resource_policy':'UNCHANGED exact nested cache-v4 policy; model-stage10GiB gate is not activated in stage1.'}
    return {**base,'status':'PREPARED_STAGE1_PENDING_INDEPENDENT_REVIEW_AND_FRESH_HUMAN','executable_scope':scope,'scope_sha256':scope_digest(scope)}


_ISSUER=object()
class Authorization:
    def __init__(self,issuer,scope,expires):
        require(issuer is _ISSUER,'Admission-issued authorization only')
        self.scope=json.loads(json.dumps(scope));self.scope_sha256=scope_digest(scope);self.phase=scope['phase'];self.expires=expires;self.verified={}
    def check(self,phase):
        require(self.phase=='A' and self.scope.get('stage')==1 and scope_digest(self.scope)==self.scope_sha256,'Stage1 scope changed')
        require(datetime.now(timezone.utc)<self.expires,'Human permission expired')
        full=any(x in phase.lower() for x in ('admission','start','complete','release'))
        for p,sha in {**self.scope['source_sha256'],**self.scope['file_sha256']}.items():
            path=Path(p);require(path.is_file(),'Bound file missing');s=path.stat();stamp=(s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_ino,s.st_dev)
            if full or self.verified.get(p)!=stamp:
                require(_hash(path)==sha,'Bound source/input changed');after=path.stat();require(stamp==(after.st_size,after.st_mtime_ns,after.st_ctime_ns,after.st_ino,after.st_dev),'File changed while hashing');self.verified[p]=stamp
        return True


def authorize(scope,human,review,*,consumed):
    require(scope.get('scope_kind')==KIND and scope.get('stage')==1 and scope.get('phase')=='A' and scope.get('action')==ACTION and scope.get('automatic_next_stage') is False,'Exact standalone stage1 scope required')
    binding=scope_digest(scope)
    require(human.get('authorized') is True and human.get('action')==ACTION and human.get('phase')=='A' and human.get('scope_sha256')==binding and human.get('cache_scope_sha256')==scope['cache_scope_sha256'] and authority.sha(human.get('human_instruction_sha256')) and human.get('approved_by'),'Fresh exact stage1/cache human instruction required')
    try:
        issued,expires=(datetime.fromisoformat(human[k].replace('Z','+00:00')) for k in ('issued_at_utc','expires_at_utc'));now=datetime.now(timezone.utc)
        valid=issued.tzinfo is not None and expires.tzinfo is not None and issued<=now<expires and (expires-issued).total_seconds()<=86400 and (now-issued).total_seconds()<=1800
    except (KeyError,ValueError,TypeError,AttributeError):valid=False
    require(valid,'Fresh bounded human authorization time required')
    require(review.get('verdict') in ('PASS','PASS_WITH_LIMITATIONS') and review.get('reviewer_id') and review['reviewer_id']!=IMPLEMENTED_BY and review.get('reviewer_kind') in ('isolated_codex','claude_code') and review.get('scope_sha256')==binding and review.get('source_sha256')==scope['source_sha256'] and review.get('input_sha256')==scope['input_sha256'],'Independent exact source/input/scope review required')
    require(callable(consumed),'Once-only permission ledger required');token=Authorization(_ISSUER,scope,expires);token.check('admission');consumed(human['human_instruction_sha256']);return token


def execute(args):
    required=('spec','scope','expected_scope_sha256','human','expected_human_sha256','review','expected_review_sha256','cache_execution','expected_cache_execution_sha256')
    require(all(getattr(args,k,None) for k in required),'NOT_AUTHORIZED: exact stage1 and nested cache artifacts required')
    for k in ('scope','human','review','cache_execution'):require(_hash(getattr(args,k))==getattr(args,'expected_'+k+'_sha256'),'Admission artifact changed')
    p=prepare(args.spec);require(not p['blockers'],'Stage1 blocked');scope=_load(args.scope);require(scope==p['executable_scope'],'Current exact preparation differs')
    spec=_load(args.spec);human=_load(args.human);nested=_load(args.cache_execution)
    require(set(nested)=={'scope_path','scope_sha256','human_path','human_sha256','review_path','review_sha256','repo_root'},'Exact nestedcache keys; no backend/test overrides')
    require(nested['scope_path']==spec['cache_contract']['path'] and nested['scope_sha256']==scope['cache_scope_file_sha256'],'Nested scope differs')
    for key in ('human','review'):require(_hash(nested[key+'_path'])==nested[key+'_sha256'],'Nested credential changed')
    nh=_load(nested['human_path']);require(nh.get('human_instruction_sha256')==human['human_instruction_sha256'] and nh.get('scope_sha256')==scope['cache_scope_sha256'],'Same fresh instruction must authorize stage1 and nestedcache')
    out=Path(scope['output']);token=authorize(scope,human,_load(args.review),consumed=authority.consume_once(out.with_name(out.name+'_permission_PRIVATE')))
    token.check('stage1_start')
    # Only cache module is imported. No model bridge, training entry or later stage dispatch.
    from mri_rl import accuracy_recovery_gpu_cache_v4 as cache4
    result=cache4.execute(**nested);token.check('cache_release_returned')
    from mri_rl.resource_supervisor import sample_macos
    sample=sample_macos();token.check('stage1_complete');out.mkdir(parents=True,exist_ok=False)
    record={'status':'PHASE_A_CACHE_ONLY_COMPLETE_STOP','stage':1,'scope_sha256':token.scope_sha256,'cache_scope_sha256':scope['cache_scope_sha256'],
      'human_instruction_sha256':human['human_instruction_sha256'],'cache_execute_returned_after_source_ordered_release':True,
      'allocator_release_bytes':'NOT_MEASURED_BY_WRAPPER','post_release_resource_sample_PRIVATE':asdict(sample),
      'cache_certificate_path_PRIVATE':str(Path(_load(spec['cache_contract']['path'])['roots']['work_root'])/'certificate.json'),
      'automatic_next_stage':False,'model_loads':0,'training_updates':0,'later_stage_permission':False}
    authority.publish_once(out/'complete_PRIVATE.json',record)
    return {'status':record['status'],'stage':1,'model_loads':0,'training_updates':0,'automatic_next_stage':False}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True);p.add_argument('--scope-output',type=Path);p.add_argument('--execute',action='store_true');p.add_argument('--scope',type=Path)
    for name in ('human','review','cache-execution'):p.add_argument('--'+name,type=Path)
    for name in ('scope','human','review','cache-execution'):p.add_argument('--expected-'+name+'-sha256')
    a=p.parse_args(argv)
    if a.execute:return execute(a)
    result=prepare(a.spec)
    if not result['blockers'] and a.scope_output:_output(a.scope_output);authority.publish_once(a.scope_output,result['executable_scope'])
    safe={k:v for k,v in result.items() if k!='executable_scope'};print(json.dumps(safe,sort_keys=True));return safe
