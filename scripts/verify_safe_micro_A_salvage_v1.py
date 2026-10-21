"""CPU-only forensic one-step qualification; never adopts a checkpoint or changes its old journal.

Reads authenticated safetensors with bounded per-tensor seek/read, not MLX, Torch
or base-model materialization. Numeric bounds below are fixed before actual use.
"""
from __future__ import annotations
import argparse,ast,copy,hashlib,json,math,struct,sys
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime,timezone
import numpy as np

OPTIMIZER_SHA='a649eeb65133d1a21eec61ab83529badd0ec50938ee6476b1e59f7ee36def1f2'
SCOPE_SHA='da7ef34ccf171922d601bb3160c99758c0873a518e58ea981f39e8f86012c279'
PRIOR_REVIEW_SHA='27697bfaa667158bb9f8856232c244bff992bd3336014237605e15daa57bb1a2'
PROFILE={'name':'Adam','learning_rate':1e-5,'betas':[.9,.999],'eps':1e-8,'bias_correction':False}
BOUNDS={'policy':'PREREGISTERED_F32_FORWARD_ROUNDING_V1','eps32':float(np.finfo(np.float32).eps),
 'm':'8*eps32*(abs(.9*m0)+abs(.1*g))+1e-12',
 'v':'8*eps32*(abs(.999*v0)+abs(.001*g*g))+1e-16',
 'w':'8*eps32*(abs(w0)+abs(lr*m1/(sqrt(v1)+1e-8)))+1e-10',
 'relaxation_after_actual_outcome':False}
DTYPES={'F32':np.dtype('<f4'),'U64':np.dtype('<u8'),'U32':np.dtype('<u4')}
NAMES={'F32':'float32','U64':'uint64','U32':'uint32'}

def require(ok,message):
    if not ok:raise ValueError(message)

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def identity(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def load(path):return json.loads(Path(path).read_text())
def unique_pairs(pairs):
    result={}
    for key,value in pairs:
        require(key not in result,'Duplicate JSON key');result[key]=value
    return result

class TensorFile:
    """Strict header validation and at most one requested tensor allocation per get."""
    def __init__(self,path):
        self.path=Path(path);self.f=self.path.open('rb')
        try:
            prefix=self.f.read(8);require(len(prefix)==8,'Truncated header length')
            n=struct.unpack('<Q',prefix)[0];require(0<n<=8*1024**2,'Unbounded header')
            self.header=json.loads(self.f.read(n),object_pairs_hook=unique_pairs);self.start=8+n
            self.keys=set(self.header)-{'__metadata__'};ranges=[]
            for key in self.keys:
                v=self.header[key];require(v['dtype'] in DTYPES,'Unsupported forensic dtype')
                shape=v['shape'];require(isinstance(shape,list) and all(type(x) is int and x>=0 for x in shape),'Invalid shape')
                a,b=v['data_offsets'];require(type(a) is int and type(b) is int and 0<=a<=b,'Invalid offsets')
                require(b-a==math.prod(shape)*DTYPES[v['dtype']].itemsize,'Shape/byte length differs');ranges.append((a,b))
            cursor=0
            for a,b in sorted(ranges):require(a==cursor,'Overlapping or noncontiguous tensor payload');cursor=b
            require(self.start+cursor==self.path.stat().st_size,'Truncated or excess payload')
        except BaseException:self.f.close();raise
    def __enter__(self):return self
    def __exit__(self,*unused):self.f.close()
    def get(self,key):
        v=self.header[key];a,b=v['data_offsets'];self.f.seek(self.start+a);raw=self.f.read(b-a)
        require(len(raw)==b-a,'Tensor changed/truncated during read')
        return np.frombuffer(raw,dtype=DTYPES[v['dtype']]).reshape(v['shape'])


def layout_index(layout,tensors):
    result={};used=set()
    def visit(node,path):
        kind=node['kind']
        if kind=='dict':
            names=[k for k,_ in node['items']];require(len(names)==len(set(names)),'Duplicate layout path')
            for k,v in node['items']:visit(v,path+(k,))
        elif kind in ('list','tuple'):
            for i,v in enumerate(node['items']):visit(v,path+(str(i),))
        elif kind=='scalar':
            value=node['value'];require(value is None or type(value) in (str,bool,int,float),'Invalid scalar')
            require(type(value) is not float or math.isfinite(value),'Nonfinite scalar');result[path]=node
        elif kind=='tensor':
            key=node['key'];require(key in tensors.keys and key not in used,'Missing/duplicate layout tensor');used.add(key)
            header=tensors.header[key]
            require(node['shape']==header['shape'] and node['dtype'].removeprefix('mlx.core.')==NAMES[header['dtype']],'Layout dtype/shape differs')
            result[path]=node
        else:raise ValueError('Unknown layout node')
    visit(layout,());require(used==tensors.keys,'Unreferenced checkpoint tensor');return result


def numpy_adam(w,g,m,v,lr,oracle=None):
    state={'m':m,'v':v}
    if oracle is None:
        mm=np.float32(.9)*m+np.float32(.1)*g;vv=np.float32(.999)*v+np.float32(.001)*np.square(g)
        return {'m':mm,'v':vv,'w':w-np.float32(lr)*mm/(np.sqrt(vv)+np.float32(1e-8))}
    opt=SimpleNamespace(learning_rate=np.array(lr,np.float32),betas=[.9,.999],eps=1e-8,bias_correction=False,step=np.array(53,np.uint64))
    result=oracle(opt,g,w,state);return {'w':result,'m':state['m'],'v':state['v']}


def compare_step(w,g,m,v,wa,ma,va,lr,oracle=None):
    arrays=(w,g,m,v,wa,ma,va)
    require(all(x.dtype==np.dtype('float32') and x.shape==w.shape and np.isfinite(x).all() for x in arrays),'Nonfinite or mismatched F32 array')
    require((v>=0).all() and (va>=0).all(),'Negative second moment')
    expected=numpy_adam(w,g,m,v,lr,oracle);e=float(np.finfo(np.float32).eps)
    delta=np.float64(lr)*expected['m'].astype(np.float64)/(np.sqrt(expected['v'].astype(np.float64))+1e-8)
    bounds={'m':8*e*(abs(.9*m.astype(np.float64))+abs(.1*g.astype(np.float64)))+1e-12,
            'v':8*e*(abs(.999*v.astype(np.float64))+abs(.001*np.square(g.astype(np.float64))))+1e-16,
            'w':8*e*(abs(w.astype(np.float64))+abs(delta))+1e-10}
    result={'pass':True}
    for key,actual in (('w',wa),('m',ma),('v',va)):
        require(np.isfinite(expected[key]).all(),'Nonfinite recurrence')
        error=abs(actual.astype(np.float64)-expected[key].astype(np.float64));bound=bounds[key]
        entry={'max_abs_error':float(error.max(initial=0)),'max_error_over_bound':float((error/bound).max(initial=0)),
               'failed_elements':int(np.count_nonzero(error>bound)),'actual_min':float(actual.min()),'actual_max':float(actual.max())}
        result[key]=entry;result['pass'] &= entry['failed_elements']==0
    return result


def pinned_adam_oracle(path):
    require(sha(path)==OPTIMIZER_SHA,'Installed optimizer source hash differs')
    tree=ast.parse(Path(path).read_text());adam=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Adam')
    node=copy.deepcopy(next(n for n in adam.body if isinstance(n,ast.FunctionDef) and n.name=='apply_single'))
    node.returns=None
    for arg in node.args.args:arg.annotation=None
    namespace={'mx':SimpleNamespace(square=np.square,sqrt=np.sqrt)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'<pinned-Adam-CPU-NumPy>','exec'),namespace)
    return namespace['apply_single']


def rng_equal(a_index,a_file,r_index,r_file):
    aa={p[1:]:n for p,n in a_index.items() if p[:1]==('rng',)}
    require(set(aa)==set(r_index),'R1 RNG leaf inventory differs')
    for path,node in aa.items():
        other=r_index[path];require(node['kind']==other['kind'],'R1 RNG layout differs')
        if node['kind']=='scalar':require(node['value']==other['value'],'Python RNG differs from R1')
        else:
            require(node['shape']==other['shape'] and node['dtype']==other['dtype'],'MLX RNG layout differs')
            require(np.array_equal(a_file.get(node['key']),r_file.get(other['key'])),'MLX RNG bits differ from R1')
    return len(aa)


def qualify(specpath,scopepath,priorpath):
    require(sha(scopepath)==SCOPE_SHA and sha(priorpath)==PRIOR_REVIEW_SHA,'Exact prior scope/review required')
    scope=load(scopepath);spec=load(specpath);prior=load(priorpath);R=Path(__file__).resolve().parents[3];S=Path(__file__).resolve().parents[1]
    B=Path(spec['output']);T=B/'stage3_PRIVATE/training_PRIVATE';A=T/'arm_A';common=T/'common_PRIVATE';checkpoint=A/'checkpoint_PRIVATE'
    runtime=load(spec['runtime_spec']);parent=Path(runtime['parent_checkpoint']);receipt=load(A/'receipt_PRIVATE.json');pr=load(parent/'receipt.json')
    files={}
    def bind(path,expected=None):
        path=Path(path);h=sha(path);require(expected is None or h==expected,'Input hash differs: '+str(path));files[str(path)]=h;return h
    bind(scopepath,SCOPE_SHA);bind(priorpath,PRIOR_REVIEW_SHA)
    for p,h in scope['file_sha256'].items():bind(p,h)
    for relative in ('src/mri_rl/safe_micro_training_v1.py','src/mri_rl/safe_micro_training_v2.py','src/mri_rl/shared_rollout_gspo_backend_v1.py','src/mri_rl/paired_z045_continuation_driver_v1.py'):
        path=S/relative;bind(path,scope['source_sha256'][str(path)])
    bind(parent/'receipt.json',runtime['parent_receipt_sha256'])
    require(pr['optimizer_updates']==52 and pr['completed_groups']==114,'Wrong original parent')
    for name in ('adapter.safetensors','adapter_config.json','optimizer/manifest.json','optimizer/state.safetensors'):bind(parent/name,pr['files_sha256'][name])
    bind(A/'receipt_PRIVATE.json',prior['evidence_FILE_sha256']['arm_A_receipt']);bind(A/'gradient_evidence_PRIVATE.json',prior['evidence_FILE_sha256']['gradient_evidence'])
    require(receipt['receipt_sha256']==identity({k:v for k,v in receipt.items() if k!='receipt_sha256'}),'A receipt digest differs')
    checkpoint_files={name:bind(checkpoint/name,h) for name,h in receipt['files_sha256'].items()}
    checkpoint_files['adapter_config.json']=bind(checkpoint/'adapter_config.json')
    gradient=load(A/'gradient_evidence_PRIVATE.json');bind(A/'joint_gradient_float32_PRIVATE.safetensors',gradient['joint_gradient_sha256'])
    jp=T/'journal_PRIVATE';binding=load(jp/'binding_PRIVATE.json');commit=load(jp/'common/commit_PRIVATE.json');reservation=jp/'common/reservation_PRIVATE.json'
    common_commit_sha=bind(jp/'common/commit_PRIVATE.json',prior['evidence_FILE_sha256']['common_commit']);bind(jp/'binding_PRIVATE.json');bind(reservation)
    require(commit['reservation_sha256']==files[str(reservation)] and commit['record_sha256']==identity({k:v for k,v in commit.items() if k!='record_sha256'}),'Common commit differs')
    require(load(reservation)['binding_sha256']==identity(binding),'Common binding differs')
    for p,h in commit['files'].items():require(Path(p).is_relative_to(common),'Foreign common artifact');bind(p,h)
    record=load(common/'common_record_PRIVATE.json');require(record==commit['value']['record'] and record['receipt_sha256']==receipt['common_receipt_sha256'],'Common receipt differs')
    require(record['parent_state']==binding['parent'] and record['rng_after_all16_digest']==receipt['rng_digest_after'],'Parent/R1 receipt differs')
    require(not (jp/'arm_A/commit_PRIVATE.json').exists() and (jp/'arm_A/reservation_PRIVATE.json').is_file(),'Old outer journal must remain incomplete')
    bind(jp/'arm_A/reservation_PRIVATE.json',prior['evidence_FILE_sha256']['arm_A_reservation'])
    require(prior['verified_actual_outcome']['A_actual_optimizer_updates']==1 and prior['verified_actual_outcome']['A_valid_completed_arm'] is False,'Prior actual-outcome evidence differs')
    require(receipt['updated'] and receipt['adam_step_after']==53 and receipt['rng_terminal']=='R1_POST_ALL16_GENERATION','Wrong A boundary')
    installed=R/'work/accuracy-recovery-venv-v1/lib/python3.13/site-packages/mlx/optimizers/optimizers.py';bind(installed,OPTIMIZER_SHA);oracle=pinned_adam_oracle(installed)
    profile_tree=ast.parse((S/'src/mri_rl/paired_z045_continuation_driver_v1.py').read_text());profile=ast.literal_eval(next(n.value for n in profile_tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PROFILE' for t in n.targets)))
    require(profile['optimizer']==PROFILE and profile['scheduler']=={'type':'constant'},'Adam profile differs')
    pm=load(parent/'optimizer/manifest.json');am=load(checkpoint/'optimizer/manifest.json');rm=load(common/'rng_after_all16/rng.json')
    require(pm['optimizer_updates']==52 and am['optimizer_updates']==53,'Manifest step differs')
    require(pm['bindings']['trainable_signature']==am['bindings']['trainable_signature'],'Trainable signature changed')
    require(am['tensors_sha256']==checkpoint_files['optimizer/state.safetensors'] and am['bindings']['adapter_sha256']==checkpoint_files['adapter.safetensors'],'A manifest payload binding differs')
    rows=[];norm2=0.;max_live_input_bytes=0
    with ExitStack() as stack:
        pweights,pstate,aweights,astate,grads,rng=[stack.enter_context(TensorFile(p)) for p in (parent/'adapter.safetensors',parent/'optimizer/state.safetensors',checkpoint/'adapter.safetensors',checkpoint/'optimizer/state.safetensors',A/'joint_gradient_float32_PRIVATE.safetensors',common/'rng_after_all16/rng.safetensors')]
        names=set(am['bindings']['trainable_signature']);require(len(names)==504 and pweights.keys==aweights.keys==grads.keys==names,'504 weight/gradient inventory differs')
        require(len(pstate.keys)==len(astate.keys)==1011,'Full state1011 tensor inventory differs')
        pi=layout_index(pm['layout'],pstate);ai=layout_index(am['layout'],astate);ri=layout_index(rm['layout'],rng)
        require(set(pi)==set(ai),'Full state leaf paths differ')
        for path in pi:
            pn,an=pi[path],ai[path];require(pn['kind']==an['kind'],'Full state kind differs')
            if pn['kind']=='tensor':require(pn['dtype']==an['dtype'] and pn['shape']==an['shape'],'Full state shape/dtype differs')
            elif path[:1]!=('rng',):require(pn['value']==an['value'],'Non-RNG state scalar differs')
        for index,reader,step in ((pi,pstate,52),(ai,astate,53)):
            require(int(reader.get(index[('optimizer','step')]['key']))==step,'Actual step array differs')
            require(math.isclose(float(reader.get(index[('optimizer','learning_rate')]['key'])),1e-5,rel_tol=1e-7),'Actual learning rate differs')
        rng_leaf_count=rng_equal(ai,astate,ri,rng)
        optimizer_tensor_paths={('optimizer','step'),('optimizer','learning_rate')}
        for name in sorted(names):
            stem=('optimizer',)+tuple(name.split('.'));mp=stem+('m',);vp=stem+('v',);optimizer_tensor_paths.update((mp,vp))
            require(mp in pi and vp in pi,'Missing per-weight m/v')
            arrays=[pweights.get(name),grads.get(name),pstate.get(pi[mp]['key']),pstate.get(pi[vp]['key']),aweights.get(name),astate.get(ai[mp]['key']),astate.get(ai[vp]['key'])]
            require(list(arrays[0].shape)==am['bindings']['trainable_signature'][name]['shape'],'Signature shape differs')
            result=compare_step(*arrays,1e-5,oracle=oracle);result['tensor']=name;result['elements']=arrays[0].size;rows.append(result)
            norm2+=float(np.sum(arrays[1].astype(np.float64)**2));max_live_input_bytes=max(max_live_input_bytes,sum(x.nbytes for x in arrays));del arrays
        require({p for p,n in ai.items() if p[:1]==('optimizer',) and n['kind']=='tensor'}==optimizer_tensor_paths,'Unexpected optimizer tensors')
    require(abs(math.sqrt(norm2)-gradient['joint_gradient_l2'])<=1e-7,'Saved joint gradient norm differs')
    allpass=all(x['pass'] for x in rows)
    aggregates={k:{'max_abs_error':max(x[k]['max_abs_error'] for x in rows),'max_error_over_bound':max(x[k]['max_error_over_bound'] for x in rows),'failed_elements':sum(x[k]['failed_elements'] for x in rows),'min':min(x[k]['actual_min'] for x in rows),'max':max(x[k]['actual_max'] for x in rows)} for k in ('w','m','v')}
    bind(__file__)
    require(not any(x.split('.')[0] in ('mlx','mlx_vlm','torch') for x in sys.modules),'Native runtime import forbidden')
    return {'schema':'safe-micro-A-salvage-cpu-forensic-v1','created_at_utc':datetime.now(timezone.utc).isoformat(),'reviewer_id':'Codex /root/resource_v2_final_review','reviewer_kind':'isolated_codex',
      'status':'CPU_QUALIFIED_SALVAGE_PENDING_NATIVE_Q_AND_HUMAN' if allpass else 'NOT_ELIGIBLE_CPU_RECURRENCE_MISMATCH','label':'SALVAGED_FROM_INCOMPLETE_TRANSACTION',
      'eligibility':'ELIGIBLE_PENDING_NATIVE_RESTORE_AND_EXPLICIT_ADOPTION_PERMISSION' if allpass else 'NOT_ELIGIBLE',
      'parent_policy_sha256':record['parent_state']['policy'],'parent_adam_digest':record['parent_state']['adam'],'parent_rng_digest':record['parent_state']['rng'],'parent_receipt_sha256':runtime['parent_receipt_sha256'],
      'common_journal_binding':binding,'common_commit_sha256':common_commit_sha,'A_receipt_sha256':files[str(A/'receipt_PRIVATE.json')],'A_checkpoint_files':checkpoint_files,
      'original_outer_journal_committed':False,'old_journal_modified':False,'file_sha256':files,'numeric_policy':BOUNDS,'optimizer_profile':PROFILE,
      'cpu_proof':{'tensor_count':len(rows),'moment_tensor_count':1008,'all_finite':True,'all_second_moments_nonnegative':True,'parent_step':52,'A_step':53,'full_state_tensor_count':1011,'R1_rng_leaf_count_equal':rng_leaf_count,'R1_python_and_MLX_bits_equal':True,'saved_gradient_L2':gradient['joint_gradient_l2'],'recomputed_gradient_L2':math.sqrt(norm2),'all_tensor_recurrences_within_preregistered_bounds':allpass,'component_aggregates':aggregates,'per_tensor':rows,'maximum_simultaneously_held_input_tensor_bytes':max_live_input_bytes},
      'counters':{'actual_optimizer_updates_already_performed':1,'new_optimizer_updates':0,'new_model_loads':0,'native_calls':0},
      'limitations':['CPU NumPy executes exact pinned installed Adam.apply_single AST; predeclared float32 rounding bounds allow CPU/MLX execution-order differences, not arbitrary numerical changes.','This proves consistency of saved adapter/m/v/step/R1 with one saved-gradient update, not correctness of a new native restore or a repaired live transaction.','Real reward/gate/common provenance and failure ordering are authenticated through the exact independently reviewed saved-failure proof and bound immutable artifacts; no image/base model/test outcomes are accessed.','The original A arm journal remains reserved and uncommitted. This new forensic artifact does not retroactively commit, adopt, rerun or authorize anything.','The CPU container-alias mechanism is confirmed separately; the unique native cause of the generic full-state restore failure remains unproven.','Native restoration of full saved state and explicit new human adoption/execution permission remain required. No MRI accuracy gain is established.'],
      'native_restore_qualified':False,'adoption_authorized':False,'additional_native_execution_authorized':False}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True);p.add_argument('--scope',type=Path,required=True);p.add_argument('--prior-review',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args(argv)
    result=qualify(a.spec,a.scope,a.prior_review)
    with a.output.open('x') as f:json.dump(result,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush()
    print(json.dumps({'status':result['status'],'output_FILE_sha256':sha(a.output),'tensor_count':result['cpu_proof']['tensor_count'],'component_aggregates':result['cpu_proof']['component_aggregates']},sort_keys=True));return 0 if result['cpu_proof']['all_tensor_recurrences_within_preregistered_bounds'] else 2

if __name__=='__main__':raise SystemExit(main())
