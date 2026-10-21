"""Future native Q->B->DEV24 salvage path. Source/CPU preparation, NOT_RUN.

No cached rollout regeneration, own-old refresh or A optimizer call is exposed.
Q changes only tiny dummy arrays, never live model weights/Adam. B starts from
disk Adam52 and finishes at its committed terminal; no final parent rollback.
Importing this module initializes no MLX or model. Exact new authority required.
"""
import ast,copy,gc,json,time
from pathlib import Path
from mri_rl import safe_micro_training_v1 as old
from mri_rl import safe_micro_training_v2 as durable
from mri_rl import safe_micro_pilot_native_v1 as pilot
from mri_rl import safe_micro_pilot_native_v2 as resumable
from mri_rl import shared_rollout_gspo_core_v1 as shared
from mri_rl.safe_micro_native_v1 import MicroNativeBackend
from mri_rl.safe_micro_restore_v1 import detach_containers
from mri_rl.safe_micro_journal_v1 import Journal,durable_mkdir

require=shared.require
OLD_ARM_SOURCE_SHA256='071b2d2bb1b2a25d342ec6416a443f9d269123afaa343c3899403d11168e26ed'
COPY_OBSERVATIONS_MAX=10000


def clone_parameter_tree(tree,trainable,copy_leaf):
    """Detach every container and copy only named trainable tensor payloads."""
    seen=set()
    def walk(value,path=''):
        if isinstance(value,dict):return {k:walk(v,(path+'.'+str(k)).lstrip('.')) for k,v in value.items()}
        if isinstance(value,list):return [walk(v,(path+'.'+str(i)).lstrip('.')) for i,v in enumerate(value)]
        if isinstance(value,tuple):return tuple(walk(v,(path+'.'+str(i)).lstrip('.')) for i,v in enumerate(value))
        if path in trainable:seen.add(path);return copy_leaf(value)
        return value
    result=walk(tree)
    require(seen==set(trainable),'Full trainable parameter paths must exist')
    return result


def drop_snapshot_references(backend):
    for name in ('_unit_snapshot','_active','_active_modality','active','_recorded'):
        if hasattr(backend,name):setattr(backend,name,None)
    if hasattr(backend,'_rollout_contexts'):backend._rollout_contexts={}


def full_identity(backend):
    from mri_rl import early_native_conditioning_probe_v1 as utility
    from mri_rl.paired_z045_f32_qualification_v1 import _optimizer,_state
    return {'parameters':utility._fingerprints(backend,dict(backend.tree_flatten(backend.model.parameters()))),
            'optimizer':_optimizer(backend,utility),'state':_state(backend,utility),
            'policy':backend.policy_sha(),'adam':backend.adam_digest(),'rng':backend.rng_digest(),
            'optimizer_step':backend.optimizer_step()}


class RecoveryBackend(MicroNativeBackend):
    """Explicit future backend: no snapshot containers are installed as live state.

    Frozen1438 parameter leaves are borrowed; 504 trainable leaves and full
    Adam/RNG leaves use the documented array __copy__ interface, materialized
    serially. Q must verify tiny-array isolation and full restore before B.
    """
    def copy_leaf(self,value):
        if not hasattr(value,'dtype') or not hasattr(value,'shape'):return value
        count=getattr(self,'copy_observations',0)
        require(count<COPY_OBSERVATIONS_MAX,'Native leaf-copy budget exhausted')
        self.copy_observations=count+1
        result=copy.copy(value);self.mx.eval(result);return result
    def restore(self,state,rng=None):
        if rng is None:return self.transaction_restore(state)
        return super().restore(detach_containers(state,copy_leaf=self.copy_leaf),detach_containers(rng,copy_leaf=self.copy_leaf))
    def transaction_snapshot(self):
        from mri_rl import early_native_conditioning_probe_v1 as utility
        from mri_rl.class_margin_aux_v2 import _state
        parameters=self.model.parameters();fingerprints=utility._fingerprints(self,dict(self.tree_flatten(parameters)))
        result={'parameters':clone_parameter_tree(parameters,set(self.trainable_signature()),self.copy_leaf),
                'optimizer':detach_containers(self.optimizer_state(),copy_leaf=self.copy_leaf),
                'rng':detach_containers(self.rng_state(),copy_leaf=self.copy_leaf),
                'parameter_fingerprints':fingerprints,'state':_state(self,utility)}
        self.mx.eval(result['optimizer'],result['rng']);return result
    def transaction_restore(self,snapshot):
        from mri_rl import early_native_conditioning_probe_v1 as utility
        from mri_rl.class_margin_aux_v2 import _state
        self.active=None
        self.model.update(clone_parameter_tree(snapshot['parameters'],set(self.trainable_signature()),self.copy_leaf))
        self.mx.eval(self.model.parameters());self.restore(snapshot['optimizer'],snapshot['rng'])
        require(utility._fingerprints(self,dict(self.tree_flatten(self.model.parameters())))==snapshot['parameter_fingerprints'] and
                _state(self,utility)==snapshot['state'],'Detached rollback full model/Adam/RNG identity mismatch')
        self.mx.clear_cache()
    def load_checkpoint(self,directory):
        from mri_rl.optimizer_checkpoint import load_optimizer_checkpoint
        directory=Path(directory);meta=json.loads((directory/'optimizer/manifest.json').read_text())
        expected={'run':self.stage_binding,'adapter_sha256':shared.file_sha256(directory/'adapter.safetensors'),'trainable_signature':self.trainable_signature()}
        require(meta['bindings']==expected,'Checkpoint binding differs')
        self.model.load_weights(str(directory/'adapter.safetensors'),strict=False)
        state,rng,_=load_optimizer_checkpoint(directory/'optimizer',expected_bindings=expected,load_arrays=self.load_arrays)
        self.restore(state,rng)
    def load_rng(self,directory):
        import random
        from mri_rl.optimizer_checkpoint import unpack_tree
        directory=Path(directory);meta=json.loads((directory/'rng.json').read_text())
        require(shared.file_sha256(directory/'rng.safetensors')==meta['tensors_sha256'],'Persisted R1 payload changed')
        rng=unpack_tree(meta['layout'],self.load_arrays(directory/'rng.safetensors'))
        require(set(rng)=={'python','mlx'} and len(rng['mlx'])==len(self.mx.random.state),'Persisted RNG inventory changed')
        before=self.adam_digest();saved=detach_containers(rng,copy_leaf=self.copy_leaf)
        random.setstate(saved['python'])
        for i,value in enumerate(saved['mlx']):self.mx.random.state[i][:]=value
        self.mx.eval(self.mx.random.state)
        require(random.getstate()==saved['python'] and all(bool(self.mx.array_equal(v,self.mx.random.state[i]).item()) for i,v in enumerate(saved['mlx'])),
                'RNG-only restore differs')
        require(self.adam_digest()==before,'RNG-only restore changed Adam')
    def close(self):
        drop_snapshot_references(self);super().close()


def arm_body_nodes():
    path=Path(old.__file__);require(shared.file_sha256(path)==OLD_ARM_SOURCE_SHA256,'Original micro arm source changed')
    original=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_arm')
    require(isinstance(original.body[-1],ast.Try) and not original.body[-1].handlers and original.body[-1].finalbody,
            'Unexpected original arm transaction shape')
    new=copy.deepcopy(original);new.name='_B_without_final_parent_rollback'
    new.body=new.body[:-1]+new.body[-1].body
    return original,new


def run_B_once(*,backend,units,common,ledger,snapshot,parent,output,before):
    require(backend.optimizer_step()==52 and old._state(backend)==parent,'B must start at full original Adam52')
    require(not (Path(output)/'arm_B').exists(),'Existing B operation is not replayable')
    _,node=arm_body_nodes();namespace=dict(vars(old))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),
                 '<original-micro-arm-B-only-without-final-rollback>','exec'),namespace)
    return namespace[node.name](arm='B',backend=backend,units=units,common=common,ledger=ledger,snapshot=snapshot,
                                parent=parent,output=output,before=before,kind=backend.evidence_kind)


def qualify_restore(backend,before):
    """Actual native Q, but only three tiny dummy arrays are mutated."""
    first=second=source=cloned=None
    try:
        before('recovery_Q_restore_qualification');mx=backend.mx;before_state=full_identity(backend)
        require(len(before_state['parameters'])==1942 and len(backend.trainable_signature())==504 and
                len(before_state['optimizer']['arrays'])==1010 and before_state['optimizer_step']==52,'Exact parent inventory required')
        fixtures=[]
        for name,dtype in (('float32',mx.float32),('uint32',mx.uint32),('uint64',mx.uint64)):
            before('recovery_Q_tiny_copy_'+name)
            source=mx.array([1,2,3],dtype=dtype);mx.eval(source)
            original=source.tolist();cloned=backend.copy_leaf(source);same_before=bool(mx.array_equal(source,cloned).item())
            cloned[0]=9;mx.eval(source,cloned)
            isolated=source.tolist()==original and cloned.tolist()!=original
            require(same_before and isolated,'Native copied leaf aliases its source: '+name)
            fixtures.append({'dtype':name,'copy_values_exact':same_before,'source_unchanged_after_copy_mutation':isolated})
            del source,cloned
        first=backend.snapshot();backend.restore(first)
        require(full_identity(backend)==before_state,'First full detached restore changed parent')
        second=backend.snapshot();backend.restore(second)
        after=full_identity(backend);require(after==before_state,'Second full detached restore changed parent')
        # Snapshot state is independently rehashed before references are dropped.
        from mri_rl import early_native_conditioning_probe_v1 as utility
        from mri_rl.optimizer_checkpoint import pack_tree
        for saved in (first,second):
            layout,arrays=pack_tree({'optimizer':saved['optimizer'],'rng':saved['rng']})
            require({'layout':layout,'arrays':utility._fingerprints(backend,arrays)}==saved['state'],'Saved snapshot mutated during restore')
        first=None;second=None;drop_snapshot_references(backend);gc.collect();mx.clear_cache()
        return {'status':'PASS','evidence_kind':backend.evidence_kind,'full_parameters_checked':1942,'trainable_parameters_checked':504,
                'optimizer_arrays_checked':1010,'tiny_copy_fixtures':fixtures,'complete_snapshots':2,'complete_restores':2,
                'state_before_PRIVATE':before_state,'state_after_PRIVATE':after,'leaf_copy_observations':backend.copy_observations,
                'generation':0,'model_TF':0,'backward':0,'optimizer_updates':0,'accuracy_improvement':'NOT_MEASURED'}
    finally:
        first=second=source=cloned=None
        drop_snapshot_references(backend);gc.collect();backend.mx.clear_cache()


def read_salvage(spec):
    """Read old immutable evidence; never open an old writer lock or repair it."""
    previous=Path(spec['previous_output']).resolve();qualification=json.loads(Path(spec['salvage_qualification']).read_text())
    require(qualification['status']=='CPU_QUALIFIED_SALVAGE_PENDING_NATIVE_Q_AND_HUMAN','CPU salvage qualification required')
    training=previous/'stage3_PRIVATE/training_PRIVATE';journal=training/'journal_PRIVATE';operation=journal/'common'
    binding=json.loads((journal/'binding_PRIVATE.json').read_text())
    require(binding==qualification['common_journal_binding'],'Old common binding differs')
    claim=operation/'reservation_PRIVATE.json';commit=operation/'commit_PRIVATE.json'
    require(shared.file_sha256(commit)==qualification['common_commit_sha256'],'Old common commit changed')
    reservation=json.loads(claim.read_text());record=json.loads(commit.read_text())
    require(reservation['key']=='common' and reservation['binding_sha256']==shared.identity(binding) and
            record['reservation_sha256']==shared.file_sha256(claim) and
            record['record_sha256']==shared.identity({k:v for k,v in record.items() if k!='record_sha256'}),'Old common commit chain changed')
    for filename,h in record['files'].items():
        path=Path(filename);require(path.resolve().is_relative_to(previous) and path.is_file() and shared.file_sha256(path)==h,'Old common file changed')
    a_dir=training/'arm_A';a_receipt=a_dir/'receipt_PRIVATE.json'
    require(shared.file_sha256(a_receipt)==qualification['A_receipt_sha256'],'Saved A receipt changed')
    receipt=json.loads(a_receipt.read_text());pilot.verify_terminal_receipt(receipt,receipt['receipt_sha256'])
    expected=qualification['A_checkpoint_files']
    require(set(expected)=={'adapter.safetensors','adapter_config.json','optimizer/manifest.json','optimizer/state.safetensors'} and
            {k:v for k,v in expected.items() if k!='adapter_config.json'}==receipt['files_sha256'],'Saved A checkpoint hash inventory differs')
    require(receipt['updated'] and receipt['adam_step_after']==53,'Saved A must be one actual update')
    for name,h in expected.items():require(shared.file_sha256(a_dir/'checkpoint_PRIVATE'/name)==h,'Saved A checkpoint changed')
    return qualification,record['value'],receipt,a_dir


def verify_loaded_terminal(backend,receipt):
    actual=full_identity(backend)
    require(len(actual['parameters'])==1942 and len(actual['optimizer']['arrays'])==1010 and len(backend.trainable_signature())==504,
            'Full loaded checkpoint inventory differs')
    require(actual['policy']==receipt['policy_sha_after'] and actual['adam']==receipt['adam_digest_after'] and
            actual['rng']==receipt['rng_digest_after'] and actual['optimizer_step']==receipt['adam_step_after'],
            'Loaded checkpoint full policy/Adam/RNG identity differs from receipt')
    return actual


def release_before_commit(backend,directory):
    drop_snapshot_references(backend);backend.close();gc.collect();backend.mx.clear_cache()
    active=int(backend.mx.get_active_memory())
    require(backend.model is None and active<128*1024**2,'Recovery allocator release failed')
    path=Path(directory)/'operation_release_SAFE.json'
    shared.publish_once(path,{'model_is_none':True,'active_MLX_bytes':active,'release_verified':True,
                             'snapshot_reference_holders_cleared':True})
    return path


def run_phase(stage,*,spec,scope,authorization,resource_gate):
    """New separately authorized Q/B/E child, never an old-stage retry."""
    from mri_rl.safe_micro_pilot_entry_v1 import Authorization,PHASES
    from mri_rl.safe_micro_recovery_entry_v1 import ADOPTION
    require(isinstance(authorization,Authorization) and authorization.parent is not None and authorization.scope==scope and
            scope['execution_stage']==PHASES[stage],'Exact issued recovery child authority required')
    require(resource_gate.authorization is authorization and resource_gate.scope==scope,'Exact recovery resource gate required')
    require(scope.get('adoption')==ADOPTION,'Fresh scope must bind the exact nested conditional A adoption contract')
    authorization.check('recovery_native_import')
    from mri_rl import accuracy_recovery_v1 as recovery
    from mri_rl.accuracy_stage_native_v1 import restore_parent
    from mri_rl.experiment_contract import load_contract
    from mri_rl.two_task_training import make_grader
    from mri_rl.paired_z045_continuation_entry_v1 import run_with_release,verify_images
    runtime=json.loads(Path(spec['runtime_spec']).read_text());micro=json.loads(Path(spec['micro_input']).read_text())
    dev=json.loads(Path(spec['dev24']).read_text());out=Path(scope['output']);durable_mkdir(out)
    binding={'scope':scope['compound_scope_sha256'],'stage':stage,'input':scope['input_sha256']}
    journal=Journal(out/'recovery_journal_PRIVATE',binding,before=resource_gate)
    try:
        journal.assert_recoverable();complete=journal.get('native_complete')
        if complete is not None:return complete
        salvage,common_value,a_receipt,a_dir=read_salvage(spec)
        parent=Path(runtime['parent_checkpoint']);parent_disk=recovery.parent_disk_snapshot(parent)
        anchor=recovery.adam52_parent_anchor(parent,runtime['parent_receipt_sha256'])
        require(anchor['receipt_sha256']==salvage['parent_receipt_sha256'],'Original parent receipt differs')
        parent_identity={k:salvage['parent_'+v] for k,v in (('policy','policy_sha256'),('adam','adam_digest'),('rng','rng_digest'))}
        calibration=json.loads(Path(runtime['calibration']).read_text());reward_sha=shared.file_sha256(runtime['calibration'])
        rows=[c['row'] for u in micro['units_PRIVATE'] for c in u['contexts']]
        images={x['record_id_PRIVATE']:x['image_binding_PRIVATE']['sha256'] for x in micro['planned_image_bindings_PRIVATE']}
        verify_images(rows,images)
        if stage!=2:
            proof=json.loads((Path(spec['output'])/'stage2_PRIVATE/terminal_PRIVATE.json').read_text())
            require(proof['status']=='PASS' and proof['qualification']['status']=='PASS' and proof['release_verified'] is True,
                    'Actual Q restore and release PASS required before adoption/B/evaluation')
        def one(label,operation,checkpoint=None,receipt=None):
            resource_gate('native_model_load')
            directory=out/(label+'_resident_%d_PRIVATE'%time.time_ns());durable_mkdir(directory);resident=[None]
            def call():
                resource_gate.claim_model_load();backend=RecoveryBackend.__new__(RecoveryBackend);resident[0]=backend
                backend.__init__((Path(checkpoint) if checkpoint else parent)/'adapter.safetensors',pilot._configuration(runtime),
                    model_path=Path(runtime['model']),contract=load_contract(Path(runtime['contract'])),seed=13,before_unit=resource_gate,
                    cache={'dir':runtime['cache_dir']})
                backend.copy_observations=0
                backend.configure_stage(grader=make_grader(calibration),expected_images=images,prompt_lengths=micro['prompt_lengths'],
                    work=directory/'rollout_work_PRIVATE',binding={'scope_sha256':authorization.sha,'parent_receipt_sha256':anchor['receipt_sha256']})
                if checkpoint is None:
                    restore_parent(backend,parent,anchor)
                    require(pilot.state_identity(backend)==parent_identity,'Disk parent full identity differs from shared rollout parent')
                else:
                    manifest=json.loads((Path(checkpoint)/'optimizer/manifest.json').read_text())
                    backend.stage_binding=manifest['bindings']['run'];backend.load_checkpoint(checkpoint)
                    actual=verify_loaded_terminal(backend,receipt)
                    shared.publish_once(directory/'actual_checkpoint_full_restore_PRIVATE.json',actual)
                return operation(backend,directory)
            value=run_with_release(call,resident,directory)
            release=json.loads((directory/'release_SAFE.json').read_text());require(release['release_verified'],'Actual model release required')
            resident[0]=None;gc.collect();return value
        if stage==2:
            value=journal.get('Q_restore')
            if value is None:
                def qualify(backend,directory):
                    def operation(_):
                        result=qualify_restore(backend,resource_gate)
                        saved=out/'Q_restore_actual_PRIVATE.json';shared.publish_once(saved,result)
                        release=release_before_commit(backend,directory)
                        return {'value':result,'files':{str(p.resolve()):shared.file_sha256(p) for p in (saved,release)}}
                    return journal.run('Q_restore',operation,reservation={'array_leaf_copies_max':10000,'optimizer_updates':0,'generation':0,'model_TF':0,'backward':0})
                value=one('parent_Q',qualify)
            result={'status':'PASS','qualification':value,'optimizer_updates':0,'adoption':'PERMITTED_ONLY_AFTER_THIS_Q_PASS_AND_EXPLICIT_FRESH_GRANT'}
        elif stage==3:
            value=journal.get('B_once')
            if value is None:
                def train(backend,directory):
                    def operation(_):
                        training_out=out/'B_training_PRIVATE';durable_mkdir(training_out)
                        budget=old.micro_budget(micro['units_PRIVATE'],micro['prompt_lengths']);ledger=old.Ledger(budget,training_out/'counters_PRIVATE.json')
                        holder=[None];common=None
                        try:
                            holder[0]=backend.snapshot();parent_state=old._state(backend)
                            common=durable.restore_common(common_value,micro['units_PRIVATE'],reward_sha,parent_state,anchor['receipt_sha256'],backend)
                            summary=run_B_once(backend=backend,units=micro['units_PRIVATE'],common=common,ledger=ledger,snapshot=holder[0],
                                parent=parent_state,output=training_out,before=resource_gate)
                            require(summary['counters']['optimizer_updates']<=1 and summary['counters']['backward_calls']<=16 and
                                    summary['counters']['teacherforced_input_positions']<=9024,'Recovery B budget exceeded')
                        finally:
                            holder[0]=None;common=None;drop_snapshot_references(backend);gc.collect();backend.mx.clear_cache()
                        release=release_before_commit(backend,directory)
                        return {'value':summary,'files':{**durable.files_under(training_out),str(release.resolve()):shared.file_sha256(release)}}
                    return journal.run('B_once',operation,reservation={'gradient_forwards':16,'backward_calls':16,'teacherforced_input_positions':9024,'optimizer_updates':1,'new_generation':0,'detached_old_forwards':0})
                value=one('fresh_parent_B',train)
            result={'status':'PASS','B':value,'optimizer_updates':int(value['updated']),'new_optimizer_updates':int(value['updated']),
                    'adopted_A_optimizer_updates':1,'adopted_A_receipt_sha256':a_receipt['receipt_sha256'],'common_generated_again':False,'old_scores_refreshed':False}
        elif stage==4:
            b_dir=Path(spec['output'])/'stage3_PRIVATE/B_training_PRIVATE/arm_B'
            b_receipt=json.loads((b_dir/'receipt_PRIVATE.json').read_text())
            b_terminal=json.loads((Path(spec['output'])/'stage3_PRIVATE/terminal_PRIVATE.json').read_text())
            pilot.verify_terminal_receipt(b_receipt,b_terminal['B']['receipt_sha256'])
            choices={'A':pilot.terminal_choice(a_receipt,parent_identity['policy'],a_dir),
                     'B':pilot.terminal_choice(b_receipt,parent_identity['policy'],b_dir)}
            receipts={'A':a_receipt,'B':b_receipt};counter=resumable.CallCounter({'generations':72,'generated_tokens':18432,'LM':18720,'vision':72,'sampler':18504},out/'calls_PRIVATE.json');metrics={}
            for label in ('Parent','A','B'):
                if label!='Parent' and choices[label]['alias_of_parent']:
                    metrics[label]={**metrics['Parent'],'alias_of':'Parent','actual_updates':0};continue
                cached=[journal.get(label+'_row_%02d'%i) for i in range(24)]
                if all(x is not None for x in cached):
                    for row in cached:resumable._check_completed_row(row,counter)
                    metrics[label]=pilot.greedy_metrics([x['item'] for x in cached])
                else:
                    checkpoint=None if label=='Parent' else choices[label]['path']
                    def evaluate(backend,directory):
                        backend.expected_image_sha256={x['row_PRIVATE']['id']:x['image_sha256'] for x in dev['records_PRIVATE']}
                        return pilot._readonly(backend,lambda:resumable._evaluate_rows(backend,label,dev['records_PRIVATE'],journal,counter,
                                               resource_gate,out/(label+'_PRIVATE'),directory),directory)
                    metrics[label]=one(label,evaluate,checkpoint,None if label=='Parent' else receipts[label])
                metrics[label]['actual_updates']=0 if label=='Parent' else int(receipts[label]['updated'])
            result={'status':'PASS','metrics':metrics,'comparison':pilot.pilot_decision(metrics),'calls':counter.used,'candidate_calls':0,
                    'adopted_A_updates':1,'new_B_updates':int(b_receipt['updated']),'interpretation':'Previously explored DEV24 engineering pilot only; Dallas retention and independent-patient generalization NOT_MEASURED.'}
        else:raise ValueError('Only Q2/B3/E4 stages are supported')
        recovery.require_parent_unchanged(parent_disk,parent);authorization.check('recovery_stage_complete')
        result.update(release_verified=True,parent_disk_unchanged=True,stage=stage,resources=resource_gate.peaks())
        return resumable.commit_native_complete(journal,out,result)
    finally:journal.close()
