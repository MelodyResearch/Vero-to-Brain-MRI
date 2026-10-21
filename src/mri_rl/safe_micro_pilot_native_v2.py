"""Dynamic-resource native pilot with durable coarse scientific milestones.

No MLX import on module import. Only issued child authority and matching guard
may enter. Partial claimed native work is never repeated. Native qualification
and accuracy remain NOT_RUN until a fresh exact scope is explicitly approved.
"""
from pathlib import Path
import gc,json,time
from mri_rl import safe_micro_pilot_native_v1 as v1
from mri_rl import safe_micro_training_v2 as training
from mri_rl import shared_rollout_gspo_core_v1 as shared
from mri_rl.safe_micro_journal_v1 import Journal,durable_mkdir

require=shared.require


class CallCounter(v1.CallCounter):
    """Finite calls survive process release/recovery without refund or reset."""
    def __init__(self,limits,path):
        self.limits=dict(limits);self.path=Path(path)
        if self.path.exists():
            saved=json.loads(self.path.read_text())
            require(saved.get('limits')==self.limits and set(saved.get('used',{}))==set(limits),'Frozen call inventory changed')
            require(all(type(v) is int and 0<=v<=limits[k] for k,v in saved['used'].items()),'Invalid persisted call counts')
            self.used=saved['used']
        else:
            self.used={k:0 for k in limits}
            shared.publish_once(self.path,{'limits':self.limits,'used':self.used})
    def charge(self,key,n=1):
        require(key in self.used and type(n) is int and n>=0 and self.used[key]+n<=self.limits[key],'Frozen native-call budget exceeded')
        used={**self.used,key:self.used[key]+n}
        shared.replace_atomic(self.path,{'limits':self.limits,'used':used});self.used=used


def _binding(scope,stage):
    return {'scope':scope['compound_scope_sha256'],'stage':stage,'input':scope['input_sha256']}


def _readonly(backend,run,directory):
    """Reuse existing full parameter/Adam/RNG integrity at unique load sessions."""
    return v1._readonly(backend,run,directory)


def _stage2(backend,micro,dev,out,before):
    # V1 qualification's fixed2 generation/6scores stays unchanged. Its whole
    # operation is journaled: partial work blocks, never reinitializes its counter.
    require(not (Path(out)/'calls_PRIVATE.json').exists(),'Incomplete Stage2 calls cannot be replayed')
    return v1._stage2(backend,micro,dev,out,before)


def _evaluate_rows(backend,label,records,journal,counter,before,out,session_directory):
    """Each entire greedy query+saved Rn is one durable indivisible milestone."""
    output=Path(out);durable_mkdir(output);results=[];last_rng=None;last_digest=None
    for i,record in enumerate(records):
        key=label+'_row_%02d'%i
        cached=journal.get(key)
        if cached is not None:
            _check_completed_row(cached,counter)
            results.append(cached['item']);last_rng=cached['rng_dir'];last_digest=cached['rng_digest'];continue
        if last_rng is not None:
            backend.load_rng(last_rng)
            require(backend.rng_digest()==last_digest,'Recovered row RNG differs')
        row=record['row_PRIVATE']
        def operation(opdir):
            start=time.monotonic()
            actual=v1.generate_one(backend,row,record['image_sha256'],greedy=True,counter=counter,before=before)
            item={'id':record['id'],'truth':row['reward_model']['ground_truth'],'text':actual['response']['text'],
                  'actual_PRIVATE':actual,'wall_seconds':time.monotonic()-start}
            p=output/('row_%02d_PRIVATE.json'%i);shared.publish_once(p,item)
            rng=Path(opdir)/'post_row_RNG_PRIVATE';durable_mkdir(rng);backend.save_rng(rng)
            value={'item':item,'rng_dir':str(rng.resolve()),'rng_digest':backend.rng_digest(),'counters':dict(counter.used),
                   'resident_session':str(Path(session_directory).resolve())}
            return {'value':value,'files':{str(p.resolve()):shared.file_sha256(p),**training.files_under(rng)}}
        value=journal.run(key,operation,reservation={'generated_responses':1,'generated_response_tokens':256})
        results.append(value['item']);last_rng=value['rng_dir'];last_digest=value['rng_digest']
    return v1.greedy_metrics(results)


def _check_completed_row(value,counter):
    require(set(value['counters'])==set(counter.used) and all(type(v) is int and 0<=v<=counter.used[k] for k,v in value['counters'].items()),
            'Recovered row exceeds persisted spent counters')
    p=Path(value['resident_session'])/'release_SAFE.json'
    require(p.is_file() and json.loads(p.read_text()).get('release_verified') is True,'Completed row lacks actual previous allocator release')


def _close_before_commit(backend,directory):
    """Stage2 commits its reusable qualification only AFTER a real release."""
    backend.close();gc.collect();backend.mx.clear_cache();active=int(backend.mx.get_active_memory())
    require(backend.model is None and active<128*1024**2,'Stage2 release before qualification commit failed')
    p=Path(directory)/'qualification_release_SAFE.json'
    shared.publish_once(p,{'model_is_none':True,'active_MLX_bytes':active,'release_verified':True})
    return p


def commit_native_complete(journal,out,result):
    """Pause before claim cannot leave an unbound final result to collide later."""
    complete=Path(out)/'native_complete_PRIVATE.json'
    def commit(_):
        require(not complete.exists(),'Uncommitted native completion file cannot be adopted')
        shared.publish_once(complete,result)
        return {'value':result,'files':{str(complete.resolve()):shared.file_sha256(complete)}}
    return journal.run('native_complete',commit)


def run_phase(stage,*,spec,scope,authorization,resource_gate):
    from mri_rl.safe_micro_pilot_entry_v1 import Authorization,PHASES
    require(isinstance(authorization,Authorization) and authorization.parent is not None and authorization.scope==scope and scope['execution_stage']==PHASES[stage],
            'Exact genuine child-stage authority required')
    require(resource_gate.authorization is authorization and resource_gate.scope==scope,'Matching dynamic resource gate required')
    authorization.check('native_import')
    from mri_rl.safe_micro_native_v1 import MicroNativeBackend
    from mri_rl.accuracy_stage_native_v1 import restore_parent
    from mri_rl import accuracy_recovery_v1 as recovery
    from mri_rl.experiment_contract import load_contract
    from mri_rl.two_task_training import make_grader
    from mri_rl.paired_z045_continuation_entry_v1 import run_with_release,verify_images
    runtime=json.loads(Path(spec['runtime_spec']).read_text());micro=json.loads(Path(spec['micro_input']).read_text())
    dev=json.loads(Path(spec['dev24']).read_text());out=Path(scope['output']);durable_mkdir(out)
    binding=_binding(scope,stage);journal=Journal(out/'native_journal_PRIVATE',binding,before=resource_gate)
    try:
        journal.assert_recoverable()
        done=journal.get('native_complete')
        if done is not None:return done
        parent=Path(runtime['parent_checkpoint']);disk_before=recovery.parent_disk_snapshot(parent)
        anchor=recovery.adam52_parent_anchor(parent,runtime['parent_receipt_sha256'])
        rows=[c['row'] for u in micro['units_PRIVATE'] for c in u['contexts']]
        images={x['record_id_PRIVATE']:x['image_binding_PRIVATE']['sha256'] for x in micro['planned_image_bindings_PRIVATE']}
        verify_images(rows,images)
        config=v1._configuration(runtime);calibration=json.loads(Path(runtime['calibration']).read_text());reward_sha=shared.file_sha256(runtime['calibration'])
        def one(label,operation,checkpoint=None):
            # Loading itself may be abandoned/released; scientific work has not been
            # claimed yet. The root's durable load-attempt cap governs retries.
            resource_gate('native_model_load')
            directory=out/(label+'_resident_%d_PRIVATE'%time.time_ns());durable_mkdir(directory);resident=[None]
            def call():
                resource_gate.claim_model_load()
                backend=MicroNativeBackend.__new__(MicroNativeBackend);resident[0]=backend
                backend.__init__((Path(checkpoint) if checkpoint else parent)/'adapter.safetensors',config,
                    model_path=Path(runtime['model']),contract=load_contract(Path(runtime['contract'])),seed=13,
                    before_unit=resource_gate,cache={'dir':runtime['cache_dir']})
                backend.configure_stage(grader=make_grader(calibration),expected_images=images,prompt_lengths=micro['prompt_lengths'],
                    work=directory/'rollout_work_PRIVATE',binding={'scope_sha256':authorization.sha,'parent_receipt_sha256':anchor['receipt_sha256']})
                if checkpoint is None:restore_parent(backend,parent,anchor)
                else:
                    manifest=json.loads((Path(checkpoint)/'optimizer/manifest.json').read_text())
                    backend.stage_binding=manifest['bindings']['run'];backend.load_checkpoint(checkpoint)
                return operation(backend,directory)
            result=run_with_release(call,resident,directory)
            release=json.loads((directory/'release_SAFE.json').read_text());require(release['release_verified'],'Actual allocator release required')
            resident[0]=None;gc.collect();return result
        if stage==2:
            qualification=journal.get('qualification')
            if qualification is None:
                def operation(backend,directory):
                    def qualify(_):
                        value=_stage2(backend,micro,dev,out,resource_gate)
                        release=_close_before_commit(backend,directory)
                        return {'value':value,'files':{str(p.resolve()):shared.file_sha256(p) for p in (out/'calls_PRIVATE.json',out/'forward_actual_PRIVATE.json',out/'readonly_integrity_PRIVATE.json',release)}}
                    return journal.run('qualification',qualify,reservation={'generated_response_tokens':512,'scorer_forwards':6,'teacherforced_input_positions':2808})
                qualification=one('parent_load',operation)
            result=qualification
        elif stage==3:
            def train(backend,directory):
                return training.execute_micro(backend=backend,units=micro['units_PRIVATE'],prompt_lengths=micro['prompt_lengths'],reward_policy_sha256=reward_sha,
                    output=out/'training_PRIVATE',before=resource_gate,expected_parent=v1.state_identity(backend),checkpoint_receipt_sha256=anchor['receipt_sha256'],
                    verify_parent_disk=lambda:recovery.require_parent_unchanged(disk_before,parent),journal_binding=binding)
            actual=one('training_model',train)
            result={'status':'PASS','training':actual,'optimizer_updates':actual['optimizer_updates']}
        elif stage==4:
            trainout=Path(spec['output'])/'stage3_PRIVATE/training_PRIVATE'
            receipts={arm:json.loads((trainout/('arm_'+arm)/'receipt_PRIVATE.json').read_text()) for arm in ('A','B')}
            completed=json.loads((Path(spec['output'])/'stage3_PRIVATE/terminal_PRIVATE.json').read_text())
            for arm in ('A','B'):v1.verify_terminal_receipt(receipts[arm],completed['training']['arms'][arm]['receipt_sha256'])
            parent_policy=receipts['A']['parent_policy_sha256'];choices={k:v1.terminal_choice(v,parent_policy,trainout/('arm_'+k)) for k,v in receipts.items()}
            counter=CallCounter({'generations':72,'generated_tokens':18432,'LM':18720,'vision':72,'sampler':18504},out/'calls_PRIVATE.json');metrics={}
            for label in ('Parent','A','B'):
                if label!='Parent' and choices[label]['alias_of_parent']:
                    metrics[label]={**metrics['Parent'],'alias_of':'Parent','actual_updates':0};continue
                completed_rows=[journal.get(label+'_row_%02d'%i) for i in range(24)]
                if all(x is not None for x in completed_rows):
                    for value in completed_rows:_check_completed_row(value,counter)
                    metrics[label]=v1.greedy_metrics([x['item'] for x in completed_rows])
                else:
                    checkpoint=None if label=='Parent' else choices[label]['path']
                    def evaluate(backend,directory):
                        backend.expected_image_sha256={x['row_PRIVATE']['id']:x['image_sha256'] for x in dev['records_PRIVATE']}
                        require(backend.policy_sha()==(parent_policy if label=='Parent' else choices[label]['policy_sha256']),'Loaded evaluation policy differs')
                        return _readonly(backend,lambda:_evaluate_rows(backend,label,dev['records_PRIVATE'],journal,counter,resource_gate,out/(label+'_PRIVATE'),directory),directory)
                    metrics[label]=one(label,evaluate,checkpoint)
                metrics[label]['actual_updates']=0 if label=='Parent' else choices[label]['updates']
            result={'status':'PASS','metrics':metrics,'comparison':v1.pilot_decision(metrics),'calls':counter.used,'candidate_calls':0,
                    'interpretation':'Explored4patient DEV24 engineering pilot only; Dallas retention NOT_MEASURED.'}
        else:raise ValueError('Only2/3/4 native stages permitted')
        recovery.require_parent_unchanged(disk_before,parent);authorization.check('stage_complete')
        result.update(release_verified=True,parent_disk_unchanged=True,stage=stage,resources=resource_gate.peaks())
        # No native calls in this final journal operation; results are durable only
        # after every model session returned a real release receipt.
        return commit_native_complete(journal,out,result)
    finally:journal.close()
