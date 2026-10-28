"""Durable micro milestones; unchanged V1 GSPO/Adam arm implementation.

Recoverable: fully committed common16+own-old+R1 and each complete arm.
An interrupted claimed common/arm is INDETERMINATE, never recomputed. No claim
of token/backward/optimizer-level lossless recovery is made. Import is CPU-only.
"""
from pathlib import Path
from types import SimpleNamespace
import json,time
from mri_rl import safe_micro_training_v1 as v1
from mri_rl import multistep_gspo_core_v1 as ms
from mri_rl import shared_rollout_gspo_core_v1 as shared
from mri_rl.safe_micro_journal_v1 import Journal,durable_mkdir

require=shared.require


def files_under(directory):
    directory=Path(directory)
    for p in sorted(directory.rglob('*'),reverse=True):
        if p.is_dir():shared.sync_dir(p)
    shared.sync_dir(directory);shared.sync_dir(directory.parent)
    return {str(p.resolve()):shared.file_sha256(p) for p in sorted(Path(directory).rglob('*')) if p.is_file()}


class Ledger(v1.Ledger):
    """Persistent spent-work counters, never replaced by zeros on recovery."""
    def __init__(self,budget,path):
        self.path=Path(path)
        self.limits={'common':budget['common'],'A':budget['arm'],'B':budget['arm']}
        if self.path.exists():
            old=json.loads(self.path.read_text())
            require(old.get('protocol')==v1.PROTOCOL and old.get('limits')==self.limits,'Frozen ledger limits changed')
            require(set(old['used'])==set(self.limits),'Ledger phase inventory changed')
            self.used=old['used']
        else:self.used={name:{k:0 for k in v1.CHARGEABLE} for name in self.limits}
        self.phases={name:shared.Counters(limits,self.used[name],lambda used,name=name:self._write(name,used))
                     for name,limits in self.limits.items()}
        if not self.path.exists():self._write('common',self.used['common'])


def _common(backend,units,lengths,reward,ledger,parent,output,before,receipt):
    directory=Path(output)/'common_PRIVATE';durable_mkdir(directory)
    backend.counters=ledger.phase('common');policy,adam,rng0=parent;raws=[];blocks=[]
    for unit in units:
        before('micro_common_generate_unit_'+str(unit['index']))
        raw=backend.generate(unit,lengths)
        ms.ownership(raw['policy_sha256'],policy)
        require(backend.policy_sha()==policy and backend.adam_digest()==adam,'Common rollout mutated parent')
        raws.append(raw)
        blocks.append(ms.build_block(unit,raw,policy_sha=policy,reward_policy_sha256=reward,checkpoint_receipt_sha256=receipt))
    rng1=backend.rng_digest();require(rng1!=rng0,'Common generation did not consume RNG')
    rng_dir=directory/'rng_after_all16';durable_mkdir(rng_dir);backend.save_rng(rng_dir)
    olds=[]
    for unit,block in zip(units,blocks):
        before('micro_common_score_old_unit_'+str(unit['index']))
        old=backend.score_old(unit,block.responses);ms.check_old(block,old)
        require(v1._state(backend)==(policy,adam,rng1),'Own-old mutated state')
        olds.append(old)
    record={'protocol':v1.PROTOCOL,'parent_state':dict(zip(('policy','adam','rng'),parent)),
            'rng_after_all16_digest':rng1,'bundle_sha256':[b.sha256 for b in blocks],
            'responses':sum(len(b.responses) for b in blocks),'counters':ledger.phase('common').snapshot()}
    record['receipt_sha256']=shared.identity(record)
    for unit,block,old in zip(units,blocks,olds):
        shared.publish_once(directory/f"bundle_unit{unit['index']}_PRIVATE.json",block.raw)
        shared.publish_once(directory/f"own_old_unit{unit['index']}_PRIVATE.json",{'bundle_sha256':block.sha256,'old':old})
    shared.publish_once(directory/'common_record_PRIVATE.json',record)
    value={'raws':raws,'olds':olds,'rng_dir':str(rng_dir.resolve()),'rng1':rng1,'record':record}
    shared.publish_once(directory/'raw_rollout_and_own_old_PRIVATE.json',value)
    return {'value':value,'files':files_under(directory)}


def restore_common(value,units,reward,parent,receipt,backend):
    record=value['record']
    require(record['receipt_sha256']==shared.identity({k:v for k,v in record.items() if k!='receipt_sha256'}),'Common receipt changed')
    require(record['parent_state']==dict(zip(('policy','adam','rng'),parent)),'Common belongs to another parent')
    blocks=[ms.build_block(u,r,policy_sha=parent[0],reward_policy_sha256=reward,checkpoint_receipt_sha256=receipt)
            for u,r in zip(units,value['raws'])]
    require(len(value['raws'])==len(value['olds'])==len(blocks)==2,'Common inventory differs')
    require(record['bundle_sha256']==[b.sha256 for b in blocks] and record['rng_after_all16_digest']==value['rng1'],'Common bundle/R1 differs')
    for unit,raw,block,old in zip(units,value['raws'],blocks,value['olds']):
        ms.check_old(block,old)
        if hasattr(backend,'_rollout_contexts'):
            backend._rollout_contexts[(unit['index'],parent[0])]=raw['contexts']
    return SimpleNamespace(blocks=blocks,olds=value['olds'],rng_dir=Path(value['rng_dir']),rng1=value['rng1'],record=record)


def execute_micro(*,backend,units,prompt_lengths,reward_policy_sha256,output,before,expected_parent,
                  checkpoint_receipt_sha256=None,verify_parent_disk=None,journal_binding=None):
    """Same scientific V1 transaction, resumed only at verified coarse commits."""
    v1._check_backend(backend);require(callable(before),'Resource guard required')
    require(isinstance(expected_parent,dict) and set(expected_parent)=={'policy','adam','rng'},'Exact parent identity required')
    require(isinstance(journal_binding,dict) and journal_binding,'Exact scope/input journal binding required')
    output=Path(output);durable_mkdir(output)
    budget=v1.micro_budget(units,prompt_lengths);parent=v1._state(backend)
    require(parent==tuple(expected_parent[k] for k in ('policy','adam','rng')) and backend.optimizer_step()==52,'Original Adam52 parent required')
    binding={**journal_binding,'protocol':v1.PROTOCOL,'parent':expected_parent,'units_sha256':shared.identity(units),
             'reward_sha256':reward_policy_sha256,'parent_receipt':checkpoint_receipt_sha256,'budget':budget}
    journal=Journal(output/'journal_PRIVATE',binding,before=before)
    try:
        journal.assert_recoverable()
        ledger=Ledger(budget,output/'counters_PRIVATE.json');snapshot=backend.snapshot()
        start={'protocol':v1.PROTOCOL,'budget':budget,'evidence_kind':backend.evidence_kind,'parent_state':expected_parent,'adam_step':52,'automatic_retry':False}
        p=output/'start_PRIVATE.json'
        if p.exists():require(json.loads(p.read_text())==start,'Micro start changed')
        else:shared.publish_once(p,start)
        try:
            common_value=journal.run('common',lambda _: _common(backend,units,prompt_lengths,reward_policy_sha256,ledger,parent,output,before,checkpoint_receipt_sha256),reservation=budget['common'])
            require(common_value['record']['counters']==ledger.phase('common').snapshot(),'Persisted common counters differ')
            backend.restore(snapshot)
            common=restore_common(common_value,units,reward_policy_sha256,parent,checkpoint_receipt_sha256,backend)
            arms={}
            for arm in v1.ARMS:
                def operation(_,arm=arm):
                    value=v1._arm(arm=arm,backend=backend,units=units,common=common,ledger=ledger,snapshot=snapshot,parent=parent,
                                  output=output,before=before,kind=backend.evidence_kind)
                    return {'value':value,'files':files_under(output/('arm_'+arm))}
                arms[arm]=journal.run('arm_'+arm,operation,reservation=budget['arm'])
                require(arms[arm]['counters']==ledger.phase(arm).snapshot(),'Persisted arm counters differ')
            require(v1._state(backend)==parent,'Parent not preserved')
            if verify_parent_disk:require(verify_parent_disk() is True,'Parent files changed')
            spent=ledger.snapshot();total={k:sum(spent[p][k] for p in spent) for k in v1.CHARGEABLE}
            require(all(total[k]<=budget['total'][k] for k in total),'Spent work exceeds original budget')
            result={'status':'ACTUAL_MICRO_TRANSACTION_COMPLETE_PENDING_INDEPENDENT_REVIEW' if backend.evidence_kind==v1.NATIVE_KIND else 'SYNTHETIC_FIXTURE_COMPLETE_NOT_NATIVE_EVIDENCE',
                    'protocol':v1.PROTOCOL,'evidence_kind':backend.evidence_kind,'arms':arms,'common_receipt_sha256':common.record['receipt_sha256'],
                    'rng_after_all16_digest':common.rng1,'parent_preserved':True,'spent_total':total,'optimizer_updates':sum(a['updated'] for a in arms.values()),
                    'next_stage_authorized':False,'small_pilot_authorized':False,'dev24_authorized':False,'automatic_stage_progression':False,'science_claim':False}
            complete=output/'micro_complete_SAFE.json'
            if complete.exists():require(json.loads(complete.read_text())==result,'Completed micro result changed')
            else:shared.publish_once(complete,result)
            return result
        except BaseException as error:
            restored=False
            try:backend.restore(snapshot);restored=v1._state(backend)==parent
            except BaseException:pass
            shared.publish_once(output/f'STOP_{time.time_ns()}_PRIVATE.json',{'error_type':type(error).__name__,'error':str(error),
                'rolled_back_to_parent':restored,'counters_spent_not_refunded':ledger.snapshot(),'automatic_retry':False,
                'recovery':'Only committed journal milestones; claimed incomplete operation is indeterminate, no replay.'})
            raise
    finally:journal.close()
