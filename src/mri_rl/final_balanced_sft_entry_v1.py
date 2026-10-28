"""Exact-input supervised sprint launcher. No native import before reviewed authority.

One candidate, one pass/seed; seed1/2 require the frozen seed0 improvement rule.
No heldout16, no final100 evaluation, no historical run/rollout retry or cache rebuild.
"""
from __future__ import annotations
import argparse
import dataclasses
import fcntl
import gc
import json
import os
import shutil
import signal
import sys
import time
from pathlib import Path
from mri_rl import final_accuracy_sprint_v1 as core
from mri_rl.shared_rollout_gspo_core_v1 import publish_once, replace_atomic

ROOT=Path('/Users/vv/Documents/Codex/2026-10-15/files-pasted-by-the-user-codex-2/outputs/Vero-to-Brain-MRI')
PRIVATE=ROOT/'local_data/final_accuracy_sprint_v1'
USER_REQUEST=Path('/Users/vv/.codex/attachments/b38bca07-be2c-4b2e-808f-bfea0661c0c3/已粘贴的文本.txt')
_ISSUER=object()


class Authorization:
    def __init__(self,scope,scope_sha,seed,*,issuer):
        core.require(issuer is _ISSUER,'Entry-issued exact authority required')
        self.scope,self.sha,self.seed=scope,scope_sha,seed
        self.started=time.monotonic()
    def check(self,name):
        core.require(time.monotonic()-self.started<core.BUDGET['run_window_seconds'],'Run operation-start window exhausted')
        ledger=json.loads((PRIVATE/'runs_ledger_PRIVATE.json').read_text())
        core.require(ledger['scope']==self.sha and time.time()-ledger['started_unix']<core.BUDGET['total_window_seconds'],
                     'Original compound operation-start window exhausted')


def resource_policy():
    from mri_rl.safe_micro_dynamic_resources_v1 import POLICY
    return {**POLICY,'version':'FINAL_SFT_COLOR_V1','compound_seconds':10800.,'model_load_attempts_max':3,
            'original_planned_model_loads':3,'extra_recovery_loads_max':0,'recoveries_max':0,
            'automatic_retry':False,'ac_required':False,
            'power_exception':'Researcher fresh explicit reply: only this SFT scope battery>=20; all other protections unchanged'}


class Gate:
    def __init__(self,authorization,runtime,out):
        from mri_rl.safe_micro_dynamic_resources_v1 import Controller
        self.authorization=authorization;self.runtime=runtime;self.out=out;self.controller=Controller();self.red=False
        self.peak_rss=0.;self.peak_mlx=0.
    def __call__(self,name):
        from mri_rl.resource_supervisor import sample_macos
        from mri_rl.accuracy_recovery_gpu_cache_v3 import os_pressure_level
        from mri_rl.safe_micro_dynamic_resources_v1 import ResourceFatal,ResourcePause
        from mri_rl.accuracy_stage_resources_v1 import assert_global_lock_held
        self.authorization.check(name);assert_global_lock_held(Path(self.runtime['global_lock']))
        if self.red:raise ResourcePause('Guardian requested release')
        sample=sample_macos();r=self.controller.decide(sample,os_pressure_level(),shutil.disk_usage(PRIVATE).free/1024**3,time.monotonic())
        if resource_policy()['ac_required'] and sample.power_source!='AC' and r['action']!='STOP':
            r={**r,'action':'WAIT','reasons':[*r['reasons'],'new_SFT_requires_AC_or_fresh_scoped_power_exception']}
        self.peak_rss=max(self.peak_rss,sample.process_rss_gib or 0);self.peak_mlx=max(self.peak_mlx,sample.mlx_peak_gib or 0)
        with (self.out/'resources_PRIVATE.jsonl').open('a') as f:
            f.write(json.dumps({'boundary':name,**r,'sample':dataclasses.asdict(sample)},sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
        if r['action']=='STOP':raise ResourceFatal(str(r['reasons']))
        if r['action']!='CONTINUE':raise ResourcePause(str(r['reasons']))


def validate(scope_path,review_path):
    scope_path=Path(scope_path);scope=json.loads(scope_path.read_text());h=core.sha(scope_path)
    core.require(scope['protocol']=='FINAL_BALANCED_PATIENT_PAIRED_SEQUENCE_SFT_V1','Wrong protocol')
    core.require(scope['budget']==core.BUDGET,'Budget changed')
    core.require(scope['resource_policy']==resource_policy(),'Local resource scope changed')
    core.require(core.sha(USER_REQUEST)==scope['user_request_sha256'],'Actual user request changed')
    review=json.loads(Path(review_path).read_text())
    core.require(review['scope_file_sha256']==h and review['reviewer_kind']=='isolated_codex' and review['verdict']=='PASS_WITH_LIMITATIONS',
                 'Independent exact-scope review required')
    core.verify_files(scope['file_sha256'])
    core.require(scope['output_root']==str(PRIVATE) and scope['sealed16_authorized'] is False,'Private bounded outputs/holdout exclusion required')
    return scope,h


def execute_child(args):
    scope,h=validate(args.scope,args.review);seed=args.seed
    core.require(seed in core.SEEDS,'Frozen seeds only')
    verify_replication(seed,h)
    core.validate_dev(json.loads((PRIVATE/'DEV24_COPY_PRIVATE.json').read_text()))
    out=PRIVATE/('seed_%d_PRIVATE'%seed);core.require(not out.exists(),'No automatic retry or overwrite')
    core.claim_run(PRIVATE/'runs_ledger_PRIVATE.json',seed,h)
    out.mkdir();runtime=json.loads((PRIVATE/'RUNTIME_PRIVATE.json').read_text())
    auth=Authorization({**scope,'execution_stage':'SFT_TRAIN'},h,seed,issuer=_ISSUER)
    gate=Gate(auth,runtime,out)
    signal.signal(signal.SIGUSR1,lambda *_:setattr(gate,'red',True))
    lock=open(runtime['global_lock'],'a+b')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    resident=[None]
    try:
        # All imports below are lazy; the constructor itself checks the guard before MLX.
        from mri_rl import final_balanced_sft_v1 as sft
        from mri_rl.safe_micro_pilot_native_v1 import _configuration
        from mri_rl.experiment_contract import load_contract
        from mri_rl.accuracy_recovery_v1 import adam52_parent_anchor,parent_disk_snapshot
        from mri_rl.paired_z045_continuation_entry_v1 import run_with_release
        from mri_rl.safe_micro_recovery_native_v1 import full_identity,drop_snapshot_references
        from mri_rl.accuracy_pilot_v2_native import generate_one_v2,call_limits,prepared_inputs_v2
        from mri_rl.safe_micro_pilot_native_v2 import CallCounter
        from mri_rl.accuracy_pilot_v2_core import condition_row,strict_label
        plan=json.loads((PRIVATE/('PLAN_seed%d_PRIVATE.json'%seed)).read_text())
        targets=json.loads((PRIVATE/'TARGETS_PRIVATE.json').read_text())
        run=sft.run_binding(plan=plan,targets=targets,parent_receipt_sha256=runtime['parent_receipt_sha256'],scope_sha256=h)
        expected_images=scope['image_sha256_PRIVATE'];parent=Path(runtime['parent_checkpoint'])
        before_disk=parent_disk_snapshot(parent);anchor=adam52_parent_anchor(parent,runtime['parent_receipt_sha256'])
        def operate():
            backend,checked=sft.make_native_backend(authorization=auth,resource_gate=gate,adapter=parent/'adapter.safetensors',
                config=_configuration(runtime),model_path=Path(runtime['model']),contract=load_contract(Path(runtime['contract'])),
                cache={'dir':runtime['cache_dir']},prompt_lengths={'mri_sequence':233},run=run)
            resident[0]=backend
            identity,eos=sft.prepare_native_parent(backend,expected_images=expected_images,prompt_lengths={'mri_sequence':233},
                work=out/'unused_rollouts_PRIVATE',run=run,parent_dir=parent,anchor=anchor)
            core.require(eos==targets['eos_id'],'Native EOS differs from CPU freeze')
            publish_once(out/'actual_parent_identity_PRIVATE.json',full_identity(backend))
            start=time.monotonic()
            result=sft.run_sft(backend=backend,plan=plan,targets=targets,binding=run,output=out/'training_PRIVATE',before=gate,
                expected_parent=identity,authority=checked)
            training_seconds=time.monotonic()-start
            drop_snapshot_references(backend);gc.collect();backend.mx.clear_cache()
            checkpoint=out/'training_PRIVATE/pairs/pair_0027/checkpoint_PRIVATE'
            cursor=sft.committed_cursor(out/'training_PRIVATE',binding_sha256=core.identity(run))
            core.require(cursor['completed_pairs']==28,'Complete one-pass terminal required')
            # Reload the actual terminal from disk BEFORE evaluation, not merely its in-memory policy.
            before_reload=full_identity(backend)
            backend.load_sft_checkpoint(checkpoint,run=run,completed_pairs=28)
            core.require(full_identity(backend)==before_reload,'Saved terminal full native restore differs')
            publish_once(out/'terminal_reload_PRIVATE.json',{'full_state_exact':True,'checkpoint_sha256':core.sha(checkpoint/'adapter.safetensors')})
            counter=CallCounter(call_limits(24),out/'eval_calls_PRIVATE.json');evalrows=[]
            records=json.loads((PRIVATE/'DEV24_COPY_PRIVATE.json').read_text())
            readonly_before=full_identity(backend);t0=time.monotonic()
            for i,item in enumerate(records):
                row=condition_row(item['row_PRIVATE'],'concise')
                actual=generate_one_v2(backend,row,item['image_sha256'],lengths={'mri_sequence':233},counter=counter,before=gate)
                record={'patient':row['extra_info']['patient_key'],'level':row['extra_info'].get('level'),
                        'truth':row['reward_model']['ground_truth'],'prediction':strict_label(actual['response']['text']),
                        'id':row['id'],'actual_PRIVATE':actual}
                publish_once(out/('eval_%02d_PRIVATE.json'%i),record);evalrows.append(record)
            readonly_after=full_identity(backend)
            core.require(readonly_before['parameters']==readonly_after['parameters'] and readonly_before['adam']==readonly_after['adam'],
                         'Evaluation changed parameters/Adam')
            m=core.metrics(evalrows)
            pairs={}
            for r in evalrows:pairs.setdefault((r['patient'],r['level']),[]).append(r)
            core.require(len(pairs)==12 and all(len(v)==2 for v in pairs.values()),'Frozen12pairedDEV contexts required')
            m['both_correct_pairs']=sum(all(x['prediction']==core.LABELS.index(x['truth']) for x in v) for v in pairs.values())
            return {'status':'TRAINED_AND_DEV_EVALUATED','scope_file_sha256':h,'method':'SUPERVISED_SFT_NOT_GSPO','seed':seed,'metrics_PRIVATE':m,
                    'training':result,'training_seconds':training_seconds,'evaluation_seconds':time.monotonic()-t0,
                    'checkpoint_sha256':core.sha(checkpoint/'adapter.safetensors'),'checkpoint_PRIVATE':str(checkpoint),
                    'resources':{'RSS_peak_GiB':gate.peak_rss,'MLX_reported_peak_GiB':gate.peak_mlx},'eval_calls':counter.used}
        result=run_with_release(operate,resident,out)
        resident[0]=None;gc.collect()
        core.require(parent_disk_snapshot(parent)==before_disk,'Historical parent changed on disk')
        core.require(json.loads((out/'release_SAFE.json').read_text())['release_verified'],'Native allocator release required')
        publish_once(out/'complete_PRIVATE.json',result)
    finally:
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


def verify_replication(seed,scope_sha):
    if not seed:return
    first=PRIVATE/'seed_0_PRIVATE/complete_PRIVATE.json'
    witness=PRIVATE/'seed_0_witness_PRIVATE/actual_exit_PRIVATE.json'
    core.require(first.is_file() and witness.is_file(),'No completed seed0 native result')
    result=json.loads(first.read_text());w=json.loads(witness.read_text())
    core.require(result['scope_file_sha256']==scope_sha and w['scope_file_sha256']==scope_sha and w['native_exit_code']==0 and
                 w['child_reaped'] and not w['guardian_stop'] and core.replication_allowed(result['metrics_PRIVATE']),
                 'Seed0 did not meet predeclared replication criterion and native completion')


def launch(args):
    scope,h=validate(args.scope,args.review)
    runtime=json.loads((PRIVATE/'RUNTIME_PRIVATE.json').read_text())
    core.require(args.seed in core.SEEDS,'Frozen seeds only')
    verify_replication(args.seed,h)
    from mri_rl.safe_micro_watchdog_v2 import witness_child
    from mri_rl.safe_micro_dynamic_resources_v1 import Controller
    from mri_rl.resource_supervisor import sample_macos
    from mri_rl.accuracy_recovery_gpu_cache_v3 import os_pressure_level
    def disk():return shutil.disk_usage(PRIVATE).free/1024**3
    controller=Controller()
    for _ in range(3):
        sample=sample_macos()
        core.require(not resource_policy()['ac_required'] or sample.power_source=='AC','Preflight HOLD: AC required')
        r=controller.decide(sample,os_pressure_level(),disk(),time.monotonic())
        core.require(r['action']=='CONTINUE','Preflight HOLD/STOP: '+str(r))
    cmd=[sys.executable,'-m','mri_rl.final_balanced_sft_entry_v1','--child','--scope',args.scope,'--review',args.review,'--seed',str(args.seed)]
    def inspect_lock():
        with open(runtime['global_lock'],'a+b') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:return {'status':'OCCUPIED'}
            fcntl.flock(lock,fcntl.LOCK_UN);return {'status':'FREE'}
    result=witness_child(cmd,PRIVATE/('seed_%d_witness_PRIVATE'%args.seed),scope_sha256=h,sampler=lambda pid:sample_macos(pid),
        level=os_pressure_level,disk_free_gib=disk,post_sampler=lambda:dataclasses.asdict(sample_macos()),lock_inspector=inspect_lock)
    print(json.dumps(result,sort_keys=True))
    return 0 if result.get('native_exit_code')==0 and not result.get('guardian_stop') and result.get('post_child_global_lock',{}).get('status')=='FREE' else 1


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scope',required=True);p.add_argument('--review',required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--child',action='store_true');a=p.parse_args()
    return execute_child(a) if a.child else launch(a)


if __name__=='__main__':raise SystemExit(main())
