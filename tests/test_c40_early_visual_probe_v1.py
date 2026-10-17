"""Isolated early diagnostic guards; no primary source/model/process mutation."""
import importlib.util
from pathlib import Path
import pytest


def mod():
 p=Path(__file__).parents[1]/'scripts/run_c40_early_visual_probe_v1.py';assert p.is_file(),'Isolated early probe missing'
 s=importlib.util.spec_from_file_location('early_probe_test',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def test_early_plan_keeps_all_original10_and_changes_only_eligibility():
 m=mod();original={'records_PRIVATE':[{'record_id':str(i)} for i in range(10)],'checkpoints':{'B28':{},'U_raw_seed7':{}},'derived_cache':{'same':True},'likelihood':{'same':True},'greedy':{'same':True}}
 p=m.early_plan(original,'/frozen/original.json',{'frozen':'same'})
 for field in ('records_PRIVATE','checkpoints','derived_cache','likelihood','greedy'):assert p[field]==original[field]
 assert p['records_per_checkpoint']==10 and p['total_candidate_forwards']==60 and p['new_null_greedy_generations']==4
 assert p['weight_updates']==0 and p['checkpoint_order']==['B28','U_raw_seed7']


def test_no_lease_fails_before_native_or_process_action(tmp_path):
 m=mod()
 with pytest.raises(ValueError):m.validate_lease(tmp_path/'missing','0'*64,{},tmp_path/'lock',None)


def test_review_guard_requires_actual_independent_digest(tmp_path):
 m=mod();p=tmp_path/'review';p.write_text('{}')
 with pytest.raises(ValueError):m.verify_review(p,'0'*64,{'kind':'early'})


def test_command_base_removes_only_runtime_provenance_arguments():
 m=mod();base=['/lexical/venv/python','-u','script.py','--checkpoint','B28','--execute']
 args=base+['--lease','lease','--expected-lease-sha256','HASH','--lease-lock-fd','9','--review-verdict','review','--expected-review-sha256','REVIEW']
 assert m.command_base(args)==base


def test_inherited_lock_identity_mismatch_blocks_before_terminal_artifacts(tmp_path,monkeypatch):
 import os,json
 m=mod();contract={'contract_sha256':'C','controller_source_sha256':'S','boundary_stage_id':'rl:P_raw:13','expected_old_parent_identity':{'pid':999999}}
 lease={'schema':'natural-arm-boundary-handoff-lease-v1','contract_sha256':'C','controller_source_sha256':'S','boundary_stage_id':'rl:P_raw:13',
  'original_parent_exited':True,'no_model_children':True,'original_parent_identity':{'pid':999999},'controller_identity':{'pid':os.getppid(),'lstart':'START','command':'CONTROLLER'},
  'queue_lock':{'path':str((tmp_path/'original.lock').resolve()),'st_dev':0,'st_ino':0}}
 path=tmp_path/'lease';path.write_text(json.dumps(lease));(tmp_path/'original.lock').write_text('')
 monkeypatch.setattr(m,'process_identity',lambda pid:lease['controller_identity'] if pid==os.getppid() else None)
 with (tmp_path/'wrong.lock').open('w') as fd:
  with pytest.raises(ValueError,match='original global lock'):m.validate_lease(path,m.sha(path),contract,tmp_path/'original.lock',fd.fileno())


def test_valid_inherited_boundary_lease_keeps_original_cursor_unchanged(tmp_path,monkeypatch):
 import json,fcntl
 m=mod();run=tmp_path/'run/boundary_0001';run.mkdir(parents=True);(run/'receipt.json').write_text(json.dumps({'completed_groups':1,'optimizer_updates':1}))
 log=tmp_path/'native.log';log.write_text('closed fixture');exitp=tmp_path/'exit.json';exitp.write_text(json.dumps({'exit_code':0,'stage_id':'rl:P_raw:13','raw_log_sha256':m.sha(log)}))
 manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'stages':[{'id':'rl:P_raw:13','output':str(run.parent),'exit_receipt':str(exitp),'raw_log':str(log)}]}))
 cursor=tmp_path/'cursor.json';cursor.write_text('{"stages":{"rl:P_raw:13":{"state":"RUNNING"}}}')
 before=cursor.read_bytes();lock=tmp_path/'remaining20.lock';lock.write_text('');base=['/lexical/python','-u','handler.py','--execute'];controller={'pid':333,'lstart':'START','command':'CONTROLLER'}
 contract={'contract_sha256':'C','controller_source_sha256':'S','boundary_stage_id':'rl:P_raw:13','expected_old_parent_identity':{'pid':444},'queue_cursor_path':str(cursor),'original_manifest_path':str(manifest),'diagnostic_commands':[base]}
 monkeypatch.setattr(m.os,'getppid',lambda:333);monkeypatch.setattr(m,'process_identity',lambda pid:controller if pid==333 else None);monkeypatch.setattr(m.sys,'orig_argv',base)
 with lock.open('a') as handle:
  fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);stat=lock.stat()
  terminal={'completed_groups':1,'optimizer_updates':1,'boundary_receipt_sha256':m.sha(run/'receipt.json'),'full_budget_terminal_verified':True,'native_terminal_status':'TOKEN_BUDGET_EXHAUSTED'}
  lease={'schema':'natural-arm-boundary-handoff-lease-v1','contract_sha256':'C','controller_source_sha256':'S','boundary_stage_id':'rl:P_raw:13','original_parent_exited':True,'no_model_children':True,
   'original_parent_identity':{'pid':444},'controller_identity':controller,'queue_lock':{'path':str(lock.resolve()),'st_dev':stat.st_dev,'st_ino':stat.st_ino},'queue_cursor_sha256':m.sha(cursor),
   'arm_exit_receipt_sha256':m.sha(exitp),'closed_native_log_sha256':m.sha(log),'completed_boundary_receipt_sha256':m.sha(run/'receipt.json'),'terminal_summary':terminal,'diagnostic_command_base_sha256':[m.h(base)]}
  p=tmp_path/'lease.json';p.write_text(json.dumps(lease));assert m.validate_lease(p,m.sha(p),contract,lock,handle.fileno())==lease
  assert cursor.read_bytes()==before
