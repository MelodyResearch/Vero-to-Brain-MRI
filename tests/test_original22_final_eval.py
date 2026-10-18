"""Synthetic metadata only: no sealed registry, model or native runtime."""
import copy,json
from pathlib import Path
import pytest


def test_review_set_requires_exact22_and_external_digest(tmp_path):
 from mri_rl.original22_final_eval import validate_review_set
 from mri_rl.two_task_pool import sha
 p=tmp_path/'reviews.json';p.write_text('{}')
 with pytest.raises(ValueError,match='digest'):validate_review_set(p,'0'*64,{'runs':[]},'a'*64)
 with pytest.raises(ValueError,match='scope set'):validate_review_set(p,sha(p),{'runs':[]},'a'*64)


def test_native_cli_is_fixed_greedy_original_interface(tmp_path):
 from mri_rl.original22_final_eval import cli_argv
 b={'python':'python','model':'model','contract_path':'contract','cache_dir':'cache','cache_certificate':'cert','cache_certificate_sha256':'c'*64}
 run={'adapter':'adapter','manifest':'manifest','output':'output','label':'ORIGINAL22_B28_FINAL100','seed':7}
 a=cli_argv(b,run,'review','r'*64)
 assert a[:4]==['python','-u','-m','mri_rl.continuous_rl_dev_eval']
 assert '--resume' not in a and '--subset-ids-file' not in a
 assert a[a.index('--seed')+1]=='7' and a[a.index('--manifest')+1]=='manifest'
 assert a[a.index('--expected-review-sha256')+1]=='r'*64


def test_failed_or_partial_stage_never_automatically_retries(tmp_path):
 from mri_rl.original22_final_eval import completed_stage
 from mri_rl.two_task_pool import sha
 output=tmp_path/'output.jsonl';log=tmp_path/'stdout.log';exitfile=tmp_path/'exit.json'
 run={'output':str(output),'log':str(log),'exit':str(exitfile),'checkpoint_id':'B28'}
 assert completed_stage(run,'a'*64) is False
 output.write_text('{}\n')
 with pytest.raises(ValueError,match='Partial'):completed_stage(run,'a'*64)
 log.write_text('closed');exitfile.write_text(json.dumps({'exit_code':1}))
 with pytest.raises(ValueError,match='failed'):completed_stage(run,'a'*64)


def test_completed_stage_authenticates_closed_output_and_log(tmp_path):
 from mri_rl.original22_final_eval import completed_stage
 from mri_rl.two_task_pool import sha
 out=tmp_path/'output';log=tmp_path/'log';w=tmp_path/'exit'
 out.write_text(''.join(json.dumps({'record_id':str(i)})+'\n' for i in range(100)));log.write_text('closed')
 run={'output':str(out),'log':str(log),'exit':str(w),'checkpoint_id':'B28'}
 receipt={'checkpoint_id':'B28','exit_code':0,'review_sha256':'a'*64,'raw_log_sha256':sha(log),'output_sha256':sha(out),'records':100}
 w.write_text(json.dumps(receipt));assert completed_stage(run,'a'*64) is True
 out.write_text(out.read_text()+'{}\n')
 with pytest.raises(ValueError,match='output'):completed_stage(run,'a'*64)
