"""External exit witness: only tiny synthetic Python children, no GPU."""
import json
import sys
import pytest
from mri_rl import safe_micro_witness_v1 as w


@pytest.mark.parametrize('code',[0,3])
def test_closed_logs_exit_and_post_release_observation(tmp_path,code):
    out=tmp_path/'witness'
    result=w.witness_child([sys.executable,'-c',f'print("synthetic"); raise SystemExit({code})'],out,
        scope_sha256='a'*64, sampler=lambda:{'available_gib':10.}, lock_inspector=lambda:{'no_open_holders':True})
    assert result['native_exit_code']==code and result['child_reaped']
    assert result['post_child_resource_sample']=={'available_gib':10.}
    assert result['automatic_next_stage'] is False
    assert result['stdout_sha256']==w.digest(out/'stdout_PRIVATE.log')
    assert result['child_peak_RSS_GiB']>0
    assert json.loads((out/'actual_exit_PRIVATE.json').read_text())==result
    with pytest.raises(FileExistsError):
        w.witness_child([sys.executable,'-c','raise Exception()'],out,scope_sha256='a'*64,
            sampler=lambda:{},lock_inspector=lambda:{})


def test_witness_does_not_claim_allocator_or_model_state_proof(tmp_path):
    result=w.witness_child([sys.executable,'-c','pass'],tmp_path/'out',scope_sha256='b'*64,
        sampler=lambda:{'mlx_active_gib':None},lock_inspector=lambda:{'no_open_holders':True})
    assert result['allocator_release_proof']=='SEPARATE_CHILD_RECEIPT_OR_NOT_MEASURED'
    assert result['parent_Adam_RNG_integrity']=='NOT_ATTESTED_BY_EXIT_WITNESS'


def test_sensor_errors_preserve_the_known_exit_and_remain_unknown(tmp_path):
    def unavailable():raise OSError('synthetic sensor unavailable')
    out=tmp_path/'out'
    result=w.witness_child([sys.executable,'-c','raise SystemExit(3)'],out,scope_sha256='c'*64,
        sampler=unavailable,lock_inspector=unavailable)
    assert result['native_exit_code']==3 and result['child_reaped']
    assert result['post_child_resource_sample']['status']=='UNAVAILABLE'
    assert 'no_open_holders' not in result['post_child_global_lock']
    assert len(result['post_child_inspection_errors'])==2
    assert json.loads((out/'actual_exit_PRIVATE.json').read_text())==result
