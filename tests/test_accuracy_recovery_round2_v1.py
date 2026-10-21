"""Round-2 core fixtures: canonical private root, original parent-arm constants, disk-backed gradients, Adam recurrence. CPU/numpy only."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
from mri_rl import accuracy_recovery_v1 as ar


def load(name):
    spec=importlib.util.spec_from_file_location(name[:-3],Path(__file__).with_name(name));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


c=load('test_accuracy_recovery_v1.py')


def make_repo(tmp_path):
    R=(tmp_path/'proj').resolve();(R/'.git/worktrees/w1').mkdir(parents=True);(R/'.git/worktrees/w1/commondir').write_text('../..\n')
    W=R/'.worktrees/w1';W.mkdir(parents=True);(W/'.git').write_text(f'gitdir: {R}/.git/worktrees/w1\n');return R,W


def test_canonical_project_root_comes_from_git_common_dir_not_the_worktree(tmp_path):
    R,W=make_repo(tmp_path)
    assert ar.canonical_project_root(W)==R and ar.canonical_project_root(R)==R and ar.canonical_project_root(W/'sub')==R
    assert ar.output_root(W)==R/'local_data/accuracy_recovery_v1'
    with pytest.raises(ValueError):ar.canonical_project_root(tmp_path/'nowhere')


def test_output_guard_binds_exact_canonical_root_and_refuses_worktree_symlink_and_traversal(tmp_path):
    R,W=make_repo(tmp_path);root=R/'local_data/accuracy_recovery_v1';root.mkdir(parents=True)
    assert ar.require_canonical_output(root/'run1',start=W)==root/'run1'
    outside=tmp_path/'outside';outside.mkdir();(root/'link').symlink_to(outside)
    for bad in (W/'local_data/accuracy_recovery_v1/run1',R/'local_data/other/run1',root,root/'../escape',tmp_path/'elsewhere',root/'link/run'):
        with pytest.raises(ValueError):ar.require_canonical_output(bad,start=W)


def test_parent_arm_constants_are_the_codex_verified_original_metadata():
    assert ar.PARENT_ARM=={'name':'P_calibrated','reward':'medgrpo_eq3','sampling':'proportional'} and ar.PARENT_SEED==13
    assert ar.PARENT_CALIBRATION_SHA256=='0d8564c5373e10b9f7fb3c004d42a45d7c4a690e972b2686309e225b43d1a136'
    assert ar.PARENT_ACTIVE_GROUP_RULE=='all-full-format_task-varies_equal-overlong'


def test_gradient_tree_is_loaded_back_from_disk_authenticated_and_norm_checked(tmp_path):
    gradient={'weight':np.array([3.,4.],dtype=np.float32)};c.archive(tmp_path,'A',gradient)
    meta=tmp_path/'gradients/arm_A_gradient_meta_PRIVATE.json'
    record,tree=ar.load_gradient_tree(meta,c.Backend().load_arrays)
    np.testing.assert_array_equal(tree['weight'],gradient['weight']);assert record['arm']=='A'
    stored=json.loads(meta.read_text());stored['gradient_l2_reported']=9.;meta.write_text(json.dumps(stored))
    with pytest.raises(ValueError,match='norm'):ar.load_gradient_tree(meta,c.Backend().load_arrays)


def test_none_gradient_loads_as_explicit_none(tmp_path):
    c.archive(tmp_path,'B',None)
    assert ar.load_gradient_tree(tmp_path/'gradients/arm_B_gradient_meta_PRIVATE.json',c.Backend().load_arrays)[1] is None


def test_archive_records_the_pre_restore_audit_evidence(tmp_path):
    meta=c.archive(tmp_path,'A',{'w':np.array([1.,2.],dtype=np.float32)})
    assert meta['pre_restore_audit']['transaction_validate_unchanged_parameters'] is True and meta['pre_restore_audit']['adam_digest_equals_parent'] is True


def test_adam_recurrence_recomputes_moments_step_and_delta_independently():
    flat_state={'w.m':np.array([.4,-.3]),'w.v':np.array([.2,.1]),'step':np.array(52)}
    grad={'w':np.array([1.,0.],dtype=np.float32)}
    nxt=ar.adam_next_state(flat_state,grad,c.PROFILE['optimizer'])
    np.testing.assert_allclose(nxt['w.m'],.9*np.array([.4,-.3])+.1*np.array([1.,0.]));assert int(nxt['step'])==53 and 'delta' in nxt
    after={k:v for k,v in nxt.items() if k!='delta'}
    assert ar.compare_adam_after(nxt,after) is True
    after['w.v']=after['w.v']+1e-3
    with pytest.raises(ValueError,match='recurrence'):ar.compare_adam_after(nxt,after)
    assert ar.adam_next_state(flat_state,None,c.PROFILE['optimizer']) is None


def test_comparison_record_has_norms_cosine_and_explicit_none_cases(tmp_path):
    c.archive(tmp_path,'A',{'w':np.array([1.,0.],dtype=np.float32)});c.archive(tmp_path,'B',{'w':np.array([1.,1.],dtype=np.float32)})
    metas={a:tmp_path/f'gradients/arm_{a}_gradient_meta_PRIVATE.json' for a in 'AB'}
    record=ar.comparison_record(metas,c.Backend().load_arrays)
    assert record['norms']['A']==pytest.approx(1.) and record['norms']['B']==pytest.approx(2**.5) and record['cosine']==pytest.approx(1/2**.5) and record['cosine_reason'] is None
    c.archive(tmp_path/'n','B',None);c.archive(tmp_path/'n','A',{'w':np.array([1.,0.],dtype=np.float32)})
    none_case=ar.comparison_record({a:tmp_path/f'n/gradients/arm_{a}_gradient_meta_PRIVATE.json' for a in 'AB'},c.Backend().load_arrays)
    assert none_case['cosine'] is None and none_case['norms']['B'] is None and 'NO_ADMITTED_GROUP' in none_case['cosine_reason']
