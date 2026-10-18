"""New causal mechanism checks; no models or medical data."""
from pathlib import Path
import ast
import math
import pytest
from mri_rl.root_cause_gspo_parity_v1 import mechanisms, detached_surrogate_fd, sample_advantages
ROOT=Path(__file__).resolve().parents[1]
# The fixture only reads official public Python source. Explicit env is required outside this checkout.
import os
OFF=Path(os.environ.get('VERO_OFFICIAL_ROOT',str(ROOT.parents[3]/'work/vero-official')))

@pytest.fixture(scope='module')
def report():
    if not (OFF/'vero-rl/verl/trainer/ppo/core_algos.py').is_file():
        pytest.skip('Set VERO_OFFICIAL_ROOT to pinned public source checkout')
    return mechanisms(ROOT,OFF)

def test_ratio_one_and_zero_scalar_loss_leave_nonzero_current_derivative(report):
    r=report['ratio_one_loss_zero_gradient_nonzero']
    assert r['forward_ratios']==[1.,1.]
    assert r['loss']==0. and r['gradient']==-1.5
    assert r['frozen_detach_finite_difference']==pytest.approx(-1.5,abs=1e-8)

def test_stale_favorable_ratios_clip_where_one_onpolicy_group_does_not(report):
    r=report['stale_minibatch_clipping_counterexample']
    assert r['forward_ratios'][0]>1.0004 and r['forward_ratios'][1]<.9997
    assert r['gradient']==0.

def test_sequence_mean_credit_dilution_is_length_normalization(report):
    r=report['sequence_credit_dilution']
    assert r['length2']==-.5 and r['length10']==-.1

def test_nonzero_image_ce_derivative_does_not_imply_nonzero_signed_rl_credit(report):
    r=report['image_derivative_does_not_prove_rl_signal']
    assert r['single_target_ce_image_derivative']!=0. and r['gradient']==0.

def test_format_only_standardized_credit_not_bounded_by_point_two(report):
    r=report['format_only_group_normalization']
    assert min(map(abs,r['format_only_advantages']))>.86
    assert r['max_difference']<1e-5

def test_group_size_changes_mixed_success_probability(report):
    r=report['group_size_active_probability']
    assert r['p_0_1_K4']==pytest.approx(.3438)
    assert r['p_0_1_K8']==pytest.approx(.56953278)
    assert r['p_0_5_K8']>r['p_0_5_K4']
    assert r['strict_all_format_q_0_8_p_0_5_K4']==pytest.approx(.3584)
    assert r['strict_all_format_q_0_8_p_0_5_K8']<r['strict_all_format_q_0_8_p_0_5_K4']
    assert r['uniform_attempts_active_share_p_0_1_vs_0_5_K4']<.3

def test_equal_length_and_recomputed_logps_do_not_prove_generation_identity(report):
    r=report['teacher_force_identity_blind_spot']
    assert r['generation_prompt_ids']!=r['teacher_force_prompt_ids']
    assert len(r['generation_prompt_ids'])==len(r['teacher_force_prompt_ids'])
    assert r['equal_length_check_passes'] and r['old_current_recompute_error']==0.

def test_zero_advantages_and_common_sensitivity_cancellation():
    assert sample_advantages([.2]*4)==[0.]*4
    assert detached_surrogate_fd([1.,-1.],[[1.],[1.]])==0.

def test_module_has_no_native_or_data_loader_imports():
    tree=ast.parse((ROOT/'src/mri_rl/root_cause_gspo_parity_v1.py').read_text())
    names=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    names += [v.name for n in ast.walk(tree) if isinstance(n,ast.Import) for v in n.names]
    assert not any(n and n.split('.')[0] in {'mlx','mlx_vlm','torch','transformers','nibabel'} for n in names)
