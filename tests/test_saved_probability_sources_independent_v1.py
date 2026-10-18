"""CPU-only tests for source/evidence interpretation, never native execution."""
import importlib.util
from pathlib import Path
import numpy as np

P = Path(__file__).resolve().parents[1]/'scripts/analyze_saved_probability_sources_independent_v1.py'
spec = importlib.util.spec_from_file_location('independent_saved_source_analysis', P)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_selected_only_vectors_cannot_identify_full_distribution():
    result = m.selected_summary([-1., -2.], [-.9, -2.1])
    assert result['distribution_equivalence'] == 'NOT_IDENTIFIABLE_FROM_SELECTED_ONLY'
    assert result['max_abs_delta'] > .09
    assert abs(result['mean_signed_delta']) < 1e-15


def test_common_shift_preserves_synthetic_categorical():
    r = m.common_shift_fixture()
    assert r['max_categorical_delta'] == 0.
    assert r['old_probability_sum'] != 1.
    assert r['old_normalized'] == r['reference_normalized']


def test_actual_frozen_logps_AST_alignment_with_identity_markers():
    backend = Path(__file__).resolve().parents[1]/'src/mri_rl/mlx_continuous_gspo.py'
    r = m.check_logps_indexing(backend, [[4, 5, 6]], [7, 8, 9])
    assert r['positions_verified'] == 3 and r['first_causal_logit_position'] == 2
    assert r['last_causal_logit_position'] == 4
    assert r['all_input_tokens_and_masks_equal']


def test_actual_cache_AST_preserves_prefix_and_cursor():
    package = Path('/Users/vv/Documents/Codex/2026-10-15/files-pasted-by-the-user-codex-2/work/mlx-venv/lib/python3.13/site-packages')
    r = m.check_cache_update(package/'mlx_vlm/models/cache.py', 7, 3)
    assert r['offsets'] == [7, 8, 9]
    assert r['no_missing_or_duplicate_prefix_values']


def test_neutral_sampler_AST_has_no_filter_or_extra_cast():
    package = Path('/Users/vv/Documents/Codex/2026-10-15/files-pasted-by-the-user-codex-2/work/mlx-venv/lib/python3.13/site-packages')
    result = m.check_sampler_input(package/'mlx_vlm/sample_utils.py')
    assert result['categorical_argument_equals_input_at_temperature1']
    assert result['native_random_draws'] == 0
