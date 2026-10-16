import numpy as np
import pytest

from mri_rl.identity_baseline import (
    cosine_score,
    rank_candidates,
    retrieval_metrics,
    spatial_signature,
    wilson_interval,
)


def test_spatial_signature_distinguishes_nonspatially_identical_volumes():
    a = np.zeros((24, 24, 24), dtype=np.float32)
    a[3:21, 3:21, 3:21] = 1
    a[6:12, 6:12, 6:12] = 5
    b = np.zeros_like(a)
    b[3:21, 3:21, 3:21] = 1
    b[12:18, 12:18, 12:18] = 5
    fa, fb = spatial_signature(a), spatial_signature(b)
    assert np.array_equal(np.sort(a.ravel()), np.sort(b.ravel()))
    assert cosine_score(fa, fa) == pytest.approx(1.0)
    assert cosine_score(fa, fb) < 0.9


def test_query_level_ranking_and_tie_policy():
    scores = np.array([[0.8, 0.1, 0.2], [0.1, 0.6, 0.2], [0.1, 0.7, 0.3]])
    metrics = retrieval_metrics(scores)
    assert metrics["queries"] == 3
    assert metrics["top1_correct"] == 2
    assert metrics["ranks_private"] == [1, 1, 2]
    assert rank_candidates([1, 1, 0]).tolist() == [0, 1, 2]


def test_wilson_interval_stays_in_unit_range():
    lo, hi = wilson_interval(0, 12)
    assert 0 <= lo <= hi <= 1
    lo, hi = wilson_interval(12, 12)
    assert 0 <= lo <= hi <= 1
