import pytest

from mri_rl.score_frozen_eval import score


def test_categorical_scoring_rejects_explanatory_or_ambiguous_answers():
    assert score("mri_sequence_identification", "FLAIR", "<think>x</think><answer>FLAIR</answer>") == (True, True, None)
    assert score("mri_sequence_identification", "FLAIR", "T1 or FLAIR") == (False, False, None)


def test_centroid_scoring_has_explicit_voxel_tolerance():
    correct, valid, distance = score("central_slice_hippocampal_centroid", [10, 20], "(13, 20)")
    assert (correct, valid, distance) == (True, True, 3.0)
    assert score("central_slice_hippocampal_centroid", [10, 20], "unknown") == (False, False, None)


def test_unknown_task_fails_closed():
    with pytest.raises(ValueError):
        score("diagnosis", "x", "x")


def test_tumor_routes_reject_unsupported_language():
    assert score("tumor_enhancing_presence", "no", "no") == (True, True, None)
    assert score("tumor_enhancing_presence", "no", "possibly no") == (False, False, None)
    assert score("tumor_grid", "center-middle", "center-middle") == (True, True, None)
    assert score("tumor_grid", "center-middle", "center") == (False, False, None)
