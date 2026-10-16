"""Behavioral checks for an engineering replay, not MRI model evaluation."""

import math

import pytest

from mri_rl.reward_replay import constant_output_probe, engineering_fixture, replay_groups, standardized_advantage


def test_positive_affine_keeps_group_advantages_without_epsilon():
    raw, active = standardized_advantage([0.0, 1.0, 2.0, 3.0])
    transformed, active_affine = standardized_advantage([5.0, 7.0, 9.0, 11.0])
    assert active and active_affine
    assert transformed == pytest.approx(raw)


def test_zero_variance_group_stays_inactive():
    advantages, active = standardized_advantage([1.0, 1.0, 1.0, 1.0])
    assert advantages == [0.0] * 4
    assert active is False


def test_equal_nonbinary_floats_do_not_create_false_gradient():
    advantages, active = standardized_advantage([0.1, 0.1, 0.1])
    assert active is False
    assert advantages == [0.0, 0.0, 0.0]


def test_epsilon_weakens_affine_invariance_at_small_reward_scale():
    raw, _ = standardized_advantage([0.0, 1.0], epsilon=0.5)
    scaled, _ = standardized_advantage([0.0, 2.0], epsilon=0.5)
    assert raw == pytest.approx([-0.5, 0.5])
    assert scaled == pytest.approx([-2 / 3, 2 / 3])


def test_replay_separates_logistic_and_format_effects():
    groups = [{"id": "a", "task": "spatial", "content": [0.0, 0.2, 0.5, 1.0],
               "format": [1, 1, 0, 1], "p50": 0.5, "iqr": 0.5}]
    result = replay_groups(groups)
    by_scheme = result["groups"][0]["schemes"]
    assert by_scheme["raw"]["advantages"] == pytest.approx(by_scheme["affine"]["advantages"])
    assert by_scheme["logistic"]["advantages"] != pytest.approx(by_scheme["raw"]["advantages"])
    assert by_scheme["raw_format"]["advantages"] != pytest.approx(by_scheme["raw"]["advantages"])
    assert by_scheme["raw_format"]["rewards"][2] == pytest.approx(0.4)


def test_replay_reports_effective_task_share_after_ties():
    groups = [
        {"id": "c0", "task": "categorical", "content": [0, 0, 0, 0], "format": [1, 1, 1, 1], "p50": 0.5, "iqr": 1},
        {"id": "c1", "task": "categorical", "content": [0, 1, 0, 1], "format": [1, 1, 1, 1], "p50": 0.5, "iqr": 1},
        {"id": "s0", "task": "spatial", "content": [0, 0.2, 0.5, 1], "format": [1, 1, 1, 1], "p50": 0.5, "iqr": 0.5},
        {"id": "s1", "task": "spatial", "content": [0.1, 0.3, 0.4, 0.8], "format": [1, 1, 1, 1], "p50": 0.5, "iqr": 0.5},
    ]
    summary = replay_groups(groups)["summary"]["raw"]
    assert summary["nominal_share"] == {"categorical": 0.5, "spatial": 0.5}
    assert summary["active_share"] == pytest.approx({"categorical": 1 / 3, "spatial": 2 / 3})
    assert summary["zero_variance_groups"] == 1


def test_format_can_activate_all_wrong_group_without_content_signal():
    replay = replay_groups(engineering_fixture())
    row = next(g for g in replay["groups"] if g["id"] == "cat-all-wrong-format-varies")
    assert row["schemes"]["raw"]["active"] is False
    assert row["schemes"]["raw_format"]["active"] is True
    assert all(value == 0 for value in row["schemes"]["raw"]["rewards"])
    summary = replay["summary"]["raw_format"]
    assert summary["content_active_groups"] == 3
    assert summary["format_only_active_groups"] == 1


def test_calibration_must_be_shared_within_declared_key():
    groups = [
        {"id": "a", "task": "spatial", "content": [0, 1], "format": [1, 1], "p50": 0.5, "iqr": 0.5},
        {"id": "b", "task": "spatial", "content": [0, 1], "format": [1, 1], "p50": 0.6, "iqr": 0.5},
    ]
    with pytest.raises(ValueError, match="calibration"):
        replay_groups(groups)
    groups[1]["calibration_key"] = "other-dataset/spatial"
    replay_groups(groups)


def test_constant_output_probe_exposes_prevalence_shortcut():
    targets = [{"task": "categorical", "target": {"label": label}}
               for label in ["T1", "T1", "T1", "FLAIR"]]
    result = constant_output_probe(targets, "T1")
    assert result == {"records": 4, "mean_reward": 0.75, "prediction": "T1"}


def test_replay_rejects_invalid_group_statistics_and_nonfinite_reward():
    with pytest.raises(ValueError, match="IQR"):
        replay_groups([{"id": "a", "task": "x", "content": [0, 1], "format": [1, 1], "p50": 0, "iqr": 0}])
    with pytest.raises(ValueError, match="finite"):
        standardized_advantage([0.0, math.nan])
