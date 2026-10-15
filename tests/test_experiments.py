import pytest

from mri_rl.experiments import paired_patient_difference, validate_arms


def test_budget_invariants_and_task_mix():
    arms = [
        {"name": "grounding_only", "tasks": {"grounding": 1.0}, "steps": 100, "group_size": 8, "max_tokens": 512},
        {"name": "anatomy_only", "tasks": {"anatomy": 1.0}, "steps": 100, "group_size": 8, "max_tokens": 512},
        {"name": "uniform", "tasks": {"grounding": 0.5, "anatomy": 0.5}, "steps": 100, "group_size": 8, "max_tokens": 512},
    ]
    assert validate_arms(arms) == {"steps": 100, "group_size": 8, "max_tokens": 512}
    arms[2]["steps"] = 101
    with pytest.raises(ValueError, match="budgets"):
        validate_arms(arms)


def test_patient_paired_difference_not_qa_pseudoreplication():
    rows = [
        {"subject_id": "a", "base": 0, "candidate": 1},
        {"subject_id": "a", "base": 0, "candidate": 1},
        {"subject_id": "b", "base": 1, "candidate": 0},
    ]
    result = paired_patient_difference(rows, n_bootstrap=100, seed=1)
    assert result["subjects"] == 2
    assert result["mean_difference"] == 0
    assert result["confidence_interval"][0] <= 0 <= result["confidence_interval"][1]
