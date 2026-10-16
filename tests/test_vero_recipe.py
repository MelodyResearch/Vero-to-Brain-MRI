"""CPU contract checks; these do not perform policy training."""

import math
import json
from collections import Counter
from pathlib import Path

import pytest

from mri_rl.vero_recipe import (
    format_reward,
    gspo_clipped_surrogate,
    overlong_penalty,
    route_accuracy,
    sample_equal_categories,
    score_response,
    validate_recipe,
)


def test_equal_category_sampling_preserves_open_ended_exposure():
    records = [
        {"id": "a", "category": "anatomy"},
        {"id": "b", "category": "anatomy"},
        {"id": "c", "category": "open_ended"},
        {"id": "d", "category": "longitudinal"},
    ]
    categories = ["anatomy", "open_ended", "longitudinal"]
    first = sample_equal_categories(records, categories, per_category=4, seed=7)
    assert Counter(row["category"] for row in first) == {"anatomy": 4, "open_ended": 4, "longitudinal": 4}
    assert first == sample_equal_categories(records, categories, per_category=4, seed=7)
    with pytest.raises(ValueError, match="missing"):
        sample_equal_categories(records, categories + ["spatial"], per_category=1)


def test_routed_rewards_fail_closed_for_missing_judge_and_unknown_route():
    assert route_accuracy({"reward_type": "exact_match", "target": {"label": "T1"}}, " t1 ") == 1.0
    assert route_accuracy({"reward_type": "numeric", "target": {"value": 3.0, "absolute_tolerance": 0.2}}, "3.1") == 1.0
    assert route_accuracy({"reward_type": "grounding_2d", "target": {"frame": 1, "box": [0, 0, 10, 10]}},
                          '{"frame": 2, "box": [0, 0, 10, 10]}') == 0.0
    with pytest.raises(ValueError, match="judge"):
        route_accuracy({"reward_type": "open_ended_judge", "target": {}}, "description")
    assert route_accuracy({"reward_type": "open_ended_judge", "target": {}}, "description",
                          judge=lambda sample, output: 0.7) == pytest.approx(0.7)
    with pytest.raises(ValueError, match="unsupported"):
        route_accuracy({"reward_type": "voxel_dice", "target": {}}, "anything")


def test_format_reward_and_overlong_match_documented_formula():
    valid = "<think>inspect slices</think><answer>\\boxed{T1}</answer>"
    assert format_reward(valid, symbolic=True) == 1.0
    assert format_reward("<think>inspect</think><answer>T1</answer>", symbolic=True) == 0.5
    assert format_reward("<think></think><answer>T1</answer>", symbolic=False) == 0.0
    assert format_reward("<think>inspect</think><answer>description</answer>", symbolic=False) == 1.0
    assert overlong_penalty(2048, max_tokens=4096) == 0.0
    assert overlong_penalty(3072, max_tokens=4096) == -0.5
    assert overlong_penalty(4096, max_tokens=4096) == -1.0


def test_composite_reward_extracts_answer_and_combines_terms():
    sample = {"reward_type": "exact_match", "target": {"label": "T1"}}
    response = "<think>inspect slices</think><answer>\\boxed{T1}</answer>"
    result = score_response(sample, response, length=100, max_tokens=4096)
    assert result == {"accuracy": 1.0, "format": 1.0, "overlong": 0.0, "total": 1.0}
    unboxed = score_response(sample, "<think>inspect</think><answer>T1</answer>", 100, 4096)
    assert unboxed["accuracy"] == 1.0
    assert unboxed["format"] == 0.5
    assert unboxed["total"] == pytest.approx(0.9)
    ground = {"reward_type": "grounding_2d", "target": {"frame": 1, "box": [0, 0, 10, 10]}}
    grounded = '<think>inspect</think><answer>{"frame": 1, "box": [0, 0, 10, 10]}</answer>'
    assert score_response(ground, grounded, 100, 4096)["total"] == 1.0
    multiple_boxes = '<think>inspect</think><answer>\\boxed{a} \\boxed{b}</answer>'
    assert format_reward(multiple_boxes, symbolic=False, grounding=True) == 0.5


def test_sequence_surrogate_clips_positive_and_negative_advantages():
    result = gspo_clipped_surrogate(
        rewards=[0.0, 1.0],
        new_logps=[[math.log(0.9995)], [math.log(1.0005)]],
        old_logps=[[0.0], [0.0]],
        clip_low=0.0003,
        clip_high=0.0004,
    )
    expected_advantage = 1 / math.sqrt(2)
    assert result["advantages"] == pytest.approx([-expected_advantage, expected_advantage])
    assert result["sequence_ratios"] == pytest.approx([0.9995, 1.0005])
    assert result["objective"] == pytest.approx(((-0.9997) + 1.0004) * expected_advantage / 2)


def test_recipe_contract_requires_open_ended_and_known_routes():
    recipe = {"categories": ["anatomy", "longitudinal", "open_ended"],
              "routes": {"anatomy": "exact_match", "longitudinal": "longitudinal_direction",
                         "open_ended": "open_ended_judge"},
              "group_size": 8, "clip_low": 0.0003, "clip_high": 0.0004,
              "max_tokens": 4096, "overlong_buffer": 2048}
    assert validate_recipe(recipe)["categories"] == 3
    without_open = dict(recipe, categories=["anatomy", "longitudinal"])
    with pytest.raises(ValueError, match="open_ended"):
        validate_recipe(without_open)


def test_saved_mri_recipe_is_coherent_and_explicitly_not_train_ready():
    path = Path(__file__).parents[1] / "configs" / "vero_mri_recipe.json"
    config = json.loads(path.read_text())
    result = validate_recipe(config)
    assert result["categories"] == 5
    assert result["equal_category_weight"] == pytest.approx(0.2)
    assert result["train_ready"] is False
    assert config["rl_weight_updates"] == "NOT_RUN"
