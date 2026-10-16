"""Optional local regression over saved *actual* MRI model generations.

Private image-linked outputs are never committed. CI without local_data skips;
on the research machine this test fails if the saved execution contract drifts.
"""

import json
from pathlib import Path

import pytest

from mri_rl.ds007045_internal_dev_eval import score_outputs
from mri_rl.vero_mri_reward import (
    _answer_and_format, compute_score_from_data_source,
)


ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "local_data" / "ds007045_bounded" / "internal_dev_eval_v1"


def rows(path):
    return {r["record_id"]: r for r in map(json.loads, path.open())}


@pytest.mark.skipif(not PRIVATE.is_dir(), reason="private real-MRI receipts unavailable")
def test_saved_real_image_generation_reward_and_prompt_contract():
    manifest = PRIVATE / "internal_dev_manifest_sft_train_matched.jsonl"
    gold = PRIVATE / "internal_dev_gold_sft_train_matched_PRIVATE.jsonl"
    final = PRIVATE / "sft28_matched_native_outputs_PRIVATE.jsonl"
    start = PRIVATE / "sft0_matched_native_outputs_PRIVATE.jsonl"
    first_probe = PRIVATE / "sft28_native_outputs_PRIVATE.jsonl"
    if not all(path.is_file() for path in (manifest, gold, final, start, first_probe)):
        pytest.skip("complete private real-generation receipts unavailable")
    scored_final = score_outputs(manifest, gold, final)
    scored_start = score_outputs(manifest, gold, start)
    assert (scored_final["rows"], scored_final["format_valid"],
            scored_final["accuracy_correct"]) == (54, 54, 36)
    assert (scored_start["rows"], scored_start["format_valid"],
            scored_start["accuracy_correct"]) == (54, 0, 0)
    actual = rows(final)
    targets = rows(gold)
    correct = incorrect = 0
    for key, row in actual.items():
        assert row["response"].startswith("<think>")
        answer, format_value = _answer_and_format(row["response"])
        assert format_value == 1 and answer
        target = targets[key]
        grade = compute_score_from_data_source(
            "OpenNeuro_ds007045", row["response"], target["ground_truth"],
            target["extra_info"])
        correct += grade["accuracy"] == 1
        incorrect += grade["accuracy"] == 0
    assert (correct, incorrect) == (36, 18)

    # These are saved model outputs, not hand-written ideal strings.
    user_only = rows(first_probe)
    box_semantic_correct = 0
    for key, row in user_only.items():
        target = targets[key]
        strict = compute_score_from_data_source(
            "OpenNeuro_ds007045", row["response"], target["ground_truth"],
            target["extra_info"])
        semantic = compute_score_from_data_source(
            "OpenNeuro_ds007045", row["response"], target["ground_truth"],
            target["extra_info"], single_box_anywhere=True)
        assert strict["accuracy"] == strict["format"] == 0
        box_semantic_correct += semantic["accuracy"]
    assert box_semantic_correct == 35


@pytest.mark.skipif(not PRIVATE.is_dir(), reason="private real-MRI receipts unavailable")
def test_saved_real_generation_prefix_and_truncation_boundaries():
    contract_dir = ROOT / "local_data" / "ds007045_bounded" / "experiment_contract_sft28_matched_v1"
    diagnostic = ROOT / "local_data" / "ds007045_bounded" / "official_prompt_format_diagnostic_v1"
    if not (contract_dir / "contract.json").is_file() or not (diagnostic / "raw_rollouts.json").is_file():
        pytest.skip("private prompt/token receipts unavailable")
    contract = json.loads((contract_dir / "contract.json").read_text())
    rendered = (contract_dir / "rendered_prompt.txt").read_text()
    assert rendered.endswith(contract["assistant_prefix"])
    assert contract["response_includes_prefill"] is False
    assert not contract["assistant_prefix"].endswith("<think>")
    rollouts = json.loads((diagnostic / "raw_rollouts.json").read_text())
    generated = [item for group in rollouts for item in group["rollouts"]]
    assert any(len(item["token_ids"]) == 512 for item in generated)
    assert all(_answer_and_format(item["text"])[1] == 0 for item in generated)


@pytest.mark.skipif(not PRIVATE.is_dir(), reason="private real-MRI receipts unavailable")
def test_saved_exploratory_task_reward_update_and_adapter_reload():
    # This validates the existing *relaxed-reward* engineering run. It is not
    # evidence that the unchanged strict Vero reward has produced an update.
    arm_dir = ROOT / "local_data" / "ds007045_bounded" / "rl_proportional_single_box_8group_v1"
    predictions = ROOT / "local_data" / "frozen_eval_v1" / "ds007045_rl_proportional_single_box_256.jsonl"
    receipt_path = arm_dir / "arm_receipt_PRIVATE.json"
    if not receipt_path.is_file() or not predictions.is_file():
        pytest.skip("exploratory update/reload receipts unavailable")
    receipt = json.loads(receipt_path.read_text())
    assert receipt["reward_profile"] == "single_box_anywhere_ablation"
    active = [row for row in receipt["group_results"] if row["optimizer_steps"]]
    assert len(active) == receipt["optimizer_steps"] == 1
    group = active[0]
    assert group["task_reward_variance"] is True
    assert sorted(group["accuracy_scores"]) == [0.0, 1.0]
    assert group["format_scores"] == [0.0, 0.0]
    assert group["before_adapter_sha256"] != group["after_adapter_sha256"]
    step = json.loads((arm_dir / f"step_{group['step']:02d}" / "receipt.json").read_text())
    assert step["gradient_l2"] > 0
    reloaded = [json.loads(line) for line in predictions.open()]
    assert len(reloaded) == 107
    assert all(row["adapter_checkpoint_sha256"] == receipt["final_adapter_sha256"] and
               row["adapter_verified_parameter_tensors"] > 0 for row in reloaded)
