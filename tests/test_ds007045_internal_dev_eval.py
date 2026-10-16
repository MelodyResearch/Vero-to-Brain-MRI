"""Synthetic patient-split tests; no real MRI labels or identifiers are committed."""

import hashlib
import importlib
import json

import pytest

ROUTES = ("mri_sequence", "mri_tumor_presence", "mri_enhancing_presence",
          "mri_tumor_grid")
TRUTHS = ("FLAIR", "yes", "no", "center-middle")


def _row(image, patient, split, route, truth, ordinal):
    return {
        "id": f"{patient}_{ordinal}", "ability": route,
        "data_source": "OpenNeuro_ds007045", "images": [str(image)],
        "prompt": [{"role": "user", "content": [
            {"type": "image", "image": str(image)},
            {"type": "text", "text": "Inspect the MRI."}]}],
        "reward_model": {"ground_truth": truth},
        "extra_info": {"patient_key": patient, "patient_lineage": "BIDS_SUBJECT_VERIFIED",
                       "internal_split": split, "reward_type": route},
    }


def _files(tmp_path):
    image = tmp_path / "scan.png"
    image.write_bytes(b"synthetic image")
    train = tmp_path / "train.verl.jsonl"
    dev = tmp_path / "internal_dev.verl.jsonl"
    train.write_text("".join(json.dumps(_row(image, "train_patient", "train", route,
                                                  truth, i)) + "\n"
                             for i, (route, truth) in enumerate(zip(ROUTES, TRUTHS))))
    dev.write_text("".join(json.dumps(_row(image, "dev_patient", "internal_dev", route,
                                                truth, i)) + "\n"
                           for i, (route, truth) in enumerate(zip(ROUTES, TRUTHS))))
    return train, dev


def test_dev_manifest_separates_gold_and_scores_verified_adapter_outputs(tmp_path):
    dev_eval = importlib.import_module("mri_rl.ds007045_internal_dev_eval")
    train, dev = _files(tmp_path)
    out = tmp_path / "private"
    receipt = dev_eval.build_manifest(train, dev, out, expected_rows=4,
                                      expected_patients=1, private_root=tmp_path)
    manifest = out / "internal_dev_manifest.jsonl"
    gold = out / "internal_dev_gold_PRIVATE.jsonl"
    manifest_rows = [json.loads(x) for x in manifest.read_text().splitlines()]
    assert len(manifest_rows) == 4
    assert all(set(row) == {"record_id", "category", "image", "prompt"}
               for row in manifest_rows)
    assert {row["record_id"] for row in manifest_rows} == {
        "dev_patient_0", "dev_patient_1", "dev_patient_2", "dev_patient_3"}
    assert receipt["dev_patients"] == 1 and receipt["train_dev_disjoint"] is True
    assert "ground_truth" not in manifest.read_text()
    assert all(row["internal_split"] == "internal_dev"
               for row in (json.loads(x) for x in gold.read_text().splitlines()))
    outputs = tmp_path / "adapter_outputs.jsonl"
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    responses = (
        r"<think>visible.</think><answer>\boxed{FLAIR}</answer>",
        r"<think>visible.</think><answer>\boxed{no}</answer>",
        r"No tags, but \boxed{no}",
        r"<think>visible.</think><answer>\boxed{center-middle}</answer>",
    )
    outputs.write_text("".join(json.dumps({
        "record_id": row["record_id"], "category": row["category"],
        "response": response, "manifest_sha256": manifest_sha,
        "prompt": row["prompt"],
        "image_sha256": hashlib.sha256((tmp_path / "scan.png").read_bytes()).hexdigest(),
        "image_placeholder_count": 1,
        "model_path": "/synthetic/qwen", "model_config_sha256": "c" * 64,
        "model_template_sha256": "d" * 64,
        "max_tokens": 256, "temperature": 0,
        "adapter_checkpoint_sha256": "a" * 64, "adapter_config_sha256": "b" * 64,
        "adapter_verified_parameter_tensors": 10, "template_override_sha256": None,
    }) + "\n" for row, response in zip(manifest_rows, responses)))
    result = dev_eval.score_outputs(manifest, gold, outputs)
    assert result["rows"] == 4 and result["patients"] == 1
    assert result["format_valid"] == 3 and result["accuracy_correct"] == 2
    assert result["by_category"]["mri_sequence"]["accuracy_correct"] == 1
    assert result["by_category"]["mri_enhancing_presence"]["format_valid"] == 0
    assert set(result["by_category"]["mri_sequence"]) == {
        "rows", "format_valid", "accuracy_correct", "mean_reward"}


@pytest.mark.parametrize("field,replacement,match", [
    ("prompt", [], "prompt or image"),
    ("image_sha256", "0" * 64, "prompt or image"),
    ("image_placeholder_count", 0, "prompt or image"),
    ("model_config_sha256", "0" * 64, "one adapter, model"),
    ("model_template_sha256", "0" * 64, "one adapter, model"),
    ("max_tokens", 128, "one adapter, model"),
    ("temperature", 1, "nondeterministic"),
])
def test_dev_scorer_rejects_mismatched_per_row_receipts(tmp_path, field, replacement, match):
    dev_eval = importlib.import_module("mri_rl.ds007045_internal_dev_eval")
    train, dev = _files(tmp_path)
    out = tmp_path / "private"
    dev_eval.build_manifest(train, dev, out, expected_rows=4,
                            expected_patients=1, private_root=tmp_path)
    manifest = out / "internal_dev_manifest.jsonl"
    gold = out / "internal_dev_gold_PRIVATE.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    image_sha = hashlib.sha256((tmp_path / "scan.png").read_bytes()).hexdigest()
    predictions = [{
        "record_id": row["record_id"], "category": row["category"],
        "response": "<think>Visible.</think><answer>\\boxed{FLAIR}</answer>",
        "manifest_sha256": manifest_sha, "prompt": row["prompt"],
        "image_sha256": image_sha, "image_placeholder_count": 1,
        "model_path": "/synthetic/qwen", "model_config_sha256": "c" * 64,
        "model_template_sha256": "d" * 64,
        "max_tokens": 256, "temperature": 0,
        "adapter_checkpoint_sha256": "a" * 64,
        "adapter_config_sha256": "b" * 64,
        "adapter_verified_parameter_tensors": 10,
        "template_override_sha256": None,
    } for row in rows]
    predictions[0][field] = replacement
    outputs = tmp_path / "outputs.jsonl"
    outputs.write_text("".join(json.dumps(row) + "\n" for row in predictions))
    with pytest.raises(ValueError, match=match):
        dev_eval.score_outputs(manifest, gold, outputs)


def test_dev_manifest_rejects_test_split_and_patient_overlap(tmp_path):
    dev_eval = importlib.import_module("mri_rl.ds007045_internal_dev_eval")
    train, dev = _files(tmp_path)
    rows = [json.loads(x) for x in dev.read_text().splitlines()]
    rows[0]["extra_info"]["internal_split"] = "test"
    dev.write_text("".join(json.dumps(x) + "\n" for x in rows))
    with pytest.raises(ValueError, match="internal_dev"):
        dev_eval.build_manifest(train, dev, tmp_path / "out", expected_rows=4,
                                expected_patients=1, private_root=tmp_path)
    rows[0]["extra_info"]["internal_split"] = "internal_dev"
    rows[0]["extra_info"]["patient_key"] = "train_patient"
    dev.write_text("".join(json.dumps(x) + "\n" for x in rows))
    with pytest.raises(ValueError, match="patient overlap"):
        dev_eval.build_manifest(train, dev, tmp_path / "out", expected_rows=4,
                                expected_patients=2, private_root=tmp_path)


def test_dev_scorer_rejects_test_gold_row(tmp_path):
    dev_eval = importlib.import_module("mri_rl.ds007045_internal_dev_eval")
    train, dev = _files(tmp_path)
    out = tmp_path / "private"
    dev_eval.build_manifest(train, dev, out, expected_rows=4,
                            expected_patients=1, private_root=tmp_path)
    gold = out / "internal_dev_gold_PRIVATE.jsonl"
    rows = [json.loads(x) for x in gold.read_text().splitlines()]
    rows[0]["internal_split"] = "test"
    gold.write_text("".join(json.dumps(x) + "\n" for x in rows))
    outputs = tmp_path / "outputs.jsonl"
    outputs.write_text(gold.read_text())
    with pytest.raises(ValueError, match="internal-dev gold"):
        dev_eval.score_outputs(out / "internal_dev_manifest.jsonl", gold,
                               outputs)


def test_sft_matched_manifest_adds_exact_training_system_without_overwriting_original(tmp_path):
    dev_eval = importlib.import_module("mri_rl.ds007045_internal_dev_eval")
    train, dev = _files(tmp_path)
    out = tmp_path / "private"
    original = dev_eval.build_manifest(train, dev, out, expected_rows=4,
                                       expected_patients=1, private_root=tmp_path)
    original_path = out / "internal_dev_manifest.jsonl"
    original_bytes = original_path.read_bytes()
    matched = dev_eval.build_manifest(
        train, dev, out, expected_rows=4, expected_patients=1,
        private_root=tmp_path, prompt_profile="sft_train_matched")
    matched_path = out / "internal_dev_manifest_sft_train_matched.jsonl"
    assert matched_path.is_file() and original_path.read_bytes() == original_bytes
    assert matched["prompt_profile"] == "sft_train_matched"
    assert matched["manifest_sha256"] != original["manifest_sha256"]
    messages = json.loads(matched_path.read_text().splitlines()[0])["prompt"]
    assert messages[0] == {"role": "system", "content":
                           "Respond with one short observation in <think> and one boxed final answer in <answer>."}
    assert messages[1]["role"] == "user"
    assert (out / "internal_dev_gold_sft_train_matched_PRIVATE.jsonl").is_file()
