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
