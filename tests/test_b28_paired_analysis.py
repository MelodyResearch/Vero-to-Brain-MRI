"""Synthetic 54-question scorer fixtures; never real model experiment results."""
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest


def module():
    path = Path(__file__).parents[1] / "scripts/analyze_b28_paired.py"
    assert path.exists(), "New B28 paired analysis implementation is missing"
    spec = importlib.util.spec_from_file_location("b28_analysis", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def write(path, rows):
    path.write_text("".join(json.dumps(x) + "\n" for x in rows))


@pytest.fixture
def synthetic(tmp_path):
    a0 = "502084c8a1394fdffd765741b9b0fbfcb6720268dfe7671c20d7a67dc5f86088"
    a28 = "244c26fd5ce2e3b02b46703d4f46a90608f2ef1a868eeee682ee4fe48a58ed73"
    image = tmp_path / "synthetic.bin"; image.write_bytes(b"SYNTHETIC IMAGE BYTES")
    adapter = tmp_path / "synthetic_adapter.safetensors"; adapter.write_bytes(b"SYNTHETIC ADAPTER NOT LOADABLE")
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    manifest, gold = tmp_path / "manifest.jsonl", tmp_path / "gold.jsonl"
    prompts, targets = [], []
    tasks = ["mri_sequence"] * 24 + ["mri_tumor_presence"] * 12 + ["mri_enhancing_presence"] * 12 + ["mri_tumor_grid"] * 6
    for i, task in enumerate(tasks):
        truth = ("FLAIR" if i % 2 == 0 else "contrast-enhanced T1") if task == "mri_sequence" else ("yes" if i % 2 == 0 else "no") if task != "mri_tumor_grid" else "center-middle"
        patient = f"synthetic_p{i//6}" if task == "mri_sequence" else f"synthetic_p{i%4}"
        key = f"{patient}_z{[45,91,136][(i%6)//2]:03d}_{'flair' if i%2==0 else 't1ce'}_sequence" if task == "mri_sequence" else f"synthetic_{i}"
        record_image = tmp_path / f"image_{i}.bin"; record_image.write_bytes(f"SYNTHETIC IMAGE {i}".encode())
        prompts.append({"record_id": key, "category": task, "image": str(record_image), "prompt": [{"role": "user", "content": [{"type": "image", "image": str(record_image)}, {"type": "text", "text": "Synthetic test only"}]}]})
        targets.append({"record_id": key, "category": task, "patient_key": patient, "internal_split": "internal_dev", "ground_truth": truth, "extra_info": {"reward_type": task}})
    write(manifest, prompts); write(gold, targets)
    outputs = {}
    for name, checkpoint in [("A0", a0), ("A28", a28), ("B28", sha(adapter))]:
        data = []
        for i, p in enumerate(prompts):
            answer = "FLAIR" if p["category"] == "mri_sequence" else "center-middle" if p["category"] == "mri_tumor_grid" else "yes"
            if name == "B28" and i == 1: answer = "contrast-enhanced T1"
            response = "synthetic invalid" if name == "A0" else "<think>Synthetic.</think><answer>\\boxed{" + answer + "}</answer>"
            data.append({**p, "response": response, "manifest_sha256": sha(manifest), "image_sha256": sha(Path(p["image"])), "image_placeholder_count": 1, "template_override_sha256": None, "adapter_verified_parameter_tensors": 504, "model_path": "/SYNTHETIC_MODEL", "model_config_sha256": "synthetic_config", "model_template_sha256": "synthetic_template", "adapter_checkpoint_sha256": checkpoint, "adapter_config_sha256": "synthetic_lora", "max_tokens": 256, "temperature": 0})
        path = tmp_path / f"{name}.jsonl"; write(path, data); outputs[name] = path
    receipt = tmp_path / "synthetic_receipt.json"
    receipt.write_text(json.dumps({"status": "SFT_ENGINEERING_CONTROL", "optimizer_steps": 28, "completed_control_28steps": True, "checkpoint_reloaded": True, "after_adapter_sha256": sha(adapter), "original_starting_adapter_sha256": a0, "schedule_sha256": "2e5d6bcb55cfb949035986fa8e65debdf13013badb04c275d195fd96be661f67", "executed_prefix_examples": 112, "exact_resume_supported": True, "internal_train_only": True, "heldout_test_used": False}))
    return {"manifest": manifest, "gold": gold, "outputs": outputs, "b28_receipt": receipt, "b28_adapter": adapter}


def test_complete_synthetic_paired_analysis(synthetic):
    result = module().analyze(**synthetic)
    assert result["models"]["A28"]["accuracy_correct"] == 30
    assert result["models"]["B28"]["accuracy_correct"] == 31
    assert result["B28_minus_A28"]["gained"] == 1
    assert result["B28_minus_A28"]["lost"] == 0
    assert result["B28_minus_A28"]["patient_mean_difference"] == pytest.approx(1 / 56)
    assert result["models"]["B28"]["termination"]["unknown"] == 54
    assert "p_value" not in result["B28_minus_A28"]


@pytest.mark.parametrize("defect", ["missing", "duplicate", "prompt", "image", "config", "checkpoint"])
def test_rejects_incomplete_or_mismatched_predictions(synthetic, defect):
    path = synthetic["outputs"]["B28"]
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    if defect == "missing": rows.pop()
    elif defect == "duplicate": rows.append(rows[0])
    elif defect == "prompt": rows[0]["prompt"] = [{"role": "user", "content": "Changed"}]
    elif defect == "image": rows[0]["image_sha256"] = "changed"
    elif defect == "config":
        for row in rows: row["model_config_sha256"] = "changed"
    elif defect == "checkpoint":
        for row in rows: row["adapter_checkpoint_sha256"] = "0" * 64
    write(path, rows)
    with pytest.raises(ValueError): module().analyze(**synthetic)


def test_rejects_partial_training_as_b28(synthetic):
    p = synthetic["b28_receipt"]; value = json.loads(p.read_text()); value["optimizer_steps"] = 11
    p.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="28"): module().analyze(**synthetic)


def test_termination_counts_are_observed_not_inferred(synthetic):
    p = synthetic["outputs"]["B28"]; rows = [json.loads(x) for x in p.read_text().splitlines()]
    for row in rows: row.update({"generation_tokens": 20, "finish_reason": "stop"})
    rows[0].update({"generation_tokens": 256, "finish_reason": "length"}); write(p, rows)
    result = module().analyze(**synthetic)
    assert result["models"]["B28"]["termination"] == {"stop": 53, "length": 1, "unknown": 0}


def test_partial_format_and_answer_only_do_not_become_full_format_accuracy(synthetic):
    p = synthetic["outputs"]["B28"]; rows = [json.loads(x) for x in p.read_text().splitlines()]
    rows[0]["response"] = "<think>Synthetic.</think><answer>FLAIR</answer>"
    rows[1]["response"] = r"\boxed{contrast-enhanced T1}"
    write(p, rows)
    result = module().analyze(**synthetic)["models"]["B28"]
    assert result["accuracy_correct"] == 30  # Existing strict-parser reward contract.
    assert result["full_format_accuracy_correct"] == 29
    assert result["format_valid"] == 52
    assert result["by_category"]["mri_sequence"]["prediction_counts"]["INVALID"] == 1
    assert result["by_category"]["mri_sequence"]["diagnostic_single_box_anywhere_prediction_counts"]["contrast-enhanced t1"] == 1


def test_unknown_finish_retains_actual_token_count(synthetic):
    p = synthetic["outputs"]["B28"]; rows = [json.loads(x) for x in p.read_text().splitlines()]
    rows[0].update({"generation_tokens": 256, "finish_reason": None}); write(p, rows)
    result = module().analyze(**synthetic)["models"]["B28"]
    assert result["termination"] == {"stop": 0, "length": 0, "unknown": 54}
    assert result["observed_generation_tokens"]["n_observed"] == 1
    assert result["observed_generation_tokens"]["max"] == 256


def controls(synthetic):
    import copy
    prompts = [json.loads(x) for x in synthetic["manifest"].read_text().splitlines()][:24]
    original = [json.loads(x) for x in synthetic["outputs"]["B28"].read_text().splitlines()][:24]
    result = {}
    for kind, suffix in [("no_image", "__no_image"), ("swap", "__same_slice_sequence_swap")]:
        manifest_rows, outputs = [], []
        for i, base in enumerate(prompts):
            row = copy.deepcopy(base); row["record_id"] += suffix
            row["image"] = None if kind == "no_image" else prompts[i ^ 1]["image"]
            content = row["prompt"][0]["content"]
            if kind == "no_image": row["prompt"][0]["content"] = [x for x in content if x["type"] != "image"]
            else: content[0]["image"] = row["image"]
            manifest_rows.append(row)
            prediction = {**original[i], **row, "image_placeholder_count": 0 if kind == "no_image" else 1,
                          "image_sha256": None if kind == "no_image" else hashlib.sha256(Path(row["image"]).read_bytes()).hexdigest()}
            # Always FLAIR means balanced 12/24 accuracy and zero directed flips.
            prediction["response"] = r"<think>Synthetic.</think><answer>\boxed{FLAIR}</answer>"
            outputs.append(prediction)
        manifest = synthetic["manifest"].parent / f"{kind}_manifest.jsonl"; write(manifest, manifest_rows)
        for row in outputs: row["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
        path = manifest.with_name(f"{kind}_outputs.jsonl"); write(path, outputs)
        result[kind] = {"manifest": manifest, "outputs": path}
    return result


def test_complete_controls_score_visible_swap_targets(synthetic):
    result = module().analyze_controls(synthetic["manifest"], synthetic["gold"], synthetic["outputs"]["B28"], controls(synthetic))
    assert result["conditions"]["swap"]["accuracy_correct"] == 12
    assert result["conditions"]["no_image"]["accuracy_correct"] == 12
    # B's one correct T1 answer flips to correct FLAIR under the actual swap.
    assert result["correct_direction_changes"] == 1


@pytest.mark.parametrize("defect", ["missing", "wrong_visible_image", "changed_prompt", "changed_config"])
def test_controls_reject_incomplete_or_unmatched_inputs(synthetic, defect):
    data = controls(synthetic); path = data["swap"]["outputs"]
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    if defect == "missing": rows.pop()
    elif defect == "wrong_visible_image": rows[0]["image_sha256"] = "wrong"
    elif defect == "changed_prompt": rows[0]["prompt"] = []
    elif defect == "changed_config": rows[0]["max_tokens"] = 128
    write(path, rows)
    with pytest.raises(ValueError): module().analyze_controls(synthetic["manifest"], synthetic["gold"], synthetic["outputs"]["B28"], data)


def test_equal_task_patient_macro_distinguishes_unequal_task_coverage():
    # Synthetic: task a has two questions for p1, one for p2; task b only p1.
    rows = {
        "a1": {"task": "a", "patient_key": "p1", "correct": 1, "full_format_correct": 1},
        "a2": {"task": "a", "patient_key": "p1", "correct": 1, "full_format_correct": 0},
        "a3": {"task": "a", "patient_key": "p2", "correct": 0, "full_format_correct": 0},
        "b1": {"task": "b", "patient_key": "p1", "correct": 0, "full_format_correct": 0},
    }
    result = module().task_patient_metrics(rows)
    assert result["equal_task_macro_question_accuracy"] == pytest.approx(1 / 3)
    assert result["equal_task_macro_within_patient_task_accuracy"] == pytest.approx(1 / 4)
    assert result["full_format_equal_task_macro_question_accuracy"] == pytest.approx(1 / 6)
    assert result["full_format_equal_task_macro_within_patient_task_accuracy"] == pytest.approx(1 / 8)
    assert result["task_patient_coverage"]["a"] == {"patients": 2, "questions": 3, "questions_per_patient_sorted": [1, 2]}
    assert result["task_patient_coverage"]["b"]["patients"] == 1


def test_control_full_format_metrics_reject_partial_format(synthetic):
    data = controls(synthetic); path = data["no_image"]["outputs"]
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    rows[0]["response"] = "<think>Synthetic.</think><answer>FLAIR</answer>"
    write(path, rows)
    result = module().analyze_controls(synthetic["manifest"], synthetic["gold"], synthetic["outputs"]["B28"], data)
    scores = result["conditions"]["no_image"]
    assert scores["accuracy_correct"] == 12
    assert scores["full_format_accuracy_correct"] == 11
    assert scores["full_format_confusion"]["flair"]["INVALID"] == 1
    assert scores["full_format_patient_mean_accuracy"] == pytest.approx(11 / 24)


def test_same_family_analysis_never_claims_cross_model_review(synthetic):
    result = module().analyze(**synthetic)
    assert result['scope'] == 'PAIRED_INTERNAL_DEV_CHECKPOINT_ANALYSIS'
    assert result['cross_model_review'] == 'NOT_VERIFIED'


def test_historical_output_guard_rejects_changed_responses(tmp_path, monkeypatch):
    m = module(); paths = {}
    for name in ['A0', 'A28']:
        path = tmp_path / (name + '.jsonl'); path.write_text('{"response":"synthetic original"}\n'); paths[name] = path
    expected = {name: m.digest(path) for name, path in paths.items()}
    monkeypatch.setattr(m, 'HISTORICAL_OUTPUT_SHA', expected)
    m.verify_historical_outputs(paths)
    paths['A28'].write_text('{"response":"synthetic edited"}\n')
    with pytest.raises(ValueError, match='Historical'):
        m.verify_historical_outputs(paths)
