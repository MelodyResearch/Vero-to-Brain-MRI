"""Fail-closed descriptive A0/A28/B28 scoring; no model execution.

Raw predictions, patient outcomes and receipts must remain under local_data.
This analyses recorded evidence; it cannot independently certify the optimizer
chain or manufacture model predictions. Four dev patients are not 54 trials.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import copy
import importlib.util
import json
from pathlib import Path
import re
from statistics import mean

from mri_rl.ds007045_internal_dev_eval import _keyed_jsonl

_spec = importlib.util.spec_from_file_location(
    "existing_balanced_audit", Path(__file__).with_name("audit_balanced_sft_comparison.py"))
_audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_audit)
digest = _audit.digest

A0_SHA = "502084c8a1394fdffd765741b9b0fbfcb6720268dfe7671c20d7a67dc5f86088"
A28_SHA = "244c26fd5ce2e3b02b46703d4f46a90608f2ef1a868eeee682ee4fe48a58ed73"
B_SCHEDULE_SHA = "2e5d6bcb55cfb949035986fa8e65debdf13013badb04c275d195fd96be661f67"
MANIFEST_SHA = "9e5194cc808ad27592549b3fab21e31fd6ba0a9582ca76fac75928b589b9734f"
GOLD_SHA = "0630ec7c91ed67f1059294422f8eb937435546bbf446155d187365862fda8961"
HISTORICAL_OUTPUT_SHA = {
    "A0": "578e523bbbedae6f221ddc5732817e15df2b2f257d4bc23a4fb77d4d85afde59",
    "A28": "c65f5e4343f8f9bc3ee6d77c71c5a87dbd794b7280f5c7c85cf937d515142d9e",
}
CATEGORY_COUNTS = {"mri_sequence": 24, "mri_tumor_presence": 12,
                   "mri_enhancing_presence": 12, "mri_tumor_grid": 6}
COMMON_FIELDS = ("model_path", "model_config_sha256", "model_template_sha256",
                 "adapter_config_sha256", "max_tokens", "temperature")
CONTROL_NAMES = {
    "no_image": ("internal_dev_manifest_sft28_sequence_no_image_PRIVATE.jsonl",
                 "sft28_sequence_no_image_outputs_PRIVATE.jsonl",
                 "420c0dc5a152d3dc3e9d5e1505525f038537bf8b06a245fdc03452fa369061d4"),
    "swap": ("internal_dev_manifest_sft28_sequence_swap_PRIVATE.jsonl",
             "sft28_sequence_swap_outputs_PRIVATE.jsonl",
             "c2f709242f25d5a3d3621bfd1f0f40f59d6ffcf6320613bfcf363d4d375aad11"),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_historical_outputs(paths):
    require(set(paths) == set(HISTORICAL_OUTPUT_SHA), "Historical output inventory changed")
    for name, expected in HISTORICAL_OUTPUT_SHA.items():
        require(digest(paths[name]) == expected, "Historical " + name + " output bytes changed")


def termination_counts(records):
    counts = {"stop": 0, "length": 0, "unknown": 0}
    tokens = []
    for row in records:
        finish, n = row.get("finish_reason"), row.get("generation_tokens")
        require(finish in {None, "stop", "length"}, "Invalid generation termination metadata")
        counts[finish or "unknown"] += 1
        if n is not None:
            require(type(n) is int and 0 <= n <= row["max_tokens"], "Invalid generation token count")
            tokens.append(n)
    return counts, ({"n_observed": len(tokens), "min": min(tokens),
                     "max": max(tokens), "mean": mean(tokens)} if tokens else
                    {"n_observed": 0, "status": "NOT_MEASURED"})


def analyze_controls(manifest, gold, original_outputs, controls):
    """Validate complete24 paired controls and score the replacement image label."""
    require(set(controls) == {"no_image", "swap"}, "Both complete control conditions required")
    original_summary = _audit.detailed_score(manifest, gold, original_outputs)
    require(original_summary["rows"] == 54 and original_summary["patients"] == 4,
            "Controls require the completed54-question four-patient original evaluation")
    all_prompts, all_gold = _keyed_jsonl(manifest), _keyed_jsonl(gold)
    base = {key: row for key, row in all_prompts.items() if row["category"] == "mri_sequence"}
    require(len(base) == 24, "Require24 frozen sequence originals")
    original = _keyed_jsonl(original_outputs)
    coordinate = {}
    for key in base:
        match = re.fullmatch(r"(.+)_z(\d+)_(flair|t1ce)_sequence", key)
        require(match is not None and match[1] == all_gold[key]["patient_key"],
                "Cannot verify sequence patient/slice lineage")
        label = "FLAIR" if match[3] == "flair" else "contrast-enhanced T1"
        require(all_gold[key]["ground_truth"] == label, "Sequence source label/ID mismatch")
        coordinate[key] = (match[1], match[2], match[3])
    outcomes = defaultdict(dict)
    conditions = {}

    def score_condition(name, records, target_labels):
        confusion = defaultdict(Counter); predictions = Counter(); correct = valid = full_correct = 0
        patients = defaultdict(list); full_patients = defaultdict(list)
        full_confusion = defaultdict(Counter)
        for key, row in records.items():
            truth = target_labels[key]
            grade = _audit.compute_score_from_data_source("OpenNeuro_ds007045", row["response"], truth,
                                                         {"reward_type": "mri_sequence"})
            answer, formatting = _audit._answer_and_format(row["response"])
            answer = answer.lower() if answer.lower() in {"flair", "contrast-enhanced t1"} else "INVALID"
            hit = int(grade["accuracy"] > 0); correct += hit; valid += formatting == 1
            full_hit = int(hit and formatting == 1); full_correct += full_hit
            confusion[truth.lower()][answer] += 1; predictions[answer] += 1
            patient = all_gold[key]["patient_key"]; patients[patient].append(hit)
            full_patients[patient].append(full_hit)
            full_confusion[truth.lower()][answer if formatting == 1 else "INVALID"] += 1
            outcomes[key][name] = {"correct": hit, "full_format_correct": full_hit,
                                   "prediction": answer, "visible_label": truth}
        termination, tokens = termination_counts(records.values())
        return {"n": 24, "accuracy_correct": correct, "full_format_accuracy_correct": full_correct,
                "format_valid": valid,
                "invalid_answers": predictions["INVALID"], "prediction_counts": dict(predictions),
                "confusion": {label: dict(values) for label, values in confusion.items()},
                "full_format_confusion": {label: dict(values) for label, values in full_confusion.items()},
                "full_format_patient_mean_accuracy": mean(mean(v) for v in full_patients.values()),
                "patient_mean_accuracy": mean(mean(v) for v in patients.values()),
                "patient_accuracy_PRIVATE": {p: mean(v) for p, v in patients.items()},
                "termination": termination, "observed_generation_tokens": tokens}

    conditions["original"] = score_condition("original", {key: original[key] for key in base},
                                             {key: all_gold[key]["ground_truth"] for key in base})
    for kind, paths in controls.items():
        control_manifest = _keyed_jsonl(paths["manifest"])
        predicted = _keyed_jsonl(paths["outputs"])
        suffix = "__no_image" if kind == "no_image" else "__same_slice_sequence_swap"
        require(set(control_manifest) == set(predicted) == {key + suffix for key in base},
                "Require complete matched24 control records")
        control_hash = digest(paths["manifest"]); keyed = {}; targets = {}
        for key, source in base.items():
            transformed = copy.deepcopy(source["prompt"])
            patient, level, modality = coordinate[key]
            if kind == "no_image":
                expected_image = None; target = all_gold[key]["ground_truth"]
            else:
                opposite = [other for other, xyz in coordinate.items()
                            if xyz[:2] == (patient, level) and xyz[2] != modality]
                require(len(opposite) == 1, "Missing unique same-patient same-slice opposite sequence")
                expected_image = base[opposite[0]]["image"]
                target = all_gold[opposite[0]]["ground_truth"]
            for message in transformed:
                if isinstance(message.get("content"), list):
                    if expected_image is None:
                        message["content"] = [x for x in message["content"] if x.get("type") != "image"]
                    else:
                        for item in message["content"]:
                            if item.get("type") == "image": item["image"] = expected_image
            control = control_manifest[key + suffix]; row = predicted[key + suffix]
            require(control["image"] == expected_image and control["prompt"] == transformed and
                    control["category"] == "mri_sequence", "Control image/prompt intervention changed")
            require(row.get("prompt") == transformed and row.get("category") == "mri_sequence" and
                    row.get("manifest_sha256") == control_hash and
                    row.get("image_sha256") == (digest(expected_image) if expected_image else None) and
                    row.get("image_placeholder_count") == (1 if expected_image else 0),
                    "Control output image/prompt/manifest receipt mismatch")
            require(all(row.get(field) == original[key].get(field) for field in
                        (*COMMON_FIELDS, "adapter_checkpoint_sha256", "template_override_sha256",
                         "adapter_verified_parameter_tensors")), "Control model/decoding configuration mismatch")
            require(isinstance(row.get("response"), str), "Nontext control response")
            keyed[key] = row; targets[key] = target
        conditions[kind] = score_condition(kind, keyed, targets)
    directed = sum(v["original"]["correct"] and v["swap"]["correct"] and
                   v["original"]["prediction"] != v["swap"]["prediction"] for v in outcomes.values())
    return {"status": "TESTED_CPU", "n": 24, "patients": 4, "conditions": conditions,
            "correct_direction_changes": directed,
            "full_format_correct_direction_changes": sum(v["original"]["full_format_correct"] and
                v["swap"]["full_format_correct"] and v["original"]["prediction"] != v["swap"]["prediction"]
                for v in outcomes.values()),
            "changed_valid_prediction": sum(v["original"]["prediction"] != v["swap"]["prediction"] and
                                             "INVALID" not in {v["original"]["prediction"], v["swap"]["prediction"]}
                                             for v in outcomes.values()),
            "paired_outcomes_PRIVATE": dict(outcomes),
            "interpretation": "Swap accuracy uses the new visible image label. Directed change requires both original and replacement answers correct. Four development patients; descriptive only."}


def task_patient_metrics(outcomes):
    """Equal task weights, separately pooling questions or averaging patients.

    A patient without an eligible question in a task is absent from that task's
    denominator, not assigned an invented zero. Coverage is reported explicitly.
    """
    grouped = defaultdict(lambda: defaultdict(list))
    for row in outcomes.values():
        grouped[row["task"]][row["patient_key"]].append(row)
    result = {"task_patient_coverage": {
        task: {"patients": len(patients), "questions": sum(map(len, patients.values())),
               "questions_per_patient_sorted": sorted(map(len, patients.values()))}
        for task, patients in grouped.items()}}
    for field, prefix in [("correct", ""), ("full_format_correct", "full_format_")]:
        question_means = {}
        patient_means = {}
        for task, patients in grouped.items():
            question_means[task] = mean(row[field] for rows in patients.values() for row in rows)
            patient_means[task] = mean(mean(row[field] for row in rows) for rows in patients.values())
        result[prefix + "equal_task_macro_question_accuracy"] = mean(question_means.values())
        result[prefix + "equal_task_macro_within_patient_task_accuracy"] = mean(patient_means.values())
        result[prefix + "within_patient_task_accuracy"] = patient_means
    result["task_patient_denominator_rule"] = "Average observed patients within each task, then equally average tasks; patients without eligible task questions are omitted for that task, never imputed. Descriptive only."
    return result


def analyze(manifest, gold, outputs, b28_receipt, b28_adapter):
    require(set(outputs) == {"A0", "A28", "B28"}, "Require exact A0/A28/B28 model inventory")
    prompts, targets = _keyed_jsonl(manifest), _keyed_jsonl(gold)
    require(len(prompts) == len(targets) == 54 and set(prompts) == set(targets), "Require complete matched54 records")
    require(Counter(x["category"] for x in targets.values()) == CATEGORY_COUNTS,
            "Four-category54-question inventory differs")
    patient_sizes = Counter(x["patient_key"] for x in targets.values())
    require(sorted(patient_sizes.values()) == [13, 13, 14, 14], "Require four frozen development patients")
    receipt = json.loads(Path(b28_receipt).read_text())
    require(receipt.get("status") == "SFT_ENGINEERING_CONTROL" and
            type(receipt.get("optimizer_steps")) is int and receipt["optimizer_steps"] == 28 and
            receipt.get("completed_control_28steps") is True and
            receipt.get("checkpoint_reloaded") is True and receipt.get("executed_prefix_examples") == 112,
            "B28 requires completed28-update112-example reload receipt; partial training is not B28")
    require(receipt.get("original_starting_adapter_sha256") == A0_SHA and
            receipt.get("schedule_sha256") == B_SCHEDULE_SHA and
            receipt.get("exact_resume_supported") is True and
            receipt.get("internal_train_only") is True and receipt.get("heldout_test_used") is False,
            "B28 training provenance contract mismatch")
    b_sha = digest(b28_adapter)
    require(receipt.get("after_adapter_sha256") == b_sha and b_sha not in {A0_SHA, A28_SHA},
            "B28 checkpoint bytes do not match distinct completed receipt")
    expected_adapters = {"A0": A0_SHA, "A28": A28_SHA, "B28": b_sha}
    predictions = {name: _keyed_jsonl(path) for name, path in outputs.items()}
    first = next(iter(predictions["A28"].values()))
    common = {key: first.get(key) for key in COMMON_FIELDS}
    require(common["max_tokens"] == 256 and common["temperature"] == 0,
            "Historical evaluation must remain greedy with256-token cap")
    for name, rows in predictions.items():
        require(len(rows) == 54 and set(rows) == set(prompts), "Require complete matched54 predictions")
        for row in rows.values():
            require(row.get("adapter_checkpoint_sha256") == expected_adapters[name],
                    "Prediction checkpoint mismatch")
            require(all(row.get(key) == value for key, value in common.items()),
                    "Cross-model model/prompt-template/LoRA/decoding config mismatch")
            require(isinstance(row.get("response"), str), "Nontext prediction")
    # Existing scorer checks each image hash, prompt, native-template receipt,
    # adapter-load verification, manifest hash, exact IDs and strict reward.
    scored = _audit.compare(manifest, gold, outputs)
    for name, summary in scored["models"].items():
        counts, tokens = termination_counts(predictions[name].values())
        summary["termination"] = counts
        summary["observed_generation_tokens"] = tokens
        summary["response_tokens"] = tokens
        summary["truncation"] = "Actual finish_reason counts; historical unknowns are not inferred"
        summary["invalid_answers"] = sum(x["invalid_answers"] for x in summary["by_category"].values())
        full_patients = defaultdict(list)
        full_confusions = defaultdict(lambda: defaultdict(Counter))
        for key, row in predictions[name].items():
            answer, formatting = _audit._answer_and_format(row["response"])
            outcome = summary["outcomes_PRIVATE"][key]
            full_correct = int(outcome["correct"] and formatting == 1)
            outcome["full_format_correct"] = full_correct
            full_patients[outcome["patient_key"]].append(full_correct)
            prediction = outcome["prediction"] if formatting == 1 else "INVALID"
            full_confusions[outcome["task"]][outcome["ground_truth"].lower()][prediction] += 1
        for task, data in summary["by_category"].items():
            data["diagnostic_single_box_anywhere_prediction_counts"] = data.pop("semantic_single_box_counts")
            data["full_format_confusion"] = {truth: dict(counts) for truth, counts in full_confusions[task].items()}
        summary["full_format_accuracy_correct"] = sum(x["strict_correct"] for x in summary["by_category"].values())
        summary["full_format_equal_task_macro_accuracy"] = mean(x["strict_correct"] / x["n"] for x in summary["by_category"].values())
        summary["full_format_worst_task_accuracy"] = min(x["strict_correct"] / x["n"] for x in summary["by_category"].values())
        summary["full_format_patient_mean_accuracy"] = mean(mean(v) for v in full_patients.values())
        summary.update(task_patient_metrics(summary["outcomes_PRIVATE"]))
        summary["metric_semantics"] = "accuracy_correct uses unchanged Vero reward parser, including its partial format0.5 branch; full_format_accuracy_correct additionally requires format1. Diagnostic answer-only extraction never contributes to either primary count."
    a = scored["models"]["A28"]["outcomes_PRIVATE"]
    b = scored["models"]["B28"]["outcomes_PRIVATE"]
    paired = scored["paired"]["A28__vs__B28"]
    changes = [{"record_id": key, "patient_key": a[key]["patient_key"], "category": a[key]["task"],
                "A28_correct": a[key]["correct"], "B28_correct": b[key]["correct"],
                "change": b[key]["correct"] - a[key]["correct"]} for key in a]
    full_patient_deltas = defaultdict(list)
    for key in a:
        full_patient_deltas[a[key]["patient_key"]].append(b[key]["full_format_correct"] - a[key]["full_format_correct"])
    return {**scored, "scope": "PAIRED_INTERNAL_DEV_CHECKPOINT_ANALYSIS", "cross_model_review": "NOT_VERIFIED", "scientific_validation": "NOT_VERIFIED",
            "B28_training_receipt_sha256": digest(b28_receipt), "B28_adapter_sha256": b_sha,
            "B28_minus_A28": {"gained": sum(x["change"] == 1 for x in changes),
                                "lost": sum(x["change"] == -1 for x in changes),
                                "unchanged": sum(x["change"] == 0 for x in changes),
                                "patient_mean_difference": paired["patient_mean_difference_b_minus_a"],
                                "equal_task_macro_question_difference": scored["models"]["B28"]["equal_task_macro_question_accuracy"] - scored["models"]["A28"]["equal_task_macro_question_accuracy"],
                                "equal_task_macro_within_patient_task_difference": scored["models"]["B28"]["equal_task_macro_within_patient_task_accuracy"] - scored["models"]["A28"]["equal_task_macro_within_patient_task_accuracy"],
                                "full_format_equal_task_macro_within_patient_task_difference": scored["models"]["B28"]["full_format_equal_task_macro_within_patient_task_accuracy"] - scored["models"]["A28"]["full_format_equal_task_macro_within_patient_task_accuracy"],
                                "full_format_patient_mean_difference": mean(mean(v) for v in full_patient_deltas.values()),
                                "full_format_gained": sum(a[key]["full_format_correct"] == 0 and b[key]["full_format_correct"] == 1 for key in a),
                                "full_format_lost": sum(a[key]["full_format_correct"] == 1 and b[key]["full_format_correct"] == 0 for key in a),
                                "patient_outcomes_PRIVATE": paired["patient_outcomes_PRIVATE"],
                                "paired_questions_PRIVATE": changes,
                                "uncertainty": "Descriptive four-patient paired differences; no independent-question p-value or confirmatory claim."},
            "limits": ["Training completion fields and final adapter are checked; full optimizer-chain verification belongs to independent engineering review.",
                       "Historical outputs lack full processor artifact fingerprints and actual generation termination metadata.",
                       "Repeatedly examined internal development cohort; historical eight test patients unused."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--b28-outputs", type=Path, required=True)
    parser.add_argument("--b28-receipt", type=Path, required=True)
    parser.add_argument("--b28-adapter", type=Path, required=True)
    parser.add_argument("--b28-no-image-outputs", type=Path)
    parser.add_argument("--b28-swap-outputs", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    private = Path(__file__).resolve().parents[1] / "local_data"
    require(args.output.resolve().is_relative_to(private.resolve()) and not args.output.exists(),
            "New private output under ignored local_data required")
    evaluation = args.root / "internal_dev_eval_v1"
    manifest = evaluation / "internal_dev_manifest_sft_train_matched.jsonl"
    gold = evaluation / "internal_dev_gold_sft_train_matched_PRIVATE.jsonl"
    require(digest(manifest) == MANIFEST_SHA and digest(gold) == GOLD_SHA,
            "Frozen54-question manifest/gold changed")
    historical_outputs = {
        "A0": evaluation / "sft0_matched_native_outputs_PRIVATE.jsonl",
        "A28": evaluation / "sft28_matched_native_outputs_PRIVATE.jsonl"}
    verify_historical_outputs(historical_outputs)
    result = analyze(manifest, gold, {**historical_outputs, "B28": args.b28_outputs},
                     args.b28_receipt, args.b28_adapter)
    require(bool(args.b28_no_image_outputs) == bool(args.b28_swap_outputs),
            "Provide both complete24 B28 control outputs or neither")
    result["sequence_controls"] = {"status": "NOT_RUN"}
    if args.b28_no_image_outputs:
        historical = {}; corrected = {}
        new_paths = {"no_image": args.b28_no_image_outputs, "swap": args.b28_swap_outputs}
        for kind, (manifest_name, output_name, expected_sha) in CONTROL_NAMES.items():
            path = evaluation / manifest_name
            require(digest(path) == expected_sha, "Frozen24-question control manifest changed")
            historical[kind] = {"manifest": path, "outputs": evaluation / output_name}
            corrected[kind] = {"manifest": path, "outputs": new_paths[kind]}
        a_controls = analyze_controls(manifest, gold, evaluation / "sft28_matched_native_outputs_PRIVATE.jsonl", historical)
        b_controls = analyze_controls(manifest, gold, args.b28_outputs, corrected)
        result["sequence_controls"] = {"A28": a_controls, "B28": b_controls,
            "B28_minus_A28": {"full_format_correct_direction_changes": b_controls["full_format_correct_direction_changes"] - a_controls["full_format_correct_direction_changes"],
                               "full_format_condition_patient_mean_differences": {kind: b_controls["conditions"][kind]["full_format_patient_mean_accuracy"] - a_controls["conditions"][kind]["full_format_patient_mean_accuracy"] for kind in ("original", "no_image", "swap")},
                               "correct_direction_changes": b_controls["correct_direction_changes"] - a_controls["correct_direction_changes"],
                               "condition_patient_mean_differences": {kind: b_controls["conditions"][kind]["patient_mean_accuracy"] - a_controls["conditions"][kind]["patient_mean_accuracy"] for kind in ("original", "no_image", "swap")}}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "scope": result["scope"],
                      "models": {name: {key: data[key] for key in
                          ("accuracy_correct", "full_format_accuracy_correct", "format_valid",
                           "equal_task_macro_question_accuracy", "equal_task_macro_within_patient_task_accuracy",
                           "full_format_equal_task_macro_question_accuracy", "full_format_equal_task_macro_within_patient_task_accuracy",
                           "task_patient_coverage", "worst_task_accuracy", "full_format_worst_task_accuracy",
                           "patient_mean_accuracy", "full_format_patient_mean_accuracy", "termination")}
                                 for name, data in result["models"].items()},
                      "gained": result["B28_minus_A28"]["gained"],
                      "lost": result["B28_minus_A28"]["lost"]}, indent=2))
