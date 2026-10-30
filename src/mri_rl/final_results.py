"""Deterministic, standard-library-only recount of authenticated public aggregates.

This validates and recomputes published counts; it neither reruns inference nor
opens private payloads. Historical final100 and repeatedly explored DEV24 remain
separate estimands. Source SHA pins intentionally fail closed on changed evidence.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path

SOURCE_SHA256 = {
    "results/FINAL_MODEL_ELIGIBILITY_REVIEW_V1.json": "f2f759b01c015139222e2136e327bba62774b6645bdac8f2698f64ebb664ded5",
    "results/ORIGINAL22_FINAL100_ACTUAL_MATRIX_V1.json": "4ce5a22c5594a12f0c8711c4f921a337b7bc63cb2636575ae52af2e1fbd02512",
    "results/ORIGINAL22_FINAL100_CLASS_FORMAT_COST_SUPPLEMENT_V1.json": "9cd855ea076def78c2ae912fb162a7702691efdc23e8c96a8ea336662437c2f9",
    "results/ORIGINAL21_TRAINING_ACTUAL_COMPLETION_V1.json": "5915fecfca893341231272e7fc3e9f96f0d123b1302c77c4d2a84592743e3c6f",
    "results/ACCURACY_PILOT_V2_ACTUAL_RESULTS_V1.json": "386c90a30bda025f27f95b1aee65a78ca9aefbbce58f2d0186fedd7bd0760af0",
    "results/FINAL_BALANCED_SFT_SEED0_ACTUAL_REVIEW_V1.json": "cc58c2e3c4faa07e48e308f29ec195eb733c0798d99543b708645ae7a7ef2f72",
}
LABELS = ["FLAIR", "contrast-enhanced T1"]


def _count(value):
    if type(value) is not int or value < 0:
        raise ValueError("Counts must be nonnegative integers")
    return value


def _equal(actual, expected, key):
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not isinstance(actual, (int, float)) or isinstance(actual, bool):
            raise ValueError(f"Inconsistent {key}")
        if not math.isfinite(actual) or not math.isclose(actual, expected, abs_tol=1e-12, rel_tol=1e-12):
            raise ValueError(f"Inconsistent {key}: {actual} vs {expected}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"Inconsistent {key}")
        for i, (a, b) in enumerate(zip(actual, expected)):
            _equal(a, b, f"{key}[{i}]")
    elif actual != expected:
        raise ValueError(f"Inconsistent {key}")


def summarize_confusion(confusion, class_n):
    """Rows=true class; columns=the two labels, then strict INVALID."""
    if len(confusion) != 2 or len(class_n) != 2 or any(len(row) != 3 for row in confusion):
        raise ValueError("Expected binary 2x3 confusion and two denominators")
    for row, n in zip(confusion, class_n):
        if sum(_count(x) for x in row) != _count(n):
            raise ValueError("Confusion row does not match class denominator")
        if n == 0:
            raise ValueError("Empty class")
    n = sum(class_n)
    correct = confusion[0][0] + confusion[1][1]
    recalls = [confusion[i][i] / class_n[i] for i in range(2)]
    invalid = sum(row[2] for row in confusion)
    return {
        "n": n, "correct": correct, "accuracy": correct / n,
        "class_n": list(class_n), "class_correct": [confusion[0][0], confusion[1][1]],
        "class_recalls": recalls, "balanced_accuracy": sum(recalls) / 2,
        "confusion": [list(row) for row in confusion],
        "invalid": invalid, "invalid_rate": invalid / n,
        "predicted_class_counts": [sum(row[j] for row in confusion) for j in range(3)],
    }


def validate_reported(reported, computed):
    """Cross-check every recognized source metric; do not trust stored accuracy."""
    aliases = {"cases": "n", "class_counts": "class_n"}
    for key, value in reported.items():
        target = aliases.get(key, key)
        if target in computed:
            _equal(value, computed[target], key)


def _pairs(profile, metrics, correct_key):
    allowed = {correct_key, "CONSTANT_FLAIR", "CONSTANT_T1", "REVERSED", "INVALID_INVOLVED"}
    if set(profile) - allowed:
        raise ValueError("Unknown pair profile")
    p = {key: _count(profile.get(key, 0)) for key in allowed}
    count = sum(p.values())
    if 2 * count != metrics["n"]:
        raise ValueError("Pair counts do not conserve images")
    # These source studies have no invalid paired predictions.
    if p["INVALID_INVOLVED"] or metrics["invalid"]:
        raise ValueError("Cannot infer complete pair confusion with invalid pairs")
    expected = [p[correct_key] + p["CONSTANT_FLAIR"], p[correct_key] + p["CONSTANT_T1"]]
    _equal(metrics["class_correct"], expected, "pair class_correct")
    return {"pairs": count, "fully_correct_pairs": p[correct_key],
            "profile": {k: profile[k] for k in sorted(profile)}}


def _cluster_interval(counts, n):
    if len(counts) != 4 or n != 24 or sum(_count(v) for v in counts) > n or any(v > 6 for v in counts):
        raise ValueError("Expected four six-image patient clusters")
    values = sorted(sum(sample) / n for sample in itertools.product(counts, repeat=4))
    def quantile(q):
        index = q * (len(values) - 1)
        lo = math.floor(index)
        hi = math.ceil(index)
        return values[lo] + (values[hi] - values[lo]) * (index - lo)
    return [quantile(.025), quantile(.975)]


def _dev(pilot, sft):
    if pilot["protocol"]["dev_images"] != 24 or pilot["protocol"]["dev_patients"] != 4:
        raise ValueError("Unexpected DEV population")
    models = {}
    for key, name in [("Parent", "Parent_Adam52"), ("A", "Direct_Vero_A54"), ("B", "Paired_Vero_B54")]:
        row = pilot["metrics"][key]
        # Profile is sufficient to recover confusion, with explicit zero invalid.
        if row["invalid"] != 0:
            raise ValueError("Nonzero invalid requires per-class confusion evidence")
        n, c = row["class_counts"], row["class_correct"]
        metric = summarize_confusion([[c[0], n[0]-c[0], 0], [n[1]-c[1], c[1], 0]], n)
        validate_reported(row, metric)
        _equal(row["FLAIR_recall"], metric["class_recalls"][0], "FLAIR_recall")
        _equal(row["T1_recall"], metric["class_recalls"][1], "T1_recall")
        pred = row["predicted_counts"]
        _equal([pred[LABELS[0]], pred[LABELS[1]], pred["INVALID"]], metric["predicted_class_counts"], "predicted_counts")
        metric["paired_image_outcomes"] = _pairs(pilot["pair_profiles"][key], metric, "DISCRIMINATES_CORRECTLY")
        metric["optimizer_updates"] = 0 if key == "Parent" else pilot["native_training"][key]["optimizer_updates"]
        metric["descriptive_four_patient_cluster_percentile95"] = None
        metric["uncertainty_status"] = "NOT_AVAILABLE_FROM_PUBLIC_AGGREGATE"
        models[name] = metric
    for source_key, name in [("Parent", "Parent_Adam52"), ("SFT_seed0", "Balanced_SFT_Adam80_seed0")]:
        row = sft[source_key]
        metric = summarize_confusion(row["confusion"], row["class_n"])
        validate_reported(row, metric)
        if sum(row["patient_correct_counts_sorted"]) != metric["correct"] or row["patient_n"] != 4:
            raise ValueError("Patient totals do not match image totals")
        interval = _cluster_interval(row["patient_correct_counts_sorted"], metric["n"])
        _equal(row["descriptive_four_patient_cluster_percentile95"], interval, "cluster interval")
        pair = _pairs(row["pair_profile"], metric, "BOTH_CORRECT")
        _equal(row["pair_n"], pair["pairs"], "pair_n")
        _equal(row["fully_correct_pairs"], pair["fully_correct_pairs"], "fully_correct_pairs")
        if source_key == "Parent":
            validate_reported(metric, models[name])
            _equal(pair["fully_correct_pairs"], models[name]["paired_image_outcomes"]["fully_correct_pairs"], "Parent pairs")
            metric = models[name]
        else:
            metric["paired_image_outcomes"] = pair
            metric["optimizer_updates"] = sft["training"]["optimizer_updates"]
        metric["descriptive_four_patient_cluster_percentile95"] = interval
        metric["uncertainty_status"] = "EXACT_4_TO_THE_4_CLUSTER_RESAMPLES_DESCRIPTIVE_NOT_POPULATION_INFERENCE"
        models[name] = metric
    _equal(pilot["actual_budget"]["optimizer_updates"], sum(models[k]["optimizer_updates"] for k in ["Direct_Vero_A54", "Paired_Vero_B54"]), "RL updates")
    _equal(sft["training"]["Adam_end"] - sft["training"]["Adam_start"], models["Balanced_SFT_Adam80_seed0"]["optimizer_updates"], "SFT updates")
    return {"status": "REPEATEDLY_EXPLORED_INTERNAL_DEV_NOT_INDEPENDENT_HOLDOUT",
            "patients": 4, "images": 24, "class_order": LABELS,
            "confusion_columns": LABELS + ["INVALID"], "metric": "STRICT_GREEDY",
            "models": models,
            "uncertainty_limit": "Four correlated patient clusters; descriptive intervals are not confidence in generalization."}


def _original(matrix, supplement, completion):
    _equal(supplement["canonical_result_sha256"], SOURCE_SHA256["results/ORIGINAL22_FINAL100_ACTUAL_MATRIX_V1.json"], "matrix binding")
    _equal(matrix["distinct_model_inference_runs"], 22, "model runs")
    _equal(supplement["unique_model_runs"], 22, "unique model runs")
    _equal(len(supplement["models"]), 22, "model tables")
    _equal(completion["RL_arms"], 18, "RL completion")
    _equal(len(completion["SFT_controls"]), 3, "SFT completion")
    _equal(sum(row["optimizer_updates"] for row in completion["SFT_controls"]), completion["SFT_updates_total"], "SFT total updates")
    families = {}
    for row in matrix["matrix"]["arm_seed_summary"]:
        arm, seed = row["arm"], row["seed"]
        model_key = "B28" if arm == "B28" else f"sft:{seed}" if arm == "same_C_SFT" else f"rl:{arm}:{seed}"
        task_metrics = {}
        for task, task_row in supplement["models"][model_key]["tasks"].items():
            labels = LABELS if task == "mri_sequence" else ["different", "same"]
            n = [task_row["class_counts"][label]["questions"] for label in labels]
            modes = {}
            for prefix in ["native", "full_format"]:
                conf = [[task_row[f"{prefix}_confusion"][label][pred] for pred in labels + ["INVALID"]] for label in labels]
                got = summarize_confusion(conf, n)
                _equal(task_row[f"{prefix}_correct"], got["correct"], f"{model_key} {task} correct")
                _equal(task_row[f"{prefix}_question_accuracy"], got["accuracy"], f"{model_key} {task} accuracy")
                for i, label in enumerate(labels):
                    _equal(task_row["class_counts"][label][f"{prefix}_recall"], got["class_recalls"][i], "class recall")
                modes[prefix] = got
            _equal(task_row["questions"], sum(n), "task questions")
            # invalid_answers follows native parser; half-format is separately counted.
            _equal(task_row["invalid_answers"], modes["native"]["invalid"], "native invalid")
            task_metrics[task] = {"class_order": labels, "metrics": modes}
        if sorted(task_metrics) != ["mri_sequence", "mri_visit_correspondence"]:
            raise ValueError("Unexpected historical task set")
        macro = sum(t["metrics"]["native"]["accuracy"] for t in task_metrics.values()) / 2
        _equal(row["equal_task_macro_native_accuracy"], macro, "native macro")
        cells = [c for c in matrix["matrix"]["cells"] if c["arm"] == arm and c["seed"] == seed]
        if len(cells) != 2:
            raise ValueError("Expected two canonical task cells")
        for cell in cells:
            t = task_metrics[cell["evaluation_task"]]["metrics"]
            _equal(cell["native_task_accuracy"], t["native"]["accuracy"], "matrix native cell")
            _equal(cell["full_format_accuracy"], t["full_format"]["accuracy"], "matrix format cell")
        members = families.setdefault(arm, {"seed_rows": [], "native_macro_mean": None, "native_macro_range": None})
        members["seed_rows"].append({"seed": seed, "equal_task_macro_native_accuracy": macro, "tasks": task_metrics})
    for arm, family in families.items():
        family["seed_rows"].sort(key=lambda row: row["seed"])
        if [row["seed"] for row in family["seed_rows"]] != [7, 11, 13]:
            raise ValueError("Missing or duplicate original seeds")
        scores = [row["equal_task_macro_native_accuracy"] for row in family["seed_rows"]]
        family["native_macro_mean"] = sum(scores) / len(scores)
        family["native_macro_range"] = [min(scores), max(scores)]
        reported = matrix["matrix"]["between_seed_summary"][arm]
        _equal(reported["macro_native_accuracy_mean"], family["native_macro_mean"], "seed mean")
        _equal(reported["macro_native_accuracy_range"], family["native_macro_range"], "seed range")
    return {"status": "COMPLETED_HISTORICAL_FROZEN_EVALUATION_NOT_RERUN",
            "unique_model_runs": 22, "questions_per_model": 100,
            "sequence_questions_per_model": 84, "Dallas_questions_per_model": 16,
            "completed_RL_arms": 18, "completed_SFT_controls": 3,
            "families": families,
            "baseline_reuse": "B28 has one inference run reused in three seed rows, not three replications.",
            "metric_limit": "native correctness and full-format correctness differ; Dallas is an engineering control, not a second qualified clinical skill.",
            "uncertainty_limit": "Seeds do not add patients. Original tasks and DEV24 are not pooled or ranked across datasets."}


def build_final_metrics(root):
    """Recount only the six SHA-pinned, public aggregate JSON inputs."""
    source = {}
    for relative, expected in SOURCE_SHA256.items():
        data = (Path(root) / relative).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"SHA256 mismatch: {relative}")
        source[relative.rsplit("/", 1)[-1]] = json.loads(data)
    return {
        "schema": "vero-to-brain-mri-final-metrics-v1",
        "reproduction": "CPU_SAVED_PUBLIC_AGGREGATE_RECOUNT_NOT_NEW_MODEL_EVALUATION",
        "source_sha256": dict(SOURCE_SHA256),
        "selected_model": {
            "name": "Parent_Adam52", "identity": "P_calibrated13 / Adam52 / boundary114",
            "adapter_sha256": "c70b426b58191446545ba6333f7595e96f4fdb4cc689ddbaf5206e276b329db6",
            "model_processor_state_binding": source["FINAL_MODEL_ELIGIBILITY_REVIEW_V1.json"]["opaque_parent_config_processor_and_base_hashes"],
            "selection_scope": "EXPLORED_DEV24_ONLY_NOT_GLOBAL_OR_INDEPENDENT_TEST_WINNER",
            "reason": "Highest observed strict greedy accuracy among the four matched DEV24 endpoints; no demonstrated RL gain.",
        },
        "explored_DEV24": _dev(source["ACCURACY_PILOT_V2_ACTUAL_RESULTS_V1.json"], source["FINAL_BALANCED_SFT_SEED0_ACTUAL_REVIEW_V1.json"]),
        "original_frozen_final100": _original(source["ORIGINAL22_FINAL100_ACTUAL_MATRIX_V1.json"], source["ORIGINAL22_FINAL100_CLASS_FORMAT_COST_SUPPLEMENT_V1.json"], source["ORIGINAL21_TRAINING_ACTUAL_COMPLETION_V1.json"]),
        "independent_holdout": {"status": source["FINAL_MODEL_ELIGIBILITY_REVIEW_V1.json"]["final_independent_evaluation_status"], "patients": 16, "images": 32,
                                "model_outputs_accessed_by_reproduction": False},
        "training_authorized": False,
        "scientific_conclusion": "NO_STABLE_MRI_ACCURACY_BENEFIT_DEMONSTRATED_BY_THESE_VERO_ADAPTATIONS",
    }


def render_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"

def main(argv=None):
    """Installed entry point; the script and package use the same CPU-only path."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Recount authenticated public saved MRI results; no model calls.")
    parser.add_argument("--output", type=Path, help="Write deterministic JSON to this path")
    parser.add_argument("--check", type=Path, help="Fail unless existing JSON equals the reproduced result")
    args = parser.parse_args(argv)
    try:
        result = build_final_metrics(Path(__file__).resolve().parents[2])
        if args.check is not None and json.loads(args.check.read_text()) != result:
            raise ValueError("Existing final metrics differ from authenticated recount")
        encoded = render_json(result)
        if args.output is not None:
            args.output.write_text(encoded)
        print(encoded, end="")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"reproduce_results: {exc}", file=sys.stderr)
        return 1
    return 0


# Saved-response class/format/cost recount retained from the pinned original22
# supplementary analyzer. Full historical authentication/analysis uses its fixed checkout.
from collections import Counter
import statistics
SAVED_LABELS={"mri_sequence":("FLAIR","contrast-enhanced T1"),"mri_visit_correspondence":("different","same")}
def require(condition,message):
    if not condition:raise ValueError(message)

def recount_saved_responses(reference,outputs,answer_and_format):
 require(len(reference)==len({r['id'] for r in reference}) and len(outputs)==len({r['record_id'] for r in outputs}) and {r['id'] for r in reference}=={r['record_id'] for r in outputs},'Exact unique saved-output coverage required');by={r['record_id']:r for r in outputs};result={}
 for task,labels in SAVED_LABELS.items():
  rows=[r for r in reference if r['ability']==task];require(rows,'Both frozen tasks required');classes={l:{'questions':0,'native_correct':0,'full_format_correct':0} for l in labels};conf={l:{p:0 for p in (*labels,'INVALID')} for l in labels};strictconf={l:{p:0 for p in (*labels,'INVALID')} for l in labels};formats=Counter();finish=Counter();invalid=0;lengths=[];times=[];caps=0;truncated=0
  for r in rows:
   o=by[r['id']];truth=r['reward_model']['ground_truth'];require(truth in labels and o['category']==task and isinstance(o['response'],str),'Reference/output task or target differs');answer,fmt=answer_and_format(o['response']);normalized=next((l for l in labels if l.lower()==answer.lower()),None);valid=fmt>0 and normalized is not None;hit=bool(valid and normalized==truth);full=bool(hit and fmt==1)
   classes[truth]['questions']+=1;classes[truth]['native_correct']+=hit;classes[truth]['full_format_correct']+=full;conf[truth][normalized if valid else 'INVALID']+=1;strictconf[truth][normalized if valid and fmt==1 else 'INVALID']+=1;invalid+=not valid;formats[str(fmt)]+=1
   ids=o.get('token_ids');cap=o.get('max_tokens');require(isinstance(ids,list) and all(type(x) is int and x>=0 for x in ids) and type(cap) is int and cap>0 and len(ids)<=cap,'Actual saved token/cap metadata malformed');require('generation_tokens' not in o or o['generation_tokens']==len(ids),'Saved generation count differs');lengths.append(len(ids));finish[o.get('finish_reason','NOT_RECORDED')]+=1;caps+=o.get('hit_max_tokens') is True;truncated+=o.get('finish_reason')=='length' and len(ids)==cap
   if 'elapsed_seconds' in o:
    t=o['elapsed_seconds'];require(type(t) in (int,float) and math.isfinite(t) and t>=0,'Nonfinite/negative saved generation-call time');times.append(t)
  for c in classes.values():
   for kind in ('native','full_format'):c[kind+'_recall']=c[kind+'_correct']/c['questions'] if c['questions'] else None
  n=len(rows);result[task]={'questions':n,'native_correct':sum(c['native_correct'] for c in classes.values()),'full_format_correct':sum(c['full_format_correct'] for c in classes.values()),'native_question_accuracy':sum(c['native_correct'] for c in classes.values())/n,'full_format_question_accuracy':sum(c['full_format_correct'] for c in classes.values())/n,'class_counts':classes,'native_confusion':conf,'full_format_confusion':strictconf,'invalid_answers':invalid,'format_score_counts':dict(formats),'format_half_records':formats['0.5'],'finish_reason_counts':dict(finish),'literal_hit_max_tokens_records':caps,'length_truncated_records':truncated,'tokens':{'records_with_actual_token_ids':n,'sum':sum(lengths),'min':min(lengths),'median':statistics.median(lengths),'max':max(lengths)},'call_time':{'records_with_actual_elapsed':len(times),'records_without_elapsed':n-len(times),'actual_elapsed_seconds_sum':math.fsum(times) if times else None,'actual_elapsed_seconds_mean':statistics.mean(times) if times else None},'aggregation':'Question/class counts only; canonical patient/cycle estimates, seed summaries and2000-bootstrap uncertainty remain in the unchanged main matrix.'}
 return result
