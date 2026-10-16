"""Read-only paired ds007045 score audit; print aggregate statistics only.

Run from the repository root after both private output files are complete::

    PYTHONPATH=src python3 scripts/ds007045_audit_stats.py \
        --repo . \
        --base local_data/frozen_eval_v1/ds007045_base_256.jsonl \
        --vero local_data/frozen_eval_v1/ds007045_vero_shared_256.jsonl

Add ``--sft local_data/frozen_eval_v1/ds007045_sft4_256.jsonl`` and
``--sft-before <complete matched 107-row before-adapter output>`` only after
both adapter predictions have completed. The optional extension verifies
both adapter checksums and training-patient exposure before scoring SFT.

The script requires ignored local manifests, gold records, images and raw model
outputs. It does not write any patient-level data or change the locked scorer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from mri_rl.score_frozen_eval import score


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_verified(path: Path, manifest: Path, source: dict) -> dict:
    predictions = rows(path)
    if len(predictions) != len(source):
        raise ValueError(f"incomplete prediction file: {path.name}")
    out = {}
    for prediction in predictions:
        key = prediction["record_id"]
        if key in out or key not in source:
            raise ValueError("unknown or duplicate record")
        target = source[key]
        if prediction["category"] != target["category"]:
            raise ValueError("category mismatch")
        if prediction["prompt"] != target["prompt"]:
            raise ValueError("prompt mismatch")
        if prediction["manifest_sha256"] != sha(manifest):
            raise ValueError("manifest hash mismatch")
        if prediction["image_sha256"] != sha(Path(target["image"])):
            raise ValueError("image hash mismatch")
        if prediction["image_placeholder_count"] != 1:
            raise ValueError("missing image token")
        if prediction["max_tokens"] != 256 or prediction["temperature"] != 0:
            raise ValueError("decoding mismatch")
        out[key] = prediction
    return out


def main() -> None:
    cli = argparse.ArgumentParser()
    cli.add_argument("--repo", type=Path, required=True)
    cli.add_argument("--base", type=Path, required=True)
    cli.add_argument("--vero", type=Path, required=True)
    cli.add_argument("--sft", type=Path, help="Optional complete 107-row base-initialized SFT adapter outputs")
    cli.add_argument("--sft-before", type=Path, help="Required matched 4-bit LoRA before-update outputs when --sft is set")
    args = cli.parse_args()
    private = args.repo / "local_data"
    manifest_path = private / "frozen_eval_v1/ds007045_heldout_manifest.jsonl"
    manifest = rows(manifest_path)
    source = {r["record_id"]: r for r in manifest}
    if len(source) != len(manifest) or len(source) != 107:
        raise ValueError("unexpected manifest records")
    private_gold = rows(private / "ds007045_bounded/visual_prompts/gold_mask_derived_PRIVATE.jsonl")
    gold = {r["prompt_id"]: r for r in private_gold}
    if any(gold[key]["split"] != "test" or gold[key]["answer"] != row["gold"] for key, row in source.items()):
        raise ValueError("private heldout gold mismatch")
    base = load_verified(args.base, manifest_path, source)
    vero = load_verified(args.vero, manifest_path, source)
    if len({r["model_config_sha256"] for r in base.values()}) != 1 or len({r["model_config_sha256"] for r in vero.values()}) != 1:
        raise ValueError("mixed model configuration")
    if len({r["model_template_sha256"] for r in base.values()}) != 1:
        raise ValueError("mixed base chat template")
    base_template = next(iter(base.values()))["model_template_sha256"]
    if any(r["template_override_sha256"] != base_template for r in vero.values()):
        raise ValueError("Vero did not use shared base chat template")
    if any(r["template_override_sha256"] is not None for r in base.values()):
        raise ValueError("base template override unexpectedly present")
    categories = ["sequence_flair_vs_t1ce", "tumor_presence", "tumor_enhancing_presence", "tumor_grid"]
    data = {}
    patient_macro = defaultdict(list)
    for category in categories:
        keys = [k for k, r in source.items() if r["category"] == category]
        per_patient = defaultdict(list)
        discordant = Counter()
        for key in keys:
            p = gold[key]["private_subject"]
            b_correct, b_valid, _ = score(category, source[key]["gold"], base[key]["response"])
            v_correct, v_valid, _ = score(category, source[key]["gold"], vero[key]["response"])
            per_patient[p].append((b_correct, b_valid, v_correct, v_valid))
            discordant["base_only" if b_correct and not v_correct else "vero_only" if v_correct and not b_correct else "both_correct" if b_correct else "both_wrong"] += 1
        if len(per_patient) != 8:
            raise ValueError("category does not cover all eight patients")
        by_patient = {
            p: {
                "base": mean(x[0] for x in values),
                "vero": mean(x[2] for x in values),
                "base_valid": mean(x[1] for x in values),
                "vero_valid": mean(x[3] for x in values),
            }
            for p, values in per_patient.items()
        }
        for p, stat in by_patient.items():
            patient_macro[p].append((stat["base"], stat["vero"]))
        train = [r for r in private_gold if r["split"] == "train" and ("sequence_flair_vs_t1ce" if r["category"] == "sequence" else "tumor_enhancing_presence" if r["category"] == "enhancing_presence" else r["category"]) == category]
        counts = Counter(str(r["answer"]) for r in train)
        if category == "sequence_flair_vs_t1ce":
            majority = "FLAIR"  # The two training classes tie; locked tie break.
        else:
            majority = counts.most_common(1)[0][0]
        baseline_by_patient = defaultdict(list)
        for key in keys:
            baseline_by_patient[gold[key]["private_subject"]].append(str(source[key]["gold"]) == majority)
        data[category] = {
            "n": len(keys),
            "n_patients": len(per_patient),
            "base_correct": sum(sum(x[0] for x in values) for values in per_patient.values()),
            "vero_correct": sum(sum(x[2] for x in values) for values in per_patient.values()),
            "base_valid": sum(sum(x[1] for x in values) for values in per_patient.values()),
            "vero_valid": sum(sum(x[3] for x in values) for values in per_patient.values()),
            "base_patient_mean": round(mean(s["base"] for s in by_patient.values()), 6),
            "vero_patient_mean": round(mean(s["vero"] for s in by_patient.values()), 6),
            "paired_patient_difference": round(mean(s["vero"] - s["base"] for s in by_patient.values()), 6),
            "paired_patient_range": [round(min(s["vero"] - s["base"] for s in by_patient.values()), 6), round(max(s["vero"] - s["base"] for s in by_patient.values()), 6)],
            "base_patient_valid_mean": round(mean(s["base_valid"] for s in by_patient.values()), 6),
            "vero_patient_valid_mean": round(mean(s["vero_valid"] for s in by_patient.values()), 6),
            "train_majority_answer": majority,
            "baseline_raw_correct": sum(sum(v) for v in baseline_by_patient.values()),
            "baseline_patient_mean": round(mean(mean(v) for v in baseline_by_patient.values()), 6),
            "discordance": dict(discordant),
        }
    if len(patient_macro) != 8 or any(len(x) != 4 for x in patient_macro.values()):
        raise ValueError("incomplete patient macro")
    overall = {
        "n_prompts": len(source),
        "n_patients": 8,
        "base_patient_category_macro": round(mean(mean(x[0] for x in values) for values in patient_macro.values()), 6),
        "vero_patient_category_macro": round(mean(mean(x[1] for x in values) for values in patient_macro.values()), 6),
        "paired_macro_difference": round(mean(mean(x[1] - x[0] for x in values) for values in patient_macro.values()), 6),
        "base_raw_correct": sum(s["base_correct"] for s in data.values()),
        "vero_raw_correct": sum(s["vero_correct"] for s in data.values()),
        "base_raw_valid": sum(s["base_valid"] for s in data.values()),
        "vero_raw_valid": sum(s["vero_valid"] for s in data.values()),
    }
    result = {"overall": overall, "categories": data}
    if args.sft is not None:
        if args.sft_before is None:
            raise ValueError("SFT effect requires matched 4-bit before-adapter inference")
        adapter_dir = private / "ds007045_bounded/sft_warmstart_4step_v1"
        training_receipt = json.loads((adapter_dir / "receipt.json").read_text())
        checkpoint_sha = sha(adapter_dir / "after_adapters.safetensors")
        before_sha = sha(adapter_dir / "before_adapters.safetensors")
        config_sha = sha(adapter_dir / "adapter_config.json")
        if checkpoint_sha != training_receipt["after_adapter_sha256"]:
            raise ValueError("SFT adapter checkpoint/receipt hash mismatch")
        if before_sha != training_receipt["before_adapter_sha256"]:
            raise ValueError("SFT before-adapter/receipt hash mismatch")
        if checkpoint_sha == before_sha:
            raise ValueError("SFT adapter did not change")
        if (training_receipt["optimizer_steps"] != 4
                or training_receipt["heldout_test_used"]
                or training_receipt["model_path_name"] != "qwen3vl8b-base"
                or training_receipt["status"] != "SFT_ENGINEERING_CONTROL"):
            raise ValueError("unexpected SFT training receipt")
        exposed = {row["patient_key"] for row in training_receipt["examples"]}
        case_crosswalk = json.loads((private / "ds007045_bounded/visual_prompts/private_manifest.json").read_text())["private_cases"]
        train_patients = {row["anonymous_token"] for row in case_crosswalk if row["split"] == "train"}
        test_patients = {row["anonymous_token"] for row in case_crosswalk if row["split"] == "test"}
        if exposed - train_patients or exposed & test_patients:
            raise ValueError("SFT exposure includes nontraining patient")
        if len(exposed) != training_receipt["distinct_patients_exposed"]:
            raise ValueError("SFT patient count mismatch")
        sft = load_verified(args.sft, manifest_path, source)
        sft_before = load_verified(args.sft_before, manifest_path, source)
        base_config = next(iter(base.values()))["model_config_sha256"]
        if any(
            row["model_config_sha256"] != base_config
            or row["model_template_sha256"] != base_template
            or row["template_override_sha256"] is not None
            or row.get("adapter_checkpoint_sha256") != checkpoint_sha
            or row.get("adapter_config_sha256") != config_sha
            or not isinstance(row.get("adapter_verified_parameter_tensors"), int)
            or row["adapter_verified_parameter_tensors"] <= 0
            for row in sft.values()
        ):
            raise ValueError("SFT inference adapter/model/template receipt mismatch")
        if any(
            row["model_config_sha256"] != base_config
            or row["model_template_sha256"] != base_template
            or row["template_override_sha256"] is not None
            or row.get("adapter_checkpoint_sha256") != before_sha
            or row.get("adapter_config_sha256") != config_sha
            or not isinstance(row.get("adapter_verified_parameter_tensors"), int)
            or row["adapter_verified_parameter_tensors"] <= 0
            for row in sft_before.values()
        ):
            raise ValueError("SFT before-update adapter/model/template receipt mismatch")
        sft_stats = {}
        sft_patient_macro = defaultdict(list)
        before_patient_macro = defaultdict(list)
        for category in categories:
            keys = [key for key, row in source.items() if row["category"] == category]
            per_patient = defaultdict(list)
            before_per_patient = defaultdict(list)
            base_per_patient = defaultdict(list)
            vero_per_patient = defaultdict(list)
            discordant_base = Counter()
            discordant_vero = Counter()
            discordant_before = Counter()
            for key in keys:
                patient = gold[key]["private_subject"]
                target = source[key]
                bc = score(category, target["gold"], base[key]["response"])[0]
                vc = score(category, target["gold"], vero[key]["response"])[0]
                sc, sv, _ = score(category, target["gold"], sft[key]["response"])
                pc, pv, _ = score(category, target["gold"], sft_before[key]["response"])
                per_patient[patient].append((sc, sv))
                before_per_patient[patient].append((pc, pv))
                base_per_patient[patient].append(bc)
                vero_per_patient[patient].append(vc)
                discordant_base["sft_only" if sc and not bc else "base_only" if bc and not sc else "same"] += 1
                discordant_vero["sft_only" if sc and not vc else "vero_only" if vc and not sc else "same"] += 1
                discordant_before["after_only" if sc and not pc else "before_only" if pc and not sc else "same"] += 1
            if len(per_patient) != 8:
                raise ValueError("SFT category does not cover all eight patients")
            patient_scores = {patient: mean(value[0] for value in items) for patient, items in per_patient.items()}
            before_scores = {patient: mean(value[0] for value in items) for patient, items in before_per_patient.items()}
            for patient, value in patient_scores.items():
                sft_patient_macro[patient].append(value)
                before_patient_macro[patient].append(before_scores[patient])
            sft_stats[category] = {
                "n": len(keys),
                "n_patients": len(per_patient),
                "sft_correct": sum(sum(value[0] for value in items) for items in per_patient.values()),
                "sft_valid": sum(sum(value[1] for value in items) for items in per_patient.values()),
                "sft_patient_mean": round(mean(patient_scores.values()), 6),
                "before_correct": sum(sum(value[0] for value in items) for items in before_per_patient.values()),
                "before_valid": sum(sum(value[1] for value in items) for items in before_per_patient.values()),
                "before_patient_mean": round(mean(before_scores.values()), 6),
                "paired_sft_minus_before": round(mean(patient_scores[p] - before_scores[p] for p in per_patient), 6),
                "paired_sft_minus_base": round(mean(patient_scores[p] - mean(base_per_patient[p]) for p in per_patient), 6),
                "paired_sft_minus_vero": round(mean(patient_scores[p] - mean(vero_per_patient[p]) for p in per_patient), 6),
                "discordance_vs_before": dict(discordant_before),
                "discordance_vs_base": dict(discordant_base),
                "discordance_vs_vero": dict(discordant_vero),
            }
        if len(sft_patient_macro) != 8 or any(len(items) != 4 for items in sft_patient_macro.values()) or any(len(items) != 4 for items in before_patient_macro.values()):
            raise ValueError("incomplete SFT patient macro")
        sft_macro = mean(mean(items) for items in sft_patient_macro.values())
        before_macro = mean(mean(items) for items in before_patient_macro.values())
        result["sft"] = {
            "status": "four-step rank-four 4-bit LoRA SFT engineering control; matched-precision before/after adapter comparison",
            "optimizer_steps": 4,
            "distinct_training_patients_exposed": len(exposed),
            "adapter_checkpoint_sha256": checkpoint_sha,
            "before_adapter_sha256": before_sha,
            "adapter_config_sha256": config_sha,
            "patient_category_macro": round(sft_macro, 6),
            "before_patient_category_macro": round(before_macro, 6),
            "paired_macro_minus_before": round(sft_macro - before_macro, 6),
            "unmatched_precision_macro_minus_base": round(sft_macro - overall["base_patient_category_macro"], 6),
            "unmatched_precision_macro_minus_vero": round(sft_macro - overall["vero_patient_category_macro"], 6),
            "raw_correct": sum(value["sft_correct"] for value in sft_stats.values()),
            "raw_valid": sum(value["sft_valid"] for value in sft_stats.values()),
            "before_raw_correct": sum(value["before_correct"] for value in sft_stats.values()),
            "before_raw_valid": sum(value["before_valid"] for value in sft_stats.values()),
            "categories": sft_stats,
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
