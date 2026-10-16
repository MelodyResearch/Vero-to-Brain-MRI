"""Private, patient-disjoint ds007045 internal-dev manifest and adapter scorer.

The image-only manifest is consumable by frozen_eval. Ground truth stays in a
separate ignored local_data file and is joined only during CPU scoring.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from mri_rl.frozen_eval import sha256
from mri_rl.mlx_sft_warmstart import SFT_SYSTEM_PROMPT
from mri_rl.vero_mri_reward import compute_score_from_data_source


ROUTES = {"mri_sequence", "mri_tumor_presence", "mri_enhancing_presence",
          "mri_tumor_grid"}


def _read_records(path: Path, expected_name: str, split: str) -> list[dict]:
    if path.name != expected_name:
        raise ValueError(f"Expected {expected_name}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"Empty {split} records")
    ids = set()
    for row in rows:
        extra = row.get("extra_info", {})
        if (row.get("data_source") != "OpenNeuro_ds007045" or
                extra.get("patient_lineage") != "BIDS_SUBJECT_VERIFIED" or
                extra.get("internal_split") != split):
            raise ValueError(f"Expected verified {split} records; test patients are excluded")
        if (row.get("ability") not in ROUTES or
                extra.get("reward_type") != row["ability"] or
                not extra.get("patient_key") or not row.get("id") or
                row["id"] in ids):
            raise ValueError(f"Malformed or duplicate {split} record")
        ids.add(row["id"])
        images = row.get("images", [])
        if len(images) != 1 or not Path(images[0]).is_file() or not row.get("prompt"):
            raise ValueError(f"Missing real MRI image or prompt in {split}")
        if "ground_truth" not in row.get("reward_model", {}):
            raise ValueError(f"Missing {split} reward ground truth")
    return rows


def build_manifest(train_path: Path, dev_path: Path, output_dir: Path,
                   expected_rows: int = 54, expected_patients: int = 4,
                   private_root: Path | None = None,
                   prompt_profile: str = "original") -> dict:
    if prompt_profile not in {"original", "sft_train_matched"}:
        raise ValueError("Unknown internal-dev prompt profile")
    private_root = private_root or Path(__file__).resolve().parents[2] / "local_data"
    if not output_dir.resolve().is_relative_to(private_root.resolve()):
        raise ValueError("Internal-dev prompts and gold must stay under ignored local_data")
    train = _read_records(train_path, "train.verl.jsonl", "train")
    dev = _read_records(dev_path, "internal_dev.verl.jsonl", "internal_dev")
    train_patients = {r["extra_info"]["patient_key"] for r in train}
    dev_patients = {r["extra_info"]["patient_key"] for r in dev}
    if train_patients & dev_patients:
        raise ValueError("Train/internal-dev patient overlap")
    if len(dev) != expected_rows or len(dev_patients) != expected_patients:
        raise ValueError("Unexpected internal dev row or patient count")
    if {r["ability"] for r in dev} != ROUTES:
        raise ValueError("Internal dev must cover all four MRI categories")
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if prompt_profile == "original" else "_sft_train_matched"
    manifest = output_dir / f"internal_dev_manifest{suffix}.jsonl"
    gold = output_dir / f"internal_dev_gold{suffix}_PRIVATE.jsonl"
    with manifest.open("w") as manifest_out, gold.open("w") as gold_out:
        for row in dev:
            prompt = (row["prompt"] if prompt_profile == "original" else
                      [{"role": "system", "content": SFT_SYSTEM_PROMPT}, *row["prompt"]])
            manifest_out.write(json.dumps({
                "record_id": row["id"], "category": row["ability"],
                "image": row["images"][0], "prompt": prompt,
            }, sort_keys=True) + "\n")
            gold_out.write(json.dumps({
                "record_id": row["id"], "category": row["ability"],
                "patient_key": row["extra_info"]["patient_key"],
                "internal_split": "internal_dev",
                "ground_truth": row["reward_model"]["ground_truth"],
                "extra_info": {"reward_type": row["extra_info"]["reward_type"]},
            }, sort_keys=True) + "\n")
    receipt = {
        "source": "OpenNeuro_ds007045", "split": "internal_dev",
        "prompt_profile": prompt_profile,
        "rows": len(dev), "dev_patients": len(dev_patients),
        "train_dev_disjoint": True,
        "category_rows": dict(Counter(r["ability"] for r in dev)),
        "manifest_sha256": sha256(manifest), "gold_sha256": sha256(gold),
        "heldout_test_used": False,
    }
    (output_dir / f"build_receipt{suffix}_PRIVATE.json").write_text(
        json.dumps(receipt, indent=2))
    return receipt


def _keyed_jsonl(path: Path) -> dict[str, dict]:
    result = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            key = row.get("record_id")
            if not key or key in result:
                raise ValueError("Missing or duplicate record ID")
            result[key] = row
    if not result:
        raise ValueError("Empty scoring input")
    return result


def score_outputs(manifest_path: Path, gold_path: Path, outputs_path: Path) -> dict:
    manifest = _keyed_jsonl(manifest_path)
    gold = _keyed_jsonl(gold_path)
    outputs = _keyed_jsonl(outputs_path)
    if any(row.get("internal_split") != "internal_dev" or not row.get("patient_key")
           for row in gold.values()):
        raise ValueError("Expected internal-dev gold and patient lineage")
    if set(manifest) != set(gold) or set(manifest) != set(outputs):
        raise ValueError("Internal-dev manifest, gold and predictions must match exactly")
    manifest_hash = sha256(manifest_path)
    adapter_shas = set()
    config_shas = set()
    model_settings = set()
    decoding_settings = set()
    buckets = defaultdict(lambda: {"rows": 0, "format_valid": 0,
                                   "accuracy_correct": 0, "reward_sum": 0.0})
    patient_scores = defaultdict(list)
    for key, row in outputs.items():
        expected = manifest[key]
        target = gold[key]
        if row.get("category") != expected["category"] or target["category"] != expected["category"]:
            raise ValueError("Category mismatch in internal-dev outputs")
        if (row.get("manifest_sha256") != manifest_hash or
                row.get("template_override_sha256") is not None or
                not isinstance(row.get("adapter_verified_parameter_tensors"), int) or
                row["adapter_verified_parameter_tensors"] <= 0):
            raise ValueError("Output lacks verified native-template adapter receipt")
        if (row.get("prompt") != expected["prompt"] or
                row.get("image_sha256") != sha256(Path(expected["image"])) or
                row.get("image_placeholder_count") != 1):
            raise ValueError("Internal-dev prompt or image receipt mismatch")
        if (not isinstance(row.get("model_path"), str) or not row["model_path"] or
                not isinstance(row.get("model_config_sha256"), str) or
                not row["model_config_sha256"] or
                not isinstance(row.get("model_template_sha256"), str) or
                not row["model_template_sha256"]):
            raise ValueError("Missing internal-dev model configuration receipt")
        if (type(row.get("max_tokens")) is not int or row["max_tokens"] <= 0 or
                row.get("temperature") != 0):
            raise ValueError("Missing or nondeterministic internal-dev decoding receipt")
        adapter_shas.add(row.get("adapter_checkpoint_sha256"))
        config_shas.add(row.get("adapter_config_sha256"))
        model_settings.add((row["model_path"], row["model_config_sha256"],
                            row["model_template_sha256"]))
        decoding_settings.add((row["max_tokens"], row["temperature"]))
        grade = compute_score_from_data_source(
            "OpenNeuro_ds007045", row["response"], target["ground_truth"],
            target["extra_info"], format_score=0.2)
        bucket = buckets[row["category"]]
        bucket["rows"] += 1
        bucket["format_valid"] += grade["format"] == 1
        bucket["accuracy_correct"] += grade["accuracy"] > 0
        bucket["reward_sum"] += grade["score"]
        patient_scores[target["patient_key"]].append(grade["accuracy"])
    if (len(adapter_shas) != 1 or None in adapter_shas or len(config_shas) != 1 or
            None in config_shas or len(model_settings) != 1 or
            len(decoding_settings) != 1):
        raise ValueError("Predictions do not share one adapter, model, and decoding configuration")
    by_category = {
        category: {"rows": bucket["rows"], "format_valid": bucket["format_valid"],
                   "accuracy_correct": bucket["accuracy_correct"],
                   "mean_reward": bucket["reward_sum"] / bucket["rows"]}
        for category, bucket in buckets.items()
    }
    return {
        "split": "internal_dev", "rows": len(outputs),
        "patients": len(patient_scores), "format_valid": sum(
            bucket["format_valid"] for bucket in by_category.values()),
        "accuracy_correct": sum(bucket["accuracy_correct"] for bucket in by_category.values()),
        "patient_mean_accuracy": sum(sum(scores) / len(scores) for scores in patient_scores.values())
                                 / len(patient_scores),
        "by_category": by_category, "manifest_sha256": manifest_hash,
        "adapter_checkpoint_sha256": next(iter(adapter_shas)),
        "native_template": True, "heldout_test_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    build = sub.add_parser("build")
    build.add_argument("--train", type=Path, required=True)
    build.add_argument("--dev", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--prompt-profile", choices=("original", "sft_train_matched"),
                       default="original")
    score = sub.add_parser("score")
    score.add_argument("--manifest", type=Path, required=True)
    score.add_argument("--gold", type=Path, required=True)
    score.add_argument("--outputs", type=Path, required=True)
    args = parser.parse_args()
    result = (build_manifest(args.train, args.dev, args.output_dir,
                             prompt_profile=args.prompt_profile)
              if args.action == "build" else
              score_outputs(args.manifest, args.gold, args.outputs))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
