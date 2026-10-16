"""Score paired MRI sequence predictions after a same-patient/slice image swap."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from mri_rl.vero_mri_reward import _answer_and_format


def read_rows(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    keyed = {row["record_id"]: row for row in rows}
    if len(keyed) != len(rows):
        raise ValueError("Duplicate record IDs")
    return keyed


def analyze(original: Path, swapped: Path, gold: Path) -> dict:
    normal = read_rows(original)
    control = read_rows(swapped)
    targets = read_rows(gold)
    suffix = "__same_slice_sequence_swap"
    if set(control) != {key + suffix for key in normal if normal[key]["category"] == "mri_sequence"}:
        raise ValueError("Swap output IDs do not match original sequence IDs")
    model_fields = ("model_path", "model_config_sha256", "model_template_sha256",
                    "adapter_checkpoint_sha256", "adapter_config_sha256",
                    "max_tokens", "temperature")
    totals = {"pairs": 0, "original_correct": 0, "swap_correct": 0,
              "prediction_changed": 0, "correct_directional_change": 0,
              "both_correct": 0, "both_format_valid": 0}
    for key, before in normal.items():
        if before["category"] != "mri_sequence":
            continue
        after = control[key + suffix]
        if any(before.get(field) != after.get(field) for field in model_fields):
            raise ValueError("Model/decoding mismatch")
        if before["prompt"][0] != after["prompt"][0] or before["prompt"][1]["content"][-1] != after["prompt"][1]["content"][-1]:
            raise ValueError("System or question changed")
        if before["image_sha256"] == after["image_sha256"]:
            raise ValueError("Image did not change")
        target = targets[key]["ground_truth"]
        opposite = "FLAIR" if target == "contrast-enhanced T1" else "contrast-enhanced T1"
        p0, f0 = _answer_and_format(before["response"])
        p1, f1 = _answer_and_format(after["response"])
        c0 = p0.lower() == target.lower()
        c1 = p1.lower() == opposite.lower()
        totals["pairs"] += 1
        totals["original_correct"] += c0
        totals["swap_correct"] += c1
        totals["prediction_changed"] += p0.lower() != p1.lower()
        totals["correct_directional_change"] += c0 and c1 and p0.lower() != p1.lower()
        totals["both_correct"] += c0 and c1
        totals["both_format_valid"] += f0 == 1 and f1 == 1
    return {**totals, "original_output_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
            "swap_output_sha256": hashlib.sha256(swapped.read_bytes()).hexdigest(),
            "gold_sha256": hashlib.sha256(gold.read_bytes()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--swapped", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.original, args.swapped, args.gold), indent=2))
