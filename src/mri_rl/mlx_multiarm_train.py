"""Deterministic patient-diverse MLX GSPO engineering arm runner.

Specialist, proportional and uniform arms consume the same internal ds007045
train pool and the exact same starting LoRA file. Each prompt group runs in a
fresh process for bounded Metal memory; all groups, including failed/zero
advantage groups, are recorded. This is not official CUDA Vero parity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

from mri_rl.mlx_gspo_smoke import _sha256
from mri_rl.vero_official_launch import DS_DOMAINS, expected_batch_counts


def build_schedule(records: Path, arm: str, groups: int, seed: int,
                   specialist_category: str | None = None) -> tuple[list[dict], dict]:
    if records.name != "train.verl.jsonl" or not 4 <= groups <= 16:
        raise ValueError("Need internal train.verl.jsonl and 4–16 prompt groups")
    if arm not in {"specialist", "proportional", "uniform"}:
        raise ValueError("Unknown GSPO arm")
    if arm == "specialist" and specialist_category not in DS_DOMAINS:
        raise ValueError("Specialist category required")
    if arm != "specialist" and specialist_category is not None:
        raise ValueError("Specialist category only valid for specialist arm")
    rows = [json.loads(line) for line in records.read_text().splitlines() if line.strip()]
    by_category: dict[str, dict[str, list[dict]]] = {category: {} for category in DS_DOMAINS}
    for row in rows:
        extra = row["extra_info"]
        if (row.get("data_source") != "OpenNeuro_ds007045" or
                extra.get("patient_lineage") != "BIDS_SUBJECT_VERIFIED" or
                extra.get("internal_split") != "train"):
            raise ValueError("Nontraining patient or source in GSPO arm")
        category = row["ability"]
        if category not in DS_DOMAINS or not Path(row["images"][0]).is_file():
            raise ValueError("Invalid category or missing real MRI image")
        by_category[category].setdefault(extra["patient_key"], []).append(row)
    counts = {category: sum(map(len, patients.values()))
              for category, patients in by_category.items()}
    if not all(counts.values()):
        raise ValueError("Need all four training categories")
    if arm == "specialist":
        weights = {specialist_category: 1.0}
    elif arm == "uniform":
        weights = {category: 0.25 for category in DS_DOMAINS}
    else:
        total = sum(counts.values())
        weights = {category: counts[category] / total for category in DS_DOMAINS}
    draws = expected_batch_counts(weights, groups)
    selected: dict[str, list[dict]] = {}
    for category, draw_count in draws.items():
        patients = by_category[category]
        if len(patients) < draw_count:
            raise ValueError("Prompt budget would reuse a patient within category")
        patient_order = sorted(patients, key=lambda key: hashlib.sha256(
            f"{seed}:{category}:{key}".encode()).hexdigest())
        selected[category] = [min(patients[key], key=lambda row: hashlib.sha256(
            f"{seed}:{row['id']}".encode()).hexdigest())
            for key in patient_order[:draw_count]]
    categories = [category for category, count in draws.items() for _ in range(count)]
    random.Random(seed).shuffle(categories)
    cursor = Counter()
    schedule = []
    for category in categories:
        schedule.append(selected[category][cursor[category]])
        cursor[category] += 1
    details = {
        "arm": arm, "specialist_category": specialist_category,
        "groups": groups, "seed": seed,
        "train_pool_category_rows": counts, "category_groups": draws,
        "distinct_patients_by_category": {category: len({
            row["extra_info"]["patient_key"] for row in items})
            for category, items in selected.items()},
        "schedule_sha256": hashlib.sha256("\n".join(row["id"] for row in schedule).encode()).hexdigest(),
        "source_test_used": False,
    }
    return schedule, details


def run(model: Path, records: Path, output: Path, starting_adapter: Path,
        arm: str, groups: int = 8, group_size: int = 2, max_tokens: int = 256,
        seed: int = 7, specialist_category: str | None = None,
        reward_profile: str = "direct_vero") -> dict:
    private_root = Path(__file__).resolve().parents[2] / "local_data"
    if not output.resolve().is_relative_to(private_root):
        raise ValueError("Raw MRI rollouts and weights must stay under ignored local_data/")
    if not starting_adapter.is_file():
        raise FileNotFoundError("A common starting LoRA adapter is required")
    if not 2 <= group_size <= 8 or not 16 <= max_tokens <= 512:
        raise ValueError("Bounded group/response budget exceeded")
    if reward_profile not in {"direct_vero", "scaled_smoke"}:
        raise ValueError("Unknown reward profile")
    schedule, details = build_schedule(records, arm, groups, seed, specialist_category)
    output.mkdir(parents=True, exist_ok=True)
    (output / "schedule_private.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in schedule))
    current_adapter = starting_adapter
    starting_sha = _sha256(starting_adapter)
    results = []
    for index, row in enumerate(schedule):
        step_dir = output / f"step_{index:02d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        step_record = step_dir / "train.verl.jsonl"
        step_record.write_text(json.dumps(row, sort_keys=True) + "\n")
        cmd = [sys.executable, "-m", "mri_rl.mlx_gspo_smoke",
               "--model", str(model), "--records", str(step_record),
               "--output", str(step_dir), "--route", row["extra_info"]["reward_type"],
               "--group-size", str(group_size), "--max-tokens", str(max_tokens),
               "--max-groups", "1", "--seed", str(seed + index),
               "--reward-profile", reward_profile,
               "--adapter-checkpoint", str(current_adapter)]
        completed = subprocess.run(cmd, env=os.environ.copy(), text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (step_dir / "process_log_PRIVATE.txt").write_text(completed.stdout)
        if completed.returncode:
            raise RuntimeError(f"GSPO step {index} failed; see private process log")
        receipt = json.loads((step_dir / "receipt.json").read_text())
        attempt = receipt["attempts"][0]
        task_values = [item["score"] for item in attempt["rewards"]]
        status = receipt["status"]
        results.append({
            "step": index, "category": row["ability"], "status": status,
            "patient_key": row["extra_info"]["patient_key"],
            "reward_totals": [item["total"] for item in attempt["rewards"]],
            "response_tokens": attempt["response_tokens"],
            "task_scores": task_values,
            "task_reward_variance": max(task_values) > min(task_values),
            "format_scores": [item["format"] for item in attempt["rewards"]],
            "accuracy_scores": [item["accuracy"] for item in attempt["rewards"]],
            "optimizer_steps": receipt.get("optimizer_steps", 0),
            "before_adapter_sha256": receipt.get("before_adapter_sha256"),
            "after_adapter_sha256": receipt.get("after_adapter_sha256"),
        })
        if receipt["policy_updated"]:
            current_adapter = step_dir / "after_adapters.safetensors"
        aggregate = {**details, "starting_adapter_sha256": starting_sha,
                     "rollouts_per_group": group_size, "max_response_tokens": max_tokens,
                     "reward_profile": reward_profile,
                     "completed_prompt_groups": len(results), "group_results": results,
                     "optimizer_steps": sum(r["optimizer_steps"] for r in results),
                     "actual_rollouts": sum(len(r["response_tokens"]) for r in results),
                     "actual_response_tokens": sum(sum(r["response_tokens"]) for r in results),
                     "task_reward_varying_groups": sum(r["task_reward_variance"] for r in results),
                     "fixed_attempted_groups_not_fixed_active_updates": True,
                     "final_adapter_sha256": _sha256(current_adapter),
                     "heldout_test_used": False, "scientific_claim": False}
        (output / "arm_receipt_PRIVATE.json").write_text(json.dumps(aggregate, indent=2))
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--starting-adapter", type=Path, required=True)
    parser.add_argument("--arm", choices=("specialist", "proportional", "uniform"), required=True)
    parser.add_argument("--specialist-category", choices=DS_DOMAINS)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--reward-profile", choices=("direct_vero", "scaled_smoke"),
                        default="direct_vero")
    args = parser.parse_args()
    receipt = run(args.model.resolve(), args.records.resolve(), args.output.resolve(),
                  args.starting_adapter.resolve(), args.arm, args.groups,
                  args.group_size, args.max_tokens, args.seed,
                  args.specialist_category, args.reward_profile)
    print(json.dumps({key: value for key, value in receipt.items()
                      if key != "group_results"}, indent=2))


if __name__ == "__main__":
    main()
