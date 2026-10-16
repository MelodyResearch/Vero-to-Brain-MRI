"""Launch MRI records through the pinned official Vero veRL GSPO trainer.

This is a CUDA launcher, not an MLX implementation. It refuses to execute on
an unpinned source revision or without CUDA. Dry-run validates the contract.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path


VERO_SHA = "025ca3c800fc95224c86323c1f7fd33dceed3090"
DOMAINS = ("anatomical_comparison", "spatial_localization",
           "tumor_enhancing_presence", "tumor_spatial_grid")
DS_DOMAINS = ("mri_sequence", "mri_tumor_presence",
              "mri_enhancing_presence", "mri_tumor_grid")


def expected_batch_counts(weights: dict[str, float], batch_size: int) -> dict[str, int]:
    """Mirror pinned PerBatchDomainSampler floor/largest-remainder counts."""
    names = list(weights)
    raw = [weights[name] * batch_size for name in names]
    counts = [math.floor(value) for value in raw]
    remainders = [value - count for value, count in zip(raw, counts)]
    for index in sorted(range(len(names)), key=lambda i: -remainders[i])[:batch_size - sum(counts)]:
        counts[index] += 1
    return dict(zip(names, counts))


def _read_categories(file: Path, domains: tuple[str, ...] = DOMAINS,
                     lineage: str = "UNVERIFIED", expected_split: str | None = None) -> Counter:
    counts: Counter = Counter()
    for line in file.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            category = row["ability"]
            if category not in domains:
                raise ValueError(f"Unverified category: {category}")
            if row["extra_info"].get("patient_lineage") != lineage:
                raise ValueError("Unexpected patient lineage marker")
            if expected_split is not None and row["extra_info"].get("internal_split") != expected_split:
                raise ValueError("Unexpected internal patient split")
            if not Path(row["images"][0]).is_file():
                raise FileNotFoundError("A training image is missing")
            counts[category] += 1
    return counts


def _patient_exposure(file: Path, draws: dict[str, int], counts: Counter) -> dict:
    per_patient: dict[str, Counter] = {}
    for line in file.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        category = row["ability"]
        key = row["extra_info"].get("patient_key")
        if key is None:
            continue
        per_patient.setdefault(category, Counter())[key] += 1
    summary = {}
    for category, patients in per_patient.items():
        expected = [draws[category] * row_count / counts[category]
                    for row_count in patients.values()]
        summary[category] = {
            "distinct_patients": len(patients),
            "rows_per_patient_min": min(patients.values()),
            "rows_per_patient_max": max(patients.values()),
            "expected_draws_per_patient_min": min(expected),
            "expected_draws_per_patient_max": max(expected),
        }
    return summary


def _patient_keys(file: Path) -> set[str]:
    return {json.loads(line)["extra_info"]["patient_key"]
            for line in file.read_text().splitlines() if line.strip()}


def build_launch(
    source: Path, data: Path, model: Path, output: Path,
    arm: str = "uniform", gpus: int = 8, steps: int = 1,
    profile: str = "legacy", specialist_category: str | None = None,
) -> tuple[list[str], dict[str, str], dict]:
    source, data, model, output = (path.resolve() for path in (source, data, model, output))
    if arm not in {"uniform", "proportional", "specialist"}:
        raise ValueError("Supported arms are uniform, proportional and specialist")
    if profile not in {"legacy", "ds007045"}:
        raise ValueError("Unknown data profile")
    domains = DS_DOMAINS if profile == "ds007045" else DOMAINS
    lineage = "BIDS_SUBJECT_VERIFIED" if profile == "ds007045" else "UNVERIFIED"
    if arm == "specialist":
        if profile != "ds007045" or specialist_category not in domains:
            raise ValueError("Specialist requires a ds007045 category")
        data = data / "specialist" / specialist_category
    elif specialist_category is not None:
        raise ValueError("Specialist category is only valid for specialist arm")
    if gpus < 1 or steps < 1:
        raise ValueError("Positive GPU and step counts required")
    training = data / "train.verl.jsonl"
    dev = data / ("internal_dev.verl.jsonl" if profile == "ds007045"
                  else "dev_scan_disjoint_only.verl.jsonl")
    counts = _read_categories(training, domains, lineage,
                              "train" if profile == "ds007045" else None)
    dev_counts = _read_categories(dev, domains, lineage,
                                  "internal_dev" if profile == "ds007045" else None)
    active = tuple(category for category in domains if category in counts)
    if (len(active) < (1 if arm == "specialist" else 2) or
            set(counts) != set(dev_counts)):
        raise ValueError("Matching train/internal-dev categories required")
    if profile == "ds007045" and _patient_keys(training) & _patient_keys(dev):
        raise ValueError("Internal training/dev patient overlap")
    if arm == "uniform":
        weights = {category: 1 / len(active) for category in active}
    else:
        total = sum(counts.values())
        weights = {category: counts[category] / total for category in active}
    official = source / "vero-rl"
    reward = Path(__file__).with_name("vero_mri_reward.py").resolve()
    if not (official / "verl/trainer/main_ppo.py").is_file():
        raise FileNotFoundError("Official veRL trainer not found")
    if not model.is_dir():
        raise FileNotFoundError("Model checkpoint directory not found")
    domain_arg = ",".join(f"{category}:{weights[category]:.12g}" for category in active)
    draws = expected_batch_counts(weights, 256)
    args = [
        sys.executable, "-m", "verl.trainer.main_ppo",
        f"--config-path={official / 'examples/model_runs/config'}",
        "--config-name=gspo_qwen3vl",
        f"data.train_files={training}", f"data.val_files={dev}",
        f"data.per_batch_domain_weights={domain_arg}",
        f"actor_rollout_ref.model.path={model}",
        "reward_model.reward_manager=vero_vllm_judge",
        f"custom_reward_function.path={reward}",
        "custom_reward_function.name=compute_score_from_data_source",
        "custom_reward_function.reward_kwargs.format_score=0.2",
        f"trainer.n_gpus_per_node={gpus}",
        f"trainer.total_training_steps={steps}",
        "trainer.logger=[console]", "trainer.val_before_train=false",
        "reward_model.reward_kwargs.llm_judge_kwargs.debug_enabled=false",
        "reward_model.reward_kwargs.llm_judge_kwargs.log_outputs=false",
    ]
    env = dict(os.environ)
    env.update({
        "TRAIN_FILES": str(training), "VAL_FILES": str(dev),
        "REPO_ROOT": str(official), "ROOT_PATH": str(output),
        "TRAINER_EXPERIMENT_NAME": f"mri_{profile}_{arm}_engineering",
        "PYTHONPATH": str(official) + os.pathsep + str(Path(__file__).resolve().parents[1]) +
        (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""),
        "WANDB_MODE": "offline",
    })
    details = {
        "official_source_sha_required": VERO_SHA,
        "arm": arm, "train_category_counts": dict(counts),
        "profile": profile, "specialist_category": specialist_category,
        "dev_category_counts": dict(dev_counts), "sampling_weights": weights,
        "official_batch_size": 256,
        "expected_category_draws_per_batch": draws,
        "draws_per_training_scan_by_category": {
            category: draws[category] / counts[category]
            for category in active
        },
        "expected_patient_exposure_per_batch": _patient_exposure(training, draws, counts),
        "proportional_equals_uniform": len(set(counts.values())) == 1,
        "patient_disjoint_internal_dev": profile == "ds007045",
        "heldout_test_used_for_training_or_validation": False,
        "scientific_evaluation_eligible": False,
        "gpus_requested": gpus, "steps_requested": steps,
        "official_policy_loss": "gspo", "official_rollout_n": 8,
        "weight_updates_executed": False,
    }
    return args, env, details


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("uniform", "proportional", "specialist"), default="uniform")
    parser.add_argument("--profile", choices=("legacy", "ds007045"), default="legacy")
    parser.add_argument("--specialist-category", choices=DS_DOMAINS)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True).strip()
    if revision != VERO_SHA:
        raise SystemExit(f"Official Vero revision {revision} differs from pinned {VERO_SHA}")
    command, env, details = build_launch(args.source, args.data, args.model, args.output,
                                         args.arm, args.gpus, args.steps,
                                         args.profile, args.specialist_category)
    if not args.execute:
        print(json.dumps(details, indent=2))
        return
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() < args.gpus:
        raise SystemExit("Official Vero training requires the requested CUDA GPUs")
    args.output.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command, cwd=args.source / "vero-rl", env=env, check=False)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
