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


def expected_batch_counts(weights: dict[str, float], batch_size: int) -> dict[str, int]:
    """Mirror pinned PerBatchDomainSampler floor/largest-remainder counts."""
    names = list(weights)
    raw = [weights[name] * batch_size for name in names]
    counts = [math.floor(value) for value in raw]
    remainders = [value - count for value, count in zip(raw, counts)]
    for index in sorted(range(len(names)), key=lambda i: -remainders[i])[:batch_size - sum(counts)]:
        counts[index] += 1
    return dict(zip(names, counts))


def _read_categories(file: Path) -> Counter:
    counts: Counter = Counter()
    for line in file.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            category = row["ability"]
            if category not in DOMAINS:
                raise ValueError(f"Unverified category: {category}")
            if row["extra_info"].get("patient_lineage") != "UNVERIFIED":
                raise ValueError("Unexpected patient lineage marker")
            if not Path(row["images"][0]).is_file():
                raise FileNotFoundError("A training image is missing")
            counts[category] += 1
    return counts


def build_launch(
    source: Path, data: Path, model: Path, output: Path,
    arm: str = "uniform", gpus: int = 8, steps: int = 1,
) -> tuple[list[str], dict[str, str], dict]:
    source, data, model, output = (path.resolve() for path in (source, data, model, output))
    if arm not in {"uniform", "proportional"}:
        raise ValueError("Supported arms are uniform and proportional")
    if gpus < 1 or steps < 1:
        raise ValueError("Positive GPU and step counts required")
    training = data / "train.verl.jsonl"
    dev = data / "dev_scan_disjoint_only.verl.jsonl"
    counts = _read_categories(training)
    dev_counts = _read_categories(dev)
    active = tuple(category for category in DOMAINS if category in counts)
    if len(active) < 2 or set(counts) != set(dev_counts):
        raise ValueError("At least two matching categories required in train and engineering dev")
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
        "TRAINER_EXPERIMENT_NAME": f"mri_{arm}_engineering",
        "PYTHONPATH": str(official) + os.pathsep + str(Path(__file__).resolve().parents[1]) +
        (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""),
        "WANDB_MODE": "offline",
    })
    details = {
        "official_source_sha_required": VERO_SHA,
        "arm": arm, "train_category_counts": dict(counts),
        "dev_category_counts": dict(dev_counts), "sampling_weights": weights,
        "official_batch_size": 256,
        "expected_category_draws_per_batch": expected_batch_counts(weights, 256),
        "draws_per_training_scan_by_category": {
            category: expected_batch_counts(weights, 256)[category] / counts[category]
            for category in active
        },
        "proportional_equals_uniform": len(set(counts.values())) == 1,
        "patient_disjoint": False, "scientific_evaluation_eligible": False,
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
    parser.add_argument("--arm", choices=("uniform", "proportional"), default="uniform")
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True).strip()
    if revision != VERO_SHA:
        raise SystemExit(f"Official Vero revision {revision} differs from pinned {VERO_SHA}")
    command, env, details = build_launch(args.source, args.data, args.model, args.output,
                                         args.arm, args.gpus, args.steps)
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
