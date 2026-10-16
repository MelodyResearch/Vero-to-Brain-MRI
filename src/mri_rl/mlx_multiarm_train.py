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
import math
import os
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

from mri_rl.mlx_gspo_smoke import _sha256, load_system_prompt_file, atomic_json, task_group_is_informative
from mri_rl.experiment_contract import require_launch_preflight, load_contract, canonical_hash
from mri_rl.vero_official_launch import DS_DOMAINS, expected_batch_counts
from mri_rl.vero_mri_reward import compute_score_from_data_source


DEFAULT_RL_SYSTEM_PROMPT = (
    "Respond using <think>brief visual reasoning</think><answer>\\boxed{final answer}</answer>.")


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


def system_prompt_cli_args(path: Path | None) -> list[str]:
    return ["--system-prompt-file", str(path)] if path is not None else []


def _verified_child(step_dir: Path, row: dict, starting_sha: str, *,
                    contract_id: str, review_sha: str, seed: int, records: Path,
                    expected_system: str, group_size: int, max_tokens: int) -> dict:
    """Check actual child artifacts before recording a resumable boundary."""
    receipt_path = step_dir / "receipt.json"
    raw_path = step_dir / "raw_rollouts.json"
    receipt = json.loads(receipt_path.read_text())
    raw = json.loads(raw_path.read_text())
    if (receipt.get("contract_id") != contract_id or receipt.get("review_sha256") != review_sha
            or receipt.get("starting_adapter_sha256") != starting_sha
            or receipt.get("seed") != seed or receipt.get("reward_profile") != "direct_vero"
            or receipt.get("temperature") != 1.0 or receipt.get("max_tokens") != max_tokens
            or receipt.get("group_size") != group_size or receipt.get("groups_attempted") != 1
            or receipt.get("training_records_sha256") != _sha256(records)
            or receipt.get("science_claim") is not False):
        raise ValueError("Child receipt does not match authorized schedule/configuration")
    attempts = receipt.get("attempts", [])
    if len(raw) != 1 or len(attempts) != 1 or raw[0].get("record_id") != row["id"]:
        raise ValueError("Child raw rollout does not match scheduled training record")
    if raw[0].get("image_sha256") != _sha256(Path(row["images"][0])):
        raise ValueError("Child MRI bytes differ from scheduled image")
    rendered, prompt_ids = raw[0].get("rendered_prompt"), raw[0].get("prompt_token_ids")
    attempt = attempts[0]
    if (not isinstance(rendered, str) or expected_system not in rendered
            or not row.get("extra_info", {}).get("question")
            or row["extra_info"]["question"] not in rendered
            or rendered.count("<|image_pad|>") != 1
            or not rendered.endswith("<|im_start|>assistant\n")
            or raw[0].get("group_index") != 0 or attempt.get("group_index") != 0
            or attempt.get("image_sha256") != raw[0]["image_sha256"]
            or attempt.get("route") != row["extra_info"]["reward_type"]
            or attempt.get("source") != row["data_source"]
            or not isinstance(prompt_ids, list) or len(prompt_ids) != 1
            or not isinstance(prompt_ids[0], list) or not prompt_ids[0]
            or any(type(token) is not int or token < 0 for token in prompt_ids[0])
            or attempt.get("prompt_tokens") != len(prompt_ids[0])
            or attempt.get("rendered_prompt_sha256") != hashlib.sha256(rendered.encode()).hexdigest()
            or attempt.get("prompt_token_ids_sha256") != hashlib.sha256(json.dumps(prompt_ids).encode()).hexdigest()):
        raise ValueError("Child raw prompt/image/token inventory linkage invalid")
    rollouts, saved = raw[0].get("rollouts", []), attempt.get("rewards", [])
    if len(rollouts) != group_size or len(saved) != group_size:
        raise ValueError("Child rollout count differs from authorized budget")
    grades = []
    for generated, stored in zip(rollouts, saved):
        tokens = generated.get("token_ids")
        if (not isinstance(tokens, list) or not 1 <= len(tokens) <= max_tokens
                or any(type(token) is not int or token < 0 for token in tokens)
                or stored.get("overlong", 0) != 0):
            raise ValueError("Child generated-token budget or direct-Vero length reward invalid")
        grade = compute_score_from_data_source(row["data_source"], generated["text"],
                    row["reward_model"]["ground_truth"], row["extra_info"])
        if any(not math.isclose(grade[key], stored[key], abs_tol=1e-9, rel_tol=0)
               for key in ("score", "accuracy", "format")):
            raise ValueError("Child saved reward differs from strict raw-output recomputation")
        grades.append({**grade, "overlong": stored.get("overlong", 0)})
    if attempt.get("response_tokens") != [len(item["token_ids"]) for item in rollouts]:
        raise ValueError("Child response token inventory mismatch")
    updated = receipt.get("policy_updated") is True
    informative = task_group_is_informative(grades)
    files = {"receipt.json": _sha256(receipt_path), "raw_rollouts.json": _sha256(raw_path),
             "train.verl.jsonl": _sha256(records)}
    if updated:
        parity = receipt.get("old_current_logp_max_abs_errors")
        if (not isinstance(parity, list) or len(parity) != group_size
                or any(type(value) not in (int, float) or not math.isfinite(value)
                       or value < 0 or value > 1e-5 for value in parity)):
            raise ValueError("Child old/current log-probability parity unverified")
        if (receipt.get("status") != "TRAINING_COMPLETED_ENGINEERING_SMOKE"
                or receipt.get("optimizer_steps") != 1 or not informative
                or not math.isfinite(receipt.get("gradient_l2", float("nan")))
                or receipt["gradient_l2"] <= 0 or receipt.get("checkpoint_reloaded") is not True):
            raise ValueError("Child update lacks task variance, finite gradient or completed reload")
        before, after = step_dir / "before_adapters.safetensors", step_dir / "after_adapters.safetensors"
        before_sha, after_sha = _sha256(before), _sha256(after)
        if (before_sha != starting_sha or before_sha == after_sha or receipt.get("before_adapter_sha256") != before_sha
                or receipt.get("after_adapter_sha256") != after_sha):
            raise ValueError("Child saved adapter change is invalid")
        reload_path = step_dir / "reload_inference_receipt.json"
        reload_receipt = json.loads(reload_path.read_text())
        if (receipt.get("reload_inference_receipt_sha256") != _sha256(reload_path)
                or reload_receipt.get("loaded_adapter_sha256") != after_sha
                or reload_receipt.get("contract_id") != contract_id
                or reload_receipt.get("image_sha256") != raw[0]["image_sha256"]
                or reload_receipt.get("actual_model_generation") is not True
                or reload_receipt.get("exact_adapter_tensor_match") is not True
                or reload_receipt.get("fresh_model_loaded") is not True
                or not reload_receipt.get("generated_text")):
            raise ValueError("Child checkpoint reload linkage invalid")
        for name in ("before_adapters.safetensors", "after_adapters.safetensors", "reload_inference_receipt.json"):
            files[name] = _sha256(step_dir / name)
        final_sha = after_sha
    else:
        if (receipt.get("status") != "NO_UPDATE_NO_TASK_VARIANCE"
                or receipt.get("optimizer_steps") != 0 or informative):
            raise ValueError("Child no-update receipt is inconsistent")
        final_sha = starting_sha
    return {"record_id": row["id"], "category": row["ability"],
            "starting_adapter_sha256": starting_sha, "final_adapter_sha256": final_sha,
            "contract_id": contract_id, "review_sha256": review_sha,
            "optimizer_steps": int(updated), "task_reward_varying": informative,
            "optimizer_reset_each_group": True, "exact_optimizer_resume": False,
            "artifact_sha256": files, "status": receipt["status"]}


def run(model: Path, records: Path, output: Path, starting_adapter: Path,
        arm: str, groups: int = 8, group_size: int = 2, max_tokens: int = 256,
        seed: int = 7, specialist_category: str | None = None,
        reward_profile: str = "direct_vero", system_prompt_file: Path | None = None,
        contract_path: Path | None = None, dev_records: Path | None = None,
        dynamic_receipt_path: Path | None = None, review_verdict_path: Path | None = None,
        expected_review_sha256: str | None = None, execution_tier: str = "C",
        phase: str = "continuation", temperature: float = 1.0,
        child_authorizations: dict[int, dict] | None = None, resume: bool = False) -> dict:
    """Run only separately authorized children; pause durably for the next review.

    Child approval digests are supplied by the trusted orchestration caller.
    Do not infer them from files or preapprove unknown future adapter hashes.
    """
    expected_system = (load_contract(contract_path).system_instruction if contract_path is not None
                       else DEFAULT_RL_SYSTEM_PROMPT)
    if system_prompt_file is not None and system_prompt_file.read_text() != expected_system:
        raise ValueError("Parent system prompt differs from contract")
    contract_id = require_launch_preflight(
        contract_path=contract_path, train=records, dev=dev_records, model_path=model,
        dynamic_receipt_path=dynamic_receipt_path, review_verdict_path=review_verdict_path,
        expected_system=expected_system, expected_max_tokens=max_tokens,
        expected_reward_profile=reward_profile, expected_starting_adapter=starting_adapter,
        expected_review_sha256=expected_review_sha256, execution_tier=execution_tier,
        phase=phase, expected_temperature=temperature,
        launch_budget={"optimizer_steps": groups, "max_groups": groups,
                       "rollouts_per_group": group_size, "max_tokens": max_tokens},
        launch_configuration={"runner": "mlx_multiarm_train", "arm": arm, "seed": seed,
                              "specialist_category": specialist_category,
                              "output": str(output.resolve())})
    if execution_tier != "C" or phase != "continuation" or reward_profile != "direct_vero" or temperature != 1.0:
        raise ValueError("Multiarm requires reviewed Tier C continuation and strict Vero temperature one")
    private_root = Path(__file__).resolve().parents[2] / "local_data"
    if not output.resolve().is_relative_to(private_root):
        raise ValueError("Raw MRI rollouts and weights must stay under ignored local_data/")
    if not 2 <= group_size <= 8 or not 16 <= max_tokens <= 512:
        raise ValueError("Bounded child response budget exceeded")
    schedule, details = build_schedule(records, arm, groups, seed, specialist_category)
    identity = {**details, "experiment_contract_id": contract_id,
                "starting_adapter_sha256": _sha256(starting_adapter),
                "training_records_sha256": _sha256(records),
                "model_config_sha256": _sha256(model / "config.json"),
                "group_size": group_size, "max_tokens": max_tokens, "temperature": temperature,
                "reward_profile": reward_profile, "system_sha256": hashlib.sha256(expected_system.encode()).hexdigest(),
                "optimizer_reset_each_group": True, "exact_optimizer_resume": False,
                "implementation_scope": "sequential one-step MLX engineering pilot; not native continuous GSPO"}
    identity_sha = canonical_hash(identity)
    state_path = output / "arm_receipt_PRIVATE.json"
    results = []
    if state_path.exists():
        if not resume:
            raise ValueError("Existing arm requires explicit resume; no historical outputs overwritten")
        state = json.loads(state_path.read_text())
        if state.get("identity_sha256") != identity_sha:
            raise ValueError("Resume schedule, model, manifest, starting adapter or budget changed")
        results = state.get("group_results", [])
        if not isinstance(results, list) or len(results) > len(schedule):
            raise ValueError("Invalid saved completed-prefix length")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("Untracked arm output exists without a validated resumable state")
    output.mkdir(parents=True, exist_ok=True)
    current_adapter = starting_adapter
    # Recompute every completed boundary from actual raw files on resume.
    for index, saved in enumerate(results):
        step_dir = output / f"step_{index:02d}"
        actual = _verified_child(step_dir, schedule[index], _sha256(current_adapter),
                                 contract_id=saved["contract_id"], review_sha=saved["review_sha256"],
                                 records=output / "manifests" / f"step_{index:02d}" / "train.verl.jsonl",
                                 expected_system=expected_system, seed=seed + index, group_size=group_size, max_tokens=max_tokens)
        if actual != saved:
            raise ValueError("Completed child artifacts or adapter chain changed since checkpoint")
        if saved["optimizer_steps"]:
            current_adapter = step_dir / "after_adapters.safetensors"

    def persist(status, pending=None):
        state = {**identity, "identity_sha256": identity_sha,
                 "status": status, "group_results": results,
                 "completed_prompt_groups": len(results),
                 "optimizer_steps": sum(item["optimizer_steps"] for item in results),
                 "actual_rollouts": len(results) * group_size,
                 "final_adapter_sha256": _sha256(current_adapter),
                 "pending_child_review": pending, "parent_review_sha256": expected_review_sha256,
                 "scientific_claim": False}
        atomic_json(state_path, state)
        return state

    for index in range(len(results), len(schedule)):
        row = schedule[index]
        step_dir = output / f"step_{index:02d}"
        manifest_dir = output / "manifests" / f"step_{index:02d}"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        step_record = manifest_dir / "train.verl.jsonl"
        payload = json.dumps(row, sort_keys=True) + "\n"
        if step_record.exists() and step_record.read_text() != payload:
            raise ValueError("Pending child manifest changed")
        if not step_record.exists():
            step_record.write_text(payload)
        authorization = (child_authorizations or {}).get(index)
        pending = {"step": index, "records": str(step_record.resolve()),
                   "output": str(step_dir.resolve()), "route": row["extra_info"]["reward_type"],
                   "seed": seed + index, "starting_adapter": str(current_adapter.resolve()),
                   "starting_adapter_sha256": _sha256(current_adapter),
                   "group_size": group_size, "max_tokens": max_tokens,
                   "execution_tier": "B", "phase": "diagnostic", "max_groups": 1,
                   "diverse_labels": False}
        persist("PAUSED_FOR_CHILD_REVIEW", pending)
        if authorization is None:
            return persist("PAUSED_FOR_CHILD_REVIEW", pending)
        child_contract = Path(authorization["contract_path"])
        child_review = Path(authorization["review_verdict_path"])
        child_digest = authorization["expected_review_sha256"]
        child_id = require_launch_preflight(
            contract_path=child_contract, train=step_record, dev=dev_records, model_path=model,
            dynamic_receipt_path=None, review_verdict_path=child_review,
            expected_system=expected_system, expected_max_tokens=max_tokens,
            expected_reward_profile=reward_profile, expected_starting_adapter=current_adapter,
            execution_tier="B", phase="diagnostic", expected_review_sha256=child_digest,
            expected_temperature=temperature,
            launch_budget={"optimizer_steps": 1, "max_groups": 1,
                           "rollouts_per_group": group_size, "max_tokens": max_tokens},
            launch_configuration={"runner": "mlx_gspo_smoke", "route": pending["route"],
                                  "seed": seed + index, "diverse_labels": False,
                                  "output": str(step_dir.resolve())})
        if not (step_dir / "receipt.json").exists():
            # A partial in-flight run is preserved for explicit recovery; never
            # silently repeat paid work or overwrite unknown adapter state.
            if (step_dir / "progress.json").exists() or (step_dir / "raw_rollouts.json").exists():
                raise ValueError("Interrupted child requires explicit recovery before resuming this arm")
            cmd = [sys.executable, "-m", "mri_rl.mlx_gspo_smoke",
                   "--model", str(model), "--records", str(step_record), "--output", str(step_dir),
                   "--route", pending["route"], "--group-size", str(group_size),
                   "--max-tokens", str(max_tokens), "--max-groups", "1", "--seed", str(seed + index),
                   "--reward-profile", reward_profile, "--adapter-checkpoint", str(current_adapter),
                   "--contract", str(child_contract), "--dev-records", str(dev_records),
                   "--review-verdict", str(child_review), "--expected-review-sha256", child_digest,
                   "--execution-tier", "B", "--phase", "diagnostic"]
            completed = subprocess.run(cmd, env=os.environ.copy(), text=True,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            step_dir.mkdir(exist_ok=True)
            (step_dir / "process_log_PRIVATE.txt").write_text(completed.stdout)
            if completed.returncode:
                persist("CHILD_FAILED_REQUIRES_RECOVERY", pending)
                raise RuntimeError(f"GSPO child {index} failed; preserved private log and receipt")
        verified = _verified_child(step_dir, row, _sha256(current_adapter),
                                   contract_id=child_id, review_sha=child_digest, records=step_record,
                                   expected_system=expected_system, seed=seed + index, group_size=group_size, max_tokens=max_tokens)
        results.append(verified)
        if verified["optimizer_steps"]:
            current_adapter = step_dir / "after_adapters.safetensors"
        persist("RUNNING_REVIEWED_PREFIX")
    return persist("TRAINING_COMPLETED_BOUNDED_ARM" if any(item["optimizer_steps"] for item in results)
                   else "NO_UPDATE_COMPLETED_BOUNDED_ARM")


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
    parser.add_argument("--reward-profile", choices=("direct_vero", "scaled_smoke",
                                                     "boxed_only_ablation",
                                                     "single_box_anywhere_ablation"),
                        default="direct_vero")
    parser.add_argument("--system-prompt-file", type=Path)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--dev-records", type=Path, required=True)
    parser.add_argument("--dynamic-receipt", type=Path, required=True)
    parser.add_argument("--review-verdict", type=Path, required=True)
    parser.add_argument("--expected-review-sha256", required=True)
    parser.add_argument("--execution-tier", choices=("C",), default="C")
    parser.add_argument("--phase", choices=("continuation",), default="continuation")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--child-authorizations", type=Path,
                        help="JSON map of step index to contract_path/review_verdict_path; excludes trusted digests")
    parser.add_argument("--child-review-digest", action="append", default=[], metavar="INDEX=SHA256",
                        help="Digest received independently from the reviewer")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    children = {int(index): value for index, value in json.loads(args.child_authorizations.read_text()).items()} if args.child_authorizations else {}
    for value in children.values():
        value.pop("expected_review_sha256", None)
    for declaration in args.child_review_digest:
        index, digest = declaration.split("=", 1)
        if int(index) not in children:
            parser.error("Child digest has no corresponding authorization paths")
        children[int(index)]["expected_review_sha256"] = digest
    if any("expected_review_sha256" not in value for value in children.values()):
        parser.error("Every child requires an independently supplied --child-review-digest")
    receipt = run(args.model.resolve(), args.records.resolve(), args.output.resolve(),
                  args.starting_adapter.resolve(), args.arm, args.groups,
                  args.group_size, args.max_tokens, args.seed,
                  args.specialist_category, args.reward_profile,
                  args.system_prompt_file.resolve() if args.system_prompt_file else None,
                  args.contract.resolve(), args.dev_records.resolve(),
                  args.dynamic_receipt.resolve(), args.review_verdict.resolve(),
                  expected_review_sha256=args.expected_review_sha256,
                  execution_tier=args.execution_tier, phase=args.phase,
                  temperature=args.temperature, child_authorizations=children, resume=args.resume)
    print(json.dumps({key: value for key, value in receipt.items()
                      if key != "group_results"}, indent=2))


if __name__ == "__main__":
    main()
