"""Bounded patient-keyed visual SFT LoRA warm-start for MRI format learning.

This is a separate supervised control, not GSPO or the official Vero trainer.
Only the ds007045 internal training JSONL is accepted. A saved adapter may
then be supplied as the identical starting state to multiple GSPO arms.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
from collections import Counter

from mri_rl.mlx_gspo_smoke import _sha256
from mri_rl.experiment_contract import require_launch_preflight, load_contract
from mri_rl.resource_supervisor import ResourceGate


CATEGORIES = ("mri_sequence", "mri_tumor_presence",
              "mri_enhancing_presence", "mri_tumor_grid")
SCHEDULE_POLICIES = ("legacy_lexical_v1", "balanced_sequence_v2")
SEQUENCE_LABELS = ("FLAIR", "contrast-enhanced T1")
SFT_SYSTEM_PROMPT = (
    "Respond with one short observation in <think> and one boxed final answer in <answer>.")


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _save_step_checkpoint(save_trainable_weights, model, output: Path,
                          step_number: int, metadata: dict) -> dict:
    """Durably preserve weights at a completed optimizer-step boundary.

    This does not capture optimizer/RNG state and is not an exact resume point.
    """
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    path = checkpoint_dir / f"step_{step_number:03d}_adapters.safetensors"
    temporary = checkpoint_dir / f"step_{step_number:03d}_adapters.partial.safetensors"
    save_trainable_weights(model, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    receipt = {**metadata, "completed_optimizer_steps": step_number,
               "adapter_sha256": _sha256(path), "adapter_path": str(path),
               "optimizer_state_saved": False, "rng_state_saved": False,
               "exact_resume_supported": False}
    _atomic_json(checkpoint_dir / f"step_{step_number:03d}_receipt.json", receipt)
    return receipt


def _schedule_train_rows(path: Path, steps: int, seed: int,
                         schedule_policy: str = "balanced_sequence_v2") -> list[list[dict]]:
    """Future v2 balances sequence labels; v1 exactly replays historic selection.

    Keep the same seed-hashed patient order, category budget and total examples.
    A missing assigned sequence label fails explicitly rather than substituting
    an available label, dropping a patient, or reading development patients.
    """
    if schedule_policy not in SCHEDULE_POLICIES:
        raise ValueError("Unknown versioned SFT schedule policy")
    if path.name != "train.verl.jsonl":
        raise ValueError("Warm-start accepts only internal train.verl.jsonl")
    if not 1 <= steps <= 28:
        raise ValueError("Bounded SFT supports 1–28 steps")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    by_category: dict[str, dict[str, list[dict]]] = {category: {} for category in CATEGORIES}
    for row in rows:
        if row.get("data_source") != "OpenNeuro_ds007045" or row["extra_info"].get(
                "patient_lineage") != "BIDS_SUBJECT_VERIFIED" or row["extra_info"].get(
                "internal_split") != "train":
            raise ValueError("Unexpected source or patient lineage")
        if not Path(row["images"][0]).is_file():
            raise FileNotFoundError("Missing MRI train image")
        if row["ability"] in CATEGORIES:
            by_category[row["ability"]].setdefault(row["extra_info"]["patient_key"], []).append(row)
    ordered = {}
    for category, patients in by_category.items():
        if len(patients) < steps:
            raise ValueError("Too few distinct train patients for SFT schedule")
        patient_keys = sorted(patients, key=lambda key: hashlib.sha256(
            f"{seed}:{category}:{key}".encode()).hexdigest())
        selected = []
        label_offset = int(hashlib.sha256(f"{seed}:sequence-label-offset".encode()).hexdigest(), 16) % 2
        for index, key in enumerate(patient_keys[:steps]):
            candidates = patients[key]
            if schedule_policy == "balanced_sequence_v2" and category == "mri_sequence":
                required_label = SEQUENCE_LABELS[(index + label_offset) % 2]
                candidates = [row for row in candidates
                              if row["reward_model"]["ground_truth"] == required_label]
                if not candidates:
                    raise ValueError("Balanced sequence schedule lacks the assigned label "
                                     f"{required_label!r} for a selected training patient; "
                                     "obtain eligible training images or revise the reviewed schedule")
            selected.append(min(candidates, key=lambda row: row["id"]))
        ordered[category] = selected
    schedule = [[ordered[category][step] for category in CATEGORIES]
                for step in range(steps)]
    if schedule_policy == "balanced_sequence_v2" and steps >= 2:
        counts = schedule_label_exposure(schedule)["mri_sequence"]
        if set(counts) != set(SEQUENCE_LABELS) or abs(counts[SEQUENCE_LABELS[0]] - counts[SEQUENCE_LABELS[1]]) > 1:
            raise ValueError("Future SFT schedule must expose both sequence labels with balanced counts")
    return schedule


def schedule_label_exposure(schedule: list[list[dict]]) -> dict[str, dict[str, int]]:
    return {category: dict(sorted(Counter(
        row["reward_model"]["ground_truth"] for step in schedule for row in step
        if row["ability"] == category).items())) for category in CATEGORIES}


def _selected_train_rows(path: Path) -> list[dict]:
    """Backward-compatible one-step selection for CPU validation."""
    return _schedule_train_rows(path, steps=1, seed=7, schedule_policy="legacy_lexical_v1")[0]


def run(model_path: Path, records: Path, output: Path, seed: int = 7,
        steps: int = 1, resource_guard: bool = True,
        contract_path: Path | None = None, dev_records: Path | None = None,
        dynamic_receipt_path: Path | None = None,
        review_verdict_path: Path | None = None,
        starting_adapter: Path | None = None,
        expected_review_sha256: str | None = None,
        execution_tier: str = "C", phase: str = "continuation",
        schedule_policy: str = "balanced_sequence_v2") -> dict:
    generation = load_contract(contract_path).generation if contract_path is not None else {"max_tokens": 256, "temperature": 0.0}
    contract_id = require_launch_preflight(
        contract_path=contract_path, train=records, dev=dev_records,
        model_path=model_path, dynamic_receipt_path=dynamic_receipt_path,
        review_verdict_path=review_verdict_path,
        expected_system=SFT_SYSTEM_PROMPT,
        expected_starting_adapter=starting_adapter, expected_review_sha256=expected_review_sha256,
        execution_tier=execution_tier, phase=phase,
        expected_max_tokens=generation["max_tokens"], expected_temperature=generation["temperature"],
        launch_budget={"optimizer_steps": steps, "max_groups": steps,
                       "rollouts_per_group": len(CATEGORIES), "max_tokens": generation["max_tokens"]},
        launch_configuration={"runner": "mlx_sft_warmstart", "seed": seed,
                              "output": str(output.resolve()), "resource_guard": resource_guard,
                              "target_max_tokens": 64, "examples_per_update": len(CATEGORIES),
                              "schedule_policy": schedule_policy})
    if execution_tier != "C" or phase != "continuation":
        raise ValueError("SFT control requires explicit Tier C continuation authorization")
    if not resource_guard:
        raise ValueError("Reviewed training requires resource supervision")
    private_root = Path(__file__).resolve().parents[2] / "local_data"
    if not output.resolve().is_relative_to(private_root):
        raise ValueError("MRI adapters and receipts must stay under ignored local_data/")
    if not model_path.is_dir():
        raise FileNotFoundError("Local base checkpoint missing")
    if schedule_policy != "balanced_sequence_v2":
        raise ValueError("Future SFT execution requires balanced_sequence_v2; legacy policy is reconstruction only")
    schedule = _schedule_train_rows(records, steps, seed, schedule_policy)
    label_exposure = schedule_label_exposure(schedule)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "receipt.json").exists() or list((output / "checkpoints").glob("step_*_receipt.json")):
        raise ValueError("Output already contains a completed or partial SFT run; choose a new directory")
    gate = ResourceGate(output / "resource_events.jsonl") if resource_guard else None
    if gate:
        gate.before_unit("model_load")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_map
    from mlx_vlm import load
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.trainer.common.model import prepare_model_for_training
    from mlx_vlm.trainer.common.utils import save_trainable_weights
    from mlx_vlm.utils import prepare_inputs, should_add_special_tokens

    mx.random.seed(seed)
    model, processor = load(str(model_path))
    model = prepare_model_for_training(model, train_type="lora", lora_rank=4,
                                       lora_alpha=8, quantization_bits=4,
                                       quantization_group_size=64, checkpoint_path=starting_adapter, verbose=False)
    gc.collect()
    mx.clear_cache()
    before = output / "before_adapters.safetensors"
    after = output / "after_adapters.safetensors"
    save_trainable_weights(model, before)
    before_sha = _sha256(before)
    model.train()

    def loss_for(model, prompt_ids, target_ids, pixel_values, model_kwargs):
        ids = mx.concatenate((prompt_ids, target_ids[None, :]), axis=1)
        inputs = ids[:, :-1]
        logits = model(inputs, pixel_values, mx.ones_like(inputs), **model_kwargs).logits.astype(mx.float32)
        labels = ids[:, 1:]
        if logits.shape[1] != labels.shape[1]:
            raise ValueError("SFT logits/labels length mismatch")
        selected = mx.take_along_axis(nn.log_softmax(logits, axis=-1),
                                      labels[..., None], axis=-1)[..., 0]
        return -mx.mean(selected[0, prompt_ids.shape[-1]-1:])

    loss_and_grad = nn.value_and_grad(model, loss_for)
    optimizer = optim.Adam(learning_rate=1e-5)
    checkpoint_metadata = {
        "model_config_sha256": _sha256(model_path / "config.json"),
        "training_records_sha256": _sha256(records),
        "seed": seed, "planned_optimizer_steps": steps,
        "schedule_policy": schedule_policy, "scheduled_label_exposure": label_exposure,
        "precision": "MLX 4-bit base, LoRA rank 4",
        "microbatch_size": 1, "gradient_accumulation_examples": len(CATEGORIES),
        "image_budget_prompt_tokens": 2048,
    }
    losses = []
    examples = []
    gradient_norms = []
    for step_index, chosen in enumerate(schedule):
        if gate:
            gate.before_unit(f"optimizer_step_{step_index + 1}")
        grad_sum = None
        for row in chosen:
            image = row["images"][0]
            question = row["extra_info"]["question"]
            messages = [
                {"role": "system", "content": SFT_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ]
            prompt = apply_chat_template(processor, model.config, messages,
                                         add_generation_prompt=True, num_images=1)
            if prompt.count("<|image_pad|>") != 1:
                raise ValueError("Expected exactly one image placeholder")
            prepared = prepare_inputs(
                processor, images=[image], prompts=prompt,
                image_token_index=getattr(model.config, "image_token_index", None),
                add_special_tokens=should_add_special_tokens(model.config.model_type, processor),
            )
            prompt_ids = prepared["input_ids"]
            if prompt_ids.shape[-1] > 2048:
                raise ValueError("Prompt image exceeds SFT budget")
            answer = row["reward_model"]["ground_truth"]
            target = f"<think>Visible MRI evidence.</think><answer>\\boxed{{{answer}}}</answer>"
            target_ids = mx.array(processor.tokenizer.encode(
                target, add_special_tokens=False), dtype=prompt_ids.dtype)
            if target_ids.size == 0 or target_ids.size > 64:
                raise ValueError("SFT target outside 1–64 token budget")
            model_kwargs = {key: value for key, value in prepared.items()
                            if key not in {"input_ids", "attention_mask", "pixel_values"}}
            loss, grad = loss_and_grad(model, prompt_ids, target_ids,
                                       prepared.get("pixel_values"), model_kwargs)
            mx.eval(loss, grad)
            losses.append(float(loss.item()))
            examples.append({"step": step_index, "category": row["ability"],
                             "patient_key": row["extra_info"]["patient_key"],
                             "prompt_tokens": int(prompt_ids.shape[-1]),
                             "target_tokens": int(target_ids.size), "target_label": answer,
                             "record_id": row["id"], "image_sha256": _sha256(Path(image)),
                             "target_sha256": hashlib.sha256(target.encode()).hexdigest()})
            scaled = tree_map(lambda x: x / len(chosen), grad)
            grad_sum = scaled if grad_sum is None else tree_map(lambda a, b: a+b, grad_sum, scaled)
            mx.eval(grad_sum)
            mx.clear_cache()
        grad_norm = math.sqrt(sum(float(mx.sum(g.astype(mx.float32) ** 2).item())
                                  for _, g in tree_flatten(grad_sum)))
        if not math.isfinite(grad_norm) or grad_norm <= 0:
            raise RuntimeError("SFT produced no finite nonzero gradient")
        gradient_norms.append(grad_norm)
        optimizer.update(model, grad_sum)
        mx.eval(model.trainable_parameters(), optimizer.state)
        mx.clear_cache()
        _save_step_checkpoint(save_trainable_weights, model, output,
                              step_index + 1, checkpoint_metadata)
    save_trainable_weights(model, after)
    after_sha = _sha256(after)
    if before_sha == after_sha:
        raise RuntimeError("SFT optimizer step did not change adapter weights")
    receipt = {
        "experiment_contract_id": contract_id, "review_sha256": expected_review_sha256,
        "starting_adapter_sha256": _sha256(starting_adapter), "execution_tier": execution_tier,
        "status": "SFT_ENGINEERING_CONTROL", "model_path_name": model_path.name,
        "source": "OpenNeuro_ds007045", "internal_train_only": True,
        "heldout_test_used": False, "categories": list(CATEGORIES),
        "examples": examples, "loss_terms": losses, "gradient_l2_by_step": gradient_norms,
        "optimizer_steps": steps, "learning_rate": 1e-5,
        "schedule_policy": schedule_policy, "label_exposure_by_route": label_exposure,
        "distinct_patients_exposed": len({example["patient_key"] for example in examples}),
        "before_adapter_sha256": before_sha, "after_adapter_sha256": after_sha,
        "trainable_adapter_format": "MLX safetensors 4-bit base LoRA rank 4",
        "periodic_adapter_checkpoints": steps,
        "exact_resume_supported": False,
        "science_claim": False,
    }
    _atomic_json(output / "receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--no-resource-guard", action="store_true")
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--dev-records", type=Path, required=True)
    parser.add_argument("--dynamic-receipt", type=Path, required=True)
    parser.add_argument("--review-verdict", type=Path, required=True)
    parser.add_argument("--starting-adapter", type=Path, required=True)
    parser.add_argument("--expected-review-sha256", required=True)
    parser.add_argument("--execution-tier", choices=("C",), default="C")
    parser.add_argument("--phase", choices=("continuation",), default="continuation")
    parser.add_argument("--schedule-policy", choices=("balanced_sequence_v2",), default="balanced_sequence_v2")
    args = parser.parse_args()
    print(json.dumps(run(args.model.resolve(), args.records.resolve(),
                         args.output.resolve(), args.seed, args.steps,
                         resource_guard=not args.no_resource_guard,
                         contract_path=args.contract.resolve(),
                         dev_records=args.dev_records.resolve(),
                         dynamic_receipt_path=args.dynamic_receipt.resolve(),
                         review_verdict_path=args.review_verdict.resolve(),
                         starting_adapter=args.starting_adapter.resolve(),
                         expected_review_sha256=args.expected_review_sha256,
                         execution_tier=args.execution_tier, phase=args.phase,
                         schedule_policy=args.schedule_policy), indent=2))


if __name__ == "__main__":
    main()
