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
import time
import random
from pathlib import Path
from collections import Counter

from mri_rl.mlx_gspo_smoke import _sha256
from mri_rl.experiment_contract import require_launch_preflight, load_contract, canonical_hash, check_resource
from mri_rl.resource_supervisor import ResourceGate, sample_macos
from mri_rl.paced_resources import PacedGate, PACED_POLICY
from mri_rl.optimizer_checkpoint import save_optimizer_checkpoint


CATEGORIES = ("mri_sequence", "mri_tumor_presence",
              "mri_enhancing_presence", "mri_tumor_grid")
SCHEDULE_POLICIES = ("legacy_lexical_v1", "balanced_sequence_v2")
SEQUENCE_LABELS = ("FLAIR", "contrast-enhanced T1")
SFT_SYSTEM_PROMPT = (
    "Respond with one short observation in <think> and one boxed final answer in <answer>.")


def _sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _sync_directory(path.parent)


def _save_step_checkpoint(save_trainable_weights, model, output: Path,
                          step_number: int, metadata: dict) -> dict:
    """Durably preserve weights at a completed optimizer-step boundary.

    This does not capture optimizer/RNG state and is not an exact resume point.
    """
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    _sync_directory(output)
    path = checkpoint_dir / f"step_{step_number:03d}_adapters.safetensors"
    temporary = checkpoint_dir / f"step_{step_number:03d}_adapters.partial.safetensors"
    save_trainable_weights(model, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _sync_directory(path.parent)
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


def schedule_inventory(schedule: list[list[dict]]) -> list[dict]:
    return [{"step": step, "category": row["ability"], "record_id": row["id"],
             "patient_key": row["extra_info"]["patient_key"],
             "image_sha256": _sha256(Path(row["images"][0])),
             "target_label": row["reward_model"]["ground_truth"],
             "question_sha256": hashlib.sha256(row["extra_info"]["question"].encode()).hexdigest()}
            for step, rows in enumerate(schedule) for row in rows]


def verify_frozen_row(row: dict, step: int, frozen: dict) -> None:
    actual = schedule_inventory([[row]])[0]
    actual["step"] = step
    if actual != frozen:
        raise ValueError("Selected SFT example changed after schedule freeze")


def require_empty_output(output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise ValueError("Refusing nonempty SFT output; preserve partial work and choose a new directory")


def require_runtime_resources():
    sample = sample_macos()
    failures = check_resource(sample)
    if sample.other_heavy_processes is None or sample.other_heavy_processes > 0:
        failures.append("concurrent model process inventory unavailable or occupied")
    if failures:
        raise RuntimeError("SFT resource gate: " + "; ".join(failures))


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
    if contract_path is None or dev_records is None:
        raise ValueError("Private contract and dev split are required")
    if not 1 <= steps <= 2:
        raise ValueError("Current post-panic authorization permits only a reviewed 1–2 step compatibility prefix")
    schedule = _schedule_train_rows(records, 28, seed, schedule_policy)
    inventory = schedule_inventory(schedule)
    prefix_binding = {"executed_prefix_sha256": canonical_hash(inventory[:4*steps]),
                      "executed_prefix_examples": 4*steps}
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
                              "schedule_policy": schedule_policy,
                              "schedule_sha256": canonical_hash(inventory),
                              "paced_policy": PACED_POLICY, **prefix_binding})
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
    label_exposure = schedule_label_exposure(schedule)
    require_empty_output(output)
    output.mkdir(parents=True, exist_ok=True)
    _sync_directory(output.parent)
    _atomic_json(output / "schedule_PRIVATE.json", {"rows": inventory, "sha256": canonical_hash(inventory)})
    gate = PacedGate(output / "resource_events.jsonl") if resource_guard else None
    if gate:
        gate.before_unit("model_load", loading=True)
    gate.before_unit("before_runtime_import", loading=True)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_map
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.trainer.common.model import prepare_model_for_training
    from mlx_vlm.trainer.common.utils import save_trainable_weights, enable_gradient_checkpointing
    from mlx_vlm.utils import prepare_inputs, should_add_special_tokens

    mx.set_memory_limit(PACED_POLICY["memory_guideline_gib"] * 1024**3)
    mx.set_cache_limit(PACED_POLICY["cache_limit_mib"] * 1024**2)
    mx.random.seed(seed)
    random.seed(seed)
    model, processor = load(str(model_path), lazy=True)
    model = prepare_model_for_training(model, train_type="lora", lora_rank=4,
                                       lora_alpha=8, quantization_bits=4,
                                       quantization_group_size=64, checkpoint_path=starting_adapter, verbose=False)
    # Realize restored quantized leaves before backprop; release lazy float sources.
    mx.eval(model.parameters())
    gc.collect(); mx.clear_cache()
    gate.before_unit("quantized_model_ready")
    enable_gradient_checkpointing(model.language_model.layers)
    saved_start = mx.load(str(starting_adapter))
    loaded_start = dict(tree_flatten(model.trainable_parameters()))
    if set(saved_start) != set(loaded_start) or not all(
            bool(mx.array_equal(value, loaded_start[name]).item()) for name, value in saved_start.items()):
        raise RuntimeError("Loaded SFT starting tensors differ from reviewed original adapter")
    del saved_start, loaded_start
    gc.collect()
    mx.clear_cache()
    before = output / "before_adapters.safetensors"
    after = output / "after_adapters.safetensors"
    save_trainable_weights(model, before)
    before_sha = _sha256(before)
    if before_sha != _sha256(starting_adapter):
        raise RuntimeError("Saved SFT before-adapter hash differs from reviewed start")
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
    optimizer = optim.Adam(learning_rate=1e-5, bias_correction=False)
    checkpoint_metadata = {
        "model_config_sha256": _sha256(model_path / "config.json"),
        "training_records_sha256": _sha256(records),
        "seed": seed, "planned_optimizer_steps": steps, **prefix_binding,
        "schedule_policy": schedule_policy, "scheduled_label_exposure": label_exposure,
        "precision": "MLX 4-bit base, LoRA rank 4",
        "microbatch_size": 1, "gradient_accumulation_examples": len(CATEGORIES),
        "image_budget_prompt_tokens": 2048,
    }
    losses = []
    examples = []
    gradient_norms = []
    previous_adapter_sha = before_sha
    for step_index, chosen in enumerate(schedule[:steps]):
        if gate:
            gate.before_unit(f"optimizer_step_{step_index + 1}")
        gate.before_unit(f"before_step_{step_index+1}")
        grad_sum = None
        for category_index, row in enumerate(chosen):
            gate.before_unit(f"step_{step_index+1}_example_{category_index+1}")
            verify_frozen_row(row, step_index, inventory[step_index * len(CATEGORIES) + category_index])
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
            if not math.isfinite(float(loss.item())):
                raise RuntimeError("Nonfinite SFT loss")
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
            del grad, scaled, loss, prepared, model_kwargs, prompt_ids, target_ids
            gc.collect(); mx.clear_cache()
            time.sleep(PACED_POLICY["example_pause_seconds"])
        grad_norm = math.sqrt(sum(float(mx.sum(g.astype(mx.float32) ** 2).item())
                                  for _, g in tree_flatten(grad_sum)))
        if not math.isfinite(grad_norm) or grad_norm <= 0:
            raise RuntimeError("SFT produced no finite nonzero gradient")
        gradient_norms.append(grad_norm)
        gate.before_unit(f"optimizer_update_{step_index+1}")
        optimizer.update(model, grad_sum)
        mx.eval(model.trainable_parameters(), optimizer.state)
        mx.clear_cache()
        boundary = _save_step_checkpoint(save_trainable_weights, model, output,
                              step_index + 1, checkpoint_metadata)
        if boundary["adapter_sha256"] == previous_adapter_sha:
            raise RuntimeError("Completed SFT step did not change adapter bytes")
        finite = all(bool(mx.all(mx.isfinite(v)).item()) for _,v in tree_flatten(model.trainable_parameters()))
        if not finite:
            raise RuntimeError("Completed SFT step has nonfinite parameters; no continuation")
        boundary.update({"previous_adapter_sha256": previous_adapter_sha,
                         "parameters_finite": True, "weight_changed": True})
        previous_adapter_sha = boundary["adapter_sha256"]
        state_rng = {"mlx": [mx.array(v) for v in mx.random.state], "python": random.getstate()}
        mx.eval(optimizer.state, state_rng["mlx"])
        signature = {k:{"shape":list(v.shape),"dtype":str(v.dtype)}
                     for k,v in tree_flatten(model.trainable_parameters())}
        save_optimizer_checkpoint(output / "checkpoints" / f"state_{step_index+1:03d}",
            optimizer_state=optimizer.state, rng_state=state_rng,
            bindings={"adapter_sha256":boundary["adapter_sha256"],"contract_id":contract_id,
                      "trainable_signature":signature,"schedule_sha256":canonical_hash(inventory),
                      "sampler_position_examples":4*(step_index+1),
                      "optimizer":{"name":"Adam","lr":1e-5,"bias_correction":False},
                      "backend":mx.__version__},
            completed_groups=step_index+1,optimizer_updates=step_index+1,
            save_arrays=lambda path,arrays:mx.save_safetensors(str(path),arrays))
        boundary.update({"optimizer_state_saved":True,"rng_state_saved":True,
                         "state_directory":f"state_{step_index+1:03d}",
                         "exact_resume_supported":False})
        _atomic_json(output / "checkpoints" / f"step_{step_index+1:03d}_receipt.json", boundary)
        _atomic_json(output / "progress_PRIVATE.json", {
            "completed_optimizer_steps": step_index + 1, "examples": examples,
            "loss_terms": losses, "gradient_l2_by_step": gradient_norms,
            "schedule_sha256": canonical_hash(inventory), "contract_id": contract_id, **prefix_binding})
        print(f"SFT completed step {step_index + 1}/{steps}; loss={sum(losses[-4:])/4:.5f}", flush=True)
        gate.after_step(step_index+1)
    save_trainable_weights(model, after)
    after_sha = _sha256(after)
    if before_sha == after_sha:
        raise RuntimeError("SFT optimizer step did not change adapter weights")
    receipt = {
        "experiment_contract_id": contract_id, "review_sha256": expected_review_sha256,
        "starting_adapter_sha256": _sha256(starting_adapter), "execution_tier": execution_tier,
        "adapter_config_sha256": _sha256(output / "adapter_config.json"),
        "status": "SFT_RELOAD_PENDING", "model_path_name": model_path.name,
        "source": "OpenNeuro_ds007045", "internal_train_only": True,
        "heldout_test_used": False, "categories": list(CATEGORIES),
        "examples": examples, "loss_terms": losses, "gradient_l2_by_step": gradient_norms,
        "optimizer_steps": steps, "learning_rate": 1e-5, **prefix_binding,
        "adam_bias_correction": False, "starting_tensors_verified": True,
        "schedule_sha256": canonical_hash(inventory),
        "schedule_policy": schedule_policy, "scheduled_full28_label_exposure_by_route": label_exposure,
        "label_exposure_by_route": schedule_label_exposure(schedule[:steps]),
        "completed_control_28steps": False,
        "distinct_patients_exposed": len({example["patient_key"] for example in examples}),
        "before_adapter_sha256": before_sha, "after_adapter_sha256": after_sha,
        "trainable_adapter_format": "MLX safetensors 4-bit base LoRA rank 4",
        "periodic_adapter_checkpoints": steps,
        "exact_resume_supported": False, "continuous_single_process_optimizer": True,
        "optimizer_state_saved_each_step": True, "paced_policy": PACED_POLICY,
        "science_claim": False,
    }
    _atomic_json(output / "receipt.json", receipt)
    before_weights, after_weights = mx.load(str(before)), mx.load(str(after))
    if set(before_weights) != set(after_weights):
        raise RuntimeError("Adapter parameter set changed")
    changed = sum(not bool(mx.array_equal(v, after_weights[k]).item()) for k,v in before_weights.items())
    finite = all(bool(mx.all(mx.isfinite(v)).item()) for v in after_weights.values())
    delta_l2 = math.sqrt(sum(float(mx.sum((after_weights[k].astype(mx.float32)-v.astype(mx.float32))**2).item()) for k,v in before_weights.items()))
    if not finite or changed == 0 or not math.isfinite(delta_l2) or delta_l2 <= 0:
        raise RuntimeError("Adapter parameter-change evidence invalid")
    receipt.update({"changed_tensors": changed, "adapter_tensor_count": len(after_weights),
                    "parameters_finite": finite, "parameter_delta_l2": delta_l2})
    del before_weights, after_weights, optimizer, loss_and_grad, grad_sum, state_rng, model
    gc.collect(); mx.clear_cache()
    gate.before_unit("fresh_checkpoint_reload", loading=True)
    reloaded, reload_processor = load(str(model_path), lazy=True)
    reloaded = prepare_model_for_training(reloaded, train_type="lora", lora_rank=4,
        lora_alpha=8, quantization_bits=4, quantization_group_size=64, checkpoint_path=after, verbose=False)
    mx.eval(reloaded.parameters()); gc.collect(); mx.clear_cache()
    gate.before_unit("reloaded_quantized_model_ready")
    reloaded.eval()
    disk=mx.load(str(after)); loaded=dict(tree_flatten(reloaded.trainable_parameters()))
    if set(disk)!=set(loaded) or not all(bool(mx.array_equal(v,loaded[k]).item()) for k,v in disk.items()):
        raise RuntimeError("Fresh SFT checkpoint reload mismatch")
    response=generate(reloaded,reload_processor,prompt,image=[image],max_tokens=256,temperature=0,verbose=False)
    if not response.text.strip():
        raise RuntimeError("Fresh SFT reload generated empty response")
    _atomic_json(output / "reload_inference_receipt.json", {
        "loaded_adapter_sha256": after_sha, "fresh_model_loaded": True,
        "exact_adapter_tensor_match": True, "adapter_tensor_count": len(disk),
        "actual_model_generation": True, "generated_text": response.text,
        "image_sha256": _sha256(Path(image)), "contract_id": contract_id,
        "record_id": row["id"], "rendered_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "max_tokens": 256, "temperature": 0})
    receipt.update({"status": "SFT_ENGINEERING_CONTROL", "checkpoint_reloaded": True,
                    "reload_inference_receipt_sha256": _sha256(output / "reload_inference_receipt.json")})
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
