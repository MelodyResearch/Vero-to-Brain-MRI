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

from mri_rl.mlx_gspo_smoke import _sha256


CATEGORIES = ("mri_sequence", "mri_tumor_presence",
              "mri_enhancing_presence", "mri_tumor_grid")


def _schedule_train_rows(path: Path, steps: int, seed: int) -> list[list[dict]]:
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
        ordered[category] = [min(patients[key], key=lambda row: row["id"])
                             for key in patient_keys[:steps]]
    return [[ordered[category][step] for category in CATEGORIES]
            for step in range(steps)]


def _selected_train_rows(path: Path) -> list[dict]:
    """Backward-compatible one-step selection for CPU validation."""
    return _schedule_train_rows(path, steps=1, seed=7)[0]


def run(model_path: Path, records: Path, output: Path, seed: int = 7,
        steps: int = 1) -> dict:
    private_root = Path(__file__).resolve().parents[2] / "local_data"
    if not output.resolve().is_relative_to(private_root):
        raise ValueError("MRI adapters and receipts must stay under ignored local_data/")
    if not model_path.is_dir():
        raise FileNotFoundError("Local base checkpoint missing")
    schedule = _schedule_train_rows(records, steps, seed)
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

    output.mkdir(parents=True, exist_ok=True)
    mx.random.seed(seed)
    model, processor = load(str(model_path))
    model = prepare_model_for_training(model, train_type="lora", lora_rank=4,
                                       lora_alpha=8, quantization_bits=4,
                                       quantization_group_size=64, verbose=False)
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
    losses = []
    examples = []
    gradient_norms = []
    for step_index, chosen in enumerate(schedule):
        grad_sum = None
        for row in chosen:
            image = row["images"][0]
            question = row["extra_info"]["question"]
            messages = [
                {"role": "system", "content": "Respond with one short observation in <think> and one boxed final answer in <answer>."},
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
                             "target_tokens": int(target_ids.size)})
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
    save_trainable_weights(model, after)
    after_sha = _sha256(after)
    if before_sha == after_sha:
        raise RuntimeError("SFT optimizer step did not change adapter weights")
    receipt = {
        "status": "SFT_ENGINEERING_CONTROL", "model_path_name": model_path.name,
        "source": "OpenNeuro_ds007045", "internal_train_only": True,
        "heldout_test_used": False, "categories": list(CATEGORIES),
        "examples": examples, "loss_terms": losses, "gradient_l2_by_step": gradient_norms,
        "optimizer_steps": steps, "learning_rate": 1e-5,
        "distinct_patients_exposed": len({example["patient_key"] for example in examples}),
        "before_adapter_sha256": before_sha, "after_adapter_sha256": after_sha,
        "trainable_adapter_format": "MLX safetensors 4-bit base LoRA rank 4",
        "science_claim": False,
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2))
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(run(args.model.resolve(), args.records.resolve(),
                         args.output.resolve(), args.seed, args.steps), indent=2))


if __name__ == "__main__":
    main()
