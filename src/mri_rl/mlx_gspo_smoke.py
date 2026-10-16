"""One real-image, one-group MLX LoRA GSPO weight-update feasibility run.

This is an independent Apple Metal port of the pinned Vero policy-loss formula,
not the official CUDA veRL trainer or a scientific MRI experiment. It is kept
small and fail-closed: a zero-variance reward group produces no update.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path

from mri_rl.vero_mri_reward import compute_score_from_data_source


VERO_SHA = "025ca3c800fc95224c86323c1f7fd33dceed3090"
ROUTE_SOURCES = {
    "mri_centroid_2d": {"MSD_Task04_Hippocampus"},
    "mri_dominance": {"MSD_Task04_Hippocampus"},
    "mri_enhancing_presence": {"MSD_Task01_BrainTumour", "OpenNeuro_ds007045"},
    "mri_tumor_grid": {"MSD_Task01_BrainTumour", "OpenNeuro_ds007045"},
    "mri_sequence": {"OpenNeuro_ds007045"},
    "mri_tumor_presence": {"OpenNeuro_ds007045"},
}


def advantages_from_rewards(rewards: list[float], epsilon: float = 1e-6) -> list[float]:
    """Match the official veRL GRPO sample standard deviation convention."""
    if len(rewards) < 2 or any(not math.isfinite(x) for x in rewards):
        raise ValueError("Need at least two finite rewards")
    mean = sum(rewards) / len(rewards)
    std = math.sqrt(sum((x - mean) ** 2 for x in rewards) / (len(rewards) - 1))
    return [(x - mean) / (std + epsilon) for x in rewards]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _records_for_route(path: Path, reward_type: str, limit: int,
                       diverse_labels: bool = False) -> list[dict]:
    if reward_type not in ROUTE_SOURCES:
        raise ValueError("Unsupported MRI reward route")
    matches = []
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("extra_info", {}).get("reward_type") == reward_type:
                if row.get("data_source") not in ROUTE_SOURCES[reward_type]:
                    raise ValueError("Reward route/source mismatch")
                if not Path(row["images"][0]).is_file():
                    raise FileNotFoundError("MRI image missing")
                expected_lineage = ("BIDS_SUBJECT_VERIFIED" if row["data_source"] == "OpenNeuro_ds007045"
                                    else "UNVERIFIED")
                if row["extra_info"].get("patient_lineage") != expected_lineage:
                    raise ValueError("Unexpected lineage status")
                if (row["data_source"] == "OpenNeuro_ds007045" and
                        row["extra_info"].get("internal_split") != "train"):
                    raise ValueError("Heldout or internal-dev record is not train eligible")
                matches.append(row)
                if len(matches) == limit and not diverse_labels:
                    return matches
    if not matches:
        raise ValueError(f"No record for route {reward_type}")
    if diverse_labels:
        labels = list(dict.fromkeys(row["reward_model"]["ground_truth"] for row in matches))
        chosen = [next(row for row in matches if row["reward_model"]["ground_truth"] == label)
                  for label in labels[:limit]]
        chosen_ids = {row["id"] for row in chosen}
        chosen.extend(row for row in matches if row["id"] not in chosen_ids)
        return chosen[:limit]
    return matches[:limit]


def run(model_path: Path, records: Path, output: Path, route: str, group_size: int,
        max_tokens: int, seed: int, max_groups: int = 3,
        template_override_path: Path | None = None,
        concise_prompt: bool = False, diverse_labels: bool = False,
        adapter_checkpoint: Path | None = None) -> dict:
    private_root = Path(__file__).resolve().parents[2] / "local_data"
    if not output.resolve().is_relative_to(private_root):
        raise ValueError("Raw MRI rollouts and adapter weights must stay under ignored local_data/")
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_map
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.trainer.common.model import prepare_model_for_training
    from mlx_vlm.trainer.common.utils import save_trainable_weights
    from mlx_vlm.utils import prepare_inputs, should_add_special_tokens

    if group_size < 2 or group_size > 8 or max_tokens < 16 or max_tokens > 512 or not 1 <= max_groups <= 3:
        raise ValueError("Bounded smoke requires group 2–8, response 16–512, and 1–3 groups")
    if not model_path.is_dir():
        raise FileNotFoundError("Local Qwen3-VL checkpoint missing")
    if adapter_checkpoint is not None and not adapter_checkpoint.is_file():
        raise FileNotFoundError("Starting LoRA adapter checkpoint missing")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    candidate_records = _records_for_route(records, route, max_groups, diverse_labels)
    template_override = None
    if template_override_path is not None:
        raw_template = template_override_path.read_text()
        template_override = (json.loads(raw_template)["chat_template"]
                             if template_override_path.suffix == ".json" else raw_template)
    output.mkdir(parents=True, exist_ok=True)
    mx.random.seed(seed)
    model, processor = load(str(model_path))
    model = prepare_model_for_training(model, train_type="lora", lora_rank=4,
                                       lora_alpha=8, quantization_bits=4,
                                       quantization_group_size=64,
                                       checkpoint_path=adapter_checkpoint, verbose=False)
    gc.collect()
    mx.clear_cache()
    model.eval()
    def logps_for(response_ids):
        ids = mx.concatenate((prompt_ids, mx.array(response_ids, dtype=prompt_ids.dtype)[None, :]), axis=1)
        mask = mx.ones_like(ids[:, :-1])
        logits = model(ids[:, :-1], pixel_values, mask, **model_kwargs).logits.astype(mx.float32)
        labels = ids[:, 1:]
        if logits.shape[1] != labels.shape[1]:
            raise ValueError("Model logits and labels differ in sequence length")
        selected = mx.take_along_axis(nn.log_softmax(logits, axis=-1), labels[..., None], axis=-1)[..., 0]
        return selected[0, prompt_length - 1:]

    attempts = []
    all_raw = []
    candidates = []
    advantages = []
    selected_group = None
    for group_index, row in enumerate(candidate_records):
        image = row["images"][0]
        question = row["extra_info"]["question"]
        system_text = ("Use exactly this short format and no preamble: "
                       "<think>one brief observation</think><answer>\\boxed{final answer}</answer>."
                       if concise_prompt else
                       "Respond using <think>brief visual reasoning</think><answer>\\boxed{final answer}</answer>.")
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": question},
        ]
        prompt = apply_chat_template(processor, model.config, messages,
                                     add_generation_prompt=True, num_images=1,
                                     **({"chat_template": template_override,
                                         "enable_thinking": False}
                                        if template_override is not None else {}))
        if prompt.count("<|image_pad|>") != 1:
            raise ValueError("Expected exactly one Qwen3-VL image placeholder")
        inputs = prepare_inputs(
            processor, images=[image], prompts=prompt,
            image_token_index=getattr(model.config, "image_token_index", None),
            add_special_tokens=should_add_special_tokens(model.config.model_type, processor),
        )
        prompt_ids = inputs["input_ids"]
        prompt_length = int(prompt_ids.shape[-1])
        if prompt_length > 2048:
            raise ValueError("Prompt image uses >2048 tokens; smoke budget exceeded")
        pixel_values = inputs.get("pixel_values")
        model_kwargs = {key: value for key, value in inputs.items()
                        if key not in {"input_ids", "attention_mask", "pixel_values"}}
        candidates = []
        for _ in range(group_size):
            result = generate(model, processor, prompt, image=[image], max_tokens=max_tokens,
                              temperature=1.0, top_p=1.0, verbose=False)
            tokens = list(result.token_ids or [])
            if not tokens or result.prompt_tokens != prompt_length:
                raise ValueError("Rollout tokens and log-probability prompt do not align")
            grade = compute_score_from_data_source(
                data_source=row["data_source"], solution_str=result.text,
                ground_truth=row["reward_model"]["ground_truth"],
                extra_info=row["extra_info"], format_score=0.2,
            )
            # Small-response scaling of the official last-buffer overlong ramp.
            buffer = max_tokens // 2
            penalty = min(-(len(tokens) - (max_tokens - buffer)) / buffer, 0.0)
            grade["overlong"] = penalty
            grade["total"] = grade["score"] + penalty
            candidates.append({"tokens": tokens, "text": result.text, "reward": grade})
            mx.clear_cache()
        rewards = [item["reward"]["total"] for item in candidates]
        advantages = advantages_from_rewards(rewards)
        active = max(advantages) - min(advantages) >= 1e-12
        attempts.append({
            "group_index": group_index, "source": row["data_source"],
            "route": route, "prompt_tokens": prompt_length,
            "response_tokens": [len(c["tokens"]) for c in candidates],
            "rewards": [c["reward"] for c in candidates],
            "advantages": advantages, "active": active,
        })
        all_raw.append({"group_index": group_index, "record_id": row["id"],
                        "rollouts": [{"text": c["text"], "token_ids": c["tokens"]}
                                     for c in candidates]})
        if active:
            for item in candidates:
                item["old_logps"] = mx.stop_gradient(logps_for(item["tokens"]))
                mx.eval(item["old_logps"])
                mx.clear_cache()
            selected_group = group_index
            break
    receipt = {
        "status": "NO_UPDATE_ZERO_VARIANCE", "official_source_sha": VERO_SHA,
        "implementation": "MLX independent LoRA port of pinned GSPO-token loss; not official veRL",
        "model_path_name": model_path.name, "image_source": row["data_source"],
        "reward_type": route, "group_size": group_size,
        "template_override_sha256": (_sha256(template_override_path)
                                     if template_override_path else None),
        "starting_adapter_sha256": (_sha256(adapter_checkpoint)
                                    if adapter_checkpoint else None),
        "concise_prompt": concise_prompt, "diverse_labels": diverse_labels,
        "candidate_label_count": len({r["reward_model"]["ground_truth"]
                                       for r in candidate_records}),
        "max_groups": max_groups, "groups_attempted": len(attempts),
        "selected_active_group": selected_group, "attempts": attempts,
        "policy_updated": False, "patient_disjoint_evaluation": False,
        "science_claim": False,
    }
    # Raw generated responses are private and must never be committed.
    (output / "raw_rollouts.json").write_text(json.dumps(all_raw, indent=2))
    if selected_group is None:
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2))
        return receipt

    before = output / "before_adapters.safetensors"
    after = output / "after_adapters.safetensors"
    save_trainable_weights(model, before)
    before_sha = _sha256(before)
    model.train()

    def one_loss(model, tokens, old_logps, advantage):
        fresh = logps_for(tokens)
        if fresh.shape != old_logps.shape:
            raise ValueError("Response log-probability shape mismatch")
        seq_log_ratio = mx.mean(fresh - old_logps)
        # Exact GSPO-token stop-gradient placement from pinned veRL core_algos.
        log_ratio = fresh - mx.stop_gradient(fresh) + mx.stop_gradient(seq_log_ratio)
        ratio = mx.exp(mx.minimum(log_ratio, 10.0))
        clipped = mx.clip(ratio, 1 - 0.0003, 1 + 0.0004)
        loss_tokens = mx.maximum(-advantage * ratio, -advantage * clipped)
        return mx.mean(loss_tokens)

    loss_and_grad = nn.value_and_grad(model, one_loss)
    grad_sum = None
    losses = []
    for item, advantage in zip(candidates, advantages):
        loss, grad = loss_and_grad(model, item["tokens"], item["old_logps"], advantage)
        mx.eval(loss, grad)
        losses.append(float(loss.item()))
        scaled = tree_map(lambda x: x / group_size, grad)
        grad_sum = scaled if grad_sum is None else tree_map(lambda a, b: a + b, grad_sum, scaled)
        mx.eval(grad_sum)
        mx.clear_cache()
    grad_norm = math.sqrt(sum(float(mx.sum(g.astype(mx.float32) ** 2).item())
                              for _, g in tree_flatten(grad_sum)))
    if not math.isfinite(grad_norm) or grad_norm <= 0:
        raise RuntimeError("Active group produced no finite nonzero gradient")
    optimizer = optim.Adam(learning_rate=1e-5)
    optimizer.update(model, grad_sum)
    mx.eval(model.trainable_parameters(), optimizer.state)
    save_trainable_weights(model, after)
    after_sha = _sha256(after)
    if before_sha == after_sha:
        raise RuntimeError("Optimizer step did not change adapter weights")
    receipt.update({
        "status": "TRAINING_COMPLETED_ENGINEERING_SMOKE", "policy_updated": True,
        "loss_terms": losses, "gradient_l2": grad_norm,
        "before_adapter_sha256": before_sha, "after_adapter_sha256": after_sha,
        "optimizer_steps": 1, "trainable_adapter_format": "MLX safetensors 4-bit base LoRA rank 4",
    })
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2))
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--route", choices=tuple(ROUTE_SOURCES),
                        default="mri_dominance")
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--max-groups", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--template-override", type=Path)
    parser.add_argument("--concise-prompt", action="store_true")
    parser.add_argument("--diverse-labels", action="store_true")
    parser.add_argument("--adapter-checkpoint", type=Path)
    args = parser.parse_args()
    result = run(args.model.resolve(), args.records.resolve(), args.output.resolve(),
                 args.route, args.group_size, args.max_tokens, args.seed, args.max_groups,
                 args.template_override.resolve() if args.template_override else None,
                 args.concise_prompt, args.diverse_labels,
                 args.adapter_checkpoint.resolve() if args.adapter_checkpoint else None)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {"attempts", "loss_terms"}}, indent=2))


if __name__ == "__main__":
    main()
