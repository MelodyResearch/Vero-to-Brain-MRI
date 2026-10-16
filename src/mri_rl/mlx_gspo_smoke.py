"""One real-image, one-group MLX LoRA GSPO weight-update feasibility run.

This is an independent Apple Metal port of the pinned Vero policy-loss formula,
not the official CUDA veRL trainer or a scientific MRI experiment. It is kept
small and fail-closed: a zero-variance reward group produces no update.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from mri_rl.vero_mri_reward import compute_score_from_data_source


VERO_SHA = "025ca3c800fc95224c86323c1f7fd33dceed3090"


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


def _first_record(path: Path, reward_type: str) -> dict:
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("extra_info", {}).get("reward_type") == reward_type:
                if not Path(row["images"][0]).is_file():
                    raise FileNotFoundError("MRI image missing")
                if row["extra_info"].get("patient_lineage") != "UNVERIFIED":
                    raise ValueError("Unexpected lineage status")
                return row
    raise ValueError(f"No record for route {reward_type}")


def run(model_path: Path, records: Path, output: Path, route: str, group_size: int,
        max_tokens: int, seed: int) -> dict:
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

    if group_size < 2 or group_size > 8 or max_tokens < 16 or max_tokens > 128:
        raise ValueError("Bounded smoke requires group 2–8 and response 16–128")
    if not model_path.is_dir():
        raise FileNotFoundError("Local Qwen3-VL checkpoint missing")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    row = _first_record(records, route)
    output.mkdir(parents=True, exist_ok=True)
    mx.random.seed(seed)
    model, processor = load(str(model_path))
    model = prepare_model_for_training(model, train_type="lora", lora_rank=4,
                                       lora_alpha=8, quantization_bits=4,
                                       quantization_group_size=64, verbose=False)
    model.eval()
    image = row["images"][0]
    question = row["extra_info"]["question"]
    messages = [
        {"role": "system", "content": "Respond using <think>brief visual reasoning</think><answer>\\boxed{final answer}</answer>."},
        {"role": "user", "content": question},
    ]
    prompt = apply_chat_template(processor, model.config, messages,
                                 add_generation_prompt=True, num_images=1)
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

    def logps_for(response_ids):
        ids = mx.concatenate((prompt_ids, mx.array(response_ids, dtype=prompt_ids.dtype)[None, :]), axis=1)
        mask = mx.ones_like(ids[:, :-1])
        logits = model(ids[:, :-1], pixel_values, mask, **model_kwargs).logits.astype(mx.float32)
        labels = ids[:, 1:]
        if logits.shape[1] != labels.shape[1]:
            raise ValueError("Model logits and labels differ in sequence length")
        selected = mx.take_along_axis(nn.log_softmax(logits, axis=-1), labels[..., None], axis=-1)[..., 0]
        return selected[0, prompt_length - 1:]

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
        old_logps = mx.stop_gradient(logps_for(tokens))
        mx.eval(old_logps)
        candidates.append({"tokens": tokens, "text": result.text, "reward": grade,
                           "old_logps": old_logps})
        mx.clear_cache()
    rewards = [item["reward"]["total"] for item in candidates]
    advantages = advantages_from_rewards(rewards)
    receipt = {
        "status": "NO_UPDATE_ZERO_VARIANCE", "official_source_sha": VERO_SHA,
        "implementation": "MLX independent LoRA port of pinned GSPO-token loss; not official veRL",
        "model_path_name": model_path.name, "image_source": row["data_source"],
        "reward_type": route, "group_size": group_size,
        "prompt_tokens": prompt_length, "response_tokens": [len(c["tokens"]) for c in candidates],
        "rewards": [c["reward"] for c in candidates], "advantages": advantages,
        "policy_updated": False, "patient_disjoint_evaluation": False,
        "science_claim": False,
    }
    # Raw generated responses are private and must never be committed.
    (output / "raw_rollouts.json").write_text(json.dumps(
        [{"text": c["text"], "token_ids": c["tokens"]} for c in candidates], indent=2))
    if max(advantages) - min(advantages) < 1e-12:
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
    parser.add_argument("--route", choices=("mri_centroid_2d", "mri_dominance",
                                            "mri_enhancing_presence", "mri_tumor_grid"),
                        default="mri_dominance")
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    result = run(args.model.resolve(), args.records.resolve(), args.output.resolve(),
                 args.route, args.group_size, args.max_tokens, args.seed)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {"rewards", "advantages", "loss_terms"}}, indent=2))


if __name__ == "__main__":
    main()
