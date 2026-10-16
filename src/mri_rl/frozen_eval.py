"""Run a pinned local VLM on a private image-linked JSONL evaluation manifest.

The manifest and generated responses stay outside Git. The public aggregate
must be produced separately after checking source and patient lineage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

from mri_rl.resource_supervisor import ResourceGate


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adapter_metadata(adapter_checkpoint: Path | None) -> dict:
    if adapter_checkpoint is None:
        return {"adapter_checkpoint_sha256": None, "adapter_config_sha256": None}
    if not adapter_checkpoint.is_file() or adapter_checkpoint.suffix != ".safetensors":
        raise FileNotFoundError("MLX LoRA adapter checkpoint missing")
    config = adapter_checkpoint.parent / "adapter_config.json"
    if not config.is_file():
        raise FileNotFoundError("MLX LoRA adapter config missing")
    return {"adapter_checkpoint_sha256": sha256(adapter_checkpoint),
            "adapter_config_sha256": sha256(config)}


def prepare_resume(output_path: Path, records: list[dict], expected: dict) -> set[str]:
    """Keep only durable, matching predictions; reject changed model/data settings.

    A torn final JSONL line is saved to a private .partial file before repair.
    A malformed earlier line or duplicate ID fails closed.
    """
    if not output_path.exists():
        return set()
    manifest_ids = {row["record_id"] for row in records}
    if len(manifest_ids) != len(records):
        raise ValueError("Manifest contains duplicate record IDs")
    raw = output_path.read_bytes()
    lines = raw.splitlines(keepends=True)
    valid = []
    seen = set()
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index != len(lines) - 1 or line.endswith(b"\n"):
                raise ValueError("Existing output has a malformed completed line") from None
            (output_path.parent / (output_path.name + ".partial")).write_bytes(line)
            break
        record_id = row.get("record_id")
        if record_id not in manifest_ids or record_id in seen:
            raise ValueError("Existing output has unknown or duplicate record ID")
        for key, value in expected.items():
            if row.get(key) != value:
                raise ValueError(f"Resume configuration mismatch: {key}")
        source = next(item for item in records if item["record_id"] == record_id)
        if row.get("prompt") != source["prompt"]:
            raise ValueError("Resume prompt mismatch")
        image = Path(source["image"]) if source.get("image") else None
        if row.get("image_sha256") != (sha256(image) if image else None):
            raise ValueError("Resume image checksum mismatch")
        if not isinstance(row.get("response"), str):
            raise ValueError("Existing output is incomplete")
        valid.append(row)
        seen.add(record_id)
    if len(valid) != len(lines) or (raw and not raw.endswith(b"\n")):
        replacement = output_path.with_name(output_path.name + ".resume-tmp")
        with replacement.open("wb") as handle:
            for row in valid:
                handle.write((json.dumps(row, ensure_ascii=False) + "\n").encode())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(replacement, output_path)
    return seen


def run(model_path: Path, manifest_path: Path, output_path: Path, max_tokens: int = 64,
        template_override_path: Path | None = None,
        adapter_checkpoint: Path | None = None, *, resume: bool = False,
        resource_guard: bool = False) -> None:
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template

    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    adapter_receipt = adapter_metadata(adapter_checkpoint)
    records = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError("empty manifest")
    if len({row["record_id"] for row in records}) != len(records):
        raise ValueError("duplicate manifest record IDs")
    for record in records:
        if (record.get("image") is not None and not Path(record["image"]).is_file()) or not record["prompt"]:
            raise ValueError("missing image or prompt")

    expected = {
        "model_path": str(model_path.resolve()),
        "model_config_sha256": sha256(model_path / "config.json"),
        "model_template_sha256": sha256(model_path / "chat_template.json") if (model_path / "chat_template.json").exists() else (sha256(model_path / "chat_template.jinja") if (model_path / "chat_template.jinja").exists() else None),
        "template_override_sha256": sha256(template_override_path) if template_override_path else None,
        "manifest_sha256": sha256(manifest_path),
        "max_tokens": max_tokens,
        "temperature": 0,
        **adapter_receipt,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = prepare_resume(output_path, records, expected) if resume else set()
    pending = [record for record in records if record["record_id"] not in completed]
    if not pending:
        return
    gate = ResourceGate(output_path.with_suffix(".resources.jsonl")) if resource_guard else None
    if gate:
        gate.before_unit("model_load")
    model, processor = load(str(model_path))
    verified_adapter_parameters = None
    if adapter_checkpoint is not None:
        import mlx.core as mx
        from mlx.utils import tree_flatten
        from mlx_vlm.trainer.common.model import prepare_model_for_training

        model = prepare_model_for_training(
            model, train_type="lora", quantization_bits=4,
            quantization_group_size=64, checkpoint_path=adapter_checkpoint,
            verbose=False,
        )
        saved = mx.load(str(adapter_checkpoint))
        loaded = dict(tree_flatten(model.trainable_parameters()))
        if set(saved) != set(loaded) or not loaded:
            raise ValueError("Loaded LoRA keys differ from checkpoint")
        for key, value in loaded.items():
            if value.shape != saved[key].shape or not bool(mx.all(value == saved[key]).item()):
                raise ValueError("Loaded LoRA weights differ from checkpoint")
        verified_adapter_parameters = len(loaded)
        model.eval()
    template_override = None
    if template_override_path is not None:
        raw_template = template_override_path.read_text()
        template_override = json.loads(raw_template)["chat_template"] if template_override_path.suffix == ".json" else raw_template
    with output_path.open("a" if resume else "w") as out:
        for record in pending:
            if gate:
                gate.before_unit(record["record_id"])
            image_path = Path(record["image"]) if record.get("image") else None
            expected_images = 1 if image_path else 0
            prompt = apply_chat_template(
                processor, model.config, record["prompt"],
                num_images=expected_images, enable_thinking=False,
                **({"chat_template": template_override} if template_override is not None else {}),
            )
            image_placeholders = prompt.count("<|image_pad|>")
            if image_placeholders != expected_images:
                raise ValueError(f"expected {expected_images} Qwen3-VL image placeholders, got {image_placeholders}")
            started = time.monotonic()
            response = generate(
                model, processor, prompt, image=[str(image_path)] if image_path else None,
                max_tokens=max_tokens, temperature=0, verbose=False,
            )
            result = {
                "record_id": record["record_id"],
                "category": record["category"],
                "image_sha256": sha256(image_path) if image_path else None,
                **expected,
                "adapter_verified_parameter_tensors": verified_adapter_parameters,
                "prompt": record["prompt"],
                "image_placeholder_count": image_placeholders,
                "response": response.text,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "python": platform.python_version(),
            }
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()
            os.fsync(out.fileno())
            print(f"completed {record['record_id']} ({record['category']})", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--template-override", type=Path)
    parser.add_argument("--adapter-checkpoint", type=Path)
    parser.add_argument("--resume", action="store_true", help="Skip only verified complete outputs")
    parser.add_argument("--resource-guard", action="store_true", help="Cooperative macOS gate between images")
    args = parser.parse_args()
    run(args.model, args.manifest, args.output, args.max_tokens,
        args.template_override, args.adapter_checkpoint,
        resume=args.resume, resource_guard=args.resource_guard)


if __name__ == "__main__":
    main()
