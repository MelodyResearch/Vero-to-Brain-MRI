"""Run a pinned local VLM on a private image-linked JSONL evaluation manifest.

The manifest and generated responses stay outside Git. The public aggregate
must be produced separately after checking source and patient lineage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(model_path: Path, manifest_path: Path, output_path: Path, max_tokens: int = 64,
        template_override_path: Path | None = None) -> None:
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template

    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    records = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError("empty manifest")
    for record in records:
        if (record.get("image") is not None and not Path(record["image"]).is_file()) or not record["prompt"]:
            raise ValueError("missing image or prompt")

    model, processor = load(str(model_path))
    template_override = None
    if template_override_path is not None:
        raw_template = template_override_path.read_text()
        template_override = json.loads(raw_template)["chat_template"] if template_override_path.suffix == ".json" else raw_template
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as out:
        for record in records:
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
                "model_path": str(model_path.resolve()),
                "model_config_sha256": sha256(model_path / "config.json"),
                "model_template_sha256": sha256(model_path / "chat_template.json") if (model_path / "chat_template.json").exists() else (sha256(model_path / "chat_template.jinja") if (model_path / "chat_template.jinja").exists() else None),
                "template_override_sha256": sha256(template_override_path) if template_override_path else None,
                "manifest_sha256": sha256(manifest_path),
                "prompt": record["prompt"],
                "image_placeholder_count": image_placeholders,
                "response": response.text,
                "max_tokens": max_tokens,
                "temperature": 0,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "python": platform.python_version(),
            }
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()
            print(f"completed {record['record_id']} ({record['category']})", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--template-override", type=Path)
    args = parser.parse_args()
    run(args.model, args.manifest, args.output, args.max_tokens, args.template_override)


if __name__ == "__main__":
    main()
