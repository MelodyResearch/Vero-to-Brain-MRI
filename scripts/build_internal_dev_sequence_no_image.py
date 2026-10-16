"""Create a private, question-identical no-image sequence control."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def build(source: Path, output: Path) -> dict:
    if not output.resolve().is_relative_to(Path(__file__).resolve().parents[1] / "local_data"):
        raise ValueError("Patient-linked control must stay under ignored local_data")
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    controls = []
    for row in rows:
        if row["category"] != "mri_sequence":
            continue
        prompt = json.loads(json.dumps(row["prompt"]))
        image_parts = [part for message in prompt if isinstance(message["content"], list)
                       for part in message["content"] if part.get("type") == "image"]
        if len(image_parts) != 1 or image_parts[0]["image"] != row["image"]:
            raise ValueError("Expected one matching image")
        for message in prompt:
            if isinstance(message["content"], list):
                message["content"] = [part for part in message["content"]
                                      if part.get("type") != "image"]
        controls.append({"record_id": row["record_id"] + "__no_image",
                         "category": row["category"], "image": None,
                         "prompt": prompt})
    if len(controls) != 24:
        raise ValueError("Expected the fixed 24 sequence questions")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in controls))
    return {"rows": len(controls), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output), indent=2))
