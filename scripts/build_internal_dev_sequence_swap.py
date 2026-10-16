"""Build private, same-patient/slice T1ce↔FLAIR counterfactual prompts.

The source manifest already pairs both sequences at each patient/slice. Only
the image changes; the question and system message stay byte-identical.
Outputs must remain under ignored local_data/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def build(source: Path, output: Path) -> dict:
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    sequence = {r["record_id"]: r for r in rows if r["category"] == "mri_sequence"}
    if not sequence or len(sequence) % 2:
        raise ValueError("Need paired sequence rows")
    swapped = []
    for record_id, row in sequence.items():
        if record_id.endswith("_flair_sequence"):
            other_id = record_id.removesuffix("_flair_sequence") + "_t1ce_sequence"
        elif record_id.endswith("_t1ce_sequence"):
            other_id = record_id.removesuffix("_t1ce_sequence") + "_flair_sequence"
        else:
            raise ValueError(f"Unexpected sequence record ID: {record_id}")
        other = sequence.get(other_id)
        if other is None or not Path(other["image"]).is_file():
            raise ValueError(f"Missing same-patient/slice counterpart for {record_id}")
        prompt = json.loads(json.dumps(row["prompt"]))
        images = [part for message in prompt for part in message["content"]
                  if isinstance(message["content"], list) and part.get("type") == "image"]
        if len(images) != 1 or images[0]["image"] != row["image"]:
            raise ValueError("Unexpected image placeholder")
        images[0]["image"] = other["image"]
        swapped.append({"record_id": record_id + "__same_slice_sequence_swap",
                        "category": row["category"], "image": other["image"],
                        "prompt": prompt})
    if not output.resolve().is_relative_to(Path(__file__).resolve().parents[1] / "local_data"):
        raise ValueError("Counterfactual patient-linked manifest must stay private")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in swapped))
    return {"rows": len(swapped), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output), indent=2))
