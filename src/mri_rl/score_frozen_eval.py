"""Score private small MRI inference diagnostics without exposing case rows."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path


def normalize_answer(response: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", response, flags=re.I | re.S)
    if match:
        response = match.group(1)
    return response.strip().replace("\\boxed{", "").rstrip("}").strip()


def score(category: str, gold: object, response: str) -> tuple[bool, bool, float | None]:
    answer = normalize_answer(response)
    if category == "central_slice_hippocampal_centroid":
        numbers = re.findall(r"(?<!\d)-?\d+(?!\d)", answer)
        if len(numbers) != 2:
            return False, False, None
        distance = math.dist([int(x) for x in numbers], gold)
        return distance <= 3, True, distance
    if category == "central_slice_anterior_posterior_dominance":
        token = answer.lower().strip(" .")
        return token == gold, token in {"anterior", "posterior"}, None
    if category == "mri_sequence_identification":
        token = answer.upper().strip(" .")
        return token == gold.upper(), token in {"T1", "FLAIR"}, None
    if category == "tumor_enhancing_presence":
        token = answer.lower().strip(" .")
        return token == gold, token in {"yes", "no"}, None
    if category == "tumor_grid":
        token = answer.lower().strip(" .")
        cells = {f"{x}-{y}" for x in ("left", "center", "right") for y in ("top", "middle", "bottom")}
        return token == gold, token in cells, None
    raise ValueError(f"unknown diagnostic category: {category}")


def summarize(manifest: Path, outputs: Path) -> dict:
    gold = {row["record_id"]: row for row in map(json.loads, manifest.read_text().splitlines())}
    seen: set[str] = set()
    buckets: dict[str, list[tuple[bool, bool, float | None]]] = defaultdict(list)
    for prediction in map(json.loads, outputs.read_text().splitlines()):
        key = prediction["record_id"]
        if key in seen or key not in gold:
            raise ValueError("duplicate or unknown record")
        seen.add(key)
        target = gold[key]
        if target["category"] != prediction["category"]:
            raise ValueError("category mismatch")
        buckets[target["category"]].append(score(target["category"], target["gold"], prediction["response"]))
    if seen != set(gold):
        raise ValueError("incomplete predictions")
    return {
        category: {
            "n": len(values),
            "correct": sum(item[0] for item in values),
            "valid": sum(item[1] for item in values),
            "mean_centroid_distance_voxels": round(sum(item[2] for item in values if item[2] is not None) / sum(item[2] is not None for item in values), 3) if any(item[2] is not None for item in values) else None,
        }
        for category, values in sorted(buckets.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.manifest, args.outputs), indent=2))


if __name__ == "__main__":
    main()
