"""Private MSD MRI records to the pinned Vero veRL training schema.

The split made here is scan-disjoint only. MSD Task04 does not expose stable
person identifiers, so this is an engineering training substrate, not a
patient-independent scientific benchmark.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


TASK_ROUTES = {
    "central_slice_hippocampal_centroid": ("mri_centroid_2d", "spatial_localization"),
    "central_slice_anterior_posterior_dominance": ("mri_dominance", "anatomical_comparison"),
}
TASK01_ROUTES = {
    "tumor_enhancing_presence": ("mri_enhancing_presence", "tumor_enhancing_presence"),
    "tumor_grid": ("mri_tumor_grid", "tumor_spatial_grid"),
}
_GRID_LABELS = {f"{x}-{y}" for x in ("left", "center", "right")
                for y in ("top", "middle", "bottom")}


def _scan_key(row: dict) -> str:
    match = re.fullmatch(r"scan_(\d+)_central\.png", Path(row["input_image"]).name)
    if match is None:
        raise ValueError("Expected the fixed MSD scan render name")
    return match.group(1)


def convert_record(row: dict, ordinal: int) -> dict:
    task = row["task_id"]
    if task not in TASK_ROUTES:
        raise ValueError(f"Unsupported MRI task: {task}")
    if row.get("split_status") != "UNVERIFIED_PATIENT_IDENTITY_NO_HELD_OUT_CLAIM":
        raise ValueError("Unexpected lineage status")
    image_path = Path(row["input_image"])
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    route, category = TASK_ROUTES[task]
    answer = row["answer"]
    if route == "mri_centroid_2d":
        if not isinstance(answer, list) or len(answer) != 2 or any(not isinstance(x, int) for x in answer):
            raise ValueError("Centroid requires two integer voxel coordinates")
        truth = f"{answer[0]},{answer[1]}"
        instruction = " Give the final two native voxel coordinates as \\boxed{i,j}."
    else:
        if answer not in {"anterior", "posterior"}:
            raise ValueError("Dominance label must be anterior or posterior")
        truth = answer
        instruction = " Give the final one-word answer as \\boxed{anterior} or \\boxed{posterior}."
    question = row["prompt"] + instruction
    return {
        "id": f"msd04_{_scan_key(row)}_{task}",
        "data_source": "MSD_Task04_Hippocampus",
        "ability": category,
        "prompt": [{"role": "user", "content": [
            {"type": "image", "image": str(image_path)},
            {"type": "text", "text": question},
        ]}],
        "images": [str(image_path)],
        "reward_model": {"style": "rule", "ground_truth": truth},
        "extra_info": {
            "index": ordinal,
            "question": question,
            "answer": truth,
            "reward_type": route,
            "task_type": category,
            "scan_key": _scan_key(row),
            "patient_lineage": "UNVERIFIED",
            "target_provenance": "official MSD Task04 label volume, fixed central slice",
        },
    }


def convert_task01_record(row: dict, ordinal: int) -> dict:
    """Convert a verified fixed-midpoint, four-sequence image/mask task."""
    category = row["category"]
    if category not in TASK01_ROUTES:
        raise ValueError(f"Unsupported Task01 category: {category}")
    image_path = Path(row["image"])
    if not image_path.is_file() or not row["record_id"].startswith(image_path.stem):
        raise ValueError("Task01 image or record linkage mismatch")
    route, domain = TASK01_ROUTES[category]
    answer = row["gold"]
    if route == "mri_enhancing_presence" and answer not in {"yes", "no"}:
        raise ValueError("Invalid enhancing-presence target")
    if route == "mri_tumor_grid" and answer not in _GRID_LABELS:
        raise ValueError("Invalid tumor grid target")
    question = row["prompt"] + " Put only the final answer in \\boxed{...}."
    return {
        "id": row["record_id"], "data_source": "MSD_Task01_BrainTumour",
        "ability": domain,
        "prompt": [{"role": "user", "content": [
            {"type": "image", "image": str(image_path)},
            {"type": "text", "text": question},
        ]}],
        "images": [str(image_path)],
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {
            "index": ordinal, "question": question, "answer": answer,
            "reward_type": route, "task_type": domain,
            "scan_key": image_path.stem, "patient_lineage": "UNVERIFIED",
            "target_provenance": "official MSD Task01 segmentation mask, fixed z=77 image-only render",
        },
    }


def convert_jsonl(source: Path, destination: Path, holdout_modulus: int = 5,
                  task01_source: Path | None = None) -> dict:
    """Write private train/dev JSONL; modulo split is engineering-only."""
    if holdout_modulus < 2:
        raise ValueError("holdout_modulus must be >=2")
    destination.mkdir(parents=True, exist_ok=True)
    seen: set[tuple[str, str]] = set()
    rows = {"train": [], "dev_scan_disjoint_only": []}
    counts = Counter()
    for ordinal, line in enumerate(source.read_text().splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        converted = convert_record(row, ordinal)
        key = (converted["data_source"], converted["extra_info"]["scan_key"], converted["ability"])
        if key in seen:
            raise ValueError(f"Duplicate scan/category pair: {key}")
        seen.add(key)
        split = "dev_scan_disjoint_only" if int(key[1]) % holdout_modulus == 0 else "train"
        rows[split].append(converted)
        counts[f"{split}/{converted['ability']}"] += 1
    if task01_source is not None:
        raw_task01 = [json.loads(line) for line in task01_source.read_text().splitlines() if line.strip()]
        scan_keys = sorted({Path(row["image"]).stem for row in raw_task01})
        if len(scan_keys) < 4:
            raise ValueError("Task01 engineering split needs at least four source scans")
        dev_keys = set(scan_keys[:max(1, len(scan_keys) // 4)])
        for ordinal, row in enumerate(raw_task01):
            converted = convert_task01_record(row, ordinal)
            key = (converted["data_source"], converted["extra_info"]["scan_key"], converted["ability"])
            if key in seen:
                raise ValueError(f"Duplicate scan/category pair: {key}")
            seen.add(key)
            split = "dev_scan_disjoint_only" if key[1] in dev_keys else "train"
            rows[split].append(converted)
            counts[f"{split}/{converted['ability']}"] += 1
    if not rows["train"] or not rows["dev_scan_disjoint_only"]:
        raise ValueError("Need nonempty train and engineering dev splits")
    for split, values in rows.items():
        (destination / f"{split}.verl.jsonl").write_text(
            "".join(json.dumps(v, sort_keys=True) + "\n" for v in values)
        )
    return {"source_rows": len(seen), "train_rows": len(rows["train"]),
            "dev_rows": len(rows["dev_scan_disjoint_only"]),
            "category_counts": dict(sorted(counts.items())),
            "patient_disjoint": False, "scientific_evaluation_eligible": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--task01-source", type=Path)
    args = parser.parse_args()
    print(json.dumps(convert_jsonl(args.source, args.destination,
                                   task01_source=args.task01_source), indent=2))


if __name__ == "__main__":
    main()
