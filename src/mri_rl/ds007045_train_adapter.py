"""Patient-keyed ds007045 MRI prompts to pinned Vero veRL records.

The eight source-designated test patients are never read into train/dev rows.
An internal four-patient dev set is chosen from the 32 training patients by a
label-blind SHA-256 sort of anonymous patient tokens.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


SOURCE = "OpenNeuro_ds007045"
ROUTES = {
    "sequence": ("mri_sequence", "mri_sequence"),
    "tumor_presence": ("mri_tumor_presence", "mri_tumor_presence"),
    "enhancing_presence": ("mri_enhancing_presence", "mri_enhancing_presence"),
    "tumor_grid": ("mri_tumor_grid", "mri_tumor_grid"),
}
GRID_LABELS = {f"{x}-{y}" for x in ("left", "center", "right")
               for y in ("top", "middle", "bottom")}


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _valid_answer(category: str, answer: str) -> bool:
    if category == "sequence":
        return answer in {"FLAIR", "contrast-enhanced T1"}
    if category in {"tumor_presence", "enhancing_presence"}:
        return answer in {"yes", "no"}
    if category == "tumor_grid":
        return answer in GRID_LABELS
    return False


def convert_ds007045(prompt_file: Path, gold_file: Path, manifest_file: Path,
                     destination: Path, internal_dev_patients: int = 4) -> dict:
    manifest = json.loads(manifest_file.read_text())
    if manifest.get("source") != "OpenNeuro ds007045 v2.0.1":
        raise ValueError("Unexpected ds007045 source/version")
    cases = manifest["private_cases"]
    if len(cases) != 40 or manifest["train_patient_count"] != 32 or manifest["heldout_patient_count"] != 8:
        raise ValueError("Expected verified 32/8 patient split")
    by_subject = {case["private_subject"]: case for case in cases}
    if (len(by_subject) != len(cases) or
            len({c["anonymous_token"] for c in cases}) != len(cases) or
            any(c["split"] not in {"train", "test"} for c in cases)):
        raise ValueError("Duplicate patient or unexpected split")
    train_cases = [c for c in cases if c["split"] == "train"]
    test_cases = [c for c in cases if c["split"] == "test"]
    if len(train_cases) != 32 or len(test_cases) != 8 or not 1 <= internal_dev_patients <= 8:
        raise ValueError("Invalid patient split or internal dev count")
    ordered = sorted(train_cases, key=lambda c: hashlib.sha256(
        c["anonymous_token"].encode()).hexdigest())
    dev_subjects = {c["private_subject"] for c in ordered[:internal_dev_patients]}

    prompts = _jsonl(prompt_file)
    gold = _jsonl(gold_file)
    prompt_ids = [r["prompt_id"] for r in prompts]
    if len(prompt_ids) != len(set(prompt_ids)) or len(gold) != len(prompts):
        raise ValueError("Prompt/gold count or uniqueness mismatch")
    prompt_by_id = {r["prompt_id"]: r for r in prompts}
    # Test labels are deliberately not copied into any training object.
    train_gold = {r["prompt_id"]: r for r in gold if r["split"] == "train"}
    if len(train_gold) != 436 or len(prompts) != 543:
        raise ValueError("Unexpected fixed ds007045 task pack size")
    if set(train_gold) != {r["prompt_id"] for r in prompts if r["split"] == "train"}:
        raise ValueError("Train prompt/gold linkage mismatch")
    if any(r["split"] != "test" for r in prompts if r["prompt_id"] not in train_gold):
        raise ValueError("Heldout prompt split mismatch")

    rows: dict[str, list[dict]] = {"train": [], "internal_dev": []}
    counts = Counter()
    subject_counts: dict[str, set[str]] = {"train": set(), "internal_dev": set()}
    for ordinal, prompt in enumerate(prompts):
        if prompt["split"] == "test":
            continue
        gold_row = train_gold[prompt["prompt_id"]]
        if any(prompt[key] != gold_row[key] for key in ("prompt_id", "split", "category")):
            raise ValueError("Prompt/gold metadata mismatch")
        subject = gold_row["private_subject"]
        case = by_subject.get(subject)
        if case is None or case["split"] != "train" or not prompt["prompt_id"].startswith(case["anonymous_token"]):
            raise ValueError("Patient lineage or prompt linkage mismatch")
        category = prompt["category"]
        if category not in ROUTES or not _valid_answer(category, gold_row["answer"]):
            raise ValueError("Unknown route or malformed answer")
        image = Path(prompt["image_path"])
        if not image.is_file():
            raise FileNotFoundError(image)
        route, domain = ROUTES[category]
        question = prompt["question"] + " Give the final answer in \\boxed{...}."
        split = "internal_dev" if subject in dev_subjects else "train"
        row = {
            "id": prompt["prompt_id"], "data_source": SOURCE, "ability": domain,
            "prompt": [{"role": "user", "content": [
                {"type": "image", "image": str(image)},
                {"type": "text", "text": question},
            ]}],
            "images": [str(image)],
            "reward_model": {"style": "rule", "ground_truth": gold_row["answer"]},
            "extra_info": {
                "index": ordinal, "question": question, "answer": gold_row["answer"],
                "reward_type": route, "task_type": domain,
                "patient_key": case["anonymous_token"],
                "patient_lineage": "BIDS_SUBJECT_VERIFIED",
                "internal_split": split,
                "target_provenance": (
                    "OpenNeuro ds007045 source modality identity, independent of tumor mask"
                    if category == "sequence" else
                    "OpenNeuro ds007045 official tumor mask at fixed geometric slice"
                ),
            },
        }
        rows[split].append(row)
        subject_counts[split].add(subject)
        counts[f"{split}/{domain}"] += 1
    if len(subject_counts["train"]) != 32-internal_dev_patients or len(subject_counts["internal_dev"]) != internal_dev_patients:
        raise ValueError("Internal patient split is incomplete")
    if not all(any(r["ability"] == category for r in rows[split])
               for split in rows for category in (v[1] for v in ROUTES.values())):
        raise ValueError("Every category must appear in both internal splits")
    destination.mkdir(parents=True, exist_ok=True)
    for split, items in rows.items():
        (destination / f"{split}.verl.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in items))
    for category in (value[1] for value in ROUTES.values()):
        specialist = destination / "specialist" / category
        specialist.mkdir(parents=True, exist_ok=True)
        for split, items in rows.items():
            selected = [row for row in items if row["ability"] == category]
            (specialist / f"{split}.verl.jsonl").write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected))
    subject_by_token = {c["anonymous_token"]: c["private_subject"] for c in train_cases}
    category_subjects = {
        category: {subject_by_token[row["extra_info"]["patient_key"]]
                   for row in rows["train"] if row["ability"] == category}
        for category in (value[1] for value in ROUTES.values())
    }
    return {
        "source": SOURCE, "source_version": "v2.0.1", "source_train_patients": 32,
        "heldout_test_patients_untouched": 8, "internal_train_patients": 32-internal_dev_patients,
        "internal_dev_patients": internal_dev_patients,
        "internal_train_rows": len(rows["train"]),
        "internal_dev_rows": len(rows["internal_dev"]),
        "category_counts": dict(sorted(counts.items())),
        "train_patient_count_by_category": {key: len(value)
                                             for key, value in category_subjects.items()},
        "test_gold_used_in_training": False,
        "patient_disjoint_internal_dev": True,
        "patient_disjoint_heldout": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(convert_ds007045(args.prompts, args.gold, args.manifest,
                                      args.destination), indent=2))


if __name__ == "__main__":
    main()
