"""CPU-only source-bound VQA-RAD task preparation; never authorizes training.

The clinical endpoint is the original evaluated QA, not a diagnosis by this
module. Components are conservative case proxies, not verified patient IDs.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
from pathlib import Path
import re

from .vero_mri_reward import _answer_and_format


def _case_components(rows):
    parent = {}

    def find(key):
        parent.setdefault(key, key)
        if parent[key] != key:
            parent[key] = find(parent[key])
        return parent[key]

    for row in rows:
        image = row.get("image_name")
        case = row.get("image_case_url")
        if image and case and case.startswith("https://medpix.nlm.nih.gov/case?id="):
            a, b = find("image:" + image), find("case:" + case)
            parent[a] = b
    return {row["image_name"]: find("image:" + row["image_name"])
            for row in rows if "image:" + row.get("image_name", "") in parent}


def build_manifest(rows, selection, image_dir, official_sha256):
    """Bind a preselected, reviewed endpoint and balanced source-case split.

    No image diagnosis, patient-independence certification, split optimization,
    download, model call, reward redesign or optimizer work happens here.
    `official_sha256` must come from the separately authenticated OSF index.
    """
    by_qid = {}
    for row in rows:
        key = str(row["qid"])
        if key in by_qid:
            raise ValueError("Duplicate source qid")
        by_qid[key] = row
    components = _case_components(rows)
    seen_components = set()
    seen_payloads = set()
    selected_qids = set()
    records = []
    counts = defaultdict(Counter)
    image_dir = Path(image_dir).resolve()
    for item in selection:
        qid = str(item["qid"])
        if qid in selected_qids:
            raise ValueError("Repeated selected qid")
        selected_qids.add(qid)
        row = by_qid[qid]
        answer = row["answer"].strip().lower()
        question = row["question"]
        if (row.get("image_organ") != "HEAD"
                or row.get("evaluation") != "evaluated"
                or row.get("answer_type", "").strip() != "CLOSED"
                or "PRES" not in row.get("question_type", "").split(",")
                or answer not in {"yes", "no"}
                or not re.search(r"\bmid(?:line|light)\b.*\bshift(?:ed|ing)?\b|\bshift(?:ed|ing)?\b.*\bmidline\b", question, re.I)):
            raise ValueError("Not a reviewed original head midline-shift yes/no QA")
        name = row["image_name"]
        component = components.get(name)
        if component is None or component in seen_components:
            raise ValueError("Missing or reused case component (including paraphrases)")
        seen_components.add(component)
        path = (image_dir / name).resolve()
        if path.parent != image_dir:
            raise ValueError("Image path must remain in bound directory")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != official_sha256[name]:
            raise ValueError("Image SHA differs from official OSF payload")
        if digest in seen_payloads:
            raise ValueError("Duplicate image payload")
        seen_payloads.add(digest)
        split = item["split"]
        if split not in {"train", "eval", "reserve"}:
            raise ValueError("Unknown split")
        counts[split][answer] += 1
        records.append({"source_qid": qid, "split": split, "question": question,
                        "ground_truth": answer, "source_image_sha256": digest,
                        "case_proxy_sha256": hashlib.sha256(component.encode()).hexdigest(),
                        "model_input": {"image": str(path), "question": question},
                        "clinical_authority": "original_VQA_RAD_evaluated_QA",
                        "technical_mri_check": "EXTERNAL_REVIEW_REQUIRED",
                        "global_patient_registry_check": "NOT_VERIFIED"})
    for split in ("train", "eval"):
        if not counts[split]["yes"] or counts[split]["yes"] != counts[split]["no"]:
            raise ValueError("TRAIN and EVAL must each be nonempty and label-balanced")
    return {"schema": "vqarad_midline_source_candidate_v1", "rows": records,
            "counts": {key: dict(value) for key, value in counts.items()},
            "split_unit": "conservative_source_case_component",
            "patient_independence": "SOURCE_PAPER_ASSERTION_NOT_REGISTRY_VERIFIED",
            "gpu_authorized": False, "training_authorized": False}


def score_response(response, target):
    """Reuse Vero's existing parser and 0.8/0.2 task/format components.

    Future strict greedy accuracy requires format=1 and accuracy=1; a malformed
    answer never counts as a strict correct response. Existing group gate is
    not changed or activated by this preparation-only callback.
    """
    if target not in {"yes", "no"}:
        raise ValueError("Expected yes/no source target")
    answer, fmt = _answer_and_format(response)
    accuracy = float(fmt > 0 and answer.strip().lower() == target)
    invalid = fmt != 1 or answer.strip().lower() not in {"yes", "no"}
    return {"score": .8 * accuracy + .2 * fmt, "accuracy": accuracy,
            "format": fmt, "invalid": invalid,
            "strict_accuracy": float(not invalid and accuracy == 1)}
