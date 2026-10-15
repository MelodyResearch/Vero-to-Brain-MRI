"""Image-linked task records, deterministic verification, and CPU evaluation.

Coordinates are pixel-edge xyxy boxes in the source frame. This module does
not interpret 2D boxes as volumetric masks or infer patient IDs from filenames.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageOps


def _box(value):
    try:
        x1, y1, x2, y2 = map(float, value)
        if not all(map(math.isfinite, (x1, y1, x2, y2))) or x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2
    except (TypeError, ValueError):
        return None


def frame_iou(prediction: dict, target: dict) -> float:
    """Frame-aware 2D IoU; mismatched/invalid frame and boxes score zero."""
    if prediction.get("frame") != target.get("frame"):
        return 0.0
    a, b = _box(prediction.get("box")), _box(target.get("box"))
    if a is None or b is None:
        return 0.0
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union


def reward(task: str, prediction: str, target: dict) -> float:
    """Return a [0,1] deterministic reward for currently eligible tasks."""
    if task == "categorical":
        return float(prediction.strip().casefold() == str(target["label"]).strip().casefold())
    if task == "longitudinal_direction":
        return float(prediction.strip().casefold() == str(target["direction"]).strip().casefold())
    if task == "numeric":
        try:
            value = float(prediction.strip())
            reference = float(target["value"])
            tolerance = float(target["absolute_tolerance"])
            return float(math.isfinite(value) and tolerance >= 0 and abs(value - reference) <= tolerance)
        except (ValueError, TypeError):
            return 0.0
    if task == "spatial":
        try:
            output = json.loads(prediction)
            return frame_iou(output, target)
        except (ValueError, TypeError, AttributeError):
            return 0.0
    raise ValueError(f"No validated verifier for task {task!r}")


def _valid_image(frame: dict) -> bool:
    try:
        with Image.open(frame["path"]) as image:
            image.verify()
        return True
    except (OSError, KeyError, TypeError, ValueError):
        return False


def _valid_spatial_target(row: dict) -> bool:
    target = row["target"]
    box = _box(target.get("box"))
    frames = [f for f in row["frames"] if f.get("index") == target.get("frame")]
    if box is None or len(frames) != 1:
        return False
    try:
        with Image.open(frames[0]["path"]) as image:
            width, height = image.size
        return 0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height
    except (OSError, KeyError, TypeError, ValueError):
        return False


def audit_records(splits: dict[str, list[dict]]) -> dict:
    """Fail closed on unknown lineage, cross-split subject overlap or images."""
    subjects = {name: {r.get("subject_id") for r in rows if r.get("subject_id")} for name, rows in splits.items()}
    overlap = set()
    names = list(subjects)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap |= subjects[left] & subjects[right]
    decoded = 0
    usable = 0
    reasons = Counter()
    seen_ids = set()
    for rows in splits.values():
        for row in rows:
            missing = [key for key in ("id", "subject_id", "study_id", "task", "target") if not row.get(key)]
            frames = row.get("frames") or []
            good_frames = sum(_valid_image(f) for f in frames)
            decoded += good_frames
            if missing:
                reasons["missing_lineage_or_task"] += 1
            elif row["id"] in seen_ids:
                reasons["duplicate_id"] += 1
            elif not frames or good_frames != len(frames):
                reasons["missing_or_undecodable_image"] += 1
            elif row["task"] == "spatial" and not _valid_spatial_target(row):
                reasons["invalid_spatial_target"] += 1
            elif row["subject_id"] in overlap:
                reasons["subject_split_overlap"] += 1
            else:
                usable += 1
            if row.get("id"):
                seen_ids.add(row["id"])
    return {"records": sum(map(len, splits.values())), "unique_subjects": len(set().union(*subjects.values())),
            "decodable_frames": decoded, "usable_records": usable,
            "subject_overlap": sorted(overlap), "exclusion_reasons": dict(reasons)}


def render_frames(frames: list[dict], output: Path, tile_size: int = 256) -> dict:
    """Render actual linked 2D frames in index order without annotation-guided selection."""
    ordered = sorted(frames, key=lambda f: f["index"])
    if not ordered or len({f["index"] for f in ordered}) != len(ordered):
        raise ValueError("Frames must have unique indices")
    canvas = Image.new("RGB", (tile_size * len(ordered), tile_size), "black")
    for slot, frame in enumerate(ordered):
        with Image.open(frame["path"]) as source:
            image = ImageOps.contain(source.convert("RGB"), (tile_size, tile_size))
            canvas.paste(image, (slot * tile_size + (tile_size - image.width) // 2, (tile_size - image.height) // 2))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return {"frame_indices": [f["index"] for f in ordered], "tile_size": tile_size,
            "source_paths": [f["path"] for f in ordered], "output": str(output)}


def balanced_sample(records: list[dict], count: int, seed: int = 0) -> list[dict]:
    """Uniform task-category sampling with replacement; deterministic seed."""
    by_task = defaultdict(list)
    for record in records:
        by_task[record["task"]].append(record)
    if not by_task or count < 0:
        raise ValueError("Nonempty records and nonnegative count required")
    rng = random.Random(seed)
    tasks = sorted(by_task)
    quotas = {task: count // len(tasks) for task in tasks}
    for task in tasks[: count % len(tasks)]:
        quotas[task] += 1
    sampled = [rng.choice(by_task[task]) for task in tasks for _ in range(quotas[task])]
    rng.shuffle(sampled)
    return sampled


def evaluate(records: list[dict], predictions: dict[str, str]) -> dict:
    """Score supplied model outputs; omit unanswered rows and report coverage."""
    scores = defaultdict(list)
    subjects = set()
    missing = []
    for row in records:
        if row["id"] not in predictions:
            missing.append(row["id"])
            continue
        subjects.add(row["subject_id"])
        scores[row["task"]].append(reward(row["task"], predictions[row["id"]], row["target"]))
    task_scores = {task: {"mean": sum(values) / len(values), "n": len(values)} for task, values in sorted(scores.items())}
    return {"subjects": len(subjects), "scored": sum(len(v) for v in scores.values()), "missing_prediction_ids": missing,
            "task_scores": task_scores, "macro_mean": sum(x["mean"] for x in task_scores.values()) / len(task_scores) if task_scores else None}
