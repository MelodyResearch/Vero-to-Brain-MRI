"""MRI-specific reward callback for the official Vero veRL reward manager.

Call signature and returned keys match Vero's custom_reward_function path.
The Vero manager applies its own overlong response penalty after this score.
"""

from __future__ import annotations

import math
import re


_FORMAT = re.compile(
    r"<think>(?P<think>(?:(?!<think>|</think>).)*)</think>\s*<answer>(?P<answer>.*?)</answer>\s*\Z",
    re.DOTALL,
)
_BOXED = re.compile(r"\\boxed\{([^{}]+)\}")
_COORD = re.compile(r"\s*([+-]?\d+)\s*[, ]\s*([+-]?\d+)\s*")
_GRID_LABELS = {f"{x}-{y}" for x in ("left", "center", "right")
                for y in ("top", "middle", "bottom")}


def _answer_and_format(response: str) -> tuple[str, float]:
    # Mirror pinned Vero's full-match tag gate and nonempty think/answer checks.
    if any(response.count(tag) != 1 for tag in
           ("<think>", "</think>", "<answer>", "</answer>")):
        return "", 0.0
    match = _FORMAT.fullmatch(response)
    if not match or not match.group("think").strip():
        return "", 0.0
    answer = match.group("answer").strip()
    if not answer:
        return "", 0.0
    boxes = _BOXED.findall(answer)
    if len(boxes) == 1 and boxes[0].strip():
        return boxes[0].strip(), 1.0
    if len(boxes) > 1:
        return boxes[-1].strip(), 0.5
    return answer, 0.5


def compute_score_from_data_source(
    data_source: str, solution_str: str, ground_truth: str,
    extra_info: dict | None = None, format_score: float = 0.2,
) -> dict[str, float]:
    """Return fail-closed verifiable task accuracy and Vero's 0.8/0.2 blend."""
    if data_source not in {"MSD_Task04_Hippocampus", "MSD_Task01_BrainTumour",
                           "OpenNeuro_ds007045"} or not isinstance(extra_info, dict):
        raise ValueError("Unsupported source or missing reward metadata")
    if not 0 <= format_score <= 1:
        raise ValueError("format_score outside [0,1]")
    route = extra_info.get("reward_type")
    if route in {"mri_centroid_2d", "mri_dominance"} and data_source != "MSD_Task04_Hippocampus":
        raise ValueError("Task04 reward route/source mismatch")
    if route in {"mri_enhancing_presence", "mri_tumor_grid"} and data_source not in {
            "MSD_Task01_BrainTumour", "OpenNeuro_ds007045"}:
        raise ValueError("Tumor reward route/source mismatch")
    if route in {"mri_sequence", "mri_tumor_presence"} and data_source != "OpenNeuro_ds007045":
        raise ValueError("ds007045 reward route/source mismatch")
    answer, formatting = _answer_and_format(solution_str)
    if route == "mri_centroid_2d":
        target = _COORD.fullmatch(ground_truth)
        if target is None:
            raise ValueError("Malformed centroid ground truth")
        predicted = _COORD.fullmatch(answer)
        distance = math.hypot(
            int(predicted.group(1)) - int(target.group(1)),
            int(predicted.group(2)) - int(target.group(2)),
        ) if predicted else float("inf")
        accuracy = float(distance <= 3.0)
    elif route == "mri_dominance":
        if ground_truth not in {"anterior", "posterior"}:
            raise ValueError("Malformed dominance ground truth")
        accuracy = float(answer.lower() == ground_truth)
        distance = None
    elif route == "mri_enhancing_presence":
        if ground_truth not in {"yes", "no"}:
            raise ValueError("Malformed enhancing-presence target or source")
        accuracy = float(answer.lower() == ground_truth)
        distance = None
    elif route == "mri_tumor_presence":
        if ground_truth not in {"yes", "no"}:
            raise ValueError("Malformed tumor-presence target")
        accuracy = float(answer.lower() == ground_truth)
        distance = None
    elif route == "mri_sequence":
        if ground_truth not in {"FLAIR", "contrast-enhanced T1"}:
            raise ValueError("Malformed MRI sequence target")
        accuracy = float(answer.lower() == ground_truth.lower())
        distance = None
    elif route == "mri_tumor_grid":
        if ground_truth not in _GRID_LABELS:
            raise ValueError("Malformed tumor-grid target")
        accuracy = float(answer.lower() == ground_truth)
        distance = None
    else:
        raise ValueError(f"Unsupported MRI reward route: {route}")
    result = {"score": (1-format_score)*accuracy + format_score*formatting,
              "accuracy": accuracy, "format": formatting}
    if distance is not None:
        result["voxel_distance"] = distance if math.isfinite(distance) else -1.0
    return result
