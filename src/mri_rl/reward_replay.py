"""Offline reward diagnostics on declared rollout groups.

This module does not generate model responses or infer MRI truth. Its demo
groups are engineering fixtures and must never be reported as VLM results.
"""

from __future__ import annotations

import json
import math
from collections import Counter

from mri_rl.core import reward


def standardized_advantage(values: list[float], epsilon: float = 0.0) -> tuple[list[float], bool]:
    """Population-standardized group advantages, with exact ties inactive."""
    if len(values) < 2 or any(not math.isfinite(v) for v in values):
        raise ValueError("Group needs at least two finite rewards")
    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and nonnegative")
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    if variance == 0:
        return [0.0] * len(values), False
    std = math.sqrt(variance)
    return [(v - mean) / (std + epsilon) for v in values], True


def _logistic(value: float, p50: float, iqr: float) -> float:
    z = 3.0 * (value - p50) / iqr
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    exp_z = math.exp(z)
    return exp_z / (1.0 + exp_z)


def replay_groups(groups: list[dict], epsilon: float = 0.0) -> dict:
    """Compare content rewards, affine scaling, logistic scaling and format mix.

    The affine transform is the declared 2*x+5 demonstration. Logistic is
    MedGRPO Eq. 3 with k=3 and caller-supplied training-derived p50/IQR.
    Format mix is Vero's 0.8 content + 0.2 format, independently applied.
    It omits overlong penalty, KL, clipping, and optimizer effects.
    """
    if not groups:
        raise ValueError("Need at least one group")
    seen = set()
    output = []
    for group in groups:
        group_id = group["id"]
        task = group["task"]
        content = [float(x) for x in group["content"]]
        fmt = [float(x) for x in group["format"]]
        p50 = float(group["p50"])
        iqr = float(group["iqr"])
        if not group_id or group_id in seen or not task:
            raise ValueError("Group IDs must be unique and task names nonempty")
        seen.add(group_id)
        if len(content) < 2 or len(content) != len(fmt):
            raise ValueError("Content and format need equal group lengths >= 2")
        if any(not math.isfinite(x) for x in content + fmt + [p50, iqr]):
            raise ValueError("Rewards and calibration statistics must be finite")
        if iqr <= 0:
            raise ValueError("IQR must be positive")
        if any(x < 0 or x > 1 for x in fmt):
            raise ValueError("Format reward must be in [0,1]")
        content_rewards = {
            "raw": content,
            "affine": [2 * x + 5 for x in content],
            "logistic": [_logistic(x, p50, iqr) for x in content],
        }
        rewards = dict(content_rewards)
        for name, scores in content_rewards.items():
            rewards[f"{name}_format"] = [0.8 * score + 0.2 * format_score
                                         for score, format_score in zip(scores, fmt)]
        schemes = {}
        for name, values in rewards.items():
            advantages, active = standardized_advantage(values, epsilon)
            schemes[name] = {"rewards": values, "advantages": advantages, "active": active}
        output.append({"id": group_id, "task": task, "schemes": schemes})

    scheme_names = output[0]["schemes"]
    nominal = Counter(row["task"] for row in output)
    summary = {}
    for scheme in scheme_names:
        active = Counter(row["task"] for row in output if row["schemes"][scheme]["active"])
        active_count = sum(active.values())
        summary[scheme] = {
            "groups": len(output),
            "active_groups": active_count,
            "zero_variance_groups": len(output) - active_count,
            "nominal_share": {task: count / len(output) for task, count in sorted(nominal.items())},
            "active_share": {task: (active[task] / active_count if active_count else 0.0)
                             for task in sorted(nominal)},
        }
    return {"groups": output, "summary": summary}


def constant_output_probe(records: list[dict], prediction: str) -> dict:
    """Show reward attainable by one answer regardless of image evidence."""
    if not records:
        raise ValueError("Need at least one record")
    scores = [reward(row["task"], prediction, row["target"]) for row in records]
    return {"records": len(records), "mean_reward": sum(scores) / len(scores), "prediction": prediction}


def engineering_fixture() -> list[dict]:
    """Small transparent toy groups; no subject, image, or model provenance."""
    return [
        {"id": "cat-all-wrong", "task": "categorical", "content": [0, 0, 0, 0],
         "format": [1, 1, 1, 1], "p50": 0.5, "iqr": 1.0},
        {"id": "cat-all-wrong-format-varies", "task": "categorical", "content": [0, 0, 0, 0],
         "format": [1, 0, 1, 0], "p50": 0.5, "iqr": 1.0},
        {"id": "cat-mixed", "task": "categorical", "content": [0, 1, 0, 1],
         "format": [1, 1, 1, 1], "p50": 0.5, "iqr": 1.0},
        {"id": "spatial-graded-a", "task": "spatial", "content": [0.0, 0.2, 0.5, 1.0],
         "format": [1, 1, 0, 1], "p50": 0.5, "iqr": 0.5},
        {"id": "spatial-graded-b", "task": "spatial", "content": [0.1, 0.3, 0.4, 0.8],
         "format": [1, 0, 1, 1], "p50": 0.5, "iqr": 0.5},
    ]


def main() -> None:
    fixture = engineering_fixture()
    constant_targets = [{"task": "categorical", "target": {"label": label}}
                        for label in ["T1", "T1", "T1", "FLAIR"]]
    print(json.dumps({"fixture_type": "ENGINEERING_ONLY_NO_MODEL_OUTPUTS",
                      "replay": replay_groups(fixture),
                      "constant_output_probe": constant_output_probe(constant_targets, "T1")},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
