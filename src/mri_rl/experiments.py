"""Controls for planned arms and patient-level paired inference."""

import random
from collections import defaultdict


def validate_arms(arms: list[dict]) -> dict:
    if not arms or len({a["name"] for a in arms}) != len(arms):
        raise ValueError("Arms must be nonempty and uniquely named")
    keys = ("steps", "group_size", "max_tokens")
    budgets = {tuple(a[k] for k in keys) for a in arms}
    if len(budgets) != 1 or any(v <= 0 for v in next(iter(budgets))):
        raise ValueError("Arms must have equal positive budgets")
    for arm in arms:
        weights = arm["tasks"]
        if not weights or any(v <= 0 for v in weights.values()) or abs(sum(weights.values()) - 1) > 1e-9:
            raise ValueError("Task weights must be positive and sum to one")
    return dict(zip(keys, next(iter(budgets))))


def paired_patient_difference(rows: list[dict], n_bootstrap: int = 2000, seed: int = 0) -> dict:
    """Patient-average score differences and percentile interval.

    For a real study, report training-seed variability separately; bootstrap
    does not account for scanner/site shift or a biased patient sample.
    """
    if n_bootstrap < 1:
        raise ValueError("Need at least one bootstrap resample")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["subject_id"]].append(float(row["candidate"]) - float(row["base"]))
    if len(grouped) < 2:
        raise ValueError("Need at least two independent subjects")
    differences = [sum(values) / len(values) for _, values in sorted(grouped.items())]
    estimate = sum(differences) / len(differences)
    rng = random.Random(seed)
    samples = sorted(sum(rng.choices(differences, k=len(differences))) / len(differences) for _ in range(n_bootstrap))
    low = samples[int(0.025 * (n_bootstrap - 1))]
    high = samples[int(0.975 * (n_bootstrap - 1))]
    return {"subjects": len(grouped), "mean_difference": estimate, "confidence_interval": [low, high]}
