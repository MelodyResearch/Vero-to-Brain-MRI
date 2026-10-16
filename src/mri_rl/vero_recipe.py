"""CPU-checkable pieces of a Vero-style multi-task MRI training contract.

No model, optimizer, rollout engine, or validated medical judge is provided.
The GSPO function checks numerical objective values, not autograd behavior.
"""

from __future__ import annotations

import math
import random
import re
from collections import defaultdict
from collections.abc import Callable

from mri_rl.core import reward


SUPPORTED_ROUTES = {
    "exact_match": "categorical",
    "numeric": "numeric",
    "grounding_2d": "spatial",
    "longitudinal_direction": "longitudinal_direction",
    "open_ended_judge": None,
}


def sample_equal_categories(records: list[dict], categories: list[str],
                            per_category: int, seed: int = 0) -> list[dict]:
    """Draw equal category counts with replacement from eligible records."""
    if not categories or len(set(categories)) != len(categories) or per_category < 1:
        raise ValueError("Need unique categories and positive per_category")
    pools = defaultdict(list)
    for row in records:
        pools[row["category"]].append(row)
    missing = sorted(set(categories) - set(pools))
    if missing:
        raise ValueError(f"missing category records: {missing}")
    rng = random.Random(seed)
    sampled = [rng.choice(pools[category]) for category in categories for _ in range(per_category)]
    rng.shuffle(sampled)
    return sampled


def route_accuracy(sample: dict, output: str,
                   judge: Callable[[dict, str], float] | None = None) -> float:
    """Dispatch a restricted subset of reward routes; unknown routes fail closed.

    External open-ended judge is required. This function does not validate
    the medical truth, image visibility, or independence of any target.
    """
    route = sample["reward_type"]
    if route not in SUPPORTED_ROUTES:
        raise ValueError(f"unsupported reward route: {route}")
    if route == "open_ended_judge":
        if judge is None:
            raise ValueError("open-ended judge required")
        score = float(judge(sample, output))
        if not math.isfinite(score) or score < 0 or score > 1:
            raise ValueError("judge score must be finite and in [0,1]")
        return score
    return reward(SUPPORTED_ROUTES[route], output, sample["target"])


_TAGGED = re.compile(r"\s*<think>(.*?)</think>\s*<answer>(.*?)</answer>\s*", re.DOTALL)
_BOXED = re.compile(r"\\boxed\{[^{}]+\}")


def format_reward(output: str, symbolic: bool, grounding: bool = False) -> float:
    """Vero-style think/answer structure and single boxed symbolic answer."""
    match = _TAGGED.fullmatch(output)
    if match is None or not match.group(1).strip() or not match.group(2).strip():
        return 0.0
    if symbolic and len(_BOXED.findall(match.group(2))) != 1:
        return 0.5
    if grounding and match.group(2).count("\\boxed") > 1:
        return 0.5
    return 1.0


def overlong_penalty(length: int, max_tokens: int, buffer: int = 2048,
                     strength: float = 1.0) -> float:
    """Paper Eq. 4 soft linear ramp; no clamp beyond max_tokens."""
    if length < 0 or max_tokens <= buffer or buffer <= 0 or strength < 0:
        raise ValueError("Invalid overlong-penalty parameters")
    return min(-(length - (max_tokens - buffer)) / buffer * strength, 0.0)


def score_response(sample: dict, response: str, length: int, max_tokens: int,
                   judge: Callable[[dict, str], float] | None = None,
                   buffer: int = 2048) -> dict:
    """Combine routed answer accuracy, format, and overlong terms.

    Caller supplies tokenizer-measured response length. This restricted
    parser does not implement every Vero answer type or its judge prompt.
    """
    route = sample["reward_type"]
    symbolic = route in {"exact_match", "numeric", "longitudinal_direction"}
    match = _TAGGED.fullmatch(response)
    answer = match.group(2).strip() if match else ""
    if symbolic:
        boxed = _BOXED.search(answer)
        if boxed:
            answer = boxed.group(0)[len("\\boxed{"):-1]
    accuracy = route_accuracy(sample, answer, judge)
    fmt = format_reward(response, symbolic, grounding=route == "grounding_2d")
    overlong = overlong_penalty(length, max_tokens, buffer)
    return {"accuracy": accuracy, "format": fmt, "overlong": overlong,
            "total": 0.8 * accuracy + 0.2 * fmt + overlong}


def gspo_clipped_surrogate(rewards: list[float], new_logps: list[list[float]],
                           old_logps: list[list[float]], clip_low: float = 0.0003,
                           clip_high: float = 0.0004, advantage_epsilon: float = 0.0) -> dict:
    """Numerical GSPO/constant-advantage GSPO-token clipped objective.

    Uses sequence-mean log probability ratios and group-standardized rewards.
    It deliberately cannot verify stop-gradient placement or weight updates.
    """
    n = len(rewards)
    if n < 2 or len(new_logps) != n or len(old_logps) != n:
        raise ValueError("Need aligned group of at least two responses")
    if any(not math.isfinite(x) for x in rewards) or advantage_epsilon < 0:
        raise ValueError("Rewards and epsilon must be valid")
    if clip_low < 0 or clip_high < 0:
        raise ValueError("Clipping bounds must be nonnegative")
    mean = sum(rewards) / n
    if min(rewards) == max(rewards):
        advantages = [0.0] * n
    else:
        # Pinned Vero veRL uses torch.std(scores_tensor), whose default is
        # Bessel-corrected sample standard deviation (correction=1).
        std = math.sqrt(sum((value - mean) ** 2 for value in rewards) / (n - 1))
        advantages = [(value - mean) / (std + advantage_epsilon) for value in rewards]
    ratios = []
    terms = []
    for fresh, old, advantage in zip(new_logps, old_logps, advantages):
        if not fresh or len(fresh) != len(old) or any(not math.isfinite(x) for x in fresh + old):
            raise ValueError("Need equal nonempty finite token log probabilities")
        ratio = math.exp(sum(a - b for a, b in zip(fresh, old)) / len(fresh))
        clipped = min(max(ratio, 1 - clip_low), 1 + clip_high)
        ratios.append(ratio)
        terms.append(min(ratio * advantage, clipped * advantage))
    return {"advantages": advantages, "sequence_ratios": ratios,
            "objective": sum(terms) / n}


def validate_recipe(recipe: dict) -> dict:
    """Validate planned category/route math without declaring training ready."""
    categories = recipe["categories"]
    routes = recipe["routes"]
    if not categories or len(set(categories)) != len(categories) or "open_ended" not in categories:
        raise ValueError("Recipe needs unique categories including open_ended")
    if set(categories) != set(routes) or any(route not in SUPPORTED_ROUTES for route in routes.values()):
        raise ValueError("Each category needs one supported reward route")
    if recipe["group_size"] < 2 or not 0 <= recipe["clip_low"] < recipe["clip_high"]:
        raise ValueError("Invalid group size or GSPO clipping")
    if recipe["max_tokens"] <= recipe["overlong_buffer"] or recipe["overlong_buffer"] <= 0:
        raise ValueError("Max tokens must exceed positive overlong buffer")
    return {"categories": len(categories), "equal_category_weight": 1 / len(categories),
            "train_ready": False, "reason": "Dataset targets, open-ended judge, model and GPU not validated"}
