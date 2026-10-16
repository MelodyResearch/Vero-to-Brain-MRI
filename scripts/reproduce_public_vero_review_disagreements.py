"""Small source-bound comparison fixtures; no model/Torch/MLX imports."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import textwrap
from types import SimpleNamespace
import numpy as np
from reproduce_public_vero_metadata_edges import load_helpers, PROMPT_SHA256


def excerpt(prompt, filename, low, high):
    matching = []
    for section in prompt.split("===== PUBLIC SOURCE ")[1:]:
        header, body = section.split(" =====", 1)
        if filename not in header:
            continue
        for line in body.splitlines():
            m = re.fullmatch(r"(\d+): ?(.*)", line)
            if m and low <= int(m[1]) <= high:
                matching.append((int(m[1]), m[2]))
    if [n for n, _ in matching] != list(range(low, high + 1)):
        raise ValueError("Excerpt inventory differs")
    return "\n".join(line for _, line in matching) + "\n"


def run(path):
    scope, _, _ = load_helpers(path)
    prompt = path.read_text()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == PROMPT_SHA256
    quota_source = textwrap.dedent(excerpt(prompt, "rl_dataset.py", 191, 200))
    sampler = {"np": np}
    exec(compile(ast.parse(quota_source), "exact_public_quota_method", "exec"), sampler)
    counts = {}
    for size in (256, 2):
        obj = SimpleNamespace(domain_weights=[1 / 6] * 6, batch_size=size)
        first = sampler["_compute_batch_counts"](obj)
        assert all(sampler["_compute_batch_counts"](obj) == first for _ in range(3))
        assert sum(first) == size
        counts[str(size)] = first
    assert sorted(counts["256"]) == [42, 42, 43, 43, 43, 43]
    assert counts["2"].count(0) == 4

    # Inject an ordinary parser exception, never an actual threaded timeout.
    def parse_error(*_args, **_kwargs):
        raise RuntimeError("fixture parser failure")
    scope["parse"] = parse_error
    direct = scope["_numeric_candidates"]("1.05")
    assert direct == [1.05]
    finite_tolerance = scope["grade_numeric"]("1", "1.05", tolerance=0.1)
    no_tolerance = scope["grade_numeric"]("1", "1.05", tolerance=None)
    assert finite_tolerance == 1 and no_tolerance == -1

    # Record raw-text reachability without evaluating any expression.
    seen = []
    class FakeExpression:
        is_real = True
        def evalf(self):
            return 3.0
    class RecordingSympy:
        def sympify(self, value):
            seen.append({"length": len(value), "sha256": hashlib.sha256(value.encode()).hexdigest()})
            return FakeExpression()
    scope["sympy"] = RecordingSympy()
    harmless_unevaluated_text = "unknown_expression_" * 100
    assert scope["_coerce_float"](harmless_unevaluated_text) == 3.0
    assert seen[0]["length"] > 200

    # Scalar finite differences of the displayed clipped surrogate, not an
    # execution of PyTorch/autodiff or proof of backend gradient equivalence.
    gradients = []
    for advantage in (-1.0, 1.0):
        for ratio in (0.999, 1.001):
            def loss(delta):
                r = math.exp(math.log(ratio) + delta)
                clipped = min(max(r, 1 - 0.0003), 1 + 0.0004)
                return max(-advantage * r, -advantage * clipped)
            derivative = (loss(1e-8) - loss(-1e-8)) / 2e-8
            active_clip = (advantage > 0 and ratio > 1.0004) or (advantage < 0 and ratio < 0.9997)
            expected = 0.0 if active_clip else -advantage * ratio
            assert math.isclose(derivative, expected, abs_tol=1e-7)
            gradients.append({"advantage": advantage, "ratio": ratio,
                              "outside_clip_interval": True, "active_clipped_branch": active_clip,
                              "surrogate_finite_difference": derivative, "expected": expected})
    gspo = ast.parse(excerpt(prompt, "core_algos.py", 1036, 1099))
    f = next(n for n in gspo.body if isinstance(n, ast.FunctionDef))
    rollout_references = sum(isinstance(n, ast.Name) and n.id == "rollout_log_probs"
                             for statement in f.body for n in ast.walk(statement))
    assert rollout_references == 0
    return {"status": "TESTED_CPU", "prompt_sha256": PROMPT_SHA256,
            "numpy_version": np.__version__, "repeated_quota_counts": counts,
            "parser_exception_direct_candidates": direct,
            "parser_exception_grade_with_tolerance": finite_tolerance,
            "parser_exception_grade_without_tolerance": no_tolerance,
            "raw_text_reaches_sympify_spy": seen,
            "sympy_expression_actually_evaluated": False,
            "scalar_surrogate_checks": gradients,
            "gspo_body_rollout_log_probs_references": rollout_references,
            "not_verified": ["pinned upstream NumPy tie ordering", "real math_verify threading/timeouts",
                             "real Sympy safety/version", "PyTorch GSPO gradient parity",
                             "judge noise independence", "metrics aggregation", "local MRI interfaces"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("New output required")
    result = run(args.prompt)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))
