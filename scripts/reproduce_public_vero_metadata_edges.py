"""Execute exact public AST helpers, with explicitly limited parser dependencies.

No model, Torch, MLX, private data, Sympy or external judge is loaded.
Input is the frozen public-only review prompt; output is aggregate fixture data.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any


PROMPT_SHA256 = "29f33b1b207be1a8ae329ff2121310d8d0bb74ae7ee01cc157766d67a527f11d"
FUNCTIONS = {
    "_extract_image_size", "_extract_tolerance", "_coerce_float",
    "_numeric_candidates", "grade_numeric",
}


def load_helpers(prompt_path: Path):
    payload = prompt_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != PROMPT_SHA256:
        raise ValueError("Public review prompt hash changed")
    sections = payload.decode().split("===== PUBLIC SOURCE ")
    selected = [s for s in sections if s.startswith(
        "vero-rl/vero_reward/math_verify_reward_type_boxed.py lines 1-1148 =====")]
    if len(selected) != 1:
        raise ValueError("Expected exactly one complete public reward excerpt")
    body = selected[0].split(" =====", 1)[1]
    numbered = [re.fullmatch(r"(\d+): ?(.*)", line) for line in body.splitlines()]
    numbered = [m for m in numbered if m]
    if [int(m[1]) for m in numbered] != list(range(1, 1149)):
        raise ValueError("Public reward excerpt line inventory differs")
    source = "\n".join(m[2] for m in numbered) + "\n"
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in FUNCTIONS]
    if {n.name for n in nodes} != FUNCTIONS:
        raise ValueError("Required public helpers absent")
    # Only the exact selected function bodies execute. No upstream imports run.
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[
        ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    ast.fix_missing_locations(module)
    parser_calls = []

    def parser_stub(text, *_args, **_kwargs):
        parser_calls.append(text)
        return []

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Fixture reached an out-of-scope dependency")

    class NoSympy:
        sympify = staticmethod(forbidden)

    scope = {"Any": Any, "parse": parser_stub, "verify": forbidden,
             "sympy": NoSympy(), "_NUMERIC_EXTRACTION_TARGETS": [],
             "_PARSE_TIMEOUT": 3, "_VERIFY_TIMEOUT": 3,
             "TimeoutException": type("FixtureTimeout", (Exception,), {}),
             "_log_timeout": forbidden}
    exec(compile(module, "public_reward_exact_ast", "exec"), scope)
    return scope, parser_calls, {
        n.name: {"start": n.lineno, "end": n.end_lineno,
                 "source_sha256": hashlib.sha256(ast.get_source_segment(source, n).encode()).hexdigest()}
        for n in nodes}


def run(prompt_path: Path) -> dict:
    scope, calls, sources = load_helpers(prompt_path)
    image_cases = [
        ("flat_list", {"image_size": [100, 200]}, None),
        ("tuple", {"image_size": (100, 200)}, (100, 200)),
        ("nested_list", {"image_sizes": [[100, 200]]}, (100, 200)),
        ("dictionary", {"image_size": {"width": 100, "height": 200}}, (100, 200)),
        ("missing", {}, None),
    ]
    image_results = []
    for name, value, expected in image_cases:
        actual = scope["_extract_image_size"](value)
        assert actual == expected, (name, actual, expected)
        image_results.append({"case": name, "actual": actual, "expected_source_behavior_confirmed": True})
    tolerance_results = []
    for name, value in [("positive_infinity", float("inf")), ("nan", float("nan"))]:
        actual = scope["_extract_tolerance"]({"tolerance": value})
        accepted = actual is not None and (math.isinf(actual) if name == "positive_infinity" else math.isnan(actual))
        assert accepted
        tolerance_results.append({"case": name, "nonfinite_value_accepted": accepted})
    numeric_cases = [
        ("infinite_tolerance_wrong_answer", "1", "999", float("inf"), 1),
        ("finite_tolerance_wrong_answer", "1", "999", 0.1, 0),
        ("finite_tolerance_close_answer", "1", "1.05", 0.1, 1),
        ("zero_gold_absolute_tolerance", "0", "0.05", 0.1, 1),
        ("nan_tolerance", "1", "999", float("nan"), 0),
        ("negative_infinite_tolerance", "1", "999", float("-inf"), -1),
    ]
    numeric_results = []
    for name, gold, pred, tol, expected in numeric_cases:
        actual = scope["grade_numeric"](gold, pred, tolerance=tol)
        assert actual == expected, (name, actual, expected)
        numeric_results.append({"case": name, "gold": gold, "prediction": pred,
                                "actual_grade": actual, "expected_source_behavior_confirmed": True})
    return {"status": "TESTED_CPU", "upstream_commit": "025ca3c800fc95224c86323c1f7fd33dceed3090",
            "prompt_sha256": PROMPT_SHA256, "exact_function_sources": sources,
            "image_size_cases": image_results, "tolerance_cases": tolerance_results,
            "numeric_cases": numeric_results, "parser_stub_calls": len(calls),
            "dependency_scope": "Exact public function AST; parse stub returns no additional candidates. Ordinary decimal strings take exact upstream float conversion. Sympy and verify calls fail this fixture if reached.",
            "not_executed": ["real math_verify parser", "complete reward manager", "grounding handler", "training", "GPU", "MRI"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("New output required; preserve prior fixture receipt")
    result = run(args.prompt)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "image_size_cases": len(result["image_size_cases"]),
                      "tolerance_cases": len(result["tolerance_cases"]), "numeric_cases": len(result["numeric_cases"]),
                      "parser_stub_calls": result["parser_stub_calls"]}, indent=2))
