"""Command-line data gate, rendering, and precomputed-prediction evaluation."""

import argparse
import json
from pathlib import Path

from .core import audit_records, evaluate, render_frames


def _jsonl(path):
    with open(path, encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(prog="mri-rl")
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit", help="Audit an image-linked manifest before any model run")
    audit.add_argument("--train", type=Path, required=True)
    audit.add_argument("--test", type=Path, required=True)
    audit.add_argument("--validation", type=Path)
    score = sub.add_parser("score", help="Score saved model predictions")
    score.add_argument("--records", type=Path, required=True)
    score.add_argument("--predictions", type=Path, required=True)
    render = sub.add_parser("render", help="Render linked genuine frames in index order")
    render.add_argument("--record", type=Path, required=True)
    render.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "audit":
        splits = {"train": _jsonl(args.train), "test": _jsonl(args.test)}
        if args.validation:
            splits["validation"] = _jsonl(args.validation)
        result = audit_records(splits)
        print(json.dumps(result, indent=2))
        return 0 if result["usable_records"] == result["records"] else 2
    if args.command == "score":
        rows = _jsonl(args.records)
        outputs = _jsonl(args.predictions)
        predictions = {row["id"]: row["prediction"] for row in outputs}
        if len(predictions) != len(outputs):
            raise ValueError("Duplicate prediction IDs")
        print(json.dumps(evaluate(rows, predictions), indent=2))
        return 0
    row = json.loads(args.record.read_text(encoding="utf-8"))
    print(json.dumps(render_frames(row["frames"], args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
