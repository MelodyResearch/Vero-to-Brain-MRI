#!/usr/bin/env python3
"""CPU-only, read-only V2 regression diagnosis over the immutable Accuracy Pilot V2 run (no MLX/model/generation/optimizer).

Writes three NEW files into --out (refuses to overwrite) and prints only the aggregate public candidate to stdout.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = WORKTREE.parents[1]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=DEFAULT_ROOT, help='canonical repository root holding local_data/')
    p.add_argument('--worktree', type=Path, default=WORKTREE)
    p.add_argument('--run', type=Path, help='immutable V2 run (default: ROOT/local_data/accuracy_pilot_v2/pilot_v2_attempt1_PRIVATE)')
    p.add_argument('--out', type=Path, help='new private output directory (default: ROOT/local_data/v2_regression_diagnosis_v1)')
    p.add_argument('--results', type=Path, help='committed public V2 results JSON (default: WORKTREE/results/ACCURACY_PILOT_V2_ACTUAL_RESULTS_V1.json)')
    p.add_argument('--expected-scope-sha256', default='afb0d7ef4af2bc581ffce761faba8975a00f32d6ff2ba9c2f4975c838c115482')
    p.add_argument('--vocab-json', type=Path, help='tokenizer vocab.json read as JSON only (default: runtime model dir)')
    p.add_argument('--tokenizer-json', type=Path, help='tokenizer.json for added tokens, read as JSON only (default: runtime model dir)')
    p.add_argument('--mlx-optimizer-source', type=Path, help='installed mlx/optimizers/optimizers.py, read as TEXT only (never imported)')
    p.add_argument('--no-regrade', action='store_true', help='skip production re-grading of response texts')
    args = p.parse_args(argv)
    root, worktree = args.root.resolve(), args.worktree.resolve()
    sys.path.insert(0, str(worktree / 'src'))
    from mri_rl import v2_regression_diagnosis_v1 as diag
    diag.install_import_guard()
    run = (args.run or root / 'local_data/accuracy_pilot_v2/pilot_v2_attempt1_PRIVATE').resolve()
    out = (args.out or root / 'local_data/v2_regression_diagnosis_v1').resolve()
    results = (args.results or worktree / 'results/ACCURACY_PILOT_V2_ACTUAL_RESULTS_V1.json').resolve()
    runtime = json.loads(Path(json.loads((root / 'local_data/accuracy_pilot_v2/PILOT_V2_SPEC_PRIVATE.json').read_text())['runtime_spec']).read_text())
    model_dir = Path(runtime['model'])
    vocab = args.vocab_json or model_dir / 'vocab.json'
    tokenizer = args.tokenizer_json or model_dir / 'tokenizer.json'
    mlx_source = args.mlx_optimizer_source or next(iter(sorted(root.glob('work/accuracy-recovery-venv-v1/lib/python3*/site-packages/mlx/optimizers/optimizers.py'))), None)
    result = diag.run_analysis(root=root, worktree=worktree, run_dir=run, out_dir=out, results_path=results, expected_scope_sha256=args.expected_scope_sha256,
                               vocab_json=vocab, tokenizer_json=tokenizer, mlx_optimizer_source=mlx_source, regrade=not args.no_regrade)
    print(json.dumps({'written_sha256': result['written_sha256'], 'public': result['public']}, sort_keys=True, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
