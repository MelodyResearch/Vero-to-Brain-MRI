# README refresh: cleanup candidates (V1)

Candidate list only. Nothing was deleted. Scope: git-tracked files in this docs worktree at base `6ee3d16` (648 files), checked by blob hash and filename pattern. Root owns dependency/provenance checks and any deletion.

## Safe cleanup candidates

- **Exact byte duplicates among non-empty tracked files:** none.
- **Tracked temporary or generated files** (`.tmp`, `.bak`, `.orig`, `.log`, `.pyc`, `.DS_Store`, `__pycache__`, `.egg-info`): none.

Result: no safe deletion candidate.

## Items reviewed and not proposed

- `docs/superpowers/plans/2026-10-15-feasibility-path.md` is the only tracked file under `docs/`. It looks like an early generated plan, but it is a historical planning record that may be referenced by provenance. Root should check before any decision.
- The old 26 KB README (git history, `6ee3d16`) held a per-arm status table and an archived 2026-10-15 gate table. Each row links to a tracked `results/` or `research/` file, and the rows are recoverable from Git. The one claim not repeated elsewhere is its "Working CPU path" CLI example (`mri-rl audit`, `render`, `score`, `dallas_pilot`). The new README keeps only the install and `pytest` commands. Restore the CLI examples if they should stay on the homepage.
- No archive copy of the old README was created, because no unique evidence would be lost.
