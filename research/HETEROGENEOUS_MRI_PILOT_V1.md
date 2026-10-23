# First real heterogeneous MRI pilot v1 (sequence FLAIR/T1ce + VQA-RAD midline shift)

**Preparation snapshot (before execution):** `SOURCE_IMPLEMENTED_CPU_SYNTHETIC_NOT_NATIVE_RUN`. CPU `prepare` returns
`PREPARED_PENDING_EXACT_REVIEW_AND_FRESH_HUMAN` with 0 blockers. The exact scope has been published privately; fresh human execution permission remains absent.

**Author:** Claude Code (claude-opus-5-5), primary implementer. Codex's source/resource review fixes and one final resource-boundary regression have been applied.
The exact-scope independent review is recorded separately; it does not grant execution permission. No MLX, model load, generation, backward or update was run.

## Actual authorized run — 2026-10-23 UTC

**Current outcome: `NO_SIGNAL / INVALID_CONTROL`. This one-shot authorization is consumed and closed.**
Claude Code (claude-opus-5-5) executed the frozen scope once. Both wrapper and native child exited 0. Actual cost was one model
load, six TRAIN greedy controls and 110 returned tokens; no sampled TRAIN rollout, scorer, backward, optimizer update or new
checkpoint was produced. The gate stopped the run before Parent midline EVAL or terminal evaluation. No retry or resampling occurred.

All six replies contained think/answer tags but omitted the required boxed answer, producing format=0.5 and strict invalid.
A separate diagnostic extraction of the already saved answer tags (not a relaxed official score) found zero prediction changes
under the two opposite-image swaps. Thus fixing boxing alone would not establish the frozen image-dependence condition.
This is not a causal finding about GSPO: no GSPO rollout or update occurred.

Historical, same-contract Parent DEV24 is 18/24 (75%), FLAIR 10/12, T1 8/12, invalid 0. It was hash/journal-reused, not rerun.
No new midline EVAL accuracy, equal-task/worst-task accuracy or accuracy improvement was measured.

Observed wrapper time: 151.433727 s; native child: 139.393693 s. Peak RSS (wait4): 5.590561 GiB; reported MLX peak:
6.221834 GiB. External monitor minimum available memory: 5.136795 GiB; minimum pressure-free: 28%. Runtime swap:
13.917295 → 16.897402 GiB (+2.980107 GiB). No guardian STOP, TERM, KILL or monitor errors occurred. Temperature was unavailable.
Model release was actually verified (model=None; active MLX=1052 bytes), child reaped and global lock free. Parent files remain
unchanged. Post-exit available memory was 11.619659 GiB. This observation does not guarantee avoidance of future OS failures.

[Actual result](../results/HETEROGENEOUS_MRI_PILOT_NATIVE_ACTUAL_V1.json) ·
[Independent actual-run review](../results/HETEROGENEOUS_MRI_PILOT_ACTUAL_INDEPENDENT_REVIEW_V1.json)

**Decision:** stop this configuration; do not expand or automatically reissue the unused budget. Any future changed input/output
contract requires a separately declared test and fresh permission. Do not infer clinical generalization, image grounding or
GSPO effectiveness from this rejected pre-update gate. The preparation record below is retained as its historical cutoff.


## Question
Can one direct Vero-style same-prompt GSPO step from Parent Adam52 use two heterogeneous MRI tasks, each with a real
verifiable reward, without hidden negative transfer? This is a feasibility check only. It makes no accuracy or
generalization claim.

## Census (OBSERVED, CPU; public VQA-RAD QA JSON SHA `948b7156…`, 2248 rows; agrees with Codex's oracle)

| Field | Count |
|---|---|
| HEAD yes/no explicit midline-shift QA (incl. `midlight`, `shifted`) | 14 (11 evaluated, 3 not evaluated) |
| Mass + shift with no explicit midline (AMBIGUOUS, never merged) | 2 |
| Admitted: evaluated + CLOSED + PRES + MRI source-text evidence | **5 components: 3 yes, 2 no** |
| All MRI incl. not-evaluated (reported, **not promoted**) | 8 components: 4 yes, 4 no |

`data_scale = DATA_TOO_SMALL_FOR_LEARNING_CLAIM`. The existing frozen 2 TRAIN + 2 EVAL split is a technical smoke test.
No 4+4 split was invented, and no not-evaluated QA was relabelled as evaluated.

## Frozen design (existing machinery; no new loss, LR, LoRA, optimizer or reward)

- **Midline prompt.** The original question is kept verbatim. The prompt appends one frozen suffix,
  `'\n\n' + 'Answer yes or no. ' + SHORT_INSTRUCTION`, using the same join as V2 `condition_row`.
- **Image-dependence gate.** Runs on TRAIN only, greedy, before any rollout, with 6 controls:
  - 2 TRUE: each TRAIN item's question with its own image;
  - 2 SWAP: the same question with the opposite-label image;
  - 2 BORDER: the same question with Codex's frozen CPU central-occlusion images.
  The BORDER images were made before any model output and are bound by qid, source SHA and control SHA. Their rectangle is
  **not** a brain mask: it may retain peripheral MRI pixels and certifies no watermark exclusion or grounding.
- **Gate rule.** PASS iff all 6 controls are strictly valid, the two TRUE answers differ, and each SWAP answer equals the TRUE
  answer of the image that was shown. Both the positive and the inverted pattern pass, because the gate measures dependence,
  not correctness. If the BORDER pair reproduces the TRUE pair, the result is `NO_SIGNAL` and no update happens. The no-image
  control is not run, because the existing path requires exactly one image.
- **Training.**
  - 16 real responses from Parent (sequence F4/T4 plus midline K4/K4).
  - The legacy `DALLAS_BLOCK` schema is reused only as transport; the receipts declare task `vqarad_midline_shift` and
    `dallas_data: false`.
  - If either task has no admitted group, the result is `NO_SIGNAL` with 0 backward calls.
  - Otherwise there is one step 52→53 with joint gradient = 0.5·g_seq + 0.5·g_mid.
  - The sequence reward uses the existing calibrated grader; the midline reward uses the raw verifier.
- **Failure handling.** Any failure, including ResourcePause/Fatal/RED, writes a STOP record and does **no in-process
  rollback**. The process is released, the Parent on disk is unchanged, spent work stays charged and nothing is replayed.
- **Evaluation.**
  - Parent DEV24 is the 24 V2 EP concise rows. Each one is authenticated read-only through its committed journal record, and
    the EP terminal is bound to the re-derived executed V2 scope.
  - Plus 2 new Parent midline EVAL calls.
  - Terminal: a fresh process runs DEV24 plus the 2 midline EVAL items. With 0 updates, the terminal is the Parent alias.
- **Outcome.** `NO_SIGNAL` if there were 0 updates. `FAIL_FORMAT_COLLAPSE` if any task gained invalid outputs.
  `NEGATIVE_TRANSFER` if FLAIR or T1 recall, or either task's balanced accuracy, fell below Parent. Otherwise `FEASIBLE`.
  Per-task correct→wrong flips are always reported.

## Measured prompt lengths (once, CPU)
The existing `native_processor_qualification_v1` helper was run with the pinned venv. MLX was not imported, torch was not
imported, and there were 0 model calls.

| Midline input | Length |
|---|---|
| TRAIN | 405 / 460 |
| EVAL | 287 / 143 |
| SWAP + BORDER controls (4) | 405, 409, 456, 460 |

Method check: the frozen V2 concise sequence rows re-measure to **233 / 233**. The measurement file
(`b27bd92d…`) is bound to the row identity and the suffix SHA.

## Exact budget

| Quantity | Limit |
|---|---|
| Model loads (1 resident at a time) | ≤2 |
| Generations | ≤50 (34 greedy + 16 sampled) |
| Generated tokens | ≤12,800 |
| Detached own-old forwards / gradient forwards | 16 / 16 |
| Backward calls | ≤16 |
| Optimizer updates | ≤1 |
| TF positions (own-old, exact per row) | 9,404 |
| Gradient TF positions | ≤9,404 |
| Greedy prefill positions | 9,047 |
| Windows (enforced, waits count) | parent 2,400 s; terminal 2,400 s; global 4,800 s |
| Declared cost-model estimate (not a measured upper bound) | 1,953 s / 1,832 s |

Resources: unchanged GREEN/YELLOW/RED policy, RSS 14 GiB, MLX 12 GiB, disk 26 GiB, battery 20%. One attempt per phase,
no recovery. The human grant must be ≤30 min old, must not appear in any historical consumed ledger, must cover the
4,800 s flow, and is consumed once.

## Prerequisites at the original CPU-preparation cutoff (historical)
1. Fresh human execution grant. The final exact-scope independent Codex review is `PASS_WITH_LIMITATIONS`.
2. The native path is untested. The 8 distinct midline inputs (TRAIN, SWAP, BORDER and EVAL) have no native F32
   qualification; the native hook refuses any length that differs from the frozen per-row value.

## CPU verification and execution boundary

Codex independently ran the five targeted CPU suites: **158 passed**, including 58 pilot tests. This includes the late-RED
case: a safety signal arriving after the last response prevents post-run native fingerprints and RNG reload. Native admission
reuses the historical installed-runtime and certified-shard checks; no cache is reconstructed. Actual greedy prompt/image/F32
traces are retained for later independent verification. The eight expanded midline contexts were independently recomputed
with the CPU processor and all matched (no MLX imported).

Scope FILE SHA256: `fd565a94a47d471e867c9f908aceed956692ac50a0e3f11ac7ca6138c75941e4`.
Canonical scope digest: `883cb2038f7ee3a4656dbf5a9a37fc285bb7a6fba5dea8a80432be856151c064`.
Full-token cost-model estimate is about 63 minutes combined; the operation-start windows total 80 minutes and include waits.
These are declared planning assumptions, not observed execution times or a guarantee against OOM.

Preparation facts: [CPU result](../results/HETEROGENEOUS_MRI_PILOT_PREPARATION_ACTUAL_V1.json),
[census review](../results/HETEROGENEOUS_MRI_CENSUS_INDEPENDENT_V1.json).
This turn performed **0 native model loads, 0 generations, 0 backward and 0 optimizer updates**.

The final Parent post-evaluation RNG reload and segment state/snapshot boundaries are guarded as well.
An earlier never-authorized scope was superseded; only the final FILE SHA above may be proposed for execution.

[Final independent scope review](../results/HETEROGENEOUS_MRI_PILOT_INDEPENDENT_REVIEW_V1.json): 58 independently run pilot tests passed; 111 source and 165 input hashes verified. This review authorizes no native execution.
