# Heterogeneous MRI pilot V2 (sequence FLAIR/T1ce + VQA-RAD midline shift)

**Status:** `SOURCE_IMPLEMENTED_CPU_SYNTHETIC_NOT_NATIVE_RUN`. CPU `prepare` returns
`PREPARED_PENDING_EXACT_REVIEW_AND_FRESH_HUMAN` (169 input files and 115 sources bound, 0 blockers). The final exact scope and independent review are recorded in the accompanying sanitized result; human execution permission is pending.

**Implementation:** Claude Code (`claude-opus-5-5`) wrote the primary core, native bridge, entry, tests, measurement and diagnosis tools. A follow-up CLI call returned `Not logged in`; Codex fixed the independent-review findings and verified the final changes. No Claude participation is claimed for those follow-up fixes. For this version, 0 model
loads, generations, teacher-forced passes, backward calls or updates were run. V1 source, results and private receipts are
unchanged.

## 1. V1 diagnosis (saved evidence only; CPU; no replay of the 6 calls)
Sanitized summary: `results/HETEROGENEOUS_MRI_PILOT_V2_CPU_ACTUAL.json`. The private proof is in the V2 private root.

**Confirmed (OBSERVED on CPU):**
- **Delivered prompts.** All 6 rendered prompts reproduce exactly to their saved SHAs: system instruction + original question
  + v1 suffix. The prompt ends with the contract's assistant prefix, `enable_thinking` was false, decoding was greedy
  (temperature 0), and the cap was 256 tokens.
- **The prompt did ask for a box.** The system instruction asks for "one boxed final answer in `<answer>`", and the v1 suffix
  (`SHORT_INSTRUCTION`) says "finish with the required boxed answer".
- **The responses.** Every answer had exactly one `<think>…</think>` and one `<answer>…</answer>`, with `\boxed` count 0,
  giving parser format 0.5. The tag-only greedy decisions changed in 0/2 swaps; raw answers remain private.
- **Pixel identities.** Each SWAP's pixels equal the other item's TRUE pixels. BORDER pixels differ from TRUE while its
  `input_ids` are equal. The two TRUE items differ in pixels.
- **Cleaning evidence.** The currently pinned Qwen3-VL processor has no `clean_output`; V1 did not explicitly save the invocation/pre-clean text, so this is source evidence, not an independent native pre/post-clean trace. A `\boxed{no}` probe round-trips through the
  pinned tokenizer exactly.
- **Token counts.** Re-encoding each saved text gives saved count − 1 tokens. This is consistent with the text plus one EOS
  token and nothing stripped. It is consistent evidence, not a proof, because re-tokenization may not reproduce the generated
  segmentation.

**Derived from an enforced invariant, not from saved metadata:** `capture_stream` only returns `stop` with the last token in
the EOS set, or `length` at exactly 256 tokens. Responses of 18–19 tokens are consistent with that stop/EOS path; the exact token/stop cause cannot be independently reconstructed from V1 saved output.

**Not available:** v1 saved only text, token count and trace. It did not persist response token IDs, finish reason, EOS id,
per-token logprobs, explicit pre-clean text/cleaner invocation or canonical yes/no margins.

**Hypotheses (untested):**
- The Parent prefers an unboxed `<answer>` for this short clinical yes/no question.
- The indirect instruction ("finish with the required boxed answer") was weaker than an explicit boxed example would be.

**On SWAP = TRUE:** identical greedy argmax decisions do not imply identical image-conditional probabilities, and no margins
were measured.

## 2. Frozen V2 midline suffix (one; original question verbatim; joined with `\n\n`)
`Look at the image and answer yes or no. Keep the reasoning to one short sentence. Reply in exactly one of these two forms:
<think>one short sentence about the image</think><answer>\boxed{yes}</answer> or <think>one short sentence about the
image</think><answer>\boxed{no}</answer>` (sha `78ad42495ce6…`).

- In the Python source this is a literal backslash (`'\\boxed'`).
- **Unchanged:** the system instruction, the sequence concise prompt, parsers, reward, LR, LoRA, Adam and GSPO. There is no
  automatic boxing or response repair.
- **Measured once on CPU** with the pinned processor helper (no MLX, no torch, 0 model calls):

  | Input | Length (tokens) |
  |---|---|
  | TRAIN | 445 / 500 |
  | EVAL | 327 / 183 |
  | SWAP / BORDER | 449, 496, 445, 500 |
  | Sequence rows (method check) | 233 / 233 |

  Each midline row is exactly +40 tokens versus v1. Identifier leaks visible to the model: 0.

## 3. Gates
- **Phase 1 (train process).** The same 6 frozen controls run greedy on Parent Adam52.
  - The **hard gate is strict format on TRUE_0 and TRUE_1** (the real TRAIN inputs). Failing it gives `FORMAT_BLOCKED`,
    before any on-policy work.
  - SWAP/BORDER answers and the v1 image-dependence classification are recorded but **never veto**.
  - BORDER is not a no-image control and not an anatomy-grounding control.
- **Phase 2.** 16 real on-policy responses are generated once: sequence F4/T4 plus midline D1/D2 K4.
  - All failures are kept and graded with the original reward. Nothing is re-drawn, repaired or filtered.
  - Raw text, token IDs, finish reason and EOS metadata are persisted. V2 refuses an unexpected callable cleaner before generation and explicitly records `clean_called:false` plus the unchanged pre-clean text.
  - **Before any backward:** both tasks need at least one inherited-gate-admitted same-prompt group that contains a
    strict-valid CORRECT and a strict-valid WRONG response. Format-only reward variance does not count. Otherwise
    `NO_GSPO_SIGNAL`.
  - The original advantages, loss and 1/8 weights are kept, with joint gradient = 0.5·g_seq + 0.5·g_mid.
  - A zero or returned-nonfinite norm gives `NO_GSPO_SIGNAL`. The actual native backend rejects nonfinite tensor leaves by exception, producing STOP; that exception is never swallowed as a scientific NO_SIGNAL result. At most one Adam step 52→53, with an atomic checkpoint.
- **Phase 3 (only after a committed update).** Fresh processes run only after validating the train journal, artifact hashes, terminal, actual reaped clean exit and allocator release:
  1. Parent midline EVAL2 under the new prompt. Parent DEV24 is the journal-authenticated V2 EP concise rows.
  2. The child runs DEV24 + EVAL2 + the same 6 controls; the controls are diagnostic only.
- **Outcome.** With an update, the run is `REAL_PILOT_COMPLETED` and the result is one of:
  - `NEGATIVE_TRANSFER`: any class recall (FLAIR, T1, yes, no) fell, or any task's invalid count rose;
  - `FEASIBLE`: equal-task strict balanced accuracy rose with no such loss;
  - `NO_ACCURACY_GAIN`: otherwise.

  Without an update, the outcome is `FORMAT_BLOCKED` or `NO_GSPO_SIGNAL`, never "completed learning". Midline EVAL has n=2,
  so no efficacy or generalization claim is possible.
- **On any error.** A STOP record is written, with no in-process rollback or fingerprint, only process release, and no replay.

## 4. Schedule and budget (3 serial fresh processes)
A two-load schedule would need either Parent midline EVAL before training, which spends evaluation even without an update,
or an in-process restore of the trained model. Both conflict with the requested rules, so V2 declares one extra Parent load.

| Quantity | Limit |
|---|---|
| Model loads (1 resident at a time) | ≤3 |
| Generations | ≤56 (40 greedy: 6 / 2 / 32; 16 sampled) |
| Generated tokens | ≤14,336 |
| Scorer forwards | 32 |
| Teacher-forced positions | 19,448 (2 × 9,724) |
| Backward calls | ≤16 |
| Optimizer updates | ≤1 |
| Greedy prefill positions | 12,282 |

- **Windows** (unchanged V2 cost model, not a measured guarantee): train 2,400 s, parent_eval 900 s, child_eval 3,000 s,
  global 7,200 s; waits count.
- **Resources:** the existing GREEN/YELLOW/RED policy is inherited unchanged (RSS 14 GiB, MLX 12 GiB, disk 26 GiB,
  battery 20%), with the global flock, the guardian, the pre-native runtime/cache check, and a fresh grant that cannot be one
  consumed by an earlier scope.

## 5. Files
- `src/mri_rl/heterogeneous_mri_pilot_v2.py`, `heterogeneous_mri_pilot_native_v2.py`, `heterogeneous_mri_pilot_entry_v2.py`:
  these compose the v1 and multistep helpers.
- `scripts/run_heterogeneous_mri_pilot_v2.py`, `scripts/measure_heterogeneous_midline_prompts_v2.py`,
  `scripts/diagnose_heterogeneous_pilot_v1_format.py`.
- `tests/test_heterogeneous_mri_pilot_v2.py`: 51 passing CPU tests. The selected six related CPU suites have 209 passing tests (not an unrestricted repository/native test run). These include observed failing regressions followed by fixes for RED boundaries, missing transaction closure, wrong/unreaped exit witnesses and checkpoint tampering.

## 6. Missing native evidence
- No V2 native run has happened.
- The 8 V2 midline F32 contexts are not natively qualified.
- Whether the Parent boxes its answer under the V2 suffix is unknown until the Phase-1 run.

## 7. Final independent scope review (no execution permission)
Isolated Codex returned **PASS_WITH_LIMITATIONS**, independently reproduced 51 V2 CPU tests and exact preparation,
and verified all115 source/169 input hashes. See [independent review](../results/HETEROGENEOUS_MRI_PILOT_V2_INDEPENDENT_REVIEW.json).

- Exact execution scope FILE SHA256: `a0ce6e0bb5874eb0414fae9ac7d9fccaa9ab85d6ece326277268a488e22603f1`.
- Canonical scope identity: `a3f8f6f49adcc62630fc6713d0850bfb097d1590e867821a45f0c275b14956c7`.
- Independent PRIVATE review SHA256: `79f43d9934444cb85c3f81b2fa989926c3ca2149e69055f16eebb3274b37d22d`.

The reviewed source is ready for one fresh explicit native authorization. No new grant, model run or accuracy result exists.
