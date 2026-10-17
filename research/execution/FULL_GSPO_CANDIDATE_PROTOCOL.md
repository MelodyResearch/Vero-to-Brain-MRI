# Full GSPO candidate protocol — PROPOSED by Claude, NOT_APPROVED, NOT_RUN

Written before any outcome of these arms. Derived from `VERO_MRI_TRANSFER_EXPERIMENT_CONTRACT.md` and `VERO_TO_MRI_PRIMARY_EXECUTION_CONTRACT.md` Stage D. It replaces the unexecutable `100 steps / group 8 / max 512` design template for the local Mac. The 1-update pilot cap covers only the single-FLAIR-group diagnostic and is not extended by this file. One reviewed approval of this manifest is intended to cover all segments (durable cursor, no per-batch review).

## Task eligibility (stated, not assumed)
- `mri_sequence`: acquisition-label (FLAIR vs contrast T1); visual answerability of the frozen pair is technical-only (clinical blind NOT_VERIFIED); the A28 evidence shows a strong FLAIR prior.
- `tumor_presence`, `enhancing_presence`, `tumor_localization`: derived from the same segmentation masks. Label-source dependence means these are **not four independent skills**.
- Consequence: the four routes are admitted only as an `ENGINEERING_MATCHED` Vero-recipe comparison on one source cohort. Scientific task admission (>=2 distinct, image-answerable, held-out-scored tasks) stays in track C; results here cannot be reported as a transfer finding until C admits tasks. B (supervised control) is the existing B28 start.

## Common design
- Start: B28 adapter (identical for all arms; Adam state fresh at step 0 per arm, then persisted by `continuous_rl` across all segments). Local profile: 4-bit base, LoRA r4/a8, Adam 1e-5, strict `direct_vero`, temperature 1, group size 2, max_tokens 256 (the contract's value). Not veRL parity.
- Data: the 28-patient / 382-row training split only; dev (4 patients, 54 questions) for evaluation only; reserved patients never used.
- Arms (4): `S` sequence-only specialist; `T` tumor-presence specialist; `P` proportional over the 4 routes (pool-size weights); `U` uniform over the 4 routes. (Enhancement/localization specialists, raw-vs-calibrated and SFT arms are deferred: each needs its own admitted task or reviewed reward.)
- Primary budget (compute-matched): **G = 56 attempted groups per arm** (112 rollouts, <=256 generated tokens each), identical schedule length, seeds and unique-image cap; schedules built before training from a seeded sampler, patient-balanced without repeating an image until the pool is exhausted. Secondary estimand: realized optimizer updates (reported, not matched, since zero-variance groups are skipped by design; the earlier strict-profile pilots produced 0 updates, so a low active fraction is a real expected outcome and is reported, never forced).
- Segments: 4 attempted groups each (14 per arm), checkpoint (adapter + Adam + RNG + sampler cursor) after every group via existing atomic boundaries; one resident model per segment run, continuing from the last boundary.
- Stopping: G attempted groups reached, or a hard failure (critical resource, nonfinite state). No early stop on reward or dev score. No dev-based checkpoint selection: the evaluated checkpoint is the final one.
- Seeds: seed 7 primary; seeds 11 and 13 only if time allows, reported separately.
- Evaluation: B28 start and the 4 final checkpoints on identical dev inputs (greedy, `continuous_rl_dev_eval`), 5 rows x dev categories matrix; report per-category accuracy, format/invalid rate, FLAIR/T1 prediction rates, termination, active-group fraction, reward variance, and patient-clustered descriptive intervals (4 dev patients: no inferential claims).
- Visual-grounding controls on the sequence route: real vs no-image vs matched-pair swap using the existing builders.

## Not decided here (needs researcher/Codex)
Approval of G and the arm list; whether to wait for track-C task admission before calling any result transfer; the per-route dev question counts (not read by me).
