# VERO_TO_MRI_ACCURACY_RECOVERY_V1 — experiment protocol (CPU preparation; NOT RUN)

Author/lead: Claude Code (primary research and coding agent); Codex = orchestrator/independent reviewer. Evidence labels: OBSERVED / REPRODUCED / INFERRED / PROPOSED / NOT_VERIFIED. This file does not change `CURRENT_RESEARCH_PLAN.md`; the Adam60→Adam52 wording is only the UNAPPLIED `research/PROPOSED_PLAN_DIFF_ADAM60_TO_ADAM52_UNAPPLIED.patch`.

## 1. Question and what is being compared
Single plan question: does organising the *same* real on-policy MRI responses into paired-prompt groups (F2T2/F2T2) change the verifiable learning signal and MRI accuracy relative to the original same-prompt groups (F4/T4)? Only the group membership differs.

- Parent: the authentic original P_calibrated13 boundary114, Adam step 52 (adapter `c70b426b…329db6`, receipt `e0696a81…cc882a`), original arm `{P_calibrated, medgrpo_eq3, proportional}`, seed 13, calibration SHA `0d8564c5…d1a136` (metadata verified by Codex; the code refuses any other value). Chosen only for training-chain ancestry; no final100/new16 output was used.
- One real F4+T4 rollout (same person, FLAIR and T1, z045) from the parent. The immutable 8-response bundle is consumed by both arms: A = {F0-3},{T0-3}; B = {F0,F1,T0,T1},{F2,F3,T2,T3}. Original GSPO loss, weight 1/8 per admitted response, ≤1 Adam update per arm from the identical restored parent, no resample, no momentum-only step.
- Both arms use the same narrow informative-group gate (accuracy varies in the valid-format subset, equal overlong, nonzero total). The parent chain used `all-full-format_task-varies_equal-overlong`: this is an explicit **stage change versus the parent that is identical in A and B**, so it is not an A/B confound. Reward, LR, LoRA, optimizer profile are unchanged.
- **Local recipe, not official Vero.** The chain is Qwen3-VL-8B + B28 language LoRA on MLX with a Vero-style recipe (heterogeneous verifiable reward, GSPO). It is not the published Princeton Vero weights nor an official veRL-equivalent. Nothing here supports a claim about Princeton Vero. A is the direct local same-prompt control. No CE / margin / auxiliary objective is added anywhere (archived v1–v3 stay archived).

## 2. Independent assessment of the hypothesis (INFERRED, from existing negative evidence and the code)
Existing evidence (OBSERVED, plan §2): 18 RL arms, 2286 groups, 362 accepted updates, no verified stable accuracy advantage; zero-variance groups, the strict format gate and the class prior were all observed; Adam60's T1 margin was negative on a pair and both real images chose FLAIR; auxiliary margin losses did not raise accuracy.

Why pairing might help (mechanism, PROPOSED): when the model answers FLAIR for both modalities, a same-prompt T1 group is all-wrong (zero variance, no signal) while a paired group F2T2 has FLAIR responses correct and T1 responses wrong, so the T1 image receives a negative advantage on its FLAIR answer. This is the only way B can add signal.

Key risk — cross-prompt baseline = prior/difficulty shift, not discrimination (INFERRED). The extra within-group variance in B comes from *prompt difficulty and the class prior*, not from the model telling the two images apart. The group mean is then a prompt-mix baseline, so B reweights tasks by difficulty. Three rival explanations must be separable: (a) B only upweights the T1 term (a reweighting, not "pairing"); (b) B shifts the shared answer-token prior (FLAIR down everywhere) and so raises T1 recall while dropping FLAIR; (c) B's gain is image-independent. One pair, one rollout and one step also gives a single noisy gradient: with Adam without bias correction and lr 1e-5 the step may simply be too small to move strict-greedy accuracy, in which case the honest result is "no detectable difference".

What falsifies Paired-Prompt in this experiment (each is a STOP, no coefficient search follows):
1. B has no more admitted groups/responses than A, or cos(gA,gB)≈1 with no B−A behavioural difference (no mechanism).
2. B−A is not positive on both strict-greedy sequence balanced accuracy and T1 recall.
3. B's FLAIR recall falls below the Parent, or Dallas accuracy falls below the Parent or A (retention).
4. The T1 gain is image-independent: the 8 swap aliases show B has fewer valid opposite-class changes than A (prior shift, not visual evidence).
5. Candidate mean margins and strict greedy disagree in sign (reported, interpreted as a prior/format effect, never as accuracy).
A GO is only "CONTINUE_CANDIDATE_REQUIRES_NEW_AUTHORIZATION" on a 4-person engineering dev set (one sequence row moves a recall by 1/12); it proves nothing about generalisation and authorizes nothing.

## 3. Execution (single combined human scope; ordinary CPU work now, no native run)
Order (driver `accuracy_recovery_driver_v1`): restore parent (state-only preflight) → ONE F4+T4 rollout → bundle committed → own-old scoring on the shared parent with rollout/scorer selected-logp parity ≤ .002 (this is the integrated on-policy numerical check, before any step) → **Phase 1** for A then B: restore R0, serial per-response backward (one image graph at a time), Adam/RNG pre-restore audit **and** `transaction_validate(parent_snapshot, False)` for the full trainable parameters, archive the full weighted gradient to disk → persist the independent comparison (norms, cosine, admitted sets; `None` explicit for an inactive arm) → **Phase 2**: only now, per arm restore R0, read the authenticated gradient back from disk, ≤1 Adam update, atomic boundary with parent Adam state hash, recurrence inputs and optimizer before/after state, restore R0. If any Phase-1 step fails there are zero optimizer updates. Parent directory is hashed before/between arms/after and must be unchanged.

Preflight decision (Codex finding 4): a pre-rollout probability probe would need up to two extra F32 forwards and add nothing the shared8 on-policy own-old parity does not (and the parity covers the real sampled contexts). So the preflight is honestly **state-only** (full parameters + Adam + Python/MLX RNG identity after restore; adapter byte hash is informational because serialization may differ) and the numerical gate is the .002 own-old parity before the first step. Limit: no numerical statement exists before the rollout; a failing parity stops the run with zero updates. No separate qualification phase and no old consumed qualification proof is used.

Independent recompute from saved artifacts: gradient L2/cosine from the archive; Adam recurrence `adam_next_state(parent optimizer state, gradient archive)` must equal the arm boundary `optimizer/` state and its delta must equal the adapter change (`accuracy_recovery_v1.adam_next_state/compare_adam_after`).

## 4. Evaluation (separate exact scope, read-only, after authenticated receipts)
`accuracy_recovery_eval_v1`: Parent52, committed A, committed B on the frozen internal-dev records (24 sequence + 16 Dallas real rows + 8 swap aliases reusing real outputs, 0 extra calls), strict temperature-0 greedy (F32 final processor), strict canonical candidate mean/sum/first, via the existing `run_checkpoint` kernels. At most 3 models, serial, each released; a no-update arm is an alias of the Parent and is not re-run; weights/Adam are audited unchanged before RNG restore. Old final100 and new16 are never read (refused by name). The analyzer applies the stop rule above to saved rows only.

## 5. Where things live
Actual outputs and private payloads: canonical project root `<R>/local_data/accuracy_recovery_v1/…` (R derived from the git common dir, symlink/traversal/worktree paths refused) so worktree cleanup cannot delete them. Source and runtime config may stay in the worktree.

## 6. Limits
CPU tests use a numpy fake; the mixed-context native F32 rollout/scorer/backward and the evaluator kernels on real weights are NOT_VERIFIED until the human-authorized run. `CROSS_MODEL_REVIEW` remains Codex's task. Nothing here is an accuracy result.
