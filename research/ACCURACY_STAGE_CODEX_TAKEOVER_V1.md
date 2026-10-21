# Bounded MRI accuracy stage — Codex implementation handoff

## Execution status

Source and CPU preparation only. New model loads, generated responses, native
backward calls, optimizer updates and measured accuracy gains in this stage: **0**.
Execution requires a fresh independently reviewed compound scope and explicit
human permission for cache recovery, A/B training and matched evaluation.

Claude Code 2.1.306 (`claude-sonnet-5-5`) actually implemented the processor
qualification, exposure planner and multistep core in
`f06a8f33690427915ad97459d3c8720a65092acc`. Its normal CLI session then reached
its quota. The researcher explicitly assigned the remainder to Codex; no waiting
for or bypassing the Claude quota is required. Codex implemented the live bridge,
local resources, cache-v4 continuation and exact execution admission.

## Data and numerical contract

The existing z045 TRAIN pool contains 56 images: both acquisition-labelled
modalities from each of 28 people. The legacy 28-image pool and historical PNGs
are preserved. No source volume is re-rendered by this stage.

The saved CPU processor qualification covers all 56 images. Each produced an
expanded 213-token prompt and 154 merged visual tokens (grid 1×28×22), with 56
distinct decoded-pixel/processor-pixel identities. All 28 modality swaps changed
the pixels while preserving the prompt. This is a NumPy reproduction of pinned
processor functions, **not a new native F32 backend qualification**. Native F32
input fingerprints and the unchanged 0.002 rollout/scorer bound are checked by
the execution path before using an on-policy update.

## One frozen comparison

Parent: authentic `P_calibrated13 / Adam52 / boundary114`, preserving its adapter,
full Adam and Python/MLX RNG. This is not an Adam60 recovery or a new base model.

| Item | Frozen ceiling per arm |
|---|---:|
| Whole patient rounds | 4 |
| People / sites in the sequence prefix | 4 / 4 |
| Sequence images | 8 (4 FLAIR, 4 enhanced T1) |
| Unchanged Dallas rows | 8 (balanced classes) |
| K4 groups | 16 (8 sequence, 8 Dallas) |
| Optimizer updates | At most 8, never a target count |
| Logical input + response generation charge | 36,160 tokens |
| Logical returned-response exposure | 16,384 tokens |
| Training operation-start window | 3,600 seconds |

Only 4/28 people and 8/56 sequence images are scheduled; the full qualified pool
is not claimed as trained. Both arms use exactly the same prefix, task exposure,
reward, learning rate, LoRA, persistent Adam and F32 policy. A uses same-image K4
groups; B uses F2/T2 paired-prompt groups within each sequence block. Dallas
remains unchanged same-prompt K4 and **ENGINEERING_ONLY**.

The first F4/T4 rollout is generated once from the common parent and shared.
Every later unit uses its own arm's current policy. Total new training response
tokens are capped at 30,720 (2,048 shared once plus 14,336 per arm); at most 128
GSPO backwards and 140,640 teacher-forced positions. Inactive groups retain their
actual rewards and cause no fabricated advantage, momentum-only update or retry.

Parent/A/B terminal evaluation reuses the same frozen internal DEV40, strict
256-token greedy decoding and canonical candidate mean/sum/first readouts.
Ceilings: 120 greedy calls, 360 candidate API calls, 30,720 returned tokens and
153,000 teacher-forced positions. Eight image-swap aliases reuse identical real
inputs and require no additional model calls. The primary benefit requires
sequence balanced accuracy, enhanced-T1 recall and candidate balanced accuracy
to improve against both A and Parent, with FLAIR and Dallas protected.

These repeatedly inspected four sequence people and four Dallas cycles support
engineering diagnosis only. No historical final100 output or new16 prediction
is used. Current availability of the private new16 payload/seal is not
re-attested by this stage; no independent-patient performance is claimed.

## Confirmed implementation fixes

1. Shared rollout consumption now checks live parameters **and Adam/R0** against
   the trusted parent after the resource guard. An isolated original six-case
   fixture changed from 5 PASS / 1 RED to 6 PASS without changing assertions.
2. The historical loader assigns an instance `generate` callable. The stage
   explicitly preserves it under a legacy name so it cannot shadow the new
   per-unit generation method. A synthetic reproduction failed before the fix.
3. Per-unit own-old probabilities and gradient evidence are durably included in
   the checkpoint receipt's file hashes. The previous core accepted these as
   arguments but did not save them.
4. Model release clears full-parameter/Adam snapshots and active input references
   before invoking the historical close routine. A synthetic retention test
   reproduced the prior leak.

The bridge uses the unchanged GSPO loss AST and serial per-response autograd,
the inherited task grader and group gate, and the existing Adam implementation.
There is no CE/margin auxiliary objective, architectural change or forced correct
rollout. Full frozen-base integrity and finite Adam/parameter checks surround
the transaction. An unfinished reservation is preserved and is not auto-retried.

## Cache and resource conditions

The fresh v4 cache scope adopts 406 retained units plus the saved first quantized
unit. Remaining work is at most 343 direct and 343 module-path quantizations;
the cumulative historical ceiling remains 688. Every payload and both final
shards must match historical SHA256; the first mismatch stops execution.

The local power exception records the actual BATTERY state and permits this
round above the proposed 20% floor. Global training policy is unchanged. Cache
headroom remains 5.5 GiB; model admission is 8 GiB/free35%, resident operations
6 GiB/free30%. Memory pressure, RSS, MLX, thermal, disk and global flock guards
remain active. Original swap baseline 13.96966796875 GiB is carried, with net
growth at most1.5 GiB and absolute swap at most16 GiB. Short shortages HOLD at a
safe boundary; risk signals STOP. This is not a promise against OS failure.

The compound operation-start limit is 10,800 seconds including cache, holds,
load, training and evaluation. Component maxima sum to12,600 seconds, so the
global limit is stricter and may prevent full completion. These are ceilings,
not a measured F32 training ETA; F32 multistep rate is **NOT_MEASURED**.

## Evidence and next action

- Processor result: `results/NATIVE_PROCESSOR_QUALIFICATION_CPU_V1.json`.
- CPU implementation: `multistep_gspo_core_v1.py`, `accuracy_stage_native_v1.py`,
  `accuracy_stage_entry_v1.py`, `accuracy_stage_resources_v1.py`.
- Cache continuation: `accuracy_recovery_gpu_cache_v4.py`.
- Private manifests, exact scopes and receipts remain in canonical `local_data`;
  patient images, IDs, prompts, labels, weights, Adam and RNG are never pushed.

The next action is a **single explicit authorization request for the reviewed
cache → finite A/B → DEV40 scope**. Passing CPU tests, having a nonzero gradient,
or reaching more updates cannot establish MRI accuracy improvement.

Final independent source/CPU/exact-scope review: **PASS_WITH_LIMITATIONS**, isolated Codex `/root/full_arm_execution_contract`; see `results/ACCURACY_STAGE_FINAL_INDEPENDENT_CPU_SCOPE_REVIEW_V1.json`. Metadata reprepare equals frozen scope (85 source and13 metadata bindings). Root final targeted verification:79 PASS in8.94s. Native NOT_RUN. Adopted cache unit0 lacks its historical ordinary receipt and six metrics; these are not fabricated, and exact historical payload reauthentication remains mandatory.
