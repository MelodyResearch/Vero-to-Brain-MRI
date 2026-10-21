# Safe micro recovery: saved A qualification, fresh B and DEV24

Date: 2026-10-21. Mission alignment: finish the existing Vero-style same-prompt versus Paired-Prompt MRI pilot; no new loss, data, model architecture or research direction.

## Actual state and evidence tiers

- **OBSERVED:** previous Stage 2 passed; A executed eight backwards and one Adam52→53 update. Its checkpoint was saved before the parent rollback failed. Old outer A transaction remains incomplete. B and DEV24 have not run.
- **REPRODUCED_CPU:** installed optimizer setter/step semantics reproduce mutable snapshot-container aliasing. This is a confirmed mechanism; the previous generic native integrity assertion does not identify it as the sole native cause.
- **REPRODUCED_CPU:** independent forensic recomputation authenticated the original parent, common16/own-old/R1, saved joint gradient and A payloads. All 504 weights and first moments matched bit for bit. The largest second-moment difference was 1.1617190264e-38, within bounds declared before computation; zero failing tensors/elements. All 628 serialized Python/MLX RNG leaves matched R1.
- **NOT_RUN:** new native restore qualification, B update and Parent/A/B DEV24. No new native model call or optimizer update was made during preparation. No MRI accuracy improvement is established.

See [saved A CPU qualification](../results/SAFE_MICRO_A_SALVAGE_CPU_INDEPENDENT_QUALIFICATION_V1.json) and [previous actual outcome](../results/SAFE_MICRO_DYNAMIC_V3_ACTUAL_RESULTS_V1.json).

## Actual fixes

1. New `safe_micro_recovery_native_v1.RecoveryBackend` integrates detached dict/list/tuple restoration and explicit materialized array copies for 504 trainable leaves, Adam and RNG. Frozen base leaves are borrowed, and all 1942 parameter fingerprints are checked. Native array copy isolation is a pending qualification, not inferred from NumPy.
2. RNG-only restoration copies/checks RNG without copying or reinstalling unchanged Adam. The independent reviewer identified the earlier unnecessary full-state copy at every DEV row; its regression failed before this correction.
3. The native dispatcher now requires the exact nested adoption contract used by the entry. A mismatched top-level boolean cannot authorize adoption.
4. B runs in a new process, loading the original disk Adam52. Its arm AST preserves the original scientific body and removes only the final in-process parent rollback. It never loads A as its training parent.
5. Snapshot, active input, rollout-context and optimizer references are cleared before allocator measurement. A successful process exit cannot replace allocator verification; each session must report model absent and active MLX memory below 128 MiB.

The old frozen implementation and evidence remain byte-unchanged. No previous A journal commit is fabricated.

## Saved A eligibility

Label: **SALVAGED_FROM_INCOMPLETE_TRANSACTION**. The CPU record only makes A eligible for a fresh explicit adoption decision and native verification. It does not repair the old transaction or grant execution permission.

The new scope conditions prospective adoption on fresh human permission plus Q PASS. Q checks the original parent restoration. A's own full policy/Adam/RNG load is independently checked in E immediately before A inference. If that check fails, no completed fair A/B accuracy comparison is claimed, even if B has been saved.

A's adapter/config/full optimizer/RNG and original parent remain private. The source verifier streams bounded tensor inputs instead of loading a base model; its bounded-input claim is not a total-process-memory bound.

## New Q → B → E contract (pending fresh authorization)

| Phase | Actual operation | Maximum new work |
|---|---|---|
| Q | One original Adam52 disk load; 3 tiny F32/U32/U64 copy-isolation fixtures; 2 full snapshots and 2 restores; full state fingerprints | 10,000 array-leaf copies; 0 model TF/generation/backward/update; 900s operation-start window |
| B | Separate original Adam52 load; saved common16, original rewards, own-old and R1; Paired-Prompt grouping | 16 gradient forwards/backwards; 9,024 TF positions; 1 update; 0 new responses or old-score refreshes; 1,800s window |
| E | Parent, saved A, saved B, serial single-model loads; frozen balanced DEV24 | 72 strict greedy calls; 18,432 returned tokens; 0 candidate scoring; 2,400s window |

The Q leaf-copy cap does not include fixed-source fingerprint/equality reads. Scientific arithmetic, reward, LR, LoRA, F32 and fixed inputs are unchanged. Inactive groups are not replaced; zero-update B can be a verified Parent alias.

Previous consumption is charged once: 673 returned tokens, 30 scorer forwards, 9,655 TF positions, 8 backwards, 1 optimizer update and 2 loads. Including this recovery, maxima become 19,105 returned tokens, 46 scorer forwards, 18,679 TF positions, 24 backwards and 2 total updates. New load attempts ≤5; old+new ≤7. No Common16 or A update is repeated. No cache recovery or old Stage-2 MRI demonstration is repeated.

All waiting counts against the original cumulative new-phase windows (900/1800/2400s, 5100s combined). At most two cooperative recoveries; only complete, hash-verified operations can be reused. An incomplete reservation is INDETERMINATE and cannot be replayed. Fatal termination is not retried.

## Mac protections

The previously approved local GREEN/YELLOW/RED controller is unchanged: no independent 9/10 GiB or ordinary-swap STOP. CRITICAL/severe risks block work and release only this experiment's process. RSS≤14 GiB, MLX≤12 GiB, cache≤256 MiB, disk≥26 GiB, battery≥20%, severe-low-memory and coupled sustained swap/memory/disk protections remain. Independent child monitoring and lock/reaping/closed-log witnesses are required. GPU atomic operations and monitor sampling retain blind spots; no OOM or kernel-panic guarantee is made.

## Accuracy interpretation

DEV24 is the previously explored four-patient engineering set, 12 FLAIR and 12 enhanced T1. Report all three endpoints, strict overall accuracy, per-class recall, balanced accuracy, invalids and actual updates. Report B−Parent and B−A with error transitions; small-sample class-preference changes are not proof of image grounding or generalization. Dallas retention is **NOT_MEASURED** by this sequence-only evaluation. No final100 or sealed new16 outputs are accessed.

## Verification and execution ownership

Root independently ran 40 focused CPU tests covering new entry/native/forensic/restore plus previous durable-resume compatibility. The new native tests include mocked Q/B dispatch, release-before-commit, no duplicate operation on replay, B formula equivalence, weak-reference cleanup, nested authorization rejection and RNG-only restore. These are CPU tests, not new native PASS evidence.

Implementation: Codex/root and Codex/root/native_resume_implementation. CPU A forensic author: Codex/root/resource_v2_final_review. Final independent exact-scope reviewer: Codex/root/recovery_exact_scope_review. Claude CLI reports loggedIn=false; CROSS_MODEL_REVIEW=NOT_VERIFIED.

Frozen scope FILE SHA256: `8be74ec824bc4daa9a11497dd07ba7501c98cc3f261f4169c2433f3840dda1c7`.
Canonical scope SHA256: `cd9f6038a680ccb02b0e857f53d9fae97855268927c7a486fee822fd34075f3e`.
99 source/runtime files and 24 metadata files are bound; payload bindings are verified before actual use. Current research plan and lock are unchanged.

[Independent exact-scope review](../results/SAFE_MICRO_RECOVERY_QBE_EXACT_SCOPE_INDEPENDENT_REVIEW_V1.json): PASS_WITH_LIMITATIONS. [CPU preparation aggregate](../results/SAFE_MICRO_RECOVERY_QBE_CPU_PREPARATION_V1.json).

The single next requested action is fresh human authorization for this exact Q→conditional A adoption→B→DEV24 scope. No grant from the failed previous run transfers.
