# Actual Vero-to-Brain-MRI micro pilot: restore recovery, A adoption, B update and DEV24

Date: 2026-10-21. Evidence: **OBSERVED_NATIVE / INDEPENDENTLY_REPRODUCED**, with limitations. Mission alignment: finish the existing matched Direct Vero-style versus Paired-Prompt adaptation pilot; no new loss/data/architecture.

## Main result

All three endpoints were evaluated on the **same frozen internal DEV24**: four TRAIN-disjoint but previously explored DEV patients, 12 FLAIR and 12 contrast-enhanced T1 images. Same F32 contract, original prompt/processor, temperature zero, 256-token cap and unchanged strict Vero format/parser. All 72 actual records are durably committed and independently authenticated.

| Endpoint | Strict MRI accuracy | FLAIR recall | Enhanced-T1 recall | Balanced accuracy | Actual optimizer updates |
|---|---:|---:|---:|---:|---:|
| Original Parent Adam52 | 10/24 (41.67%) | 5/12 (41.67%) | 5/12 (41.67%) | 41.67% | 0 |
| Direct Vero-style A | 11/24 (45.83%) | 5/12 (41.67%) | 6/12 (50.00%) | 45.83% | 1, previously saved and now qualified/adopted |
| Paired-Prompt B | 12/24 (50.00%) | 5/12 (41.67%) | 7/12 (58.33%) | 50.00% | 1, newly executed |

B is **+8.33 percentage points versus Parent and +4.17 points versus A** in strict/balanced accuracy. T1 recall is +16.67 points versus Parent and +8.33 versus A; FLAIR recall is unchanged. These are two and one additional correct predictions respectively, not a demonstrated stable population effect.

Independent strict parsing agrees for all 72 outputs. B versus A has one valid-wrong→correct transition and no lost correct answer. B versus Parent has two valid-wrong→correct transitions and no lost correct answer; it also exchanges one wrong→invalid and one invalid→wrong case. The two additional correct cases are enhanced T1. Each endpoint has nine invalid generations, all reaching the cap. The gain is not explained by fewer invalid outputs or a relaxed parser.

**Image grounding remains unresolved.** B predicts T1 on 9/24 inputs versus 8/24 for A/Parent (invalids included in denominator). A class-preference shift could explain a one-case boundary crossing. No new candidate/margin, swap/null-image or Dallas evaluation was authorized. Do not claim improved candidate discrimination, protected Dallas retention, anatomical reasoning or independent-patient generalization.

## Native restore and legitimate adoption of saved A

The original A outer transaction remains incomplete. The saved one-step checkpoint was CPU-qualified against the authenticated original parent, joint gradient and R1; all 504 weights/first moments were bitexact and second moments passed predeclared bounds. Fresh human permission explicitly allowed conditional adoption.

The new native Q passed three actual copy-isolation fixtures (float32/uint32/uint64), two full snapshots and restores, 1942 parameter fingerprints, 1010 Adam arrays and complete Python/MLX RNG checks. It used 7074 leaf copies and no model forward/generation/backward/update.

A was separately loaded during E; its full parameter inventory, policy/Adam/RNG identity and Adam53 matched the old receipt before inference. The new private adoption record binds that native proof, the fresh human grant and Q/E actual completions. Label: **SALVAGED_FROM_INCOMPLETE_TRANSACTION**. No old journal commit was written and no A update was repeated.

The new restore path detaches mutable containers, explicitly copies trainable/Adam/RNG leaves and restores RNG without reinstalling Adam. B uses an independent process from original disk Adam52 instead of rolling the updated A process back. This qualifies the actual isolated execution path, not every historical rollback implementation or the unique cause of the previous combined integrity failure.

## B really trained under the matched protocol

- Same original P_calibrated13 / Adam52 / boundary114 parent and unchanged real TRAIN pair plus two Dallas engineering controls.
- Original Common16/own-old/R1 reused; zero regenerated answers and zero refreshed old-score forwards.
- Same reward, loss, LR, LoRA, F32, task weights and response exposure. Only same-prompt versus Paired-Prompt grouping differs.
- A and B each admitted one sequence group and one Dallas group, eight responses and eight backwards, followed by one Adam52→53 update. The gain did not come from giving B more updates.
- New B actual cost: eight gradient forwards/backwards, 2712 TF positions, one update. Full checkpoint/Adam/RNG saved and journal committed. Independent CPU recomputation of its 504 weights/1008 moments passed; R1 matched all 628 serialized RNG leaves.
- B adapter SHA and complete receipt hashes are in the aggregate; all actual tensor payloads remain private locally. Parent and historical A files remain unchanged.

## Actual cost and Mac resources

| Phase | Native elapsed |
|---|---:|
| Q restore qualification | 81.84 s |
| B fresh load + training + checkpoint/release | 137.37 s |
| All DEV24 evaluation, three serial endpoints | 2384.31 s |
| Whole new controller run including handoffs | 2635.56 s (43.93 min) |

E completed inside its frozen 2400-second operation-start window. There were five new serial model loads, zero recoveries/retries, no cache recovery, no extra A training and no expansion.

DEV generation used 2932 / 2800 / 2756 returned tokens for Parent / A / B, total 8488. B used 6.0% fewer tokens than Parent and 1.6% fewer than A, but generation wall times were 692.20 / 681.16 / 697.03 seconds: **no latency improvement is demonstrated**. Do not compare old A whole-stage time (which included common generation) directly to B time.

Independent resource aggregation covers 18,958 child/guardian observations: 16,470 GREEN, 2,488 YELLOW, zero STOP/RED. Exact wait4 child RSS peak 5.9696 GiB; MLX peak 10.35595 GiB; minimum available 5.26535 GiB and free 28%. Swap 14.81317→peak15.62701→post15.51764 GiB (peak growth 0.81384 GiB). RSS and MLX are different counters and must not be added as a physical-memory estimate.

All five model sessions passed allocator release (active MLX≤1050 bytes), all phase/root processes exited0 and were reaped, shared lock was free. No TERM/KILL, other-app termination, reboot or OS-protection changes occurred. Thermal telemetry remained unavailable; these observations are not a future no-OOM guarantee.

## Decision and remaining limitation

**POSITIVE_ENGINEERING_SIGNAL_NO_AUTOMATIC_SCALE.** B improved this exact small internal comparison while retaining FLAIR recall. It is reasonable to discuss one further tightly bounded matched replication, but this result does not justify large-scale training or a reproducibility/generalization claim. No further execution is authorized by these results.

Only one TRAIN patient pair and one shared draw were used; four previously explored DEV patients provide limited uncertainty information. Dallas retention and independent new16 prediction results remain unmeasured. The next decision should preserve the same comparison and resolve whether the extra T1 correction persists across patients without trading away FLAIR/Dallas. No new experiment is started here; no preprocessing, loss redesign or sealed-test tuning was performed.

## Evidence and reviewers

- [Actual aggregate](../results/SAFE_MICRO_RECOVERY_QBE_ACTUAL_RESULTS_V1.json)
- [Final independent native/accuracy review](../results/SAFE_MICRO_RECOVERY_QBE_FINAL_ACTUAL_INDEPENDENT_REVIEW_V1.json)
- [A adoption and full resource review](../results/SAFE_MICRO_RECOVERY_QBE_ADOPTION_RESOURCE_INDEPENDENT_REVIEW_V1.json)
- [Q actual review](../results/SAFE_MICRO_RECOVERY_Q_NATIVE_ACTUAL_INDEPENDENT_REVIEW_V1.json)
- [B actual review](../results/SAFE_MICRO_RECOVERY_B_NATIVE_ACTUAL_INDEPENDENT_REVIEW_V1.json)

Source commit: `748922f61bf12f75d76c62322ec92dd9ae3eee16`. Exact executed scope FILE: `8be74ec824bc4daa9a11497dd07ba7501c98cc3f261f4169c2433f3840dda1c7`.

Implementation/execution: Codex/root and Codex/root/native_resume_implementation. Independent saved-outcome reviewer: Codex/root/recovery_exact_scope_review; independent A CPU author: Codex/root/resource_v2_final_review. Claude was not logged in; **CROSS_MODEL_REVIEW=NOT_VERIFIED**. Frozen plan/lock unchanged; facts appended to experiment log. No raw patient IDs, images, response/token arrays, weights, Adam/RNG payloads or credentials are committed.
