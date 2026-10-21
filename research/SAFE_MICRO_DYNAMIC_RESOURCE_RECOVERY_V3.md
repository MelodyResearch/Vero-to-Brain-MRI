# Micro pilot: dynamic OS color and exact committed recovery v3

**IMPLEMENTED / TESTED_CPU only; new model loading, training and accuracy NOT_RUN. Fresh exact human permission required.** Prior STOP attempts are closed evidence, never resumed by this version. Stage1's exact historical cache is reused without quantization.

## Local resource contract

| State | Action |
|---|---|
| GREEN: OS NORMAL (1) | Continue serial computation if independent emergency protections pass. |
| YELLOW: OS WARNING (2) | Continue with one model and the unchanged low MLX cache (256 MiB). Available<9/10GiB, free<50%, and ordinary swap growth are not standalone blocks. Coupled risk or unknown sensors pauses initiation of new work. |
| RED: OS CRITICAL (4) | Block new computation. Request cooperative child release at its next safe boundary; no resident-model waiting for recovery. |

No global policy or historical source is changed. RSS14GiB, activeMLX12GiB, disk26GiB (6forecast+20reserve), battery20%, unknown-power/other-heavy/thermal checks remain. Severe available<2GiB, available<4GiB for30s, or pressure-free≤15% override even GREEN. Allocation failures/nonfinite tensors remain fatal. These are emergency conditions, not a9GiB admission forecast.

**Explicit swap change:** the previous original-baseline15.46966796875GiB and absolute16GiB ceilings are recorded historical context and are no longer independent STOPs in this local scope. Swap remains sampled. Two consecutive intervals≥5s at≥0.25GiB/min combined with disk<30GiB, or available<4GiB AND free<20%, trigger emergency STOP. Ordinary swap alone cannot stop a run; disk<26GiB always stops. No system settings, swap files or unrelated applications are changed.

Independent non-MLX guardian samples every~2s. RED/unknown/coupled pause requests SIGUSR1 **only to its owned unreaped child**; the child sets a flag and releases at a safe boundary. If unresponsive, TERM after10s, KILL afteranother10s. Hard danger/logging failure sends TERM immediately, including escalation after a cooperative request. Forced termination, missing exit/logs, monitor failures or ambiguous operations never auto-resume. Parent cannot observe child MLX allocator; child readings/release receipts remain separate. Sampling and atomic kernel/GPU calls cannot guarantee against OOM or OS failure.

## Durable recovery and its limits

A single-writer journal binds scope, source/input identity, parent and operation identity. It fsyncs a reservation before work and publishes a hash-bound manifest last, after immutable payload/state files and directories are synced. Completed results are hash-checked and reused. Spent counters persist; reservations are never refunded. Controller and model-load ledgers share the original run/deadline.

Supported recovery milestones:

- Stage2: complete fixed MRI/F32 qualification with real allocator release.
- Stage3: complete common16 real responses, own-old scores and R1; then each complete arm checkpoint/Adam/RNG/receipt. Arm A is not recomputed when recovering B. Both arms still restore the exact original Adam52 parent independently.
- Stage4: each complete frozen greedy query with output/RNG/counters and previous session release proof. No repeat inference for committed rows.

**A claimed operation without its complete commit is INDETERMINATE_NO_REPLAY.** A partially saved generation, backward, optimizer step or checkpoint is preserved and blocks automatic continuation. This is not lossless token-level recovery. No synthetic advantage, resampling, checkpoint substitution, or silent abandonment is allowed. A recovered arm checkpoint is not used as the other arm's parent.

Nonresident controller waits for3 consecutive samples permitting work, every5s; neither9/10GiB nor120s ordinary-HOLD timeout exists. Waits count against the same original stage900/1800/2400s and compound5100s operation-start windows. Only exact closed/reaped exit75 with pause receipt, unchanged closed logs, unlocked model lock, no forced signal and no monitor error can be recovered. Every claimed scientific operation must also be complete or unstarted. Controller restarts reconcile the previous attempt before dispatch; missing/fatal exits block.

At most2 automatic recoveries,3 controller launches, and **7 actual model-construction attempts total** (original5 planned loads plus2 recovery loads). This is an explicit bounded loading-cost change, not extra training. Deadlines and call/token budgets never reset. True fatal STOP is not retried under this scope.

## Fixed scientific comparison

Parent: authentic P_calibrated13 / Adam52 / boundary114. Same first whole TRAIN round: one z045 FLAIR/T1 pair + two Dallas engineering-control records. Same real pixels, prompts, F32 contract, reward, GSPO loss/gate, LR, LoRA, persistent Adam/RNG and joint .5sequence+.5Dallas weighting. Only A same-prompt vs B paired-prompt grouping differs. No supervised auxiliary loss or new sampling intervention.

| Stage | Maximum scientific work |
|---|---|
|2|2 generated MRI responses,512 returned tokens,6scorer forwards,2808TF positions,0updates.|
|3|16shared responses once,4096returned tokens,48scorer forwards including32gradient forwards,27072TF positions,32backwards,at most1Adam update/arm.|
|4|All frozenDEV24:12FLAIR+12T1,Parent/A/B strictgreedy,up to72generations/18432returned tokens,0candidate scoring. Exact verified0-update aliases labeled.|

Combined23040returned tokens/54scorer forwards/29880TF positions/32backwards/2updates. No extra scientific calls on recovery. StagePASS/release/closedexit and fresh resource checks precede the next stage. PARTIAL probability/image-sensitivity evidence is not PASS and blocks progression. No oldfinal100 or sealednew16 predictions are used.

DEV24 is a repeatedly explored4patient engineering pilot. Primary readout:accuracy,FLAIR/T1recall,balancedaccuracy,invalidresponses,actualupdates,time/RSS/MLX/swap. Candidate accuracy and Dallas retention are **not measured** by this greedy-only scope. No automatic expansion or generalization claim follows.

## Evidence and status

Implementation: `safe_micro_dynamic_resources_v1`, `safe_micro_journal_v1`, `safe_micro_watchdog_v2`, `safe_micro_pilot_entry_v3`, `safe_micro_pilot_native_v2`, `safe_micro_training_v2`.

CPU tests exercise color transitions, cooperative owned-child termination, durable claims, interrupted/ambiguous operations, unchanged scientific endpoints and resumed A/B counts. They do not establish native execution, MRI accuracy or hardware safety. Claude Code2.1.306 is installed but actual authentication is unavailable; implementation/review this round is Codex, not Claude.

Frozen primary plan and historical evidence remain unchanged. Exact independent review/scope and human execution permission are separate artifacts; no local approval flag is human authorization.

Final frozen scope FILE SHA256: `da7ef34ccf171922d601bb3160c99758c0873a518e58ea981f39e8f86012c279`; canonical: `bc01fd061005f888fe2d9236f1c23f5d9e654be4d36927f23d78f280ff6c376b`. Independent review: **PASS_WITH_LIMITATIONS**, Codex `/root/resource_v2_final_review`, exact95source/20metadata bindings. [Preparation evidence](../results/SAFE_MICRO_DYNAMIC_RESOURCE_V3_PREPARATION_ACTUAL_V1.json) and [independent review](../results/SAFE_MICRO_DYNAMIC_RESOURCE_V3_INDEPENDENT_REVIEW_V1.json). Fresh human execution permission remainsNOT_ISSUED.
