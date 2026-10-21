# Stage 2–4: smallest real Vero-to-MRI pilot

Status: **SOURCE / TESTED_CPU; native loading, training and accuracy NOT_RUN.**

This version implements the researcher's 2026-10-21 direction. Stage4 now means **DEV24 greedy accuracy**, not the optional larger Small-Pilot expansion. The frozen primary research plan and lock are unchanged. The Stage1 cache grant is consumed and cannot authorize this experiment.

## Fixed comparison

Parent: authentic P_calibrated13 / Adam52 / boundary114. Original z045 TRAIN inputs; one paired patient, FLAIR+enhancedT1 and two matched Dallas engineering-control rows. No new loss, reward, LR, LoRA, architecture or forced-pair sampler. A/B use the already versioned micro batch granularity:0.5sequence+0.5Dallas before at most one total Adam step each, inactive block not renormalized. Only sequence grouping differs between arms. This is a local Vero-style adaptation, not a claim of full official Vero parity.

## Frozen stages and maxima

| Stage | Operations | Returned tokens | TF positions | Backward | Updates | Operation-start window |
|---|---|---:|---:|---:|---:|---:|
|2—load/forward|One parent load, two real-image responses, six scorer forwards|512|2,808|0|0|900s|
|3—micro A/B|16 shared responses once;16 own-old+at most32 gradient forwards;one resident model restored separately for each arm|4,096|27,072|32|2 total /1 each|1,800s|
|4—accuracy|Up to3 serial model loads; all24 real sequence DEV inputs each;72 greedy calls,0candidate forwards|18,432|0|0|0|2,400s|

Combined maxima:23,040returned tokens,54scorer forwards,29,880TF positions,32backwards,2updates,5serial model loads,5,100s operation-start window. Zero cache quantization. Limits are not targets; each stage must actually PASS, release, exit and pass fresh resources before progression. An atomic native call/cleanup can extend beyond an operation-start deadline; these are not hard-kill or guaranteed-completion times.

Stage2 preserves exact existing F32 probability tolerance0.002. Two saved responses are scored under their own and opposite true images (four forwards). Two further forwards score the same predeclared canonical FLAIR token sequence under both images. The latter is a same-target image-sensitivity check, not a complete class margin or anatomical-grounding proof. A zero measured effect is PARTIAL and stops progression. Responses may be wrong; no reward or answer is manufactured.

Stage3 separately restores parent weights/fullAdam/R0 for A/B, persists the common post-generation R1, saves gradients and atomic terminal checkpoints/optimizer/RNG, and enforces at mostone update each. Inactive or zero-gradient arms remain zero-update. Both sequence and Dallas rollout/scorer parity are checked on real common responses before updates using the unchanged mechanism.

## Accuracy readout and interpretation

DEV24 is all24 previously frozen sequence rows,12FLAIR+12enhancedT1 from4TRAIN-disjoint DEV people. No new outcome-dependent subset; no old final100 or sealed new16 predictions. Parent/A/B use the same strictVero parser, originalprompt,F32,temp0,256cap.

Report overall accuracy,FLAIRrecall,T1recall,balancedaccuracy,invalidresponses,actualupdates,token/time/memory costs andB−A/B−Parent. A zero-update arm reuses Parent inference only after exact unchanged policy identity and terminal-receipt verification; aliases are labeled, not counted as separate calls. The greedy-only scope does not measure candidate accuracy or Dallas retention.

Predeclared positive screening rule: B balancedaccuracy andT1recall must both exceed Parent andA; FLAIRrecall must not fall relative to either. Even then, only a small explored-DEV engineering signal is supported. Image-grounded generalization, Dallas retention and scaling remain unproved; no automatic expansion follows. No improvement is preserved as a negative/inconclusive pilot result, not followed by an automatic extra step.

## Authority, resources and state safety

The new compound grant must explicitly include stages2,3,4 and exact reviewed scope. Separate child permission markers and durable exit witnesses prevent rerunning completed or failed stages. No runtime backend/test override, stage skip or retry is exposed. Each model load requires at least10GiB available and40%pressure-free; resident floor6GiB/free35%,RSS≤14GiB,MLX active≤12GiB,localcache256MiB,original swapbaseline13.96966796875GiB/netcap1.5/effectivecap15.46966796875GiB/absolute16GiB,20GiBdiskreserve,battery≥20%,singleGPUflock and OS/thermal safeguards. Ordinary swaprate alone does not STOP; coupled sustained deterioration invokes boundedHOLD, genuinehardcap/CRITICAL STOP. No OS changes or unrelated app termination.

At08:01:29UTC actualread-only snapshot:available10.4103GiB/free57%,swap14.63458GiB,OS1,battery70%BATTERY,nootherheavy/lockfree,disk294.69GiB. This passed the loadingpreflight at observation; execution must resample. ThermalSENSOR_UNAVAILABLE is not NORMAL; MLX was not initialized. Macstability is not guaranteed by threshold checks.

Cachecertificate/finalshards,originalAdam52adapter/AdamRNG/receipt,processor/config/contract/calibration,imageSHA and realprepared pixel/context fingerprints are checked before native use. Frozen source closure includes twelve existing third-party F32 runtime pins, not packageversions alone. Child release is checked even on failure, with real childexit/log/RSS/resource/lock witnesses.

## Evidence

- `src/mri_rl/safe_micro_pilot_entry_v1.py`: CPU preparation, exact combined human/review admission, perstage child authority and sequential external witnesses.
- `src/mri_rl/safe_micro_pilot_native_v1.py`: bounded native load/forward, existing micro transaction, greedy-only allDEV24 evaluator.
- `scripts/run_safe_micro_pilot_v1.py`: fixed production entry; requires pinned interpreter and exact credential filehashes for execution.
- [Preparation aggregate](../results/SAFE_MICRO_PILOT234_PREPARATION_ACTUAL_V1.json).
- [Independent scope review](../results/SAFE_MICRO_PILOT234_SOURCE_SCOPE_INDEPENDENT_REVIEW_V1.json).

Scope FILE SHA256: `3927e15080b582e57be370151abe8fc34a6833f1bbcb8c4991681fd93ea8c01c`.
Canonical SHA256: `3e14e49641f43720dcf6ef681bbb63727bfcd90ef60aabe21b7ce876fc7eca2a`.

No new human grant has been issued. Current actual Parent/A/B accuracy is NOT_RUN.
