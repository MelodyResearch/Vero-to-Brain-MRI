# Dynamic-resource micro pilot — actual Stage2 PASS, Stage3 restore failure

## Actual outcome

The exact authorized scope `da7ef34ccf171922d601bb3160c99758c0873a518e58ea981f39e8f86012c279` ran once. No cache recovery was repeated. Stage2 completed; Stage3 made one real Direct Vero update and saved its new checkpoint, then failed restoring the shared parent. B and DEV24 were not executed. This is a failed/incomplete matched experiment, not an accuracy result.

| Endpoint | Real updates | DEV24 accuracy / FLAIR / T1 |
|---|---:|---|
|Original Adam52 parent|0|NOT_RUN|
|Direct Vero A saved Adam53|1|NOT_RUN; enclosing arm transaction not committed|
|Paired-Prompt B|0|NOT_RUN; not started|

## Stage2: actual load and finite forward qualification

- Full original Adam52 loaded from certified historical cache.
- Two genuine TRAIN MRI responses,55 returnedtokens,6F32 scorer calls/1408TFpositions,0updates.
- Rollout/scorer errors0.0000424385071 and0.0000131130219, below0.002.
- Same fixed-target image delta0.00476684939: finite image sensitivity, not full class discrimination or anatomical grounding.
- Full1942parameter/1010Adam-array fingerprints and Adam layout unchanged. RNG restore passed the source identity assertion; readonly JSON lacks a separately saved pre/post RNG fingerprint pair.
- Real exit0/reaped,release1096activeMLXbytes/modelnone. Peak exact RSS5.91937GiB,reportedMLX6.00541GiB. Independent saved-only review PASS_WITH_LIMITATIONS.

## Stage3: real GSPO update, then transaction failure

One common draw produced16responses/618tokens. Own-old scoring used16forwards/5546TFpositions. Original verifiable rewards and sample-std+epsilon advantages were retained. A admitted one sequence and oneDallas group,8responses total. Native8backwards/8gradientforwards/2701TFpositions produced finite joint gradient L2=0.38172549327. One Adam52→53 update and adapter/fullAdam/R1 checkpoint were actually written; all saved checkpoint file hashes match their receipt.

The subsequent `finally: backend.restore(snapshot)` failed `Rollback failed full model/Adam/RNG integrity`. A receipt exists, but its outer operation journal has only a reservation: **INDETERMINATE_NO_REPLAY**, not a successful complete arm. No automatic adoption, repeat update or repeated rollout occurred. B never started. NativeStage3exit1 and rootexit1 were reaped; model lock is free. Stage3 allocator release check failed (modelreferenceNone but5,260,969,528activebytes). Process exit/reaping is separate lifetime-release evidence, not successful allocator cleanup.

Original parent canonical files still match their original receipt by independent CPU streaming checks. The saved child is preserved privately and has not been reloaded/evaluated.

## Resources and total spent cost

All1801saved child-boundary+independent-guardian observations were GREEN/YELLOW; noRED,guardianSTOP,TERM orKILL. Minimumavailable5.78966GiB/free32%; exact-childRSSpeak5.91937GiB,reportedMLXpeak10.35595GiB. Swap12.00006→peak15.96203GiB. It did not independently stop the new color policy. Postrootavailable14.67416GiB/free59%,battery90%; thermal unavailable remains disclosed. No user apps closed or OS protections changed.

Actual totals:2loads,18responses,673returnedtokens,30scorerforwards,9655TFpositions,8backwards,1update,458.77srootwall. Zero recovery attempts. No final100 or new16 outcomes were accessed.

## Directly relevant CPU repair

A confirmed source/CPU defect can corrupt a rollback target: installed MLX retains the dict assigned to `optimizer.state`, then replaces its step/m/v entries during an update. The historical restore passed saved snapshot containers directly to that setter. Exact installed Optimizer/Adam AST executed with NumPy reproduces saved-step52→53 corruption; reassignment cannot restore52.

New future-only `safe_micro_restore_v1` detaches mutable containers before restoration. Six CPU tests pass, including exact installed Adam formulas, complete moment/step/RNG restoration and repeatable independent A/B starts. The helper is not wired into the consumed frozen scope. Array leaves are borrowed by default; a backend with in-place leaf mutations requires an explicit leaf-copy contract.

Classification: **container-alias bug CONFIRMED in source/CPU; contribution to this generic native restore failure LIKELY, not uniquely isolated**. No repaired native run, adopted A transaction or MRI accuracy improvement is claimed.

## Next decision

Do not rerun A or regenerate the shared bundle. Any continuation needs a fresh exact scope: qualify the corrected state restoration, independently decide whether the saved A endpoint can be explicitly adopted despite its failed enclosing transaction, then execute only genuinely unspent B work and matched accuracy evaluation if authorized. The consumed fatal attempt cannot resume automatically. This is a restore-integrity correction, not a new loss, reward or MRI research direction.

[Actual aggregate](../results/SAFE_MICRO_DYNAMIC_V3_ACTUAL_RESULTS_V1.json) · [Stage2 review](../results/SAFE_MICRO_DYNAMIC_V3_STAGE2_ACTUAL_INDEPENDENT_REVIEW_V1.json) · [Stage3 review](../results/SAFE_MICRO_DYNAMIC_V3_STAGE3_ACTUAL_FAILURE_INDEPENDENT_REVIEW_V1.json) · [CPU alias proof](../results/SAFE_MICRO_RESTORE_CONTAINER_ALIAS_CPU_PROOF_V1.json)

[Future-only CPU restore fix independent review](../results/SAFE_MICRO_RESTORE_CONTAINER_ALIAS_CPU_INDEPENDENT_REVIEW_V1.json): PASS_WITH_LIMITATIONS,6independenttests. No native repair or checkpoint adoption permission.
