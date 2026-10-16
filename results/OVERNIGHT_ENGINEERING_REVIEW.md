# Overnight independent engineering review

## Scope and current verdict

Reviewer: isolated Codex engineering reviewer; no Claude review is claimed. Review uses bounded CPU metadata/source inspection without MLX imports, model loading, GPU execution or MRI-volume loading. The researcher’s latest overnight directive authorizes staged Control B continuation and separate Candidate C research, subject to exact restoration, scoped independent review and live resource checks. The prior one-step approval does not automatically approve a new stage.

**PASS_WITH_LIMITATIONS — exactly one resumed update, cumulative step two, is approved below.** The initial requirements record is retained; subsequent stages need new scoped review.

## Verified starting boundary

The previous independent report, `results/SAFE_RESUME_INDEPENDENT_REVIEW.md`, records verification of the completed `balanced_sft_prefix1_paced_v3` run. It completed exactly one Adam update on the first four frozen Control B examples. Saved adapter SHA-256 is `1c7c27bd4c4a902026334d49c88eb8d602678ab517880823d92c614e3de12c04`; optimizer/RNG tensor SHA-256 is `62096c52176adeb8b94d50cc9fa95cf25482695b219afdb9c33071c57f59fb6f`. Its sidecar contains 504 matching moment pairs, scalar step 1, float32 learning rate approximately 1e-5, Python and MLX RNG state, and sampler cursor 4. The complete 112-example schedule hash is `2e5d6bcb55cfb949035986fa8e65debdf13013badb04c275d195fd96be661f67`.

Saved-state integrity and the prior successful adapter reload are verified; an exact native continuation of that run has not yet been demonstrated. No tensor/model replay was performed for this initial review.

## Required restoration and stage invariants

1. Bind actual starting adapter, optimizer payload and manifest bytes, producer receipt, contract, model/processor preparation, adapter configuration, complete trainable signature and frozen schedule. Preserve old producer source identity and explicitly identify the reviewed new consumer source. A migration must not relabel the old checkpoint as produced by current code.
2. Require cumulative completed step 1 and cursor 4 before step two. Distinguish stage updates, cumulative updates, starting cursor, ending cursor, executed range hash and cumulative prefix hash. One newly executed update consumes precisely schedule positions 5–8, ending at cumulative step 2 and cursor 8. No repeated completed examples.
3. Restore every Adam moment, step counter and learning-rate state. Bind beta1, beta2, epsilon and bias-correction semantics in the reviewed compatibility envelope, including explicit treatment of defaults omitted by the old manifest. Compare state inventory against the actual model trainables; reject missing, extra, wrong-shape or wrong-dtype moments.
4. Reconstruct and materialize model/optimizer first; restore all used RNGs immediately before the next stochastic action, without a later reseed. Preserve the fixed schedule; include any actually used NumPy RNG. Do not restore a seed in place of generator state.
5. The tiny native CPU fixture must compare uninterrupted updates 1+2 against save/reconstruct/restore/update 2, including parameters, complete optimizer state, loss, gradients, cursor and subsequent Python/MLX random draws. Corruption and identity mismatches must fail closed. Fixture success is engineering evidence, not MRI training or learning.
6. Preserve all completed checkpoints. Each reviewed stage produces a new immutable adapter/state boundary and receipts recording actual newly executed and cumulative budgets. Validate finite losses/gradients/parameters, a changed adapter digest, complete saved state, reload evidence and resource observations before considering expansion.
7. Retain one model worker, conservative resource gates and paced execution. A resource failure pauses the affected configuration; it does not authorize automatic retry. Resource-boundary observations are not a certified whole-run temperature or memory maximum.

## Scientific limits

Control B remains the label-only sequence balancing intervention; the other 84 historical examples are unchanged. Candidate C admission and visual answerability are independent scientific questions. Exact engineering recovery does not establish MRI generalization, clinical validity, or improvements in model performance. Full B28 or matched evaluation may only be claimed after actual completion and verified receipts.

## Restoration implementation review

The new consumer restores native Adam state only after loading, quantizing and validating the starting adapter, with the expected moment inventory derived from the actual model trainables. It checks backend identity, explicit Adam semantics, finite optimizer tensors, nonnegative second moments and native assignment equality, then restores Python and MLX RNG state and checks equality. SFT uses exactly those two RNG inventories; NumPy state is not silently omitted from an active NumPy training path. The training loop selects `schedule[start_step:end_step]` and preserves cumulative history.

Independent CPU verification: **54 passed, two native tests skipped in 0.30 seconds**, with an import blocker for MLX/MLX-VLM and single-thread environment settings. The lead’s native fixture log reports **two passed in 0.11 seconds** after the final restoration checks were added. Direct fixture source inspection confirms comparison of loss, gradients, parameters, complete optimizer state, cursor and subsequent Python/MLX draws against an uninterrupted tiny CPU linear-model path. This verifies the fixture’s numerical recovery scope; it does not establish full-model uninterrupted-versus-resumed equality.

The reviewer verified the immutable original producer approval and every one of its eleven preserved source snapshots. The new approval will bind twelve current consumer source files separately. Explicit migration records preserve the old producer identity and Adam defaults. Actual static data/contract preflight and four negative probes rejected wrong stage start, stage range digest, cumulative prefix length and optimizer semantics.

Two receipt issues were corrected during review: resume support is now consistent between boundary and final receipts, and single-process continuity is explicitly stage-local. A post-generation resource sample is added. A further issue found before expanding to two-update stages was that intermediate progress used the planned stage-end prefix; the lead is correcting each saved boundary to describe its actual completed prefix. Approval is withheld until that correction is frozen and verified.


## Final step-two scoped approval

The intermediate-prefix defect is fixed: both the step checkpoint metadata and progress receipt now use the actual completed prefix. Independent final CPU verification: **55 passed, two native tests skipped in 0.40 seconds**, with MLX imports blocked. Refreshed native fixture evidence reports **two passed in 0.21 seconds**; its source hashes match the frozen implementation and tests. Actual static preflight passes, and all four independent negative configuration probes fail closed.

**PASS_WITH_LIMITATIONS** for exactly **one new resumed optimizer update**, starting from verified step 1 / cursor 4 and consuming Control B examples 5–8, ending at cumulative step 2 / cursor 8. Output: `local_data/ds007045_bounded/balanced_sft_resume_step002_v1`. One sequential fresh checkpoint reload and one greedy training-image generation capped at 256 tokens are included. No automatic third update, two-update stage, C training or RL expansion is authorized by this approval. The lead must pass current live resource checks and monitor execution.

Reviewer-authored private approval: `overnight_resume_review_v1/approval_step002.json`, SHA-256 **`4ee4e17fc21996b538c704d00a0a401512727ff2ba0f036fbf07ee9c9160ee51`**. It binds twelve frozen consumer sources, the unchanged contract, exact step-one adapter and state/receipt digests, native fixture evidence, explicit producer-to-consumer migration, full schedule, stage range, cumulative prefix, fixed Adam semantics, pacing and unique output. The earlier draft was never transmitted as authorization and was replaced before this final digest was issued.

New stage range hash: `6d7ddb1655f5d7403a5c9d622236383ef08c3607d6eb5f07f40d1c9cfa97420b`. Cumulative eight-example prefix hash: `88e1a394950a8603de21ccbcff28e69b58b40e757b7a875bf3ac519bac76cb24`.

The reviewer-authored private launcher `overnight_resume_review_v1/launch_step002.py` embeds the separately supplied final digest and defaults to preflight only; `--execute` is for the lead’s single monitored model worker. The reviewer did not launch it or import MLX. Completion remains **NOT_RUN** until actual artifacts are independently audited. Native fixture equivalence and validated state restoration do not claim an unmeasured uninterrupted full-Qwen parity experiment.

## Independent step-two result

**PASS_WITH_LIMITATIONS — B_STEP2_TRAINING_COMPLETED.** Actual receipts and byte-level checks verify one new update, two cumulative updates, and cursor 8. The original four-example history is unchanged. The restore receipt reports step 1, cursor 4, all 504 moment pairs, the saved learning rate and restored Python/MLX RNG. The reviewed code enforces those checks before consuming examples 5–8.

Independent recomputation gives new-step loss mean **6.819051146507263**, cumulative loss mean **6.7832823395729065**, and gradient L2 **7.546680441626037**. Bounded per-tensor CPU reads verify all 504 adapter tensors changed and remain finite; float64 parameter delta L2 is **0.11447650611342731**. The after-adapter SHA-256 is `70b92a14ca2837676ab8582b7fb23a39a0f9c676bbeccfa458ba291ae436e660`.

The saved optimizer payload SHA-256 is `cbb6d876eff76ca958249fc9297a8e7b58f231e5be94536a7266e12b2e436e69`; its manifest SHA-256 is `fa221429e915b94444c1bc1d9b0521b95c97f061c231dfba2e94812d18850c35`. All 1,011 tensor headers, offsets, dtypes, shapes and layout references agree. All 504 complete moment pairs are finite, with nonnegative second moments; native step is 2, float32 LR approximately 1e-5, and sampler cursor 8. Python RNG structure and MLX RNG header are valid. No reviewer native-runtime/model execution occurred.

The fresh reload receipt records exact matching of all 504 adapter tensors, followed by actual generation of 55 tokens ending with `stop`. The selected training example/image and adapter match the executed range. Strict reward recomputation is **0 accuracy, 0 format, 0 total score**. This one training-image generation is compatibility evidence, not a model evaluation or learning result.

All **15 resource decisions** reproduce from the saved samples and policy. One `HOLD` for swap growth at quantized-model-ready was followed by `CONTINUE`; no stop occurred. Minimum sampled available memory was **7.40349 GiB**, minimum pressure-free fraction **37%**. Swap moved from **0.81488 GiB** to a maximum **5.19666 GiB**, ending at **5.18885 GiB**. Largest recorded MLX peak was **11.55774 GiB**. The new final generation sample records available memory **8.65211 GiB**, pressure-free **45%**, AC power and zero detected competing heavyweight processes. Thermal telemetry remains unavailable. Subsequent samples **58.12 seconds apart** show exactly stable swap **5.11854 GiB**, with available memory **13.98975 then 12.82420 GiB**, pressure-free **68% then 69%**, AC/100% battery and no competing heavyweight process. Both pass loading gates; swap remains material and must continue to be monitored.

Private independent audit: `overnight_resume_review_v1/postrun_step002_independent_verification.json`. Full-Qwen uninterrupted-versus-restored numerical equality remains unmeasured; the real continuation establishes validated state restoration and a completed next update.

## Step-three scoped approval

**PASS_WITH_LIMITATIONS** for exactly one new update from the verified step-two boundary, consuming frozen examples 9–12 and ending at cumulative step 3 / cursor 12. All twelve source hashes are unchanged. Actual static preflight passes. The output is `balanced_sft_resume_step003_v1`; one sequential fresh reload and <=256-token greedy training-image generation are included. Current live loading/resource gates remain mandatory. No automatic fourth update or two-update stage is approved.

Reviewer-authored approval: `overnight_resume_review_v1/approval_step003.json`; independently supplied SHA-256 **`ba3d3a1551c07126b0fcb1bf271dbb763b1692f623bfca7092481bba830f89f4`**. Its dynamic evidence binds the step-two independent audit, immutable parent approval, actual state/receipt hashes and two stable post-run resource samples. Stage range hash: `39323b88683932bfe93e6f9f4c7a6da2b17eae287c4da694cb690f4fd3457f83`; cumulative twelve-example prefix hash: `b55c886c66c941e9b58e3cf552b7c0412f048e2c30f0c442fe2393939618fbb0`. Reviewer launcher `launch_step003.py` defaults to preflight; execution belongs to the lead’s monitored worker. Step three remains NOT_RUN until actual receipts are inspected.


## Research covenant addendum review

The reviewer read the researcher’s cross-model addendum and the fetched covenant at revision **0ed14a2**, along with the primary execution contract, AGENTS, CLAUDE and engineering protocol. The six-part step-three mission alignment record is present in `research/execution/OVERNIGHT_DECISION_LEDGER.md`. Step three is tagged **ENGINEERING_UNBLOCK / B_LABEL_CONTROL**: it supports an eventual controlled SFT baseline for the locked Vero-to-MRI question; it does not replace the task-transfer study or establish learned MRI reasoning. Frozen variables, actual update provenance, falsifiers, required later baselines and finite resource conditions are specified.

**NO_ISSUE_FOUND within the exact step-three engineering scope.** All twelve bound source hashes and the independently issued approval digest remain unchanged after fetching documentation. The existing one-update approval therefore remains valid, subject to fresh live resource checks. No scope expansion follows from this addendum review.

Reviewer provenance is an isolated **Codex / GPT-family agent**, the same model family as the implementer. This is independent-agent engineering review, not independent-model-family verification. **CROSS_MODEL_REVIEW=NOT_VERIFIED** for these experiments. The lead reports authenticated Claude access through an authorized keychain check, but no research payload has been sent; authentication alone does not override the earlier denied unpublished-data sharing. No Claude findings or cross-model consensus are claimed here. The covenant expressly permits the current interim scoped review and ongoing B/C work while a legitimate distinct-family review remains unavailable or incomplete.

## Independent step-three result

**REPRODUCED / PASS_WITH_LIMITATIONS — three cumulative updates completed, cursor 12.** The independently inspected third-stage artifacts preserve all prior history and consume exactly examples 9–12. New-step mean loss is **6.615137100219727**, cumulative mean **6.72723392645518**, gradient L2 **7.811427946835844**. All 504 adapter tensors changed and remain finite; independently accumulated float64 delta L2 is **0.1373981192716968**.

After-adapter SHA-256: `381b0ca7711099bfd0bf33f33605c4791b24a8a4d51f2c0b8a52738f968e792b`. Optimizer payload SHA-256: `f67abc4fb9ad37ad8b35e6e00d613351732de9f203eb6b5e76fb43123dbbfaf2`. Manifest SHA-256: `2aa16872a09de6e918123685c63c27d434cb1e08cbde9017614c198212ffc313`. All 1,011 tensor layout/header entries, complete 504 moment pairs, finite/nonnegative moment checks, native step 3, learning rate, RNG inventory and cursor binding pass independent CPU inspection. The restore receipt binds step 2/cursor 8. Fresh adapter reload and actual generation are recorded; strict generation scores remain **0/0/0**, not evidence of learning.

All 14 saved resource decisions reproduce with no HOLD. Minimum sampled available memory is **8.10381 GiB**, minimum pressure-free percentage **40%**. Swap increases **4.91528 → 6.98498 GiB**; final available memory is **8.42723 GiB**, pressure-free **43%**, AC/100% battery and zero detected competing model processes. Largest recorded MLX peak is **11.55774 GiB**. Thermal telemetry remains unavailable. Nonzero and increasing inter-run swap requires fresh post-run stability checks before expansion.

Private verification: `overnight_resume_review_v1/postrun_step003_independent_verification.json`. No reviewer MLX import/model execution or claim of full-Qwen uninterrupted parity. Steps 4–5 are not yet approved in this section; their separate exact scope and current resource evidence must be recorded first.

## Exact two-update stage: steps four and five

**PASS_WITH_LIMITATIONS**, under **B_LABEL_CONTROL / ENGINEERING_UNBLOCK**, for exactly two additional updates from cumulative step 3/cursor 12 to step 5/cursor 20, consuming frozen schedule examples 13–20. This preserves the controlled SFT baseline needed for the Vero-to-MRI study; no new scientific claim or changed data/precision/objective is introduced. All twelve sources remain identical to the reviewed implementation. The prior intermediate-prefix correction is present at both checkpoint and progress publication. Step four must save a full immutable boundary with cursor 16 before step five proceeds; the existing pacing includes its 15-second step-four pause. Each example and optimizer update remains resource-gated.

Three post-step-three samples spanning **36.61 seconds** show swap **6.88342 → 6.86779 → 6.86779 GiB**, available memory **11.96130 / 12.17310 / 11.92831 GiB**, pressure-free **62 / 66 / 66%**, AC power, full battery and zero detected competing model workers. All samples independently pass loading gates. This supports only the finite stage; swap persistence and unavailable thermal telemetry remain limitations, and fresh live launch checks remain required.

Reviewer-authored approval: `overnight_resume_review_v1/approval_step005.json`, SHA-256 **`6a8105ed4bfd2533c5af225d8d531240c4208049788298e33e87b050845b1737`**. Reviewer launcher `launch_step005.py` explicitly calls `steps=2` with parent `balanced_sft_resume_step003_v1` and new output `balanced_sft_resume_step005_v1`; AST and saved budget checks confirm that scope. Actual static preflight passes. The independently authored dynamic evidence binds the step-three audit, parent state/receipt/approval and stable resource samples.

Stage range hash: `143f43fc2d8a276ffd53da32ceeb51e5ee1ff8ca7b606d7e4fae4b7615cf38ce`; cumulative twenty-example prefix hash: `b779c6f380c26740d4440d8dddf0896fe2ba95ee080e1cae49c1f3f927065ace`. One sequential fresh reload and <=256-token greedy training-image generation at stage end are included. **No step six or automatic subsequent stage is authorized.** The implementation’s native tiny-fixture equality remains valid; full-Qwen uninterrupted parity remains unmeasured. Steps four and five remain NOT_RUN until actual receipts are independently audited.
