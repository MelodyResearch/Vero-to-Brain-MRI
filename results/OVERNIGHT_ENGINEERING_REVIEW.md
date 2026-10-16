# Overnight independent engineering review

<!-- CURRENT_ENGINEERING_STATUS -->
**Latest verified training: 19 cumulative updates, cursor 76.** The completed stage approval is consumed; further updates require a separate approval after fresh resource evidence. Evaluation and controls have source/CPU reviews only. Earlier approvals do not authorize replay or expansion.
<!-- END_CURRENT_ENGINEERING_STATUS -->

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


## Subsequent bounded-stage ledger

All rows below are **REPRODUCED / PASS_WITH_LIMITATIONS**, based on primary private receipts, independent checksums/header/layout inspection and bounded per-tensor CPU checks. Each intermediate adapter/state and actual completed prefix is verified; complete 504 Adam moment pairs, learning rate, RNG inventory, restored starting cursor and unchanged source approval remain required. Values are engineering observations, not held-out model-performance estimates.

| Newly completed steps | Cumulative cursor | Per-step mean losses | Per-step gradient L2 | Changed finite tensors per step | Strict final reload accuracy/format/reward | Resource observations |
| --- | --- | --- | --- | --- | --- | --- |
| 4–5 | 20 | 6.711098671 / 6.355120540 | 8.701938941 / 9.788064628 | 504 / 504 | 0 / 0 / 0 | 22 decisions, no HOLD; min available 8.65620 GiB, min pressure-free 47%; swap 6.82092 → max 7.89313 → end 7.86188 GiB |
| 6–7 | 28 | 6.131338596 / 5.590974331 | 11.241118678 / 11.608965319 | 504 / 504 | 0 / 0 / 0 | 22 decisions, no HOLD; min available 8.80391 GiB, min pressure-free 49%; swap 7.47906 → max 7.87671 → end 7.79858 GiB |
| 8–9 | 36 | 4.912723184 / 4.154455423 | 19.708411036 / 10.760168337 | 504 / 504 | 0 / 0 / 0 | 22 decisions, 0 HOLD; min available 9.08488 GiB, min pressure-free 45%; swap 7.68915 → max 9.22645 → end 9.18738 GiB |
| 10–11 | 44 | 3.444637895 / 2.975314081 | 10.326782028 / 7.904877372 | 504 / 504 | 0 / 0 / 0 | 22 decisions, 0 HOLD; min available 8.66185 GiB, min pressure-free 46%; swap 8.97638 → max 9.25903 → end 9.22596 GiB |
| 12–13 | 52 | 2.427459776 / 1.856112719 | 6.558643483 / 5.279714036 | 504 / 504 | 1 / 1 / 1 | 22 decisions, 0 HOLD; min available 9.58792 GiB, min pressure-free 49%; swap 9.13221 → max 9.98877 → end 9.95752 GiB |
| 14–15 | 60 | 1.727716565 / 1.391658962 | 5.077487420 / 4.231651757 | 504 / 504 | 0 / 0 / 0 | 22 decisions, 0 HOLD; min available 9.44507 GiB, min pressure-free 49%; swap 9.92627 → max 9.92627 → end 9.91064 GiB |
| 16–17 | 68 | 1.265112400 / 0.916878432 | 4.084682287 / 3.385609708 | 504 / 504 | 0 / 1 / 0.2 | 22 decisions, 0 HOLD; min available 9.96506 GiB, min pressure-free 51%; swap 9.70746 → max 9.70746 → end 9.66840 GiB |
| 18–19 | 76 | 0.904681936 / 0.732752539 | 3.173778016 / 3.029052296 | 504 / 504 | 0 / 1 / 0.2 | 22 decisions, 0 HOLD; min available 10.02870 GiB, min pressure-free 52%; swap 9.65277 → max 9.65277 → end 9.64496 GiB |
<!-- ENGINEERING_STAGE_ROWS -->

Stage 4–5 saves verified intermediate cursor 16 and final cursor 20, with distinct changed adapter hashes and complete 1,011-tensor state payloads. Final adapter SHA-256: `89694d04924c1327f638a7ca9b622c5f3c96aa3c6fc894115de0946b03f8767b`. Final optimizer payload SHA-256: `836d1b1169b32ba76f3b6603fb0209fede25ef37b6cbd489d6aaaa59ca583eb6`. Full per-step hashes and CPU results are in `overnight_resume_review_v1/postrun_step005_independent_verification.json`. Float64 aggregate adapter delta L2 is 0.31993798362058434. All 22 resource decisions reproduce; final-generation telemetry is present, AC/100% battery and no competing heavyweight process were recorded, and thermal telemetry remains unavailable.

### Authorization ledger after step five

**PASS_WITH_LIMITATIONS for exactly steps 6–7**, frozen examples 21–28, ending at cumulative step 7/cursor 28, with full intermediate state at step 6/cursor 24 and one sequential final reload generation capped at 256 tokens. All twelve code hashes remain unchanged; actual static preflight and launcher budget inspection pass. Two post-step-five samples 39.90 seconds apart show decreasing swap (7.78375 to approximately 7.7056 GiB), available memory above 12.6 GiB, pressure-free 64–67%, AC/100% and no competing model process. Fresh launch gates remain mandatory, and the lead must finish concurrent local data work before loading the model.

Approval: `overnight_resume_review_v1/approval_step007.json`; digest **`b0b3a84f085b0644a9b955902bb6dcc65b26025c08dcf511f711ab0de398e40c`**. Launcher: `launch_step007.py`; parent `balanced_sft_resume_step005_v1`, output `balanced_sft_resume_step007_v1`, explicit `steps=2`. Stage-range hash `ce76d33200c94211daf2afb73fe854df89b1388d41f72d1766f660db33625bf9`; cumulative prefix hash `26f042cd472af22209da87322b5b5962131dcd9886bed2612bcedf5793d489e5`. No step 8 or subsequent stage is authorized. This remains B_LABEL_CONTROL / ENGINEERING_UNBLOCK under the covenant.

A minor inherited explanatory string in earlier approval metadata said “cumulative8” after scopes expanded; actual machine-enforced prefix counts/hashes and launch budgets were correct. The new approval text now derives that explanatory count from its exact scope. Prior approvals and executed evidence are preserved unchanged.

Step 6–7 private audit: `overnight_resume_review_v1/postrun_step007_independent_verification.json`. Both boundaries and actual prefixes pass, at cursors 24 and 28. Final adapter SHA-256 `e5f444649ccb865e49d9e29f35d6cdd30968112d429b44ae3bc12db1f9002a68`; final optimizer payload SHA-256 `60c6f993e3c65d435a6c479a65ddf6ee411886034c96c2d5f5b4c9548369b7c1`. No code changes or scientific-success claims accompany this result.


## Later scoped approvals

Each row is **PASS_WITH_LIMITATIONS** for only the named new updates and one sequential final reload/generation (<=256 tokens). Each separately authored approval binds unchanged implementation hashes, parent artifacts, independent audit, actual schedule range/cumulative prefix and post-run stability samples. Actual static preflight and launcher budget assertions pass. Live resource gates remain mandatory; further stages require separate review. These remain B_LABEL_CONTROL / ENGINEERING_UNBLOCK, with no approval for the new evaluation helper, C training or RL.

| New updates | Starting → ending cursor | Output / private approval | Independently supplied approval SHA-256 | Post-run stability supporting scope |
| --- | --- | --- | --- | --- |
| 8–9 | 28 → 36 | `balanced_sft_resume_step009_v1` / `approval_step009.json` | `fa54a27f03d67b68466e14a54fdbe1aadeb698de6c589bdd3a0c0e2edb3741b7` | Two samples 36.34 seconds apart; swap 7.79858 → 7.79077 GiB; available 14.14900 → 13.9044 GiB, pressure-free 68 → 69%; AC/100%, competing model workers 0 |
| 10–11 | 36 → 44 | `balanced_sft_resume_step011_v1` / `approval_step011.json` | `140a401abdc20eb5375d25f341e598711279732cf3d10615e55ba9efd8efc5fa` | Two samples 61.10 seconds apart; swap 9.17957 → 8.98419 GiB; latest available 13.0999 GiB, pressure-free 72%; AC/100%, competing model workers 0 |
| 12–13 | 44 → 52 | `balanced_sft_resume_step013_v1` / `approval_step013.json` | `90019ef8069f04910071f5233ed2fab300784a65e0ee0a0480db2fd4626af9f7` | Two samples 64.96 seconds apart; swap 9.2103 → 9.1322 GiB; latest available 12.7705 GiB, pressure-free 67%; AC/100%, competing model workers 0 |
| 14–15 | 52 → 60 | `balanced_sft_resume_step015_v1` / `approval_step015.json` | `4658714abed1f1a9a5a85017547be0536e328e798d8e15ea52128a19533cf950` | Two samples 57.07 seconds apart; swap exactly stable 9.92627 GiB; latest available 14.399 GiB, pressure-free 67%; AC/100%, competing model workers 0 |
| 16–17 | 60 → 68 | `balanced_sft_resume_step017_v1` / `approval_step017.json` | `bf255f844df656b28388e471cb46cc0f20b43ca8f0cbb8e4ccedf8cfcf24979e` | Two samples 70.88 seconds apart; swap 9.76215 → 9.72309 GiB; latest available 13.559 GiB, pressure-free 69%; AC/100%, competing model workers 0 |
| 18–19 | 68 → 76 | `balanced_sft_resume_step019_v1` / `approval_step019.json` | `26a6e46678be1f92fa0b8c852aae8f6ce05fe6c38607342ff5a929976df81123` | Post-C-decoding samples 73.98 seconds apart; swap exactly stable 9.65277 GiB; latest available 14.191 GiB, pressure-free 69%; AC/100%, competing model workers 0 |
<!-- ENGINEERING_APPROVAL_ROWS -->

Step 8–9 launcher `overnight_resume_review_v1/launch_step009.py` explicitly requests two updates from step007. Approval remains conditional on the same conservative gates; no step10 is approved. The separately developed evaluation helper is outside these twelve bound training sources and has not been reviewed or authorized by this approval.

Stage 8–9 actual boundaries, checksum/header/finite state checks and full per-step hashes are preserved in `overnight_resume_review_v1/postrun_step009_independent_verification.json`. Final adapter SHA-256 `2e223a7eba50de729d76149c305e6a8f2d1f2de1536be33930919e9295767bf3`; final optimizer payload SHA-256 `995a8d2ee339f4336620f1ab7d94e15fb0e1c1b0e514f6c3136631f4423200ed`. **REPRODUCED / PASS_WITH_LIMITATIONS** for actual completion, without a learning or full-Qwen uninterrupted-parity claim.

Exact steps10–11 approval includes the intermediate step10/cursor40 state, followed by final step11/cursor44 and sequential reload. All twelve source hashes remain unchanged; actual static preflight and explicit two-update launcher checks pass. No step12 or evaluation execution is approved.

## Future matched evaluator review — execution not authorized

**CONFIRMED_DEFECT (P1) → REPRODUCED correction.** The first `paced_frozen_eval.py` plan pinned the manifest bytes but recomputed current image hashes without comparing them to preserved A0/A28 outputs. A manifest containing unchanged paths could therefore admit changed image pixels into a supposedly matched comparison. The implementer reproduced this in an integrated `prepare_plan` regression: the pre-fix version accepted changed pixels and the rejection test failed. Competing explanation checked: actual current data have not drifted; this reviewer independently matched all 54 current image hashes, prompts and categories against both immutable historical output files.

Minimal correction is confined to the new evaluator: pin both historical output-file digests and require exact 54 unique IDs, categories, prompts, current pixel hashes, model/template/decoding profile, image placeholders, verified tensor count and baseline adapter/config identities. The two historical digests are now included in every expected output and exact approval scope. Independent Step D verification inspected source SHA-256 **`645b80af9b1ae80141bfcb484461cf5266f60c65a83df241168275249c14ec73`**, confirmed guard placement before resume/planning, and ran **26 tests passed in 0.40 seconds** with MLX/MLX-VLM imports blocked. The actual 54-record historical-baseline verifier passes. Regression coverage includes changed pixels, baseline bytes, prompt/profile/adapter changes and duplicate IDs.

Installed MLX-VLM source inspection confirms that historical `add_generation_prompt` already defaults to true and that synchronous `GenerationResult` returns the recorded token, finish-reason and peak-memory fields. The explicit new prompt setting is therefore compatible with the historical evaluator. Backend peak-memory units are decimal GB, distinct from the resource supervisor’s GiB; an exact 256-token count is not by itself a truncation flag. Empty textual answers are retained as failures, not selectively regenerated.

**PASS_WITH_LIMITATIONS for source/CPU correction only; evaluation execution NOT_VERIFIED / NOT_RUN.** A genuinely completed and independently audited B28, exact new record scope, existing-output digest, fresh resources and separate reviewer approval remain necessary. The new evaluator is outside the twelve frozen training files, which were not changed. Four internal-development patients and 54 correlated questions remain exploratory; this engineering fix does not establish MRI learning or cross-model review of private B/C evidence.

Stage 10–11 actual boundaries, checksum/header/finite state checks and full per-step hashes are preserved in `overnight_resume_review_v1/postrun_step011_independent_verification.json`. Final adapter SHA-256 `a47bd3ce1ff84b3f098d409b78b81795ffe49185f201f2f9552c8d285d1f6f37`; final optimizer payload SHA-256 `3c5186b062339031476652c480ad5ab765fc9b4320f6d380ebce3e588427a5ac`. **REPRODUCED / PASS_WITH_LIMITATIONS** for actual completion, without a learning or full-Qwen uninterrupted-parity claim.

Exactly steps12–13 are separately approved, with intermediate step12/cursor48 and final step13/cursor52. Code hashes remain identical after commit495db13; static preflight and explicit two-update launcher assertions pass. No step14 or evaluation execution is approved.

Stage 12–13 actual boundaries, checksum/header/finite state checks and full per-step hashes are preserved in `overnight_resume_review_v1/postrun_step013_independent_verification.json`. Final adapter SHA-256 `99d90bd28c9964e8ec6f1034d9d3aa1e81a51996cb068c7c735b9e168ec3bba6`; final optimizer payload SHA-256 `7596e7261eab381f9775582fb23c21934644cdfb9e51574b571974c321d6f6e7`. **REPRODUCED / PASS_WITH_LIMITATIONS** for actual completion, without a learning or full-Qwen uninterrupted-parity claim.

Stage12–13 independently exposes27 cumulative patients. Its final single-training-image reload score is1/1/1 under the unchanged strict parser; this is not a development score or evidence of pixel grounding. The fixed28-step schedule, selection and loss remain unchanged. Exactly14–15 are approved after stable resources, unchanged 12 source hashes and passing staticpreflight; intermediate cursor56 and final60 are required. No step16 is approved.

Stage 14–15 actual boundaries, checksum/header/finite state checks and full per-step hashes are preserved in `overnight_resume_review_v1/postrun_step015_independent_verification.json`. Final adapter SHA-256 `21e09cacbb31970f0cd3603073015855f7c80bc62f1e808b0b7e05b58145ba59`; final optimizer payload SHA-256 `d2cb23fe9f9548d531e789479ccb9ab383f9ec7087bbcbc9121847acd56d53c9`. **REPRODUCED / PASS_WITH_LIMITATIONS** for actual completion, without a learning or full-Qwen uninterrupted-parity claim.

## Future sequence-control evaluator review

**NO_ISSUE_FOUND / PASS_WITH_LIMITATIONS for source and CPU checks only.** Reviewed `paced_sequence_controls.py` SHA-256 `30d8ca43298842a553ded6ba20a17e2caa18fe0671ec056c6fcb53836976553c`: genuine B28 and all 54 matching primary predictions are required before controls; immutable 24-row no-image and same-patient/same-slice opposite-sequence manifests, historical A28 control outputs and private gold identity are verified. Exact source, pending IDs, mode, current image hashes, primary-output digest, existing control-output digest and 1–4-record budget enter the independently approved scope. The no-image path requires zero image placeholders; swap requires one. Actual finish/token/peak metadata and post-generation gates remain explicit, and output is persisted before a resource HOLD/STOP.

Independent checks: **9 tests passed in 0.06 seconds** with MLX/MLX-VLM imports blocked. Both actual 24-row control manifests equal transformations of the frozen primary records; historical baseline pixel/prompt/profile checks and private gold hash pass. No model was run. Control inference remains **NOT_VERIFIED / NOT_RUN** until genuine audited B28, complete 54 primary outputs and a new exact execution approval. Baseline correctness does not establish image grounding; later analysis must distinguish answer changes from correct directional label flips.

Exactly steps 16–17 approved with intermediate cursor 64 and final 68, unchanged 12 source hashes, passing actual static preflight and explicit two-update launcher. No step 18 or evaluation execution approved.

Stage 16–17 actual boundaries, checksum/header/finite state checks and full per-step hashes are preserved in `overnight_resume_review_v1/postrun_step017_independent_verification.json`. Final adapter SHA-256 `62f25fb23a922a80afd91c7cb4f5b06c2c72632207a3d02261bdb058388464f8`; final optimizer payload SHA-256 `c50efdfd83150c0c5e3466700a68bf160aa54a3813afb8ea0cc4f9ddc7f11fdf`. **REPRODUCED / PASS_WITH_LIMITATIONS** for actual completion, without a learning or full-Qwen uninterrupted-parity claim.

## Paired-analysis provenance corrections

Lightweight source review identified two bounded issues in the new analysis script: the label `CROSS_MODEL_REVIEW` could imply distinct-family review of private checkpoint results, and matching profile fields alone would not reject edited historical response text. The statistics author added failing regressions, renamed the scope to `PAIRED_INTERNAL_DEV_CHECKPOINT_ANALYSIS`, explicitly records `cross_model_review=NOT_VERIFIED`, and makes the CLI verify exact immutable A0/A28 output digests before analysis. This does not change scoring rules or training.

Independent Step D: source SHA-256 `935dc33d74a8e0f890d590d78c72271040b35d3335f262e222c5fed0c881b600` inspected; **20 paired-analysis tests passed in 0.41 seconds** with MLX imports blocked. Source checks support the distinctions between existing parser accuracy, full-format accuracy, diagnostic answer-only extraction, question-pooled versus within-task patient means, and observed versus unknown termination. Synthetic test results are not model scores. Actual B28 paired numerical analysis remains NOT_RUN until complete predictions exist and are independently recomputed.

Exactly steps18–19 are approved using the fresh `postC_poststep017_resources_PRIVATE.jsonl` evidence after decoding exited. Intermediate cursor72 and final76, unchanged12 source hashes, actual static preflight and explicit two-update launcher are verified. No step20 or evaluation execution is approved.

Stage 18–19 actual boundaries, checksum/header/finite state checks and full per-step hashes are preserved in `overnight_resume_review_v1/postrun_step019_independent_verification.json`. Final adapter SHA-256 `2a1b2c3d22cf45945ffa990ce9be1686d368baae7201bf37f4c37ad8b6923c53`; final optimizer payload SHA-256 `8e909dff319cd9d86f7fb55fb79bccd02920d9b909bfab1acdeca0b32c4a8a3c`. **REPRODUCED / PASS_WITH_LIMITATIONS** for actual completion, without a learning or full-Qwen uninterrupted-parity claim.
