# Independent review of post-SFT28 strict reward activation

Reviewer: isolated Codex agent `/root/activation_reviewer`. This is a local independent engineering review, not a Claude review. No MRI data or unpublished material were sent to Claude. Review scope is one bounded Tier B diagnostic using the preserved SFT28 adapter, actual training images, strict `direct_vero` reward, at most three rollout groups and one optimizer step. It does not authorize Tier C training or establish MRI generalization.

## Initial independent source inspection

Status before implementation changes: **NOT_VERIFIED** for launch. The old smoke path lacks a versioned Tier B review gate and checkpoint reload verification. Its active-group gate uses total reward variation; format-only reward variation can therefore trigger an optimizer update. A bounded task-reward activation test must require variation in actual task correctness and separately disclose format contributions. The prior matched-prompt contract declares greedy temperature 0 and cannot authorize the intended stochastic temperature 1 sampling unchanged.

The reviewer read `mlx_gspo_smoke.py`, `experiment_contract.py`, `vero_mri_reward.py`, `mlx_sft_warmstart.py`, the actual SFT28 receipt and adapter bytes, local model/adapter configs, the local train/dev manifests, and installed MLX-VLM generation and adapter-loading source directly. This assessment was sent to the implementer before the final code was supplied.

## Independently reproduced static and CPU checks

- Preserved SFT28 adapter SHA-256: `244c26fd5ce2e3b02b46703d4f46a90608f2ef1a868eeee682ee4fe48a58ed73`; actual bytes match its completed 28-step receipt. LoRA rank 4, scale 2, dropout 0; base attention dropout 0.
- Existing contract artifacts verify against the stored hashes. All 382 training records from 28 patients and 54 development records from four separate patients pass manifest, lineage, image decoding and positive/negative strict reward checks; 436 image references decoded. This independently confirms manifest consistency; it is not a new visual label audit or proof of upstream label correctness.
- For `mri_sequence`, first three diverse-label candidates are FLAIR / contrast-enhanced T1 / FLAIR, all from one eligible training patient. This is sufficient to test a technical update but cannot represent patient generalization. The question in `extra_info` equals the actual user-prompt question for each candidate.
- Installed MLX-VLM sampling defaults are `top_k=0`, `min_p=0`, no repetition penalty. Explicit temperature 1 and top-p 1 therefore use the same unwarped policy probabilities as the teacher-forced old/current log-probability calculation. Generation returns EOS in `token_ids` even though the decoder removes it from displayed text; training includes that sampled token.
- Old log probabilities are materialized before updates and stopped from receiving gradients. Current loss uses the pinned GSPO-token stop-gradient construction and response mean; group advantages use sample standard deviation. This is a source inspection of a small Apple Metal LoRA port, not numerical CUDA-veRL parity or the full native Vero recipe.

## Required bounded-run acceptance conditions

The final launcher must bind code, contract, starting adapter, training manifest, route and budget to a separate review. It must use the matching SFT system instruction and native assistant prefix, preserve actual rollout token IDs, record prompt/image hashes, skip zero task-variance and format-only groups, check finite gradients before updating, save changed adapter bytes and reload them with prediction/log-probability agreement. Current power, memory and thermal checks must run before model loading and each heavy unit. A no-update result is valid evidence and does not justify silently expanding the budget.

Final source review and actual-run verification are pending in this initial entry. No review authorization JSON has been issued yet.

## Corrected code inspection before exact authorization

The corrected diagnostic selects a group only if every candidate has format score 1, both correct and incorrect answers occur, and overlong penalties are equal. Thus only the MRI task term distinguishes candidate rewards. Independent CPU examples confirm rejection of format-only, equal-answer and length-confounded groups. For two rewards 1.0 and 0.2, the sample-standard-deviation advantages are approximately +0.7071055 and -0.7071055.

The new contract identity is `056485976b5fff7229cb83232217234550f509eed3254c7e2408f42c848001b6`: temperature 1, maximum 256 tokens, exact SFT system instruction, native assistant prefix, and the preserved train/dev manifests. The reviewer independently reran its artifact/data checks successfully. All four local base weight shards were separately streamed through SHA-256 and matched their cached Hugging Face etags at revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`. The private verification inventory is `experiment_contract_strict_activation_v2/independent_base_shard_audit.json`.

New runtime checks bind the prompt to the contract, use explicit unwarped sampling, materialize old log probabilities, check their equality with current teacher-forced probabilities before updating, enforce finite nonzero gradients, preserve partial evidence, and reload the saved adapter into a fresh base model with exact adapter-tensor equality. Same-parameter teacher-forced old/current parity does not itself establish numerical equality between incremental cached generation and teacher-forced forward passes; this and CUDA-veRL numerical parity remain explicit limits of the bounded diagnostic. A successful update will require separate actual receipt inspection before any expansion.

## Final prelaunch verdict

**PASS_WITH_LIMITATIONS** for the exact first Tier B diagnostic, following final source inspection and **134 passing CPU tests**. The reviewer authored `experiment_contract_strict_activation_v2/approval.json` and separately transmitted its SHA-256 `42725005872de4301b8376747f82bf97fde39f40c3edd07f9a11694ee8f9aecd` to the implementer. The live preflight accepted that actual review against the exact current source hashes, contract, SFT28 starting adapter, route, output directory and budget. This is an operational trust boundary relying on the independent reviewer message; local JSON is not a cryptographic proof of agent identity.

Authorized scope: sequence route, seed 7, deterministic diverse-label candidate selection, temperature 1, group size 2, maximum three groups, 256 generated tokens per candidate and at most one optimizer update. The output is `strict_activation_sft28_n2_v2` under ignored private data. A previous dynamic weight update is deliberately not required for this first diagnostic; the runtime must prove task variation, gradient finiteness, changed weights and reload before expansion.

**Tier C and multiarm child-contract delegation remain NOT_VERIFIED and are not authorized by this review.** The current multiarm runner explicitly blocks because each child's changed manifest/output/starting adapter still needs a separately verified delegation implementation. Active-group filtering is a diagnostic choice, not a claim that the full native Vero sampling recipe has been reproduced. Unavailable thermal telemetry is disclosed; no temperature is inferred. Current AC and memory preflight passed, with checks repeated by the runner at heavy-unit boundaries. Actual experiment results remain pending until receipts are independently inspected below.

## Actual n2 result: independently verified

**PASS_WITH_LIMITATIONS** for the correctness of this bounded no-update result. Raw receipt SHA-256 is `4531add153b93aeedd2e78073b58d9c6724875e9357aa9062cd04a25bbd5a0cf`; raw rollout SHA-256 is `e5c4938f299fe2b3901cc6121a04c727f619147a633d88646f755e801fd65212`. Independent replay matched image bytes, prompt hashes, training eligibility, strict rewards and zero advantages. There were **six of six strict-valid answers, four of six correct answers, and three of three zero-variance groups**. Every answer was FLAIR. Reasoning wording varied, showing that output text diversity alone did not produce useful task-reward diversity. No optimizer update occurred and no adapter checkpoint was created; gradient and reload gates were correctly not exercised. All four logged resource samples were NORMAL on AC.

One further independently scoped diagnostic was authorized: four samples per group, two groups, seed 11, same starting SFT28 checkpoint, system instruction and 256-token cap, on the first two training-only prompts covering both sequence labels. Review SHA-256: `4293e9efe24e765c2c44661b27c6469bb720f871fb345859a246fc269987a506`. This increases stochastic coverage to investigate the n2 result, with at most one update. It is not authorization for a longer run. Both seed and group size change, so this cannot estimate a causal group-size effect. If task variance remains absent, further automatic scaling is prohibited under this approval; diagnosis should turn to answer bias and visual/task construction.

## Independent schedule finding: sequence supervision was one class

Reconstructing `_schedule_train_rows(train, 28, 7)` reproduces all **112 step/category/patient triples** in the actual SFT28 receipt. The 28 sequence targets are **28 FLAIR and zero contrast-enhanced T1**. The sampler chooses the lexicographically smallest record ID within each patient/category; the FLAIR record sorts ahead of T1. Thus balancing categories and patients did not balance labels within the sequence category. For comparison, tumor presence targets are 14 yes / 14 no; enhancing-presence targets are 11 yes / 17 no.

This is a concrete training-data selection weakness and a plausible explanation for the observed FLAIR answer mode. It does not prove that all bias is caused by the schedule; no matched balanced-label SFT intervention has been run. Preserve SFT28 and its results. Any future supervised control must declare its label sampling and use a separately named experiment; do not silently replace or retrain the preserved checkpoint. The private reproducible aggregate evidence is `experiment_contract_strict_activation_v2/independent_sft28_schedule_audit.json`.

## Actual n4 result and final activation verdict

**PASS_WITH_LIMITATIONS** for the correctness of the second bounded no-update result. Receipt SHA-256: `0288416839c9d465238856aa2b8511612c57a9a250cb3c7400df182781fc8ab0`; raw rollout SHA-256: `f728fe94bde272f39fa8c1d65fdbba9f2038bf491a608d2a7d65ed74c9305b24`. Independent replay verified eight strict-valid answers, four correct answers, two zero-variance groups, zero optimizer steps and no saved adapter. All answers normalize to FLAIR, including a nested `\text{FLAIR}` answer correctly accepted by the pinned strict parser. All three logged resource samples were NORMAL on AC.

Across both diagnostics: **14/14 strict-valid answers, 8/14 task-correct answers, 5/5 zero-variance groups and zero optimizer updates**. These are correlated training-patient diagnostic observations, not independent-patient accuracy estimates. The starting adapter was loaded and checked against its saved tensors by the runtime. **Gradient magnitude, changed post-update weights and post-update checkpoint reload remain NOT_VERIFIED because no update was attempted.** The correct engineering response was to skip zero-advantage groups; no useful task gradient was demonstrated. Additional automatic sample-count or training-duration expansion is not authorized by this review.

The schedule exposure reconstruction has two independent receipt cross-checks: all 112 step/category/patient entries match, and every actual sequence example records 16 target tokens. The pinned tokenizer gives 16 tokens for the historical FLAIR target template and 19 for contrast-enhanced T1. Historical source at commit `3906df3852462055661108bf112e85bcac62b076` uses the same template and lexical selection. The historical receipt does not contain source record IDs/image hashes, so exact image-row linkage is reconstructed from code and manifests rather than directly recorded; that is a reproducibility limitation. Future SFT receipts should record those identifiers and hashes under ignored private data.

## Future SFT schedule correction: CPU review only

**PASS_WITH_LIMITATIONS** for the versioned schedule implementation. Independent execution on the actual training manifest confirms that `legacy_lexical_v1` still reproduces the historical 28 FLAIR / zero T1 exposure, while `balanced_sequence_v2` produces 14 FLAIR / 14 T1 with the same 112-example budget, identical per-category patient order and unchanged non-sequence records. Forty-four focused regression tests pass, including missing-label and nontraining-patient rejection. Future private receipts now save source record IDs, image hashes, target labels and target hashes.

This review does not authorize a new SFT job and no SFT28 retraining occurred. It does not establish that balancing labels will improve visual understanding; a separately reviewed matched experiment would be required. Changing the SFT source makes the previous diagnostic approval source hashes stale, correctly preventing their reuse for another model launch. Historical approval files and raw results remain preserved.

## Separate tumor-presence diagnostic approval

The sequence-specific failure does not establish failure of other MRI rewards. A separate **PASS_WITH_LIMITATIONS** authorizes a tumor-presence diagnostic using three exact original training records from three patients, all at the same axial level, selected before candidate generation: one zero-mask slice and two large positive masks (1,017 and 2,320 native mask pixels). The reviewer independently loaded all source NIfTI files, verified matching affines/shapes, recomputed those mask counts, checked nine source-file hashes and three PNG hashes, and visually inspected all paired images. The positive slices contain conspicuous focal abnormalities. Zero annotation on the negative is a mask-derived target, not independent clinical proof of tumor absence.

Contract identity: `18cd0fa85f821b83bbd94e9f7480307d4c9dce3e7e442eb68a9f2b019a27d47e`. Review SHA-256: `6455ad723a704e3a85adefaee8b22048600e2d9f6a53b19ae3cfa26ec3c7664d`. Scope is group size two, maximum three groups, 256-token cap, seed 7 and at most one update from the same preserved SFT28 adapter. The new live preflight passed. All six exact reviewed source files were copied to the private contract's `source_snapshot` and rehashed against the approval. No previous stale approval was reused. There is no authorization for automatic expansion; actual reward/gradient/reload receipts still require review.

The reviewer also independently decoded all 14 saved sequence rollout token-ID lists using the pinned local tokenizer. Every decoded response matches the text that was scored, after omitting special EOS tokens and trimming whitespace. This checks text/token alignment in the actual evidence rather than relying only on saved reward scalars.

## Actual tumor-presence update: independently verified

**PASS_WITH_LIMITATIONS** for one genuine strict task-reward GSPO-token diagnostic update. The first training-only negative slice produced two strictly formatted answers, `yes` and `no`, with recomputed task accuracy `[0, 1]`, equal format reward `[1, 1]`, equal zero overlong penalties and total rewards `[0.2, 1.0]`. Token IDs independently decode to the scored text. The sample-standard-deviation advantages match the saved values. Old/current teacher-forced log-probability maximum absolute errors were `[0.0, 0.0]`, and the finite gradient L2 norm was **1.3902573521807733**.

The runtime performed exactly one optimizer step. Independent CPU safetensors comparison confirms the before adapter is tensorwise equal to the preserved SFT28 checkpoint, every one of the 504 adapter tensors changed, all 10,911,744 adapter elements remain finite, total parameter-change L2 is **0.10120659851975045**, and maximum absolute change is **0.00003162305802106857**. The before SHA is `244c26fd5ce2e3b02b46703d4f46a90608f2ef1a868eeee682ee4fe48a58ed73`; after SHA is `1eb17b675f02d464eb8db0a4da2a3a0dd2262610d87c10de5c8988cfefc07446`.

A fresh base-model reload restored all 504 saved adapter tensors exactly and generated a strictly formatted `no` answer on the same image. The reviewer inspected the reload implementation and actual receipt, verified its image/adapter/hash linkage, and independently compared saved tensors; the reviewer did not launch a second GPU reload. All four resource samples were NORMAL on AC. Source receipt SHA: `dd99c720ce3281bafcd93b8968d126d9b20e1e159bbf2ff176489e912893937f`; raw rollout SHA: `6afbfd9bf85ca9899c24c247a8f7c2b7375ade13a7fe51de5a41be0d100dbb4e`.

The reviewer-authored `independent_dynamic_evidence.json` has SHA `4de4a69a37cc428c50d30236b0eebf2568a28af52275c74844133171a656553f` and binds verified claims to these actual files. It supplies evidence, not authorization for another training job. A single informative group proves technical strict-reward activation; it does not prove MRI generalization, improved visual representations, cross-task transfer or a balancing advantage.

An additional port distinction is now explicit: installed MLX `Adam` defaults to `bias_correction=False`. The observed first-step maximum change is consistent with that setting at the declared learning rate. This is not a claim of optimizer equivalence to native Vero Adam/AdamW; keep the actual result unchanged and declare this before any matched training comparison.

## Narrow development inference review

**PASS_WITH_LIMITATIONS** for one fixed 54-row internal-development comparison using the new after adapter, existing matched SFT prompts, native template, greedy decoding and 256-token cap, with resource guarding and current AC verification. No previously examined eight-patient test set is authorized. The `frozen_eval` implementation verifies exact loaded adapter tensors and records prompt/image/model/decoder hashes; no optimizer executes.

The existing aggregate scorer counts `format_valid` when the format score is greater than zero, which includes half-credit outputs. Consequently, strict-format validity must be computed as format score exactly one; a corrected or separate field is required before labeling the post-update count strict. This inference review does not authorize model selection or further training based on the four development patients.

## Subsequent interface corrections

The independently reported half-credit formatting issue is corrected: aggregate strict validity now requires format score exactly one, with a regression for partial-format outputs. Fourteen focused scorer/activation tests passed. The smoke runner now records `ACTIVE_GROUP_GRADIENT_PENDING` and the saved before-checkpoint hash during parity/backpropagation, avoiding a misleading zero-update status while an active group is still running. Historical completed receipts remain unchanged.

Initial review of the new multiarm child-delegation implementation is **NOT_VERIFIED** pending fixes: the parent stages `train.verl.jsonl` inside the child's output directory, while the smoke runner rejects every nonempty output; and the child verifier initially did not require the saved before-checkpoint hash to equal the authorized starting checkpoint. The reviewer also requested raw prompt/token evidence and old/current log-probability checks at resume boundaries. These are concrete source findings reported before any multiarm launch approval.

## Independently reproduced development comparison

**PASS_WITH_LIMITATIONS** for the descriptive before/after measurements. The reviewer compared every row's model path/config/template hashes, adapter-config hash, prompt, image hash, visual placeholder count, manifest hash, temperature and token cap; these match across runs. Only the expected adapter weights differ. Both checkpoints produce 54/54 strict-valid answers. Correct answers change **36/54 → 37/54**:

| Evaluation category | Before | After |
|---|---:|---:|
| MRI sequence | 15/24 | 15/24 |
| Tumor presence | 10/12 | 11/12 |
| Enhancing presence | 8/12 | 9/12 |
| Tumor grid | 3/6 | 2/6 |

Exactly three correctness outcomes change: two improve and one regresses. Equal-category accuracy remains **0.65625 → 0.65625**; worst-category accuracy falls **0.5 → 0.3333**. Mean per-patient question accuracy changes **0.66346 → 0.68407**. One development patient gains two correct answers, another loses one, and two are unchanged. All 55 resource samples are NORMAL on AC.

The post-update prediction file SHA is `51823d6e6ebe4800ec17366ba3446ccf7030baf23c46cf9c5c3e58e639fff280`; the preserved pre-update file SHA is `c65f5e4343f8f9bc3ee6d77c71c5a87dbd794b7280f5c7c85cf937d515142d9e`. This is a descriptive cross-task tradeoff after one training-group update on four repeatedly examined development patients. It is not a significance result, robust negative-transfer demonstration, new independent test evaluation, or proof of improved MRI representations. Aggregate accuracy alone would conceal the unchanged task macro and worsened weakest task. The full cross-task transfer matrix and controlled mixing comparison remain unrun.

## Final multiarm CPU orchestration review

**PASS_WITH_LIMITATIONS** for CPU orchestration after the reported defects were fixed. Child manifests now live outside the run-output directory, and an integration test reaches the real smoke runner's output checks using inert MLX stubs. The verifier requires the before-checkpoint SHA to equal the authorized starting adapter, finite old/current log-probability differences within tolerance, actual question and system text, exactly one image placeholder, prompt/token/image hash linkage and valid response-token inventories. Resume checks revalidate completed child artifacts and reject tampering. Forty-nine focused tests passed in the reviewer's final run.

This is not a model-launch approval. Parent and child reviews still require exact scopes and independently supplied digests; no unknown future checkpoint is preapproved. Optimizer moments reset between child groups and exact optimizer resume is false, now explicitly recorded. Incomplete children are preserved and require explicit recovery; automated completion of a partially executed child is not implemented. Completed child boundaries can be resumed without repeating their updates. These limitations prevent claiming native continuous Vero optimization from this bounded orchestration port.

A narrow additional no-image control is approved for diagnosis only. Independently removing the image content from all 12 development tumor-presence questions yields one identical message list; the labels are six yes and six no. One deterministic generation per adapter therefore evaluates the unique no-image prompt efficiently. Report two actual generations total and their constant-prompt scores against the 12 labels; do not create fictitious per-patient inference records. The real-image before/after result remains the primary measured comparison.

## Optional no-image control execution status

**NOT_RUN — power gate rejected before model loading.** After the category selector was corrected, the live power check observed an AC-header/battery-detail conflict: the battery was at 85% and explicitly reported discharging. No model generation occurred and no prediction outputs were produced. The prepared private script is `presence_no_image_control_v1/run_control.py`. The gate was honored; no retry or inference result is claimed. The proposed two-generation control remains unexecuted until safe power conditions are verified.
