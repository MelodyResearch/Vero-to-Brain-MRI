# Answer-likelihood sprint — independent engineering and scientific review

Status: `REVIEW_REQUIREMENTS_PREPARED`; experiment `NOT_RUN` by this reviewer. Review type: isolated same-family Codex agent; private cross-model review `NOT_VERIFIED`. No MLX/MLX-VLM import, model execution, MRI-volume loading or training. No model-launch approval is granted by this document.

## Mission alignment and scope

The training-only diagnostic identifies whether supervision changed conditional label preference, image sensitivity or primarily target formatting before further Vero-to-MRI RL work. It preserves A0/A28/B28 and the negative development/control results. The competing explanations remain supervision/optimization, decoding, training-versus-development shift, image-independent priors, task suitability and implementation error; none is selected in advance. No sealed prospective or historically examined test patients may select cases, metrics or thresholds. B changed sequence-image/label exposure, not only text labels.

## Checkpoint identities independently verified

Actual files were streamed through SHA-256 and safetensors headers inspected without decoding weight arrays. All three contain504 F32 tensors with identical names/shapes and43,708,920 file bytes; common adapter configuration SHA `032becabb7cbb6d2af6f3451eb127e5f944551d42eb51eaee3e8d75b839b2f6a`.

| Checkpoint | Actual adapter SHA-256 |
| --- | --- |
| A0 original starting adapter | `502084c8a1394fdffd765741b9b0fbfcb6720268dfe7671c20d7a67dc5f86088` |
| Historical A28 | `244c26fd5ce2e3b02b46703d4f46a90608f2ef1a868eeee682ee4fe48a58ed73` |
| Corrected B28 | `b316db3d5eadf2e1fc0cbe80ccc93c7f7405858769f7d32b39a137dea3d6c572` |

Private exact paths and identity receipt: `local_data/ds007045_bounded/likelihood_independent_review_v1/checkpoint_identity_PRIVATE.json`. These checks establish saved-file identity; each actual process must still verify exact504-tensor adapter assignment and report base/config/processor/tokenizer/template identities. A0 is the original starting adapter, without inventing a new pure-base equivalence test.

## Required estimands and limitations

1. Use one fixed, versioned assistant prefix preceding the label. A natural source-matched choice is the existing generic observation plus answer/box opening. Both candidates must have exactly the same prompt and prefix tokens. No generated rationale, candidate-specific observation or full-target suffix may enter the primary label-only comparison.
2. Save each label token's conditional log probability, raw sum, summed NLL and mean token log probability/NLL. Prespecify a length-aware comparison, such as mean-logp margin `T1 minus FLAIR`, before outputs. Report raw-sum margin alongside it. Mean normalization is a tokenizer-dependent heuristic and can favor a different candidate; report disagreements, not a uniquely unbiased class probability.
3. Softmax of raw sums gives relative mass for these two candidate token strings under the stated prefix; softmax of mean scores is a different relative preference statistic. Neither is calibrated probability over unrestricted model answers. Label entropy with the exact score transformation, natural-log units and two-candidate denominator; it is not total model uncertainty.
4. Record separate prefix, label and suffix loss components and reconcile their weighted sums with whole-target CE. Closing-format tokens are conditioned on the candidate label; a shared textual suffix does not imply shared probability or causal formatting-gradient dominance. Token count, token CE and gradient contribution are different quantities. Gradient attribution requires a separate scoped experiment.
5. Primary unit is the matched patient/slice pair. With margin m=T1−FLAIR, image separation `m(T1-image)−m(FLAIR-image)` measures directionally appropriate preference movement; also report whether each image is correctly preferred. Positive separation alone need not mean both classes are recognized. Checkpoint differences should be paired on identical visible inputs. Training fit is not patient generalization.
6. Same-question swaps are visible-input permutations of the original pair and must use replacement-image truth. They provide execution/direction checks, not additional independent examples. Identical no-image prompts are one unique conditioning input per checkpoint; repeated rows are replication, not independent patient evidence. Any other-patient control needs a fixed mapping and displayed-image truth before scores are seen.

## Token and visual-conditioning requirements

Existing SFT source concatenates native prepared prompt IDs with target IDs, forwards `ids[:-1]`, selects targets from `ids[1:]` and masks before prompt_length−1. For label span [a,b) within the target, selected predictions are [prompt_length+a−1,prompt_length+b−1). Required regression tests must reject off-by-one indices and prompt/closing-tag contamination.

The actual pinned tokenizer must verify that the common prefix is a token prefix of both complete targets, that the label span decodes exactly, and that no token crosses either label boundary. Reject incompatible concatenation/tokenization rather than silently approximating spans. This must cover both raw text tokenization and the native prepared prompt including expanded image placeholders. Save token IDs, decoded pieces or offsets, prompt/prefix lengths, span indices and special-token behavior. Historical FLAIR2/T1five-token counts are expectations to recheck, not hardcoded correctness.

Installed MLX-VLM source inspection shows `prepare_inputs` has distinct image and text-only paths; Qwen's `should_add_special_tokens` returns true under the current utility. Native preparation must match that behavior rather than use an untested text-only tokenizer shortcut for MRI. The image branch must provide finite nonempty pixels, `image_grid_thw`, expanded image-token count consistent with spatial merging, and all required model kwargs. No-image must have no pixel tensor or image placeholders. Preserve actual attention masks; with one unpadded example an all-one assertion can justify the existing SFT convention. Reject padding assumptions silently applied to batches.

Qwen3-VL `Model.__call__` recomputes vision/deepstack embeddings and mRoPE from full IDs and image-grid metadata when no cache is supplied. Use fresh noncached forwards for this first diagnostic and no stale cached image features across controls. Forward the full causal input, but slice only needed logit rows before float32 log-softmax where practical to limit temporary memory. No backward pass or optimizer belongs in this read-only probe.

## Smallest actual-forward acceptance evidence

The first launch must bind one checkpoint, a predetermined training-only subset, exact candidate/prefix definitions, numerical tolerance, number of candidate forwards, optional bounded generation count, output path and current source hashes. A first stage of at most two unique input conditions and two candidates per condition is a useful upper bound; serial checkpoint changes and later conditions require new exact scopes. This is a recommendation, not authorization.

For every first-stage condition require:

- Exact adapter assignment, eval mode and recorded prepared prompt/pixel/grid/mask identities and shapes.
- Finite selected token log probabilities <=0, raw/mean/NLL/entropy arithmetic independently reproducible from saved token values.
- Alignment evidence: first label prediction comes immediately after the common fixed prefix; the final label token is included and the closing brace/tag excluded.
- Shared-prefix consistency across two complete candidate forwards. Both candidate first-token log probabilities extracted at the same causal position should agree across the two forward contexts within a declared tolerance. Prefix scores must not change because future candidate tokens were appended.
- Whole-target CE equals the weighted decomposition into fixed prefix, answer and suffix; optional SFT-equivalent loss on the same target must match the corresponding saved token sum within tolerance.
- A repeated identical visible input or swap-permutation check where included; numerical differences must be bounded before interpreting image sensitivity. No demand that genuinely different images yield different scores, since that would prejudge the hypothesis.
- Actual greedy outputs, when included, generated from the original prompt without forced label prefix, with explicit native256-token/stop metadata. Forced-prefix preference and free generation answer are separate results.
- Evaluate arrays before recording/timing, persist each completed output before post-forward gate, release large logits and caches between candidate forwards. Apply existing live gates before imports/load and each candidate forward, after computation, after generation if any, and on model release. Single heavyweight process;18GiB memory guideline/256MiB cache and8GiB35% load/6GiB30% unit rules remain. Thermal unavailable stays explicit.

## Review decision

`PASS` for the three saved checkpoint identities and source-derived review requirements only. Scorer implementation, actual native visual processing, likelihood values and first model launch remain `NOT_VERIFIED / NOT_RUN` until exact source, CPU regressions, input contract and finite approval are independently reviewed. No old B training/evaluation approval covers this new experiment.

## First scorer source review — independent findings before launch

Current focused tests independently pass9/9 in0.02s with MLX/MLX-VLM/Torch/Transformers imports blocked. Actual tokenizer-only replay reproduces11 common-prefix tokens,2 FLAIR/5 T1 label tokens and3 suffix tokens. The ordered contract places the first real pair first and the single no-image condition third; the earlier unrun contract remains preserved.

**L01 CONFIRMED_DEFECT (P1, reproducibility):** installed generation code is a package, but backend binding looks for `generate.py` and silently skips it. Actual scope contains six backend files, omitting `generate/dispatch.py`, `generate/ar.py` and `generate/common.py`, which own generation/cache execution. Required correction: bind actual required files and reject absent required dependencies.

**L02 CONFIRMED_DEFECT (P1, resume integrity):** `completed_prefix` only checks truthiness of candidate/comparison dictionaries. An independent reproduction with one candidate's impossible +999 mean log probability and a NaN comparison is accepted as one completed record. The existing numeric-evidence test did not exercise changed numeric values. Required correction: validate complete candidate/partition/token evidence and recompute summaries/comparisons/oracle consistency plus frozen record fields before counting a persisted row. Preserve the failing-then-passing case. A fresh first run has no prefix, but correcting before immutable identity freeze avoids a preventable later migration.

**NO_ISSUE_FOUND in inspected causal/cache path:** Qwen3-VL full forwards create fresh visual features and explicit mRoPE positions without a KVcache. The language wrapper passes those positions into its inner model, which constructs a causal mask; the caller's all-one2D mask is not being used as a noncausal additive attention matrix in this installed implementation. Native actual-forward consistency still requires the prefix oracle and declared0.002 tolerance. No runtime correctness is inferred solely from this source inspection.

No launch approval issued pending these bounded corrections and final scope/resource review. The original source hashes and independent reproduction are retained in `likelihood_independent_review_v1/initial_source_review_PRIVATE.json`.

## Independent correction recheck — L01/L02 resolved

**REPRODUCED / PASS_WITH_LIMITATIONS for source and CPU behavior.** Required installed generation dispatch/ar/common files are now included in nine backend hashes and absence fails closed. Resume checks both candidates, actual-tokenizer partitions, causal indices, finite nonpositive token log probabilities, recomputed summaries and preference/entropy values, oracle agreement and frozen record fields; the same validator runs before output persistence. Independent14-test run passes in0.02s with native/model imports blocked. Additional reviewer tests reject the original +999/NaN case, changed real-tokenizer partition, changed frozen prompt and an intentionally absent generation dispatch file. Actual ordered57-record contract plan and pinned tokenizer replay pass.

The proposed first scope remains A0, first matched pair only: two inputs, six teacher-forced forwards including two prefix-only oracles, two greedy generations capped256 tokens each, zero updates. Contract SHA `32c2a636fa1b51b06b2bc4ce576e1084782dd148964e6e662bfd169a60e94e85`. No third/no-image input or later checkpoint is covered. Native values remain NOT_RUN; this is preparation for the actual alignment/resource probe.

**Launch HOLD:** current actual resource sample reports BATTERY91%, which fails the existing AC gate. No model-launch approval has been issued. Fresh passing AC samples are required before an exact approval is bound; CPU review remains permitted. Source/retest evidence is saved in `likelihood_independent_review_v1/source_correction_recheck_PRIVATE.json`.

## Bounded battery amendment: independent first-probe admission

The researcher subsequently authorized a separate read-only battery diagnostic. The original BATTERY91% HOLD remains accurate under the earlier AC requirement. The new opt-in gate was independently source-reviewed, and 24 focused tests plus eight independently specified threshold cases passed with native model imports blocked. The global training resource gate is unchanged. Two genuine initial samples at least30 seconds apart must both pass before MLX imports; no watt measurement or missing thermal reading is inferred.

**PASS_WITH_LIMITATIONS for exactly A0's first two frozen real-image inputs:** six forward calls and two greedy generations capped at256 tokens each, zero updates. The ordered contract remains `32c2a636fa1b51b06b2bc4ce576e1084782dd148964e6e662bfd169a60e94e85`. Exact source/backend hashes, output absence, checkpoint identity, finite budget and battery policy are bound into the independently verified approval. Native execution remains NOT_RUN at approval; no next stage is authorized.

Approval SHA256: `f7b816807cf05c347e7a462122612cf863d07900041626e17eabbe20d6088858`, private `likelihood_independent_review_v1/approval_A0_battery_002.json`. Launcher: `launch_A0_battery_002.sh` beside it. Runtime must reject start below75%, stop at65% or below, the declared rapid drain/stage drain/swap limits, and every existing memory/concurrency/thermal reason. The earlier tiny native CPU fixture must have released before the lead launches this process. Failed numerical consistency currently stops before detailed candidate values are persisted; this is an accepted diagnostic limitation for the first bounded probe, requiring preserved failure logs and review before any rerun. Same-family engineering review is not clinical or cross-model validation.

## Actual A0 first attempt: resource STOP before first forward

Actual log and four resource events independently verify three admission CONTINUE events followed by STOP at `quantized_model_ready`. No output JSONL exists: **zero candidate forwards, zero prefix-oracle forwards, zero generations, zero completed records and zero updates**. During the 5.480-second model-loading interval, swap grew 1.766416 GiB, available memory fell 11.485→7.745 GiB and pressure-free percentage fell64→38. Battery stayed88%; other-heavy inventory stayed zero. Reported MLX active/peak allocations were4.818/6.845 GiB.

Both `sustained_swap_growth` and `stage_swap_growth` were real gate conditions. **The stop behavior passes; continuing or retrying this configuration is not approved.** This observation does not establish that the load transient was harmless, nor that subsequent forwards would be safe. Post-process recovery can establish recovery, not validate overriding the stopped gate. No scientific likelihood result was obtained.

The failure occurs before the runner's current cleanup `try/finally`, so this trace has no `model_released` event; the lead reported process exit1. Moving cleanup coverage to include model setup would improve failure/release evidence. Any lower-peak materialization revision must preserve the saved quantization/adapter identity, undergo independent checks and receive a new finite approval. Caps must not be raised merely to pass this trace. Private audit: `likelihood_independent_review_v1/A0_first_attempt_STOP_PRIVATE.json`.

## Serial materialization revision: separately bounded approval

Independent source review confirms the new helper evaluates existing parameter graphs one leaf at a time, without changing numerical conversion, base quantization or adapter settings. Existing memory/battery/swap caps are unchanged. A concrete failure-cleanup defect was independently reproduced: an observer exception retained the helper's parameter list through its traceback after the caller deleted the model. The author added a failing regression and cleared helper references in `finally`; the reviewer's separate weak-reference reproduction then passed. Setup is now covered by caller cleanup, with observations every64 leaves and at the final leaf.

Fresh focused verification: **26 tests passed** under a model-import blocker. The lead's saved native CPU fixture and executed source were inspected: nine quantized tensors plus three quantized forward outputs matched exactly, with final helper hash `d2d446e1910e9a280015465ffab0617c94ba81f80af6fb391b35e3742c8b931b`. This small fixture does not establish lower Qwen peak memory or full-model runtime equivalence.

**PASS_WITH_LIMITATIONS** for the explicit revised loading experiment and at most the same first two A0 inputs. Mandatory fresh two-sample admission and all stop conditions apply; failure causes exit, no automatic retry. No completed likelihood data exist from the prior attempt. Private approval `approval_A0_battery_serial_v2_002.json`, SHA256 `74ad9db753ba60685640a6f146eb93fcc8cd1d01d5a0934ed3b9ddd3e2c47124`, and launcher `launch_A0_battery_serial_v2_002.sh` are under `likelihood_independent_review_v1/`. The launcher retains the literal venv interpreter path rather than resolving its Python symlink.

## Actual serial A0 first pair: numerical and resource replay

**PASS_WITH_LIMITATIONS; two records completed.** A separate model-free implementation recomputed every per-token partition, answer/fixed/prefix/suffix/whole-target score, candidate preference/entropy and oracle disagreement from actual saved values. Both maximum prefix disagreements were exactly0; both generations stopped after12 tokens. Prompt-token, ones-mask and grid fingerprints reproduced; both real PNG hashes matched, and distinct native pixel fingerprints were recorded. Pixel tensor preprocessing was not independently rerendered. Clean source-bound completion reached the native before/after adapter equality assertions.

The current attempt's48 resource events all replay as CONTINUE. Battery87→86%, minimum available6.192GiB, minimum pressure-free34%; swap7.293086→7.287656GiB with maximum7.295469GiB; reported MLX peak5.977558GiB. Release active memory was approximately0.000001GiB. This observed lower peak differs from the failed bulk-materialization attempt; differing initial system conditions prevent assigning the entire resource difference causally to the loader. The narrow memory reserve remains a concrete limitation.

A0 mean T1-minus-FLAIR margins were−2.67880906 and−2.07975275 for the frozen FLAIR/T1 pair; raw margins−13.39404924 and−10.39881256. This single training-patient pair is a diagnostic, not a scientific efficacy estimate. Output SHA256 `663f3847809b408741f934d28d7d9c0aba0270c03078e89af1822eba3c1b1c9b`; independent audit SHA256 `ed97ff92d3c0af197746a52af0c62de808673324a3917e2d3a77350bce8c4ddc`.

Following the lead's finite efficiency request, the next authorized A0 scope is exactly seven frozen inputs (unique no-image plus the next three fixed pairs), cumulative9, with unchanged caps and source. It supersedes the prepared but unused one-record continuation. This avoids a separate full model reload for no-image while preserving every per-forward, per-generation and battery/resource stop. No A28/B28 launch is preauthorized; each follows actual preceding-stage audit.
