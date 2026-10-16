# Post-SFT28 strict Vero reward activation

Status: **`TRAINING_COMPLETED` for one bounded exploratory MLX/LoRA update**, with independent Codex verification; **not scientifically validated**. The historical SFT28 checkpoint, its 54/54 matched-format and 36/54 task-correct development results, the original failed probe and earlier zero-update RL arms remain unchanged. No SFT retraining was performed.

## Execution policy and actual review

The former unconditional rejection was replaced with `mri-execution-tiers-v1`: reproducible CPU work (A), capped exploratory model work (B), and reviewed substantial/confirmatory training (C). The first Tier B diagnostic needs actual data/prompt/reward/resource checks and separate engineering approval, but does not need a previous optimizer receipt. Any update must establish within-group task variation, finite gradients, changed weights and fresh reload before expansion. The independent reviewer was **`/root/activation_reviewer`, an isolated Codex agent**. Claude did not review these runs; no earlier sharing denial was bypassed.

Each training approval bound the contract, exact code, starting adapter, generation settings, route, seed, output directory and finite budget. The expected approval digest came from the actual reviewer's message, independently of the approval file. This protects against stale or substituted evidence within trusted orchestration; it is not cryptographic protection against a malicious process running as the same OS user. Exact reviewed source snapshots and private receipts are preserved under each ignored contract directory. The reviewer independently checked all four local base-model shard hashes against the cached pinned Hugging Face revision, the SFT28 adapter, patient splits, rendered images and masks.

All runs used the preserved SFT28 adapter `244c26fd5ce2e3b02b46703d4f46a90608f2ef1a868eeee682ee4fe48a58ed73`, the matching SFT system instruction, native Qwen3-VL template, `direct_vero` reward, temperature 1, top-p 1, top-k 0, no minimum-p/repetition penalty, and a 256-token ceiling. Each permitted at most one update and stopped immediately after an eligible group. To isolate task reward, the diagnostic accepted only groups whose answers all had strict format score 1 and whose task accuracies differed. This diagnostic selection rule is stricter than native Vero's group handling. Length penalties were zero at these response lengths.

## Runs that actually executed

| Diagnostic | Attempted groups × rollouts | Strict format | Task-correct samples | Zero-variance groups | Optimizer updates |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sequence, seed 7 | 3 × 2 | 6/6 | 4/6 | 3/3 | 0 |
| Sequence, seed 11 | 2 × 4 | 8/8 | 4/8 | 2/2 | 0 |
| Same-level tumor presence, seed 7 | 1 × 2, then stopped | 2/2 | 1/2 | 0/1 | **1** |

Both sequence runs answered FLAIR on every rollout. Text varied, but answers and within-group task rewards did not; none hit the token ceiling. The second probe changed both seed and group size and therefore cannot estimate a causal group-size effect. Combined, these are 14 strict-format outputs and five zero-variance groups from one training patient, not 14 independent cases. No automatic further sequence scaling followed.

For the separate presence diagnostic, the data agent selected three original training records at z=91, across three patients, before seeing their model predictions: one zero-mask negative and two positive slices with 1,017 and 2,320 native mask pixels. Independent mask/affine/hash checks and local inspection supported diagnostic suitability. All shared the axial level, removing level alone as a discriminator within this tiny cohort; a constant yes still scores 2/3 and other shortcuts remain. The run stopped on the first, negative slice after finding an eligible group, so **only one training image and two rollouts were used for the actual update**. The other two selected images were not executed. A slice-negative tumor patient is not a healthy control, and these are source-mask references rather than new clinical adjudications.

## Verified optimizer and checkpoint evidence

The presence group had task accuracies `[0, 1]`, equal strict-format scores `[1, 1]`, rewards `[0.2, 1.0]` and sample-standard-deviation-normalized advantages of opposite signs. Thus the varying reward term was task correctness. The old/current log-probability maximum absolute differences before updating were `[0.0, 0.0]`; the gradient L2 norm was **1.3902573521807733**. One optimizer step changed the adapter hash to **`1eb17b675f02d464eb8db0a4da2a3a0dd2262610d87c10de5c8988cfefc07446`**.

Independent CPU inspection found all **504 adapter tensors** changed, all **10,911,744** adapter elements finite, total parameter-delta L2 **0.1012065985**, and maximum absolute delta **3.1623058e-5**. A fresh base model plus saved adapter load matched every saved adapter tensor, then generated a real-image response. The reload response was `no`; there was no paired pre-update greedy response on this exact training slice, so this is not evidence that the update corrected that slice.

The MLX port uses `mlx.optimizers.Adam(learning_rate=1e-5)` whose installed default is `bias_correction=False`. The observed first-step scale is consistent with that setting. This is a disclosed optimizer difference; full numerical parity with upstream Vero/veRL or AdamW has not been established. The run also uses a 4-bit base, rank-4 language LoRA, frozen visual parameters, small groups and short responses. It demonstrates a local strict task-reward update, not completion of Vero's full native recipe transfer.

All three training runs were on verified AC/charging power and logged NORMAL resource samples (4, 3 and 4 respectively). Thermal telemetry was unavailable and remains explicitly unknown. No healthy task was killed. Raw outputs, masks, checkpoints and patient-linked records remain ignored locally. `scripts/audit_strict_activation.py` replays raw outputs against original labels and checks saved hashes; independent findings are in `STRICT_ACTIVATION_INDEPENDENT_REVIEW.md`.

## Fixed internal-development comparison after the update

A separately reviewed Tier B inference used the existing matched-prompt 54-question manifest from four internal-development patients, the same base/precision/native template, greedy decoding and 256-token cap. The previous eight examined test patients were not used. Every prediction verified the exact loaded adapter tensors and saved model/prompt/image hashes. All 55 resource samples were NORMAL.

| Evaluation task | Preserved SFT28 | After one presence update | Change in correct answers |
| --- | ---: | ---: | ---: |
| Sequence recognition | 15/24 | 15/24 | 0 |
| Tumor presence | 10/12 | 11/12 | +1 |
| Enhancing presence | 8/12 | 9/12 | +1 |
| Tumor grid | 3/6 | 2/6 | −1 |
| All questions | 36/54 | 37/54 | +1 |
| Strict format score exactly 1 | 54/54 | 54/54 | 0 |

Exactly three extracted answers changed: two incorrect-to-correct and one correct-to-incorrect. **Equal-category accuracy is unchanged at 65.625%; worst-task accuracy drops from 50% to 33.33%.** Equal-patient mean accuracy across that patient's questions changes from 0.66346 to 0.68407, which uses a different weighting from equal-category accuracy. Four development patients and correlated mask-derived tasks cannot establish a generalization gain. This is one descriptive row of a prospective cross-task analysis: a presence update affected presence, enhancement and grid, with an observed grid regression. It is not a specialist/proportional/uniform comparison or robust evidence of positive or negative transfer.

Independent review also found that the development scorer counted partial format credit (0.5) as “format valid.” A failing regression reproduced this reporting error; the aggregate now counts only `format == 1`. The historical SFT28 outputs and both sides of this new comparison are all fully formatted, so their 54/54 counts are unchanged. The strict reward implementation was not relaxed.

## New root cause: the SFT sequence exposure was one-class

The historical scheduler chose the lexicographically smallest record ID for each task/patient. For sequence recognition that selected FLAIR every time: **28 FLAIR and zero contrast-enhanced T1 examples**, despite balanced sequence availability in the full training manifest. All 112 saved step/category/patient receipt tuples match the reconstructed historical schedule; all 28 actual sequence target lengths are 16 tokens, matching the pinned FLAIR target encoding versus 19 tokens for T1. Historical receipts did not log exact record IDs, so exact image-row identity was reconstructed from the source schedule rather than directly recorded.

This exposure defect is a plausible contributor to the FLAIR bias, not a causal proof. The future scheduler now has an explicit `balanced_sequence_v2` policy: **14 FLAIR and 14 T1** across the same 28 ordered patients and 112 total task examples, with unchanged non-sequence selections. It fails before model loading if an assigned class is unavailable. `legacy_lexical_v1` preserves historical replay. Future receipts add record IDs, image hashes, target labels and target hashes. This correction was CPU-tested and independently reviewed; **no corrected SFT model was trained**. Fair future comparisons must share the revised exposure and common starting weights; a gain relative to old SFT28 could not be attributed solely to RL.

## Scientific decision

The strict reward/optimizer interface can activate on an eligible MRI task after SFT28. Sequence failure is task-specific in these probes, so it must not block unrelated eligible MRI tasks. The next main experiment still needs reviewed full multi-category data, a defensible common SFT initialization, matched attempted/informative updates and token budgets, plus cross-task and no-image/swap controls. The current three-row dynamic receipt cannot be relabeled as approval for the full multi-category training contract.

MRI scientific benefit and independent-patient generalization remain unestablished. The AI/ML result is a verified task-driven update plus an identified supervision-sampling defect, with no causal balancing benefit or new algorithm claim. The experimental value is that output-format repair, sequence-label exposure, task-reward activation and checkpoint integrity can now be tested separately.

## Final verification and pending control

The final CPU suite passed **154 tests** (`PYTHONPATH=src python3 -m pytest -q`); `git diff --check` was clean. The independent reviewer cleared the corrected sequential orchestration only for tested CPU behavior, with optimizer moments resetting between children and incomplete children requiring explicit recovery. No multiarm model launch was approved or executed in this stage.

The reviewer confirmed that removing images from all 12 development presence prompts yields one identical text prompt, with six `yes` and six `no` references. A bounded control was approved for exactly one greedy no-image generation per adapter (two generations total), to report a constant-prompt baseline without inventing per-patient inference. **This control is NOT_RUN.** The live power check reported an AC header but an explicitly discharging battery at 85%; the corrected gate stopped execution before loading a model. No no-image output was generated and no interpretation of its result is possible. Earlier completed runs retain their own verified AC receipts. Thermal telemetry remains unavailable, not established safe by measurement.

Substantial matched multi-task training remains NOT_RUN: it needs a reviewed full-scope dynamic contract, corrected common sequence exposure, continuous optimizer-state handling for native-parity claims, and eligible power/compute. Official CUDA veRL execution and open-ended retention are also unrun. The project remains active; the bounded update is a completed engineering experiment, not completion of Vero-to-MRI transfer.
