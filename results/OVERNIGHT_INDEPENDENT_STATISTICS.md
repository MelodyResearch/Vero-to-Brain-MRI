# Independent statistical check of the saved MRI experiments

Status: `TESTED_CPU` audit of existing private predictions and receipts, 2026-10-15 local time. This audit launched no model job and made no training update. It reports aggregates only; patient keys, images, gold rows, predictions, and adapter files remain in ignored `local_data/`.

## Recomputed historical held-out results

I read all six complete 107-row prediction files under `local_data/frozen_eval_v1/`, the frozen held-out manifest, and the private gold crosswalk. For every row I checked unique record ID, prompt, manifest SHA-256, and image SHA-256. I independently normalized the exact-answer outputs and computed correctness, validity, and the equal-patient/equal-category mean over 8 patients × 4 routes. Counts match the existing published audits.

| Checkpoint | Correct / 107 | Exact-answer valid / 107 | Equal-patient/category accuracy | Relevant paired comparison |
| --- | ---: | ---: | ---: | --- |
| Frozen BF16 Qwen3-VL base | 45 | 96 | 0.333333 | Base versus Vero: 43 both correct, 2 base only, 1 Vero only |
| Frozen BF16 released Vero | 44 | 96 | 0.328125 | Difference from frozen base: −0.005208 |
| Matched 4-bit LoRA starting adapter | 53 | 100 | 0.432292 | SFT4 changes one incorrect grid item to correct |
| Four-step SFT adapter | 54 | 101 | 0.447917 | Difference from matched starting adapter: +0.015625 |
| Exploratory proportional RL adapter | 54 | 101 | 0.447917 | Scored outcomes identical to SFT4 |
| Exploratory uniform RL adapter | 53 | 100 | 0.432292 | One SFT4-correct grid item becomes incorrect |

The frozen base/Vero comparison joins different training histories. The 4-bit starting adapter is already 10 percentage points above frozen BF16 base on this metric; the gain cannot be assigned to the four SFT steps. The explored test set has only eight patients, one per site, and the one-item paired differences do not support significance or negative-transfer claims. These eight patients have influenced subsequent research decisions, so they are no longer a fresh prospective test for revised prompts or rewards.

## Actual RL optimization and denominators

I independently counted step receipts for each arm and compared them with arm-level receipts. All arms attempted 8 groups × 2 image-conditioned rollouts; actual response tokens were unequal. With the strict direct reward, the specialist, proportional, and uniform arms each had **0/8 varying groups and 0 optimizer steps** (response tokens 2,132 / 2,835 / 2,989). Their final adapter hashes equal the SFT4 start. These arms have no learned transfer matrix.

Under the **post hoc single-box reward**, specialist/proportional/uniform had **0/8, 1/8, and 2/8 varying groups**, respectively, and **0, 1, and 2 optimizer steps** (response tokens 2,132 / 2,719 / 3,181). The proportional and uniform checkpoint hashes differ from the common start. Only these two arms produced real exploratory RL updates. Their unequal active updates, route exposures, and token work prevent attributing the one-item score difference to uniform versus proportional sampling.

## 28-step SFT and internal development baseline

The 28-step receipt contains **112 image/question examples**, four routes per step, from **28 distinct internal-training patients**; its exposed patient keys are a subset of training and have **zero overlap** with four internal-development and eight historical test patients. It records 28 finite, nonzero gradient norms. Both saved before/after adapter byte hashes match the receipt and differ from each other. Mean of the first four saved teacher-forced loss terms is 6.9009; mean of the last four is 0.1670. This is evidence of a completed weight-updating SFT run, not of generalization. The receipt does not save optimizer and RNG state for exact continuation.

The first **original-prompt** internal-development inference file has 54 verified rows from four patients: 24 sequence, 12 tumor-presence, 12 enhancing-presence, and 6 tumor-grid. Each output matches its manifest prompt/image hash and the same SFT28 adapter hash; decoding used temperature 0 and 256-token maximum. None of the 54 newly generated strings contains `<think>` or `<answer>` tags, although 48 contain a boxed answer. Consequently **strict Vero format and strict task accuracy are both 0/54**. This does **not** establish that 28-step SFT failed to learn a format: the original development prompt omitted the SFT training system instruction. A newly built, separate matched-system manifest should be evaluated with the same model and decoding before that causal conclusion.

As an explicitly exploratory semantic diagnostic, the existing single-box-anywhere extractor finds **35/54 correct**: sequence 15/24, tumor presence 11/12, enhancing presence 9/12, grid 0/6. Its equal-patient/category macro is **0.572917** across 16 patient-category cells. A training-majority-answer control scores **26/54**, macro **0.416667**, on this same development pack. These 35 answers are **not valid strict Vero rewards**; boxed extraction was introduced after the strict gate failed. No SFT28 matched starting-adapter inference or no-image/image-swap controls have yet established an SFT28 effect or visual grounding.

## Discrepancy and interpretation ledger

| Issue | Audit result | Consequence |
| --- | --- | --- |
| Question rows treated as independent | 107 historical test questions come from 8 patients; 54 development questions from 4 patients. Three tumor tasks share mask lineage. | Use patient as the unit for uncertainty, with category aggregation; do not report question-level significance. |
| Original-prompt SFT28 format score | 0/54 strict versus 48/54 boxed and 35/54 post hoc box-correct. | Wrapper absence is observed; prompt mismatch and generation-prefix behavior remain competing causes. |
| Historical test reuse | The eight test patients were used during reward/prompt development. | Preserve those results as exploratory; require a new untouched cohort after protocol freeze. |
| Internal-dev scorer validation | `score_outputs` now checks each prompt, image SHA-256, image placeholder, deterministic decoding, and common model/adapter settings. The original 54-row file passes these stronger checks. | Keep the same validation for later matched-prompt and cross-run comparisons. |
| Balance effect | Strict arms made zero updates; exploratory arms made 0/1/2 updates. | No causal evidence for task balancing or cross-task transfer yet. |
| Visual grounding | The 28-step semantic score exceeds the simple majority control but has no matched before-adapter or counterfactual-image comparison. | Accuracy can still reflect priors, patient/source features, or answer shortcuts; test images and labels separately. |

The next statistical gate is a **paired, same-prompt, same-precision internal-development comparison** of the 28-step adapter with its saved starting adapter, including strict format, post hoc semantic correctness, and no-image/swapped-image controls. Keep development findings exploratory; only a future frozen protocol and untouched patients can support a generalization claim.
