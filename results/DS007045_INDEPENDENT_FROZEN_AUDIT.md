# Independent patient-level frozen MRI diagnostic

Status: `INFERENCE_COMPLETED` for the paired 107-question base/Vero checkpoint comparison; `TRAINING_COMPLETED` for the separate four-step SFT LoRA engineering control below; `SCIENTIFICALLY_VALIDATED` does not apply. Audit date: 2026-10-15. The frozen-model comparison is **not** Vero-to-MRI recipe transfer or a weight-updating RL result.

## Dataset and receipt checks

The bounded official OpenNeuro `ds007045` v2.0.1 source supplies 40 patients across eight sites, with 32 training and eight held-out patients, one held out per site; see the [source and label audit](../data/PATIENT_KEYED_MRI_ELIGIBILITY.md) for official links and version caveats. I independently reread all **120 MRI/mask payload files** and confirmed their local byte lengths and SHA-256 hashes against the private pinned Git-annex receipt. Each patient has the three expected files. Across the private prompt and gold crosswalk, there are no train/test patient-key or prompt-ID overlaps; 288 training and 72 test rendered PNG hashes have zero overlap. The 107 held-out image/question pairs use 72 distinct PNGs. All four categories occur in all eight held-out patients; tumor-grid questions number one or two per patient. Four prompt templates contain no subject IDs or mask/label text. This establishes a **within-source patient-held-out engineering comparison**, subject to the source and label limits below.

The analysis specification and control manifest were fixed in ignored private files before Vero outputs were read. The primary descriptive statistic is the unweighted mean of each patient's accuracy within each category, followed by an equal-category macro average. A training-only majority answer was fixed per category. The independent scoring script accepted all 214 main prediction rows only after checking uniqueness, exact prompt/category, current image and manifest SHA-256, one image placeholder, `temperature=0`, `max_tokens=256`, and Vero's shared override of the base chat template. The same `score_frozen_eval.py` parser was used for both checkpoints. Recorded placeholder and file hashes do not prove identical internal visual-token counts or processor behavior.

Private main-output file digests for reproducibility: base SHA-256 `48a2937d4c9326fd5b201bc0b1782136433aaae5e86e3d2e0f66f1d89ba392c4`, Vero shared-template SHA-256 `9a854ab39a85cf9f8ee49155ad1c75e4fb7c72631f1df1f30c35f335f8b0afef`. The local model configuration hashes in those receipts are `5cd452860dc1e9c29dd71cc3cef7f39b338b7a40793f7a260655c2d3568f3661` and `b8bfebee7659d5b82bef87492df8d8b3d9597fce45363e5e2a4997057c9c5d02`, respectively. Private output files themselves are excluded from Git.

## Paired results

| Category | Held-out questions / patients | Training-majority baseline on heldout | Base correct / valid | Vero correct / valid | Base patient mean | Vero patient mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FLAIR vs contrast-enhanced T1 | 48 / 8 | 24/48; 50.0% patient mean | 26 / 48 | 25 / 48 | 54.2% | 52.1% |
| Any tumor on fixed slice | 24 / 8 | 11/24; 45.8% | 10 / 24 | 11 / 24 | 41.7% | 45.8% |
| Enhancing tumor on fixed slice | 24 / 8 | 16/24; 66.7% | 9 / 24 | 8 / 24 | 37.5% | 33.3% |
| Entire-tumor centroid in 3 × 3 grid | 11 / 8 | 2/11; 18.75% patient mean | 0 correct / 0 valid | 0 correct / 0 valid | 0% | 0% |
| **Raw pooled, descriptive only** | **107 / 8** | Categories not exchangeable | **45 correct, 96 valid** | **44 correct, 96 valid** | **33.3% category macro** | **32.8% category macro** |

The paired patient-category macro difference (Vero minus base) is **−0.52 percentage points**. Seven of eight patients have the same macro score under both checkpoints; one has a −4.17-point difference. At the individual-question level, sequence has one base-only correct; any-tumor has one Vero-only correct; enhancing presence has one base-only correct; all other pairs agree in correctness. The worst category is the grid task at 0% for both. With eight patients and one per site, these differences have no useful precision and do not support a significance claim.

The simple majority baseline's category macro is **45.3%**, above either model on this selected task pack. Both checkpoints mostly emit fixed answers: base/Vero chose FLAIR on 46/48 and 45/48 sequence prompts; “yes” on 23/24 and 24/24 any-tumor prompts; “yes” on 23/24 and 24/24 enhancing-tumor prompts. The enhancing task's training-majority answer is “no,” explaining its especially poor performance relative to a trivial baseline. These answer distributions warrant visual-ablation controls before calling any correct response image-grounded.

Both checkpoints returned **zero parseable grid cells in 11 prompts**. Their grid responses were long explanatory text (median roughly 900 characters), frequently appearing to stop mid-answer under the configured 256-token maximum; exact generated-token counts were not recorded. This is an output-format/budget failure under the prespecified strict exact-cell scorer, not isolated evidence that either model cannot spatially localize a tumor. A larger or differently prompted evaluation must lock a parser and separate format compliance from spatial correctness before touching a new held-out set.

## Locked visual-input controls

The private prespecified manifest contains 16 within-patient z=91 FLAIR/T1ce image swaps (two per held-out patient, same question and original gold, with a separate expected answer for the swapped image) plus one no-image and one uniform-gray image for each of the three tumor categories on a fixed first held-out patient/slice. The manifest SHA-256 was fixed before the main model comparison: `52c46f848ba03c00363758ac688cc540137c7916253aa8c4123953693702f82d`. Both checkpoints completed all 22 controls under the same shared-template and 256-token settings. The independent audit checked unique records, exact prompts, source-image and manifest hashes, image-placeholder counts, and decoding settings. Private control output SHA-256 digests are base `c70b3c25fadd1d85c8eb5195d89d931ae20581918b4af13f5ac8c16a9cc8e258` and Vero `9cb643123f3b615b5d99619f84c8f15ddc6632b4768b23db23042e7607c0918c`.

| Prespecified control | Base | Vero | Interpretation |
| --- | ---: | ---: | --- |
| Swapped-sequence image: answer matches the **new visible image**, not the original gold | 9/16 | 8/16 | Each swapped set has eight FLAIR and eight T1ce targets; always saying FLAIR gives 8/16. |
| Sequence answer changed relative to the real-image paired question | 2/16 | 4/16 | Only two changes for each model moved toward the new visible-image label. |
| No-image tumor controls: correct / valid across any-tumor, enhancing-tumor, grid | 2/3; 2/3 | 2/3; 2/3 | Both models correctly said “yes” for the two presence questions without pixels; grid was invalid. |
| Gray-image tumor controls: correct / valid across the same three questions | 0/3; 2/3 | 1/3; 3/3 | Both said “no” for the two presence questions, which were wrong on this one selected slice; Vero's gray-image grid answer happened to match gold. |

Swapped-image outputs remained heavily FLAIR-biased: 15/16 for base and 14/16 for Vero. A single-patient gray/no-image tumor probe cannot establish general visual reliance or lack of it. The controls nevertheless rule out treating these frozen scores as convincing image-grounded tumor reasoning; correctness can occur without the tumor image, and a visually counterfactual sequence swap seldom changes the answer. Control outcomes were excluded from the 107-question main metrics.

## Matched-precision four-step SFT adapter control

A separate base-initialized **4-bit, rank-4 LoRA SFT** engineering run updated adapter weights for four optimizer steps using 16 real image/question/answer examples from **14 of the 32 training patients** across the four categories. Its private receipt records zero held-out patients used in training, four finite loss/gradient steps, and different before/after checkpoint SHA-256 values: before `502084c8a1394fdffd765741b9b0fbfcb6720268dfe7671c20d7a67dc5f86088`, after `39c5c48796a53bd82ef3184f54a266935b30f686af841b850bd981aa061ed8ca`. The independent audit verified those bytes, the adapter config, training-only anonymous patient-token crosswalk, and that **504 saved adapter tensors** were exactly reloaded for each of 107 before and 107 after predictions. Every inference row matched the same held-out prompt/image/manifest hashes, base model configuration and chat template, one image placeholder, `temperature=0`, and 256-token maximum.

| Category | Matched 4-bit LoRA **before** correct / valid | **After four SFT steps** correct / valid | Patient-mean accuracy before → after |
| --- | ---: | ---: | ---: |
| FLAIR vs contrast-enhanced T1 | 26 / 48 | 26 / 48 | 54.2% → 54.2% |
| Any tumor on fixed slice | 13 / 24 | 13 / 24 | 54.2% → 54.2% |
| Enhancing tumor on fixed slice | 11 / 24 | 11 / 24 | 45.8% → 45.8% |
| Tumor-centroid 3 × 3 grid | 3 correct / 4 valid of 11 | 4 correct / 5 valid of 11 | 18.75% → 25.0% |
| **Raw total / equal-category patient macro** | **53 correct; 100 valid / 43.23%** | **54 correct; 101 valid / 44.79%** | **+1.5625 percentage points** |

The four steps changed **six of 11** grid response strings and no responses in the other three categories. Exactly **one** grid answer changed from incorrect/invalid to correct/valid; no other correctness or validity changed. The patient-category macro improvement is consequently one patient's grid score, not a broad cross-category improvement. The training-majority macro baseline is 45.31%, slightly above this after-SFT result. This is an actual weight-update and matched-precision before/after inference receipt, yet four steps and one changed correct answer do not establish a useful SFT learning curve or generalization effect. No SFT-specific no-image or swapped-image controls were run, so visual reliance remains unverified for this adapter.

The SFT before/after outputs have private SHA-256 `ff2d5e20c0ee9598571eda8f0d432f6ab156294063027e6d0b277f78e3c1a9bd` and `c6ce22532949aebd6bfa5c516d20f78dcefbc5bccf0a5d707bfdbec5c37ccbc3`, respectively. The larger numerical gap between 4-bit LoRA SFT **after** (44.79%) and the earlier BF16 frozen base (33.33%) or Vero (32.81%) is **not** attributable to four SFT steps: the matched 4-bit LoRA **before** state already scored 43.23%. Use the before/after contrast for any local SFT effect, and keep BF16 frozen comparisons descriptive.

## Scope and significance

The sequence label is supported independently by BIDS acquisition metadata. The three tumor tasks share an expert-reviewed, algorithm-initialized tumor mask; they are correlated derivatives of one reference source, not three independent biological labels. Images are registered processed derivatives, and scanner/site appearance may yield shortcuts. This convenience sample has only one held-out patient per site, no external-patient crosswalk with MSD, no established lack of pretraining exposure, and no clinical endpoint. The fixed slice positions were set before mask inspection, but a central slice grid task still needs visual and format validation.

- **Scientific significance:** No clinical, neurobiological, or longitudinal conclusion follows. Accuracy here is not tumor diagnostic performance.
- **AI/ML significance:** This is a within-source, patient-keyed frozen base/Vero MRI comparison plus a four-step SFT adapter engineering control. Neither frozen checkpoint demonstrates a robust advantage here; both fail to exceed simple category baselines on the macro metric. The matched SFT before/after change is one correct grid answer. No balanced GSPO or MRI RL comparison occurred, so the Vero task-balanced GSPO hypothesis remains untested.
- **Experimental significance:** The same-source, patient-disjoint data infrastructure now supports controlled training experiments without waiting for longitudinal change labels. First resolve grid output-format failure and image reliance on a separate development subset, then lock a larger patient/site evaluation and compare identical-base SFT, single-task RL, proportional-mixture GSPO, balanced GSPO, and MedGRPO/active-group controls with matched image-token/rollout budgets and full cross-task evaluation. Do not tune on these eight test patients.

Raw patient keys, images, masks, and generations remain in ignored local files and must not be committed.
