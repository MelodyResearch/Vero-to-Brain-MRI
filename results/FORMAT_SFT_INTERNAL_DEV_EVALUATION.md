# 28-step format SFT: matched internal-development evaluation

Status: `TRAINING_COMPLETED` for SFT; `INFERENCE_COMPLETED` for the paired development and visual-control runs; exploratory, not a prospective patient test. All images and predictions are under ignored `local_data/`. The previously examined eight test patients were not used here.

## Completed training and equal-settings comparison

The 28-step run used 112 real-image/question examples from 28 internal-training patients, four routes, a 4-bit Qwen3-VL-8B base and rank-4 language LoRA. Twenty-eight finite nonzero gradient norms and changed adapter bytes verify optimization. The saved starting/final adapter SHA-256 values are `502084c8a1394fdffd765741b9b0fbfcb6720268dfe7671c20d7a67dc5f86088` and `244c26fd5ce2e3b02b46703d4f46a90608f2ef1a868eeee682ee4fe48a58ed73`. First-four/last-four mean teacher-forced losses were 6.9009/0.1670. The original run did **not** save optimizer or RNG state for exact resume.

The fixed internal-development manifest contains 54 image-linked questions from four patient-disjoint development patients: 24 sequence, 12 tumor-presence, 12 enhancing-presence and six 3×3 tumor-grid. The matched-prompt manifest SHA-256 is `9e5194cc808ad27592549b3fab21e31fd6ba0a9582ca76fac75928b589b9734f`. Both adapter runs used the same 4-bit base, native Qwen3-VL chat template, MRI images, system instruction, greedy decoding and 256-token cap. The strengthened scorer checks each prompt, image SHA, visual placeholder, deterministic settings and model/adapter receipt.

| Adapter and prompt | Strict format | Strict task-correct | Patient mean accuracy | Sequence | Tumor presence | Enhancement | Grid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Starting adapter, **matched SFT system instruction** | 0/54 | 0/54 | 0 | 0/24 | 0/12 | 0/12 | 0/6 |
| 28-step adapter, **matched SFT system instruction** | **54/54** | **36/54** | **0.6635** | 15/24 | 10/12 | 8/12 | 3/6 |
| 28-step adapter, original user-only prompt | 0/54 | 0/54 | 0 | 0/24 | 0/12 | 0/12 | 0/6 |

The user-only probe had 48/54 boxed answers and 35/54 correct under a **post hoc single-box** extraction diagnostic. The missing system instruction explains the immediate 0/54 wrapper failure in that probe: the same SFT28 weights and images produce 54/54 strict wrappers when the training instruction is included. The starting adapter still fails the strict contract under the matched instruction, so the 28-step SFT was necessary for this observed format recovery. This does not prove the factual answers came from MRI pixels.

## What the SFT learned, and did not establish

All 54 matched SFT28 responses contained `<think>`, `<answer>` and a boxed final answer. In 51/54, the reasoning span was the identical generic sentence `Visible MRI evidence.` The SFT target itself used that fixed phrase, so the run is primarily **format and short-answer supervision**, not validated anatomical reasoning supervision. The 54 generated responses averaged 66.5 characters and maxed at 81 characters; the inference receipt recorded a 256-token ceiling but not per-response token count, so exact cap-hit frequency is `NOT_VERIFIED` despite no visibly cut-off answer.

The sequence predictions were FLAIR for 21/24 images and contrast-enhanced T1 for 3/24; the true sequence labels were balanced 12/12. The model has a substantial FLAIR bias. Tumor-presence and enhancement share mask lineage with grid targets; their scores are not three independent clinical findings. The fixed axial slice and tiny positive masks create additional shortcut/answerability concerns documented in the image audit.

## Same-patient/slice counterfactual image controls

To test pixel dependence, each of the 24 sequence questions was rerun with the opposite sequence image from the **same patient and same axial slice**, while holding the system and question text fixed. The control manifest SHA-256 is `c2f709242f25d5a3d3621bfd1f0f40f59d6ffcf6320613bfcf363d4d375aad11`. Original/swap response file hashes are `c65f5e4343f8f9bc3ee6d77c71c5a87dbd794b7280f5c7c85cf937d515142d9e` / `6249117b8543277b4858a5d76497c81314be5e896446bb2a20af919a3e32f179`.

The no-image manifest SHA-256 is `420c0dc5a152d3dc3e9d5e1505525f038537bf8b06a245fdc03452fa369061d4`; its output SHA-256 is `607888eddd082c5d04e9320134362099bb5dc6a411c58bb5534caefce11835d8`.

| Sequence image condition | Correct against visible sequence | Prediction changed from original | Strict format |
| --- | ---: | ---: | ---: |
| Original image | 15/24 | reference | 24/24 |
| Same-patient/slice opposite-sequence image | 15/24 | **6/24**; all six changed in the correct direction | 24/24 |
| No image, same question | **12/24** | **3/24** from original | 24/24 |

The no-image model answered FLAIR on **all 24** sequence questions; its 12/24 score is exactly the balanced class-prior baseline. Correct images improve only three questions beyond that baseline, and 18/24 paired sequence swaps leave the prediction unchanged. Thus SFT28 shows **partial, weak visual dependence** for sequence recognition; format recovery is much stronger than MRI visual discrimination. Four development patients and repeated slices are too few for a generalization claim. A more informative subsequent control is a larger prospective, patient/site-heldout sequence cohort with protocol cues balanced and a simple supervised image classifier.

## Decision

**GO** for a bounded strict-reward *activation* diagnostic on training patients: the SFT28 checkpoint can now emit valid strict Vero-format answers and some task-correct answers. **Do not GO** yet for a full balanced multi-task effect claim: task-reward variance within real rollouts, gradient/update correctness, two independent visually answerable routes, and resistance to slice-position/mask shortcuts remain unverified. Preserve the original user-only probe and all earlier zero-update RL arms as historical results; neither should be silently rescored into a successful direct transfer experiment.
