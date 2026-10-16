# Independent audit of the frozen MRI model diagnostic

Status: `INFERENCE_COMPLETED` for the 24 paired, image-conditional frozen predictions and 12 control predictions; `TRAINING_COMPLETED` and `SCIENTIFICALLY_VALIDATED` do not apply. Audit date: 2026-10-15. Private case manifests, images, masks, and raw generations remain under ignored `local_data/frozen_eval_v1/`.

## Receipt and scoring checks

I independently recomputed every score with `src/mri_rl/score_frozen_eval.py` from the private manifest and raw output JSONL. The base and released Vero checkpoint each produced one response for the same 24 record IDs. For all 48 main predictions, recorded manifest SHA-256, prompt text, source image SHA-256, and one Qwen3-VL image placeholder match the current private inputs. Both used deterministic `temperature=0` and `max_tokens=256`. Vero was given the base checkpoint's chat template as a shared-template override; the override hash equals the base template hash. This checks question/image identity and common decoding settings, **not** equality of effective visual-token counts or processor behavior. The two checkpoint configuration hashes differ, as expected.

The 24 prompts span **15 distinct rendered images**, from three sources. Multiple questions share the same tumor image, and the two hippocampus tasks partly share images. Individual predictions are therefore not 24 independent patients. The exact patient lineage of the Medical Segmentation Decathlon sources is not verified. This is a deliberately small, selected diagnostic, not a prespecified held-out cohort or an estimate with useful precision.

| Image task and source | n | Base correct / valid | Vero correct / valid | Simple sample baseline |
| --- | ---: | ---: | ---: | ---: |
| Hippocampal anterior/posterior dominance, MSD Task04 | 2 | 1 / 2 | 1 / 2 | Sample majority 1 / 2; whole Task04 mask prevalence 246 / 260 posterior |
| Hippocampal centroid within 3 native voxels, MSD Task04 | 2 | 0 / 2 | 0 / 2 | Native image-grid center 0 / 2 |
| T1 versus FLAIR sequence, Dallas | 4 | 2 / 4 | 2 / 4 | Sample majority 2 / 4 |
| Enhancing tumor present on fixed slice, MSD Task01 | 8 | 7 / 8 | 6 / 8 | Sample majority “yes” 6 / 8 |
| Entire-tumor center in FLAIR-panel 3 × 3 grid, MSD Task01 | 8 | 2 / 8 | 0 / 8 | Sample majority grid cell 4 / 8 |
| **Pooled descriptive total** | **24** | **12 / 24; 24 / 24 valid** | **9 / 24; 17 / 24 valid** | Categories and sources are not exchangeable |

The paired correctness difference is three base-only correct and zero Vero-only correct cases. The five-category macro accuracy is 42.5% for base and 35.0% for Vero, a descriptive −7.5 percentage-point difference; pooled accuracy differs by −12.5 points. No inferential superiority claim is warranted. The worst category is 0% for both. Among the two parseable centroid outputs, Vero's mean distance was 6.14 voxels versus 24.16 for base, yet neither met the ≤3-voxel criterion. The task's rendered 2D crop does not establish native 3D spatial understanding.

The largest format difference is the tumor-grid task: base gave eight valid cells, while Vero gave one valid cell and seven explanatory or incomplete outputs under the strict short-answer parser. Several appear to stop mid-explanation under the configured 256-token cap; generated token counts were not saved. Thus the 0/8 tumor-grid result mixes spatial failure with answer-format/budget failure; it does not isolate image reasoning. A separate native-Vero-template probe at 256 tokens on one centroid question also yielded an incomplete/invalid answer, and the earlier 48-token native-template probe covered only three prompts. Neither probe is a matched full-set ablation. Increasing token budget and scoring both exact short answers and spatial content would clarify this failure mode, but any new parser or prompt must be locked before an independent test set.

## Visual-input controls

The same three selected questions had no-image and uniform-gray-image versions for each checkpoint, with the same prompt wording and shared-template Vero setting. On the selected anterior-dominance question, **both models answered “anterior” correctly with real, absent, and gray images**. On the selected Dallas T1 question, both said “FLAIR” with real, absent, and gray images, hence all three were wrong. The centroid no-image responses were parseable but wrong, while gray-image responses were invalid. These six matched control pairs identify strong answer priors for the category probes; they do not establish visual reliance for the tumor tasks, which lack matching controls. Four Dallas images from two subjects cannot support protocol-robust sequence recognition.

## Parser and source audit

The strict categorical parser correctly rejects ambiguous free text; all five main routes use the same scorer for both checkpoints. **At audit time**, the centroid parser accepted any response containing exactly two integers, including prose with two unrelated numbers, and did not check coordinate bounds. Its reported mean distance conditions on parseable outputs. This did not create a correct centroid answer here, but called for an explicit coordinate-only grammar and source-image dimensions before larger evaluation. `summarize()` checks record uniqueness and category identity but does not independently verify prompt, image, checkpoint, or manifest hashes; this audit performed those checks separately. The image placeholder assertion proves insertion of one image token marker, not that all model-specific visual tensors were equal. No metadata-only, shuffled-image, source-identification, or tumor no-image control was completed in this diagnostic.

**Post-audit fix:** The centroid parser was tightened to a coordinate-only grammar and the 24-case scores were recomputed unchanged. Source-image coordinate bounds are still not enforced and remain required before larger spatial evaluation.

The Task04 dominance and centroid answers derive from the **same segmentation source** and are not independent task labels. The Task04 crop is hippocampus-centered, and its whole-source dominance is 94.6% posterior. Task01's two answers come from the same tumor mask on the same eight volumes; case-label provenance has not been independently traced for every MSD item. Dallas sequence truth comes from acquisition metadata, but site/contrast shortcuts remain. Source and task are confounded across the three collections.

## Interpretation and next experiment

- **Scientific significance:** None established for brain anatomy, tumor assessment, or longitudinal change. No patient-independent external validation, clinical endpoint, expert adjudication, or outcome analysis was run.
- **AI/ML significance:** The frozen comparison demonstrates a reproducible, image-linked evaluation path on three real MRI sources and a Vero format/budget sensitivity on this tiny task set. It does **not** test Vero recipe transfer, GSPO, task mixing, negative transfer, or an updated MRI checkpoint.
- **Experimental significance:** The next fair inference study needs a locked patient/source split; larger, class-balanced samples; native and shared template arms with comparable answer budgets; image-absent, gray, shuffled, and source-only controls for every task; and a preregistered parser with invalid-rate reporting. The actual transfer claim then needs weight-updating base-initialized single-task, proportional, balanced, SFT, and MedGRPO/active-group arms with matched rollout and visual-token budgets and a full cross-task matrix.

No model weights were updated in this frozen comparison. The private raw generations, manifest, local checkpoint files, and medical images must not be committed.
