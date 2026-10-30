# Vero-to-Brain-MRI

**Verifiable Reinforcement Learning for Brain MRI Visual Reasoning**

An independent research project investigating how Princeton Vero-style heterogeneous verifiable reinforcement learning and GSPO can be adapted to image-grounded brain MRI reasoning. This is not an official Princeton collaboration or a clinically validated model.

## Research question

Can verifiable reinforcement learning improve MRI sequence discrimination while preserving image dependence and cross-task performance? The completed experiments did **not** establish a stable MRI accuracy advantage. Training and method exploration are closed.

## Final selected model

**P_calibrated13 / Adam52 / boundary114** is retained among the four verified same-contract development endpoints. Adapter SHA256:

`c70b426b58191446545ba6333f7595e96f4fdb4cc689ddbaf5206e276b329db6`

[Model handoff](research/FINAL_MODEL_HANDOFF_V1.md) and [state/config qualification](results/FINAL_MODEL_ELIGIBILITY_REVIEW_V1.json) identify its exact base, processor, adapter and full Adam/RNG. Private weights and patient data are not distributed in Git.

## Verified accuracy and scope

DEV24 contains 24 images from **four TRAIN-disjoint, repeatedly explored development patients**, with three levels and both modalities. It is not an untouched clinical test.

| Model | Strict accuracy / balanced accuracy | FLAIR recall | Enhanced T1 recall | Invalid | Correct modality pairs |
|---|---:|---:|---:|---:|---:|
| **Selected Parent Adam52** | **18/24 — 75.00%** | 10/12 | **8/12** | 0/24 | **7/12** |
| Direct GSPO A54 | 16/24 — 66.67% | 9/12 | 7/12 | 0/24 | 5/12 |
| Paired-Prompt GSPO B54 | 16/24 — 66.67% | 9/12 | 7/12 | 0/24 | 5/12 |
| Balanced paired SFT Adam80 | 15/24 — 62.50% | **12/12** | 3/12 | 0/24 | 3/12 |

The earlier 18 RL arms, three SFT controls and 22 frozen 100-question endpoints are a **separate historical evaluation**: 2,286 attempted RL groups and 362 accepted updates. All seeds and native/full-format scoring distinctions remain in [machine metrics](results/FINAL_METRICS.json). These results are not pooled with DEV24 or used to select a test-best seed.

## Key findings

- Direct and Paired-Prompt GSPO both regressed on matched DEV24. No tested continuation improved the retained Parent.
- Dense balanced paired SFT completed 28 updates but increased FLAIR preference; it is a negative control, not a GSPO improvement.
- Sparse modality exposure and uneven K4 correctness signal were limitations. The unique causal effects of inherited moments, target weighting and adaptation capacity remain unresolved.
- Dallas is an engineering control; a second clinically qualified heterogeneous learning task was not established.

## Paper and evidence

[Internal paper draft](research/FINAL_ACCURACY_PAPER_DRAFT_V1.md) · [Final results](research/FINAL_ACCURACY_RESULTS_V1.md) · [Evidence index](research/EVIDENCE_INDEX.md) · [Historical summary](research/HISTORICAL_SUMMARY.md)

## Reproduction

```bash
python -m pip install -e '.[test]'
python scripts/reproduce_results.py --check results/FINAL_METRICS.json
python -m pytest -q
python scripts/evaluate.py --validate-only
```

The CPU recount verifies saved SHA-bound numerical evidence. The evaluator defaults to fail-closed validation. Native evaluation additionally requires private inputs, fresh review/permission and an explicit immutable historical engine dependency; see [reproducibility](REPRODUCIBILITY.md). The current source packaging is CPU-verified, **not newly native-qualified**.

The maintained package has seven source files for result recount, strict scoring, F32 checks, plan protection and read-only evaluation. Retired training/debug implementations are accessible at fixed Git commits; native evaluation still depends on 90 exact historical source files, exported explicitly rather than maintained in the current tree. The [single deletion/dependency manifest](results/SIMPLIFICATION_MANIFEST.json) records this boundary.

## Limitations

Independent 16-person/32-image evaluation remains **NOT_RUN**: original execution metadata is not currently located and explicit final unsealing confirmation is required. No replacement patients, output-based exclusions, clinical validation or reliable generalization claim is permitted. Finite-context MLX numerical checks do not establish full official Vero equivalence. This PRIVATE repository is a reproducible negative development study, not a public model release.
