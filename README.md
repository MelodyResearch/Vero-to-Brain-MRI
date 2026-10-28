# Vero-to-Brain-MRI

**Verifiable Reinforcement Learning for Brain MRI Visual Reasoning**

An independent research project investigating how Princeton Vero-style heterogeneous verifiable reinforcement learning and GSPO can be adapted to image-grounded brain MRI reasoning. This is not an official Princeton collaboration or a clinically validated model.

## Final selected model

**P_calibrated13 / Adam52 / boundary114** is the sole principal model: the best verified endpoint among four models evaluated under the same Concise Prompt/F32/strict greedy DEV24 contract. Adapter SHA256:

`c70b426b58191446545ba6333f7595e96f4fdb4cc689ddbaf5206e276b329db6`

Its adapter, full Adam/RNG, exact4-bitbase, processor and configuration identities are verified in the [model eligibility record](results/FINAL_MODEL_ELIGIBILITY_REVIEW_V1.json). [Model handoff](research/FINAL_MODEL_HANDOFF_V1.md) describes authorized private access; weights and patient data are not in Git.

## Best verified MRI results

DEV24 contains24images from **four TRAIN-disjoint, repeatedly explored development patients**, at three levels and both modalities. These are development results, not an untouched clinical test.

| Model | Strict accuracy / balanced accuracy | FLAIR recall | Enhanced T1 recall | Invalid | Correct modality pairs |
|---|---:|---:|---:|---:|---:|
| **Selected Parent Adam52** | **18/24 —75.00%** | 10/12 | **8/12** | 0/24 | **7/12** |
| Direct GSPO A54 | 16/24 —66.67% | 9/12 | 7/12 | 0/24 | 5/12 |
| Paired-Prompt GSPO B54 | 16/24 —66.67% | 9/12 | 7/12 | 0/24 | 5/12 |
| Balanced paired SFT Adam80 | 15/24 —62.50% | **12/12** | 3/12 | 0/24 | 3/12 |

**No tested continuation improved this Parent. Training and method exploration are closed.** The final SFT completed28realupdates but strengthened FLAIR preference; it is retained as a negative control. SFT results are not GSPO results.

The earlier18RL+3SFT cohort and22frozen100-question endpoints are a separate historical evaluation. B28 scored52/84sequence questions; the three historicalSFTseeds scored54/84,57/84,52/84. Allseeds and full-format/native scoring distinctions remain in [machine metrics](results/FINAL_METRICS.json). These percentages are not pooled with DEV24 or used to pick a new test-bestseed. HistoricalSFT lacks matchedDEV24 outputs, so the selected Parent is not claimed globally best.

## Main findings and limitations

- Original sparse modality exposure and uneven K4correctness signal were real limitations. Correcting full paired exposure with this dense SFT was insufficient.
- Direct and Paired-Prompt GSPO both regressed on the matched development readout. Local MLX numerical checks do not establish full officialVero parity.
- Class bias, template/label weighting, inherited moments and limited adaptation remain competing mechanisms; their unique causal contributions were not isolated.
- Dallas is an engineering control. Midline/SLAKE work is paused; no second clinically qualified heterogeneous learning task or clinical reasoning claim is established.
- Independent16-person/32-image evaluation is **NOT_RUN**: original execution metadata is not currently located, and explicit final unsealing confirmation is still required. No replacement patients or output-based exclusions are allowed.

## Results, paper and evidence

- [Final results and research report](research/FINAL_ACCURACY_RESULTS_V1.md)
- [Internal paper draft](research/FINAL_ACCURACY_PAPER_DRAFT_V1.md)
- [Machine metrics](results/FINAL_METRICS.json) and [independent model/state review](results/FINAL_MODEL_ELIGIBILITY_REVIEW_V1.json)
- [Condensed scientific history](research/HISTORICAL_SUMMARY.md) and [evidence index](research/EVIDENCE_INDEX.md)
- [Model handoff](research/FINAL_MODEL_HANDOFF_V1.md)
- [Collaborator summary draft](research/COLLABORATOR_SUMMARY.md), not sent

## Reproduce the saved results

```bash
python -m pip install -e '.[test]'
python scripts/reproduce_results.py --check results/FINAL_METRICS.json
pytest -q
```

This is an independent CPU recount of SHA-bound saved evidence, not a new inference run. It cannot turn development results into independent generalization.

```bash
python scripts/evaluate.py --validate-only
```

The final evaluator defaults to non-native validation and currently reports missing original inputs/confirmation. Native evaluation requires exact private payloads, independent scope review, fresh researcher approval and resource protections. Its new wrapper is CPU-tested, not newly native-qualified. See [reproducibility and execution boundaries](REPRODUCIBILITY.md) and the [fixed final protocol](configs/final_evaluation_v1.json).

The final source retains the unchanged native compatibility and original22recount dependencies. Obsolete process logs/launchers are removed by a separate cleanup PR; their immutable history is indexed, not rewritten. Main is not auto-merged and this PRIVATE repository is not a public model release.
