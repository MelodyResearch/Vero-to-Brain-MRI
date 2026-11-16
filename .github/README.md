# Vero-to-Brain-MRI

**Verifiable reinforcement learning for brain MRI visual reasoning**

A controlled development study of image-grounded MRI sequence discrimination: **can Vero-style verifiable reinforcement learning and GSPO improve the distinction between FLAIR and contrast-enhanced T1 images?**

**Status:** Completed development study; training and method exploration are closed. The repository provides code and saved evidence, not a clinically validated model or a release of patient data and checkpoint weights.

[Results and evidence](../README.md#verified-accuracy-and-scope) · [Reproducibility](../REPRODUCIBILITY.md) · [Detailed technical overview](../README.md)

## Study at a glance

| Question | Evidence |
| --- | --- |
| What was compared? | The retained Parent Adam52 endpoint, direct GSPO, Paired-Prompt GSPO, and balanced paired supervised fine-tuning |
| What was evaluated? | FLAIR versus contrast-enhanced T1 on a repeatedly explored, training-disjoint development set: 24 images from four patients |
| What was found? | Neither tested GSPO continuation improved the retained parent on this matched development comparison |
| What remains untested? | The independent 16-person / 32-image evaluation remains **NOT_RUN** |

## Matched development results

| Endpoint | Correct / total | Accuracy |
| --- | ---: | ---: |
| Retained Parent Adam52 | 18 / 24 | 75.00% |
| Direct GSPO A54 | 16 / 24 | 66.67% |
| Paired-Prompt GSPO B54 | 16 / 24 | 66.67% |
| Balanced paired SFT Adam80 | 15 / 24 | 62.50% |

**Interpretation:** These are development results, not an untouched test or a clinical validation. The study does not establish a stable MRI accuracy advantage for GSPO. Its earlier 18 RL arms, three SFT controls, and 22 frozen 100-question endpoints form a separate historical evaluation and are not pooled with this table.

## Reproduce the saved evidence

From the repository root, the documented CPU checks are:

```bash
python -m pip install -e '.[test]'
python scripts/reproduce_results.py --check results/FINAL_METRICS.json
python -m pytest -q
python scripts/evaluate.py --validate-only
```

These commands inspect saved evidence and validation logic; they do not constitute a fresh MRI evaluation. Native inference requires private inputs and separate authorization. See the [reproduction boundary and requirements](../REPRODUCIBILITY.md).

## Explore the repository

| Location | Contents |
| --- | --- |
| [research/](../research/) | Final results, model handoff, evidence index, and historical study summaries |
| [results/](../results/) | Saved metrics, eligibility evidence, and provenance manifests |
| [src/](../src/) | Maintained result-recount, scoring, and validation code |
| [tests/](../tests/) | Retained CPU tests |

This is an independent adaptation study, not an official Princeton collaboration. Negative comparisons, evaluation limits, and historical evidence are retained rather than presented as performance gains.
