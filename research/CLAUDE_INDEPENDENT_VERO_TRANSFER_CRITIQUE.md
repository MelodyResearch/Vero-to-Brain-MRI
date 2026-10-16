# Claude Code independent Vero-to-MRI methods critique

**Date:** 2026-10-15. **Status:** independent, aggregate-only scientific review; no neural inference or training was performed by Claude. Codex requested this review after the Vero-to-MRI priority correction. Claude's response was generated before Codex supplied its own conclusions.

## Provenance and access boundary

- The installed Claude Code CLI at `/opt/homebrew/bin/claude` returned `loggedIn: true`, `authMethod: claude.ai` when queried on the host. The sandbox's separate credential context returned `loggedIn: false`; the host result enabled the run.
- Invocation: `claude --tools '' --strict-mcp-config -p` with a 4,635-byte, subject-free prompt supplied on stdin. The command exited 0 and yielded a 9,428-byte written response. No file tools or MCP tools were available to Claude for this review.
- Claude received only the primary research objective and aggregate data/method facts: Task04 260 image/mask pairs and 14:246 dominance labels; unresolved patient linkage; Dallas aggregate 20 subjects/40 sessions/44 volumes; Task01 metadata counts; and the implementation status of the sampling/reward/GSPO surrogate. **No MRI image, individual receipt, subject identifier, model output, or source code was sent.** Its response explicitly states it read no papers, code, or skill. This is a methods critique of supplied facts, not independent verification of those facts or a Norman paper audit.
- Raw CLI response is kept in the ignored workspace `work/claude_aggregate_review_raw.txt`, outside Git. The summaries below retain its substantive disagreements.

## Independent findings

1. **Task admission.** Claude classified both Task04 questions as engineering probes: dominance is 94.6% posterior and centroid may be explained by a target-centered crop. They are two answer formats from one mask, so they cannot establish transfer among independent task categories. Dallas sequence recognition has an acquisition shortcut; longitudinal change lacks independent truth. Task01 tumor tasks remain candidates after local mask and lineage validation. Claude judged that no current task has yet passed a strong scientific admission gate.
2. **Shortcut tests.** For dominance, use a constant posterior comparator and report balanced accuracy, anterior recall, MCC and answer entropy alongside accuracy. Failure would be approximately 94–95% accuracy with zero anterior recall and balanced accuracy near 50%. For centroid, compare with fixed image-center and training-set mean locations; inspect prediction variance, truth correlation and translation/recropping response. Add no-image, target-occluded, geometry/intensity-only and label-shuffle probes. A rising composite reward with flat answer accuracy or template repetition would indicate reward hacking.
3. **Causal training controls.** Match common starting weights, image/question set, token and rollout budgets, update count, unique patient/image exposure, and per-task **effective nonzero-advantage group counts**. Contrast uniform versus proportional mixing; each specialist evaluated on all tasks; same-data SFT; reward calibration; active-group filtering; and ratio-preserving filtering. Record both attempted and effective category proportions. Source-specific labels may let the model recognize cohort rather than learn transfer, so at least one task replicated across sources or distinct tasks within a source is needed for stronger attribution.
4. **Inference now.** Claude supported bounded frozen-model inference for engineering feasibility and reward-signal diagnosis, explicitly without patient-generalization claims. It recommended patient lineage resolution before a held-out comparison, then base/Vero scoring against simple baselines and no-image/occlusion controls. It warned that no transfer or algorithm novelty conclusion follows from such a pilot.
5. **Significance thresholds.** MRI utility requires patient-separated, visually grounded, shortcut-beating results and an independently meaningful reference. AI/ML transfer requires a filled cross-task matrix from common weights with seed variation and exposure accounting. Algorithm innovation requires a distinct mechanism that survives ablation against direct Vero transfer, reward calibration, active-group and ratio-preserving controls. The current pure-Python GSPO objective is an implementation check, not training or an algorithmic contribution.

## Discrepancy ledger

| Optimistic interpretation to avoid | Claude's independent assessment | Decisive evidence |
| --- | --- | --- |
| Initial CPU contract equals complete Vero transfer | No rollout, autograd, optimizer or actual weight update, and official source parity is unverified | Pinned official implementation comparison plus end-to-end weight-updating run and checkpoint/log receipt |
| 260 Task04 images equal 260 independent patients | Only volume count is established | Patient/crop lineage or explicitly qualified volume-level engineering analysis |
| Two Task04 prompts demonstrate cross-category transfer | Shared image and segmentation truth; answer formats alone are weakly distinct | A separate admitted task with independent target provenance, ideally crossed by source |
| Dominance accuracy can demonstrate reasoning | Posterior prevalence alone gives 246/260 = 94.6% | Minority recall and balanced accuracy above prevalence/geometry/no-image controls |
| Centroid is nontrivial localization | Centered crop may make constant center competitive | Lower error than center/mean baseline and sensitivity to controlled translation |
| Task/category gain implies MRI generalization | Source, site and acquisition may explain the gain | Patient-separated matrix with source-balanced or replicated tasks and visual ablations |
| Equal mixing alone is novel | It is a transferred prior method | Reproducible distinct mechanism beyond Vero and calibration/filtering controls |

## Codex comparison and action

Codex agrees with the critique's central distinction: the current Task04 pair is an **engineering substrate**, not sufficient evidence of scientific cross-task transfer, and `94.6%` accuracy is not a useful primary result. Codex also agrees that reward and exposure audits must accompany any RL comparison.

Codex differs on sequencing. Patient lineage is necessary before patient-generalization or confirmatory training claims, but it need **not delay frozen base/Vero inference on verified real images as an explicitly labeled engineering diagnostic**. The immediate falsifiable comparison is both checkpoints on the same Task04 renders and Dallas sequence prompts, against majority/center/no-image baselines, with no patient-level inference claim. In parallel, seek independently labeled second-category data and verify lineage. This preserves the user's priority to execute feasible model work now.

No result, citation, patient-level statistic, or model score has been inferred from Claude's critique. Any subsequent actual inference or training result must be reported separately.
