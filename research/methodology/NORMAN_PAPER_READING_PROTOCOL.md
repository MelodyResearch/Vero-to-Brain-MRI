# Project paper-reading protocol

This is **our independently implemented adaptation of a three-layer research reading method**, not the original Norman author's skill. It is tool independent and can be followed by Codex, Claude Code, or any independent reviewer. The project skill at `.agents/skills/norman-paper-reading/SKILL.md` is its short entry point.

## Source and scope record

Start each audit with exact title, authors, identifier, version and date, publication status, primary text URL, supplement and code URLs, and access level for each source. Resolve the identifier before interpreting the paper. Mark a mismatch `SOURCE_MISMATCH`; inaccessible evidence `NOT_VERIFIED`. Distinguish preprint versions, changed repositories, and reported from independently reproduced results. A PDF page is the PDF's page number; also give printed page if different.

## Layer 1 — Reconstruct the documented research process

Answer: What problem was targeted? What prior methods and assumptions failed? What exact gap remained? Why did the documented design choose its architecture, representation, loss/reward, optimizer (SFT, DPO, GRPO, GSPO, etc.), data and baselines? Which components are inherited? Which alternatives were feasible, and how would they alter the mechanism or result? Ground this reconstruction in Methods, equations, configuration, code and the authors' stated motivations. State your own inferred rationale as an inference, never as access to private decisions.

## Layer 2 — Audit the evidence claim by claim

For each central claim, cite the exact figure/table/ablation/section and page, and transcribe critical numbers with units and comparison direction. Evaluate same starting model, size, training data, supervision, visual tokens, rollouts, optimizer updates, compute, hyperparameter selection and prompt/decoding. Check metrics, seeds, confidence intervals, independent subjects and studies, split leakage, preprocessing, 2D versus native 3D evidence, repeated-scan validity, cross-source transfer, and missing negative results. Assign one label:

- **Direct Evidence:** the reported experiment directly measures the stated outcome with a sufficiently described comparison.
- **Conditional Evidence:** the observation is real but depends on a control, assumption, cohort, or implementation choice that limits the claim.
- **Author Interpretation:** a mechanism or generalization inferred by authors, not isolated by their evidence.
- **Not Established:** source missing, comparison absent, or claim exceeds what was measured.

Do not fill missing tables or confidence intervals from memory. An abstract, search snippet, or secondary summary does not establish a full-text claim. Record unsuccessful source retrieval explicitly.

## Layer 3 — Derive the next experiment

State the most consequential unresolved question, falsifiable hypothesis, smallest decisive experiment, necessary real data and reference, model and simple controls, critical ablations, primary metric and independent unit, predicted supportive and falsifying results, plus GO/REFORMULATE/KILL decision. Prefer a cheap no-image, leakage, label permutation, or matched-budget control before expensive RL when it can answer the question. Define a comparison that separates the proposed mechanism from generic normalization, additional supervision, or format learning.

## Fourth check — independent methodological novelty

Locate novelty precisely: architecture, representation, objective/reward, optimization, data construction, sampling, verification, evaluation or scientific discovery. Name the closest inherited components and competing primary paper/code. Identify the necessary technical difference, remaining unsolved problem, mechanism for transfer to real 3D/multi-sequence/longitudinal MRI, and why the proposed transfer may just be an application. Specify the smallest result that the strongest existing method could not already produce. Absence of an identical project name is not novelty evidence.

## MRI-specific cautions and output contract

Patient identity is the inference and split unit unless a stronger independent unit is justified. A stack of 2D slices is not native 3D reasoning; frames within one scan are not longitudinal visits. A scalar derived from FreeSurfer is a processed image measurement, not an independent diagnosis or a lesion mask. Grounding needs correct voxel/frame geometry and annotations. Clinical change needs repeatability, acquisition comparability, and actual repeat visits.

Each paper audit should use the headings `Verified metadata and access`, `Layer 1 — Research process`, `Layer 2 — Evidence audit`, `Layer 3 — Next experiment`, `Independent novelty and MRI transfer`, and `Missing evidence and uncertainties`. A cross-paper synthesis must reconcile disagreements and answer what is established, what remains open, and what experiment would change the project's decision. A section-completeness checker can detect omissions; it cannot validate scientific truth.
