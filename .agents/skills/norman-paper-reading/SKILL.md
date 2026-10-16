---
name: norman-paper-reading
description: Use when auditing research papers for the brain MRI verifiable RL project, especially methodological novelty, strength of evidence, and the next falsifiable experiment.
---

# Norman-style paper reading — independent project adaptation

This is **our independently implemented adaptation of a three-layer reading method**, not the original author's skill. Use it for primary-paper audits and cross-paper research decisions. Read [the full protocol](../../../research/methodology/NORMAN_PAPER_READING_PROTOCOL.md) before producing an audit.

For each paper, verify exact title, identifier, version/date, venue if any, and code/data links. Record what was actually accessible: metadata, abstract, full text, supplement, implementation. If an identifier points elsewhere, mark `SOURCE_MISMATCH`. Never label abstract-only reading as deep reading.

Keep four distinct analyses:

1. **Research-process reconstruction:** Infer the documented problem, prior limitations, design choices, objective/optimizer, data, baselines, and credible alternatives from Methods, equations, and code. Do not invent authors' private reasoning.
2. **Evidence audit:** Map each important claim to the exact table, figure, section, and PDF page, with reported numbers only when checked. Examine budget/model/data comparability, subject leakage, preprocessing, seeds, 2D/3D and longitudinal scope. Label conclusions `Direct Evidence`, `Conditional Evidence`, `Author Interpretation`, or `Not Established`.
3. **Next experiment:** Give the smallest test that separates competing explanations, including data, reference, model controls, unit of inference, support/falsification outcome, and GO/REFORMULATE/KILL condition.
4. **Independent novelty check:** Locate the exact new component, inherited components, necessary technical difference, remaining gap, MRI transfer mechanism, novelty risk, and minimal demonstration beyond the strongest prior method.

Use primary sources and distinguish paper claims from your inference. Write `NOT_VERIFIED` where full evidence cannot be checked. Preserve source URLs and version. Never turn a dataset or benchmark paper into a claimed optimizer contribution. Keep patient-level identifiers, restricted data, and images out of Git.
