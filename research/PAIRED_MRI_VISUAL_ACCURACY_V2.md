# Paired MRI visual-accuracy V2 — isolated accuracy intervention

**Research goal:** make the Qwen3-VL MRI model distinguish actual image content, not merely change its FLAIR/T1 answer prior. **State:** implementation and synthetic CPU tests only. Real MLX training, gradient parity, free-generation evaluation, and any accuracy gain: **NOT_RUN / NOT_VERIFIED**.

## Why this change, not another engineering loop?

The common B28 start correctly identified 15/24 sequence images on four internal-development people, strongly favoring FLAIR (12/12 FLAIR, 3/12 T1). U_raw seed7 scored 13/24 (7/12 FLAIR, 6/12 T1) without evidence of increased visual discrimination. The original GSPO arms cannot be retrospectively altered.

**Single intervention:** start from the exact B28 adapter; use actual matched FLAIR and T1 images of the **same training patient and axial level**, train the existing language LoRA with a **label-token-only supervised contrastive objective**. Keep the frozen vision encoder and the original image/prompt contract.

For image \(x\), let \(s_y(x)\) be mean log-likelihood over *only* the tokens of label \(y\), conditioned on the same fixed assistant prefix. For each matched patient pair, define \(m_F=s_{\mathrm{T1}}(x_F)-s_F(x_F)\), \(m_T=s_{\mathrm{T1}}(x_T)-s_F(x_T)\). Optimize:

\[
L=\operatorname{softplus}(-(m_T-m_F))+
\frac{1}{2}\left[\operatorname{softplus}(m_F)+\operatorname{softplus}(-m_T)\right].
\]

The first term **cannot be improved by a purely image-independent common label-prior shift**. The second term pushes each actual image toward its true class. These are distinct from the historical 14/14 class exposure and from Vero-GSPO. A patient-pair separation gain alone is **not** a test-set accuracy gain.

The native backend evaluates one image/candidate graph at a time, forms the analytically verified four-score gradient and takes one real Adam update per pair. It uses a fixed future-only token sentinel to equalize candidate forward **shapes**, addressing the previously observed prefix-shape sensitivity without changing which label positions are scored. Whether MLX logits/gradients are invariant under this padding on the actual model remains an essential **first native diagnostic**, not an established result.

## Files

- \`src/mri_rl/paired_sequence_accuracy_v2.py\` — strict train-only patient/level pairing; objective, gradient coefficients, native backend, candidate scorer.
- \`src/mri_rl/paired_sequence_accuracy_v2_entry.py\` — source-bound plan/scope, existing resource and reviewer gates, atomic LoRA+Adam+RNG checkpoint/resume, patient-level native development scoring.
- \`tests/test_paired_sequence_accuracy_v2*.py\` — synthetic pairing/label-leak rejection; analytic gradient finite differences; evaluation distinguishes candidate preference from **actual greedy MRI answer accuracy**.

The original 18 RL arms, 3 SFT arms, historical receipts, raw data, and \`main\` are **untouched**.

## Run on the existing Mac only after the current heavyweight arm releases the model

Use the existing Python/MLX environment; run one heavyweight process at a time. Supply the **real local source 382-row training + matching internal-development manifests** and an experiment contract that cryptographically binds exactly those manifests. The C76 mixed-task subset is **not** an adequate paired training source because it intentionally carries only one sequence label per patient. Do not invent missing pairs or change the existing final100 seal.

Define the local paths in the shell, including \`TRAIN\`, \`DEV\`, \`MODEL\`, \`B28_ADAPTER\`, \`CONTRACT\`, and a private \`OUTPUT\` under \`local_data/\`. First inspect the cohort without loading any model:

\`\`\`bash
PYTHONPATH=src python -m mri_rl.paired_sequence_accuracy_v2_entry plan \
  --train "$TRAIN" --dev "$DEV" --model "$MODEL" \
  --contract "$CONTRACT" --adapter "$B28_ADAPTER" --output "$OUTPUT" \
  --seed 7 --steps 28
\`\`\`

Before training, bind and independently review **one small exact V2 scope** using \`scope\` (and \`--scope-output\`). Use the existing certified quantized cache arguments when applicable. Train with \`--max-new-steps 1\` and an authenticated \`--review-verdict\` and \`--expected-review-sha256\`; subsequent steps need \`--resume\`, a fresh scope and the same frozen schedule. Do not bypass legitimate Mac training resource safeguards.

For **before/after native accuracy**, generate a separate \`eval-scope\`, then run \`eval\` with its reviewed scope and an explicit \`--evaluation-adapter\` (B28 initially; verified new V2 boundary afterward). The output is private JSONL with per-row candidate scores **and real greedy responses**; summaries report FLAIR/T1 recall, overall accuracy, and patient-macro accuracy. Do not compare a candidate-score result against a greedy baseline as if the endpoints were identical.

A first one-update native pilot must verify finite real Qwen gradients, numerical padding invariance, changed adapter bytes and persisted Adam/RNG before more compute. Run fixed-budget V2 versus B28 on the **same** patient-disjoint development rows. Keep the dev cohort exploratory; neither a four-patient win nor repeated tuning on it suffices for publication. Freeze the chosen procedure before a one-time independent test.

## Go / stop criterion

- **GO (exploratory):** Actual image-specific T1 recall improves **without a larger FLAIR loss**, and both image-swap response dependence and strict greedy answer accuracy improve relative to B28 on the fixed development rows. Otherwise examine candidate-vs-greedy disagreement rather than celebrating a margin change.
- **REFORMULATE:** fixed vision features cannot support paired discrimination, or improvements follow nonbrain/acquisition shortcuts. Consider one limited vision-adaptation experiment only after this falsifier.
- **NO SCIENTIFIC CLAIM:** no real MLX update or no independent accuracy measurement. No sealed final test may be opened to tune the V2 intervention.

**Paper context:** This is a *targeted accuracy repair baseline* for the main Vero-to-MRI recipe-transfer question, not a claimed new architecture, completed ablation, demonstrated generalization, or publishable outcome by itself.
