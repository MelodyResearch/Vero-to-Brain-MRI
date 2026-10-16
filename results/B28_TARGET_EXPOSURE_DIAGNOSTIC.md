# B28 target-token exposure and saved-loss diagnostic

Status: `TESTED_CPU`; evidence: `OBSERVED` for saved receipts and tokenizer replay, source-derived for the loss-mask algebra. Tag: `EXPLORATORY_DIAGNOSTIC`. This audit was performed before reading B's primary correctness scores. It changes no targets, model, training code or evaluation selection. It uses only saved training evidence, source and the local tokenizer/template artifacts; no model or MRI arrays were loaded.

## What is directly verified

All112 B28 examples match the frozen schedule in step, category, patient key, record ID, image hash and target label. Their saved target SHA256 and token counts match a fresh tokenizer-only encoding of the source-defined target. The sequence route contains **14 FLAIR and14 contrast-enhanced T1** targets. There is no evidence here of the corrected sequence label silently reverting to FLAIR.

Historical A28 saved112 scalar example losses, category/patient/step and token counts, but **did not originally log target labels or target hashes**. Its label identities are reconstructed from the existing independently audited historical selection record, matched by `(step, category, patient)` rather than file-list position. That reconstruction's original-receipt hash matches the actual A28 receipt. All84 nonsequence B identities, images and labels match it. The historical-label conclusion therefore retains the explicit qualification `reconstructed_not_originally_logged`.

The target string is fixed observation/format text surrounding the class label. Fresh local `tokenizers`0.23.3 encoding reproduces every saved A/B target length. The fixed part has **14 tokens**: four tokens for the generic observation, and ten for tags, box syntax and closing text. No answer-boundary-spanning tokens occur for either sequence label.

| Sequence target | Presentations A28 / B28 | Target tokens each | Fixed tokens each | Answer tokens each |
|---|---:|---:|---:|---:|
| FLAIR | 28 /14 | 16 | 14 | 2 (`FL`, `AIR`) |
| Contrast-enhanced T1 | 0 /14 | 19 | 14 | 5 (`contrast`, `-enh`, `anced`, space+`T`, `1`) |
| Sequence total | 28 /28 | A448 /B490 | A392 /B392 | A56 /B98 |
| All four routes | 112 /112 | A1,736 /B1,778 | See private per-target receipt | See private per-target receipt |

The updates and number of presentations are matched; **target-token work is not identical**. B has42 more supervised tokens because14 sequence answers are longer. It is therefore precise to call this an equal-update/equal-example intervention, rather than an equal-token-budget experiment.

## Assistant-only loss and native prefix

The source uses the native Qwen template with `add_generation_prompt=True`; the inspected template ends with `<|im_start|>assistant\n`, without prefilled `<think>`. A CPU Jinja rendering of the saved template confirmed that suffix and exactly one image placeholder. This checks the template artifact, not the runtime visual processor tensor contents.

In `mlx_sft_warmstart.py::loss_for`, prompt and target tokens are concatenated; inputs omit the final token and labels omit the first. Log probabilities are then selected starting at `prompt_length−1`. Thus the first included logit predicts the first assistant target token, and the final included logit predicts the final target token. Prompt/user/system/image tokens receive **no direct cross-entropy targets**. They still condition the forward pass and can affect gradients; masking the prompt does not remove image conditioning. A small integer-index replay checked this alignment for16-,19- and15-token targets without importing MLX.

The loss is the mean negative log probability over **all assistant target tokens**, followed by equal1/4 averaging of each example's gradient in the four-example update. There is no class-answer-only weighting. For a FLAIR example, answer-token terms occupy2/16 of the scalar mean; for T1 they occupy5/19. These are static coefficients, **not measured fractions of loss, gradient norm or learning**. The target is encoded without added special tokens and ends at the answer closing tag; no EOS/im_end token is appended or directly supervised. Whether that affects generation termination is unmeasured here.

## Saved sequence loss distribution

These values are recomputed from the actual saved **whole-target per-example mean cross-entropies**, linked by category and step. They are online losses at different training states, not a common-checkpoint evaluation. They do not isolate answer tokens from shared formatting tokens.

| Whole-target sequence loss | A28 | B28 |
|---|---:|---:|
| Examples | 28 | 28 |
| Mean | 2.816188 | 2.755610 |
| Median | 1.822435 | 1.864160 |
| Minimum / maximum | 0.100699 /6.874905 | 0.139999 /6.931138 |
| First four sequence examples, mean | 6.692559 | 6.383625 |
| Last four sequence examples, mean | 0.179735 | 0.248374 |

Within B, FLAIR's14 whole-target losses average2.749975; T1's14 average2.761245. The last four **within each label** average0.330785 and0.464222 respectively. Those subsets occur at different steps and cannot establish a causal class difficulty difference. No per-token losses, answer-only CE, answer-token logits or token-specific gradient decomposition were saved. Consequently, the tempting explanation “format tokens dominated optimization and suppressed T1 learning” remains a **hypothesis**, not a finding. Likewise, low aggregate teacher-forced loss does not prove free-generation image recognition.

## Minimal future discriminating test

First finish the frozen primary and image-control analyses. If weak image dependence persists, a bounded, independently reviewed **read-only teacher-forcing probe** could evaluate both sequence answers under fixed training-only real, absent and same-patient swapped images at A0/A28/B28. Save token-level CE with explicit fixed/answer partitions and first-answer-token likelihood margins, while keeping the current native prefix and all weights unchanged. This separates a format-loss explanation, class-prior preference and image-conditioned likelihood sensitivity without adding training or retuning prompts. It is **PROPOSED / NOT_RUN**; it requires the normal model/resource approval. A change to answer weighting or EOS supervision would be a later separately declared intervention, not a correction justified by the present counts alone.

## Mission Alignment Check

1. **Main question:** verify the supervision interface before interpreting image-dependent task learning and subsequent Vero RL transfer.
2. **Claim/falsifier:** actual B sequence targets are14/14 and assistant-only; any mismatched frozen label, target hash/count or loss boundary would refute the contract.
3. **Isolation:** replay only immutable training receipts and tokenizer/source artifacts; no score-driven selection, code changes or target edits.
4. **Provenance:** private source/hash-bound audit and separate template/index receipt below; historical label identity is explicitly reconstructed.
5. **Baseline:** compare A/B fixed-versus-answer token counts and saved scalar losses, without inferring unrecorded token losses or image understanding.
6. **Resources/decision:** a single bounded tokenizer-only CPU pass, with thread caps and MLX/MLX-VLM/Torch/Transformers/NumPy/nibabel imports blocked. Preserve the current training; only propose a read-only forward test if complete evaluation warrants it.

## Reproducibility and limits

Private directory: `local_data/ds007045_bounded/b28_target_exposure_audit_v1/`. The audit script retains deterministic assertions and hashes of both training receipts, historical reconstruction, frozen B schedule, tokenizer, template and training source.

- `receipt_PRIVATE.json` SHA256: `20b4e645fb0e283d3166f0995ce100dd57835054912239ea0cfc73211830d27c`.
- `prefix_mask_PRIVATE.json` SHA256: `60bab55e5096019b47f91978bcad47962b56b59ffc646a61c678a021ae4b4020`.
- No per-patient identities or model generations are included here.
- Independent replay completed by the isolated Codex methods/statistics reviewer; scope and limits below. This document is not independent approval of the full training interface. Existing B28 chain verification remains separate.


## Independent verification

**`REPRODUCED`; verdict `PASS_WITH_LIMITATIONS` for token exposure, recorded scalar losses and source-index alignment.** An independent implementation reconstructed historical labels directly from the seed-7 lexical training selection and matched `(step, category, patient)` to the saved A28 receipt. This confirms the reconstruction while preserving its limitation: A28 labels and target hashes were not originally recorded. The review independently checked all224 A/B target lengths, all112 B target hashes and frozen schedule bindings, all84 unchanged nonsequence identities/labels/image hashes, and every reported category/label scalar-loss summary.

The reviewer evaluated only the literal target f-string extracted from the source AST and encoded it with the local tokenizer. It reproduced1,736 versus1,778 supervised tokens, 14 fixed tokens per sequence target, two versus five answer tokens, four generic-observation tokens and no answer-boundary-spanning tokens. It separately rendered a synthetic native-template fixture and reproduced the assistant suffix and one image placeholder; integer index replay reproduces exactly the assistant-only target positions. No model, visual processor forward pass, NumPy, MRI array, Torch or MLX was imported.

The interpretation is appropriately bounded: equal updates/examples do not imply equal supervised-token work; static token coefficients are not observed loss/gradient fractions; online whole-target losses at different training states are not common-checkpoint accuracy. Missing direct EOS supervision is a source fact, not an established stopping defect. The separately verified B28 primary outputs all report normal stopping, so this audit supplies no evidence of a B28 truncation problem. A read-only token-level teacher-forcing probe remains proposed and requires a separate scope; this replay authorizes no model launch and changes no targets.

Private independent script and receipt: `overnight_method_stats_v1/independent_b28_target_exposure.py` and `independent_b28_target_exposure_PRIVATE.json`. The producer receipt remains unchanged at SHA-256 `20b4e645fb0e283d3166f0995ce100dd57835054912239ea0cfc73211830d27c`. No partial B28 image controls were scored during this verification.
