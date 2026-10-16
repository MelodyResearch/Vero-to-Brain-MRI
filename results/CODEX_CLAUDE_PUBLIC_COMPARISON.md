# Independent Codex–Claude public Vero comparison

This compares two initially independent readings of the **same public-only excerpts** at Vero commit `025ca3c800fc95224c86323c1f7fd33dceed3090`, prompt SHA-256 `29f33b1b207be1a8ae329ff2121310d8d0bb74ae7ee01cc157766d67a527f11d`.

- Genuine authenticated Claude Code response: `CLAUDE_PUBLIC_VERO_REVIEW.md`, model `claude-sonnet-5-5`, supplied public excerpts only, tools disabled. Its original findings are preserved.
- Independent Codex initial response: `CODEX_PUBLIC_VERO_REVIEW.md`, frozen SHA-256 `e4cc86ec16dbd80889cbb1efd4b04484bc091162d383ac9049011cc9f8ac18d1`. It was written before reading Claude's response and remains unchanged.
- Codex first reproduced its two original metadata findings in `PUBLIC_VERO_COUNTEREXAMPLES.md` **before** reading Claude. Only then did the lead authorize this comparison.

Agreement is not proof. The follow-up fixtures below resolve particular behaviors; missing dependency/runtime context remains unknown. Neither review covers or approves private B/C data, local SFT restoration, local RL, MRI generalization or checkpoint receipts. Claude did not receive those materials.

## Genuine agreements

Both independently reconstructed fixed per-batch domain quotas, strict outer-tag format with partial boxed credit, GRPO sample-standard-deviation normalization, the detached GSPO sequence ratio with token gradients, and reuse of an actor-owned optimizer inside updates. Both found the inherited Qwen response-length mismatch and unknown-route fallback worth investigating. Both declined to treat the dummy judge helper's zero as a demonstrated failure because the judge manager is absent. Both withheld claims about full checkpoint/restart fidelity and numerical framework parity.

These agreements concern displayed source semantics. They do not establish the actual sampler/dataloader wiring, effective Hydra configuration, judge results or successful training.

## Findings unique to each initial review

**Codex:** conventional flat-list image-size metadata is discarded; nonfinite tolerance is accepted. Both now have executed exact-helper AST counterexamples, with external-parser stubbing clearly disclosed. Claude's initial report did not identify them.

**Claude:** unbounded text reaches `sympify` in the tolerance path; possible thread-sensitive parser timeouts; optional rollout importance weights appear ignored by GSPO; mixed-precision scaler observability; multiple-choice first-parsed-answer semantics. These extend the review. They are not automatically accepted as demonstrated full-pipeline defects.

## Disputes and clarifications resolved with bounded fixtures

Executed script: `scripts/reproduce_public_vero_review_disagreements.py`. Results: `PUBLIC_VERO_REVIEW_COMPARISON_FIXTURES.json`. This is lightweight CPU work with exact public AST helpers and explicitly identified stubs or scalar mathematical checks; no Torch, MLX, actual Sympy expression, model or judge ran.

| Question | Executed/static evidence | Resolution |
|---|---|---|
| Is the batch256 remainder bias real? | Exact `_compute_batch_counts` under NumPy2.3.2 gives `[43,43,43,43,42,42]` repeatedly; batch2 gives `[1,1,0,0,0,0]`. | Fixed quantization is verified. The exact tied-domain order is not a guarantee of NumPy API stability or a check of upstream pinned dependencies. Calling the unavoidable256/6 rounding a defect assumes exact long-run equality was required. Batch2 starvation is a concrete transfer hazard. |
| Does parser failure always remove all numeric candidates, as part of Claude F3 suggests? | Injected ordinary `parse` exception into exact public helpers. `_numeric_candidates("1.05")` still yields `[1.05]`; tolerance0.1 grading gold1/pred1.05 returns1. Without tolerance, the same parser exception gives−1. | The blanket claim is false for directly float-convertible values on the tolerance path. Actual math_verify signal/thread behavior is still `NOT_VERIFIED`; no threading test against a pinned real dependency ran. |
| Does long raw text reach `sympify` without the200-character guard? | An exact `_coerce_float` call passed a harmless1900-character nonnumeric string unchanged to a recording `sympify` stand-in. The stand-in did no parsing or evaluation. | Reachability and absence of a local length guard are verified. Actual arbitrary-code execution, denial of service, dependency sanitization and worker isolation were not tested. Claude's security severity remains conditional on those missing facts. Do not execute its adversarial stress examples on this Mac. |
| Does every ratio outside the clipping interval have zero gradient? | Scalar finite differences of the displayed surrogate: positive advantage/high ratio and negative advantage/low ratio yield0; negative advantage/high ratio yields+1.001, positive advantage/low ratio yields−0.999. | Only the **active clipped branch** has zero gradient. An out-of-range ratio alone is insufficient. This clarifies Claude F9 wording; it does not identify a GSPO implementation bug. No PyTorch autodiff parity claim follows. |
| Does supplied GSPO use `rollout_log_probs`? | AST inspection finds zero body references to that argument. The displayed actor can require/pass it when optional TIS is configured. | The optional correction is not implemented in the supplied GSPO function. Actual activation/default of TIS and any outer correction are unverified. Do not claim an observed default-run defect. |

### Remaining assertions that must stay conditional

- **Judge noise:** stochastic judge settings are visible. Independence of errors across rollouts, their magnitude and how normalization changes useful gradients are not established. Shared ambiguity/systematic judge bias can correlate errors. Injected failures and repeated fixed-answer grading are future tests, not existing results.
- **Loss metrics:** the actor emits microbatch-weighted loss values. Whether the eventual metric is underreported depends on the omitted aggregation code; summing those weights can be correct. Claude's cosmetic underreport claim is not verified by the excerpts.
- **Large-ratio negative-advantage loss:** the source clamps log ratio at10, so with finite fixed advantage the displayed surrogate magnitude is bounded by the corresponding finite exponential scale. “Unbounded up to e10” is imprecise; the relevant concern is a large capped update, not an infinite ratio in this expression.
- **Multiple-choice truth:** choosing the first parsed item is visible, but multi-select examples and actual parser output were not supplied. No corpus error is established.
- **Overlong penalty:** both see data cap18432 versus inherited reward value16384. Actual manager consumption/override/clamping is absent, so predicted penalties remain conditional.
- **Safety fixes versus fidelity:** making the judge deterministic, rejecting missing domains, changing batch rounding or changing parsers may be defensible design choices, but they alter upstream behavior. Record them as explicit changes rather than describing every recommendation as an invariant already implemented by Vero.

## Scoped conclusion

The two independent reviews agree on the core displayed recipe while finding complementary boundary issues. Executed fixtures confirm the Codex metadata counterexamples and refine several Claude assertions. **`PASS_WITH_LIMITATIONS`** applies to this source-level reconstruction and the specified small CPU fixtures. **`NOT_VERIFIED`** remains the status of full upstream runtime correctness, judge integration, actual restart fidelity, scientific transfer and every private local B/C interface.

The initial reports remain intact so the reader can distinguish independent first conclusions from later evidence-based reconciliation. No consensus vote substitutes for a regression test or real experimental evidence.
