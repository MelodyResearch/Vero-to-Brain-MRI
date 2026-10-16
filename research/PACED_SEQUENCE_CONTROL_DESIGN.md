# B28 sequence pixel-dependence controls — bounded implementation

Tags: `B_LABEL_CONTROL`, `EXPLORATORY_DIAGNOSTIC`. Status: `IMPLEMENTED`, `TESTED_CPU`; B28 control inference: `NOT_RUN`. This adds one new module and its tests; the reviewed primary evaluator and training source are unchanged.

## Mission Alignment Check

1. **Main-question link:** Determine whether correction of sequence-label exposure improves responses to actual MRI pixels, a prerequisite for interpreting later Vero task-mixture gains.
2. **Claim and falsifier:** Compare real-image answers with question-identical no-image and same-patient/slice opposite-sequence images. A prior-only explanation remains plausible if outputs ignore the changed visible sequence; arbitrary answer changes do not establish grounding. Correct-direction change requires correct original and replacement-image answers.
3. **Change isolation:** Use the existing frozen24-question control manifests and historical A28 controls. The checkpoint must be actual completed B28, the same initialization lineage as the primary matched comparison, with native prompt/template,4-bit loading, LoRA rank4/alpha8, temperature0 and256-token cap unchanged. B/C datasets remain separate.
4. **Provenance:** Require completed and identity-validated matched54 B28 outputs before any control model load. Bind those output bytes, fullB28 evidence, each entire24-record control manifest, source labels' private SHA256, immutable A28 control output hashes, current pixels, exact transformed prompts and source hashes. Gold answers are never passed to model prompts.
5. **Baselines:** Preserve A28 original15/24, no-image12/24 and six correct-direction swaps out of24. Score the swap against the replacement image's acquisition label. Four patients and paired repeated images remain descriptive; control questions are not new independent patients. Since both modalities already appear in the original24, swaps permute paired inputs and are not a separate generalization cohort.
6. **Resources and decision:** One model, at most four new records per independently reviewed stage; full manifests are never sliced or rewritten. Use lazy loading followed by4-bit adapter restoration before materialization, exact504-tensor equality, existing low cache/memory guideline and PacedGate before/after generation. Preserve each response before any post-generation HOLD/STOP. No automatic full24 expansion or model execution was performed during implementation.

## Frozen source identities

| Condition | Manifest SHA256 | Historical A28 outputs SHA256 |
|---|---|---|
| No image | `420c0dc5a152d3dc3e9d5e1505525f038537bf8b06a245fdc03452fa369061d4` | `607888eddd082c5d04e9320134362099bb5dc6a411c58bb5534caefce11835d8` |
| Same-slice sequence swap | `c2f709242f25d5a3d3621bfd1f0f40f59d6ffcf6320613bfcf363d4d375aad11` | `6249117b8543277b4858a5b76497c81314be5e896446bb2a20af919a3e32f179` |

Historical adapter: `244c26fd5ce2e3b02b46703d4f46a90608f2ef1a868eeee682ee4fe48a58ed73`. Shared adapter configuration: `032becabb7cbb6d2af6f3451eb127e5f944551d42eb51eaee3e8d75b839b2f6a`.

Private labels remain in the frozen matched internal-development gold JSONL, SHA256 `0630ec7c91ed67f1059294422f8eb937435546bbf446155d187365862fda8961`. The runner checks this hash only; analysis separately validates source patient/slice/modality labels and applies visible-image targets.

One communicated swap-output hash had a transcription error (`a5d` instead of actual `a5b`). Reading the actual immutable file caught it; the implementation was corrected to the actual source digest above. No baseline data was rewritten.

## Runner and approval

New module: `src/mri_rl/paced_sequence_controls.py`.

Inputs: `--model`, `--primary-manifest`, `--primary-outputs`, `--output`, `--b28-run`, `--contract`, `--mode {no_image,swap}`, optional `--max-new-records` (1–4, default4), `--resume`, and `--plan-only`.

Plan preparation invokes the unchanged B28 primary verifier. Missing/incomplete54 predictions, changed model/checkpoint/profile, invalid baseline pixels, or altered control intervention fail before MLX import. Derived control rows must equal the frozen manifests exactly:

- No-image: remove exactly the source image part, preserve system and question; runtime expects zero image placeholders and sends no image payload.
- Swap: select the opposite modality with the same patient/slice ID, replace exactly the image part, preserve system and question; runtime expects one image placeholder.

Execution requires an independent approval JSON with `kind: paced_sequence_controls_v1`, actual reviewer identity, `PASS` or `PASS_WITH_LIMITATIONS`, and the exact scope dictionary emitted by `--plan-only`, plus its supplied SHA256. Scope binds mode, primary output hash, baseline/control hashes, exact pending IDs/pixels, budget, source and resource settings. A no-image approval cannot authorize swapped images or a larger stage.

The output schema retains the primary/historical scorer fields and adds mode, completed-primary provenance, actual token counts, actual optional finish reason, process peak memory and native prompt hash. Explicit resume uses the existing checked-output helper; completed IDs are skipped only when model/data/settings match. Data and raw generations remain private.

## Analysis compatibility and checks

The analysis-only `scripts/analyze_b28_paired.py` validates both complete24-condition outputs and scores new visible labels. It reports original/no-image/swap confusion, invalid and format counts, per-patient accuracy and directional changes. Unknown termination reasons must remain unknown; a backend token count is not permission to infer stop/length.

Nine focused CPU tests pass: no-image removal without question mutation; exact opposite-sequence pairing and missing-pair rejection; baseline pixel/prompt/digest/decoding drift; incomplete primary rejection; full-primary adapter/metrics checks; mode-specific review binding; unknown mode rejection; and model-free module import. Direct checks on the actual24+24 frozen manifests reproduce their transformations and match all corresponding historical output pixel/profile receipts. No B28 controls have run, and no control execution approval is implied by these tests.

## Independent source/CPU review

The isolated Codex engineering reviewer inspected new control-module SHA256 `30d8ca43298842a553ded6ba20a17e2caa18fe0671ec056c6fcb53836976553c`, ran nine tests under an MLX import blocker and directly checked both actual24-record transformations, immutable A28 control receipts/pixels/profiles and the private gold hash. Verdict: `NO_ISSUE_FOUND` within the source/CPU scope. Runtime correctness and execution approval remain `NOT_VERIFIED` until actual B28 and completed54 primary outputs exist. This review is not a claim of Claude involvement.
