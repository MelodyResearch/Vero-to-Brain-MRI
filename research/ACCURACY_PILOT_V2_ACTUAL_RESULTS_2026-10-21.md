# Accuracy Pilot V2 — actual native results (2026-10-21 PDT)

## Result

**NO_GO: MRI-adapted Paired-Prompt GSPO did not improve this matched pilot.** Both trained endpoints were worse than the same-prompt Parent. No further training was started.

| Same F32 / concise prompt / DEV24 | Correct | Strict accuracy / balanced accuracy | FLAIR recall | Enhanced T1 recall | Invalid | Actual updates |
|---|---:|---:|---:|---:|---:|---:|
| Parent Adam52 | 18/24 | 75.00% | 10/12 (83.33%) | 8/12 (66.67%) | 0/24 | 0 |
| A: Direct Vero-style same-prompt GSPO | 16/24 | 66.67% | 9/12 (75.00%) | 7/12 (58.33%) | 0/24 | 2 |
| B: MRI-adapted Paired-Prompt GSPO | 16/24 | 66.67% | 9/12 (75.00%) | 7/12 (58.33%) | 0/24 | 2 |

This is 24 images, balanced 12/12, from **four previously explored internal DEV patients**. It is an engineering pilot, not an independent generalization or clinical-reasoning result. A and B have different saved policies; their 24 class predictions are identical. Their full response texts are identical on 23/24 cases.

## Separate prompt effect from RL effect

The original Adam52 was evaluated first on the same fixed TRAIN8 under the original and the single predeclared concise prompt. Original: 1/8 correct, FLAIR 1/4, T1 0/4, invalid 7/8. Concise: 3/8, FLAIR 2/4, T1 1/4, invalid 1/8. The frozen rule selected concise without consulting DEV24.

The literal task-question addition was:

> Keep the reasoning brief (at most two short sentences) and finish with the required boxed answer.

System format, strict verifier, 256-token cap and temperature 0 were preserved. The question gained exactly 20 prompt tokens.

For the **unchanged Parent on the same DEV24**, saved original-prompt results were 10/24, FLAIR 5/12, T1 5/12, invalid 9/24. Concise yielded 18/24, FLAIR 10/12, T1 8/12, invalid 0/24. Independent verification matched all image/input bindings and full parameter/Adam integrity. The +8 correct comprises six invalid-to-correct and two valid-wrong-to-correct transitions, with no previously correct answer lost. Three other invalid answers became valid but wrong.

Thus the **+33.33 percentage points is prompt/readout improvement, not Vero weight-learning improvement**. Under that identical concise condition, both trained models lost two correct answers and gained none: one FLAIR and one T1. B−A is zero.

## Paired image-conditioned evidence

The main evaluation contains 12 same-patient/same-level FLAIR/T1 pairs. All 36 model-pair checks have matching non-pixel prompt/input/attention/grid fingerprints and different pixel payloads. This permits labeled reuse as `PAIRED_IMAGE_SUBSTITUTION_FROM_MAIN_EVALUATION`; it is not a new independent image intervention.

| Pair category, denominator 12 | Parent | A | B |
|---|---:|---:|---:|
| Both modalities classified correctly | 7 | 5 | 5 |
| Constant FLAIR | 3 | 4 | 4 |
| Constant T1 | 1 | 2 | 2 |
| Reversed | 1 | 1 | 1 |
| Invalid involved | 0 | 0 | 0 |

A/B each lose two correct-discriminating pairs: one becomes constant FLAIR and one becomes constant T1. There are no gains. All three models predict 14 FLAIR and 10 T1 overall, so the losses cannot be explained by a changed aggregate class count alone. There is **no Paired-Prompt-specific image-discrimination improvement**, and no anatomical-grounding claim.

## Actual training and mechanism

Parent: preserved **P_calibrated13 / Adam52 / boundary114**. Each arm freshly restored the same full disk parent, serially. Four TRAIN patients, eight original z045 images, and eight unchanged Dallas engineering-control records were exposed. Reward, GSPO loss, LR, LoRA, F32, prompt, data exposure and budgets were equal. Only sequence grouping differs between A and B. No CE/margin auxiliary objective was used.

The first 32 on-policy responses were generated once and shared. On those identical responses A admitted 5 groups / 20 responses; B admitted 7 / 28. After the first update, each arm generated its own next 32 responses from its current policy. All checkpoints include adapter, Adam and RNG and have chained receipts.

| Actual training cost | A | B |
|---|---:|---:|
| Adam steps | 52→53→54 | 52→53→54 |
| Updates | 2 | 2 |
| Effective groups | 9 | 14 |
| Backward calls | 36 | 56 |
| Scoring TF positions, including own-old | 27,710 | 31,338 |
| Own second-segment returned tokens | 2,418 | 1,549 |
| Arm process wall seconds, excludes shared C | 1,021.976 | 935.778 |

Common C cost: 32 generated responses / 1,465 returned tokens / 32 old forwards / 11,641 TF positions / 549.373 seconds. Counting it once is essential. More effective groups in B did **not** improve DEV accuracy; accepted groups and gradient norms are not performance evidence.

## Compute and Mac resources

Actual complete P→C→A→B→EP→EA→EB controller wall: **3,956.212 seconds (65.94 minutes)**. Actual totals: **184 responses, 9,899 returned tokens, 96 old + 92 gradient = 188 scorer forwards, 70,689 TF positions, 92 backwards, four updates, seven loads, zero recoveries**. All are within the approved maxima.

Each stage has actual exit 0, reaped child, closed-log hashes, authenticated terminal and native release verification; the controller also exited 0 and the shared lock had no holders. No retry, cache recovery or extra patient/model call occurred.

- wait4 peak native-child RSS: **5.92143 GiB**.
- Native-reported peak MLX: **10.35662 GiB**, below the 12 GiB cap.
- Sampled minimum available memory: **5.25034 GiB**; minimum pressure-free: **29%**.
- Observed OS levels: GREEN/NORMAL and YELLOW/WARNING; no CRITICAL or guardian stop.
- Preflight swap: **14.86133 GiB**; sampled peak **19.62726 GiB**, final **18.70483 GiB**. Peak growth was **4.76593 GiB**. Ordinary swap growth was permitted by the approved dynamic policy; it is reported, not hidden.
- After controller exit: available **16.65929 GiB**, pressure-free **71%**, AC / battery 100%, no other heavy processes reported.
- Thermal sensor: **SENSOR_UNAVAILABLE**. External monitoring cannot directly attest child MLX usage; MLX peaks/release come from the separately authenticated native records. No claim of zero OS risk is made.

DEV generation tokens / seconds: Parent **713 / 187.636**, A **739 / 191.896**, B **738 / 188.880**. Original-prompt Parent was **2,932 / 692.195**. The large token/time reduction belongs to concise prompting. B is not more accurate or faster than the concise Parent; shorter own training generations are not accuracy-efficiency gains.

## Decision and limits

The frozen scale rule yields **NO_GO**. This configuration will not be automatically expanded. The evidence supports a useful bounded generation correction, but not a Paired-Prompt RL accuracy benefit. It does not prove Paired-Prompt universally ineffective. Only four repeatedly explored DEV patients and one fixed run were evaluated.

Candidate margins, no-image controls and post-training Dallas retention were outside this scope and remain unmeasured. Dallas is ENGINEERING_ONLY. Old final100 and sealed new16 predictions were not accessed. No external baseline, new preprocessing or loss search occurred. The frozen research plan and lock remain byte-identical.

## Execution, evidence and review

Claude Code **2.1.306 / claude-opus-5-5** actually implemented the V2 path (commits `8c5af12c46c63c7f9590fa2f79d49e73c490850a`, `1b018e644f3e44674b0e3bcf88ef1fdabf41da77`) and launched the exact approved native command once through normal host permissions. The CLI monitor has no recorded final exit and later was no longer running; **no Claude final numerical approval is claimed**. The detached durable native controller completed all seven stages independently. Codex continued monitoring and reproduced saved numerical results; isolated **Codex /root/v2_actual_verifier** independently checked them.

- Approved scope FILE: `afb0d7ef4af2bc581ffce761faba8975a00f32d6ff2ba9c2f4975c838c115482`.
- Native source HEAD: `a10b4a2bc42cf6dc27871070313095ac69173878`.
- Actual private analysis SHA: `e67e5ed074ffecb3a78ece0e09195e7f69ed9129dfc1d7b44dc52a3fa272dc0b`.
- External controller exit SHA: `3e064206862be17d35f0e2d97af4d5372b93f23cb4346e4d6e776e67128967b7`.
- [Sanitized actual results](../results/ACCURACY_PILOT_V2_ACTUAL_RESULTS_V1.json).
- [Independent actual-results review](../results/ACCURACY_PILOT_V2_ACTUAL_INDEPENDENT_REVIEW_V1.json).
- Source: `src/mri_rl/accuracy_pilot_v2_{entry,native,core}.py`; saved-output analysis is CPU-only. All 104 frozen scientific source hashes were checked.

Native weights, Adam/RNG, response tokens, patient-linked outputs and detailed receipts remain private local assets and are not committed.
