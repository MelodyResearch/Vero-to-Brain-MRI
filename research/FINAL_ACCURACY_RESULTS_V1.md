# Final accuracy sprint: actual results and model selection

2026-10-28. **Native training completed; no accuracy improvement.** One supervised candidate was tested. The maximum of two candidates was a cap, not a requirement to run an unsupported second method. No additional seeds are admitted. This is an internal development result; the independent 16-person evaluation is NOT_RUN: original execution metadata is not currently located, and researcher confirmation is still required.

## Actual matched development results

All rows use the same Concise Prompt, F32 path, processor, strict parser, greedy temperature 0 and 256-token cap, on all DEV24 images: four TRAIN-disjoint but repeatedly explored people, three levels (45/91/136), two modalities. The Parent and historical GSPO rows reuse authenticated existing outputs. Only the SFT row is newly generated here.

| Endpoint | Strict / balanced accuracy | FLAIR recall | Enhanced T1 recall | Invalid | Both modalities correct | Updates after Adam52 |
|---|---:|---:|---:|---:|---:|---:|
| Parent Adam52 | 18/24 (75.00%) | 10/12 | 8/12 | 0/24 | 7/12 pairs | 0 |
| Direct same-prompt GSPO A54 | 16/24 (66.67%) | 9/12 | 7/12 | 0/24 | 5/12 pairs | 2 historical |
| Paired-Prompt GSPO B54 | 16/24 (66.67%) | 9/12 | 7/12 | 0/24 | 5/12 pairs | 2 historical |
| Patient-paired balanced SFT seed0 / Adam80 | 15/24 (62.50%) | 12/12 | 3/12 | 0/24 | 3/12 pairs | 28 new |

The new model gained two FLAIR answers and lost five T1 answers relative to Parent. Thirteen answers remained correct and four remained wrong. Predicted labels moved from 14 FLAIR / 10 T1 to 21 FLAIR / 3 T1. Thus full class-balanced TRAIN exposure did **not** prevent a stronger FLAIR preference at greedy inference.

Confusion matrices (rows true FLAIR/T1; columns predicted FLAIR/T1/INVALID):

- Parent: `[[10,2,0],[4,8,0]]`.
- New SFT: `[[12,0,0],[9,3,0]]`.

Sorted per-patient correct counts out of six images: Parent `[3,4,5,6]`, SFT `[3,4,4,4]`. Paired patient accuracy changes sorted are `[-0.5,-1/6,0,+1/6]`. Exact 4^4 cluster-resampling percentile ranges are Parent 58.33–91.67%, SFT 54.17–66.67%, and paired difference −37.50 to +8.33 percentage points. These are descriptive ranges for four explored clusters, not proof of population generalization or a significance test.

## Candidate, real computation and stopping rule

The candidate averaged two own-image mean full-response CE gradients per patient (FLAIR and enhanced T1), then made one persistent Adam update. All 28 TRAIN people were exposed to both z045 modalities. The fixed target contains the same generic reasoning prefix, tags and EOS for both classes. The model retained the frozen vision encoder and language LoRA, Adam52 moments and 1e-5 LR. This is **supervised SFT, not GSPO** and not the archived margin auxiliary variants.

Actual native computation: 56 distinct TRAIN images, 56 scorer/backward calls, 14,028 teacher-forced positions, 28 updates (Adam52→80), 24 DEV generations and 417 returned tokens. Training phase 372.913 s; evaluation phase 119.222 s; whole child 557.305 s. No new TRAIN sampling, cache recovery or historical experiment rerun occurred. One earlier entry failure consumed zero native calls; its technical repair retained the original compound deadline and evidence.

The predeclared replication gate required >=19/24 correct, FLAIR>=10/12, T1>=8/12, invalid=0 and >=7 fully correct pairs. The candidate fails overall, T1 and pair thresholds. **Seeds 1 and 2 are NOT_RUN_BY_FROZEN_STOP_RULE.** No second candidate, target search, optimizer reset or extension is started.

## Mission alignment and evidence tiers

1. **Main-question link / B_LABEL_CONTROL:** the researcher explicitly authorized a final accuracy-first supervised control; its gains cannot be attributed to Vero GSPO.
2. **Claim/falsifier:** complete paired dense supervision should improve strict class discrimination. The frozen class-protected gate failed.
3. **Change isolation:** parent, prompt, LoRA and numerical readout stay fixed; training exposure/target/task allocation differ from historical GSPO, so causal method superiority is not tested.
4. **Provenance:** actual native outputs and checkpoint receipts are OBSERVED; independent parsing and payload hashing are REPRODUCED.
5. **Shortcut/baseline:** stronger FLAIR prediction bias and fewer correct modality pairs are OBSERVED. A unique optimizer or visual-path explanation is INFERRED/NOT_VERIFIED.
6. **Resources/decision:** one bounded native run completed; the failed predeclared rule stops replication. Sealed evaluation is PROPOSED and requires confirmation.

## What the evidence supports

**Confirmed:** original sparse modality exposure was a data-design defect; historical K4 correctness signal was uneven; prompt format affected strict accuracy; Direct/Paired GSPO both regressed on the matched DEV24; the new dense balanced full-response SFT also regressed, with increased FLAIR predictions. Therefore paired exposure alone is not sufficient under this specific supervision and inherited state.

**Plausible, not causally isolated:** inherited Adam moments, full-response template weighting, unequal label token lengths (17/20 response tokens including EOS), language-only adaptation and source/level shift can contribute. No new margin, no-image, independent randomized swap, gradient attribution or momentum ablation was run here. Do not claim a disconnected vision path, anatomical grounding, or identify one of these hypotheses as the unique cause.

Main paired-image evaluation gives 3/12 correctly distinguished pairs versus Parent 7/12; the remaining nine SFT pairs both predict FLAIR. This is behavioral image-conditioned discrimination evidence under the paired input contract, not proof of brain anatomy reasoning. Dallas retention is **NOT_MEASURED** for this sequence-only SFT; the model cannot be claimed to preserve cross-task ability.

## Best model selection and fair comparison

**Retain Parent Adam52 as the best endpoint among these same-contract DEV24 comparisons.** Its advantage is +8.33 pp over each historical GSPO arm and +12.50 pp over this new SFT. It is not a new training improvement and is not a claim that Adam52 is globally best among every historical model.

The original formal sequence test has 84 images from 14 people. B28 was 52/84 (FLAIR42/42, T110/42). All three historical SFT seeds were54/84,57/84,52/84. All are preserved, without choosing the best test seed for a new claim. Those formal84 values cannot be ranked against DEV24 percentages. A same-contract DEV24 evaluation of all relevant historical SFT controls is **NOT_RUN**. No matched-budget causal SFT-versus-GSPO conclusion is supported: this candidate changes exposure, supervision, task allocation and update count. No new16 or formal100 predictions were opened for model selection.

## Checkpoint, safety and reproducibility

Terminal adapter SHA256: `7a6c414bae6790f88fd002bbcdc68a887aac5ecda29108bddb7971bbe4f683e8`.
Parent adapter SHA256: `c70b426b58191446545ba6333f7595e96f4fdb4cc689ddbaf5206e276b329db6`.
Exact execution scope: `6b6d51b44ccaf242c4540fa93afad0964f3e33c9016e4a2fb8f444538e077366`.

Independent review checked all28 full checkpoint sets, hash-chained receipts, cursors, state continuity and the unchanged10 bound Parent files. Actual native terminal reload was exact. The release receipt records model=None and1,050 active MLX bytes. The child exited0, was reaped, and the global lock was free. These are actual saved runtime attestations plus independent file checks; the reviewer did not rerun the model.

Peak wait4 RSS5.8243GiB, native reported MLX peak7.9008GiB. Swap was17.0425GiB initially,21.2609GiB finally, peak21.2999GiB (+4.2574GiB peak growth). Guardian minimum available memory5.4974GiB and pressure-free27%; no CRITICAL/STOP/TERM/KILL. All actual samples wereAC, although a fresh SFT-only battery>=20% exception had been granted. Temperature telemetry was unavailable. This completed cleanly; it does not guarantee future OOM or system safety.

## Execution and evidence index

- [Actual independent aggregate](../results/FINAL_BALANCED_SFT_SEED0_ACTUAL_REVIEW_V1.json): metrics, transitions, costs, checkpoint and receipt hashes.
- [Reviewed entry repair](../results/FINAL_SFT_ENTRY_REPAIR_V2_REVIEW_SAFE.json): exact scope and zero-native failure.
- [Candidate source](https://github.com/MelodyResearch/Vero-to-Brain-MRI/blob/b03d0b30a6f0de59f23ae2eb76a5b9563364fa06/src/mri_rl/final_balanced_sft_v1.py), [entry](https://github.com/MelodyResearch/Vero-to-Brain-MRI/blob/b03d0b30a6f0de59f23ae2eb76a5b9563364fa06/src/mri_rl/final_balanced_sft_entry_v1.py), [analysis/ledger](https://github.com/MelodyResearch/Vero-to-Brain-MRI/blob/b03d0b30a6f0de59f23ae2eb76a5b9563364fa06/src/mri_rl/final_accuracy_sprint_v1.py).
- [Candidate CPU tests](https://github.com/MelodyResearch/Vero-to-Brain-MRI/blob/b03d0b30a6f0de59f23ae2eb76a5b9563364fa06/tests/test_final_balanced_sft_v1.py), [entry/metrics tests](https://github.com/MelodyResearch/Vero-to-Brain-MRI/blob/b03d0b30a6f0de59f23ae2eb76a5b9563364fa06/tests/test_final_accuracy_sprint_v1.py).
- [Matched GSPO actual results](../results/ACCURACY_PILOT_V2_ACTUAL_RESULTS_V1.json).
- [Historical formal matrix](../results/ORIGINAL22_FINAL100_ACTUAL_MATRIX_V1.json), [class/format/cost](../results/ORIGINAL22_FINAL100_CLASS_FORMAT_COST_SUPPLEMENT_V1.json).
- [Model handoff and sealed-evaluation boundary](FINAL_MODEL_HANDOFF_V1.md).
- [Evidence-based paper draft](FINAL_ACCURACY_PAPER_DRAFT_V1.md).

Claude Code actually used `claude-opus-5-5` for candidate implementation/design and37CPUtests. Codex implemented the exact launcher, executed native training and evaluation, fixed the zero-native module-identity issue and integrated results. A separately identified Codex reviewer independently verified actual metrics and payloads. No clinician review or independent cross-model review of private artifacts is claimed.

## Remaining work and decision

**No credible new accuracy gain; no expansion.** The current deliverable is a reproducible negative-result development study and the retained Parent. Independent patient performance, a same-contract historical-SFT baseline, and clinical/cross-task capability remain unestablished. A final sealed16 evaluation cannot begin before the researcher confirms a final checkpoint and once-only protocol. Do not replace this boundary with another unapproved training search.

## Selective main-branch publication

The final delivery includes CPU result reproduction, a fail-closed read-only evaluator and the byte-identical native compatibility dependency closure. Historical training entrypoints are not exposed as current package commands. The preserved research commit remains the immutable provenance for actual training; the new final evaluation wrapper is not newly native-qualified and cannot run with missing inputs/confirmation.

## Final closeout decision

Training and method search are ended by the final researcher directive. Parent Adam52 is the sole principal model within the verified selection scope. Independent16 evaluation remains NOT_RUN_INPUT_METADATA_NOT_LOCATED_AND_NOT_UNSEALED. Historical SFT matched evaluation is not fabricated or pooled with formal84; it lacks matched saved outputs and a fresh exact reviewed native execution scope. No further model comparison is launched in this closeout. The final machine table is [FINAL_METRICS](../results/FINAL_METRICS.json); CPU recount and [reproduction instructions](../REPRODUCIBILITY.md) are provided.
