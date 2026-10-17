# Vero-to-MRI Research Covenant — anti-drift, evidence, and cross-model review

**Status:** researcher-directed standing research guardrail. **Scope:** the PRIVATE `Longitudinal-BrainMRI-Verifiable-RL` repository. This covenant supplements, but does not replace, `research/VERO_TO_MRI_PRIMARY_EXECUTION_CONTRACT.md`, `research/VERO_MRI_TRANSFER_EXPERIMENT_CONTRACT.md`, `AGENTS.md` and `research/ENGINEERING_REVIEW_PROTOCOL.md`. A later explicit researcher decision may amend it; agents must record the change, not silently reinterpret it.

## 1. Locked principal research question

**Test whether Princeton Vero's heterogeneous, task-routed, category-balanced verifiable RL/GSPO training recipe transfers to genuinely image-grounded brain MRI and improves patient-held-out cross-task performance relative to same-data SFT, task specialists, proportional sampling and appropriate reward/active-group controls, without destroying grounding or open-ended ability.**

This is an independent transfer investigation, **not** an official Vero collaboration, not a demonstrated novel RL algorithm, and not a clinical MRI model claim.

The research must ultimately include real MRI pixels, valid patient/label provenance, distinct visually answerable task categories, actual reward-driven weight updates, honest Vero/MLX implementation distinctions, matched training/evaluation contracts and a train-task x eval-task matrix. A passed CPU test, rewritten plan, one optimizer update or output formatting success is not proof of transfer.

**Hard anti-drift rules:**

- Do **not** pivot into a standalone longitudinal verifier, tumor binary classifier, dataset-only effort, safety infrastructure project, speculative new optimizer or broad literature-review loop. Longitudinal verification is an optional eligible task, not the parent objective.
- Do **not** treat four reward routes as four independent visual skills merely because their names differ; three tumor routes currently share one mask/image provenance.
- Do **not** promote format compliance, synthetic gradients, local inference, small engineering pilots, more unit tests or checkpoint hashes to a scientific learning result.
- Do **not** erase historical negative results, change denominators after seeing outcomes, cherry-pick checkpoints or relabel a relaxed reward ablation as native Vero.
- Do **not** substitute a vaguely plausible experimental narrative for runnable code, verified data and saved raw receipts.
- Any proposed new architecture, task family or study pivot must state precisely how it improves the locked question and which direct baseline failed. Scope changes outside this contract require a clear researcher decision, not agent self-authorization.

## 2. Experiment lanes: preserve separability

**A — historical controls:** immutable historical SFT28 (28 FLAIR/0 T1, low-slice bias), prior frozen comparisons and strict/relaxed RL results. Never overwrite or retroactively reinterpret.

**B — corrected supervision control:** `balanced_sequence_v2`: 14 FLAIR and 14 T1 sequence examples; same 28 training patients, 112 examples, same remaining 84 task examples, common original initialization, documented optimizer/compute. This specifically tests *changed sequence-example/label exposure*; it does **not** solve the other tasks' axial slice bias. After the verified first step, save and **prove** complete optimizer/RNG/sampler restoration before treating a separate process as continuous SFT. Completion and 54-question internal-development comparisons are future facts until actual receipts exist.

**C — scientifically admissible MRI task redesign:** remove/measure axial-level shortcuts, image invisibility, mask-derived dependence, spatial grid boundaries, class/site imbalance and patient linkage limitations. Preserve task cards and independent reviewer decisions. The prior candidate C exact 28-patient/three-level constraints were not satisfiable for both binary routes; choose an explicitly distinct feasible subset or new legitimate training data rather than weakening clinical labels simply to fill quota. C is a future scientific comparison, **not** a replacement for B's controlled causal contrast.

B and C advance independently when possible; never combine their sample-selection interventions under one causal label.

## 3. Mandatory question before each substantial change

Complete a **Mission Alignment Check** in the experiment record, preferably 6 short entries:

1. **Main-question link:** how does this change identify Vero-to-MRI recipe transfer, or unblock a required controlled experiment?
2. **Claim and falsifier:** what exact hypothesis, rival explanation and measurable negative result?
3. **Change isolation:** what changed, what is frozen, and which historical experiment it may fairly be compared against?
4. **Evidence provenance:** exact real-data/source/patient/model/adapter/reward/prompt hashes; which steps actually ran?
5. **Shortcut & baseline:** class-prior, slice-level, same-patient swap/no-image, simpler model, SFT or RL arm as appropriate?
6. **Resources and decision:** finite compute/review budget, Mac-safe serial execution, checkpoint/resume contract and GO/HOLD/REFORMULATE criterion.

Use research tags on each milestone: `PRIMARY_VERO_TRANSFER`, `B_LABEL_CONTROL`, `C_TASK_ELIGIBILITY`, `ENGINEERING_UNBLOCK`, `EXPLORATORY_DIAGNOSTIC`. If a change has no defensible tag and main-question link, do not consume heavyweight model compute for it.

## 4. Evidence and anti-hallucination covenant

Every consequential claim should be tagged with evidence tier:

- `OBSERVED`: specific code version and actual saved output/receipt; report units, denominators, patient count, run budget and artifact hashes.
- `REPRODUCED`: another reviewer independently recalculated a result from primary artifacts or recreated the failure/correction. Name reviewer and exact evidence inspected.
- `INFERRED`: plausible mechanism supported by observations, **not** a causal conclusion; state competing explanations.
- `PROPOSED`: design or hypothesis **NOT_RUN**.
- `NOT_VERIFIED`: missing data, inaccessible raw record, external access, experiment or review.
- `CONTRADICTED`: actual evidence refutes the claim; preserve rather than edit away.

A markdown report is **not** itself proof of a claimed GPU run. For model results, retain private model-output/weight/optimizer receipts and the permitted aggregate audit. Don't make up figures, clinical labels, p-values, image-grounded reasoning, genuine second-model verdicts, exact optimizer parity, uptime or crash prevention. Separate **format pass**, **task accuracy**, **true image response sensitivity**, **gradient/update**, **patient generalization** and **recipe fidelity**.

Before a final statement, reconcile: `N questions != N patients`; `N reward routes != N independent tasks`; `changed tensors != improved skill`; `saved optimizer state != verified restored update`; `absence of an observed crash != proof of hardware safety`.

When uncertain or contradictory, write `NOT_VERIFIED` and a bounded decisive check, not a guessed story. Reuse existing data/literature audits; do not needlessly audit 13 papers again.

## 5. Genuine cross-model review and researcher-directed Claude execution

**Role assignment is flexible:** the researcher may designate either Codex (GPT) **or Claude Code** as the primary implementation/execution lead. When Claude leads, it may work on eligible code/data, implement debugging, run scoped local inference and GSPO, and push verified changes under the same hardware/data/review contracts. When Codex leads, Claude can serve as the independent second model family; when Claude leads, seek a genuinely distinct model family for independent verification when available. A second agent of the **same** underlying model is an independent agent, not a cross-model validation.

**Project/public data versus service access:** OpenNeuro ds007045 is a public dataset according to the source audit; the project's unpublished analyses, Gitignored raw receipts, patient-linked records and private GitHub repository are distinct artifacts. Public dataset status does not automatically authorize disclosure of every private derived file to any outside service. The researcher explicitly wants Claude Code to participate in this project. Use the normal authenticated Claude workflow and applicable user/service/organizational approval for the **specific** files or actions. Claude should be allowed to use the authorized public MRI data and appropriately approved project materials; do not require a blanket prohibition merely because the dataset name is ds007045.

**Historical approval incident is not a global ban:** one earlier attempt to transmit a **specific unpublished ds007045 payload** for an external Claude review was rejected before Claude ran. That incident does not establish a permanent prohibition on all Claude access, on public MRI data, or on a newly authorized workflow. Determine exactly which payload/transfer and approval mechanism failed, request legitimate renewed approval through the normal mechanism if applicable, and proceed when it is actually granted. **Do not evade a still-applicable denial by disguising the refused payload, changing transport channels to bypass controls, or falsely recording authorization.** The old `loggedIn: false` status was local to an earlier CLI environment; check current state instead of treating it as permanent. Genuine Claude public-source review has since occurred, but did not validate private B/C results.

**Independent critique is adversarial:** each available distinct model reviews actual permissible source, data provenance, prompt/reward/optimizer code, scoring evidence and study design independently, ideally before reading another model's preferred conclusion. Require concrete counterexamples, falsifying tests and independently recomputed results; `LGTM` is insufficient. When Claude is lead, an authorized independent Codex or other model reviews as feasible; if cross-family review is not available for that exact payload, record `CROSS_MODEL_REVIEW=NOT_VERIFIED` without blocking otherwise approved work.

**Decision protocol:**
1. Freeze exact source/config/diff and hashes; provide independently permitted raw evidence without a leading diagnosis.
2. Record each actual reviewer's initial `CONFIRMED_DEFECT`, `PLAUSIBLE_CONCERN`, `DISAGREEMENT`, `NO_ISSUE_FOUND` or `NOT_VERIFIED` verdict with evidence references.
3. Resolve material disagreement with the smallest safe reproducible test.
4. Fix verified P0/P1 defects with failing-then-passing tests and review corrected source/receipts, not summaries.
5. Preserve reviewer identities, permission scope, unresolved disagreements and negative findings. Do not invent multi-model consensus.

P0 correctness, patient leakage, hardware safety and irreproducible optimizer state gate affected model training; P1 methodological confounds require an isolated control or explicit qualification; P2 documentation does not block real progress. An unavailable Claude integration never imposes a blanket stop on legitimate CPU/data tasks or already authorized model execution.

## 6. Hardware discipline and scientific continuity

The researcher explicitly authorizes carefully paced local MLX because no remote GPU is presently available, but a recent macOS watchdog kernel panic has **no established cause**. Never promise it cannot recur.

- Only **one** heavyweight VLM/MLX job at a time; no parallel model-family weights loaded into Mac RAM; run Claude/Codex code review when the heavy worker is idle or light enough to avoid competing resource pressure.
- Permit bounded, independently reviewed B stages, not unbounded automatically expanding overnight training. Use live memory/pressure/swap/power indicators and record unavailable thermal telemetry honestly.
- Save crash-consistent adapter + **complete** optimizer moments/RNG/sampler/LR state and verify restored-versus-uninterrupted next-update equivalence before claiming 28-step continuous SFT. No silent Adam reset.
- Critical resource condition or renewed system instability stops the offending hardware configuration; preserve receipts, do not blindly repeat; continue CPU/data/statistics tasks.
- No global Mac shutdown/reboot, thermal protections bypass, paid GPU, unauthorized data sharing or publication.
- Routine CPU work must not be globally blocked because Claude is unavailable or a separate model tier is paused.

## 7. Overnight priorities and required morning deliverable

Priority: (1) verify saved B step-001 state and exact restore; (2) one genuinely resumed B step-002 plus independent receipts; (3) successive **small reviewed** B stages if safe; (4) C feasible task subsets, visual evidence and shortcut diagnostics; (5) B28 held-out *internal-development* comparison and counterfactual controls **only if complete**; (6) later match-controlled Vero GSPO transfer matrix after task admission. Do not claim B28, C validation or GSPO results prematurely.

The morning brief in Chinese and English must contain real new experiments completed, exact number of updates/subjects/images, comparison against historical controls, B/C changes kept separate, individual task metrics if evaluated, distinct-family reviewer status, the most consequential disagreements/defects and corrections, Mac resource observations, and next single decisive experiment. Link exact private GitHub sources without exposing private data. If incomplete, state the last verified checkpoint and next bounded step instead of inventing a polished success.

**Covenant check:** every new heavy experiment must advance the locked Vero-to-MRI question under verifiable evidence and reviewed, finite resource conditions. This covenant is intended to *detect and correct* drift and hallucinations; no text instruction can guarantee a model will never make an error.
