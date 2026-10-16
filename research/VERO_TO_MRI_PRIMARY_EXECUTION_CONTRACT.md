# Vero-to-MRI — primary execution contract (researcher-directed)

**Authoritative research objective:** Systematically transfer Princeton Vero's **multi-task visual reinforcement-learning training recipe** to genuinely image-grounded brain MRI tasks and test whether the original task-diversity / negative-transfer findings hold. This research is independent and is **not** an official Princeton collaboration.

This is a **research-direction correction**, not an assertion of experimental completion or methodological novelty. Prior 13-paper audits, Dallas data, verifier experiments and negative findings remain intact. Historical scientific conclusions are not overwritten.

## Scientific question and bounded claims

**Primary RQ:** Under the same model initialization, eligible MRI images, total rollout/visual-token/optimizer budgets and held-out subjects, does Vero-style **task-balanced sampling + task-routed verifiable rewards + GSPO** outperform single-category RL and proportional multi-task RL on cross-skill MRI generalization?

**Secondary RQs:** Which MRI categories are helped/harmed? Does training on short categorical/localization answers erode open-ended MRI description? Are apparent balance gains better explained by MedGRPO-style nonlinear reward calibration, active rollout-group filtering, image-budget differences, SFT or extra patient exposure? Do results transport to distinct sites/sequence protocols and genuine longitudinal inputs?

A systematic method transfer is a legitimate empirical objective; **using existing ingredients is not automatically a new RL algorithm**. Preserve strong prior-art baselines (Vero, Med-R1, BrReMark, LoV3D, MedVol-R1, MedGRPO).

## Priority correction

Do **not** redefine the project as a standalone longitudinal anatomical-change verifier. Acquisition-aware change verification is **one optional MRI task/verifier** with its own eligibility gates. A blocked longitudinal biological-change label must **not** stop separate image-grounded, legally available MRI classification, localization, anatomy and open-ended task work.

Do not force six categories: retain only genuinely distinct, sufficiently supported task groups with image-linked, source-valid answers. Never turn structured nonimage numerics into artificial image inputs.

## Existing evidence to preserve (verify latest local/remote before using)

- Dallas ds004856 1.3.0 genuine NIfTI acquisition, image provenance, multimodal/longitudinal geometry and rewards have CPU pilots.
- The current repo README at the time of this contract reports **20 subjects, 40 sessions, 44 volumes** (40 paired T1 + four FLAIR); registration/identity tasks still have scientific QC and label limitations.
- Thirteen Norman-style source-grounded paper audits, independently scoped Claude critique, statistics/novelty audits.
- **No neural Vero-vs-base MRI inference or weight-updating GSPO/GRPO training has yet been verified.** Do not turn CPU pass counts, identity retrieval, an image render or replayed toy rewards into neural outcomes.

## Stage A — Identify runnable MRI tasks *in parallel*, not serially behind longitudinal labels

Create a registry with for each task: original source/license, actual decoded image/volume, patient-study-sequence identity, hidden target/ref evidence, genuine visual answerability, verifier type, splits, per-category counts and difficulty, held-out independence, and failure/abstention behavior.

Candidate routes:
1. MRI sequence/anatomy recognition → choice/string; challenge with metadata-free images.
2. Image/volume spatial region grounding → verified frame-aware 2D IoU or actual 3D mask/coordinate targets; never relabel 2D IoU as volumetric accuracy.
3. Anatomical measurement / normality category → numeric tolerance only if independently calibrated and visibly answerable; keep FreeSurfer references hidden and avoid unsupported clinical diagnoses.
4. Paired longitudinal evidence → same-patient visit/order/sequence and repeatability-aware change, **only when trustworthy change targets exist**.
5. Cross-sequence evidence → T1+FLAIR or other truly co-available sequences and independently supported questions.
6. MRI description/explanation → held-out, expert-validated factual criteria and a separately audited judge; no invented lesions, sizes or unsupported mechanisms.

**Fast-first dataset strategy:** Check available verified Dallas for limited category tasks; in parallel seek a legitimate public MRI annotation source with visible images+categorical/spatial targets (e.g. source-authorized Calgary-Campinas/CC359, available public brain segmentation or permitted fastMRI-derived resources). Record licensing and subject IDs. Never use BrainTRACE test or SGMRI/NeuroQA test for training/tuning; their missing/upstream images are independent blockers, not global project blockers. Keep restricted images/patient records outside Git.

No model accuracy claims on a two-subject pilot or tests whose gold is the same trivial metadata rule being evaluated.

## Stage B — Real base-model / released-Vero checkpoint comparison

First deliver actual saved **image-conditional output** on a small eligible image-linked diagnostic; then expand to patient-separated held-out evaluation only when labels and sample size permit.

- Compare matching Qwen3-VL-8B-Instruct base and zlab-princeton/Vero-Qwen3I-8B with pinned upstream revisions. Same rendered images, image tokens, decoding, evaluation, parser and questions; record native-template and shared-template ablations where relevant.
- Include no-image, metadata-only, blank-image and shuffled-image controls; diagnose hallucination and shortcut behavior.
- Attempt existing authorized local inference including properly tested MLX/Metal multimodal runtime, verifying weight/processor compatibility, model identity, actual visual tensors and generation receipts; unsupported conversion is a genuine blocker. Any small CPU/quantized feasibility run must be labeled exploratory and not misrepresented as matched checkpoint evaluation.
- No paid GPU rental, endpoint purchase or restricted-data authorization without researcher consent.

## Stage C — Transfer the *actual Vero recipe*, not just invoke a checkpoint

Pin the official Vero source commit, inspect sampler, reward dispatch, chat templates, GSPO policy loss, rollout grouping, format/accuracy/length terms and launcher. Record any differences between paper and current code. Implement image+question+hidden-target schema and task-specific reward routes that actually feed optimizer rollouts.

Implement/test the training interface with:
- At least two independently verifiable, heterogeneous MRI categories (third when available); no requirement to wait for longitudinal label.
- Uniform category sampler **and** natural/proportional sampler; detect repeated patients and effective unique-image exposure.
- Typed verifier routes (string/choice, numerical-with-evidence-tolerance, 2D/3D grounding, genuinely validated judge where applicable); trace format failures, hallucination, reward ties, nonzero-variance rollout group fractions.
- Actual GSPO training adapter, pinned base model and reproducible configs. An end-to-end **weight-changing** miniature engineering run is a separate smoke test; scientific training is conditional on real-data labels, held-out patients and authorized compute.
- Standard GRPO and MedGRPO-style nonlinear normalization as fair alternatives, plus zero-variance group filtering matched effective-update controls. Positive affine reward changes are largely invariant under per-group standardization (subject to epsilon); demonstrate mechanism, don't assert it.

## Stage D — Controlled cross-task experiment

All arms from **identical starting weights**, same source pool and split, matched total group rollouts, useful visual tokens, optimizer steps and GPU-hours:
1. Base, untrained.
2. Same-data domain SFT.
3. RL for each eligible task alone.
4. Task-size proportional multi-task GSPO.
5. Equal-category balanced multi-task GSPO.
6. Balanced with genuine task-routed rewards/ablations where that differs from arm 5.
7. Faithful MedGRPO-style nonlinear calibration and zero-variance/active-group control.
8. Retention ablation including an actual open-ended task if validated.

Evaluate EVERY trained checkpoint on EVERY held-out task (not just trained category). Primary outcome: patient-averaged category macro / worst-category performance and off-diagonal transfer deltas. Secondary: per-task error, invalid/refusal, hallucination, format, 2D vs true 3D metric, reward variance, active group count; uncertainty clustered at patient and variation over seeds. Register contrast/margins before held-out test. Never cherry-pick a test checkpoint.

Only later add strong longitudinal/BrainTRACE out-of-source evaluation if image access, independent labels and a sealed protocol exist.

## Multi-agent operating contract

- **Codex:** lead engineer; pinned GitHub, real image access, data/model adapters, verified inference and training, reproducibility.
- **Claude Code:** independent design/novelty critique and review of raw outputs; verify actual authentication. Do not attribute other agents' work to Claude.
- **Literature + novelty agent:** use installed global `$norman-paper-reading` and existing 13-paper audits; update only targeted new prior art or a real contradiction, no new review treadmill.
- **MRI data agent:** independent image-lineage/geometry/annotation and patient-split validity; cannot approve novel labels based on scanner ID.
- **RL agent:** Vero training contract and group/reward correctness.
- **Statistics + impact:** blind model-metric recomputation; separate ML transfer claim from clinical/biological claims.

Agents should conduct independent criticism on material decisions and publish a small discrepancy ledger; do not create redundant status essays.

## Immediate engineering queue (execute until externally blocked)

0. `git fetch` and reconcile this new remote directive with local work; preserve newer scientific artifacts. Run existing tests.
1. Inventory actual MRI target categories, mark eligible vs partial vs blocked. Find the fastest **two nonidentical real image-linked tasks**, not necessarily longitudinal. Use available data authorization.
2. Build model runners and save a tiny real MRI **generated prediction receipt** with controls on a compatible authorized runtime. If unavailable, finish tested exact runtime commands and continue 3–5.
3. Implement/test MRI-to-Vero dataloader, balanced/proportional sampler, task reward router, GSPO training adapter/config, parser and logging. Run unit/integration tests. A no-weight-update replay is not training.
4. Challenge reward shortcuts, source-level leakage, imbalanced effective groups. Fix failing gates and document limitations.
5. Run fair frozen-checkpoint and then modest authorized weight-updating comparisons on eligible data; progress toward cross-task matrix, not standalone longitudinal verifier.
6. Independent scientific and statistical review. Continue feasible work after each commit.

**Completion is not** a new README, 13 review documents, two subjects, a fake reward, passing CPU tests or a speculative GO/KILL memo. Report what actually ran, evidence receipts, Git commit, unresolved external permissions/compute and precisely which experiments remain NOT_RUN. Do not promise asynchronous execution outside the current session.

**Never:** invent MRI diagnosis, access restricted images without consent, mix held-out test labels into training, modify Story A or NeuroBrain, claim original Vero code executed without verification, rent GPUs automatically, publish the private repo, or claim novel ML algorithms without appropriate controls.

## Headless operation: no desktop/browser handoffs for public research

**Explicit researcher direction:** Run routine research autonomously through background-compatible, noninteractive CLI, APIs and existing GitHub credentials. Do not open the researcher's browser tabs, control the desktop GUI, summon manual clicks, or request repeated permission for openly accessible material. The project repository presently contains code, literature audits and aggregate scientific records, not a licensed bundle of MRI patient scans; this fact should not be used to block routine public-source operations. Treat each *future source* according to its actual access terms.

Allowed without repetitive user interaction, within the active execution session:
- `gh`/GitHub API and `git`: search/read open-source Vero/medical-RL repositories, pin commits, inspect pretrained-model metadata, make ordinary commits/pushes **to the existing PRIVATE MRI project only**.
- `curl`/`wget`/`python`/dataset client APIs: inspect/download arXiv open papers, Hugging Face *public and authorized* datasets/checkpoints, OpenNeuro CC0 data via official endpoints, public data-host/object-store files. Retry legitimate API methods, honor rate limits, use staged checksummed retrievals, verify real MRI image bytes rather than symbolic pointers. Prefer CLI/DataLad/git-annex for annex-backed NIfTI.
- Locally preprocess licensed images, derive eligibility counts, implement Vero training adapters, run CPU tests and compatible already-authorized local inference, work with independent Codex/Claude agents, update GitHub issue/research receipts.

**Access split:** PUBLIC / ALREADY_AUTHORIZED / EXTERNAL_APPROVAL_NEEDED / NOT_PERMITTED. A public metadata page is not necessarily permission to use hidden scan bytes. If fastMRI/MR-RATE/ADNI/MIRIAD requires researcher-specific approval, skip that specific source and continue alternative genuinely public MRI image+annotation sources without involving the researcher, unless no feasible route remains. Never forge approvals, autoaccept a DUA, circumvent access controls, make payments, rent GPUs or change repository visibility.

The *first scientific priority* remains systematic Vero-to-MRI task-balanced GSPO transfer, not a standalone anatomical change verifier. Headless data sourcing must be directed toward **two verified heterogeneous image-answer task families**, with independent patient/source checks; a public MRI collection without ground-truth tasks is still only a feasibility source. Run real-model generation when compatible authorized compute permits, then actual weight-updating comparisons; label blocked/not run honestly.

Do not finish early merely because the desktop/browser route is unavailable. Preserve reproducibility and do not promise out-of-session autonomous activity.

