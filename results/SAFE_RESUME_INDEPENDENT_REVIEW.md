# Independent safe-resume review — 2026-10-15

Reviewer: isolated Codex agent `/root/safe_resume_reviewer`. This review uses source, small synthetic CPU fixtures, existing aggregate audits and training-manifest metadata. No MLX/Metal import, model loading, MRI-volume loading, inference, training, native runtime test, external sharing or crash-log reproduction was performed by this reviewer. The initial review below occurred under the post-panic local-model prohibition. A later explicit researcher directive and remote `AGENTS.md` revision `37a9828` reauthorize carefully bounded, serial local MLX work. Historical local launch approval is not reused; the new exact staged configuration requires a fresh scoped review.

## Initial findings requiring correction before optimizer integration

The following were reproduced against the initial uncommitted `optimizer_checkpoint.py`; they are not hypothetical failures inferred from naming.

1. **P1 — Missing trainable moments are silently accepted.** `_validate` checks that at least one `m`/`v` pair exists and validates each discovered pair. It does not compare paths, shapes and dtypes against the expected complete trainable parameter inventory. A state containing only `weights.present` saves and reloads successfully while bindings declare both `present` and `omitted`. A resumed optimizer can initialize the missing parameter's moments from zero and silently break continuity. Require an explicit expected inventory and exact agreement at save and load, with a documented initialization exception only for zero effective updates.
2. **P1 — Seed-only records pass the advertised RNG-state gate.** `rng_state={'seed': 7}` saves and reloads successfully. This cannot recover the next stochastic draw. Require typed, backend-specific state schemas, the complete declared RNG inventory, and sampler cursor/state. Reject seeds presented as resumable state. Preserve generic serialization separately if desired, but do not describe it as RNG recovery validation.
3. **P2 — Publication can overwrite a destination created after the existence check.** An injected creation of an empty destination immediately before `os.rename` reproduces successful replacement. The helper's immutable/no-overwrite guarantee is therefore false under contention. Use an exclusive publication mechanism or explicit cooperative lock with a recheck and test the competing-writer boundary.
4. **P1 — Directory entries are not synchronized before publishing the checkpoint.** Instrumented ordering is `file fsync, file fsync, rename, parent-directory fsync`. The temporary checkpoint directory itself is never synchronized. File sync alone does not establish the claimed crash-durability of its directory entries. Sync the completed temporary directory before publication and its parent afterwards. This is especially material after the actual system panic; no crash was induced to demonstrate it.
5. **P2 — SFT completed-step durability also omits containing-directory sync.** `_atomic_json` and `_save_step_checkpoint` sync file contents then rename without syncing the containing directory. Add directory synchronization and a CPU call-order regression, or weaken the durability claim. These checkpoints explicitly omit optimizer/RNG state and must remain unsuitable for exact resume.

The initial checkpoint utility is **NOT_VERIFIED for integration as an exact continuous-training checkpoint**. Existing tests passing do not cover these failure modes. The historical native MLX CPU receipt is preserved evidence from before the prohibition, not a fresh test or permission to repeat it.

## Corrected supervised control

Independent replay of the actual training manifest verifies 112 examples, 28 patients per category, corrected sequence exposure 14 FLAIR / 14 contrast-enhanced T1, identical patient ordering, and all 84 non-sequence rows identical. This replay only reads JSON and checks image-file existence; it does not load MRI arrays or weights.

| Historical category | Labels | Selected axial positions |
| --- | --- | --- |
| Sequence | FLAIR 28 | z045: 28 |
| Tumor presence | yes 14, no 14 | z045: 28 |
| Enhancing presence | yes 11, no 17 | z045: 28 |
| Tumor grid | left-middle 11, right-middle 8, center-middle 5, center-top 3, left-bottom 1 | z045: 14, z091: 14 |

Each historical category uses 28 distinct image paths. Presence and enhancement nevertheless reuse the same patient/slice images across tasks. Exact content duplicates and mask-area distributions belong in the lead's full 112-example audit; distinct paths alone do not establish distinct images.

The source of `sft_evidence.py` scopes historical supervised evidence to the balanced SFT28 runner and cannot authorize RL through the dedicated dispatch branch. Historical receipt order, losses/gradients, adapter hashes and reload artifacts are checked; selected future images, questions and targets are frozen. Historical record/image identity remains reconstructed because the original receipt lacks original image hashes. The historical original-adapter receipt hash is `502084c8a1394fdffd765741b9b0fbfcb6720268dfe7671c20d7a67dc5f86088`; this review reads its metadata and does not decode weights.

The corrected `balanced_sft_28step_v2` output has no completed-progress receipt or final receipt at initial inspection. No completed corrected update is claimed. Prior local Tier C approval was superseded by the user's post-panic restrictions. The latest directive now permits a new carefully paced local configuration; remote GPU is unavailable and remote preparation is no longer the immediate priority.

## Scientific task admission

The corrected schedule is a controlled sequence-exposure repair. It preserves lexical axial-level bias in the other 84 examples. It must not be described as a fully debiased MRI training set or an isolated label-weighting intervention: 14 sequence images and their targets change together.

- **Sequence discrimination:** candidate for primary visual evaluation only after paired same-patient/level balanced labels, hidden modality paths/titles, fixed rendering, patient-held-out evaluation, class-prior baselines, and image-swap/blank controls. Source/preprocessing shortcuts and external pretraining overlap remain unverified.
- **Tumor presence:** retain current tasks as mask-derived slice engineering endpoints. The published training-only axial baseline ties historical SFT28 on development tumor presence (10/12), and all source patients have tumors. A negative slice is not a healthy patient. A primary endpoint needs varied levels, within-level positive/negative support and a prespecified visibility rule or independent adjudication.
- **Enhancing presence:** do not admit any-positive-voxel labels to the primary visual reasoning matrix without answerability checks. Existing evidence includes one-voxel positives and the axial baseline exceeds historical SFT28 (10/12 versus 8/12 on development). Computable mask truth does not establish visible evidence.
- **Tumor grid:** conditional question eligibility exposes tumor presence; the exact categorical centroid can lie near a boundary or outside the lesion. Keep it engineering-only until clear reference definitions, boundary tolerance or exclusion rules fixed from training/development, and visibility adjudication exist. The present grid is lesion localization, not anatomical localization.
- **Anatomical localization:** no new primary task is admitted by the existing grid label. Require a separately defined anatomy reference, orientation convention, visible landmark and reference verification before inclusion.

Four repeatedly inspected development patients and the historical eight exploratory patients cannot support a new confirmatory claim. The metadata-only 14-patient cohort remains unreleased. Freeze selection, endpoint eligibility, renderer, model/settings, pixel controls, scoring and patient-level uncertainty before accessing its images or labels. Site-balanced sampling defines a site-balanced estimand. The existing fixed three-level endpoint may be retained as a separately labeled diagnostic but does not cure the shortcut.

## Bounded validation

One thread-capped Python process ran `test_optimizer_checkpoint.py`, `test_sft_supervised_evidence.py` and `test_balanced_sft_comparison_audit.py` with `MRI_TEST_MLX_CPU=0` and an import hook rejecting both `mlx` and `mlx_vlm`: **16 passed, 1 skipped in 0.22 seconds**. The native MLX test was skipped. The requested `nice` priority reduction was denied by the sandbox; work remained bounded and completed in under a second. No concurrent CPU-intensive reviewer work was launched.

Additional tiny NumPy reproduction established the four optimizer defects above. Its arrays have two elements; it uses no model or MRI data. Parent-directory syncing happened after rename, confirming the absence of a temporary-directory sync before it. A save callback injected the empty-directory race deterministically; no stress test was used.

The first revision's local-execution guards were checked to precede MLX imports. The latest explicit authorization now requires replacing that blanket prohibition with the exact reviewed serial configuration. This initial review is an engineering and scientific limitation record, not SFT launch authorization.

## Follow-up under the renewed serial-local authorization

The lead corrected optimizer publication using exclusive directory reservation, a manifest published last, and synchronization of the temporary, destination and parent directories. Save and load now compare complete Adam moment paths/shapes/dtypes to an expected trainable signature. Seed-only state is rejected. A second independently reproduced defect, malformed NumPy state `{'bit_generator': 'PCG64', 'state': 7}`, was fixed by constructing the supported CPU bit generator and verifying its state setter accepts the structure. SFT file publication now synchronizes containing directories. Exact full-run recovery still depends on runner bindings, all used random generators, sampler position, scheduler and shared model/state publication.

The updated focused suite initially passed **21 tests with one native test skipped** under an explicit import hook rejecting MLX/MLX-VLM. Subsequent new staged-run implementation is a separate review scope.

For the newly authorized tiny compatibility check only, the reviewer permits the lead to run one small synthetic MLX CPU checkpoint/resume and deterministic checkpointed-versus-uncheckpointed forward/gradient fixture, with no pretrained model or MRI data. Setting the default array device to CPU does not prove that importing MLX avoids runtime Metal initialization. This permission is limited to that compatibility check and does not authorize SFT28 by itself.

Source inspection supports language-layer activation recomputation as a way to lower retained training activations without changing image resolution, precision, targets or optimizer semantics. The installed `mlx_vlm.trainer.common.utils.grad_checkpoint` wraps every instance of a layer class and also affects later reload instances. Restrict it to language decoder layers; the broader model helper also wraps vision blocks. The saved LoRA configuration has zero dropout, and the inspected Qwen3-VL decoder has no dropout or mutable cache when SFT supplies no cache. Record the implementation change and verify a small deterministic forward/gradient equivalence fixture; no full-model bitwise equivalence is claimed from that fixture.

Materialize the restored quantized parameter tree before the first gradient, release safe temporaries, and sample headroom again. The installed loader defaults to eager loading, while quantization restoration builds lazy operations; a lazy base load followed by restored quantization and evaluation avoids intentionally retaining the eager floating-point base before training. This is a memory-lifecycle finding, not an attribution of the watchdog panic. The installed MLX documentation explicitly describes its memory limit as a guideline; it is not a guaranteed allocation ceiling or a guarantee against system failure.

The preferred new staged design keeps one live model and one Adam instance across short stages. Adapter-only checkpoints remain clearly labeled as insufficient for exact cross-process resume. Any warning handling must preserve that optimizer state in the live process or stop with an explicit incomplete-run status. A first critical resource signal must prevent the next heavy unit even before multi-sample hysteresis reaches its critical state.

## Latest scope correction: audit B and C before one real step

The subsequent researcher directive explicitly prioritizes the newly established axial-level selection defect. No full 28-step approval was issued. The next real-model scope is at most one independently reviewed compatibility step after both schedules have been audited:

- **Control B:** preserve `balanced_sequence_v2`, the same 28 patients, 14/14 sequence labels and all 84 original non-sequence examples. It isolates the intended sequence-example selection change while retaining low-slice bias. Associated sequence images and labels change together.
- **Candidate C:** create a separate versioned training-only schedule spanning suitable levels and labels, with explicit visibility eligibility and ambiguous positives excluded or marked unresolved. A mask-area threshold by itself cannot establish visually answerable lesions. Do not inspect held-out predictions or collapse this redesign into B.

A one-step B-prefix compatibility result must bind the complete frozen 28-step schedule and identify the executed first four examples. It is an engineering result, not completed B28, and its saved optimizer/RNG sidecar does not establish cross-process resume until that runner path is implemented and verified. Further real updates require a separately reviewed expansion after the smaller result and sampling design pass. The prior proposal to continue automatically to 28 on healthy resources is superseded.

### Independent B/C audit result

The expanded historical audit has 112 rows with 28 distinct patients per category, 70 unique image hashes and 42 uses of already used images across categories. No identical image-plus-question pair is repeated. Native mask affine orientation is LAS for all 112; source-axis z values are 45/91/136, and the saved renderer applies `rot90` and 2× enlargement. Native affine orientation must not be confused with a fresh canonical orientation of the rendered panel.

The selected historical tumor-positive masks have minimum area 268 native slice pixels; selected enhancing-positive masks have minimum area 257. The previously documented one-voxel enhancement ambiguity belongs to other rows in the broader pack, not these selected 112 historical examples. These selected area statistics still do not establish clinical visibility. The new private rows retain both tumor and enhancing area, question hashes and explicit `VISIBILITY_NOT_ADJUDICATED` status.

Candidate C's 49 proposed rows all exactly match the 382 training-only references. Its sequence subset contains 28 patients, 14 FLAIR / 14 T1, and level counts z45=10, z91=9, z136=9. Its 21-patient grid subset remains engineering-only and is not class-balanced. The candidate explicitly has `gpu_admitted=false` and `visual_adjudication_complete=false`; it is not an executable matched 112-example replacement.

An independent breadth-first augmenting-path max-flow implementation checked every positive three-level quota split, separately from the production depth-first matching algorithm. With 28 distinct patients, exact 14/14 binary labels, within-level class balance at all three levels, negatives with zero mask area and positives with at least 1,000 native mask pixels, no complete 28-patient assignment exists. The largest flow across those quotas is 27 for tumor presence and 19 for enhancement. Enhancement has only 2/3/0 eligible positive reference rows at z45/z91/z136. This is infeasibility under the declared candidate constraints, not evidence that every slice-aware design is impossible. Visibility review and a separately justified revision are needed before admitting C.

The final focused reviewer CPU pass at this point is **22 passed, one native MLX test skipped in 0.13 seconds**, with MLX/MLX-VLM imports explicitly blocked. A prior invocation named the just-removed blanket-ban test file and collected no tests; the reported passing count comes from the corrected invocation. No reviewer model or native-runtime execution occurred.

## Exact one-step authorization

**PASS_WITH_LIMITATIONS for one paced Control B compatibility update only**, followed by one sequential fresh-checkpoint reload and one greedy training-image generation capped at 256 tokens. No second update, full B28 run, Candidate C run, RL or multiarm expansion is authorized. The lead must verify current actual resource conditions and retain monitoring during execution. Another serious instability or persistent hold does not permit an automatic retry.

The reviewer independently authored `local_data/ds007045_bounded/paced_sft_review_v1/approval_prefix1.json` and transmitted its SHA-256 through the separate reviewer message: **`3413e06b3205b779af66b9d2e19eaf60faf0d27d6783838ec8d2a192a6d57a54`**. The approval binds all eleven source files, the unchanged contract and original adapter, the complete 112-example schedule, the first four executed examples, exact pacing policy and the unique output directory `balanced_sft_prefix1_paced_v3`. Source copies are preserved in the private `source_snapshot_prefix1` directory. Any scope change invalidates approval.

The full schedule SHA-256 is `2e5d6bcb55cfb949035986fa8e65debdf13013badb04c275d195fd96be661f67`; the four-example executed prefix SHA-256 is `a27b301f679be9126101144c3ba39d2f369ed88abf165b6f8a639247a9a12bbf`. A hash-bound prefix prevents treating the full 14/14 schedule as evidence that the one-step job executed all 112 examples.

Final independent checks passed **26 tests with one native MLX test skipped in 0.09 seconds**, with imports of MLX and MLX-VLM blocked. The reviewer reran actual static contract/data preflight and supervised-evidence verification, including selected image hashes, original/historical adapter hashes and 54-row historical reload provenance. Weight arrays and models were not decoded or loaded. The previously verified approximately 16 GB base shards were not rehashed again; that limitation is explicit in the approval.

The latest runner samples conservative resource thresholds immediately before every image and optimizer update. It checks changed adapter bytes and finite parameters at each completed step, writes optimizer/RNG state and the schedule cursor, and preserves `exact_resume_supported=false` until an actual cross-process recovery runner is verified. The single-step boundary includes four finite per-example losses and one finite positive gradient norm. The completed B28 flag remains false.

The reviewer-authored private `launch_prefix1.py` defaults to a read-only preflight; `--execute` explicitly enables only the approved one-step scope. Its syntax and paths were checked without running it. The reviewer did not launch the model and becomes idle during the lead's model work. A training completion claim remains pending independent inspection of the actual resulting receipts.

## Independent post-run verdict

**PASS_WITH_LIMITATIONS: the approved one-step engineering compatibility run completed.** No additional training is authorized by this verdict. The actual receipt, progress file, saved complete schedule and four executed examples match the approved source, contract, prefix hash and one-update budget. All eleven current source hashes still match the reviewed snapshot. The interrupted earlier directory and historical SFT28 remain separate; this result is not completed B28.

The reviewer recomputed the four-loss mean as **6.74751353263855** and verified a finite positive gradient L2 of **8.834030061690534**. The saved adapters contain 504 matching tensors and 10,911,744 elements. Bounded per-tensor CPU reads, without MLX or a model, independently verify that all values are finite and **252 tensors changed**. Independent float64 accumulation gives parameter-change L2 **0.07449211499523938**, agreeing with the runtime's float32-based value within tolerance; the maximum absolute parameter change is **0.00003162274515489116**.

| Verified artifact | SHA-256 |
| --- | --- |
| Original/before adapter | `502084c8a1394fdffd765741b9b0fbfcb6720268dfe7671c20d7a67dc5f86088` |
| Completed step and final after adapter, identical bytes | `1c7c27bd4c4a902026334d49c88eb8d602678ab517880823d92c614e3de12c04` |
| Optimizer/RNG tensor payload | `62096c52176adeb8b94d50cc9fa95cf25482695b219afdb9c33071c57f59fb6f` |
| Fresh reload inference receipt | `0239ad1c85a8886eddc3eac9fe8e262c606f55128de54e9e96cc4f7c523a6915` |

The optimizer sidecar contains **1,011 tensors**, including 504 complete Adam moment pairs, scalar step 1, learning rate approximately 1e-5 in float32, and the MLX RNG array. Every header dtype, shape, byte range and manifest tensor reference matches; the payload checksum matches the manifest. All moment values are finite and second moments are nonnegative. The complete moment inventory matches the adapter signature. Python RNG structure is accepted by a fresh Python RNG object; the MLX RNG header has the required uint32 shape. Adapter identity, contract identity, full schedule hash and **sampler cursor 4** are bound in the manifest. These checks establish saved state integrity, not tested cross-process recovery of this real training run. `exact_resume_supported=false` remains accurate.

The fresh reload receipt records exact equality of all 504 saved adapter tensors before actual generation. Its image hash and record match the fourth executed training example. The reviewer independently reconstructed the prompt using the pinned native Jinja template and verified its SHA-256, along with the unchanged greedy decoding and 256-token cap. The generated response is nonempty. This verifies the recorded reload path and successful generation; the reviewer did not reload a model. The strict parser independently scores the response **0 accuracy, 0 format, 0 total reward**. A successful compatibility step does not imply task learning or improved image dependence.

All **13 recorded resource-boundary samples** independently reproduce `CONTINUE` / `NORMAL` with no warning reasons. All report AC power and zero other detected heavyweight processes. Minimum sampled available memory is **9.4148 GiB**; minimum pressure-free percentage is **48%**. The largest recorded MLX peak counter is **11.4764 GiB**. Swap rose from zero to **0.8930 GiB**, then was **0.8852 GiB** at the last recorded boundary. Thermal telemetry remained `SENSOR_UNAVAILABLE`. The completed-step sample occurs more than 30 seconds after the pre-update sample, consistent with the approved pause. Before the fresh reload, active MLX allocation had fallen to about one KiB, supporting release of the training model before the next load.

These are boundary observations. The final event occurs before the reload generation; there is no post-generation sample in this run's resource log. Consequently **11.4764 GiB is the largest recorded peak, not a verified whole-run maximum**, and unavailable thermal telemetry cannot certify temperature. No inference about the cause of the earlier kernel panic follows.

The independent machine-readable verification is preserved privately as `paced_sft_review_v1/postrun_prefix1_independent_verification.json`, SHA-256 **`d6e2524c93ae3142360987ead76a0ba4ee6640b87fc2425ead2d48784dbe53a9`**. Inspection used small metadata structures and per-tensor CPU buffers; no MLX import, model load, MRI-volume load or additional GPU experiment was performed by the reviewer.

### Exact next bounded action

First implement and CPU-test a real restoration path for the saved adapter, complete Adam state, all used RNGs, fixed learning-rate settings and the frozen schedule cursor. Verify uninterrupted-versus-restored equivalence on the permitted tiny fixture and reject missing or mismatched state. Preserve the one-step checkpoint as the starting boundary. After a fresh independent review of that implementation and actual resources, the next model experiment should be **one resumed update on Control B schedule examples 5–8**, followed by checkpoint integrity verification. That experiment is not currently authorized or executable as an exact continuation; resetting Adam and replaying from the original adapter would answer a different question.

Candidate C remains a separate, incomplete design with visibility pending. Continue its training-only eligibility review and revise its infeasible constraints transparently. Neither that limitation nor the successful one-step compatibility result warrants silently merging C with B or expanding to full 28-step training.
