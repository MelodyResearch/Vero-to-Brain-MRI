# Post-panic serial MRI training compatibility result

**Executed:** `balanced_sft_prefix1_paced_v3`, one optimizer update on four real training images, followed by one fresh checkpoint reload and one image-conditioned generation. Process exited 0. **Not executed:** completed corrected B28, Candidate C training, new development/test evaluation or further GSPO. No automatic second stage was started.

This is `TRAINING_COMPLETED` for the one-step engineering prefix and `INFERENCE_COMPLETED` for its reload diagnostic. It is not a scientific Vero-transfer result. The source and exact configuration were independently approved by isolated Codex reviewer `/root/safe_resume_reviewer`, verdict `PASS_WITH_LIMITATIONS`, approval SHA-256 `3413e06b3205b779af66b9d2e19eaf60faf0d27d6783838ec8d2a192a6d57a54`. Claude did not review this payload. See the independent post-run review in `SAFE_RESUME_INDEPENDENT_REVIEW.md` and machine-readable aggregate `PACED_SFT_PREFIX1_RESULT.json`.

## Actual result

| Check | Observed |
|---|---|
| Starting checkpoint | Preserved original pre-SFT adapter; SHA `502084c8a1394fdffd765741b9b0fbfcb6720268dfe7671c20d7a67dc5f86088` |
| Data exposure | Four examples, four train patients, one example from each of four categories; first sequence example is T1, so this prefix is not itself label-balanced |
| Optimizer | One continuous MLX Adam object; LR 1e-5, bias correction false; microbatch 1 with four averaged gradients |
| Losses | 6.12026882, 6.87427282, 6.95437431, 7.04113817; mean **6.74751353** |
| Gradient L2 | **8.83403006**, finite and nonzero |
| Weight change | **252 of 504 adapter tensors** changed; all finite; parameter-delta L2 **0.07449212** |
| Final adapter SHA-256 | `1c7c27bd4c4a902026334d49c88eb8d602678ab517880823d92c614e3de12c04` |
| Saved update | Step-001 checkpoint hash equals final adapter hash; progress, optimizer moments, MLX/Python RNG and frozen sampler cursor saved |
| Reload | Fresh sequential base+adapter load; exact equality for all **504** adapter tensors; real training-image generation, greedy, cap 256 tokens |
| Strict Vero reward on reload output | **0.0** total / task / format: required think/boxed structure missing. The literal grid label matches its target, but the strict gate correctly rejects it. This single seen training image is not an accuracy estimate. |
| Full B28 complete | **False** |
| Exact cross-process resume supported | **False**; state is saved, but a trainer restore entry point and next-update equivalence remain unverified |

The full B112 schedule and executed four-row prefix have separate hashes. B preserves all 84 historical non-sequence selections, including their slice bias. Candidate C was independently audited and remains ineligible for training under its current visibility and feasibility constraints; see `HISTORICAL_SAMPLING_AND_ADMISSION.md`.

## Resource observations and limits

The new implementation loads the base lazily, materializes the restored quantized weights, uses language-layer activation recomputation, releases evaluated temporaries, limits the allocator cache, and pauses between images and after the update. These are declared execution changes; no new image resizing, target shortening or optimizer recurrence was introduced. Synthetic CPU loss/gradient equality does not prove full-Qwen bitwise equivalence.

Thirteen boundary samples all recorded `CONTINUE`, AC power and no competing heavyweight process. Minimum recorded available memory was **9.4148 GiB** and minimum pressure-free memory **48%**. Maximum MLX-reported peak observed at those boundaries was **11.4764 GiB**. Swap rose from zero to **0.8930 GiB**, then **0.8852 GiB**. The last in-process sample precedes the final reload generation, so these are boundary observations, not a certified whole-run or system-wide peak. No thermal sensor was available. A later model-free sample showed AC, no heavyweight process, 13.0287 GiB available and 0.8774 GiB swap.

The 18 GiB MLX memory setting is a guideline, not a hard allocation guarantee. This attempt finished normally; a user-space monitor cannot rule out another OS watchdog failure. No reboot, shutdown, kernel-protection or global power-management setting was changed.

## Preserved interruption and checkpoints

`balanced_sft_28step_v2` was left untouched. It contains its frozen schedule and original starting adapter; there is no independently verified completed-step receipt or final receipt. This does not prove that no device operations occurred before the crash.

Post-reboot streaming hashes verified the historical SFT28 final adapter (`244c26fd…58ed73`), the strict-presence RL adapter (`1eb17b67…f07446`) and original initialization. The private recovery receipt's earlier `model_restart_authorized=false` records the temporary post-panic directive; it is superseded by the user's later explicit bounded-local authorization, not silently rewritten. Raw panic logs remain local. The watchdog timeout is observed; its cause is not established.

## Verification and exact next bounded work

- Fresh default suite: **180 passed, one native MLX test skipped**. The reviewer separately ran 26 focused CPU tests with native imports blocked.
- Explicitly authorized tiny native CPU Adam/RNG continuation: one test passed. A separate four-dimensional layer fixture matched deterministic loss and gradients with/without activation checkpointing.
- Next engineering work: integrate loading the saved step-001 optimizer/RNG/sampler state, bind the execution source/configuration and constant learning-rate semantics, and verify corruption rejection plus continuous-versus-restored next-update equality on a tiny native fixture. Saving state alone does not prove recovery.
- **Next intended model experiment:** after a new independent scope and live preflight, exactly one update on B's next four frozen rows from the verified step-001 adapter **and restored Adam/RNG/cursor**, in a fresh output directory; then save and verify. This is not executable as an exact continuation until the restore path is implemented and independently checked. Do not reset Adam and call it step two of matched B28. No automatic 28-step expansion.
- Continue train-only C data work independently: obtain support at appropriate levels or prespecify a smaller matched subset, with visual answerability review. Do not relax mask-area eligibility merely to complete 28 slots.

**Scientific significance:** the verified axial sampling bias changes how historical results must be interpreted. **AI/ML significance:** no balancing or GSPO transfer benefit has yet been shown. **Experimental significance:** one actual post-panic weight update, saved full optimizer sidecar and exact checkpoint reload now support a cautiously scoped next stage. The Vero-to-MRI research remains active and incomplete.
