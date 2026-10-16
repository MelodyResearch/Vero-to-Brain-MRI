# Paced B28 matched evaluation — implementation awaiting independent review

Tags: `B_LABEL_CONTROL`, `ENGINEERING_UNBLOCK`. Status: `IMPLEMENTED`, `TESTED_CPU`; actual model inference: `NOT_RUN`. This module cannot issue its own approval. Existing training source and the historical `frozen_eval.py` remain unchanged.

## Mission Alignment Check

1. **Main-question link:** B28 is the label-exposure SFT control needed before interpreting Vero-to-MRI RL mixture results. Matching its historical54-question development comparison isolates sequence-example exposure while retaining other sampling biases.
2. **Claim/falsifier:** The wrapper should evaluate the completed B28 checkpoint under exactly the historical manifest, system/template and deterministic generation contract. Changed identities, incomplete28-update history, duplicate predictions, reset manifest subsets or missing reload evidence invalidate that comparison and fail before loading a model.
3. **Change isolation:** Evaluation loading/pacing and durable outputs change; the full manifest and its SHA256, native prompt,4-bit base/LoRA settings,256-token generation budget and temperature0 remain frozen. A partial B checkpoint is explicitly ineligible. This is SFT model evaluation, not GSPO or a test of clinical generalization.
4. **Provenance:** Bind B28 receipt, adapter/config,112-example schedule, optimizer-state28 manifest/checksum, reload receipt, model/processor/tokenizer/template files, full manifest/image inventory, source hashes and per-stage reviewer digest. MRI/generations stay private. Prior pipeline provenance pins base weights; this wrapper does not rehash every full base-weight shard on every stage.
5. **Shortcuts/baselines:** Preserve A0/A28 matched results. Report format, answer and pixel-dependence separately. This first wrapper handles the fixed54 development examples only; swapped/no-image/blank controls require separately reviewed manifests and are not silently substituted. Four patients remain exploratory;54 questions are not54 independent patients.
6. **Resources/decision:** Default maximum four new records, hard cap four per reviewed stage, one lazily loaded model. Restore4-bit representation before materializing,18GiB MLX memory guideline/256MiB cache, PacedGate before load/every generation/after generation/model release, two-second inter-record pause. A memory guideline is not a hard allocation guarantee. HOLD/STOP preserve durable outputs; independent reapproval is required for each changed pending-record scope.

## Interface

New module: `src/mri_rl/paced_frozen_eval.py`.

Plan preparation validates genuine B28 completion before importing MLX. It requires all112 executed examples and28 finite gradient norms, the exact frozen B schedule and original initialization, matching final optimizer-state cursor112 and update count28, and the separately hashed successful reload receipt. It then accepts only the existing matched manifest SHA256:

`9e5194cc808ad27592549b3fab21e31fd6ba0a9582ca76fac75928b589b9734f`.

`--plan-only` outputs the exact proposed review scope; it does not run a model. This currently cannot succeed with an incomplete B checkpoint. The command takes `--model`, `--manifest`, `--output`, `--b28-run`, `--contract`, optional `--resume`, and `--max-new-records` (1–4, default4).

Execution additionally requires independently supplied `--approval` and `--approval-sha256`. The approval schema is:

```json
{
  "kind": "paced_frozen_eval_v1",
  "reviewer": "actual independent reviewer identity",
  "verdict": "PASS_WITH_LIMITATIONS",
  "scope": "the exact dictionary emitted by --plan-only"
}
```

The shown scope string is explanatory; an actual approval must contain the exact JSON object, not that string. Allowed verdicts are `PASS` and `PASS_WITH_LIMITATIONS`. Scope includes new record IDs, their image hashes, existing output SHA256, budget, source hashes and resource policy. Reusing an approval for changed source, output or pending records fails. Review authenticity remains the external independent-review process; a JSON validator is not an identity-authentication system.

## Incremental durability and resume

The entire54-record manifest remains immutable. Each stage selects the next unfinished IDs in original order and generates at most its approved limit. Existing output without `--resume` is rejected. The historical `prepare_resume` helper rejects unknown/duplicate IDs, prompt/image/config mismatches and malformed completed lines. A torn final line is preserved privately and repaired; the containing directory is fsynced by this wrapper.

Each completed response is appended and file-fsynced before the post-generation gate, so a subsequent resource STOP does not lose the response. Resource events are recorded separately; if a crash occurs after output persistence but before the post-generation observation, the response remains valid while the missing resource observation must remain honestly unavailable. A fresh reviewed stage rechecks resources and skips durable completed IDs. No automatic retry or background runner is installed.

## Actual generation measurements

Installed source was inspected directly:

- `mlx_vlm/generate/common.py`, `GenerationResult`: `generation_tokens`, `prompt_tokens`, `total_tokens`, `peak_memory`, optional `finish_reason`.
- `mlx_vlm/generate/dispatch.py`: the synchronous return copies those fields from the final stream response; backend peak uses `mx.get_peak_memory()/1e9`.
- `mlx_vlm/prompt_utils.py`: `add_generation_prompt=True` is the existing default, made explicit by this wrapper.

Output preserves the historical scoring schema and adds actual backend counts, `finish_reason`, peak memory in **decimal GB**, native rendered-prompt hash and observed MLX peak bytes. Peak is the process high-water mark, not a per-record allocation delta. `truncated` is true only when the actual reason is `length`, false for `stop`, and null when unavailable; token count equal to256 does not alone imply truncation.

No fabricated fallback token count or finish reason is permitted. A model may return an empty textual answer; it is retained and graded as invalid rather than selectively regenerated.

## CPU verification and remaining gate

Twenty-six focused tests cover incomplete/invalid B28 history, finite-history requirements, true finish-reason semantics, missing metrics,1–4-record budgeting, immutable resume settings, duplicate rejection, torn-tail recovery, exact approval scope, and CPU-only module import. They do not load MLX or a VLM and do not establish runtime generation success. Independent engineering review must inspect this source and exact future B28 artifacts before any model launch.

## Independent P1 correction: freeze pixels against historical comparisons

**Observed defect:** Pinning the54-record manifest fixes image paths but not the bytes at those paths. The first implementation recomputed a new image inventory without requiring it to equal the actual A0/A28 inputs. An integrated CPU regression changed a fixture image while retaining identical manifest bytes; pre-fix `prepare_plan` accepted it and the expected rejection test failed (`DID NOT RAISE ValueError`). Completed-B28 and model-configuration prerequisites were stubbed in this isolated regression; the exercised manifest/pixel planning logic was real.

**Consequence:** A later image replacement could invalidate the matched comparison while producing a new internally consistent B28 inventory. No actual MRI drift was discovered: all current54 image hashes match both historical output files.

**Minimal correction:** The new module now verifies both immutable baseline output file SHA256 values, exactly54 unique matching IDs, category, prompt, current image SHA256, model path/config/template, null template override,256-token temperature0 decoding, one image placeholder,504 verified adapter tensors, pinned A0/A28 adapter hashes and the common adapter-configuration hash. Both baseline file digests enter prediction metadata and the independently reviewed scope. Neither baseline file is rewritten.

- A0 outputs SHA256: `578e523bbbedae6f221ddc5732817e15df2b2f257d4bc23a4fb77d4d85afde59`.
- A28 outputs SHA256: `c65f5e4343f8f9bc3ee6d77c71c5a87dbd794b7280f5c7c85cf937d515142d9e`.

**Regression evidence:**26 focused CPU tests pass, including the original pixel-drift reproduction, baseline digest drift, ID duplicates, prompt/category/model/decoding/adapter provenance mismatch and positive digest binding. Direct checking of the actual54 current images against both immutable historical files also passes. Independent Step D verification completed: the isolated Codex engineering reviewer inspected source SHA256 `645b80af9b1ae80141bfcb484461cf5266f60c65a83df241168275249c14ec73`, ran all26 tests under an explicit MLX/MLX-VLM import blocker (26 passed), and directly verified the real54 records against both immutable baselines. The P1 correction is independently verified. Model execution remains unapproved until actual B28 and its exact future review scope exist.
