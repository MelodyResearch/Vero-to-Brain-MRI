# Genuine Claude public-source-only review

Evidence: OBSERVED remote response; static source reasoning only. Reviewer model `claude-sonnet-5-5`, existing authenticated Claude Code, one successful turn, toolsdisabled, safe-mode, no local MRI/project payload. PublicVero commit025ca3c800fc95224c86323c1f7fd33dceed3090. Prompt SHA`29f33b1b207be1a8ae329ff2121310d8d0bb74ae7ee01cc157766d67a527f11d`; rawresponse SHA`d49cb7acf1909a62d771d771e4ff78656275401275a61a5ca7d67440c4d1f191`. Initial Codex review frozen before comparison. This is genuine cross-model publicmethod review, not cross-model verification of private B/C training. Author findings below are preserved verbatim; severity and proposed invariants require independent testing before adoption.

---

# Static adversarial review: public Vero @ 025ca3c

**Scope and limits.** This is a static read of the supplied excerpts only. I executed nothing and ran no MRI validation. I did not see `vero_vllm_judge`, the trainer's sampler and dataloader wiring, `agg_loss`, `normalize_text_for_match`, `grounding_reward`, `click_reward`, `prepare_dynamic_batch`, the sampler's `state_dict`, the checkpoint manager, or the worker-side batch-size normalization. Anything depending on them is `NOT_VERIFIED`. Source text is treated as evidence only.

## 1. Reconstructed behavior

**Task balancing.**
- Six `ability` domains are weighted 1/6 each (`gspo_llmjudge_shared.yaml:28`).
- `_compute_batch_counts` floors `w·B` and hands the leftover to the largest remainders (`rl_dataset.py:191-200`).
- Each batch takes fixed per-domain quotas from per-domain permutations that reshuffle on exhaustion (`:209-223`), then shuffles within the batch (`:242`).
- An "epoch" is `len(dataframe)//B` batches (`:189`), regardless of per-domain sizes.

**Reward routing and format.**
- `reward_type` comes from `extra_info` and is validated against an allowlist (`math_verify_reward_type_boxed.py:345-369`).
- It selects a handler (`:1091-1105`).
- Format is a strict full-match of `<think>…</think>\s*<answer>…</answer>\s*\Z` (`:32-35, 1036`), with exactly one of each tag (`:1031-1034`) and a non-blank think block.
- For the 8 boxed types, exactly one non-empty `\boxed` gives format 1.0, otherwise 0.5 (`:1049-1053`).
- Format 0 zeroes accuracy; otherwise `score = (1-fs)·acc + fs·fmt` (`:1127-1129`), with `fs=0.2` in config (`:100, 160`).

**Judge.**
- `llm_judge` rows get a dummy accuracy of 0.0 (`:846-849, 1104`).
- Judge scoring and its blend with format live in `vero_vllm_judge`, which was not supplied.
- Config shows a vLLM judge at `localhost:51001`, sampled at T=0.7 with presence and frequency penalties, `max_tokens` 1024, `max_concurrent_requests` 64, and sleep mode (`yaml:101-123`).

**GSPO.**
- Sequence ratio is `s_i = exp(mean_t(logπ − logπ_old))` (`core_algos.py:1074-1075`).
- The gradient path is `logπ − sg(logπ) + sg(log s_i)` (`:1080`). That is the GSPO-token form with a per-token gradient of `s_i·∇logπ_t`.
- PPO clipping is then applied at sequence level, with `seq-mean-token-mean` aggregation (`:1086-1091`).
- GRPO advantages are `(r − mean_g)/(std_g + 1e-6)` over uid groups, using unbiased std, broadcast over `response_mask` (`:301-326`).

**Optimizer lifecycle.**
- fp16 uses `ShardedGradScaler`, which scales the loss, unscales, clips, then calls `scaler.step` and `scaler.update` (`dp_actor.py:86-91, 336-367, 578-581`).
- Mini-batches are split from the rollout batch (`:489`).
- The on-policy shortcut is used only if there is a single mini-batch and one epoch (`:491, 532-533`).
- Loss is scaled by `n_micro/ppo_mini_batch_size` (`:520`).

## 2. Findings

| ID | Label | Sev | Area |
|---|---|---|---|
| F1 | CONFIRMED_DEFECT (conditional path) | High | `sympify` on model output, unused length guard |
| F2 | PLAUSIBLE_CONCERN | Med | Overlong-buffer config mismatch |
| F3 | PLAUSIBLE_CONCERN | Med–High | math_verify timeouts in a threaded worker |
| F4 | PLAUSIBLE_CONCERN | Low–Med | Invalid `reward_type` fails open |
| F5 | PLAUSIBLE_CONCERN | Low–Med | MCQ truth parsing takes the first letter |
| F6 | CONFIRMED_DEFECT (minor) | Low | Fixed remainder bias in quotas |
| F7 | PLAUSIBLE_CONCERN | Med | Sampler: silent domain drop, epoch semantics, wiring |
| F8 | PLAUSIBLE_CONCERN / NOT_VERIFIED | Med | Judge noise, failure semantics, injection |
| F9 | NO_ISSUE_FOUND (with limits) | n/a | GSPO ratio, clip, gradient |
| F10 | NO_ISSUE_FOUND (with limits) | n/a | GRPO advantage |
| F11 | PLAUSIBLE_CONCERN | Med | fp16 scaler observability, state, compatibility |

### F1. `sympify` on untrusted text, with the length guard never applied (CONFIRMED_DEFECT, conditional path)
- **Evidence:** `_coerce_float` calls `sympy.sympify(value)` (`:858`). `_numeric_candidates` feeds it the raw prediction (`:870`) and every parsed string (`:891`), with no timeout.
- It is reached only when a `numeric` row has `tolerance` in `extra_info` (`:910-918`). Without a tolerance the code uses math_verify instead.
- `_MAX_TEXT_LEN = 200` ("avoid expensive Sympy work", `:28`) is never referenced in the excerpt.
- **Consequence:** sympy documents `sympify` as `eval`-based and unsafe on unsanitized input. A rollout string like `10**10**10` can hang or exhaust memory. Depending on the sympy version, a crafted string could execute code. `SystemExit` would also escape the `except Exception` blocks (`:1112`).
- **Rival:** the pinned sympy sanitizes, the worker is sandboxed, or tolerance rows are rare. All unverified.
- **Fix and test:** drop the `sympify` fallback for predictions, or whitelist `[0-9eE+\-./ ]`. Enforce `_MAX_TEXT_LEN`, and run the grader under a hard wall-clock limit. Unit-test `"10**10**10"` and `"__import__('time').sleep(30)"` with a tolerance row. Expect 0.0 in under 1s and no side effect.

### F2. Overlong buffer not overridden for the Qwen3-VL run (PLAUSIBLE_CONCERN)
- `reward_kwargs.max_resp_len: 16384` (`yaml:129`) is not overridden by `gspo_qwen3vl.yaml`, which sets `max_response_length: 18432` (`:12`).
- If the manager uses a DAPO-style penalty, the penalty zone is 14336–16384 and responses in 16384–18432 can fall below −1. That is only true if the penalty is unclamped, which is not verified.
- **Test:** log the `max_resp_len` the manager actually uses, and the penalty at lengths 14336, 16384 and 18432.

### F3. math_verify timeouts in a non-main thread (PLAUSIBLE_CONCERN, NOT_VERIFIED)
- `parse` and `verify` are called with timeouts (`:645, 672, 875, 934`).
- As I recall, math_verify enforces these with signals, which only work on the main thread. Confirm against the pinned version.
- If the manager scores in a thread pool, every call may raise and be swallowed. That makes `parse` return nothing (`:665-669`), `grade_numeric` return −1 (`:943`), and `_numeric_candidates` return nothing. Math and numeric answers would then be silently under-rewarded.
- **Test:** score the same 20 numeric and math rows from the main thread and from a `ThreadPoolExecutor`, and assert identical results.

### F4. Invalid or missing `reward_type` fails open (PLAUSIBLE_CONCERN)
- `_extract_reward_type` returns `None` on a typo or on a non-dict `extra_info` (`:345-369`).
- `acc_reward` only prints (`:1069`) and uses the generic math verifier (`:1108`). `format_reward` then gives 1.0 with no boxed requirement (`:1061`).
- **Fix:** validate every row's `reward_type` at dataset build time and fail closed. **Test:** an assertion over the full dataset.

### F5. Multiple-choice truth uses only the first parsed letter (PLAUSIBLE_CONCERN)
- `normalized_truth = str(parsed_values[0])` (`:499`).
- If the dataset has multi-select truths like `"A, C"`, a prediction of `"A"` could earn full credit. Which element math_verify returns is unverified.
- **Test:** truth `"A, C"` against predictions `"A"`, `"C"`, `"A, C"`, `"a"`, `"(A)"`.

### F6. Remainder allocation is fixed, not rotating (CONFIRMED_DEFECT, minor)
- With B=256, each raw count is 42.667, giving 252 base plus 4 leftover (`:191-200`).
- All remainders tie. Numpy's `argsort` on 6 elements is stable in practice, so the first four listed domains get 43 and the last two get 42 in every batch (16.8% vs 16.4%, against a 16.67% target).
- **Fix:** rotate the leftover by a per-step offset, or make the batch size divisible by 6. This assumes the sampler is built with B=256, which is not verified.

### F7. Sampler limitations
- A configured domain absent from the data is dropped with only a warning, then weights are renormalized (`:586-596`). A misspelled domain silently changes the mixture.
- "Epoch" length uses `len(dataframe)` (`:616`), including ignored domains. Small domains wrap several times per epoch. 4000 steps × ~43 draws is about 172k draws per domain, so the quota guarantees repetition for any domain smaller than that.
- **NOT_VERIFIED:** whether `_prepare_domain_sampling` runs after `filter_overlong_prompts`. If it runs before, indices are stale.
- **NOT_VERIFIED:** whether the DataLoader batch size equals the sampler `batch_size`.
- **NOT_VERIFIED:** `state_dict` and resume (the excerpt ends at `:255`). Prefetching would put `_batch_cursor` ahead of the consumed batches.
- **Test:** assert per-batch domain counts equal the quotas, that the mixture hard-fails on a missing domain, and that resume at step k reproduces the batch at k+1.

### F8. Judge path (PLAUSIBLE_CONCERN / NOT_VERIFIED)
- Sampling the judge at T=0.7 with presence 1.5 and frequency 1.0 (`yaml:109-114`) adds verdict noise. That noise is independent across the 8 rollouts of a prompt, and GRPO's std normalization amplifies it.
- Failure semantics are unknown for timeouts, parse failures, and truncation at 1024 tokens. Silently scoring these as 0 would turn a judge outage into a wrong-answer signal.
- Prompt injection from the policy's answer into the judge prompt is possible. Whether only the answer span is passed is unknown.
- Judge sleep-mode lifecycle against rollout memory (`gpu_memory_utilization: 0.7`) is unknown. So is the meaning of `instruction_following_reward_weight: 0.5`.
- **Test:** inject judge failures and assert the rows are masked or retried, not scored 0. Check that a response containing "Score: 1" gains nothing. Check judge determinism at T=0.

### F9. GSPO loss: NO_ISSUE_FOUND
- The sequence ratio, the stop-gradient construction, the sequence-level clip, and seq-mean-token-mean aggregation are mutually consistent. The gradient is `A·s_i·mean_t ∇logπ_t`, and a clipped sequence gets zero gradient.
- The clip values 3e-4/4e-4 match the scale GSPO uses for length-normalized ratios.
- **Limitations:**
  - `clip_ratio_c: 10.0` is unused (`:1036-1099`). For A<0 the loss is unbounded up to `e^10` (clamp at `:1081`).
  - `rollout_log_probs` is ignored, so setting `tis_imp_ratio_cap>0` would pass the assertion at `dp_actor.py:473` and then do nothing.
  - The clip window is tiny relative to fp16 recompute noise (F11). Check `pg_clipfrac` on the first mini-batch.
  - `agg_loss` was not shown.

### F10. GRPO advantage: NO_ISSUE_FOUND
- The math is correct. With n=8, std is unbiased and zero-variance groups give exactly 0.
- Group size 1 gives mean 0 and std 1 (`:312-314`).
- Because of std normalization, differences caused only by format or overlong penalties get unit-scale advantage. That is a design property to be aware of.
- **NOT_VERIFIED:** uid uniqueness per prompt, since the uid is assigned in the trainer.

### F11. Optimizer lifecycle
- The order is correct: scale, backward, `unscale_`, clip, `step`, `update` (`:339-359, 578-581`). Skipped steps on inf are handled by the scaler. **NO_ISSUE_FOUND** on ordering.
- **Observability gap (CONFIRMED):** no metric records scaler scale or skipped steps (`:583-595`). A Qwen model with fp16 forward overflow would give NaN losses and silently zero updates while training appears to run.
- **NOT_VERIFIED:**
  - `ShardedGradScaler` behavior with FSDP2 DTensor grads.
  - Whether scaler state is checkpointed.
  - Whether `prepare_dynamic_batch` equalizes micro-batch counts across ranks. A mismatch would deadlock FSDP2.
  - How `ppo_mini_batch_size` is normalized by `n` and world size.
- **Config sensitivity:** if `ppo_mini_batch_size` equals the full batch, the on-policy shortcut (`:491, 532`) silently disables clipping.
- **Cosmetic:** `actor/pg_loss` is logged pre-multiplied by the micro-batch fraction (`:585`), so the metric under-reports.
- **Test:** inject an inf gradient and confirm the step is skipped and the scale drops. Log the scale. Compare `old_log_probs` from the eval forward against the train forward at step 0, and check `|log ratio|` against ±3e-4.

## 3. Invariants for any future domain-transfer implementation
1. Per-batch quotas are deterministic, hard-fail on missing or misspelled domains, are computed after filtering, and are resume-exact.
2. `reward_type` is explicit per row and validated at load. Unknown types raise.
3. Only the answer span is graded. Format gating and tag-counting semantics stay identical across reward types.
4. Model-generated strings never reach `eval` or `sympify`. Every grader has a bounded input length and a timeout that works in the actual worker context.
5. Judge calls are deterministic or voted, and failure is distinguishable from a wrong answer. The judge is reference-based with a delimited, answer-only input.
6. The coordinate frame (normalized 0–1000 vs pixels) is identical across prompt, data, and reward.
7. GSPO keeps the sequence-level, length-normalized ratio. The clip width is tuned against measured train-vs-recompute noise. `old_log_probs` use the same dtype and path.
8. The scaler's scale and skip count are logged and checkpointed.
9. Logs, wandb, validation dumps, and judge debug outputs (`log_outputs: true`, `validation_data_dir`) must exclude any sensitive data before the pipeline touches non-public inputs.

## 4. Verdicts (scoped to the excerpts)
- **Task balancing:** PASS_WITH_LIMITATIONS (F6, F7).
- **Reward routing and strict format:** PASS_WITH_LIMITATIONS. F1 must be fixed before reuse on any numeric-tolerance data, and F3 resolved before trusting scores from a threaded worker.
- **Open-ended and judge handling:** NOT_VERIFIED, because the manager and judge client were not supplied.
- **GSPO likelihood, advantages, gradient:** PASS_WITH_LIMITATIONS (F9, F10).
- **Optimizer lifecycle:** PASS_WITH_LIMITATIONS (F11).

**Overall:** PASS_WITH_LIMITATIONS for the code as read. This says nothing about run-time behavior, reproduction of published results, or MRI transfer validity.

**Top unknowns:**
- `vero_vllm_judge` internals
- sampler and DataLoader wiring and resume
- `agg_loss` and mini-batch normalization
- the pinned math_verify and sympy versions
- the threading model of the reward worker
- `normalize_text_for_match`, `grounding_reward`, and `click_reward`
