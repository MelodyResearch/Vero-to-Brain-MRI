# Minimal correction for continuous local GSPO optimization

**Original design status:** `PROPOSED` correction at the time of the review below. **2026-10-16 update:** a separately versioned implementation now has CPU and tiny native fixture evidence plus scoped independent `PASS_WITH_LIMITATIONS`; see [implementation](../results/CONTINUOUS_RL_IMPLEMENTATION.md) and [independent review](../results/CONTINUOUS_RL_INDEPENDENT_ENGINEERING_REVIEW.md). Real-Qwen integration and MRI RL remain `NOT_VERIFIED` / `NOT_RUN`. The dated design evidence below is preserved. Review tag: `ENGINEERING_UNBLOCK`. Reviewer: independent Codex method/statistics agent. `CROSS_MODEL_REVIEW=NOT_VERIFIED` for private/local interfaces. The genuine Claude review covered public upstream source only and does not approve this correction. This document preserves the historical one-update experiments and does not alter B or C.

## Mission alignment

1. **Question:** eliminate an optimizer restart confound before interpreting specialist/proportional/uniform MRI RL comparisons.
2. **Claim/falsifier:** consecutive local updates can retain complete optimizer and sampling state; an interrupted/restored next update must match uninterrupted execution on a controlled fixture. Missing moments, cursor/RNG divergence, duplicate updates or mismatched weights refute continuity.
3. **Isolation:** first repair state continuity while retaining the explicitly declared local Adam/LoRA objective and fixed strict reward. Matching upstream AdamW/minibatching is a separately versioned fidelity change, not silently folded into the continuity correction.
4. **Evidence:** inspected local source and pinned upstream Vero commit `025ca3c800fc95224c86323c1f7fd33dceed3090`; no model execution in this review. Existing CPU tests ran with MLX, mlx_vlm, Torch and Transformers imports blocked.
5. **Rivals/controls:** optimizer restart, effective update imbalance, active-group filtering, format-only reward, image shortcuts and different rollout budgets remain distinct explanations.
6. **Resources/decision:** one heavy model at a time, bounded existing approval tiers, completed-group checkpoint boundaries and independent review. No paid compute, private external sharing, repeated paper audit or development-answer-driven rollout selection.

## 1. Observed lifecycle and precise consequences

Inspected local source SHA-256: `mlx_gspo_smoke.py` = `4e9e424d683ca9c2ec7748b8fe6ba79e12e458514d3218e7057720968d2dbb1a`; `mlx_multiarm_train.py` = `2a982352e0820bc09a0ac6029c95c8519f0e3e06fbf50c19c86c9c86b1e48bde`. The upstream working tree was clean; core/actor hashes match the earlier pinned public audit.

Local citations refer to the source inspected for this design; implementation changes require renewed hashes/review.

- `src/mri_rl/mlx_multiarm_train.py:268` verifies earlier child receipts and advances only the adapter pointer; `:333` launches a fresh `mlx_gspo_smoke` process per prompt group; `:353` carries forward the next adapter. The receipts accurately declare `optimizer_reset_each_group=True` and `exact_optimizer_resume=False` at `:201` and `:249`.
- `src/mri_rl/mlx_gspo_smoke.py:228` seeds and loads a model, verifies the supplied adapter, and generates responses. `:343` stores detached old log probabilities before any update. `:392` checks old/current agreement. `:432` constructs a new Adam, executes exactly one update and saves adapter weights. `:451` deletes the optimizer before a fresh model reload. There is no optimizer-state output/input in this path.
- **Confirmed R06:** adapter continuity is not Adam continuity. Old moments and the update counter are absent in each new child. The existing SFT/native checkpoint repairs do not change this RL path.
- **Additional source-derived limitation:** every actual update in this path is evaluated at approximately unit sequence importance ratio, because old/current are checked before the sole optimizer update. Carrying Adam across fresh one-group updates will repair continuity but will still leave clipping inactive at each initial on-policy loss evaluation. This is not an algebraic GSPO defect: the upstream actor explicitly has a one-minibatch, one-epoch on-policy branch too. It means a successful one-update diagnostic does **not** empirically verify clipping under changed policy or reproduce the upstream multi-minibatch rollout/update cycle.
- `task_group_is_informative` (`mlx_gspo_smoke.py:128`) requires every completion to have full format, differing task accuracy and equal overlong penalty. This is a useful task-gradient diagnostic gate. It is a **selection intervention** and must not be described as unmodified Vero sampling/training.

Existing CPU numerical evidence independently re-executed here: `test_continuous_and_resumed_match_and_reset_does_not` proves exact resumed equality on the declared NumPy Adam fixture and a detectable reset difference. It does not prove the full RL driver restores state, reproduce a Qwen gradient, or validate native MLX serialization in this review.

## 2. Pinned upstream invariants versus the present local approximation

Primary sources below are local copies verified at the pinned upstream commit, under `work/vero-official` adjacent to this checkout's outputs directory. Public links pin the same commit.

| Component | Pinned Vero source | Current local path / required interpretation |
|---|---|---|
| Advantages | `core_algos.py:264–326`: response outcome sums; per-prompt sample SD (`torch.std`, correction 1), epsilon 1e-6 when normalization enabled; response mask | `advantages_from_rewards`: matching group mean/sample-SD convention for group size ≥2. Actual pipeline routing/group identity must also be checked. |
| GSPO-token surrogate | `core_algos.py:1037–1101`: masked mean log ratio, token log probability minus detached self plus detached sequence mean, upper log clamp 10, asymmetric clipping, sequence-mean/token-mean | Local `one_loss`: corresponding algebra for one unpadded response and equal weighting across completions; scalar tests do not establish native autograd parity. |
| Fixed old policy within rollout batch | `dp_actor.py:487–535`: split batch into minibatches, retain supplied old probabilities except the explicit single-minibatch/one-epoch shortcut | Local regenerates under current weights each group, then one update. No stale-probability defect at temperature 1, but different batch/update schedule. Do not overwrite old probabilities between minibatches of a future retained rollout block. |
| Persistent optimizer | `dp_actor.py:61`, `:494–596`: optimizer stored on actor and reused; per-minibatch `zero_grad` resets gradients, not moments | Local creates Adam per child. Minimal correction must preserve complete state and distinguish attempts from updates. |
| Optimizer recipe | `fsdp_workers.py:565–587`: AdamW, configured betas/weight decay and scheduler; shared YAML lr 1e-6, warmup 40; FSDP optimizer defaults weight decay .01 | Local Adam lr 1e-5, no explicit scheduler/weight decay and no global norm clipping. State repair alone is not optimizer-recipe parity. Make installed Adam defaults explicit and bound to receipts; preserve them for the isolated correction. |
| Gradient clipping | `dp_actor.py:336–366` clips global norm; `dp_actor.yaml:31` default 1.0 | Local computes norm and verifies it is finite/nonzero but does not clip it. A later faithful profile needs separately tested global clipping. |
| Sampler / exposure | shared YAML batch 256, six category weights; dataset sampler uses batch quotas | Local 4–16 prompt-group schedule, patient-within-category uniqueness, MRI routes and serial updates. Small quotas and active-group rejection can change effective category weights. |
| Rollout / lengths | shared YAML temperature 1, n=8; Qwen3-VL YAML max response 18432; reward manager max 16384/buffer2048 | Local strict path enforces temperature1, group2–8, cap16–512 and typically256. `direct_vero` preserves long reward thresholds, so its overlong term is inactive at these short caps. Do not call a scaled penalty the original recipe. |
| Open-ended retention | Public configuration includes judge-routed tasks and mixed categories | Current local MRI route set lacks scientifically admitted open-ended retention. Restoring optimizer state does not close this gap. |

Sources: [GSPO and advantages](https://github.com/zlab-princeton/vero/blob/025ca3c800fc95224c86323c1f7fd33dceed3090/vero-rl/verl/trainer/ppo/core_algos.py), [actor lifecycle](https://github.com/zlab-princeton/vero/blob/025ca3c800fc95224c86323c1f7fd33dceed3090/vero-rl/verl/workers/actor/dp_actor.py), [optimizer construction](https://github.com/zlab-princeton/vero/blob/025ca3c800fc95224c86323c1f7fd33dceed3090/vero-rl/verl/workers/fsdp_workers.py), [shared configuration](https://github.com/zlab-princeton/vero/blob/025ca3c800fc95224c86323c1f7fd33dceed3090/vero-rl/examples/model_runs/config/gspo_llmjudge_shared.yaml), [Qwen3-VL overrides](https://github.com/zlab-princeton/vero/blob/025ca3c800fc95224c86323c1f7fd33dceed3090/vero-rl/examples/model_runs/config/gspo_qwen3vl.yaml).

The public shared/default configuration is not proof of every released training launch's resolved overrides. Faithful reproduction must save the fully resolved configuration. Temperature is already locked to1 in the reviewed local path; adding another temperature later requires scaling logits consistently before both old/current likelihoods, not merely changing generation.

## 3. Minimal continuity correction design — no algorithm change

Prefer adding a versioned continuous RL driver while preserving the old smoke and multiarm historical semantics. Reuse `optimizer_checkpoint.py` backend-neutral serialization and the proven SFT state-binding patterns, without claiming that reuse alone validates RL integration.

### State owned by one run

- One model and one optimizer per live process, instantiated once outside the group loop. Preserve the current declared Adam hyperparameters for this repair, with explicit beta/epsilon/bias-correction settings instead of relying on hidden defaults. Constant LR is an explicit schedule type with no hidden scheduler.
- A fixed immutable train-only schedule with a cursor for **attempted prompt groups**, a separate **effective optimizer update count**, and raw rollout provenance for every attempted group, including skipped groups.
- Native MLX RNG state and any used Python/NumPy RNG state. A seed alone is insufficient. Restore RNG **after** model/optimizer initialization and reload checks, since setup may consume randomness. A deterministic per-group seed policy is a distinct explicitly defined procedure, not evidence of exact state restoration.
- Bind base/processor/template/source versions, frozen initial adapter, current adapter, complete moment inventory/dtypes, reward/parser/prompt contract, candidate schedule, precision, trainable parameter signature and optimizer configuration.

### Per-group transition

1. Verify current authorized boundary and resources; generate only the next predetermined training prompt's bounded group. Record actual token IDs/text, image/prompt hashes, current adapter identity, token counts and finish reason (unknown stays unknown).
2. Recompute strict rewards from saved raw text. Record format, task accuracy, overlong and total separately. If the diagnostic criterion fails, persist the attempted-group cursor and post-generation RNG, retain optimizer/adapter state exactly, increment **no** optimizer/scheduler update counter and record a skipped group. Do not call `optimizer.update` with a zero gradient: existing momentum/weight decay could still move weights.
3. For a qualifying diagnostic group, freeze old probabilities and advantages from the same policy/template/temperature/masks. Verify old/current parity before the first update, finite nonzero gradient, and exactly one planned optimizer update. Evaluate lazy arrays before serializing.
4. Commit an immutable boundary containing adapter, optimizer, RNG, cursor and digest-linked raw-rollout/reward receipt. Only advance the public progress pointer after the entire boundary has been validated and atomically committed. A crash between saving adapter and optimizer must leave an incomplete boundary that cannot be resumed as completed.
5. At a reviewed process boundary, stop/release the old model, load only one new model, verify adapter and moment tensors, restore RNG/cursor, then continue. A deterministic reload inference may consume RNG/cache state; restore the saved training RNG afterward and prove the next draw still matches. Never overlap two live heavy model instances.

Checkpoint at completed group/update boundaries only. Do not attempt partial-gradient or mid-rollout resume in the minimal version. Preserve interrupted raw receipts and orphan artifacts; do not overwrite/retry an already committed group. If a precommit attempt is replayed from its earlier complete boundary, record it as a restart and keep the original partial evidence.

### Later fidelity stage, separately scoped

After continuity passes, a faithful GSPO update schedule requires a small frozen rollout block generated before parameter changes, detached stored old probabilities for all block responses, and sequential minibatches/gradient accumulation with equal sequence weighting. Retain old probabilities when moving to a later minibatch; fresh likelihoods must reflect the changed model. Use CPU/tiny-model evidence to exercise nonunit ratios and clipping before any additional MRI model load. Merely repeating updates on one easy MRI group to make clipping activate is not a substitute for the prescribed rollout/minibatch design.

AdamW/warmup/global clipping, sampler semantics, short response caps, LoRA/quantization, batch scale and open-ended task support must each be recorded as matched or deviating. Full upstream hyperparameter-scale parity may remain compute-limited; call the local result a bounded recipe approximation with a fidelity table.

## 4. Decisive regression tests

Tests below are **proposed for the corrected driver**, unless explicitly listed as already executed. Use failing-first tests with backend injection; do not load MRI or large models.

| Test | Decisive failure / acceptance |
|---|---|
| Driver owns persistent optimizer | A fake optimizer counts constructions and retains moments across two active groups. Exactly one construction per live run; reset version fails. Test public driver behavior, not a detached serialization helper only. |
| Interrupted versus uninterrupted | Non-collinear gradients, active→inactive→active groups and an RNG-consuming reload fixture. Compare exact final parameters, moment state, update count, next group and next random draw. Intentional adapter-only restart must differ. |
| Inactive group is a true no-update | After an active group creates nonzero moments, inactive processing leaves adapter/moments/step/LR unchanged but advances attempt cursor and RNG. A zero-gradient optimizer call must fail this test. |
| Atomic boundary / wrong lineage | Inject failures after adapter save, after optimizer save and before commit; reject incomplete, mismatched-source, modified-adapter, missing-moment, wrong-shape/dtype, duplicate-group and counter-tampered boundaries before decoding arrays or updating. |
| Reward/eligibility decomposition | Full-format correct+incorrect with equal penalties qualifies; format-only/length-only variation, nonfinite values, constant task rewards or nontraining lineage do not. A rejected group's raw evidence remains in receipt. |
| Group identity and sample SD | Two different prompts with different reward means never share normalization. Compare known sample-SD results; epsilon behavior and groups of constant rewards are explicit. |
| Sequence-weighted gradient | Unequal response lengths and masked padding: equal sequence weight, no prompt/padding contribution; compare vector result against sum of per-response gradients. |
| GSPO detached-gradient parity | For fixed cached sequence ratios, verify positive/negative advantages inside/outside asymmetric clip; clamp boundaries; zero gradient only on the active clipped branch. Ordinary finite differences that recompute a stop-gradient value are not a complete autograd parity check. |
| Frozen old policy after an update | Two minibatches from one frozen rollout block; second minibatch current probabilities change while saved old probabilities do not. An implementation refreshing old probabilities must fail. This test belongs to the later fidelity stage. |
| No evaluation-driven sampler | Dev/reserved patient records or mismatched train/image hashes fail before model import; selection is frozen before generation and independent of evaluation outputs. |
| Resource/scope guard | Missing/revoked exact review digest or critical gate blocks model load/update; previous immutable boundary survives; only one live heavy worker. |

**Executed during this review:** existing `test_optimizer_checkpoint.py`, `test_strict_activation.py`, `test_vero_recipe.py`, `test_multiarm_reviewed_resume.py`: **33 passed, 1 native MLX test skipped**, with explicit import blocking for MLX/MLX-VLM/Torch/Transformers. The existing scalar GSPO public fixtures also remain preserved. No newly corrected RL driver exists, and none of the proposed driver acceptance tests is claimed passing. A separately scoped tiny native CPU test is still necessary to verify actual MLX autograd/state integration; it must not be confused with a full MRI model comparison.

## 5. Train-only informative-group diagnostic and stopping rule

The next strict diagnostic must use a checkpoint selected by the predeclared experiment (for example genuine completed B28 if that is the fixed initialization), **not the checkpoint or prompt that scores best on internal development**. Development answers, the historical eight test patients and the sealed prospective cohort cannot select rollout cases, rewards or response budgets.

Freeze training candidates by source/patient identity and a deterministic seed before new generation. Require image-linked eligible train records, hidden labels excluded from the actual rendered prompt, unchanged native system/assistant prefix and strict `direct_vero` reward. Do not admit the C cohort clinically merely because its geometry and hashes pass.

For a bounded **diagnostic**, retain the existing conservative task-informative rule:

- At least two completions for the same training prompt; finite rewards and nonempty token sequences.
- Full strict format for all completions, at least one correct and one incorrect task answer, identical overlong contributions; nonzero finite total advantages.
- Matching image/prompt/tokenization/temperature provenance; no target leakage. Preserve actual termination metadata; unknown termination is a limitation, not a reason to infer successful stopping.
- A finite predeclared candidate/group/token budget and zero adaptive replacements based on dev results. The existing maximum of three candidate groups is a possible bounded diagnostic limit, not authorization to launch. If none qualifies, save `NO_UPDATE_NO_TASK_VARIANCE`; do not relax format, reward correctness or extend rollouts indefinitely.

To demonstrate continuity on real MRI eventually requires two qualifying updates and a reviewed state transition, not merely two attempted groups. Failure to obtain two within the approved fixed attempt budget means that evidence remains incomplete; it does not permit cherry-picking unlimited attempts.

For the eventual **recipe-transfer comparison**, this all-valid mixed-correctness filter must either be removed in the direct-Vero arm or explicitly declared as an active-group ablation applied comparably across arms. Record attempted and active groups per category, actual generated/visual tokens, patient/image exposures, optimizer steps and effective category weights. Equal attempted-group budgets do not guarantee equal updates; equal update targets reached by rejection sampling do not guarantee equal rollout cost. Prespecify the primary budget estimand, cap both resources, and report the other quantities rather than claiming all budgets match. MedGRPO reward normalization is a separate control, not a justification for deleting zero-variance groups without accounting.

## 6. GO / HOLD / KILL

- **GO to a bounded corrected-driver diagnostic:** failing-first CPU integration tests and tiny native correctness evidence pass; an independent reviewer binds the precise implementation/configuration/checkpoint/budget; current resources pass. This document grants no training approval.
- **HOLD affected continuity experiment:** state mismatch, partial/corrupt checkpoint, fake/zero update, prompt/likelihood inconsistency, absent eligible training group or critical resource gate. Continue independent data/statistics work.
- **KILL a continuous-RL interpretation:** any second child silently starts Adam at step zero after a prior update, reuses a consumed group, or claims exact resume without matching actual restored state. Preserve those runs as explicitly labeled reset-optimizer pilots.
- **KILL a full-Vero-transfer claim from this pilot:** no eligible heterogeneous task support, no matched multiarm budgets, no open-ended retention assessment, no cross-task matrix, or improvements explained only by formatting/priors without independent patient evidence. These criteria limit claims; they do not halt the project.
