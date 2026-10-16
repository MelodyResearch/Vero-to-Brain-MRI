# Independent audit of attempted ds007045 GSPO arms

**Status:** `INFERENCE_COMPLETED` for 24 on-policy prompt groups and 48 image-conditional rollouts on local Metal; **`TRAINING_COMPLETED` does not apply to these three RL arms** because each made zero optimizer steps and retained identical adapter bytes. Audit date: 2026-10-15. The earlier four-step SFT weight update is separately documented in [the matched-precision audit](DS007045_INDEPENDENT_FROZEN_AUDIT.md). This audit reports only aggregate details; schedules, patient tokens, images, generations, and model weights remain under ignored `local_data/`.

## Receipt validation

I reread each arm-level receipt, all 24 step-level receipts, and the private schedules. Each schedule hash recomputed from its ordered record IDs; each step's input matched the corresponding schedule record. The shared internal training pool had **382 image-linked prompts from 28 training patients** (168 sequence, 84 any-tumor, 84 enhancing-tumor, 46 grid). All selected records were from the verified OpenNeuro `ds007045` **training** split, with eight distinct patient keys per arm and no held-out patient-token overlap. The common starting adapter SHA-256 matched the saved four-step SFT adapter: `39c5c48796a53bd82ef3184f54a266935b30f686af841b850bd981aa061ed8ca`. All three final-adapter hashes equal this same starting hash, with no later checkpoint update. The source/test-used flags were false.

Each arm used seed 7, `direct_vero` reward profile, the local short system prompt, eight attempted groups, group size two, and a 256-token maximum response. Step receipts confirmed one image-linked prompt per group, two generated responses, nonempty token IDs, aligned prompt tokens, and eight `NO_UPDATE_ZERO_VARIANCE` outcomes per arm. The direct-profile overlong window is 18,432 tokens with a 2,048-token buffer, so it did not penalize these ≤256-token responses. The current Apple MLX runner is an independent LoRA/GSPO-token engineering port; it is **not** an execution of the official CUDA veRL trainer or a full Vero recipe parity result.

| Attempted arm | Prompt groups by category: sequence / any tumor / enhancing / grid | Distinct training patients | Rollouts | Prompt tokens, once per group | Response tokens | Responses at 256-token cap | Groups with task-reward variance | Optimizer steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sequence specialist | 8 / 0 / 0 / 0 | 8 | 16 | 1,712 | 2,132 | 0 | 0/8 | **0** |
| Proportional mixture | 3 / 2 / 2 / 1 | 8 | 16 | 2,790 | 2,835 | 2 | 0/8 | **0** |
| Uniform mixture | 2 / 2 / 2 / 2 | 8 | 16 | 3,044 | 2,989 | 4 | 0/8 | **0** |

The group/rollout/response **caps** matched, but actual token work did not: prompt tokens per rollout were generated twice per group and differed by image/prompt type, and total response tokens rose from 2,132 to 2,989 across arms. These are bounded engineering attempts, not compute-matched scientific training arms.

## Why no RL update occurred

Every one of the **48 rollout rewards had task accuracy 0.0, format score 0.0, overlong penalty 0.0, and total reward 0.0**. Every two-response group therefore had zero standardized advantage and was skipped by the fail-closed optimizer. The MRI reward callback requires a complete `<think>...</think><answer>...</answer>` wrapper and then evaluates the boxed or plain final answer; zero format scores mean no rollout satisfied its strict wrapper gate. The receipts do not independently establish whether any unformatted response contained a semantically correct MRI answer. The direct-profile short prompt, the four-step SFT starting adapter, and 256-token cap did not elicit an active group in this bounded run.

This is a **reward-interface failure observed under one local prompt and short budget**, not evidence that uniform balancing helps or harms MRI reasoning. Changing the formatting prompt, adding a format warmstart, or loosening the verifier are distinct interventions that must be separately declared and rerun with identical arms and matched effective token/update budgets. A separate single-patient official-prompt probe appears below. The immediate technical target is a nonzero, valid task-reward group on training patients before any held-out cross-task transfer claim.

## Separate post hoc pinned-official-prompt diagnostic

After the three fixed arms, a **separate** train-only sequence diagnostic used the exact `system_prompt_chatting.txt` from the pinned official Vero source commit `025ca3c800fc95224c86323c1f7fd33dceed3090`. I rehashed that file in the local official checkout and matched its SHA-256 `3dc7c55d78ae580385b72b98d65317ceb2acc36874e4c00b4ad8ede787d6d7ad` to both the runner's pinned constant and the diagnostic receipt. Its three record IDs matched the private **training** manifest and came from **one** training patient across three images. The starting adapter hash was the same as the three arms above.

The diagnostic attempted **three prompt groups and six rollouts**, with group size two, direct reward and a **512-token response cap** (one response reached the cap). Prompt length was 680 tokens in each group; total response tokens were 2,087. All six accuracy, format and total rewards were still **0.0**; all advantages were zero; no group was active and **no optimizer step** occurred. Its status was `NO_UPDATE_ZERO_VARIANCE`; receipt SHA-256 is `8e00b167c4a76be9e552d71ef2f2ce0d957adbbe63a0a8c6ec008b3c96b3ea0d`. This bounded, one-patient, longer-budget follow-up shows that replacing the local short system prompt with the pinned official file **alone** did not solve the wrapper/reward bottleneck in this setup. It is not a fourth matched training arm, an official veRL run, or a patient-independent test.

## Evidence scope and next gate

### Separate exploratory single-box reward ablation

After the direct-profile failure, all three schedules were rerun from the same four-step SFT adapter with an explicitly post hoc `single_box_anywhere_ablation` reward route. This route can award task accuracy to a single `\boxed{...}` answer anywhere in the response, even when the Vero-style answer wrapper is absent; format reward remained zero. It is a diagnostic reward-interface change, **not** a direct Vero recipe replication. I checked all three arm receipts, all 24 step receipts, schedule and training-record links, before/after checkpoint SHA chains, and finite gradient/loss values at active steps. The eight selected patient tokens in each arm remained in the internal training split, with no held-out overlap. All arms used the same SFT adapter SHA-256 `39c5c48796a53bd82ef3184f54a266935b30f686af841b850bd981aa061ed8ca`, seed 7, eight attempted image-linked groups, two rollouts per group, and a 256-token response cap.

| Exploratory arm | Sequence / any tumor / enhancing / grid groups | Accuracy-positive rollouts | Reward-varying groups | Actual optimizer steps and updated task routes | Response tokens | Cap hits | Final adapter SHA-256 |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| Sequence specialist | 8 / 0 / 0 / 0 | 4/16 | 0/8 | 0; no update | 2,132 | 0 | Same as SFT start |
| Proportional mixture | 3 / 2 / 2 / 1 | 3/16 | 1/8 | 1; sequence | 2,719 | 2 | `0f17bd3ebe6ed81e0facec5b8de348b52efc57c7c677aacbc177ccd8076761b4` |
| Uniform mixture | 2 / 2 / 2 / 2 | 2/16 | 2/8 | 2; sequence, any tumor | 3,181 | 5 | `65714601f06eb745452b8337fbfaae97fa9b92f97aa1ab47b82dee70730bd8be` |

All 48 exploratory format scores and overlong penalties remained zero. The positive total rewards came only from task accuracy, at 0.8; each active group contained one positive and one zero task reward. The proportional arm had one nonzero finite gradient (L2 norm 0.402532); the uniform arm had two (0.321746 and 0.308344), and their adapter files changed. The specialist's four positive rollouts fell in groups without within-group reward variance, so it still made no update. Private arm-receipt SHA-256 digests (specialist, proportional, uniform) are `67dddec49ad4cb8a821f109fdcfffaf692831eabb73a6a0c65dbf2bed5ebeceb`, `c629311459b10915d933ea46f26a923a749358b541443914a0f3066bcba3acd2`, and `264a67ce0d0099e19179a9bb9fd2d93f99799aa436bafee78fead4608fa84fb0`.

These are genuine local LoRA optimizer updates in two arms. They do **not** isolate task balance: the arms have unequal active updates (0/1/2), unequal actual prompt/response token work, and different routes that received an update. A held-out evaluation of the two changed checkpoints may describe these particular runs, but cannot establish that uniform sampling caused a generalization benefit. The original direct-profile arms above still have no trained cross-task transfer matrix.

### Held-out evaluation of the two changed exploratory checkpoints

Both final adapters were evaluated on the same locked 107 image-linked prompts from **eight independent held-out patients**, with the same 256-token deterministic decoding and base chat template as the matched four-step SFT adapter. The read-only audit script required exactly the manifest's 107 unique records, verified prompt, image, and manifest hashes for every row, and matched each inference adapter SHA-256 to its saved final training checkpoint. Both arms also matched the same SFT starting adapter and adapter configuration. The scorer and patient/category aggregation were locked before these outputs were read.

| Checkpoint | Optimizer steps after SFT | Raw correct / 107 | Valid / 107 | Patient/category macro accuracy | Worst category patient mean | Paired macro difference from SFT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Matched SFT start | 0 | 54 | 101 | 0.447917 | 0.250000 | reference |
| Proportional exploratory RL | 1 | 54 | 101 | 0.447917 | 0.250000 | 0.000000 |
| Uniform exploratory RL | 2 | 53 | 100 | 0.432292 | 0.187500 | −0.015625 |

For **sequence** (48 prompts), **any-tumor presence** (24), and **enhancing-tumor presence** (24), all three checkpoints had identical *scored correctness and validity* on every held-out prompt. Their patient-mean accuracies were respectively 0.541667, 0.541667, and 0.458333. On **tumor grid** (11 prompts), SFT and proportional each had 4 correct / 5 valid, patient mean 0.250000; uniform had 3 correct / 4 valid, patient mean 0.187500. Paired discordance was exactly **one SFT/proportional-only correct grid item**, with zero uniform-only correct items across all categories. Proportional and uniform therefore differ on one scored grid item; this is not evidence of a systematic cross-task effect.

The uniform checkpoint's one-grid-item decrement occurred despite two task-reward-driven training updates. It is an observed score change for this post hoc run, **not** proof that uniform task sampling causes negative transfer: the proportional and uniform arms differed in active updates (1 versus 2), response tokens (2,719 versus 3,181), and which task routes updated. The held-out sample includes only eight patients from the same dataset source. There is no uncertainty estimate capable of supporting an MRI scientific claim from one discordant grid prompt. The intended direct-profile three-arm transfer experiment remains untrained and has no learned cross-task transfer matrix.

- **Scientific significance:** None yet for brain tumor diagnosis, anatomical understanding, or longitudinal MRI. The direct-profile arms have no trained RL checkpoint; the exploratory ablation has not established patient-independent benefit.
- **AI/ML significance:** The intended direct-profile arms exposed a reproducible all-zero reward bottleneck. The post hoc single-box route produced real task-driven optimizer steps in two arms. Neither result identifies a task-balancing effect, multi-task transfer, or negative transfer. No new RL algorithm is motivated by this result alone.
- **Experimental significance:** The direct-profile viability gate failed: 0/24 groups carried reward variance and 0 optimizer steps occurred. The exploratory route established nonzero task-reward gradients but left active updates and token work unequal. Next obtain valid-format and task-correct variance under the intended reward profile, then rerun arms from identical starting weights with matched effective rollouts, visual/response tokens, active groups, optimizer steps, and patient exposure. Evaluate every trained checkpoint on every held-out task, alongside same-data SFT and MedGRPO/active-group controls.

**Distinct completion conclusion:** The specialist, proportional and uniform schedules all completed their planned *rollout attempts*, but **none produced a trained RL checkpoint**. Consequently a Vero-to-MRI learned cross-task transfer matrix remains `NOT_RUN`; the identical final adapter SHA values are not three independent trained model outcomes.

Private arm-receipt SHA-256 digests (specialist, proportional, uniform respectively): `d7f1bc8cbacf2703e98c00796c65e877f155f8f13a56e8e94dca0d713bbc0122`, `6e95500c5ae53b1ea4513eee0a511882372eecb28ac82dfd027e42b40f0a61ea`, `41c76a8e9d3f53def850d337ad8d7d72f66e5c73caa05f8b75270394ddbf5a29`.
