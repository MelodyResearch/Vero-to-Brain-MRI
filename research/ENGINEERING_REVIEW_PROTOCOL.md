# Independent engineering review before substantial MRI training

The project lead (Codex) implements data, prompts, rewards, SFT/GSPO, model runs and receipts. A separate Claude Code review is required for substantial new training **when the exact code/data payload is permitted by the approval controls**. Review code and raw permitted receipts before the lead's preferred diagnosis wherever possible. Never interpret a code-review pass as validation of MRI scientific generalization.

## Required cycle

1. Codex freezes a versioned experiment contract and produces the source diff, tests, deidentified or permitted data-lineage audit, exact model/adapter hashes, a bounded real-image rollout receipt, and resource sample.
2. Independent reviewer checks image/label and patient integrity; SFT/inference/RL prompt and token boundaries; strict parser/reward decomposition; GSPO old/current log probabilities, advantages and gradient flow; checkpoint reload; fair budgets; and visual shortcut controls. Record a concrete failure example or unverified point.
3. Codex fixes verified defects with a failing-then-passing regression test. The reviewer verifies the corrected artifact, rather than approving a summary of the fix.
4. The review verdict is one of `PASS`, `PASS_WITH_LIMITATIONS`, `FAIL`, or `NOT_VERIFIED`, scoped to a named contract and component. `FAIL` blocks the affected training job. `NOT_VERIFIED` cannot authorize a substantial job. A bounded diagnostic may run to obtain the missing evidence if its data/resource gates pass and its result is labeled exploratory.
5. A substantial SFT/RL launch requires the reviewed contract, static data/prompt/reward checks, an actual image-conditioned group with task-reward variance, finite task-related gradients, changed adapter bytes and checkpoint reload, plus adequate AC power and memory. If one gate fails, continue other CPU/data/scientific work.

## Sharing and reviewer identity

A prior attempt to send the unpublished ds007045 design to third-party Claude was **rejected by automatic approval review** before execution. Do not retry that denied payload through a different route or attribute any result to Claude. An isolated Codex agent can perform an accurately labeled interim review, but the Claude verdict remains `NOT_VERIFIED` for that payload until an allowed review channel is available. Source code, aggregate findings and non-patient test fixtures may be considered separately only through applicable approval controls; denial of a specific payload is not permission to disguise it.

On 2026-10-15 this task's `claude auth status` returned `loggedIn: false`. The researcher had reported a sign-in earlier, but this CLI environment cannot currently authenticate Claude Code. No new Claude request was run in this cycle. Authentication and the payload approval must both be resolved before attributing a new verdict to Claude.

## Current audit ledger

| Component/configuration | Independent reviewer | Verdict | Evidence / limit |
| --- | --- | --- | --- |
| SFT28 first user-only internal-dev probe versus matched system instruction | Isolated Codex engineering agent | `PASS_WITH_LIMITATIONS` | Paired receipts show 0/54 strict format for user-only and 54/54 for matched instruction with the same final adapter; this isolates the prompt mismatch for the probe. See `results/INDEPENDENT_CONTRACT_REVIEW.md`. Does not explain the earlier SFT4 direct-RL zero groups. |
| Pinned Vero reward parser and MLX GSPO formula | Independent Codex RL agent | `PASS_WITH_LIMITATIONS` for CPU parser golden cases; `NOT_VERIFIED` for numerical optimizer parity | Nested-box defect fixed and 15 focused tests pass. Official CUDA/full-parameter behavior differs from local Metal/LoRA; no cross-framework gradient parity. See `results/VERO_FORMAT_AND_REWARD_CONTRACT_AUDIT.md`. |
| ds007045 image/label/split engineering integrity | Independent Codex MRI agent | `PASS_WITH_LIMITATIONS` | Render/mask arithmetic and 28/4/8 patient split verified; strong axial-level shortcut and tiny-mask/grid ambiguity limit visual/scientific validity. See `data/MRI_VISUAL_ANSWERABILITY_AND_SHORTCUT_AUDIT.md`. |
| Unpublished ds007045 review by Claude Code | Claude Code | `NOT_VERIFIED` | Automatic approval review rejected sharing the earlier new-dataset payload before Claude ran. No Claude verdict exists for this configuration. |
| Substantial multi-task GSPO comparison | None | `NOT_VERIFIED` | Valid strict answers now exist after SFT28, but live task-reward variance, optimizer/checkpoint reload, full data answerability and independent approval are not yet established. |

Historical results are immutable. A repaired prompt does not retroactively turn prior zero-update arms into weight-updating experiments or the explored eight-patient test into a prospective test.
