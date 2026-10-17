# Independent engineering review and scoped execution

Policy version: `mri-execution-tiers-v1`. **The researcher-designated lead, Claude Code or Codex, implements and executes.** A distinct reviewer inspects source and permitted raw evidence before seeing the lead's preferred diagnosis where feasible. Claude may act as execution lead or independent reviewer when currently authenticated and legitimately authorized for the exact materials. The researcher's 2026-10-15 correction explicitly permits an isolated Codex reviewer for local work when Claude is unavailable. A review pass never establishes MRI scientific generalization.

## Execution tiers

| Tier | Scope | Required evidence and approval |
| --- | --- | --- |
| A | CPU tests, image/data audits, reward replay, statistics | Reproducible automated static checks. No Claude requirement; battery power does not block CPU work. The Tier A result never authorizes a model launcher. |
| B diagnostic | First bounded real-image model diagnostic | Exact static data/prompt/reward contract, positive and negative parser tests, safe AC/memory/thermal preflight, separate approval bound to current code/config/starting adapter/budget. At most one optimizer update, three groups, eight total rollouts and 512 response tokens per rollout. A previous update receipt is intentionally not required. |
| B continuation | Bounded continuation after a diagnostic | All B gates plus independently reviewed raw on-policy generations, strict-format correct/incorrect answers within the selected group, task accuracy variation, finite nonzero gradient, changed adapter bytes and fresh reload inference. At most three updates, three groups, eight total rollouts and 512 tokens. |
| C | Substantial or confirmatory training | Reproducible dynamic evidence plus explicit independent authorization for the exact finite budget, matched comparison checks, acceptance criteria, resources and patient/data validity. No unconditional Claude-only block; use a permitted explicitly authorized independent reviewer and accurately record identity. |

The initial B diagnostic must enforce task variation and finite gradients internally before updating, then verify actual changed weights and checkpoint reload. A zero-variance or nonfinite-gradient result authorizes no expansion. Strict format alone and length-only reward differences do not satisfy task-reward variation. Increase generation diversity or change training-only eligible tasks only with recorded justification and a newly bound review.

## Required cycle and approval provenance

1. Codex freezes the contract, source, run configuration, output destination, exact starting adapter and finite budget, then runs preliminary CPU tests.
2. A separate reviewer directly inspects those artifacts, identifies defects and scientific limits, and writes an approval under ignored `local_data`. Its verdict is `PASS`, `PASS_WITH_LIMITATIONS`, `FAIL`, or `NOT_VERIFIED`. Limited approval names the precise limits. `FAIL` and `NOT_VERIFIED` cannot authorize affected model work.
3. Codex fixes verified defects and adds meaningful regression cases; reviewer verifies corrections. Disagreements are retained and resolved through reproducible tests.
4. The reviewer independently sends the SHA-256 of their authored approval through the orchestration message. The launcher receives that expected digest separately from the candidate file. The approval binds `review_scope(...)`: execution policy, contract ID, source hashes, generation config, tier/phase, budget, launch configuration, starting adapter and (for continuation) dynamic receipt hash. Changes invalidate approval.
5. Launch rechecks actual static data/prompt/reward/resource gates and, for expansion, recomputes strict raw-output rewards, selected-group task variation, adapter bytes and checkpoint reload linkage before loading a model.

This is a trusted orchestration boundary, not cryptographic reviewer identity against a process controlled by the same OS user. A reviewer-name field alone is insufficient; an implementer computing a digest from their own purported approval is not independent review. Unit fixtures exercise acceptance/rejection branches and are never scientific receipts. Unrelated CPU/data work continues if a model gate fails.

## Sharing and historical reviewer status

An earlier automatic approval review rejected one specific attempted outbound disclosure of **unpublished ds007045 research material** to Claude before that run executed. It is **not** a general prohibition against accessing genuinely public OpenNeuro ds007045 data, participating as a newly authorized Claude Code lead, or obtaining legitimate renewed approval through the normal mechanism. For each proposed private payload or transfer, verify current Claude authentication and applicable approvals; do not circumvent an active denial, conceal rejected content, or attribute a nonexistent review to Claude. The earlier `loggedIn: false` result was local to that prior task environment; a later authenticated Claude public-source review exists. Keep exact scope/provenance. Where access is not granted, report the particular blocker and continue all other authorized local B/C work.

## Current audit ledger

| Component/configuration | Independent reviewer | Verdict | Evidence / limit |
| --- | --- | --- | --- |
| SFT28 first user-only internal-dev probe versus matched system instruction | Isolated Codex engineering agent | `PASS_WITH_LIMITATIONS` | Paired receipts show 0/54 strict format for user-only and 54/54 for matched instruction with the same final adapter; this isolates the prompt mismatch for the probe. See `results/INDEPENDENT_CONTRACT_REVIEW.md`. Does not explain the earlier SFT4 direct-RL zero groups. |
| Pinned Vero reward parser and MLX GSPO formula | Independent Codex RL agent | `PASS_WITH_LIMITATIONS` for CPU parser golden cases; `NOT_VERIFIED` for numerical optimizer parity | Nested-box defect fixed and 15 focused tests pass. Official CUDA/full-parameter behavior differs from local Metal/LoRA; no cross-framework gradient parity. See `results/VERO_FORMAT_AND_REWARD_CONTRACT_AUDIT.md`. |
| ds007045 image/label/split engineering integrity | Independent Codex MRI agent | `PASS_WITH_LIMITATIONS` | Render/mask arithmetic and 28/4/8 patient split verified; strong axial-level shortcut and tiny-mask/grid ambiguity limit visual/scientific validity. See `data/MRI_VISUAL_ANSWERABILITY_AND_SHORTCUT_AUDIT.md`. |
| Unpublished ds007045 review by Claude Code | Claude Code | `NOT_VERIFIED` | Automatic approval review rejected sharing the earlier new-dataset payload before Claude ran. No Claude verdict exists for this configuration. |
| Substantial multi-task GSPO comparison | None | `NOT_VERIFIED` | Valid strict answers now exist after SFT28, but live task-reward variance, optimizer/checkpoint reload, full data answerability and independent approval are not yet established. |

Historical results are immutable. A repaired prompt does not retroactively turn prior zero-update arms into weight-updating experiments or the explored eight-patient test into a prospective test.
