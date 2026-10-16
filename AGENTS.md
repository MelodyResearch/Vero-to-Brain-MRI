# Project guardrails

## Independent training-interface review

For substantial new SFT or RL jobs, follow [research/ENGINEERING_REVIEW_PROTOCOL.md](research/ENGINEERING_REVIEW_PROTOCOL.md): Codex implements; Claude Code independently verifies only when the exact review material is permitted by approval controls. Record a scoped `PASS`, `PASS_WITH_LIMITATIONS`, `FAIL`, or `NOT_VERIFIED` verdict. A previous attempt to share unpublished ds007045 material with Claude was denied by automatic approval review; do not bypass it or claim Claude reviewed this configuration. An isolated Codex review is explicitly labeled interim and does not make a scientifically invalid experiment valid. Require real-image reward variance, task-related gradient/weight update, checkpoint reload, patient/data checks and resource readiness before a substantial launch. A failed gate blocks the affected job while independent work continues.

## Headless public-source access — standing researcher instructions

The researcher authorizes **routine noninteractive research operations** using existing repository access and legitimately public data: GitHub API/search/clone/fetch, arXiv/publisher open paper downloads, Hugging Face public repository/dataset APIs, OpenNeuro public downloads, authorized public object stores, and standard CLI tools (`git`, `gh`, `curl`, `wget`, `python`, `huggingface_hub`, DataLad/git-annex when needed). Conduct these **headlessly**, without opening the researcher's desktop browser, taking over the screen, waiting for clicks, or asking approval for each public search/download/normal commit/push to this existing PRIVATE repo. Preserve rate limits, provenance, exact hashes, licenses and disk/bandwidth prudence. Publicly downloadable MRI images are not automatically restricted merely because they are MRI; verify actual source-specific terms rather than assuming all images require a DUA.

Use APIs and official download endpoints to fetch **actual MRI payload bytes**, not Git-annex pointers, dataset cards or synthetic substitutes. Retrieve in bounded stages; first verify images, labels and lineage; keep large image payloads, source-linked subject records, checkpoints and credentials out of Git (even for public datasets) pending explicit redistribution review.

Do **not** bypass paywalls, gated datasets, sign-in or DUA requirements, autoaccept a new binding agreement on the researcher's behalf, make payments, provision paid GPUs, change account permissions, or change visibility to public. Use already authorized noninteractive credentials where they apply; if a specific source needs new interactive authorization, mark only that source BLOCKED and continue with other public sources rather than opening a website or ending the project. Report a user action only when truly unavoidable.

This is authority for routine project work, **not** a claim that the assistant can run indefinitely after a session ends or that GitHub itself contains all MRI image bytes.

## Primary research objective: systematic Vero-to-MRI training-recipe transfer

**Do not drift into a standalone longitudinal-change-verifier project.** The source of truth is [research/VERO_TO_MRI_PRIMARY_EXECUTION_CONTRACT.md](research/VERO_TO_MRI_PRIMARY_EXECUTION_CONTRACT.md). The top-level question is whether *full Vero-style task taxonomy, category balancing, task-routed verifiable rewards and actual GSPO training* systematically transfer to real brain MRI and preserve cross-task capability relative to same-data SFT, single-task RL, proportional mixing and MedGRPO/active-group controls. Longitudinal MRI is an important category **only when its scientific labels are eligible**; its blockage does not block other genuine image-grounded MRI tasks. Do not call model checkpoint inference 'method transfer' or CPU reward replay 'RL training'. The full 13-paper audit is already complete; use the Norman skill for targeted source questions, not repeated review cycles. Keep the existing data, evidence and negative results intact.

- For scientific literature audits, read `.agents/skills/norman-paper-reading/SKILL.md` and its linked `research/methodology/NORMAN_PAPER_READING_PROTOCOL.md` before writing. This is our independent project adaptation, not the original external Norman skill. Verify primary-source access level, exact paper version, claim-to-table/page evidence, and independent novelty; use `NOT_VERIFIED` or `SOURCE_MISMATCH` when warranted.

- Treat BrainTRACE as evaluation-only. Never train, tune rewards, select prompts or choose checkpoints on its test content.
- Do not commit medical images, patient/study identifiers, reports, restricted labels, model weights or credentials.
- Do not claim patient independence without a verified patient key and cross-source lineage check.
- Keep 2D frame IoU distinct from 3D Dice/IoU. A slice stack is not a native 3D encoder.
- Use status labels `IMPLEMENTED`, `TESTED_CPU`, `INFERENCE_COMPLETED`, `TRAINING_COMPLETED`, `SCIENTIFICALLY_VALIDATED`, `DATA_BLOCKED`, `GPU_BLOCKED`, `NOT_RUN` precisely.
- Do not claim an RL result unless model weights were updated and the run logs and checkpoint are verified.
