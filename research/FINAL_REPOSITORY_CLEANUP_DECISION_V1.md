# Evidence-preserving repository cleanup decision

2026-10-28. This PR is stacked on the actual-results PR, after the real training/development decision. **Deleted files:0. Moved files:0. Deleted worktrees/branches/model assets:0.** No safe deletion is manufactured to satisfy a quota.

[Machine inventory](../results/FINAL_REPOSITORY_CLEANUP_INVENTORY_V1.json) compares the exact main, selective-results and research-source Git trees. It checks identical blob payloads, empty tracked files and tracked cache/bytecode artifacts. Unique reports, checkpoints receipts, negative findings, source/tests and data contracts remain KEEP. README navigation is consolidated by the results PR, without deleting evidence.

`research/CURRENT_DEVELOPMENT.md` is a possible historical archive candidate on the source branch, absent the main/results trees. No move is performed: complete hash/reference dependency evidence is not established. It is not a current launch permission. Old branch guidance must be read with dated researcher decisions and actual results.

No `local_data/`, private runtime, historical weights, optimizer/RNG, raw predictions or ignored caches were cleaned. No worktree registration, branch ref or source interface was removed. This avoids repeating the earlier loss of non-Git runtime assets. Shared-rollout, Adam52/60 and historical negative evidence retain their existing storage and Git references.

## Review boundary

This is a cleanup classification and preservation PR, **not** a claim that code was pruned or a dependency reachability audit completed. If no file satisfies dependency/provenance/active-use/unique-evidence/recoverability/link checks, preservation is the actual safe outcome. Destructive pruning remains unperformed. The pending sealed16 decision is unaffected. No native computation was launched for cleanup.
