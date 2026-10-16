# Project guardrails

- For scientific literature audits, read `.agents/skills/norman-paper-reading/SKILL.md` and its linked `research/methodology/NORMAN_PAPER_READING_PROTOCOL.md` before writing. This is our independent project adaptation, not the original external Norman skill. Verify primary-source access level, exact paper version, claim-to-table/page evidence, and independent novelty; use `NOT_VERIFIED` or `SOURCE_MISMATCH` when warranted.

- Treat BrainTRACE as evaluation-only. Never train, tune rewards, select prompts or choose checkpoints on its test content.
- Do not commit medical images, patient/study identifiers, reports, restricted labels, model weights or credentials.
- Do not claim patient independence without a verified patient key and cross-source lineage check.
- Keep 2D frame IoU distinct from 3D Dice/IoU. A slice stack is not a native 3D encoder.
- Use status labels `IMPLEMENTED`, `TESTED_CPU`, `INFERENCE_COMPLETED`, `TRAINING_COMPLETED`, `SCIENTIFICALLY_VALIDATED`, `DATA_BLOCKED`, `GPU_BLOCKED`, `NOT_RUN` precisely.
- Do not claim an RL result unless model weights were updated and the run logs and checkpoint are verified.
