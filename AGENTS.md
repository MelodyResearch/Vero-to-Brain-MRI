# Project guardrails

- Treat BrainTRACE as evaluation-only. Never train, tune rewards, select prompts or choose checkpoints on its test content.
- Do not commit medical images, patient/study identifiers, reports, restricted labels, model weights or credentials.
- Do not claim patient independence without a verified patient key and cross-source lineage check.
- Keep 2D frame IoU distinct from 3D Dice/IoU. A slice stack is not a native 3D encoder.
- Use status labels `IMPLEMENTED`, `TESTED_CPU`, `INFERENCE_COMPLETED`, `TRAINING_COMPLETED`, `SCIENTIFICALLY_VALIDATED`, `DATA_BLOCKED`, `GPU_BLOCKED`, `NOT_RUN` precisely.
- Do not claim an RL result unless model weights were updated and the run logs and checkpoint are verified.
