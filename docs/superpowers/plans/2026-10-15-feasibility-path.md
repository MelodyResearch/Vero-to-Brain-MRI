# MRI feasibility and first comparison plan

1. Audit official dataset files, rights, source image links and patient grouping. Store only aggregate access facts in Git.
2. Implement strict linked-image JSONL audit, frame rendering, NIfTI geometry, task verifiers and a patient-level metric. Test on clearly synthetic engineering fixtures.
3. Obtain approved real MRI image access and create a local-only lineage manifest. Run decoding, volume/frame, bounding-box and split checks; report exact usable counts.
4. Pin matching Qwen3-VL and Vero checkpoint revisions and Vero code commit. Freeze evidence and generation settings. Run paired inference and patient-level comparison.
5. Only if steps 3–4 support it, implement Vero-compatible online RL updates and train matched-budget single-task and mixture arms. Log actual weight changes, token cost and all transfer cells.
6. Hold BrainTRACE test untouched until MR-RATE access, source overlap audit and a frozen protocol. Then use its official renderer/scorer once.

Step 1–2 are CPU feasibility work. Step 3 is currently data-blocked. Steps 4–6 are `NOT_RUN`. The project request itself supplied the design intent and explicit instruction to execute; no additional design approval was needed to do local reversible work.
