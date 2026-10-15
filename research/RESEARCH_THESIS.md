# Thesis and present decision

**Question:** On genuine, patient-separated brain MRI, does task-balanced online verifiable RL improve a task-to-task transfer matrix relative to single-task and proportional-mixture RL, while preserving spatial grounding and explanation quality?

Vero demonstrates multi-category balance and negative-transfer patterns on general visual data. Med-R1, BrReMark, LoV3D, MedVol-R1 and MedGRPO already cover major ingredients. The possible contribution is a controlled empirical transfer result in MRI, or a genuinely distinct operation motivated by a reproduced failure of existing methods. Neither is established here.

**Current recommendation: REFORMULATE until Gate A.** Preserve the longitudinal objective, but first secure and audit legitimately usable MRI images plus patient identities. The immediate scientific experiment is paired Qwen3-VL versus matching Vero inference on two real MRI task families with identical evidence and inference budgets. This measures pretrained transfer, not method transfer. A task-mixing RL claim requires subsequent actual weight updates under matched compute.

**Decision gates:** Gate A, eligible real images and patient-separated task labels, is blocked. Gates B–F are `NOT_RUN`. Do not infer a longitudinal result from multi-frame SGMRI-VQA; its frames are within a volume rather than visits.
