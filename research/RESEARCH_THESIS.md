# Thesis and present decision

**Question:** On genuine, patient-separated brain MRI, does task-balanced online verifiable RL improve a task-to-task transfer matrix relative to single-task and proportional-mixture RL, while preserving spatial grounding and explanation quality?

Vero demonstrates multi-category balance and negative-transfer patterns on general visual data. Med-R1, BrReMark, LoV3D, MedVol-R1 and MedGRPO already cover major ingredients. The possible contribution is a controlled empirical transfer result in MRI, or a genuinely distinct operation motivated by a reproduced failure of existing methods. Neither is established here.

**Current recommendation: GO for real-image feasibility; REFORMULATE the scientific claim.** The Dallas pilot now has authentic paired T1 and FLAIR images for two subjects. Expand to a patient-separated cohort with image-visible, validated targets and no-image controls before paired Qwen3-VL versus Vero inference. That comparison would measure pretrained transfer, not method transfer. A task-mixing RL claim requires subsequent actual weight updates under matched compute.

**Decision gates:** The real-image access and lineage portion of Gate A is met for the eight-volume Dallas pilot; eligible task labels and a patient-separated evaluation set remain blocked. Gates B–F are `NOT_RUN`. Do not infer a longitudinal result from multi-frame SGMRI-VQA; its frames are within a volume rather than visits.
