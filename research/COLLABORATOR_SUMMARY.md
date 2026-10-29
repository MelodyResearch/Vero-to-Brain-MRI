# Collaborator email — ready for the researcher to send

Subject: Vero-to-Brain-MRI — final results and reproducible research materials

Hi [Name],

I have completed a small empirical study of adapting Princeton Vero-style verifiable reinforcement learning and GSPO to brain MRI sequence recognition. The goal was to improve image-grounded accuracy while monitoring both FLAIR and contrast-enhanced T1 performance.

On our repeatedly explored development set of 24 images from four TRAIN-disjoint patients, the selected Adam52 parent achieved 18/24 (75.0%), with FLAIR recall 10/12 and enhanced-T1 recall 8/12. Matched Direct and Paired-Prompt GSPO endpoints each achieved 16/24 (66.7%), and a balanced paired-SFT control achieved 15/24 (62.5%). All four had zero invalid answers. None of these continuations improved on the parent; the SFT control particularly reduced T1 recall. We retain these negative results and do not attribute supervised training to GSPO.

The final materials and simplified source tree are now on main in the [private repository](https://github.com/MelodyResearch/Vero-to-Brain-MRI). The [paper draft](https://github.com/MelodyResearch/Vero-to-Brain-MRI/blob/main/research/FINAL_ACCURACY_PAPER_DRAFT_V1.md), [full results](https://github.com/MelodyResearch/Vero-to-Brain-MRI/blob/main/research/FINAL_ACCURACY_RESULTS_V1.md), [reproduction instructions](https://github.com/MelodyResearch/Vero-to-Brain-MRI/blob/main/REPRODUCIBILITY.md) and [model handoff](https://github.com/MelodyResearch/Vero-to-Brain-MRI/blob/main/research/FINAL_MODEL_HANDOFF_V1.md) are available to authorized repository collaborators. Results PR #6 and cleanup PR #7 are merged. Access must be granted separately; these links are not public, and no model weights or patient records are distributed through Git.

The earlier 22-model formal evaluation is reported separately and is not pooled with DEV24. An independent 16-patient evaluation has not run: its original input metadata has not been located despite a bounded recovery check, and explicit unsealing confirmation is still required. CPU reproduction verifies the saved metrics; actual inference additionally requires the authenticated private model/data assets and an approved execution scope. We make no clinical-performance or reliable-generalization claim.

I would welcome your feedback on the conclusions and limitations.

Best,
Melody

---
Ready-to-send draft only; no message sent, recipient selected, repository permission changed or private asset transferred.
