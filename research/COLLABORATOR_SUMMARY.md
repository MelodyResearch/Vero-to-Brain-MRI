# Collaborator summary — draft for the researcher to send

Subject: Vero-to-Brain-MRI — final empirical results and reproducibility

Hi,

I have completed a bounded study of adapting Vero-style verifiable reinforcement learning to brain MRI sequence recognition. On our repeatedly explored, TRAIN-disjoint development set (24 images from four people), the retained Adam52 parent scored75.0%; matched Direct and Paired-Prompt GSPO endpoints each scored66.7%, and a subsequent balanced paired-SFT control scored62.5%. None demonstrated an improvement over the parent. We report all class recalls and negative results, without pooling this development set with our earlier22-model formal evaluation.

The final source, result-reproduction entry, model identity/configuration hashes, paper draft and evidence index are in the private repository: [Vero-to-Brain-MRI](https://github.com/MelodyResearch/Vero-to-Brain-MRI). [Results PR](https://github.com/MelodyResearch/Vero-to-Brain-MRI/pull/6) and [cleanup PR](https://github.com/MelodyResearch/Vero-to-Brain-MRI/pull/7) are drafts for review, not merged releases. Repository access is required; no public share link or weight distribution is implied.

The independent16-person evaluation has not run: original execution metadata needs to be located and verified, and the checkpoint/protocol require my explicit unsealing confirmation. The project does not claim clinical diagnostic performance, independent generalization or a successful new RL method. CPU reproduction of the saved metrics is available; native inference also needs the authorized private model/data assets.

I would welcome feedback on the empirical conclusions and limitations.

Best,
Melody

---
Draft only. Not sent, and no repository permissions changed.
