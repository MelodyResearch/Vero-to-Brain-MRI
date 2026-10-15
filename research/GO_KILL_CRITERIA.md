# Preregistered decision criteria

- **GO for baseline evaluation:** approved MRI images, verified QA links and at least two different verifiable categories with held-out subjects; fixed prompt/render/decoding and matching base–Vero checkpoint pair.
- **GO for empirical RL study:** same starting weights, valid subject splits, equal rollout and visual-token budgets, each single-task arm and proportional/uniform mixtures; patient-cluster intervals and all transfer cells reported.
- **GO for a new method claim:** a distinct technical operation improves independent-patient outcomes beyond Vero-style balance and a faithful MedGRPO normalization control, with a mechanism ablation.
- **REFORMULATE:** only 2D/multi-frame QA is accessible, longitudinal labels are not visually supportable, or method ingredients remain prior art without a differentiated result.
- **KILL a specific claim:** leakage, unverifiable image lineage, shortcut-maximizable rewards, absent model weight updates, missing matched controls, or a hypothesized benefit whose patient-cluster interval fails the preregistered margin.
