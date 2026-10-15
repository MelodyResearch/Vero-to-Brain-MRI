# Falsifiable hypotheses

- **H1, interference:** A single-task MRI RL checkpoint changes held-out performance on another supported task by a predeclared meaningful margin, with patient-level uncertainty excluding no effect. Report every off-diagonal cell and training-seed variation.
- **H2, balanced training:** At matched starts, image evidence and total rollout/token/optimizer budget, uniform task mixing improves equal-category macro score and worst-category score versus task-size proportional mixing, without a grounding or hallucination regression. A larger data pool alone is not evidence.
- **H3, reward design:** A specified verifier/calibration change improves stable effective updates and patient-held-out outcomes beyond a faithful MedGRPO-style control. Positive affine scaling in within-task group normalization is not presumed to change rankings.

These are hypotheses. All results are `NOT_RUN` while real linked MRI data and training compute are absent.
