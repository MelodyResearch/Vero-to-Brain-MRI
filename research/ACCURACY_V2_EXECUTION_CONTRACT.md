# Accuracy V2: researcher directive and finite first experiments

The researcher directive supplied on 2026-10-17 makes demonstrably better image-grounded MRI accuracy the primary acceptance criterion. Update counts, training rewards and engineering checks are supporting evidence. The original eighteen RL arms and three SFT arms remain frozen controls; their configurations and negative results must be preserved.

## Separate V2 evidence

- Actual paired image-only baseline: fit only on the existing train split, with same-person/same-level sequence pairs and matched Dallas cycles. Expanded sequence supervision (168 images versus the original mixed-pool 28) is declared; classifier versus VLM accuracy differences are not a matched-budget algorithm gain.
- Actual read-only B28/U7 diagnostic: reuse the nine saved B28 records; complete only its missing null-correspondence record and U7's ten fixed records. Preserve original candidate values and the legacy failed oracle. A separately tagged shape control is permitted only for its reproduced case. Score substitutions using displayed-image truth. Report greedy accuracy separately from candidate selection and mean/raw/first-label margins.
- Frozen Qwen feature diagnostic: first extract mean merged visual features from the lexically selected four training sequence patients and all four fixed internal-development sequence patients, at three paired levels per patient (24 train and24 development images). Fit a fixed-alpha dual-ridge head and pixel/intensity/null controls on exactly those same training patients. Spatial token pooling and Dallas feature extraction are outside this first scope. A failed mean-feature probe cannot prove the entire visual encoder inadequate.
- Minimal image-paired supervised control: prepare a separately versioned, two-update language-LoRA pilot with opposite-label same-person/same-level training pairs and a no-update B28 baseline. Native launch requires the preceding actual evidence, focused numerical tests, one scoped independent review and existing hardware gates. No full vision unfreezing or V2 sweep.

## Acceptance and limitations

Report sequence accuracy, FLAIR/T1 recall, correspondence accuracy, same/different recall, confusion matrices, equal-task patient/cycle macro, worst-task accuracy and paired uncertainty. Prior shifts, image-substitution effects and shortcut controls are separate outcomes. A larger margin or more updates is not accuracy improvement. Nonspatial intensity success does not demonstrate anatomy reasoning. Dallas person correspondence is not biological-change reasoning.

The repeatedly examined development cohort is exploratory even when train/development patient keys are disjoint. Do not call its improvement stable independent generalization. Hyperparameters and input selection must be frozen before the corresponding outcome is inspected; no hard negatives from development or test.

Keep final100 sealed throughout V2 model development. Freeze the selected V2 method or predeclare a separate independent V2 holdout before final outcome inspection. Preserve the original exact22-checkpoint registry and release interface; any additional V2 evaluation requires its own explicit scope and cannot silently change that original registry.

## Execution

One heavyweight process only. Original training remains AC-only. A recoverable power-hold exit is resumed only from authenticated model, full optimizer, RNG, exact cursor and one-time partial-work accounting. A new diagnostic may use an actually idle slot, then release its model before original recovery proceeds. No unrelated apps or OS protections are changed. Preserve every prediction and checkpoint, including negative results. Claude general scientific critique is distinguished from code, private-data and numerical review; actual reviewer identities and access limits must be recorded.
