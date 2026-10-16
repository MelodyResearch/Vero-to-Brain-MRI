# Training-only answer-likelihood recovery experiment

Tag: `EXPLORATORY_DIAGNOSTIC / B_LABEL_CONTROL`. Frozen before new model scores on 2026-10-16. This diagnoses the supervised starting interface for eventual Vero transfer; it is not another SFT run or a new algorithm.

## Fixed comparison

- Saved A0/A28/B28 adapters, identical 4-bit Qwen3-VL-8B base and rank4/alpha8 language adapters; no weight updates.
- All28 existing training patients, matched FLAIR/T1ce images at z45. Patient order is SHA256 of `7:mri_sequence:patient`, with FLAIR then T1ce. No score-based selection.
- One unique no-image prompt follows the first pair. Total57 unique inputs per checkpoint. A same-patient opposite-sequence swap and a deterministic next-patient opposite-sequence replacement are exact permutations of these already evaluated visible image/prompt inputs; derived controls must be labeled as such, never claimed as additional independent inference or patients.
- The primary unit is the28 patient/slice pairs. No-image is one common reference per checkpoint. Exact sequence-task exposure and any prior paired-panel pixel exposure are different facts; all z45 frames were seen in supervised panels, even when they lacked direct sequence supervision.
- z91/z136 are separately named potential diagnostics, not additions selected after viewing scores. Existing z91 panel exposure must be reported; z136 currently has no known A/B frame exposure. Neither is a new patient cohort.

## Likelihood estimands and alignment

Use the unchanged native system instruction and question. Append the common fixed target prefix `<think>Visible MRI evidence.</think><answer>\boxed{`, then either `FLAIR` or `contrast-enhanced T1`, then `}</answer>`. No generated reasoning enters the likelihood comparison. Full target segmentation must equal independently tokenized prefix + label + suffix; fail on a crossing token.

Primary length-aware score: mean log probability of label tokens only; margin = T1 score − FLAIR score. Report raw label log-probability sums and their margin separately. Also report first-label-token margin as a prespecified secondary boundary diagnostic. Mean token log probability addresses unequal token counts but is not a proper class posterior and may favor predictable multi-token completions. Raw sums retain length effects. Neither may be selected retrospectively as the winning metric.

Softmax over the two raw or mean scores supplies candidate-relative preferences and entropy in nats, not calibrated probabilities over all possible responses. Report disagreements between raw and mean rankings. A zero margin is a tie, not an arbitrarily correct class.

Save prefix-, answer-, suffix- and whole-target losses at the same frozen checkpoint. Closing-token loss is conditioned on the candidate label. Token counts and losses do not identify gradient dominance. The optional gradient-contribution experiment is separately scoped and NOT_RUN.

For every input, two candidate forwards are checked against a third prefix-only forward. Save both first-label-token probabilities at the common causal index, plus shared-prefix selected token scores. Maximum absolute log-probability disagreement must be≤0.002; this is a numerical consistency tolerance, not a scientific improvement margin. Check native expanded visual tokens/grid, pixels and token indices. A separate native greedy generation uses the same unprefilled assistant prompt and256-token limit.

## Interpretation and stopping

Report per-class margins, A0→A28 and A28→B28 changes, ground-truth signed margins, patient-pair separation `margin(T1 image)−margin(FLAIR image)`, real-minus-no-image shifts and directional preference changes. Pair substitutions use the displayed image's reference. Contrast hypotheses H1–H6 without choosing one beforehand. Training-only results cannot establish held-out generalization; compare with preserved development findings without retuning against them.

Initial scope: at most two unique real-image inputs at A0, six forwards and two bounded generations. Independent actual numerical/visual/provenance/resource review precedes expansion. Subsequent stages have explicit finite record counts and an independently supplied approval digest; no automatic expansion. One heavyweight model at a time, existing resource gates, fsynced incremental private outputs and post-release observations. A failed numerical or resource gate stops only the affected model probe.

Do not open reserved prospective patients, retrain B28, relax Vero rewards or infer a new optimizer from this diagnostic. Commit source/tests and safe aggregates only. Full specialism/proportional/uniform GSPO comparison remains the parent objective.

## Researcher amendment — bounded battery inference (2026-10-16)

The researcher explicitly authorized this read-only diagnostic on battery; substantial SFT/GSPO training remains AC-only. The opt-in `--battery-diagnostic` is bound into the independent approval and run identity. It retains the existing memory, concurrency and thermal checks, takes two initial samples at least 30 seconds apart, requires battery >=75% at stage start, and stops at <=65%. Additional conservative diagnostic limits are >=5 percentage points consumed per stage, >=2 points lost over >=60 seconds at >=1 point/minute, or >=1 GiB net swap growth per stage. Stops occur at computation boundaries; this cannot guarantee prevention of an OS failure. Every completed record is fsynced. No macOS power/protection settings change. Power in watts and unavailable thermal telemetry are not inferred from battery percentage. Initial scope remains two image inputs and zero weight updates; expansion needs actual numerical/resource review and a new exact-scope approval.

Mission alignment: `EXPLORATORY_DIAGNOSTIC`, `ENGINEERING_UNBLOCK`. This measures whether the B exposure intervention altered answer likelihoods before committing to further training, while retaining the original Vero-to-MRI objective. The battery exception changes resource admission only; MRI records, checkpoint identities, scoring, prompts and greedy budgets remain frozen.

### Loading correction after actual resource STOP

The first battery attempt exited before all forwards at quantized-model materialization: available RAM11.485→7.745 GiB, pressure-free64→38%, swap5.566→7.332 GiB, battery88%, MLX active4.818/peak6.845 GiB. Both existing swap stops fired. No result is inferred from this attempt and it is not repeated under the same loading configuration.

The separately versioned `serial_parameter_v2` implementation evaluates already defined parameter graphs one tensor at a time, clears cache/collects between leaves, and samples resources every64 leaves. It changes neither base/adapter values nor quantization specification, inference precision or resource thresholds. A tiny native CPU fixture checked nine quantized tensors and three quantized forward results for exact equality; full-model numerical/resource behavior remains an experiment. Loading now falls under cleanup coverage; the serial helper drops parameter references from retained exception frames (independently found failing weak-reference regression, then corrected). A new exact approval is required for this version.
