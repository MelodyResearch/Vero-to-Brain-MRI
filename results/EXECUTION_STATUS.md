# Execution record — 2026-10-15

## Verified local execution

- **IMPLEMENTED, TESTED_CPU:** `PYTHONPATH=src pytest -q` returned `10 passed in 0.11s` (Python 3.13.6, macOS Apple M3 Pro). It tests invalid and matching frame IoU, reward parsing, image decoding and split overlap, box bounds, render ordering, NIfTI RAS orientation/affine roundtrip, balanced sampler, budget validation and patient-unit paired comparison.
- **TESTED_CPU engineering smoke:** `mri_rl.cli audit` decoded two generated 8×8 PNG *engineering fixtures* with distinct dummy subjects; `mri_rl.cli score` scored one supplied dummy prediction. Both commands exited zero. Fixture outputs were removed before commit and have no scientific meaning.
- `torch.backends.mps.is_available()` and `torch.cuda.is_available()` both returned `False` in this environment. Machine is Apple M3 Pro with 36 GB memory. No GPU endpoint or VLM training runtime was identified. Vero's documented training configuration uses 8×80GB A100/H100; that is an upstream reference, not a measured requirement for this reduced experiment.
- Shell `curl` to Hugging Face failed with `Could not resolve host`; no dataset files were downloaded by shell. Source audit used browsable primary web pages. Locally verified genuine MRI images/subjects/studies/volumes/tasks: **0/0/0/0/0**.
- Claude Code 2.1.301 is installed but `claude auth status` returned `loggedIn: false`; Claude did not participate.

## Unexecuted science

`INFERENCE_COMPLETED`: none. `TRAINING_COMPLETED`: none. `SCIENTIFICALLY_VALIDATED`: none. Qwen3-VL versus Vero, SGMRI MRI evaluation, single-task RL, balanced RL, MedGRPO control, transfer matrix and BrainTRACE longitudinal generalization are all **NOT_RUN**. Lack of legitimate image-linked MRI data is `DATA_BLOCKED`; large-model execution is `GPU_BLOCKED`. The public benchmark's publisher counts in `DATASET_MANIFEST.json` are not local usable counts.

## Repository publication

The local Git repository is initialized independently. Private GitHub creation and remote push remain blocked by automatic approval review of the final browser action. `gh auth status` also reported an invalid local CLI token. No GitHub repository URL or remote parity can be claimed until explicit approval and authenticated push succeed.
