# Patient-keyed ds007045 interface for Vero-to-MRI transfer

Status (2026-10-15): **real MRI training records converted and official launcher dry-run validated; no ds007045 model weights updated yet.** This interface uses OpenNeuro ds007045 v2.0.1 MRI and official tumor masks. It is separate from the earlier MSD engineering smoke, whose one real LoRA update was driven only by an overlong penalty.

## Patient split and task labels

The private prompt pack contains 40 BIDS patients, with **32 source-designated train** and **8 source-designated test** patients. [`ds007045_train_adapter.py`](../src/mri_rl/ds007045_train_adapter.py) chooses four internal dev patients from the 32 training patients by a label-blind SHA-256 ordering of anonymous patient tokens. It writes **382 internal train** and **54 internal dev** records from 28 and 4 disjoint patients. The eight heldout patients and their labels are never written to either veRL file or specialist files. The heldout pack is reserved for final evaluation; internal dev is for engineering/tuning. Data, patient keys and rendered images remain under ignored `local_data/ds007045_bounded/`.

The four routes are: source sequence identity (FLAIR versus contrast-enhanced T1, independently labeled from image modality), slice tumor presence (yes/no), enhancing tumor presence (yes/no), and tumor location in a 3×3 image grid. The latter three derive from the official mask at fixed geometric slices. The source sequence task can be solved from acquisition/appearance cues and should be interpreted separately from tumor reasoning. Every route passes through [`vero_mri_reward.py`](../src/mri_rl/vero_mri_reward.py), preserving the pinned Vero format gate: an untagged boxed answer receives zero task accuracy. The rewards are programmatically verifiable against labels not included in the prompt image.

| Route | Internal train rows | Internal dev rows | Expected official batch-256 draws, uniform | Proportional |
|---|---:|---:|---:|---:|
| Sequence | 168 | 24 | 64 | 113 |
| Tumor presence | 84 | 12 | 64 | 56 |
| Enhancing presence | 84 | 12 | 64 | 56 |
| Tumor grid | 46 | 6 | 64 | 31 |

Uniform sampling draws **1.39 tumor-grid rows per available grid record per batch**, versus 0.674 under proportional sampling. All 28 internal train patients have at least one grid-positive slice, but each has only one or two grid rows. Uniform thus gives **1.39–2.78 expected grid draws per patient per batch**; proportional gives **0.674–1.35**. A grid specialist at official batch 256 gives **5.57–11.13** draws per grid-positive patient. These are category-sampler expectations, not unique patient counts or executed SGD exposures. Repeated grid-positive patient exposure could mimic a task-balancing benefit, so future evaluation must report both nominal category draws and unique patient exposure.

## Matched training arms and status

[`vero_official_launch.py`](../src/mri_rl/vero_official_launch.py) accepts `--profile ds007045` and validates a uniform mixture, a proportional mixture, and each category specialist against the pinned official Vero source SHA `025ca3c800fc95224c86323c1f7fd33dceed3090`. Each specialist has category-specific train and internal-dev files; all arms use the same source checkpoint path and reward callback. The launcher dry-run passed for uniform, proportional and tumor-grid specialist. It cannot execute the official veRL CUDA trainer on this Apple Metal host. These are prepared configs, **not completed RL comparisons**.

[`mlx_sft_warmstart.py`](../src/mri_rl/mlx_sft_warmstart.py) is a separate 4-bit rank-4 LoRA supervised control prepared for **1–8 optimizer steps**, each with one teacher-forced, image-conditioned example per category. A seed-fixed schedule selects a different training patient per category at each step, logs the actual patient exposure privately, and saves before/after adapters only if weights change. It is CPU-checked but **has not been run on Metal**. [`mlx_gspo_smoke.py`](../src/mri_rl/mlx_gspo_smoke.py) accepts verified ds007045 train records and an optional starting LoRA checkpoint.

[`mlx_multiarm_train.py`](../src/mri_rl/mlx_multiarm_train.py) prepares a deterministic **4–16 prompt-group** specialist, proportional, or uniform schedule before generation. At eight groups, the exact category-group counts are **8 grid** for a grid specialist, **3/2/2/1** for proportional, and **2/2/2/2** for uniform (in sequence/tumor-presence/enhancing-presence/grid order). Within each category, it selects distinct patients and records every zero-advantage group; each group runs in a fresh MLX process and can update from the exact preceding adapter. All arms require the **same starting adapter SHA-256**. The MLX path remains an independent small-budget implementation, not official Vero parity. A successful group update counts as a task-learning diagnostic only if its advantage includes **nonzero task accuracy or valid-format reward variance**, rather than solely an overlong penalty.

The first controlled sequence after heldout frozen inference is: save one common base 4-bit LoRA state; run the SFT control separately; launch specialist, proportional and uniform GSPO arms from an identical chosen state; record each prompt and group reward including accuracy, format and overlong components; score only patient-heldout predictions; compare the full four-by-four task transfer matrix, worst-task accuracy and effective unique patient exposure. The present dataset permits patient-heldout engineering comparisons, but one-step or few-step runs cannot establish a scientific or AI-method contribution by themselves.

CPU verification: full repository suite **62 passed**; Python compile and diff checks passed. Private conversion counts were checked against all 543 prompt/gold entries. The MLX schedules were constructed without Metal execution; no ds007045 optimizer step, trained model comparison, or cross-task transfer result has yet been observed.
