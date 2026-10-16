# Independent review: balanced sequence SFT28 control

> **Historical pre-panic review, superseded for launch.** The approved 28-step attempt was interrupted and has no verified completed-update receipt. The latest researcher directive authorizes tiny serial stages only. See `SAFE_RESUME_INDEPENDENT_REVIEW.md` and `PACED_SFT_PREFIX1_RESULT.md` for the new exact one-step scope and actual completed result. Do not reuse the historical approval below.

Reviewer: isolated Codex engineering agent `/root/balanced_sft_reviewer`. Scope: the new `balanced_sequence_v2` supervised control, original starting adapter, 28 updates and 112 examples. No material was sent to Claude and no Claude approval is asserted. This review is independent source and artifact inspection; the reviewer does not run a second model job.

## Initial verdict and source findings

**NOT_VERIFIED for launch, pending exact corrected implementation and scoped approval.** The existing launcher correctly rejects the proposed job through its continuation checks: available successful GSPO evidence uses a three-record presence manifest and starts from the historical **final** SFT28 adapter. The requested matched supervised control uses the full training manifest and the original **before** adapter. Reusing that RL evidence with altered identities would be invalid. A dedicated supervised-evidence validator may authorize this supervised control using actual historical supervised updates and fresh reload inference, while preserving all RL continuation requirements.

Additional requirements identified before implementation: freeze and hash the selected 112-record schedule and actual image bytes before model loading; bind that schedule to the independent review; verify loaded starting tensors before updating; persist losses and gradient evidence at completed step boundaries; record the actual historical Adam semantics; verify final numerical parameter changes and a fresh model reload; preserve all historical artifacts and reject an existing new output directory with experiment data.

## Independently reproduced historical and schedule evidence

The actual historical receipt has 112 finite losses, 28 finite positive gradient norms, 28 optimizer steps and 28 training patients. Both actual adapter files match the recorded hashes:

- Starting adapter: `502084c8a1394fdffd765741b9b0fbfcb6720268dfe7671c20d7a67dc5f86088`.
- Historical SFT28 final adapter: `244c26fd5ce2e3b02b46703d4f46a90608f2ef1a868eeee682ee4fe48a58ed73`.
- Historical receipt: `6555de20b80d483238b9c358e7abc08f71d1754c581e7f7c2c88fe8a1f0aa687`.

Independent CPU decoding of the safetensors files found 504 matching tensor names/shapes, all 504 tensors changed, all 10,911,744 elements finite, parameter-change L2 **3.4431796378**, and maximum absolute change **0.0018424486**. This confirms actual historical parameter changes without loading a model.

Direct execution of both schedule policies against the actual full training manifest at seed 7 confirms:

| Check | Result |
| --- | --- |
| Historical sequence exposure | 28 FLAIR, 0 contrast-enhanced T1 |
| Corrected sequence exposure | 14 FLAIR, 14 contrast-enhanced T1 |
| Total selected examples | 112; 28 per category |
| Non-sequence records | All 84 exactly identical |
| Per-category patient order | Identical |
| Sequence records changed | Exactly 14 |
| Historical receipt linkage | All 112 step/category/patient triples match legacy replay |

The unchanged presence targets are 14 no / 14 yes; enhancement targets are 17 no / 11 yes. The historical receipt did not save exact record IDs or image hashes. Historical image identity therefore remains a reconstruction from source and manifest, supported by receipt patient/order/token counts, rather than an original exact-image log. This limitation cannot be repaired retroactively.

The historical matched development output was rechecked against the manifest, gold references, image bytes and current strict scorer: **54/54 strict-format outputs and 36/54 correct**, from four development patients. All 54 receipts report verification of 504 loaded adapter tensors against the historical final adapter. The reviewer inspected the actual frozen inference code performing tensor-key, shape and elementwise equality before generation. The reviewer did not repeat that historical model load. Output SHA-256 is `c65f5e4343f8f9bc3ee6d77c71c5a87dbd794b7280f5c7c85cf937d515142d9e`.

Reproducible private audit script, selected-record inventory and results are under ignored `local_data/ds007045_bounded/balanced_sft_independent_review_v1/`. They include patient-linked record and image hashes and must not be committed.

## Required comparison and acceptance limits

The exact comparison must preserve base checkpoint revision, original adapter tensors, rank 4 / alpha 8 LoRA, 4-bit/group-64 quantization, Adam learning rate 1e-5 with historical `bias_correction=False`, seed 7, patient order, four-example averaged gradients, 28 updates, system instruction, target template, image preparation and all 84 non-sequence records. The intervention is **sequence-label exposure with associated sequence-image and answer changes**; it cannot isolate label balance from changed visual examples.

Completion requires all 112 selected examples, finite per-example losses and 28 finite positive gradients, exactly 28 optimizer steps, durable checkpoints, unchanged original before tensors, finite changed final tensors, fresh final reload with exact tensor equality and actual bounded image generation, and resource receipts. Any failed gate blocks the affected model launch or completion claim. No RL expansion is authorized by this SFT review.

The four development patients remain exploratory and repeatedly examined. Tumor-mask and axial-level shortcuts remain scientific limits. Neither engineering approval nor successful supervised optimization establishes learned MRI appearance, patient generalization, or a Vero balancing advantage. Exact resume is unsupported unless optimizer and RNG state are separately preserved and verified.

## Final exact-scope prelaunch review

**PASS_WITH_LIMITATIONS for one new Tier C supervised control, conditional on current runtime resource gates.** The reported source defects were corrected and independently re-inspected. The reviewer reran the full CPU suite: **170 passed, one skipped**. All four actual base checkpoint shards were independently streamed through SHA-256 again and matched the pinned revision inventory. The actual contract/data preflight passed: 382 training rows from 28 patients, 54 development rows from four separate patients, and 436 decoded image references.

The new dedicated `supervised_control_v1` evidence route is restricted to `mlx_sft_warmstart`, Tier C, the balanced 28-step control. It checks actual historical supervised losses/gradients, adapter bytes, reconstructed schedule linkage, and actual image/prompt/model-bound reload inference. It cannot authorize an RL runner; existing RL task-variance and raw-rollout continuation checks remain in place. The reviewer independently authored the evidence wrapper, with SHA-256 `eb8158d705b50e32e73346752dabb9b5f96b3747610b6a699ea40a4eadd61c59`.

The final runner freezes the complete schedule before model imports, rechecks each selected image/target/question against that inventory, rejects any nonempty output destination, verifies exact starting tensors and serialized bytes, preserves progress after every completed update, uses explicit historical Adam `bias_correction=False`, and checks finite nonzero parameter changes. Completion remains pending until a fresh model reload exactly matches every saved adapter tensor and generates a nonempty real-image response. The runner rechecks AC power, memory and competing model processes before heavy units and before each optimizer update. Loss of a required resource gate preserves partial outputs and prevents continuation.

The starting adapter configuration was also independently inspected: rank 4, scale 2 (alpha 8), dropout 0, 252 LoRA target modules, and 4-bit/group-64 affine base quantization. Its SHA-256 `032becabb7cbb6d2af6f3451eb127e5f944551d42eb51eaee3e8d75b839b2f6a` is now part of the common review scope, so changed checkpoint-loading configuration invalidates approval.

The exact new contract ID is `babad5d1ab01a36aa5ffb04f86096f3efab10011876a9014fcd6d52689119517`; selected schedule identity is `2e5d6bcb55cfb949035986fa8e65debdf13013badb04c275d195fd96be661f67`. Authorized output is the distinct private directory `local_data/ds007045_bounded/balanced_sft_28step_v2`. The approval covers 28 optimizer updates, 112 training examples, seed 7, original starting adapter, and one greedy reload verification generation capped at 256 tokens.

The reviewer authored the private approval and independently transmitted SHA-256 **`4fc88acf3572bb57ae349731379b1f143885bde0d2a3b76ece4c7d64d133bee2`** to the implementation lead. All nine bound source files were copied into the review's private `source_snapshot` and rehashed. Any source, contract, budget, schedule, starting adapter/configuration, evidence, or destination change invalidates this approval.

At approval time, the last live power observation reported battery discharging. This approval does not override that condition: model work remains deferred until the actual launcher verifies safe AC power. Unavailable thermal telemetry remains unknown. No corrected training completion or new scientific result is asserted in this prelaunch entry. Actual outputs require subsequent independent inspection.
