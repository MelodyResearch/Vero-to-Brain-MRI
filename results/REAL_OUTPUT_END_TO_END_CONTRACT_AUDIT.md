# Real-output end-to-end contract regression

Status: `TESTED_CPU` against **saved, actual model generations** from real MRI images; no new Metal job. The optional local test `tests/test_real_mri_contract_integration.py` runs only when ignored private receipts exist, so the public test suite cannot falsely claim these checks on machines without the MRI files. On this host, all **3 integration tests passed**.

| Required case | Saved real-generation evidence | Outcome |
| --- | --- | --- |
| Correct answer, strict Vero format | SFT28 matched-prompt internal-dev outputs and hidden gold | 36/54 correct and strict-valid; task accuracy 1, format 1, total reward 1 on each correct row. |
| Incorrect answer, strict Vero format | Same outputs/gold | 18/54 incorrect but strict-valid; task accuracy 0, format 1, total reward 0.2. |
| Semantically correct, invalid strict format | SFT28 original user-only prompt outputs/gold | 35/54 correct under a **post hoc** single-box parser, while strict accuracy/format/reward remain zero. |
| Native assistant prefix and prefill | CPU rendered tokenizer artifact plus real SFT28 outputs | Prompt ends `<|im_start|>assistant\n` with no partial `<think>` tag; saved generated text begins `<think>`. The contract records `response_includes_prefill=false`. This excludes the proposed hidden prefilled opening tag for this tested native profile; full visual-token tensor hashing remains unverified. |
| Truncated response | Earlier actual official-prompt diagnostic `raw_rollouts.json` | At least one generated rollout has 512 token IDs, exactly its cap, and all diagnostic texts fail the strict format gate. This is an earlier SFT4 configuration, not evidence SFT28 truncates. |
| Multiple conflicting boxed answers | No saved model rollout with two boxes found in 57 inspected private rollout files | `NOT_OBSERVED` in real outputs; synthetic golden parser tests cover the multiple-box rule. Do not call it a real-generation integration success. |
| Reward variation and finite update | Earlier **post hoc single-box** proportional arm | One training group had accuracy rewards 0 and 0.8, format 0 for both, a finite nonzero gradient, one optimizer step and different before/after adapter hashes. This proves task-reward update under the declared ablation, **not** under strict Vero format. |
| Checkpoint reload | Saved 107-row inference after that exploratory update | All rows record the final adapter SHA and verified 504 loaded parameter tensors. This verifies reload in inference but does not supply optimizer/RNG state for exact training resume. |

The stricter post-SFT28 rollout-group and optimizer integration case is **`NOT_RUN`**. Static reward examples and cross-prompt correct/incorrect responses do not establish within-group reward variance. The battery is discharging and the versioned launch preflight has not cleared its resource and dynamic gates. The next bounded activation test should run only under an exact contract and safe power state; substantial specialist/proportional/uniform training remains blocked by that gate and independent review status.
