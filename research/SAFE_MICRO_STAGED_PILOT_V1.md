# Safe micro pilot: staged preparation — 2026-10-21

## Actual status

The previous three-hour compound scope is revoked. This revision is source/CPU preparation. **No model load, GPU operation, generated response, native backward or optimizer update occurred this round. MRI accuracy is NOT_RUN.**

The authentic parent remains **P_calibrated13 / Adam52 / boundary114**. Lost Adam60 is not claimed restored. Existing verified legacy z045 TRAIN inputs are reused; no renderer, loss, reward, LR, LoRA or architecture change is introduced. The frozen research plan and lock remain unchanged.

## Stages and authority

1. **CACHE_ONLY:** executable and independently reviewed; fresh explicit researcher grant pending. Adopt 406 retained units and saved exact unit0; at most343 direct+343 module quantizations,686 new/688 cumulative. Historical unit0 missing ordinary receipt/six metrics remains disclosed. Every payload and final two shards must match historical SHA. First mismatch stops; no retry.1800s operation-start window is not a guaranteed wall-clock completion deadline.
2. **SINGLE_LOAD_FORWARD:** separate future exact scope and grant after cache certification and exit/release/resource checks. Not executable through the Stage1 token.
3. **MICRO_TRAIN:** transaction/native adapter implemented and CPU tested; production stage-specific admission and exact runtime anchor still require completion/review after prerequisites. No current training permission.
4. **SMALL_PILOT:** future optional expansion only after genuine evidence and new permission.
5. **DEV24_EVAL:** frozen all24 existing sequence DEV inputs, separate exact evaluation permission. No automatic progression.

Stage1 scope FILE SHA256: `49dad3571308ed23af9efb1cec6fd78cbeb8e66e7760a724953fb3316d332419`.
Canonical scope: `d372c429c46536c928297e5066f088e92dfff15fff0291dcd4f9e3c16bb576ee`.
[Independent Stage1 review](../results/SAFE_MICRO_STAGE1_CACHE_ONLY_INDEPENDENT_REVIEW_V1.json): PASS_WITH_LIMITATIONS, source/CPU only.

## Micro comparison and cost caps

Use the first complete predeclared round: one TRAIN person, FLAIR and enhancedT1, plus two matched unchanged Dallas engineering-control records. Generate the common16 responses once: sequence8 and Dallas8. A and B each restore the same parent, then run serially with at most **one total optimizer update per arm**. Only sequence grouping differs.

To combine both task exposures under that cap, this micro version forms `g_joint=0.5*g_sequence+0.5*g_Dallas`; inactive blocks are not renormalized. This is an explicit common change in batch/update granularity relative to the old per-block driver. Zero/no active joint gradient produces zero updates, never a fabricated or momentum-only learning claim.

Prospective training cap:4096 returned tokens,16 detached-old forwards, at most32 gradient forwards/backwards,27072 teacher-forced positions across the common rollout and both arms. These are ceilings, not targets. Proposed micro operation-start window1800s, actual elapsed time unknown until native run.

DEV24 is all12 FLAIR+12 enhancedT1 from four TRAIN-disjoint existing DEV people; no outcome-dependent subset. Parent/A/B use the same F32/strict evaluator. Prospective separate evaluation caps:72 greedy,216 candidate API calls,18432 returned tokens,91800 TF positions;2400s operation-start window. This repeatedly explored DEV supports engineering screening only. Sequence-only DEV24 cannot establish Dallas retention or independent-patient generalization. Old final100 and sealed new16 outputs remain unused.

## Resource evidence

Six actual read-only samples over26.19s observed9.58–9.76GiB available,53–54% pressure-free and declining swap. Fresh Pacific2026-10-21 00:19:43 sample:9.5848GiB available,54% pressure-free, OS pressure1,swap14.7049GiB, battery78% BATTERY,other-heavy0; global lock had no open holders. Monitor RSS is not model RSS; MLX active/peak was unmeasured because MLX was not imported. Thermal sensor is UNAVAILABLE, not declared normal.

The unchanged cache policy keeps5.5GiB/free30%, original swap baseline13.96966796875GiB+1.5GiB => **15.46966796875GiB effective cap**, absolute16GiB, RSS/MLX/disk/thermal/lock protection and bounded HOLD. Battery floor20% is local to this round. Ordinary swap-rate warnings alone do not terminate work. Fresh execution admission must resample.

Proposed model loading requires10GiB available/free40%, so the recorded9.5848GiB would HOLD for loading. Small data/update count does not reduce the base model loading peak. No OS-stability guarantee is claimed.

## Implementation and review evidence

Claude Code2.1.306 actually ran Sonnet5.5 for25 turns and committed `4f95b4a382e95ee3155242b75edcd98ae560f823`: micro transaction, lazy native adapter and25 synthetic tests. This was not strict test-first development; later mutation checks are recorded separately. Codex implemented stage1 admission, resource helper and exit witness and verified46 focused synthetic tests in2.18s.

An isolated Codex reviewer found that post-exit sensor exceptions could prevent a durable exit receipt. The fix writes the known exit/log hashes first, then records failed sensors as UNAVAILABLE. The unchanged counterexample and four witness tests passed5/5 independently. No GPU/model verification is inferred from these tests.

[Sanitized actual preparation evidence](../results/SAFE_MICRO_STAGED_PILOT_PREPARATION_ACTUAL_V1.json).

**Decision: INCONCLUSIVE. No actual Parent/A/B accuracy exists for this pilot, and scaling is not supported yet. Next requested action is CACHE_ONLY, followed by release and a separate decision.**
