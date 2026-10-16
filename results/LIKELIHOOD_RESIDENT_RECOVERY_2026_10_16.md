# A0/A28/B28 likelihood recovery: execution record

Tags: `EXPLORATORY_DIAGNOSTIC`, `B_LABEL_CONTROL`, `ENGINEERING_UNBLOCK`.

## Mission alignment

This diagnostic tests whether sequence-balanced SFT changes conditional answer preferences hidden by greedy MRI labels. It informs the supervised starting interface for the parent Vero-to-MRI transfer study. It is zero-update inference on previously exposed training patients; it is neither GSPO training nor evidence of patient generalization. Common model/tokenizer, adapters, MRI inputs, prompt, candidate prefix, score definitions and generation budgets stay frozen.

The alternative to useful image discrimination is a shared label-prior shift. Falsification uses the frozen matched image pairs, unique no-image reference, raw/mean/first-token scores and exact-input swap permutations. No thresholds or patients are selected by outcome.

## Preserved actual results

The first matched training patient has three saved inputs at each checkpoint, independently verified in `ANSWER_LIKELIHOOD_FIRST_PAIR_INDEPENDENT_REVIEW.md`. B28 changes T1-minus-FLAIR mean margins by +0.607633 and +0.599828 relative to A28 on the two images, versus +0.659703 without an image. Final labels remain FLAIR; pair separation changes from 0.549963 to 0.542158. This is compatible with a shared prior shift and does not establish improved discrimination.

Earlier bulk-loading and serial-loading resource stops are preserved. No outputs are attributed to failed attempts. A historical repeated A0 pair exists from before the latest directive; the current continuation adopts existing pilot bytes and does not regenerate those inputs.

## Resource correction

The supervisor's `sustained_swap_growth` reason can be emitted from one adjacent-sample rate measurement. The battery-only correction separates isolated load allocation from a sustained episode. It does not require accumulated swap to return to zero. Absolute process-stage swap growth, available-memory headroom, pressure, battery/drain, thermal and single-worker limits remain binding. There is no change to global macOS protections or training power policy.

One model is retained across small inference chunks, with durable rows and cumulative resource accounting. Reviewer checks occur under standing authorization; no new user click is requested per chunk. A finite review timeout or genuine resource stop preserves the completed prefix and releases the model. No monitoring scheme guarantees prevention of another kernel panic.

## Provenance and independent review

Byte-exact pilot adoption retains historical row identities and pins the original approval and actual-result audit. The complete numerical record-loop AST is independently identical to the approved pilot. Only the pinned prefix may use its old identity; subsequent rows must use the current source-bound identity. No old records are relabeled as newly executed.

Engineering and numerical review here is by independent **Codex agents of the same model family**. `CROSS_MODEL_REVIEW=NOT_VERIFIED`; the earlier private-sharing denial is respected. No Claude review is claimed.

## Execution status

**OBSERVED / independently REPRODUCED:** A28 continued from3 to42 saved inputs, adding39 real-image inputs,117 candidate/oracle forwards and39 greedy generations in one resident process. The 483.24-second stage includes20 complete patient pairs, one unique no-image reference and one unmatched trailing FLAIR image. Original pilot bytes remain exact. A0 and B28 each remain3/57; the complete three-checkpoint comparison is **NOT_COMPLETED**.

The requested transient-warning behavior actually occurred: a sole loading warning at parameter1942 was logged and admitted (available6.242GiB, pressure-free34%). The process had289 CONTINUE,1 WARNING,0 HOLD and2 STOP resource events; it was not unloaded/reloaded because of the warning. Swap net growth stayed at most0.266GiB. Minimum available6.235GiB, minimum pressure-free34%, MLX peak5.978GiB. These are observations, not a hardware-safety guarantee.

The final stop was **battery**, not swap:78→73% reached the existing five-point stage budget; a75→73% window over119.753seconds also met the1point/minute rapid-drain threshold. The42nd output had already been fsynced. Cleanup reduced active MLX allocation to approximately0.000001GiB and the process exited; no heavy worker remained. The new-stage >=75% rule blocks B28/A0 at73%, and no retry is launched. No charger request or global power change is made.

All42 numerical records independently pass the scoped checks. Adapter equality was last explicitly measured at boundary35; the final7 outputs precede the interrupted end-boundary equality check. The source has no optimizer or weight updates, but no post-stop equality measurement is invented. Temperature telemetry and watts are unavailable. The final partial batch remains a stopped batch, not a completed8-record scope.

CPU regression result after all reviewed changes: **380 passed,3 skipped**. Native/full-model numerical evidence comes from the actual42 rows, separately from those CPU tests. The frozen target remains57 unique inputs per checkpoint (56 images from28 training patients plus one no-image input). Full multi-task Vero GSPO transfer, held-out generalization and substantial new SFT remain NOT_RUN in this diagnostic.


## 中文科研状态

本次执行完成了门控修复后的真实续跑：A28新增39条、累计42/57条；同一个模型完成全部新增计算，旧结果未重跑。一次实际加载增速告警被记录并放行，没有因它暂停或重新加载。停止原因是电量78%→73%达到累计预算，同时满足快速耗电阈值。B28与A0后续阶段未运行，当前电量低于既定75%新阶段启动门槛。

20个完整训练患者对的A28严格答案为24/40正确（FLAIR20/20，T1为4/20），全部40条格式有效；另1张未配对FLAIR也正确。该单检查点结果显示标签偏向，但不能证明B28改善或削弱了全队列图像区分。已完成的首个三检查点患者对显示B28的T1条件偏好提高，且无图像偏好移动同样明显；这一对比仍仅限一个训练患者。

Scientific Significance：记录了实际输入依赖与强类别偏向，不能外推至独立患者或临床能力。AI/ML Significance：这是监督起点的诊断，尚未完成Vero多任务GSPO方法迁移或证明优化算法创新。Experimental Significance：连续保存与独立复核支持后续从精确未完成位置恢复，修复了孤立swap告警引发不必要重载的问题。

## English technical summary

The reviewed resident diagnostic executed39 new A28 inputs (117 teacher-forced candidate/oracle forwards and39 greedy generations) with zero updates. Existing pilot bytes and identities were preserved. One real load-time rate warning was admitted without HOLD or reload. The process stopped after42 total records because its unchanged battery budget and rapid-drain threshold fired; model memory was released. A0/B28 remain at3 saved inputs each. The complete three-checkpoint cohort comparison and full Vero-to-MRI RL experiment remain unfinished. Numerical review is independent Codex review, not a claimed Claude-family review.


## Authoritative analysis and next eligible continuation

Use `A28_LIKELIHOOD_PARTIAL42_AGGREGATE_V3.json` for the partial single-checkpoint analysis. V1/V2 are retained audit history; their scientific values agree, while V3 binds the final analysis sources after the stale-approval-scope correction. Both offline analyzers now reject an obsolete approval that does not cover the saved row range, incorrect next-record slices and modified previously completed prefix bytes. Actual A28 through42 uses the valid through43 approval, so no model result was discarded or rerun for this correction.

Resume is **NOT_RUN**: once current resource and new-stage battery admission are eligible and independently rechecked, continue A28 from record43 (15 inputs remaining), then B28 and A0 sequentially from record4. The previous process battery budget cannot be reset within a running session. No future execution or completion is implied by this handoff. No substantial SFT/GSPO starts under the battery diagnostic exception.
