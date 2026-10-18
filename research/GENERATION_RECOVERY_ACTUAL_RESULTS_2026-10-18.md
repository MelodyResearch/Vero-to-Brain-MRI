# 生成预算对照：实际准确率恢复与续训决定

研究标签：`PRIMARY_VERO_TRANSFER / EXPLORATORY_DIAGNOSTIC`。证据：实际原生推理输出，保存结果的独立数值核验另见对应 JSON。没有读取 sealed final100。

## 当前状态与权重连续性

原始18个 RL 臂和3个 SFT 对照已全部完成并独立核验。三个 SFT 的实际更新数为120、120、121，总计505,105 charged tokens；原始训练剩余时间为0。原始控制臂分别从共同 B28 起点训练，不是一个连续优化链。

另行版本化的改善链为 P_calibrated13 Adam52 → 混合格式准入对照 Adam56 → 均匀采样对照 Adam60。此前两段续训均继承已训练504个 LoRA 张量、完整 Adam 和 RNG，未初始化新模型。本轮仅评估现有 Adam60，**0次梯度计算、0次 optimizer 更新**。B28、Adam52、Adam56、Adam60 全部保留。

Adam60 终点为 accuracy worktree 下 `local_data/uniform_balance_continuation_v1/run/boundary_0008`，adapter SHA256 `1741ff2feb2f569cfe062b92cb79b62f61f83463e6683a928d2c458a5f547310`。原生只读任务实际退出0，504个可训练张量逐一未变，模型释放已确认。

## 实际比较

固定开发集为24个序列问题／4人及16个 Dallas 问题／4个匹配周期。旧基线输出直接复用，未重跑。以下全部保留严格 Vero think/answer、单一 boxed answer 评分。

| 权重与生成条件 | 序列 strict greedy | 序列 candidate mean | Dallas greedy | 等任务宏平均 | 序列 invalid |
|---|---:|---:|---:|---:|---:|
| B28，原提示256 | 15/24 | 15/24 | 8/16 | 56.25% | 0 |
| Adam52，原提示256 | 9/24 | 15/24 | 8/16 | 43.75% | 7 |
| Adam56，原提示256 | 11/24 | 15/24 | 7/16 | 44.79% | 7 |
| Adam60，原提示256 | 10/24 | 15/24 | 7/16 | 42.71% | 6 |
| 同一 Adam60，原提示512 | 10/24 | 原值15/24 | 7/16 | 42.71% | 6 |
| 同一 Adam60，简短推理256 | **17/24** | **14/24** | **8/16** | **60.42%** | **1** |

简短推理只在 user 问题尾部追加固定句子：

> Keep the reasoning brief (at most two short sentences) and finish with the required boxed answer.

没有改 system、图像、标签、温度、assistant prefix、解析器或权重。没有将两个干预合并。原提示512的候选分数直接引用同一 Adam60 的原提示候选结果；简短提示的25条候选记录由75次新前向实际计算。

Adam60 原提示256 → 简短提示256：

- FLAIR recall：6/12 → **10/12**；T1 recall：4/12 → **7/12**。
- Dallas same recall：0/8 → **0/8**；different recall：7/8 → **8/8**。Dallas 仍没有有效的 same 识别能力。
- 最差任务准确率：41.67% → **50%**。
- 序列增加7个正确答案：2个原先无效答案变正确，5个原先有效但错误答案变正确，没有丢失原正确序列答案。
- 主要 mean candidate：15/24 → **14/24**；raw 和 first-label candidate 都仍为15/24。不能把自由生成改善解释为已提升视觉判别能力。

原提示512没有救回任何答案：40/40共享 token 前缀相同，36条原已完成响应逐字不变，4条触顶响应延长至512后仍触顶。Adam60 原提示的另外2条无效输出早在配置内 EOS 停止，只有10 tokens，仍未闭合 think 或生成 answer；增加上限不解决这种失败。

## 计算成本

同一40张真实图：

| 条件 | 实际生成 tokens | 记录的生成调用时间 |
|---|---:|---:|
| Adam60 原提示256 | 2,038 | 历史未逐调用记录 |
| Adam60 原提示512 | 3,062 | 209.078秒 |
| Adam60 简短推理256 | 1,286 | 122.088秒 |

简短推理比同权重原提示256少用 **36.9%** 的实际生成 tokens，同时 strict greedy 增加7/24；512多用50.2% tokens而没有准确率收益。不能从历史资源日志重建原提示256的纯生成时间。两种新条件调用时间来自实际 monotonic 计时，但它们顺序执行、共享模型，不能据此宣称消除了缓存等全部时间混杂。

另各有1次真实 no-image 生成；计入后512为3,086 tokens／210.666秒，简短推理为1,307 tokens／123.622秒。简短提示候选前向另耗85.908秒。整个有限对照实际用时490.552秒，包括加载、监测及候选工作。

348次资源采样均为 NORMAL/CONTINUE，other-heavy为0，可用内存最低13.28 GiB，MLX峰值5.90 GiB；温度传感器未提供可用状态，不能保证不会发生系统故障。没有操作无关应用或系统保护。

## 根因：支持什么，仍不能断言什么

**已确认的配置失配：** direct-Vero长度惩罚在16,384 tokens以内为0，从16,385开始为负，256／512上限都触发不了它。accuracy与format仍会改变组内奖励，因此不能说超长无效输出完全没有负面信号。

**不能认定是训练失败的唯一原因：** 原始18臂9,144条训练响应仅39条在256触顶；它们都不满足完整格式，处于被拒绝组。对同一保存组做假想缩放长度惩罚检查，没有原先接受的362组因此失去等长度惩罚准入条件。这不是新策略轨迹实验。

**本轮支持生成策略是可修正瓶颈：** 不换权重即可将 strict sequence 由41.67%提升到70.83%，同时两类 recall 增加；仅增上限失败。它是提示／生成契约的收益，**不是新训练权重的收益**。

**视觉学习和负迁移仍未解决：** 简短提示下主要候选指标下降1例，mean配对图像分离只增加0.017963，而无图像 margin 改变0.374243。各版本的单个 no-image 生成都偏向 FLAIR。Adam60相比Adam56的 Dallas same recall 从6/8降到0/8；均匀采样和增加更新不能作为成功证据。

已有视觉特征可分类、一个训练图像对的原生 label-CE 梯度会随 real/swapped/zero改变，均不支持直接断言“Qwen无法适配 MRI”。原始28条序列材料每人仅一个模态，现有28人×3层面×2模态的168张配对训练图覆盖更广，是下一项训练暴露控制的依据；不是已证明的单一因果解释。两步配对SFT的负结果继续保留，不能盲目扩训。

## 决定、剩余工作与 ETA

1. **不扩大原uniform／准入配置，也不采用单纯512延长。**
2. 保留简短推理作为已验证的探索性推理配置；不改写原始22-checkpoint正式协议，不把其收益归到GSPO权重训练。
3. 本轮不追加GSPO更新：主要candidate没有改善且Dallas same识别仍为0，未满足扩大该配置的门槛。下一有限训练版本应先把已有配对覆盖与跨任务保留问题分开验证，明确父权重、兼容Adam/RNG、新阶段、组数／更新数／token上限；不能默认Adam60优于Adam52或B28。
4. 原始22个最终checkpoint的元数据清单已齐全，final100仍封存。原始18RL+3SFT的统一正式准确率和迁移矩阵尚未执行，不能从完成训练推断哪一臂最好。
5. 剩余原始训练0；本轮Adam60评估、两项生成对照均实际完成。方法选择冻结后的原22×100正式评估，既有原生日志估计为乐观5.7小时、正常8.6小时、保守17.2小时，另需结果核验／推送；不是完成时限，不含新的方法迭代和资源HOLD。

## 证据与边界

- `results/ORIGINAL21_TRAINING_ACTUAL_COMPLETION_V1.json`
- `results/B28_ADAM52_ADAM56_ADAM60_SAVED_COMPARISON_V1.json` 与对应独立核验。
- `results/ADAM60_GENERATION_RECOVERY_ACTUAL_COMPARISON_V1.json`
- `results/SAVED_NULL_GENERATION_AND_ADAM60_INVALID_SUPPLEMENT_V1.json`
- `results/ADAM60_GENERATION_CONTROLS_AND_NULL_SUPPLEMENT_INDEPENDENT_REVIEW_V1.json`：实际独立 Codex 保存结果核验 `PASS_WITH_LIMITATIONS`，SHA256 `46290a80834e0b71e92a89174838779d9849f5a80d947db6d74505dbe43a650c`；82条新生成、25条新候选、全部40条前缀、计时、配对统计及无图像补充均独立复算。
- 生成对照私有实际 exit0／闭合日志／完整回执：accuracy worktree 的 `local_data/generation_recovery_v1/actual/`。原始影像、逐例回答、token数组、权重和optimizer／RNG均不推送。

Mission Alignment：识别Vero-to-MRI的生成／验证接口瓶颈；比较固定权重的单因素干预；保留冻结历史与严格评分；实际运行可复核；no-image及简单特征证据不升级为解剖理解；按真实准确率和跨任务退化停止扩展。内部开发集已反复查看，以上均为探索性结果，没有独立患者泛化或临床结论。当前私有结果的独立核验由准确标明身份的Codex执行，`CROSS_MODEL_REVIEW=NOT_VERIFIED`，不虚构Claude审查。
