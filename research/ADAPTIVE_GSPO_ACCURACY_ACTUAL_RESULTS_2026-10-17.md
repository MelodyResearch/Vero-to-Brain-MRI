# 已有权重上的 GSPO 续训：实际准确率与停止判据

## 实际执行与连续性

本轮没有重新初始化基础模型。实际起点是原始 `P_calibrated seed13` 的 boundary114、Adam52、已训练语言 LoRA；继承完整权重、Adam moments、Python/MLX RNG。原始18个 RL arms 是从共同 B28 出发的独立对照，不是一个模型的18段连续训练。当前实现是 Qwen3-VL-8B-Instruct + B28 adapter 上的本地 Vero-style/GSPO recipe transfer，并非发布的 Princeton Vero checkpoint。

独立版本只修改组准入：格式有效的回答子集中必须存在准确率差异，同时保留总体差异、长度与方差检查。奖励0.8 accuracy +0.2 format、校准、K4、学习率、语言 LoRA 和原采样顺序不变。本段最多16组、4次新更新，实际7组即达到4次更新上限，Adam52→56，11,785 tokens，无部分工作债务；序列3组/1更新，Dallas4组/3更新。原始预算、权重、日志和队列恢复信息均保留。

独立 CPU 验证重建了4次保存的 Adam/权重递推，最大权重误差0，504个张量每次均改变；初始完整 optimizer/RNG 与父快照一致。它没有独立重跑原生 backward，因此不是所有底层数值路径的完全验证。训练、父评估、子评估均实际 exit0，模型已释放。

## 同一冻结开发集、同一读取方式的结果

| 指标 | Pcal13父模型 | 续训后Adam56 |
|---|---:|---:|
| 序列 strict greedy |9/24 =37.50%|11/24 =45.83%|
| FLAIR greedy recall |5/12|6/12|
| T1 greedy recall |4/12|5/12|
| 序列无效输出 |7/24|7/24|
| Dallas greedy |8/16 =50.00%|7/16 =43.75%|
| same recall |8/8|6/8|
| different recall |0/8|1/8|
| 等任务 greedy macro |43.75%|44.79%|
| worst-task accuracy |37.50%|43.75%|
| 序列 mean-label candidate |15/24 =62.50%|15/24 =62.50%|

24条 candidate 预测全部相同，raw/first 读取也都是15/24。Mean-label 配对分离度1.564465→1.563217；唯一无图像查询的 margin −2.026427→−2.026127。没有证实视觉判别能力改善。序列 greedy 的2条增益伴随 Dallas 退化，不能宣称整体准确率恢复，也不能把强制候选读取相对自由生成的收益算作权重训练收益。

该开发集已反复观察，只有4位序列患者和4个 Dallas 配对循环；不是新的独立泛化证据。Dallas 是身份对应任务，不是临床变化检测。最终 sealed100 未访问。

## 有证据的瓶颈与未决问题

- 已确认：历史A28为28 FLAIR/0 T1；B28平衡14/14后仍不足。原始混合池只使用28张序列图，而已有同患者、同层面配对监督可用。它是冻结匹配设计的限制，不是不存在更多图像。
- 已确认：冻结 Qwen视觉特征的固定小队列分类为23/24，匹配直方图为24/24。可分类信息存在；强度/扫描协议捷径仍可能解释成功，不能据此证明解剖推理。
- 已确认：生成读取和候选读取存在明显差异；父模型7条序列输出达到256-token上限且无效。旧SFT含大量固定格式目标，但静态token比例不能独自证明梯度被格式主导。
- 已确认：旧严格门控会跳过本轮全部7个已保存组；新门控实际产生4次更新。这个事实只证明准入/有效曝光改变。不同策略后续轨迹并非匹配反事实，不能证明新门控因果优越。
- 已观察：当前有限续训出现跨任务退化；不能把更多更新等同于MRI学习，也不能将退化唯一归因于某一任务更新。
- 已实际测量：在预先指定的一对训练图像上，原生标签CE损失真实配对0.162280，交换像素并固定真值后1.995218，零归一化像素后0.943613；配对分离度分别为+3.665878、−3.665878、0。交换后的候选分数矩阵精确反转，说明这条标签损失路径真实使用了图像。
- 该原生诊断执行6次串行单图梯度、12次候选前向、零次更新。真实/交换/零像素梯度范数分别1.485867/15.114225/3.191291，后两者相对真实的cosine为−0.886774/−0.363174。独立CPU复算全部3×504梯度张量，并直接将504权重和1011 optimizer/RNG指纹与Adam56快照字节核对；状态未变，进程exit0，释放后MLX活动分配1032 bytes。
- 未决：language-only LoRA 的总体容量是否充分、完整GSPO损失的图像信用分配以及 rollout/scorer 一致性。单训练配对的标签CE梯度敏感性不等于GSPO原生有限差分验证、MRI准确率改善或泛化证明；目前没有依据直接解冻整个视觉编码器。

## 决定与后续执行

本段未通过“提高准确率且没有明显任务退化”的继续条件，停止沿该配置盲目扩训；保留父和子全部状态。上述原生梯度诊断已实际完成。下一项有限配置控制正在准备：从已有Adam56权重继续，仅将类别采样从proportional切换为既有uniform配方，数据、loss、reward、LoRA保持不变；最多16组/4次更新，必须独立精确批准后才能执行。它不是原始U_calibrated13独立对照臂，也尚无新准确率结果。原始对照队列保持可恢复。

本轮数值复核为真实独立 Codex 执行；没有声称 Claude 读取或核验这些私有原始结果。跨模型原始结果审查仍为 NOT_VERIFIED。

## 证据

- `results/ADAPTIVE_ACCURACY_GATE_ACTUAL_CONTINUITY_INDEPENDENT_PROOF_V1.json`
- `results/ADAPTIVE_ACCURACY_GATE_ACTUAL_TRAINING_EFFICIENCY_V1.json`
- `results/ADAPTIVE_ACCURACY_GATE_PARENT_CHILD_ACTUAL_OUTCOMES_V1.json`
- `results/ADAPTIVE_ACCURACY_GATE_PARENT_CHILD_SAVED_OUTCOME_INDEPENDENT_VERIFICATION_FINAL.json`
- `results/ADAPTIVE_PARENT_PCAL13_ACTUAL_BASELINE_INDEPENDENT.json`
- `results/ADAPTIVE_ACTUAL_COMPLETION_AND_REMAINING_ETA_V1.json`
- `results/PAIRED_NATIVE_LABEL_CE_GRADIENT_ACTUAL_SAVED_INDEPENDENT_VERIFICATION_V1.json`

原生快照：独立 accuracy 分支的 `local_data/adaptive_accuracy_gate_v1/run/boundary_0007/`；adapter SHA256 `23b6ce522f770bf39acbadd9b66ccf00529de7822a5bb4173e7c4b96cd8618a7`，完整 optimizer/RNG 位于同目录的 `optimizer/`。医学图像、患者关联原始输出和权重不进入 Git。
