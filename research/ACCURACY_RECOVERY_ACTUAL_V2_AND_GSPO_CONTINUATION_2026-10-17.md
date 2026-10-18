# 实际准确率恢复：两步结果与已有 GSPO 权重续训

## 已完成的真实干预

V2 从已训练 B28 adapter 开始，冻结视觉编码器，在8张训练图／4个同患者同层面配对／2名训练患者上，执行2次语言 LoRA 更新。目标为两个类别的 mean-label 分数交叉熵，排除格式前后缀的直接 loss。Adam0→1→2、全部504个适配器张量的变化、3份优化器/RNG状态和3688-token账本已独立核验；没有从零初始化基础模型。

相同冻结开发输入、相同 prompt、相同读出：

| 指标 | B28 | 两步后 |
|---|---:|---:|
| 主指标：24图 mean-label 候选准确率 |15/24 (62.50%)|14/24 (58.33%)|
| 候选 FLAIR 召回 |12/12|9/12|
| 候选 T1 召回 |3/12|5/12|
| 独立报告的生成序列准确率 |15/24|14/24|
| 生成 FLAIR / T1 召回 |12/12；3/12|10/12；4/12|
| Dallas 生成准确率 |8/16|8/16|
| Dallas same / different 召回 |0%；100%|0%；100%|
| 等任务生成宏平均 |56.25%|54.17%|
| 最差任务生成准确率 |50%|50%|

主指标下降4.17个百分点；四个序列患者的描述性配对区间为[-25,+12.5]个百分点。mean 配对图像分离度0.894069573→1.203716134，无图 T1−FLAIR margin -1.281400330→-1.055226174。图像条件分数发生变化，但准确率没有改善；不以先验移动、增大 margin 或非零更新数宣布成功。未执行空白图像控制。开发集已反复查看，不代表稳定泛化。

## 已有证据支持的限制

- 原始 A28 序列监督为28FLAIR/0T1；B28实际为14/14，原生 assistant-only whole-target loss覆盖14固定token及FLAIR2／T1五个答案token。静态权重不是实测梯度占比，不能断言全部学习被格式支配。
- 原混合训练池保留28人各一个序列样例，完整来源已有28人×三层面×两个模态的168图。这是冻结暴露设计的局限，不是数据不存在；不能凭两步多因素干预唯一归因。
- 冻结Qwen mean视觉特征固定Ridge头为23/24，匹配直方图为24/24。它反驳当前特征完全没有可分信息，但不证明完整模型能达到该准确率，也不排除强度／采集捷径。
- 原U_raw7由15/24到13/24；有效更新、格式门和类别偏好存在实际不均衡。已验证Adam连续性不等于验证全部原生视觉反向或KV生成／teacher-forcing数值一致性。
- 当前本地起点是Qwen3-VL-8B-Instruct+B28 adapter，再实施版本化MLX/LoRA GSPO配方迁移；不是加载发布的Vero检查点，也不是官方分布式veRL训练的等价证明。

## 下一项单变量 GSPO 阶段

两步配对SFT没有准确率改善，因此不盲目延长。研究者已授权从最近完成的P_calibrated13（114组、实际Adam52）权重和兼容的完整优化器/RNG继续，独立记录新阶段，不修改原18RL+3SFT。

历史114组的严格准入实际更新52组。去掉整组全格式门会纳入83组，但其中16个新增组没有格式合格回答内部的正确／错误变化。限定有效子集确有类别变化后，反事实准入为67组：序列7→12、Dallas45→55。此为保存rollout重放，不是未来更新数或准确率收益。

新版本只放宽该准入条件，保留准确率变化、等长惩罚、非零总优势以及原奖励、校准、类别采样、K4、学习率和原生GSPO数学。继续原种子13的后续16组，新增最多36160tokens／4次更新；模型权重与实际Adam52继承，子日志相对计数与实际52+n分开。先实际父检查点评估，再续训和同协议比较。**本文形成时父检查点评估正在执行；新GSPO原生更新尚未完成，不能宣称恢复或改善成功。**

训练 source 在独立 `research/mri-adaptive-accuracy-gate` 工作树，原队列在已完成边界保存恢复参数。没有中断未提交 optimizer group。最终100仍封存。

## 证据

- 两步训练数值证明：`results/PAIRED_DISCRIMINATION_V2_ACTUAL_TWO_UPDATE_INDEPENDENT_NUMERICAL_PROOF.json`。
- 完整实际准确率：`results/PAIRED_TWO_UPDATE_FIXED_EVALUATION_ACTUAL_RESULTS_V2.json`。
- 独立输出复算：`results/PAIRED_TWO_UPDATE_FIXED_EVALUATION_INDEPENDENT_VERIFICATION.json`。
- 格式门反事实与排除的负对照：`results/P_CALIBRATED13_FORMAT_GATE_COUNTERFACTUAL_REPLAY.json`、`results/P_CALIBRATED13_NARROW_VALID_SUBSET_COUNTERFACTUAL_REPLAY.json`。
- 基于原生资源事件的条件ETA：`results/LIVE_FINITE_STUDY_NATIVE_TIMING_ETA_2026-10-17.json`；不是commit时间，进一步方法迭代及资源HOLD另计。

实际数值核验人为独立Codex；Claude此前只读取两步目标源码与测试提供科学批判，没有核验私有模型输出。本阶段跨模型原始数值核验为NOT_VERIFIED。
