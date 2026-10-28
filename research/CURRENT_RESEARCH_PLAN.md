# 目前的研究计划 | Current Research Plan

版本：1；首次冻结：2026-10-20。研究者本轮《Freeze Current Research Plan and Resume Original Research Direction》明确授权首次创建；源指令SHA256见计划锁。**本文件是当前研究路线与下一步实验优先级的唯一权威入口，不是自动更新的工作日志。** 安全、隐私、资源保护与人工模型执行授权继续适用。

## 1. FIXED RESEARCH OBJECTIVE

**Can an evidence-based adaptation of Princeton Vero-style heterogeneous verifiable RL / GSPO improve real image-grounded brain MRI accuracy and training efficiency over direct Vero-style transfer and matched controls?**

核心内容：异质MRI任务组织、类别平衡与任务采样、真实MRI图像条件下的可验证奖励、GSPO有效group构造与学习效率、跨任务迁移和负迁移、真实准确率与独立患者泛化。每个新方法必须指出它改变了哪项Vero训练机制，并与直接迁移和匹配控制比较。

FLAIR/增强T1采集模态识别是当前可验证任务之一，不等于整个MRI医学推理。本项目不能变为独立分类器、一般监督loss设计、answer-margin或Adam优化项目、纯数据整理/工程审计项目。实际本地链为Qwen3-VL-8B+B28 language LoRA上的MLX Vero-style recipe transfer，不是发布的Princeton Vero权重或官方veRL完整等价实现。

## 2. VERIFIED CURRENT RESULTS

下表保留已完成事实，不重算或利用final100选择方法。OBSERVED=实际保存运行/数据；REPRODUCED=独立复算；INFERRED=有竞争解释的机制判断；NOT_VERIFIED=缺失证据。PROPOSED只用于尚未执行的方法。

|结论|证据等级及范围|固定证据入口|
|---|---|---|
|18个独立RL臂及3个SFT对照均已完成；各从共同B28初始化，不能串成连续模型。|OBSERVED / REPRODUCED|results/ORIGINAL21_TRAINING_ACTUAL_COMPLETION_V1.json|
|原始22端点正式final100已执行；尚未证明原始本地Vero-style GSPO稳定、跨seed可重复的MRI准确率优势。保留全部正负端点。|OBSERVED / REPRODUCED；有限本地任务/实现的结论，不外推Vero普遍无效|research/ORIGINAL22_FINAL100_ACTUAL_RESULTS_2026-10-18.md|
|原18臂2286次rollout group，362次接受更新；更多更新不等于正确性提高。|OBSERVED / REPRODUCED|results/ORIGINAL21_TRAINING_ACTUAL_COMPLETION_V1.json及其指向的逐臂proof|
|正确/错误类别差异不足、零方差、strict format gate与类别先验均被观察；它们各自对准确率失败的因果占比未确定。|OBSERVED；主要失败原因排序仅INFERRED|research/PAIRED_Z045_QUALIFICATION_AND_EXPOSURE_2026-10-20.md；results/COMPLETED_RAW12_TRAINING_SIGNAL_SYNTHESIS.json|
|新F32 rollout/scorer在已测TRAIN上下文达到原0.002门槛；不是所有上下文/完整训练路径的PASS。最近两组0/4正确，on-policy有效组反向资格仍PARTIAL。|OBSERVED / REPRODUCED|results/PAIRED_Z045_F32_QUALIFICATION_ACTUAL_INDEPENDENT_REVIEW_V1.json|
|保存的真实GSPO梯度在已测实例中受图像条件调制；局部类别margin方向不等于有限Adam更新或accuracy收益。零像素比较不证明解剖grounding。|OBSERVED / REPRODUCED|research/GSPO_CLASS_MARGIN_AND_SPATIAL_ONTOLOGY_2026-10-20.md|
|Adam60一个TRAIN配对上T1正确margin为负，两张真图都选FLAIR，无图参考强偏FLAIR。|OBSERVED / REPRODUCED；单配对，不代表总体|results/PAIRED_SIGNAL_ADAM60_CANONICAL_ACTUAL_V1.json|
|v1/v2/v3辅助目标均未证明提高真实MRI accuracy；v3仅有旧Adam状态下局部TOTAL方向可行，新增增量仍削弱FLAIR/Dallas。|OBSERVED（原生v1/v2）/ REPRODUCED（v3 CPU）；准确率收益NOT_VERIFIED|research/CLASS_MARGIN_V2_NATIVE_ACTUAL_RESULTS_2026-10-20.md；research/MIXTURE_V3_ACTUAL_CPU_RESULTS_2026-10-20.md|
|新16人32张z045配对MRI已实际取得并完成技术验证；模型输出未用于方法选择。保留所有16人及已有QC标记，不按预测排除。|OBSERVED / REPRODUCED；来源命名空间内与54历史身份分离；全局外部重复NOT_VERIFIED|results/PAIRED_SIGNAL_HOLDOUT16_TECHNICAL_V1.json；results/PAIRED_SIGNAL_HOLDOUT16_INDEPENDENT_REVIEW_V1.json|
|Dallas whole-head身份对应保持ENGINEERING_ONLY，不冒充脑内结构/临床变化推理；临床空间任务仍缺独立可答性、可见性和端点资格。|OBSERVED技术准备；科学资格NOT_VERIFIED|results/SECOND_TASK_ELIGIBILITY_INDEPENDENT_REVIEW_V1.json；research/GSPO_CLASS_MARGIN_AND_SPATIAL_ONTOLOGY_2026-10-20.md|

旧thesis、契约中的“尚未训练/下一步先诊断”等为其历史时点，不覆盖本表实际完成事实。完整证据保留原文件和私有receipts；不重写历史故事。

## 3. RESEARCH PATH

**MRI DATA/TASK QUALIFICATION → VERO-STYLE GSPO MECHANISM → CONTROLLED MRI TRAINING → MATCHED ACCURACY EVALUATION → CROSS-TASK TRANSFER → PAPER**

v1/v2/v3监督辅助loss路线状态统一为 **EXPLORATORY_ARCHIVED — NOT PRIMARY METHOD**，见research/EXPLORATORY_METHOD_ARCHIVE.md。不删除源码、结果、梯度、权重或Adam/RNG；不继续其系数搜索或新的原生验证。不得把监督CE/margin收益冒充Vero GSPO收益。

任务资格工作服务于真实RL比较，不能无限替代训练与准确率验证。完整异质任务结论至少需要两个科学合格MRI视觉任务；sequence+Dallas工程控制不能自动满足该条件。单序列试验只能支持该序列任务的迁移结论。

## 4. CURRENT SCIENTIFIC HYPOTHESIS

**在MRI二分类及强类别先验条件下，原始Vero-style same-image K4经常无法获得真实正确/错误差异，使视觉任务学习信号稀疏。** 这是待检验机制，非已确定的唯一因果根因。

下一候选：将同患者、同层面FLAIR/T1的真实on-policy响应组织成配对prompt group。研究能否通过改变group baseline/advantage的归一化范围，在保留GSPO主体时改善可验证视觉信号及准确率。

必须保留真实像素、真实on-policy生成、真实答案与reward归属、原GSPO loss/importance ratio的可追溯性。不伪造正确回答、advantages或新患者；不添加CE/margin辅助目标。凡跨prompt grouping均标为 **PROPOSED VERO-STYLE ADAPTATION**，不是官方same-prompt GSPO，也不是已成立的创新贡献。

关键竞争解释：组内方差可能仅来自prompt难度/类别先验；跨prompt baseline、随机std、筛选与sequence平均可能重加权任务难度。新增active group可能抵消或强化无图类别偏好。数学/训练契约不成立时停止，不强行训练。

## 5. SUCCESS CRITERIA

1. 同协议MRI strict greedy balanced accuracy提高。
2. 增强T1 recall提高，FLAIR没有不可接受下降；阈值在未来准确率结果前独立冻结。
3. Candidate discrimination、正确/错误类别margin提供一致改善证据；mean/sum/first分开。
4. 真图、同人反模态换图与无图/适当控制区分类别先验和图像响应；不能凭zero梯度宣称解剖grounding。
5. 实际可验证信号、每次有效更新与准确率收益的生成/视觉/TF token、时间成本得到量化。
6. Dallas工程保留表现、equal-task/worst-task及未来科学合格异质任务无隐藏负迁移。
7. 新独立患者上一次预声明父子/方法比较可获得可信结果，报告患者聚类不确定性与全部失败。

CPU PASS、gradient norm、accepted updates、reward或loss变化均不是主要成功。内部反复查看的40条仅工程诊断；旧final100不再用于选择任何方法、prompt、系数、预算或checkpoint。

## 6. CURRENT SINGLE NEXT EXPERIMENT

**比较原始Vero same-prompt GSPO与MRI Paired-Prompt Group-Relative GSPO，判断是否改变真实可验证学习信号及后续MRI准确率。**

先做完全匹配CPU机制对照：同一对、每图4个响应，共8个；原两组F4/T4，对照两组F2T2。只有同checkpoint/数值策略/输入条件下真实对应的保存rollout才可作实际离线对照；否则记NOT_AVAILABLE，合成反例须标SYNTHETIC，不制造历史证据。

机制与源代码检验通过后，仅准备一次有限匹配A/B小试：相同已训练父checkpoint（Adam60按训练链连续性预定，不按final挑优）、F32策略、架构/LoRA、Adam/RNG、reward/LR、真实TRAIN配对图像及每类曝光。每arm有限8个sequence响应与相同Dallas控制响应；有效更新数为结果，不以重抽/补训强求匹配。两arm分别恢复同一父状态，串行单模型，不把原18臂重跑。

未来原生执行必须具备源码入口、真实输入/状态哈希、完整持久化与资源预算、独立审查及**新的明确训练授权**。本轮只有CPU准备；没有任何模型调用、训练或更新许可。具体可执行准备与剩余资格写入research/PAIRED_PROMPT_MATCHED_EXPERIMENT_V1.md，不得据其变化自动改写主计划。

## 7. STOP / GO CONDITIONS

- 停止v1/v2/v3系数优化和监督loss主线；不继续新原生辅助验证。
- 不重复已完成18RL、3SFT及final100，不重新初始化模型，不覆盖任何历史状态。
- 不因active groups增加而扩训；没有真实准确率收益或有明显类别/跨任务退化即停止同配置扩展。
- 不秘密丢弃全部错误/格式失败组，不重抽直到成功，不伪造奖励。
- 无匹配对照不作方法因果判断；没有至少两个合格科学任务不作完整异质MRI迁移结论。
- 新16人模型输出在方法、数据处理、预算、终点、评价规则冻结前保持不用于开发；之后全员一次预声明比较，不随结果继续调参。
- CPU机制FAIL/PARTIAL或历史匹配记录NOT_AVAILABLE仅写实验日志，不能被包装为原生或accuracy PASS；是否改变主线由研究者决定。

## 8. PLAN CHANGE CONTROL

本次首次创建由研究者附件明确要求。未来只有以下条件**全部**满足，才可提出修改，而非自动改写：

1. 新真实实验已完成；PROPOSED或CPU-only代码准备不够。
2. 保存可复核原始结果、实际模型运行记录及准确率或决策关键机制证据。
3. 完成独立复核，明确reviewer真实身份与范围。
4. 明确说明结果如何影响假设、方法或实验决定，提供新旧计划diff。
5. **研究者明确批准该次修改**；不能自行写approved=true充当批准。

日常CPU、原生、失败、代码、checkpoint、资源和技术问题只追加到research/EXPERIMENT_LOG.md。提交检查以Git基线为锚，拒绝普通提交修改/删除本计划、锁或改写既有日志前缀；输出diff和缺失的人工批准状态。仓库检查不是人工身份认证，也不能保证恶意绕过不可能。获准的未来计划修订需单独交付证据、diff及实际人工批准记录，不能用本地JSON自授权。
