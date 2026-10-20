# GSPO 类别 margin 与 MRI 空间任务 ontology：实际核验

## 范围与决策

继承 `cc41a50c47489a3e3481d03f1518e9d999c329e8`，工作分支 `research/vero-mri-margin-ontology-v1`。本轮新增训练 **0**、optimizer updates **0**、生成回答 **0**。未重复旧 B2/B3、18 RL、3 SFT、final100 或 MRI 图像盲审。B28/Adam52/56/60、历史 source、Adam/RNG 与训练结果保留。

**当前结论：所检查两个 TRAIN 输入中，保存 GSPO 梯度的欧氏下降方向提高了正确相对错误类别的 margin；Dallas 的完整回答 logp 下降并不等于类别判别受损。因此没有证据支持修改当前 GSPO loss，本轮没有实现或启用新的答案信用分配 loss。** 这不证明有限 Adam 更新后的 accuracy 提升，也不代表全训练分布均有利。

Mission Alignment：`ENGINEERING_UNBLOCK / C_TASK_ELIGIBILITY`。固定历史 TRAIN 输入、权重和保存梯度，区分类别判别与非答案文本信用，再验证第二个可验证 MRI 任务的来源与端点。原模型仍为 Qwen3-VL-8B + 已训练 B28 language LoRA 的本地 Vero-style GSPO recipe transfer，不是 Princeton 发布权重或完整官方 veRL parity。独立复核为隔离 Codex；跨模型家族复核 `NOT_VERIFIED`。

## 1. 为什么保存的 B3 还不能回答 margin 问题

旧 B3 给出了完整 response 与其中正确类别 token 的梯度，没有同一前缀下 canonical 错误类别的反事实梯度。一个合法归一化 categorical CPU 反例证明：保持正确类与 group 梯度完全相同，未观测错误类梯度可使 margin 方向为 −1 或 +1。因此不能由 `−0.087605` 或 `+0.001564` 单独推断类别受损。

本次严格固定：

- 原两个 pre-rollout TRAIN checkpoint、图像、native prompt、保存的 REAL/ZERO full-group 梯度。
- COMMON 使用既有固定答案前缀；SAVED_RESPONSE0 保留历史 response0 类别前的精确推理上下文。Sequence 两种前缀相同而去重；Dallas 保留两种独立前缀。
- 正确与错误候选使用相同前缀，只计 canonical 类别 span；不计 prefix、suffix、EOS，不宽松解析。
- 主指标 `M_mean = mean(logp_correct) − mean(logp_wrong)`，继承旧候选读取方式；它是长度归一化分数，不是完整类别事件 log probability。另报 joint label-span `M_sum` 与下一 token `M_first`。
- 每个新梯度 h 都与保存 g 做 `D M[−g] = −dot(g,h)`。没有重算 full-group GSPO 反向。Sequence 标签长度 2/5，first 使用单独共享因果 logit 向量；Dallas 两类各一 token，mean=sum=first。

`−g` 是欧氏 group 梯度方向，**不是 persistent Adam 的预条件方向或实际有限参数步**。新 F32 fixed-sample surrogate 不等于历史 BF16 on-policy 梯度。

## 2. 实际原生执行与 Dallas 数值资格

单次原生工具实际退出 **0**，用时 **214.151101 秒**：33 次 LM API 调用、16 次视觉 embedding、14 次新类别/first 反向。LM API 次数不含 gradient-checkpoint 内部层重算。1 次模型加载，无参数更新；模型释放后 active MLX **1,050 bytes**。

先执行此前未覆盖的 Dallas `P=405/T=17` 保存上下文，F32 KV/full 资格通过后才计算 margin：

| 数值比较 | 实测最大差 |
|---|---:|
| 实际 AR F32 vs centered full scorer，全词表 logp | 0.000198364258 |
| 实际 AR F32 vs centered full scorer，selected logp | 0.000043869019 |
| F64 重算的 normalized 全词表 logp | 0.000199921757 |
| F64 categorical TV | 0.000008812108 |

保持原 **0.002** 门槛；17/17 top-1 相同、top-5 集合 overlap=1。实际 cache K/V 与输出为 F32，因果位置、pixel/prompt/token/grid 绑定一致。这只新增一个 Dallas 上下文的资格；没有把候选的不同长度形状或所有模型上下文一并批准，更未替换历史训练器。

101 条资源记录中，98 次 CONTINUE、3 次因加载期间 swap 增长而 HOLD；门槛恢复后在同一进程继续，没有重启模型或修改门槛。全部样本 supervisor 为 NORMAL、AC 供电且无其他 heavy process。最低 available 6.487 GiB、最大 active 5.039 GiB、peak 10.125 GiB；swap 10.045→11.650 GiB。MLX active/peak 各2条记录缺失，极值仅使用有效观测。温度传感器不可用，不声称低温或硬件风险已排除。

## 3. 类别 margin 的实际结果

以下是真实 MRI 输入、沿保存的 REAL group 方向 `−g_REAL`；单位是未归一化参数方向的局部导数，不是准确率百分点：

| TRAIN 上下文 | M_mean | M_sum | M_first | D M_mean | D M_sum | D M_first |
|---|---:|---:|---:|---:|---:|---:|
| Sequence COMMON（亦为 SAVED） | 0.078578 | 0.934877 | 0.892180 | +12.255905 | +41.693154 | +45.117666 |
| Dallas COMMON | 3.750874 | 3.750874 | 3.750874 | +49.146564 | +49.146564 | +49.146564 |
| Dallas SAVED_RESPONSE0 | 7.154451 | 7.154451 | 7.154451 | +33.044614 | +33.044614 | +33.044614 |

### Dallas 冲突的直接答案

在相同历史推理前缀下：

- 旧完整 response mean-logp 局部导数：**−0.08760523**。
- 旧答案支持分量（仍除完整 T=17）：**+0.00156433**。
- 新直接测得正确类 logp 导数：**+0.02659395**。
- 新直接测得错误类 logp 导数：**−33.01801986**。
- 所以正确减错误 margin 导数：**+33.04461381**。

旧答案分量乘17约为 +0.02659357，与新正确类测量同向且非常接近；这是观测比较，不由 score 相同自动推出 gradient parity。完整回答的负方向主要涉及非答案文本，不是这个同前缀类别对比受损的证据。没有依据因此改 GSPO loss。

### 图像条件与类别先验分别报告

| 上下文 | M_mean：REAL | M_mean：ZERO | D M_mean：REAL | D M_mean：ZERO |
|---|---:|---:|---:|---:|
| Sequence COMMON | 0.078578 | 0.822171 | +12.255905 | +4.758984 |
| Dallas COMMON | 3.750874 | 3.871475 | +49.146564 | +27.142236 |
| Dallas SAVED_RESPONSE0 | 7.154451 | 7.704088 | +33.044614 | +17.140942 |

方向均固定为保存的 `−g_REAL`，仅改变新 margin 函数的像素条件。零归一化像素是 OOD 控制，不是无图 prompt；历史推理文本保持。REAL−ZERO margin 的局部方向差为正，但当前基线 margin 在 ZERO 下反而更大。说明图像会调制这个局部信号，同时类别/文本先验仍重要；不能把正导数或零图像也正确解释成 MRI 理解。

完整交叉矩阵也计算了保存 `−g_ZERO` 的方向，保留在独立证据中。全部六个图像/前缀条件、两个保存方向、三种 margin 的36个导数均为正；这不是36个独立统计样本。两组 TRAIN 诊断无统计泛化能力；未运行新 greedy accuracy，也未测试 Adam52/56/60 的新学习效果。

## 4. 第二 MRI 任务：来源已收窄，科学资格仍 HOLD

[正式来源全文 Data Records](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC13498554/fullTextXML) 明确数值字典：0背景、1坏死核、2非增强/水肿肿瘤、3对比增强肿瘤。论文对应 ds007045 v1.0.1（337人），本地为 v2.0.1（363人）。不能套用通用 BraTS 印象或把版本视为相同。

先前按本地 TRAIN 身份映射逐人发出公共请求的动作被自动审批拒绝，执行数 0。随后独立正常审批通过仅公开固定版本完整目录的下载；所有本地患者匹配和 payload SHA 比对离线完成，没有外发 TRAIN 选择映射：

- 28/28 本地 mask 指针绑定当前 release，payload SHA/size 与既有 packet 一致。
- 26/28 annotation 指针与论文版本相同；2个是后来新增，其数值3的精确版本语义仍未验证。
- 对15条优先空间候选，对应13条论文版本同源、2条新增版本。这个限制针对增强子类，不自动否定 `mask>0` 的 whole-union 空间算术。

### 实际代码和候选修正

新代码明确 panel-relative **image-edge** 坐标，以 mask 前景像素等权计算的几何质心表示 panel display 像素坐标；不是组织物理质量或世界坐标测量。边界使用半开 thirds、边界归右/下；union、各组件和可见 focus 是不同端点。保留旧 prompt/target，将新算术端点、差异和 HOLD 原因单独存入新 manifest，不覆盖历史标签。

在46个 mask 阳性平面中：14个多2D组件、2个 union 质心落在 mask 外、8个组件网格与 union 不同、15个距边界不足10 display pixels；1个非优先候选对新旧坐标约定敏感，已显式 HOLD。这是约定敏感反例，不据此追认历史标签为软件错误。

15条优先候选均为单2D组件、质心 on-mask、组件网格一致、距边界≥10 pixels，旧/新网格变化 **0**。13个来源体积中7个有多个26连通 mask 组件；不等于临床多灶。214个既有 mask 派生 target 均可重现，原 target 改写 **0**。

实际工程修正还包括 pointer-lineage 检查同时要求 symlink mode 与内容 blob 匹配；同 blob、错误 file mode 的 CPU 反例先失败后修复，防止把非注释链接自动宣布 ontology 已证实。可见病灶中心、临床多灶、阴性真值不能由上述程序生成。

新版15条候选与84条空白专家表、单独的第二遍端点 key 已在本地准备。MRI 图像解码、重渲染、重复盲审均 **0**；只读取 TRAIN masks 做新来源/坐标计算。所有15条仍缺专家 frame visibility 与 question/annotation endpoint alignment 裁决，科学准入 **0**。Dallas 仍是工程身份对应任务，没有新的脑内身份可答性资格。

## 5. 下一阶段唯一优先改动与恢复门槛

`READY_FOR_TRAINING=NO`。没有观察到本次 GSPO 原始 group 方向损害 canonical margin，故保留当前 loss，不加入答案信用分配修正，也不追加训练。

未来最值得隔离的单一训练因素仍是**序列监督样本组成**：若既有同患者/同层面配对 FLAIR/T1 图像完成所需科学资格，检验相对原一人一例暴露的差异。具体协议须预先绑定 baseline registry、患者和 slice-level 分布、任务预算及暴露次数；不能同时从单层面扩成三层面再把效果单独归因于模态配对。它针对已知覆盖限制，不保证改善，也不同时改变奖励、LR、sampler、LoRA 或数值策略。此前小 SFT 负结果保留，不能把配对图像本身当作已验证有效的方法。

先决条件是这批序列数据和第二个科学任务的科学资格、共同 rollout/scorer 数值合约和新的预声明独立患者验证。新 F32 合约属于独立前置资格，不能与数据改动合并为一个因果因素。旧 final100 不再用于方法或父权重选择。

父权重规则须在未来新授权前声明；Adam60只能按延续52→56→60的理由选择，不能称其最佳。B28继续保留共同对照，Adam52/56/60全部保留，状态兼容才继承。当前不启动任何训练。

## 6. 可复核证据与实现

- `src/mri_rl/class_margin_v1.py`、`scripts/run_class_margin_v1.py`：新类别 span、共享 first-logit、保存方向 cross-dot，以及 Dallas F32 资格；没有 optimizer/loss 修改入口。
- `results/CLASS_MARGIN_ACTUAL_INDEPENDENT_REVIEW_V1.json`：闭合退出、19次 Dallas LM 加14次 margin LM、14份新梯度、28次 cross-dot、全词表与状态独立复算。
- `results/MARGIN_IDENTIFIABILITY_INDEPENDENT_MATH_V1.json`：为什么旧 full/label 证据不足以推出 margin 的归一化 categorical 反例。
- `src/mri_rl/spatial_ontology_resolution_v1.py`、`scripts/build_spatial_ontology_resolution_v1.py`：正式来源绑定、独立端点和保留原 target 的候选修正。
- `results/SPATIAL_ONTOLOGY_RESOLUTION_V1.json`、`results/SPATIAL_ONTOLOGY_RESOLUTION_INDEPENDENT_REVIEW_V1.json`：作者实际构建和 root 独立13个来源 mask/15条候选复算、28个版本指针核验、84张空白表身份验证。

新的原生输出仅在本地 `local_data/class_margin_v1/run_v2_PRIVATE`；15条候选、84份专家表和独立答案 key 在 `local_data/spatial_ontology_resolution_v1/candidate_v2`。图像、患者键、token IDs、原始回答、词表数组、梯度、模型权重和 Adam/RNG 均不进入 Git。
