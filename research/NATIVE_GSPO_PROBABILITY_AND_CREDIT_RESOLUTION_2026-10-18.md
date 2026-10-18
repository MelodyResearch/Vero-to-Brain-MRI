# Vero-to-MRI：概率路径与视觉信用分配的实测修复

## 范围与研究目标

本阶段只使用此前绑定的 TRAIN 输入、已训练 B28 adapter 和历史 pre-rollout 状态。新增模型训练、optimizer updates、回答生成均为 0。没有读取 final100 来选择样本、精度、任务、父权重或方法。目标是修复 Vero-style GSPO 迁移链路的可复现性和测量问题；数值通过不代表 MRI 准确率提高。

当前模型是 Qwen3-VL-8B-Instruct、4-bit 基座、已训练 B28 language LoRA 与本地 MLX Vero-style GSPO 实现；不是 Princeton 发布的 Vero 权重，也不声称官方 veRL 全路径等价。B28、Adam52、Adam56、Adam60 及原始实验保持原样。

证据级别：`ENGINEERING_UNBLOCK / ACTUAL_NATIVE_READONLY_NUMERICS`。Mission Alignment：验证实际多任务 GSPO 的视觉条件与有符号学习信号，保留任务资格门槛，不转为独立 MRI 分类模型。独立复核由如实标注身份的隔离 Codex reviewer 完成；`CROSS_MODEL_REVIEW=NOT_VERIFIED`。

## 1. 两个历史概率异常的分解

仅凭历史 selected-token logp 无法判断完整 categorical 分布。本次实际保存 5 份完整词表测量：同一条 TRAIN response 的 17 个因果位置，每个位置 151,936 个词表项。共 55 次 language-model API 前向、2 次视觉 embedding、1 次文本 embedding 一致性检查；没有生成、反向或更新。实际退出 0，耗时 85.539 秒，模型释放后 MLX active 为 1,050 bytes。

- 原 BF16 KV 与 frozen F32-normalized full-TF 的 selected-token 最大差为 **0.100353874266**。同 KV logits 的 BF16 与 F32 归一化差为 **0.100337982178**。BF16 logp 隐含总概率质量约为 **0.932188–1.105543**；最大共同 log-mass 偏移约 **0.100336284**。这是直接测到的精度问题。
- 对 BF16 原生 logp 重新归一化后，与相同 logits 的 F32 分布仍有小残差：最大 TV **0.000083919**，selected residual **0.0000179283**。因此不能称整个词表严格只有一个共同常数偏移。这里是 score-implied 数学分布，未测实际采样频率；历史完整 logits/实际采样分布未保存，本轮 source 重放不能证明当时的完整分布。
- 原 stable KV 与 full-TF 的 **0.009733378887** 不能由共同归一化常数全部解释。完整词表比较的最大 TV 为 **0.009957547**、最大 normalized logp 差约 **0.500000**。17 个位置的 top-1 均相同，也不意味着分布相同。
- 旧 forced-KV 诊断使用整段 prompt prefill；当前已安装 AR 默认实际为 `212+1` 分块，再逐 token 解码。校正为真实分块后 selected 差仍为 **0.007351398468**，没有通过原门槛。分块差异是诊断方法的混杂因素，不足以解释全部误差。
- CPU 实际 AST 与原生路径检查未发现这条 `P=213/T=17` 输入的 token、mask、图像、cache offset 或 RoPE 因果位置错位。没有定位到某一个 attention/RoPE kernel 的唯一原因；不能将精度路径差异直接归因于 MRI 准确率下降。

另一个测量修正：frozen scorer 实际使用 F32 `nn.log_softmax`，旧对照直接做 `logit-logsumexp`。新增同 logits 双归一化对照并绑定实际实现，避免将测量算法差异称作 scorer 错误。

## 2. 实际最小修正与同输入验证

新版本诊断显式复用同一视觉前缀，复制并提升输入 embedding/deepstack 激活为 F32，绑定实际 causal position 与 cache，使用实际 centered F32 scorer。未修改模型权重，也未修改冻结 GSPO 源码或历史数值门槛。

| 同一 TRAIN 输入比较 | 最大 selected logp 差 | 最大全词表 logp 差 | 最大 TV | 结果 |
|---|---:|---:|---:|---|
| 旧 BF16 KV / frozen scorer | 0.1003538743 | 不由历史 selected 值推断 | 不由历史 selected 值推断 | 原 FAIL 保留 |
| 新 matched F32 KV / centered F32 scorer | 0.0000202656 | 0.0002746582 | 0.0000129953（F64 重算） | 通过原 0.002 门槛 |

F32 matched KV/full 的最大 JS 为 **1.5034e-10**，17/17 位置 top-1 相同、top-5 集合 overlap 为 1.0；旧三组 selected 数值均逐项复现，误差 0。表中 logp 差来自实际 F32 AR/scorer，TV/JS 来自保存的 raw logits 经 F64 归一化复算。独立 NumPy 全词表复核检查 112 项统计，最大算术差 **3.87e-15**。

这项修正是**新的数值策略**，不是与旧 BF16 完全等价的归一化替换：新 F32 控制与旧 BF16 full 的最大 TV 为 **0.072795541**。激活精度、共享前缀、显式位置是绑定的组合控制，未单独隔离某个 kernel。原 BF16 路径仍 FAIL；目前仅第一条 sequence 的 17 个位置通过 KV/full 原生资格。Dallas、其他长度与上下文的 KV 一致性没有自动获证。

历史 GSPO 的 old logp 由 teacher-forced scorer 重算，并未直接使用保存的 generation-KV logp。因此不能把上述差值当作历史训练中已经证实的 importance-ratio/clipping 错误。

## 3. 完整 GSPO 梯度诊断

B3 修正版已实际完成：外部工具退出 0，耗时 **227.441948 秒**，两组共 **16 次 detached-old 前向、24 次 backward、40 次 scorer API 前向**。梯度 checkpoint 内部层重算不另算 API 次数，实际时间和资源覆盖全部工作。没有新生成或 optimizer 更新。全部 1,942 个模型参数、504 个 adapter 参数及各组 Adam/Python+MLX RNG 指纹保持一致；模型释放后 MLX active 为 **1,052 bytes**。103 条资源记录全部 `NORMAL/CONTINUE`，最低 available **8.855 GiB**，最大观测 active **4.951 GiB**、最大 peak **10.206 GiB**，swap 保持 **12.386 GiB**。其中 2 条缺少 MLX active/peak 遥测，极值只使用有效观测；温度传感器不可用不等于已经测得低温。

B3 固定使用原先绑定的 sequence、Dallas 两个历史 TRAIN group、各自 pre-rollout B28 adapter/Adam0/RNG、保存的 response IDs 和 advantages。新 F32 scorer 在每种图像条件下计算自己的 detached old logp。因此它测量的是 **historical fixed samples / new-F32 surrogate reference** 的真实原生梯度，不是重现历史 BF16 梯度，也不是新 F32 on-policy rollout。

第一条实际 B3 尝试在 `label_support` 的空 token 前缀准备中触发 `tokens[-1]` 越界，尚未进入 scorer 或 backward。旧 source、实际失败、完整参数一致性与释放证据保留。此失败没有 group-final Adam/RNG 指纹，明确标为未验证。后续独立 v2 只补空前缀 guard、将元数据准备移入状态审计 `try/finally`，不改变 scorer、loss 或容差。针对实际 decoder AST、EOS、非类别答案及首次 scorer 前失败路径新增 CPU 回归验证。

### 实际 group 梯度与信用方向

| 指标 | Sequence 历史 TRAIN group | Dallas 历史 TRAIN group |
|---|---:|---:|
| 原图 full-group 梯度 L2 | 1.646326 | 1.525601 |
| 零归一化像素 full-group 梯度 L2 | 1.496461 | 1.423742 |
| 原图/零像素梯度 cosine | 0.797870 | 0.848530 |
| 原图−零像素梯度差 L2 | 1.009167 | 0.817545 |
| 原图正/负优势梯度和 cosine | +0.105959 | −0.478106 |
| group norm / 各响应 norm 之和 | 0.649573 | 0.491393 |
| strict-label 投影 group L2 | 0.257340 | 0.985218 |
| full-minus-label remainder L2 | 1.527278 | 1.224174 |
| full-response 局部方向与保存正确性相符 | 4/4 | 3/4 |

图像改变使实际 group 梯度发生明确变化；两组没有完全抵消。Dallas 的有符号正/负响应和存在部分反向抵消。这里的 norm 比率不是互斥梯度贡献百分比；向量非正交，不能由此称“某百分比梯度来自格式”。

在所有 own-old/current logp 逐项误差均为 **0** 的前提下，当前 ratio 为 1，使用保存的 native GSPO loss AST 的真实反向梯度，检查无穷小下降方向 `−g`：

- Sequence 三个正确回答的 mean-logp 方向导数为 **+0.059437、+1.826977、+1.544495**；一个 strict-nonclass 错误回答为 **−6.084089**。错误回答不强行解析为合法类别，label mask 为 0。
- Dallas 三个正确回答为 **−0.087605、+1.192515、+0.421432**；错误回答为 **−5.697788**。第一条正确回答的 strict-label 分量为 **+0.00156433**，remainder 为 **−0.08916956**，合成为负方向。这是一个真实的序列级信用冲突反例：有正优势不保证合成 group 下降方向提高每个正确响应的平均 logp。
- 所有可严格分离的类别分量在这两组都沿保存正确性所需的方向；但这不是完整候选类别 margin 的导数，也不是一次真实参数更新后的正确率。

remainder 由 full-minus-label 得到，包含推理、格式、其他文本和整个 nonclass 响应，**不是单独测量的 format-only gradient**。原 label CE 诊断未重复。本次证明的是两组固定样本的新 F32 surrogate 路径有图像依赖及局部信用冲突；没有证明历史 BF16 训练效果、全 GSPO rollout/scorer 跨上下文资格或真实 MRI accuracy 改善。

## 4. 第二个 MRI 任务的实际资格推进

没有重复渲染 168 张 sequence 图像，也没有重复 84 对盲态检查。此前技术检查的 168 张图像、28 人×3 层面×2 模态配对保持；166 张有非专家视觉候选证据、2 张严重覆盖不足，专家资格仍待定。Dallas 48 条继续是 whole-head 身份对应工程任务，没有新 brain-only 科学资格。

本次利用既有 TRAIN 证据生成可复现候选文件：214 条原字段不变的病灶技术记录；严格空间复核优先子集 15 条，覆盖 13 人、6 sites。复用既有 7 张 MRI-only 复核表，为 84 个平面建立空白专家裁决 packet，答案键分开保留，实际专家裁决 **0**。

基于既有非专家 MRI-only notes 和 mask geometry 的 84 平面互斥复核队列：

| 分区 | 数量 |
|---|---:|
| 空间复核候选，待专家 | 15 |
| 阳性几何复杂，待裁决 | 22 |
| mask 阳性，非专家在 contact-sheet 分辨率未见显著 focus | 5 |
| cavity/rim 病因歧义 | 4 |
| mask 阴性但 MRI 看似异常 | 5 |
| mask 阴性，需专家阴性裁决 | 32 |
| 解剖覆盖不足 | 1 |

还生成同 site、同层面、不同患者与不同空间答案的控制索引：全部 grid 候选有 14 对，严格 15 条子集有 4 对。这是控制样本清单，尚无新的模型或专家控制结果。

来源 README 明确为术前、病理确认的原发 GBM，分割经神经放射科医生复核/修正，但本轮既有本地 source docs 中未找到显式数值 `0/1/2/3` 标签字典。因此 `mask==3` 是否等于增强肿瘤仍为 **NOT_VERIFIED**。AI 复核中的 cavity/rim 描述不能被解释成术后诊断。2D 可见病灶与阴性真值，以及冻结问题“所有肿瘤区域中心所处的 3×3 网格”是否能以 whole-mask-union 全局质心作端点，仍需专家证据。

**第二个异质 MRI 科学任务批准数仍为 0。** 同一 mask 派生的三个标签路由不构成三个独立任务。以上 manifest 可用于后续资格复核，`training_authorized=false`。

## 5. 决策和剩余门槛

当前 `READY_FOR_TRAINING=NO`。通过数值诊断也不会自动授权续训。已经确定的任务门槛是第二个真正可答、视觉依赖明确且标签端点可靠的 MRI 任务尚未建立。B3 已排除“所检查两个新 F32 surrogate group 完全不受图像影响”，但没有排除信用分配风险；Dallas 已出现一个正确响应局部方向反转。数值路径还需要在未来实际 rollout/scorer 的共同合约下完成更广上下文资格，不能直接将诊断 scorer 热替换进训练。

| 解释 | 证据状态 | 对 MRI 准确率失败的结论 |
|---|---|---|
| A. KV/scorer 数值路径不一致 | **CONFIRMED**，原 BF16 仍 FAIL；新 matched F32 单实例通过 | 历史训练因果影响 **NOT_VERIFIED**，不是已证实的唯一根因 |
| B. GSPO 视觉信用分配/抵消 | **CONFIRMED（局部）**，原图梯度不同；Dallas 1/4 full-response 方向不符；没有完全抵消 | 跨组、跨 seed 的主因 **NOT_VERIFIED**，需要保留风险 |
| C. 任务和监督资格不足 | **CONFIRMED**，第二异质科学任务批准 0；原 28 条序列未充分覆盖既有配对图像 | 是当前训练恢复的首要科学阻塞；扩大样本能否提高准确率尚未证明 |
| D. 奖励稀疏/format gate | **CONFIRMED** 的历史限制：362/2,286 groups 接受更新 | 本次不重复既有精确重放；不能仅靠更多更新推断收益 |
| E. 模型适配容量不足 | **NOT_VERIFIED** | 本次图像敏感梯度反驳所检查路径“完全不用图像”，但未排除容量限制 |

以未来共同数值合约、第二任务资格和新授权通过为条件，暂定未来唯一优先训练因素为**序列样本组成：原一人一例改为既有同患者同层面的成对模态监督**，并先完成样本资格和新的独立患者验证设计。它针对已确认的视觉曝光覆盖限制，不能提前声称提高分类或解剖推理。第二任务未经科学批准仍不能启动多任务实验；不得同时改变 reward、LR、sampler 和 LoRA。数值策略修复需先作为共同前置合约独立资格，不把其分布变化与数据干预混为一个实验。没有新授权，本阶段不启动训练。

父权重必须按预先声明的目的选择，不能使用旧 final100：若未来明确延续原 adaptive 链，可声明 Adam60 为 literal-continuity 父权重，而不是声称它更优；B28 保留共同基线。Adam52/56/60 全保留，兼容状态才继承 Adam/RNG；改变数值合约必须明确新阶段。未来准确率验证需要预先冻结新的独立患者集合与跨来源 lineage 检查；此集合尚未构建，不能声称已经独立。

## 6. 可审查证据

- `results/NATIVE_PROBABILITY_RESOLUTION_ACTUAL_INDEPENDENT_REVIEW_V1.json`
- `results/NATIVE_PROBABILITY_DISTRIBUTION_INDEPENDENT_REVIEW_V1.json`
- `results/SAVED_NATIVE_PROBABILITY_SOURCE_INDEPENDENT_ANALYSIS_V1.json`
- `results/CONDITIONAL_GSPO_PRE_SCORER_FAILURE_INDEPENDENT_REVIEW_V1.json`
- `results/CONDITIONAL_GSPO_GRADIENT_ACTUAL_INDEPENDENT_REVIEW_V2.json`
- `results/GSPO_SHARED_PREFIX_LENGTH_CREDIT_CPU_V1.json`
- `results/SECOND_MRI_TASK_REVIEW_QUEUE_V1.json`
- `results/SECOND_TASK_ELIGIBILITY_INDEPENDENT_REVIEW_V1.json`

所有图像、病例键、token IDs、逐词表 logits、逐响应梯度、权重、Adam/RNG 和原始运行日志仅保存在 private local_data，不进入 Git。
