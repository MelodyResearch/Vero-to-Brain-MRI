# 全局 Norman Skill 对既有 13 篇精读的增量核查

核查日期：2026-10-15。直接读取已安装的 [`norman-paper-reading` 全局 Skill](/Users/vv/.agents/skills/norman-paper-reading/SKILL.md)及其 `references/evidence-audit.md`、`references/novelty-and-transfer.md`；没有安装、复制或创建同名 Skill。本文件是对[逐篇审计](./)和[原文页码复核](SOURCE_RECONCILIATION.md)的增量核查，不把既有文档的“已读”冒充本轮重新执行的完整 PDF/代码复现实验。所有模型性能数字都是**论文报告值**，本项目未复现。

## 核查口径与完成情况

Skill 要求的条目为 Paper Information → Scientific Question → Research Reconstruction → Method Explanation → Claim–Evidence ledger → Limitations → Novelty Assessment → Research Card。现有 13 个文件均有版本/原文/代码入口、研究问题与方法重建、至少一个定位到 Results 表或图的主张审计、局限、迁移判断和下一实验。**“有三层段落”不等于完成全部证据核验**：不少文件的 ledger 只有三列或散文形式，缺少每个结论的数据单位、独立性、算力/信息对齐、变异与开放检查。下表按证据充分程度而非按文件是否存在标记。

| 论文（对应审计文件） | 原文 / Methods、Results、图表及补充材料 | 作者代码 | 可核对的关键证据与本轮最需要补的核查 |
|---|---|---|---|
| [MedVLM-R1 v2](2502.19634-medvlm-r1.md) | 既有审计记录完整 PDF/HTML、Methods/Results/Limitations；补充材料单独核查 `UNVERIFIED` | [官方仓库](https://github.com/JZPeterPan/MedVLM-R1)已定位，训练执行 `NOT_RUN` | Table 1 PDF p.8 的 MRI→跨模态数值可查；同等提示格式、患者去重、种子和 rationale fidelity 未定。 |
| [Med-R1 v5](2503.13939-med-r1.md) | 既有 HTML 精读；[v5 PDF](https://arxiv.org/pdf/2503.13939v5)的 Tables I/IV/VIII 页码由[复核](SOURCE_RECONCILIATION.md)确认 | [官方仓库](https://github.com/Yuxiang-Lai117/Med-R1)已定位，训练配置对齐未定 | PDF §IV-A p.6 写 328×328，README 写 384×384；要锁定实际 checkpoint/训练配置才能解释对照。 |
| [MedGRPO v4](2512.06581-medgrpo.md) | 既有全文、Appendix、Tables 2–3 PDF p.7 已查 | [官方仓库](https://github.com/UII-AI/MedGRPO-Code)仅核到推理参考，训练代码未核 | NAP 从 .442 降至 .405；无归一化组崩溃**不隔离**“跨任务尺度”原因，因组内正仿射变换标准化后通常不变。 |
| [MedMO v2](2602.06965-medmo.md) | 既有 22 页 PDF、Appendices A–E、Figs 5–6/Table 4 已查 | [官方仓库](https://github.com/genmilab/MedMO)已定位，未运行 | Table 4 中两个框 IoU 增量各 +0.4 点；强 SFT 阶段和资源差异使跨模型总提升不能归于 GRPO。需明确 `KILL`：匹配 SFT 后空间 RL 无增益。 |
| [LoV3D v2](2603.12071-lov3d.md) | 既有 13 页 PDF，Methods/Tables 1–3/Conclusion 已查；独立补充材料 `UNVERIFIED` | [官方仓库](https://github.com/Anonymous-TEVC/LoV-3D)已定位；权重/运行未核 | Table 2 DPO 相对 Stage 1b 诊断仅 +0.4 pp；MMSE/CDR-SB/APOE 输入破坏与纯影像 CNN 的信息对齐。需明确 `KILL`：双访视影像不优于同元数据的当前影像。 |
| [Vero v3](2604.04917-vero.md) | 既有全文、Appendix、Tables 2–4、Figs 7/9 已查 | [官方仓库](https://github.com/zlab-princeton/vero)数据路由/训练文档已查，未跑 | 均匀类别比例是该自然图像实验的条件结果；不能据此宣称 MRI 上机制成立。应加入零优势后的**有效任务比例**对照。 |
| [SGMRI-VQA v1](2604.15808-sgmri-vqa.md) | 既有 HTML/Appendix；[PDF](https://arxiv.org/pdf/2604.15808v1) §3.3.1、Tables 2–4 页码另行复核 | [官方仓库](https://github.com/lamawmouk/SGMRI-VQA)已定位，训练未跑 | 医生全面复核的是 222+155 个**测试**体积，训练只记录超过 300 个的迭代复核；V-Score 是帧索引二维框，不是 3D Dice。 |
| [NeuroQA v1](2605.20525-neuroqa.md) | 既有 HTML/Appendix；[PDF](https://arxiv.org/pdf/2605.20525v1) Tables 2/4/14 和 Appendix E 已复核 | [官方仓库](https://github.com/mhabbasiit/neuroqa)已定位，构建流程未运行 | text-only 49.4% 为 test-public，与全池 44.6%/random 39.5% 分母不同；README/v1 第一轮人工审查数量冲突待固定版本。 |
| [MedVol-R1 v1](2605.26621-medvol-r1.md) | 既有 10 页 PDF，Methods/Tables 1–2/Conclusion 已查；独立补充材料 `UNVERIFIED` | [官方仓库](https://github.com/AdvancingEther/MedVol-R1)现提供 CT-ORG/KiTS23 的 SFT→GRPO 脚本；[发布自审](https://github.com/AdvancingEther/MedVol-R1/blob/main/docs/RELEASE_REVIEW.md)承认配置/评估/复现未闭环，未运行 | mask-derived anchor 和局部 Dice 需要标注 mask；原文“matched boxes”进 MedSAM2 的训练/奖励实现仍需逐行检查以排除参考框进入预测路径。`KILL`：同标注 SFT selector 匹配 RL。 |
| [BrReMark v1](2606.25894-brremark.md) | 既有全文/消融 Tables 1–2 已查；补充材料单独核查 `UNVERIFIED` | [官方仓库](https://github.com/fdu-farm/BrReMark)已定位，未跑 | 异常切片由标注最大病灶面积选出，不等于全体积搜索。纯 SFT 两轮控制与跨集患者去重未定。 |
| [TractoGraphVLM v1](2608.18166-tractographvlm.md) | 既有 PDF/HTML/Supplement、Tables 1/6 已查 | [官方仓库](https://github.com/AS-Lab/Marthi-et-al-2026-TractoGraphVLM-Unified-Vision-Language-White-Matter-Tractography)已定位，未跑 | 衍生扩散纤维束图，非 T1/FLAIR 纵向原始影像；不能把有监督方向特征消融迁移成 RL 或 interval-change 证据。 |
| [NV-Reason-CT v2](2609.27511-nv-reason-ct.md) | 既有 49 页 PDF、Fig. 6/10/§6 已查；独立补充材料 `UNVERIFIED` | [官方训练仓库](https://github.com/NVIDIA-Medtech/NV-Reason-CT)已定位，未跑 | 原生 3D CT + finding-set GRPO 是直接相关架构/奖励前例；无决定性的匹配 SFT-only/坐标消融。`KILL`：固定 checkpoint 下坐标/GRPO 无增益。 |
| [BrainTRACE v1](2610.06571-braintrace.md) | 既有 35 页 PDF/Appendices A–G、Tables 2–4/Appendix C 已查 | [官方仓库](https://github.com/lanqz7766/BrainTRACE)已读，未运行 | 论文与现行 README 的数据账号及 `gpt-4o`/`gpt-4o-mini` 评判器默认值不一致；准确复现榜单 `UNVERIFIED`。70 病例六步任务须封存只测。 |

**状态解释：**“已查”指前一轮审计可追溯记录，本轮核对了文件、三层结构及若干原文主张；不表示医学影像、代码或指标已独立复现。各文献的具体页码与数值以逐篇文档及 [SOURCE_RECONCILIATION](SOURCE_RECONCILIATION.md) 为准。表中未能确认 supplement、患者独立性、训练实现或 checkpoint 对应性的项目保持 `UNVERIFIED`，不能用“已读全文”消除。

## 本轮方法指纹搜索：新增强前例

检索日期 2026-10-15；索引为 arXiv、ICML/PMLR 官方 proceedings、CVPR 论文入口和作者 GitHub。检索式包括 `longitudinal brain MRI multi task reinforcement learning verifiable reward GRPO voxel 2026`、`brain MRI longitudinal paired volume reinforcement learning verifier task balanced reward normalization 2026`、`medical multimodal GRPO task balanced reward calibration`、`Multi-Task GRPO ratio preserving sampler zero advantage`。方法指纹：**组内标准化优势 + 多任务采样/权重更新 + 异质 verifier + 零优势组筛除后的有效梯度比例**；目标任务指纹：**同一患者两次原生 3D 脑 MRI，序列/配准/扫描协议受控，输出可独立验证的区域与超出测量误差的变化**。

1. **[MT-GRPO, ICML 2026 / arXiv:2602.05547v3](https://arxiv.org/html/2602.05547v3)** 是本轮发现的关键相同方法/不同任务先例。论文 §§2–5、Algorithms 1–2 明确把弱任务奖励与改善量用于动态权重，并用 ratio-preserving sampler 在剔除零优势组后维持有效任务比例。§6 与 Appendix D.2/D.4 给出 3/9 任务、单组件消融和采样实现；Appendix D.5/Table 1 在固定 80 小时下给出 worst-task 53.6（MT-GRPO）对 43.2（DAPO）、34.5（GRPO），同时承认每步约 10.2% 时间开销。作者[代码](https://github.com/rsshyam/MT-GRPO)包含修改的 VeRL 与运行脚本，未执行。**判定：Verified Direct Prior Art**，仅对“多任务 GRPO 的有效任务比例/最弱任务优化”这一算法机制；**不是**纵向 3D MRI 同任务的直接先例。任何把 uniform 采样或零优势补偿本身包装成原创算法的主张须撤回。
2. **[MedLoc-R1, arXiv:2603.28120v1](https://arxiv.org/html/2603.28120)** §3/Eqs. 3–11、§4/Table 1 和附录，把医疗二维视觉定位中固定 IoU 门槛导致的零奖励问题改为性能驱动的阈值课程；[代码](https://github.com/MembrAI/MedLoc-R1)已定位，未执行。**判定：Related Prior Art**，针对 MRI 空间 reward 稀疏性和课程而非跨任务/纵向变化。若提出自适应 IoU 门槛作为新方法，必须比对该原文。
3. 既有 13 篇中的 [Vero](2604.04917-vero.md)、[MedGRPO](2512.06581-medgrpo.md)、[LoV3D](2603.12071-lov3d.md) 和 [BrainTRACE](2610.06571-braintrace.md) 分别占据均匀类混合/路由、医疗异质奖励、纵向 T1 验证训练、纵向多序列评价。对“上述四者完整组合已有文献”当前只能判 **No Direct Match Found in this bounded search**，不是全球首创声明；代码快照和后续版本可能改变边界。

**需要修正的研究设计：**不能只对比 Vero uniform 与 MedGRPO logistic。加入 MT-GRPO 的固定均匀+ratio-preserving、动态加权+ratio-preserving 两臂，并报告每类的提议采样比、零优势过滤率、实际非零梯度占比、重复患者暴露和总 rollout/GPU 时间。否则“均匀混合减轻负迁移”可能只是过滤后有效梯度比改变。

## 三张优先 Research Card（均为待检验假设）

### 1. 影像依赖的真实纵向变化

- **Unresolved Gap / Scientific Question：** LoV3D 的临床元数据及既往 FreeSurfer 标签可提供强捷径；BrainTRACE 是评价而非训练。正确的既往影像是否对独立判定的区域变化提供增量信息？
- **Hypothesis / Potential Method：** 患者、次序、序列、世界坐标和扫描协议一致的两访视输入优于同元数据的当前扫描；输出区域、方向、幅度区间和“变化不足以判定”状态。
- **Minimal Experiment / Baselines：** 先不训练；在更大、患者隔离、独立 QC 的重复 MRI 上用冻结模型或确定性管线对比真配对、仅当前、无影像、时间反转、错患者、错序列，并分层记录超出 repeatability 的变化。Dallas 现有 2 人/8 卷仅证明读取可行，不用于统计结论。
- **Evaluation / Success / Kill：** 预注册患者级可判定变化准确率、误报率、覆盖率、按采集协议分层和患者 bootstrap。只有真配对优于所有捷径且错配被拒绝才 GO；若错患者/时间反转同样好，`KILL` 影像纵向机制。下一步：先建立独立标注与测量误差门槛。

### 2. 纵向 verifier 的来源/协议拒绝能力

- **Unresolved Gap / Scientific Question：** Vero 路由 reward 以答案可验证为前提；影像定量标签若与 reward 共源，会制造循环“正确”。
- **Hypothesis / Potential Method：** 几何、患者/访视/序列 lineage 与不确定性联合 verifier 能拒绝高分但证据错误的输出，优于仅用数字方向或二维 IoU。
- **Minimal Experiment / Baselines：** 不训练模型；盲化 challenge set 包含真配对、swap、反序、注册偏移、协议改变、亚阈值变化、编造体积。比较纯数值、纯坐标、联合 verifier；真值由独立管线/QC 提供。
- **Evaluation / Success / Kill：** 主要终点为患者级不支持变化的误接受率，同时报可判定覆盖率。预先固定可接受的误接受上限与覆盖下限；若错患者/错访视被接受即 `KILL` 当前 reward。下一步：把 Dallas 现有 FLAIR 间距不一致纳入反例库，但不把两人案例当作有效率估计。

### 3. 任务均衡的增量效应

- **Unresolved Gap / Scientific Question：** Vero 的 uniform、MedGRPO 的非线性 reward 和 MT-GRPO 的有效比例校正可能产生相似的宏平均改善；MRI 上何者必要？
- **Hypothesis / Potential Method：** 仅在 verified MRI 任务之间存在可重复负迁移时，保留独立影像证据且控制 reward/有效梯度比例的新策略才可能优于现成方法。
- **Minimal Experiment / Baselines：** 同初始化、同训练例、同视觉 token/rollout、等 GPU 时间：SFT、单任务 RL、自然比例、Vero uniform、MedGRPO logistic、uniform+MT-GRPO ratio-preserving、MT-GRPO 动态加权；至少三个 seed，患者级独立测试。比较原始/仿射/logistic 回放以拆开校准与采样。
- **Evaluation / Success / Kill：** 主要终点预先指定 patient-macro 和 worst-task 表现；辅以错误医学断言、报告事实及有效梯度比例。增量需在独立患者/站点上超过上述强基线，且机制消融定位新增部分。若 MT-GRPO/MedGRPO 或相同数据 SFT 匹配，则 `KILL` 独立算法创新，保留工程迁移/数据贡献。下一步：先验证两个以上 MRI reward，随后才投入 GPU RL。

## 本轮结论的证据标签

- **Direct observation in cited sources：** 13 篇逐项原文已记录于既有审计；本轮新增 MT-GRPO §§2–6/Appendix D 和 MedLoc-R1 §3–4 的原文及官方代码入口。表格数值均为作者报告。
- **Author interpretation：** 例如 MedGRPO 将失败主要归因于跨数据集 reward 尺度、LoV3D 把高诊断表现解释为 MRI grounding；两者现有消融不能完全隔离这些机制。
- **Our inference：** 纵向 MRI 的可发表方法空间应放在独立证据、采集感知变化与有效梯度机制的交互，而非纯 uniform mix；这是由上述方法指纹与数据条件推得，尚无 MRI 模型实验证实。
- **UNVERIFIED / NOT_RUN：** 13 个作者训练流水线未本地复现；若干仓库没有可验证的训练代码或固定快照；所有 MRI VLM 推理和 RL 训练仍未执行。文献阶段已完成增量核查，但整体 MRI 研究继续开放。
