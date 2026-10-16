# Novelty Agent 独立审查：纵向 MRI 验证器与多任务 RL

审查日：2026-10-15。方法：实际读取全局 `/Users/vv/.agents/skills/norman-paper-reading/SKILL.md`、`references/novelty-and-transfer.md`、`references/evidence-audit.md`；随后独立检索原论文 Methods、Results、消融、附录及可见官方代码入口。在形成本文判断前未读取本轮 Claude 或其他 Agent 的新结论。本文是对方法优先权的定界，不是 13 篇论文的逐篇完整重审，也不是模型实验结果。

## 研究对象与方法指纹

项目的终极问题是：单一模型能否基于同一患者的多时点、3D、多序列脑 MRI，给出有影像证据支持、可复核且对扫描协议变化稳健的变化判断；训练方法拟借鉴 Vero 的任务路由奖励与均匀混合。当前 Dallas ds004856 实证仅有 2 人、4 次访视、8 个真实体数据，证明访问、配对和渲染可行；未有训练、临床变化金标准或重复性阈值，见仓库 `results/EXECUTION_STATUS.md`。不可把现有数据可用性误写成研究终点。

拟议机制拆开才可检验优先权：(A) 有纵向身份、时序、采集协议和空间证据检查的验证器；(B) 多 MRI 任务路由 reward；(C) 固定均匀/自适应任务权重及有效梯度批次构造；(D) 3D 定位或区域变化的 RL 优化；(E) 下游患者级、跨扫描仪的脑影像变化性能。输出形式、监督标签、推理输入（是否给历史 FreeSurfer 或临床资料）属于方法指纹，不能仅凭“GRPO”或“纵向 MRI”归并。

## 检索记录及范围

2026-10-15 检索 arXiv HTML/论文、PMLR、CVPR 开放论文、作者 GitHub 入口，组合查询：`MT-GRPO 2602.05547`, `MedLoc-R1 2603.28120`, `Vero uniform task mixing reward router`, `MedGRPO reward normalization`, `longitudinal brain MRI verifier acquisition scanner protocol`, `3D medical visual grounding reinforcement learning GRPO`, `longitudinal brain MRI verifiable rewards RL`, `LoV3D verifier longitudinal`, `BrainTRACE longitudinal evaluation`, `MedVol-R1`, `MedGround-R1`, `BrReMark`。这是针对已知技术及近邻的搜索，并非整个文献宇宙的穷尽搜索。部分官方代码入口能确认存在，未逐文件审计训练实现；代码一致性结论标为 **UNVERIFIED**。

## 近邻先例：方法而非题名比较

| 原始来源及核验位置 | 与本项目关系 | 已直接观察的机制与证据 | 边界/不应声称 |
| --- | --- | --- | --- |
| [LoV3D, arXiv 2603.12071](https://arxiv.org/html/2603.12071), §2.2–2.5, §3, Table 1；亦见 [PDF](https://arxiv.org/pdf/2603.12071) | **Same task + related verifier，Verified Direct Prior Art**：纵向 3D 脑 MRI 的结构化变化、解剖与临床一致性验证 | 3D 编码器接 LLM；模型可见前访视解剖标签，当前 FreeSurfer 量值只用于 verifier。C1–C3 检查区域文字/标签、方向及阈值一致性；按区域 Z 分数给容差，复合分数选偏好对做 DPO。§3 使用患者级 ADNI split，且包含纵向方向评价。 | 不得把“首次纵向 3D 脑 MRI 自动验证”当创新。其 DPO、病程假设、前次标签及临床元数据与拟议纯影像 RL 不同；采集变化和独立生物真值未由此解决。 |
| [BrainTRACE, arXiv 2610.06571](https://arxiv.org/html/2610.06571), §3, §4, Appendix G | **Same task + different method，Verified Direct Prior Art**：报告支撑的纵向、多序列、3D 脑 MRI 证据链评测 | 病例/访视对齐，隐藏报告及量值参考，L1–L5 操作与广义/3D/病例三轨；§4 报告多模型失败，Appendix G 明示报告并非像素标注、量值有采集/配准误差。 | 不可再称首次纵向脑 MRI VQA/评测；它并未证明某种 RL 改进，且原始影像另受 MR-RATE 获取条件限制。 |
| [Vero, arXiv 2604.04917](https://arxiv.org/html/2604.04917), §3–5, Table 2/4；[训练说明](https://github.com/zlab-princeton/vero/blob/main/docs/TRAINING.md) | **Same method + different task，Verified Direct Prior Art**：任务路由奖励、均匀视觉任务混合、GSPO | Table 2 在其五域混合中均匀权重平均增益最高；Table 4(b) 路由奖励高于统一 math_verify；Table 4(c) GSPO 略高于 GRPO/DAPO，§5 有等样本混合与单域对照。 | 均匀比例只是该数据/指标下的经验结果，非 MRI 的优化定理；不能把简单迁移称新算法。其等样本对照仍需在 MRI 复现有效梯度量、患者级结果。 |
| [MT-GRPO, ICML 2026/PMLR 306](https://proceedings.mlr.press/v306/ramesh26b.html), [arXiv v3](https://arxiv.org/html/2602.05547), §2–5, Eq. 4–8, Algorithm 1–2, Appendix D.2/D.4/D.5 | **Same multi-task RL mechanism + different task，Verified Direct Prior Art**：比“均匀更公平”更近的算法竞争者 | 学习任务权重 `z`，以任务 reward 与改善量更新；过滤零 advantage 后 ratio-preserving sampler 保持实际 batch 比例。Appendix D.2 单独消融 reweight 与 sampler：均匀 RPS 已显著强于 DAPO，而只自适应加权提高 worst task 却损平均分；D.5 明示逐步重采样开销。 | 固定均匀输入占比≠均匀有效梯度；在 MRI 提出“控制有效任务份额”须对照 MT-GRPO/RPS。PMLR 正式版与 arXiv v3 可能有差异，本文具体式/数取可访问 arXiv v3；正式版细节一致性 **UNVERIFIED**。 |
| [MedGRPO, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Su_MedGRPO_Multi-Task_Reinforcement_Learning_for_Heterogeneous_Medical_Video_Understanding_CVPR_2026_paper.pdf), [arXiv HTML](https://arxiv.org/html/2512.06581), §3.2–3.4 Eq. 1–3, §4/Table 3 | **Same medical multi-task RL, different organ/input**；奖励校准 **Verified Direct Prior Art** | dataset-task 基线 median/IQR 经过 sigmoid 变换；Table 3 去除归一化训练崩塌；使用 GRPO、医学 LLM judge。 | 不能仅把跨任务 reward 标度校准称新。数学注意：组内标准化对同一组的正仿射缩放不变；MedGRPO 的非线性 sigmoid 可以改变组内相对差异，但为何改善不能只归因于跨任务均值对齐；须实测优势分布/过滤率。 |
| [MedLoc-R1, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Yang_MedLoc-R1_Performance-Aware_Curriculum_Reward_Scheduling_for_GRPO-Based_Medical_Visual_Grounding_CVPR_2026_paper.pdf), [arXiv](https://arxiv.org/html/2603.28120), §3 Eq. 3–8, §4.3/Table 2–3；[官方仓库](https://github.com/MembrAI/MedLoc-R1) | **Same medical spatial RL + different 2D targets**，reward sparsity 的 **Related Prior Art** | 固定 IoU 阈值使组内全零；以滑窗命中率、稳定性、IoU margin 渐进提高阈值。HAM10000 消融去掉某判断降低 A@0.5。 | 3D MRI 定位不能只套 IoU reward 声称创新；动态阈值也已有。其三数据为皮肤/骨 X 线/超声 2D，并未解决体素坐标/时序配准。代码仓库入口可见，训练代码逐文件核验 **UNVERIFIED**。 |
| [MedVol-R1, arXiv 2605.26621](https://arxiv.org/html/2605.26621), §2.3–2.5 Eq. 2–3 | **Same volumetric RL family + different task**，3D evidence reward 的 **Verified Direct Prior Art** | 多切片输入，LVLM 选关键轴位片与 2D box，冻结 MedSAM2 跨切片传播 3D mask；GRPO reward 涵盖片选择、box 和跨切片一致性。 | 不可称首次以 RL 奖励 3D 医学证据；本文主要是分割，不是跨访视变化。是否其输出具有原生体素级验证要看目标定义，不能等同于本项目。 |
| [MedGround-R1, MICCAI 2025](https://papers.miccai.org/miccai-2025/paper/1222_paper.pdf), [arXiv](https://arxiv.org/html/2507.02994), 方法及 §3.2 | **Same spatial RL + different task**，**Related Prior Art** | 医学图像 grounding 的空间/语义 GRPO，含 M3D-RefSeg 数据。 | “医学空间 GRPO 从零开始”不成立；未确认其纵向 3D 脑 MRI 或采集漂移验证。 |
| [BrReMark, arXiv 2606.25894](https://arxiv.org/html/2606.25894), §2.4 Eq. 1–2, §3 | **Same brain MRI RL + different longitudinal task**，**Verified Direct Prior Art** | 脑 MRI 异常 box、描述、诊断；format+IoU+LLM judge 奖励，并在模态认错时关闭语义奖励。 | “脑 MRI 使用 GRPO/模态门控奖励”不是新贡献；其主要任务为横断面异常，未建立纵向患者变化。 |
| [Longitudinal ComBat, NeuroImage 2020](https://www.sciencedirect.com/science/article/pii/S1053811920306157), 方法/模拟与 ADNI 应用；[脑 MRI 测量可靠性研究](https://pubmed.ncbi.nlm.nih.gov/34848299/) | **Same acquisition confound + different method**，**Related Prior Art** | 多扫描仪加性/乘性效应会影响纵向皮层厚度；重扫、序列、处理流影响测量。 | “发现扫描协议混杂”不新；新 verifier 至少须在匹配/不匹配采集和 test-retest 条件下证明比已有质控/校正更可靠。 |

## Claim–Evidence ledger 与直接判定

| 主张 | 证据类型及位置 | 判定及缺失控制 |
| --- | --- | --- |
| “均匀多任务 RL 是在 MRI 上有优势的方法创新” | **作者观察**：Vero Table 2 其视觉五域均匀混合平均优；**直接观察**：MT-GRPO Appendix D.2 在非 MRI 任务中均匀 post-filter RPS 有效，自适应权重改变 worst/average tradeoff。 | **未证实于 MRI**。需同预算比较按数据比例、固定均匀、post-filter 均匀、MT-GRPO、MedGRPO 校准；同时报最弱任务而非仅宏平均。 |
| “结构化纵向脑 MRI verifier 是空白” | **直接观察**：LoV3D §2.2–2.5 已做代码可验条件、区域/纵向奖励、容差与 DPO。 | **被直接先例推翻**。剩余空间是独立影像可见证据、采集/配准/重复性不确定性以及必要控制的严格测试，而非“有 verifier”。 |
| “MRI 空间 RL 有方法首创性” | **直接观察**：BrReMark §2.4、MedVol-R1 §2.5、MedLoc-R1 §3 及 MedGround-R1 方法已有 GRPO + 定位奖励。 | **被先例压缩**。3D 体素坐标、跨访视对应和校准拒答若有新算法与实证才可另论。 |
| “BrainTRACE 指出真实缺口” | **作者观察**：BrainTRACE §4 的闭式/开放式/病例得分差距；**作者限制**：Appendix G 提醒报告不是独立影像重读。 | 支持**评测难题存在**，不自动支持某 RL 算法可解。必须防止仅学习报告措辞或标签泄漏。 |

## 三个候选研究卡（按优先次序）

### H1：带重复性约束的 acquisition-aware 纵向影像验证（最高科学价值；创新风险中）

- **Unresolved Gap / Scientific Question**：LoV3D 有结构逻辑验证，但原始影像测量和跨扫描协议差异能否被验证器区分仍未确立。影像可见的真实改变能否与配准/协议伪变化分离？
- **Falsifiable Hypothesis / Potential Method**：将同患者身份、访视时间、序列/体素参数、注册误差、test-retest 不确定性映射为可审计的资格门与“变化/稳定/不可判定”输出；只有超越预设重复性界限才奖励方向。拒答不是错误答案的廉价替代，应单独评估 coverage-risk。
- **Minimal Experiment / Baselines**：先建立独立验证器研究，不训练 RL。取得较大患者级纵向 T1/FLAIR 队列与同日/短期重扫、人工或高质量独立量值；预注册可比较扫描对。和原始差分、配准后差分、FreeSurfer/LoV3D 式阈值逻辑、采集元数据简单门控比较；注入患者互换、访视互换、协议漂移、下阈值微差和影像无变化负对照。Dallas 2 人只用于流程烟测。
- **Evaluation / Success**：以患者为单位测错误方向率、不可判定覆盖率、校准、跨协议亚组、错误接受率及 bootstrap/分层不确定性；与最强门控基线相比，在固定覆盖率下显著降低错误变化，且对真正改变保留合理敏感性。
- **Kill Criteria / Next Action**：如 test-retest 分布无法可靠估计，或简单协议门控/重复性阈值达到同等性能，或验证器只在合成扰动有用而真实访视无增益，则不声称新方法。下一步先扩 Dallas 与重扫来源，制定数据字典及资格标准。

### H2：不确定性感知的有效任务份额（AI/ML 价值潜在较高；创新风险高）

- **Gap / Question**：Vero 固定均匀采样不保证在 MRI 各任务产生均匀有效梯度；MT-GRPO 已解决零 advantage 后比例，但不处理影像标签可靠性差异。可靠性与任务进度联合控制能否胜过两者？
- **Hypothesis / Method**：在 MT-GRPO 的 post-filter 实际占比约束之上，再以独立估计的标签误差或资格置信度限定每任务贡献；不允许用测试集结果调权。保持病例级采样，按验证器资格计有效任务份额。
- **Minimal Experiment / Baselines**：同初始化、相同患者分割、相同有效 rollout/token/GPU 时数下比较比例混合、Vero 均匀、MedGRPO sigmoid 奖励校准、MT-GRPO RPS/IWU、二者组合及本法；2–3 个真正有参考答案的任务，单任务及 SFT 上界/下界。先模拟标注噪声作机制测试，再在独立 MRI 测试集中验证。
- **Evaluation / Success**：预定义 worst-task 正确率/定位与宏平均，零 advantage 比率、有效 batch 占比、reward calibration、患者级置信区间与训练成本；需在现实噪声和外部协议变化下优于**最强**控制且无另一任务实质退化。
- **Kill / Next Action**：若优势消失于 post-filter RPS、简单噪声过滤或相等有效计算预算，或仅在合成数据成立，则停止算法创新主张。下一步先测基座模型在各 MRI reward 中的零方差与噪声分布，确定是否存在可优化问题。

### H3：纵向 3D 证据的反事实一致性（科学/AI 交叉价值；创新风险中高）

- **Gap / Question**：MedVol-R1 的跨切片一致性与 LoV3D 的纵向逻辑均不足以证明模型真的使用正确访视和空间证据。能否使一个错误证据链无法得到高 reward？
- **Hypothesis / Method**：输出须指向可回映原生体素坐标的局部体积证据，并在患者/访视互换、时间反转、邻近无关切片、模态移除等反事实下满足预期的答案改变/稳定约束。先做验证器判别，再考虑引入训练 reward。
- **Minimal Experiment / Baselines**：来自独立区域 mask 或专家标注的真实配对病例；对照标准答案 reward、2D IoU、LoV3D 式结构检查、MedVol-R1 式多切片一致性。报告正确案例、干预后应改变案例及无关扰动不应改变案例。
- **Success / Kill / Next Action**：预定义证据位置与方向同时正确率，以及对干预的敏感性/特异性；如果反事实 reward 被坐标模板或单帧捷径满足，或在真实影像上不超过结构化基线，则停止。下一步核验可得的人工 mask 与时点配准质量。

## 最终 scoped novelty 判断

已核实的**直接先例**覆盖“纵向 3D 脑 MRI 结构化 verifier”（LoV3D）、“纵向多序列脑 MRI 证据评测”（BrainTRACE）、“医学/脑 MRI 空间 GRPO”（MedVol-R1、BrReMark 等）、“均匀视觉多任务奖励路由”（Vero）、“医疗多任务 reward 归一化”（MedGRPO）及“过滤后有效任务比例控制”（MT-GRPO）。因此，原样组合的算法首创性主张应 **KILL/REFORMULATE**。上述三个更窄的机制在本次搜索中为 **No Direct Match Found**，但这是限定于记录和检索词的判断，特别是 H2 与 MT-GRPO 关系极近；不得写成“全球首次”。目前所有 H1–H3 的 MRI 增益均 **UNVERIFIED**，需按研究卡先建立可靠目标和相等计算的反证实验。
