# Vero-to-Brain-MRI：停止扩训后的CPU根因核查与修复

研究标签：`PRIMARY_VERO_TRANSFER / C_TASK_ELIGIBILITY`。基线为main `6ee3d1668f17189b24eaf1e850644a2cfbe5f69b`；独立分支 `research/vero-mri-root-cause-cpu-v1`。本阶段实际模型训练、原生模型推理和新增模型optimizer updates均为0。没有重新执行原始实验或final100，也没有从final100选择数据、算法、prompt或checkpoint。

## 1. ROOT CAUSE RANKING

以下区分“已确认的准备／机制限制”和“已证明导致准确率失败的唯一因果根因”。后者尚未得到证明。

| 排名 | 判断 | 实际支持 | 尚不能声称 |
|---|---|---|---|
| P0：任务与监督设计不足 | **CONFIRMED：准备限制；LIKELY：主要贡献因素** | 主池只有28条单模态／单层面序列＋48条whole-head身份对应；序列全局14/14平衡却每患者只有一个真值；缺少两类独立完成视觉资格审查的MRI任务 | 二分类天然不可学、所有图像不可答、单一数据原因已解释全部失败 |
| P1：有效判别信号稀疏且准入受格式耦合 | **CONFIRMED：实际机制；LIKELY：学习效率限制** | 2,286组只接受362次更新；其中348组同时含有效正确／错误类别；另101个具有此类判别差异的组被组中其他响应的格式问题连带拒绝 | 放宽门控一定提升MRI准确率、392组都可恢复、更多更新就是成功 |
| P2/P3：完整GSPO执行路径与适配容量 | **UNVERIFIED：主要失败原因；部分配方差异CONFIRMED** | 官方与本地提取出的loss公式CPU梯度一致；本地单组单次更新缺少官方跨minibatch的stale-ratio阶段；量化、LoRA、冻结视觉、优化器、LR／warmup明显不同 | 已找到native GSPO公式bug、冻结视觉必然错误、CE图像梯度已证明完整GSPO视觉学习 |

生成上限256与本地长度惩罚16384之后才触发，是确定配置失配；但39/9,144条训练响应触顶，不能把它升级为全部失败的唯一原因。

## 2. DATA FIX：实际准备了什么

### 原主池为何只有28＋48

`two_task_pool.extract_frozen_sequence/assemble`直接继承已完成B28的28条序列schedule，明确要求14/14；这些记录均为z045、每患者一条。Dallas为12个两人匹配周期，每周期四个有向same/different组合，共48条。这个池是冻结对照选择，不是现有图像库存的上限。

官方 [DATA.md](https://github.com/zlab-princeton/vero/blob/025ca3c800fc95224c86323c1f7fd33dceed3090/docs/DATA.md) 的通用数据集筛选排除纯二分类和小于1,000样本的数据集，另要求图像与问题的相关性、明确可答性及可验证答案。我们的有限池未复现其规模、任务多样性和筛选证据。**二分类仍可成为合理识别任务，不能仅凭标签数判定无效**；官方问题筛选提示本身也允许视觉证据清楚的医学二分类问题。强度／对比度是模态识别的合法信息，简单直方图有效不能自动证明数据违规，但不能据此宣称解剖推理。

### 新候选清单：真实存在的TRAIN图像，非新训练集批准

- 从既有TRAIN注册表构建216条技术候选：168条序列＋48条Dallas；没有新增患者、下载图像或拟合分类器。
- 序列仍是28名患者，每人z045／091／136的FLAIR和增强T1，共84个同患者同层面配对；标签84/84平衡。168个PNG逐一解码、哈希和模型可见prompt／隐藏标签／来源记录核对。
- 原28条的经验`H(label|patient)=0`，患者多数类查表在训练记录上可达28/28；新168条为1 bit和84/168。配对消除了这一特定患者—标签列联关系。它不是未见患者模型准确率，也不证明历史VLM利用了患者身份。
- 八个既有来源元数据站点分组内，site×level多数类上限为0.5；这不等于完成跨扫描仪／协议控制或全局人员去重。
- 实际观察发现一个患者z136的两张图仅约1.92%和2.13%非零前景，已逐行标记稀疏解剖警告。5%仅为工程描述，不是临床排除阈值；不把非空像素等同于可答。
- Dallas48对的左右拼接像素与既有48个source panels逐一一致，source panel文件字节哈希已核对，BIDS身份／visit／两人周期标签有效；whole-head可见face/skull/neck，brain-only可答性和扫描仪／站点控制仍未验证。
- 新候选**科学资格批准行数为0**，意思是尚无充分独立资格证据，不是证明216张图均不可用。`training_rows()`拒绝任何非空候选包，篡改`INDEPENDENTLY_VERIFIED`状态字符串也不能放行。

私有候选：`local_data/root_cause_cpu_v1/data/candidate_v1/`；清单SHA256 `e33259ef98ecdb4e8931edcf8648d2044cbc80cecbee15554551e65d5234e258`。公开证据：`results/ROOT_CAUSE_DATA_TRAIN_ONLY_V1.json`。

### 病灶／定位为何不准入

既有TRAIN几何清单包含tumor presence84行（46yes／38no）、enhancing presence84行（31yes／53no）和tumor grid46行（六个占用网格类）。它们仍缺少独立image-linked可见性、歧义／拒答标注、临床合理阴性、坐标框架与多灶／边界裁决以及source/site/level捷径控制。非零mask是几何证据，不是可见病灶真值。没有把这些行伪装成第三类合格任务。

可复现builder仅读取固定TRAIN资源、TRAIN PNG及Dallas panels/QC；不读final100。探索定位文件时曾解析混合source-heldout QC／internal-dev字段所在JSON的完整字节，只选择TRAIN记录和控制字段；未使用非TRAIN像素或测试输出。不能把这段探索描述为“从未读过任何混合元数据字节”。上游NIfTI重新渲染、mask独立重读和专家可答性裁决本轮均未执行。

## 3. REWARD FIX：精确重放，不凭感觉改奖励

已重放18个arm×seed的全部2,286个TRAIN group、9,144条响应；certified streaming／EOS解码、grade、sample-SD advantage、accept/reject、更新游标及token债务逐一匹配保存回执。

| 实际组数 | 序列 | Dallas | 合计 |
|---|---:|---:|---:|
| 尝试组 | 1,139 | 1,147 | 2,286 |
| 接受更新 | 193 | 169 | 362 |
| 首个拒绝原因：非全格式有效 | 347 | 219 | 566 |
| 首个拒绝原因：task accuracy恒定 | 599 | 759 | 1,358 |
| 仅格式改变total reward的组 | 76 | 98 | 174 |
| 被格式拒绝、但含有效正确和错误类别 | 61 | 40 | 101 |

全局接受率15.84%，但不能用它衡量图像理解。174个format-only组原门控已经全部拒绝，没有证据支持“历史optimizer主要执行纯格式组”。

392个被格式拒绝且native accuracy分数变化的组必须拆开：101个真实有效类别混合、287个只因格式无效被强制记0分、4个正格式非类别答案不明。362个已接受组中348个含有效正确／错误类别，14个是正确类别与完整格式非类别答案的对比。后一类并不是确认的视觉判别学习。

**未复现历史reward／advantage实现bug，因此没有主观改写0.8accuracy＋0.2format或历史门控。** 新离线分析器修复了证据分类：非零advantage、格式变化、合法类别对比与实际图像依赖不再混称“有效MRI学习”。101个被拒绝组是候选信息损失证据，不是未经试验即可恢复的训练收益。

在已接受组上，format固定为1且overlong相等，二值accuracy的严格单调calibration只剩两个值，等价于正仿射变换。相同保存组的raw／calibrated advantage最大差为`4.133233401093506e-6`（epsilon=1e-6）；去掉epsilon仅`4.440892098500626e-16`。因此不能把此控制视为充分区分非线性MedGRPO机制的实验。

### 实际代码修复：长度契约拒绝静默失配

本地历史direct profile为max18432、buffer2048，首次负惩罚在16385；0..256共257个实际公式值全部为0。新增版本化`validate_length_contract()`拒绝声称enabled但在rollout cap内不可达的配置；要么显式记录disabled，要么另行论证可达契约。精确边界、非法预算和显式模式均有CPU回归测试。

没有自动选择新的buffer、缩放惩罚、放宽格式或更改历史结果；新校验尚未接入原冻结训练器，也不授予任何训练许可。官方当前配置还存在reward max16384与Qwen data cap18432分开的事实，源码默认首次负值14337；不能把本地16385误称为官方实际运行参数等价。

### 任务投入：不要把token记账当视觉FLOPs

- Charged tokens3,012,459＝completed3,010,292＋measured partial845＋保守imputed1,322；两个中断尝试各计一次。
- 序列／Dallas charged tokens分别1,070,270／1,942,189。逻辑uniform block的有限前缀不保证实际attempt或accepted更新完全均衡；逐arm×seed×task表已公开。
- 展开image-pad位置每完成组计一次分别175,406／330,336；乘实际K4响应共2,022,968，再有已知partial576。它们是prompt位置计数，不是测得的vision encoder总算量。
- 按冻结源码和实际组数派生9,144次generate调用，另1,448 old＋1,448 current＋1,448 gradient likelihood模型调用。实际缓存、重复视觉编码、未知partial视觉工作及FLOPs仍`NOT_VERIFIED`。
- 所有362个接受回执均保存正且有限的gradient L2；逐arm×seed×task min／median／max见`ROOT_CAUSE_SAVED_GRADIENT_NORMS_V1.json`。范数不能相加冒充任务净梯度、视觉贡献或Adam方向。

## 4. VERO PARITY与CPU反例

官方和历史本地pin均为`025ca3c800fc95224c86323c1f7fd33dceed3090`。已读官方DATA／TRAINING、过滤脚本／prompts、配置、sampler、actor与GSPO loss。

| 机制 | 官方源码默认 | 本地实际 | 判断 |
|---|---|---|---|
| Rollout group／任务 | K8，六类，每batch加权配额 | K4，两类，串行有限前缀 | 确定配方差异，不等于代码错误 |
| 优化批次 | 256 prompts，128 prompt minibatch，默认两个actor minibatch | 单on-policy group仅一次update | 本地clip初始不活跃；不能宣称完整官方优化轨迹等价 |
| GSPO公式 | sequence geometric ratio、detach surrogate、asymmetric clip | 提取的同构表达式 | CPU局部公式／梯度一致，未证实native视觉全路径 |
| 参数／优化器 | 训练配置vision不冻结、AdamW／bias correction、LR1e-6和40步warmup | 4-bit frozen base、language LoRA rank4、Adam无bias correction／decay，constant1e-5 | 确定不同；没有证据直接判定哪项导致失败 |
| 长度 | reward max16384／buffer2048，Qwen data cap18432 | direct max18432／buffer2048，真实cap256 | 确认配置差异和不可触发本地惩罚 |
| 输入一致性 | 分布式rollout／actor栈 | generate与teacher-forced scorer分别处理输入，检查长度及重算old=current | 缺少生成分布logp／KV／真实输入身份等价证据 |

新CPU机制fixture的实际结果：

1. forward ratio=1、group scalar loss=0，梯度仍为-1.5；冻结detach值的有限差分一致。**不能说ratio恒1使训练没有梯度。**
2. favorable stale mean-logratio移动±0.001时，窄clip使对照梯度变0。本地fresh group没有这段stale轨迹，属于调度差异，不是loss公式bug。
3. 只保留label-token敏感性的反例中，加入固定wrapper使长度2→10，梯度被sequence平均稀释5倍；这是两种公式共有行为，不是已测得真实格式梯度支配。
4. 个别响应具有非零图像导数／CE梯度，仍可能在正负group advantage加权后抵消。因此过去真实TRAIN pair的CE梯度不证明完整视觉GSPO信用分配。
5. K8使二分类混合成功概率增加，但在假想独立format成功率q=.8、类别成功率p=.5时，全格式门控的资格概率从K4的.3584降至K8的.16646144。不能盲目增大group。
6. 相同长度、不同prompt token IDs也可通过重算old=current；这是现有检查不充分的CPU反例，不是实际生成输入错位已发生的证据。

未发现足以正当修改历史GSPO loss的数值错误；也未证明所有原生路径正确。实际rollout/scorer分布一致性、KV-cache数值、图像条件GSPO梯度和完整视觉适配仍待另行允许的原生验证。

## 5. READINESS与未来父checkpoint

| 恢复问题 | 本阶段结论 |
|---|---|
| 至少两个科学可答且视觉依赖已验证的任务？ | **NO／NOT_VERIFIED**；216条仅技术候选，不能把source身份或非零mask当充分证据 |
| 患者／site捷径已排除？ | 特定患者—标签列联关系已修正；强度／协议／whole-head线索及可答性仍未解决 |
| 奖励可区分类别且能提供信号？ | 二值task可区分，348次更新有合法类别对比；门控损失101组；图像因果贡献仍未验证 |
| GSPO数值机制已通过？ | 小型CPU公式／导数通过，362个历史正梯度范数已核对；完整native视觉路径仍未通过等价证明 |
| 主要根因有唯一因果证据？ | 没有；准备限制和信号稀疏已确认，其相对因果贡献未隔离 |
| 现在可以启动训练？ | **NO**。即使上述门槛以后通过，也必须先汇报并等待研究者新决定 |

若未来坚持当前改善链的权重连续性，可预先声明已认证Adam60为父权重候选，理由仅是52→56→60链的既有终点，**不是最终测试挑出的优胜模型**。同架构／优化状态兼容时继承adapter、完整Adam和RNG；若目标或优化器阶段改变，须明确新阶段和状态处理，绝不重置基础权重或伪装成冻结臂连续结果。当前没有生成任何可执行训练启动命令。

## 6. 修复范围与证据边界

本分支的实际改动为：TRAIN候选builder／资格validator、完整历史reward replay、严格类别信息分类、长度契约validator、GSPO CPU反例及保存梯度范数摘要。所有历史模型源码、18RL／3SFT配置、checkpoint、adapter、optimizer和RNG不变。

独立Codex复核对数据、重放、公式和修复分别绑定源码／证据哈希；`CROSS_MODEL_REVIEW=NOT_VERIFIED`，不虚构Claude审查。公开仅提交非敏感代码、汇总和审查，不提交影像、患者ID、逐条回答、token数组或模型状态。方法未因CPU测试通过而获训练批准，真实准确率改善本轮**NOT_TESTED**。

Mission Alignment：保留系统Vero-to-MRI目标；用固定TRAIN来源追溯数据和实际学习信号；按旧记录复现而不重跑模型；只有可证实的校验／准备缺陷才修；不从final100调参；新训练、推理和模型optimizer更新均为0。
