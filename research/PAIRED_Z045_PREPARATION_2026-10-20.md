# Vero-to-MRI：固定 z045 的配对模态曝光修正

## 本轮状态

`research/vero-mri-paired-z045-v1` 继承已验证 `d43379d9e6e8a62d10c755f121862c482afc268a`。这是 CPU 数据、执行代码与前瞻评价准备：**模型加载0、生成0、真实训练0、optimizer updates0**。没有启动历史队列、重跑18RL/3SFT/final100，没有修改 GSPO loss、reward、LR、LoRA 或历史权重/Adam/RNG。

研究标签：`C_TASK_ELIGIBILITY / ENGINEERING_UNBLOCK / PROPOSED_NOT_RUN`。目标是检验同患者同层面双模态支持是否改善 sequence 尤其增强T1召回，同时保留FLAIR和Dallas表现。病例采集模态识别不是病灶诊断；不额外要求放射科专家诊断才能使用已认证采集标签。Dallas仍为 `ENGINEERING_ONLY` 身份对应，不冒称第二个科学MRI推理任务。

## 1. 已实际构建的数据

| 项目 | 历史序列池 | 新配对序列池 |
|---|---:|---:|
| TRAIN患者 | 28 | 同28 |
| 层面 | z045 | 仅z045 |
| 记录 | 28 | 56 |
| FLAIR / 增强T1 | 14 / 14 | 28 / 28 |
| 每名患者模态 | 一种、1条 | 两种、各1条 |
| 有两模态的患者 | 0 | 28 |

原28个图像/标签成员保留；使用旧审查生成的同像素 opaque 文件，文件路径和记录元数据被单独版本化。原28和76条历史文件原字节保存。48条Dallas保持原始字节与对象内容不变，mixed候选为104条。没有引入z091/z136、新患者、新任务或新图像增强。

56条全部与既有来源NIfTI/annex/sidecar/像素等价审查绑定，并实际复查当前PNG SHA、364×436 RGB文件头、sidecar SHA、同患者/层面/模态关系。复用z045的28组既有非专家覆盖判断，没有重渲染、重盲审或新解码MRI像素/体积/mask。官方采集说明的post-Gd run-02_T1w与FLAIR来源支持目标标签。强度/采集对比可作为合法序列判别信息，不宣称解剖推理或排除了site/protocol捷径。

旧池中模态可被患者键完全查表记忆，新池这种TRAIN重代入查表上限由1变成.5；这是数据列联表性质，**不是模型曾使用患者捷径的因果证明，也不是准确率结果**。

## 2. 实际曝光与唯一数据变量

沿用Adam60已完成的uniform seed13阶段：旧8组中2组sequence、6组Dallas；后续固定原任务槽位 `[8:40]`，不是把采样器重置到 `[0:32]`。这32槽包含18sequence、14Dallas。两类的逻辑权重仍各.5，有限前缀不强制16/16。

序列源流复用既有 `build_category_schedule`：128次/类的抽取、耗尽重洗牌、完整256逻辑块洗牌，然后过滤sequence子流、跳过已发生的2次，映射至固定任务槽。对原76池生成的控制逐条恢复原source occurrence（仅新顶层occurrence ID不同）；新104池使用同一已有实现。所有Dallas槽内容逐条一致。没有新独立row RNG、配额、强制配对邻接或根据模型表现挑图。

这是**任务槽位匹配的新数据池绑定**，不是官方采样器旧row RNG的无缝恢复。模型的Python/MLX RNG、Adam状态另行完整继承。初版独立sequence queue会改变原逻辑块的早期重复分布，已用反例纠正并保留PRE_REVIEW私有预览，没有用于任何模型。

完整32槽的实际候选曝光：18次sequence=8FLAIR+10T1，16张唯一序列图像、2次重复、12人，其中4人两种模态均曝光。原池同槽也是8/10、16图/2重复，但为16人、0双模态。因此此小预算不保证看完56条或全部28对；若先达到8次更新等限制，实际曝光更少。不得把扩大的候选池当作实际全部训练量。

## 3. 预先固定评价，不打开旧final100

主指标为严格Vero自由生成的sequence balanced accuracy，分别报FLAIR和增强T1 recall、invalid及混淆矩阵。保持原system/question、`<think>/<answer>`与boxed格式、temperature0、max_tokens256、严格format=1；不使用简短prompt、放宽parser或512预算。

保存后统计实现为独立 `paired_z045_saved_evaluation_v1.py`，已用已知混淆、截断、tie、错误alias、不同数值合约及成本越界的合成CPU反例验证。它只接收已经由外部实际exit/closed-log/output/source/scope审核认证的完整输出；内部状态布尔字段不能代替原生状态审计。尚未运行任何真实父子统计。

另报同一共同答案前缀的candidate mean-label balanced accuracy和正确减错误margin；sum和first为次要读取方式。mean不是类别概率。Dallas16条报告accuracy及same/different各自recall；equal-task宏、最差任务与患者/配对cycle聚类不确定性分开计算。固定2000 bootstrap、seed7，不能把旧开发集描述性区间称为独立泛化。

现有开发输入：24sequence/4人、16Dallas/4cycle，共40个唯一输入。8个z045交换对照的图像和通用问题，恰好等于对应另一模态的真实输入，故**复用其输出，新增交换推理0**。报告预测变化、两侧均正确、显示图像真值正确率及paired margin separation；“回答改变”不等于正确。

父Adam60和唯一有限终点必须采用**同一个新数值合约、输入、prompt、解码和评分**；不能把旧BF16父结果与F32子结果相减称为训练收益。每checkpoint40greedy、40candidate查询；候选至多80类评分+40prefix oracle=120次评分API。两个checkpoint共80greedy、240次评分API，上限20,480生成tokens；候选输入位置单列最多102,000（条件于已资格化的输入长度和固定前缀/标签），不与训练ledger冒充相同成本单位。

已有相同合约父输出若能真实认证则复用，禁止无理由重复基线。终点只选预算先到时的最终checkpoint，不按dev挑最佳中间步。候选不同长度/oracle的数值资格仍须完成，不因两个旧17-token上下文通过而自动批准。

独立的 `paired_z045_matched_f32_eval_v1.py` 已实现CPU可准备、未来精确授权才可执行的原生入口：同F32 greedy与候选label scorer、完整prompt/image绑定、候选共同prefix及独立oracle两道`.002`门槛、串行父子加载、全部参数/Adam不变与RNG恢复/释放审计。未知终点时仅生成 `PENDING_EXACT_COMPLETED_TERMINAL` inspect范围，不能执行。旧backend的BF16评价接口仍明确拒绝，未改动已冻结的资格诊断源码。原生实际评价资格和输出均为 `NOT_RUN`。

`scripts/analyze_paired_z045_saved_v1.py` 提供保存后入口，必须先收到实际exit0、闭合stdout、完整输出、scope、父子状态/释放审计的外部hash绑定及独立completion-spec复核，才执行聚合算术。未来缺失hash或临时把状态字段写成true，不能作为已完成证据。本轮未创建真实completion-spec或执行许可。

GO条件必须同时满足：sequence严格balanced accuracy、增强T1 recall和mean candidate balanced accuracy都提高；FLAIR recall、Dallas accuracy及same/different recalls均不下降。否则 `STOP_SAME_CONFIGURATION`，不追加相同配置训练。GO也不是自动扩训授权。一次父子工程小试不能把额外优化与配对数据的因果效应完全分开，没有额外matched旧池训练或多seed被授权。

### 新独立患者验证

状态 `NOT_ESTABLISHED`。前瞻方案：在来源元数据上排除所有历史train/dev/final/feature/probe/裁决曝光患者及可确认的重复源，按8个既有站点各2人选择16个全新患者、每人z045两模态。站点内按固定salt `paired-z045-holdout-v1` 的SHA顺序选取，不看预测；若站点可用性不足，先修改并冻结可用性规则，再看任何输出。

目前缺完整的全历史身份排除并集和新配对payload资格证据，公共目录指针不等于未曝光或图像合格。没有下载/声明这个验证集已经建立。现有TRAIN/反复使用dev仅作工程诊断；old final100不用于任何选择。Dallas现有dev只能衡量工程保留，不能产生新的独立患者推理结论。

## 4. 父权重与有限预算

父checkpoint为实际uniform阶段 `boundary_0008`：Adam60、absolute groups129、adapter SHA `1741ff2feb2f569cfe062b92cb79b62f61f83463e6683a928d2c458a5f547310`。选它仅为52→56→60连续性，不是旧final100或dev表现最好。B28、Adam52、56、60全部保留。

未来原生入口需要验证全部adapter/fullAdam/Python+MLX RNG、配置/梯度签名兼容并恢复，不能仅加载LoRA或重置Adam。本轮只认证小receipt/manifest和已有独立证明，未重读大型参数数组；未来实际执行仍须重新逐项验证。

一次新阶段取先达到的限制停止：**32 attempted groups、8 accepted updates、70,784 charged tokens、3000秒操作启动截止**，仍为K4/max256。完整32槽prompt预留38,016、response预留32,768。213/405展开输入长度是旧原生测量+相同尺寸/文本的条件推导；新的processor展开和视觉tokens必须原生重核，若不同则停止而非静默扩大cap。CPU实际tokenizer文本：问题27、标签2/5、固定canonical target16/19。

3000秒从本实验最初启动计时，绝对截止journal跨进程恢复不重置，包含等待和HOLD；仅在完整checkpoint边界停止，保留60秒开组余量。最后原子组/保存/释放可能越过截止，**不是硬杀进程的50分钟保证**。每组提前预留，pending measured/imputed工作只计一次；恢复不重复更新。既有AC/内存/pressure/绝对与阶段swap/热与异常门槛、全局flock、单模型限制保持。

## 5. F32是独立数值前置条件

原历史BF16路径保留；独立F32激活/缓存/输出后端使用已有已核验的full scorer，保留原生成采样、EOS和流式detokenizer，捕获实际on-policy selected logps，在任何更新之前比较rollout/scorer，并委托原GSPO loss与Adam实现。新代码的CPU fixtures、source checks或两个旧上下文PASS都不能当作新原生资格。

当前只有CPU实现，**完整on-policy生成/KV/full/GSPO反向的原生资格未运行**。新训练入口必须先认证实际零更新资格证据及独立复核，再验证新的明确human训练授权和精确scope；不能编辑一个JSON PASS就激活、不能退回旧BF16。

唯一应先考虑的模型执行是独立、最多两条预先固定TRAIN输入的零更新F32资格诊断；这仍需精确源码/预算复核和另行明确授权。资格真正通过后，才请求一次上述有限续训的授权。本轮没有生成授权、没有自动启动诊断或训练。

该只读资格范围已固定：每任务第一条TRAIN输入、每组K4，最多2048个返回response tokens、2080次language-model调用、24次vision embedding、16次full scorer、8次只读GSPO反向，optimizer updates=0，1200秒开操作窗口并保留原子收尾。安装包AR的额外lookahead单列，最多2056次sampler draws；不把lookahead当作返回token。源绑定的独立静态工程审查不是实际数值PASS、不是human授权。若真实采样恰为inactive/zero-variance组，则记录PARTIAL，不额外重采样或构造advantages。

## 6. 耗时与计算成本估计

依据真实uniform历史资源时间轴：7个完整相邻组起点间隔平均31.57秒、中位25.55秒、p90 50.73秒；第8组仅有≥47.01秒截尾记录，不伪造完整均值。加载至首组约60.36秒。旧8组共12,957 charged tokens/4更新；这不是新池接受率预测。

条件于接近旧速度，完整32组约15–30分钟；8更新上限可能使其更早结束，F32新延迟尚未实测。新阶段3000秒是50分钟开组截止；HOLD或原子末组另记。历史一套40greedy+25candidate/null资源跨度约300秒，不等于新流程精确wall。新增父子40candidate的评价规划约10–30分钟，总评价开组窗口最多40分钟。训练加父子评价规划约25–60分钟；保守预算约90分钟加原子收尾，**不含独立F32资格、用户等待或后续新数据收集**，不是置信区间或完成承诺。

## 7. 证据与状态

- `results/PAIRED_Z045_DATA_V1.json` 和独立数据复核：实际56清单、来源/旧证据绑定、旧32控制恢复和真实曝光。
- `results/PAIRED_Z045_EVALUATION_PREPARATION_V1.json`：固定40输入/8复用aliases、严格指标、GO规则及新holdout缺口。
- `results/PAIRED_Z045_HISTORICAL_BUDGET_INDEPENDENT_V1.json`：历史真实token/时间、未来cap条件，不使用Git commit时间。
- `results/PAIRED_Z045_EXECUTION_PREPARATION_V1.json`：CPU编译的Adam60/数据/源码/配置/预算绑定，`NOT_RUN`。

所有图像、患者键、原始label清单、原始回答、token/梯度及权重/Adam/RNG留在忽略目录。独立reviewers为隔离Codex；`CROSS_MODEL_REVIEW=NOT_VERIFIED`，不假称Claude批准。`READY_FOR_TRAINING=NO`，关键剩余前置条件是新原生数值合约；独立验证集尚不能支撑泛化声明。本阶段没有新的准确率提升结果。
