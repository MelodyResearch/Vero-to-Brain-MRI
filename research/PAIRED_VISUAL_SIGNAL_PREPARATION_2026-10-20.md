# MRI 配对视觉信号：实际 CPU 结果与尚未运行的干预

## 最新实际原生结果（仅诊断，已结束）

研究者随后明确授权这一次评分及局部电池>=60%例外。最终独立review的scope_v4文件SHA `834be758cb6fc660d498b0e1a9f909dbf622647956accfeb0d7d7aab7dc651b2`，继承已有Adam60，实际exit0，外部72.012秒；12次TF评分/2,811输入位置，**0生成、0反向、0optimizer更新**。74次资源采样全NORMAL/CONTINUE、电量均94%。1942参数和完整Adam/Python+MLX RNG在restore前完全一致，恢复后仍相同；释放后model=None/active1052B。v1–v3均未执行，不重试。

| 固定TRAIN条件 | mean正确减错误margin | mean/raw/first预测 |
|---|---:|---|
| 增强T1真实图 | −1.321572 | 都是FLAIR，错误 |
| 同人z045 FLAIR图 | +1.405040 | 都是FLAIR，正确 |
| 同question无图参考 | 无图没有正确类别；FLAIR−T1=+1.964856 | 都是FLAIR |
| 原Dallas same图 | −0.643669 | 都是different，错误 |

两模态图的FLAIR−T1分离为mean+0.083468、raw+0.416683、first+0.416554。真实T1相对null的FLAIR偏置下降0.643284，但FLAIR图也下降0.559816；这是两个真实图相对文本NULL的共同分数偏移（也包含图像token/输入形状差异），再加较小的同形状模态图像特异差异；不能将real-vsNULL偏移隔离解释为图像机制，两张图都没有越过类别决策边界。全部候选prefix/oracle记录误差0（未保存原始prefix向量，独立复核验证的是已认证摘要与保存的first分数）。

**判定：** 对这一个TRAIN配对，错误不只是开放生成随机采错：共享canonical前缀下T1也已经低于FLAIR，并且三个读出均一致。存在较小图像条件差异，不能说完全不看图，也不能由此断言整个Qwen视觉编码器缺乏信息。既有8次回答全正常EOS，本例0/4不应归因于256-token截断。Dallas本轮没有swap/null，不能据此断言其视觉依赖。

没有本轮自由生成、权重训练或准确率改善。仅一个sequence患者/一条Dallas输入，不是独立泛化测量。配对辅助目标仍为显式监督候选，特别是单纯增加配对分离未必消除共同FLAIR偏置，因此当前不扩训。实际结果与独立复核见 `PAIRED_SIGNAL_ADAM60_CANONICAL_ACTUAL_V1.json` 和 `PAIRED_SIGNAL_CANONICAL_CANDIDATE_ACTUAL_INDEPENDENT_REVIEW_V1.json`。

## 实际完成

本轮模型训练 **0**、optimizer updates **0**。继承 `e345b89c8cd3b8c04a81f8448b3352824e923c3d`，不重跑旧 F32、B2/B3、18 RL、3 SFT 或 final100。

保存的八条 TRAIN 回答全部答错且正常 EOS；sequence 三条格式合格、一条不合格，Dallas 四条格式合格。读取实际 response 对齐的完整词表 logits，用 F64 logsumexp 重算答案位置：

| 条件 | 正确类别减错误类别的实际答案上下文分数 |
|---|---|
| Sequence 第一类别 token | −7.219707、−5.661572、−7.287600、−5.100401 |
| Sequence 完整 mean-label margin **上界** | −1.443698、−1.131150、−1.457242、−1.018068 |
| Dallas 单 token 完整 margin | −11.541479、−0.917667、−13.479046、−2.493296 |

因此，在这些**实际生成的思考历史**中，错误类别已占优，不能解释为“正确类别分数更高而随机采错”。Sequence 的另一条多 token 路径未生成，用后续 logp≤0 得到严格上界，没有编造反事实 logits。在这次保存答案分析时，共同 canonical 前缀下的真实图、同人换图和无图分数尚未测量；现由顶部这次已完成的诊断补充。不能据保存答案单独断言编码器或整个类别映射没有信息。

## 唯一候选干预：显式配对辅助目标

保留原 GSPO、reward、LR、LoRA 和原28+48池的实际主日程。只为 sequence 槽附加同患者 z045 另一模态的监督，模型提示不含患者、站点或路径身份。两图分别计算两个 canonical label 的均分，排除固定格式、前缀、EOS：

`d=((s_F(I_F)-s_T(I_F))-(s_F(I_T)-s_T(I_T)))/2`

`L_pair=softplus(-d)`；未来预声明 `L=L_GSPO+0.1 L_pair`，不搜索系数。

这是 **Vero-style GSPO 的显式监督辅助目标**，不是原始纯 GSPO reward。0/4 正确时不伪造正确 rollout 或 advantage，仍可通过真实标签提供配对判别约束。公共类别常数偏移被抵消；若参数的图像 Jacobian 相同，真实参数梯度也会抵消。CPU 反例同时证明：配对分离增大仍可能损害 T1 绝对 margin，强类别先验、图像特异捷径和 Dallas 共享参数干扰仍需实际评价排除。

原 gate 继续决定 GSPO 分支是否为零；新增监督分支若单独产生更新，必须单列为 auxiliary-only，不能计作原 GSPO accepted group。当前仅 CPU 可微表达式与数值接口，尚无原生训练调用。

“先验抵消”仅指最终score空间中两张图共有的加性类别偏移；不保证网络/logit先验改变后，受归一化、类别2/5个token及条件历史影响的非线性分数也抵消。

只实现独立参考数学、可微表达式、私有配对路由和成本准备，**未连接原生 trainer**。历史 B28 两更新独立 CE 的负结果保持；本方案不据此声称会成功。额外配对曝光与辅助数学没有曝光匹配控制时不能拆成单一因果解释。

## 曝光与成本

原强制配对32槽：18 sequence、14 Dallas、9对；8-update上限最快在第8槽停止，只接触2对。32-update上限只表示不由该计数提前截断，不要求32次更新，资源、时间、token仍可提前停止。两者主 rollout 预留均70,784；任何已接触图像都不自动算有效梯度曝光。

推荐的辅助目标使用原28池主32槽，18 sequence涉及16人；只在辅助端补齐16对、32张图。它不同时采用强制9对采样。单图串行两遍求值/反向，加独立 prefix oracle，完整辅助预算40,680 TF位置，合计111,464；不能声称仍在70,784内。

未来若获准，先原日程前4槽（2 sequence/2 Dallas），最多2次联合更新、最多4,096返回 rollout token、主计费9,040加辅助4,520=13,560位置、20分钟 operation-start上限，先到即停；无最少更新要求。这是 rollout输入/返回与额外辅助TF的约定账本，不是所有计算：原GSPO scorer/autograd重算、未返回lookahead和内部kernel另记，尚不能给出完整原生算力成本。10–20分钟只是含未测量反向成本的规划窗口。此前真实两组 F32 只读执行307.59秒且0 backward，不是新训练速度证据。父子完整40条工程评价按相同 F32 合约预留80 greedy、240 candidate API、最多20,480生成token与102,000 TF位置，40分钟操作启动上限；不是已运行成本。

## 评价、条件与决策

新16人/32图验证输入按固定7站点配额和 SHA 排序、排除54历史来源身份后取得实际 payload；它与旧40开发集及已打开 final100分离。所有新模型输出仍未运行。公开技术证明明确未知外部/跨来源历史身份、采集强度/站点等限制；不将16人当作临床泛化结论。

Root在未读左右模态映射的前提下首次目视检查全部32张新图：未见全黑/空图，4对有有限解剖覆盖、模糊或配对FOV不对称的质量标记。固定z045是低层面，不能称全脑覆盖或完美配准。保留全部预选病例、0替换；若以后评价，主结果仍含全部16人，可另按预先保存的质量标记分层。这是非临床视觉QA，不是放射科专家裁决。

未来父子都使用同一真实资格化 F32 合约，主报 strict greedy balanced accuracy、T1/FLAIR recall；另报 mean/raw/first candidate accuracy和 margin、同人换图方向、Dallas recall、macro/worst task、invalid与实际成本。没有 T1和候选改善，或 FLAIR/Dallas退化，就不扩训。Adam60只因完整权重/Adam/RNG链连续而作为预声明父模型，未用 final100优选。

实际新输入封存后，新增 `PAIRED_SIGNAL_FUTURE_MATCHED48_EVALUATION_V1.json` 将下一次主要评价固定为新32张sequence和既有16条Dallas工程保留输入，替代前述旧40工程评价的成本方案，不要求两套都跑。父子总96次greedy、294次candidate TF（含各1个无图参考）、最多24,576返回token和124,950 TF位置，50分钟总操作启动窗口，仍未授权/运行。全部16新患者固定纳入，不按结果删质量flag病例；预声明一次比较失败就结束该干预，不在新验证集上反复调参。终点checkpoint尚不存在，必须另绑真实terminal和独立审查，现有输入清单不能启动评价。

**READY FOR TRAINING=NO。** 原生 active F32 GSPO backward仍PARTIAL；辅助原生串行梯度集成/状态原子性未验证；本次 canonical TRAIN 前向已完成且单次许可已消耗，任何额外模型干预需要新的人类授权。CPU通过或新验证图像合格均不自动授权训练。

执行前，4上下文探针的v1静态审查发现运行contract未重绑的工程缺口，以同长度contract篡改的失败测试修正为v2，全部15运行输入与63source文件重新绑定。v2曾因人类授权pending及原AC门槛HOLD而未运行；研究者新回复明确批准一次只读诊断及局部电池例外后，经v3/v4局部门槛测试和精确复核，只执行上述v4一次。全局训练电源门槛始终未变。v4在AC和BATTERY下均要求已知电量>=60%，其余保护原样。

Mission alignment：`PRIMARY_VERO_TRANSFER` / `EXPLORATORY_DIAGNOSTIC`。观察层级为保存 TRAIN 数值重算和新来源技术验证；辅助准确率效应 `NOT_RUN`。实际独立 reviewer及哈希见results，跨模型家族审查 `NOT_VERIFIED`。
