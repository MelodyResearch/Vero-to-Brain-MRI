# TRAIN资格与原生GSPO路径：实际检查、修复及停止结论

研究标签：PRIMARY_VERO_TRANSFER / C_TASK_ELIGIBILITY / ENGINEERING_UNBLOCK。继承已核实的CPU修复 `e10194aeac2491ccc7d85b7bdb546261cb7a6fae`，独立分支 `research/vero-mri-native-path-qualification-v1`。本轮新模型训练0、新回答生成0、新optimizer updates0。只读MLX诊断真实启动一次，因预先冻结的数值门槛失败停止；没有为补齐结果而自动重试。

## 1. 结论与证据等级

**READY FOR TRAINING = NO。B2原生数值一致性FAIL，B3完整GSPO视觉梯度NOT_RUN，Task B整体PARTIAL。** 此结论与CPU公式检查PASS不矛盾；CPU公式不等于原生rollout/scorer等价。

| 当前优先级 | 结论 | 已有直接证据与限制 |
|---|---|---|
| P0：原生概率路径 | CONFIRMED：本次复现不一致；历史准确率失败的因果贡献仍UNVERIFIED | 同权重、同图像/IDs下，BF16增量KV与F32完整TF最大logp差0.1003538743；F32归一化后仍差0.0097333789，超过冻结0.002。仅测到第一组第一回答，不外推全部旧rollout。 |
| P0：任务资格与监督定义 | CONFIRMED：准备限制；LIKELY：学习质量瓶颈 | 序列源标签与像素正确、166张有非专家配对可答性支持；Dallas没有验证brain-only身份依据；mask-derived空间标签不能替代可见病灶/临床阴性。 |
| P1/P3：奖励信用与模型容量 | 机制限制CONFIRMED，相对因果贡献UNVERIFIED | 旧CPU重放结论保留；本次未到真实GSPO反向，不得声称视觉group梯度弱、抵消率高或Qwen缺乏MRI能力。 |

已确认的BF16归一化精度问题可做局部工程修复；残余KV/full差异尚未定位至具体kernel、attention、shape或位置实现，不能凭猜测宣称已解决。

## 2. TASK A1/A2：168张序列TRAIN图像

### 来源、像素及输入

对28个TRAIN患者的56个NIfTI和56个侧车逐项核验官方annex SHA、字节数和模态来源。原renderer的subject哈希规则只用来匹配候选28人的原始目录；没有读取非TRAIN影像/侧车或final100。官方README的run-02为注钆后T1，FLAIR来源与既有derivative命名和原代码一致。

168张PNG全部由源volume重建，**168/168逐像素一致，最大差0**。两模态在28患者内shape与affine一致；56源为LAS。实际渲染为原生z045/091/136、np.rot90、各volume非零体素stride16的0.5/99.5百分位窗口、round/clamp uint8、2倍bilinear RGB。没有新裁剪、重采样或mask叠加；这证明历史渲染忠实，不证明该窗口/层面最适合学习。

原始filename含模态，但native message AST只使用system与跨类别恒定question，路径只传入图像loader；168张原PNG没有嵌入文本metadata。新候选提供opaque无语义资产。静态检查不冒充168条全部原生tokenizer实测。

### 真正执行的盲态配对检查

生成84个随机ID、左右顺序随机的独立配对图及11页contact sheets，真值单独保存。Root实际查看全部11页，未看key先冻结判断，SHA `382a5379f42d4d4e5d8fc62cfdead6ea64c51c8300c621bc4b632087ad79be56`，再由另一agent解盲并由独立reviewer重算。

- 83对作答，83对两侧均正确；166个作答侧正确。
- 1对因几乎全黑、仅有零碎组织而ABSTAIN，隔离2张图。
- 78对high、5对medium、1对none。
- **这是非专家AI、已知每对一张FLAIR一张T1ce的配对识别，不是VLM准确率、独立单图测试、临床裁决或独立患者泛化。**

已有166张coverage候选、2张abstention及完整168张未来loader清单，逐行保留可答性状态、隐藏真值和TRAIN来源，全部training_authorized=false及EXPERT_PENDING；不把猜对标签自动升级为科学训练许可。

### 强度/协议与泄漏控制

28患者、3层面、8来源site、9个manufacturer/model/field组合均在FLAIR/T1ce两类间平衡。原28条每人一种模态的列联关系与新168条每人两模态的差别保留在前轮证据中；本轮核对原始侧车补齐真实采集分组。

固定无参数16-bin直方图及0..1强度摘要，进行排除同患者全部图像的TRAIN内部1NN检索：分别**153/168**与**152/168**。没有拟合head、scaler或optimizer，没有调参。它们说明采集对比信号具有可辨性；不能区分合法模态对比与所有协议/归一化捷径，也不是未触碰cohort泛化结果。

## 3. TASK A3：Dallas工程降级

24个TRAIN人、48个源NIfTI、48个跨visit pair全部重开验证，源panel与原pair PNG重建逐像素一致；BIDS身份与wave2→wave1方向/隐藏same-different标签核对通过。实际查看全部48对wholehead图。

脑室/脑沟结构可见，但脸、颅骨、颈部线索也存在。独立目录生成固定cranial ROI遮挡并实际检查，发现它**截掉脑组织且残留颅外线索**。该控制判为失败，不称skull stripping或brain-only。现有TRAIN侧车不足以证明scanner/site匹配。

**Dallas仍为工程性wholehead跨visit身份对应任务；不是已通过的MRI脑结构推理或临床纵向变化任务。** 缺少有效脑内隔离与身份可答性对照，不能仅因identity元数据正确就批准。

## 4. TASK A4：空间/病灶任务真实检查

检查全部28人、84层面、56 MRI和28 mask，对应214条既有TRAIN routes：presence84、enhancement84、grid46。源affine、LAS方向、slice index、np.rot90、2倍显示及pair frame offset均一致，无重采样；所有214条几何引用与固定TRAIN记录相符。包括Dallas在内104个MRI header的qform/sform差异均未超过0.001mm。

先实际看84个MRI-only plane并冻结非专家观察，再看84个mask overlay：

| 非专家MRI观察与mask技术交叉 | Plane数 |
|---|---:|
| 明显/不对称异常且mask positive | 41 |
| mask positive但MRI-only未明显识别焦点 | 5 |
| mask negative但可见异常 | 5 |
| mask negative且未见明显焦点 | 33 |

可见异常可能是术腔、水肿、伪影或其他改变，不能推断临床肿瘤真值；最后33个也不是专家确认阴性。几何检查还标记全局质心落在mask外、靠近网格边界、多连通灶以及全局/最大连通区网格不同等问题，保持原标签不变并逐行留在待裁决队列。三条route共用同一mask来源，不是三份独立临床gold。

现阶段不能确认第二个科学可接受的异质MRI任务。确切缺口是独立可见性/歧义/阴性与术腔定义、空间标注约定及捷径控制；不是缺少文件哈希。新候选不会把这些问题静默转换成批准。

## 5. TASK B：真实原生执行与停止

### 预定输入、预算和真实状态

按冻结TRAIN ledger顺序选择最早accepted informative sequence与correspondence group，不依据dev/final100。两组各自真实pre-rollout checkpoint拥有相同已训练B28 adapter、GSPO Adam0和各自RNG；计划一次resident加载并逐组恢复真实状态，非重新初始化模型。

独立review绑定25个源码摘要、53个输入摘要、scope及预算：最多2组、8 KV prefills、136 KV token forwards、8 full-TF、8 ZERO-old、24 backward、1800秒安全边界检查、0生成/0updates。用户接电后实核AC，资源/排他锁通过才启动。

### 实际测得的输入与logp

第一组实际generation-prepare与teacher-forced-prepare的input IDs、pixel tensor、image grid、mask等逐tensor哈希相同；完整prompt IDs与保存rollout一致。没有用仅长度或old=current自一致代替此检查。

第一条保存回答17 tokens的原生增量KV评分与完整TF比较：

| 比较 | Max abs | Mean abs | 超过0.002的位置数 |
|---|---:|---:|---:|
| 原生BF16 KV vs F32完整TF | 0.10035387426614761 | 0.017341758463978733 | 9 |
| 同一次KV logits，BF16 vs F32归一化 | 0.10033798217773438 | 0.01722279716940487 | 9 |
| F32 KV vs F32完整TF | 0.009733378887176514 | 0.0012734010297137388 | 3 |

17个位置均记录到BF16 logits/BF16原生logp与F32 stable/TF dtype。差异逐位置可分解，F32控制能解释一部分误差，但仍不通过原阈值。不能把残差归因到某个特定kernel或宣称已经解决。

历史生成时的条件logp与完整pixel/grid/mask指纹没有保存。本次路径复现FAIL与历史可追溯性PARTIAL必须分开，不能补造历史一致性。

### 真实调用、完整性与安全

进程真实exit1，因数值ValueError主动STOP，没有自动重试。实际只完成1 KV prefill、16个单token KV forward、1 full TF；**backward0，生成0，optimizer updates0**。没有进入B3，没有计算完整GSPO图像梯度、抵消或label-gradient贡献，不能以旧CE实验或CPU测试填补。

1,942个模型参数指纹前后完全一致；504个adapter张量及Adam/Python/MLX RNG均不变；模型释放成功，active MLX剩1,050 bytes。82次资源记录全部NORMAL/CONTINUE/AC；最低可用8.79GiB、采样MLX peak5.59GiB，thermal为SENSOR_UNAVAILABLE。到STOP实测约61.6秒。停止原因明确为数值门槛；已保存资源记录未显示资源触发STOP或新系统不稳定，thermal仍为SENSOR_UNAVAILABLE。

数据范围边界：真正选组和模型输入全部TRAIN。CPU prepare曾完整解析既有混合probe-plan JSON，只消费其中model/cache/contract配置，没有用其dev条目选样；不能声称整个native准备过程零混合元数据字节读取。未读取final100/非TRAIN图像或正式测试输出。

## 6. TASK C：实际代码修复与尚未修复部分

1. **数据准备：**opaque资产、完整来源/像素/空间核对、盲审真值分离、coverage拒答隔离、科学/视觉/技术状态分层及TRAIN-only候选；不覆盖原数据，不虚构专家标签。
2. **诊断器释放：**CPU weakref反例复现异常traceback保留model，现于保存原错误与完整性后清理已unwind帧；真实数值STOP仍成功释放，原异常未被二次release错误遮盖。
3. **预算：**资源HOLD返回后复检截止时间，fakeclock回归通过。边界检查不承诺能中断不可返回的单个native kernel。
4. **独立未来概率修复：**新增FP32LogitsProcessor，供被检视的普通AR分支作为最后custom logits processor，将已有logits转F32后由既有AR归一化，不改权重或架构。旧冻结代码和v3scope均保留。

BF16 round-to-nearest-even CPU反例logits[64,63]本来可精确表示：旧归一化相对F64参考误差0.1867383125，F32路径约0.0000012519。它证明返回的logp绝对值可因归一化舍入而不准确，不声称bitwise模拟MLX reduction kernel。真实同-forward的F32分支提供了上述部分误差下降证据。

这里还需区分AR产生的logp数值与categorical实际采样分布：统一加性偏移不改变categorical。上述两logit反例的误差恰好是共同偏移，重新精确归一化后类别分布相同，因此该反例不能证明答案分布改变或历史低准确率因果。即使剔除这个归一化偏移因素，本次F32 KV对F32 TF的0.009733残差仍不通过冻结门槛。

**未来hook已IMPLEMENTED/TESTED_CPU，尚未在生成器中原生执行，不覆盖batch/speculative/fused等其他路径，不授予训练许可。残余0.009733仍未解决，B3仍NOT_RUN，MRI准确率收益NOT_TESTED。**

## 7. TASK D：唯一下一优先与父权重规则

**唯一下一优先：定位并修复剩余KV/full概率路径差异，再验证完整GSPO视觉导数。** 不先改reward、扩大数据训练、增加updates、调全部LR或解冻视觉塔。当前不能指定一个已经获准的续训配置。

若未来允许沿既有adaptive链继续，可预先以血缘规则选Adam60：它是52→56→60既有终点，绝不按已看过的final100挑优。B28/Adam52/56/60全部保存；诊断使用B28只因它是所选历史组真实pre-state。优化器/RNG兼容时保留；改变目标/状态结构时明确新阶段及cursor，不重新初始化权重。

如以后获准有限续训，首个单因素应是经验证一致的rollout/scorer概率数值契约；前提仍是至少两个合格任务及另行许可。随后才可单独检验已有166张配对视觉监督相对原28条的影响，不能同时改reward/sampling/LR来掩盖因果。

独立患者验证应从未用于既有train/dev/final100的新患者建立，先按真实身份/跨源lineage排除重叠，固定site/protocol分层、任务资格、prompt/读取方式、主要指标、预算与停止规则，再看模型输出。内部TRAIN检索和反复检查的dev均不能作为此独立验证，原final100保持历史正式证据，不再用于选择新方法。

## 8. 交付与独立复核

- 非空168行未来训练格式序列清单，166 coverage候选与2拒答隔离；Dallas/空间262行资格索引及完整原TRAIN记录保留。所有清单training_authorized=false。
- 统一版本化候选bundle保留所有430条技术候选、来源hidden target与逐行qualification，不能自动启动训练。
- TaskA由独立CPU reviewer重新计算所有LOPO邻居/混淆、盲态解盲、清单分区及几何汇总；没有冒充重复打开全部NIfTI。
- TaskB由独立reviewer检查精确launch scope，再从实际闭合输出重算17位置误差、dtype分解、状态、资源与释放；没有重复原生执行或反向。
- 全部审查真实身份为Codex；CROSS_MODEL_REVIEW=NOT_VERIFIED，没有虚构Claude或临床专家。
- 只提交源码、脱敏汇总与review。影像、原始ID、逐条真值/输出、token arrays、权重、Adam/RNG和梯度均保持PRIVATE。

Mission Alignment：继续Vero-to-MRI多任务可验证GSPO迁移；真实TRAIN资格与原生路径是前置，不转成独立分类器项目。训练0、更新0，不用最终测试挑方法。**本轮完成了可执行的检查和局部修复，但未完成完整原生GSPO路径证明；训练准备状态NO，等待研究者下一步决定。**
