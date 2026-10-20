# 配对z045：一次F32资格结果与实际曝光修订

## 本轮结论

研究标签：`ENGINEERING_UNBLOCK / EXPLORATORY_DIAGNOSTIC`。完成了用户授权的唯一一次零更新原生检查及CPU日程/身份清单工作。**模型加载1、生成8个回答、返回216tokens、GSPO backward0、optimizer updates0、训练0**。没有重抽、重试、追加模型、重跑历史18RL/3SFT/final100，未改loss/reward/LR/LoRA。所有MRI、身份映射、原始回答、概率矩阵、Adam/RNG留在忽略目录。

**概率一致性PASS，完整新数值资格PARTIAL；当前不批准训练。** 两个真实组均被既有MRI信息门槛判定inactive，未为填满预算制造advantages或反向。

## 1. 实际F32数值结果

固定范围与上轮相同：scope FILE `865aeda5a772c9a917370bacddfd84309ac51ac225cd52732e0ee0e2f56f3511`，canonical `d218806a395680f9e11b9acfc09b66353d570323b3db1c69bc782924af03900a`。源码57项、元数据13项、独立review与输入SHA在启动前后认证；本轮新日程没有替换这两个诊断输入。

| 指标 | Sequence | Dallas工程对应 |
|---|---:|---:|
| 真实回答 / 返回tokens | 4 / 86 | 4 / 130 |
| F64归一化全词表最大logp差 | 0.000167836705 | 0.000263787556 |
| 最大TV | 0.0000162651 | 0.0000195905 |
| 原阈值 | 0.002 | 0.002 |
| 实际有效组 | 否 | 否 |
| 本轮只读反向 | 0 | 0 |

全部216个被比较位置的词表top1一致；实际返回216、含安装包lookahead的sampler draws224、LM调用240、vision调用16、full scorer8。8个回答均正常EOS结束，未触及256上限。这里只验证这些实际上下文，不能推广为任意F32上下文或MRI准确率结论。

Sequence的4个raw accuracy都是0，format为`[0,1,1,1]`，优势约`[-1.499985,.499995,.499995,.499995]`，非零差异全部来自格式；有效组规则正确拒绝它。Dallas同样4个raw accuracy都是0，format全1、total相同、优势全0。因此不能说“两组都零方差”，也不能把概率PASS换成GSPO反向PASS。

外部实际exit0，闭合stdout SHA `ec92eca62148aa82cf6bf53229e253ce83937eb9e5f13020ddbb0de0bf1eb20f`。外部witness elapsed约307.590158秒，包含资源HOLD，不是训练速度。全部1942模型参数与Adam保持相同，真实生成RNG消耗先记录，随后恢复Adam60的Python/MLX RNG，状态一致。释放后`model=None`、MLX active1100bytes。

资源537条：526CONTINUE、11HOLD；525NORMAL、12WARNING。两次内存余量HOLD、九次swap增长HOLD后由既有门槛自然放行，没有重启或降保护。全部AC、otherheavy0，温度537条均`SENSOR_UNAVAILABLE`，不能声称有温度正常读数或全程NORMAL。

独立Codex从8个响应各自保存的生成端／完整TF词表logits、实际token IDs与真实奖励重新归一化计算分布指标、全部位置及状态/来源/退出证据，结论见 `results/PAIRED_Z045_F32_QUALIFICATION_ACTUAL_INDEPENDENT_REVIEW_V1.json`。不是Claude或跨模型家族验证。

## 2. 为什么原32/8方案不足

以下均为**已构建日程的输入曝光上限**，本轮没有训练；“被尝试输入”不是“获得了有效梯度”。

| 日程 | groups | seq / Dallas | 序列唯一图像 / 患者 | 完整患者对 | 满日程charged预留 |
|---|---:|---:|---:|---:|---:|
| 原28池、同32槽 | 32 | 18 / 14 | 16 / 16 | 0 | 70,784 |
| 上轮56池、原采样 | 32 | 18 / 14 | 16 / 12 | 4 | 70,784 |
| 新强制配对、同32任务槽 | 32 | 18 / 14 | 18 / 9 | 9 | 70,784 |
| 只作成本比较的较长方案 | 46 | 32 / 14 | 32 / 16 | 16 | 97,048 |
| 只作成本比较的全56序列曝光 | 70 | 56 / 14 | 56 / 28 | 28 | 142,072 |

推荐候选为**同32任务槽的强制配对v2**：固定历史任务顺序和14条Dallas occurrence对象；在sequence子序列中按同患者FLAIR/T1成对出现，不要求全局相邻。按8个TRAIN站点轮转、站点内固定SHA顺序选择9人，每对方向交替，实际9FLAIR/9T1，无重复序列图像。没有新患者、层面、任务或额外类别quota。明确这是新的采样干预，不是与原Vero row sampler或旧row RNG完全相同。

代价：32槽从原来的12个患者降为9个，换取完整双模态配对；不能把患者覆盖与配对效应拆开归因。46/70组还改变了任务投入比例且超出旧70,784cap，不推荐自动扩大。

**8更新早停仍会遮蔽配对曝光。** 新日程第8组时仅完整2对，第16组4对，第28组8对，第32组9对。如果保留旧8-update先到停止条件，不能承诺9对。为检验完整32槽，另列了“32attempts、最多32updates”的候选封顶（每组最多一次，不以达到32更新为目标），token70,784、开操作窗口3000秒仍不变；这会改变预算封顶，**不是已批准的旧32/8入口，也未修改该入口或取得训练许可**。若研究者坚持8更新硬cap，则只报告实际配对数，日程未完成应判`INCONCLUSIVE_FOR_PLANNED9_PAIR_EXPOSURE`，不能偷加组数或只读rollout来冒充训练曝光。

即使完整尝试9对，只有实际有效组才有梯度；未来必须按患者/模态分别报告attempted与accepted覆盖，不能把两个模态都被生成过算成两个模态都学会了。当前没有依据扩大同配置训练。

## 3. 准确率比较及新患者验证

父权重仍预先固定Adam60，只为52→56→60连续性，不用旧final100挑选。保留全部原权重、Adam/RNG。未来父子必须使用同一个已实际资格化的新F32合约、相同prompt/temp0/max256、严格Vero格式和相同输入。报告严格sequence balanced/greedy accuracy、FLAIR/T1 recall、candidate mean balanced accuracy和正确减错误margin（sum/first次要）、交换敏感性、Dallas各类recall、宏/最差任务及实际token/时间成本。FLAIR或Dallas退化，或者candidate不改善，则停止同配置扩训。

现有40条dev和8个复用swap aliases仅作反复查看过的工程诊断，没有同预算旧池续训对照，父子差分不能证明配对数据的因果效应。完整F32反向资格、真实同F32父子评价仍未完成。本轮没有新的准确率改善或泛化结果。

实际建立了23个历史元数据输入的排除目录：原40例来源partition、TRAIN/dev、最终序列身份registry、先前feature/probe输入、168审查与空间候选，加保守排除的既有raw目录。原匿名键规则为`sha256(BIDS subject)[:12]`并与旧40例crosswalk核验；最终14例使用另一opaque键命名，已改为通过批准的source subject crosswalk加入，强制核对14个唯一身份。初版仅识别40人的预览私有保留，最终排除54个使用或保留身份；不是宣称54人都训练过。

来源363人中剩309个元数据候选，均有FLAIR/T1ce sidecar；本地成对gzip payload为0，没有读取新MRI像素或下载。BA剩0、FO剩1、ME剩1，其余BO162/KA101/NSK32/JS4/TO8。因此原“8站点各2人”的16人规则客观不可行，必须在任何新预测之前另行冻结可行站点可用性规则。当前**new holdout NOT_ESTABLISHED，选定新患者0**。

该清单明确限定在已枚举历史目录，不冒称排除了未登记的人工/外部曝光；跨数据源全局患者身份仍NOT_VERIFIED。需补真实payload/配对几何/渲染标签资格与跨源lineage检查。没有读取旧final100输出、分数或用其标签选择新方法；最终registry仅投影身份作排除。

## 4. 成本与下一步决定

旧真实uniform历史7个完整间隔均值31.57秒、中位25.55、p9050.73；不能把这次无反向的307秒资格耗时当作训练速度。按旧速度32组含加载约17.8分钟，逐组按旧p90约28.1分钟；规划15–30分钟、开操作上限50分钟加原子收尾，实际F32 backward时间未知。匹配父子评价另规划10–30分钟，HOLD/新患者准备另计。70,784是prompt/visual38,016加K4返回response32,768的条件预留，实际展开213/405若变化仍须STOP，不能暗扩cap。

**当前建议：不启动续训。** 优先保留本次概率PASS与梯度PARTIAL；不得用剩余预算重抽或自动再跑。若以后补反向证据，需另行明确授权一个复用已保存真实response/advantage、无新采样的范围，并区分数值梯度验证与MRI正确性学习信号。新的强制配对32槽只是优先候选，不是已批准训练。后续即使批准工程小试，也只能作局部准确率诊断；新独立患者验证集真正建立前，不宣布泛化成功。

Mission Alignment：该改动仍服务于Vero-style多任务GSPO的MRI迁移；检验配对曝光能否改善T1且保留FLAIR/Dallas；loss/reward/LR/LoRA冻结；反例为candidate不升或任务退化；没有matched旧池控制则拒绝因果叙事；所有新训练继续等待数值前置条件与新的明确人类决定。
