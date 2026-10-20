# Adam60 混合辅助目标：实际CPU结果（2026-10-20）

## 范围与研究目标

Mission alignment: PRIMARY_VERO_TRANSFER。这是Vero-style GSPO的显式监督辅助目标，不是原始纯Vero GSPO。继承已训练Adam60，保留原主池28条sequence+48条Dallas、固定日程、reward、gate、GSPO loss、LR、LoRA。Dallas继续为ENGINEERING_ONLY。

本轮新增模型加载、原生forward、backward、generation、训练和optimizer updates全部为0。只读上轮实际保存的TRAIN margin VJP与Adam60 moments，没有打开final100或新16人输出。继承分支a6ebe8d12e0dc7a8e1633acb64e5730cc5c6897b；新分支research/vero-mri-protected-mixture-cpu-v3。

## 实际可行性与系数选择

先冻结判据和选择规则，后计算107个w预览，实际CPU耗时34.10秒。先混合梯度再执行逐参数Adam F32运算，F64归约；绝不插值两个最终Adam方向。目标为原L_GSPO + .1*((1-w)*L_v1+w*L_v2)。

保存的一个真实TRAIN配对与一个Dallas输入上，TOTAL方向零边界数值区间为[0.4014438391, 0.5128168464]；要求三项均>=1e-6的内缩区间为[0.4014863968, 0.5127726197]。边界二分括号宽约5.96e-8。17点定位网格和33个区间内部点均按预声明执行；这不是排除任意狭窄漏采样连通分量的全局证明。

按“最宽内缩区间中点、六位小数”规则，固定w=0.457130。没有用dev/final准确率搜索系数。

|正确类别mean margin的一阶方向|T1|FLAIR|Dallas|
|---|---:|---:|---:|
|g=0，旧Adam动量基线|−0.011919110|+0.010489396|+0.031854243|
|混合辅助梯度后的TOTAL Adam61方向|+0.001306137|+0.001258635|+0.029486729|
|TOTAL减去g=0基线|+0.013225247|−0.009230761|−0.002367514|
|欧氏负辅助梯度方向|+1.890200097|−1.396719534|−0.432138671|

**局部TOTAL可行，但新增信号本身没有保护FLAIR和Dallas。** 两者依靠旧动量保有正总方向。Dallas相对基线损失0.002367514的一阶margin方向。不能称为“无负迁移”，也不能将局部方向用于下一模型状态而不重算。

缩放辅助梯度L2=.185404014；TOTAL Adam位移L2=.024784822；相对g=0增量L2=.002581295。新端点包络上限分别为.342440863及.024884931，当前余量.157036850/.000100108。历史路径没有绝对L2硬上限；这些是新预览限制，不是原训练器已有的裁剪，也不证明硬件安全。

F64与另一F32混合顺序的候选TOTAL Adam margin导数差约2e-9，远小于1e-6内缩保护；边界附近仍须报告舍入敏感性。独立审查重算13个候选/端点/边界/敏感性组合，与主结果最大差1.964e-13；独立核查完整107次缓存记录，但未声称另跑全部107次。

T1原mean margin=−1.321572210；即使线性外推这一次TOTAL方向，也仍约−1.320266，未跨分类边界。这不是有限参数步的实测，更不是准确率提升。

## 实际代码与保护

- mixture_adam_cpu_v3.py：同点梯度合成、真实Adam预览、TOTAL/g0/增量分别计算。
- paired_classification_mix_v3.py：冻结w、alpha的可微v3损失与导数，保留v1/v2。
- class_margin_mix_v3.py：休眠原生backend，复用原v2原子事务driver；合成GSPO+辅助梯度后只允许一次Adam调用，inactive组的辅助更新标AUXILIARY_ONLY。没有启动模型或生成新的权重。
- 未来每步先取得当前模型的margin VJP与当前Adam状态，再预览实际JOINT方向；未达保护线或超包络即停止，不修改advantage、不投影梯度、不伪造rollout。sequence槽保护当次辅助配对；Dallas槽保护固定参考配对；Dallas始终保护固定case。逐目标记录身份，不能冒充全类别保护。
- 额外保护VJP消耗原40TF/20backward/19860位置硬预算，不能自动扩预算以完成步骤。事务driver负责异常回滚和完整checkpoint发布；CPU验证不能代替原生原子性资格。

Root实际30个针对性CPU测试通过（9新backend+4混合+17旧原子接口）。独立review加入2个真实NumPy Adam预览fixture，总计32通过；负FLAIR反例在optimizer调用前拒绝。休眠backend不等于已审查的可直接启动训练CLI。

## 排除只对一对有效：下一有限验证准备

固定规则排除原梯度患者及其站点，取字典序前三个可用TRAIN站点BA/FO/JS，每站首个既有患者：共3人、3站、6张z045图像。另固定Dallas same/different各一例；其存储patientkey不同，但跨case真实身份独立性NOT_VERIFIED。仅核验既有元数据及PNG字节hash，没有重复168图审核。保留QC标记，不按模型输出排除。

下一只读检查的**输入及调用预算草案**为16个F32 TF forward、8个margin VJP，预计4382位置/硬上限6800，0生成、0更新，20分钟操作窗口。它尚不是精确review过的原生runner/launch scope，也没有新的执行许可。需要先绑定完整源码、状态、资源、实际调用路径并审查，再申请一次明确只读授权。若其他TRAIN病例失败，不自动按这些病例重调w。

## 未来微型试验与评价

目前不具备直接训练资格。若后续跨TRAIN只读验证与精确原生路径审查通过，再由研究者单独决定是否批准：继承Adam60完整权重/LoRA/Adam/Python+MLX RNG；原日程index8:12共2sequence+2Dallas；最多4尝试group、**2次TOTAL optimizer更新**（JOINT/AUXILIARY_ONLY/GSPO_ONLY均计入），任何硬限先到即停。2步可能只接触一个配对，不能声称训练全部56张。

主rollout输入视觉+文本及返回token上限9040，返回最多4096；辅助TF位置1812；两者计数口径合计10852。全部TF（含辅助及额外保护/scorer）共用19860位置、40TF、20backward上限；1812辅助TF位置是其子上限；1200秒是操作启动截止，不是可抢占的绝对运行硬墙，原子group及释放可能越界，不能承诺20分钟完全结束。不能将TF位置计作生成token。初两槽均active时调用上界28TF/14backward/12500TF位置。10–20分钟仅为未来单次操作规划窗口，不是实测v3速度或完成承诺。工程父子评价沿用历史约20–30分钟规划，实际取决于输出长度。

父子必须用同一F32、prompt/decoding/score比较内部40条：strict greedy balanced accuracy、T1/FLAIR recall、candidate mean/sum/first、margin与换图方向、Dallas same/different、worst task、token/时间。内部集只能工程诊断；没有等预算旧池对照，父子变化不能证明配对辅助目标的因果收益。

新16人32图保持模型输出封存；只有方法、终点和协议冻结后一次全员父子评价，包含已有QC标记图，不按预测删病例。不再利用旧final100选方法。

## 决策

CPU局部可行性成立；训练READY=NO。缺少跨TRAIN原生方向验证、实际联合路径资格、精确微型launch scope及新的研究者训练许可。当前真实MRI accuracy提升=未证明，本轮未做准确率评价。下一步值得做的是固定系数的跨TRAIN零更新验证，而不是立即续训或扩大系数搜索。

证据：results/MIXTURE_ADAM60_FEASIBILITY_V3_CPU_V1.json、独立数值review、MIXTURE_V3_BROADER_TRAIN_AND_MICRO_PREPARATION_V3.json及独立源码/准备review。私有原始梯度、moments、患者清单均不进入Git。CROSS_MODEL_REVIEW=NOT_VERIFIED；独立审查实际由Codex隔离reviewer执行。
