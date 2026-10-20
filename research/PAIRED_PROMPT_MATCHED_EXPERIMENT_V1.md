# Paired-Prompt GSPO：一次有限、匹配的实验准备

状态：**PROPOSED VERO-STYLE ADAPTATION / CPU_PREPARED / NATIVE_NOT_IMPLEMENTED / NOT_AUTHORIZED**。这份执行准备与实验日志均不改变CURRENT_RESEARCH_PLAN.md。研究标签PRIMARY_VERO_TRANSFER；真实MRI准确率收益NOT_VERIFIED。本轮训练/updates/模型调用=0。

## 假设、竞争解释与主对照

假设：same-prompt K4经常零正确/零差异时，配对输入的真实可验证奖励差异可能提供有用的条件信号。竞争解释：差异只来自prompt难度/类别先验，归一化导致偏置、抵消或负迁移，active组增加但accuracy不变。CPU反例已证明这些风险实际存在，故不按active-rate授权训练，不把方法视为已成立的创新。

仅比较分组机制；没有CE/margin辅助，reward/calibration、ratio/clip、模型、LR、LoRA均不改。两arm均使用**Adam60继承的valid-format-subset group gate**，不是重现历史18臂all-format-strict gate或官方Vero所有实现细节。此选择在两arm一致，不偷偷增加第二项gate干预。严格生成/评分格式仍不放宽。

同一对TRAIN z045 F/T图像，每图总4个response：

|arm|sequence两组|每组执行方式|
|---|---|---|
|A SAME_PROMPT_LOCAL_VERO_STYLE_CONTROL|F4，然后T4|当前policy生成该K4，按该组实际reward评估是否更新一次|
|B PAIRED_PROMPT_ADAPTATION|F2T2，然后F2T2|当前policy下先收齐该K4各自条件的真实回答，再按该混合组reward评估是否更新一次|

随后两arm各执行相同两个Dallas原TRAIN记录（same/different各K4）。总计每arm4个尝试组、16个response，最多4个真实GSPO更新、最低0次。无监督辅助更新；不重抽至成功，不以增加groups补齐updates。

完全固定八回答CPU对照使用同checkpoint的F4/T4，两个group loss平均后每response权重1/8。**未来每K4单独optimizer step仍使用原1/4权重**；不能复用CPU block的1/8使有效LR减半。不能先收齐八回答、更新第一组后仍把第二组称作当前on-policy。A/B第一步后的response/状态轨迹会不同，这是group方法诱发的差别，不承诺逐token相同。原18臂不重跑。

## 输入、父状态与实际证据

CPU准备入口：`scripts/prepare_paired_prompt_ab_v1.py`。它只构建源绑定方案，`--execute`会拒绝；不是已实现的原生训练入口。`paired_prompt_groups_v1.py`提供response各自prompt/image/tokens/oldlogps/reward归属的CPU路由与loss系数，不能将历史单一self.active输入后端直接当作mixed-context训练器。

- 由固定字典序、在结果前选择一名既有TRAIN患者的z045 F/T配对；Dallas取原source-order首个same/different。四个输入两arm完全相同，IDs/路径/labels仅PRIVATE；模型仍只看真实图像和同一泛化问题，不见身份/site/文件名。
- 实际固定输入v2 SHA：`f4257ce14dd2918e5efd1bb01d27d86a1fcbbe890c881c527f07f2bbe3e63c5c`。V1仅历史准备，完整八回答同point要求只适用于离线固定block，不是原生每组更新规则。
- 父权重固定Adam60（52→56→60延续链的终点），仅据血缘选择，不据旧final100。adapter SHA `1741ff2feb2f569cfe062b92cb79b62f61f83463e6683a928d2c458a5f547310`，receipt SHA `3c2567723e7bc90a65464c8c3645f49ad91c38a0ad886961c508bc1f8a0e3879`。本轮只核对既有receipt，未重新读取权重/Adam数组。
- A/B各自从同一完整父权重、LoRA、Adam60和Python/MLX RNG恢复，串行单模型；新阶段记录新的group日程cursor。不重新初始化基础模型或偷偷重置Adam。
- 21份命名历史证据的有界核对：18臂sequence源每人只一种模态；实际F32只有一个sequence K4和一个Dallas K4；候选/VJP探针0生成。**真实完全匹配八回答block=0，NOT_AVAILABLE**。不跨checkpoint、prompt或数值策略拼接，不声称扫遍所有磁盘历史。

## 相同预算、成本口径与停止

|每arm上限/预留|数量|口径|
|---|---:|---|
|尝试组 / responses|4 / 16|2 sequence＋2 Dallas；必须完整走完才有完整计划曝光|
|FLAIR/T1/Dallas响应曝光|4 / 4 / 8|1个sequence患者配对，不是28人或56张全部训练|
|更新|0–4|不因active不同补训；纯GSPO，无AUXILIARY_ONLY|
|返回token|4096|16×256；保留EOS与lookahead的真实计数|
|输入token历史预留|4944|8×213+8×405；新runner须实际重测，变化即STOP|
|输入＋返回token预留|9040|两arm合计18080；不包含下列额外TF/内部重算|
|generation LM API预留|4160|保守预留，不宣称物理kernel次数；须用实际安装路径追踪确认|
|TF / backward / TF位置|64 / 16 / 42240|最坏全部active，含对齐、own-old/current及反向调用；梯度checkpoint内部重算不在API次数内|
|vision embedding API预留|80|生成16＋TF64；不允许跨图缓存视觉特征捷径|
|操作启动截止|1200秒|每arm20分钟；原子group、释放可能越界，不是可抢占硬墙|

这些是未来scope的保守预留，**尚无mixed-context原生调用计数资格**，不能直接当执行批准。新后端若需额外调用，先报告并重新审查，不自动扩预算。

历史同设备8组运行中7个完整group间隔mean31.57秒、median25.55秒、p90=50.73秒，加载到首组60.36秒；这些含checkpoint/监测且来自旧数值路径。另一次F32两组无更新资格运行实际307.59秒。它们不测得本候选速度。A/B合计训练先按**20–40分钟操作规划**，不保证完成，资源HOLD/加载/释放可能延长。三端点父/A/B工程评价再按约**30–45分钟规划**；F32新评价实际延迟NOT_MEASURED，不用Git时间估计。

任意内存/swap/温度/电源/其他heavy异常、概率差超过原0.002、状态不一致、token或调用预算不足均在安全边界STOP，保留原始错误与已完成状态，不自动重试。全参数/Adam/RNG恢复与原子checkpoint必须先通过独立源审查；失败不得留下半次更新。若某arm没有完成四组曝光，记INCOMPLETE_MATCH，不通过补跑或删组制造同预算；部分成本与结果保留。

## 冻结评价与可否证判断

首先用同一F32合约对父、A、B全部终点评价内部40条（24sequence+16Dallas），只作工程诊断。严格Vero think/answer/boxed、greedy256/temp0、parser不变。主要比较B−A，另报A−parent和B−parent，不能把共同配对数据的收益单独归因于grouping。

必须报告：strict greedy balanced accuracy、T1/FLAIR recall、candidate balanced accuracy及mean/sum/first（2/5 label长度不同）、正确−错误margin、同人反模态换图方向、适当无图reference、Dallas same/different recall、equal-task及worst-task、invalid、各任务实际tokens/有效组/更新/时间。两arm用同等硬上限；实际token成本可以不同，只能如实报告准确率−成本向量，不叫实际FLOPs完全相等。

工程GO仅用于是否值得更大独立研究决策：B的strict balanced accuracy、T1 recall、primary candidate balanced accuracy须同时高于A和parent；FLAIR及Dallas准确率/两类recall均不低于A和parent。若无信号、不可评估或任何保护项失败，停止同配置扩大，不调prompt/系数/种子/预算。微型单pair/单seed不能证明泛化或新算法成立。

新16人32图的模型输出保持封存。主方法、A/B预算、完整预算终点、数据处理、评价/统计及失败规则全部冻结并获准后，才可一次预声明比较父/A/B全员，不根据QC或预测排除已有4对标记。此时32新sequence+既有16Dallas构成48条/模型；Dallas仍是重复工程保留，不是新科学任务。可沿用旧每端点49candidate查询（含1无图）的147TF与48greedy预算，三个端点上限441TF/144greedy/36864返回token；**这仅是代数准备，不是已批准的新评价scope**。报告16患者配对聚类不确定性、全部模型结果，不挑最佳；看后不能继续用该集合调参。完整异质迁移结论仍缺第二个合格科学MRI任务。

## 训练前缺项与唯一许可边界

已完成：固定计划、历史归档、CPU数学/反例/来源路由、命名证据可用性审计、匹配输入与预算准备。仍缺：mixed-context原生generation/scorer/梯度与原子事务实现、精确native调用/资源/状态scope、相同F32全评价路径资格及独立审查。不得把旧F32 probability PASS当作这些新路径的PASS。

因此当前READY_FOR_TRAINING=NO，未申请/获得训练许可。完成缺项后只能请求这一次明确的A/B有限scope授权，不包括32/46/70组、多seed、旧实验重跑或新辅助目标。若数学/路由出现无法解决的契约错误，停止本候选并将具体证据提交研究者；不能用日志偷偷改主计划。
