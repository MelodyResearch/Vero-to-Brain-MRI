# Vero-to-MRI：逐类 margin 辅助目标 v2

## 范围与唯一干预

继承 `7e594e6742c7a39f6a505a4d731c0b0750cb9c32`。本阶段实际训练与 optimizer updates 必须为0。旧v1、原GSPO损失、reward、LR、LoRA、历史18RL/3SFT/final100、B28/Adam52/56/60保持不变。新机制是显式监督辅助，不是原始纯Vero GSPO。

同患者z045两模态、同generic问题和canonical prefix：`mF=sF(IF)-sT(IF)`，`mT=sT(IT)-sF(IT)`，类别分数是label-only mean logp。`L2=(softplus(-mF)+softplus(-mT))/2`，`L_total=L_GSPO+0.1L2`。固定格式、前缀、EOS不参与辅助损失。mean为主，sum与first分开报告，不互相替代。

v1仅约束平均margin，可以接受T1仍为负。v2直接惩罚两个独立错误margin，而且不再抵消共有FLAIR score偏置。score空间导数方向正确并不保证共享参数或Adam方向正确；无图像信息的模型也能通过先验中和降低v2，不能据此宣称视觉学习。

## 新的零更新原生资格：先审查，再请求一次授权

固定上一轮TRAIN配对、既有Dallas输入和一个同shape零归一化pixel control。不新增患者、不生成回答、不读新16人图像/标签、不利用final100。复用旧F/T/Dallas候选分数并检查原点一致性，不重复F32概率路径审计。

按每张图单独构图的正确类别mean-margin VJP，组合v1/v2参数梯度；swap使用已计算图像的严格符号/行交换别名，不追加计算。blank用于识别先验梯度，不赋予真实视觉正确标签。逐项报告mF/mT/Dallas沿欧氏方向、保留Adam60 moments的解析Adam方向，以及相对g=0旧动量基线的增量；禁止optimizer.step和参数扰动。sum/first方向未计算时明确NOT_MEASURED。

具体调用、硬预算、所有输入/源文件/Adam状态和资源检查须由实际准备scope冻结；没有人类对该新scope的明确许可就不加载模型。旧候选评分及其电池例外许可已消耗，不能重复使用。状态必须在任何restore前核验模型1942参数、完整Adam及Python/MLX RNG未变，再核验恢复与释放。失败不自动重试，不扩大样本。

## 可执行微试路径，但本轮禁止执行更新

未来只从Adam60/fullAdam/RNG继承。原28sequence+48Dallas主池、固定uniform历史index8:12四槽（2sequence/2Dallas）保持原样。sequence槽只附加同患者z045另一模态，明确增加辅助监督，不同时使用强制9对采样。

GSPO原gate决定原分支；监督分支可单独有效，记为AUXILIARY_ONLY，不能伪造原GSPO优势或accepted group。任何真实Adam调用，无论JOINT/AUXILIARY_ONLY/GSPO_ONLY，都计同一个最多2TOTALsteps。inactiveGSPO且aux梯度全0则NO_UPDATE；原activeGSPO或最终joint梯度非有限/为0则停止并恢复当前group，不能用旧动量制造视觉学习。

所有分支在同一原点计算并求和后只step一次。事务覆盖可训练参数、完整Adam、RNG和cursor；未发布失败回滚，已完整rename的checkpoint作为已提交状态，resume不得重复更新；保存的完整rollout可精确复用。历史父文件不覆盖。

准备预算：最多4attempted groups、2TOTALupdates、K4/256，main输入+返回预留9040；辅助至多1812 TF位置，所有GSPO+aux至多40 full-TF API调用/19860 TF位置、20backward。内部checkpoint重算/未返回ARlookahead单列，不能把10852当完整计算量。1200s操作启动窗口，原子调用/回滚释放可超时。先到限制即停，未完成四槽或未产生两步也保留；不补跑。10–20分钟仅规划窗口，未测新训练吞吐。

## 微试评价与新患者封存

微试父子使用同一F32合约和原40条内部工程输入、同prompt/greedy256/strictVero scorer。冻结报告T1/FLAIR recall、strict greedy balanced accuracy、mean/raw/first候选balanced accuracy/margin、同患者换图、Dallas两类recall、macro/worst task、invalid和实际成本。改善不了候选/T1或出现类别/任务退化，不自动扩训；不以loss下降或更多updates判成功。

新16人/32图只引用已发布seal与manifest hashes，不在本阶段打开其payload、labels或outcomes，不用于loss、系数、父checkpoint或阈值选择。只有方法、确切父/子checkpoint和正式评价流程全部冻结后，另行授权一次预声明父子评价，全16人纳入，4对质量flag不剔除。原future48输入准备不是自动评价许可；不得用其正式结果再调方法。

## Mission alignment / 证据层级

`PRIMARY_VERO_TRANSFER`：用可验证采集模态补充稀疏GSPO视觉类别信号，不更换基础模型或研究方向。唯一方法干预是配对监督辅助分支；额外配对曝光与辅助数学没有等曝光对照时不能单独归因。当前CPU数学为TESTED_CPU，原生资格、真实准确率和训练收益必须依实际记录更新。即使零更新资格通过，也不自动授权微试或新16人正式评价。
