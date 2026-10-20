# T1 类别偏置：逐类 margin 辅助 v2 的实际准备

## 已确认的问题和修正

继承 `research/vero-mri-paired-signal-v1` 的 `7e594e6742c7a39f6a505a4d731c0b0750cb9c32`。本轮新增训练、optimizer updates、模型重新初始化均为 **0**；没有打开新16人图像/标签，没有读取 final100 进行方法选择。

旧 Adam60 的固定 TRAIN 配对：FLAIR 正确 margin `+1.4050400871`，增强T1正确 margin `−1.3215722103`。这是该配对的实际类别判别错误，不仅是开放式生成随机失败；不能据一个配对推广全部患者。

旧 v1 `softplus(−(mF+mT)/2)` 仅约束两者平均值，不能保证各自正确。新独立 v2 为 `L2=.5[softplus(−mF)+softplus(−mT)]`，未来联合目标固定 `L_GSPO+.1L2`。原 GSPO 主损失、reward、LR、LoRA 和历史实验均未改。v2 是显式监督辅助，不是原始纯 Vero GSPO。

同一旧评分点上，CPU 实际重算 `L1=.6724979108`、`L2=.8887110315`。对共同 FLAIR score 偏置的导数：v1 为0，v2 为 `+.2962128094`。v2 对 T1 错误 margin 施加更强的分数空间纠正。但共享模型参数、Adam 动量和类别先验仍可能抵消或反转某类效果；这不是准确率改善证据。

## 已完成的代码和证据

- `paired_classification_signal_v2.py`：稳定可微目标、解析导数、固定 `.1` 系数和真实更新类型。旧v1保留。
- `paired_class_margin_probe_v2.py`：新的一次只读原生 margin VJP/Adam 方向入口，待精确独立审查和新的人类授权；不复用上轮已消耗许可。
- `class_margin_aux_v2{,_driver,_entry}.py`：可执行的独立 F32 原生集成。原主池28+48和历史 index8:12 四槽不变，仅sequence槽附加同患者z045另一模态。梯度串行计算后合并，只进行一次持久Adam调用；实际执行尚未授权。
- CPU事务夹具覆盖精确续接、异常回滚、原子发布后不重复更新、完整rollout复用及成本只计一次。CPU夹具不是原生joint更新或原生恢复通过的证据。
- 独立数学复核：`CLASSIFICATION_MARGIN_V2_CPU_INDEPENDENT_REVIEW_V1.json`，有限差分最大误差 `3.914e−12`；确认无图像先验中和和共享参数伤害T1两个失败边界。

Root 最终实际运行三份新测试文件：**42 tests PASS（0.25s）**，`git diff --check` 通过。其中9个loss测试、16个只读probe测试、17个integration测试。当前CPU事务证明为 `CLASS_MARGIN_AUX_V2_CPU_TRANSACTIONS_V2.json`；V1保留为修复前版本记录。

每张图的正确/错误类别使用同一输入、prompt、canonical prefix。辅助仅包含label token；mean为主，sum/first独立报告，不将长度读取变化冒充学习。模型不可见patient/site/path/隐藏标签。

## 新原生证据：尚未运行

拟固定既有 TRAIN F/T配对、原Dallas输入和同shape的零归一化pixel control。最多11次full-TF前向、4次margin backward、4675个TF位置硬上限（完整计划预期2868）、1200秒操作启动窗口，0生成、0参数扰动、0optimizer更新。swap由已算的反模态梯度作符号/行交换，不追加模型调用。

分别报告两个正确类别margin及Dallas margin沿v1/v2欧氏方向、真实Adam60 moments解析预条件方向、`g=0`旧动量基线和增量的导数。没有运行前不报告这些数值。均值梯度不冒充sum/first梯度；blank是OOD控制，不是有真实标签的无图像任务。该诊断也不能冒充完整GSPO+辅助joint原生训练资格。

精确待授权scope：文件SHA256 `0dc854bcd5834ecb8f27fb38a867304084f7c2772f2be00a5c1ce08e55670d1f`；canonical SHA256 `d4ee98b6e4db3e11e625c9701aaeea274e39f7cc322ea80dbb49ba25e49fc6dc`。只读scope提出局部电池≥60%条件，仍须本次明确授权；所有内存/swap/温度/其他重进程检查保持，不改全局训练电源门槛。

## 后续唯一微试及停止规则

未来另行授权后，从完整Adam60/LoRA/Adam/Python+MLXRNG继承，最多4原槽、2次**总**optimizer更新（JOINT/AUXILIARY_ONLY/GSPO_ONLY合计），不强制9对采样。主rollout输入+返回预留9040，辅助TF位置1812；全部full-TF至多40调用/19860位置、backward至多20。内部重算与AR lookahead另列，不能把10852说成总计算量。1200秒操作启动上限先到即停，单次原子调用与释放可超时；不自动补组。

四个候选槽最多触及2组患者配对；若头两个槽均更新，则两步上限立即停止，实际只触及1组配对及1个Dallas槽。不能把准备好的56张图或全部四槽称为已训练曝光。

inactive GSPO不伪造优势；辅助非零才可记AUXILIARY_ONLY。零辅助且GSPO无效则NO_UPDATE，joint全梯度抵消或非有限则停止/回滚。所有历史父状态只读保留。

10–20分钟仅是微试操作窗口，不是已测的新路径吞吐。后续父子同F32内部40工程评价，参考历史原生每模型约8.5分钟的生成与加载，另加候选评分，规划约20–30分钟，长输出/资源hold可更久。若采用每模型40次greedy、cap256，则父子新生成上限为20480 tokens；canonical候选与oracle TF位置另列，精确评价scope仍须冻结和审查，不能用AR token上限代表全部推理成本。当前没有测得新的训练速度或准确率改善。

新16人32图保持封存，仅引用旧seal/manifest hashes。正式父子评价必须先冻结方法、确切checkpoint和评价流程，另行授权一次，全16人纳入，4对质量flag不得预测后剔除。内部工程结果不能宣称独立患者泛化。

**当前训练就绪：NO。** 新原生梯度方向尚待授权和实测；原生joint/on-policy辅助训练尚未实测；训练本身还需要单独的人类决定。若T1 candidate不改善或FLAIR/Dallas明显退化，不扩训同配置。
