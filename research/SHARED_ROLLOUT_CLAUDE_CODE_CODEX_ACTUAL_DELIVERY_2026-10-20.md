# Shared-rollout GSPO：Claude 实现与 Codex 独立复核实际交付

状态：**原生源码已实现、CPU 已测；原生资格 NOT_RUN，训练未授权。** 本轮 MRI 模型加载0、原生前向0、原生反向0、optimizer updates0，真实准确率尚无新提高证据。未读取新16人预测或旧final100，未改冻结主计划及锁。

## 真实协作

已实际调用现有已登录 Claude Code 2.1.303，CLI报告模型 `claude-sonnet-5-5`，真实 session `141d22e8-4a62-4305-88a6-266b0f743b1e`。正常 acceptEdits/工具权限，无权限绕过。Claude 是主要源码和测试实现者，提交：

- `f17cf47bfad558fa7b08498c068e8a833adb74cf`：核心、混合上下文F32后端、单批driver、入口、评价及CPU夹具。
- `f77ce44d194d34b55e3dc69cf530c1682e883069`：资格PARTIAL、R1保存、完整bundle采用、原子提交、评价接口修订。
- `d985a76c6ba4816ed75be810530c826f7bb3c2c6`：恢复前Adam/RNG审计、不可变科学输入与执行元数据分离、采用证据重算。

Codex /root 阅读实现、运行测试、复现问题并要求Claude修复；独立 Codex /root/full_arm_execution_contract 在提交归档快照上复核。Codex最后补齐schema、真实prepare CLI和R1证据哈希链/强制采用路径，提交 `4ebdd868218f2bdb46484cfbfc191ec1cac4db72`、`800d5a2566b69837d0beb520c4a49854fa0c7607`、`490c502b127628963b389e7f142717d728ed2b06`。

真实Claude工具记录含7个需审批shell结果及工具schema/sed错误；它们不是患者数据共享拒绝。源码编辑通过本来允许的Edit完成。没有将患者图像、标识、raw回答、模型权重、Adam/RNG或凭据发给Claude。作者“没有工具拒绝”的概括不作为事实采用；公开provenance保留准确分类，原始日志私有。

## 已实现的方法

实验 subtype 固定为 `SHARED_ROLLOUT_BATCHED_ONE_STEP_GSPO_MECHANISM_CONTROL`，不同于未来逐K4立即更新的在线GSPO。

1. 固定已选TRAIN同患者z045配对。Adam60仅因训练连续性预定，不按旧final100挑选。
2. 同一不变父模型下只生成F4+T4，共8个真实响应，保存各自prompt/image/input指纹、token、真实生成logp、EOS及reward。
3. A：F4/T4；B：F0F1T0T1/F2F3T2T3。两个K4 loss等权平均，每个获准响应只计一次1/8。
4. 每条响应均使用自己的图像、prompt、视觉位置和old/current logp。原GSPO loss AST、reward/calibration、继承Adam60 gate、LR、LoRA保持不变，不加入任何辅助loss。
5. 两臂独立恢复相同完整Adam60及R0，汇总后各最多一次更新。生成后R1单独持久化；确定性阶段在恢复之前核验Adam/RNG，不能用恢复掩盖意外变化。
6. 真实无优势或总梯度为零则0更新，不重抽、不伪造优势、不执行仅靠旧动量的步骤。资格缺少有效反向覆盖记PARTIAL并不自动解锁训练。
7. 完整资格bundle未来只在另行训练授权后采用。生产训练入口强制该路径，8条回答不再重采，own-old不重算；新调用与继承成本分开。R1元数据和tensor哈希也逐项核验。
8. boundary原子rename就是提交点；marker随后失败也不重复更新。半rollout/未提交臂保留证据并停止，不自动重试。

## 实际验证与修复

首提交在独立21项反例中11失败；首修订15通过/6失败；之后原21项全通过。新增R1证据篡改反例又复现2项失败并完成修补。历史FAIL保留，不覆盖为PASS。

Root最终针对性CPU测试：**108 passed /0 failed**，包括共享8绑定、1/8、原loss自动微分双数夹具、状态故障、原子发布、资格/采用、评价CLI及主计划guard。它们使用NumPy/轻量模拟后端，不是实际MRI训练或原生资格。

真实CPU `prepare` 也已用预定私有TRAIN输入元数据执行：资格与未来采用训练两种spec均成功；69个源码/运行时文件绑定，科学输入哈希相同、执行scope不同。此步只读来源/receipt/manifest/cache证书及源码，没有读取权重/Adam数组。

独立复核实际执行23项独立反例与80项作者测试，共103项通过；与root的108项集合有重叠，不相加作唯一测试数。

正式独立源码/反例复核见 `results/SHARED_ROLLOUT_FINAL_CODEX_INDEPENDENT_REVIEW_V1.json`；root实际CPU证据见 `results/SHARED_ROLLOUT_ROOT_ACTUAL_CPU_VERIFICATION_V1.json`；真实Claude参与见 `results/SHARED_ROLLOUT_CLAUDE_CODE_ACTUAL_PARTICIPATION_V1.json`。任何源码/CPU PASS都不代表新混合路径已经原生通过。

## 下一次唯一请求的执行动作：零更新原生资格

源码冻结于 `490c502b127628963b389e7f142717d728ed2b06`。私有scope：`local_data/shared_rollout_gspo/preparation_QUALIFIER_FROZEN_V2_PRIVATE_qualify_scope.json`。

- Scope文件SHA256：`32157f37c0d17e6bee9daa3dd12c777a811611ddcf63448d66cf9d5423f0cb99`。
- Canonical scope SHA256：`1413c560b9358ca3bf97757ba35e385db4fd35733e359c25449d211677f948cb`。
- 最多8条响应、2048个返回token、2056次sampler draw、2080次generation LM API、8次生成vision embedding。
- own-old评分8次；两臂合计最多16次梯度前向及16次只读backward，故完整scorer上限24、TF位置11232、总vision API32。
- **0 optimizer updates**；单模型/单图梯度图；1800秒操作启动期限，不自动重试或扩大。原子模型调用/资源释放可能越过启动期限，此期限不是硬杀进程保证。
- 保留全局锁、AC、电池/内存/swap/温度/压力保护；执行前重新核验资源。之前电池或原生授权均已消耗，不复用。
- 每一条真实rollout-vs-own-scorer仍须≤0.002，确定性own-old-vs-current≤1e-5，完整模型/Adam/RNG审计与释放必须成功。

只有独立scope复核和研究者新授权同时到位才可加载模型。通过后仍需另一项明确训练许可，才能各臂最多一次Adam60→61；采用资格生成的同8条回答，新增生成和own-old评分均为0。后续原生评价亦保留独立、确切的授权门槛。

## 准确率评价与成本解释

已写出parent/A/B匹配F32评价CLI，固定原256/temp0/严格Vero解析，canonical mean/sum/first分开。比较B−A、A−Parent、B−Parent：sequence严格balanced accuracy、T1/FLAIR recall、candidate与margin、swap及Dallas保留表现。无更新臂直接作为父模型别名，避免重复推理。最多3个真实端点/120次greedy/360次候选/前缀oracle评分API、30720返回token、153000 TF位置，操作启动期限最多3600秒；这些是前瞻上限，不是本轮测得耗时。

**尚无新训练和模型评价，accuracy尚未提高。** 当前准备仅让受控实验能够执行。新16人输出继续封存；单TRAIN配对的一次更新只支持机制/工程判断，不能证明跨患者或解剖视觉推理收益。出现有效组或非零梯度也不是MRI准确率成功。
