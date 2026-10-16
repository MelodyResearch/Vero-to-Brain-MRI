# Vero 配方向脑 MRI 的初始训练契约

**状态：IMPLEMENTED / TESTED_CPU；MODEL_INFERENCE、RL_TRAINING：NOT_RUN。** 本阶段实现 Vero 方法迁移的可检查接口与数值目标，不是模型训练，也没有生成 MRI 答案或更新任何权重。纵向变化是五个拟议 MRI 类别之一；总体研究问题仍是 Vero 式**任务均衡、多奖励路由、开放式输出保留和 GSPO 优化**能否在真正患者分离的脑 MRI 上改善跨任务转移。

## 来源版本和需先解决的差异

- 固定论文：[Vero arXiv:2604.04917v3](https://arxiv.org/html/2604.04917v3)，特别是 §3.1–3.3、Fig. 2、Table 2、Table 4、§5–6、Appendix A.5/B.2/B.3。论文的训练主体是六类各 100K 样本的 Vero-600K，任务权重与 GSPO/GRPO 消融表仅覆盖**五类**，不能当成六类实验的直接证据。
- 官方公开 [仓库](https://github.com/zlab-princeton/vero)、[训练文档](https://github.com/zlab-princeton/vero/blob/main/docs/TRAINING.md)和[数据格式/路由文档](https://github.com/zlab-princeton/vero/blob/main/docs/DATA.md)于 2026-10-15 查阅；仓库提交 SHA **UNVERIFIED**，论文与当前 `main` 的训练代码等价性 **UNVERIFIED**。当前数据文档宣称“六类均匀抽样 +5.8”，但论文 Table 2/A.5 的 +5.8 来自五类（不含 Captioning/IF）；以版本化原文为消融范围依据。
- 论文 Appendix B.2 正文写 2,343 步，Table B2 为列出的模型写 2,000 步；当前训练文档写 4,000 步。尚未锁定配置和实际代码版本，不应任选一个声称原样复现。论文 Table B1 列 8 rollouts/提示、训练 batch 256、PPO mini-batch 128、学习率 1e-6、warmup 40、clip 0.0003/0.0004、KL=0；这些是来源参数，**不是本地运行日志**。
- Vero 的[训练样本结构](https://github.com/zlab-princeton/vero/blob/main/docs/DATA.md)包含 `prompt`、`images`、`reward_type`、`data_source`、`domain`、`answer`。官方路由有十类；本地仅实现与现有 MRI 可审计代码相接的五条接口，并对未实现类型拒绝运行。

## 原方法 → 初始 MRI 契约

| 原方法关键机制 | 本地实现 | 现在的边界 |
| --- | --- | --- |
| 六类任务各 100K 的均匀类别分配，[Fig. 2/§3.2](https://arxiv.org/html/2604.04917v3) | [`sample_equal_categories`](../src/mri_rl/vero_recipe.py) 对配置内每类抽取相同数量，固定种子，可小数据重复抽样；[配置](../configs/vero_mri_recipe.json) 为五个**拟议 MRI 类别**，每类 20%。 | 这不是 600K 数据复现。小样本重复抽样会重复受试者；必须先验证患者级划分、样本独立性及类别标签。`open_ended` 明确保留。 |
| 十类 metadata 驱动奖励路由，[§3.3/官方 DATA](https://github.com/zlab-princeton/vero/blob/main/docs/DATA.md) | `route_accuracy` 支持 `exact_match`、`numeric`、`grounding_2d`、`longitudinal_direction` 和需外部回调的 `open_ended_judge`；未知路由失败。 | 其中 2D grounding 只是帧级代理，**不是 3D voxel verifier**；方向标签尚无可重复性阈值，不能当临床变化真值。开放式评判器尚未验证，缺回调直接失败。其余五条 Vero 路由未实现。 |
| 0.8 正确性 + 0.2 格式 + 超长项，[Eqs. 3–4](https://arxiv.org/html/2604.04917v3) | `score_response` 解析 `<think>/<answer>`、单个 boxed 符号答案，并把路由分、格式分和线性 overlong 项组合；响应长度须调用方提供 tokenizer 实测值。 | 未实现 Vero 完整的符号、动作及 judge 解析；未实现 overlong 外其他隐含训练辅助项。输入过长可得负奖励，符合公式。 |
| GSPO/GSPO-token 序列重要性比率和非对称 clip，[§3.3 Eq. 2，Appendix B.2 Eqs. 6–9](https://arxiv.org/html/2604.04917v3) | `gspo_clipped_surrogate` 计算每序列 token log-ratio 均值的指数、组标准化优势及 clipped `min` 数值目标；使用 0.0003/0.0004 缺省值。 | 纯 Python 数值核验**不能**证明 GSPO-token 的 stop-gradient 放置、自动微分、优化器、梯度聚合或参数更新与官方训练代码等价。标准差采用总体约定；官方实际数值实现需代码锁版核实。 |

在 [测试](/Users/vv/Documents/Codex/2026-10-15/files-pasted-by-the-user-codex-2/outputs/Longitudinal-BrainMRI-Verifiable-RL/tests/test_vero_recipe.py) 中，等额采样对开放式和纵向类别都保留 4 条工程记录；无类别样本时失败。格式有效且答案正确时组合分为 1.0；答案未 boxed 时为 0.9；`max_tokens=4096`、`buffer=2048` 时长度 3072 的超长惩罚为 −0.5。手算的二响应 GSPO 例子中比率 0.9995/1.0005 分别触发下界/上界处理；按官方 veRL `torch.std` 默认样本标准差，优势为 ±0.70710678，目标约 0.00024749。这些都是**代码机制检查**，不是影像表现或训练结果。

## 运行门槛与下一实验

配置中的 `starting_checkpoint`、`data_revision`、`steps` 和 `batch_size` 保持空值，训练状态为 `NOT_RUN`。进入正式迁移前，需：（1）按患者隔离并扩大真实 MRI 任务集；（2）为 anatomy、quantitative、spatial、longitudinal 和 open-ended 逐项取得独立且可靠的目标与奖励；（3）确定经审查的开放式 judge，做奖励黑客/无图/错访视控制；（4）锁定官方代码 SHA、模型权重、视觉 token 预算和实际训练日程；（5）建立相同数据/算力的 SFT、单任务 RL、比例混合 RL、Vero 等额混合、MedGRPO 校准、MT-GRPO 比例保持采样基线；（6）取得可核验的 GPU 运行、checkpoint 和日志。纵向任务无法满足门槛时，应暂时从可训练类别中剔除并**重新声明类别权重与假设**，不能用任意方向标签补齐。

**当前结论：** Vero 方法迁移的初始软件契约已经有可运行的 CPU 检查；真正的 Vero-to-MRI 训练、模型输出、跨任务迁移收益和脑 MRI 科学结论均未建立。
