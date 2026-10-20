# 实验日志 | Experiment Log

Append-only。此文件记录事实、失败、实现及资源，不改变CURRENT_RESEARCH_PLAN.md的路线。纠错只能追加说明并引用旧条目，不能回写旧事实。历史实验保留原报告/receipts；本日志于2026-10-20新建，不伪装为更早的现场日志。

## 2026-10-20 / INITIAL_DIRECTION_FREEZE / PRIMARY_VERO_TRANSFER

- 明确研究者指令：创建唯一主计划、归档监督auxv1/v2/v3、转向Vero same-prompt与paired-prompt grouping机制的可否证匹配比较。首次创建依据指令SHA256 `4ef822f5ec91c49713092d9091e6907512bed56c125782b1af6552dcc1d601b0`，不是agent自行推断新目标。
- 实际git fetch成功；origin/research/vero-mri-protected-mixture-cpu-v3=1f39f5ec748ad4b7686cadf5778096cb16309d75，origin/main=6ee3d1668f17189b24eaf1e850644a2cfbe5f69b。新独立worktree/branch=research/vero-mri-plan-paired-group-v1，继承全部历史材料。
- 阅读并核对covenant、primary execution、transfer experiment contract、thesis，以及原22实际结果、F32两组实际资格、TRAIN/空间任务及holdout16技术/独立证据。只读已有汇总，不读取final100原始预测或新16人输入/输出。
- 实现顺序：固定计划与历史状态标记 → Git基线/日志前缀守卫及RED/GREEN测试 → 完全匹配CPU分组/风险反例 → 核对真实可比保存rollout是否存在 → 准备有限A/B来源/预算/停止规则 → 独立复核与推送。研究者已明确要求执行以上准备，无模型启动授权。
- 治理守卫观察到RED（新module缺失）后GREEN，6项针对性CPU测试通过；plan+lock同时修改并放入approved=true仍被拒绝，日志改写/截断/删除被拒绝。模型加载/原生调用/训练/updates=0。
- 本日志先记录准备事实，后续CPU机制与复核结果单独追加；不因这些结果回写主计划。

## 2026-10-20 / PLAN_GUARD_BASELINE_EDGE / ENGINEERING_UNBLOCK

- 新增不完整Git基线反例：旧计划存在但旧锁或日志缺失时不得重新bootstrap。实际RED为缺少INVALID_BASELINE；修复后7项CPU测试通过。初次创建与未来计划修订权限仍分开。
- 版本化pre-commit脚本及push/PR CI加入仓库；未修改全局/共享Git hooks配置，未声称已设置GitHub required-status branch protection。这是意外漂移检测，不是人类身份认证或不可绕过的权限系统。

## 2026-10-20 / PLAN_GUARD_MULTI_COMMIT_FIX / REPRODUCED

- 独立reviewer在临时Git仓库复现：新branch只比较HEAD^会漏掉更早commit的plan+lock篡改，后续普通代码commit可能通过。保留此真实失败，不把初版7测试当完备证明。
- 修复为逐个first-parent治理文件变更检查；CI不再依赖新branch HEAD^ fallback。staged检查先验整个已提交历史，再强制当前HEAD与index比较，不能用旧--base重新bootstrap。
- 两个Git集成反例观察RED→GREEN（多commit隐藏变更；首次授权staged创建）。当前root9项测试通过。该守卫仍是本仓库误改检测，不声称能够防止刻意修改守卫自身或替代研究者身份审查。
- 没有模型加载、原生forward/backward、生成或optimizer更新。

## 2026-10-20 / PAIRED_PROMPT_CPU_MECHANISM / EXPLORATORY_DIAGNOSTIC

- 已实现不可拆的response自身context/prompt/image/token/own-old logp/reward绑定、固定F4/T4八回答same/mixed分组、原sample-std advantages和ratio/clip的CPU surrogate及AST导数oracle；没有CE或margin辅助目标，没有真实模型调用。
- 合成F全对/T全错：same准入0/2，paired准入2/2；若响应Jacobians相同，局部梯度仍0。共享类别先验反例可损害T1。精确枚举256个合成Bernoulli结果显示组归一化重加权及共享参数方向翻转。这些是数学反例，不是MRI运行或准确率。
- 实际接口修复：允许保留历史format=.5及校准后的accuracy，并单列raw_accuracy；不重算reward。gate归属明确为继承Adam60 valid-format-subset predicate，与历史18臂strict gate不同；两未来arm使用同一当前gate以隔离group因素，不伪称官方原实现。
- fixed8 CPU block归一化1/8；未来原生每K4更新仍1/4，禁止误用CPU block权重改变学习率。原生mixed-context后端和事务入口尚未实现，不能把CPU路由接口当已可训练。
- root实际28项pytest通过（17机制、9治理、2预算），0.45秒；来源审计18个CPU契约fixture另行通过。没有从CPU PASS推断accuracy提高或自动GO。

## 2026-10-20 / SAVED_MATCHED_AVAILABILITY_AND_AB_PREPARATION / ENGINEERING_UNBLOCK

- 审计21份固定命名证据：2286个原RL group中1139个sequence group，但原28序列每人仅一种模态；实际F32只有一个sequenceK4和一个DallasK4。可验证同checkpoint同条件的F4/T4八回答block=0：NOT_AVAILABLE_IN_ENUMERATED_AUTHENTICATED_SOURCES。未重读全部2286rollout，不作全磁盘不存在断言。
- 固定一个既有TRAIN z045 F/T配对和两个Dallas原输入，选择不依赖模型输出；影像不重渲染/解码，权重与optimizer数组不读取。数据清单不进入Git。
- v1准备中“八回答同point”的要求已区分：仅离线固定block需要；未来原生训练是每K4当前policy不变，组后可更新、后续轨迹不同。旧v1证据及脚本PRIVATE快照保留，v2是当前输入准备。
- 未来A/B各从同一Adam60/fullstate开始，4尝试group/16response/最多4实际更新，FLAIR4/T1四/Dallas8；每arm9040输入+返回token预留（4944+4096），两arm18080。原生TF/反向/资源资格与新授权均未完成，不执行训练。
- 真实CPU prepare入口已退出0，输出PROPOSED_CPU_PREPARED_NATIVE_NOT_IMPLEMENTED_NOT_AUTHORIZED；本轮全程model/native/generation/training/optimizer updates=0，未访问旧final100原始结果或新16人模型输出。主计划未因本次CPU发现修改。

## 2026-10-20 / INDEPENDENT_REVIEW_AND_FREEZE_DELIVERY / REPRODUCED

- 隔离Codex /root/full_arm_execution_contract实际独立复核：治理15个测试（root9+独立Git6）、最终机制/准备19项pytest、来源18个CPU契约；独立枚举256个数学状态复现偏置/方向翻转，未调用模型。
- 治理review SHA282feaab8517b3cb4fb7fe6858f8fb922bb143105d3f1620d226e92c2fbc25fd；机制/来源/实验准备review SHA9d600ced1142b850e4b3201ff64c5ff511b52007d9313875093d4cd5eb5cdb69。结论均PASS_WITH_LIMITATIONS，不授予训练/原生运行或主计划修改许可，CROSS_MODEL_REVIEW=NOT_VERIFIED。
- 独立复现机制CLI与AB准备CLI结果hash，--execute实际拒绝。源码明确未来native未实现，保持READY_FOR_TRAINING=NO；没有将参数/方向/accepted group当accuracy提升。
- 只发布当前availabilityV2；本轮未发布的V1中间材料原字节保留在PRIVATE目录，未删改历史Git事实。主计划初始冻结后SHA保持c7e96b76d8a821ae6c86a997e7b7f220c233295af073c360ae5ef76f4bfbe86e，没有因CPU结果自动重写。


## 2026-10-20 — Actual Claude Code shared-rollout native implementation; no model execution

- Request: implement `SHARED_ROLLOUT_BATCHED_ONE_STEP_GSPO_MECHANISM_CONTROL` inside the unchanged frozen plan; no auxiliary losses, historical reruns or current model permission.
- Actual Claude Code 2.1.303 / reported `claude-sonnet-5-5`, session141d22e8-4a62-4305-88a6-266b0f743b1e implemented and committed f17cf47/f77ce44/d985a76. Codex independently reproduced faults, requested actual revisions and added schema/CLI/R1-adoption regression fixes; final implementation490c502b127628963b389e7f142717d728ed2b06.
- Implemented: one sharedF4+T4 immutable bundle; A=F4/T4, B=F0F1T0T1/F2F3T2T3; own-image/prompt F32 scoring, inherited gate/loss/reward/LR/LoRA,1/8 each admitted response; exact parent/R0 restore,<=1step/arm only after new authorization; no-gradient=no update/no resample. Full atomic boundary commit and exact qualification-bundle adoption with separate inherited/new counters.
- Actual rootCPU108PASS. Separate independent Codex source review23adversarial+80author=103PASS (overlapping root set; not211unique). Initial FAIL snapshots retained; repaired pre-restoreAdam/RNG, malformed bundle, z045/parent binding, renamecommit, PARTIAL qualification, Q→T immutable input identity, evaluation CLI and R1 hash adoption.
- Actual nonmocked metadata preparation Q/T bothPASS,69source hashes exact, same predeclared TRAIN pair/scientificinputs and distinctinvocations. No image/weight/Adam payload reads. Final reviewPWL source/CPU engineering only: results/SHARED_ROLLOUT_FINAL_CODEX_INDEPENDENT_REVIEW_V1.json.
- Exact future zero-update qualifier FILEscopeSHA32157f37c0d17e6bee9daa3dd12c777a811611ddcf63448d66cf9d5423f0cb99:<=8responses/2048returnedtokens/8ownold+16gradientforwards/16backwards/11232TFpositions/1800soperation-start,0updates,existingglobalflock/AC/resourcegates/no retry. No human authorization exists; nativeNOT_RUN/trainingNOT_READY.
- This turn actualMRI model loads0/nativeforward0/nativebackward0/optimizerupdates0/accuracygainNOT_MEASURED. new16outputs sealed, final100untouched, CURRENT_RESEARCH_PLAN.md and lockbyteunchanged.
- Evidence: results/SHARED_ROLLOUT_CLAUDE_CODE_ACTUAL_PARTICIPATION_V1.json; results/SHARED_ROLLOUT_ROOT_ACTUAL_CPU_VERIFICATION_V1.json; research/SHARED_ROLLOUT_CLAUDE_CODE_CODEX_ACTUAL_DELIVERY_2026-10-20.md. Code-only collaboration is genuine; approval-required Claude shell results recorded, no patient-sharing denial bypass or fabricated native/accuracy PASS.
