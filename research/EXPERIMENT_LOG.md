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


## 2026-10-20 / SHARED_ROLLOUT_NATIVE_ZERO_UPDATE_ACTUAL / LIMITED_INTEGRATION

- 原AC scope未执行。研究者明确批准本次局部电池例外后，以独立复核的新scope32b03d90/canonicald8940abf执行一次；保留所有内存/swap/温度与低电量停止保护，全局训练门槛未改。局部代码e860988；root90项相关CPU测试、独立12项针对性检查通过，非重复模型资格。
- 实际单进程退出0（2026-10-20T23:00:13.372904Z），344.795秒；8真实响应/162返回tokens，16scorer前向/8只读backward/3709TF位置，optimizer updates=0，无重试。结束模型None、MLXactive1096B。
- rollout/own-contextTF最大差4.863739013671875e-05<.002，ownold/current差0。A/B各1个有效K4，gradientL2分别.9864109546776391/1.0637626054069058；本draw没有有效组数量优势，不是accuracy提升。
- FLAIR和T1各采样2/4正确；5完整格式/3无效格式，8EOS、0caps。运行时1942参数指纹/fullAdam/PythonMLXRNG一致、父磁盘4项artifacthash不变。完整梯度数组未持久化，saved-onlyreview不能独立重算nativeL2。
- 公开聚合和中文实际报告已生成，PRIVATE原始bundle/ownold/R1/reward/计数/log/exit/release保留。A/B训练与后续评价未授权、未运行；plan/lock不变，新16输出继续封存，旧final100未用于方法选择。

## 2026-10-20 / README_REFRESH_SAFE_CLEANUP / DOCS_ONLY

- 实际ClaudeCode session2e433b8c-cd0d-4992-ac2d-dfa9fca26634负责README/清理候选，commit9327ee38d99a7d45d98745d3d2073066de0d13cd；Codex独立核对8方法×3seed/48任务格、B28类别recall、2286/362和15个本地/固定Git链接。保留真实工具approval-required结果，不伪造模型审查。
- 在独立docs worktree更新展示名Vero-to-Brain-MRI与8节首页；26,324B旧README保留Git6ee3d16固定链接，不复制出新档案。主计划用已有不可变commit链接，不在main创建竞争计划或复制研究代码。
- 对648既有tracked文件的metadata/blob检查：0完全重复组、0明确cache/temp/空文件；因此既有文件删除0。仅去掉本轮新生成且已被最终review替代的清理proposal1份，仍可从9327commit恢复；最终相对main净删除0。未删改数据、历史模型/状态、梯度/负结果、实验依赖或既有日志。
- 清理时原生进程在另一个worktree继续；没有停止/重启进程、锁抢占或第二模型。最终native已自然退出0释放。文档分支和main PR状态将在实际推送/合并后追加。

- 实际资格saved-only独立review已完成：Codex /root/full_arm_execution_contract，公开SHA0fe61d6a811b074d6feb0d3f232d62f50085f23892b4913f2a31789d402758ad，PWL。独立重算reward/gate/advantage/1/8loss/parity/counters/source70hash；runtime-only状态与gradientnorm认证边界明确。459资源samples中448CONTINUE/11HOLD，456NORMAL/3WARNING，按原gate正常暂停后继续；没有掩盖HOLD或宣称保证安全。
- README最终actual-status狭窄复核已完成：README SHA179cf90a3a7585bde0025b0e5b293ec28aeed2a54ae1a7153d84fc715e73f07c；独立docsreview SHA7b0f1c91f47c121d9273ac2b86740c462742489950d3608d1e44537cc79c834b。主计划/锁hash保持c7e96b76/6844b67a。


## 2026-10-20 / DOCS_PR_ACTUAL_MERGE / VERIFIED

- PRIVATE docs branch docs/vero-to-brain-mri-readme-refresh-v1 已实际推送61f76a63a7270744e2f1cdcd1e0e6acf1d63a586。PR#4仅README+两份aggregate文档review/verification，MERGEABLE/CLEAN，无configuredchecks，未修改branchrules；已按研究者授权正常merge，非force/squash/historyrewrite。
- PR https://github.com/MelodyResearch/Longitudinal-BrainMRI-Verifiable-RL/pull/4 于2026-10-20T23:09:16Z实际MERGED，main c92b133d1a8fa1524c4a6a650d9c6c84dada3d03。GitHubAPI读取默认README bytes与reviewed本地逐字节相等：SHA179cf90a3a7585bde0025b0e5b293ec28aeed2a54ae1a7153d84fc715e73f07c，10790B（原26324B）。
- native qualification源/结果保持独立researchbranch；实际结果commit cadf78babdfdfdfed89c54dab4d0d8bca7cfdb37已推送并远端exact。没有把未训练MLX代码合入main，没有新的训练/模型调用/optimizer更新。R原main checkout未强切换，S/D均独立；原模型状态/数据/负结果/冻结plan与锁完整保留。


## 2026-10-20 / REPOSITORY_RENAME / ADMINISTRATIVE_VERIFIED

- 研究者明确授权后实际执行 gh repo rename：MelodyResearch/Longitudinal-BrainMRI-Verifiable-RL → MelodyResearch/Vero-to-Brain-MRI。现URL https://github.com/MelodyResearch/Vero-to-Brain-MRI ，同repositoryID1421422665；PRIVATE/main不变，无新repo/删除/forcepush/历史重写。
- rename前后所有17branchrefs、0tags、3PR历史、1独立Issue、1workflow/4Actionsruns的ID/sha逐项相等。rename-only main仍c92b133d1a8fa1524c4a6a650d9c6c84dada3d03，research仍612cb69fbb79a0d981fc41a866c76eddfebd38a5。所有历史训练/正式结果/Adam60/冻结plan与日志refs保留；本地模型数据路径未重命名。
- 实际发现2个独立clone（主clone及ResearchScratch），合计17checkout/worktree；新HTTPS origin统一更新并各fetch退出0，所有17remote-v核对一致。home元数据搜索排除了cache/runtime，macOS Photos保护目录未检查；没有其他外置卷。其他worktree既有untracked文件保持原样，不强切他人分支。
- 已登录浏览器直接进入新PRIVATE页面，About新描述可见，旧URL实际跳转新URL。homepage/topics原空值保留。About为Vero-style verifiable reinforcement learning for image-grounded brain MRI reasoning.
- README当前9个导航URL更换canonical新repo，commit/pinned内容不变；current scripts/src/config/CI扫描0旧repo调用。历史covenant原签发scope、初次push事实、旧README核验hash和receipt/local路径保留。docs PR#5仅README+行政聚合记录，实际正常merge于2026-10-20T23:31:21Z，main b8b0493754451befaf5e61285ca8bb05570ca36d；不是rename重写历史。
- rename新增模型调用/optimizer更新/rollout/评价=0，没有停止科研进程或抢占锁。源证据PRIVATE before/after/API/remoteinventory保留；公开记录results/REPOSITORY_RENAME_ACTUAL_VERIFICATION_2026-10-20.json在main。

## 2026-10-20 / NEXT_AB_PREPARATION_DIRECTIVE / HUMAN_DECISION_RECORDED

- 新研究者附件SHA9ed64a48104aa99fb617dec7cb2dfee50b87e68237a983fee6d24b4fc608de3e明确转入最小真实A/B更新准备，不重复qualification、不重抽响应、不改辅助loss。阶段为ENGINEERING PASS / NATIVE INTEGRATION PASS / ACTIVE-GROUP BENEFIT NOT_OBSERVED_THIS_DRAW / METHOD BENEFIT INCONCLUSIVE / ACCURACY NOT_TESTED。
- 复用已认证8response bundle，两臂同Adam60/fullAdam/R0，最多各1step，执行前必须新独立scope review和研究者一次明确训练授权；目前仍无step许可。CPU准备必须保存A/B真梯度/cos在首step前；不匹配本pair的旧marginVJP不得复用或伪造方向结论。
- 原资格proof/source70项及原失败/部分证据不改写；必要新driver/证据delta采用一次精确source兼容审查，不能把新代码伪称已经native-qualified。新16模型输出继续封存，原final100不参与选择，冻结主计划内容/锁仍未修改。

## 2026-10-20 / LIGHTWEIGHT_FIND_SKILLS / COMPLETED_NO_INSTALL

- 新研究者指令要求find-skills但不能阻塞A/B。已真实执行8组CLI搜索（research/RL/GSPO/Python数值测试/MLX/PyTorch/GitHub/reproducibility），全退出0；检查4候选pinnedSKILL正文+来源metadata，发现阶段结束，新增安装0。
- 优先复用已有TDD/systematic-debugging/code-review/verification，研究侧Norman/novelty/experiment-design按问题使用。通用MLX不覆盖VLMGSPO+fullstate，verl/TRL仅参考不换backend；scientificpytest候选含错误approx断言，已最小CPU反例验证，未盲从。
- 独立技能发现者Codex /root/rc_data；actualClaude skilluse本项NOT_VERIFIED，不伪造使用证明。没有上传MRI/权重/Adam/PRIVATEreceipt、没有加载模型或新增训练，见results/SKILL_DISCOVERY_LIGHTWEIGHT_V1.json。


## 2026-10-20 / CANONICAL_LOCAL_REMOTE_NORMALIZATION / HUMAN_AUTHORIZED

- 研究者明确决定不恢复旧worktrees；本轮没有旧目录恢复/搜索/重建、没有模型重跑。主目录安全rename为Vero-to-Brain-MRI，同inode/HEAD/refs/config/cleanstatus，随后main正常ff到b8b0493；非重新clone。
- 当前仅main checkout和一个必要的新.worktrees/shared-rollout-ab，后者从已保存当前research分支6808f58取tracked源码/计划/日志。15个此前消失的历史worktree没有重建；当前prune检查0stale registration。同步当前README canonicalURL，新增CURRENT_DEVELOPMENT.md，不修改冻结plan/lock。
- 远端18→6branch：删除12个完整被保留分支包含的refs（2merged docs、1被lossfix包含的旧V3experiment、9旧研究refs）；逐tip ancestry核验，局部冗余branch refs删除11。保留main/current及4个仍有unique unmerged历史的分支。关闭superseded PR2/3而不merge，保留所有讨论/PR历史；Issue1仍是未完成整体研究目标，保持open。
- 已验证ResearchScratch duplicate clone无untracked/ignored/currenthandles、全部refs可由保留主仓历史访问后删除1clone/338files/6,709,248allocatedbytes。另删除4个ignored缓存目录/190files/3,596,565logicalbytes；不扫描删除local_data/work/checkpoint中的独有科研资产。current origin新HTTPS，fetch origin --prune实际退出0。
- 本轮只核验现存canonical主仓local_data/work中的当前A/B必要资产，不搜索旧worktree/隔离区：找到0个shared bundle，0个匹配1741ff2f...的adaptiveAdam60状态；3个step60 optimizer实际是SFT对照，不能冒充Adam60。因此tracked source与冻结计划存在，但当前不能声称bundle/fullAdam60仍可运行，未申请或执行A/B训练，不重rollout/不换parent/不制造scope。
- 主线仍minimal real A/B optimizer update；行政整理不更改方法/reward/采样/loss/预算，new16与final100没有用于选择。新模型调用/optimizer更新=0。精确删除/保留清单见results/REPOSITORY_NORMALIZATION_ACTUAL_V1.json；已明确结束旧worktree恢复工作。


## 2026-10-20 / REQUIRED_RUNTIME_RECOVERY / BLOCKED_TRASH_ACCESS

- 最新研究者附件仅授权恢复原shared8 bundle和adaptive Adam60必要payload，不重建旧worktrees、不重qualification、不训练。实际fetch origin --prune成功，六远端refs核实；全部2418本地Git对象（含unreachable）、1289blob逐项检查，最大198016B，无二进制、LFSpointer、目标adapterSHA或bundle schema。12个已删除branch tips仍可由保留refs访问；GitHub保留aggregate evidence，不包含runtime payload。
- 限定canonical repo与原outputs parent的metadata扫描0个bundle文件、0个qualification/uniform目录；现有三个SFT step60不匹配adaptive parent。没有读取new16/final100 outcomes，没有替换parent或伪造state。
- 已按新授权尝试Trash：shell返回macOS Operation not permitted；Finder初次连接成功，但GoTo未确认进入Trash，随后出现用户改变app控制提示与ScreenCaptureKit -3812。Trash内容未读取，因此本轮状态严格为BLOCKED_TRASH_ACCESS_NOT_UNRECOVERABLE，不能宣称Trash空/原资产不可恢复，也不启动Plan B。已请求研究者用Finder检查或复制候选至canonical local_data/recovery_staging。
- 原bundle canonical SHA f8305d64...与adaptive adapter SHA1741ff2f...仍为唯一接受目标；没有copy/覆盖runtime，源码、权重、Adam/RNG、冻结plan/lock不变。新增model/generation/backward/optimizer updates全部0。精确metadata证据PRIVATE存于local_data/recovery_staging；sanitized结果见results/REQUIRED_RUNTIME_ASSET_RECOVERY_ACTUAL_V1.json。

- 同轮后续Finder重新绑定成功并实际打开Trash，显示200顶层items，因此已排除Trash为空的推测；但目标文件尚未完成查找。下一操作再次USER_CHANGED_APPLICATION，实际AX变为外部输入的/Users/vv/.keras/ GoToWindow。本任务停止争用并请求暂停其他Finder控制或复制必要候选。最终状态更新为BLOCKED_TRASH_SEARCH_CONCURRENT_FINDER_CONTROL；未宣告unrecoverable/未启动PlanB。

- 同轮独立Codex /root/full_arm_execution_contract复算Git/current-path metadata并核对脱敏表述，结果PASS_WITH_LIMITATIONS仅针对恢复证据，不是资产恢复或A/B许可。review SHA ad696d32fe68fb25b09e47b68bfdef5886a6039a64bc1e5eb64c3fe128873ceb；保留Trash未完成查找及A/B NOT_READY边界。


## 2026-10-20 / EXACT_RUNTIME_DELETION_TRACE_AND_RECOVERY_CLOSURE / NO_MODEL_ACTION

- 已读取四份原始cleanup报告/inventory，并核对实际cleanup executor与成功调用记录。原shared qualification准确在旧.worktrees/vero-mri-shared-rollout-native-v1/local_data/shared_rollout_gspo/qualification_run_v1；adaptive Adam60准确在旧.worktrees/mri-adaptive-accuracy-gate/local_data/uniform_balance_continuation_v1/run/boundary_0008。二者属于Batch4，不在Batch3的local_data/models。Batch4用git worktree remove --force再rmdir/prune，DELETION_BYPASSED_TRASH。成功launcher窗口为2026-10-20T23:39:41.153Z至23:40:44.192Z；这是包含删除的调用时间，并非逐文件syscall时间。
- 预删除与最终machine inventory为15 linked worktrees；Claude摘要16及字节多122880的差异保留，不隐去。Batch3另以shutil.rmtree删除模型cache；Batch6还删除native env/quant cache。原始报告、失败/拒绝尝试与成功执行均保留，不把早期exit0拒绝当删除成功。
- Finder实际遍历Trash至末尾、执行准确目录/文件名搜索，manifest.json有35个非目标positive controls；未恢复/删除/PutBack任何项目。Trash索引不是字节级穷尽，但直接删除已确认，停止重复Trash循环。quarantine91条全为另一Memory项目；Time Machine无destination/machine directory，Data卷当前0snapshot，root三个OS-update不当作用户数据备份。3个已知副本根metadata检查0目标，Git证据不含private runtime。
- ORIGINAL_RUNTIME_ASSETS_UNRECOVERABLE仅指已检查、当前可用来源；未排除离线外部备份或未执行的磁盘取证。bundle canonical f8305d64...与adaptive adapter1741ff2f...均未找回，恢复runtime文件0；原A/B NOT_READY，不拿三个SFT step60替代。
- 只提出NEW_MATCHED_PARENT_AND_SHARED_ROLLOUT_AB_V1，未执行/未授权：现存原P_calibrated13 boundary114/Adam52候选adapter/receipt与原独立proof匹配，9文件hash全PASS；CPU读取完整state通过504moments/1011state tensors/有限值/非负second moments/Python+MLXRNG/精确inventory，未native restore。选择理由仅原训练链祖先连续性，不用final100；不能复用旧Adam60授权。native base/runtime/quant依赖需另行认证恢复。
- 新增model load/generation/backward/optimizer updates/qualification/cleanup均0，冻结plan和lock未改。详见results/REQUIRED_RUNTIME_EXACT_DELETION_TRACE_AND_RECOVERY_V1.json及独立review；原BLOCKED_TRASH历史报告保留，由本轮精确证据状态取代。

- 最终独立Codex /root/full_arm_execution_contract对精确删除trace及新report SHA537b76900ef10c480e8e16e922d3a17384e0ebb6e54ba9643f1fd2c6d448d0de作PASS_WITH_LIMITATIONS，review SHA8ee5604adf278676867e500c55f711ed0e6868b25776ba7e1fa18cab9bda9749。P52数组及新增Finder检查标明operator evidence未独立重跑；不把恢复来源窄结论扩大为全局取证不可能。
