# Shared-rollout GSPO 零更新原生资格：实际结果（2026-10-20）

## 执行与边界

研究者单次授权及本次局部电池例外经过精确scope复核后执行。实际进程退出0，用时344.795秒；没有重试、训练或optimizer更新。模型已释放，MLX active1096 bytes。全局训练电源策略未修改。

共用一个既定TRAIN z045 FLAIR/T1配对、Adam60完整父状态；A为F4/T4，B为F0F1T0T1/F2F3T2T3。真实8条响应只生成一次，每条保留自己的图像/prompt/old-logp/reward归属，两臂从相同父状态只计算梯度，不执行step。

## 实际数值

|项目|结果|
|---|---:|
|真实响应 / 返回tokens|8 / 162|
|scorer前向 / 只读backward|16 / 8|
|teacher-forced输入位置|3709|
|optimizer updates|0|
|rollout/scorer最大logp绝对差|0.00004863739013671875（阈值0.002）|
|own-old/current最大差|0|
|A有效K4 / gradient L2|1 / 0.9864109546776391|
|B有效K4 / gradient L2|1 / 1.0637626054069058|
|FLAIR采样正确 / T1采样正确|2/4 / 2/4|
|完整格式 / 格式无效 / 长度截断|5 / 3 / 0|

两臂各1个准入组，这个固定draw没有显示有效组数量优势。不同梯度范数不证明更好的类别方向或准确率。8个采样TRAIN回答的正确率不是greedy或candidate评价，更不是泛化提升。

## 状态完整性与限制

运行时全参数指纹、完整Adam及Python/MLX RNG检查报告与父状态一致，结束时释放模型；外部exit/closed-log与父adapter/optimizer磁盘hash经过核验。原生完整梯度数组没有保存，saved-only reviewer不能独立重算原生gradient L2或重跑autograd；其复核范围为保存的绑定、真实reward/gate/advantage/计数/parity及状态检查证据。

公开聚合：`results/SHARED_ROLLOUT_ACTUAL_READONLY_QUALIFICATION_V1.json`；独立复核见同目录`SHARED_ROLLOUT_ACTUAL_READONLY_QUALIFICATION_INDEPENDENT_REVIEW_V1.json`。原始患者关联响应、token、状态及完整receipts留PRIVATE。

## 下一步

本次只取得有限真实原生集成证据，没有MRI准确率提升结果。未启动A/B训练、未执行后续模型评价，未访问新16人输出或旧final100原始输出用于选择。当前单次资格权限已消费；任何未来A/B更新和评价均须新的明确研究者授权。
