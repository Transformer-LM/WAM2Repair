# WAM2Repair 研究交接记录

更新时间：2026-09-15

## 当前目标

在现有 H1/VLA×WAM 总目标下，重新启动 WAM2Repair 数据集与模型实验。研究命题是：WAM imagined future 中的物理幻觉（接触不成立、相对 6D 姿态错误、滑落、穿透/不可达）可能改变 VLA 候选动作排序；训练一个两阶段 action-conditioned physical-state repair model，修复 imagined state，并验证修复后是否改善 VLA 的候选动作选择和闭环执行。

## 两阶段方法

1. Stage 1：从真实/仿真未来状态施加通用状态扰动，学习恢复几何、接触、关系和滑落状态。
2. Stage 2：用重新生成的 WAM 特有错误（时序漂移、接触幻觉、对象/末端相对姿态偏差、多步误差）适应 WAM error distribution。
3. 下游：将 repaired imagined state 输入候选动作评分/重排序，比较 raw WAM、oracle state、repair model。

## 标签与损失

真实未来 RGB、PointMap/depth、对象和末端 6D pose、相对 pose、接触/关系状态、slip 事件、碰撞/穿透/可达性标签，以及每个候选动作的真实执行结果。旋转使用 SO(3) geodesic loss（不要直接 quaternion MSE），并加入 relative-pose、contact BCE、slip BCE、时序平滑与物理约束损失。

## 已有证据（不能当作最终主结果）

旧的 E1 诊断实验只有 18 个 paired scenes：baseline 15/18，raw WAM 12/18，anchored 13/18；raw action L2 约 3.1044，anchored 约 1.1827。它只证明 imagined state 可能改变 StarVLA 下游动作且造成伤害，尚未训练 learned repair，也没有 WAM2Repair 闭环主结果。

## 当前文件

- `refine-logs/EXPERIMENT_PLAN.md`：实验计划与 gates
- `refine-logs/DATASET_DESIGN.md`：数据集设计
- `refine-logs/EXPERIMENT_TRACKER.md`：实验追踪表
- `implementation/`：E0/E1 诊断和 WAM 输入脚本
- `AUTORESEARCH_STATUS.md`、`E1_RESULT.md`、`E1_AUDIT.md`：旧诊断记录

## 服务器与安全规则

- SSH：Windows OpenSSH，`__WAM2REPAIR_USER__@__REMOTE_HOST__`，只读写 `__WAM2REPAIR_ROOT__`
- 禁止 root/sudo/su/.bashrc/global Conda/system CUDA/shared dirs/真实机器人自主运动
- 个人代码：`__WAM2REPAIR_ROOT__/workspace`
- A100 0–3；每次启动前重新查 `nvidia-smi`；优先 2、3；显存 <500 MiB、利用率 <=5%、无 compute process 才可用；先小规模 sanity，再扩大
- W&B=false；服务器无外网，不安装在线依赖
- 之前用户已删除个人目录下 `runs/`、`checkpoints/`、`tmp/`、`logs/`，并停止旧 OpenPI/CF-DynAlign 任务；不要恢复或覆盖这些内容

## 下一步（从第一个未完成 gate 开始）

1. 检查个人 workspace 中现有 simulator、数据和 WAM 代码；由于旧 FastWAM checkpoint 已删除，先确认是否有可复用模型或从 Stage-1 synthetic corruption 开始。
2. 在远端个人目录生成最小 factual trajectory/state bank 与 corruption bank。
3. E0：单批 overfit + 标签/旋转/接触指标 sanity；失败就停止扩展。
4. E1：Stage 1/Stage 2 repair 训练；只在通过 E0 后运行。
5. E2：oracle/raw/repair 的候选动作排序与闭环模拟比较。
6. 所有结果写入个人目录 JSON/CSV/Markdown，并经过 experiment-audit 与 result-to-claim 后才能形成结论。

## 诚信边界

目前不能声称 WAM2Repair 已有效、已提升成功率或已具备论文新颖性；这些都要由上述实验和审计决定。若 Stage 1 能改善状态指标但不能改善真实动作排序，应停止把它包装成 VLA 方法。
